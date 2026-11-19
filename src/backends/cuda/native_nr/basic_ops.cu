#include "native_nr.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace rw::native_nr {
namespace {
__global__ void input_nchw_kernel(const float* input,float* output,size_t count,
    int height,int width,int channels,int padded_height,int padded_width) {
    const size_t padded_pixels=size_t(padded_height)*padded_width;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x) {
        const size_t pixel=i%padded_pixels;
        const int channel=i/padded_pixels,row=pixel/padded_width,column=pixel%padded_width;
        const int y=min(max(row<height ? row : 2*height-row-2,0),height-1);
        const int x=min(max(column<width ? column : 2*width-column-2,0),width-1);
        output[i]=input[(size_t(y)*width+x)*channels+channel];
    }
}
__global__ void output_hwc_kernel(const float* input, float* output, size_t count,
    int padded_height, int padded_width, int width) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=size_t(blockDim.x)*gridDim.x) {
        const size_t pixel=i/3, channel=i%3;
        const size_t source=channel*size_t(padded_height)*padded_width+
                            (pixel/width)*padded_width+pixel%width;
        const float value=input[source];
        output[i]=value;
        if (!isfinite(value) || value<0.0f || value>1.0f)
            atomicExch(reinterpret_cast<unsigned int*>(output+count), __float_as_uint(1.0f));
    }
}
__global__ void validate_hwc_inputs_kernel(const float* color,size_t colors,
    const float* motion,size_t motions,const float* validity,size_t validities,float* flag) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<colors;i+=size_t(blockDim.x)*gridDim.x) {
        const float c=color[i];
        bool invalid=!isfinite(c) || c<0.0f || c>1.0f;
        if(i<motions)invalid|=!isfinite(motion[i]);
        if(i<validities) {
            const float v=validity[i];invalid|=!isfinite(v) || v<0.0f || v>1.0f;
        }
        if(invalid)atomicExch(reinterpret_cast<unsigned int*>(flag),__float_as_uint(2.0f));
    }
}
__global__ void convert_f32_to_f16_kernel(const float* input, __half* output, size_t count) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(blockDim.x) * gridDim.x)
        output[index] = __float2half_rn(input[index]);
}
__global__ void convert_f32_to_fp8_kernel(const float* input, unsigned char* output, size_t count) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(blockDim.x) * gridDim.x) {
        __nv_fp8_e4m3 encoded;
        encoded.__x = __nv_cvt_float_to_fp8(input[index], __NV_SATFINITE, __NV_E4M3);
        output[index] = encoded.__x;
    }
}
__global__ void convert_f16_to_f32_kernel(const __half* input, float* output, size_t count) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(blockDim.x) * gridDim.x)
        output[index] = __half2float(input[index]);
}
__device__ float rounded_half(float value) { return __half2float(__float2half_rn(value)); }
__device__ float rounded_e4m3(float value) {
    // torch.clamp preserves NaN; fminf/fmaxf alone would replace it with a bound.
    if (!isnan(value)) value = fminf(fmaxf(value, -448.0f), 448.0f);
    __nv_fp8_e4m3 encoded;
    encoded.__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    return static_cast<float>(encoded);
}
enum class Publication { Half, E4M3, HalfE4M3, ActivateE4M3 };
template<Publication quantization>
__global__ void publish(const float* input, float* output, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        float value = input[i];
        if constexpr (quantization == Publication::Half || quantization == Publication::HalfE4M3)
            value = rounded_half(value);
        if constexpr (quantization == Publication::ActivateE4M3) {
            const float t = isnan(value) ? value : fminf(fmaxf(value, -4.0f), 4.0f);
            const float inner = __fmaf_rn(fabsf(t), -0.055908203125f, 0.447265625f);
            value = __fmul_rn(value, __fmaf_rn(t, inner, 0.89453125f));
        }
        if constexpr (quantization != Publication::Half) value = rounded_e4m3(value);
        output[i] = value;
    }
}
template<Publication quantization>
Tensor publication(Context& context, const Tensor& input) {
    if (input.empty()) throw std::invalid_argument("cannot publish an empty native NR tensor");
    if (!context.owns(input)) throw std::invalid_argument("native NR publication belongs to another context");
    const auto count = input.numel();
    auto output = context.empty(input.shape);
    const auto blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
    publish<quantization><<<blocks, 256, 0, context.stream()>>>(input.data(), output.data(), count);
    cuda_check(cudaGetLastError(), "native NR publication kernel");
    return output;
}
template<Publication quantization>
void publication_into(Context& context, const Tensor& input, Tensor& output, size_t offset) {
    if (input.empty() || output.empty()) throw std::invalid_argument("cannot publish into an empty native NR tensor");
    if (!context.owns(input) || !context.owns(output)) throw std::invalid_argument("native NR publication belongs to another context");
    const auto count = input.numel();
    const auto output_count = output.numel();
    if (offset > output_count || count > output_count - offset)
        throw std::invalid_argument("native NR publication slice exceeds output tensor");
    const auto blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
    publish<quantization><<<blocks, 256, 0, context.stream()>>>(input.data(), output.data() + offset, count);
    cuda_check(cudaGetLastError(), "cuda publication into owned tensor slice");
}
int blas_dimension(int64_t value) {
    if (value <= 0 || value > INT_MAX) throw std::invalid_argument("native NR GEMM dimension exceeds cuBLAS int range");
    return static_cast<int>(value);
}
}

void convert_f32_to_f16(cudaStream_t stream, const float* input, void* output, size_t count) {
    if (!input || !output || !count) throw std::invalid_argument("native NR F16 conversion requires nonempty buffers");
    const auto blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
    convert_f32_to_f16_kernel<<<blocks, 256, 0, stream>>>(input, static_cast<__half*>(output), count);
    cuda_check(cudaGetLastError(), "native NR explicit-half conversion");
}
void convert_f32_to_fp8(cudaStream_t stream, const float* input, void* output, size_t count) {
    if (!input || !output || !count) throw std::invalid_argument("native NR FP8 conversion requires nonempty buffers");
    const auto blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
    convert_f32_to_fp8_kernel<<<blocks, 256, 0, stream>>>(input, static_cast<unsigned char*>(output), count);
    cuda_check(cudaGetLastError(), "native NR explicit-FP8 conversion");
}
void convert_f16_to_f32(cudaStream_t stream, const void* input, float* output, size_t count) {
    if (!input || !output || !count) throw std::invalid_argument("native NR F16 output conversion requires nonempty buffers");
    const auto blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
    convert_f16_to_f32_kernel<<<blocks, 256, 0, stream>>>(static_cast<const __half*>(input), output, count);
    cuda_check(cudaGetLastError(), "native NR explicit-FP8 output conversion");
}

bool fp8_matmul(Context& context, const Tensor& input, const Parameter& weight,
                int rows, int outputs, int channels, Tensor& output) {
    if (!context.explicit_fp8_gemm()) return false;
    auto* input_fp8 = context.fp8_activation(input.numel());
    const auto* weight_fp8 = context.fp8_parameter(weight);
    convert_f32_to_fp8(context.stream(), input.data(), input_fp8, input.numel());
    auto* output_f16 = context.half_activation(output.numel());
    const auto lt = context.blas_lt();
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, d_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    const auto cleanup = [&] {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (operation) cublasLtMatmulDescDestroy(operation);
        if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
        if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
        if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    };
    try {
        blas_check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                   "cublasLtMatmulDescCreate(native NR FP8)");
        const auto transpose_a = CUBLAS_OP_T, transpose_b = CUBLAS_OP_N;
        blas_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
            &transpose_a, sizeof(transpose_a)), "cublasLtMatmulDescSetAttribute(native NR FP8 A)");
        blas_check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
            &transpose_b, sizeof(transpose_b)), "cublasLtMatmulDescSetAttribute(native NR FP8 B)");
        // Weight [outputs, channels] and input [rows, channels] are already
        // column-major [channels, outputs] and [channels, rows] byte layouts.
        // TN produces [outputs, rows], whose contiguous bytes are the desired
        // row-major [rows, outputs] output.
        blas_check(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_8F_E4M3,
            channels, outputs, channels), "cublasLtMatrixLayoutCreate(native NR FP8 A)");
        blas_check(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_8F_E4M3,
            channels, rows, channels), "cublasLtMatrixLayoutCreate(native NR FP8 B)");
        blas_check(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_16F,
            outputs, rows, outputs), "cublasLtMatrixLayoutCreate(native NR FP8 D)");
        blas_check(cublasLtMatmulPreferenceCreate(&preference),
                   "cublasLtMatmulPreferenceCreate(native NR FP8)");
        const size_t workspace_bytes = context.blas_workspace_bytes();
        blas_check(cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)),
            "cublasLtMatmulPreferenceSetAttribute(native NR FP8)");
        cublasLtMatmulHeuristicResult_t heuristic{};
        int count = 0;
        blas_check(cublasLtMatmulAlgoGetHeuristic(lt, operation, a_layout, b_layout,
            d_layout, d_layout, preference, 1, &heuristic, &count),
            "cublasLtMatmulAlgoGetHeuristic(native NR FP8)");
        if (!count) { cleanup(); return false; }
        const float alpha = 1.0f, beta = 0.0f;
        blas_check(cublasLtMatmul(lt, operation, &alpha, weight_fp8, a_layout,
            input_fp8, b_layout, &beta, output_f16, d_layout, output_f16, d_layout,
            &heuristic.algo, context.blas_workspace(), workspace_bytes, context.stream()),
            "cublasLtMatmul(native NR FP8)");
        convert_f16_to_f32(context.stream(), output_f16, output.data(), output.numel());
        cleanup();
        return true;
    } catch (...) {
        cleanup();
        throw;
    }
}

Tensor narrow(Context& context, const Tensor& input) { return publication<Publication::Half>(context, input); }
Tensor round8(Context& context, const Tensor& input) { return publication<Publication::E4M3>(context, input); }
Tensor store(Context& context, const Tensor& input) { return publication<Publication::HalfE4M3>(context, input); }
void store_into(Context& context, const Tensor& input, Tensor& output, size_t offset) {
    publication_into<Publication::HalfE4M3>(context, input, output, offset);
}
Tensor activate(Context& context, const Tensor& input) { return publication<Publication::ActivateE4M3>(context, input); }

Tensor linear_last(Context& context, const Tensor& input, const Parameter& weight) {
    if (input.empty() || weight.empty() || input.shape.empty() || weight.shape().size() != 2)
        throw std::invalid_argument("native NR linear_last requires input[...,K], weight[N,K]");
    if (!context.owns(input))
        throw std::invalid_argument("native NR linear_last operands belong to another context");
    context.validate_parameter(weight);
    const auto input_count = input.numel();
    weight.numel();
    const int k = blas_dimension(input.shape.back());
    const int n = blas_dimension(weight.shape()[0]);
    if (weight.shape()[1] != k) throw std::invalid_argument("native NR linear_last K mismatch");
    const int m = blas_dimension(static_cast<int64_t>(input_count / static_cast<size_t>(k)));
    auto output_shape = input.shape;
    output_shape.back() = n;
    auto output = context.empty(std::move(output_shape));
    if (context.explicit_fp8_gemm() && fp8_matmul(context, input, weight, m, n, k, output)) return output;
    const float alpha = 1.0f, beta = 0.0f;
    if (context.explicit_half_gemm()) {
        const auto stream = context.stream();
        const auto* half_weight = context.half_parameter(weight);
        auto* half_input = context.half_activation(input.numel());
        convert_f32_to_f16(stream, input.data(), half_input, input.numel());
        blas_check(cublasGemmEx(context.blas(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
            &alpha, half_weight, CUDA_R_16F, k, half_input, CUDA_R_16F, k,
            &beta, output.data(), CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(native NR linear_last explicit F16)");
        return output;
    }
    const auto compute = context.half_math() ? CUBLAS_COMPUTE_32F_FAST_16F :
                         context.fast_math() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F_PEDANTIC;
    const auto algorithm = context.fast_math() ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    // Row-major Y = X W^T is column-major Y^T = W X^T.
    blas_check(cublasGemmEx(context.blas(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        &alpha, weight.data(), CUDA_R_32F, k, input.data(), CUDA_R_32F, k,
        &beta, output.data(), CUDA_R_32F, n, compute, algorithm),
        context.half_math() ? "cublasGemmEx(native NR linear_last FAST_16F)" :
        context.fast_math() ? "cublasGemmEx(native NR linear_last TF32 tensor op)" :
                              "cublasGemmEx(native NR linear_last FP32 pedantic)");
    return output;
}

Tensor linear_nchw(Context& context, const Tensor& input, const Parameter& weight) {
    if (input.empty() || weight.empty() || input.shape.size() != 4 || input.shape[0] != 1 || weight.shape().size() != 2)
        throw std::invalid_argument("native NR linear_nchw requires [1,C,H,W] and weight[N,C]");
    if (!context.owns(input))
        throw std::invalid_argument("native NR linear_nchw operands belong to another context");
    context.validate_parameter(weight);
    input.numel(); weight.numel();
    const int channels = blas_dimension(input.shape[1]);
    const int outputs = blas_dimension(weight.shape()[0]);
    if (weight.shape()[1] != channels) throw std::invalid_argument("native NR linear_nchw channel mismatch");
    if (input.shape[2] > INT_MAX / input.shape[3]) throw std::invalid_argument("native NR linear_nchw spatial count exceeds cuBLAS range");
    const int spatial = blas_dimension(input.shape[2] * input.shape[3]);
    auto output = context.empty({1, outputs, input.shape[2], input.shape[3]});
    if (context.explicit_fp8_gemm() && fp8_matmul(context, input, weight, spatial, outputs, channels, output)) return output;
    const float alpha = 1.0f, beta = 0.0f;
    if (context.explicit_half_gemm()) {
        const auto stream = context.stream();
        const auto* half_weight = context.half_parameter(weight);
        auto* half_input = context.half_activation(input.numel());
        convert_f32_to_f16(stream, input.data(), half_input, input.numel());
        blas_check(cublasGemmEx(context.blas(), CUBLAS_OP_N, CUBLAS_OP_N, spatial, outputs, channels,
            &alpha, half_input, CUDA_R_16F, spatial, half_weight, CUDA_R_16F, channels,
            &beta, output.data(), CUDA_R_32F, spatial, CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx(native NR linear_nchw explicit F16)");
        return output;
    }
    const auto compute = context.half_math() ? CUBLAS_COMPUTE_32F_FAST_16F :
                         context.fast_math() ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F_PEDANTIC;
    const auto algorithm = context.fast_math() ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    // Input [C,T] is seen as column-major [T,C]; output [N,T] as [T,N].
    blas_check(cublasGemmEx(context.blas(), CUBLAS_OP_N, CUBLAS_OP_N, spatial, outputs, channels,
        &alpha, input.data(), CUDA_R_32F, spatial, weight.data(), CUDA_R_32F, channels,
        &beta, output.data(), CUDA_R_32F, spatial, compute, algorithm),
        context.half_math() ? "cublasGemmEx(native NR linear_nchw FAST_16F)" :
        context.fast_math() ? "cublasGemmEx(native NR linear_nchw TF32 tensor op)" :
                              "cublasGemmEx(native NR linear_nchw FP32 pedantic)");
    return output;
}
Tensor input_nchw(Context& context,const Tensor& input,int padded_height,int padded_width) {
    if(!context.owns(input) || input.shape.size()!=4 || input.shape[0]!=1 ||
       input.shape[3]<1 || input.shape[3]>3 || input.shape[1]>padded_height || input.shape[2]>padded_width)
        throw std::invalid_argument("NR input HWC/padded geometry mismatch");
    auto output=context.empty({1,input.shape[3],padded_height,padded_width});
    const auto count=output.numel();
    input_nchw_kernel<<<static_cast<unsigned>(std::min<size_t>((count+255)/256,65535)),256,0,context.stream()>>>(
        input.data(),output.data(),count,input.shape[1],input.shape[2],input.shape[3],padded_height,padded_width);
    cuda_check(cudaGetLastError(),"mirror and pack NR input on GPU");
    return output;
}
void validate_hwc_inputs(Context& context,const Tensor& color,const Tensor* motion,
    const Tensor* validity,Tensor& delivery) {
    if(!context.owns(color) || !context.owns(delivery) ||
       (motion && !context.owns(*motion)) || (validity && !context.owns(*validity)) ||
       delivery.numel()!=color.numel()+1)
        throw std::invalid_argument("CUDA producer validation storage mismatch");
    validate_hwc_inputs_kernel<<<static_cast<unsigned>(std::min<size_t>((color.numel()+255)/256,65535)),256,0,context.stream()>>>(
        color.data(),color.numel(),motion?motion->data():nullptr,motion?motion->numel():0,
        validity?validity->data():nullptr,validity?validity->numel():0,delivery.data()+color.numel());
    cuda_check(cudaGetLastError(),"validate CUDA producer inputs");
}

Tensor output_hwc(Context& context, const Tensor& input, int height, int width) {
    if (!context.owns(input) || input.shape.size()!=4 || input.shape[0]!=1 || input.shape[1]!=3 ||
        height<=0 || width<=0 || height>input.shape[2] || width>input.shape[3])
        throw std::invalid_argument("NR output crop does not match the NCHW tensor");
    const size_t count=size_t(height)*width*3;
    auto output=context.empty({static_cast<int64_t>(count+1)});
    const auto stream=context.stream();
    cuda_check(cudaMemsetAsync(output.data()+count, 0, sizeof(float), stream), "reset NR output validity");
    output_hwc_kernel<<<static_cast<unsigned>(std::min<size_t>((count+255)/256,65535)),256,0,stream>>>(
        input.data(),output.data(),count,input.shape[2],input.shape[3],width);
    cuda_check(cudaGetLastError(), "crop NR output to HWC");
    return output;
}
} // namespace rw::native_nr
