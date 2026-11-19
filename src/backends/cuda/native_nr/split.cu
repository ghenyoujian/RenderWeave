// Native CUDA arithmetic port of dlssnr/layers/split_swin.py at
// mauri870/dlssnr-pytorch 3beef7935a8ef976b47ab78b906e75600c52d2de.
// Copyright (c) 2026 Mauri de Souza Meneguzzo. MIT License.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "native_nr.hpp"
#include "rw/rw_api.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <stdexcept>

// Executable semantic source: mauri870/dlssnr-pytorch at
// 3beef7935a8ef976b47ab78b906e75600c52d2de, layers/split_swin.py.
// In particular, Split attention's denominator is FP32 in that executable
// source. It must not be replaced by Swin/Vit's different half reductions.
namespace rw::native_nr {
namespace {
constexpr int channels = 512, heads = 16, head_dimensions = 32;

void split_gemm_strided_batched(Context& context, int m, int n, int k, const float* alpha,
    const float* a, int lda, long long stride_a, const float* b, int ldb, long long stride_b,
    const float* beta, float* c, int ldc, long long stride_c, int batch_count, const char* operation) {
    if (context.half_math()) {
        blas_check(cublasGemmStridedBatchedEx(context.blas(), CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k, alpha, a, CUDA_R_32F, lda, stride_a, b, CUDA_R_32F, ldb, stride_b,
            beta, c, CUDA_R_32F, ldc, stride_c, batch_count,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), operation);
    } else {
        blas_check(cublasSgemmStridedBatched(context.blas(), CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
            beta, c, ldc, stride_c, batch_count), operation);
    }
}

unsigned blocks(size_t count) {
    if (!count || (count - 1) / 256 + 1 > UINT_MAX)
        throw std::invalid_argument("split CUDA launch extent out of range");
    return static_cast<unsigned>((count - 1) / 256 + 1);
}
struct Map { int channels, height, width; size_t pixels; };
Map map_shape(const Tensor& tensor) {
    if (tensor.empty() || tensor.shape.size() != 4 || tensor.shape[0] != 1 ||
        tensor.shape[1] <= 0 || tensor.shape[1] > INT_MAX ||
        tensor.shape[2] <= 0 || tensor.shape[2] > INT_MAX ||
        tensor.shape[3] <= 0 || tensor.shape[3] > INT_MAX)
        throw std::invalid_argument("split expects positive [1,C,H,W]");
    tensor.numel();
    return {static_cast<int>(tensor.shape[1]), static_cast<int>(tensor.shape[2]),
            static_cast<int>(tensor.shape[3]),
            static_cast<size_t>(tensor.shape[2]) * static_cast<size_t>(tensor.shape[3])};
}
std::string cache_key(const LayerSpec& spec, const std::string& suffix) {
    return "split/block" + std::to_string(spec.block) + "/layer" +
        std::to_string(spec.layer) + "/" + suffix;
}
uint32_t bit(uint32_t value, unsigned from, unsigned to) {
    return ((value >> from) & 1u) << to;
}
size_t slice_source(unsigned row, unsigned column) {
    const unsigned lane = 4 * (row % 8) + (column % 8) / 2;
    const unsigned in_slice = column % 32;
    const unsigned byte = 2 * (in_slice / 8) + in_slice % 2;
    const unsigned slice = (column / 32) * 32 + row / 16;
    return size_t(slice) * 512 + lane * 16 + ((row % 16) / 8) * 8 + byte;
}

struct GroupedMatrices {
    std::vector<float> a;
    std::array<std::vector<float>, 8> q0, q2;
};
GroupedMatrices grouped_matrices(Weights& weights, const LayerSpec& spec) {
    const auto first = weights.raw(spec.block, spec.layer, "ffwd_a");
    const auto second = weights.raw(spec.block, spec.layer, "ffwd_b");
    constexpr size_t half_bytes = size_t(512) * 512;
    if (first.size() != half_bytes || second.size() != half_bytes)
        throw std::invalid_argument("split grouped FFN raw byte count mismatch");
    std::vector<uint8_t> record(2 * half_bytes);
    for (unsigned row = 0; row < 512; ++row) for (unsigned column = 0; column < 512; ++column) {
        const size_t source = slice_source(row, column);
        record[source] = first[size_t(row) * 512 + column];
        record[half_bytes + source] = second[size_t(row) * 512 + column];
    }
    GroupedMatrices result;
    result.a.resize(8 * 64 * 512);
    for (unsigned group = 0; group < 8; ++group) {
        result.q0[group].resize(256 * 64);
        result.q2[group].resize(64 * 256);
        for (unsigned row = 0; row < 64; ++row) for (unsigned k = 0; k < 512; ++k) {
            const auto offset = bit(k,0,0)|bit(k,1,4)|bit(k,2,5)|bit(k,3,1)|bit(k,4,2)|
                bit(k,5,14)|bit(k,6,15)|bit(k,7,16)|bit(k,8,17)|
                bit(row,0,3)|bit(row,1,6)|bit(row,2,7)|bit(row,3,8)|bit(row,4,9)|bit(row,5,10)|
                bit(group,0,11)|bit(group,1,12)|bit(group,2,13);
            result.a[(size_t(group) * 64 + row) * 512 + k] = rw_e4m3_decode(record[offset]);
        }
        for (unsigned j = 0; j < 256; ++j) for (unsigned r = 0; r < 64; ++r) {
            const auto offset = 262144 + group * 16384 +
                (bit(r,0,1)|bit(r,1,0)|bit(r,2,4)|bit(r,3,5)|bit(r,4,2)|bit(r,5,13)|
                 bit(j,0,6)|bit(j,1,3)|bit(j,2,9)|bit(j,3,7)|bit(j,4,8)|bit(j,5,10)|bit(j,6,11)|bit(j,7,12));
            result.q0[group][size_t(j) * 64 + r] = rw_e4m3_decode(record[offset]);
        }
        for (unsigned n = 0; n < 64; ++n) for (unsigned j = 0; j < 256; ++j) {
            const auto offset = 393216 + group * 16384 +
                (bit(j,0,0)|bit(j,1,1)|bit(j,2,2)|bit(j,3,4)|bit(j,4,5)|bit(j,5,11)|bit(j,6,12)|bit(j,7,13)|
                 bit(n,0,6)|bit(n,1,7)|bit(n,2,8)|bit(n,3,3)|bit(n,4,9)|bit(n,5,10));
            result.q2[group][size_t(n) * 256 + j] = rw_e4m3_decode(record[offset]);
        }
    }
    return result;
}

__device__ float quant8(float value) {
    __nv_fp8_e4m3 encoded;
    encoded.__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    return static_cast<float>(encoded);
}
__device__ float half_value(float value) { return __half2float(__float2half_rn(value)); }

__global__ void split_activation(const float* input, float* output, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const __half x = __float2half_rn(input[index]);
    const float wide_x = __half2float(x);
    const __half t = isnan(wide_x) ? x : __float2half_rn(fminf(fmaxf(wide_x, -4.0f), 4.0f));
    const __half u = __hfma(__float2half_rn(-0.055908203125f),
        __float2half_rn(fabsf(__half2float(t))), __float2half_rn(0.447265625f));
    const __half v = __hfma(t, u, __float2half_rn(0.89453125f));
    output[index] = quant8(__half2float(__hmul(x, v)));
}

__global__ void projection_residual(const float* product, const float* residual,
    const float* gain, float* output, size_t count, size_t pixels) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float term = residual ? __fmul_rn(residual[index], gain[index / pixels]) : 0.0f;
    // The source does two FP32 operations before narrowing, not an FP16 FMA.
    output[index] = half_value(__fadd_rn(product[index], term));
}

__global__ void pad_map(const float* input, float* output, int c, int h, int w,
    int out_h, int out_w, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t p = index % (size_t(out_h) * out_w);
    const int row = static_cast<int>(p / out_w), column = static_cast<int>(p % out_w);
    const size_t channel = index / (size_t(out_h) * out_w);
    output[index] = channel < size_t(c) && row < h && column < w
        ? input[channel * size_t(h) * w + size_t(row) * w + column] : 0.0f;
}

__global__ void to_windows(const float* input, float* output, int h, int w,
    int columns, int top, int left, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int channel = static_cast<int>(index % channels);
    const size_t token_index = index / channels;
    const int token = static_cast<int>(token_index % 64);
    const size_t window = token_index / 64;
    const int tile = token / 16, within = token % 16;
    const int row = static_cast<int>(window / columns) * 8 + (tile / 2) * 4 + within / 4 - top;
    const int column = static_cast<int>(window % columns) * 8 + (tile % 2) * 4 + within % 4 - left;
    output[index] = row >= 0 && column >= 0 && row < h && column < w
        ? input[(size_t(channel) * h + row) * w + column] : 0.0f;
}
__global__ void from_windows(const float* input, float* output, int h, int w,
    int columns, int top, int left, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t pixels = size_t(h) * w;
    const size_t channel = index / pixels, p = index % pixels;
    const int row = static_cast<int>(p / w) + top, column = static_cast<int>(p % w) + left;
    const size_t window = size_t(row / 8) * columns + column / 8;
    const int y = row % 8, x = column % 8;
    const int token = (y / 4) * 32 + (x / 4) * 16 + (y % 4) * 4 + x % 4;
    output[index] = input[(window * 64 + token) * channels + channel];
}

__device__ __half inverse_norm32(const float* values) {
    __half squares[32], pairs[16], eights[8], quads[4];
    #pragma unroll
    for (int d = 0; d < 32; ++d) {
        const __half value = __float2half_rn(values[d]);
        squares[d] = __hmul(value, value);
    }
    #pragma unroll
    for (int d = 0; d < 16; ++d) pairs[d] = __hadd(squares[d], squares[d + 16]);
    #pragma unroll
    for (int d = 0; d < 8; ++d) eights[d] = __hadd(pairs[d], pairs[d + 8]);
    #pragma unroll
    for (int d = 0; d < 4; ++d) quads[d] = __hadd(eights[d], eights[d + 4]);
    const __half total = __hadd(__hadd(quads[0],quads[2]),__hadd(quads[1],quads[3]));
    const float raw_squared = __half2float(total);
    const float squared = isnan(raw_squared) ? raw_squared : fmaxf(raw_squared, 6.198883056640625e-05f);
    return __float2half_rn(rsqrtf(squared));
}

__global__ void normalize_qkv(const float* input, const float* scale, float* output, size_t vectors) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= vectors) return;
    const int head = static_cast<int>(index % heads);
    const size_t token = index / heads;
    const float* raw = input + token * (3 * channels) + head * 96;
    float* result = output + token * (3 * channels) + head * 96;
    const __half query_inverse = inverse_norm32(raw), key_inverse = inverse_norm32(raw + 32);
    const __half temperature = __float2half_rn(scale[head]);
    #pragma unroll
    for (int d = 0; d < 32; ++d) {
        const __half query = __hmul(__hmul(__float2half_rn(raw[d]), query_inverse), temperature);
        const __half key = __hmul(__float2half_rn(raw[32+d]), key_inverse);
        result[d] = quant8(__half2float(query));
        result[32+d] = quant8(__half2float(key));
        result[64+d] = quant8(half_value(raw[64+d]));
    }
}

__device__ float exponential(float logit) {
    const __half affine = __hfma(__float2half_rn(logit), __float2half_rn(0.044921875f),
                                 __float2half_rn(1.30078125f));
    const float affine_float = __half2float(affine);
    const __half clipped = isnan(affine_float) ? affine :
        __float2half_rn(fminf(fmaxf(affine_float,1.03125f),1.5693359375f));
    const unsigned short bits = static_cast<unsigned short>((__half_as_ushort(clipped) & 0x3ffu) << 5);
    return __half2float(__ushort_as_half(bits));
}

#if RW_NR_HOPPER_WGMMA_ENABLED
#include "split_hopper.cuh"
#endif

// One warp per (window,head,query), O(64) shared memory, no N^2 tensor.
__global__ void split_attention(const float* qkv, const float* bias, float* output) {
    const unsigned index = blockIdx.x;
    const int query = index % 64, head = (index / 64) % heads;
    const size_t window = index / (64 * heads);
    const int d = threadIdx.x;
    __shared__ float probabilities[64];
    const float q = qkv[(window * 64 + query) * (3 * channels) + head * 96 + d];
    for (int key = 0; key < 64; ++key) {
        const float k = qkv[(window * 64 + key) * (3 * channels) + head * 96 + 32 + d];
        float score = __fmul_rn(q, k);
        #pragma unroll
        for (int delta = 16; delta > 0; delta /= 2)
            score = __fadd_rn(score, __shfl_down_sync(0xffffffffu, score, delta));
        if (d == 0) probabilities[key] = exponential(__fadd_rn(score,bias[(head * 64 + query) * 64 + key]));
    }
    __syncthreads();
    if (d == 0) {
        float halves[2];
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            float column[8], pair[4];
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                float sum = __fadd_rn(probabilities[half*8+c],probabilities[16+half*8+c]);
                sum = __fadd_rn(sum,probabilities[32+half*8+c]);
                column[c] = __fadd_rn(sum,probabilities[48+half*8+c]);
            }
            #pragma unroll
            for (int p = 0; p < 4; ++p) pair[p] = __fadd_rn(column[2*p],column[2*p+1]);
            halves[half] = __fadd_rn(__fadd_rn(pair[0],pair[1]),__fadd_rn(pair[2],pair[3]));
        }
        const float total = __fadd_rn(halves[0],halves[1]);
        const float denominator = isnan(total) ? total : fmaxf(total,6.103515625e-05f);
        const float inverse = __fdiv_rn(1.0f,denominator);
        for (int key = 0; key < 64; ++key)
            probabilities[key] = quant8(half_value(__fmul_rn(probabilities[key],inverse)));
    }
    __syncthreads();
    float accumulator = 0.0f;
    for (int key = 0; key < 64; ++key) {
        const float value = qkv[(window * 64 + key) * (3 * channels) + head * 96 + 64 + d];
        accumulator = __fmaf_rn(probabilities[key],value,accumulator);
    }
    output[(window * 64 + query) * channels + head * 32 + d] = quant8(half_value(accumulator));
}

// Four query warps share one K/V tile for a window/head.  The source reduction
// order is retained inside each warp (the same shuffle-down tree and scalar
// key/PV loop as split_attention); only the read-only K/V loads are shared.
// This is opt-in because the exact end-to-end H200 timing must be measured
// before changing the default path.
__global__ void split_attention_fused4(const float* qkv, const float* bias, float* output) {
    const unsigned batch = blockIdx.x;
    const int warp = static_cast<int>(threadIdx.x >> 5);
    const int d = static_cast<int>(threadIdx.x & 31);
    const int query_base = warp * 16;
    const int head = static_cast<int>((batch / 1u) % heads);
    const size_t window = static_cast<size_t>(batch) / heads;
    __shared__ float key_tile[64][32];
    __shared__ float value_tile[64][32];
    __shared__ float probabilities[4][64];
    for (int index = static_cast<int>(threadIdx.x); index < 64 * 32; index += 128) {
        const int key = index / 32;
        const int dim = index % 32;
        const size_t base = (window * 64ull + static_cast<size_t>(key)) * (3 * channels) +
            static_cast<size_t>(head) * 96ull;
        key_tile[key][dim] = qkv[base + 32ull + static_cast<size_t>(dim)];
        value_tile[key][dim] = qkv[base + 64ull + static_cast<size_t>(dim)];
    }
    __syncthreads();
    for (int query = query_base; query < query_base + 16; ++query) {
        const size_t qbase = (window * 64ull + static_cast<size_t>(query)) * (3 * channels) +
            static_cast<size_t>(head) * 96ull;
        const float q = qkv[qbase + static_cast<size_t>(d)];
        for (int key = 0; key < 64; ++key) {
            float score = __fmul_rn(q, key_tile[key][d]);
            #pragma unroll
            for (int delta = 16; delta > 0; delta /= 2)
                score = __fadd_rn(score, __shfl_down_sync(0xffffffffu, score, delta));
            if (d == 0)
                probabilities[warp][key] = exponential(__fadd_rn(
                    score, bias[(head * 64 + query) * 64 + key]));
        }
        __syncthreads();
        if (d == 0) {
            float halves[2];
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                float column[8], pair[4];
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    float sum = __fadd_rn(probabilities[warp][half * 8 + c], probabilities[warp][16 + half * 8 + c]);
                    sum = __fadd_rn(sum, probabilities[warp][32 + half * 8 + c]);
                    column[c] = __fadd_rn(sum, probabilities[warp][48 + half * 8 + c]);
                }
                #pragma unroll
                for (int p = 0; p < 4; ++p) pair[p] = __fadd_rn(column[2 * p], column[2 * p + 1]);
                halves[half] = __fadd_rn(__fadd_rn(pair[0], pair[1]), __fadd_rn(pair[2], pair[3]));
            }
            const float total = __fadd_rn(halves[0], halves[1]);
            const float denominator = isnan(total) ? total : fmaxf(total, 6.103515625e-05f);
            const float inverse = __fdiv_rn(1.0f, denominator);
            for (int key = 0; key < 64; ++key)
                probabilities[warp][key] = quant8(half_value(__fmul_rn(probabilities[warp][key], inverse)));
        }
        __syncthreads();
        float accumulator = 0.0f;
        for (int key = 0; key < 64; ++key)
            accumulator = __fmaf_rn(probabilities[warp][key], value_tile[key][d], accumulator);
        output[(window * 64ull + static_cast<size_t>(query)) * channels +
               static_cast<size_t>(head * 32 + d)] = quant8(half_value(accumulator));
        __syncthreads();
    }
}

bool split_fused_attention_requested() {
    const char* value = std::getenv("RW_NR_SPLIT_FUSED_ATTN");
    return value && std::strcmp(value, "1") == 0;
}

Tensor projection(Context& context, Weights& weights, const LayerSpec& spec,
                  const Tensor& input, const Tensor* residual, int output_channels) {
    const auto shape = map_shape(input);
    if (shape.channels != channels) throw std::invalid_argument("split projection requires C512");
    const Parameter matrix = weights.e4m3(context,spec.block,spec.layer,"weight",{output_channels,channels});
    const Tensor product = linear_nchw(context,input,matrix);
    Parameter gain;
    if (residual) {
        const auto other = map_shape(*residual);
        if (other.channels != output_channels || other.height != shape.height || other.width != shape.width)
            throw std::invalid_argument("split projection residual shape mismatch");
        gain = weights.f16(context,spec.block,spec.layer,"skip_weight",{output_channels});
        context.validate_parameter(gain);
    }
    Tensor output = context.empty(product.shape);
    projection_residual<<<blocks(output.numel()),256,0,context.stream()>>>(product.data(),
        residual ? residual->data() : nullptr,gain.empty() ? nullptr : gain.data(),output.data(),output.numel(),shape.pixels);
    cuda_check(cudaGetLastError(),"split projection residual");
    return output;
}

// The split FFN is eight independent 64->256->64 blocks.  The source layout
// stores each block contiguously, so a strided-batched GEMM can execute all
// blocks without materialising eight group_input tensors or issuing one GEMM
// per group.  This keeps the exact row-major weight interpretation used by
// linear_nchw while reducing the launch count from 16 to 2 per FFN node.
Tensor grouped_linear_nchw(Context& context, const Tensor& input, const Parameter& weight,
                           int groups, int input_channels, int output_channels) {
    const auto shape = map_shape(input);
    if (groups <= 0 || input_channels <= 0 || output_channels <= 0 ||
        shape.channels != groups * input_channels ||
        weight.shape() != std::vector<int64_t>{groups * output_channels, input_channels})
        throw std::invalid_argument("split grouped linear shape mismatch");
    context.validate_parameter(weight);
    const int spatial = static_cast<int>(shape.pixels);
    const int n = output_channels, k = input_channels;
    auto output = context.empty({1, groups * output_channels, shape.height, shape.width});
    const float alpha = 1.0f, beta = 0.0f;
    const long long input_stride = static_cast<long long>(input_channels) * spatial;
    const long long weight_stride = static_cast<long long>(input_channels) * output_channels;
    const long long output_stride = static_cast<long long>(output_channels) * spatial;
    split_gemm_strided_batched(context,
        spatial, n, k, &alpha, input.data(), spatial, input_stride,
        weight.data(), k, weight_stride, &beta, output.data(), spatial,
        output_stride, groups, "split grouped linear GEMM");
    return output;
}

LayerOutput feed_forward(Context& context, Weights& weights, const LayerSpec& spec,const Tensor& input) {
    const auto shape = map_shape(input);
    if (shape.channels != channels) throw std::invalid_argument("split FFN requires C512");
    const std::string prepared_key=cache_key(spec,"prepared_ffwd_parameters_v2_batched");
    const std::array<int64_t,4> configuration={channels,8,64,256};
    const auto prepared=weights.find_prepared(context,prepared_key,configuration);
    if (prepared && prepared->tensors.size()!=3)
        throw std::invalid_argument("split prepared FFN parameter count mismatch");
    GroupedMatrices matrices;
    if (!prepared) matrices=grouped_matrices(weights,spec);
    const Parameter matrix=prepared ? prepared->tensors[0] :
        weights.derived(context,cache_key(spec,"ffwd/A"),{512,512},matrices.a);
    std::vector<float> grouped_q0, grouped_q2;
    Parameter expansion, contraction;
    if (prepared) {
        expansion=prepared->tensors[1];
        contraction=prepared->tensors[2];
    } else {
        grouped_q0.resize(size_t(8) * 256 * 64);
        grouped_q2.resize(size_t(8) * 64 * 256);
        for (int group=0; group<8; ++group) {
            std::copy(matrices.q0[group].begin(), matrices.q0[group].end(),
                grouped_q0.begin()+size_t(group)*256*64);
            std::copy(matrices.q2[group].begin(), matrices.q2[group].end(),
                grouped_q2.begin()+size_t(group)*64*256);
        }
        expansion=weights.derived(context,cache_key(spec,"ffwd/Q0_batched"),{8*256,64},grouped_q0);
        contraction=weights.derived(context,cache_key(spec,"ffwd/Q2_batched"),{8*64,256},grouped_q2);
    }
    if (!prepared) {
        const std::array<Parameter,3> batched={matrix,expansion,contraction};
        weights.remember_prepared(context,prepared_key,configuration,batched);
    }
    const Tensor reduced = store(context,linear_nchw(context,input,matrix));
    const Tensor hidden_raw = grouped_linear_nchw(context,reduced,expansion,8,64,256);
    Tensor hidden = context.empty(hidden_raw.shape);
    split_activation<<<blocks(hidden.numel()),256,0,context.stream()>>>(hidden_raw.data(),hidden.data(),hidden.numel());
    cuda_check(cudaGetLastError(),"split cubic FP16 activation");
    const Tensor grouped_output = grouped_linear_nchw(context,hidden,contraction,8,256,64);
    const Tensor result = store(context,grouped_output);
    return {result,input};
}

LayerOutput attention(Context& context, Weights& weights, const LayerSpec& spec,const Tensor& input) {
    const auto shape = map_shape(input);
    if (shape.channels != channels || shape.height > INT_MAX-11 || shape.width > INT_MAX-11 ||
        (spec.shift_rows != 0 && spec.shift_rows != -1) ||
        (spec.shift_columns != 0 && spec.shift_columns != -1)) throw std::invalid_argument("split window input/shift invalid");
    const int top = -spec.shift_rows*4, left = -spec.shift_columns*4;
    const int window_rows = (shape.height + top + 7)/8, window_columns = (shape.width + left + 7)/8;
    const size_t window_count = size_t(window_rows)*window_columns;
    if (window_count > UINT_MAX / (64u * heads)) throw std::invalid_argument("split attention launch too large");
    Tensor tokens = context.empty({static_cast<int64_t>(window_count),64,channels});
    to_windows<<<blocks(tokens.numel()),256,0,context.stream()>>>(input.data(),tokens.data(),shape.height,shape.width,
        window_columns,top,left,tokens.numel());
    cuda_check(cudaGetLastError(),"split pack windows");
    const Parameter matrix = weights.e4m3(context,spec.block,spec.layer,"qkv",{3*channels,channels});
    const Tensor raw = linear_last(context,tokens,matrix);
    const std::string prepared_key=cache_key(spec,"prepared_attention_parameters_v1");
    const std::array<int64_t,3> configuration={channels,heads,64};
    Parameter scale,bias;
    if (const auto prepared=weights.find_prepared(context,prepared_key,configuration)) {
        if (prepared->tensors.size()!=2) throw std::invalid_argument("split prepared attention parameter count mismatch");
        scale=prepared->tensors[0]; bias=prepared->tensors[1];
    } else {
        const auto scales = weights.host_f32(spec.block,spec.layer,"tail");
        if (scales.size() != heads) throw std::invalid_argument("split head scale count mismatch");
        scale=weights.derived(context,cache_key(spec,"attention/scales"),{heads},scales);
        const auto packed_bias = weights.host_f16(spec.block,spec.layer,"attn_pos_bias");
        if (packed_bias.size() != size_t(heads)*4096) throw std::invalid_argument("split attention bias count mismatch");
        std::vector<float> unpacked_bias(packed_bias.size());
        for (int head = 0; head < heads; ++head) for (int i = 0; i < 64; ++i) for (int j = 0; j < 64; ++j) {
            const int lane=4*((i%16)%8)+((j%16)%8)/2;
            const int slot=(j%2)|(((i%16)/8)<<1)|(((j%16)/8)<<2);
            const int source=(4*(i/16)+j/16)*256+lane*8+slot;
            unpacked_bias[(head*64+i)*64+j]=packed_bias[head*4096+source];
        }
        bias=weights.derived(context,cache_key(spec,"attention/bias"),{heads,64,64},unpacked_bias);
        const std::array<Parameter,2> prepared_tensors={scale,bias};
        weights.remember_prepared(context,prepared_key,configuration,prepared_tensors);
    }
    context.validate_parameter(scale);
    context.validate_parameter(bias);
    Tensor normalized;
    Tensor values = context.empty(tokens.shape);
    const char* tensor_core=std::getenv("RW_NR_SPLIT_TENSOR_CORE");
#if RW_NR_HOPPER_WGMMA_ENABLED
    if(tensor_core && std::strcmp(tensor_core,"1")==0)
        split_attention_hopper<<<static_cast<unsigned>(window_count*heads),128,0,context.stream()>>>(
            raw.data(),scale.data(),bias.data(),values.data());
    else
#endif
    {
    normalized=context.empty(raw.shape);
    normalize_qkv<<<blocks(window_count*64*heads),256,0,context.stream()>>>(raw.data(),scale.data(),normalized.data(),window_count*64*heads);
    cuda_check(cudaGetLastError(),"split normalize QKV");
    if (split_fused_attention_requested()) {
        split_attention_fused4<<<static_cast<unsigned>(window_count * heads), 128, 0, context.stream()>>>(
            normalized.data(), bias.data(), values.data());
    } else {
        split_attention<<<static_cast<unsigned>(window_count*64*heads),32,0,context.stream()>>>(
            normalized.data(),bias.data(),values.data());
    }
    }
    cuda_check(cudaGetLastError(),"split source-specific window attention");
    Tensor output = context.empty(input.shape);
    from_windows<<<blocks(output.numel()),256,0,context.stream()>>>(values.data(),output.data(),shape.height,shape.width,
        window_columns,top,left,output.numel());
    cuda_check(cudaGetLastError(),"split unpack windows");
    return {output,{}};
}
} // namespace

LayerOutput run_split(Context& context, Weights& weights, const LayerSpec& spec,
                      const LayerOutput& input,const LayerOutput* skip) {
    if (!context.owns(input.main) || (!input.skip.empty() && !context.owns(input.skip)) ||
        (skip && ((!skip->main.empty() && !context.owns(skip->main)) ||
                  (!skip->skip.empty() && !context.owns(skip->skip)))))
        throw std::invalid_argument("split tensor belongs to another native context");
    if (spec.kind == "SplitFfwd") return feed_forward(context,weights,spec,input.main);
    if (spec.kind == "SplitQKVAttn") return attention(context,weights,spec,input.main);
    if (spec.kind == "SplitFfwdProj") {
        const Tensor* residual = !input.skip.empty() ? &input.skip : (skip ? &skip->main : nullptr);
        if (!residual || residual->empty()) throw std::invalid_argument("split feed-forward projection missing residual");
        return {round8(context,projection(context,weights,spec,input.main,residual,channels)),{}};
    }
    if (spec.kind == "SplitProj" || spec.kind == "SplitProjPool") {
        if (!skip || skip->main.empty()) throw std::invalid_argument("split attention projection missing residual");
        const Tensor projected = projection(context,weights,spec,input.main,&skip->main,channels);
        const Tensor published = round8(context,projected);
        if (spec.kind == "SplitProj") return {published,{}};
        const auto shape = map_shape(projected);
        if (shape.height%2 || shape.width%2) throw std::invalid_argument("split pool requires even input dimensions");
        const Tensor pooled = pool2x2(context,projected);
        const auto half = map_shape(pooled);
        const int padded_h=(half.height+3)/4*4, padded_w=(half.width+3)/4*4;
        if (padded_h==half.height && padded_w==half.width) return {published,pooled};
        Tensor padded = context.empty({1,channels,padded_h,padded_w});
        pad_map<<<blocks(padded.numel()),256,0,context.stream()>>>(pooled.data(),padded.data(),channels,
            half.height,half.width,padded_h,padded_w,padded.numel());
        cuda_check(cudaGetLastError(),"split zero-pad pooled output");
        return {published,padded};
    }
    if (spec.kind == "SplitFinalHead") {
        if (skip && skip->skip.empty()) throw std::invalid_argument("split final head missing pooled tuple member");
        const Tensor& pooled = skip ? skip->skip : input.main;
        return {round8(context,projection(context,weights,spec,pooled,nullptr,1024)),{}};
    }
    throw std::invalid_argument("unknown split layer kind: "+spec.kind);
}
} // namespace rw::native_nr
