#include "native_nr.hpp"
#include "numpy_math_device.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace rw::native_nr {
namespace {
__device__ float hround(float value) { return __half2float(__float2half_rn(value)); }
__device__ float clamp_nan(float value,float low,float high) {
    return isnan(value) ? value : fminf(fmaxf(value,low),high);
}
#include "temporal_history.cuh"

// The 16->32 pre-lift is a low-arithmetic-intensity projection over a large
// image.  This opt-in path keeps the source's half input/weight publication and
// writes the narrow result directly, avoiding a separate GEMM output traversal.
__global__ void pre_lift_half_narrow_kernel(const float* input, const float* weight,
    float* output, size_t spatial) {
    const int lane = static_cast<int>(threadIdx.x & 31);
    const int warp = static_cast<int>(threadIdx.x >> 5);
    const int channel = static_cast<int>(blockIdx.y) * 8 + warp;
    const size_t pixel = static_cast<size_t>(blockIdx.x) * 32ull + static_cast<size_t>(lane);
    if (channel >= 32 || pixel >= spatial) return;
    float accumulator = 0.0f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const float x = __half2float(__float2half_rn(input[static_cast<size_t>(k) * spatial + pixel]));
        const float w = __half2float(__float2half_rn(weight[static_cast<size_t>(channel) * 16ull + k]));
        accumulator = __fmaf_rn(w, x, accumulator);
    }
    output[static_cast<size_t>(channel) * spatial + pixel] = __half2float(__float2half_rn(accumulator));
}

bool pre_lift_custom_requested() {
    const char* value = std::getenv("RW_NR_PRE_LIFT_CUSTOM");
    return value && std::strcmp(value, "1") == 0;
}

__device__ uint32_t pcg(uint32_t value) {
    value = (value >> ((value >> 28u) + 4u)) ^ value;
    return value * UINT32_C(0x108ef2d9);
}
__device__ float unit(uint32_t stream) {
    const uint32_t value = pcg(stream);
    return __fmul_rn(__uint2float_rn(((value >> 30u) ^ (value >> 8u)) + 1u),0x1p-24f);
}
__global__ void gaussian_noise_kernel(float* result,int height,int width,uint32_t seed) {
    const size_t pixels = static_cast<size_t>(height) * width;
    const size_t pixel=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pixel>=pixels) return;
    const auto x=static_cast<uint32_t>(pixel%width),y=static_cast<uint32_t>(pixel/width);
    const uint32_t base=(x*UINT32_C(0x8da6b343))^(seed*UINT32_C(0x9e3779b9))^
        (y*UINT32_C(0xd8163841))^UINT32_C(0x243f6a88);
    const uint32_t mixed=pcg(base),hash=(mixed>>22u)^mixed;
    const float ua=unit(hash*UINT32_C(0x2c9277b5)+UINT32_C(0xac564b05));
    const float ub=unit(hash*UINT32_C(0xfa6dc5f9)+UINT32_C(0x4712a88e));
    const float uc=unit(hash*UINT32_C(0xcaa5b80d)+UINT32_C(0x21dd796b));
    const float ud=unit(hash*UINT32_C(0x83232c31)+UINT32_C(0x3463e0ac));
    namespace npy=numpy_math_device;
    const float radius_a=npy::numpy_sqrtf(__fmul_rn(-2.0f,npy::numpy_logf(ua)));
    const float radius_c=npy::numpy_sqrtf(__fmul_rn(-2.0f,npy::numpy_logf(uc)));
    const float angle_a=__fmul_rn(ub,6.28318530718f),angle_c=__fmul_rn(ud,6.28318530718f);
    result[pixel]=__fmul_rn(radius_a,npy::numpy_cosf(angle_a));
    result[pixels+pixel]=__fmul_rn(radius_a,npy::numpy_sinf(angle_a));
    result[2*pixels+pixel]=__fmul_rn(radius_c,npy::numpy_cosf(angle_c));
}

__device__ float sample_previous(const float* previous, size_t plane, int width, int height,
    float x, float y, int channel) {
    x = fminf(fmaxf(x, 0.0f), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0f), static_cast<float>(height - 1));
    const int x0 = static_cast<int>(floorf(x)), y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
    const float tx = x - static_cast<float>(x0), ty = y - static_cast<float>(y0);
    const auto at = [&](int xx, int yy) {
        return previous[static_cast<size_t>(channel) * plane + static_cast<size_t>(yy) * width + xx];
    };
    const float a = __fadd_rn(at(x0, y0), __fmul_rn(tx, __fsub_rn(at(x1, y0), at(x0, y0))));
    const float b = __fadd_rn(at(x0, y1), __fmul_rn(tx, __fsub_rn(at(x1, y1), at(x0, y1))));
    return __fadd_rn(a, __fmul_rn(ty, __fsub_rn(b, a)));
}

__global__ void features_kernel(const float* image, const float* previous, const float* motion,
    const float* motion_validity,
    const float* reprojected, const float* noise, float* output, size_t pixels, int width, int height,
    float style, float tone, float structure, float skin, float other,
    bool synthetic_noise, bool previous_color, bool motion_warp) {
    const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    for (int channel = 0; channel < 3; ++channel) {
        const float source = synthetic_noise
            ? noise[static_cast<size_t>(channel) * pixels + pixel]
            : image[static_cast<size_t>(channel) * pixels + pixel];
        output[static_cast<size_t>(channel) * pixels + pixel] = hround(source);
        const float centered = hround(__fmul_rn(hround(__fsub_rn(hround(image[static_cast<size_t>(channel) * pixels + pixel]),0.5f)),0.125f));
        const int px = static_cast<int>(pixel % width), py = static_cast<int>(pixel / width);
        const float dx = motion_warp ? motion[pixel] : 0.0f;
        const float dy = motion_warp ? motion[pixels + pixel] : 0.0f;
        const bool valid_motion = !motion_warp || !motion_validity || motion_validity[pixel] >= 0.5f;
        const size_t offset = static_cast<size_t>(channel) * pixels + pixel;
        const float previous_sample = reprojected
            ? (reprojected[3 * pixels + pixel] != 0.f ? reprojected[offset] : image[offset])
            : (previous_color && motion_warp && valid_motion)
            ? sample_previous(previous, pixels, width, height, static_cast<float>(px) + dx,
                static_cast<float>(py) + dy, channel)
            : (previous_color ? (motion_warp && !valid_motion
                    ? image[static_cast<size_t>(channel) * pixels + pixel]
                    : previous[static_cast<size_t>(channel) * pixels + pixel])
                              : image[static_cast<size_t>(channel) * pixels + pixel]);
        const float previous_centered = previous_color
            ? hround(__fmul_rn(hround(__fsub_rn(hround(previous_sample),0.5f)),0.125f))
            : centered;
        output[static_cast<size_t>(4 + channel) * pixels + pixel] = centered;
        output[static_cast<size_t>(7 + channel) * pixels + pixel] = previous_centered;
    }
    output[3 * pixels + pixel] = 1.0f;
    output[10 * pixels + pixel] = hround(style);
    output[11 * pixels + pixel] = hround(tone);
    output[12 * pixels + pixel] = hround(structure);
    output[13 * pixels + pixel] = hround(skin);
    output[14 * pixels + pixel] = hround(other);
    output[15 * pixels + pixel] = 0.0f;
}

template<bool synthetic_noise>
__global__ void stateless_features_kernel(const float* image, const float* noise,
    float* output, size_t pixels, float style, float tone, float structure,
    float skin, float other) {
    const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    #pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        const size_t offset = static_cast<size_t>(channel) * pixels + pixel;
        const float source = synthetic_noise ? noise[offset] : image[offset];
        output[offset] = hround(source);
        const float centered = hround(__fmul_rn(
            hround(__fsub_rn(hround(image[offset]), 0.5f)), 0.125f));
        output[static_cast<size_t>(4 + channel) * pixels + pixel] = centered;
        output[static_cast<size_t>(7 + channel) * pixels + pixel] = centered;
    }
    output[3 * pixels + pixel] = 1.0f;
    output[10 * pixels + pixel] = hround(style);
    output[11 * pixels + pixel] = hround(tone);
    output[12 * pixels + pixel] = hround(structure);
    output[13 * pixels + pixel] = hround(skin);
    output[14 * pixels + pixel] = hround(other);
    output[15 * pixels + pixel] = 0.0f;
}

__global__ void post_mix_kernel(const float* input, const float* skip, const float* main_gain,
    const float* skip_gain, float* output, int low_h, int low_w, int height, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pixels = static_cast<size_t>(height) * width;
    if (index >= pixels * 32) return;
    const int channel = static_cast<int>(index / pixels);
    const size_t pixel = index % pixels;
    const int y = static_cast<int>(pixel / width), x = static_cast<int>(pixel % width);
    const int iy = min(y / 2, low_h - 1), ix = min(x / 2, low_w - 1);
    const float up = input[(static_cast<size_t>(channel) * low_h + iy) * low_w + ix];
    const float main = hround(__fmul_rn(up, main_gain[channel]));
    // All operands here are f16-exact; native half FMA provides the single
    // rounding represented by the source's exact double product plus add.
    output[index] = __half2float(__hfma(__float2half_rn(skip[index]),
        __float2half_rn(skip_gain[channel]), __float2half_rn(main)));
}

__global__ void post_mix_pixel_kernel(const float* input, const float* skip,
    const float* main_gain, const float* skip_gain, float* output,
    int low_h, int low_w, int height, int width) {
    const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pixels = static_cast<size_t>(height) * width;
    if (pixel >= pixels) return;
    const int y = static_cast<int>(pixel / static_cast<size_t>(width));
    const int x = static_cast<int>(pixel % static_cast<size_t>(width));
    const int iy = min(y / 2, low_h - 1), ix = min(x / 2, low_w - 1);
    #pragma unroll
    for (int channel = 0; channel < 32; ++channel) {
        const size_t index = static_cast<size_t>(channel) * pixels + pixel;
        const float up = input[(static_cast<size_t>(channel) * low_h + iy) * low_w + ix];
        const float main = hround(__fmul_rn(up, main_gain[channel]));
        output[index] = __half2float(__hfma(__float2half_rn(skip[index]),
            __float2half_rn(skip_gain[channel]), __float2half_rn(main)));
    }
}

bool pixel_post_mix_requested() {
    const char* value = std::getenv("RW_NR_PIXEL_POST_MIX");
    return !value || std::strcmp(value, "0") != 0;
}

__global__ void compose_kernel(const float* image, const float* residual, float* output,
    size_t pixels, float intensity) {
    const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    float original[3], mixed[3];
    for (int channel = 0; channel < 3; ++channel) {
        const size_t index = static_cast<size_t>(channel) * pixels + pixel;
        original[channel] = image[index];
        const float neural = clamp_nan(__fadd_rn(original[channel],
            __fmul_rn(residual[index],post_residual_scale)),0.0f,1.0f);
        mixed[channel] = clamp_nan(__fadd_rn(original[channel],
            __fmul_rn(fminf(intensity,1.0f), __fsub_rn(neural,original[channel]))),0.0f,1.0f);
    }
    if (intensity > 1.0f) {
        const float luminance = __fadd_rn(__fadd_rn(__fmul_rn(mixed[0],0.2126f),
            __fmul_rn(mixed[1],0.7152f)),__fmul_rn(mixed[2],0.0722f));
        const float original_luminance = __fadd_rn(__fadd_rn(__fmul_rn(original[0],0.2126f),
            __fmul_rn(original[1],0.7152f)),__fmul_rn(original[2],0.0722f));
        const float ratio = __fdiv_rn(__fadd_rn(luminance,0x1p-9f),__fadd_rn(original_luminance,0x1p-9f));
        const float guarded = isnan(ratio) ? ratio : fmaxf(ratio,1.0e-6f);
        const float amplification = clamp_nan(powf(guarded,intensity),0.5f,2.0f);
        for (int channel = 0; channel < 3; ++channel)
            mixed[channel] = clamp_nan(__fdiv_rn(__fmul_rn(mixed[channel],amplification),guarded),0.0f,1.0f);
    }
    for (int channel = 0; channel < 3; ++channel)
        output[static_cast<size_t>(channel) * pixels + pixel] = mixed[channel];
}

void validate_image(const Tensor& image) {
    image.numel();
    if (image.shape.size() != 4 || image.shape[0] != 1 || image.shape[1] != 3 ||
        image.shape[2] <= 0 || image.shape[3] <= 0 || image.shape[2] > 16384 || image.shape[3] > 16384)
        throw std::invalid_argument("pre/post requires bounded contiguous [1,3,H,W]");
}
void validate_controls(const Controls& controls) {
    for (float value : {controls.intensity,controls.local_tone,controls.local_structure,controls.skin_structure})
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite NR control");
    if (controls.intensity < 0 || controls.intensity > 2 || controls.style < 0 || controls.style > 2 ||
        controls.local_tone < 0 || controls.local_tone > 2 || controls.local_structure < 0 ||
        controls.local_structure > 2 || controls.skin_structure < -1 || controls.skin_structure > 2)
        throw std::invalid_argument("NR controls outside pinned source contract");
}
}

Tensor gaussian_noise_f32(Context& context,int height,int width,uint32_t seed) {
    if(height<=0 || width<=0 || height>16384 || width>16384)
        throw std::invalid_argument("NR counter RNG requires bounded positive H/W <=16384");
    auto result=context.empty({1,3,height,width});
    const size_t pixels=static_cast<size_t>(height)*width;
    gaussian_noise_kernel<<<static_cast<unsigned>((pixels+255)/256),256,0,context.stream()>>>(
        result.data(),height,width,seed);
    cuda_check(cudaGetLastError(),"NR GPU counter RNG with source-specific RN math");
    return result;
}

Tensor input_features(Context& context, const Tensor& image, const Controls& controls,
                      const Tensor* previous_image, const Tensor* motion_image,
                      const Tensor* motion_validity_image, const Tensor* reprojected) {
    validate_image(image); validate_controls(controls);
    if(!context.owns(image)) throw std::invalid_argument("image belongs to another execution context");
    if (controls.previous_color != (previous_image != nullptr))
        throw std::invalid_argument("previous colour control and tensor presence differ");
    if (controls.motion_warp != (motion_image != nullptr))
        throw std::invalid_argument("motion warp control and tensor presence differ");
    if (controls.motion_validity != (motion_validity_image != nullptr))
        throw std::invalid_argument("motion validity control and tensor presence differ");
    if (previous_image) {
        validate_image(*previous_image);
        if (!context.owns(*previous_image) || previous_image->shape != image.shape)
            throw std::invalid_argument("previous colour must match current image geometry/context");
    }
    if (motion_image && (!context.owns(*motion_image) ||
        motion_image->shape != std::vector<int64_t>({1,2,image.shape[2],image.shape[3]})))
        throw std::invalid_argument("motion must match padded image geometry/context");
    if (motion_validity_image && (!context.owns(*motion_validity_image) ||
        motion_validity_image->shape != std::vector<int64_t>({1,1,image.shape[2],image.shape[3]})))
        throw std::invalid_argument("motion validity must match padded image geometry/context");
    const int height=static_cast<int>(image.shape[2]), width=static_cast<int>(image.shape[3]);
    Tensor noise;
    if (controls.synthetic_noise) noise=gaussian_noise_f32(context,height,width,controls.seed);
    auto result=context.empty({1,16,height,width});
    const size_t pixels=static_cast<size_t>(height)*width;
    const float structure=controls.auto_mask ? 1.0f : controls.local_structure;
    const float skin=controls.auto_mask ? (controls.skin_structure<0 ? controls.local_structure : controls.skin_structure) : -1.0f;
    const float other=controls.auto_mask ? controls.local_structure : -1.0f;
    if (!previous_image && !motion_image && !motion_validity_image) {
        if (controls.synthetic_noise)
            stateless_features_kernel<true><<<static_cast<unsigned>((pixels+255)/256),256,0,context.stream()>>>(
                image.data(),noise.data(),result.data(),pixels,controls.style/128.0f,
                controls.local_tone,structure,skin,other);
        else
            stateless_features_kernel<false><<<static_cast<unsigned>((pixels+255)/256),256,0,context.stream()>>>(
                image.data(),nullptr,result.data(),pixels,controls.style/128.0f,
                controls.local_tone,structure,skin,other);
    } else {
        features_kernel<<<static_cast<unsigned>((pixels+255)/256),256,0,context.stream()>>>(
            image.data(),previous_image ? previous_image->data() : nullptr,
            motion_image ? motion_image->data() : nullptr,
            motion_validity_image ? motion_validity_image->data() : nullptr,
            reprojected ? reprojected->data() : nullptr,
            controls.synthetic_noise ? noise.data() : nullptr,result.data(),pixels,width,height,
            controls.style/128.0f,controls.local_tone,structure,skin,other,
            controls.synthetic_noise,controls.previous_color,controls.motion_warp);
    }
    cuda_check(cudaGetLastError(),"NR feature lanes");
    return result;
}

LayerOutput run_pre(Context& context, Weights& weights, const LayerSpec& spec,
    const Tensor& image, const Controls& controls, const Tensor* explicit_features,
    const Tensor* previous_image, const Tensor* motion_image,
    const Tensor* motion_validity_image,
    PreBlockDiagnostics* diagnostics, const Tensor* reprojected) {
    validate_image(image); validate_controls(controls);
    if(!context.owns(image)) throw std::invalid_argument("pre-block image belongs to another context");
    if (spec.kind!="PreBlock") throw std::invalid_argument("wrong pre-block kind");
    const auto stored=weights.host_f16(spec.block,spec.layer,"input_lift");
    if (stored.size()!=512) throw std::runtime_error("input lift byte count differs from fixed source");
    std::vector<float> lift(512);
    for (int channel=0;channel<32;++channel) for(int k=0;k<16;++k) {
        const size_t offset=(channel/16)*256+32*(channel%8)+8*((k%8)/2)+4*((channel/8)%2)+(k%2)+2*(k/8);
        lift[static_cast<size_t>(channel)*16+k]=stored[offset];
    }
    const auto matrix=weights.derived(context,"pre/"+std::to_string(spec.block)+"/lift",{32,16},lift);
    const Tensor features=explicit_features ? *explicit_features : input_features(context,image,controls,
        previous_image,motion_image,motion_validity_image,reprojected);
    const std::vector<int64_t> expected={1,16,image.shape[2],image.shape[3]};
    if(features.shape!=expected) throw std::invalid_argument("explicit NR features have wrong dimensions");
    Tensor lifted;
    const size_t spatial = static_cast<size_t>(features.shape[2]) * static_cast<size_t>(features.shape[3]);
    if (pre_lift_custom_requested() && matrix.shape() == std::vector<int64_t>{32,16}) {
        lifted = context.empty({1,32,features.shape[2],features.shape[3]});
        const dim3 grid(static_cast<unsigned>((spatial + 31) / 32), 4, 1);
        pre_lift_half_narrow_kernel<<<grid,256,0,context.stream()>>>(
            features.data(),matrix.data(),lifted.data(),spatial);
        cuda_check(cudaGetLastError(),"NR custom pre-lift half projection");
    } else {
        lifted = narrow(context,linear_nchw(context,features,matrix));
    }
    if(diagnostics) { diagnostics->features=features; diagnostics->lifted=lifted; }
    const auto parameters=load_swin(context,weights,spec.block,spec.layer,32,1);
    const auto body=swin_body(context,parameters,lifted,spec.shift_rows,spec.shift_columns);
    return {pool2x2(context,body.accumulator),round8(context,body.wide)};
}

Tensor reproject_sdr_history(Context& context, const Tensor& previous, const Tensor& motion,
                             const Tensor* validity, int valid_width, int valid_height) {
    validate_image(previous);
    const int height=static_cast<int>(previous.shape[2]), width=static_cast<int>(previous.shape[3]);
    if (!context.owns(previous) || !context.owns(motion) || motion.shape!=std::vector<int64_t>{1,2,height,width} ||
        (validity && (!context.owns(*validity) || validity->shape!=std::vector<int64_t>{1,1,height,width})) ||
        valid_width<=0 || valid_width>width || valid_height<=0 || valid_height>height)
        throw std::invalid_argument("NR temporal history and guide geometry differ");
    auto result=context.empty({1,4,height,width});
    const size_t pixels=static_cast<size_t>(width)*height;
    temporal_history::reproject<<<static_cast<unsigned>((pixels+255)/256),256,0,context.stream()>>>(
        previous.data(),motion.data(),validity ? validity->data() : nullptr,result.data(),
        width,height,valid_width,valid_height);
    cuda_check(cudaGetLastError(),"NR five-tap history reprojection");
    return result;
}

LayerOutput run_post(Context& context, Weights& weights, const LayerSpec& spec,
    const Tensor& input, const LayerOutput* skip, const Tensor& image, const Controls& controls,
    PostBlockDiagnostics* diagnostics, const Tensor* reprojected, int temporal_width, int temporal_height) {
    validate_image(image); validate_controls(controls);
    if(spec.kind!="PostBlock" || !skip || skip->skip.empty()) throw std::invalid_argument("post-block missing pre-block skip");
    const auto& original_skip=skip->skip;
    input.numel(); original_skip.numel();
    if(!context.owns(image) || !context.owns(input) || !context.owns(original_skip))
        throw std::invalid_argument("post-block operands belong to another execution context");
    if(input.shape.size()!=4 || input.shape[0]!=1 || input.shape[1]!=32 ||
        original_skip.shape!=std::vector<int64_t>({1,32,image.shape[2],image.shape[3]}))
        throw std::invalid_argument("post-block channel/geometry mismatch");
    const auto main_gain=weights.f16(context,spec.block,spec.layer,"main_gain",{32});
    const auto skip_gain=weights.f16(context,spec.block,spec.layer,"skip_gain",{32});
    context.validate_parameter(main_gain);
    context.validate_parameter(skip_gain);
    auto blend=context.empty(original_skip.shape);
    const size_t elements=blend.numel();
    if (pixel_post_mix_requested())
        post_mix_pixel_kernel<<<static_cast<unsigned>((static_cast<size_t>(image.shape[2]) * image.shape[3] + 255) / 256),256,0,context.stream()>>>(
            input.data(),original_skip.data(),main_gain.data(),skip_gain.data(),blend.data(),
            static_cast<int>(input.shape[2]),static_cast<int>(input.shape[3]),
            static_cast<int>(image.shape[2]),static_cast<int>(image.shape[3]));
    else
        post_mix_kernel<<<static_cast<unsigned>((elements+255)/256),256,0,context.stream()>>>(
            input.data(),original_skip.data(),main_gain.data(),skip_gain.data(),blend.data(),
            static_cast<int>(input.shape[2]),static_cast<int>(input.shape[3]),
            static_cast<int>(image.shape[2]),static_cast<int>(image.shape[3]));
    cuda_check(cudaGetLastError(),"NR post-block nearest/skip mix");
    auto stored=weights.host_f16(spec.block,spec.layer,"out_project");
    if(stored.size()!=16*32) throw std::runtime_error("post-block output projection byte count changed");
    const bool temporal=temporal_width>0;
    const int heads=temporal ? 4 : 3;
    stored.resize(heads*32);
    const auto project=weights.derived(context,"post/"+std::to_string(spec.block)+
        (temporal ? "/rgba_project" : "/rgb_project"),{heads,32},stored);
    const auto parameters=load_swin(context,weights,spec.block,spec.layer,32,1);
    const auto body=swin_body(context,parameters,blend,spec.shift_rows,spec.shift_columns);
    Tensor residual;
    if(weights.network()=="ht") {
        // The original source's 32MiB workspace selects a different reduction
        // for this RGB projection. Keep all earlier GEMMs on their original
        // handle/workspace; this separately owned slot handle is prepared
        // before any full-graph submission, never allocated on this hot path.
        if(body.wide.shape.size()!=4 || body.wide.shape[0]!=1 || body.wide.shape[1]!=32 ||
            project.shape()!=std::vector<int64_t>{heads,32} || !context.owns(body.wide) ||
            body.wide.shape[2]>std::numeric_limits<int>::max()/body.wide.shape[3])
            throw std::invalid_argument("fixed HT final RGB projection shape/owner mismatch");
        context.validate_parameter(project);
        const int spatial=static_cast<int>(body.wide.shape[2]*body.wide.shape[3]);
        residual=context.empty({1,heads,body.wide.shape[2],body.wide.shape[3]});
        const float alpha=1.0f,beta=0.0f;
        blas_check(cublasSgemm(context.post_projection_blas(),CUBLAS_OP_N,CUBLAS_OP_N,spatial,heads,32,
            &alpha,body.wide.data(),spatial,project.data(),32,&beta,residual.data(),spatial),
            "HT final RGB Sgemm with persistent slot32MiB workspace");
    } else residual=linear_nchw(context,body.wide,project);
    if(diagnostics) { diagnostics->blend=blend; diagnostics->wide=body.wide; diagnostics->residual=residual; }
    auto output=context.empty(image.shape);
    const size_t pixels=static_cast<size_t>(image.shape[2])*image.shape[3];
    if (temporal) {
        const auto blend_scale=weights.host_f16(spec.block,spec.layer,"blend_scale");
        const float scale=std::isfinite(blend_scale.at(0)) ? std::clamp(blend_scale[0],0.f,1.f) : 0.f;
        auto history=context.empty({1,temporal_height,temporal_width,3});
        temporal_history::compose<<<static_cast<unsigned>((pixels+255)/256),256,0,context.stream()>>>(
            image.data(),residual.data(),reprojected ? reprojected->data() : nullptr,
            output.data(),history.data(),scale,static_cast<int>(image.shape[3]),
            static_cast<int>(image.shape[2]),temporal_width,temporal_height);
        cuda_check(cudaGetLastError(),"NR four-head SDR history composition");
        return {output,{},history};
    }
    compose_kernel<<<static_cast<unsigned>((pixels+255)/256),256,0,context.stream()>>>(
        image.data(),residual.data(),output.data(),pixels,controls.intensity);
    cuda_check(cudaGetLastError(),"NR source RGB-only composition");
    return {output,{}};
}
} // namespace rw::native_nr
