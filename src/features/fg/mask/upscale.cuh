#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rw::fg::mask {

struct UpscaleExtent {
    unsigned height;
    unsigned width;
};

struct alignas(16) UpscaleChannels {
    __half2 pair[4];
};

struct UpscaleAxis {
    unsigned lower;
    unsigned upper;
    __half2 fraction;
    __half2 complement;
};

__device__ inline UpscaleAxis upscale_axis(unsigned position, unsigned input, unsigned valid_output) {
    float coordinate = 0.0f;
    if (valid_output > 1) {
        const float denominator = __uint2float_rn(valid_output - 1);
        float inverse;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(denominator));
        const float scale = __fmul_rn(__uint2float_rn(input - 1), inverse);
        coordinate = __fmul_rn(__uint2float_rn(position), scale);
    }
    unsigned lower = __float2uint_rz(coordinate);
    unsigned upper = __float2uint_rz(__fadd_rn(coordinate, 1.0f));
    if (upper >= input) lower = upper = input - 1;
    const float lower_float = __uint2float_rn(lower);
    const float fraction = __fadd_rn(coordinate, -lower_float);
    const float complement = __fadd_rn(__fadd_rn(-coordinate, lower_float), 1.0f);
    return {lower, upper, __float2half2_rn(fraction), __float2half2_rn(complement)};
}

__device__ inline __half2 upscale_mix(__half2 lower, __half2 upper, UpscaleAxis axis) {
    const auto weighted_upper = __hmul2_rn(axis.fraction, upper);
    return __hfma2(axis.complement, lower, weighted_upper);
}

template<unsigned Channels>
static __global__ void upscale_align_corners(const UpscaleChannels* input, UpscaleChannels* output,
                                             UpscaleExtent source, UpscaleExtent valid, UpscaleExtent destination) {
    constexpr unsigned groups = Channels / 8;
    const unsigned group = threadIdx.x;
    const unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned y = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= destination.width || y >= destination.height) return;
    const auto out = (size_t(y) * destination.width + x) * groups + group;
    UpscaleChannels result;
    if (x < valid.width && y < valid.height) {
        const auto sx = upscale_axis(x, source.width, valid.width);
        const auto sy = upscale_axis(y, source.height, valid.height);
        const auto upper_row = size_t(sy.lower) * source.width;
        const auto lower_row = size_t(sy.upper) * source.width;
        const auto top_left = input[(upper_row + sx.lower) * groups + group];
        const auto top_right = input[(upper_row + sx.upper) * groups + group];
        const auto bottom_left = input[(lower_row + sx.lower) * groups + group];
        const auto bottom_right = input[(lower_row + sx.upper) * groups + group];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const auto top = upscale_mix(top_left.pair[i], top_right.pair[i], sx);
            const auto bottom = upscale_mix(bottom_left.pair[i], bottom_right.pair[i], sx);
            result.pair[i] = upscale_mix(top, bottom, sy);
        }
    } else {
#pragma unroll
        for (auto& pair : result.pair) pair = __float2half2_rn(0.0f);
    }
    output[out] = result;
}

template<unsigned Channels>
inline cudaError_t launch_upscale(const UpscaleChannels* input, UpscaleChannels* output, UpscaleExtent source,
                                  UpscaleExtent valid, UpscaleExtent destination, cudaStream_t stream) {
    static_assert(Channels == 32 || Channels == 64 || Channels == 128 || Channels == 256);
    const dim3 block(Channels / 8, 5, 5);
    const dim3 grid(1, (destination.width - 1) / block.y + 1, (destination.height - 1) / block.z + 1);
    upscale_align_corners<Channels><<<grid, block, 0, stream>>>(input, output, source, valid, destination);
    return cudaGetLastError();
}

}
