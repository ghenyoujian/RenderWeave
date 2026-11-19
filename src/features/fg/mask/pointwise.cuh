#pragma once

#include "fp16_mma.cuh"

namespace rw::fg::mask {

struct PointwiseExtent {
    unsigned width;
    unsigned height;
};

static __global__ void pointwise_fp16(const uint32_t* packed, const uint32_t* input, uint32_t* output,
                                      size_t pixel_count,
                                      unsigned input_channels, unsigned output_channels, bool leaky) {
    constexpr unsigned warp_size = 32;
    const unsigned lane = threadIdx.x % warp_size;
    const unsigned row = lane / 4;
    const unsigned pair = lane % 4;
    const unsigned group = blockIdx.y;
    const size_t first_pixel = (size_t(blockIdx.x) * blockDim.x / warp_size + threadIdx.x / warp_size) * 16;
    if (first_pixel >= pixel_count) return;
    const auto p0 = first_pixel + row;
    const auto p1 = p0 + 8;
    const unsigned input_pairs = input_channels / 2;
    const unsigned output_pairs = output_channels / 2;
    // The carrier stores a 128-half fragment for every eight output and
    // sixteen input channels. Each lane reads its two adjacent packed words.
    const auto weights = packed + size_t(group) * input_channels * 4 + row * 8 + pair * 2;
    const auto bias = packed + size_t(input_channels) * output_channels + size_t(group) * 64 + pair;
    uint32_t low = *bias;
    uint32_t high = low;
    for (unsigned k = 0; k < input_pairs; k += 8) {
        const auto k_pair = k + pair;
        const uint32_t a0 = p0 < pixel_count ? input[p0 * input_pairs + k_pair] : 0;
        const uint32_t a1 = p1 < pixel_count ? input[p1 * input_pairs + k_pair] : 0;
        const uint32_t a2 = p0 < pixel_count ? input[p0 * input_pairs + k_pair + 4] : 0;
        const uint32_t a3 = p1 < pixel_count ? input[p1 * input_pairs + k_pair + 4] : 0;
        const auto fragment = weights + k * 8;
        const uint32_t b0 = fragment[0];
        const uint32_t b1 = fragment[1];
        accumulate_f16(low, high, a0, a1, a2, a3, b0, b1);
    }
    if (leaky) {
        low = mask_leaky(low);
        high = mask_leaky(high);
    }
    const unsigned out_pair = group * 4 + pair;
    if (p0 < pixel_count) output[p0 * output_pairs + out_pair] = low;
    if (p1 < pixel_count) output[p1 * output_pairs + out_pair] = high;
}

// The host binds identical positive extents for the input and output. NHWC
// channels are multiples of 16 and 8 respectively. Keep the original packed
// parameter resource intact.
inline cudaError_t launch_pointwise(const uint32_t* packed, const uint32_t* input, uint32_t* output,
                                    PointwiseExtent extent, unsigned input_channels,
                                    unsigned output_channels, bool leaky,
                                    cudaStream_t stream) {
    constexpr unsigned threads = 128;
    constexpr unsigned pixels_per_block = threads / 32 * 16;
    const auto pixels = size_t(extent.width) * extent.height;
    const dim3 grid((pixels - 1) / pixels_per_block + 1, output_channels / 8);
    pointwise_fp16<<<grid, threads, 0, stream>>>(packed, input, output, pixels,
                                                input_channels, output_channels, leaky);
    return cudaGetLastError();
}

}
