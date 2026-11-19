#pragma once

#include "fp16_mma.cuh"

namespace rw::fg::mask {

struct SpatialExtent {
    unsigned width;
    unsigned height;
};

enum class SpatialAccumulation {
    channel_then_column,
    column_then_channel,
};

__device__ inline const uint32_t* spatial_pixel(const uint32_t* input, SpatialExtent extent,
                                                unsigned pairs, int x, int y) {
    if (unsigned(x) >= extent.width || unsigned(y) >= extent.height) return nullptr;
    return input + (size_t(y) * extent.width + unsigned(x)) * pairs;
}

template<SpatialAccumulation Order>
static __global__ void spatial_fp16(const uint32_t* packed, const uint32_t* input, uint32_t* output,
                                    SpatialExtent extent, unsigned input_channels, unsigned output_channels) {
    constexpr unsigned warp_size = 32;
    const unsigned lane = threadIdx.x % warp_size;
    const unsigned row = lane / 4;
    const unsigned pair = lane % 4;
    const unsigned group = blockIdx.y;
    const size_t pixels = size_t(extent.width) * extent.height;
    const size_t first = (size_t(blockIdx.x) * blockDim.x / warp_size + threadIdx.x / warp_size) * 16;
    if (first >= pixels) return;
    const size_t p0 = first + row;
    const size_t p1 = p0 + 8;
    const int x0 = p0 % extent.width;
    const int y0 = p0 / extent.width;
    const int x1 = p1 % extent.width;
    const int y1 = p1 / extent.width;

    const unsigned input_pairs = input_channels / 2;
    const unsigned output_pairs = output_channels / 2;
    const unsigned blocks = input_channels / 16;
    const size_t inner_channels = size_t(input_channels) * 9;
    const auto weights = packed + size_t(group) * inner_channels * 4 + row * 8 + pair * 2;
    const auto bias = packed + inner_channels * output_channels + size_t(group) * 64 + pair;
    uint32_t low = *bias;
    uint32_t high = low;
    for (unsigned i = 0; i < blocks * 3; ++i) {
        const unsigned block = Order == SpatialAccumulation::channel_then_column ? i / 3 : i % blocks;
        const unsigned column = Order == SpatialAccumulation::channel_then_column ? i % 3 : i / blocks;
        const unsigned k = block * 8 + pair;
        for (unsigned tap_row = 0; tap_row < 3; ++tap_row) {
            const auto tap = tap_row * 3 + column;
            const auto a = spatial_pixel(input, extent, input_pairs, x0 + int(column) - 1, y0 + int(tap_row) - 1);
            const auto b = spatial_pixel(input, extent, input_pairs, x1 + int(column) - 1, y1 + int(tap_row) - 1);
            const uint32_t a0 = p0 < pixels && a ? a[k] : 0;
            const uint32_t a1 = p1 < pixels && b ? b[k] : 0;
            const uint32_t a2 = p0 < pixels && a ? a[k + 4] : 0;
            const uint32_t a3 = p1 < pixels && b ? b[k + 4] : 0;
            const auto fragment = weights + (tap * blocks + block) * 64;
            accumulate_f16(low, high, a0, a1, a2, a3, fragment[0], fragment[1]);
        }
    }
    const unsigned out_pair = group * 4 + pair;
    if (p0 < pixels) output[p0 * output_pairs + out_pair] = mask_leaky(low);
    if (p1 < pixels) output[p1 * output_pairs + out_pair] = mask_leaky(high);
}

template<SpatialAccumulation Order>
inline cudaError_t launch_spatial(const uint32_t* packed, const uint32_t* input, uint32_t* output,
                                  SpatialExtent extent, unsigned input_channels, unsigned output_channels,
                                  cudaStream_t stream) {
    constexpr unsigned threads = 128;
    constexpr unsigned pixels_per_block = threads / 32 * 16;
    const auto pixels = size_t(extent.width) * extent.height;
    const dim3 grid((pixels - 1) / pixels_per_block + 1, output_channels / 8);
    spatial_fp16<Order><<<grid, threads, 0, stream>>>(packed, input, output, extent,
                                                     input_channels, output_channels);
    return cudaGetLastError();
}

}
