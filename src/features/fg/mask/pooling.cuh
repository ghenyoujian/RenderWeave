#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rw::fg::mask {

// The original reduction uses ordered comparison masks, half multiply and
// half FMA in that order. This retains its signed-zero and exceptional behavior.
__device__ inline __half2 pool_select(__half2 current, __half2 sample) {
    const auto from_current = __hgt2(current, sample);
    const auto from_sample = __hle2(current, sample);
    const auto selected_sample = __hmul2_rn(from_sample, sample);
    return __hfma2(from_current, current, selected_sample);
}

static __global__ void pool_2x2(const __half2* input, __half2* output,
                                int in_width, int in_height, int out_width,
                                int out_height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height) return;

    const int channel_pairs = channels / 2;
    const int left = 2 * x;
    const int top = 2 * y;
    const int right = min(left + 2, in_width);
    const int bottom = min(top + 2, in_height);
    const auto first = (size_t(top) * in_width + left) * channel_pairs;
    const auto dst = (size_t(y) * out_width + x) * channel_pairs;
    for (int pair = threadIdx.z; pair < channel_pairs; pair += blockDim.z) {
        auto value = input[first + pair];
        for (int row = top; row < bottom; ++row) {
            const int start = left + (row == top);
            for (int col = start; col < right; ++col) {
                const auto offset = (size_t(row) * in_width + col) * channel_pairs + pair;
                value = pool_select(value, input[offset]);
            }
        }
        output[dst + pair] = value;
    }
}

// The descriptor provides positive extents, even channels, and output
// coordinates whose 2x2 windows start inside the input.
inline cudaError_t launch_pool_2x2(const __half2* input, __half2* output,
                                  int in_width, int in_height, int out_width,
                                  int out_height, int channels, cudaStream_t stream) {
    const dim3 block(4, 4, 16);
    const dim3 grid((out_width - 1) / block.x + 1, (out_height - 1) / block.y + 1, 1);
    pool_2x2<<<grid, block, 0, stream>>>(input, output, in_width, in_height, out_width, out_height, channels);
    return cudaGetLastError();
}

}
