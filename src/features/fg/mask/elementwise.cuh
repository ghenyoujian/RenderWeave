#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rw::fg::mask {

struct ElementExtent {
    unsigned width;
    unsigned height;
};

template<unsigned Channels>
static __global__ void add_repeated(const __half2* left, const __half2* right, __half2* output,
                                    ElementExtent a, ElementExtent b, ElementExtent destination) {
    constexpr unsigned pairs = Channels / 2;
    const unsigned pair = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned x = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned y = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= destination.width || y >= destination.height || pair >= pairs) return;

    const auto ai = (size_t(y % a.height) * a.width + x % a.width) * pairs + pair;
    const auto bi = (size_t(y % b.height) * b.width + x % b.width) * pairs + pair;
    const auto oi = (size_t(y) * destination.width + x) * pairs + pair;
    output[oi] = __hadd2_rn(left[ai], right[bi]);
}

template<unsigned Channels>
inline cudaError_t launch_add_repeated(const __half2* left, const __half2* right, __half2* output,
                                       ElementExtent a, ElementExtent b, ElementExtent destination,
                                       cudaStream_t stream) {
    static_assert(Channels == 32 || Channels == 64 || Channels == 128 || Channels == 256);
    const dim3 block(32, 4, 2);
    const dim3 grid((Channels / 2 + block.x - 1) / block.x,
                    (destination.width - 1) / block.y + 1, (destination.height - 1) / block.z + 1);
    add_repeated<Channels><<<grid, block, 0, stream>>>(left, right, output, a, b, destination);
    return cudaGetLastError();
}

}
