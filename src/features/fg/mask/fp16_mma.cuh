#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace rw::fg::mask {

__device__ inline uint32_t mask_leaky(uint32_t value) {
    // Both binary16 lanes use the original rounded slope, 0x211f.
    constexpr uint32_t slope = 0x211f211fu;
    uint32_t negative, result;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(negative) : "r"(value), "r"(slope));
    asm("max.f16x2 %0, %1, %2;" : "=r"(result) : "r"(value), "r"(negative));
    return result;
}

__device__ inline void accumulate_f16(uint32_t& low, uint32_t& high,
                                      uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                      uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
                 : "+r"(low), "+r"(high)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

}
