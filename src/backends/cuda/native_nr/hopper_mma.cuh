#pragma once
// SPDX-License-Identifier: MIT
// Hopper FP8 execution. Shared layout and the issue/fence sequence reuse the
// project's tools/probes/hopper_fp8_wgmma primitive, without its probe host.

inline bool hopper_wide_mma_requested() {
    const char* value=std::getenv("RW_NR_HOPPER_WIDE_MMA");
    return value && std::strcmp(value,"1")==0;
}
inline bool hopper_coalesced_requested() {
    const char* value=std::getenv("RW_NR_HOPPER_COALESCED_INPUT");
    return value && std::strcmp(value,"1")==0;
}
inline bool hopper_batched_mma_requested() {
    const char* value=std::getenv("RW_NR_HOPPER_BATCHED_MMA");
    return value && std::strcmp(value,"1")==0;
}

__device__ __forceinline__ int hopper_e4_offset(int row, int k) {
    return (row % 8) * 16 + (row / 8) * 256 + (k / 16) * 128 + k % 16;
}
__device__ __forceinline__ unsigned char hopper_encode(float value) {
    return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ void hopper_publish_shared() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
    __syncthreads();
}
__device__ __forceinline__ void hopper_e4_mma(float (&d)[4], const unsigned char* a,
                                            const unsigned char* b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    const uint64_t ad = (static_cast<uint64_t>(__cvta_generic_to_shared(a)) >> 4) |
                        (uint64_t(8) << 16) | (uint64_t(16) << 32);
    const uint64_t bd = (static_cast<uint64_t>(__cvta_generic_to_shared(b)) >> 4) |
                        (uint64_t(8) << 16) | (uint64_t(16) << 32);
    asm volatile("{\n.reg .pred keep_c;\nsetp.ne.b32 keep_c,1,0;\n"
                 "wgmma.fence.sync.aligned;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3},%4,%5,keep_c,1,1;\n"
                 "wgmma.commit_group.sync.aligned;\n"
                 "wgmma.wait_group.sync.aligned 0;\n}"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "l"(ad), "l"(bd) : "memory");
#else
    asm volatile("trap;");
#endif
}

__device__ __forceinline__ void hopper_e4_mma(float (&d)[16], const unsigned char* a,
                                            const unsigned char* b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    const uint64_t ad = (static_cast<uint64_t>(__cvta_generic_to_shared(a)) >> 4) |
                        (uint64_t(8) << 16) | (uint64_t(16) << 32);
    const uint64_t bd = (static_cast<uint64_t>(__cvta_generic_to_shared(b)) >> 4) |
                        (uint64_t(8) << 16) | (uint64_t(16) << 32);
    asm volatile("{\n.reg .pred keep_c;\nsetp.ne.b32 keep_c,1,0;\n"
                 "wgmma.fence.sync.aligned;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},%16,%17,keep_c,1,1;\n"
                 "wgmma.commit_group.sync.aligned;\n"
                 "wgmma.wait_group.sync.aligned 0;\n}"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "l"(ad), "l"(bd) : "memory");
#else
    asm volatile("trap;");
#endif
}

__device__ __forceinline__ void hopper_f16_mma(float (&d)[16], const unsigned char* a,
                                            const unsigned char* b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    const uint64_t ad = (static_cast<uint64_t>(__cvta_generic_to_shared(a)) >> 4) |
                        (uint64_t(8) << 16) | (uint64_t(16) << 32);
    const uint64_t bd = (static_cast<uint64_t>(__cvta_generic_to_shared(b)) >> 4) |
                        (uint64_t(8) << 16) | (uint64_t(16) << 32);
    asm volatile("{\n.reg .pred keep_c;\nsetp.ne.b32 keep_c,1,0;\n"
                 "wgmma.fence.sync.aligned;\n"
                 "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},%16,%17,keep_c,1,1,0,0;\n"
                 "wgmma.commit_group.sync.aligned;\n"
                 "wgmma.wait_group.sync.aligned 0;\n}"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                 : "l"(ad), "l"(bd) : "memory");
#else
    asm volatile("trap;");
#endif
}

// Consecutive K tiles consume immutable shared operands and the same FP32
// accumulator. Keep their instruction order, but retire the group once.
#define RW_HOPPER_MMA_D "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}"
#define RW_HOPPER_MMA32(A,B) "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 " RW_HOPPER_MMA_D "," A "," B ",keep_c,1,1;\n"
#define RW_HOPPER_MMA_D_ARGS "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), \
    "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), \
    "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])

template<int KTiles,int AStride,int BStride>
__device__ __forceinline__ void hopper_e4_mma_tiles(float (&d)[16],
    const unsigned char* a,const unsigned char* b,bool batched) {
    static_assert(KTiles==2 || KTiles==4);
    if(!batched) {
        #pragma unroll
        for(int k=0;k<KTiles;++k)hopper_e4_mma(d,a+k*AStride,b+k*BStride);
        return;
    }
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    const uint64_t ad=(static_cast<uint64_t>(__cvta_generic_to_shared(a))>>4) |
                      (uint64_t(8)<<16) | (uint64_t(16)<<32);
    const uint64_t bd=(static_cast<uint64_t>(__cvta_generic_to_shared(b))>>4) |
                      (uint64_t(8)<<16) | (uint64_t(16)<<32);
    if constexpr(KTiles==2) {
        asm volatile("{\n.reg .pred keep_c;\nsetp.ne.b32 keep_c,1,0;\n"
            "wgmma.fence.sync.aligned;\n"
            RW_HOPPER_MMA32("%16","%17") RW_HOPPER_MMA32("%18","%19")
            "wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n}"
            : RW_HOPPER_MMA_D_ARGS
            : "l"(ad),"l"(bd),"l"(ad+AStride/16),"l"(bd+BStride/16) : "memory");
    } else {
        asm volatile("{\n.reg .pred keep_c;\nsetp.ne.b32 keep_c,1,0;\n"
            "wgmma.fence.sync.aligned;\n"
            RW_HOPPER_MMA32("%16","%17") RW_HOPPER_MMA32("%18","%19")
            RW_HOPPER_MMA32("%20","%21") RW_HOPPER_MMA32("%22","%23")
            "wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n}"
            : RW_HOPPER_MMA_D_ARGS
            : "l"(ad),"l"(bd),"l"(ad+AStride/16),"l"(bd+BStride/16),
              "l"(ad+2*AStride/16),"l"(bd+2*BStride/16),
              "l"(ad+3*AStride/16),"l"(bd+3*BStride/16) : "memory");
    }
#else
    asm volatile("trap;");
#endif
}
#undef RW_HOPPER_MMA_D_ARGS
#undef RW_HOPPER_MMA32
#undef RW_HOPPER_MMA_D
