// SPDX-License-Identifier: MIT
// Swin attention with packed E4M3 operands and FP32 Tensor Core accumulators.
// Fragment coordinates follow OpenDLSS-NR scripts/ptx/swin.py, commit
// 9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1 (MIT, Copyright 2026 maan).
// This lowering retains this backend's normalization/publication recipe;
// it does not introduce the upstream port's different native-f16 recipe.
// Included inside rw::native_nr's anonymous namespace, after Swin helpers.

template<bool PackedFP8> struct AttentionMmaElement;
template<> struct AttentionMmaElement<true> { using type = unsigned char; };
template<> struct AttentionMmaElement<false> { using type = __half; };

__device__ __forceinline__ float attention_probability_reciprocal(const float* row) {
    float pairs[4][2][2];
    for (int pair = 0; pair < 4; ++pair)
        for (int member = 0; member < 2; ++member)
            for (int half = 0; half < 2; ++half) {
                const int key = pair * 4 + member * 2 + half;
                float total = half_sum(row[key], row[16 + key]);
                total = half_sum(total, row[32 + key]);
                pairs[pair][member][half] = half_sum(total, row[48 + key]);
            }
    float members[2];
    for (int member = 0; member < 2; ++member) {
        float halves[2];
        for (int half = 0; half < 2; ++half) {
            float sum = half_sum(pairs[0][member][half], pairs[1][member][half]);
            sum = half_sum(sum, pairs[2][member][half]);
            halves[half] = half_sum(sum, pairs[3][member][half]);
        }
        members[member] = half_sum(halves[0], halves[1]);
    }
    const float denominator = half_sum(members[0], members[1]);
    return __fdiv_rn(1.0f, isnan(denominator) ? denominator : fmaxf(denominator, kSumFloor));
}

template<bool PackedFP8>
__device__ __forceinline__ typename AttentionMmaElement<PackedFP8>::type mma_operand(float value) {
    if constexpr (PackedFP8)
        return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    else
        return __float2half_rn(fp8_round(value));
}

// A: rows x K, B: columns x K (column-major K x columns). Adjacent four
// lanes own K fragments of the same row/column. Output is a 16 x 8 tile.
template<bool PackedFP8, class Element>
__device__ __forceinline__ void attention_mma(float (&d)[4], const Element* a,
    const Element* b, int a_stride, int b_stride, int row, int column, int k) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int group = lane >> 2, part = lane & 3;
    const int base_k = k + part * (PackedFP8 ? 4 : 2);
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(a + (row + group) * a_stride + base_k);
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(a + (row + group + 8) * a_stride + base_k);
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(a + (row + group) * a_stride + base_k + (PackedFP8 ? 16 : 8));
    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(a + (row + group + 8) * a_stride + base_k + (PackedFP8 ? 16 : 8));
    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(b + (column + group) * b_stride + base_k);
    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(b + (column + group) * b_stride + base_k + (PackedFP8 ? 16 : 8));
    if constexpr (PackedFP8) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
        // Unsupported devices must not execute a replacement or identity.
        asm volatile("trap;");
#endif
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}

template<bool PackedFP8>
__global__ __launch_bounds__(128, 2) void fused_window_attention_tensor_core(
    const float* raw_input, const float* head_scale, const float* exponent_bias,
    float* output, size_t windows, int heads, int channels) {
    using Element = typename AttentionMmaElement<PackedFP8>::type;
    const size_t batch = static_cast<size_t>(blockIdx.x);
    const size_t window = batch / static_cast<size_t>(heads);
    const int head = static_cast<int>(batch % static_cast<size_t>(heads));
    const int thread = static_cast<int>(threadIdx.x), lane = thread & 31;
    const int row_base = (thread >> 5) * 16, group = lane >> 2, part = lane & 3;
    if (window >= windows) return;
    __shared__ __align__(16) Element queries[64][36];
    __shared__ __align__(16) Element keys[64][36];
    __shared__ __align__(16) Element values_transposed[32][68];
    __shared__ __align__(16) Element probabilities[64][68];
    __shared__ float scores[64][65];
    __shared__ float reciprocals[64];
    const size_t token_base = window * 64ull * static_cast<size_t>(heads) * 96ull;
    if (thread < 64) {
        const size_t base = token_base + static_cast<size_t>(thread) * heads * 96ull + head * 96ull;
        const float query_scale = __fmul_rn(norm_scale(raw_input + base), head_scale[head]);
        const float key_scale = norm_scale(raw_input + base + 32);
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension) {
            queries[thread][dimension] = mma_operand<PackedFP8>(__fmul_rn(raw_input[base + dimension], query_scale));
            keys[thread][dimension] = mma_operand<PackedFP8>(__fmul_rn(raw_input[base + 32 + dimension], key_scale));
            values_transposed[dimension][thread] = mma_operand<PackedFP8>(half_round(raw_input[base + 64 + dimension]));
        }
    }
    __syncthreads();
    #pragma unroll
    for (int column = 0; column < 64; column += 8) {
        float accumulator[4] = {};
        #pragma unroll
        for (int k = 0; k < 32; k += PackedFP8 ? 32 : 16)
            attention_mma<PackedFP8>(accumulator, &queries[0][0], &keys[0][0],
                                     36, 36, row_base, column, k);
        #pragma unroll
        for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component) {
                const int query = row_base + group + half * 8, key = column + part * 2 + component;
                const float bias = exponent_bias[(static_cast<size_t>(head) * 64 + query) * 64 + key];
                const float ramp = clamp_preserving_nan(half_round(__fmaf_rn(accumulator[half * 2 + component], kExpSlope, bias)), kExpLow, kExpHigh);
                const unsigned short bits = __half_as_ushort(__float2half_rn(ramp));
                scores[query][key] = __half2float(__ushort_as_half(static_cast<unsigned short>((bits & 0x3ffu) << 5u)));
            }
    }
    __syncthreads();
    if (thread < 64) {
        float pairs[4][2][2];
        for (int pair = 0; pair < 4; ++pair)
            for (int member = 0; member < 2; ++member)
                for (int half = 0; half < 2; ++half) {
                    const int key = pair * 4 + member * 2 + half;
                    float total = scores[thread][key];
                    total = half_sum(total, scores[thread][16 + key]);
                    total = half_sum(total, scores[thread][32 + key]);
                    pairs[pair][member][half] = half_sum(total, scores[thread][48 + key]);
                }
        float members[2];
        for (int member = 0; member < 2; ++member) {
            float halves[2];
            for (int half = 0; half < 2; ++half) {
                float sum = half_sum(pairs[0][member][half], pairs[1][member][half]);
                sum = half_sum(sum, pairs[2][member][half]);
                halves[half] = half_sum(sum, pairs[3][member][half]);
            }
            members[member] = half_sum(halves[0], halves[1]);
        }
        const float denominator = half_sum(members[0], members[1]);
        const float clamped = isnan(denominator) ? denominator : fmaxf(denominator, kSumFloor);
        reciprocals[thread] = __fdiv_rn(1.0f, clamped);
    }
    __syncthreads();
    for (int index = thread; index < 64 * 64; index += 128) {
        const int query = index / 64, key = index % 64;
        probabilities[query][key] = mma_operand<PackedFP8>(half_round(__fmul_rn(scores[query][key], reciprocals[query])));
    }
    __syncthreads();
    #pragma unroll
    for (int column = 0; column < 32; column += 8) {
        float accumulator[4] = {};
        #pragma unroll
        for (int k = 0; k < 64; k += PackedFP8 ? 32 : 16)
            attention_mma<PackedFP8>(accumulator, &probabilities[0][0], &values_transposed[0][0],
                                     68, 68, row_base, column, k);
        #pragma unroll
        for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component)
                output[(window * 64ull + row_base + group + half * 8) * channels +
                       head * 32 + column + part * 2 + component] = accumulator[half * 2 + component];
    }
}
