// SPDX-License-Identifier: MIT
#include "hopper_mma.cuh"
#include "swin_c32_full.cuh"
#include "swin_c64_mlp.cuh"
#include "swin_expert_front.cuh"

template<int MmaColumns=8, bool Coalesced=false>
__global__ __launch_bounds__(128, 2) void fused_window_attention_hopper(
    const float* raw_input, const float* head_scale, const float* exponent_bias,
    float* output, size_t windows, int heads, int channels) {
    const size_t batch = static_cast<size_t>(blockIdx.x), window = batch / heads;
    const int head = static_cast<int>(batch % heads), thread = static_cast<int>(threadIdx.x);
    const int row_base = (thread / 32) * 16 + (thread % 32) / 4, column_pair = (thread % 4) * 2;
    if (window >= windows) return;
    __shared__ __align__(16) unsigned char queries[2048], keys[2048], values[2][1024];
    __shared__ __align__(16) unsigned char probability[2][2048];
    union AttentionWork {
        float raw[Coalesced ? 64 : 1][Coalesced ? 97 : 1];
        struct { float scores[64][65],reciprocal[64]; } attention;
    };
    __shared__ AttentionWork work;
    auto& scores=work.attention.scores;
    auto& reciprocal=work.attention.reciprocal;
    if constexpr(Coalesced) {
        for(int index=thread;index<64*96;index+=128) {
            const int row=index/96,dimension=index%96;
            work.raw[row][dimension]=raw_input[(window*64ull+row)*heads*96ull+head*96ull+dimension];
        }
        __syncthreads();
    }
    if (thread < 64) {
        const size_t base = (window * 64ull + thread) * heads * 96ull + head * 96ull;
        const float* raw=Coalesced ? work.raw[thread] : raw_input+base;
        const float qs = __fmul_rn(norm_scale(raw), head_scale[head]);
        const float ks = norm_scale(raw + 32);
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension) {
            queries[hopper_e4_offset(thread, dimension)] = hopper_encode(__fmul_rn(raw[dimension], qs));
            keys[hopper_e4_offset(thread, dimension)] = hopper_encode(__fmul_rn(raw[32 + dimension], ks));
            values[thread / 32][hopper_e4_offset(dimension, thread % 32)] = hopper_encode(half_round(raw[64 + dimension]));
        }
    }
    hopper_publish_shared();
    #pragma unroll
    for (int column = 0; column < 64; column += MmaColumns) {
        float accumulator[MmaColumns/2] = {};
        hopper_e4_mma(accumulator, queries, keys + (column / 8) * 256);
        #pragma unroll
        for (int tile = 0; tile < MmaColumns/8; ++tile)
            #pragma unroll
            for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component) {
                const int query = row_base + half * 8, key = column + tile * 8 + column_pair + component;
                const float bias = exponent_bias[(static_cast<size_t>(head) * 64 + query) * 64 + key];
                const float ramp = clamp_preserving_nan(half_round(__fmaf_rn(accumulator[tile * 4 + half * 2 + component], kExpSlope, bias)), kExpLow, kExpHigh);
                const unsigned short bits = __half_as_ushort(__float2half_rn(ramp));
                scores[query][key] = __half2float(__ushort_as_half(static_cast<unsigned short>((bits & 0x3ffu) << 5u)));
            }
    }
    __syncthreads();
    if (thread < 64) reciprocal[thread] = attention_probability_reciprocal(scores[thread]);
    __syncthreads();
    for (int index = thread; index < 64 * 64; index += 128) {
        const int row = index / 64, key = index % 64;
        probability[key / 32][hopper_e4_offset(row, key % 32)] = hopper_encode(half_round(__fmul_rn(scores[row][key], reciprocal[row])));
    }
    hopper_publish_shared();
    #pragma unroll
    for (int column = 0; column < 32; column += MmaColumns) {
        float accumulator[MmaColumns/2] = {};
        #pragma unroll
        for (int k = 0; k < 2; ++k)
            hopper_e4_mma(accumulator, probability[k], values[k] + (column / 8) * 256);
        #pragma unroll
        for (int tile = 0; tile < MmaColumns/8; ++tile)
            #pragma unroll
            for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component)
                output[(window * 64ull + row_base + half * 8) * channels + head * 32 + column + tile * 8 + column_pair + component] = accumulator[tile * 4 + half * 2 + component];
    }
}

template<int MmaColumns=8, bool Coalesced=false>
__global__ __launch_bounds__(128, 2) void fused_swin_c32_mlp_qkv_hopper(
    const float* input, const float* expand, const float* contract, const float* qkv,
    const float* mlp_scale, float* residual_output, float* qkv_output,
    size_t windows, int height, int width, int grid_columns, int front_rows, int front_columns) {
    const size_t window = static_cast<size_t>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int row_base = (thread / 32) * 16 + (thread % 32) / 4, column_pair = (thread % 4) * 2;
    if (window >= windows) return;
    __shared__ __align__(16) unsigned char state[2048], hidden[4][2048], weights[4096];
    __shared__ float residual[64][Coalesced ? 33 : 32];
    for (int index = thread; index < 64 * 32; index += 128) {
        const int token = index % 64, channel = index / 64;
        const int row = static_cast<int>(window / grid_columns) * 8 + (token / 32) * 4 + (token / 4) % 4 - front_rows;
        const int column = static_cast<int>(window % grid_columns) * 8 + ((token / 16) % 2) * 4 + token % 4 - front_columns;
        const float value = row >= 0 && row < height && column >= 0 && column < width
            ? input[(static_cast<size_t>(channel) * height + row) * width + column] : 0.0f;
        residual[token][channel] = value;
        state[hopper_e4_offset(token, channel)] = hopper_encode(value);
    }
    for (int index = thread; index < 128 * 32; index += 128)
        weights[hopper_e4_offset(index / 32, index % 32)] = hopper_encode(expand[index]);
    hopper_publish_shared();
    for (int column = 0; column < 128; column += MmaColumns) {
        float accumulator[MmaColumns/2] = {};
        hopper_e4_mma(accumulator, state, weights + (column / 8) * 256);
        #pragma unroll
        for (int tile = 0; tile < MmaColumns/8; ++tile)
            #pragma unroll
            for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component) {
                const int row = row_base + half * 8, channel = column + tile * 8 + column_pair + component;
                const float value = accumulator[tile * 4 + half * 2 + component], t = clamp_preserving_nan(value, -4.0f, 4.0f);
                const float inner = __fmaf_rn(fabsf(t), -0.055908203125f, 0.447265625f);
                hidden[channel / 32][hopper_e4_offset(row, channel % 32)] = hopper_encode(__fmul_rn(value, __fmaf_rn(t, inner, 0.89453125f)));
            }
    }
    __syncthreads();
    for (int index = thread; index < 32 * 128; index += 128) {
        const int column = index / 128, k = index % 128;
        weights[(k / 32) * 1024 + hopper_e4_offset(column, k % 32)] = hopper_encode(contract[index]);
    }
    hopper_publish_shared();
    #pragma unroll
    for (int column = 0; column < 32; column += MmaColumns) {
        float accumulator[MmaColumns/2] = {};
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            hopper_e4_mma(accumulator, hidden[k], weights + k * 1024 + (column / 8) * 256);
        #pragma unroll
        for (int tile = 0; tile < MmaColumns/8; ++tile)
            #pragma unroll
            for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component) {
                const int row = row_base + half * 8, channel = column + tile * 8 + column_pair + component;
                const float value = __fadd_rn(__fmul_rn(residual[row][channel], mlp_scale[channel]), accumulator[tile * 4 + half * 2 + component]);
                residual_output[(window * 64ull + row) * 32 + channel] = value;
                state[hopper_e4_offset(row, channel)] = hopper_encode(half_round(value));
            }
    }
    __syncthreads();
    for (int index = thread; index < 96 * 32; index += 128)
        weights[hopper_e4_offset(index / 32, index % 32)] = hopper_encode(qkv[index]);
    hopper_publish_shared();
    for (int column = 0; column < 96; column += MmaColumns) {
        float accumulator[MmaColumns/2] = {};
        hopper_e4_mma(accumulator, state, weights + (column / 8) * 256);
        #pragma unroll
        for (int tile = 0; tile < MmaColumns/8; ++tile)
            #pragma unroll
            for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component)
                qkv_output[(window * 64ull + row_base + half * 8) * 96 + column + tile * 8 + column_pair + component] = accumulator[tile * 4 + half * 2 + component];
    }
}
