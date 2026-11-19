// SPDX-License-Identifier: MIT
// C32 MLP + residual publication + QKV projection, one CTA per window.
// Uses the MMA fragment helper in swin_mma.cuh and the existing source recipe.
// Intermediate expansion and quantized state stay in shared memory.

template<bool PackedFP8>
__global__ __launch_bounds__(128, 2) void fused_swin_c32_mlp_qkv(
    const float* input, const float* expand, const float* contract,
    const float* qkv, const float* mlp_scale, float* residual_output, float* qkv_output,
    size_t windows, int height, int width, int grid_columns, int front_rows, int front_columns) {
    using Element = typename AttentionMmaElement<PackedFP8>::type;
    const size_t window = static_cast<size_t>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x), lane = thread & 31;
    const int row_base = (thread >> 5) * 16, group = lane >> 2, part = lane & 3;
    if (window >= windows) return;
    __shared__ __align__(16) Element state[64][36];
    __shared__ __align__(16) Element hidden[64][132];
    __shared__ __align__(16) Element weights[128 * 36];
    __shared__ float residual[64][32];
    for (int index = thread; index < 64 * 32; index += 128) {
        const int token = index % 64, channel = index / 64;
        const int row = static_cast<int>(window / grid_columns) * 8 + (token / 32) * 4 + (token / 4) % 4 - front_rows;
        const int column = static_cast<int>(window % grid_columns) * 8 + ((token / 16) % 2) * 4 + token % 4 - front_columns;
        const float value = row >= 0 && row < height && column >= 0 && column < width
            ? input[(static_cast<size_t>(channel) * height + row) * width + column] : 0.0f;
        residual[token][channel] = value;
        state[token][channel] = mma_operand<PackedFP8>(value);
    }
    for (int index = thread; index < 128 * 32; index += 128)
        weights[(index / 32) * 36 + index % 32] = mma_operand<PackedFP8>(expand[index]);
    __syncthreads();
    for (int column = 0; column < 128; column += 8) {
        float accumulator[4] = {};
        #pragma unroll
        for (int k = 0; k < 32; k += PackedFP8 ? 32 : 16)
            attention_mma<PackedFP8>(accumulator, &state[0][0], weights, 36, 36, row_base, column, k);
        #pragma unroll
        for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component) {
                const float value = accumulator[half * 2 + component];
                const float t = clamp_preserving_nan(value, -4.0f, 4.0f);
                const float inner = __fmaf_rn(fabsf(t), -0.055908203125f, 0.447265625f);
                const float activated = __fmul_rn(value, __fmaf_rn(t, inner, 0.89453125f));
                hidden[row_base + group + half * 8][column + part * 2 + component] = mma_operand<PackedFP8>(activated);
            }
    }
    __syncthreads();
    for (int index = thread; index < 32 * 128; index += 128)
        weights[(index / 128) * 132 + index % 128] = mma_operand<PackedFP8>(contract[index]);
    __syncthreads();
    #pragma unroll
    for (int column = 0; column < 32; column += 8) {
        float accumulator[4] = {};
        #pragma unroll
        for (int k = 0; k < 128; k += PackedFP8 ? 32 : 16)
            attention_mma<PackedFP8>(accumulator, &hidden[0][0], weights, 132, 132, row_base, column, k);
        #pragma unroll
        for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component) {
                const int row = row_base + group + half * 8, channel = column + part * 2 + component;
                const float value = __fadd_rn(__fmul_rn(residual[row][channel], mlp_scale[channel]), accumulator[half * 2 + component]);
                residual_output[(window * 64ull + row) * 32 + channel] = value;
                state[row][channel] = mma_operand<PackedFP8>(half_round(value));
            }
    }
    __syncthreads();
    for (int index = thread; index < 96 * 32; index += 128)
        weights[(index / 32) * 36 + index % 32] = mma_operand<PackedFP8>(qkv[index]);
    __syncthreads();
    for (int column = 0; column < 96; column += 8) {
        float accumulator[4] = {};
        #pragma unroll
        for (int k = 0; k < 32; k += PackedFP8 ? 32 : 16)
            attention_mma<PackedFP8>(accumulator, &state[0][0], weights, 36, 36, row_base, column, k);
        #pragma unroll
        for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int component = 0; component < 2; ++component)
                qkv_output[(window * 64ull + row_base + group + half * 8) * 96 + column + part * 2 + component] = accumulator[half * 2 + component];
    }
}
