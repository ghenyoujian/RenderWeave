// Native CUDA arithmetic port of dlssnr/layers/vit.py at
// mauri870/dlssnr-pytorch 3beef7935a8ef976b47ab78b906e75600c52d2de.
// Copyright (c) 2026 Mauri de Souza Meneguzzo. MIT License.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "native_nr.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace rw::native_nr {
namespace {
constexpr unsigned warp_mask = 0xffffffffu;
constexpr int head_dimensions = 32;

__device__ float half_round(float value) {
    return __half2float(__float2half_rn(value));
}

__device__ float fp8_round(float value) {
    __nv_fp8_e4m3 result;
    result.__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    return static_cast<float>(result);
}

__global__ void expand_activation(const float* input, float* output, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    const float clipped = isnan(x) ? x : fminf(fmaxf(x, -4.0f), 4.0f);
    // The source evaluates this polynomial as separate eager FP32 operations;
    // it differs from the fused cubic in the window-block family.
    const float first = __fmul_rn(-0.055908203125f, fabsf(clipped));
    const float second = __fmul_rn(first, clipped);
    const float third = __fmul_rn(0.447265625f, clipped);
    const float polynomial = __fadd_rn(__fadd_rn(second, third), 0.89453125f);
    output[index] = fp8_round(half_round(__fmul_rn(x, polynomial)));
}

__global__ void gained_residual(const float* projection, const float* residual,
                                const float* gain, float* output,
                                size_t count, int spatial) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t channel = index / static_cast<size_t>(spatial);
    // gain * residual is published as FP32 before the add in the pinned source.
    const float gated = __fmul_rn(gain[channel], residual[index]);
    output[index] = fp8_round(half_round(__fadd_rn(projection[index], gated)));
}

__device__ float inverse_norm(float value) {
    const unsigned lane = threadIdx.x & 31u;
    float sum = half_round(__fmul_rn(value, value));
    // Pairs d,d+16 followed by the source's 8,4,2,1 half butterfly.
    for (unsigned offset = 16; offset != 0; offset >>= 1u) {
        const float other = __shfl_down_sync(warp_mask, sum, offset);
        if (lane < offset) sum = half_round(__fadd_rn(sum, other));
    }
    const float inverse = isnan(sum) ? sum : half_round(rsqrtf(fmaxf(sum, 6.198883056640625e-05f)));
    return __shfl_sync(warp_mask, inverse, 0);
}

__global__ void normalise_qkv(const float* projected_query, const float* projected_key,
                              const float* projected_value, const float* temperature,
                              float* output, int channels, int tokens) {
    const int token = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int dimension = static_cast<int>(threadIdx.x);
    const size_t index = static_cast<size_t>(head * head_dimensions + dimension) * tokens + token;
    const float query = half_round(projected_query[index]);
    const float key = half_round(projected_key[index]);
    const float value = half_round(projected_value[index]);
    const float query_inverse = inverse_norm(query);
    const float key_inverse = inverse_norm(key);
    const float sqrt_head = half_round(5.6568542494923801952f);
    float query_scaled = half_round(__fmul_rn(query, sqrt_head));
    query_scaled = half_round(__fmul_rn(query_scaled, query_inverse));
    query_scaled = half_round(__fmul_rn(query_scaled, half_round(temperature[head])));
    const size_t plane = static_cast<size_t>(channels) * tokens;
    output[index] = fp8_round(query_scaled);
    output[plane + index] = fp8_round(half_round(__fmul_rn(key, key_inverse)));
    output[2 * plane + index] = fp8_round(value);
}

__global__ void normalise_qkv_fused(const float* projected, const float* temperature,
                                    float* output, int channels, int tokens) {
    const int token = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int dimension = static_cast<int>(threadIdx.x);
    const size_t index = static_cast<size_t>(head * head_dimensions + dimension) * tokens + token;
    const size_t plane = static_cast<size_t>(channels) * tokens;
    const float query = half_round(projected[index]);
    const float key = half_round(projected[plane + index]);
    const float value = half_round(projected[2 * plane + index]);
    const float query_inverse = inverse_norm(query);
    const float key_inverse = inverse_norm(key);
    const float sqrt_head = half_round(5.6568542494923801952f);
    float query_scaled = half_round(__fmul_rn(query, sqrt_head));
    query_scaled = half_round(__fmul_rn(query_scaled, query_inverse));
    query_scaled = half_round(__fmul_rn(query_scaled, half_round(temperature[head])));
    output[index] = fp8_round(query_scaled);
    output[plane + index] = fp8_round(half_round(__fmul_rn(key, key_inverse)));
    output[2 * plane + index] = fp8_round(value);
}

__device__ float approximate_exponential(float logit) {
    // The source narrows the dot product, then forms an exact binary affine
    // expression in FP64 and rounds onto a 1/1024 grid, ties to even.
    const double input = static_cast<double>(half_round(logit));
    const double product = __dmul_rn(input, 0.08953857421875);
    const double ramp = __dmul_rn(__dadd_rn(product, 1.708984375), 1024.0);
    if (isnan(ramp)) return nanf("");
    // Clamp before the integer conversion also defines +/-infinity safely;
    // the endpoints are integers, so this commutes with the source RNE step.
    const int rounded = ramp <= 1474.0 ? 1474 : (ramp >= 2025.0 ? 2025 : __double2int_rn(ramp));
    const unsigned short bits = static_cast<unsigned short>((rounded - 1024) << 4);
    return __half2float(__ushort_as_half(bits));
}

__device__ float group_denominator(const float* probability) {
    float partial[8];
    for (int column = 0; column < 8; ++column) {
        float value = half_round(__fadd_rn(probability[column], probability[column + 8]));
        for (int block = 1; block < 4; ++block) {
            const float pair = half_round(__fadd_rn(probability[block * 16 + column],
                                                    probability[block * 16 + column + 8]));
            value = half_round(__fadd_rn(value, pair));
        }
        partial[column] = value;
    }
    const float even = half_round(__fadd_rn(half_round(__fadd_rn(
        half_round(__fadd_rn(partial[0], partial[2])), partial[4])), partial[6]));
    const float odd = half_round(__fadd_rn(half_round(__fadd_rn(
        half_round(__fadd_rn(partial[1], partial[3])), partial[5])), partial[7]));
    return half_round(__fadd_rn(even, odd));
}

#if RW_NR_HOPPER_WGMMA_ENABLED
#include "vit_hopper.cuh"
#endif

// One warp handles a (query, head). Keys stream in native groups of 64, so the
// implementation uses O(C*T) tensors and 256 bytes of per-block denominator
// scratch rather than materializing an O(heads*T*T) matrix. This compatibility
// kernel is intentionally unoptimized; dot-order differences need measurement.
__global__ void global_attention(const float* qkv, float* output,
                                 int channels, int tokens, int padded_tokens) {
    const int query = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int dimension = static_cast<int>(threadIdx.x);
    const size_t channel = static_cast<size_t>(head * head_dimensions + dimension);
    const size_t plane = static_cast<size_t>(channels) * tokens;
    const float q = qkv[channel * tokens + query];
    float numerator = 0.0f;
    float denominator = 0.0f;
    __shared__ float probabilities[64];

    for (int group = 0; group < padded_tokens; group += 64) {
        for (int member = 0; member < 64; ++member) {
            const int key_index = group + member;
            const float key = key_index < tokens ? qkv[plane + channel * tokens + key_index] : 0.0f;
            float logit = __fmul_rn(q, key);
            for (unsigned offset = 16; offset != 0; offset >>= 1u)
                logit = __fadd_rn(logit, __shfl_down_sync(warp_mask, logit, offset));
            logit = __shfl_sync(warp_mask, logit, 0);
            const float probability = approximate_exponential(logit);
            if (dimension == 0) probabilities[member] = probability;
            const float value = key_index < tokens ? qkv[2 * plane + channel * tokens + key_index] : 0.0f;
            numerator = __fmaf_rn(fp8_round(probability), value, numerator);
        }
        __syncwarp(warp_mask);
        if (dimension == 0) {
            const float sum = group_denominator(probabilities);
            denominator = group == 0 ? sum : half_round(__fadd_rn(denominator, sum));
        }
        __syncwarp(warp_mask);
    }
    denominator = __shfl_sync(warp_mask, denominator, 0);
    // approximate_exponential(0) is exactly 0.083984375 before this half hop.
    const float padding = half_round((padded_tokens - tokens) * 0.083984375f);
    denominator = half_round(__fadd_rn(denominator, -padding));
    const float bounded_denominator = isnan(denominator) ? denominator :
        fmaxf(denominator, 6.198883056640625e-05f);
    const float inverse = half_round(__fdiv_rn(1.0f, bounded_denominator));
    const float result = half_round(__fmul_rn(half_round(numerator), inverse));
    output[channel * tokens + query] = fp8_round(result);
}

// The original compatibility kernel reads K/V in a channel-major stride for
// every key.  At 1080p this leaves the H200 memory path under-filled even
// though the arithmetic is small.  One warp loads each 64-key tile with a
// coalesced key sweep into a padded shared-memory transpose, then reuses that
// tile for the exact source reduction order.  The published arithmetic and
// padding correction are unchanged.
__global__ __launch_bounds__(32, 4) void global_attention_tiled(
    const float* qkv, float* output, int channels, int tokens, int padded_tokens) {
    const int query = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int dimension = static_cast<int>(threadIdx.x);
    const size_t channel = static_cast<size_t>(head * head_dimensions + dimension);
    const size_t plane = static_cast<size_t>(channels) * tokens;
    const float q = qkv[channel * tokens + query];
    float numerator = 0.0f;
    float denominator = 0.0f;
    __shared__ float key_tile[64][33];
    __shared__ float value_tile[64][33];
    __shared__ float probabilities[64];
    for (int group = 0; group < padded_tokens; group += 64) {
        #pragma unroll
        for (int source_dimension = 0; source_dimension < 32; ++source_dimension) {
            const int first = group + dimension;
            const int second = first + 32;
            key_tile[dimension][source_dimension] = first < tokens
                ? qkv[plane + (static_cast<size_t>(head * head_dimensions + source_dimension) * tokens) + first]
                : 0.0f;
            key_tile[dimension + 32][source_dimension] = second < tokens
                ? qkv[plane + (static_cast<size_t>(head * head_dimensions + source_dimension) * tokens) + second]
                : 0.0f;
            value_tile[dimension][source_dimension] = first < tokens
                ? qkv[2 * plane + (static_cast<size_t>(head * head_dimensions + source_dimension) * tokens) + first]
                : 0.0f;
            value_tile[dimension + 32][source_dimension] = second < tokens
                ? qkv[2 * plane + (static_cast<size_t>(head * head_dimensions + source_dimension) * tokens) + second]
                : 0.0f;
        }
        __syncwarp(warp_mask);
        #pragma unroll
        for (int member = 0; member < 64; ++member) {
            float logit = __fmul_rn(q, key_tile[member][dimension]);
            for (unsigned offset = 16; offset != 0; offset >>= 1u)
                logit = __fadd_rn(logit, __shfl_down_sync(warp_mask, logit, offset));
            logit = __shfl_sync(warp_mask, logit, 0);
            const float probability = approximate_exponential(logit);
            if (dimension == 0) probabilities[member] = probability;
            numerator = __fmaf_rn(fp8_round(probability), value_tile[member][dimension], numerator);
        }
        __syncwarp(warp_mask);
        if (dimension == 0) {
            const float sum = group_denominator(probabilities);
            denominator = group == 0 ? sum : half_round(__fadd_rn(denominator, sum));
        }
        __syncwarp(warp_mask);
    }
    denominator = __shfl_sync(warp_mask, denominator, 0);
    const float padding = half_round((padded_tokens - tokens) * 0.083984375f);
    denominator = half_round(__fadd_rn(denominator, -padding));
    const float bounded_denominator = isnan(denominator) ? denominator :
        fmaxf(denominator, 6.198883056640625e-05f);
    const float inverse = half_round(__fdiv_rn(1.0f, bounded_denominator));
    const float result = half_round(__fmul_rn(half_round(numerator), inverse));
    output[channel * tokens + query] = fp8_round(result);
}

// Process four query rows per block so the 64-key K/V tile is fetched once and
// shared by four warps. Each warp retains the original source dot-product tree,
// probability grouping, key order, and final publication sequence.
__global__ __launch_bounds__(128, 2) void global_attention_tiled_multi_query(
    const float* qkv, float* output, int channels, int tokens, int padded_tokens) {
    constexpr int queries_per_block = 4;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int dimension = thread & 31;
    const int query = static_cast<int>(blockIdx.x) * queries_per_block + warp;
    const int head = static_cast<int>(blockIdx.y);
    const size_t plane = static_cast<size_t>(channels) * tokens;
    __shared__ float key_tile[64][33];
    __shared__ float value_tile[64][33];
    __shared__ float probabilities[queries_per_block][64];
    const float q = query < tokens
        ? qkv[(static_cast<size_t>(head) * 32 + dimension) * tokens + query]
        : 0.0f;
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int group = 0; group < padded_tokens; group += 64) {
        for (int index = thread; index < 4096; index += 128) {
            const int member = index / 32;
            const int source_dimension = index % 32;
            if (member < 64) {
                const int key_index = group + member;
                const size_t source = (static_cast<size_t>(head) * 32 + source_dimension) * tokens + key_index;
                key_tile[member][source_dimension] = key_index < tokens ? qkv[plane + source] : 0.0f;
                value_tile[member][source_dimension] = key_index < tokens ? qkv[2 * plane + source] : 0.0f;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int member = 0; member < 64; ++member) {
            float logit = __fmul_rn(q, key_tile[member][dimension]);
            for (unsigned offset = 16; offset != 0; offset >>= 1u)
                logit = __fadd_rn(logit, __shfl_down_sync(warp_mask, logit, offset));
            logit = __shfl_sync(warp_mask, logit, 0);
            const float probability = approximate_exponential(logit);
            if (dimension == 0) probabilities[warp][member] = probability;
            numerator = __fmaf_rn(fp8_round(probability), value_tile[member][dimension], numerator);
        }
        __syncthreads();
        if (dimension == 0) {
            const float sum = group_denominator(probabilities[warp]);
            denominator = group == 0 ? sum : half_round(__fadd_rn(denominator, sum));
        }
        __syncthreads();
    }
    denominator = __shfl_sync(warp_mask, denominator, 0);
    const float padding = half_round((padded_tokens - tokens) * 0.083984375f);
    denominator = half_round(__fadd_rn(denominator, -padding));
    const float bounded_denominator = isnan(denominator) ? denominator :
        fmaxf(denominator, 6.198883056640625e-05f);
    const float inverse = half_round(__fdiv_rn(1.0f, bounded_denominator));
    if (query < tokens) {
        const size_t channel = static_cast<size_t>(head) * 32 + dimension;
        const float result = half_round(__fmul_rn(half_round(numerator), inverse));
        output[channel * tokens + query] = fp8_round(result);
    }
}

// Experimental 64-thread/query variant. Each thread owns one key in the
// 64-key tile, while the final value reduction keeps the source's key order.
// RW_NR_VIT_TILED_ATTN64=1 enables it for an exact-output A/B measurement.
__global__ __launch_bounds__(64, 2) void global_attention_tiled_64(
    const float* qkv, float* output, int channels, int tokens, int padded_tokens) {
    const int query = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (query >= tokens) return;
    __shared__ float query_tile[32];
    __shared__ float key_tile[64][33];
    __shared__ float value_tile[64][33];
    __shared__ float probability[64];
    __shared__ float denominator_shared;
    const size_t plane = static_cast<size_t>(channels) * tokens;
    if (lane < 32)
        query_tile[lane] = qkv[(static_cast<size_t>(head) * 32 + lane) * tokens + query];
    float numerator = 0.0f;
    for (int group = 0; group < padded_tokens; group += 64) {
        for (int index = lane; index < 2048; index += 64) {
            const int member = index / 32;
            const int dimension = index % 32;
            if (member < 64) {
                const int key_index = group + member;
                const size_t source = static_cast<size_t>(head * 32 + dimension) * tokens + key_index;
                key_tile[member][dimension] = key_index < tokens ? qkv[plane + source] : 0.0f;
                value_tile[member][dimension] = key_index < tokens ? qkv[2 * plane + source] : 0.0f;
            }
        }
        __syncthreads();
        if (lane < 64) {
            float partial[32];
            #pragma unroll
            for (int dimension = 0; dimension < 32; ++dimension)
                partial[dimension] = __fmul_rn(query_tile[dimension], key_tile[lane][dimension]);
            #pragma unroll
            for (int offset = 16; offset != 0; offset >>= 1)
                #pragma unroll
                for (int dimension = 0; dimension < offset; ++dimension)
                    partial[dimension] = __fadd_rn(partial[dimension], partial[dimension + offset]);
            probability[lane] = approximate_exponential(partial[0]);
        }
        __syncthreads();
        if (lane == 0) {
            const float sum = group_denominator(probability);
            denominator_shared = group == 0 ? sum : half_round(__fadd_rn(denominator_shared, sum));
        }
        __syncthreads();
        if (lane < 32)
            for (int key = 0; key < 64; ++key)
                numerator = __fmaf_rn(fp8_round(probability[key]), value_tile[key][lane], numerator);
        __syncthreads();
    }
    float denominator = denominator_shared;
    const float padding = half_round((padded_tokens - tokens) * 0.083984375f);
    denominator = half_round(__fadd_rn(denominator, -padding));
    const float bounded_denominator = isnan(denominator) ? denominator :
        fmaxf(denominator, 6.198883056640625e-05f);
    const float inverse = half_round(__fdiv_rn(1.0f, bounded_denominator));
    if (lane < 32) {
        const size_t channel = static_cast<size_t>(head) * 32 + lane;
        const float result = half_round(__fmul_rn(half_round(numerator), inverse));
        output[channel * tokens + query] = fp8_round(result);
    }
}

bool tiled_global_attention_requested() {
    const char* value = std::getenv("RW_NR_VIT_TILED_ATTN");
    return !value || std::strcmp(value, "0") != 0;
}

bool tiled_global_attention_64_requested() {
    const char* value = std::getenv("RW_NR_VIT_TILED_ATTN64");
    return value && std::strcmp(value, "1") == 0;
}

bool tiled_global_attention_multi_query_requested() {
    const char* value = std::getenv("RW_NR_VIT_TILED_ATTN_MULTI");
    return !value || std::strcmp(value, "0") != 0;
}

bool fused_qkv_requested() {
    const char* value = std::getenv("RW_NR_VIT_FUSED_QKV");
    return value && std::strcmp(value, "1") == 0;
}

Parameter fused_qkv_weight(Context& context, Weights& weights, const LayerSpec& spec,
                           int channels) {
    const std::string key = "vit/" + std::to_string(spec.block) + "/" +
        std::to_string(spec.layer) + "/qkv_fused_v1";
    const std::array<int64_t, 2> configuration = {channels, 3 * channels};
    if (const auto prepared = weights.find_prepared(context, key, configuration)) {
        if (prepared->tensors.size() != 1)
            throw std::invalid_argument("VitQKV fused prepared parameter count mismatch");
        return prepared->tensors[0];
    }
    const auto query = weights.host_e4m3(spec.block, spec.layer, "q");
    const auto key_matrix = weights.host_e4m3(spec.block, spec.layer, "k");
    const auto value = weights.host_e4m3(spec.block, spec.layer, "v");
    const size_t matrix_size = static_cast<size_t>(channels) * channels;
    if (query.size() != matrix_size || key_matrix.size() != matrix_size ||
        value.size() != matrix_size)
        throw std::invalid_argument("VitQKV fused source matrix size mismatch");
    std::vector<float> fused;
    fused.reserve(matrix_size * 3);
    fused.insert(fused.end(), query.begin(), query.end());
    fused.insert(fused.end(), key_matrix.begin(), key_matrix.end());
    fused.insert(fused.end(), value.begin(), value.end());
    const auto parameter = weights.derived(context, key, {3 * channels, channels}, fused);
    const std::array<Parameter, 1> prepared_tensors = {parameter};
    weights.remember_prepared(context, key, configuration, prepared_tensors);
    return parameter;
}

struct MapShape { int channels, height, width, tokens; };
MapShape checked_map(const Tensor& value, const char* operation) {
    if (value.empty() || value.shape.size() != 4 || value.shape[0] != 1 ||
        value.shape[1] <= 0 || value.shape[2] <= 0 || value.shape[3] <= 0 ||
        value.shape[1] > INT_MAX || value.shape[2] > INT_MAX || value.shape[3] > INT_MAX ||
        value.shape[2] > (INT_MAX - 63) / value.shape[3])
        throw std::invalid_argument(std::string(operation) + ": expected a bounded contiguous [1,C,H,W] map");
    // Tensor metadata is public to internal graph builders. Verify that a
    // rewritten shape still describes the actual allocation before indexing.
    value.numel();
    return {static_cast<int>(value.shape[1]), static_cast<int>(value.shape[2]),
            static_cast<int>(value.shape[3]), static_cast<int>(value.shape[2] * value.shape[3])};
}
unsigned blocks_for(size_t count) {
    if (count == 0 || (count - 1) / 256 + 1 > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("ViT pointwise launch size exceeds CUDA grid bounds");
    return static_cast<unsigned>((count - 1) / 256 + 1);
}

Tensor attention(Context& context, const LayerSpec& spec, const Tensor& input) {
    const auto shape = checked_map(input, "VitAttention");
    const int channels = spec.channels;
    if (channels <= 0 || channels % head_dimensions != 0 || channels > INT_MAX / 3 ||
        shape.channels != 3 * channels || channels / head_dimensions > 65535 ||
        spec.input_channels != channels || spec.output_channels != channels)
        throw std::invalid_argument("VitAttention: incompatible channels/head geometry");
    Tensor output = context.empty({1, channels, shape.height, shape.width});
    const int padded = ((shape.tokens + 63) / 64) * 64;
    const char* tensor_core=std::getenv("RW_NR_VIT_TENSOR_CORE");
#if RW_NR_HOPPER_WGMMA_ENABLED
    if (tensor_core && std::strcmp(tensor_core,"1")==0) {
        const dim3 grid((shape.tokens+63)/64,channels/head_dimensions);
        if(hopper_wide_mma_requested())
            global_attention_hopper<32><<<grid,128,0,context.stream()>>>(input.data(),output.data(),channels,shape.tokens,padded);
        else
            global_attention_hopper<8><<<grid,128,0,context.stream()>>>(input.data(),output.data(),channels,shape.tokens,padded);
    }
    else
#endif
    if (tiled_global_attention_multi_query_requested())
        global_attention_tiled_multi_query<<<dim3((shape.tokens + 3) / 4, channels / head_dimensions), 128, 0, context.stream()>>>(
            input.data(), output.data(), channels, shape.tokens, padded);
    else if (tiled_global_attention_64_requested())
        global_attention_tiled_64<<<dim3(shape.tokens, channels / head_dimensions), 64, 0, context.stream()>>>(
            input.data(), output.data(), channels, shape.tokens, padded);
    else if (tiled_global_attention_requested())
        global_attention_tiled<<<dim3(shape.tokens, channels / head_dimensions), head_dimensions, 0, context.stream()>>>(
            input.data(), output.data(), channels, shape.tokens, padded);
    else
        global_attention<<<dim3(shape.tokens, channels / head_dimensions), head_dimensions, 0, context.stream()>>>(
            input.data(), output.data(), channels, shape.tokens, padded);
    cuda_check(cudaGetLastError(), "native NR global ViT attention");
    return output;
}
} // namespace

LayerOutput run_vit(Context& context, Weights& weights, const LayerSpec& spec,
                    const Tensor& input, const LayerOutput* skip) {
    if (!context.owns(input))
        throw std::invalid_argument("ViT input belongs to another native context");
    const auto shape = checked_map(input, "run_vit");
    if (spec.kind == "VitFfnContract" || spec.kind == "VitProjection") {
        if (!skip || !context.owns(skip->main))
            throw std::invalid_argument(spec.kind + ": residual belongs to another native context");
        checked_map(skip->main, "ViT residual");
    }
    if (spec.kind == "VitAttention") return {attention(context, spec, input), {}};
    if (spec.kind != "VitFfnExpand" && spec.kind != "VitFfnContract" &&
        spec.kind != "VitProjection" && spec.kind != "VitQKV")
        throw std::invalid_argument("unimplemented ViT layer kind: " + spec.kind);
    if (spec.input_channels <= 0 || shape.channels != spec.input_channels || spec.output_channels <= 0)
        throw std::invalid_argument("ViT layer input/output channels do not match its frozen plan");

    if (spec.kind == "VitQKV") {
        const int channels = spec.channels;
        if (channels != shape.channels || spec.output_channels != channels ||
            channels % head_dimensions != 0 || channels > INT_MAX / 3 ||
            channels / head_dimensions > 65535)
            throw std::invalid_argument("VitQKV: incompatible channels/head geometry");
        // These three immutable matrices are the exact rows of the source's
        // concatenated projection. Different cuBLAS choices remain observable
        // through the differential captures; no hidden quantization is applied.
        const Parameter temperature = weights.f32(context, spec.block, spec.layer, "header", {channels / head_dimensions});
        context.validate_parameter(temperature);
        Tensor output = context.empty({1, 3 * channels, shape.height, shape.width});
        if (fused_qkv_requested()) {
            const Parameter fused_weight = fused_qkv_weight(context, weights, spec, channels);
            const Tensor projected = linear_nchw(context, input, fused_weight);
            normalise_qkv_fused<<<dim3(shape.tokens, channels / head_dimensions), head_dimensions, 0, context.stream()>>>(
                projected.data(), temperature.data(), output.data(), channels, shape.tokens);
        } else {
            const Parameter query_weight = weights.e4m3(context, spec.block, spec.layer, "q", {channels, channels});
            const Parameter key_weight = weights.e4m3(context, spec.block, spec.layer, "k", {channels, channels});
            const Parameter value_weight = weights.e4m3(context, spec.block, spec.layer, "v", {channels, channels});
            const Tensor query = linear_nchw(context, input, query_weight);
            const Tensor key = linear_nchw(context, input, key_weight);
            const Tensor value = linear_nchw(context, input, value_weight);
            normalise_qkv<<<dim3(shape.tokens, channels / head_dimensions), head_dimensions, 0, context.stream()>>>(
                query.data(), key.data(), value.data(), temperature.data(), output.data(), channels, shape.tokens);
        }
        cuda_check(cudaGetLastError(), "native NR ViT QKV normalization/publication");
        return {output, {}};
    }

    const Parameter weight = weights.e4m3(context, spec.block, spec.layer, "weight",
                                     {spec.output_channels, spec.input_channels});
    const Tensor projection = linear_nchw(context, input, weight);
    Tensor output = context.empty(projection.shape);
    if (spec.kind == "VitFfnExpand") {
        expand_activation<<<blocks_for(projection.numel()), 256, 0, context.stream()>>>(
            projection.data(), output.data(), projection.numel());
        cuda_check(cudaGetLastError(), "native NR ViT FFN cubic/publication");
    } else {
        if (!skip || skip->main.empty() || skip->main.shape != projection.shape)
            throw std::invalid_argument(spec.kind + ": missing or incompatible explicit residual");
        const Parameter gain = weights.f16(context, spec.block, spec.layer, "skip_weight", {spec.output_channels});
        context.validate_parameter(gain);
        gained_residual<<<blocks_for(projection.numel()), 256, 0, context.stream()>>>(
            projection.data(), skip->main.data(), gain.data(), output.data(), output.numel(), shape.tokens);
        cuda_check(cudaGetLastError(), "native NR ViT gained residual/publication");
    }
    return {output, {}};
}
} // namespace rw::native_nr
