// Native CUDA lowering of the pinned 3beef793 stateless Swin arithmetic.
// Matrix operands remain exact decoded E4M3 values; every source publication,
// window boundary, and half-precision reduction below is explicit. This file
// does not assert vendor parity or temporal/production qualification.
// Mathematical/source reference: mauri870/dlssnr-pytorch commit
// 3beef7935a8ef976b47ab78b906e75600c52d2de, dlssnr/layers/swin.py.
//
// MIT License
// Copyright (c) 2026 Mauri de Souza Meneguzzo
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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rw::native_nr {
namespace {
constexpr int kWindow = 8, kTokens = 64, kHeadDimension = 32;
constexpr float kExpSlope = 0.044921875f;
constexpr float kExpOffset = 1.30078125f;
constexpr float kExpLow = 1.03125f, kExpHigh = 1.5693359375f;
constexpr float kNormFloor = 0.00006198883056640625f;
constexpr float kSumFloor = 0.00006103515625f;

__device__ float half_round(float value) {
    return __half2float(__float2half_rn(value));
}
__device__ float fp8_round(float value) {
    __nv_fp8_e4m3 encoded;
    encoded.__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    return static_cast<float>(encoded);
}
__device__ float half_sum(float a, float b) {
    return half_round(__fadd_rn(a, b));
}
__device__ float clamp_preserving_nan(float value, float low, float high) {
    return isnan(value) ? value : fminf(fmaxf(value,low),high);
}
unsigned blocks(size_t count) {
    const size_t result = (count + 255) / 256;
    if (result > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Swin kernel grid exceeds CUDA x dimension");
    return static_cast<unsigned>(result);
}
void launched(const char* name) { cuda_check(cudaGetLastError(), name); }

void swin_gemm_batched(Context& context, cublasOperation_t op_a, cublasOperation_t op_b,
    int m, int n, int k, const float* alpha, const float* const* a, int lda,
    const float* const* b, int ldb, const float* beta, float* const* c, int ldc,
    int batch_count, const char* operation) {
    if (context.half_math()) {
        blas_check(cublasGemmBatchedEx(context.blas(), op_a, op_b, m, n, k,
            static_cast<const void*>(alpha), reinterpret_cast<const void* const*>(a), CUDA_R_32F, lda,
            reinterpret_cast<const void* const*>(b), CUDA_R_32F, ldb, static_cast<const void*>(beta),
            reinterpret_cast<void* const*>(c), CUDA_R_32F, ldc,
            batch_count, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), operation);
    } else {
        blas_check(cublasSgemmBatched(context.blas(), op_a, op_b, m, n, k,
            alpha, a, lda, b, ldb, beta, c, ldc, batch_count), operation);
    }
}

void swin_gemm_strided_batched(Context& context, cublasOperation_t op_a, cublasOperation_t op_b,
    int m, int n, int k, const float* alpha, const float* a, int lda, long long stride_a,
    const float* b, int ldb, long long stride_b, const float* beta, float* c, int ldc,
    long long stride_c, int batch_count, const char* operation) {
    if (context.half_math()) {
        blas_check(cublasGemmStridedBatchedEx(context.blas(), op_a, op_b, m, n, k,
            alpha, a, CUDA_R_32F, lda, stride_a, b, CUDA_R_32F, ldb, stride_b,
            beta, c, CUDA_R_32F, ldc, stride_c, batch_count,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), operation);
    } else {
        blas_check(cublasSgemmStridedBatched(context.blas(), op_a, op_b, m, n, k,
            alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count), operation);
    }
}

void swin_gemm(Context& context, cublasOperation_t op_a, cublasOperation_t op_b,
    int m, int n, int k, const float* alpha, const float* a, int lda,
    const float* b, int ldb, const float* beta, float* c, int ldc, const char* operation) {
    if (context.half_math()) {
        blas_check(cublasGemmEx(context.blas(), op_a, op_b, m, n, k,
            alpha, a, CUDA_R_32F, lda, b, CUDA_R_32F, ldb, beta, c, CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), operation);
    } else {
        blas_check(cublasSgemm(context.blas(), op_a, op_b, m, n, k,
            alpha, a, lda, b, ldb, beta, c, ldc), operation);
    }
}

struct Grid {
    int rows, columns, front_rows, front_columns;
};
bool tiled_window_layout_requested();
bool tiled_window_unpack_requested();
Grid grid_for(int height, int width, int shift_rows, int shift_columns) {
    if (height <= 0 || width <= 0 || (shift_rows != 0 && shift_rows != -1) ||
        (shift_columns != 0 && shift_columns != -1))
        throw std::runtime_error("unsupported Swin window geometry/phase");
    const int front_rows = -4 * shift_rows, front_columns = -4 * shift_columns;
    return {(height + front_rows + 7) / 8, (width + front_columns + 7) / 8,
            front_rows, front_columns};
}

// Token index is [tile_y, tile_x, pixel_y, pixel_x], not row-major 8x8.
__global__ void pack_windows_kernel(const float* input, float* output, size_t count,
    int channels, int height, int width, int grid_columns, int front_rows, int front_columns) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int channel = static_cast<int>(index % channels);
    const size_t token_index = index / channels;
    const int token = static_cast<int>(token_index % 64);
    const size_t window = token_index / 64;
    const int tile_y = token / 32, tile_x = (token / 16) % 2;
    const int y_in_tile = (token / 4) % 4, x_in_tile = token % 4;
    const int row = static_cast<int>(window / grid_columns) * 8 + tile_y * 4 + y_in_tile - front_rows;
    const int column = static_cast<int>(window % grid_columns) * 8 + tile_x * 4 + x_in_tile - front_columns;
    output[index] = row >= 0 && row < height && column >= 0 && column < width
        ? input[(static_cast<size_t>(channel) * height + row) * width + column] : 0.0f;
}
__global__ void pack_windows_quantized_kernel(const float* input, float* output, size_t count,
    int channels, int height, int width, int grid_columns, int front_rows, int front_columns) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int channel = static_cast<int>(index % channels);
    const size_t token_index = index / channels;
    const int token = static_cast<int>(token_index % 64);
    const size_t window = token_index / 64;
    const int tile_y = token / 32, tile_x = (token / 16) % 2;
    const int y_in_tile = (token / 4) % 4, x_in_tile = token % 4;
    const int row = static_cast<int>(window / grid_columns) * 8 + tile_y * 4 + y_in_tile - front_rows;
    const int column = static_cast<int>(window % grid_columns) * 8 + tile_x * 4 + x_in_tile - front_columns;
    float value = row >= 0 && row < height && column >= 0 && column < width
        ? input[(static_cast<size_t>(channel) * height + row) * width + column] : 0.0f;
    if (!isnan(value)) value = fminf(fmaxf(value, -448.0f), 448.0f);
    output[index] = fp8_round(value);
}

template<int Channels, int ChannelBits>
__global__ void pack_windows_quantized_tiled_kernel(const float* input, float* output,
    size_t windows, int height, int width, int grid_columns, int front_rows, int front_columns) {
    const size_t window = static_cast<size_t>(blockIdx.x);
    if (window >= windows) return;
    const size_t token_base = window * static_cast<size_t>(64 * Channels);
    for (int local = static_cast<int>(threadIdx.x); local < 64 * Channels; local += blockDim.x) {
        const int token = local >> ChannelBits;
        const int channel = local & (Channels - 1);
        const int tile_y = token / 32, tile_x = (token / 16) % 2;
        const int y_in_tile = (token / 4) % 4, x_in_tile = token % 4;
        const int row = static_cast<int>(window / static_cast<size_t>(grid_columns)) * 8 +
            tile_y * 4 + y_in_tile - front_rows;
        const int column = static_cast<int>(window % static_cast<size_t>(grid_columns)) * 8 +
            tile_x * 4 + x_in_tile - front_columns;
        float value = row >= 0 && row < height && column >= 0 && column < width
            ? input[(static_cast<size_t>(channel) * height + row) * width + column] : 0.0f;
        if (!isnan(value)) value = fminf(fmaxf(value, -448.0f), 448.0f);
        output[token_base + static_cast<size_t>(local)] = fp8_round(value);
    }
}
__global__ void unpack_windows_kernel(const float* input, float* output, size_t count,
    int channels, int height, int width, int grid_columns, int front_rows, int front_columns) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int column = static_cast<int>(index % width);
    const int row = static_cast<int>((index / width) % height);
    const int channel = static_cast<int>(index / (static_cast<size_t>(height) * width));
    const int padded_row = row + front_rows, padded_column = column + front_columns;
    const int y = padded_row % 8, x = padded_column % 8;
    const int token = ((y / 4) * 2 + x / 4) * 16 + (y % 4) * 4 + x % 4;
    const size_t window = static_cast<size_t>(padded_row / 8) * grid_columns + padded_column / 8;
    output[index] = input[(window * 64 + token) * channels + channel];
}
__global__ void unpack_windows_narrow_kernel(const float* source, float* raw_output,
    float* narrow_output, size_t elements, int channels, int height, int width,
    int grid_columns, int front_rows, int front_columns) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int column = static_cast<int>(index % width);
    const int row = static_cast<int>((index / width) % height);
    const int channel = static_cast<int>(index / (static_cast<size_t>(height) * width));
    const int padded_row = row + front_rows, padded_column = column + front_columns;
    const int y = padded_row % 8, x = padded_column % 8;
    const int token = ((y / 4) * 2 + x / 4) * 16 + (y % 4) * 4 + x % 4;
    const size_t window = static_cast<size_t>(padded_row / 8) * grid_columns + padded_column / 8;
    const float value = source[(window * 64 + token) * channels + channel];
    raw_output[index] = value;
    narrow_output[index] = half_round(value);
}

template<int Channels, int ChannelBits>
__global__ void unpack_windows_narrow_tiled_kernel(const float* source, float* raw_output,
    float* narrow_output, size_t windows, int height, int width, int grid_columns,
    int front_rows, int front_columns) {
    const size_t window = static_cast<size_t>(blockIdx.x);
    if (window >= windows) return;
    const int window_row = static_cast<int>(window / static_cast<size_t>(grid_columns));
    const int window_column = static_cast<int>(window % static_cast<size_t>(grid_columns));
    for (int local = static_cast<int>(threadIdx.x); local < 64 * Channels; local += blockDim.x) {
        const int token = local >> ChannelBits;
        const int channel = local & (Channels - 1);
        const int tile_y = token / 32, tile_x = (token / 16) % 2;
        const int y_in_tile = (token / 4) % 4, x_in_tile = token % 4;
        const int row = window_row * 8 + tile_y * 4 + y_in_tile - front_rows;
        const int column = window_column * 8 + tile_x * 4 + x_in_tile - front_columns;
        if (row < 0 || row >= height || column < 0 || column >= width) continue;
        const size_t output = (static_cast<size_t>(channel) * height + row) * width + column;
        const float value = source[(window * 64ull + static_cast<size_t>(token)) * Channels + channel];
        raw_output[output] = value;
        narrow_output[output] = half_round(value);
    }
}

__global__ void residual_add_unpack_narrow_kernel(
    const float* residual, const float* scale, const float* product,
    float* raw_output, float* narrow_output, size_t elements, int channels,
    int height, int width, int grid_columns, int front_rows, int front_columns) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int column = static_cast<int>(index % width);
    const int row = static_cast<int>((index / width) % height);
    const int channel = static_cast<int>(index / (static_cast<size_t>(height) * width));
    const int padded_row = row + front_rows, padded_column = column + front_columns;
    const int y = padded_row % 8, x = padded_column % 8;
    const int token = ((y / 4) * 2 + x / 4) * 16 + (y % 4) * 4 + x % 4;
    const size_t window = static_cast<size_t>(padded_row / 8) * grid_columns + padded_column / 8;
    const size_t source = (window * 64ull + static_cast<size_t>(token)) *
        static_cast<size_t>(channels) + static_cast<size_t>(channel);
    const float value = __fadd_rn(__fmul_rn(residual[source], scale[channel]), product[source]);
    raw_output[index] = value;
    narrow_output[index] = half_round(value);
}
Tensor pack_windows(Context& context, const Tensor& input, const Grid& grid) {
    const int channels = static_cast<int>(input.shape[1]);
    const int height = static_cast<int>(input.shape[2]), width = static_cast<int>(input.shape[3]);
    auto output = context.empty({static_cast<int64_t>(grid.rows) * grid.columns, 64, channels});
    pack_windows_kernel<<<blocks(output.numel()),256,0,context.stream()>>>(input.data(),output.data(),output.numel(),
        channels,height,width,grid.columns,grid.front_rows,grid.front_columns);
    launched("Swin pack windows");
    return output;
}
Tensor pack_windows_quantized(Context& context, const Tensor& input, const Grid& grid) {
    const int channels = static_cast<int>(input.shape[1]);
    const int height = static_cast<int>(input.shape[2]), width = static_cast<int>(input.shape[3]);
    auto output = context.empty({static_cast<int64_t>(grid.rows) * grid.columns, 64, channels});
    if (tiled_window_layout_requested()) {
        const size_t windows = static_cast<size_t>(grid.rows) * static_cast<size_t>(grid.columns);
        switch (channels) {
        case 32: pack_windows_quantized_tiled_kernel<32,5><<<static_cast<unsigned>(windows),256,0,context.stream()>>>(
            input.data(),output.data(),windows,height,width,grid.columns,grid.front_rows,grid.front_columns); break;
        case 64: pack_windows_quantized_tiled_kernel<64,6><<<static_cast<unsigned>(windows),256,0,context.stream()>>>(
            input.data(),output.data(),windows,height,width,grid.columns,grid.front_rows,grid.front_columns); break;
        case 128: pack_windows_quantized_tiled_kernel<128,7><<<static_cast<unsigned>(windows),256,0,context.stream()>>>(
            input.data(),output.data(),windows,height,width,grid.columns,grid.front_rows,grid.front_columns); break;
        case 256: pack_windows_quantized_tiled_kernel<256,8><<<static_cast<unsigned>(windows),256,0,context.stream()>>>(
            input.data(),output.data(),windows,height,width,grid.columns,grid.front_rows,grid.front_columns); break;
        default: throw std::runtime_error("unsupported tiled Swin channel width");
        }
    } else {
        pack_windows_quantized_kernel<<<blocks(output.numel()),256,0,context.stream()>>>(input.data(),output.data(),output.numel(),
            channels,height,width,grid.columns,grid.front_rows,grid.front_columns);
    }
    launched("Swin pack and quantize windows");
    return output;
}
Tensor unpack_windows(Context& context, const Tensor& input, const Grid& grid, int height, int width) {
    const int channels = static_cast<int>(input.shape.back());
    auto output = context.empty({1,channels,height,width});
    unpack_windows_kernel<<<blocks(output.numel()),256,0,context.stream()>>>(input.data(),output.data(),output.numel(),
        channels,height,width,grid.columns,grid.front_rows,grid.front_columns);
    launched("Swin unpack windows");
    return output;
}
void unpack_windows_narrow(Context& context, const Tensor& input, const Grid& grid,
                           int height, int width, Tensor& raw, Tensor& narrowed) {
    const int channels = static_cast<int>(input.shape.back());
    raw = context.empty({1,channels,height,width});
    narrowed = context.empty(raw.shape);
    const size_t count = raw.numel();
    if (tiled_window_unpack_requested()) {
        const size_t windows = static_cast<size_t>(grid.rows) * static_cast<size_t>(grid.columns);
        switch (channels) {
        case 32: unpack_windows_narrow_tiled_kernel<32,5><<<static_cast<unsigned>(windows),256,0,context.stream()>>>(
            input.data(),raw.data(),narrowed.data(),windows,height,width,grid.columns,grid.front_rows,grid.front_columns); break;
        case 64: unpack_windows_narrow_tiled_kernel<64,6><<<static_cast<unsigned>(windows),256,0,context.stream()>>>(
            input.data(),raw.data(),narrowed.data(),windows,height,width,grid.columns,grid.front_rows,grid.front_columns); break;
        case 128: unpack_windows_narrow_tiled_kernel<128,7><<<static_cast<unsigned>(windows),256,0,context.stream()>>>(
            input.data(),raw.data(),narrowed.data(),windows,height,width,grid.columns,grid.front_rows,grid.front_columns); break;
        case 256: unpack_windows_narrow_tiled_kernel<256,8><<<static_cast<unsigned>(windows),256,0,context.stream()>>>(
            input.data(),raw.data(),narrowed.data(),windows,height,width,grid.columns,grid.front_rows,grid.front_columns); break;
        default: throw std::runtime_error("unsupported tiled Swin channel width");
        }
    } else {
        unpack_windows_narrow_kernel<<<blocks(count),256,0,context.stream()>>>(input.data(),raw.data(),narrowed.data(),count,channels,height,width,
            grid.columns,grid.front_rows,grid.front_columns);
    }
    launched("Swin unpack and narrow windows");
}

bool fused_residual_unpack_requested() {
    const char* value = std::getenv("RW_NR_SWIN_FUSED_RESIDUAL_UNPACK");
    return value && std::strcmp(value, "1") == 0;
}

void residual_add_unpack_narrow(Context& context, const Tensor& residual,
    const Parameter& scale, const Tensor& product, const Grid& grid,
    int height, int width, Tensor& raw, Tensor& narrowed) {
    if (!context.owns(residual) || !context.owns(product) || residual.shape != product.shape ||
        residual.shape.size() != 3 || residual.shape[1] != 64 ||
        scale.numel() != static_cast<size_t>(residual.shape[2]))
        throw std::runtime_error("Swin fused residual/unpack shape mismatch");
    context.validate_parameter(scale);
    const int channels = static_cast<int>(residual.shape[2]);
    raw = context.empty({1, channels, height, width});
    narrowed = context.empty(raw.shape);
    residual_add_unpack_narrow_kernel<<<blocks(raw.numel()),256,0,context.stream()>>>(
        residual.data(), scale.data(), product.data(), raw.data(), narrowed.data(),
        raw.numel(), channels, height, width, grid.columns, grid.front_rows, grid.front_columns);
    launched("Swin fused residual add/unpack/narrow");
}

__global__ void residual_add_kernel(const float* residual, const float* scale,
    const float* product, float* output, size_t count, int channels) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = __fadd_rn(__fmul_rn(residual[index],scale[index % channels]),product[index]);
}
Tensor residual_add(Context& context, const Tensor& residual, const Parameter& scale, const Tensor& product) {
    if (!context.owns(residual) || !context.owns(product) ||
        residual.shape.empty() || residual.shape != product.shape ||
        scale.numel() != static_cast<size_t>(residual.shape.back()))
        throw std::runtime_error("Swin residual add shape mismatch");
    context.validate_parameter(scale);
    auto output = context.empty(product.shape);
    residual_add_kernel<<<blocks(output.numel()),256,0,context.stream()>>>(residual.data(),scale.data(),product.data(),
        output.data(),output.numel(),static_cast<int>(product.shape.back()));
    launched("Swin residual add");
    return output;
}

__global__ void residual_add_input_kernel(const float* input, const float* scale,
    const float* product, float* output, size_t count, int channels, int height,
    int width, int grid_columns, int front_rows, int front_columns) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const size_t token_index = index / static_cast<size_t>(channels);
    const int channel = static_cast<int>(index % static_cast<size_t>(channels));
    const int token = static_cast<int>(token_index % 64);
    const size_t window = token_index / 64;
    const int tile_y = token / 32, tile_x = (token / 16) % 2;
    const int y_in_tile = (token / 4) % 4, x_in_tile = token % 4;
    const int row = static_cast<int>(window / grid_columns) * 8 + tile_y * 4 + y_in_tile - front_rows;
    const int column = static_cast<int>(window % grid_columns) * 8 + tile_x * 4 + x_in_tile - front_columns;
    const float residual = row >= 0 && row < height && column >= 0 && column < width
        ? input[(static_cast<size_t>(channel) * height + row) * width + column] : 0.0f;
    output[index] = __fadd_rn(__fmul_rn(residual, scale[channel]), product[index]);
}

Tensor residual_add_input(Context& context, const Tensor& input, const Grid& grid,
                          const Parameter& scale, const Tensor& product) {
    const int channels = static_cast<int>(input.shape[1]);
    const int height = static_cast<int>(input.shape[2]), width = static_cast<int>(input.shape[3]);
    auto output = context.empty(product.shape);
    residual_add_input_kernel<<<blocks(output.numel()),256,0,context.stream()>>>(input.data(),scale.data(),product.data(),output.data(),
        output.numel(),channels,height,width,grid.columns,grid.front_rows,grid.front_columns);
    launched("Swin direct NCHW residual add");
    return output;
}

bool direct_residual_add_requested() {
    const char* value = std::getenv("RW_NR_SWIN_DIRECT_RESIDUAL");
    return !value || std::strcmp(value, "0") != 0;
}

bool strided_middle_gemm_requested() {
    const char* value = std::getenv("RW_NR_SWIN_MID_STRIDED_BATCH");
    return value && std::strcmp(value, "1") == 0;
}

// Source norm: 4 interleaved FP32 FMA chains for each of two halves, then
// ((chain0+chain1)+(chain2+chain3)) in FP32, each half narrowed, f16 half sum.
__device__ float norm_scale(const float* vector) {
    float chains[4][2] = {};
    for (int group = 0; group < 2; ++group)
        for (int fragment = 0; fragment < 2; ++fragment)
            for (int chain = 0; chain < 4; ++chain)
                for (int half = 0; half < 2; ++half) {
                    const float value = vector[fragment * 16 + group * 8 + chain * 2 + half];
                    chains[chain][half] = __fmaf_rn(value,value,chains[chain][half]);
                }
    float halves[2];
    for (int half = 0; half < 2; ++half)
        halves[half] = half_round(__fadd_rn(
            __fadd_rn(chains[0][half],chains[1][half]),
            __fadd_rn(chains[2][half],chains[3][half])));
    const float half_total=half_sum(halves[0],halves[1]);
    const float total=isnan(half_total)?half_total:fmaxf(half_total,kNormFloor);
    const float root = __double2float_rn(sqrt(static_cast<double>(total)));
    return __fdiv_rn(1.0f,root);
}
__global__ void normalize_qkv_kernel(const float* input, const float* head_scale,
    float* output, size_t vector_count, int heads) {
    const size_t vector_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector_index >= vector_count) return;
    const int head = static_cast<int>(vector_index % heads);
    const float* source = input + vector_index * 96;
    float* destination = output + vector_index * 96;
    const float query_scale = __fmul_rn(norm_scale(source),head_scale[head]);
    const float key_scale = norm_scale(source + 32);
    for (int dimension = 0; dimension < 32; ++dimension) {
        destination[dimension] = fp8_round(__fmul_rn(source[dimension],query_scale));
        destination[32+dimension] = fp8_round(__fmul_rn(source[32+dimension],key_scale));
        destination[64+dimension] = fp8_round(half_round(source[64+dimension]));
    }
}
__global__ void attention_numerator_kernel(const float* logits, const float* exponent_bias,
    float* numerator, size_t count, int heads) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int key = static_cast<int>(index % 64), query = static_cast<int>((index / 64) % 64);
    const int head = static_cast<int>((index / 4096) % heads);
    const float bias = exponent_bias[(static_cast<size_t>(head) * 64 + query) * 64 + key];
    const float ramp=clamp_preserving_nan(half_round(__fmaf_rn(logits[index],kExpSlope,bias)),kExpLow,kExpHigh);
    const unsigned short bits = __half_as_ushort(__float2half_rn(ramp));
    numerator[index] = __half2float(__ushort_as_half(static_cast<unsigned short>((bits & 0x3ffu) << 5u)));
}
__global__ void attention_probability_kernel(const float* numerator, float* probability,
    size_t query_count) {
    const size_t query = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query >= query_count) return;
    const float* row = numerator + query * 64;
    float pairs[4][2][2];
    for (int pair = 0; pair < 4; ++pair)
        for (int member = 0; member < 2; ++member)
            for (int half = 0; half < 2; ++half) {
                const int key = pair * 4 + member * 2 + half;
                float total = row[key];
                total = half_sum(total,row[16+key]);
                total = half_sum(total,row[32+key]);
                pairs[pair][member][half] = half_sum(total,row[48+key]);
            }
    float members[2];
    for (int member = 0; member < 2; ++member) {
        float halves[2];
        for (int half = 0; half < 2; ++half) {
            float sum = half_sum(pairs[0][member][half],pairs[1][member][half]);
            sum = half_sum(sum,pairs[2][member][half]);
            halves[half] = half_sum(sum,pairs[3][member][half]);
        }
        members[member] = half_sum(halves[0],halves[1]);
    }
    const float denominator = half_sum(members[0],members[1]);
    const float clamped=isnan(denominator)?denominator:fmaxf(denominator,kSumFloor);
    const float reciprocal=__fdiv_rn(1.0f,clamped);
    for (int key = 0; key < 64; ++key)
        probability[query*64+key] = fp8_round(half_round(__fmul_rn(row[key],reciprocal)));
}

// Head-1 windows dominate the full-resolution 1080p path.  The reference
// implementation materialises QK logits, the ordered softmax probabilities,
// and the PV product through three global-memory phases plus two batched GEMM
// calls.  For one head, one thread owns a query row, so the complete source
// ordered reduction can be kept in a 64x64 shared tile and the intermediate
// tensors disappear.  It is enabled by default for the H200 native path;
// RW_NR_SWIN_FUSED_ATTN=0 selects the reference batched-GEMM path.
__global__ __launch_bounds__(64, 2) void fused_window_attention_head1(
    const float* input, const float* exponent_bias, float* output, size_t windows) {
    const size_t window = static_cast<size_t>(blockIdx.x);
    const int query = static_cast<int>(threadIdx.x);
    if (window >= windows || query >= 64) return;
    __shared__ float tile[64][96];
    __shared__ float probability[64][64];
    const size_t token_base = window * 64ull * 96ull;
    float query_vector[32];
    for (int dimension = 0; dimension < 96; ++dimension)
        tile[query][dimension] = input[token_base + static_cast<size_t>(query) * 96ull + dimension];
    __syncthreads();
    #pragma unroll
    for (int dimension = 0; dimension < 32; ++dimension)
        query_vector[dimension] = tile[query][dimension];
    for (int key = 0; key < 64; ++key) {
        float logit = 0.0f;
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension)
            logit = __fadd_rn(logit, __fmul_rn(query_vector[dimension],
                                                tile[key][32 + dimension]));
        const float bias = exponent_bias[static_cast<size_t>(query) * 64ull + key];
        const float ramp = clamp_preserving_nan(
            half_round(__fmaf_rn(logit, kExpSlope, bias)), kExpLow, kExpHigh);
        const unsigned short bits = __half_as_ushort(__float2half_rn(ramp));
        const float numerator = __half2float(__ushort_as_half(
            static_cast<unsigned short>((bits & 0x3ffu) << 5u)));
        probability[query][key] = numerator;
    }
    // Match attention_probability_kernel's fixed source grouping and half
    // rounding rather than relying on a compiler-selected reduction tree.
    float pairs[4][2][2];
    for (int pair = 0; pair < 4; ++pair)
        for (int member = 0; member < 2; ++member)
            for (int half = 0; half < 2; ++half) {
                const int key = pair * 4 + member * 2 + half;
                float total = probability[query][key];
                total = half_sum(total, probability[query][16 + key]);
                total = half_sum(total, probability[query][32 + key]);
                pairs[pair][member][half] = half_sum(total, probability[query][48 + key]);
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
    const float reciprocal = __fdiv_rn(1.0f, clamped);
    for (int key = 0; key < 64; ++key)
        probability[query][key] = fp8_round(half_round(
            __fmul_rn(probability[query][key], reciprocal)));
    for (int dimension = 0; dimension < 32; ++dimension) {
        float accumulator = 0.0f;
        for (int key = 0; key < 64; ++key)
            accumulator = __fmaf_rn(probability[query][key],
                                    tile[key][64 + dimension], accumulator);
        output[(window * 64ull + static_cast<size_t>(query)) * 32ull + dimension] = accumulator;
    }
}

// Head-1 variant with 128 loader/query threads.  The first 64 threads retain
// the exact head1 query, reduction and publication order; the extra warp only
// cooperatively fills the shared Q/K/V tile so the large 1080p windows do not
// make each query lane perform a strided 96-float load sequence.
__global__ __launch_bounds__(128, 2) void fused_window_attention_head1_128(
    const float* raw_input, const float* head_scale, const float* exponent_bias,
    float* output, size_t windows) {
    const size_t window = static_cast<size_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (window >= windows) return;
    __shared__ float tile[64][96];
    __shared__ float probability[64][64];
    const size_t token_base = window * 64ull * 96ull;
    for (int index = lane; index < 64 * 96; index += 128)
        tile[index / 96][index % 96] = raw_input[token_base + static_cast<size_t>(index)];
    __syncthreads();
    if (lane < 64) {
        const float query_scale = __fmul_rn(norm_scale(tile[lane]), head_scale[0]);
        const float key_scale = norm_scale(tile[lane] + 32);
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension) {
            tile[lane][dimension] = fp8_round(__fmul_rn(tile[lane][dimension], query_scale));
            tile[lane][32 + dimension] = fp8_round(__fmul_rn(tile[lane][32 + dimension], key_scale));
            tile[lane][64 + dimension] = fp8_round(half_round(tile[lane][64 + dimension]));
        }
    }
    __syncthreads();
    if (lane >= 64) return;
    float query_vector[32];
    #pragma unroll
    for (int dimension = 0; dimension < 32; ++dimension)
        query_vector[dimension] = tile[lane][dimension];
    for (int key = 0; key < 64; ++key) {
        float logit = 0.0f;
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension)
            logit = __fadd_rn(logit, __fmul_rn(query_vector[dimension], tile[key][32 + dimension]));
        const float bias = exponent_bias[static_cast<size_t>(lane) * 64ull + key];
        const float ramp = clamp_preserving_nan(half_round(__fmaf_rn(logit, kExpSlope, bias)), kExpLow, kExpHigh);
        const unsigned short bits = __half_as_ushort(__float2half_rn(ramp));
        probability[lane][key] = __half2float(__ushort_as_half(static_cast<unsigned short>((bits & 0x3ffu) << 5u)));
    }
    float pairs[4][2][2];
    for (int pair = 0; pair < 4; ++pair)
        for (int member = 0; member < 2; ++member)
            for (int half = 0; half < 2; ++half) {
                const int key = pair * 4 + member * 2 + half;
                float total = probability[lane][key];
                total = half_sum(total, probability[lane][16 + key]);
                total = half_sum(total, probability[lane][32 + key]);
                pairs[pair][member][half] = half_sum(total, probability[lane][48 + key]);
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
    const float reciprocal = __fdiv_rn(1.0f, clamped);
    for (int key = 0; key < 64; ++key)
        probability[lane][key] = fp8_round(half_round(__fmul_rn(probability[lane][key], reciprocal)));
    for (int dimension = 0; dimension < 32; ++dimension) {
        float accumulator = 0.0f;
        for (int key = 0; key < 64; ++key)
            accumulator = __fmaf_rn(probability[lane][key], tile[key][64 + dimension], accumulator);
        output[(window * 64ull + static_cast<size_t>(lane)) * 32ull + dimension] = accumulator;
    }
}

// General window variant for heads 2/4/8.  A 256-thread block cooperatively
// loads one head's K/V tile, then the first 64 threads each own a query row.
// It removes the QK/softmax/PV intermediate tensors just like the head-1 path;
// RW_NR_SWIN_FUSED_ALL_ATTN=0 keeps the existing cuBLAS batched path available
// for differential qualification.
__global__ __launch_bounds__(256, 2) void fused_window_attention_all_heads(
    const float* input, const float* exponent_bias, float* output,
    size_t windows, int heads, int channels) {
    const size_t batch = static_cast<size_t>(blockIdx.x);
    const size_t window = batch / static_cast<size_t>(heads);
    const int head = static_cast<int>(batch % static_cast<size_t>(heads));
    const int lane = static_cast<int>(threadIdx.x);
    if (window >= windows) return;
    __shared__ float key_tile[64][33];
    __shared__ float value_tile[64][33];
    __shared__ float probability[64][64];
    const size_t token_base = window * 64ull * static_cast<size_t>(heads) * 96ull;
    for (int index = lane; index < 64 * 64; index += 256) {
        const int token = index / 64;
        const int component = index % 64;
        const size_t source = token_base + static_cast<size_t>(token) * heads * 96ull +
            static_cast<size_t>(head) * 96ull + 32ull +
            static_cast<size_t>(component < 32 ? component : 32 + component - 32);
        if (component < 32) key_tile[token][component] = input[source];
        else value_tile[token][component - 32] = input[source];
    }
    __syncthreads();
    if (lane >= 64) return;
    const size_t query_base = token_base + static_cast<size_t>(lane) * heads * 96ull +
        static_cast<size_t>(head) * 96ull;
    float query_vector[32];
    #pragma unroll
    for (int dimension = 0; dimension < 32; ++dimension)
        query_vector[dimension] = input[query_base + dimension];
    for (int key = 0; key < 64; ++key) {
        float logit = 0.0f;
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension)
            logit = __fadd_rn(logit, __fmul_rn(query_vector[dimension], key_tile[key][dimension]));
        const float bias = exponent_bias[(static_cast<size_t>(head) * 64ull + lane) * 64ull + key];
        const float ramp = clamp_preserving_nan(
            half_round(__fmaf_rn(logit, kExpSlope, bias)), kExpLow, kExpHigh);
        const unsigned short bits = __half_as_ushort(__float2half_rn(ramp));
        probability[lane][key] = __half2float(__ushort_as_half(
            static_cast<unsigned short>((bits & 0x3ffu) << 5u)));
    }
    float pairs[4][2][2];
    for (int pair = 0; pair < 4; ++pair)
        for (int member = 0; member < 2; ++member)
            for (int half = 0; half < 2; ++half) {
                const int key = pair * 4 + member * 2 + half;
                float total = probability[lane][key];
                total = half_sum(total, probability[lane][16 + key]);
                total = half_sum(total, probability[lane][32 + key]);
                pairs[pair][member][half] = half_sum(total, probability[lane][48 + key]);
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
    const float reciprocal = __fdiv_rn(1.0f, clamped);
    for (int key = 0; key < 64; ++key)
        probability[lane][key] = fp8_round(half_round(
            __fmul_rn(probability[lane][key], reciprocal)));
    for (int dimension = 0; dimension < 32; ++dimension) {
        float accumulator = 0.0f;
        for (int key = 0; key < 64; ++key)
            accumulator = __fmaf_rn(probability[lane][key], value_tile[key][dimension], accumulator);
        output[(window * 64ull + static_cast<size_t>(lane)) * static_cast<size_t>(channels) +
               static_cast<size_t>(head * 32 + dimension)] = accumulator;
    }
}

__global__ __launch_bounds__(256, 2) void fused_window_attention_all_heads_raw(
    const float* raw_input, const float* head_scale, const float* exponent_bias,
    float* output, size_t windows, int heads, int channels) {
    const size_t batch = static_cast<size_t>(blockIdx.x);
    const size_t window = batch / static_cast<size_t>(heads);
    const int head = static_cast<int>(batch % static_cast<size_t>(heads));
    const int lane = static_cast<int>(threadIdx.x);
    if (window >= windows) return;
    __shared__ float key_tile[64][33];
    __shared__ float value_tile[64][33];
    __shared__ float probability[64][64];
    __shared__ float query_tile[64][32];
    const size_t token_base = window * 64ull * static_cast<size_t>(heads) * 96ull;
    for (int index = lane; index < 64 * 64; index += 256) {
        const int token = index / 64;
        const int component = index % 64;
        const size_t vector = token_base + static_cast<size_t>(token) * heads * 96ull +
            static_cast<size_t>(head) * 96ull;
        if (component < 32)
            key_tile[token][component] = raw_input[vector + 32ull + static_cast<size_t>(component)];
        else
            value_tile[token][component - 32] = fp8_round(half_round(
                raw_input[vector + 64ull + static_cast<size_t>(component - 32)]));
    }
    __syncthreads();
    if (lane < 64) {
        const size_t query_base = token_base + static_cast<size_t>(lane) * heads * 96ull +
            static_cast<size_t>(head) * 96ull;
        const float query_scale = __fmul_rn(norm_scale(raw_input + query_base), head_scale[head]);
        const float key_scale = norm_scale(key_tile[lane]);
        for (int dimension = 0; dimension < 32; ++dimension) {
            key_tile[lane][dimension] = fp8_round(__fmul_rn(key_tile[lane][dimension], key_scale));
        }
        for (int dimension = 0; dimension < 32; ++dimension)
            query_tile[lane][dimension] = fp8_round(__fmul_rn(
                raw_input[query_base + dimension], query_scale));
    }
    __syncthreads();
    if (lane >= 64) return;
    const size_t query_base = token_base + static_cast<size_t>(lane) * heads * 96ull +
        static_cast<size_t>(head) * 96ull;
    (void)query_base;
    for (int key = 0; key < 64; ++key) {
        float logit = 0.0f;
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension)
            logit = __fadd_rn(logit, __fmul_rn(query_tile[lane][dimension], key_tile[key][dimension]));
        const float bias = exponent_bias[(static_cast<size_t>(head) * 64ull + lane) * 64ull + key];
        const float ramp = clamp_preserving_nan(
            half_round(__fmaf_rn(logit, kExpSlope, bias)), kExpLow, kExpHigh);
        const unsigned short bits = __half_as_ushort(__float2half_rn(ramp));
        probability[lane][key] = __half2float(__ushort_as_half(
            static_cast<unsigned short>((bits & 0x3ffu) << 5u)));
    }
    float pairs[4][2][2];
    for (int pair = 0; pair < 4; ++pair)
        for (int member = 0; member < 2; ++member)
            for (int half = 0; half < 2; ++half) {
                const int key = pair * 4 + member * 2 + half;
                float total = probability[lane][key];
                total = half_sum(total, probability[lane][16 + key]);
                total = half_sum(total, probability[lane][32 + key]);
                pairs[pair][member][half] = half_sum(total, probability[lane][48 + key]);
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
    const float reciprocal = __fdiv_rn(1.0f, clamped);
    for (int key = 0; key < 64; ++key)
        probability[lane][key] = fp8_round(half_round(
            __fmul_rn(probability[lane][key], reciprocal)));
    for (int dimension = 0; dimension < 32; ++dimension) {
        float accumulator = 0.0f;
        for (int key = 0; key < 64; ++key)
            accumulator = __fmaf_rn(probability[lane][key], value_tile[key][dimension], accumulator);
        output[(window * 64ull + static_cast<size_t>(lane)) * static_cast<size_t>(channels) +
               static_cast<size_t>(head * 32 + dimension)] = accumulator;
    }
}

// Same source arithmetic as fused_window_attention_all_heads_raw, but one
// warp pair (64 threads) owns the complete block.  The previous 256-thread
// launch used 192 threads only for the cooperative K/V load and left them
// idle for the query/softmax/PV work.  Keeping one lane per query preserves
// the source reduction order while allowing the scheduler to resident more
// independent windows.
template<int Threads, bool SharedQuery, bool FixedHead1 = false, int ProbabilityStride = 64,
         bool CooperativePV = false>
__global__ __launch_bounds__(Threads, 2) void fused_window_attention_all_heads_raw_compact(
    const float* raw_input, const float* head_scale, const float* exponent_bias,
    float* output, size_t windows, int heads, int channels) {
    const size_t batch = static_cast<size_t>(blockIdx.x);
    const size_t window = FixedHead1 ? batch : batch / static_cast<size_t>(heads);
    const int head = FixedHead1 ? 0 : static_cast<int>(batch % static_cast<size_t>(heads));
    const int lane = static_cast<int>(threadIdx.x);
    if (window >= windows) return;
    __shared__ float key_tile[64][33];
    __shared__ float value_tile[64][33];
    // One thread owns one query row. A stride of 64 maps every lane's
    // probability[row][key] to the same bank; stride 65 distributes them
    // across all 32 banks without changing any arithmetic or reduction order.
    __shared__ float probability[64][ProbabilityStride];
    extern __shared__ float shared_query[];
    float (*query_tile)[32] = reinterpret_cast<float (*)[32]>(shared_query);
    const size_t token_base = window * 64ull * (FixedHead1 ? 96ull : static_cast<size_t>(heads) * 96ull);
    for (int index = lane; index < 64 * 64; index += Threads) {
        const int token = index / 64;
        const int component = index % 64;
        const size_t vector = token_base + static_cast<size_t>(token) * (FixedHead1 ? 96ull : static_cast<size_t>(heads) * 96ull) +
            static_cast<size_t>(head) * 96ull;
        if (component < 32)
            key_tile[token][component] = raw_input[vector + 32ull + static_cast<size_t>(component)];
        else
            value_tile[token][component - 32] = fp8_round(half_round(
                raw_input[vector + 64ull + static_cast<size_t>(component - 32)]));
    }
    __syncthreads();
    float query_vector[32];
    if (lane < 64) {
        const size_t query_base = token_base + static_cast<size_t>(lane) * (FixedHead1 ? 96ull : static_cast<size_t>(heads) * 96ull) +
            static_cast<size_t>(head) * 96ull;
        const float query_scale = __fmul_rn(norm_scale(raw_input + query_base), head_scale[head]);
        const float key_scale = norm_scale(key_tile[lane]);
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension) {
            key_tile[lane][dimension] = fp8_round(__fmul_rn(key_tile[lane][dimension], key_scale));
            const float query = fp8_round(__fmul_rn(raw_input[query_base + dimension], query_scale));
            if constexpr (SharedQuery) query_tile[lane][dimension] = query;
            else query_vector[dimension] = query;
        }
    }
    __syncthreads();
    if constexpr (!CooperativePV) {
        if (lane >= 64) return;
    }
    if (lane < 64) {
        for (int key = 0; key < 64; ++key) {
            float logit = 0.0f;
            #pragma unroll
            for (int dimension = 0; dimension < 32; ++dimension)
                if constexpr (SharedQuery)
                    logit = __fadd_rn(logit, __fmul_rn(query_tile[lane][dimension], key_tile[key][dimension]));
                else
                    logit = __fadd_rn(logit, __fmul_rn(query_vector[dimension], key_tile[key][dimension]));
            const float bias = exponent_bias[(static_cast<size_t>(head) * 64ull + lane) * 64ull + key];
            const float ramp = clamp_preserving_nan(
                half_round(__fmaf_rn(logit, kExpSlope, bias)), kExpLow, kExpHigh);
            const unsigned short bits = __half_as_ushort(__float2half_rn(ramp));
            probability[lane][key] = __half2float(__ushort_as_half(
                static_cast<unsigned short>((bits & 0x3ffu) << 5u)));
        }
        float pairs[4][2][2];
        for (int pair = 0; pair < 4; ++pair)
            for (int member = 0; member < 2; ++member)
                for (int half = 0; half < 2; ++half) {
                    const int key = pair * 4 + member * 2 + half;
                    float total = probability[lane][key];
                    total = half_sum(total, probability[lane][16 + key]);
                    total = half_sum(total, probability[lane][32 + key]);
                    pairs[pair][member][half] = half_sum(total, probability[lane][48 + key]);
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
        const float reciprocal = __fdiv_rn(1.0f, clamped);
        for (int key = 0; key < 64; ++key)
            probability[lane][key] = fp8_round(half_round(
                __fmul_rn(probability[lane][key], reciprocal)));
    }
    if constexpr (CooperativePV) {
        __syncthreads();
        // A warp now owns the 32 output channels of one query. Probability
        // reads broadcast, V reads and output writes are contiguous, and each
        // accumulator visits the same 64 keys in the original FMA order.
        for (int index = lane; index < 64 * 32; index += Threads) {
            const int query = index / 32, dimension = index % 32;
            float accumulator = 0.0f;
            for (int key = 0; key < 64; ++key)
                accumulator = __fmaf_rn(probability[query][key], value_tile[key][dimension], accumulator);
            output[(window * 64ull + static_cast<size_t>(query)) * static_cast<size_t>(FixedHead1 ? 32 : channels) +
                   static_cast<size_t>(head * 32 + dimension)] = accumulator;
        }
    } else {
        for (int dimension = 0; dimension < 32; ++dimension) {
            float accumulator = 0.0f;
            for (int key = 0; key < 64; ++key)
                accumulator = __fmaf_rn(probability[lane][key], value_tile[key][dimension], accumulator);
            output[(window * 64ull + static_cast<size_t>(lane)) * static_cast<size_t>(FixedHead1 ? 32 : channels) +
                   static_cast<size_t>(head * 32 + dimension)] = accumulator;
        }
    }
}

// One window/head per CTA, one query per warp. QK lanes own different keys;
// PV lanes own different channels. Every dot/FMA and half reduction retains
// the source order. K/V normalization is shared by all 64 queries, and the
// raw K tile is reused for normalized Q after K has been transposed.
__global__ __launch_bounds__(256, 2) void fused_window_attention_warp_queries(
    const float* raw_input, const float* head_scale, const float* exponent_bias,
    float* output, size_t windows, int heads, int channels) {
    const size_t batch = static_cast<size_t>(blockIdx.x);
    const size_t window = batch / static_cast<size_t>(heads);
    const int head = static_cast<int>(batch % static_cast<size_t>(heads));
    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread % 32;
    if (window >= windows) return;
    __shared__ float raw_key[64][33];
    __shared__ float key_transposed[32][65];
    __shared__ float value[64][33];
    __shared__ float probability[64][65];
    __shared__ float reciprocal[64];
    const size_t token_base = window * 64ull * static_cast<size_t>(heads) * 96ull;
    for (int index = thread; index < 64 * 64; index += 256) {
        const int token = index / 64, component = index % 64;
        const size_t vector = token_base + static_cast<size_t>(token) * heads * 96ull + head * 96ull;
        if (component < 32) raw_key[token][component] = raw_input[vector + 32 + component];
        else value[token][component - 32] = fp8_round(half_round(raw_input[vector + 32 + component]));
    }
    __syncthreads();
    if (thread < 64) {
        const float scale = norm_scale(raw_key[thread]);
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension)
            key_transposed[dimension][thread] = fp8_round(__fmul_rn(raw_key[thread][dimension], scale));
    }
    __syncthreads();
    if (thread < 64) {
        const size_t base = token_base + static_cast<size_t>(thread) * heads * 96ull + head * 96ull;
        const float scale = __fmul_rn(norm_scale(raw_input + base), head_scale[head]);
        #pragma unroll
        for (int dimension = 0; dimension < 32; ++dimension)
            raw_key[thread][dimension] = fp8_round(__fmul_rn(raw_input[base + dimension], scale));
    }
    __syncthreads();
    for (int query = thread / 32; query < 64; query += 8) {
        #pragma unroll
        for (int key_half = 0; key_half < 2; ++key_half) {
            const int key = lane + key_half * 32;
            float logit = 0.0f;
            #pragma unroll
            for (int dimension = 0; dimension < 32; ++dimension)
                logit = __fadd_rn(logit, __fmul_rn(raw_key[query][dimension], key_transposed[dimension][key]));
            const float bias = exponent_bias[(static_cast<size_t>(head) * 64 + query) * 64 + key];
            const float ramp = clamp_preserving_nan(half_round(__fmaf_rn(logit, kExpSlope, bias)), kExpLow, kExpHigh);
            const unsigned short bits = __half_as_ushort(__float2half_rn(ramp));
            probability[query][key] = __half2float(__ushort_as_half(
                static_cast<unsigned short>((bits & 0x3ffu) << 5u)));
        }
    }
    __syncthreads();
    if (thread < 64) {
        float pairs[4][2][2];
        for (int pair = 0; pair < 4; ++pair)
            for (int member = 0; member < 2; ++member)
                for (int half = 0; half < 2; ++half) {
                    const int key = pair * 4 + member * 2 + half;
                    float total = probability[thread][key];
                    total = half_sum(total, probability[thread][16 + key]);
                    total = half_sum(total, probability[thread][32 + key]);
                    pairs[pair][member][half] = half_sum(total, probability[thread][48 + key]);
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
        reciprocal[thread] = __fdiv_rn(1.0f, clamped);
    }
    __syncthreads();
    for (int query = thread / 32; query < 64; query += 8) {
        #pragma unroll
        for (int key_half = 0; key_half < 2; ++key_half) {
            const int key = lane + key_half * 32;
            probability[query][key] = fp8_round(half_round(__fmul_rn(probability[query][key], reciprocal[query])));
        }
    }
    __syncthreads();
    for (int query = thread / 32; query < 64; query += 8) {
        float accumulator = 0.0f;
        for (int key = 0; key < 64; ++key)
            accumulator = __fmaf_rn(probability[query][key], value[key][lane], accumulator);
        output[(window * 64ull + query) * channels + head * 32 + lane] = accumulator;
    }
}

#include "swin_mma.cuh"
#include "swin_c32_mlp.cuh"
#include "swin_hopper.cuh"

bool fused_c32_mlp_requested() {
    const char* value = std::getenv("RW_NR_SWIN_C32_FUSED_MLP_QKV");
    return value && std::strcmp(value, "1") == 0;
}

bool tensor_core_attention_requested() {
    const char* value = std::getenv("RW_NR_SWIN_TENSOR_CORE");
    return value && std::strcmp(value, "1") == 0;
}

bool fused_window_attention_requested() {
    const char* value = std::getenv("RW_NR_SWIN_FUSED_ATTN");
    return !value || std::strcmp(value, "0") != 0;
}

bool fused_window_attention_all_requested() {
    const char* value = std::getenv("RW_NR_SWIN_FUSED_ALL_ATTN");
    return !value || std::strcmp(value, "0") != 0;
}

bool fused_window_attention_norm_requested() {
    const char* value = std::getenv("RW_NR_SWIN_FUSED_NORM");
    return !value || std::strcmp(value, "0") != 0;
}

bool fused_window_attention_64_requested() {
    const char* value = std::getenv("RW_NR_SWIN_FUSED_ALL_ATTN_64");
    return value && std::strcmp(value, "1") == 0;
}

bool fused_window_attention_128_requested() {
    const char* value = std::getenv("RW_NR_SWIN_FUSED_ALL_ATTN_128");
    return value && std::strcmp(value, "1") == 0;
}
bool fused_window_attention_head1_norm_requested() {
    const char* value = std::getenv("RW_NR_SWIN_HEAD1_FUSED_NORM");
    return value && std::strcmp(value, "1") == 0;
}

bool padded_probability_requested() {
    const char* value = std::getenv("RW_NR_SWIN_PADDED_PROBABILITY");
    return value && std::strcmp(value, "1") == 0;
}

bool cooperative_pv_requested() {
    const char* value = std::getenv("RW_NR_SWIN_COOPERATIVE_PV");
    return value && std::strcmp(value, "1") == 0;
}

bool warp_queries_requested() {
    const char* value = std::getenv("RW_NR_SWIN_WARP_QUERIES");
    return value && std::strcmp(value, "1") == 0;
}

bool fused_window_attention_shared_query_requested() {
    const char* value = std::getenv("RW_NR_SWIN_FUSED_ALL_ATTN_SHARED_Q");
    return value && std::strcmp(value, "1") == 0;
}

bool fused_window_attention_head1_128_requested() {
    const char* value = std::getenv("RW_NR_SWIN_HEAD1_128");
    return value && std::strcmp(value, "1") == 0;
}

bool fused_window_layout_requested() {
    const char* value = std::getenv("RW_NR_SWIN_FUSED_LAYOUT");
    return !value || std::strcmp(value, "0") != 0;
}

bool tiled_window_layout_requested() {
    const char* value = std::getenv("RW_NR_SWIN_TILED_LAYOUT");
    return value && std::strcmp(value, "1") == 0;
}

bool tiled_window_unpack_requested() {
    const char* value = std::getenv("RW_NR_SWIN_TILED_UNPACK");
    return value && std::strcmp(value, "1") == 0;
}

__global__ void pool_kernel(const float* input, float* output, size_t count,
    int output_height, int output_width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int column = static_cast<int>(index % output_width);
    const int row = static_cast<int>((index / output_width) % output_height);
    const size_t channel = index / (static_cast<size_t>(output_height) * output_width);
    const size_t origin = (channel * (2 * output_height) + 2 * row) * (2 * output_width) + 2 * column;
    const float first = half_round(__fadd_rn(input[origin],input[origin+1]));
    const float second = half_round(__fadd_rn(input[origin+2*output_width],input[origin+2*output_width+1]));
    output[index] = fp8_round(__fmul_rn(half_sum(first,second),0.25f));
}
__global__ void zero_pad_kernel(const float* input, float* output, size_t count,
    int input_height, int input_width, int output_height, int output_width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int column = static_cast<int>(index % output_width);
    const int row = static_cast<int>((index / output_width) % output_height);
    const size_t channel = index / (static_cast<size_t>(output_height) * output_width);
    output[index] = row < input_height && column < input_width
        ? input[(channel * input_height + row) * input_width + column] : 0.0f;
}
__global__ void upsample_blend_kernel(const float* projected, const float* skip,
    const float* gain, float* output, size_t count,
    int input_height, int input_width, int output_height, int output_width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int column = static_cast<int>(index % output_width);
    const int row = static_cast<int>((index / output_width) % output_height);
    const size_t channel = index / (static_cast<size_t>(output_height) * output_width);
    const int source_row = min(row/2,input_height-1), source_column = min(column/2,input_width-1);
    const float other = projected[(channel * input_height + source_row) * input_width + source_column];
    // Source computes the skip product and add in float64, then one f16 rounding.
    const double exact = __dadd_rn(__dmul_rn(static_cast<double>(skip[index]),static_cast<double>(gain[channel])),
                                  static_cast<double>(other));
    output[index] = __half2float(__double2half(exact));
}
void require_feature(const Tensor& tensor, int channels, const char* operation) {
    if (tensor.shape.size() != 4 || tensor.shape[0] != 1 || tensor.shape[1] != channels ||
        tensor.shape[2] <= 0 || tensor.shape[3] <= 0 ||
        tensor.shape[2] > std::numeric_limits<int>::max()-16 || tensor.shape[3] > std::numeric_limits<int>::max()-16)
        throw std::runtime_error(std::string(operation)+": invalid NCHW feature shape");
    (void)tensor.numel(); // Reject altered shapes that exceed their actual allocation.
}
std::string key_for(int block, int layer, const char* suffix) {
    return "swin/b"+std::to_string(block)+"/l"+std::to_string(layer)+"/"+suffix;
}
}

SwinParameters load_swin(Context& context, Weights& weights, int block, int layer, int channels, int heads) {
    if ((channels != 32 && channels != 64 && channels != 128 && channels != 256) || heads != channels/32)
        throw std::runtime_error("unsupported Swin parameter width/head count");
    const std::string prepared_key=key_for(block,layer,"prepared_parameters_v1");
    const std::array<int64_t,2> configuration={channels,heads};
    if (const auto prepared=weights.find_prepared(context,prepared_key,configuration)) {
        if (prepared->tensors.size()!=9) throw std::runtime_error("Swin prepared parameter count mismatch");
        SwinParameters result;
        result.channels=channels; result.heads=heads;
        result.expand=prepared->tensors[0]; result.mid=prepared->tensors[1];
        result.contract=prepared->tensors[2]; result.qkv=prepared->tensors[3];
        result.out_projection=prepared->tensors[4]; result.exponent_bias=prepared->tensors[5];
        result.head_scale=prepared->tensors[6]; result.mlp_scale=prepared->tensors[7];
        result.attention_scale=prepared->tensors[8];
        return result;
    }
    SwinParameters params;
    params.channels=channels; params.heads=heads;
    const int hidden=4*channels;
    params.expand=weights.e4m3(context,block,layer,"mlp_expand",{hidden,channels});
    if (heads>1) params.mid=weights.e4m3(context,block,layer,"mlp_mid",{channels,hidden/heads});
    params.contract=weights.e4m3(context,block,layer,"mlp_contract",{channels,heads>1?channels:hidden});
    params.qkv=weights.e4m3(context,block,layer,"qkv",{3*channels,channels});
    params.out_projection=weights.e4m3(context,block,layer,"attn_out_proj",{channels,channels});
    const auto raw_bias=weights.host_f16(block,layer,"attn_pos_bias");
    auto residual=weights.host_f16(block,layer,"residual_scale");
    auto attention=weights.host_f16(block,layer,"attn_residual_scale");
    auto scales=weights.host_f32(block,layer,"scalars_b");
    if (raw_bias.size()!=static_cast<size_t>(heads)*4096 || attention.size()<static_cast<size_t>(channels) ||
        scales.size()<static_cast<size_t>(heads)) throw std::runtime_error("Swin scalar/bias parameter size mismatch");
    const bool padded=residual.size()>=8 && std::all_of(residual.begin(),residual.begin()+8,[](float v){return v==0.0f;});
    const size_t start=padded?8:0;
    if (residual.size()<start+channels) throw std::runtime_error("Swin residual scale table truncated");
    std::vector<float> mlp(residual.begin()+start,residual.begin()+start+channels);
    attention.resize(channels); scales.resize(heads);
    std::vector<float> exponent_bias(static_cast<size_t>(heads)*4096);
    for (int head=0;head<heads;++head) for (int query=0;query<64;++query) for (int key=0;key<64;++key) {
        const int lane=4*((query%16)%8)+((key%16)%8)/2;
        const int slot=(key%2)|(((query%16)/8)<<1)|(((key%16)/8)<<2);
        const int source=(4*(query/16)+(key/16))*256+lane*8+slot;
        exponent_bias[(static_cast<size_t>(head)*64+query)*64+key]=
            std::fma(raw_bias[static_cast<size_t>(head)*4096+source],kExpSlope,kExpOffset);
    }
    params.exponent_bias=weights.derived(context,key_for(block,layer,"exp_bias"),{heads,64,64},exponent_bias);
    params.head_scale=weights.derived(context,key_for(block,layer,"head_scale"),{heads},scales);
    params.mlp_scale=weights.derived(context,key_for(block,layer,"mlp_scale"),{channels},mlp);
    params.attention_scale=weights.derived(context,key_for(block,layer,"attention_scale"),{channels},attention);
    const std::array<Parameter,9> prepared_tensors={params.expand,params.mid,params.contract,params.qkv,
        params.out_projection,params.exponent_bias,params.head_scale,params.mlp_scale,params.attention_scale};
    weights.remember_prepared(context,prepared_key,configuration,prepared_tensors);
    return params;
}

Tensor window_attention(Context& context, const Tensor& input,
    const Parameter& head_scale, const Parameter& exponent_bias, int heads) {
    if (!context.owns(input) ||
        heads<1 || heads>32 || input.shape.size()!=3 || input.shape[1]!=64 ||
        input.shape[2]!=static_cast<int64_t>(heads)*96 || input.shape[0]<=0 ||
        head_scale.numel()!=static_cast<size_t>(heads) || exponent_bias.numel()!=static_cast<size_t>(heads)*4096)
        throw std::runtime_error("window attention parameter/shape mismatch");
    context.validate_parameter(head_scale);
    context.validate_parameter(exponent_bias);
    const int64_t windows=input.shape[0];
    if (windows>std::numeric_limits<int>::max()) throw std::runtime_error("attention batch exceeds cuBLAS integer range");
    if (fused_window_attention_all_requested()) {
        auto output = context.empty({windows, 64, static_cast<int64_t>(heads) * 32});
        if (fused_window_attention_norm_requested()) {
            const auto grid = static_cast<unsigned>(windows * heads);
            if (tensor_core_attention_requested()) {
#if RW_NR_HOPPER_WGMMA_ENABLED
                if(hopper_coalesced_requested())
                    fused_window_attention_hopper<32,true><<<grid,128,0,context.stream()>>>(
                        input.data(),head_scale.data(),exponent_bias.data(),output.data(),
                        static_cast<size_t>(windows),heads,heads*32);
                else if(hopper_wide_mma_requested())
                    fused_window_attention_hopper<32><<<grid,128,0,context.stream()>>>(
                        input.data(),head_scale.data(),exponent_bias.data(),output.data(),
                        static_cast<size_t>(windows),heads,heads*32);
                else
                fused_window_attention_hopper<8><<<grid, 128, 0, context.stream()>>>(
#else
                fused_window_attention_tensor_core<true><<<grid, 128, 0, context.stream()>>>(
#endif
                    input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                    static_cast<size_t>(windows), heads, heads * 32);
            } else if (warp_queries_requested()) {
                fused_window_attention_warp_queries<<<grid, 256, 0, context.stream()>>>(
                    input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                    static_cast<size_t>(windows), heads, heads * 32);
            } else if (cooperative_pv_requested()) {
                fused_window_attention_all_heads_raw_compact<256, false, false, 65, true><<<grid, 256, 0, context.stream()>>>(
                    input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                    static_cast<size_t>(windows), heads, heads * 32);
            } else if (padded_probability_requested()) {
                if (heads == 1 && fused_window_attention_head1_norm_requested())
                    fused_window_attention_all_heads_raw_compact<64, false, true, 65><<<grid, 64, 0, context.stream()>>>(
                        input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                        static_cast<size_t>(windows), heads, heads * 32);
                else
                    fused_window_attention_all_heads_raw_compact<64, false, false, 65><<<grid, 64, 0, context.stream()>>>(
                        input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                        static_cast<size_t>(windows), heads, heads * 32);
            } else if (heads == 1 && fused_window_attention_head1_norm_requested())
                fused_window_attention_all_heads_raw_compact<64, false, true><<<static_cast<unsigned>(windows), 64, 0, context.stream()>>>(
                    input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                    static_cast<size_t>(windows), heads, heads * 32);
            else if (heads == 1 && fused_window_attention_head1_128_requested())
                fused_window_attention_head1_128<<<static_cast<unsigned>(windows), 128, 0, context.stream()>>>(
                    input.data(), head_scale.data(), exponent_bias.data(), output.data(), static_cast<size_t>(windows));
            else if (fused_window_attention_shared_query_requested())
                fused_window_attention_all_heads_raw_compact<128, true><<<grid, 128,
                    64 * 32 * sizeof(float), context.stream()>>>(
                    input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                    static_cast<size_t>(windows), heads, heads * 32);
            else if (fused_window_attention_64_requested())
                fused_window_attention_all_heads_raw_compact<64, false><<<grid, 64, 0, context.stream()>>>(
                    input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                    static_cast<size_t>(windows), heads, heads * 32);
            else if (fused_window_attention_128_requested())
                fused_window_attention_all_heads_raw_compact<128, false><<<grid, 128, 0, context.stream()>>>(
                    input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                    static_cast<size_t>(windows), heads, heads * 32);
            else
                fused_window_attention_all_heads_raw<<<grid, 256, 0, context.stream()>>>(
                    input.data(), head_scale.data(), exponent_bias.data(), output.data(),
                    static_cast<size_t>(windows), heads, heads * 32);
        } else {
            auto normalized=context.empty(input.shape);
            const size_t vectors=input.numel()/96;
            normalize_qkv_kernel<<<blocks(vectors),256,0,context.stream()>>>(input.data(),head_scale.data(),normalized.data(),vectors,heads);
            launched("Swin normalize QKV");
            fused_window_attention_all_heads<<<static_cast<unsigned>(windows * heads), 256, 0, context.stream()>>>(
                normalized.data(), exponent_bias.data(), output.data(), static_cast<size_t>(windows), heads,
                heads * 32);
        }
        launched("Swin fused multi-head window attention");
        return store(context, output);
    }
    auto normalized=context.empty(input.shape);
    const size_t vectors=input.numel()/96;
    normalize_qkv_kernel<<<blocks(vectors),256,0,context.stream()>>>(input.data(),head_scale.data(),normalized.data(),vectors,heads);
    launched("Swin normalize QKV");
    if (heads == 1 && fused_window_attention_requested()) {
        auto output = context.empty({windows, 64, static_cast<int64_t>(heads) * 32});
        fused_window_attention_head1<<<static_cast<unsigned>(windows), 64, 0, context.stream()>>>(
            normalized.data(), exponent_bias.data(), output.data(), static_cast<size_t>(windows));
        launched("Swin fused head-1 window attention");
        return store(context, output);
    }
    auto numerators=context.empty({windows,heads,64,64});
    const float alpha=1.0f,beta=0.0f;
    // q,k each use strided rows inside per-token head/QKV packing.  Compute
    // K^T*Q in column-major, which stores the desired row-major [query,key].
    const bool pointer_batch = [] {
        const char* value = std::getenv("RW_NR_SWIN_POINTER_BATCH");
        return value && std::strcmp(value, "1") == 0;
    }();
    const size_t batch_size = static_cast<size_t>(windows) * static_cast<size_t>(heads);
    if (batch_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Swin attention batch exceeds cuBLAS integer range");
    if (pointer_batch) {
        std::vector<const float*> qk_a(batch_size), qk_b(batch_size);
        std::vector<float*> qk_c(batch_size);
        for (size_t window=0; window<static_cast<size_t>(windows); ++window) for (int head=0; head<heads; ++head) {
            const size_t batch=window*static_cast<size_t>(heads)+static_cast<size_t>(head);
            const size_t token_base=window*64ull*static_cast<size_t>(heads)*96ull;
            qk_a[batch]=normalized.data()+token_base+head*96+32;
            qk_b[batch]=normalized.data()+token_base+head*96;
            qk_c[batch]=numerators.data()+window*static_cast<size_t>(heads)*4096ull+static_cast<size_t>(head)*4096ull;
        }
        auto* workspace=static_cast<unsigned char*>(context.batched_pointer_workspace(batch_size));
        auto* stream=context.stream();
        const size_t pointer_bytes=batch_size*sizeof(void*);
        cuda_check(cudaMemcpyAsync(workspace,qk_a.data(),pointer_bytes,cudaMemcpyHostToDevice,stream),"Swin QK pointer A upload");
        cuda_check(cudaMemcpyAsync(workspace+pointer_bytes,qk_b.data(),pointer_bytes,cudaMemcpyHostToDevice,stream),"Swin QK pointer B upload");
        cuda_check(cudaMemcpyAsync(workspace+2*pointer_bytes,qk_c.data(),pointer_bytes,cudaMemcpyHostToDevice,stream),"Swin QK pointer C upload");
        swin_gemm_batched(context,CUBLAS_OP_T,CUBLAS_OP_N,64,64,32,&alpha,
            reinterpret_cast<const float* const*>(workspace),heads*96,
            reinterpret_cast<const float* const*>(workspace+pointer_bytes),heads*96,
            &beta,reinterpret_cast<float* const*>(workspace+2*pointer_bytes),64,
            static_cast<int>(batch_size),"Swin QK device pointer batch");
    } else {
        for (int head=0; head<heads; ++head)
            swin_gemm_strided_batched(context,CUBLAS_OP_T,CUBLAS_OP_N,
                64,64,32,&alpha,normalized.data()+head*96+32,heads*96,
                static_cast<long long>(64)*heads*96,normalized.data()+head*96,heads*96,
                static_cast<long long>(64)*heads*96,&beta,numerators.data()+head*4096,64,
                static_cast<long long>(heads)*4096,static_cast<int>(windows),"Swin QK GEMM");
    }
    attention_numerator_kernel<<<blocks(numerators.numel()),256,0,context.stream()>>>(numerators.data(),exponent_bias.data(),
        numerators.data(),numerators.numel(),heads);
    launched("Swin numerator");
    auto probabilities=context.empty(numerators.shape);
    const size_t queries=numerators.numel()/64;
    attention_probability_kernel<<<blocks(queries),256,0,context.stream()>>>(numerators.data(),probabilities.data(),queries);
    launched("Swin ordered softmax");
    auto output=context.empty({windows,64,static_cast<int64_t>(heads)*32});
    if (pointer_batch) {
        std::vector<const float*> pv_a(batch_size), pv_b(batch_size);
        std::vector<float*> pv_c(batch_size);
        for (size_t window=0; window<static_cast<size_t>(windows); ++window) for (int head=0; head<heads; ++head) {
            const size_t batch=window*static_cast<size_t>(heads)+static_cast<size_t>(head);
            const size_t token_base=window*64ull*static_cast<size_t>(heads)*96ull;
            pv_a[batch]=normalized.data()+token_base+head*96+64;
            pv_b[batch]=probabilities.data()+window*static_cast<size_t>(heads)*4096ull+static_cast<size_t>(head)*4096ull;
            pv_c[batch]=output.data()+window*64ull*static_cast<size_t>(heads)*32ull+static_cast<size_t>(head)*32;
        }
        auto* workspace=static_cast<unsigned char*>(context.batched_pointer_workspace(batch_size));
        auto* stream=context.stream();
        const size_t pointer_bytes=batch_size*sizeof(void*);
        cuda_check(cudaMemcpyAsync(workspace+3*pointer_bytes,pv_a.data(),pointer_bytes,cudaMemcpyHostToDevice,stream),"Swin PV pointer A upload");
        cuda_check(cudaMemcpyAsync(workspace+4*pointer_bytes,pv_b.data(),pointer_bytes,cudaMemcpyHostToDevice,stream),"Swin PV pointer B upload");
        cuda_check(cudaMemcpyAsync(workspace+5*pointer_bytes,pv_c.data(),pointer_bytes,cudaMemcpyHostToDevice,stream),"Swin PV pointer C upload");
        swin_gemm_batched(context,CUBLAS_OP_N,CUBLAS_OP_N,32,64,64,&alpha,
            reinterpret_cast<const float* const*>(workspace+3*pointer_bytes),heads*96,
            reinterpret_cast<const float* const*>(workspace+4*pointer_bytes),64,
            &beta,reinterpret_cast<float* const*>(workspace+5*pointer_bytes),heads*32,
            static_cast<int>(batch_size),"Swin PV device pointer batch");
    } else {
        for (int head=0; head<heads; ++head)
            swin_gemm_strided_batched(context,CUBLAS_OP_N,CUBLAS_OP_N,
                32,64,64,&alpha,normalized.data()+head*96+64,heads*96,
                static_cast<long long>(64)*heads*96,probabilities.data()+head*4096,64,
                static_cast<long long>(heads)*4096,&beta,output.data()+head*32,heads*32,
                static_cast<long long>(64)*heads*32,static_cast<int>(windows),"Swin PV GEMM");
    }
    return store(context,output);
}

SwinOutput swin_body(Context& context, const SwinParameters& params, const Tensor& input,
    int shift_rows, int shift_columns) {
    require_feature(input,params.channels,"Swin body");
    const int channels=params.channels, heads=params.heads;
    if (!context.owns(input) || heads!=channels/32 ||
        (channels!=32 && channels!=64 && channels!=128 && channels!=256) ||
        params.mlp_scale.numel()!=static_cast<size_t>(channels) ||
        params.attention_scale.numel()!=static_cast<size_t>(channels))
        throw std::runtime_error("Swin body parameter width mismatch");
    const std::vector<int64_t> expand_shape={4*channels,channels};
    const std::vector<int64_t> contract_shape={channels,heads>1?channels:4*channels};
    const std::vector<int64_t> qkv_shape={3*channels,channels},projection_shape={channels,channels};
    if (params.expand.shape()!=expand_shape || params.contract.shape()!=contract_shape ||
        params.qkv.shape()!=qkv_shape || params.out_projection.shape()!=projection_shape ||
        (heads>1 && params.mid.shape()!=std::vector<int64_t>{channels,4*channels/heads}))
        throw std::runtime_error("Swin body matrix parameter shape mismatch");
    for (const auto* parameter : {&params.expand,&params.contract,&params.qkv,&params.out_projection,
                                  &params.exponent_bias,&params.head_scale,&params.mlp_scale,&params.attention_scale}) {
        context.validate_parameter(*parameter);
        (void)parameter->numel();
    }
    if (heads>1) { context.validate_parameter(params.mid); (void)params.mid.numel(); }
    const int height=static_cast<int>(input.shape[2]), width=static_cast<int>(input.shape[3]);
    const auto grid=grid_for(height,width,shift_rows,shift_columns);
#if RW_NR_HOPPER_WGMMA_ENABLED
    const char* full_block=std::getenv("RW_NR_SWIN_C32_FULL_BLOCK");
    if(channels==32 && context.half_math() && full_block && std::strcmp(full_block,"1")==0) {
        SwinOutput output{context.empty(input.shape),context.empty(input.shape)};
        fused_swin_c32_full_hopper<<<static_cast<unsigned>(grid.rows*grid.columns),128,0,context.stream()>>>(
            input.data(),params.expand.data(),params.contract.data(),params.qkv.data(),params.out_projection.data(),
            params.mlp_scale.data(),params.attention_scale.data(),params.head_scale.data(),params.exponent_bias.data(),
            output.accumulator.data(),output.wide.data(),height,width,grid.columns,grid.front_rows,grid.front_columns,
            hopper_batched_mma_requested());
        launched("Swin C32 full window block");
        return output;
    }
#endif
    Tensor y, y_quantized, qkv, quantized, residual, hidden, mlp_product;
    const char* c64_mlp=std::getenv("RW_NR_SWIN_C64_FUSED_MLP");
#if RW_NR_HOPPER_WGMMA_ENABLED
    if(channels==64 && context.half_math() && c64_mlp && std::strcmp(c64_mlp,"1")==0) {
        const int64_t windows=static_cast<int64_t>(grid.rows)*grid.columns;
        y_quantized=context.empty({windows,64,64});qkv=context.empty({windows,64,192});
        fused_swin_c64_mlp_qkv_hopper<<<static_cast<unsigned>(windows),128,0,context.stream()>>>(
            input.data(),params.expand.data(),params.mid.data(),params.contract.data(),params.qkv.data(),
            params.mlp_scale.data(),y_quantized.data(),qkv.data(),height,width,
            grid.columns,grid.front_rows,grid.front_columns,hopper_batched_mma_requested());
        launched("Swin C64 fused expert MLP/residual/QKV");
    } else
#endif
    if (channels == 32 && fused_c32_mlp_requested()) {
        const size_t windows = static_cast<size_t>(grid.rows) * grid.columns;
        y = context.empty({static_cast<int64_t>(windows), 64, 32});
        qkv = context.empty({static_cast<int64_t>(windows), 64, 96});
#if RW_NR_HOPPER_WGMMA_ENABLED
        if(hopper_coalesced_requested())
            fused_swin_c32_mlp_qkv_hopper<32,true><<<static_cast<unsigned>(windows),128,0,context.stream()>>>(
                input.data(),params.expand.data(),params.contract.data(),params.qkv.data(),
                params.mlp_scale.data(),y.data(),qkv.data(),windows,height,width,
                grid.columns,grid.front_rows,grid.front_columns);
        else if(hopper_wide_mma_requested())
            fused_swin_c32_mlp_qkv_hopper<32><<<static_cast<unsigned>(windows),128,0,context.stream()>>>(
                input.data(),params.expand.data(),params.contract.data(),params.qkv.data(),
                params.mlp_scale.data(),y.data(),qkv.data(),windows,height,width,
                grid.columns,grid.front_rows,grid.front_columns);
        else
        fused_swin_c32_mlp_qkv_hopper<8><<<static_cast<unsigned>(windows), 128, 0, context.stream()>>>(
#else
        fused_swin_c32_mlp_qkv<true><<<static_cast<unsigned>(windows), 128, 0, context.stream()>>>(
#endif
            input.data(), params.expand.data(), params.contract.data(), params.qkv.data(),
            params.mlp_scale.data(), y.data(), qkv.data(), windows, height, width,
            grid.columns, grid.front_rows, grid.front_columns);
        launched("Swin C32 fused MLP/residual/QKV");
    } else {
        const char* expert_front=std::getenv("RW_NR_SWIN_EXPERT_FRONT");
#if RW_NR_HOPPER_WGMMA_ENABLED
        if(channels>=128 && context.half_math() && expert_front && std::strcmp(expert_front,"1")==0) {
            const int64_t windows=static_cast<int64_t>(grid.rows)*grid.columns;
            hidden=context.empty({windows,64,channels});
            if(!direct_residual_add_requested())residual=pack_windows(context,input,grid);
            if(channels==128)
                fused_swin_expert_front_hopper<128><<<static_cast<unsigned>(windows*heads),128,4096,context.stream()>>>(
                    input.data(),params.expand.data(),params.mid.data(),hidden.data(),height,width,
                    grid.columns,grid.front_rows,grid.front_columns);
            else
                fused_swin_expert_front_hopper<256><<<static_cast<unsigned>(windows*heads),128,8192,context.stream()>>>(
                    input.data(),params.expand.data(),params.mid.data(),hidden.data(),height,width,
                    grid.columns,grid.front_rows,grid.front_columns);
            launched("Swin per-head expand/activation/middle publication");
        } else
#endif
        {
        quantized=fused_window_layout_requested()
            ? pack_windows_quantized(context,input,grid)
            : pack_windows(context,round8(context,input),grid);
        if (!direct_residual_add_requested()) residual=pack_windows(context,input,grid);
        hidden=activate(context,linear_last(context,quantized,params.expand));
        if (heads>1) {
            auto projected=context.empty({hidden.shape[0],64,channels});
            const int64_t rows=hidden.shape[0]*64;
            if (rows>std::numeric_limits<int>::max()) throw std::runtime_error("Swin rows exceed cuBLAS integer range");
            const int inner=4*channels/heads;
            const float alpha=1.0f,beta=0.0f;
            if (strided_middle_gemm_requested()) {
                swin_gemm_strided_batched(context,CUBLAS_OP_T,CUBLAS_OP_N,
                    32,static_cast<int>(rows),inner,&alpha,params.mid.data(),inner,
                    static_cast<long long>(32)*inner,hidden.data()+0,4*channels,
                    static_cast<long long>(inner),&beta,projected.data()+0,channels,
                    static_cast<long long>(32),heads,"Swin strided-batched middle GEMM");
            } else {
                for (int head=0;head<heads;++head)
                    swin_gemm(context,CUBLAS_OP_T,CUBLAS_OP_N,
                        32,static_cast<int>(rows),inner,&alpha,params.mid.data()+static_cast<size_t>(head)*32*inner,inner,
                        hidden.data()+head*inner,4*channels,&beta,projected.data()+head*32,channels,"Swin grouped middle GEMM");
            }
            hidden=store(context,projected);
        }
        }
        mlp_product=linear_last(context,hidden,params.contract);
        y=direct_residual_add_requested()
            ? residual_add_input(context,input,grid,params.mlp_scale,mlp_product)
            : residual_add(context,residual,params.mlp_scale,mlp_product);
        y_quantized=store(context,y);
        qkv=linear_last(context,y_quantized,params.qkv);
    }
    const Tensor& attention_skip=heads==1?y:y_quantized;
    auto attention=window_attention(context,qkv,params.head_scale,params.exponent_bias,heads);
    auto attention_product=linear_last(context,attention,params.out_projection);
    if (fused_window_layout_requested()) {
        Tensor accumulator, narrowed;
        if (fused_residual_unpack_requested())
            residual_add_unpack_narrow(context,attention_skip,params.attention_scale,
                attention_product,grid,height,width,accumulator,narrowed);
        else {
            auto published=residual_add(context,attention_skip,params.attention_scale,attention_product);
            unpack_windows_narrow(context,published,grid,height,width,accumulator,narrowed);
        }
        return {narrowed,accumulator};
    }
    auto published=residual_add(context,attention_skip,params.attention_scale,attention_product);
    auto accumulator=unpack_windows(context,published,grid,height,width);
    return {narrow(context,accumulator),accumulator};
}

Tensor pool2x2(Context& context, const Tensor& input) {
    if (!context.owns(input) || input.shape.size()!=4 || input.shape[0]!=1 || input.shape[1]<=0 ||
        input.shape[2]<=0 || input.shape[3]<=0 || input.shape[2]%2 || input.shape[3]%2 ||
        input.shape[2]>std::numeric_limits<int>::max() || input.shape[3]>std::numeric_limits<int>::max())
        throw std::runtime_error("pool2x2 requires even positive NCHW spatial dimensions");
    (void)input.numel();
    const int height=static_cast<int>(input.shape[2]/2),width=static_cast<int>(input.shape[3]/2);
    auto output=context.empty({1,input.shape[1],height,width});
    pool_kernel<<<blocks(output.numel()),256,0,context.stream()>>>(input.data(),output.data(),output.numel(),height,width);
    launched("Swin pool2x2");
    return output;
}

LayerOutput run_swin(Context& context, Weights& weights, const LayerSpec& spec,
    const Tensor& input, const LayerOutput* skip) {
    if (spec.kind!="Swin"+std::to_string(spec.heads)+"H")
        throw std::runtime_error("unknown or inconsistent Swin layer kind");
    auto params=load_swin(context,weights,spec.block,spec.layer,spec.channels,spec.heads);
    if (spec.variant=="ds") {
        auto body=swin_body(context,params,round8(context,input),spec.shift_rows,spec.shift_columns);
        auto pooled=pool2x2(context,body.accumulator);
        auto resample=weights.e4m3(context,spec.block,spec.layer,"resample",{2*spec.channels,spec.channels});
        auto projected=store(context,linear_nchw(context,pooled,resample));
        if (spec.out_height<projected.shape[2] || spec.out_width<projected.shape[3])
            throw std::runtime_error("Swin downsample native geometry would crop unexpectedly");
        auto output=context.empty({1,2*spec.channels,spec.out_height,spec.out_width});
        zero_pad_kernel<<<blocks(output.numel()),256,0,context.stream()>>>(projected.data(),output.data(),output.numel(),
            static_cast<int>(projected.shape[2]),static_cast<int>(projected.shape[3]),spec.out_height,spec.out_width);
        launched("Swin downsample native zero padding");
        return {output,round8(context,body.wide)};
    }
    if (spec.variant=="upsample") {
        if (!skip) throw std::runtime_error("Swin upsample missing encoder skip");
        const Tensor& encoder=skip->skip.empty()?skip->main:skip->skip;
        if (!context.owns(input) || !context.owns(encoder))
            throw std::runtime_error("Swin upsample input belongs to another context");
        require_feature(encoder,spec.channels,"Swin encoder skip");
        require_feature(input,2*spec.channels,"Swin upsample input");
        auto resample=weights.e4m3(context,spec.block,spec.layer,"resample",{spec.channels,2*spec.channels});
        auto projected=narrow(context,linear_nchw(context,input,resample));
        const std::string gain_key=key_for(spec.block,spec.layer,"prepared_upsample_gain_v1");
        const std::array<int64_t,1> gain_configuration={spec.channels};
        Parameter gain_tensor;
        if (const auto prepared=weights.find_prepared(context,gain_key,gain_configuration)) {
            if (prepared->tensors.size()!=1) throw std::runtime_error("Swin prepared gain count mismatch");
            gain_tensor=prepared->tensors[0];
        } else {
            auto gain=weights.host_f16(spec.block,spec.layer,"upsample_gain");
            if (gain.size()<static_cast<size_t>(spec.channels)) {
                const auto residual=weights.host_f16(spec.block,spec.layer,"residual_scale");
                const size_t missing=spec.channels-gain.size();
                if (residual.size()<missing) throw std::runtime_error("Swin upsample gain table truncated");
                gain.insert(gain.begin(),residual.end()-missing,residual.end());
            }
            gain.resize(spec.channels);
            gain_tensor=weights.derived(context,key_for(spec.block,spec.layer,"upsample_gain"),{spec.channels},gain);
            const std::array<Parameter,1> prepared_tensors={gain_tensor};
            weights.remember_prepared(context,gain_key,gain_configuration,prepared_tensors);
        }
        context.validate_parameter(gain_tensor);
        auto blend=context.empty(encoder.shape);
        upsample_blend_kernel<<<blocks(blend.numel()),256,0,context.stream()>>>(projected.data(),encoder.data(),gain_tensor.data(),
            blend.data(),blend.numel(),static_cast<int>(projected.shape[2]),static_cast<int>(projected.shape[3]),
            static_cast<int>(encoder.shape[2]),static_cast<int>(encoder.shape[3]));
        launched("Swin upsample fused half blend");
        auto body=swin_body(context,params,spec.channels==32?blend:round8(context,blend),spec.shift_rows,spec.shift_columns);
        return {round8(context,body.wide),{}};
    }
    if (spec.variant!="" && spec.variant!="inpview" && spec.variant!="outview")
        throw std::runtime_error("unknown Swin layer variant");
    auto body=swin_body(context,params,round8(context,input),spec.shift_rows,spec.shift_columns);
    return {round8(context,body.wide),{}};
}
} // namespace rw::native_nr
