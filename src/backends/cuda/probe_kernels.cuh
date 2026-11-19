#pragma once

// Small, unfused arithmetic probes. These kernels are NOT a whole NR backend
// and do not establish the accumulator/rounding contract of a native NR GEMM.
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace rw::cuda_probe {

__device__ inline float decode_e4m3(unsigned char bits) {
    __nv_fp8_e4m3 value;
    value.__x = bits;
    return static_cast<float>(value);
}

__global__ void convert(const unsigned char* codes, float* decoded,
                        const float* values, unsigned char* encoded,
                        unsigned code_count, unsigned value_count) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < code_count) decoded[index] = decode_e4m3(codes[index]);
    if (index < value_count) {
        encoded[index] = __nv_cvt_float_to_fp8(values[index], __NV_SATFINITE,
                                              __NV_E4M3);
    }
}

__global__ void gemm(const unsigned char* a, const unsigned char* b, float* c,
                     unsigned char* published, unsigned m, unsigned n,
                     unsigned k) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m * n) return;
    const unsigned row = index / n;
    const unsigned column = index % n;
    float accumulator = 0.0F;
    for (unsigned inner = 0; inner < k; ++inner) {
        const float product = __fmul_rn(decode_e4m3(a[row * k + inner]),
                                       decode_e4m3(b[inner * n + column]));
        accumulator = __fadd_rn(accumulator, product);
    }
    c[index] = accumulator;
    published[index] = __nv_cvt_float_to_fp8(accumulator, __NV_SATFINITE,
                                            __NV_E4M3);
}

// Pixel-centre coordinates with an explicitly declared clamp boundary. This is
// a generic sampler probe; a model requiring Catmull-Rom must not use it instead.
__global__ void bilinear_clamp(const float* image, unsigned width,
                               unsigned height, const float2* coordinates,
                               const unsigned char* input_valid, float* output,
                               unsigned char* output_valid, unsigned count) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float2 coordinate = coordinates[index];
    if (!input_valid[index] || !isfinite(coordinate.x) ||
        !isfinite(coordinate.y)) {
        output_valid[index] = 0;
        output[index] = nanf("");
        return;
    }
    const float x = fminf(fmaxf(coordinate.x, 0.0F), float(width - 1));
    const float y = fminf(fmaxf(coordinate.y, 0.0F), float(height - 1));
    const unsigned x0 = static_cast<unsigned>(floorf(x));
    const unsigned y0 = static_cast<unsigned>(floorf(y));
    const unsigned x1 = x0 + unsigned(x0 + 1 < width);
    const unsigned y1 = y0 + unsigned(y0 + 1 < height);
    const float dx = x - float(x0);
    const float dy = y - float(y0);
    const float upper = __fadd_rn(
        __fmul_rn(1.0F - dx, image[y0 * width + x0]),
        __fmul_rn(dx, image[y0 * width + x1]));
    const float lower = __fadd_rn(
        __fmul_rn(1.0F - dx, image[y1 * width + x0]),
        __fmul_rn(dx, image[y1 * width + x1]));
    output[index] = __fadd_rn(__fmul_rn(1.0F - dy, upper),
                              __fmul_rn(dy, lower));
    output_valid[index] = 1;
}

}  // namespace rw::cuda_probe
