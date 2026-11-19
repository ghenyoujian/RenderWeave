#pragma once

// Native CUDA lowering of the source-specific NumPy v2.3.5 float32 SIMD math
// profile in numpy_math.hpp. No libm approximation or NumPy runtime is called.
// NumPy source commit: c3d60fc8393f3ca3306b8ce8b6453d43737e3d90.
// Source paths, exact hashes and qualification scopes: numpy_math_source.json.
// Required CUDA build mode: no fast math, --ftz=false, precise division/sqrt.
// All arithmetic below specifies IEEE binary32 RN explicitly, including the
// separate multiply/add sites and only the source's designated fused sites.
//
// Copyright (c) 2005-2025, NumPy Developers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the NumPy Developers nor the names of any contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cuda_runtime.h>

namespace rw::native_nr::numpy_math_device {

__device__ __forceinline__ float quiet_nan() { return __uint_as_float(0x7fc00000u); }
__device__ __forceinline__ bool finite(float input) {
    return (__float_as_uint(input) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ float numpy_logf(float input) {
    // This is the exact positive-normal unit domain supplied by the PCG field.
    // Unsupported domains propagate NaN rather than invoke another algorithm.
    if (!finite(input) || input < 0x1p-24f || input > 1.0f) return quiet_nan();
    const unsigned bits = __float_as_uint(input);
    float exponent = __int2float_rn(static_cast<int>((bits >> 23) & 255u) - 126);
    float x = __uint_as_float((bits & 0x007fffffu) | (126u << 23));
    if (x <= 0.707106781186547524400844362104849039f) {
        x = __fadd_rn(x, x);
        exponent = __fsub_rn(exponent, 1.0f);
    }
    x = __fsub_rn(x, 1.0f);
    float numerator = __fmaf_rn(2.589979117907922693523e-02f, x, 3.808837741388407920751e-01f);
    numerator = __fmaf_rn(numerator, x, 1.480000633576506585156e+00f);
    numerator = __fmaf_rn(numerator, x, 2.112677543073053063722e+00f);
    numerator = __fmaf_rn(numerator, x, 9.999999999999998702752e-01f);
    numerator = __fmaf_rn(numerator, x, 0.0f);
    float denominator = __fmaf_rn(5.875095403124574342950e-03f, x, 1.546476374983906719538e-01f);
    denominator = __fmaf_rn(denominator, x, 9.864942958519418960339e-01f);
    denominator = __fmaf_rn(denominator, x, 2.453006071784736363091e+00f);
    denominator = __fmaf_rn(denominator, x, 2.612677543073109236779e+00f);
    denominator = __fmaf_rn(denominator, x, 1.0f);
    return __fmaf_rn(exponent, 0.693147180559945309417232121458176568f,
                      __fdiv_rn(numerator, denominator));
}

__device__ __forceinline__ float cosine_polynomial(float squared) {
    float result = __fmaf_rn(0x1.98e616p-16f, squared, -0x1.6c06dcp-10f);
    result = __fmaf_rn(result, squared, 0x1.55553cp-5f);
    result = __fmaf_rn(result, squared, -0x1p-1f);
    return __fmaf_rn(result, squared, 1.0f);
}
__device__ __forceinline__ float sine_polynomial(float value, float squared) {
    float result = __fmaf_rn(0x1.7d3bbcp-19f, squared, -0x1.a06bbap-13f);
    result = __fmaf_rn(result, squared, 0x1.11119ap-7f);
    result = __fmaf_rn(result, squared, -0x1.555556p-3f);
    result = __fmaf_rn(result, squared, 0.0f);
    return __fmaf_rn(result, value, value);
}

__device__ __forceinline__ float sincos_angle_f32(float input, bool cosine) {
    if (!finite(input) || input < 0.0f || input > 6.28318530718f) return quiet_nan();
    float quadrant = __fmul_rn(input, 0x1.45f306p-1f);
    quadrant = __fsub_rn(__fadd_rn(quadrant, 0x1.8p+23f), 0x1.8p+23f);
    float reduced = __fmaf_rn(quadrant, -0x1.921fb0p+0f, input);
    reduced = __fmaf_rn(quadrant, -0x1.5110b4p-22f, reduced);
    reduced = __fmaf_rn(quadrant, -0x1.846988p-48f, reduced);
    const float squared = __fmul_rn(reduced, reduced);
    const float sine = sine_polynomial(reduced, squared);
    const float cos = cosine_polynomial(squared);
    int selected = __float2int_rn(quadrant);
    if (cosine) ++selected;
    float result = (selected & 1) == 0 ? sine : cos;
    if ((selected & 2) == 2) result = __fsub_rn(0.0f, result);
    return result;
}
__device__ __forceinline__ float numpy_sinf(float input) { return sincos_angle_f32(input, false); }
__device__ __forceinline__ float numpy_cosf(float input) { return sincos_angle_f32(input, true); }
__device__ __forceinline__ float numpy_sqrtf(float input) {
    if (!finite(input) || input < 0.0f) return quiet_nan();
    return __fsqrt_rn(input);
}
} // namespace rw::native_nr::numpy_math_device
