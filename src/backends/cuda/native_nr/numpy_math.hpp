#pragma once

// Proposed source-specific scalar port for differential evaluation only.
// Enable only after the installed NumPy float32 ufunc dispatch and FP environment
// are recorded and the complete RNG-domain probe passes. This header is not
// automatically selected by the native renderer.
//
// Algorithms/constants: NumPy v2.3.5, commit
// c3d60fc8393f3ca3306b8ce8b6453d43737e3d90:
//   numpy/_core/src/umath/loops_exponent_log.dispatch.c.src
//   numpy/_core/src/umath/npy_simd_data.h
//   numpy/_core/src/umath/loops_trigonometric.dispatch.cpp
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

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace rw::native_nr::numpy_math {

inline constexpr const char* source_commit = "c3d60fc8393f3ca3306b8ce8b6453d43737e3d90";

inline void require_round_to_nearest() {
    if (std::fegetround() != FE_TONEAREST)
        throw std::runtime_error("NumPy SIMD source profile requires FE_TONEAREST");
}

// Volatile binary32 publications prohibit contraction of the source's separate
// Mul/Add/Sub/Div steps. Only the explicitly named std::fma calls are fused.
inline float add(float a, float b) { volatile float value = a + b; return value; }
inline float sub(float a, float b) { volatile float value = a - b; return value; }
inline float mul(float a, float b) { volatile float value = a * b; return value; }
inline float div(float a, float b) { volatile float value = a / b; return value; }

// NumPy AVX2_FMA3 and AVX512F use the same rational polynomial and instruction
// order on positive normal inputs. The counter RNG supplies exactly [2^-24,1],
// so special/denormal dispatch behavior is deliberately outside this helper.
inline float numpy_logf(float input) {
    if (!std::isfinite(input) || input < 0x1p-24f || input > 1.0f)
        throw std::invalid_argument("NumPy log helper requires a counter-RNG unit value in [2^-24,1]");
    const uint32_t bits = std::bit_cast<uint32_t>(input);
    float exponent = static_cast<float>(static_cast<int>((bits >> 23) & 255u) - 126);
    float x = std::bit_cast<float>((bits & 0x007fffffu) | (126u << 23));
    if (x <= 0.707106781186547524400844362104849039f) {
        x = add(x, x);
        exponent = sub(exponent, 1.0f);
    }
    x = sub(x, 1.0f);
    float numerator = std::fmaf(2.589979117907922693523e-02f, x, 3.808837741388407920751e-01f);
    numerator = std::fmaf(numerator, x, 1.480000633576506585156e+00f);
    numerator = std::fmaf(numerator, x, 2.112677543073053063722e+00f);
    numerator = std::fmaf(numerator, x, 9.999999999999998702752e-01f);
    numerator = std::fmaf(numerator, x, 0.0f);
    float denominator = std::fmaf(5.875095403124574342950e-03f, x, 1.546476374983906719538e-01f);
    denominator = std::fmaf(denominator, x, 9.864942958519418960339e-01f);
    denominator = std::fmaf(denominator, x, 2.453006071784736363091e+00f);
    denominator = std::fmaf(denominator, x, 2.612677543073109236779e+00f);
    denominator = std::fmaf(denominator, x, 1.0f);
    return std::fmaf(exponent, 0.693147180559945309417232121458176568f,
                    div(numerator, denominator));
}

inline float cosine_polynomial(float squared) {
    float result = std::fmaf(0x1.98e616p-16f, squared, -0x1.6c06dcp-10f);
    result = std::fmaf(result, squared, 0x1.55553cp-5f);
    result = std::fmaf(result, squared, -0x1p-1f);
    return std::fmaf(result, squared, 1.0f);
}
inline float sine_polynomial(float value, float squared) {
    float result = std::fmaf(0x1.7d3bbcp-19f, squared, -0x1.a06bbap-13f);
    result = std::fmaf(result, squared, 0x1.11119ap-7f);
    result = std::fmaf(result, squared, -0x1.555556p-3f);
    result = std::fmaf(result, squared, 0.0f);
    return std::fmaf(result, value, value);
}

// NumPy's float32 FMA SIMD path uses Cody-Waite reduction and these polynomials
// for every counter-RNG angle. No libc or SVML trigonometric call is involved.
inline float sincos_angle_f32(float input, bool cosine) {
    if (!std::isfinite(input) || input < 0.0f || input > 6.28318530718f)
        throw std::invalid_argument("NumPy trig helper requires a counter-RNG angle in [0,2*pi_f32]");
    float quadrant = mul(input, 0x1.45f306p-1f);
    quadrant = sub(add(quadrant, 0x1.8p+23f), 0x1.8p+23f);
    float reduced = std::fmaf(quadrant, -0x1.921fb0p+0f, input);
    reduced = std::fmaf(quadrant, -0x1.5110b4p-22f, reduced);
    reduced = std::fmaf(quadrant, -0x1.846988p-48f, reduced);
    const float squared = mul(reduced, reduced);
    const float sine = sine_polynomial(reduced, squared);
    const float cos = cosine_polynomial(squared);
    int selected = static_cast<int>(quadrant);
    if (cosine) ++selected;
    float result = (selected & 1) == 0 ? sine : cos;
    if ((selected & 2) == 2) result = sub(0.0f, result);
    return result;
}
inline float numpy_sinf(float input) { return sincos_angle_f32(input, false); }
inline float numpy_cosf(float input) { return sincos_angle_f32(input, true); }
inline float numpy_sqrtf(float input) {
    if (!std::isfinite(input) || input < 0.0f)
        throw std::invalid_argument("NumPy sqrt helper requires finite nonnegative input");
    volatile float result = std::sqrt(input);
    return result;
}
} // namespace rw::native_nr::numpy_math
