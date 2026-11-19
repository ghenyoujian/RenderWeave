#include "rw/rw_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
unsigned nearest_even(float x) {
    const auto low = static_cast<unsigned>(std::floor(x));
    const float fraction = x - static_cast<float>(low);
    return low + static_cast<unsigned>(fraction > 0.5f || (fraction == 0.5f && (low & 1u)));
}
bool product_count(uint32_t a, uint32_t b, size_t& out) {
    if (a == 0 || b == 0 || uint64_t(a) * b > std::numeric_limits<size_t>::max()) return false;
    out = static_cast<size_t>(uint64_t(a) * b);
    return true;
}
bool overlaps(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (!a || !b || a_bytes == 0 || b_bytes == 0) return false;
    const auto pa = reinterpret_cast<uintptr_t>(a), pb = reinterpret_cast<uintptr_t>(b);
    // Difference comparisons avoid pointer arithmetic or overflow on untrusted sizes.
    return pa <= pb ? pb - pa < a_bytes : pa - pb < b_bytes;
}
}

extern "C" {
float rw_e4m3_decode(uint8_t code) {
    const unsigned exponent = (code >> 3u) & 15u;
    const unsigned mantissa = code & 7u;
    float value;
    if (exponent == 15u && mantissa == 7u) value = std::numeric_limits<float>::quiet_NaN();
    else if (exponent == 0u) value = std::ldexp(static_cast<float>(mantissa), -9);
    else value = std::ldexp(static_cast<float>(8u + mantissa), static_cast<int>(exponent) - 10);
    return std::copysign(value, (code & 128u) ? -1.0f : 1.0f);
}

uint8_t rw_e4m3_encode(float value) {
    const uint8_t sign = std::signbit(value) ? 128u : 0u;
    if (std::isnan(value)) return static_cast<uint8_t>(sign | 127u);
    const float magnitude = std::min(std::fabs(value), 448.0f);
    if (magnitude < std::ldexp(1.0f, -6))
        return static_cast<uint8_t>(sign | nearest_even(std::ldexp(magnitude, 9)));
    int power;
    std::frexp(magnitude, &power);
    unsigned exponent = static_cast<unsigned>(power + 6);
    unsigned significand = nearest_even(std::ldexp(magnitude, 4 - power));
    if (significand == 16u) { ++exponent; significand = 8u; }
    return static_cast<uint8_t>(sign | (exponent << 3u) | (significand - 8u));
}

rw_status_t rw_e4m3_decode_array(const uint8_t* input, float* output, size_t count) {
    if (count == 0) return RW_OK;
    if (!input || !output) return RW_INVALID_ARGUMENT;
    if (count > std::numeric_limits<size_t>::max() / sizeof(float)) return RW_OVERFLOW;
    if (overlaps(input, count, output, count * sizeof(float))) return RW_INVALID_ARGUMENT;
    for (size_t i = 0; i < count; ++i) output[i] = rw_e4m3_decode(input[i]);
    return RW_OK;
}

rw_status_t rw_e4m3_encode_array(const float* input, uint8_t* output, size_t count) {
    if (count == 0) return RW_OK;
    if (!input || !output) return RW_INVALID_ARGUMENT;
    if (count > std::numeric_limits<size_t>::max() / sizeof(float)) return RW_OVERFLOW;
    if (overlaps(input, count * sizeof(float), output, count)) return RW_INVALID_ARGUMENT;
    for (size_t i = 0; i < count; ++i) output[i] = rw_e4m3_encode(input[i]);
    return RW_OK;
}

rw_status_t rw_reference_gemm_e4m3(
    const uint8_t* a, size_t a_count, const uint8_t* b, size_t b_count,
    const float* residual, size_t residual_count, float* output, size_t output_count,
    uint32_t m, uint32_t n, uint32_t k) {
    size_t mk, kn, mn;
    if (!product_count(m, k, mk) || !product_count(k, n, kn) || !product_count(m, n, mn))
        return (m == 0 || n == 0 || k == 0) ? RW_INVALID_ARGUMENT : RW_OVERFLOW;
    if (!a || !b || !output) return RW_INVALID_ARGUMENT;
    if (mk > a_count || kn > b_count || mn > output_count || (residual && mn > residual_count))
        return RW_OUT_OF_BOUNDS;
    if (mn > std::numeric_limits<size_t>::max() / sizeof(float)) return RW_OVERFLOW;
    const size_t output_bytes = mn * sizeof(float);
    if (overlaps(a, mk, output, output_bytes) || overlaps(b, kn, output, output_bytes) ||
        overlaps(residual, residual ? output_bytes : 0, output, output_bytes)) return RW_INVALID_ARGUMENT;
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t column = 0; column < n; ++column) {
            const size_t index = static_cast<size_t>(row) * n + column;
            float sum = residual ? residual[index] : 0.0f;
            for (uint32_t inner = 0; inner < k; ++inner)
                sum = std::fma(rw_e4m3_decode(a[static_cast<size_t>(row) * k + inner]),
                               rw_e4m3_decode(b[static_cast<size_t>(inner) * n + column]), sum);
            output[index] = sum;
        }
    }
    return RW_OK;
}

rw_status_t rw_reference_bilinear_clamp_f32(const float* image, size_t image_count,
    uint32_t width, uint32_t height, float x, float y, float* output) {
    size_t elements;
    if (!image || !output || !std::isfinite(x) || !std::isfinite(y)) return RW_INVALID_ARGUMENT;
    if (!product_count(width, height, elements))
        return (width == 0 || height == 0) ? RW_INVALID_ARGUMENT : RW_OVERFLOW;
    // Float coordinates cannot address every pixel center beyond this domain.
    // Reject before float(width-1) could round upward and overflow an index cast.
    constexpr uint32_t max_exact_axis = UINT32_C(1) << 24;
    if (width > max_exact_axis || height > max_exact_axis) return RW_UNSUPPORTED;
    if (elements > image_count) return RW_OUT_OF_BOUNDS;
    x = std::clamp(x, 0.0f, static_cast<float>(width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(height - 1));
    const auto x0 = static_cast<uint32_t>(std::floor(x)), y0 = static_cast<uint32_t>(std::floor(y));
    const auto x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    const float dx = x - static_cast<float>(x0), dy = y - static_cast<float>(y0);
    const float top = std::lerp(image[static_cast<size_t>(y0) * width + x0],
                                image[static_cast<size_t>(y0) * width + x1], dx);
    const float bottom = std::lerp(image[static_cast<size_t>(y1) * width + x0],
                                   image[static_cast<size_t>(y1) * width + x1], dx);
    *output = std::lerp(top, bottom, dy);
    return RW_OK;
}

rw_status_t rw_mirror_index(int64_t coordinate, uint32_t extent, uint32_t* output) {
    if (!output || extent == 0) return RW_INVALID_ARGUMENT;
    if (extent == 1) { *output = 0; return RW_OK; }
    const int64_t period = 2 * (static_cast<int64_t>(extent) - 1);
    int64_t index = coordinate % period;
    if (index < 0) index += period;
    *output = static_cast<uint32_t>(index < extent ? index : period - index);
    return RW_OK;
}
}
