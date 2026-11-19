#pragma once

#include <cstdint>
#include <utility>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace rw::detail {
inline std::pair<uint64_t, uint64_t> pts_magnitude(int64_t ticks, uint64_t factor) {
    const uint64_t word = static_cast<uint64_t>(ticks);
    const uint64_t value = ticks < 0 ? uint64_t{0} - word : word;
#if defined(_MSC_VER)
    uint64_t high = 0;
    const uint64_t low = _umul128(value, factor, &high);
    return {high, low};
#else
    const auto product = static_cast<__uint128_t>(value) * factor;
    return {static_cast<uint64_t>(product >> 64), static_cast<uint64_t>(product)};
#endif
}

// Time-base numerators and denominators are positive int32 values.
inline bool later_pts(int64_t a, int32_t a_num, int32_t a_den,
                      int64_t b, int32_t b_num, int32_t b_den) {
    if ((a < 0) != (b < 0)) return a >= 0;
    const auto left = pts_magnitude(a, uint64_t(a_num) * b_den);
    const auto right = pts_magnitude(b, uint64_t(b_num) * a_den);
    return a < 0 ? left < right : left > right;
}
}
