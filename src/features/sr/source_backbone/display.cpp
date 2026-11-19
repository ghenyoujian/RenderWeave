#include <rw/rw_native_sr.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

uint8_t display_code(float value) {
    const float scaled = std::clamp(value, 0.0f, 1.0f) * 255.0f;
    return static_cast<uint8_t>(static_cast<double>(scaled) + 0.5);
}

}

rw_status_t rw_sr_display_rgb8(const float* rgba, uint64_t rgba_elements,
                              uint8_t* rgb, uint64_t rgb_bytes, rw_sr_display_info* info) {
    if (!rgba || !rgb || !info || !rgba_elements || rgba_elements % 4 ||
        rgba_elements > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return RW_INVALID_ARGUMENT;
    if (info->struct_size < sizeof(*info) || info->abi_version != RW_ABI_VERSION)
        return RW_ABI_MISMATCH;
    const auto pixels = static_cast<std::size_t>(rgba_elements / 4);
    if (rgb_bytes < pixels * 3) return RW_BUFFER_TOO_SMALL;

    float minimum = rgba[0], maximum = rgba[0];
    uint64_t clamped = 0;
    for (std::size_t i = 0; i < pixels; ++i) {
        const auto* source = rgba + 4 * i;
        if (!std::isfinite(source[0]) || !std::isfinite(source[1]) ||
            !std::isfinite(source[2]) || !std::isfinite(source[3])) return RW_OUTPUT_INVALID;
        const auto [lo, hi] = std::minmax({source[0], source[1], source[2], source[3]});
        minimum = std::min(minimum, lo);
        maximum = std::max(maximum, hi);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const float value = source[channel];
            clamped += value < 0 || value > 1;
            rgb[3 * i + channel] = display_code(value);
        }
    }
    info->min_rgba = minimum;
    info->max_rgba = maximum;
    info->clamped_rgb_values = clamped;
    return RW_OK;
}
