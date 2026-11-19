// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "reconstruction.hpp"

#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

namespace rw::sr::source_backbone::detail {

struct ReconstructionParameters {
    alignas(8) unsigned char block[328]{};
    std::uint32_t dimensions[8]{};

    template<class T> void set(int offset, T value) { std::memcpy(block + offset, &value, sizeof(T)); }
};

inline ReconstructionParameters reconstruction_parameters(int width, int height, bool reset) {
    const auto g = reconstruction_geometry(width, height);
    const auto ow = g.output.width, oh = g.output.height;
    ReconstructionParameters p{};
    const auto u = [](int value) { return std::uint32_t(value); };
    const std::uint32_t dims[]{u(width), u(height), u(ow), u(oh), u(g.neural.width), u(g.neural.height),
                               u(g.history.width), u(g.history.height)};
    std::memcpy(p.dimensions, dims, sizeof(dims));
    const float x = width * (1.f / ow), y = height * (1.f / oh);
    p.set(0, 1.f / x); p.set(4, 1.f / y);
    for (int offset : {16, 20, 24, 28}) p.set(offset, -.5f);
    p.set(32, static_cast<unsigned char>(reset));
    for (int offset : {40, 56}) { p.set(offset, x); p.set(offset + 4, y); }
    p.set(48, x * .5f); p.set(52, y * .5f); p.set(64, .5f); p.set(68, .5f);
    p.set(76, ~std::uint32_t(0)); p.set(80, dims[4]); p.set(84, dims[5]);
    for (int offset : {104, 120, 216}) { p.set(offset, dims[0] - 1); p.set(offset + 4, dims[1] - 1); }
    p.set(136, dims[2]); p.set(140, dims[3]);
    p.set(144, 1.f / dims[6]); p.set(148, 1.f / dims[7]); p.set(152, 1.f / ow); p.set(156, 1.f / oh);
    p.set(160, 1.f); p.set(164, 1.f); p.set(168, 1.f); p.set(172, 1.37f); p.set(176, .02f);
    p.set(180, __half2float(__float2half_rn(.02f)));
    constexpr float half_values[]{.04f, 1.f / .96f, 8.f, -32.f, -32.f};
    for (int i = 0; i < 5; ++i) p.set(184 + 2 * i, __float2half_rn(half_values[i]));
    constexpr int offsets[]{200, 224, 232, 240, 248, 288, 296, 304, 312};
    constexpr std::uint64_t handles[]{1, 2, 3, 4, 5, 7, 8, 9, 6};
    for (int i = 0; i < 9; ++i) p.set(offsets[i], handles[i]);
    p.set(272, 1ull << 32); p.set(280, 2ull << 32);
    return p;
}

} // namespace rw::sr::source_backbone::detail
