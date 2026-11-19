// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backbone.hpp"

namespace rw::sr::source_backbone {

struct PrepareInput {
    int width, height;
    const float* color_rgb;                 // SDR [H,W,3], code values [0,1].
    const float* motion_xy;                 // [H,W,2], current to previous.
    const float* depth;                     // [H,W], smaller means nearer.
    const std::uint16_t* previous_ycocg;     // FP16 [align16(2H),align16(2W),4].
    const std::uint16_t* previous_hidden;    // FP16 [neural H,neural W,4].
    float jitter_x, jitter_y;               // Low resolution pixels.
    float motion_scale_x, motion_scale_y;   // Input motion to output pixels.
    float motion_bias_x, motion_bias_y;     // Added before motion scale.
    float exposure;
    bool reset;
};

// Mutable device addresses for one slot's next graph launch. The graph caches
// only the binding allocation; the session continues to own all six planes.
struct HistoryBinding {
    const std::uint16_t* previous_ycocg;
    const std::uint16_t* previous_luma;
    const std::uint16_t* previous_hidden;
    std::uint16_t* next_ycocg;
    std::uint16_t* next_luma;
    std::uint16_t* next_hidden;
};

// Build the K 16-channel input field from current color and committed history.
// Output is FP16 [network_geometry(height),network_geometry(width),16].
cudaError_t prepare(const PrepareInput& input,std::uint16_t* prepared,cudaStream_t stream,
                    const HistoryBinding* history=nullptr);

} // namespace rw::sr::source_backbone
