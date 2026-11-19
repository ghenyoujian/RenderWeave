#pragma once

// Isolated mathematical components; not an enabled temporal NR profile.
#include "native_nr.hpp"

namespace rw::native_nr::temporal_probe {

struct MotionRegion {
    int x = 0, y = 0, width = 0, height = 0;
    float native_scale_x = 1.0f, native_scale_y = 1.0f;
    bool present = true;
    bool current_to_previous = true;
};

struct WarpResult {
    Tensor sampled;       // [1,3,H,W], unrounded mathematical RGB
    Tensor positions;     // [1,2,H,W], history texel-centre coordinates
    Tensor motion_indices;// [1,2,H,W], sampled allocation integer indices
};

// Float32 software bilinear, explicit RN operations; not CUDA tex interpolation.
Tensor sample_5tap_math(Context&, const Tensor& history, const Tensor& positions);
WarpResult warp_math(Context&, const Tensor& history, const Tensor& motion,
                     int output_width, int output_height, const MotionRegion&);

struct BlendResult {
    Tensor temporal;      // provisional F32 math value; native codec unresolved
    Tensor display;
    Tensor gate;
};
BlendResult blend_math(Context&, const Tensor& current, const Tensor& residual_rgb,
                       const Tensor& logit, const Tensor& warped_history,
                       float blend_scale, float intensity, bool history_usable);
} // namespace rw::native_nr::temporal_probe
