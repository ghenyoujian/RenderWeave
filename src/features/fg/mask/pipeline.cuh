#pragma once

#include "graph.cuh"
#include "input.cuh"
#include "output.cuh"

namespace rw::fg::mask {

// Geometry uses the exact mask plan. The optional interpolation texture in
// output_guides can view input_output.product_guide's array (resource 0x16).
inline cudaError_t launch_mask_pipeline(const MaskPlan& plan,
                                       const std::array<const uint32_t*, weight_count>& weights,
                                       MaskInputTextures input_textures, MaskInputGeometry input_geometry,
                                       MaskInputOutput input_output, uint32_t* mask, void* workspace,
                                       MaskOutputGeometry output_geometry, MaskOutputGuides output_guides,
                                       MaskOutputSurfaces surfaces, cudaStream_t stream) {
    input_geometry.mask_width = plan.input.width;
    input_geometry.mask_height = plan.input.height;
    output_geometry.input_width = plan.outputs.back().width;
    output_geometry.input_height = plan.outputs.back().height;
    if (const auto status = launch_mask_input(input_textures, input_geometry, input_output, stream);
        status != cudaSuccess) return status;
    if (const auto status = launch_mask(plan, weights, input_output.packed, mask, workspace, stream);
        status != cudaSuccess) return status;
    return launch_mask_output(reinterpret_cast<const __half*>(mask), output_geometry, output_guides, surfaces, stream);
}

}
