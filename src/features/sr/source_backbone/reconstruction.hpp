// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "prepare.hpp"

namespace rw::sr::source_backbone {

inline constexpr std::size_t filter_elements=32768;

struct SurfaceGeometry {
    Geometry output, history, luma, neural;
};

SurfaceGeometry reconstruction_geometry(int width,int height);
std::vector<std::uint16_t> decode_reconstruction_filter(std::span<const std::byte> raw);

struct ReconstructionOutput {
    float* rgba;                   // F32 [2H,2W,4].
    std::uint16_t* ycocg;           // FP16 [history H,history W,4].
    std::uint16_t* luma;            // FP16 [luma H,luma W].
    std::uint16_t* hidden;          // FP16 [neural H,neural W,4].
};

// This source profile binds zero jitter, unit exposure and input-pixel
// motion scaled by 2. The caller provides separate previous and next planes.
cudaError_t reconstruct(const PrepareInput& input,const std::uint16_t* previous_luma,
                        const std::uint16_t* field,const std::uint16_t* filter,
                        ReconstructionOutput output,cudaStream_t stream,const HistoryBinding* history=nullptr);

cudaError_t enqueue_frame(const PrepareInput& input,const std::uint16_t* previous_luma,
                         unsigned phase,const Weights& layout,const std::uint16_t* device_weights,
                         const std::uint16_t* device_filter,std::uint16_t* prepared,std::uint16_t* field,
                         ReconstructionOutput output,void* workspace,std::size_t bytes,cudaStream_t stream,
                         BackboneOptions backbone_options={},const cudaEvent_t* stage_ends=nullptr,
                         const HistoryBinding* history=nullptr);

} // namespace rw::sr::source_backbone
