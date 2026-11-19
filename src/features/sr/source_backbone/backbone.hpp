// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rw::sr::source_backbone {

inline constexpr int stage_count = 11;
inline constexpr int input_channels = 16;
inline constexpr int reconstruction_channels = 40;
inline constexpr std::size_t packed_bytes = 3545824;

struct HeadWeights {
    std::size_t query{}, key{}, value{}, position{}, projection{};
};

// Offsets count FP16 elements in the decoded, row-major weight allocation.
struct StageWeights {
    std::size_t embedding{}, embedding_bias{}, norm{}, attention_bias{};
    std::size_t ff_scale{}, ff_output_bias{}, ff_up{}, ff_bias{}, ff_down{};
    std::size_t merge{}, merge_bias{}, output{}, output_bias{};
    std::array<HeadWeights, 8> heads{};
};

struct Weights {
    std::vector<std::uint16_t> elements;
    std::array<StageWeights, stage_count> stages;
};

// Decode the original captured 310.7.0 K backbone's exact parameter bytes.
Weights decode_original(std::span<const std::byte> packed);

struct Geometry {
    int width, height;  // Neural field, after SR's own geometry calculation.
};

Geometry network_geometry(int input_width, int input_height);
std::size_t workspace_bytes(Geometry geometry, unsigned phase);

struct BackboneOptions {
    bool tiled=false;
    bool parallel_reduce=false;
    bool reuse_matrix_rows=false;
    bool original_fp16=false;
};

// Reference diagnostics borrow both first-layer outputs until this workspace is reused.
struct FirstEncoderOutput {
    const std::uint16_t* skip = nullptr;
    const std::uint16_t* downsampled = nullptr;
};

cudaError_t enqueue_first_encoder(Geometry geometry, unsigned phase,
                                  const Weights& layout, const std::uint16_t* device_weights,
                                  const std::uint16_t* prepared, void* workspace, std::size_t bytes,
                                  cudaStream_t stream, FirstEncoderOutput& output,
                                  BackboneOptions options = {}, bool tensor_core = false);

// The diagnostic callback runs on the caller before the next stage consumes these
// buffers. It can install a reference input for an independent layer comparison.
using StageObserver = cudaError_t (*)(void* user, int stage, std::uint16_t* output,
                                       std::uint16_t* encoder_skip, cudaStream_t stream);

struct BackboneProbeArithmetic {
    unsigned tensor_core_stages = 0;
    bool half_attention_probability = false;
    bool balanced_wide_norm = false;
    bool fused_half_activation = false;
};

cudaError_t enqueue_backbone_probe(Geometry geometry, unsigned phase,
                                   const Weights& layout, const std::uint16_t* device_weights,
                                   const std::uint16_t* prepared, std::uint16_t* reconstruction,
                                   void* workspace, std::size_t bytes, cudaStream_t stream,
                                   BackboneOptions options, BackboneProbeArithmetic arithmetic,
                                   StageObserver observe, void* user);

// Submit the 16 → 40 channel neural backbone on the caller's stream. Each
// slot owns its workspace. The uploaded FP16 weights are immutable and may
// be shared by slots. Input and output are row-major HWC device tensors.
cudaError_t enqueue(Geometry geometry, unsigned phase,
                    const Weights& layout, const std::uint16_t* device_weights,
                    const std::uint16_t* prepared, std::uint16_t* reconstruction,
                    void* workspace, std::size_t bytes, cudaStream_t stream, BackboneOptions options={});

} // namespace rw::sr::source_backbone
