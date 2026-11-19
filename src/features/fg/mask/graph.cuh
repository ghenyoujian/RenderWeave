#pragma once

#include "elementwise.cuh"
#include "pointwise.cuh"
#include "pooling.cuh"
#include "spatial.cuh"
#include "upscale.cuh"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace rw::fg::mask {

constexpr size_t scratch_slots = 7;
constexpr size_t input_slot = 7;
constexpr size_t output_slot = 8;
constexpr size_t weight_count = 17;

enum class MaskOperator { spatial, pointwise, pool, upscale, add };

struct MaskStage {
    MaskOperator operation;
    unsigned left, right, destination, level, channels, original_slot;
    int layer = -1;
};

// Scratch slots 0..6 correspond to the seven original resources at
// 0x1250, 0x1268, 0x1280, 0x1298, 0x12b0, 0x12c8 and 0x12e0.
inline constexpr std::array mask_stages{
    MaskStage{MaskOperator::spatial,   7, 0, 0, 0,  32, 0x0fb8,  0},
    MaskStage{MaskOperator::pool,      0, 0, 1, 1,  32, 0x0fd0},
    MaskStage{MaskOperator::pointwise, 1, 0, 2, 1,  64, 0x0fe8,  1},
    MaskStage{MaskOperator::pool,      2, 0, 1, 2,  64, 0x0fd0},
    MaskStage{MaskOperator::spatial,   1, 0, 3, 2, 128, 0x1008,  2},
    MaskStage{MaskOperator::pool,      3, 0, 1, 3, 128, 0x0fd0},
    MaskStage{MaskOperator::spatial,   1, 0, 4, 3, 128, 0x1028,  3},
    MaskStage{MaskOperator::pool,      4, 0, 1, 4, 128, 0x0fd0},
    MaskStage{MaskOperator::spatial,   1, 0, 5, 4, 256, 0x1048,  4},
    MaskStage{MaskOperator::pool,      5, 0, 1, 5, 256, 0x0fd0},
    MaskStage{MaskOperator::pointwise, 1, 0, 6, 5, 512, 0x1068,  5},
    MaskStage{MaskOperator::pointwise, 6, 0, 1, 5, 256, 0x1080,  6},
    MaskStage{MaskOperator::upscale,   1, 0, 6, 4, 256, 0x1098},
    MaskStage{MaskOperator::add,       6, 5, 1, 4, 256, 0x10b0},
    MaskStage{MaskOperator::pointwise, 1, 0, 5, 4, 256, 0x10c8,  7},
    MaskStage{MaskOperator::pointwise, 5, 0, 6, 4, 128, 0x10e0,  8},
    MaskStage{MaskOperator::upscale,   6, 0, 1, 3, 128, 0x10f8},
    MaskStage{MaskOperator::add,       1, 4, 5, 3, 128, 0x1110},
    MaskStage{MaskOperator::pointwise, 5, 0, 4, 3, 128, 0x1128,  9},
    MaskStage{MaskOperator::pointwise, 4, 0, 5, 3, 128, 0x1128, 10},
    MaskStage{MaskOperator::upscale,   5, 0, 1, 2, 128, 0x10f8},
    MaskStage{MaskOperator::add,       1, 3, 4, 2, 128, 0x1110},
    MaskStage{MaskOperator::pointwise, 4, 0, 1, 2, 128, 0x1160, 11},
    MaskStage{MaskOperator::pointwise, 1, 0, 3, 2,  64, 0x1178, 12},
    MaskStage{MaskOperator::upscale,   3, 0, 1, 1,  64, 0x1190},
    MaskStage{MaskOperator::add,       1, 2, 3, 1,  64, 0x11a8},
    MaskStage{MaskOperator::pointwise, 3, 0, 1, 1,  64, 0x11c0, 13},
    MaskStage{MaskOperator::pointwise, 1, 0, 4, 1,  32, 0x11d8, 14},
    MaskStage{MaskOperator::upscale,   4, 0, 1, 0,  32, 0x11f0},
    MaskStage{MaskOperator::add,       1, 0, 2, 0,  32, 0x1208},
    MaskStage{MaskOperator::pointwise, 2, 0, 0, 0,  32, 0x1220, 15},
    MaskStage{MaskOperator::pointwise, 0, 0, 8, 0,  16, 0x1238, 16},
};

struct MaskTensor {
    unsigned width, height, channels;

    size_t bytes() const { return size_t(width) * height * channels * sizeof(uint16_t); }
};

struct MaskPlan {
    MaskTensor input;
    std::array<MaskTensor, mask_stages.size()> outputs;
    std::array<size_t, scratch_slots> scratch_offsets;
    std::array<size_t, scratch_slots> scratch_sizes;
    size_t workspace_bytes;
};

inline MaskPlan plan_mask(unsigned width, unsigned height) {
    // Five floor divisions must leave nonempty inputs. All seven scratch
    // capacities and the external tensors are bounded by the full extent.
    if (width < 32 || height < 32 ||
        size_t(width) > std::numeric_limits<size_t>::max() / height / (512 * sizeof(uint16_t) * scratch_slots)) {
        throw std::invalid_argument("FG mask extent cannot be represented");
    }
    MaskPlan plan{};
    plan.input = {width, height, 48};
    for (size_t i = 0; i < mask_stages.size(); ++i) {
        const auto& stage = mask_stages[i];
        const bool doubled = stage.operation == MaskOperator::upscale;
        const unsigned source_level = stage.level + unsigned(doubled);
        const MaskTensor shape{(width >> source_level) << unsigned(doubled),
                               (height >> source_level) << unsigned(doubled), stage.channels};
        plan.outputs[i] = shape;
        if (stage.destination < scratch_slots) {
            auto& bytes = plan.scratch_sizes[stage.destination];
            bytes = std::max(bytes, shape.bytes());
        }
    }
    for (size_t i = 0; i < scratch_slots; ++i) {
        plan.scratch_offsets[i] = plan.workspace_bytes;
        plan.workspace_bytes += plan.scratch_sizes[i];
    }
    return plan;
}

template<unsigned Channels>
cudaError_t launch_mask_decoder(const MaskStage& stage, const MaskTensor& left, const MaskTensor& right,
                                const MaskTensor& out, const uint32_t* a, const uint32_t* b, uint32_t* dst,
                                cudaStream_t stream) {
    if (stage.operation == MaskOperator::upscale) {
        return launch_upscale<Channels>(reinterpret_cast<const UpscaleChannels*>(a),
                                        reinterpret_cast<UpscaleChannels*>(dst),
                                        {left.height, left.width}, {left.height * 2, left.width * 2},
                                        {out.height, out.width}, stream);
    }
    return launch_add_repeated<Channels>(reinterpret_cast<const __half2*>(a), reinterpret_cast<const __half2*>(b),
                                         reinterpret_cast<__half2*>(dst), {left.width, left.height},
                                         {right.width, right.height}, {out.width, out.height}, stream);
}

// The caller retains the original weights, both device tensors, and this
// execution slot's aligned device workspace through stream completion.
template<class Observe>
cudaError_t launch_mask(const MaskPlan& plan, const std::array<const uint32_t*, weight_count>& weights,
                        const uint32_t* input, uint32_t* output, void* workspace,
                        cudaStream_t stream, Observe observe) {
    std::array<uint32_t*, output_slot + 1> buffers{};
    std::array<MaskTensor, output_slot + 1> shapes{};
    for (size_t i = 0; i < scratch_slots; ++i) {
        buffers[i] = reinterpret_cast<uint32_t*>(static_cast<char*>(workspace) + plan.scratch_offsets[i]);
    }
    buffers[output_slot] = output;
    shapes[input_slot] = plan.input;

    for (size_t i = 0; i < mask_stages.size(); ++i) {
        const auto& stage = mask_stages[i];
        const auto left = shapes[stage.left];
        const auto right = shapes[stage.right];
        const auto out = plan.outputs[i];
        const auto* src = stage.left == input_slot ? input : buffers[stage.left];
        auto* dst = buffers[stage.destination];
        cudaError_t status;
        switch (stage.operation) {
        case MaskOperator::spatial:
            if (stage.layer == 4) {
                status = launch_spatial<SpatialAccumulation::column_then_channel>(
                    weights[stage.layer], src, dst, {out.width, out.height}, left.channels, out.channels, stream);
            } else {
                status = launch_spatial<SpatialAccumulation::channel_then_column>(
                    weights[stage.layer], src, dst, {out.width, out.height}, left.channels, out.channels, stream);
            }
            break;
        case MaskOperator::pointwise:
            status = launch_pointwise(weights[stage.layer], src, dst, {out.width, out.height},
                                       left.channels, out.channels, stage.layer != 16, stream);
            break;
        case MaskOperator::pool:
            status = launch_pool_2x2(reinterpret_cast<const __half2*>(src), reinterpret_cast<__half2*>(dst),
                                      left.width, left.height, out.width, out.height, out.channels, stream);
            break;
        case MaskOperator::upscale:
        case MaskOperator::add:
            switch (out.channels) {
            case 32: status = launch_mask_decoder<32>(stage, left, right, out, src, buffers[stage.right], dst, stream); break;
            case 64: status = launch_mask_decoder<64>(stage, left, right, out, src, buffers[stage.right], dst, stream); break;
            case 128: status = launch_mask_decoder<128>(stage, left, right, out, src, buffers[stage.right], dst, stream); break;
            default: status = launch_mask_decoder<256>(stage, left, right, out, src, buffers[stage.right], dst, stream); break;
            }
            break;
        }
        if (status != cudaSuccess) return status;
        shapes[stage.destination] = out;
        status = observe(i, dst, out.bytes(), stream);
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

inline cudaError_t launch_mask(const MaskPlan& plan, const std::array<const uint32_t*, weight_count>& weights,
                               const uint32_t* input, uint32_t* output, void* workspace, cudaStream_t stream) {
    return launch_mask(plan, weights, input, output, workspace, stream,
                       [](size_t, const uint32_t*, size_t, cudaStream_t) { return cudaSuccess; });
}

}
