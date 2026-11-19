#include "rw/rw_native_nr_temporal.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
uint64_t checks = 0;
#define REQUIRE(value) do { ++checks; if (!(value)) { \
    std::cerr << __FILE__ << ':' << __LINE__ << ": " #value " failed\n"; \
    return 1; } } while (false)

template <size_t N>
rw_tensor_view tensor(float (&data)[N], int64_t height, int64_t width, int64_t channels) {
    rw_tensor_view result{};
    result.struct_size = sizeof(result);
    result.dtype = RW_DTYPE_F32;
    result.device_kind = RW_DEVICE_CPU;
    result.device_ordinal = 0;
    result.ndim = 3;
    result.access_mode = RW_ACCESS_READ;
    result.shape[0] = height;
    result.shape[1] = width;
    result.shape[2] = channels;
    result.strides_bytes[2] = sizeof(float);
    result.strides_bytes[1] = channels * sizeof(float);
    result.strides_bytes[0] = width * channels * sizeof(float);
    result.base = data;
    result.allocation_bytes = sizeof(data);
    return result;
}

rw_nr_temporal_resource resource(uint32_t semantic, rw_tensor_view view, uint64_t owner,
                                 uint64_t session, uint64_t epoch, uint64_t frame) {
    rw_nr_temporal_resource result{};
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    result.resource.struct_size = sizeof(result.resource);
    result.resource.resource_semantic = semantic;
    result.resource.validity = RW_VALIDITY_VALID;
    result.resource.provenance = RW_PROVENANCE_RENDERER_DERIVED;
    result.resource.tensor = view;
    result.resource.metadata.struct_size = sizeof(result.resource.metadata);
    result.resource.metadata.color_space = RW_COLOR_SRGB_CODE;
    result.resource.metadata.dynamic_range = RW_DYNAMIC_RANGE_SDR;
    result.resource.metadata.depth_definition = RW_DEPTH_VIEW_Z;
    result.resource.metadata.motion_direction = RW_MOTION_CURRENT_TO_PREVIOUS;
    result.resource.metadata.motion_units = RW_MOTION_PIXELS;
    result.resource.metadata.jitter_included = RW_JITTER_EXCLUDED;
    result.owner_id = owner;
    result.owner_session_id = session;
    result.owner_epoch = epoch;
    result.owner_frame_id = frame;
    return result;
}

void digest(uint8_t* value, uint8_t seed) {
    for (size_t i = 0; i < 32; ++i) value[i] = static_cast<uint8_t>(seed + i);
}

rw_nr_temporal_frame base_frame() {
    rw_nr_temporal_frame result{};
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    result.session_id = 7;
    result.sequence_id = 9;
    result.frame_id = 0;
    result.epoch = 1;
    result.expected_predecessor = RW_NR_TEMPORAL_NO_PREDECESSOR;
    result.pts = 0;
    result.time_base_num = 1;
    result.time_base_den = 30;
    result.scene_epoch = 1;
    result.device_kind = RW_DEVICE_CPU;
    result.device_ordinal = 0;
    result.required_semantics = RW_SEMANTIC_BIT(RW_RESOURCE_COLOR);
    result.motion_direction = RW_MOTION_CURRENT_TO_PREVIOUS;
    result.motion_units = RW_MOTION_PIXELS;
    result.jitter_included = RW_JITTER_EXCLUDED;
    result.jitter_explicitly_disabled = 1;
    result.exposure_scale_bits = 0x3f800000u;
    result.exposure_explicit_no_transform = 1;
    digest(result.sequence_identity, 1);
    digest(result.model_fingerprint, 33);
    digest(result.rng_algorithm, 65);
    digest(result.history_codec_sha256, 97);
    digest(result.sampler_sha256, 129);
    return result;
}

int run() {
    float color_data[2 * 3 * 4]{};
    float motion_data[2 * 3 * 2]{};
    float history_data[2 * 3 * 4]{};
    auto color = resource(RW_RESOURCE_COLOR, tensor(color_data, 2, 3, 4), 11, 7, 1, 0);
    auto motion = resource(RW_RESOURCE_MOTION, tensor(motion_data, 2, 3, 2), 12, 7, 1, 0);
    auto history = resource(RW_RESOURCE_HISTORY, tensor(history_data, 2, 3, 4), 99, 7, 1, 0);
    std::array<rw_nr_temporal_resource, 3> resources{color, motion, history};

    auto frame = base_frame();
    frame.reset = 1;
    frame.resources = resources.data();
    frame.resource_count = 1;
    rw_nr_temporal_preflight_result result{};
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    REQUIRE(rw_nr_temporal_preflight(&frame, nullptr, &result) == RW_OK);
    REQUIRE(result.code == RW_NR_TEMPORAL_CODE_NATIVE_UNAVAILABLE);
    REQUIRE(result.native_execution_available == 0);

    rw_nr_temporal_state before{};
    before.struct_size = sizeof(before);
    before.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    rw_nr_temporal_state after{};
    after.struct_size = sizeof(after);
    after.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    const auto before_invalid_flag = before;
    REQUIRE(rw_nr_temporal_commit(&before, &frame, 2, 1, 99, &after) == RW_INVALID_ARGUMENT);
    REQUIRE(std::memcmp(&before, &before_invalid_flag, sizeof(before)) == 0);
    REQUIRE(rw_nr_temporal_commit(&before, &frame, 1, 2, 99, &after) == RW_INVALID_ARGUMENT);
    REQUIRE(rw_nr_temporal_commit(&before, &frame, 1, 1, 99, &after) == RW_OK);
    REQUIRE(after.valid && after.predecessor == 0 && after.history_owner_id == 99);

    frame.reset = 0;
    frame.frame_id = 1;
    frame.expected_predecessor = 0;
    frame.pts = 1;
    frame.required_semantics |= RW_SEMANTIC_BIT(RW_RESOURCE_MOTION);
    resources[0] = color;
    resources[1] = motion;
    resources[2] = history;
    frame.resources = resources.data();
    frame.resource_count = 3;
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    REQUIRE(rw_nr_temporal_preflight(&frame, &after, &result) == RW_OK);
    REQUIRE(result.code == RW_NR_TEMPORAL_CODE_NATIVE_UNAVAILABLE);

    /* Missing motion and zero motion are different states. */
    resources[1] = history;
    resources[2] = motion;
    frame.resource_count = 2;
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    REQUIRE(rw_nr_temporal_preflight(&frame, &after, &result) == RW_INVALID_ARGUMENT);
    REQUIRE(result.code == RW_NR_TEMPORAL_CODE_MISSING_MOTION);
    resources[1] = motion;
    resources[2] = history;
    frame.resource_count = 3;
    std::fill(std::begin(motion_data), std::end(motion_data), 0.0f);
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    REQUIRE(rw_nr_temporal_preflight(&frame, &after, &result) == RW_OK);

    /* Wrong history owner and non-monotonic PTS fail independently. */
    resources[2].owner_id = 100;
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    REQUIRE(rw_nr_temporal_preflight(&frame, &after, &result) == RW_INVALID_ARGUMENT);
    REQUIRE(result.code == RW_NR_TEMPORAL_CODE_HISTORY);
    resources[2].owner_id = 99;
    frame.pts = 0;
    result = {};
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    REQUIRE(rw_nr_temporal_preflight(&frame, &after, &result) == RW_INVALID_ARGUMENT);
    REQUIRE(result.code == RW_NR_TEMPORAL_CODE_PTS);
    frame.pts = 1;

    /* A late result cannot mutate state, even when a newer epoch is present. */
    auto snapshot = after;
    auto stale = frame;
    stale.epoch = 2;
    REQUIRE(rw_nr_temporal_commit(&after, &stale, 1, 1, 101, &after) == RW_STALE);
    REQUIRE(std::memcmp(&after, &snapshot, sizeof(after)) == 0);

    rw_nr_temporal_memory_plan plan{};
    plan.struct_size = sizeof(plan);
    plan.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    plan.budget_bytes = 1000;
    plan.shared_weights_bytes = 100;
    plan.bytes_per_slot = 10;
    plan.slot_count = 2;
    plan.committed_history_bytes = 20;
    plan.provisional_history_bytes = 5;
    plan.session_count = 3;
    plan.import_output_bytes = 30;
    plan.graph_handle_bytes = 40;
    plan.reserve_quarantine_bytes = 50;
    uint64_t required = 0;
    REQUIRE(rw_nr_temporal_memory_required(&plan, &required) == RW_OK);
    REQUIRE(required == 100 + 20 + (25 * 3) + 30 + 40 + 50);
    plan.budget_bytes = required - 1;
    REQUIRE(rw_nr_temporal_memory_required(&plan, &required) == RW_RESOURCE_EXHAUSTED);
    plan.budget_bytes = std::numeric_limits<uint64_t>::max();
    plan.bytes_per_slot = std::numeric_limits<uint64_t>::max();
    plan.slot_count = 2;
    REQUIRE(rw_nr_temporal_memory_required(&plan, &required) == RW_OVERFLOW);

    /* The public ABI2 session owns the same CPU history/epoch control plane. */
    rw_nr_temporal_session_t session=nullptr;
    REQUIRE(rw_nr_temporal_session_create(7, 1, &session) == RW_OK);
    rw_nr_temporal_state session_state{};
    session_state.struct_size=sizeof(session_state);
    session_state.abi_version=RW_NR_TEMPORAL_ABI_VERSION;
    REQUIRE(rw_nr_temporal_session_snapshot(session, &session_state) == RW_OK);
    REQUIRE(session_state.valid == 0 && session_state.session_id == 7 && session_state.epoch == 1);
    auto session_result=rw_nr_temporal_preflight_result{};
    session_result.struct_size=sizeof(session_result);
    session_result.abi_version=RW_NR_TEMPORAL_ABI_VERSION;
    auto session_reset=base_frame();
    session_reset.reset=1;
    session_reset.resources=resources.data();
    session_reset.resource_count=1;
    REQUIRE(rw_nr_temporal_session_preflight(session, &session_reset, &session_result) == RW_OK);
    REQUIRE(session_result.code == RW_NR_TEMPORAL_CODE_NATIVE_UNAVAILABLE);
    session_state={}; session_state.struct_size=sizeof(session_state);
    session_state.abi_version=RW_NR_TEMPORAL_ABI_VERSION;
    REQUIRE(rw_nr_temporal_session_commit(session, &session_reset, 1, 1, 99, &session_state) == RW_OK);
    REQUIRE(session_state.valid && session_state.predecessor == 0 && session_state.history_owner_id == 99);
    REQUIRE(rw_nr_temporal_session_snapshot(session, &session_state) == RW_OK && session_state.valid);
    REQUIRE(rw_nr_temporal_session_reset(session, 2) == RW_OK);
    REQUIRE(rw_nr_temporal_session_snapshot(session, &session_state) == RW_OK &&
            session_state.valid == 0 && session_state.epoch == 2);
    rw_nr_temporal_session_release(session);
    std::cout << "temporal_preflight_checks=" << checks << "\n";
    return 0;
}
}

int main() { return run(); }
