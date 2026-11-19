#include "rw/rw_native_nr_temporal.h"
#include "pts_order.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

struct rw_nr_temporal_session {
    std::mutex mutex;
    rw_nr_temporal_state state{};
    uint64_t session_id=0;
};

namespace {
using Digest = std::array<uint8_t, 32>;

bool nonzero(const uint8_t* value) {
    for (size_t i = 0; i < 32; ++i) if (value[i] != 0) return true;
    return false;
}

bool equal_digest(const uint8_t* left, const uint8_t* right) {
    return std::memcmp(left, right, 32) == 0;
}

bool add(uint64_t left, uint64_t right, uint64_t& out) {
    if (left > std::numeric_limits<uint64_t>::max() - right) return false;
    out = left + right;
    return true;
}

bool multiply(uint64_t left, uint64_t right, uint64_t& out) {
    if (right && left > std::numeric_limits<uint64_t>::max() / right) return false;
    out = left * right;
    return true;
}

bool later_pts(const rw_nr_temporal_frame& frame, const rw_nr_temporal_state& state) {
    return rw::detail::later_pts(frame.pts, frame.time_base_num, frame.time_base_den,
                                 state.pts, state.time_base_num, state.time_base_den);
}

void initialize_result(const rw_nr_temporal_frame& frame,
                       rw_nr_temporal_preflight_result& result) {
    std::memset(&result, 0, sizeof(result));
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    result.native_execution_available = RW_NR_TEMPORAL_NATIVE_EXECUTION_AVAILABLE;
    result.required_semantics = frame.required_semantics;
}

rw_status_t fail(rw_nr_temporal_preflight_result& result, uint32_t code) {
    result.code = code;
    result.history_commit_allowed = 0;
    return code == RW_NR_TEMPORAL_CODE_ABI_MISMATCH ? RW_ABI_MISMATCH :
           code == RW_NR_TEMPORAL_CODE_STALE ? RW_STALE : RW_INVALID_ARGUMENT;
}

const rw_nr_temporal_resource* find_resource(const rw_nr_temporal_frame& frame,
                                             uint32_t semantic) {
    const rw_nr_temporal_resource* found = nullptr;
    for (uint32_t i = 0; i < frame.resource_count; ++i) {
        const auto& row = frame.resources[i];
        if (row.resource.resource_semantic != semantic) continue;
        if (found) return nullptr; // duplicate semantics are invalid
        found = &row;
    }
    return found;
}

uint32_t validate_resource(const rw_nr_temporal_frame& frame,
                           const rw_nr_temporal_resource& row,
                           const rw_nr_temporal_resource* color) {
    const auto& resource = row.resource;
    if (row.struct_size != sizeof(row) || row.abi_version != RW_NR_TEMPORAL_ABI_VERSION)
        return RW_NR_TEMPORAL_CODE_ABI_MISMATCH;
    if (resource.struct_size != sizeof(resource) || resource.validity > RW_VALIDITY_INVALID)
        return RW_NR_TEMPORAL_CODE_RESOURCE;
    if (resource.validity == RW_VALIDITY_ABSENT) {
        if (row.owner_id || row.owner_session_id || row.owner_epoch || row.owner_frame_id)
            return RW_NR_TEMPORAL_CODE_OWNER;
        return RW_NR_TEMPORAL_CODE_VALID;
    }
    if (resource.validity != RW_VALIDITY_VALID || row.owner_id == 0 ||
        row.owner_session_id != frame.session_id ||
        resource.provenance < RW_PROVENANCE_CAPTURED ||
        resource.provenance > RW_PROVENANCE_RENDERER_DERIVED)
        return RW_NR_TEMPORAL_CODE_OWNER;
    if (resource.metadata.struct_size < sizeof(resource.metadata) || resource.metadata.reserved)
        return RW_NR_TEMPORAL_CODE_RESOURCE;
    uint64_t required = 0;
    const auto status = rw_validate_tensor(&resource.tensor, &required);
    if (status != RW_OK) return RW_NR_TEMPORAL_CODE_RESOURCE;
    if (resource.tensor.device_kind != frame.device_kind ||
        resource.tensor.device_ordinal != frame.device_ordinal)
        return RW_NR_TEMPORAL_CODE_RESOURCE;
    if (color && resource.resource_semantic != RW_RESOURCE_COLOR &&
        (resource.tensor.ndim < 2 || color->resource.tensor.ndim < 2 ||
         resource.tensor.shape[0] != color->resource.tensor.shape[0] ||
         resource.tensor.shape[1] != color->resource.tensor.shape[1]))
        return RW_NR_TEMPORAL_CODE_SHAPE;
    if (resource.resource_semantic == RW_RESOURCE_COLOR ||
        resource.resource_semantic == RW_RESOURCE_HISTORY) {
        if (resource.tensor.ndim != 3 ||
            (resource.tensor.shape[2] != 3 && resource.tensor.shape[2] != 4) ||
            resource.metadata.color_space != RW_COLOR_SRGB_CODE ||
            resource.metadata.dynamic_range != RW_DYNAMIC_RANGE_SDR)
            return RW_NR_TEMPORAL_CODE_SHAPE;
    } else if (resource.resource_semantic == RW_RESOURCE_DEPTH) {
        if (resource.tensor.ndim != 2 ||
            (resource.tensor.dtype != RW_DTYPE_F16 && resource.tensor.dtype != RW_DTYPE_F32) ||
            resource.metadata.depth_definition != RW_DEPTH_VIEW_Z)
            return RW_NR_TEMPORAL_CODE_SHAPE;
    } else if (resource.resource_semantic == RW_RESOURCE_MOTION) {
        if (resource.tensor.ndim != 3 || resource.tensor.shape[2] != 2 ||
            resource.metadata.motion_direction != frame.motion_direction ||
            resource.metadata.motion_units != frame.motion_units ||
            resource.metadata.jitter_included != frame.jitter_included)
            return RW_NR_TEMPORAL_CODE_CONTROL;
    } else if (resource.resource_semantic == RW_RESOURCE_VALIDITY) {
        if (resource.tensor.ndim != 2 || resource.tensor.dtype != RW_DTYPE_U8)
            return RW_NR_TEMPORAL_CODE_SHAPE;
    } else {
        return RW_NR_TEMPORAL_CODE_RESOURCE;
    }
    return RW_NR_TEMPORAL_CODE_VALID;
}

uint32_t validate_frame(const rw_nr_temporal_frame& frame,
                        const rw_nr_temporal_state* state,
                        rw_nr_temporal_preflight_result& result) {
    initialize_result(frame, result);
    if (frame.struct_size != sizeof(frame) || frame.abi_version != RW_NR_TEMPORAL_ABI_VERSION ||
        frame.flags || frame.reserved || frame.session_id == 0 || frame.sequence_id == 0 ||
        frame.frame_id == RW_NR_TEMPORAL_NO_PREDECESSOR || frame.epoch == 0 ||
        frame.time_base_num <= 0 || frame.time_base_den <= 0 ||
        frame.scene_epoch == 0 || frame.device_kind < RW_DEVICE_CPU ||
        frame.device_kind > RW_DEVICE_ASCEND || frame.device_ordinal < 0 ||
        !frame.resources || frame.resource_count == 0 || frame.resource_count > 64 ||
        !nonzero(frame.sequence_identity) || !nonzero(frame.model_fingerprint) ||
        !nonzero(frame.rng_algorithm) || !nonzero(frame.history_codec_sha256) ||
        !nonzero(frame.sampler_sha256))
        return fail(result, RW_NR_TEMPORAL_CODE_INVALID_ARGUMENT);
    if (frame.required_semantics & ~((UINT64_C(1) << (RW_RESOURCE_CAMERA + 1)) - 2))
        return fail(result, RW_NR_TEMPORAL_CODE_RESOURCE);
    if ((frame.motion_direction != RW_MOTION_CURRENT_TO_PREVIOUS) ||
        frame.motion_units != RW_MOTION_PIXELS || frame.jitter_included != RW_JITTER_EXCLUDED ||
        !frame.jitter_explicitly_disabled || frame.jitter_x_bits || frame.jitter_y_bits ||
        frame.exposure_scale_bits != 0x3f800000u || !frame.exposure_explicit_no_transform)
        return fail(result, RW_NR_TEMPORAL_CODE_CONTROL);

    const auto* color = find_resource(frame, RW_RESOURCE_COLOR);
    if (!color || color->resource.validity != RW_VALIDITY_VALID)
        return fail(result, RW_NR_TEMPORAL_CODE_MISSING_COLOR);
    uint64_t present = 0;
    for (uint32_t i = 0; i < frame.resource_count; ++i) {
        const auto& row = frame.resources[i];
        const uint32_t semantic = row.resource.resource_semantic;
        if (semantic < RW_RESOURCE_COLOR || semantic > RW_RESOURCE_CAMERA ||
            (present & RW_SEMANTIC_BIT(semantic)))
            return fail(result, RW_NR_TEMPORAL_CODE_RESOURCE);
        const auto code = validate_resource(frame, row, color);
        if (code != RW_NR_TEMPORAL_CODE_VALID) return fail(result, code);
        if (row.resource.validity == RW_VALIDITY_VALID) present |= RW_SEMANTIC_BIT(semantic);
    }
    result.present_semantics = present;
    if (!frame.reset && !(present & RW_SEMANTIC_BIT(RW_RESOURCE_MOTION)))
        return fail(result, RW_NR_TEMPORAL_CODE_MISSING_MOTION);
    if ((present & frame.required_semantics) != frame.required_semantics)
        return fail(result, RW_NR_TEMPORAL_CODE_RESOURCE);
    if (!(color->resource.tensor.ndim == 3 && color->resource.tensor.shape[2] == 4))
        return fail(result, RW_NR_TEMPORAL_CODE_SHAPE);

    const auto* history = find_resource(frame, RW_RESOURCE_HISTORY);
    if (frame.reset) {
        if (frame.expected_predecessor != RW_NR_TEMPORAL_NO_PREDECESSOR ||
            (history && history->resource.validity == RW_VALIDITY_VALID))
            return fail(result, RW_NR_TEMPORAL_CODE_RESET);
        if (state && state->valid && (frame.epoch <= state->epoch ||
            frame.session_id != state->session_id))
            return fail(result, RW_NR_TEMPORAL_CODE_RESET);
    } else {
        if (!state || state->struct_size != sizeof(*state) ||
            state->abi_version != RW_NR_TEMPORAL_ABI_VERSION || !state->valid ||
            state->session_id != frame.session_id || state->sequence_id != frame.sequence_id ||
            state->epoch != frame.epoch || state->predecessor != frame.expected_predecessor)
            return fail(result, RW_NR_TEMPORAL_CODE_STALE);
        if (!(present & RW_SEMANTIC_BIT(RW_RESOURCE_MOTION)))
            return fail(result, RW_NR_TEMPORAL_CODE_MISSING_MOTION);
        if (!history || history->resource.validity != RW_VALIDITY_VALID)
            return fail(result, RW_NR_TEMPORAL_CODE_MISSING_HISTORY);
        if (history->owner_id != state->history_owner_id ||
            history->owner_epoch != state->history_epoch ||
            history->owner_frame_id != state->history_frame ||
            history->owner_frame_id != frame.expected_predecessor ||
            !equal_digest(frame.history_codec_sha256, state->history_codec_sha256) ||
            !equal_digest(frame.sampler_sha256, state->sampler_sha256) ||
            !equal_digest(frame.sequence_identity, state->sequence_identity))
            return fail(result, RW_NR_TEMPORAL_CODE_HISTORY);
        if (!later_pts(frame, *state)) return fail(result, RW_NR_TEMPORAL_CODE_PTS);
    }
    result.code = RW_NR_TEMPORAL_CODE_NATIVE_UNAVAILABLE;
    result.history_commit_allowed = 0;
    return RW_OK;
}
}

extern "C" {
rw_status_t rw_nr_temporal_preflight(const rw_nr_temporal_frame* frame,
                                     const rw_nr_temporal_state* state,
                                     rw_nr_temporal_preflight_result* result) {
    if (!frame || !result || result->struct_size != sizeof(*result) ||
        result->abi_version != RW_NR_TEMPORAL_ABI_VERSION)
        return RW_ABI_MISMATCH;
    return validate_frame(*frame, state, *result);
}

rw_status_t rw_nr_temporal_commit(const rw_nr_temporal_state* before,
                                  const rw_nr_temporal_frame* frame,
                                  uint32_t physical_completion, uint32_t output_valid,
                                  uint64_t next_history_owner_id,
                                  rw_nr_temporal_state* after) {
    if (!before || !frame || !after || after->struct_size != sizeof(*after) ||
        after->abi_version != RW_NR_TEMPORAL_ABI_VERSION)
        return RW_ABI_MISMATCH;
    if (physical_completion > 1 || output_valid > 1)
        return RW_INVALID_ARGUMENT;
    rw_nr_temporal_preflight_result result{};
    result.struct_size = sizeof(result);
    result.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    const auto status = rw_nr_temporal_preflight(frame, before, &result);
    if (status != RW_OK) return status;
    if (!physical_completion) return RW_NOT_READY;
    if (!output_valid) return RW_OUTPUT_INVALID;
    if (!next_history_owner_id) return RW_INVALID_ARGUMENT;
    for (uint32_t i = 0; i < frame->resource_count; ++i)
        if (frame->resources[i].owner_id == next_history_owner_id)
            return RW_INVALID_ARGUMENT;
    rw_nr_temporal_state next{};
    next.struct_size = sizeof(next);
    next.abi_version = RW_NR_TEMPORAL_ABI_VERSION;
    next.valid = 1;
    next.session_id = frame->session_id;
    next.sequence_id = frame->sequence_id;
    next.epoch = frame->epoch;
    next.predecessor = frame->frame_id;
    next.pts = frame->pts;
    next.time_base_num = frame->time_base_num;
    next.time_base_den = frame->time_base_den;
    next.scene_epoch = frame->scene_epoch;
    next.history_owner_id = next_history_owner_id;
    next.history_epoch = frame->epoch;
    next.history_frame = frame->frame_id;
    std::memcpy(next.sequence_identity, frame->sequence_identity, 32);
    std::memcpy(next.model_fingerprint, frame->model_fingerprint, 32);
    std::memcpy(next.rng_algorithm, frame->rng_algorithm, 32);
    std::memcpy(next.history_codec_sha256, frame->history_codec_sha256, 32);
    std::memcpy(next.sampler_sha256, frame->sampler_sha256, 32);
    *after = next;
    return RW_OK;
}

rw_status_t rw_nr_temporal_memory_required(const rw_nr_temporal_memory_plan* plan,
                                           uint64_t* required_bytes) {
    if (!plan || !required_bytes) return RW_INVALID_ARGUMENT;
    *required_bytes = 0;
    if (plan->struct_size != sizeof(*plan) || plan->abi_version != RW_NR_TEMPORAL_ABI_VERSION)
        return RW_ABI_MISMATCH;
    uint64_t slots = 0, per_session = 0, total = plan->shared_weights_bytes;
    if (!multiply(plan->bytes_per_slot, plan->slot_count, slots) ||
        !add(plan->committed_history_bytes, plan->provisional_history_bytes, per_session) ||
        !multiply(per_session, plan->session_count, per_session) ||
        !add(total, slots, total) || !add(total, per_session, total) ||
        !add(total, plan->import_output_bytes, total) ||
        !add(total, plan->graph_handle_bytes, total) ||
        !add(total, plan->reserve_quarantine_bytes, total))
        return RW_OVERFLOW;
    *required_bytes = total;
    return total <= plan->budget_bytes ? RW_OK : RW_RESOURCE_EXHAUSTED;
}

rw_status_t rw_nr_temporal_session_create(uint64_t session_id, uint64_t epoch,
                                           rw_nr_temporal_session_t* output) {
    if (!output || !session_id || !epoch) return RW_INVALID_ARGUMENT;
    *output=nullptr;
    auto* session=new (std::nothrow) rw_nr_temporal_session;
    if (!session) return RW_RESOURCE_EXHAUSTED;
    session->session_id=session_id;
    session->state.struct_size=sizeof(session->state);
    session->state.abi_version=RW_NR_TEMPORAL_ABI_VERSION;
    session->state.session_id=session_id;
    session->state.epoch=epoch;
    session->state.predecessor=RW_NR_TEMPORAL_NO_PREDECESSOR;
    *output=session;
    return RW_OK;
}

rw_status_t rw_nr_temporal_session_snapshot(rw_nr_temporal_session_t session,
                                             rw_nr_temporal_state* output) {
    if (!session || !output || output->struct_size!=sizeof(*output) ||
        output->abi_version!=RW_NR_TEMPORAL_ABI_VERSION) return RW_ABI_MISMATCH;
    std::lock_guard lock(session->mutex); *output=session->state; return RW_OK;
}

rw_status_t rw_nr_temporal_session_reset(rw_nr_temporal_session_t session,
                                          uint64_t epoch) {
    if (!session || !epoch) return RW_INVALID_ARGUMENT;
    std::lock_guard lock(session->mutex);
    if (session->state.valid && epoch<=session->state.epoch) return RW_STALE;
    const auto identity=session->session_id;
    session->state={};
    session->state.struct_size=sizeof(session->state);
    session->state.abi_version=RW_NR_TEMPORAL_ABI_VERSION;
    session->state.session_id=identity;
    session->state.epoch=epoch;
    session->state.predecessor=RW_NR_TEMPORAL_NO_PREDECESSOR;
    return RW_OK;
}

rw_status_t rw_nr_temporal_session_preflight(rw_nr_temporal_session_t session,
                                              const rw_nr_temporal_frame* frame,
                                              rw_nr_temporal_preflight_result* result) {
    if (!session || !frame) return RW_INVALID_ARGUMENT;
    std::lock_guard lock(session->mutex);
    if (frame->session_id!=session->session_id) return RW_INVALID_ARGUMENT;
    const rw_nr_temporal_state* state=session->state.valid?&session->state:nullptr;
    return rw_nr_temporal_preflight(frame,state,result);
}

rw_status_t rw_nr_temporal_session_commit(rw_nr_temporal_session_t session,
                                           const rw_nr_temporal_frame* frame,
                                           uint32_t physical_completion,
                                           uint32_t output_valid,
                                           uint64_t next_history_owner_id,
                                           rw_nr_temporal_state* output) {
    if (!session || !frame || !output || output->struct_size!=sizeof(*output) ||
        output->abi_version!=RW_NR_TEMPORAL_ABI_VERSION) return RW_ABI_MISMATCH;
    std::lock_guard lock(session->mutex);
    if (frame->session_id!=session->session_id) return RW_INVALID_ARGUMENT;
    rw_nr_temporal_state next{};
    next.struct_size=sizeof(next); next.abi_version=RW_NR_TEMPORAL_ABI_VERSION;
    const auto status=rw_nr_temporal_commit(&session->state,frame,physical_completion,
                                             output_valid,next_history_owner_id,&next);
    if (status!=RW_OK) return status;
    session->state=next; *output=next; return RW_OK;
}

void rw_nr_temporal_session_release(rw_nr_temporal_session_t session) { delete session; }
}
