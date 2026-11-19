#include "rw/rw_api.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <utility>

struct rw_context { rw_context_desc descriptor; };

namespace {
bool add(uint64_t a, uint64_t b, uint64_t& out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    out = a + b; return true;
}
bool multiply(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b) return false;
    out = a * b; return true;
}
uint64_t item_bytes(uint32_t dtype) {
    switch (dtype) { case RW_DTYPE_U8: case RW_DTYPE_E4M3FN: return 1;
        case RW_DTYPE_F16: return 2; case RW_DTYPE_F32: return 4; default: return 0; }
}
bool valid_device(uint32_t device, int32_t ordinal) {
    return device >= RW_DEVICE_CPU && device <= RW_DEVICE_ASCEND && ordinal >= 0 &&
        (device != RW_DEVICE_CPU || ordinal == 0);
}
}

extern "C" {
uint32_t rw_abi_version(void) { return RW_ABI_VERSION; }
const char* rw_status_string(rw_status_t status) {
    switch (status) {
        case RW_OK: return "OK"; case RW_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case RW_ABI_MISMATCH: return "ABI_MISMATCH"; case RW_UNSUPPORTED: return "UNSUPPORTED";
        case RW_RESOURCE_EXHAUSTED: return "RESOURCE_EXHAUSTED"; case RW_OVERFLOW: return "OVERFLOW";
        case RW_OUT_OF_BOUNDS: return "OUT_OF_BOUNDS"; case RW_NOT_READY: return "NOT_READY";
        case RW_STALE: return "STALE"; case RW_CANCELLED: return "CANCELLED";
        case RW_OUTPUT_INVALID: return "OUTPUT_INVALID"; case RW_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case RW_BUFFER_TOO_SMALL: return "BUFFER_TOO_SMALL"; default: return "UNKNOWN_STATUS";
    }
}
rw_status_t rw_context_create(const rw_context_desc* descriptor, rw_context_t* output) {
    if (!output) return RW_INVALID_ARGUMENT;
    *output = nullptr;
    if (!descriptor) return RW_INVALID_ARGUMENT;
    if (descriptor->struct_size < sizeof(*descriptor) || descriptor->abi_version != RW_ABI_VERSION)
        return RW_ABI_MISMATCH;
    if (!valid_device(descriptor->device_kind, descriptor->device_ordinal) || descriptor->flags ||
        descriptor->execution_slots == 0 || descriptor->max_sessions == 0 || descriptor->max_inflight == 0)
        return RW_INVALID_ARGUMENT;
    if (descriptor->device_kind != RW_DEVICE_CPU) return RW_UNSUPPORTED;
    // Arbitrary control JSON is not implemented by this diagnostic context.
    if (descriptor->config_json && descriptor->config_json[0] != '\0') return RW_UNSUPPORTED;
    auto* context = new (std::nothrow) rw_context{*descriptor};
    if (!context) return RW_RESOURCE_EXHAUSTED;
    context->descriptor.config_json = nullptr;
    *output = context;
    return RW_OK;
}
rw_status_t rw_context_query(rw_context_t context, const char* key, char* json, size_t* bytes) {
    if (!context || !key || !bytes) return RW_INVALID_ARGUMENT;
    if (std::strcmp(key, "capabilities") != 0) return RW_UNSUPPORTED;
    constexpr const char* capabilities =
        "{\"project_id\":\"renderweave\",\"abi_version\":1,"
        "\"scope\":\"cpu_contract_and_arithmetic_reference\","
        "\"native_model_execution_available\":false,\"supported_features\":[],"
        "\"tools\":[\"e4m3fn_rne_satfinite\",\"tensor_bounds\",\"frame_contract\","
        "\"budget_arithmetic\",\"state_transaction_guard\"]}";
    const size_t needed = std::strlen(capabilities) + 1;
    const size_t available = *bytes;
    *bytes = needed;
    if (!json) return RW_OK;
    if (available < needed) return RW_BUFFER_TOO_SMALL;
    std::memcpy(json, capabilities, needed);
    return RW_OK;
}
void rw_context_release(rw_context_t context) { delete context; }

rw_status_t rw_validate_tensor(const rw_tensor_view* tensor, uint64_t* required) {
    if (!tensor || !required) return RW_INVALID_ARGUMENT;
    *required = 0;
    if (tensor->struct_size < sizeof(*tensor)) return RW_ABI_MISMATCH;
    const uint64_t bytes = item_bytes(tensor->dtype);
    if (!bytes || !valid_device(tensor->device_kind, tensor->device_ordinal) || !tensor->base ||
        tensor->ndim == 0 || tensor->ndim > RW_MAX_TENSOR_RANK ||
        tensor->access_mode < RW_ACCESS_READ || tensor->access_mode > RW_ACCESS_READ_WRITE)
        return RW_INVALID_ARGUMENT;
    if (tensor->byte_offset % bytes) return RW_INVALID_ARGUMENT;
    const auto address = reinterpret_cast<uintptr_t>(tensor->base);
    if (tensor->allocation_bytes > std::numeric_limits<uintptr_t>::max() - address)
        return RW_OVERFLOW;
    if (tensor->byte_offset > tensor->allocation_bytes) return RW_OUT_OF_BOUNDS;
    if ((address + tensor->byte_offset) % bytes) return RW_INVALID_ARGUMENT;
    std::array<std::pair<uint64_t, uint64_t>, RW_MAX_TENSOR_RANK> axes{};
    for (uint32_t i = 0; i < tensor->ndim; ++i) {
        if (tensor->shape[i] <= 0 || tensor->strides_bytes[i] <= 0 ||
            static_cast<uint64_t>(tensor->strides_bytes[i]) % bytes) return RW_INVALID_ARGUMENT;
        axes[i] = {static_cast<uint64_t>(tensor->strides_bytes[i]), static_cast<uint64_t>(tensor->shape[i])};
    }
    std::sort(axes.begin(), axes.begin() + tensor->ndim);
    uint64_t span = bytes;
    for (uint32_t i = 0; i < tensor->ndim; ++i) {
        const auto [stride, dimension] = axes[i];
        if (dimension == 1) continue;
        if (stride < span) return RW_INVALID_ARGUMENT; // overlapping/broadcast views require another profile
        uint64_t extension;
        if (!multiply(dimension - 1, stride, extension) || !add(span, extension, span)) return RW_OVERFLOW;
    }
    if (!add(tensor->byte_offset, span, *required)) return RW_OVERFLOW;
    return *required <= tensor->allocation_bytes ? RW_OK : RW_OUT_OF_BOUNDS;
}

rw_status_t rw_validate_frame(const rw_frame_desc* frame, const rw_frame_contract* contract) {
    if (!frame || !contract) return RW_INVALID_ARGUMENT;
    if (frame->struct_size < sizeof(*frame) || contract->struct_size < sizeof(*contract) ||
        contract->abi_version != RW_ABI_VERSION) return RW_ABI_MISMATCH;
    if (frame->flags || frame->reserved || contract->reserved || contract->temporal > 1 ||
        contract->require_ready_event > 1 || !valid_device(contract->device_kind, contract->device_ordinal) ||
        frame->time_base_num <= 0 || frame->time_base_den <= 0 || !frame->resources ||
        frame->resource_count == 0 || frame->resource_count > 64) return RW_INVALID_ARGUMENT;
    constexpr uint64_t supported_mask = (UINT64_C(1) << (RW_RESOURCE_CAMERA + 1)) - 2;
    if (contract->required_semantics & ~supported_mask) return RW_UNSUPPORTED;
    if (contract->color_space < RW_COLOR_SRGB_CODE || contract->color_space > RW_COLOR_LINEAR_REC2020 ||
        contract->dynamic_range < RW_DYNAMIC_RANGE_SDR || contract->dynamic_range > RW_DYNAMIC_RANGE_HDR)
        return RW_INVALID_ARGUMENT;
    uint64_t present = 0, valid = 0;
    const rw_resource* color = nullptr;
    for (uint32_t i = 0; i < frame->resource_count; ++i) {
        const auto& resource = frame->resources[i];
        // Resources are an array without an independent element-stride field.
        // ABI v1 therefore rejects extended elements instead of misindexing them.
        if (resource.struct_size != sizeof(resource)) return RW_ABI_MISMATCH;
        const auto semantic = resource.resource_semantic;
        if (semantic < RW_RESOURCE_COLOR || semantic > RW_RESOURCE_CAMERA || resource.validity > RW_VALIDITY_INVALID)
            return RW_INVALID_ARGUMENT;
        const auto bit = RW_SEMANTIC_BIT(semantic);
        if (present & bit) return RW_INVALID_ARGUMENT;
        present |= bit;
        if (resource.validity != RW_VALIDITY_VALID) continue;
        if (resource.provenance < RW_PROVENANCE_CAPTURED || resource.provenance > RW_PROVENANCE_USER)
            return RW_INVALID_ARGUMENT;
        uint64_t bytes;
        const auto status = rw_validate_tensor(&resource.tensor, &bytes);
        if (status != RW_OK) return status;
        if (resource.tensor.device_kind != contract->device_kind ||
            resource.tensor.device_ordinal != contract->device_ordinal) return RW_INVALID_ARGUMENT;
        if ((resource.tensor.access_mode & RW_ACCESS_READ) == 0) return RW_INVALID_ARGUMENT;
        if (contract->require_ready_event && !resource.tensor.ready_event) return RW_INVALID_ARGUMENT;
        if (resource.metadata.struct_size < sizeof(resource.metadata)) return RW_ABI_MISMATCH;
        if (resource.metadata.reserved) return RW_INVALID_ARGUMENT;
        if (semantic == RW_RESOURCE_COLOR || semantic == RW_RESOURCE_HISTORY) {
            if (resource.tensor.ndim != 3 ||
                (resource.tensor.shape[2] != 3 && resource.tensor.shape[2] != 4) ||
                resource.metadata.color_space != contract->color_space ||
                resource.metadata.dynamic_range != contract->dynamic_range) return RW_INVALID_ARGUMENT;
            if (semantic == RW_RESOURCE_COLOR) color = &resource;
        }
        if (semantic == RW_RESOURCE_DEPTH) {
            if (resource.tensor.ndim != 2 ||
                (resource.tensor.dtype != RW_DTYPE_F16 && resource.tensor.dtype != RW_DTYPE_F32) ||
                resource.metadata.depth_definition < RW_DEPTH_VIEW_Z ||
                resource.metadata.depth_definition > RW_DEPTH_EUCLIDEAN) return RW_INVALID_ARGUMENT;
        }
        if (semantic == RW_RESOURCE_VALIDITY &&
            (resource.tensor.ndim != 2 || resource.tensor.dtype != RW_DTYPE_U8)) return RW_INVALID_ARGUMENT;
        if (semantic == RW_RESOURCE_MOTION) {
            if (contract->motion_direction < RW_MOTION_CURRENT_TO_PREVIOUS ||
                contract->motion_direction > RW_MOTION_PREVIOUS_TO_CURRENT ||
                contract->motion_units < RW_MOTION_PIXELS || contract->motion_units > RW_MOTION_NDC ||
                contract->jitter_included < RW_JITTER_INCLUDED || contract->jitter_included > RW_JITTER_EXCLUDED ||
                resource.tensor.ndim != 3 || resource.tensor.shape[2] != 2 ||
                resource.metadata.motion_direction != contract->motion_direction ||
                resource.metadata.motion_units != contract->motion_units ||
                resource.metadata.jitter_included != contract->jitter_included) return RW_INVALID_ARGUMENT;
        }
        valid |= bit;
    }
    if (!color || (valid & contract->required_semantics) != contract->required_semantics)
        return RW_INVALID_ARGUMENT;
    if (contract->temporal && !(contract->required_semantics & RW_SEMANTIC_BIT(RW_RESOURCE_MOTION)))
        return RW_INVALID_ARGUMENT;
    for (uint32_t i = 0; i < frame->resource_count; ++i) {
        const auto& resource = frame->resources[i];
        if (resource.validity != RW_VALIDITY_VALID) continue;
        if (resource.resource_semantic >= RW_RESOURCE_DEPTH && resource.resource_semantic <= RW_RESOURCE_VALIDITY) {
            if (resource.tensor.ndim < 2 || resource.tensor.shape[0] != color->tensor.shape[0] ||
                resource.tensor.shape[1] != color->tensor.shape[1]) return RW_INVALID_ARGUMENT;
        }
    }
    return RW_OK;
}

rw_status_t rw_memory_plan_required(const rw_memory_plan* plan, uint64_t* required) {
    if (!plan || !required) return RW_INVALID_ARGUMENT;
    *required = 0;
    if (plan->struct_size < sizeof(*plan) || plan->abi_version != RW_ABI_VERSION) return RW_ABI_MISMATCH;
    uint64_t slots, histories;
    if (!multiply(plan->bytes_per_slot, plan->slot_count, slots) ||
        !multiply(plan->bytes_per_session, plan->session_count, histories)) return RW_OVERFLOW;
    for (const auto component : {plan->shared_weights_bytes, slots, histories, plan->import_output_bytes,
                                plan->graph_handle_bytes, plan->reserve_quarantine_bytes})
        if (!add(*required, component, *required)) return RW_OVERFLOW;
    return *required <= plan->budget_bytes ? RW_OK : RW_RESOURCE_EXHAUSTED;
}
}
