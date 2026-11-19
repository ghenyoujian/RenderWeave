#pragma once
#include <cstdint>
#include <type_traits>

namespace rw::native_nr {
inline constexpr uint32_t max_node_measurements = 152;
// Observation data has no ownership of tensors, CUDA handles or strings.
// Node kind comes from the immutable plan at the recorded index.
enum class NodeMeasurementState : uint32_t {
    dispatch_started = 1, physically_complete = 2, failed = 3
};
inline constexpr const char* node_measurement_state_name(NodeMeasurementState state) noexcept {
    switch (state) {
    case NodeMeasurementState::dispatch_started: return "DISPATCH_STARTED";
    case NodeMeasurementState::physically_complete: return "PHYSICALLY_COMPLETE";
    case NodeMeasurementState::failed: return "FAILED";
    }
    return "INVALID_OBSERVATION_STATE";
}
struct NodeMeasurement {
    int32_t index = -1;
    NodeMeasurementState status = NodeMeasurementState::dispatch_started;
    uint64_t dispatch_start_ns = 0, dispatch_end_ns = 0;
    float device_stream_span_ms = 0;
    bool device_span_collected = false;
    uint64_t live_owned_bytes = 0, peak_owned_bytes = 0;
};
struct GraphExecutionSummary {
    uint64_t execution_serial = 0;
    uint32_t started_nodes = 0, physically_completed_nodes = 0;
    int32_t active_node = -1, failed_node = -1;
    bool detail_requested = false, graph_entered = false, completed = false;
};
static_assert(std::is_trivially_copyable_v<NodeMeasurement>);
static_assert(std::is_trivially_copyable_v<GraphExecutionSummary>);
} // namespace rw::native_nr
