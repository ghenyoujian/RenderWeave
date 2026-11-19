#pragma once
#include "rw/rw_native_nr.h"
#include "node_metrics.hpp"
#include "../../core/slot_capacity.hpp"
#include <cstdint>
#include <stdexcept>

namespace rw::nr_sdk {
// Pure CPU policy shared by admission and its boundary checks. Observation
// selection never chooses an execution slot count or a different model.
struct MetricsPolicy {
    bool valid = false;
    uint32_t execution_slots = 0;
    bool collect_node_metrics = false;
};
inline constexpr MetricsPolicy metrics_policy(uint32_t flags) noexcept {
    constexpr uint32_t allowed = RW_NR_EXPERIMENTAL | RW_NR_TWO_SLOTS_EXPERIMENTAL | RW_NR_NODE_DETAIL | RW_NR_EIGHT_SLOTS_EXPERIMENTAL;
    if (!(flags & RW_NR_EXPERIMENTAL) || (flags & ~allowed)) return {};
    if ((flags & RW_NR_TWO_SLOTS_EXPERIMENTAL) && (flags & RW_NR_EIGHT_SLOTS_EXPERIMENTAL)) return {};
    return {true, (flags & RW_NR_EIGHT_SLOTS_EXPERIMENTAL) ? 8u : (flags & RW_NR_TWO_SLOTS_EXPERIMENTAL) ? 2u : 1u,
            bool(flags & RW_NR_NODE_DETAIL)};
}
inline uint64_t node_metrics_host_reservation_bytes(uint32_t slots) {
    if (slots != 1 && slots != 2 && slots != 8) throw std::invalid_argument("node metric reservation requires one, two or eight execution slots");
    // Graph and published slot snapshot each own one fixed POD array. Both
    // modes reserve this same conservative envelope before any GPU admission.
    return rw::detail::checked_slot_product(2ull * native_nr::max_node_measurements * sizeof(native_nr::NodeMeasurement),slots);
}
} // namespace rw::nr_sdk
