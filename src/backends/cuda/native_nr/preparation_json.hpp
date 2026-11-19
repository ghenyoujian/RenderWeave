#pragma once
#include "native_nr.hpp"
#include <nlohmann/json.hpp>

namespace rw::native_nr {
inline nlohmann::json preparation_json(const PreparationStatistics& s) {
    return {{"enabled",s.enabled},{"lookups",s.lookups},{"hits",s.hits},
        {"misses",s.misses},{"bypasses",s.bypasses},
        {"completed_preparations",s.completed_preparations},{"publications",s.publications},
        {"duplicate_publications",s.duplicate_publications},{"rejections",s.rejections},
        {"entries",s.entries},{"entry_limit",s.entry_limit},
        {"host_payload_bytes",s.host_payload_bytes},
        {"host_container_capacity_bytes",s.host_container_capacity_bytes},
        {"host_allocator_allowance_bytes",s.host_allocator_allowance_bytes},
        {"host_accounted_bytes",s.host_accounted_bytes},
        {"host_accounted_peak_bytes",s.host_accounted_peak_bytes},
        {"host_byte_limit",s.host_byte_limit},
        {"additional_device_allocation_bytes",s.additional_device_allocation_bytes},
        {"accounting_scope","bounded metadata plus allocator allowance; RSS is measured separately"},
        {"sharing_scope","one immutable Weights instance and its registered Context; no multislot claim"}};
}
}
