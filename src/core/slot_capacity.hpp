#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace rw::detail {
inline constexpr uint32_t maximum_execution_slots = 8;
inline constexpr bool bounded_slot_capacity(uint32_t count) noexcept {
    return count != 0 && count <= maximum_execution_slots;
}
inline constexpr bool slot_binding_has_capacity(size_t occupied,uint32_t limit,bool duplicate) noexcept {
    return bounded_slot_capacity(limit) && occupied < limit && !duplicate;
}
inline constexpr bool request_has_capacity(uint64_t inflight,uint32_t maximum,bool session_pending) noexcept {
    return maximum != 0 && !session_pending && inflight < maximum;
}
inline uint64_t checked_slot_product(uint64_t per_slot,uint32_t slots) {
    if(!bounded_slot_capacity(slots))throw std::invalid_argument("execution slot count outside bounded capacity");
    if(per_slot > std::numeric_limits<uint64_t>::max()/slots)
        throw std::overflow_error("execution slot resource product overflow");
    return per_slot*slots;
}
} // namespace rw::detail
