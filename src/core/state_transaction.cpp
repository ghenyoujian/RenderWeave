#include "rw/rw_api.h"
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>

struct rw_state_guard {
    struct Pending { uint64_t epoch, generation, predecessor, frame; bool cancelled; };
    std::mutex mutex;
    uint64_t epoch, generation = 1, predecessor, next_ticket = 1;
    std::unordered_map<uint64_t, Pending> pending;
    rw_state_guard(uint64_t e, uint64_t p) : epoch(e), predecessor(p) {}
};

extern "C" {
rw_status_t rw_state_guard_create(uint64_t epoch, uint64_t predecessor, rw_state_guard_t* output) {
    if (!output) return RW_INVALID_ARGUMENT;
    *output = new (std::nothrow) rw_state_guard(epoch, predecessor);
    return *output ? RW_OK : RW_RESOURCE_EXHAUSTED;
}
rw_status_t rw_state_guard_snapshot(rw_state_guard_t guard, rw_state_snapshot* snapshot) {
    if (!guard || !snapshot) return RW_INVALID_ARGUMENT;
    if (snapshot->struct_size < sizeof(*snapshot) || snapshot->abi_version != RW_ABI_VERSION)
        return RW_ABI_MISMATCH;
    std::lock_guard lock(guard->mutex);
    snapshot->epoch = guard->epoch; snapshot->generation = guard->generation;
    snapshot->predecessor = guard->predecessor; snapshot->pending_count = guard->pending.size();
    return RW_OK;
}
rw_status_t rw_state_guard_begin(rw_state_guard_t guard, uint64_t epoch,
    uint64_t expected_predecessor, uint64_t frame_id, uint64_t* ticket) {
    if (!guard || !ticket) return RW_INVALID_ARGUMENT;
    *ticket = 0;
    std::lock_guard lock(guard->mutex);
    if (epoch != guard->epoch || expected_predecessor != guard->predecessor) return RW_STALE;
    if (frame_id <= expected_predecessor) return RW_INVALID_ARGUMENT;
    if (guard->next_ticket == std::numeric_limits<uint64_t>::max()) return RW_RESOURCE_EXHAUSTED;
    // Guard metadata itself has an explicit bound, separate from the backend byte budget.
    if (guard->pending.size() >= 1024) return RW_RESOURCE_EXHAUSTED;
    const uint64_t candidate = guard->next_ticket;
    try {
        guard->pending.emplace(candidate, rw_state_guard::Pending{
            epoch, guard->generation, expected_predecessor, frame_id, false});
    } catch (const std::bad_alloc&) { return RW_RESOURCE_EXHAUSTED; }
    ++guard->next_ticket;
    *ticket = candidate;
    return RW_OK;
}
rw_status_t rw_state_guard_cancel(rw_state_guard_t guard, uint64_t ticket) {
    if (!guard || ticket == 0) return RW_INVALID_ARGUMENT;
    std::lock_guard lock(guard->mutex);
    const auto found = guard->pending.find(ticket);
    if (found == guard->pending.end()) return RW_INVALID_ARGUMENT;
    found->second.cancelled = true;
    return RW_OK;
}
rw_status_t rw_state_guard_reset(rw_state_guard_t guard, uint64_t new_epoch, uint64_t predecessor) {
    if (!guard) return RW_INVALID_ARGUMENT;
    std::lock_guard lock(guard->mutex);
    if (new_epoch <= guard->epoch) return RW_INVALID_ARGUMENT;
    if (guard->generation == std::numeric_limits<uint64_t>::max()) return RW_RESOURCE_EXHAUSTED;
    ++guard->generation; guard->epoch = new_epoch; guard->predecessor = predecessor;
    // Old tickets remain alive until their physical completion is confirmed.
    return RW_OK;
}
rw_status_t rw_state_guard_complete(rw_state_guard_t guard, uint64_t ticket,
    uint32_t physical_completion, uint32_t output_valid) {
    if (!guard || ticket == 0 || physical_completion > 1 || output_valid > 1) return RW_INVALID_ARGUMENT;
    std::lock_guard lock(guard->mutex);
    const auto found = guard->pending.find(ticket);
    if (found == guard->pending.end()) return RW_INVALID_ARGUMENT;
    if (!physical_completion) return RW_NOT_READY;
    const auto task = found->second;
    guard->pending.erase(found);
    if (task.cancelled) return RW_CANCELLED;
    if (task.epoch != guard->epoch || task.generation != guard->generation ||
        task.predecessor != guard->predecessor) return RW_STALE;
    if (!output_valid) return RW_OUTPUT_INVALID;
    guard->predecessor = task.frame;
    return RW_OK;
}
rw_status_t rw_state_guard_release(rw_state_guard_t guard) {
    if (!guard) return RW_OK;
    {
        std::lock_guard lock(guard->mutex);
        if (!guard->pending.empty()) return RW_NOT_READY;
    }
    // The owner must serialize destruction against new begin/snapshot calls.
    delete guard;
    return RW_OK;
}
}
