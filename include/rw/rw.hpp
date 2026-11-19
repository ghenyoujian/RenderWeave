#ifndef RW_HPP
#define RW_HPP
#include "rw_api.h"
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rw {
inline void check(rw_status_t status) {
    if (status != RW_OK) throw std::runtime_error(rw_status_string(status));
}

// The initial runtime owns CPU contract/reference tools. No NR execution is
// advertised until a registered native model backend is available.
class Runtime {
    rw_context_t context_ = nullptr;
public:
    explicit Runtime(uint64_t memory_budget_bytes = 0) {
        rw_context_desc descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.abi_version = RW_ABI_VERSION;
        descriptor.device_kind = RW_DEVICE_CPU;
        descriptor.device_ordinal = 0;
        descriptor.memory_budget_bytes = memory_budget_bytes;
        descriptor.max_sessions = descriptor.max_inflight = descriptor.execution_slots = 1;
        check(rw_context_create(&descriptor, &context_));
    }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&& other) noexcept : context_(std::exchange(other.context_, nullptr)) {}
    Runtime& operator=(Runtime&& other) noexcept {
        if (this != &other) {
            rw_context_release(context_);
            context_ = std::exchange(other.context_, nullptr);
        }
        return *this;
    }
    ~Runtime() noexcept { rw_context_release(context_); }
    std::string capabilities() const {
        size_t bytes = 0;
        check(rw_context_query(context_, "capabilities", nullptr, &bytes));
        std::vector<char> data(bytes);
        check(rw_context_query(context_, "capabilities", data.data(), &bytes));
        return std::string(data.data());
    }
    rw_context_t get() const noexcept { return context_; }
};
}
#endif
