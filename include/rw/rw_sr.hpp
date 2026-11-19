#ifndef RW_SR_HPP
#define RW_SR_HPP

#include "rw_native_sr.h"
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace rw::sr {
class Error : public std::runtime_error {
public:
    explicit Error(rw_status_t status)
        : std::runtime_error(*rw_sr_last_error() ? rw_sr_last_error() :
                            "RenderWeave SR status " + std::to_string(status)), status(status) {}
    const rw_status_t status;
};

namespace detail {
inline void check(rw_status_t status) {
    if (status != RW_OK) throw Error(status);
}
template<class Handle, auto Release>
using Owner = std::unique_ptr<std::remove_pointer_t<Handle>, std::integral_constant<decltype(Release), Release>>;
}

class Job {
    friend class Session;
    detail::Owner<rw_sr_job_t, rw_sr_job_release> handle_;
    explicit Job(rw_sr_job_t handle) : handle_(handle) {}
public:
    bool wait(std::span<float> rgba, int64_t timeout_ms = -1) {
        const auto status = rw_sr_job_wait(handle_.get(), timeout_ms, rgba.data(), rgba.size());
        if (status == RW_NOT_READY) return false;
        detail::check(status);
        return true;
    }
    [[nodiscard]] std::optional<rw_sr_cuda_view> wait_cuda(int64_t timeout_ms = -1) {
        rw_sr_cuda_view view{};
        view.struct_size = sizeof(view);
        view.abi_version = RW_ABI_VERSION;
        const auto status = rw_sr_job_wait_cuda(handle_.get(), timeout_ms, &view);
        if (status == RW_NOT_READY) return std::nullopt;
        detail::check(status);
        return view;
    }
    rw_sr_job_info info() const {
        rw_sr_job_info info{};
        info.struct_size = sizeof(info);
        info.abi_version = RW_ABI_VERSION;
        detail::check(rw_sr_job_get_info(handle_.get(), &info));
        return info;
    }
    void cancel() { detail::check(rw_sr_job_cancel(handle_.get())); }
};

class Session {
    friend class Runtime;
    detail::Owner<rw_sr_session_t, rw_sr_session_release> handle_;
    explicit Session(rw_sr_session_t handle) : handle_(handle) {}
public:
    void reset(uint64_t epoch) { detail::check(rw_sr_session_reset(handle_.get(), epoch)); }
    Job submit(const rw_sr_frame& frame) {
        rw_sr_job_t job = nullptr;
        detail::check(rw_sr_submit(handle_.get(), &frame, &job));
        return Job(job);
    }
    Job submit_cuda(const rw_sr_frame& frame, void* ready_event = nullptr) {
        const rw_sr_cuda_input input{sizeof(rw_sr_cuda_input), RW_ABI_VERSION, ready_event};
        rw_sr_job_t job = nullptr;
        detail::check(rw_sr_submit_cuda(handle_.get(), &frame, &input, &job));
        return Job(job);
    }
};

class Runtime {
    detail::Owner<rw_sr_runtime_t, rw_sr_runtime_release> handle_;
public:
    explicit Runtime(const rw_sr_runtime_desc& desc) {
        rw_sr_runtime_t runtime = nullptr;
        detail::check(rw_sr_runtime_create(&desc, &runtime));
        handle_.reset(runtime);
    }
    Runtime(const rw_sr_runtime_desc& desc, std::span<const std::byte> weights, std::span<const std::byte> filter) {
        rw_sr_runtime_t runtime = nullptr;
        detail::check(rw_sr_runtime_create_from_memory(&desc, weights.data(), weights.size(),
                                                       filter.data(), filter.size(), &runtime));
        handle_.reset(runtime);
    }
    Session session(uint64_t epoch) {
        rw_sr_session_t session = nullptr;
        detail::check(rw_sr_session_create(handle_.get(), epoch, &session));
        return Session(session);
    }
};
}

#endif
