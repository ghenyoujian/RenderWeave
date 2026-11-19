#include "native_nr.hpp"
#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace rw::native_nr {
namespace {
constexpr size_t default_workspace_bytes = 4 * 1024 * 1024;
constexpr size_t minimum_workspace_bytes = 4 * 1024 * 1024;
constexpr size_t maximum_workspace_bytes = 256 * 1024 * 1024;
constexpr size_t max_host_staging_entries = 1024;

void create_execution_stream(cudaStream_t* stream) {
    int priority=0;
    if(const char* value=std::getenv("RW_NR_STREAM_PRIORITY");value && std::strcmp(value,"high")==0) {
        int least=0,greatest=0;
        cuda_check(cudaDeviceGetStreamPriorityRange(&least,&greatest),"query NR execution stream priority range");
        priority=greatest;
    }
    cuda_check(cudaStreamCreateWithPriority(stream,cudaStreamNonBlocking,priority),"create NR execution stream with priority");
}

size_t shape_elements(const std::vector<int64_t>& shape) {
    if (shape.size() > 8) throw std::invalid_argument("native NR tensor rank exceeds 8");
    size_t count = 1;
    for (const auto extent : shape) {
        if (extent <= 0) throw std::invalid_argument("native NR tensor extents must be positive");
        if (static_cast<uint64_t>(extent) > std::numeric_limits<size_t>::max() / count)
            throw std::overflow_error("native NR tensor element product overflows");
        count *= static_cast<size_t>(extent);
    }
    if (count > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::overflow_error("native NR tensor byte product overflows");
    return count;
}

std::vector<int64_t> resolved_shape(std::vector<int64_t> shape, size_t expected) {
    size_t known = 1;
    size_t unknown = shape.size();
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] == -1) {
            if (unknown != shape.size()) throw std::invalid_argument("multiple inferred tensor axes");
            unknown = axis;
            continue;
        }
        if (shape[axis] <= 0 || static_cast<uint64_t>(shape[axis]) > std::numeric_limits<size_t>::max() / known)
            throw std::invalid_argument("invalid tensor reshape extent");
        known *= static_cast<size_t>(shape[axis]);
    }
    if (unknown != shape.size()) {
        if (expected % known || expected / known > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            throw std::invalid_argument("inferred tensor reshape is not integral");
        shape[unknown] = static_cast<int64_t>(expected / known);
    }
    if (shape_elements(shape) != expected) throw std::invalid_argument("tensor reshape changes element count");
    return shape;
}
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
}

struct DeviceState {
    struct HostStaging {
        HostStaging(size_t count, std::shared_ptr<float> destination)
            : values(destination ? 0 : count), external(std::move(destination)), elements(count) {}
        std::vector<float> values;
        std::shared_ptr<float> external;
        size_t elements;
        bool transfer_attempted = false;
        bool quarantined = false;
        float* data() noexcept { return external ? external.get() : values.data(); }
        size_t bytes() const noexcept { return elements * sizeof(float); }
    };
    int ordinal = 0;
    uintptr_t actual_context = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t node_timing_begin = nullptr, node_timing_end = nullptr;
    Context::NodeTimingInfo node_timing;
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t blas_lt = nullptr;
    void* workspace = nullptr;
    uint64_t main_allocated_bytes = 0;
    cublasHandle_t post_projection_blas = nullptr;
    void* post_projection_workspace = nullptr;
    bool post_preparation_attempted = false, post_prepared = false, post_initialization_failed = false;
    uint64_t post_admitted_bytes = 0, post_allocated_bytes = 0, post_successful_preparations = 0;
    uint64_t budget = 0, live = 0, peak = 0, quarantined = 0;
    uint64_t host_budget = 0, host_live = 0, host_peak = 0, host_quarantined = 0;
    std::shared_ptr<SharedDeviceOwner> shared_device;
    SharedDeviceIdentity shared_identity{};
    uint64_t shared_slot = 0;
    bool slot_control_registered = false;
    DeviceBudgetReservation main_workspace_charge, post_workspace_charge;
    std::unique_ptr<SlotParameterBinding> parameter_binding;
    WeightReadPin active_read;
    bool resources_retired = false, device_initialization_attempted = false;
    bool closed = false, retirement_attempted = false, retirement_succeeded = false, retirement_failed = false;
    bool stream_ordered_memory = false;
    bool fast_math = false;
    bool half_math = false;
    bool tensor_cache = false;
    bool deferred_tensor_cache = false;
    bool explicit_half_gemm = false;
    bool explicit_fp8_gemm = false;
    // A successful stream drain is reusable as the physical-completion proof
    // for all live payload allocations until another operation asks for the
    // stream. This avoids synchronizing once per Tensor destructor at frame
    // teardown.
    bool stream_drained = false;
    // Entry metadata has a fixed 1024-record bound in addition to the byte
    // budget. Registry insertion and all host allocation occur before DMA.
    std::vector<std::unique_ptr<HostStaging>> host_staging;
    std::unordered_map<size_t, std::vector<float*>> tensor_free_blocks;
    uint64_t tensor_cached_bytes = 0;
    std::unordered_map<size_t, std::vector<float*>> tensor_pending_blocks;
    uint64_t tensor_pending_bytes = 0;
    std::unordered_map<const float*, std::pair<void*, size_t>> half_parameters;
    std::vector<std::pair<void*, size_t>> half_activation_blocks;
    void* half_activation_current = nullptr;
    size_t half_activation_capacity = 0;
    uint64_t half_parameter_bytes = 0, half_activation_bytes = 0;
    std::unordered_map<const float*, std::pair<void*, size_t>> fp8_parameters;
    std::vector<std::pair<void*, size_t>> fp8_activation_blocks;
    void* fp8_activation_current = nullptr;
    size_t fp8_activation_capacity = 0;
    uint64_t fp8_parameter_bytes = 0, fp8_activation_bytes = 0;
    void* batched_pointer_workspace = nullptr;
    size_t batched_pointer_capacity = 0;
    uint64_t batched_pointer_bytes = 0;
    cudaError_t retirement_error = cudaSuccess;
    mutable std::mutex mutex;

    void quarantine_domain(cudaError_t status) noexcept {
        if (retirement_error == cudaSuccess)
            retirement_error = status == cudaSuccess ? cudaErrorUnknown : status;
        if (node_timing.preparation_attempted) {
            node_timing.failed = true;
            node_timing.prepared = false;
            if (node_timing.last_cuda_error == 0)
                node_timing.last_cuda_error = static_cast<int>(retirement_error);
        }
        main_workspace_charge.quarantine();
        post_workspace_charge.quarantine();
        if (shared_device) shared_device->quarantine();
    }

    bool retire_node_timing_after_drain() noexcept {
        // Only retire_resources, after its successful raw stream drain and
        // current-context check, may call this. On failure retain the exact
        // remaining handles in this State; never retry an uncertain destroy.
        if (node_timing.failed) return false;
        if (node_timing_begin) {
            const auto status = cudaEventDestroy(node_timing_begin);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            node_timing_begin = nullptr;
            ++node_timing.destroyed_events;
        }
        if (node_timing_end) {
            const auto status = cudaEventDestroy(node_timing_end);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            node_timing_end = nullptr;
            ++node_timing.destroyed_events;
        }
        node_timing.prepared = false;
        node_timing.active = false;
        node_timing.active_run = 0;
        node_timing.active_node = -1;
        return true;
    }

    void select_device() {
        { std::lock_guard lock(mutex); check_ready(); }
        const auto selected = cudaSetDevice(ordinal);
        if (selected != cudaSuccess) {
            std::lock_guard lock(mutex);
            quarantine_domain(selected);
            cuda_check(selected, "cudaSetDevice(native NR Context)");
        }
        if (shared_device) {
            try { shared_device->check_current_context(); }
            catch (...) {
                std::lock_guard lock(mutex);
                quarantine_domain(cudaErrorUnknown);
                throw;
            }
        } else if (actual_context) {
            CUcontext current = nullptr;
            if (cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
                reinterpret_cast<uintptr_t>(current) != actual_context) {
                std::lock_guard lock(mutex);
                quarantine_domain(cudaErrorUnknown);
                throw std::runtime_error("native NR legacy Context actual CUDA context changed");
            }
        }
    }

    bool current_context_matches() const noexcept {
        if (shared_device) {
            try { shared_device->check_current_context(); return true; }
            catch (...) { return false; }
        }
        if (!actual_context) return true; // No stream was created yet.
        CUcontext current = nullptr;
        return cuCtxGetCurrent(&current) == CUDA_SUCCESS &&
            reinterpret_cast<uintptr_t>(current) == actual_context;
    }

    bool rollback_known_workspace_oom(bool post_only) noexcept {
        // Caller has observed exactly MemoryAllocation + null target at the
        // startup cudaMalloc, before any transfer into that allocation. This
        // helper supplies the remaining positive cleanup proof; every failed
        // step leaves its real handles/reservation registered for quarantine.
        if ((post_only ? post_projection_workspace : workspace) || retirement_error != cudaSuccess ||
            (shared_device && shared_device->quarantined()) || cudaSetDevice(ordinal) != cudaSuccess ||
            !current_context_matches() || !stream || cudaStreamSynchronize(stream) != cudaSuccess)
            return false;
        auto& handle = post_only ? post_projection_blas : blas;
        if (!handle || cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) return false;
        handle = nullptr;
        if (post_only) {
            post_workspace_charge.no_allocation_confirmed();
            post_preparation_attempted = false;
            post_prepared = false;
            post_initialization_failed = false;
            post_admitted_bytes = post_allocated_bytes = 0;
        } else {
            if (cudaStreamDestroy(stream) != cudaSuccess) return false;
            stream = nullptr;
            if (slot_control_registered) {
                shared_device->retire_slot_control();
                slot_control_registered = false;
            }
            main_workspace_charge.no_allocation_confirmed();
            device_initialization_attempted = false;
        }
        return true;
    }

    bool retire_resources() noexcept {
        if (resources_retired) return true;
        if (retirement_failed || retirement_error != cudaSuccess) return false;
        if (!device_initialization_attempted && !stream && !node_timing_begin && !node_timing_end &&
            !blas && !blas_lt && !workspace && !post_projection_blas &&
            !post_projection_workspace && !parameter_binding && !active_read.read_id() && !slot_control_registered &&
            tensor_free_blocks.empty() && tensor_pending_blocks.empty() && half_parameters.empty() && half_activation_blocks.empty() &&
            fp8_parameters.empty() && fp8_activation_blocks.empty() &&
            !batched_pointer_workspace) {
            resources_retired = true;
            return true;
        }
        if (shared_device && shared_device->quarantined()) {
            quarantine_domain(cudaErrorUnknown);
            return false;
        }
        const auto selected = cudaSetDevice(ordinal);
        bool context_matches = selected == cudaSuccess;
        if (context_matches) context_matches = current_context_matches();
        // All Tensor owners have retired before DeviceState can be destroyed.
        // A failed completion never frees possibly device-owned memory.
        const bool complete = !post_initialization_failed && context_matches &&
            (!stream || cudaStreamSynchronize(stream) == cudaSuccess);
        if (!complete) {
            quarantine_domain(selected == cudaSuccess ? cudaErrorUnknown : selected);
            return false;
        }
        if (!retire_node_timing_after_drain()) return false;
        try {
            // Read completion and graph/binding retirement use different sealed
            // proofs. Neither a cancelled request nor destruction supplies one.
            if (active_read.read_id()) {
                active_read.physically_complete(PhysicalDrainProof(shared_identity,
                    reinterpret_cast<uintptr_t>(stream), shared_slot, active_read.read_id()));
                active_read = WeightReadPin{};
            }
            if (parameter_binding) {
                parameter_binding->retire_after_drain(PhysicalDrainProof(shared_identity,
                    reinterpret_cast<uintptr_t>(stream), shared_slot, 0));
                parameter_binding.reset();
            }
        } catch (...) {
            quarantine_domain(cudaErrorUnknown);
            return false;
        }
        // Handles may refer to their private workspace. Destroy both handles
        // before either buffer, after physical completion was confirmed.
        if (post_projection_blas) {
            if (cublasDestroy(post_projection_blas) != CUBLAS_STATUS_SUCCESS) {
                quarantine_domain(cudaErrorUnknown); return false;
            }
            post_projection_blas = nullptr;
            post_prepared = false;
        }
        if (blas) {
            if (cublasDestroy(blas) != CUBLAS_STATUS_SUCCESS) {
                quarantine_domain(cudaErrorUnknown); return false;
            }
            blas = nullptr;
        }
        if (blas_lt) {
            if (cublasLtDestroy(blas_lt) != CUBLAS_STATUS_SUCCESS) {
                quarantine_domain(cudaErrorUnknown); return false;
            }
            blas_lt = nullptr;
        }
        if (post_projection_workspace) {
            const auto status = cudaFree(post_projection_workspace);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            post_projection_workspace = nullptr;
            post_workspace_charge.released_after_physical_free();
            live -= post_allocated_bytes;
            post_allocated_bytes = 0;
            post_admitted_bytes = 0;
        }
        if (workspace) {
            const auto status = cudaFree(workspace);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            workspace = nullptr;
            main_workspace_charge.released_after_physical_free();
            live -= main_allocated_bytes;
            main_allocated_bytes = 0;
        }
        for (auto& [bytes, blocks] : tensor_free_blocks) {
            for (auto* pointer : blocks) {
                const auto status = cudaFree(pointer);
                if (status != cudaSuccess) { quarantine_domain(status); return false; }
                tensor_cached_bytes -= bytes;
            }
        }
        tensor_free_blocks.clear();
        for (auto& [bytes, blocks] : tensor_pending_blocks) {
            for (auto* pointer : blocks) {
                const auto status = cudaFree(pointer);
                if (status != cudaSuccess) { quarantine_domain(status); return false; }
                tensor_pending_bytes -= bytes;
            }
        }
        tensor_pending_blocks.clear();
        for (const auto& [source, record] : half_parameters) {
            const auto status = cudaFree(record.first);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            half_parameter_bytes -= record.second;
        }
        half_parameters.clear();
        for (const auto& [pointer, bytes] : half_activation_blocks) {
            const auto status = cudaFree(pointer);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            half_activation_bytes -= bytes;
        }
        half_activation_blocks.clear();
        half_activation_current = nullptr;
        half_activation_capacity = 0;
        for (const auto& [source, record] : fp8_parameters) {
            const auto status = cudaFree(record.first);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            fp8_parameter_bytes -= record.second;
        }
        fp8_parameters.clear();
        for (const auto& [pointer, bytes] : fp8_activation_blocks) {
            const auto status = cudaFree(pointer);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            fp8_activation_bytes -= bytes;
        }
        fp8_activation_blocks.clear();
        fp8_activation_current = nullptr;
        fp8_activation_capacity = 0;
        if (batched_pointer_workspace) {
            const auto status = cudaFree(batched_pointer_workspace);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            batched_pointer_workspace = nullptr;
            batched_pointer_capacity = 0;
            batched_pointer_bytes = 0;
        }
        if (stream) {
            const auto status = cudaStreamDestroy(stream);
            if (status != cudaSuccess) { quarantine_domain(status); return false; }
            stream = nullptr;
        }
        if (slot_control_registered) {
            shared_device->retire_slot_control();
            slot_control_registered = false;
        }
        resources_retired = true;
        return true;
    }

    ~DeviceState() noexcept {
        if (!retire_resources()) {
            // An unsuccessful sync proves neither DMA nor kernels have stopped.
            // Release these exact unique_ptr records without destroying their
            // vectors, preserving DMA addresses until process cleanup. This is
            // bounded by host_budget/max_host_staging_entries; no failure-path
            // allocation or process-wide quarantine registry is necessary.
            for (auto& staging : host_staging) (void)staging.release();
            // Handles and workspace may still be referenced by active work.
            // Retain them too; never reset/kill a device to force cleanup.
            // Context States use a custom deleter and never enter their
            // destructor on this branch: the EXACT whole State, owner, budget
            // reservations, binding/read pins and staging remain retained.
        }
    }

    void check_ready() const {
        if (closed) throw std::runtime_error("native NR Context is closed after explicit resource retirement");
        if (shared_device && shared_device->quarantined())
            throw std::runtime_error("native NR shared device ownership domain is quarantined");
        if (retirement_error != cudaSuccess)
            throw std::runtime_error("native NR context quarantined after unconfirmed completion: " +
                                     std::string(cudaGetErrorString(retirement_error)));
    }

    HostStaging* make_staging(size_t count, std::shared_ptr<float> destination = {}) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(float))
            throw std::overflow_error("native NR host staging byte product overflows");
        const uint64_t bytes = count * sizeof(float);
        std::lock_guard lock(mutex);
        check_ready();
        if (host_staging.size() >= max_host_staging_entries || bytes > host_budget ||
            host_live > host_budget - bytes) {
            throw DeviceResourceExhausted("native NR bounded host staging budget exhausted before DMA");
        }
        auto entry = std::make_unique<HostStaging>(count, std::move(destination));
        auto* pointer = entry.get();
        host_staging.push_back(std::move(entry));
        host_live += bytes;
        host_peak = std::max(host_peak, host_live);
        return pointer;
    }

    void begin_transfer(HostStaging* staging) {
        std::lock_guard lock(mutex);
        check_ready();
        // Set BEFORE cudaMemcpyAsync: that API can return an earlier async
        // error, so a failing return is not proof that its addresses are unused.
        staging->transfer_attempted = true;
    }

    void quarantine_transfer(HostStaging* staging, cudaError_t status) noexcept {
        std::lock_guard lock(mutex);
        if (!staging->quarantined) {
            staging->quarantined = true;
            host_quarantined += staging->bytes();
        }
        quarantine_domain(status);
    }

    void abandon_unsubmitted_staging(HostStaging* staging) noexcept {
        if (!staging) return;
        std::lock_guard lock(mutex);
        for (auto found = host_staging.begin(); found != host_staging.end(); ++found) {
            if (found->get() == staging) {
                if (staging->transfer_attempted) return;
                // This allocation has never been handed to a DMA API.
                host_live -= staging->bytes();
                host_staging.erase(found);
                return;
            }
        }
    }

    std::vector<float> finish_staging(HostStaging* staging, bool return_values) {
        // Only callers that have just confirmed cudaStreamSynchronize success
        // may retire a submitted record or transfer ownership to a result.
        std::lock_guard lock(mutex);
        for (auto found = host_staging.begin(); found != host_staging.end(); ++found) {
            if (found->get() != staging) continue;
            const uint64_t bytes = staging->bytes();
            std::vector<float> result;
            if (return_values) result = std::move(staging->values);
            host_live -= bytes;
            if (staging->quarantined) host_quarantined -= bytes;
            host_staging.erase(found);
            return result;
        }
        throw std::logic_error("native NR host staging record disappeared before completion");
    }

    bool retire(float* pointer, size_t bytes, bool live_accounted, DeviceBudgetReservation& charge) noexcept {
        {
            std::lock_guard lock(mutex);
            if (retirement_error != cudaSuccess || retirement_failed ||
                (shared_device && shared_device->quarantined())) {
                quarantined += bytes;
                charge.quarantine();
                quarantine_domain(cudaErrorUnknown);
                return false;
            }
        }
        const auto selected = cudaSetDevice(ordinal);
        bool context_matches = selected == cudaSuccess;
        if (context_matches) context_matches = current_context_matches();
        cudaError_t status = cudaSuccess;
        if (!context_matches) status = selected == cudaSuccess ? cudaErrorUnknown : selected;
        else if (!stream_ordered_memory && !deferred_tensor_cache) {
            bool drained = false;
            {
                std::lock_guard lock(mutex);
                drained = stream_drained;
            }
            if (!drained) status = cudaStreamSynchronize(stream);
        }
        if (status == cudaSuccess) {
            if (stream_ordered_memory) {
                // The free is ordered after prior work on this context stream;
                // Context retirement still performs one final physical drain.
                status = cudaFreeAsync(pointer, stream);
            } else if (tensor_cache) {
                // The stream was physically drained above, so this exact
                // allocation can be reused by a same-size activation next frame.
                try {
                    auto& blocks = deferred_tensor_cache ? tensor_pending_blocks[bytes] : tensor_free_blocks[bytes];
                    blocks.push_back(pointer);
                    if (deferred_tensor_cache) tensor_pending_bytes += bytes;
                    else tensor_cached_bytes += bytes;
                } catch (...) {
                    // In deferred mode the stream may still reference this
                    // pointer; retain the exact owner as quarantined rather
                    // than issuing an unsafe cudaFree.
                    if (deferred_tensor_cache) {
                        quarantined += bytes;
                        charge.quarantine();
                        quarantine_domain(cudaErrorMemoryAllocation);
                        return false;
                    }
                    status = cudaFree(pointer);
                }
            } else {
                status = cudaFree(pointer);
            }
        }
        std::lock_guard lock(mutex);
        if (status == cudaSuccess) {
            if (live_accounted) live -= bytes;
            if (stream_ordered_memory) stream_drained = false;
            charge.released_after_physical_free();
            return true;
        }
        else {
            // This allocation is never reused by this context. Its outstanding
            // bytes remain charged against admission until process teardown.
            quarantined += bytes;
            charge.quarantine();
            quarantine_domain(status);
            return false;
        }
    }
};

struct DeviceStorage {
    std::shared_ptr<DeviceState> owner;
    float* pointer = nullptr;
    size_t count = 0;
    bool live_accounted = false;
    DeviceBudgetReservation charge;
    ~DeviceStorage() noexcept { if (pointer) owner->retire(pointer, count * sizeof(float), live_accounted, charge); }
};

namespace {
std::shared_ptr<DeviceState> make_context_state() {
    // The control block is allocated before CUDA resources exist. On an
    // unconfirmed drain retain the exact whole object, including its primary
    // context owner and charged reservations, without a global registry.
    return std::shared_ptr<DeviceState>(new DeviceState, [](DeviceState* state) noexcept {
        if (state->retire_resources()) delete state;
    });
}
bool stream_ordered_memory_requested() noexcept {
    const char* value = std::getenv("RW_NR_STREAM_ORDERED_MEMORY");
    return value && std::strcmp(value, "1") == 0;
}
bool fast_math_requested() noexcept {
    const char* value = std::getenv("RW_NR_GPU_FAST_MATH");
    return value && std::strcmp(value, "1") == 0;
}
bool half_math_requested() noexcept {
    const char* value = std::getenv("RW_NR_GPU_HALF_MATH");
    return value && std::strcmp(value, "1") == 0;
}
bool tensor_cache_requested() noexcept {
    const char* value = std::getenv("RW_NR_TENSOR_CACHE");
    return value && std::strcmp(value, "1") == 0;
}
bool deferred_tensor_cache_requested() noexcept {
    const char* value = std::getenv("RW_NR_DEFERRED_TENSOR_CACHE");
    return value && std::strcmp(value, "1") == 0;
}
bool explicit_half_gemm_requested() noexcept {
    const char* value = std::getenv("RW_NR_EXPLICIT_HALF_GEMM");
    return value && std::strcmp(value, "1") == 0;
}
bool explicit_fp8_gemm_requested() noexcept {
    const char* value = std::getenv("RW_NR_EXPLICIT_FP8_GEMM");
    return value && std::strcmp(value, "1") == 0;
}
bool explicit_fp8_gemm_supported(const cudaDeviceProp& properties) noexcept {
    return properties.major*10+properties.minor>=89;
}
size_t cublas_workspace_bytes_requested() noexcept {
    const char* value = std::getenv("RW_NR_CUBLAS_WORKSPACE_MB");
    if (!value || !*value) return default_workspace_bytes;
    char* end = nullptr;
    const unsigned long long mb = std::strtoull(value, &end, 10);
    if (!end || *end || mb == 0) return default_workspace_bytes;
    const unsigned long long bytes = mb * 1024ull * 1024ull;
    if (bytes < minimum_workspace_bytes || bytes > maximum_workspace_bytes) return default_workspace_bytes;
    return static_cast<size_t>(bytes);
}
std::shared_ptr<DeviceStorage> make_device_storage() {
    return std::shared_ptr<DeviceStorage>(new DeviceStorage, [](DeviceStorage* storage) noexcept {
        if (storage->pointer) {
            if (!storage->owner->retire(storage->pointer, storage->count * sizeof(float), storage->live_accounted, storage->charge))
                return; // Exact uncertain allocation + reservation + slot owner.
            storage->pointer = nullptr;
        }
        delete storage;
    });
}
}

float* Tensor::data() const {
    if (!storage) throw std::invalid_argument("empty native NR tensor has no data");
    return storage->pointer;
}
size_t Tensor::numel() const {
    if (!storage) return 0;
    const auto count = shape_elements(shape);
    if (count != storage->count) throw std::invalid_argument("native NR tensor shape does not match allocation");
    return count;
}
Tensor Tensor::reshape(std::vector<int64_t> dimensions) const {
    if (!storage) throw std::invalid_argument("cannot reshape an empty native NR tensor");
    return Tensor{storage, resolved_shape(std::move(dimensions), numel())};
}

Context::Context(uint64_t memory_budget_bytes, int device_ordinal, uint64_t host_staging_budget_bytes)
    : state_(make_context_state()) {
    const size_t workspace_bytes = cublas_workspace_bytes_requested();
    if (memory_budget_bytes < workspace_bytes) throw DeviceResourceExhausted("native NR budget is smaller than cuBLAS workspace");
    int visible = 0;
    cuda_check(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (device_ordinal < 0 || device_ordinal >= visible) throw std::invalid_argument("native NR device ordinal is not visible");
    state_->ordinal = device_ordinal;
    state_->stream_ordered_memory = stream_ordered_memory_requested();
    state_->half_math = half_math_requested();
    state_->fast_math = fast_math_requested() || state_->half_math;
    state_->tensor_cache = tensor_cache_requested();
    state_->deferred_tensor_cache = state_->tensor_cache && deferred_tensor_cache_requested();
    state_->explicit_half_gemm = explicit_half_gemm_requested();
    state_->budget = memory_budget_bytes;
    state_->host_budget = host_staging_budget_bytes;
    state_->host_staging.reserve(max_host_staging_entries);
    cuda_check(cudaSetDevice(device_ordinal), "cudaSetDevice");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device_ordinal), "cudaGetDeviceProperties");
    state_->explicit_fp8_gemm = explicit_fp8_gemm_requested() && explicit_fp8_gemm_supported(properties);
    if (memory_budget_bytes > properties.totalGlobalMem)
        throw std::invalid_argument("native NR requested budget exceeds physical device memory");
    try {
        state_->device_initialization_attempted = true;
        create_execution_stream(&state_->stream);
        CUcontext current = nullptr;
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || !current)
            throw std::runtime_error("legacy native NR stream has no confirmed actual CUDA context");
        state_->actual_context = reinterpret_cast<uintptr_t>(current);
        blas_check(cublasCreate(&state_->blas), "cublasCreate");
        if (state_->explicit_fp8_gemm)
            blas_check(cublasLtCreate(&state_->blas_lt), "cublasLtCreate(native NR explicit FP8)");
        blas_check(cublasSetStream(state_->blas, state_->stream), "cublasSetStream");
        blas_check(cublasSetPointerMode(state_->blas, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        blas_check(cublasSetMathMode(state_->blas,
            state_->fast_math ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH),
            state_->fast_math ? "cublasSetMathMode(TF32)" : "cublasSetMathMode(PEDANTIC)");
        blas_check(cublasSetAtomicsMode(state_->blas,
            state_->fast_math ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED),
            state_->fast_math ? "cublasSetAtomicsMode(allowed)" : "cublasSetAtomicsMode(disabled)");
        const auto status = cudaMalloc(&state_->workspace, workspace_bytes);
        if (status == cudaErrorMemoryAllocation && !state_->workspace && state_->rollback_known_workspace_oom(false))
            throw DeviceResourceExhausted("known pre-transfer CUDA main workspace OOM; exact profile may retry");
        cuda_check(status, "cudaMalloc(cuBLAS workspace)");
        state_->main_allocated_bytes = workspace_bytes;
        state_->live = state_->peak = workspace_bytes;
        blas_check(cublasSetWorkspace(state_->blas, state_->workspace, workspace_bytes), "cublasSetWorkspace");
    } catch (const DeviceResourceExhausted&) { throw; }
    catch (...) {
        state_->retirement_failed = true;
        state_->quarantine_domain(cudaErrorUnknown);
        throw;
    }
}

Context::Context(std::shared_ptr<SharedDeviceOwner> owner, uint64_t slot_id,
                 uint64_t slot_memory_budget_bytes, uint64_t host_staging_budget_bytes)
    : state_(make_context_state()) {
    if (!owner) throw std::invalid_argument("shared native NR Context requires a DeviceOwner");
    if (!SharedDeviceOwner::is_canonical_owner(owner))
        throw std::invalid_argument("shared native NR Context requires the canonical DeviceOwner control block");
    const size_t workspace_bytes = cublas_workspace_bytes_requested();
    if (slot_memory_budget_bytes < workspace_bytes)
        throw DeviceResourceExhausted("shared native NR slot budget is smaller than cuBLAS workspace");
    if (slot_memory_budget_bytes > owner->budget()->statistics().limit_bytes)
        throw std::invalid_argument("shared native NR slot budget is outside the device budget");
    state_->shared_device = std::move(owner);
    state_->stream_ordered_memory = stream_ordered_memory_requested();
    state_->half_math = half_math_requested();
    state_->fast_math = fast_math_requested() || state_->half_math;
    state_->tensor_cache = tensor_cache_requested();
    state_->deferred_tensor_cache = state_->tensor_cache && deferred_tensor_cache_requested();
    state_->explicit_half_gemm = explicit_half_gemm_requested();
    state_->ordinal = state_->shared_device->device_ordinal();
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, state_->ordinal), "query shared CUDA device");
    state_->explicit_fp8_gemm = explicit_fp8_gemm_requested() && explicit_fp8_gemm_supported(properties);
    state_->shared_identity = state_->shared_device->identity();
    state_->actual_context = state_->shared_identity.actual_cuda_context;
    state_->shared_slot = slot_id;
    state_->budget = slot_memory_budget_bytes;
    state_->host_budget = host_staging_budget_bytes;
    state_->host_staging.reserve(max_host_staging_entries);
    state_->main_workspace_charge = state_->shared_device->budget()->reserve(
        DeviceChargeKind::slot_workspace, slot_id, workspace_bytes);
    state_->shared_device->retain_slot_control();
    state_->slot_control_registered = true;
    try {
        state_->select_device();
        state_->device_initialization_attempted = true;
        create_execution_stream(&state_->stream);
        blas_check(cublasCreate(&state_->blas), "cublasCreate(shared slot)");
        if (state_->explicit_fp8_gemm)
            blas_check(cublasLtCreate(&state_->blas_lt), "cublasLtCreate(shared explicit FP8)");
        blas_check(cublasSetStream(state_->blas, state_->stream), "cublasSetStream(shared slot)");
        blas_check(cublasSetPointerMode(state_->blas, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode(shared slot)");
        blas_check(cublasSetMathMode(state_->blas,
            state_->fast_math ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH),
            state_->fast_math ? "cublasSetMathMode(shared slot TF32)" : "cublasSetMathMode(shared slot PEDANTIC)");
        blas_check(cublasSetAtomicsMode(state_->blas,
            state_->fast_math ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED),
            state_->fast_math ? "cublasSetAtomicsMode(shared slot allowed)" :
                                "cublasSetAtomicsMode(shared slot disabled)");
        state_->main_workspace_charge.allocation_attempted();
        const auto status = cudaMalloc(&state_->workspace, workspace_bytes);
        if (status == cudaErrorMemoryAllocation && !state_->workspace && state_->rollback_known_workspace_oom(false))
            throw DeviceResourceExhausted("known pre-transfer CUDA shared main workspace OOM; exact profile may retry");
        if (status != cudaSuccess) {
            state_->quarantine_domain(status);
            cuda_check(status, "cudaMalloc(shared slot cuBLAS workspace)");
        }
        state_->main_workspace_charge.allocation_succeeded();
        state_->main_allocated_bytes = workspace_bytes;
        state_->live = state_->peak = workspace_bytes;
        blas_check(cublasSetWorkspace(state_->blas, state_->workspace, workspace_bytes), "cublasSetWorkspace(shared slot)");
    } catch (const DeviceResourceExhausted&) { throw; }
    catch (...) {
        state_->retirement_failed = true;
        state_->quarantine_domain(cudaErrorUnknown);
        throw;
    }
}

void Context::bind_slot_parameters(std::unique_ptr<SlotParameterBinding> binding) {
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (!state_->shared_device || !binding || state_->parameter_binding || state_->active_read.read_id())
        throw std::invalid_argument("native NR shared parameter binding requires an unbound shared slot");
    if (binding->device_identity() != state_->shared_identity || binding->slot_id() != state_->shared_slot ||
        !binding->bound_to_stream(state_->stream))
        throw std::invalid_argument("native NR shared parameter binding device/slot/stream mismatch");
    state_->parameter_binding = std::move(binding);
}

void Context::begin_weight_read(WeightReadStamp stamp) {
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (!state_->shared_device || !state_->parameter_binding || state_->active_read.read_id())
        throw std::logic_error("native NR shared weight read requires a bound idle slot");
    state_->active_read = state_->parameter_binding->begin_read(stamp);
}

void Context::finish_weight_read() {
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (!state_->shared_device || !state_->parameter_binding || !state_->active_read.read_id())
        throw std::logic_error("native NR shared weight read has no active pin to finish");
    const auto status = cudaStreamSynchronize(state_->stream);
    if (status != cudaSuccess) {
        state_->quarantine_domain(status);
        cuda_check(status, "cudaStreamSynchronize(shared weight physical drain)");
    }
    try {
        state_->active_read.physically_complete(PhysicalDrainProof(state_->shared_identity,
            reinterpret_cast<uintptr_t>(state_->stream), state_->shared_slot, state_->active_read.read_id()));
        state_->active_read = WeightReadPin{};
    } catch (...) {
        state_->quarantine_domain(cudaErrorUnknown);
        throw;
    }
}

void Context::validate_parameter(const Parameter& parameter) const {
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (parameter.empty()) throw std::invalid_argument("native NR cannot read an empty parameter");
    if (const auto* legacy = parameter.legacy_tensor()) {
        if (!owns(*legacy)) throw std::invalid_argument("native NR legacy parameter belongs to another Context");
        (void)legacy->numel();
        return;
    }
    const auto* shared = parameter.shared_view();
    if (!shared || !state_->shared_device || !state_->parameter_binding || !state_->active_read.read_id())
        throw std::invalid_argument("native NR shared parameter requires the slot's active model read pin");
    state_->parameter_binding->validate_and_wait(*shared);
}

std::shared_ptr<SharedDeviceOwner> Context::shared_owner() const { return state_->shared_device; }
uint64_t Context::slot_id() const { return state_->shared_slot; }
bool Context::has_active_weight_read() const {
    std::lock_guard lock(state_->mutex);
    return state_->active_read.read_id() != 0;
}

bool Context::retire_owned_resources() noexcept {
    std::lock_guard lock(state_->mutex);
    state_->closed = true;
    state_->retirement_attempted = true;
    if (state_->resources_retired) {
        state_->retirement_succeeded = true;
        return true;
    }
    const uint64_t persistent_bytes = state_->main_allocated_bytes + state_->post_allocated_bytes;
    // A remaining payload owner or staging record means the caller has not
    // reached final retirement. Never destroy its stream/handle underneath it.
    if (state_->retirement_failed || state_->retirement_error != cudaSuccess ||
        state_->active_read.read_id() || state_->live != persistent_bytes ||
        state_->host_live || !state_->host_staging.empty()) {
        state_->retirement_failed = true;
        state_->quarantine_domain(cudaErrorUnknown);
        return false;
    }
    const bool success = state_->retire_resources();
    state_->retirement_succeeded = success;
    state_->retirement_failed = !success;
    if (!success) state_->quarantine_domain(cudaErrorUnknown);
    return success;
}

Context::RetirementInfo Context::retirement_info() const {
    std::lock_guard lock(state_->mutex);
    return {state_->closed, state_->retirement_attempted, state_->retirement_succeeded, state_->retirement_failed};
}

cudaStream_t Context::stream() const {
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    // Any caller requesting the execution stream may enqueue work after this
    // point. Invalidate the last drain proof before handing it out.
    state_->stream_drained = false;
    return state_->stream;
}
cublasHandle_t Context::blas() const {
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    return state_->blas;
}

void Context::prepare_post_projection_blas() {
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (state_->post_prepared) return;
    if (state_->post_preparation_attempted)
        throw std::runtime_error("post projection cuBLAS preparation already attempted and is not ready");
    // Admission precedes creation of either the handle or GPU buffer. The
    // mutex excludes other allocations throughout this fixed-size operation.
    if (post_projection_workspace_bytes > state_->budget ||
        state_->live > state_->budget - post_projection_workspace_bytes) {
        throw DeviceResourceExhausted("post projection workspace budget exhausted before allocation");
    }
    if (state_->shared_device)
        state_->post_workspace_charge = state_->shared_device->budget()->reserve(
            DeviceChargeKind::slot_workspace, state_->shared_slot, post_projection_workspace_bytes);
    state_->post_preparation_attempted = true;
    state_->post_admitted_bytes = post_projection_workspace_bytes;
    try {
        // This is startup preparation, not a hot-path fallback. Prior work must
        // be physically complete before configuring a new persistent handle.
        auto status = cudaStreamSynchronize(state_->stream);
        if (status != cudaSuccess) {
            state_->retirement_error = status;
            cuda_check(status, "cudaStreamSynchronize(before post projection preparation)");
        }
        // State fields own each successful/partially reported resource from the
        // creation call onward; no temporary Tensor can retire this workspace.
        blas_check(cublasCreate(&state_->post_projection_blas), "cublasCreate(post projection)");
        blas_check(cublasSetStream(state_->post_projection_blas, state_->stream), "cublasSetStream(post projection)");
        blas_check(cublasSetPointerMode(state_->post_projection_blas, CUBLAS_POINTER_MODE_HOST),
                   "cublasSetPointerMode(post projection)");
        blas_check(cublasSetMathMode(state_->post_projection_blas,
                       state_->fast_math ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH),
                   state_->fast_math ? "cublasSetMathMode(post projection TF32)" :
                                       "cublasSetMathMode(post projection PEDANTIC)");
        blas_check(cublasSetAtomicsMode(state_->post_projection_blas,
                       state_->fast_math ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED),
                   state_->fast_math ? "cublasSetAtomicsMode(post projection allowed)" :
                                       "cublasSetAtomicsMode(post projection disabled)");
        state_->post_workspace_charge.allocation_attempted();
        status = cudaMalloc(&state_->post_projection_workspace, post_projection_workspace_bytes);
        if (status == cudaErrorMemoryAllocation && !state_->post_projection_workspace &&
            state_->rollback_known_workspace_oom(true))
            throw DeviceResourceExhausted("known pre-transfer CUDA post workspace OOM; same Context/profile may retry");
        if (status != cudaSuccess) {
            state_->retirement_error = status;
            cuda_check(status, "cudaMalloc(post projection persistent workspace)");
        }
        state_->post_workspace_charge.allocation_succeeded();
        state_->post_allocated_bytes = post_projection_workspace_bytes;
        state_->live += post_projection_workspace_bytes;
        state_->peak = std::max(state_->peak, state_->live);
        blas_check(cublasSetWorkspace(state_->post_projection_blas, state_->post_projection_workspace,
                                     post_projection_workspace_bytes), "cublasSetWorkspace(post projection)");
        state_->post_prepared = true;
        ++state_->post_successful_preparations;
    } catch (const DeviceResourceExhausted&) { throw; }
    catch (...) {
        // A failed setup call can report prior asynchronous errors. Do not
        // destroy/recreate a partial handle, return its budget, or infer that
        // the device has stopped. This bounded State stays quarantined through
        // process cleanup even if a later stream-only query appears complete.
        state_->post_initialization_failed = true;
        state_->quarantined += state_->post_allocated_bytes;
        state_->quarantine_domain(cudaErrorUnknown);
        throw;
    }
}

cublasHandle_t Context::post_projection_blas() const {
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (!state_->post_prepared)
        throw std::runtime_error("post projection cuBLAS handle is not prepared");
    return state_->post_projection_blas;
}

Context::PostProjectionBlasInfo Context::post_projection_info() const {
    std::lock_guard lock(state_->mutex);
    return {state_->post_preparation_attempted, state_->post_prepared, state_->post_initialization_failed,
            state_->post_admitted_bytes, state_->post_allocated_bytes, state_->post_successful_preparations,
            state_->post_projection_workspace, state_->workspace};
}

void Context::prepare_node_timing() {
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    auto& timing = state_->node_timing;
    if (timing.active) throw std::logic_error("cannot prepare node timing during an active node");
    if (timing.prepared) return;
    if (timing.preparation_attempted)
        throw std::runtime_error("node timing preparation already attempted and is not ready");
    timing.preparation_attempted = true;
    // Existing uploads/work may already reference this stream. A setup failure
    // is never classified as a no-access case or cleaned by a stack destructor.
    auto status = cudaStreamSynchronize(state_->stream);
    if (status != cudaSuccess) {
        state_->quarantine_domain(status);
        cuda_check(status, "cudaStreamSynchronize(before node timing preparation)");
    }
    status = cudaEventCreate(&state_->node_timing_begin);
    if (state_->node_timing_begin) ++timing.created_events;
    if (status != cudaSuccess) {
        state_->quarantine_domain(status);
        cuda_check(status, "cudaEventCreate(node timing begin)");
    }
    if (!state_->node_timing_begin) {
        state_->quarantine_domain(cudaErrorUnknown);
        throw std::runtime_error("node timing begin creation returned success without an owned handle");
    }
    status = cudaEventCreate(&state_->node_timing_end);
    if (state_->node_timing_end) ++timing.created_events;
    if (status != cudaSuccess) {
        state_->quarantine_domain(status);
        cuda_check(status, "cudaEventCreate(node timing end)");
    }
    if (!state_->node_timing_end) {
        state_->quarantine_domain(cudaErrorUnknown);
        throw std::runtime_error("node timing end creation returned success without an owned handle");
    }
    timing.prepared = true;
}

void Context::begin_node_timing(uint64_t run_id, int node) {
    if (node < 0) throw std::invalid_argument("node timing index must be nonnegative");
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    auto& timing = state_->node_timing;
    if (!timing.prepared || timing.failed || !state_->node_timing_begin || !state_->node_timing_end)
        throw std::logic_error("node timing requires explicit successful preparation");
    if (timing.active) throw std::logic_error("node timing already has an active run/node");
    // Publish identity before record: a failing record may still report earlier
    // asynchronous work, so ownership/active identity cannot be cleared on error.
    timing.active = true;
    timing.active_run = run_id;
    timing.active_node = node;
    const auto status = cudaEventRecord(state_->node_timing_begin, state_->stream);
    if (status != cudaSuccess) {
        state_->quarantine_domain(status);
        cuda_check(status, "cudaEventRecord(node timing begin)");
    }
    ++timing.begin_records;
}

float Context::finish_node_timing(uint64_t run_id, int node) {
    if (node < 0) throw std::invalid_argument("node timing index must be nonnegative");
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    auto& timing = state_->node_timing;
    if (!timing.prepared || timing.failed || !timing.active ||
        timing.active_run != run_id || timing.active_node != node)
        throw std::logic_error("node timing finish requires the exact active run/node");
    auto status = cudaEventRecord(state_->node_timing_end, state_->stream);
    if (status != cudaSuccess) {
        state_->quarantine_domain(status);
        cuda_check(status, "cudaEventRecord(node timing end)");
    }
    ++timing.end_records;
    // This is the original per-node physical boundary, independent of timing
    // metadata. Do not call public stream()/synchronize() under the State lock.
    status = cudaStreamSynchronize(state_->stream);
    if (status != cudaSuccess) {
        state_->quarantine_domain(status);
        cuda_check(status, "cudaStreamSynchronize(node timing end)");
    }
    float elapsed = 0;
    status = cudaEventElapsedTime(&elapsed, state_->node_timing_begin, state_->node_timing_end);
    if (status != cudaSuccess) {
        state_->quarantine_domain(status);
        cuda_check(status, "cudaEventElapsedTime(node timing)");
    }
    ++timing.elapsed_reads;
    if (!std::isfinite(elapsed) || elapsed < 0) {
        state_->quarantine_domain(cudaErrorUnknown);
        throw std::runtime_error("node timing returned an invalid elapsed duration");
    }
    timing.active = false;
    timing.active_run = 0;
    timing.active_node = -1;
    return elapsed;
}

Context::NodeTimingInfo Context::node_timing_info() const {
    std::lock_guard lock(state_->mutex);
    return state_->node_timing;
}

const void* Context::identity() const { return state_.get(); }
bool Context::owns(const Tensor& tensor) const {
    return tensor.storage && tensor.storage->owner == state_;
}

Tensor Context::empty(std::vector<int64_t> shape) {
    const auto count = shape_elements(shape), bytes = count * sizeof(float);
    auto allocation = make_device_storage();
    allocation->owner = state_;
    allocation->count = count;
    state_->select_device();
    std::unique_lock lock(state_->mutex);
    state_->check_ready();
    float* cached = nullptr;
    if (state_->tensor_cache) {
        auto found = state_->tensor_free_blocks.find(bytes);
        if (found != state_->tensor_free_blocks.end() && !found->second.empty()) {
            cached = found->second.back();
            found->second.pop_back();
            state_->tensor_cached_bytes -= bytes;
            if (found->second.empty()) state_->tensor_free_blocks.erase(found);
        }
    }
    const uint64_t physical_after = state_->live + state_->tensor_cached_bytes + state_->tensor_pending_bytes + (cached ? 0 : bytes);
    if (bytes > state_->budget || physical_after > state_->budget) {
        if (cached) {
            state_->tensor_free_blocks[bytes].push_back(cached);
            state_->tensor_cached_bytes += bytes;
        }
        throw DeviceResourceExhausted("native NR tensor budget exhausted before allocation");
    }
    if (state_->shared_device)
        allocation->charge = state_->shared_device->budget()->reserve(
            DeviceChargeKind::slot_payload, state_->shared_slot, bytes);
    allocation->charge.allocation_attempted();
    void* allocated = cached;
    cudaError_t status = cudaSuccess;
    if (!allocated) {
        status = state_->stream_ordered_memory
            ? cudaMallocAsync(&allocated, bytes, state_->stream)
            : cudaMalloc(&allocated, bytes);
    }
    allocation->pointer = static_cast<float*>(allocated);
    if (status != cudaSuccess) {
        allocation->charge.quarantine();
        state_->quarantine_domain(status);
        cuda_check(status, state_->stream_ordered_memory
            ? "cudaMallocAsync(native NR tensor)" : "cudaMalloc(native NR tensor)");
    }
    allocation->charge.allocation_succeeded();
    if (state_->stream_ordered_memory) {
        state_->stream_drained = false;
    }
    state_->live += bytes;
    allocation->live_accounted = true;
    state_->peak = std::max(state_->peak, state_->live + state_->tensor_cached_bytes + state_->tensor_pending_bytes);
    lock.unlock();
    Tensor result{std::move(allocation), std::move(shape)};
    // Captured cudaMallocAsync/cudaFreeAsync nodes own their allocations and
    // encode last-use ordering. Legacy cudaMalloc addresses need a slot lease.
    if (capture_allocations_ && !state_->stream_ordered_memory) capture_allocations_->push_back(result);
    return result;
}

void Context::begin_capture(std::vector<Tensor>& allocations) {
    if (capture_active() || !tensor_cache_enabled() || !deferred_tensor_cache())
        throw std::logic_error("CUDA graph capture requires a prepared deferred tensor cache");
    cuda_check(cudaStreamBeginCapture(stream(), cudaStreamCaptureModeThreadLocal),
               "begin native NR CUDA graph capture");
    capture_allocations_ = &allocations;
}

cudaGraph_t Context::end_capture() {
    capture_allocations_ = nullptr;
    cudaGraph_t graph = nullptr;
    const auto status = cudaStreamEndCapture(stream(), &graph);
    if (status != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        cuda_check(status, "end native NR CUDA graph capture");
    }
    return graph;
}

std::vector<Tensor> Context::upload_batch(std::span<const UploadRequest> requests) {
    if (requests.empty()) return {};
    state_->select_device();
    std::vector<DeviceState::HostStaging*> staging;
    staging.reserve(requests.size());
    std::vector<Tensor> tensors;
    tensors.reserve(requests.size());
    try {
        // Allocate and copy all caller data before handing any address to CUDA.
        // Every staging record stays owned by DeviceState until the one shared
        // stream drain below, so no host pointer can be reused early.
        for (const auto& request : requests) {
            auto* entry = state_->make_staging(request.values.size());
            staging.push_back(entry);
            std::copy(request.values.begin(), request.values.end(), entry->values.begin());
            tensors.push_back(empty(resolved_shape(request.shape, request.values.size())));
        }
        const auto copy_stream = stream();
        for (size_t index = 0; index < requests.size(); ++index) {
            auto* entry = staging[index];
            state_->begin_transfer(entry);
            const auto status = cudaMemcpyAsync(tensors[index].data(), entry->values.data(),
                                                entry->bytes(), cudaMemcpyHostToDevice, copy_stream);
            if (status != cudaSuccess) {
                state_->quarantine_transfer(entry, status);
                cuda_check(status, "cudaMemcpyAsync(native NR owned upload batch)");
            }
        }
        const auto status = cudaStreamSynchronize(copy_stream);
        if (status != cudaSuccess) {
            for (auto* entry : staging) state_->quarantine_transfer(entry, status);
            cuda_check(status, "cudaStreamSynchronize(native NR owned upload batch)");
        }
        {
            std::lock_guard lock(state_->mutex);
            state_->stream_drained = true;
        }
        for (auto* entry : staging) state_->finish_staging(entry, false);
        return tensors;
    } catch (...) {
        // An attempted transfer remains registered for quarantine. Only records
        // that never reached a CUDA API may be released immediately.
        for (auto* entry : staging) state_->abandon_unsubmitted_staging(entry);
        throw;
    }
}

std::vector<Tensor> Context::upload_batch_borrowed(std::span<const UploadRequest> requests) {
    std::vector<Tensor> tensors;
    tensors.reserve(requests.size());
    for(const auto& request:requests)
        tensors.push_back(empty(resolved_shape(request.shape,request.values.size())));
    const auto copy_stream=stream();
    for(size_t index=0;index<requests.size();++index) {
        const auto& source=requests[index].values;
        const auto status=cudaMemcpyAsync(tensors[index].data(),source.data(),source.size_bytes(),
                                         cudaMemcpyHostToDevice,copy_stream);
        if(status!=cudaSuccess) {
            std::lock_guard lock(state_->mutex);
            state_->quarantine_domain(status);
            cuda_check(status,"upload SDK-owned native NR input");
        }
    }
    synchronize();
    return tensors;
}

Tensor Context::upload(std::vector<int64_t> shape, std::span<const float> values) {
    UploadRequest request{std::move(shape), values};
    auto tensors = upload_batch(std::span<const UploadRequest>(&request, 1));
    return std::move(tensors.front());
}
std::vector<float> Context::download(const Tensor& tensor) {
    return download_staged(tensor, {});
}

void Context::download_into_pinned(const Tensor& tensor, const std::shared_ptr<float>& destination) {
    (void)download_staged(tensor, destination);
}

std::vector<float> Context::download_staged(const Tensor& tensor, const std::shared_ptr<float>& destination) {
    if (!owns(tensor))
        throw std::invalid_argument("native NR download belongs to another context");
    auto* staging = state_->make_staging(tensor.numel(), destination);
    try {
        const auto copy_stream = stream();
        state_->begin_transfer(staging);
        auto status = cudaMemcpyAsync(staging->data(), tensor.data(), staging->bytes(),
                                      cudaMemcpyDeviceToHost, copy_stream);
        if (status != cudaSuccess) {
            state_->quarantine_transfer(staging, status);
            cuda_check(status, "cudaMemcpyAsync(native NR owned download)");
        }
        status = cudaStreamSynchronize(copy_stream);
        if (status != cudaSuccess) {
            state_->quarantine_transfer(staging, status);
            cuda_check(status, "cudaStreamSynchronize(native NR owned download)");
        }
        {
            std::lock_guard lock(state_->mutex);
            state_->stream_drained = true;
        }
        auto values = state_->finish_staging(staging, true);
        staging = nullptr;
        return values;
    } catch (...) {
        state_->abandon_unsubmitted_staging(staging);
        throw;
    }
}
void Context::synchronize() {
    const auto status = cudaStreamSynchronize(stream());
    if (status != cudaSuccess) {
        std::lock_guard lock(state_->mutex);
        state_->quarantine_domain(status);
        cuda_check(status, "cudaStreamSynchronize(native NR)");
    }
    std::lock_guard lock(state_->mutex);
    if (state_->deferred_tensor_cache && !state_->tensor_pending_blocks.empty()) {
        for (auto& [bytes, blocks] : state_->tensor_pending_blocks) {
            auto& reusable = state_->tensor_free_blocks[bytes];
            reusable.insert(reusable.end(), blocks.begin(), blocks.end());
            state_->tensor_cached_bytes += static_cast<uint64_t>(blocks.size()) * bytes;
        }
        state_->tensor_pending_blocks.clear();
        state_->tensor_pending_bytes = 0;
    }
    state_->stream_drained = true;
}
uint64_t Context::peak_live_bytes() const { std::lock_guard lock(state_->mutex); return state_->peak; }
uint64_t Context::live_bytes() const { std::lock_guard lock(state_->mutex); return state_->live + state_->tensor_cached_bytes + state_->tensor_pending_bytes; }
uint64_t Context::budget_bytes() const { return state_->budget; }
uint64_t Context::host_staging_live_bytes() const { std::lock_guard lock(state_->mutex); return state_->host_live; }
uint64_t Context::host_staging_peak_bytes() const { std::lock_guard lock(state_->mutex); return state_->host_peak; }
uint64_t Context::host_staging_budget_bytes() const { return state_->host_budget; }
uint64_t Context::host_staging_quarantined_bytes() const { std::lock_guard lock(state_->mutex); return state_->host_quarantined; }
bool Context::stream_ordered_memory() const { std::lock_guard lock(state_->mutex); return state_->stream_ordered_memory; }
bool Context::fast_math() const { std::lock_guard lock(state_->mutex); return state_->fast_math; }
bool Context::half_math() const { std::lock_guard lock(state_->mutex); return state_->half_math; }
bool Context::tensor_cache_enabled() const { std::lock_guard lock(state_->mutex); return state_->tensor_cache; }
uint64_t Context::tensor_cached_bytes() const { std::lock_guard lock(state_->mutex); return state_->tensor_cached_bytes; }
bool Context::deferred_tensor_cache() const { std::lock_guard lock(state_->mutex); return state_->deferred_tensor_cache; }
bool Context::explicit_half_gemm() const { std::lock_guard lock(state_->mutex); return state_->explicit_half_gemm; }
bool Context::explicit_fp8_gemm() const { std::lock_guard lock(state_->mutex); return state_->explicit_fp8_gemm; }
cublasLtHandle_t Context::blas_lt() const {
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (!state_->blas_lt) throw std::runtime_error("native NR explicit FP8 cuBLASLt handle is not prepared");
    return state_->blas_lt;
}
void* Context::blas_workspace() const {
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    return state_->workspace;
}
size_t Context::blas_workspace_bytes() const {
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    return static_cast<size_t>(state_->main_allocated_bytes);
}
const void* Context::half_parameter(const Parameter& parameter) const {
    state_->select_device();
    const float* source = parameter.data();
    const size_t elements = parameter.numel();
    if (!source || !elements) throw std::invalid_argument("native NR half parameter is empty");
    const size_t bytes = elements * sizeof(uint16_t);
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (const auto found = state_->half_parameters.find(source); found != state_->half_parameters.end())
        return found->second.first;
    if (bytes > state_->budget || state_->live + state_->tensor_cached_bytes + state_->tensor_pending_bytes +
        state_->half_parameter_bytes + state_->half_activation_bytes > state_->budget - bytes)
        throw DeviceResourceExhausted("native NR explicit-half parameter cache exhausted");
    void* converted = nullptr;
    cuda_check(cudaMalloc(&converted, bytes), "cudaMalloc(native NR half parameter)");
    try {
        state_->half_parameters.emplace(source, std::make_pair(converted, bytes));
    } catch (...) {
        cudaFree(converted);
        throw;
    }
    state_->half_parameter_bytes += bytes;
    convert_f32_to_f16(state_->stream, source, converted, elements);
    return converted;
}
void* Context::half_activation(size_t elements) const {
    if (!elements) throw std::invalid_argument("native NR half activation is empty");
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (state_->half_activation_capacity >= elements) return state_->half_activation_current;
    const size_t bytes = elements * sizeof(uint16_t);
    if (bytes > state_->budget || state_->live + state_->tensor_cached_bytes + state_->tensor_pending_bytes +
        state_->half_parameter_bytes + state_->half_activation_bytes > state_->budget - bytes)
        throw DeviceResourceExhausted("native NR explicit-half activation workspace exhausted");
    void* converted = nullptr;
    cuda_check(cudaMalloc(&converted, bytes), "cudaMalloc(native NR half activation)");
    try {
        state_->half_activation_blocks.emplace_back(converted, bytes);
    } catch (...) {
        cudaFree(converted);
        throw;
    }
    state_->half_activation_current = converted;
    state_->half_activation_capacity = elements;
    state_->half_activation_bytes += bytes;
    return converted;
}
const void* Context::fp8_parameter(const Parameter& parameter) const {
    state_->select_device();
    const float* source = parameter.data();
    const size_t elements = parameter.numel();
    if (!source || !elements) throw std::invalid_argument("native NR explicit-FP8 parameter is empty");
    const size_t bytes = elements;
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (!state_->explicit_fp8_gemm || !state_->blas_lt)
        throw std::runtime_error("native NR explicit-FP8 path is not enabled");
    if (const auto found = state_->fp8_parameters.find(source); found != state_->fp8_parameters.end())
        return found->second.first;
    if (bytes > state_->budget || state_->live + state_->tensor_cached_bytes + state_->tensor_pending_bytes +
        state_->half_parameter_bytes + state_->half_activation_bytes + state_->fp8_parameter_bytes +
        state_->fp8_activation_bytes > state_->budget - bytes)
        throw DeviceResourceExhausted("native NR explicit-FP8 parameter cache exhausted");
    void* converted = nullptr;
    cuda_check(cudaMalloc(&converted, bytes), "cudaMalloc(native NR explicit-FP8 parameter)");
    try {
        state_->fp8_parameters.emplace(source, std::make_pair(converted, bytes));
    } catch (...) {
        cudaFree(converted);
        throw;
    }
    state_->fp8_parameter_bytes += bytes;
    convert_f32_to_fp8(state_->stream, source, converted, elements);
    return converted;
}
void* Context::fp8_activation(size_t elements) const {
    if (!elements) throw std::invalid_argument("native NR explicit-FP8 activation is empty");
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (!state_->explicit_fp8_gemm || !state_->blas_lt)
        throw std::runtime_error("native NR explicit-FP8 path is not enabled");
    if (state_->fp8_activation_capacity >= elements) return state_->fp8_activation_current;
    const size_t bytes = elements;
    if (bytes > state_->budget || state_->live + state_->tensor_cached_bytes + state_->tensor_pending_bytes +
        state_->half_parameter_bytes + state_->half_activation_bytes + state_->fp8_parameter_bytes +
        state_->fp8_activation_bytes > state_->budget - bytes)
        throw DeviceResourceExhausted("native NR explicit-FP8 activation workspace exhausted");
    void* converted = nullptr;
    cuda_check(cudaMalloc(&converted, bytes), "cudaMalloc(native NR explicit-FP8 activation)");
    try {
        state_->fp8_activation_blocks.emplace_back(converted, bytes);
    } catch (...) {
        cudaFree(converted);
        throw;
    }
    state_->fp8_activation_current = converted;
    state_->fp8_activation_capacity = elements;
    state_->fp8_activation_bytes += bytes;
    return converted;
}
void* Context::batched_pointer_workspace(size_t pointer_count) const {
    if (!pointer_count) throw std::invalid_argument("native NR pointer batch is empty");
    state_->select_device();
    std::lock_guard lock(state_->mutex);
    state_->check_ready();
    if (state_->batched_pointer_capacity >= pointer_count) return state_->batched_pointer_workspace;
    const size_t bytes = pointer_count * sizeof(void*) * 6;
    if (bytes > state_->budget || state_->live + state_->tensor_cached_bytes + state_->tensor_pending_bytes +
        state_->half_parameter_bytes + state_->half_activation_bytes + state_->batched_pointer_bytes >
        state_->budget - bytes)
        throw DeviceResourceExhausted("native NR pointer batch workspace exhausted");
    if (state_->batched_pointer_workspace) {
        cuda_check(cudaStreamSynchronize(state_->stream), "cudaStreamSynchronize(native NR pointer batch resize)");
        cuda_check(cudaFree(state_->batched_pointer_workspace), "cudaFree(native NR pointer batch resize)");
        state_->batched_pointer_workspace = nullptr;
        state_->batched_pointer_capacity = 0;
        state_->batched_pointer_bytes = 0;
    }
    void* replacement = nullptr;
    cuda_check(cudaMalloc(&replacement, bytes), "cudaMalloc(native NR pointer batch workspace)");
    state_->batched_pointer_workspace = replacement;
    state_->batched_pointer_capacity = pointer_count;
    state_->batched_pointer_bytes = bytes;
    return replacement;
}
} // namespace rw::native_nr
