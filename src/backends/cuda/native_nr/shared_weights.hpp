#pragma once
// RW040 opt-in production ownership API; not an installed public SDK header.
#include "../../../features/nr/weights.hpp"
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <vector>

namespace rw::native_nr {
class Context;
class Weights;
struct Tensor;
struct DeviceState;
class SharedDeviceOwner;
class DeviceWeightPool;
class SlotParameterBinding;
struct PreparationStatistics;
class DeviceResourceExhausted final : public std::bad_alloc {
    const char* message_;
public:
    explicit DeviceResourceExhausted(const char* static_message) noexcept:message_(static_message){}
    const char* what() const noexcept override{return message_;}
};

enum class DeviceChargeKind : uint8_t { immutable_weights, slot_payload, slot_workspace, caller_reserve };
struct DeviceBudgetStatistics {
    uint64_t limit_bytes=0, charged_bytes=0, charged_peak_bytes=0;
    uint64_t pending_bytes=0, allocated_live_bytes=0, quarantined_bytes=0;
    // Disjoint charged categories; quarantine is a diagnostic subset, not an
    // additional term in admission. slot_payload includes activation + I/O.
    std::array<uint64_t,4> charged_by_kind{};
};
class SharedDeviceBudget;
class DeviceBudgetReservation final {
    struct Impl;
    std::shared_ptr<Impl> implementation_;
    explicit DeviceBudgetReservation(std::shared_ptr<Impl>);
    friend class SharedDeviceBudget;
    friend class DeviceWeightPool;
    friend class ImmutableParameterView;
    friend class Context;
    friend struct DeviceState;
    void allocation_attempted() noexcept;
    void allocation_succeeded() noexcept;
    void no_allocation_confirmed() noexcept;
    void released_after_physical_free() noexcept;
    void quarantine() noexcept;
public:
    DeviceBudgetReservation() noexcept=default;
    DeviceBudgetReservation(DeviceBudgetReservation&&) noexcept=default;
    DeviceBudgetReservation& operator=(DeviceBudgetReservation&&) noexcept=default;
    DeviceBudgetReservation(const DeviceBudgetReservation&)=delete;
    DeviceBudgetReservation& operator=(const DeviceBudgetReservation&)=delete;
    ~DeviceBudgetReservation();
    uint64_t bytes() const noexcept;
    explicit operator bool() const noexcept;
};
class SharedDeviceBudget final {
    struct Impl;
    std::shared_ptr<Impl> implementation_;
    explicit SharedDeviceBudget(uint64_t);
    friend class SharedDeviceOwner;
    friend class DeviceBudgetReservation;
public:
    // The returned record is registered/charged before any cudaMalloc. A
    // never-attempted record can retire on destruction. An attempted record
    // requires the backend's explicit no-allocation/free proof or stays Q.
    DeviceBudgetReservation reserve(DeviceChargeKind,uint64_t slot_id,uint64_t bytes);
    DeviceBudgetStatistics statistics() const;
};

struct SharedDeviceIdentity {
    std::array<uint8_t,16> uuid{};
    uintptr_t actual_cuda_context=0;
    uint64_t owner_generation=0, device_epoch=0;
    bool operator==(const SharedDeviceIdentity&) const = default;
};
// Pure CPU predicate, also exercised without a CUDA owner. The caller must
// retain the real owner transaction lock through publication/quarantine.
inline bool publication_identity_current(bool ready,bool quarantined,
    const SharedDeviceIdentity& current,const SharedDeviceIdentity& submitted) noexcept {
    return ready && !quarantined && current==submitted;
}
class SharedDeviceOwner final : public std::enable_shared_from_this<SharedDeviceOwner> {
    struct Impl;
    std::unique_ptr<Impl> implementation_;
    explicit SharedDeviceOwner(std::unique_ptr<Impl>);
    friend class DeviceWeightPool;
    friend class Context;
    friend struct DeviceState;
    void retain_slot_control();
    void retire_slot_control() noexcept;
    void retain_pool_control();
    void retire_pool_control() noexcept;
public:
    ~SharedDeviceOwner();
    static std::shared_ptr<SharedDeviceOwner> acquire(int visible_device_ordinal,uint64_t device_budget_bytes);
    static bool is_canonical_owner(const std::shared_ptr<SharedDeviceOwner>&) noexcept;
    SharedDeviceIdentity identity() const noexcept;
    int device_ordinal() const noexcept;
    std::shared_ptr<SharedDeviceBudget> budget() const noexcept;
    bool quarantined() const noexcept;
    bool accepting() const noexcept;
    bool retired() const noexcept;
    bool retire_owned_context() noexcept;
    // Invalidate the WHOLE shared CUDA-context domain, never only one stream.
    // No allocation/diagnostic string construction on this path.
    void quarantine() noexcept;
    void check_current_context() const;
    bool commit_if_current(const SharedDeviceIdentity& submitted,
        void(*publish)(void*) noexcept,void* payload) noexcept;
    uint64_t quarantine_waiters() const noexcept;
};

class ImmutableParameterView final {
    struct Impl;
    std::shared_ptr<const Impl> implementation_;
    explicit ImmutableParameterView(std::shared_ptr<const Impl>);
    static void destroy(Impl*) noexcept;
    friend class DeviceWeightPool;
    friend class SlotParameterBinding;
    friend class Parameter;
public:
    ImmutableParameterView() noexcept=default;
    const float* data() const;
    const std::vector<int64_t>& shape() const noexcept;
    uint64_t numel() const noexcept;
    bool empty() const noexcept;
    SharedDeviceIdentity device_identity() const noexcept;
    const ModelFingerprint& model_fingerprint() const;
    // No raw-pointer construction, mutable pointer, or activation conversion.
};

// Host-kernel facade. A legacy Tensor stays a SAME-SLOT readonly operand;
// only the sealed shared alternative can cross slots. There is no conversion
// back to mutable activation Tensor. Numeric GPU arithmetic is unchanged.
class Parameter final {
    struct Impl;
    std::shared_ptr<const Impl> implementation_;
public:
    Parameter() noexcept=default;
    Parameter(const Tensor&);
    Parameter(ImmutableParameterView);
    const float* data() const;
    const std::vector<int64_t>& shape() const noexcept;
    uint64_t numel() const;
    bool empty() const noexcept;
    const Tensor* legacy_tensor() const noexcept;
    const ImmutableParameterView* shared_view() const noexcept;
    bool same_storage_owner(const Parameter&) const noexcept;
};

struct WeightReadStamp { uint64_t session_id=0, session_epoch=0, expected_predecessor=0; };
class PhysicalDrainProof final {
    SharedDeviceIdentity identity_{};
    uintptr_t stream_=0;
    uint64_t slot_id_=0, read_id_=0;
    PhysicalDrainProof(SharedDeviceIdentity,uintptr_t,uint64_t,uint64_t) noexcept;
    friend class Context;
    friend struct DeviceState;
    friend class SlotParameterBinding;
    friend class WeightReadPin;
};
class WeightReadPin final {
    struct Impl;
    std::unique_ptr<Impl> implementation_;
    explicit WeightReadPin(std::unique_ptr<Impl>);
    friend class SlotParameterBinding;
    friend class Context;
    friend struct DeviceState;
    void physically_complete(const PhysicalDrainProof&);
public:
    WeightReadPin() noexcept;
    WeightReadPin(WeightReadPin&&) noexcept;
    WeightReadPin& operator=(WeightReadPin&&) noexcept;
    WeightReadPin(const WeightReadPin&)=delete;
    WeightReadPin& operator=(const WeightReadPin&)=delete;
    ~WeightReadPin(); // Unconfirmed last owner retains exact record + pool.
    uint64_t read_id() const noexcept;
    uint64_t slot_id() const noexcept;
    SharedDeviceIdentity submitted_device_identity() const noexcept;
};
class SlotParameterBinding final {
    struct Impl;
    std::unique_ptr<Impl> implementation_;
    explicit SlotParameterBinding(std::unique_ptr<Impl>);
    friend class DeviceWeightPool;
    friend class Context;
    friend struct DeviceState;
    friend class WeightReadPin;
    void retire_after_drain(const PhysicalDrainProof&);
public:
    ~SlotParameterBinding(); // Unconfirmed graph pin never retires on drop.
    WeightReadPin begin_read(WeightReadStamp);
    // Requires this slot's active read and exact canonical owner/context/epoch;
    // enqueues wait on an already-recorded ready-event before the kernel read.
    void validate_and_wait(const ImmutableParameterView&);
    bool accepts(const ImmutableParameterView&) const noexcept;
    bool bound_to_stream(cudaStream_t) const noexcept;
    uint64_t slot_id() const noexcept;
    SharedDeviceIdentity device_identity() const noexcept;
};

struct SharedWeightStatistics {
    uint64_t entries=0, entry_limit=2048, uploads=0, allocations=0, frees=0;
    uint64_t weight_live_bytes=0, weight_peak_bytes=0, weight_quarantined_bytes=0;
    uint64_t graph_pins=0, active_reads=0, physically_completed_reads=0;
    uint64_t owner_rejections=0, context_rejections=0;
    uint64_t upload_host_live_bytes=0, upload_host_peak_bytes=0, upload_host_limit_bytes=0;
    bool retirement_attempted=false,retirement_succeeded=false,retirement_failed=false;
};
class DeviceWeightPool final : public std::enable_shared_from_this<DeviceWeightPool> {
    struct Impl;
    std::unique_ptr<Impl> implementation_;
    explicit DeviceWeightPool(std::unique_ptr<Impl>);
    friend class Weights;
    friend class SlotParameterBinding;
    friend class WeightReadPin;
    // Only the real source decoder can publish a shared parameter. The key
    // includes raw logical identity/shape, conversion recipe and exact plan for
    // derived constants. No arbitrary activation upload/adoption API exists.
    ImmutableParameterView acquire_parameter(const std::string& logical_key,const std::string& conversion,
        std::vector<int64_t> shape,bool geometry_dependent,const std::function<std::vector<float>()>& decode,
        std::span<const float> expected_derived_values={});
public:
    ~DeviceWeightPool();
    static std::shared_ptr<DeviceWeightPool> acquire(RegisteredModelOwner,
        std::shared_ptr<SharedDeviceOwner>,uint64_t immutable_weight_budget_bytes,
        uint64_t upload_host_budget_bytes,uint32_t max_slots=2);
    Weights& weights() const;
    RegisteredModelOwner model() const noexcept;
    std::shared_ptr<SharedDeviceOwner> device() const noexcept;
    std::unique_ptr<SlotParameterBinding> bind_slot(uint64_t slot_id,cudaStream_t);
    SharedWeightStatistics statistics() const;
    PreparationStatistics preparation_statistics() const;
    bool retire_after_slots_drained() noexcept;
};

} // namespace rw::native_nr
