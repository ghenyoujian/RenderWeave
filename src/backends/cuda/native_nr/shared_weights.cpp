#include "shared_weights.hpp"
#include "native_nr.hpp"
#include "../../../core/slot_capacity.hpp"
#include <cuda.h>
#include <openssl/evp.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cfenv>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>
#include <unordered_set>

namespace rw::native_nr {
namespace {
void require(bool condition,const char* message){if(!condition)throw std::invalid_argument(message);}
void driver_check(CUresult status,const char* message){if(status!=CUDA_SUCCESS)throw std::runtime_error(message);}
template<class A,class B>bool same_owner(const std::shared_ptr<A>& a,const std::shared_ptr<B>& b)noexcept{
    return a.use_count()!=0 && b.use_count()!=0 && !a.owner_before(b) && !b.owner_before(a);
}
uint64_t checked_bytes(const std::vector<int64_t>& shape){
    require(!shape.empty() && shape.size()<=8,"shared parameter rank outside bound");uint64_t count=1;
    for(auto extent:shape){require(extent>0,"shared parameter extent is not positive");
        if(uint64_t(extent)>UINT64_MAX/count)throw std::overflow_error("shared parameter shape overflow");count*=uint64_t(extent);}
    if(count>SIZE_MAX/sizeof(float))throw std::overflow_error("shared parameter byte size overflow");return count*sizeof(float);
}
std::string sha_bytes(const void* data,size_t count){
    std::array<uint8_t,32> bytes{};unsigned size=0;
    if(EVP_Digest(data,count,bytes.data(),&size,EVP_sha256(),nullptr)!=1 || size!=32)throw std::runtime_error("shared parameter fingerprint failed");
    constexpr char hex[]="0123456789abcdef";std::string text;for(auto value:bytes){text+=hex[value>>4];text+=hex[value&15];}return text;
}
struct DeviceRegistryRecord {std::weak_ptr<SharedDeviceOwner> owner;bool quarantined=false,retirement_complete=false;uint64_t generation=0;};
std::mutex& device_registry_mutex(){static auto* value=new std::mutex;return *value;}
std::map<std::array<uint8_t,16>,DeviceRegistryRecord>& devices(){static auto* value=new std::map<std::array<uint8_t,16>,DeviceRegistryRecord>;return *value;}
class CurrentDevice {
    std::shared_ptr<SharedDeviceOwner> owner_;CUcontext context_=nullptr;bool pushed_=false;
public:
    explicit CurrentDevice(std::shared_ptr<SharedDeviceOwner> owner,bool initializing=false):owner_(std::move(owner)){
        require(!owner_->quarantined() && (initializing || owner_->accepting()),"shared device is quarantined/retired");
        context_=reinterpret_cast<CUcontext>(owner_->identity().actual_cuda_context);
        driver_check(cuCtxPushCurrent(context_),"push shared primary CUDA context");pushed_=true;
        try{CUcontext current=nullptr;driver_check(cuCtxGetCurrent(&current),"verify selected shared primary context");require(current==context_ && current,"foreign shared CUDA context");}
        catch(...){CUcontext ignored=nullptr;if(cuCtxPopCurrent(&ignored)!=CUDA_SUCCESS || ignored!=context_)owner_->quarantine();pushed_=false;throw;}
    }
    ~CurrentDevice(){if(pushed_){CUcontext popped=nullptr;if(cuCtxPopCurrent(&popped)!=CUDA_SUCCESS || popped!=context_)owner_->quarantine();}}
};
}

struct SharedDeviceBudget::Impl {
    mutable std::mutex mutex;DeviceBudgetStatistics statistics;
    std::weak_ptr<SharedDeviceOwner> owner;
    explicit Impl(uint64_t limit){statistics.limit_bytes=limit;}
};
struct DeviceBudgetReservation::Impl {
    std::shared_ptr<SharedDeviceBudget::Impl> budget;
    DeviceChargeKind kind=DeviceChargeKind::slot_payload;uint64_t slot=0,bytes=0;
    bool charged=false,attempted=false,allocated=false,quarantined=false,released=false;
    ~Impl()noexcept{
        if(!charged || released)return;
        try{
            std::lock_guard lock(budget->mutex);auto& stats=budget->statistics;
            if(!attempted){stats.charged_bytes-=bytes;stats.pending_bytes-=bytes;stats.charged_by_kind[unsigned(kind)]-=bytes;if(quarantined)stats.quarantined_bytes-=bytes;released=true;}
            else if(!quarantined){stats.quarantined_bytes+=bytes;quarantined=true;}
        }catch(...){}
        if(attempted)if(auto owner=budget->owner.lock())owner->quarantine();
    }
};
DeviceBudgetReservation::DeviceBudgetReservation(std::shared_ptr<Impl> value):implementation_(std::move(value)){}
DeviceBudgetReservation::~DeviceBudgetReservation()=default;
uint64_t DeviceBudgetReservation::bytes()const noexcept{return implementation_?implementation_->bytes:0;}
DeviceBudgetReservation::operator bool()const noexcept{return implementation_ && !implementation_->released;}
void DeviceBudgetReservation::allocation_attempted()noexcept{if(implementation_)implementation_->attempted=true;}
void DeviceBudgetReservation::allocation_succeeded()noexcept{
    if(!implementation_)return;auto& entry=*implementation_;std::lock_guard lock(entry.budget->mutex);
    if(entry.released || entry.allocated)return;entry.attempted=true;entry.allocated=true;
    entry.budget->statistics.pending_bytes-=entry.bytes;entry.budget->statistics.allocated_live_bytes+=entry.bytes;
}
void DeviceBudgetReservation::no_allocation_confirmed()noexcept{
    if(!implementation_)return;auto& entry=*implementation_;std::lock_guard lock(entry.budget->mutex);
    if(entry.released || entry.allocated)return;auto& stats=entry.budget->statistics;
    stats.charged_bytes-=entry.bytes;stats.pending_bytes-=entry.bytes;stats.charged_by_kind[unsigned(entry.kind)]-=entry.bytes;
    if(entry.quarantined)stats.quarantined_bytes-=entry.bytes;entry.released=true;
}
void DeviceBudgetReservation::released_after_physical_free()noexcept{
    if(!implementation_)return;auto& entry=*implementation_;std::lock_guard lock(entry.budget->mutex);
    if(entry.released)return;auto& stats=entry.budget->statistics;stats.charged_bytes-=entry.bytes;
    if(entry.allocated)stats.allocated_live_bytes-=entry.bytes;else stats.pending_bytes-=entry.bytes;
    stats.charged_by_kind[unsigned(entry.kind)]-=entry.bytes;if(entry.quarantined)stats.quarantined_bytes-=entry.bytes;entry.released=true;
}
void DeviceBudgetReservation::quarantine()noexcept{
    if(!implementation_)return;auto& entry=*implementation_;
    {std::lock_guard lock(entry.budget->mutex);if(!entry.released && !entry.quarantined){entry.budget->statistics.quarantined_bytes+=entry.bytes;entry.quarantined=true;}}
    if(auto owner=entry.budget->owner.lock())owner->quarantine();
}
SharedDeviceBudget::SharedDeviceBudget(uint64_t limit):implementation_(std::make_shared<Impl>(limit)){}
DeviceBudgetReservation SharedDeviceBudget::reserve(DeviceChargeKind kind,uint64_t slot,uint64_t bytes){
    require(bytes && unsigned(kind)<4,"invalid shared device reservation");
    auto entry=std::make_shared<DeviceBudgetReservation::Impl>();entry->budget=implementation_;entry->kind=kind;entry->slot=slot;entry->bytes=bytes;
    const auto owner=implementation_->owner.lock();require(owner && owner->accepting(),"shared device admission has no live accepting owner");
    {std::lock_guard lock(implementation_->mutex);auto& stats=implementation_->statistics;
        if(bytes>stats.limit_bytes || stats.charged_bytes>stats.limit_bytes-bytes)throw DeviceResourceExhausted("aggregate shared-device allocation budget exhausted");
        stats.charged_bytes+=bytes;stats.pending_bytes+=bytes;stats.charged_by_kind[unsigned(kind)]+=bytes;
        stats.charged_peak_bytes=std::max(stats.charged_peak_bytes,stats.charged_bytes);entry->charged=true;}
    return DeviceBudgetReservation(std::move(entry));
}
DeviceBudgetStatistics SharedDeviceBudget::statistics()const{
    // Drop a possible last Owner only AFTER unlocking its budget; the Owner's
    // checked destructor reads this ledger and must never recurse on its mutex.
    const auto owner=implementation_->owner.lock();
    std::lock_guard lock(implementation_->mutex);auto result=implementation_->statistics;
    if(owner && owner->quarantined())result.quarantined_bytes=result.charged_bytes;
    return result;
}

struct SharedDeviceOwner::Impl {
    int ordinal=0;CUdevice device=0;CUcontext context=nullptr;bool retained=false;std::atomic<bool> ready{false};
    std::array<uint8_t,16> uuid{};uint64_t generation=0;
    std::atomic<uint64_t> epoch{1};std::atomic<bool> quarantined{false};
    mutable std::mutex transaction_mutex;
    std::atomic<uint64_t> quarantine_waiters{0};
    std::atomic<uint64_t> pool_controls{0},slot_controls{0};
    std::atomic<bool> retirement_attempted{false},retirement_succeeded{false},retirement_failed{false};
    std::shared_ptr<SharedDeviceBudget> budget;
    std::shared_ptr<SharedDeviceOwner> self_quarantine;
};
SharedDeviceOwner::SharedDeviceOwner(std::unique_ptr<Impl> value):implementation_(std::move(value)){}
SharedDeviceOwner::~SharedDeviceOwner(){
    if(implementation_ && !implementation_->retirement_succeeded.load() && !retire_owned_context())(void)implementation_.release();
}
bool SharedDeviceOwner::retire_owned_context()noexcept{
    if(!implementation_)return true;auto& value=*implementation_;
    try{
        {std::lock_guard lock(value.transaction_mutex);
            if(value.retirement_succeeded.load())return true;
            if(value.retirement_attempted.load() || value.quarantined.load())return false;
            // Slot/workspace/W frees have their own checked physical proofs.
            // Do not release this primary retain while any owned control or
            // charged allocation/pending reservation remains.
            if(value.pool_controls.load() || value.slot_controls.load() || value.budget->statistics().charged_bytes)return false;
            value.retirement_attempted=true;value.ready=false;
        }
        if(value.retained){driver_check(cuDevicePrimaryCtxRelease(value.device),"checked shared primary reference release");value.retained=false;}
        {std::lock_guard lock(value.transaction_mutex);
            if(value.quarantined.load()){value.retirement_failed=true;return false;}
            value.retirement_succeeded=true;
        }
        std::lock_guard lock(device_registry_mutex());auto found=devices().find(value.uuid);
        if(found!=devices().end() && found->second.generation==value.generation)found->second.retirement_complete=true;
        return true;
    }catch(...){value.retirement_failed=true;quarantine();return false;}
}
std::shared_ptr<SharedDeviceOwner> SharedDeviceOwner::acquire(int ordinal,uint64_t limit){
    require(ordinal>=0 && limit,"explicit shared-device ordinal and budget required");
    driver_check(cuInit(0),"initialize shared-device driver");CUdevice device=0;CUuuid raw{};
    driver_check(cuDeviceGet(&device,ordinal),"select visible shared device");driver_check(cuDeviceGetUuid(&raw,device),"query actual shared-device UUID");
    std::array<uint8_t,16> uuid{};std::copy(std::begin(raw.bytes),std::end(raw.bytes),uuid.begin());
    cudaDeviceProp properties{};cuda_check(cudaGetDeviceProperties(&properties,ordinal),"query shared CUDA device");
#if defined(RW_NR_HOPPER_WGMMA_ENABLED)
    require(properties.major==9 && properties.minor==0,"this NR build requires compute capability 9.0 (SM90a)");
#endif
    require(limit<=properties.totalGlobalMem,"shared device budget exceeds physical memory before owner registration");
    std::array<uint8_t,16> runtime_uuid{};std::copy(std::begin(properties.uuid.bytes),std::end(properties.uuid.bytes),runtime_uuid.begin());
    require(runtime_uuid==uuid,"runtime and driver visible-device UUID differ");
    std::unique_lock registry_lock(device_registry_mutex());auto& registry=devices();auto found=registry.find(uuid);
    if(found!=registry.end()){
        require(!found->second.quarantined,"this physical CUDA context domain is quarantined");
        if(auto owner=found->second.owner.lock();owner && !owner->retired()){
            registry_lock.unlock();
            require(!owner->quarantined() && owner->implementation_->ready,"shared device preparation is pending/quarantined");
            require(owner->budget()->statistics().limit_bytes==limit,"cannot silently change a live device budget");return owner;
        }
        require(found->second.retirement_complete,"previous shared primary owner is still physically retiring");
    }
    auto state=std::make_unique<Impl>();state->ordinal=ordinal;state->device=device;state->uuid=uuid;
    state->budget=std::shared_ptr<SharedDeviceBudget>(new SharedDeviceBudget(limit));
    static std::atomic<uint64_t> generations{1};state->generation=generations.fetch_add(1);
    auto owner=std::shared_ptr<SharedDeviceOwner>(new SharedDeviceOwner(std::move(state)));
    owner->implementation_->budget->implementation_->owner=owner;
    registry.insert_or_assign(uuid,DeviceRegistryRecord{owner,false,false,owner->implementation_->generation});registry_lock.unlock();
    try{
        auto& value=*owner->implementation_;driver_check(cuDevicePrimaryCtxRetain(&value.context,device),"retain actual shared primary context");value.retained=true;
        {CurrentDevice current(owner,true);}
        value.ready=true;return owner;
    }catch(...){owner->quarantine();owner->implementation_->self_quarantine=owner;throw;}
}
SharedDeviceIdentity SharedDeviceOwner::identity()const noexcept{
    const auto& value=*implementation_;return {value.uuid,reinterpret_cast<uintptr_t>(value.context),value.generation,value.epoch.load()};
}
bool SharedDeviceOwner::is_canonical_owner(const std::shared_ptr<SharedDeviceOwner>& candidate)noexcept{
    if(!candidate)return false;
    try{std::lock_guard lock(device_registry_mutex());for(const auto& [uuid,record]:devices()){
        (void)uuid;
        // A wrong control block must not temporarily become the last real
        // Owner and run GPU retirement while this registry mutex is held.
        if(record.owner.owner_before(candidate) || candidate.owner_before(record.owner))continue;
        if(auto actual=record.owner.lock())if(actual.get()==candidate.get() && actual->accepting())return true;
    }}catch(...){}return false;
}
int SharedDeviceOwner::device_ordinal()const noexcept{return implementation_->ordinal;}
std::shared_ptr<SharedDeviceBudget> SharedDeviceOwner::budget()const noexcept{return implementation_->budget;}
bool SharedDeviceOwner::quarantined()const noexcept{return implementation_->quarantined.load();}
bool SharedDeviceOwner::accepting()const noexcept{return implementation_->ready.load() && !implementation_->retirement_attempted.load() && !quarantined();}
bool SharedDeviceOwner::retired()const noexcept{return implementation_->retirement_succeeded.load();}
void SharedDeviceOwner::retain_slot_control(){
    std::lock_guard lock(implementation_->transaction_mutex);require(accepting(),"cannot register a slot on a closing device owner");implementation_->slot_controls.fetch_add(1);
}
void SharedDeviceOwner::retire_slot_control()noexcept{
    if(implementation_->slot_controls.fetch_sub(1)==0){implementation_->slot_controls.fetch_add(1);quarantine();}
}
void SharedDeviceOwner::retain_pool_control(){
    std::lock_guard lock(implementation_->transaction_mutex);require(accepting(),"cannot register an upload stream on a closing device owner");implementation_->pool_controls.fetch_add(1);
}
void SharedDeviceOwner::retire_pool_control()noexcept{
    if(implementation_->pool_controls.fetch_sub(1)==0){implementation_->pool_controls.fetch_add(1);quarantine();}
}
void SharedDeviceOwner::quarantine()noexcept{
    auto& value=*implementation_;
    try{
        {std::unique_lock lock(value.transaction_mutex,std::defer_lock);
            if(!lock.try_lock()){
                value.quarantine_waiters.fetch_add(1);
                try{lock.lock();}catch(...){value.quarantine_waiters.fetch_sub(1);throw;}
                value.quarantine_waiters.fetch_sub(1);
            }
            if(!value.quarantined.exchange(true))value.epoch.fetch_add(1);}
        // Never hold domain transaction lock while taking diagnostic registry
        // lock; callbacks hold Runtime -> domain and never call owner/GPU APIs.
        std::lock_guard lock(device_registry_mutex());auto found=devices().find(value.uuid);
        if(found!=devices().end() && found->second.generation==value.generation)found->second.quarantined=true;
    }catch(...){value.quarantined=true;}
}
bool SharedDeviceOwner::commit_if_current(const SharedDeviceIdentity& submitted,void(*publish)(void*)noexcept,void* payload)noexcept{
    if(!publish)return false;auto& value=*implementation_;
    try{
        std::lock_guard lock(value.transaction_mutex);
        if(!publication_identity_current(value.ready.load(),value.quarantined.load(),identity(),submitted))return false;
        publish(payload);return true;
    }catch(...){return false;}
}
uint64_t SharedDeviceOwner::quarantine_waiters()const noexcept{return implementation_->quarantine_waiters.load();}
void SharedDeviceOwner::check_current_context()const{
    require(accepting(),"shared CUDA-context owner is quarantined/retired");CUcontext current=nullptr;
    driver_check(cuCtxGetCurrent(&current),"query actual shared CUDA context");require(current==implementation_->context && current,"foreign actual CUDA context refused");
}

struct Parameter::Impl {Tensor legacy;ImmutableParameterView shared;bool uses_shared=false;};
Parameter::Parameter(const Tensor& value){if(!value.empty() || !value.shape.empty() || value.storage.use_count()!=0){auto impl=std::make_shared<Impl>();impl->legacy=value;implementation_=std::move(impl);}}
Parameter::Parameter(ImmutableParameterView value){if(!value.empty()){auto impl=std::make_shared<Impl>();impl->shared=std::move(value);impl->uses_shared=true;implementation_=std::move(impl);}}
const float* Parameter::data()const{require(bool(implementation_),"empty parameter has no data");return implementation_->uses_shared?implementation_->shared.data():implementation_->legacy.data();}
const std::vector<int64_t>& Parameter::shape()const noexcept{static const std::vector<int64_t> empty;return !implementation_?empty:implementation_->uses_shared?implementation_->shared.shape():implementation_->legacy.shape;}
uint64_t Parameter::numel()const{return !implementation_?0:implementation_->uses_shared?implementation_->shared.numel():implementation_->legacy.numel();}
bool Parameter::empty()const noexcept{return !implementation_ || (!implementation_->uses_shared && implementation_->legacy.empty());}
const Tensor* Parameter::legacy_tensor()const noexcept{return implementation_ && !implementation_->uses_shared?&implementation_->legacy:nullptr;}
const ImmutableParameterView* Parameter::shared_view()const noexcept{return implementation_ && implementation_->uses_shared?&implementation_->shared:nullptr;}
bool Parameter::same_storage_owner(const Parameter& other)const noexcept{
    if(!implementation_ || !other.implementation_)return !implementation_ && !other.implementation_;
    if(implementation_->uses_shared!=other.implementation_->uses_shared)return false;
    if(implementation_->uses_shared)return same_owner(implementation_->shared.implementation_,other.implementation_->shared.implementation_);
    return implementation_->legacy.storage.get()==other.implementation_->legacy.storage.get() && same_owner(implementation_->legacy.storage,other.implementation_->legacy.storage);
}

namespace {
struct PoolCounters {
    std::atomic<uint64_t> allocations{0},frees{0},uploads{0},live{0},peak{0},quarantined{0};
};
struct PoolRegistryRecord {
    std::weak_ptr<DeviceWeightPool> pool;
    std::shared_ptr<PoolCounters> allocation_lifetime;
};
std::mutex& pool_registry_mutex(){static auto* value=new std::mutex;return *value;}
std::map<std::string,PoolRegistryRecord>& pools(){static auto* value=new std::map<std::string,PoolRegistryRecord>;return *value;}
std::string pool_key(const RegisteredModelOwner& model,const std::shared_ptr<SharedDeviceOwner>& device){
    const auto& fp=model->fingerprint();const auto id=device->identity();std::string key;
    for(auto byte:id.uuid)key+=std::to_string(byte)+",";
    key+=":"+std::to_string(id.actual_cuda_context)+":"+std::to_string(id.owner_generation)+":"+std::to_string(id.device_epoch)+":";
    return key+fp.original_file_sha256+fp.topology_sha256+fp.exact_plan_sha256+fp.quantization_precision_layout_sha256+fp.lowering_recipe_sha256;
}
}
struct ImmutableParameterView::Impl {
    RegisteredModelOwner model;std::shared_ptr<SharedDeviceOwner> device;std::shared_ptr<PoolCounters> counters;
    SharedDeviceIdentity identity{};std::vector<int64_t> shape;std::string payload_sha256;uint64_t bytes=0,host_charged_bytes=0;
    float* pointer=nullptr;cudaEvent_t ready=nullptr;std::vector<float> staging;
    DeviceBudgetReservation charge;
    bool cuda_setup_attempted=false,recorded=false,upload_attempted=false,retirement_authorized=false;
};
void ImmutableParameterView::destroy(Impl* value)noexcept{
    if(!value)return;
    if(!value->pointer && !value->ready && !value->upload_attempted){delete value;return;}
    if(value->device->quarantined() || !value->retirement_authorized){
        if(value->pointer)value->counters->quarantined+=value->bytes;
        value->charge.quarantine();value->device->quarantine();return;
    }
    try{
        CurrentDevice current(value->device);
        if(value->recorded && cudaEventSynchronize(value->ready)!=cudaSuccess){value->charge.quarantine();value->device->quarantine();return;}
        if(value->pointer){
            if(cudaFree(value->pointer)!=cudaSuccess){value->charge.quarantine();value->device->quarantine();return;}
            value->pointer=nullptr;value->charge.released_after_physical_free();value->counters->frees++;value->counters->live-=value->bytes;
        }
        if(value->ready && cudaEventDestroy(value->ready)!=cudaSuccess){value->device->quarantine();return;}
    }catch(...){value->charge.quarantine();value->device->quarantine();return;}
    delete value;
}
ImmutableParameterView::ImmutableParameterView(std::shared_ptr<const Impl> value):implementation_(std::move(value)){}
const float* ImmutableParameterView::data()const{
    require(implementation_ && implementation_->recorded && !implementation_->device->quarantined(),"shared parameter is empty/unready/quarantined");return implementation_->pointer;
}
const std::vector<int64_t>& ImmutableParameterView::shape()const noexcept{static const std::vector<int64_t> empty;return implementation_?implementation_->shape:empty;}
uint64_t ImmutableParameterView::numel()const noexcept{return implementation_?implementation_->bytes/sizeof(float):0;}
bool ImmutableParameterView::empty()const noexcept{return !implementation_;}
SharedDeviceIdentity ImmutableParameterView::device_identity()const noexcept{return implementation_?implementation_->identity:SharedDeviceIdentity{};}
const ModelFingerprint& ImmutableParameterView::model_fingerprint()const{require(bool(implementation_),"empty shared parameter has no model");return implementation_->model->fingerprint();}

struct DeviceWeightPool::Impl {
    RegisteredModelOwner model;std::shared_ptr<SharedDeviceOwner> device;std::shared_ptr<PoolCounters> counters;
    std::shared_ptr<Weights> decoder;cudaStream_t upload=nullptr;
    bool upload_control_registered=false;
    uint64_t weight_limit=0,host_limit=0,host_live=0,host_peak=0;
    uint32_t slot_limit=2;std::atomic<bool> ready{false};
    mutable std::mutex mutex;
    std::map<std::string,std::shared_ptr<ImmutableParameterView::Impl>> entries;
    std::map<uint64_t,bool> slots;
    uint64_t graph_pins=0,active_reads=0,completed_reads=0,next_read=1,owner_rejections=0,context_rejections=0;
    std::shared_ptr<DeviceWeightPool> self_quarantine;
    std::atomic<bool> retirement_attempted{false},retirement_succeeded{false},retirement_failed{false};
    PreparationStatistics preparation_snapshot{};bool preparation_snapshot_valid=false;
    void retire_upload_staging(bool synchronize){
        if(synchronize)cuda_check(cudaStreamSynchronize(upload),"drain shared original-weight uploads");
        for(auto& [key,entry]:entries){(void)key;if(entry->staging.empty() || !entry->recorded)continue;
            const auto state=synchronize?cudaSuccess:cudaEventQuery(entry->ready);
            if(state==cudaErrorNotReady)continue;cuda_check(state,"query immutable upload readiness");
            host_live-=entry->host_charged_bytes;entry->host_charged_bytes=0;std::vector<float>().swap(entry->staging);
        }
    }
};
DeviceWeightPool::DeviceWeightPool(std::unique_ptr<Impl> value):implementation_(std::move(value)){}
DeviceWeightPool::~DeviceWeightPool(){
    if(implementation_ && !implementation_->retirement_succeeded && !retire_after_slots_drained())(void)implementation_.release();
}
bool DeviceWeightPool::retire_after_slots_drained()noexcept{
    if(!implementation_)return true;auto& value=*implementation_;
    std::shared_ptr<Weights> decoder;
    bool cpu_only=false;
    try{
        {std::lock_guard lock(value.mutex);
            if(value.retirement_succeeded)return true;
            cpu_only=!value.upload_control_registered && !value.upload && value.entries.empty();
            if(value.retirement_attempted || (!cpu_only && value.device->quarantined()))return false;
            if(value.graph_pins || value.active_reads)return false;
            value.retirement_attempted=true;value.ready=false;
            decoder=value.decoder;
        }
        // Preserve Weights -> Pool lock order. Closing was published under the
        // pool lock, but do not hold it while taking the decoder's cache lock.
        if(decoder){
            const auto snapshot=decoder->preparation_statistics();
            {std::lock_guard lock(value.mutex);value.preparation_snapshot=snapshot;value.preparation_snapshot_valid=true;}
            decoder->clear_device_cache_for_retirement();
        }
        if(cpu_only){std::lock_guard lock(value.mutex);value.decoder.reset();value.host_live=0;value.retirement_succeeded=true;return true;}
        CurrentDevice current(value.device);
        std::lock_guard lock(value.mutex);
        if(value.upload)cuda_check(cudaStreamSynchronize(value.upload),"final immutable upload drain");
        for(auto& [key,entry]:value.entries){(void)key;entry->retirement_authorized=true;}
        value.entries.clear();
        require(!value.device->quarantined() && value.counters->live.load()==0,
            "immutable views or uncertain frees remain after checked pool retirement");
        if(value.upload){cuda_check(cudaStreamDestroy(value.upload),"destroy immutable upload stream");value.upload=nullptr;}
        value.host_live=0;value.decoder.reset();
        if(value.upload_control_registered){value.device->retire_pool_control();value.upload_control_registered=false;}
        value.retirement_succeeded=true;return true;
    }catch(...){value.retirement_failed=true;value.device->quarantine();return false;}
}
std::shared_ptr<DeviceWeightPool> DeviceWeightPool::acquire(RegisteredModelOwner model,
    std::shared_ptr<SharedDeviceOwner> device,uint64_t weight_limit,uint64_t host_limit,uint32_t slot_limit){
    require(ModelRegistry::is_registered_owner(model) && SharedDeviceOwner::is_canonical_owner(device),"unregistered model or non-owning device control block refused before dereference");
    require(!device->quarantined() && weight_limit && host_limit && rw::detail::bounded_slot_capacity(slot_limit),
            "invalid bounded shared-weight pool profile");
    require(weight_limit<=device->budget()->statistics().limit_bytes,"shared W limit exceeds aggregate device budget");
    const auto key=pool_key(model,device);std::unique_lock registry_lock(pool_registry_mutex());auto& registry=pools();
    if(const auto found=registry.find(key);found!=registry.end()){
        if(auto existing=found->second.pool.lock()){
            const auto& value=*existing->implementation_;
            require(value.ready.load() && !value.device->quarantined(),"canonical shared-weight pool is pending/quarantined");
            require(same_owner(value.model,model) && value.model.get()==model.get() && same_owner(value.device,device) && value.device.get()==device.get(),"same fingerprint with noncanonical model/device control block refused");
            require(value.weight_limit==weight_limit && value.host_limit==host_limit && value.slot_limit==slot_limit,"cannot mutate a live shared-pool capacity");return existing;
        }
        require(!found->second.allocation_lifetime || found->second.allocation_lifetime->live.load()==0,
            "previous immutable parameter views still retain W; refusing a duplicate pool upload");
    }
    auto state=std::make_unique<Impl>();state->model=std::move(model);state->device=std::move(device);
    state->weight_limit=weight_limit;state->host_limit=host_limit;state->slot_limit=slot_limit;state->counters=std::make_shared<PoolCounters>();
    auto pool=std::shared_ptr<DeviceWeightPool>(new DeviceWeightPool(std::move(state)));
    registry.insert_or_assign(key,PoolRegistryRecord{pool,pool->implementation_->counters});registry_lock.unlock();
    bool cuda_setup_attempted=false;
    try{
        auto& value=*pool->implementation_;value.decoder=std::make_shared<Weights>(value.model->weights_path());
        value.decoder->attach_shared_pool(pool,value.model);
        value.device->retain_pool_control();value.upload_control_registered=true;
        cuda_setup_attempted=true;
        {CurrentDevice current(value.device);cuda_check(cudaStreamCreateWithFlags(&value.upload,cudaStreamNonBlocking),"create shared immutable upload stream");}
        value.ready=true;return pool;
    }catch(...){
        // A stream-creation error may report previous async failure. The exact
        // pool+decoder+device stay owned, never reconstructed as a second W.
        if(cuda_setup_attempted){pool->implementation_->device->quarantine();pool->implementation_->self_quarantine=pool;}throw;
    }
}
Weights& DeviceWeightPool::weights()const{require(implementation_->ready && implementation_->decoder && !implementation_->device->quarantined(),"shared source decoder is not ready");return *implementation_->decoder;}
RegisteredModelOwner DeviceWeightPool::model()const noexcept{return implementation_->model;}
std::shared_ptr<SharedDeviceOwner> DeviceWeightPool::device()const noexcept{return implementation_->device;}
ImmutableParameterView DeviceWeightPool::acquire_parameter(const std::string& logical,const std::string& conversion,
    std::vector<int64_t> shape,bool geometry_dependent,const std::function<std::vector<float>()>& decode,std::span<const float> expected_values){
    require(!logical.empty() && logical.size()<=256 && !conversion.empty() && conversion.size()<=128 && std::fegetround()==FE_TONEAREST,"invalid shared parameter key/rounding contract");
    const uint64_t bytes=checked_bytes(shape);auto& value=*implementation_;
    require(!geometry_dependent || expected_values.size_bytes()==bytes,"derived immutable key requires exact candidate bytes");
    const auto expected_sha=geometry_dependent?sha_bytes(expected_values.data(),expected_values.size_bytes()):std::string{};
    std::string key=logical+":"+conversion;
    if(geometry_dependent)key+=":"+value.model->fingerprint().exact_plan_sha256+":"+value.model->fingerprint().lowering_recipe_sha256;
    for(auto extent:shape)key+=":"+std::to_string(extent);
    CurrentDevice current(value.device);std::lock_guard lock(value.mutex);
    require(value.ready && !value.device->quarantined(),"immutable model pool not ready");
    if(const auto found=value.entries.find(key);found!=value.entries.end()){
        require(found->second->recorded,"registered unready parameter cannot be retried/replaced");
        require(!geometry_dependent || found->second->payload_sha256==expected_sha,"immutable derived bytes changed after a partial host-cache publication");
        return ImmutableParameterView(found->second);
    }
    if(value.entries.size()>=2048)throw DeviceResourceExhausted("immutable parameter entry limit exceeded");
    if(bytes>value.weight_limit || value.counters->live.load()>value.weight_limit-bytes)throw DeviceResourceExhausted("immutable W budget exhausted before upload");
    if(bytes>value.host_limit)throw DeviceResourceExhausted("single parameter exceeds bounded upload staging");
    try{value.retire_upload_staging(false);if(value.host_live>value.host_limit-bytes)value.retire_upload_staging(true);}
    catch(...){value.device->quarantine();value.self_quarantine=shared_from_this();throw;}
    if(value.host_live>value.host_limit-bytes)throw DeviceResourceExhausted("immutable upload host budget exhausted");
    auto raw=new ImmutableParameterView::Impl;raw->model=value.model;raw->device=value.device;raw->counters=value.counters;
    raw->identity=value.device->identity();raw->shape=std::move(shape);raw->bytes=bytes;
    std::shared_ptr<ImmutableParameterView::Impl> entry(raw,ImmutableParameterView::destroy);
    // Register pending identity before host decode/allocation or CUDA calls.
    value.entries.emplace(key,entry);
    bool clean_pre_upload_oom=false;
    try{
        entry->staging=decode();require(entry->staging.size()==bytes/sizeof(float),"source parameter decode changed exact shape");
        entry->payload_sha256=sha_bytes(entry->staging.data(),bytes);
        require(!geometry_dependent || entry->payload_sha256==expected_sha,"derived source decoding changed candidate bytes");
        const uint64_t host_bytes=entry->staging.capacity()*sizeof(float);
        if(host_bytes>value.host_limit-value.host_live)throw DeviceResourceExhausted("actual immutable staging capacity exceeds host budget");
        value.host_live+=host_bytes;entry->host_charged_bytes=host_bytes;value.host_peak=std::max(value.host_peak,value.host_live);
        entry->charge=value.device->budget()->reserve(DeviceChargeKind::immutable_weights,0,bytes);
        entry->cuda_setup_attempted=true;
        cuda_check(cudaEventCreateWithFlags(&entry->ready,cudaEventDisableTiming),"create immutable ready event");
        entry->charge.allocation_attempted();
        const auto allocated=cudaMalloc(reinterpret_cast<void**>(&entry->pointer),bytes);
        if(allocated==cudaErrorMemoryAllocation && !entry->pointer && !entry->upload_attempted){
            // Exactly the synchronous pre-H2D OOM case: prove older pool
            // uploads drained and the never-recorded event destroyed before
            // returning a reservation. Any cleanup/foreign epoch error falls
            // through the ordinary whole-domain quarantine catch below.
            value.retire_upload_staging(true);
            cuda_check(cudaEventDestroy(entry->ready),"clean unused ready event after pre-upload allocation OOM");entry->ready=nullptr;
            const auto last=cudaGetLastError();
            if(last!=cudaSuccess && last!=cudaErrorMemoryAllocation)cuda_check(last,"unexpected error while retiring known pre-upload OOM");
            require(!value.device->quarantined(),"device domain changed during OOM cleanup");
            entry->charge.no_allocation_confirmed();value.host_live-=entry->host_charged_bytes;entry->host_charged_bytes=0;
            std::vector<float>().swap(entry->staging);value.entries.erase(key);clean_pre_upload_oom=true;
            throw DeviceResourceExhausted("known pre-upload CUDA allocation OOM; request rejected without model fallback");
        }
        if(allocated!=cudaSuccess)cuda_check(allocated,"allocate immutable W once");
        entry->charge.allocation_succeeded();value.counters->allocations++;value.counters->live+=bytes;
        value.counters->peak=std::max(value.counters->peak.load(),value.counters->live.load());
        entry->upload_attempted=true;value.counters->uploads++;
        cuda_check(cudaMemcpyAsync(entry->pointer,entry->staging.data(),bytes,cudaMemcpyHostToDevice,value.upload),"upload owned original immutable parameter");
        cuda_check(cudaEventRecord(entry->ready,value.upload),"record immutable ready before publishing view");
        entry->recorded=true;return ImmutableParameterView(entry);
    }catch(...){
        if(clean_pre_upload_oom)throw;
        // No model fallback, duplicate preparation or failure-path allocation.
        // The entry owns the REAL DMA source before enqueue, and exact pool
        // self-retention keeps it through any unconfirmed asynchronous access.
        if(entry->cuda_setup_attempted || entry->ready || entry->pointer || entry->upload_attempted){entry->charge.quarantine();value.device->quarantine();value.self_quarantine=shared_from_this();}
        else{value.host_live-=entry->host_charged_bytes;entry->host_charged_bytes=0;value.entries.erase(key);}
        throw;
    }
}

struct SlotParameterBinding::Impl {
    std::shared_ptr<DeviceWeightPool> pool;SharedDeviceIdentity identity{};cudaStream_t stream=nullptr;
    uint64_t slot=0,active_read=0;bool registered=false,retired=false;
    std::unordered_set<const void*> ready_parameters;
};
struct WeightReadPin::Impl {
    std::shared_ptr<DeviceWeightPool> pool;SlotParameterBinding::Impl* slot=nullptr;
    SharedDeviceIdentity identity{};WeightReadStamp stamp{};uint64_t read=0;bool completed=false;
};
PhysicalDrainProof::PhysicalDrainProof(SharedDeviceIdentity identity,uintptr_t stream,uint64_t slot,uint64_t read)noexcept
    :identity_(identity),stream_(stream),slot_id_(slot),read_id_(read){}
SlotParameterBinding::SlotParameterBinding(std::unique_ptr<Impl> value):implementation_(std::move(value)){}
SlotParameterBinding::~SlotParameterBinding(){
    if(implementation_ && implementation_->registered && !implementation_->retired){implementation_->pool->device()->quarantine();(void)implementation_.release();}
}
std::unique_ptr<SlotParameterBinding> DeviceWeightPool::bind_slot(uint64_t slot,cudaStream_t stream){
    require(stream && !implementation_->device->quarantined(),"shared slot requires a real owned stream");implementation_->device->check_current_context();
    auto state=std::make_unique<SlotParameterBinding::Impl>();state->pool=shared_from_this();state->identity=implementation_->device->identity();state->stream=stream;state->slot=slot;
    auto binding=std::unique_ptr<SlotParameterBinding>(new SlotParameterBinding(std::move(state)));
    std::lock_guard lock(implementation_->mutex);
    require(implementation_->ready && rw::detail::slot_binding_has_capacity(implementation_->slots.size(),
        implementation_->slot_limit,implementation_->slots.count(slot)!=0),"shared slot identity/capacity already registered");
    implementation_->slots.emplace(slot,true);implementation_->graph_pins++;binding->implementation_->registered=true;return binding;
}
WeightReadPin::WeightReadPin()noexcept=default;
WeightReadPin::WeightReadPin(std::unique_ptr<Impl> value):implementation_(std::move(value)){}
WeightReadPin::WeightReadPin(WeightReadPin&&)noexcept=default;
WeightReadPin& WeightReadPin::operator=(WeightReadPin&& other)noexcept{
    if(this!=&other){this->~WeightReadPin();new(this)WeightReadPin(std::move(other));}return *this;
}
WeightReadPin::~WeightReadPin(){if(implementation_ && !implementation_->completed){implementation_->pool->device()->quarantine();(void)implementation_.release();}}
uint64_t WeightReadPin::read_id()const noexcept{return implementation_?implementation_->read:0;}
uint64_t WeightReadPin::slot_id()const noexcept{return implementation_?implementation_->slot->slot:0;}
SharedDeviceIdentity WeightReadPin::submitted_device_identity()const noexcept{return implementation_?implementation_->identity:SharedDeviceIdentity{};}
WeightReadPin SlotParameterBinding::begin_read(WeightReadStamp stamp){
    require(implementation_ && !implementation_->retired,"shared slot graph binding retired");auto& slot=*implementation_;auto& pool=*slot.pool->implementation_;
    slot.pool->device()->check_current_context();require(slot.identity==slot.pool->device()->identity(),"late device generation/epoch read refused");
    auto state=std::make_unique<WeightReadPin::Impl>();state->pool=slot.pool;state->slot=&slot;state->identity=slot.identity;state->stamp=stamp;
    std::lock_guard lock(pool.mutex);require(!slot.active_read,"shared execution slot already has an in-flight reader");
    require(pool.next_read!=UINT64_MAX,"shared read generation exhausted");state->read=pool.next_read++;slot.active_read=state->read;pool.active_reads++;
    return WeightReadPin(std::move(state));
}
void WeightReadPin::physically_complete(const PhysicalDrainProof& proof){
    require(implementation_ && !implementation_->completed,"shared read already completed/empty");auto& read=*implementation_;auto& pool=*read.pool->implementation_;
    require(proof.identity_==read.identity && proof.stream_==reinterpret_cast<uintptr_t>(read.slot->stream) &&
        proof.slot_id_==read.slot->slot && proof.read_id_==read.read,"physical drain proof owner/stream/read mismatch");
    std::lock_guard lock(pool.mutex);require(read.slot->active_read==read.read && pool.active_reads,"physical reader record mismatch");
    read.slot->active_read=0;pool.active_reads--;pool.completed_reads++;read.completed=true;
}
bool SlotParameterBinding::accepts(const ImmutableParameterView& parameter)const noexcept{
    if(!implementation_ || implementation_->retired || !parameter.implementation_)return false;
    const auto& slot=*implementation_;const auto& entry=*parameter.implementation_;const auto& pool=*slot.pool->implementation_;
    return !pool.device->quarantined() && entry.recorded && same_owner(entry.device,pool.device) && entry.device.get()==pool.device.get() &&
        same_owner(entry.model,pool.model) && entry.model.get()==pool.model.get() && entry.identity==slot.identity && slot.identity==pool.device->identity();
}
bool SlotParameterBinding::bound_to_stream(cudaStream_t stream)const noexcept{return implementation_ && implementation_->stream==stream;}
uint64_t SlotParameterBinding::slot_id()const noexcept{return implementation_?implementation_->slot:0;}
SharedDeviceIdentity SlotParameterBinding::device_identity()const noexcept{return implementation_?implementation_->identity:SharedDeviceIdentity{};}
void SlotParameterBinding::validate_and_wait(const ImmutableParameterView& parameter){
    require(accepts(parameter),"shared readonly parameter has foreign model/device/context/epoch");auto& slot=*implementation_;
    slot.pool->device()->check_current_context();
    {std::lock_guard lock(slot.pool->implementation_->mutex);require(slot.active_read,"parameter read requires an internally registered physical read pin");}
    // The immutable entry and this stream both live for the entire binding.
    // One ordered wait establishes readiness for every subsequent read on
    // the same stream, including the graph prepared after a warmup drain.
    const void* identity=parameter.implementation_.get();
    if(slot.ready_parameters.count(identity))return;
    const auto status=cudaStreamWaitEvent(slot.stream,parameter.implementation_->ready,0);
    if(status!=cudaSuccess){slot.pool->device()->quarantine();cuda_check(status,"wait already-recorded immutable ready event");}
    slot.ready_parameters.insert(identity);
}
void SlotParameterBinding::retire_after_drain(const PhysicalDrainProof& proof){
    require(implementation_ && !implementation_->retired,"shared graph binding already retired");auto& slot=*implementation_;auto& pool=*slot.pool->implementation_;
    require(proof.identity_==slot.identity && proof.slot_id_==slot.slot && proof.stream_==reinterpret_cast<uintptr_t>(slot.stream) && proof.read_id_==0,
        "graph retirement needs exact final slot physical-drain proof");
    std::lock_guard lock(pool.mutex);require(!slot.active_read && pool.graph_pins && pool.slots.count(slot.slot),"active physical reader blocks graph unpin");
    pool.graph_pins--;pool.slots.erase(slot.slot);slot.retired=true;
}
SharedWeightStatistics DeviceWeightPool::statistics()const{
    const auto& value=*implementation_;std::lock_guard lock(value.mutex);SharedWeightStatistics result;
    result.entries=value.entries.size();result.allocations=value.counters->allocations;result.frees=value.counters->frees;result.uploads=value.counters->uploads;
    result.weight_live_bytes=value.counters->live;result.weight_peak_bytes=value.counters->peak;result.weight_quarantined_bytes=value.counters->quarantined;
    if(value.device->quarantined()){
        result.weight_quarantined_bytes=0;
        for(const auto& [key,entry]:value.entries){(void)key;if(entry->charge)result.weight_quarantined_bytes+=entry->charge.bytes();}
        // A failed checked deleter deliberately retains its exact allocation
        // outside a cleared entry table. Keep known live allocations visible;
        // entry charges additionally cover unconfirmed pending allocations.
        result.weight_quarantined_bytes=std::max(result.weight_quarantined_bytes,result.weight_live_bytes);
    }
    result.graph_pins=value.graph_pins;result.active_reads=value.active_reads;result.physically_completed_reads=value.completed_reads;
    result.owner_rejections=value.owner_rejections;result.context_rejections=value.context_rejections;
    result.upload_host_live_bytes=value.host_live;result.upload_host_peak_bytes=value.host_peak;result.upload_host_limit_bytes=value.host_limit;
    result.retirement_attempted=value.retirement_attempted.load();result.retirement_succeeded=value.retirement_succeeded.load();result.retirement_failed=value.retirement_failed.load();return result;
}
PreparationStatistics DeviceWeightPool::preparation_statistics()const{
    std::shared_ptr<Weights> decoder;
    {std::lock_guard lock(implementation_->mutex);
        if(implementation_->preparation_snapshot_valid)return implementation_->preparation_snapshot;
        decoder=implementation_->decoder;}
    if(!decoder)throw std::logic_error("source decoder unavailable during partial shared-pool setup");
    return decoder->preparation_statistics();
}
} // namespace rw::native_nr
