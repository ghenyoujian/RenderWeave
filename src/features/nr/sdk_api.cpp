#include "rw/rw_native_nr.h"
#include "native_graph.hpp"
#include "metrics_policy.hpp"
#include "input_validation.hpp"
#include "../../backends/cuda/native_nr/preparation_json.hpp"
#include "../../backends/cuda/native_nr/lowering_recipe.hpp"
#include "../../backends/cuda/native_nr/shared_weights.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#if defined(__linux__)
#include <sys/resource.h>
#endif
#ifndef RW_NATIVE_BUILD_COMMIT
#error "Build the native NR SDK from its frozen CMake target with RW_NATIVE_BUILD_COMMIT"
#endif

namespace rw::nr_sdk {
using Json=nlohmann::json;
namespace fs=std::filesystem;
using native_nr::Context;
using native_nr::Graph;
using native_nr::SourcePlan;
using native_nr::Weights;
using native_nr::Tensor;
thread_local std::array<char,2048> last_error{};
void copy_text(char* destination,size_t capacity,const char* text) noexcept {
    if(!capacity) return;
    size_t used=0;
    if(text) while(used+1<capacity && text[used]) { destination[used]=text[used]; ++used; }
    destination[used]=0;
}
void append_text(char* destination,size_t capacity,const char* text) noexcept {
    size_t used=0; while(used<capacity && destination[used]) ++used;
    if(used<capacity) copy_text(destination+used,capacity-used,text);
}
void set_error(const char* text) noexcept { copy_text(last_error.data(),last_error.size(),text); }
constexpr const char* original_dll="2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f";
constexpr const char* original_resource="836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4";
constexpr const char* pinned_source="3beef7935a8ef976b47ab78b906e75600c52d2de";

uint64_t now_ns() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
struct StatusError:std::runtime_error {
    rw_status_t status;
    StatusError(rw_status_t s,const char* message):std::runtime_error(message),status(s) {}
};
template<class F> rw_status_t boundary(F&& function) noexcept {
    last_error[0]=0;
    try { return function(); }
    catch(const StatusError& e) { set_error(e.what()); return e.status; }
    catch(const native_nr::DeviceResourceExhausted& e) {set_error(e.what());return RW_RESOURCE_EXHAUSTED;}
    catch(const std::bad_alloc&) { set_error("host allocation failed"); return RW_RESOURCE_EXHAUSTED; }
    catch(const std::overflow_error& e) { set_error(e.what()); return RW_OVERFLOW; }
    catch(const std::invalid_argument& e) { set_error(e.what()); return RW_INVALID_ARGUMENT; }
    catch(const std::exception& e) { set_error(e.what()); return RW_INTERNAL_ERROR; }
    catch(...) { set_error("unexpected exception stopped at native NR C boundary"); return RW_INTERNAL_ERROR; }
}
uint64_t multiply(uint64_t a,uint64_t b) {
    if(b && a>UINT64_MAX/b) throw std::overflow_error("SDK size product overflow");
    return a*b;
}
uint64_t add(uint64_t a,uint64_t b) {
    if(a>UINT64_MAX-b) throw std::overflow_error("SDK size sum overflow");
    return a+b;
}
bool is_sha(const char* text) {
    return text && std::strlen(text)==64 && std::all_of(text,text+64,[](char c){
        return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
bool within(const fs::path& path,const fs::path& root) {
    const auto relative=path.lexically_relative(root);
    return !relative.empty() && !relative.is_absolute() && *relative.begin()!="..";
}
fs::path checked_path(const char* text,const char* optional_root_environment) {
    if(!text || !*text || !fs::path(text).is_absolute()) throw std::invalid_argument("explicit absolute asset path required");
    const auto path=fs::canonical(text);
    if(!fs::is_regular_file(path)) throw std::invalid_argument("asset must be a regular file");
    if(const char* root=std::getenv(optional_root_environment);root && *root) {
        const auto canonical=fs::canonical(root);
        if(canonical!=fs::path(root) || !within(path,canonical))
            throw std::invalid_argument("asset realpath escapes declared project/public root");
    }
    return path;
}
std::string sha256(const fs::path& path) {
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    if(!context || EVP_DigestInit_ex(context.get(),EVP_sha256(),nullptr)!=1)
        throw std::runtime_error("SHA256 initialization failed");
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::invalid_argument("cannot open asset for identity verification");
    std::array<char,65536> block{};
    while(input) {
        input.read(block.data(),block.size());
        if(input.gcount() && EVP_DigestUpdate(context.get(),block.data(),static_cast<size_t>(input.gcount()))!=1)
            throw std::runtime_error("SHA256 update failed");
    }
    if(!input.eof()) throw std::runtime_error("asset identity read failed");
    std::array<unsigned char,EVP_MAX_MD_SIZE> digest{}; unsigned count=0;
    if(EVP_DigestFinal_ex(context.get(),digest.data(),&count)!=1 || count!=32)
        throw std::runtime_error("SHA256 finalization failed");
    constexpr char hex[]="0123456789abcdef"; std::string result;
    for(unsigned i=0;i<count;++i) { result+=hex[digest[i]>>4]; result+=hex[digest[i]&15]; }
    return result;
}
Json json_file(const fs::path& path,uint64_t maximum) {
    if(fs::file_size(path)>maximum) throw std::invalid_argument("metadata exceeds bounded parser input");
    std::vector<std::set<std::string>> keys;
    auto callback=[&](int depth,Json::parse_event_t event,Json& parsed) {
        if(depth>48) throw std::invalid_argument("metadata nesting exceeds bound");
        if(event==Json::parse_event_t::object_start) keys.emplace_back();
        else if(event==Json::parse_event_t::object_end) keys.pop_back();
        else if(event==Json::parse_event_t::key && (keys.empty() || !keys.back().insert(parsed.get<std::string>()).second))
            throw std::invalid_argument("duplicate metadata key");
        return true;
    };
    std::ifstream input(path); return Json::parse(input,callback);
}
void error_text(rw_nr_job_info& info,const char* text) noexcept {
    copy_text(info.error,sizeof(info.error),text);
}
rw_nr_job_info initial_info() {
    rw_nr_job_info result{}; result.struct_size=sizeof(result); result.abi_version=RW_ABI_VERSION;
    result.state=RW_NR_JOB_QUEUED; result.status=RW_NOT_READY; result.submit_host_ns=now_ns(); return result;
}
bool valid_timeout(int64_t milliseconds) { return milliseconds>=-1 && milliseconds<=86400000; }

struct HostBudget {
    const uint64_t limit;
    std::atomic<uint64_t> reserved{0};
    explicit HostBudget(uint64_t value):limit(value) {}
    bool reserve(uint64_t count) {
        auto current=reserved.load();
        do { if(count>limit || current>limit-count) return false; }
        while(!reserved.compare_exchange_weak(current,current+count));
        return true;
    }
};
struct Reservation {
    std::shared_ptr<HostBudget> owner;
    uint64_t bytes;
    Reservation(std::shared_ptr<HostBudget> budget,uint64_t count):owner(std::move(budget)),bytes(count) {
        if(!owner->reserve(count)) throw StatusError(RW_RESOURCE_EXHAUSTED,"SDK host input/output retention budget exhausted");
    }
    ~Reservation() { owner->reserved.fetch_sub(bytes); }
};
std::mutex& device_mutex() { static auto* value=new std::mutex; return *value; }
std::set<std::string>& device_claims() { static auto* value=new std::set<std::string>; return *value; }
struct DeviceClaim {
    std::string uuid;
    explicit DeviceClaim(std::string key):uuid(std::move(key)) {
        std::lock_guard lock(device_mutex());
        if(!device_claims().insert(uuid).second)
            throw StatusError(RW_RESOURCE_EXHAUSTED,"this process already owns an NR runtime on the physical device");
    }
    ~DeviceClaim() { std::lock_guard lock(device_mutex()); device_claims().erase(uuid); }
};

struct Runtime;
struct CudaInputLease {
    rw_nr_cuda_input input{};
    const float* color=nullptr;
    bool adopted=false;
    ~CudaInputLease() {if(adopted && input.release)input.release(input.release_context);}
};
struct HostColor {
    std::vector<float> values;
    std::shared_ptr<float> pinned;
    size_t pinned_elements=0;

    std::span<const float> view() const {
        if(pinned) return {pinned.get(),pinned_elements};
        return values;
    }
};
struct Session {
    uint64_t id=0,epoch=0,predecessor=RW_NR_NO_PREDECESSOR;
    uint32_t seed=0;
    bool accepting=true,pending=false;
    bool has_history=false;
    uint64_t history_frame=RW_NR_NO_PREDECESSOR;
    std::shared_ptr<const HostColor> history;
    std::shared_ptr<Reservation> history_reservation;
    Tensor gpu_history;
};
struct Result {
    std::shared_ptr<HostColor> color=std::make_shared<HostColor>();
    uint32_t width=0,height=0;
    uint64_t session=0,frame=0,epoch=0;
    std::shared_ptr<Reservation> reservation;
};
struct Job {
    std::shared_ptr<Session> session;
    std::vector<float> input;
    std::vector<float> previous;
    std::shared_ptr<const HostColor> previous_history;
    Tensor previous_gpu_history,next_gpu_history;
    std::shared_ptr<CudaInputLease> cuda_input;
    std::vector<float> motion;
    std::vector<float> motion_validity;
    bool has_previous=false;
    bool has_motion=false;
    bool has_motion_validity=false;
    bool use_session_history=false;
    std::shared_ptr<Reservation> history_reservation;
    std::shared_ptr<HostColor> history_copy;
    std::shared_ptr<Reservation> reservation;
    std::shared_ptr<Result> result;
    rw_nr_job_info info=initial_info();
    std::condition_variable changed;
    bool cancelled=false,done=false,device_stream_span_collected=false;
    uint64_t slot_id=UINT64_MAX;
    uint64_t input_h2d_bytes=0,producer_snapshot_bytes=0,resident_history_input_bytes=0;
    native_nr::SharedDeviceIdentity submitted_device_identity{};
    double device_begin_relative_ms=0,device_end_relative_ms=0;
};
struct TraceRecord {
    rw_nr_job_info info{};
    uint64_t slot_id=UINT64_MAX;
    double device_begin_relative_ms=0,device_end_relative_ms=0;
    bool device_stream_span_collected=false;
    uint64_t input_h2d_bytes=0,producer_snapshot_bytes=0,resident_history_input_bytes=0;
};
struct NodeSnapshot {
    std::array<native_nr::NodeMeasurement,native_nr::max_node_measurements> nodes{};
    native_nr::GraphExecutionSummary execution{};
    size_t count=0;
    uint64_t session_id=0,frame_id=0,epoch=0,expected_predecessor=RW_NR_NO_PREDECESSOR;
    rw_status_t job_status=RW_NOT_READY;
    bool available=false,graph_invoked=false,execution_identity_confirmed=false,bounds_valid=true,physical_completion_confirmed=false;
    void reset(const Job& job) noexcept {
        count=0;execution={};available=false;graph_invoked=false;execution_identity_confirmed=false;bounds_valid=true;
        session_id=job.info.session_id;frame_id=job.info.frame_id;epoch=job.info.epoch;
        expected_predecessor=job.info.expected_predecessor;job_status=RW_NOT_READY;
        physical_completion_confirmed=false;
    }
};
struct Slot {
    uint64_t id=0;
    int stream_priority=0;
    uintptr_t stream_identity=0,begin_event_identity=0,end_event_identity=0;
    uintptr_t main_workspace_identity=0,post_workspace_identity=0;
    std::unique_ptr<Context> context;
    std::unique_ptr<Graph> graph;
    std::thread worker;
    cudaEvent_t begin=nullptr,end=nullptr;
    std::shared_ptr<Job> active_job;
    NodeSnapshot node_snapshot;
    // Worker-only marker; queries read the terminal snapshot instead.
    bool graph_invoked=false;
    uint64_t graph_prior_execution_serial=0;
    bool finished=false,quarantined=false;
    cudaGraph_t captured_graph=nullptr;
    cudaGraphExec_t captured_execution=nullptr;
    std::vector<Tensor> captured_allocations;
    std::array<Tensor,4> captured_inputs;
    native_nr::LayerOutput captured_output;
    Tensor captured_delivery;
    std::array<uint32_t,7> captured_key{};
    std::atomic<uint64_t> capture_count{0},replay_count{0},captured_node_count{0};
    std::atomic<uint64_t> captured_allocation_nodes{0},captured_free_nodes{0};
    std::vector<std::array<cudaEvent_t,2>> captured_profile_events;
    std::vector<float> captured_profile_spans;
    uint64_t profile_frame_id=0,profile_epoch=0,profile_execution_serial=0;
    // One result can remain the current temporal history while the next is
    // copied. A consumer can retain either allocation independently.
    std::array<std::shared_ptr<float>,2> readback_pool;
    std::atomic<uint64_t> readback_pinned_bytes{0},readback_pinned_jobs{0};
    std::shared_ptr<float> acquire_readback(size_t elements,const std::shared_ptr<HostBudget>& host) {
        for(auto& entry:readback_pool) if(entry && entry.use_count()==1) return entry;
        for(auto& entry:readback_pool) {
            if(entry) continue;
            const auto bytes=multiply(elements,sizeof(float));
            std::shared_ptr<Reservation> charge;
            try { charge=std::make_shared<Reservation>(host,bytes); }
            catch(const StatusError& error) {
                if(error.status==RW_RESOURCE_EXHAUSTED) return {};
                throw;
            }
            float* destination=nullptr;
            const auto status=cudaMallocHost(reinterpret_cast<void**>(&destination),bytes);
            if(status==cudaErrorMemoryAllocation) {
                (void)cudaGetLastError();
                return {};
            }
            native_nr::cuda_check(status,"allocate NR pinned readback");
            entry=std::shared_ptr<float>(destination,[hold=std::move(charge)](float* pointer) {
                (void)cudaFreeHost(pointer);
            });
            readback_pinned_bytes+=bytes;
            return entry;
        }
        return {};
    }
    bool retire_capture() noexcept {
        if(!captured_graph && !captured_execution && captured_allocations.empty() &&
           std::all_of(captured_inputs.begin(),captured_inputs.end(),[](const Tensor& value){return value.empty();}))return true;
        try {context->synchronize();} catch(...){return false;}
        if(captured_execution) {
            if(cudaGraphExecDestroy(captured_execution)!=cudaSuccess)return false;
            captured_execution=nullptr;
        }
        if(captured_graph) {
            if(cudaGraphDestroy(captured_graph)!=cudaSuccess)return false;
            captured_graph=nullptr;
        }
        captured_output={};captured_delivery={};captured_inputs={};captured_allocations.clear();
        return true;
    }
    bool retire_events() noexcept {
        if(!begin && !end)return true;
        try {if(!context)return false;context->synchronize();}
        catch(...){return false;}
        if(!retire_capture())return false;
        for(auto& pair:captured_profile_events)for(auto& event:pair) {
            if(event && cudaEventDestroy(event)!=cudaSuccess)return false;
            event=nullptr;
        }
        if(begin) {if(cudaEventDestroy(begin)!=cudaSuccess)return false;begin=nullptr;}
        if(end) {if(cudaEventDestroy(end)!=cudaSuccess)return false;end=nullptr;}
        readback_pool={};
        return true;
    }
    ~Slot() {
        if(worker.joinable())worker.join();
        // Runtime checks this while its no-allocation group-retention owner
        // still exists. The fallback never drops a possibly referenced graph
        // or Context merely because a destructor cannot report an error.
        if(!retire_events()){(void)graph.release();(void)context.release();return;}
        // Graphs/pins are private to a slot. Context outlives the graph and
        // proves stream drain before retiring its shared model binding.
        graph.reset();context.reset();
    }
};
struct FailureRetention {
    std::unique_ptr<DeviceClaim> claim;
    std::shared_ptr<native_nr::SharedDeviceOwner> device;
    native_nr::DeviceBudgetReservation caller_reserve;
    native_nr::RegisteredModelOwner registered_model;
    std::shared_ptr<native_nr::DeviceWeightPool> weights;
    std::unique_ptr<Weights> legacy_weights;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<std::shared_ptr<Job>> quarantine;
    std::shared_ptr<Reservation> fixed_host;
    cudaEvent_t timeline=nullptr;
};
struct Counters {
    uint64_t arrived=0,admitted=0,rejected=0,valid_outputs=0,failed=0,cancelled=0,stale=0;
    uint64_t wait_timeouts=0,running=0,inflight=0,quarantined=0;
};
struct Runtime:std::enable_shared_from_this<Runtime> {
    std::mutex mutex,join_mutex;
    std::condition_variable available,finished;
    std::deque<std::shared_ptr<Job>> queue;
    std::vector<std::shared_ptr<Job>> quarantine;
    std::vector<TraceRecord> traces;
    std::vector<std::weak_ptr<Session>> sessions;
    Counters counters;
    bool input_phase_detail=false;
    bool gpu_history=false;
    bool sdr_temporal_head=false;
    uint64_t input_phase_samples=0,input_minor_faults=0,input_fault_samples=0;
    std::array<uint64_t,4> input_phase_totals{},input_phase_max{};
    std::shared_ptr<HostBudget> host;
    std::shared_ptr<Reservation> fixed_host;
    std::unique_ptr<DeviceClaim> claim;
    std::shared_ptr<native_nr::SharedDeviceOwner> shared_device;
    native_nr::DeviceBudgetReservation caller_reserve;
    native_nr::RegisteredModelOwner registered_model;
    std::shared_ptr<native_nr::DeviceWeightPool> shared_weights;
    std::unique_ptr<Weights> legacy_weights;
    std::vector<std::unique_ptr<Slot>> slots;
    SourcePlan plan_metadata;
    native_nr::PreparationStatistics final_preparation{};
    cudaEvent_t timeline_origin=nullptr;
    Json identity;
    uint64_t next_session=1;
    uint32_t max_sessions=0,max_inflight=0,trace_capacity=0;
    size_t trace_count=0,trace_cursor=0;
    uint32_t finished_workers=0;
    bool collect_node_metrics=false;
    bool closing=false,failed=false;
    bool resources_finalizing=false,resources_retired=false,cleanup_failed=false;
    std::array<char,1024> failure{};
    std::shared_ptr<Runtime> self_quarantine;
    // Allocated before any CUDA owner/handle. Failure cleanup moves only into
    // this already-existing empty owner, then intentionally retains it.
    std::unique_ptr<FailureRetention> failure_retention=std::make_unique<FailureRetention>();

    ~Runtime() {
        {std::lock_guard lock(mutex);closing=true;available.notify_all();}
        for(auto& slot:slots)if(slot->worker.joinable())slot->worker.join();
        bool safe=resources_retired;
        if(!safe && retire_events()) {
            for(auto& slot:slots)slot->graph.reset();
            safe=retire_gpu_group();
        }
        if(!safe) {
            if(shared_device)shared_device->quarantine();
            auto& hold=*failure_retention;
            hold.claim=std::move(claim);hold.device=std::move(shared_device);hold.caller_reserve=std::move(caller_reserve);
            hold.registered_model=std::move(registered_model);hold.weights=std::move(shared_weights);
            hold.legacy_weights=std::move(legacy_weights);hold.slots=std::move(slots);hold.quarantine=std::move(quarantine);
            hold.fixed_host=std::move(fixed_host);hold.timeline=timeline_origin;timeline_origin=nullptr;
            (void)failure_retention.release();
        }
    }
    bool retire_events() noexcept {
        for(auto& slot:slots)if(!slot->retire_events())return false;
        if(timeline_origin) {
            try {if(slots.empty() || !slots.front()->context)return false;slots.front()->context->synchronize();}
            catch(...){return false;}
            if(cudaEventDestroy(timeline_origin)!=cudaSuccess)return false;timeline_origin=nullptr;
        }
        return true;
    }
    bool retire_gpu_group() noexcept {
        try {
            {std::lock_guard lock(mutex);
                for(auto& weak:sessions)if(auto session=weak.lock())session->gpu_history={};}
            if(legacy_weights)legacy_weights->clear_device_cache_for_retirement();
            for(auto& slot:slots)if(slot->context && !slot->context->retire_owned_resources())return false;
            if(shared_weights && !shared_weights->retire_after_slots_drained())return false;
            caller_reserve=native_nr::DeviceBudgetReservation{};
            if(shared_device && !shared_device->retire_owned_context())return false;
            return true;
        } catch(...) {return false;}
    }
    Weights& weights() const {return shared_weights?shared_weights->weights():*legacy_weights;}
    const SourcePlan& plan() const {return plan_metadata;}
    bool domain_failed() const noexcept {return shared_device && shared_device->quarantined();}
    void trace(const rw_nr_job_info& info,uint64_t slot=UINT64_MAX,double begin_ms=0,double end_ms=0,bool span_collected=false,
               uint64_t input_h2d=0,uint64_t producer_snapshot=0,uint64_t resident_history=0) noexcept {
        traces[trace_cursor]={info,slot,begin_ms,end_ms,span_collected,input_h2d,producer_snapshot,resident_history}; trace_cursor=(trace_cursor+1)%trace_capacity;
        trace_count=std::min(trace_count+1,static_cast<size_t>(trace_capacity));
    }
    void record_rejection(rw_nr_job_info info,rw_status_t status,const char* message) {
        ++counters.rejected; info.status=status; info.state=RW_NR_JOB_FAILED;
        info.complete_host_ns=now_ns(); error_text(info,message); trace(info);
    }
    void complete(const std::shared_ptr<Job>& job,rw_status_t status,const char* message,
                  bool physical,bool gpu_submitted) noexcept {
        job->info.status=status;
        job->info.physical_completion_confirmed=physical;
        job->info.gpu_submitted=gpu_submitted;
        if(!gpu_submitted || physical) {
            if(!job->info.complete_host_ns) job->info.complete_host_ns=now_ns();
        }
        if(status==RW_OK) { job->info.state=RW_NR_JOB_COMPLETED; ++counters.valid_outputs; }
        else if(status==RW_CANCELLED) { job->info.state=RW_NR_JOB_CANCELLED; ++counters.cancelled; }
        else if(status==RW_STALE) { job->info.state=RW_NR_JOB_STALE; ++counters.stale; }
        else { job->info.state=RW_NR_JOB_FAILED; ++counters.failed; }
        error_text(job->info,message);
        job->done=true;
        if(!gpu_submitted || physical) {
            job->session->pending=false;
            job->previous_gpu_history={};job->next_gpu_history={};
            --counters.inflight;
        }
        trace(job->info,job->slot_id,job->device_begin_relative_ms,job->device_end_relative_ms,
              job->device_stream_span_collected,job->input_h2d_bytes,job->producer_snapshot_bytes,job->resident_history_input_bytes); job->changed.notify_all();
    }
    // Called only by this slot's worker while holding Runtime::mutex, after
    // terminal status is assigned. Fixed POD copies cannot allocate or touch
    // CUDA; no query reads Graph's worker-owned observation array.
    void publish_node_snapshot(Slot& slot,const Job& job) noexcept {
        auto& snapshot=slot.node_snapshot;
        snapshot.job_status=job.info.status;
        snapshot.physical_completion_confirmed=bool(job.info.physical_completion_confirmed);
        snapshot.graph_invoked=slot.graph_invoked;
        if(slot.graph_invoked && slot.graph) {
            const auto& execution=slot.graph->execution_summary();
            snapshot.execution_identity_confirmed=execution.execution_serial!=slot.graph_prior_execution_serial && execution.graph_entered;
            if(snapshot.execution_identity_confirmed)snapshot.execution=execution;
            else snapshot.bounds_valid=false;
            if(collect_node_metrics && snapshot.execution_identity_confirmed) {
                const auto measurements=slot.graph->measurements();
                snapshot.bounds_valid=measurements.size()<=snapshot.nodes.size();
                for(const auto& measurement:measurements) {
                    if(measurement.index<0 || static_cast<size_t>(measurement.index)>=plan_metadata.layers.size())
                        snapshot.bounds_valid=false;
                }
                if(snapshot.bounds_valid) {
                    std::copy(measurements.begin(),measurements.end(),snapshot.nodes.begin());
                    snapshot.count=measurements.size();
                }
            }
        }
        snapshot.available=true;
    }
    void run(Slot&) noexcept;
    void run_loop(Slot&);
    void emergency_stop(Slot&,const char*) noexcept;
    void start_workers() {
        const auto owner=shared_from_this();
        try {for(auto& slot:slots){auto* current=slot.get();current->worker=std::thread([owner,current]{owner->run(*current);});}}
        catch(...) {
            {std::lock_guard lock(mutex);closing=true;failed=true;available.notify_all();}
            for(auto& slot:slots)if(slot->worker.joinable())slot->worker.join();
            throw;
        }
    }
    rw_status_t close(int64_t timeout) {
        if(!valid_timeout(timeout)) throw std::invalid_argument("timeout must be -1 or 0..86400000 milliseconds");
        {
            std::unique_lock lock(mutex);
            closing=true; available.notify_all();
            const auto drained=[&]{return finished_workers==slots.size() && counters.inflight==counters.quarantined;};
            if(timeout<0) finished.wait(lock,drained);
            else if(!finished.wait_for(lock,std::chrono::milliseconds(timeout),drained))
                return RW_NOT_READY;
        }
        std::lock_guard join_lock(join_mutex);
        for(auto& slot:slots)if(slot->worker.joinable())slot->worker.join();
        if(!retire_events()) {
            if(shared_device)shared_device->quarantine();
            std::lock_guard lock(mutex);failed=true;self_quarantine=shared_from_this();
            if(failure[0]=='\0') copy_text(failure.data(),failure.size(),"slot event cleanup could not be physically confirmed; whole execution group retained");
            return RW_INTERNAL_ERROR;
        }
        std::lock_guard lock(mutex);
        return quarantine.empty() ? RW_OK : RW_INTERNAL_ERROR;
    }
    rw_status_t retire_resources() {
        std::lock_guard join_lock(join_mutex);
        {
            std::lock_guard lock(mutex);
            if(resources_retired)return RW_OK;
            if(cleanup_failed)return RW_INTERNAL_ERROR;
            if(finished_workers!=slots.size() || counters.running || counters.inflight!=counters.quarantined || counters.quarantined)
                return RW_NOT_READY;
            final_preparation=shared_weights?shared_weights->preparation_statistics():legacy_weights->preparation_statistics();
            resources_finalizing=true;
            for(auto& slot:slots)slot->graph.reset();
        }
        // All worker-local activation/read owners are gone. Clearing legacy W
        // first exposes any failed payload retirement to the still-live slot
        // Context, rather than hiding it behind a later destructor.
        const bool safe=retire_gpu_group();
        {
            std::lock_guard lock(mutex);resources_finalizing=false;resources_retired=safe;cleanup_failed=!safe;
            if(safe) {
                // CPU-only child handles may outlive the user Runtime handle.
                // All physical owners retired successfully, so those queries
                // must not keep this device's SDK admission claim occupied.
                claim.reset();
            } else {
                if(shared_device)shared_device->quarantine();
                failed=true;self_quarantine=shared_from_this();
                if(failure[0]=='\0') copy_text(failure.data(),failure.size(),"checked Context/pool/device retirement failed; execution group and claim retained");
            }
        }
        return safe?RW_OK:RW_INTERNAL_ERROR;
    }
};
std::vector<float> pad_hwc_to_chw(std::span<const float> input,const SourcePlan& plan) {
    const size_t padded=static_cast<size_t>(plan.padded_width)*plan.padded_height;
    std::vector<float> output(padded*3);
    auto mirror=[](int index,int extent) {
        return std::clamp(index<extent ? index : 2*extent-index-2,0,extent-1);
    };
    for(int y=0;y<plan.padded_height;++y) for(int x=0;x<plan.padded_width;++x) {
        const int iy=mirror(y,plan.valid_height),ix=mirror(x,plan.valid_width);
        const size_t source=(static_cast<size_t>(iy)*plan.valid_width+ix)*3;
        const size_t target=static_cast<size_t>(y)*plan.padded_width+x;
        for(size_t channel=0;channel<3;++channel) output[channel*padded+target]=input[source+channel];
    }
    return output;
}
std::vector<float> pad_hwc2_to_chw(const std::vector<float>& input,const SourcePlan& plan) {
    const size_t padded=static_cast<size_t>(plan.padded_width)*plan.padded_height;
    std::vector<float> output(padded*2);
    auto mirror=[](int index,int extent) {
        return std::clamp(index<extent ? index : 2*extent-index-2,0,extent-1);
    };
    for(int y=0;y<plan.padded_height;++y) for(int x=0;x<plan.padded_width;++x) {
        const int iy=mirror(y,plan.valid_height),ix=mirror(x,plan.valid_width);
        const size_t source=(static_cast<size_t>(iy)*plan.valid_width+ix)*2;
        const size_t target=static_cast<size_t>(y)*plan.padded_width+x;
        output[target]=input[source]; output[padded+target]=input[source+1];
    }
    return output;
}
std::vector<float> pad_hwc1_to_chw(const std::vector<float>& input,const SourcePlan& plan) {
    const size_t padded=static_cast<size_t>(plan.padded_width)*plan.padded_height;
    std::vector<float> output(padded);
    auto mirror=[](int index,int extent) {
        return std::clamp(index<extent ? index : 2*extent-index-2,0,extent-1);
    };
    for(int y=0;y<plan.padded_height;++y) for(int x=0;x<plan.padded_width;++x) {
        const int iy=mirror(y,plan.valid_height),ix=mirror(x,plan.valid_width);
        output[static_cast<size_t>(y)*plan.padded_width+x]=input[static_cast<size_t>(iy)*plan.valid_width+ix];
    }
    return output;
}
void Runtime::run(Slot& slot) noexcept {
    try { run_loop(slot); }
    catch(const std::exception& error) { emergency_stop(slot,error.what()); }
    catch(...) { emergency_stop(slot,"unexpected worker failure"); }
}
void Runtime::emergency_stop(Slot& slot,const char* message) noexcept {
    auto* context=slot.context.get();
    auto& active_job=slot.active_job;
    bool physical=false;
    try { context->synchronize(); if(context->has_active_weight_read())context->finish_weight_read(); physical=true; } catch(...) {}
    if(shared_device && !physical)shared_device->quarantine();
    try {
        std::lock_guard lock(mutex);
        failed=true; closing=true; if(failure[0]=='\0') copy_text(failure.data(),failure.size(),message);
        if(!physical){slot.quarantined=true;self_quarantine=weak_from_this().lock();}
        if(active_job && !active_job->done) {
            const bool possibly_submitted=active_job->info.state==RW_NR_JOB_RUNNING;
            if(counters.running)--counters.running;
            if(possibly_submitted && !physical) {
                ++counters.quarantined; quarantine.push_back(active_job);slot.quarantined=true;
                self_quarantine=weak_from_this().lock();
            }
            complete(active_job,RW_INTERNAL_ERROR,failure.data(),physical,possibly_submitted);
            publish_node_snapshot(slot,*active_job);
            if(!possibly_submitted || physical) {
                std::vector<float>().swap(active_job->input);
                std::vector<float>().swap(active_job->previous);
                active_job->previous_history.reset();
                std::vector<float>().swap(active_job->motion);
                std::vector<float>().swap(active_job->motion_validity);
            }
        }
        active_job.reset();
        while(!queue.empty()) {
            auto pending=queue.front(); queue.pop_front();
            complete(pending,RW_INTERNAL_ERROR,failure.data(),false,false);
            std::vector<float>().swap(pending->input);
            std::vector<float>().swap(pending->previous);
            pending->previous_history.reset();
            std::vector<float>().swap(pending->motion);
            std::vector<float>().swap(pending->motion_validity);
        }
        if(!slot.finished){slot.finished=true;++finished_workers;} finished.notify_all();available.notify_all();
    } catch(...) {
        // Error reporting must not terminate the process or destroy unconfirmed
        // owners. Mutex/system corruption is not claimed recoverable; retain
        // the runtime and wake bounded waiters without allocating diagnostics.
        self_quarantine=weak_from_this().lock();
        finished.notify_all();
    }
}
void Runtime::run_loop(Slot& slot) {
    auto* context=slot.context.get();
    auto* graph=slot.graph.get();
    auto& active_job=slot.active_job;
    const auto slot_begin=slot.begin,slot_end=slot.end;
    auto& bound_weights=weights();
    for(;;) {
        std::shared_ptr<Job> job;
        bool skip=false;
        {
            std::unique_lock lock(mutex);
            available.wait(lock,[&]{return closing || !queue.empty();});
            if(queue.empty() && closing) break;
            job=queue.front(); queue.pop_front();
            active_job=job;job->slot_id=slot.id;
            slot.graph_invoked=false;slot.node_snapshot.reset(*job);
            if(domain_failed()){
                failed=true;closing=true;
                // Preserve the first CUDA/retirement error. The domain
                // quarantine message is a secondary consequence.
                if(failure[0]=='\0') copy_text(failure.data(),failure.size(),"shared CUDA device domain is quarantined");
            }
            if(failed || job->cancelled || job->session->epoch!=job->info.epoch ||
               job->session->predecessor!=job->info.expected_predecessor) {
                const auto status=failed ? RW_INTERNAL_ERROR : job->cancelled ? RW_CANCELLED : RW_STALE;
                complete(job,status,failed ? failure.data() : "not submitted to GPU",false,false);
                publish_node_snapshot(slot,*job);
                std::vector<float>().swap(job->input);
                std::vector<float>().swap(job->previous);
                job->previous_history.reset();
                std::vector<float>().swap(job->motion);
                std::vector<float>().swap(job->motion_validity);
                active_job.reset();
                skip=true;
            } else {
                ++counters.running; job->info.state=RW_NR_JOB_RUNNING;
            }
        }
        if(skip) {job->cuda_input.reset();continue;}
        bool submitted=false,physical=false;
        rw_status_t status=RW_OK;
        std::array<char,1024> error{};
        std::shared_ptr<Result> output;
        try {
            const auto& plan=graph->plan();
            const auto enabled=[](const char* name) {
                const char* value=std::getenv(name);return value && std::strcmp(value,"1")==0;
            };
            const bool gpu_input=sdr_temporal_head || enabled("RW_NR_GPU_INPUT_LAYOUT");
            const std::span<const float> previous_source=job->previous_history ? job->previous_history->view()
                : std::span<const float>(job->previous);
            std::vector<float> padded;
            if(!gpu_input)padded=pad_hwc_to_chw(job->input,plan);
            std::vector<float> previous_padded;
            if(job->has_previous && !gpu_input) previous_padded=pad_hwc_to_chw(previous_source,plan);
            std::vector<float> motion_padded;
            if(job->has_motion && !gpu_input) motion_padded=pad_hwc2_to_chw(job->motion,plan);
            std::vector<float> motion_validity_padded;
            if(job->has_motion_validity && !gpu_input) motion_validity_padded=pad_hwc1_to_chw(job->motion_validity,plan);
            if(shared_weights) {
                job->submitted_device_identity=shared_device->identity();
                context->begin_weight_read({job->info.session_id,job->info.epoch,job->info.expected_predecessor});
            }
            // Set submitted before an upload can enqueue asynchronous accesses.
            submitted=true;
            // Upload all frame inputs on the same stream and drain once.  The
            // old path called Context::upload independently, forcing a full
            // cudaStreamSynchronize after every colour/previous/motion input.
            // Keeping the host spans alive until upload_batch returns retains
            // the existing ownership and failure semantics while removing up
            // to three stream drains from every frame.
            std::vector<native_nr::Context::UploadRequest> uploads;
            uploads.reserve(1 + (job->has_previous ? 1 : 0) +
                            (job->has_motion ? 1 : 0) + (job->has_motion_validity ? 1 : 0));
            const auto upload_plane=[&](int channels,std::span<const float> source,std::span<const float> packed) {
                uploads.push_back({gpu_input ? std::vector<int64_t>{1,plan.valid_height,plan.valid_width,channels}
                    : std::vector<int64_t>{1,channels,plan.padded_height,plan.padded_width},
                    gpu_input ? source : packed});
            };
            const bool resident_previous=gpu_input && job->has_previous && !job->previous_gpu_history.empty();
            if(!job->cuda_input) {
                upload_plane(3,job->input,padded);
                if(job->has_previous && !resident_previous)upload_plane(3,previous_source,previous_padded);
                if(job->has_motion)upload_plane(2,job->motion,motion_padded);
                if(job->has_motion_validity)upload_plane(1,job->motion_validity,motion_validity_padded);
            } else if(job->has_previous && !resident_previous)upload_plane(3,previous_source,previous_padded);
            auto uploaded=uploads.empty()?std::vector<Tensor>{}:
                gpu_input ? context->upload_batch_borrowed(uploads) : context->upload_batch(uploads);
            for(const auto& request:uploads)job->input_h2d_bytes+=request.values.size_bytes();
            size_t upload_index=0;
            Tensor image,previous_image,motion_image,motion_validity_image;
            if(job->cuda_input) {
                const auto& producer=*job->cuda_input;
                if(producer.input.ready_event)
                    native_nr::cuda_check(cudaStreamWaitEvent(context->stream(),static_cast<cudaEvent_t>(producer.input.ready_event),0),
                        "wait for CUDA producer input");
                const auto import_plane=[&](const float* source,int channels) {
                    auto owned=context->empty({1,plan.valid_height,plan.valid_width,channels});
                    native_nr::cuda_check(cudaMemcpyAsync(owned.data(),source,owned.numel()*sizeof(float),
                        cudaMemcpyDeviceToDevice,context->stream()),"snapshot CUDA producer plane");
                    job->producer_snapshot_bytes+=owned.numel()*sizeof(float);
                    return owned;
                };
                image=import_plane(producer.color,3);
                if(job->has_motion)motion_image=import_plane(producer.input.motion_hwc2,2);
                if(job->has_motion_validity)motion_validity_image=import_plane(producer.input.motion_validity,1);
            } else image=std::move(uploaded[upload_index++]);
            if(resident_previous) {
                job->resident_history_input_bytes=job->previous_gpu_history.numel()*sizeof(float);
                // The graph's fixed input is mutable. Never let its first
                // capture alias committed session history: a rejected frame
                // can leave that history live while another session uses it.
                previous_image=context->empty(job->previous_gpu_history.shape);
                native_nr::cuda_check(cudaMemcpyAsync(previous_image.data(),job->previous_gpu_history.data(),
                    previous_image.numel()*sizeof(float),cudaMemcpyDeviceToDevice,context->stream()),"snapshot same-device session history into slot");
            } else if(job->has_previous)previous_image=std::move(uploaded[upload_index++]);
            if(!job->cuda_input) {
                if(job->has_motion)motion_image=std::move(uploaded[upload_index++]);
                if(job->has_motion_validity)motion_validity_image=std::move(uploaded[upload_index++]);
            }
            std::vector<float>().swap(padded);
            std::vector<float>().swap(previous_padded);
            std::vector<float>().swap(motion_padded);
            std::vector<float>().swap(motion_validity_padded);
            native_nr::RunOptions options; options.controls.seed=job->session->seed;
            if(const char* feature_mode=std::getenv("RW_NR_EXPERIMENTAL_FEATURE_MODE")) {
                if(std::strcmp(feature_mode,"rgb") == 0) options.controls.synthetic_noise=false;
                else if(std::strcmp(feature_mode,"noise") != 0)
                    throw std::invalid_argument("RW_NR_EXPERIMENTAL_FEATURE_MODE must be noise or rgb");
            }
            options.controls.previous_color=job->has_previous;
            options.controls.motion_warp=job->has_motion;
            options.controls.motion_validity=job->has_motion_validity;
            options.sdr_temporal_head=sdr_temporal_head && job->use_session_history;
            options.previous_image=job->has_previous ? &previous_image : nullptr;
            options.motion_image=job->has_motion ? &motion_image : nullptr;
            options.motion_validity_image=job->has_motion_validity ? &motion_validity_image : nullptr;
            // Experimental diagnostic only: keep the frozen default intensity=1.0
            // unless an explicit process-local probe override is requested.
            if(const char* override_text=std::getenv("RW_NR_EXPERIMENTAL_INTENSITY")) {
                char* end=nullptr;
                const float value=std::strtof(override_text,&end);
                if(end==override_text || *end!='\0' || !std::isfinite(value) || value<0.0f || value>2.0f)
                    throw std::invalid_argument("RW_NR_EXPERIMENTAL_INTENSITY must be a finite value in [0,2]");
                options.controls.intensity=value;
            }
            options.collect_node_metrics=collect_node_metrics;
            options.defer_completion=true;
            const bool gpu_layout=sdr_temporal_head || enabled("RW_NR_GPU_OUTPUT_LAYOUT");
            const bool cuda_replay=enabled("RW_NR_CUDA_GRAPH") && !collect_node_metrics;
            const auto execute_graph=[&](const Tensor& source,native_nr::RunOptions run_options) {
                if(context->capture_active() && !slot.captured_profile_events.empty())
                    run_options.node_boundary=[&](int index,bool begin) {
                        native_nr::cuda_check(cudaEventRecordWithFlags(slot.captured_profile_events.at(index)[begin?0:1],context->stream(),cudaEventRecordExternal),
                            "record captured semantic node boundary");
                    };
                Tensor current=source,previous,motion,validity;
                if(gpu_input) {
                    current=native_nr::input_nchw(*context,source,plan.padded_height,plan.padded_width);
                    if(run_options.previous_image) {
                        previous=native_nr::input_nchw(*context,*run_options.previous_image,plan.padded_height,plan.padded_width);
                        run_options.previous_image=&previous;
                    }
                    if(run_options.motion_image) {
                        motion=native_nr::input_nchw(*context,*run_options.motion_image,plan.padded_height,plan.padded_width);
                        run_options.motion_image=&motion;
                    }
                    if(run_options.motion_validity_image) {
                        validity=native_nr::input_nchw(*context,*run_options.motion_validity_image,plan.padded_height,plan.padded_width);
                        run_options.motion_validity_image=&validity;
                    }
                }
                return graph->execute(*context,bound_weights,current,run_options,{});
            };
            {
                std::lock_guard lock(mutex);
                job->info.gpu_submitted=1; job->info.device_execution_host_ns=now_ns();
            }
            slot.graph_prior_execution_serial=graph->execution_summary().execution_serial;
            slot.graph_invoked=true;
            if(cuda_replay) {
                const auto& c=options.controls;
                const std::array<uint32_t,7> key={std::bit_cast<uint32_t>(c.intensity),
                    std::bit_cast<uint32_t>(c.local_tone),std::bit_cast<uint32_t>(c.local_structure),
                    std::bit_cast<uint32_t>(c.skin_structure),static_cast<uint32_t>(c.style),c.seed,
                    static_cast<uint32_t>(c.auto_mask | (c.synthetic_noise<<1) | (c.previous_color<<2) |
                        (c.motion_warp<<3) | (c.motion_validity<<4) | (gpu_layout<<5) | (gpu_input<<6) |
                        (options.sdr_temporal_head<<7))};
                if(slot.captured_execution && key!=slot.captured_key && !slot.retire_capture())
                    throw std::runtime_error("retire previous NR captured profile failed");
                std::array<Tensor*,4> inputs={&image,&previous_image,&motion_image,&motion_validity_image};
                for(size_t i=0;i<inputs.size();++i) {
                    if(inputs[i]->empty())continue;
                    if(slot.captured_execution) {
                        native_nr::cuda_check(cudaMemcpyAsync(slot.captured_inputs[i].data(),inputs[i]->data(),
                            inputs[i]->numel()*sizeof(float),cudaMemcpyDeviceToDevice,context->stream()),
                            "update NR captured input");
                    } else slot.captured_inputs[i]=*inputs[i];
                }
                options.previous_image=job->has_previous ? &slot.captured_inputs[1] : nullptr;
                options.motion_image=job->has_motion ? &slot.captured_inputs[2] : nullptr;
                options.motion_validity_image=job->has_motion_validity ? &slot.captured_inputs[3] : nullptr;
                if(!slot.captured_execution) {
                    // Populate immutable parameter caches and the exact-size
                    // activation cache before recording allocation-free work.
                    {
                        auto warm=execute_graph(slot.captured_inputs[0],options);
                        Tensor delivery;
                        if(gpu_layout)delivery=native_nr::output_hwc(*context,warm.main,plan.valid_height,plan.valid_width);
                        context->synchronize();
                    }
                    context->synchronize();
                    context->begin_capture(slot.captured_allocations);
                    try {
                        slot.captured_output=execute_graph(slot.captured_inputs[0],options);
                        if(gpu_layout)slot.captured_delivery=native_nr::output_hwc(*context,
                            slot.captured_output.main,plan.valid_height,plan.valid_width);
                        slot.captured_graph=context->end_capture();
                    } catch(...) {
                        if(context->capture_active()) {
                            try {slot.captured_graph=context->end_capture();}catch(...){}
                        }
                        throw;
                    }
                    const auto instantiate_flags=context->stream_ordered_memory()
                        ? cudaGraphInstantiateFlagAutoFreeOnLaunch : 0;
                    native_nr::cuda_check(cudaGraphInstantiateWithFlags(&slot.captured_execution,slot.captured_graph,instantiate_flags),
                        "instantiate native NR CUDA graph");
                    size_t nodes=0;
                    native_nr::cuda_check(cudaGraphGetNodes(slot.captured_graph,nullptr,&nodes),"count captured NR nodes");
                    std::vector<cudaGraphNode_t> graph_nodes(nodes);
                    native_nr::cuda_check(cudaGraphGetNodes(slot.captured_graph,graph_nodes.data(),&nodes),"read captured NR nodes");
                    uint64_t allocations=0,frees=0;
                    for(const auto node:graph_nodes) {
                        cudaGraphNodeType type;
                        native_nr::cuda_check(cudaGraphNodeGetType(node,&type),"read captured NR node type");
                        allocations+=type==cudaGraphNodeTypeMemAlloc;frees+=type==cudaGraphNodeTypeMemFree;
                    }
                    slot.captured_allocation_nodes=allocations;slot.captured_free_nodes=frees;
                    slot.captured_node_count=nodes;slot.captured_key=key;++slot.capture_count;
                }
            }
            native_nr::cuda_check(cudaEventRecord(slot_begin,context->stream()),"record async NR slot begin");
            native_nr::LayerOutput result;
            Tensor delivery;
            if(cuda_replay) {
                graph->note_replay();
                native_nr::cuda_check(cudaGraphLaunch(slot.captured_execution,context->stream()),"replay native NR CUDA graph");
                ++slot.replay_count;
                result=slot.captured_output;delivery=slot.captured_delivery;
            } else {
                result=execute_graph(image,options);
                if(gpu_layout)delivery=native_nr::output_hwc(*context,result.main,plan.valid_height,plan.valid_width);
            }
            // download() performs the required stream drain for the output
            // copy. Avoid synchronizing once before and once inside that copy.
            float elapsed=0;
            if(job->cuda_input)native_nr::validate_hwc_inputs(*context,image,
                job->has_motion?&motion_image:nullptr,job->has_motion_validity?&motion_validity_image:nullptr,delivery);
            if(options.sdr_temporal_head) {
                if(cuda_replay) {
                    job->next_gpu_history=context->empty(result.next_history.shape);
                    native_nr::cuda_check(cudaMemcpyAsync(job->next_gpu_history.data(),result.next_history.data(),
                        result.next_history.numel()*sizeof(float),cudaMemcpyDeviceToDevice,context->stream()),
                        "snapshot NR temporal history from replay workspace");
                } else job->next_gpu_history=std::move(result.next_history);
            } else if(job->use_session_history && gpu_layout && (gpu_history || job->cuda_input)) {
                job->next_gpu_history=context->empty({1,plan.valid_height,plan.valid_width,3});
                native_nr::cuda_check(cudaMemcpyAsync(job->next_gpu_history.data(),delivery.data(),
                    job->next_gpu_history.numel()*sizeof(float),cudaMemcpyDeviceToDevice,context->stream()),"retain next GPU RGB history");
            }
            native_nr::cuda_check(cudaEventRecord(slot_end,context->stream()),"record async NR slot end");
            const auto& readback=gpu_layout ? delivery : result.main;
            auto pinned=gpu_layout && enabled("RW_NR_PINNED_READBACK")
                ? slot.acquire_readback(readback.numel(),host) : std::shared_ptr<float>{};
            std::vector<float> raw;
            if(pinned) {
                context->download_into_pinned(readback,pinned);
                ++slot.readback_pinned_jobs;
            } else raw=context->download(readback);
            physical=true;
            native_nr::cuda_check(cudaEventElapsedTime(&elapsed,slot_begin,slot_end),"measure async NR slot span");
            {
                std::lock_guard lock(mutex);job->info.device_stream_span_ms=elapsed;
                job->device_stream_span_collected=true;
            }
            if(cuda_replay && !slot.captured_profile_events.empty()) {
                std::vector<float> spans(slot.captured_profile_events.size());
                for(size_t index=0;index<spans.size();++index)
                    native_nr::cuda_check(cudaEventElapsedTime(&spans[index],slot.captured_profile_events[index][0],slot.captured_profile_events[index][1]),
                        "read captured semantic node span after frame completion");
                std::lock_guard lock(mutex);
                slot.captured_profile_spans=std::move(spans);slot.profile_frame_id=job->info.frame_id;
                slot.profile_epoch=job->info.epoch;slot.profile_execution_serial=graph->execution_summary().execution_serial;
            }
            if(timeline_origin) {
                float start=0,end=0;
                native_nr::cuda_check(cudaEventElapsedTime(&start,timeline_origin,slot_begin),"measure slot begin relative to fixed origin");
                native_nr::cuda_check(cudaEventElapsedTime(&end,timeline_origin,slot_end),"measure slot end relative to fixed origin");
                job->device_begin_relative_ms=start;job->device_end_relative_ms=end;
            }
            {
                std::lock_guard lock(mutex); job->info.complete_host_ns=now_ns();
            }
            if(result.main.shape!=std::vector<int64_t>({1,3,plan.padded_height,plan.padded_width}))
                throw std::runtime_error("complete NR output does not match frozen NCHW plan");
            output=std::make_shared<Result>(); output->width=plan.valid_width; output->height=plan.valid_height;
            output->session=job->session->id; output->frame=job->info.frame_id; output->epoch=job->info.epoch;
            output->reservation=job->reservation;
            if(gpu_layout) {
                const auto sentinel=pinned ? pinned.get()[readback.numel()-1] : raw.back();
                if(sentinel==2.0f)throw StatusError(RW_INVALID_ARGUMENT,"CUDA producer input contains non-finite or out-of-range values");
                if(sentinel!=0.0f)
                    throw std::runtime_error("NR output is non-finite or outside frozen SDR range");
                if(pinned) {
                    output->color->pinned_elements=readback.numel()-1;
                    output->color->pinned=std::move(pinned);
                } else {
                    raw.pop_back();
                    output->color->values=std::move(raw);
                }
            } else {
                output->color->values.resize(static_cast<size_t>(plan.valid_width)*plan.valid_height*3);
                const size_t pixels=static_cast<size_t>(plan.padded_width)*plan.padded_height;
                for(int y=0;y<plan.valid_height;++y) for(int x=0;x<plan.valid_width;++x) for(size_t c=0;c<3;++c) {
                    const float value=raw[c*pixels+static_cast<size_t>(y)*plan.padded_width+x];
                    if(!std::isfinite(value) || value<0 || value>1)
                        throw std::runtime_error("NR output is non-finite or outside frozen SDR range");
                    output->color->values[(static_cast<size_t>(y)*plan.valid_width+x)*3+c]=value;
                }
            }
            if(job->use_session_history && !options.sdr_temporal_head) {
                job->history_reservation=std::make_shared<Reservation>(host,
                    multiply(output->color->view().size(),sizeof(float)));
                job->history_copy=output->color;
            }
        } catch(const StatusError& e) {
            copy_text(error.data(),error.size(),e.what());status=e.status;
            try {context->synchronize();physical=true;}catch(...){physical=false;}
        } catch(const native_nr::DeviceResourceExhausted& e) {
            copy_text(error.data(),error.size(),e.what());status=RW_RESOURCE_EXHAUSTED;
            try {context->synchronize();physical=true;}catch(...){physical=false;}
        } catch(const std::bad_alloc& e) {
            copy_text(error.data(),error.size(),e.what());status=RW_RESOURCE_EXHAUSTED;
            try {context->synchronize();physical=true;}catch(...){physical=false;}
        } catch(const std::exception& e) {
            copy_text(error.data(),error.size(),e.what()); status=RW_INTERNAL_ERROR;
            try { context->synchronize(); physical=true; }
            catch(const std::exception& drain_error) {
                physical=false; append_text(error.data(),error.size(),"; unconfirmed physical completion: ");
                append_text(error.data(),error.size(),drain_error.what());
            }
        } catch(...) {
            copy_text(error.data(),error.size(),"unexpected native graph exception"); status=RW_INTERNAL_ERROR;
            try { context->synchronize(); physical=true; } catch(...) { physical=false; }
        }
        if(status==RW_OK) {
            // Check again after local Tensor retirement. A failed CUDA free or
            // device completion can quarantine the backend during destruction.
            try { context->synchronize(); }
            catch(const std::exception& e) { status=RW_INTERNAL_ERROR; physical=false; copy_text(error.data(),error.size(),e.what()); }
        }
        if(context->has_active_weight_read()) {
            try {context->finish_weight_read();}
            catch(const std::exception& e){
                status=RW_INTERNAL_ERROR;physical=false;
                if(error[0]) {append_text(error.data(),error.size(),"; weight read retirement: ");append_text(error.data(),error.size(),e.what());}
                else copy_text(error.data(),error.size(),e.what());
            }
            catch(...){status=RW_INTERNAL_ERROR;physical=false;append_text(error.data(),error.size(),"; shared weight read could not retire physically");}
        }
        if(shared_device && !physical && submitted)shared_device->quarantine();
        std::shared_ptr<CudaInputLease> completed_cuda_input;
        {
            std::lock_guard lock(mutex);
            --counters.running;
            if(!physical){slot.quarantined=true;self_quarantine=shared_from_this();}
            if(status==RW_OK && (failed || domain_failed())) {
                status=RW_INTERNAL_ERROR;copy_text(error.data(),error.size(),"runtime/device failure in another execution slot");
            }
            const bool recoverable_resource=shared_weights && status==RW_RESOURCE_EXHAUSTED && physical && !domain_failed();
            const bool recoverable_input=job->cuda_input && status==RW_INVALID_ARGUMENT && physical && !domain_failed();
            if(status!=RW_OK && !recoverable_resource && !recoverable_input) {if(!failed)failure=error; failed=true; closing=true; available.notify_all();}
            if(submitted && !physical) {
                ++counters.quarantined; quarantine.push_back(job);slot.quarantined=true;
                // This self-reference is intentional: a CUDA error did not
                // prove that the driver stopped accessing any owned memory.
                // It requires no error-path allocation and lasts to process
                // teardown, preventing replacement runtimes/device reuse.
                self_quarantine=shared_from_this();
                complete(job,RW_INTERNAL_ERROR,error.data(),false,true);
            } else {
                if(status==RW_OK && job->cancelled) { status=RW_CANCELLED; copy_text(error.data(),error.size(),"running cancellation drained without publishing"); }
                if(status==RW_OK && (job->session->epoch!=job->info.epoch ||
                   job->session->predecessor!=job->info.expected_predecessor)) {
                    status=RW_STALE; copy_text(error.data(),error.size(),"epoch or expected predecessor changed before commit");
                }
                bool published=false;
                if(status==RW_OK) {
                    struct PublishPayload {
                        Runtime* runtime;
                        const std::shared_ptr<Job>* job;
                        std::shared_ptr<Result>* output;
                        bool physical,submitted;
                    } payload{this,&job,&output,physical,submitted};
                    const auto publish=+[](void* opaque) noexcept {
                        auto& value=*static_cast<PublishPayload*>(opaque);
                        const auto& current=*value.job;
                        current->result=std::move(*value.output);
                        if(current->use_session_history) {
                            current->session->gpu_history=std::move(current->next_gpu_history);
                            current->session->history=std::move(current->history_copy);
                            current->session->history_reservation=std::move(current->history_reservation);
                            current->session->has_history=true;
                            current->session->history_frame=current->info.frame_id;
                        }
                        current->session->predecessor=current->info.frame_id;
                        current->info.output_available_host_ns=now_ns();
                        value.runtime->complete(current,RW_OK,"",value.physical,value.submitted);
                    };
                    // Runtime mutex already seals session/cancel state. The
                    // device guard additionally seals Q/epoch through result,
                    // predecessor, done/status/trace publication as one action.
                    if(shared_device)published=shared_device->commit_if_current(job->submitted_device_identity,publish,&payload);
                    else {publish(&payload);published=true;}
                    if(!published) {
                        status=RW_INTERNAL_ERROR;failed=true;closing=true;self_quarantine=shared_from_this();
                        copy_text(error.data(),error.size(),"submitted shared device epoch invalidated before publication");
                        if(failure[0]=='\0') copy_text(failure.data(),failure.size(),error.data());available.notify_all();
                    }
                }
                if(!published)complete(job,status,error.data(),physical,submitted);
                std::vector<float>().swap(job->input);
                std::vector<float>().swap(job->previous);
                job->previous_history.reset();
                job->previous_gpu_history={};job->next_gpu_history={};
                completed_cuda_input=std::move(job->cuda_input);
                std::vector<float>().swap(job->motion);
                std::vector<float>().swap(job->motion_validity);
            }
            publish_node_snapshot(slot,*job);
            active_job.reset();
        }
    }
    std::lock_guard lock(mutex); if(!slot.finished){slot.finished=true;++finished_workers;} finished.notify_all();available.notify_all();
}
std::shared_ptr<Runtime> create_runtime(const rw_nr_runtime_desc& desc) {
    if(desc.struct_size!=sizeof(desc) || desc.abi_version!=RW_ABI_VERSION)
        throw StatusError(RW_ABI_MISMATCH,"native NR runtime ABI mismatch");
    const auto metrics=metrics_policy(desc.flags);
    const bool shared_slots=metrics.execution_slots>1;
    const uint32_t slot_count=metrics.execution_slots;
    if(!metrics.valid || desc.reserved || desc.device_ordinal<0 ||
       !desc.max_sessions || desc.max_sessions>1024 || !desc.max_inflight || desc.max_inflight>64 ||
       !desc.trace_capacity || desc.trace_capacity>4096 || !desc.host_budget_bytes ||
       desc.host_budget_bytes>UINT64_C(137438953472) || !desc.device_budget_bytes)
        throw std::invalid_argument("explicit experimental flag and bounded runtime budgets required");
    if(!is_sha(desc.weights_sha256) || !is_sha(desc.parameter_manifest_sha256) || !is_sha(desc.plan_sha256))
        throw std::invalid_argument("all model and semantic-plan file identities require full lowercase SHA256");
    const auto weights_path=checked_path(desc.weights_path,"PROJECT_DOWNLOADED_WEIGHTS_ROOT");
    const auto manifest_path=checked_path(desc.parameter_manifest_path,"PROJECT_STORAGE_ROOT");
    const auto plan_path=checked_path(desc.plan_path,"PROJECT_STORAGE_ROOT");
    if(sha256(weights_path)!=desc.weights_sha256 || sha256(manifest_path)!=desc.parameter_manifest_sha256 ||
       sha256(plan_path)!=desc.plan_sha256) throw std::invalid_argument("original model or plan file SHA256 mismatch");
    const auto registration=json_file(manifest_path,8*1024*1024);
    if(registration.at("artifact_kind")!="native_nr_parameter_inventory" || registration.at("variant")!="ht" ||
       registration.at("parameters_sha256")!=desc.weights_sha256 ||
       fs::canonical(registration.at("parameters_path").get<std::string>())!=weights_path ||
       registration.at("dll_sha256")!=original_dll || registration.at("resource_sha256")!=original_resource ||
       registration.at("authenticode").at("status")!="verified_nvidia" ||
       registration.at("extractor").at("commit")!=pinned_source ||
       registration.at("extractor").at("inverse_parameter_packing")!="byte_exact" ||
       registration.at("parameter_updates_allowed")!=false || registration.at("inference_only")!=true)
        throw std::invalid_argument("original trusted NVIDIA HT parameter registration is incomplete or mismatched");
    if(shared_slots && desc.max_inflight<slot_count)throw std::invalid_argument("multislot experiment requires at least one admitted request per execution slot");
    auto plan=SourcePlan::load(plan_path);
    if(plan.network!="ht") throw StatusError(RW_UNSUPPORTED,"initial async SDK admits only frozen original HT");
    if(plan.layers.size()>native_nr::max_node_measurements)
        throw StatusError(RW_UNSUPPORTED,"verified graph exceeds the fixed node observation capacity");
    const auto plan_document=json_file(plan_path,16*1024*1024);
    if(plan_document.at("parameter_inventory").at("sha256")!=desc.weights_sha256)
        throw std::invalid_argument("semantic plan is not bound to the actual original parameter file");
    const char* temporal_setting=std::getenv("RW_NR_SDR_TEMPORAL_HEAD");
    const bool sdr_temporal_head=temporal_setting && std::strcmp(temporal_setting,"1")==0;
    const auto slot_bound=native_nr::lowering_memory_bound(plan.modelled_memory_bound,true);
    uint64_t native_bound=rw::detail::checked_slot_product(slot_bound,slot_count);
    uint64_t temporal_extra=0;
    if(sdr_temporal_head) {
        const auto padded_pixels=multiply(plan.padded_width,plan.padded_height);
        const auto history_bytes=multiply(12,multiply(plan.valid_width,plan.valid_height));
        const auto per_slot=add(add(multiply(20,padded_pixels),multiply(2,history_bytes)),512);
        temporal_extra=add(multiply(per_slot,slot_count),multiply(history_bytes,desc.max_sessions));
        native_bound=add(native_bound,temporal_extra);
    }
    if(native_bound>desc.device_budget_bytes)
        throw StatusError(RW_RESOURCE_EXHAUSTED,"source allocation envelope plus post-only workspace exceeds declared device budget");
    // Both modes validate actual graph contents, not just a caller-supplied
    // file hash or the declared semantic_graph_sha256 field.
    native_nr::ModelRegistrationRequest model_request;
    model_request.weights_path=weights_path;model_request.parameter_manifest_path=manifest_path;model_request.plan_path=plan_path;
    model_request.weights_sha256=desc.weights_sha256;model_request.parameter_manifest_sha256=desc.parameter_manifest_sha256;
    model_request.plan_sha256=desc.plan_sha256;
    model_request.lowering_recipe_sha256=native_nr::lowering_recipe_json("ht").at("recipe_fingerprint_sha256").get<std::string>();
    const auto verified_model=native_nr::ModelRegistry::register_original_ht(model_request);
    uint64_t caller_reserve_bytes=0;
    if(shared_slots){
        const auto& declared_reserve=plan_document.at("resources").at("Q_caller_reserve_bytes");
        if(!declared_reserve.is_number_unsigned())
            throw std::invalid_argument("caller reserve must be a nonnegative integer byte count");
        caller_reserve_bytes=rw::detail::checked_slot_product(declared_reserve.get<uint64_t>(),slot_count);
        if(caller_reserve_bytes>desc.device_budget_bytes)
            throw StatusError(RW_RESOURCE_EXHAUSTED,"all slot caller reservations exceed the declared device budget before ownership");
    }
    auto runtime=std::make_shared<Runtime>();
    runtime->registered_model=verified_model;
    runtime->plan_metadata=plan;
    runtime->collect_node_metrics=metrics.collect_node_metrics;
    if(const char* setting=std::getenv("RW_NR_GPU_HISTORY"))runtime->gpu_history=std::strcmp(setting,"1")==0;
    runtime->sdr_temporal_head=sdr_temporal_head;
    if(const char* setting=std::getenv("RW_NR_INPUT_PHASE_DETAIL"))runtime->input_phase_detail=std::strcmp(setting,"1")==0;
    runtime->max_sessions=desc.max_sessions; runtime->max_inflight=desc.max_inflight;
    runtime->trace_capacity=desc.trace_capacity; runtime->host=std::make_shared<HostBudget>(desc.host_budget_bytes);
    runtime->traces.resize(desc.trace_capacity); runtime->sessions.reserve(desc.max_sessions);
    runtime->quarantine.reserve(slot_count);runtime->slots.reserve(slot_count);
    // Reserve the optional registry's maximum and one bounded construction
    // candidate before admitting any weights. Disabled/enabled controls use
    // the same conservative host admission; actual bytes are queried below.
    constexpr uint64_t prepared_metadata_reservation=8ull*1024*1024+64ull*1024;
    const auto node_metrics_reservation=node_metrics_host_reservation_bytes(slot_count);
    // Charge complete private control objects, including the two fixed arrays
    // and their request/summary metadata. The array figure is a subset, not a
    // second charge or a claim about driver event allocations.
    const auto slot_control_reservation=multiply(slot_count,add(sizeof(Slot),sizeof(Graph)));
    const auto node_timing_state_reservation=multiply(slot_count,
        add(sizeof(Context::NodeTimingInfo),2*sizeof(cudaEvent_t)));
    if(slot_control_reservation<node_metrics_reservation)
        throw std::logic_error("slot control reservation does not cover node observation arrays");
    const auto fixed_host=add(node_timing_state_reservation,add(slot_control_reservation,
        add(prepared_metadata_reservation,add(multiply(fs::file_size(weights_path),8),
        add(multiply(desc.trace_capacity,sizeof(TraceRecord)),multiply(desc.max_sessions,512))))));
    runtime->fixed_host=std::make_shared<Reservation>(runtime->host,fixed_host);
    cudaDeviceProp properties{};
    native_nr::cuda_check(cudaGetDeviceProperties(&properties,desc.device_ordinal),"query async NR CUDA device");
#if defined(RW_NR_HOPPER_WGMMA_ENABLED)
    if(properties.major!=9 || properties.minor!=0)
        throw StatusError(RW_UNSUPPORTED,"this NR build requires compute capability 9.0 (SM90a)");
#endif
    if(desc.device_budget_bytes>properties.totalGlobalMem)
        throw StatusError(RW_RESOURCE_EXHAUSTED,"declared device budget exceeds physical memory before device claim");
    constexpr char hex[]="0123456789abcdef"; std::string uuid;
    for(unsigned char c:properties.uuid.bytes) { uuid+=hex[c>>4]; uuid+=hex[c&15]; }
    runtime->claim=std::make_unique<DeviceClaim>(uuid);
    int native_blas_version=0;
    bool shared_factory_started=false;
    auto failed_startup=[&](bool known_resource_failure) {
        bool retain=runtime->domain_failed() || (!known_resource_failure && shared_factory_started && !runtime->shared_weights);
        for(const auto& slot:runtime->slots) {
            if(!slot->context) {if(!known_resource_failure)retain=true;continue;}
            if(slot->context->post_projection_info().initialization_failed){retain=true;continue;}
            try {slot->context->synchronize();if(slot->context->has_active_weight_read())slot->context->finish_weight_read();}
            catch(...){retain=true;}
        }
        if(!retain && !runtime->retire_events())retain=true;
        if(retain) {
            if(runtime->shared_device)runtime->shared_device->quarantine();
            runtime->failed=true;runtime->closing=true;runtime->self_quarantine=runtime;
        }
    };
    try {
        if(shared_slots) {
            const uint64_t weight_limit=verified_model->source_weight_bound_bytes();
            if(!weight_limit || weight_limit>desc.device_budget_bytes)throw StatusError(RW_RESOURCE_EXHAUSTED,"registered immutable W envelope does not fit runtime budget");
            // The conservative fixed host model reservation above covers raw
            // mmap, bounded decode temporaries and this explicit upload cap.
            const auto upload_limit=multiply(fs::file_size(weights_path),4);
            shared_factory_started=true;
            runtime->shared_device=native_nr::SharedDeviceOwner::acquire(desc.device_ordinal,desc.device_budget_bytes);
            if(caller_reserve_bytes)runtime->caller_reserve=runtime->shared_device->budget()->reserve(
                native_nr::DeviceChargeKind::caller_reserve,UINT64_MAX,caller_reserve_bytes);
            runtime->shared_weights=native_nr::DeviceWeightPool::acquire(verified_model,runtime->shared_device,
                weight_limit,upload_limit,slot_count);
        } else runtime->legacy_weights=std::make_unique<Weights>(weights_path);
        for(uint64_t index=0;index<slot_count;++index) {
            auto slot=std::make_unique<Slot>();slot->id=index;
            runtime->slots.push_back(std::move(slot));
            auto& current=*runtime->slots.back();
            if(shared_slots) current.context=std::make_unique<Context>(runtime->shared_device,index,desc.device_budget_bytes);
            else current.context=std::make_unique<Context>(desc.device_budget_bytes,desc.device_ordinal);
            native_nr::blas_check(cublasGetVersion(current.context->blas(),&native_blas_version),"query SDK native cuBLAS profile");
            if(native_blas_version!=native_nr::nr_required_cublas_version())
                throw StatusError(RW_UNSUPPORTED,"loaded cuBLAS version differs from the compiled NR lowering recipe");
            current.context->prepare_post_projection_blas();
            if(runtime->collect_node_metrics)current.context->prepare_node_timing();
            const auto workspace_info=current.context->post_projection_info();
            current.main_workspace_identity=reinterpret_cast<uintptr_t>(workspace_info.main_workspace_address);
            current.post_workspace_identity=reinterpret_cast<uintptr_t>(workspace_info.workspace_address);
            if(shared_slots)current.context->bind_slot_parameters(runtime->shared_weights->bind_slot(index,current.context->stream()));
            current.graph=std::make_unique<Graph>(plan);
            current.stream_identity=reinterpret_cast<uintptr_t>(current.context->stream());
            native_nr::cuda_check(cudaStreamGetPriority(current.context->stream(),&current.stream_priority),"query actual NR slot priority");
            native_nr::cuda_check(cudaEventCreate(&current.begin),"create private SDK slot begin event");
            current.begin_event_identity=reinterpret_cast<uintptr_t>(current.begin);
            native_nr::cuda_check(cudaEventCreate(&current.end),"create private SDK slot completion event");
            current.end_event_identity=reinterpret_cast<uintptr_t>(current.end);
            if(const char* value=std::getenv("RW_NR_CAPTURE_NODE_DETAIL");value && std::strcmp(value,"1")==0) {
                current.captured_profile_events.resize(plan.layers.size());
                for(auto& pair:current.captured_profile_events)for(auto& event:pair)
                    native_nr::cuda_check(cudaEventCreate(&event),"create captured semantic node event");
            }
        }
        if(shared_slots) {
            native_nr::cuda_check(cudaEventCreate(&runtime->timeline_origin),"create fixed stream timeline origin");
            native_nr::cuda_check(cudaEventRecord(runtime->timeline_origin,runtime->slots.front()->context->stream()),"record fixed stream timeline origin");
            runtime->slots.front()->context->synchronize();
        }
    } catch(const native_nr::DeviceResourceExhausted&) {failed_startup(true);throw;}
    catch(...) {failed_startup(false);throw;}
    runtime->identity={{"project_id","renderweave"},{"implementation_kind","native_self"},
        {"support_status","experimental_unqualified"},{"backend","own_cpp_cuda_cublas_no_tensor_framework"},
        {"compiled_source_commit",RW_NATIVE_BUILD_COMMIT},{"abi_version",RW_ABI_VERSION},
        {"network","ht"},{"input_profile",sdr_temporal_head ? "f32_hwc_srgb_sdr_temporal_head_candidate" :
            "f32_hwc_srgb_stateless_reset"},{"execution_slots",slot_count},
        {"execution_slot_policy_version",RW_NR_EXECUTION_SLOT_POLICY_VERSION},
        {"configured_max_sessions",desc.max_sessions},{"configured_max_inflight",desc.max_inflight},
        {"session_history_storage",runtime->gpu_history?"immutable CPU RGB plus transactional GPU RGB":"shared immutable completed CPU RGB output; CUDA input retains transactional GPU RGB"},
        {"cuda_producer_input_available",true},{"cuda_producer_input_storage","borrowed device HWC, producer event, physical-completion release callback"},
        {"gpu_history_enabled",runtime->gpu_history},
        {"gpu_history_capacity_bound_bytes",multiply(add(desc.max_sessions,desc.max_inflight),multiply(multiply(plan.valid_width,plan.valid_height),12))},
        {"input_scan_backend",input_scanner().name},
        {"native_vendor_equivalence_verified",false},{"native_temporal_qualified",false},{"production_qualified",false},
        {"weights_sha256",desc.weights_sha256},{"parameter_manifest_sha256",desc.parameter_manifest_sha256},
        {"plan_file_sha256",desc.plan_sha256},{"semantic_graph_sha256",runtime->plan().semantic_graph_sha256},
        {"device_ordinal",desc.device_ordinal},{"gpu_uuid",uuid},{"device_name",properties.name},
        {"compute_capability",{properties.major,properties.minor}},
        {"width",runtime->plan().valid_width},{"height",runtime->plan().valid_height},
        {"source_memory_envelope_bytes",runtime->plan().modelled_memory_bound},
        {"native_memory_envelope_bytes",native_bound},{"native_lowering_extra_workspace_bytes",multiply(native_nr::post_projection_workspace_bytes,slot_count)},
        {"lowering_recipe",native_nr::lowering_recipe_json("ht")},{"native_cublas_version",native_blas_version},
        {"sdr_temporal_head_candidate",runtime->sdr_temporal_head},
        {"sdr_temporal_head_additional_device_envelope_bytes",temporal_extra},
        {"model_weight_storage_policy",shared_slots?"canonical per-device immutable parameter pool; explicit private execution slots":"one legacy immutable cache per runtime; second runtime on physical device refused"},
        {"shared_weight_pool_enabled",shared_slots},{"device_budget_scope",shared_slots?"single shared budget across immutable W and every private slot":"single Context owned budget"},
        {"two_slot_concurrency_qualified",false},{"eight_slot_concurrency_qualified",false},
        {"host_fixed_model_and_control_reservation_bytes",fixed_host},
        {"host_prepared_metadata_reservation_bytes",prepared_metadata_reservation},
        {"host_node_metrics_array_reservation_bytes",node_metrics_reservation},
        {"host_private_slot_and_graph_control_reservation_bytes",slot_control_reservation},
        {"host_node_timing_state_reservation_bytes",node_timing_state_reservation},
        {"memory_scope","owned SDK reservations and backend allocation budget; no production admission certification"}};
    return runtime;
}
Json trace_json(const TraceRecord& record) {
    const auto& info=record.info;
    return {{"session_id",info.session_id},{"frame_id",info.frame_id},{"epoch",info.epoch},
        {"expected_predecessor",info.expected_predecessor},{"state",info.state},{"status",info.status},
        {"gpu_submitted",bool(info.gpu_submitted)},{"physical_completion_confirmed",bool(info.physical_completion_confirmed)},
        {"submit_host_ns",info.submit_host_ns},{"admitted_host_ns",info.admitted_host_ns},
        {"device_execution_host_ns",info.device_execution_host_ns},{"complete_host_ns",info.complete_host_ns},
        {"output_available_host_ns",info.output_available_host_ns},{"first_wait_timeout_host_ns",info.first_wait_timeout_host_ns},
        {"device_stream_span_collected",record.device_stream_span_collected},
        {"device_stream_span_ms",record.device_stream_span_collected?Json(info.device_stream_span_ms):Json()},{"error",info.error},
        {"execution_slot",record.slot_id==UINT64_MAX?Json():Json(record.slot_id)},
        {"input_h2d_bytes",record.input_h2d_bytes},{"producer_snapshot_d2d_bytes",record.producer_snapshot_bytes},
        {"resident_gpu_history_input_bytes",record.resident_history_input_bytes},
        {"device_interval_collected",record.device_end_relative_ms>record.device_begin_relative_ms},
        {"device_begin_relative_ms",record.device_begin_relative_ms},{"device_end_relative_ms",record.device_end_relative_ms},
        {"device_interval_scope","CUDA stream interval; not proof of concurrent SM execution"}};
}
Json node_snapshot_json(const NodeSnapshot& snapshot,const SourcePlan& plan,bool detail_requested) {
    const bool collected=detail_requested && snapshot.available && snapshot.graph_invoked &&
        snapshot.execution_identity_confirmed && snapshot.bounds_valid && snapshot.count!=0;
    Json value={{"snapshot_available",snapshot.available},{"session_id",snapshot.session_id},
        {"frame_id",snapshot.frame_id},{"epoch",snapshot.epoch},{"expected_predecessor",snapshot.expected_predecessor},
        {"job_status",snapshot.job_status},{"physical_completion_confirmed",snapshot.physical_completion_confirmed},
        {"graph_invoked",snapshot.graph_invoked},{"node_detail_requested",detail_requested},
        {"execution_identity_confirmed",snapshot.execution_identity_confirmed},
        {"node_detail_collected",collected},{"bounds_valid",snapshot.bounds_valid},
        {"node_count",snapshot.count},{"execution_summary",Json()},{"nodes",Json()},
        {"scope","latest dequeued request in this slot; terminal snapshot only, overwritten on the next dequeue"}};
    if(snapshot.available && snapshot.graph_invoked && snapshot.execution_identity_confirmed) {
        const auto& execution=snapshot.execution;
        value["execution_summary"]={{"execution_serial",execution.execution_serial},
            {"started_nodes",execution.started_nodes},{"physically_completed_nodes",execution.physically_completed_nodes},
            {"active_node",execution.active_node},{"failed_node",execution.failed_node},
            {"detail_requested",execution.detail_requested},{"graph_entered",execution.graph_entered},
            {"completed",execution.completed}};
    }
    if(detail_requested && snapshot.available && snapshot.graph_invoked && snapshot.execution_identity_confirmed && snapshot.bounds_valid) {
        value["nodes"]=Json::array();
        for(size_t i=0;i<snapshot.count;++i) {
            const auto& node=snapshot.nodes[i];
            value["nodes"].push_back({{"index",node.index},{"kind",plan.layers[static_cast<size_t>(node.index)].kind},
                {"status",native_nr::node_measurement_state_name(node.status)},
                {"host_dispatch_start_ns",node.dispatch_start_ns},{"host_dispatch_end_ns",node.dispatch_end_ns},
                {"device_span_collected",node.device_span_collected},
                {"device_stream_span_ms",node.device_span_collected?Json(node.device_stream_span_ms):Json()},
                {"live_owned_bytes",node.live_owned_bytes},{"peak_owned_bytes",node.peak_owned_bytes}});
        }
    }
    return value;
}
Json node_timing_json(const Context& context) {
    const auto timing=context.node_timing_info();
    return {{"preparation_attempted",timing.preparation_attempted},{"prepared",timing.prepared},
        {"active",timing.active},{"failed",timing.failed},{"created_events",timing.created_events},
        {"destroyed_events",timing.destroyed_events},{"begin_records",timing.begin_records},
        {"end_records",timing.end_records},{"elapsed_reads",timing.elapsed_reads},
        {"active_run",timing.active_run},{"active_node",timing.active_node},{"last_cuda_error",timing.last_cuda_error},
        {"scope","slot-private optional node event instrumentation; metadata getter performs no CUDA work"}};
}
} // namespace rw::nr_sdk

struct rw_nr_runtime { std::shared_ptr<rw::nr_sdk::Runtime> state; };
struct rw_nr_session { std::shared_ptr<rw::nr_sdk::Runtime> owner; std::shared_ptr<rw::nr_sdk::Session> state; };
struct rw_nr_job { std::shared_ptr<rw::nr_sdk::Runtime> owner; std::shared_ptr<rw::nr_sdk::Job> state; };
struct rw_nr_result { std::shared_ptr<rw::nr_sdk::Result> state; };
using namespace rw::nr_sdk;

extern "C" {
const char* rw_nr_last_error(void) { return last_error.data(); }
rw_status_t rw_nr_runtime_create(const rw_nr_runtime_desc* desc,rw_nr_runtime_t* output) {
    return boundary([&]() -> rw_status_t {
        if(!output) return RW_INVALID_ARGUMENT; *output=nullptr;
        if(!desc) return RW_INVALID_ARGUMENT;
        auto handle=std::make_unique<rw_nr_runtime>(); handle->state=create_runtime(*desc);
        handle->state->start_workers();
        *output=handle.release(); return RW_OK;
    });
}
rw_status_t rw_nr_runtime_query(rw_nr_runtime_t handle,char* output,size_t* bytes) {
    return boundary([&]() -> rw_status_t {
        if(!handle || !bytes) return RW_INVALID_ARGUMENT;
        auto state=handle->state; std::lock_guard lock(state->mutex);
        auto value=state->identity; const auto& c=state->counters;
        uint64_t resident_history_bytes=0,resident_history_sessions=0;
        for(const auto& weak:state->sessions)if(auto session=weak.lock();session && !session->gpu_history.empty()) {
            ++resident_history_sessions;resident_history_bytes+=session->gpu_history.numel()*sizeof(float);
        }
        value["resident_gpu_history_bytes"]=resident_history_bytes;
        value["resident_gpu_history_sessions"]=resident_history_sessions;
        if(state->input_phase_detail)value["input_phase_detail"]={
            {"samples",state->input_phase_samples},{"total_ns",state->input_phase_totals},{"max_ns",state->input_phase_max},
            {"phase_names",{"color_allocation_copy","previous_capture_or_copy","guide_allocation_copy","value_scan"}},
            {"thread_minor_faults",state->input_minor_faults},{"minor_fault_samples",state->input_fault_samples},
            {"scope","CPU ownership/validation completed; includes warmup; request outcomes remain in all counters/traces"}};
        value["observability"]={{"mode",state->collect_node_metrics?"node_detail":"summary"},
            {"node_detail_enabled",state->collect_node_metrics},{"node_measurement_capacity",rw::native_nr::max_node_measurements},
            {"mandatory_request_observations_enabled",true},{"last_slot_snapshot_only",true},
            {"all_request_trace_history_complete",false},{"node_timing_overhead_qualified",false}};
        value["counters"]={{"arrived",c.arrived},{"admitted",c.admitted},{"rejected",c.rejected},
            {"valid_outputs",c.valid_outputs},{"failed",c.failed},{"cancelled",c.cancelled},{"stale",c.stale},
            {"wait_timeouts",c.wait_timeouts},{"running",c.running},{"inflight",c.inflight},{"quarantined",c.quarantined}};
        value["host_reserved_bytes"]=state->host->reserved.load(); value["host_budget_bytes"]=state->host->limit;
        uint64_t live=0,peak=0,host_live=0,host_peak=0,host_limit=0,host_quarantined=0;
        value["slots"]=Json::array();
        for(const auto& slot:state->slots) {
            const auto& context=*slot->context;
            const auto retirement=context.retirement_info();
            live=add(live,context.live_bytes());peak=add(peak,context.peak_live_bytes());
            host_live=add(host_live,context.host_staging_live_bytes());host_peak=add(host_peak,context.host_staging_peak_bytes());
            host_limit=add(host_limit,context.host_staging_budget_bytes());host_quarantined=add(host_quarantined,context.host_staging_quarantined_bytes());
            value["slots"].push_back({{"slot_id",slot->id},{"stream_priority",slot->stream_priority},{"finished",slot->finished},{"quarantined",slot->quarantined},
                {"context_identity",reinterpret_cast<uintptr_t>(slot->context->identity())},
                {"graph_identity",reinterpret_cast<uintptr_t>(slot->graph.get())},
                {"stream_identity",slot->stream_identity},{"begin_event_identity",slot->begin_event_identity},
                {"end_event_identity",slot->end_event_identity},
                {"main_workspace_identity",slot->main_workspace_identity},{"post_workspace_identity",slot->post_workspace_identity},
                {"resource_identity_scope","fixed creation identities; retirement status states whether their physical resources remain live"},
                {"owned_device_live_bytes",context.live_bytes()},{"owned_device_peak_bytes",context.peak_live_bytes()},
                {"tensor_cache_enabled",context.tensor_cache_enabled()},{"tensor_cached_bytes",context.tensor_cached_bytes()},
                {"deferred_tensor_cache",context.deferred_tensor_cache()},
                {"cuda_graph_captures",slot->capture_count.load()},{"cuda_graph_replays",slot->replay_count.load()},
                {"cuda_graph_nodes",slot->captured_node_count.load()},
                {"cuda_graph_allocation_nodes",slot->captured_allocation_nodes.load()},
                {"cuda_graph_free_nodes",slot->captured_free_nodes.load()},
                {"readback_pinned_bytes",slot->readback_pinned_bytes.load()},
                {"readback_pinned_jobs",slot->readback_pinned_jobs.load()},
                {"stream_ordered_memory",context.stream_ordered_memory()},
                {"node_timing",node_timing_json(context)},
                {"node_observation",node_snapshot_json(slot->node_snapshot,state->plan_metadata,state->collect_node_metrics)},
                {"resource_retirement",{{"closed",retirement.closed},{"attempted",retirement.attempted},
                    {"succeeded",retirement.succeeded},{"failed",retirement.failed}}},
                {"post_projection_workspace",rw::native_nr::post_projection_json(context)}});
        }
        for(size_t index=0;index<state->slots.size();++index) {
            const auto& slot=*state->slots[index];
            if(slot.captured_profile_spans.empty())continue;
            Json profile={{"frame_id",slot.profile_frame_id},{"epoch",slot.profile_epoch},
                {"execution_serial",slot.profile_execution_serial},
                {"scope","captured semantic-boundary CUDA events; read after physical frame completion; instrumented diagnostic"},
                {"nodes",Json::array()}};
            for(size_t node=0;node<slot.captured_profile_spans.size();++node)
                profile["nodes"].push_back({{"index",node},{"kind",state->plan_metadata.layers.at(node).kind},
                    {"device_stream_span_ms",slot.captured_profile_spans[node]}});
            value["slots"][index]["captured_node_profile"]=std::move(profile);
        }
        value["slots_owned_device_live_bytes"]=live;value["slots_owned_device_peak_bytes_sum"]=peak;
        value["device_peak_scope"]="conservative sum of slot peaks plus W peak, not simultaneous physical peak; charged peak also includes pending and caller reserve";
        value["immutable_preparation"]=rw::native_nr::preparation_json(state->resources_finalizing || state->resources_retired || !state->legacy_weights && !state->shared_weights?
            state->final_preparation:state->shared_weights?state->shared_weights->preparation_statistics():state->legacy_weights->preparation_statistics());
        if(state->shared_weights)value["immutable_preparation"]["sharing_scope"]="canonical immutable model parameter pool; prepared metadata guarded by one Weights mutex; activations and Graph measurements stay private to each slot";
        value["resources_retired"]=state->resources_retired;value["cleanup_failed"]=state->cleanup_failed;
        value["post_projection_workspace"]=rw::native_nr::post_projection_json(*state->slots.front()->context);
        if(state->shared_weights) {
            const auto weights=state->shared_weights->statistics();
            const auto budget=state->shared_device->budget()->statistics();
            live=add(live,weights.weight_live_bytes);peak=add(peak,weights.weight_peak_bytes);
            host_live=add(host_live,weights.upload_host_live_bytes);host_peak=add(host_peak,weights.upload_host_peak_bytes);
            host_limit=add(host_limit,weights.upload_host_limit_bytes);
            value["shared_weights"]={{"entries",weights.entries},{"uploads",weights.uploads},{"allocations",weights.allocations},{"frees",weights.frees},
                {"weight_live_bytes",weights.weight_live_bytes},{"weight_peak_bytes",weights.weight_peak_bytes},{"weight_quarantined_bytes",weights.weight_quarantined_bytes},
                {"graph_pins",weights.graph_pins},{"active_reads",weights.active_reads},{"physically_completed_reads",weights.physically_completed_reads},
                {"owner_rejections",weights.owner_rejections},{"context_rejections",weights.context_rejections},
                {"upload_host_live_bytes",weights.upload_host_live_bytes},{"upload_host_peak_bytes",weights.upload_host_peak_bytes},
                {"upload_host_limit_bytes",weights.upload_host_limit_bytes}};
            value["shared_device_budget"]={{"limit_bytes",budget.limit_bytes},{"charged_bytes",budget.charged_bytes},{"charged_peak_bytes",budget.charged_peak_bytes},
                {"pending_bytes",budget.pending_bytes},{"allocated_live_bytes",budget.allocated_live_bytes},{"quarantined_bytes",budget.quarantined_bytes},
                {"charged_by_kind",budget.charged_by_kind}};
        }
        value["owned_device_live_bytes"]=live;value["owned_device_peak_bytes"]=peak;
        value["accepting"]=!state->closing && !state->failed && !state->domain_failed(); value["failed"]=state->failed || state->domain_failed();
        value["failure"]=state->failure.data(); value["trace_capacity"]=state->trace_capacity;
        value["host_staging_live_bytes"]=host_live;value["host_staging_peak_bytes"]=host_peak;
        value["host_staging_budget_bytes"]=host_limit;value["host_staging_quarantined_bytes"]=host_quarantined;
        value["host_staging_scope"]="slots plus shared W upload staging once; peak is conservative sum, fixed host reservation is not measured usage";
        value["traces"]=Json::array();
        const size_t oldest=state->trace_count==state->trace_capacity ? state->trace_cursor : 0;
        for(size_t i=0;i<state->trace_count;++i)
            value["traces"].push_back(trace_json(state->traces[(oldest+i)%state->trace_capacity]));
        const auto text=value.dump(); const size_t available=*bytes; *bytes=text.size()+1;
        if(!output) return RW_OK; if(available<*bytes) return RW_BUFFER_TOO_SMALL;
        std::memcpy(output,text.c_str(),*bytes); return RW_OK;
    });
}
rw_status_t rw_nr_runtime_close(rw_nr_runtime_t handle,int64_t timeout) {
    return boundary([&]() -> rw_status_t { return handle ? handle->state->close(timeout) : RW_INVALID_ARGUMENT; });
}
rw_status_t rw_nr_runtime_release(rw_nr_runtime_t handle) {
    return boundary([&]() -> rw_status_t {
        if(!handle) return RW_OK; const auto state=handle->state;
        auto status=state->close(-1);
        if(status==RW_OK)status=state->retire_resources();
        delete handle;return status;
    });
}
rw_status_t rw_nr_session_create(rw_nr_runtime_t handle,const rw_nr_session_desc* desc,rw_nr_session_t* output) {
    return boundary([&]() -> rw_status_t {
        if(!output) return RW_INVALID_ARGUMENT; *output=nullptr;
        if(!handle || !desc) return RW_INVALID_ARGUMENT;
        if(desc->struct_size!=sizeof(*desc) || desc->abi_version!=RW_ABI_VERSION) return RW_ABI_MISMATCH;
        if(desc->flags!=RW_NR_STATELESS_RESET || desc->seed!=0) return RW_UNSUPPORTED;
        auto runtime=handle->state; auto session=std::make_shared<Session>();
        auto result=std::make_unique<rw_nr_session>();
        std::lock_guard lock(runtime->mutex);
        if(runtime->closing || runtime->failed || runtime->domain_failed()) return RW_NOT_READY;
        std::erase_if(runtime->sessions,[](const auto& item){return item.expired();});
        if(runtime->sessions.size()>=runtime->max_sessions || runtime->next_session==UINT64_MAX) return RW_RESOURCE_EXHAUSTED;
        session->id=runtime->next_session++; session->epoch=desc->epoch;
        session->predecessor=desc->predecessor; session->seed=desc->seed;
        runtime->sessions.push_back(session); result->owner=runtime; result->state=std::move(session);
        *output=result.release(); return RW_OK;
    });
}
rw_status_t rw_nr_session_snapshot(rw_nr_session_t handle,rw_nr_session_state* output) {
    return boundary([&]() -> rw_status_t {
        if(!handle || !output) return RW_INVALID_ARGUMENT;
        if(output->struct_size!=sizeof(*output) || output->abi_version!=RW_ABI_VERSION) return RW_ABI_MISMATCH;
        std::lock_guard lock(handle->owner->mutex); const auto& s=*handle->state;
        output->session_id=s.id; output->epoch=s.epoch; output->predecessor=s.predecessor;
        output->pending=s.pending; output->accepting=s.accepting && !handle->owner->closing && !handle->owner->failed && !handle->owner->domain_failed(); return RW_OK;
    });
}
rw_status_t rw_nr_session_reset(rw_nr_session_t handle,uint64_t epoch,uint64_t predecessor) {
    return boundary([&]() -> rw_status_t {
        if(!handle) return RW_INVALID_ARGUMENT; std::lock_guard lock(handle->owner->mutex);
        if(!handle->state->accepting || handle->owner->closing || handle->owner->failed || handle->owner->domain_failed()) return RW_NOT_READY;
        if(epoch<=handle->state->epoch) return RW_INVALID_ARGUMENT;
        handle->state->epoch=epoch; handle->state->predecessor=predecessor;
        handle->state->has_history=false; handle->state->history_frame=RW_NR_NO_PREDECESSOR;
        handle->state->history.reset(); handle->state->history_reservation.reset();
        handle->state->gpu_history={};
        return RW_OK;
    });
}
void rw_nr_session_release(rw_nr_session_t handle) {
    boundary([&]() -> rw_status_t {
        if(handle) { { std::lock_guard lock(handle->owner->mutex); handle->state->accepting=false; } delete handle; }
        return RW_OK;
    });
}
static rw_status_t submit_impl(rw_nr_session_t handle,const rw_nr_frame* frame,
                                const float* previous_color,uint64_t previous_elements,
                                const float* motion,uint64_t motion_elements,
                                const float* motion_validity,uint64_t motion_validity_elements,
                                bool temporal_request,bool session_history,
                                rw_nr_job_t* output,const rw_nr_cuda_input* cuda_input=nullptr) {
    return boundary([&]() -> rw_status_t {
        if(output) *output=nullptr;
        if(!handle) return RW_INVALID_ARGUMENT;
        auto runtime=handle->owner; auto session=handle->state;
        auto info=initial_info(); info.session_id=session->id;
        std::shared_ptr<Reservation> reservation;
        std::shared_ptr<Job> job;
        bool has_previous=false;
        bool has_motion=false;
        bool has_motion_validity=false;
        {
            std::lock_guard lock(runtime->mutex); ++runtime->counters.arrived;
            auto reject=[&](rw_status_t code,const char* reason) {
                runtime->record_rejection(info,code,reason); set_error(reason); return code;
            };
            if(!output || !frame) return reject(RW_INVALID_ARGUMENT,"missing frame or output job pointer");
            if(frame->struct_size!=sizeof(*frame) || frame->abi_version!=RW_ABI_VERSION)
                return reject(RW_ABI_MISMATCH,"frame ABI mismatch");
            if(cuda_input && (cuda_input->struct_size!=sizeof(*cuda_input) || cuda_input->abi_version!=RW_ABI_VERSION))
                return reject(RW_ABI_MISMATCH,"CUDA input ABI mismatch");
            if(cuda_input) {
                const auto enabled=[](const char* name) {const char* v=std::getenv(name);return v && std::strcmp(v,"1")==0;};
                if(!cuda_input->release || !enabled("RW_NR_GPU_INPUT_LAYOUT") || !enabled("RW_NR_GPU_OUTPUT_LAYOUT"))
                    return reject(RW_INVALID_ARGUMENT,"CUDA input requires a release callback and GPU input/output layouts");
                for(const auto* pointer:{frame->color_hwc,motion,motion_validity}) {
                    if(!pointer)continue;
                    cudaPointerAttributes attributes{};
                    const auto status=cudaPointerGetAttributes(&attributes,pointer);
                    if(status!=cudaSuccess || attributes.type!=cudaMemoryTypeDevice || attributes.device!=runtime->identity.at("device_ordinal").get<int>()) {
                        if(status!=cudaSuccess)(void)cudaGetLastError();
                        return reject(RW_INVALID_ARGUMENT,"CUDA input plane must belong to this runtime's device");
                    }
                }
            }
            info.frame_id=frame->frame_id; info.epoch=frame->epoch; info.expected_predecessor=frame->expected_predecessor;
            const bool stateless=(frame->flags==RW_NR_STATELESS_RESET);
            const bool temporal=(frame->flags==RW_NR_TEMPORAL_RGB_HISTORY);
            if(frame->reserved || (!stateless && !(temporal_request && temporal)))
                return reject(RW_UNSUPPORTED,"unsupported NR input mode");
            if(temporal_request && !temporal)
                return reject(RW_UNSUPPORTED,"temporal submit requires the temporal frame flag");
            has_previous=temporal_request && previous_color;
            if (temporal_request && session_history && frame->expected_predecessor!=RW_NR_NO_PREDECESSOR) {
                if (!session->has_history || session->history_frame!=frame->expected_predecessor)
                    return reject(RW_STALE,"session-owned history is missing or stale");
                has_previous=true;
            }
            has_motion=temporal_request && motion;
            has_motion_validity=temporal_request && motion_validity;
            if(runtime->sdr_temporal_head && temporal_request &&
               (!session_history || (has_previous && !has_motion)))
                return reject(RW_UNSUPPORTED,"SDR temporal head requires SDK session history and pixel motion after reset");
            if(temporal_request && ((frame->expected_predecessor==RW_NR_NO_PREDECESSOR) != !has_previous))
                return reject(RW_INVALID_ARGUMENT,"temporal previous colour must match predecessor presence");
            if(has_motion && !has_previous)
                return reject(RW_INVALID_ARGUMENT,"motion requires temporal previous colour");
            if(has_motion_validity && !has_motion)
                return reject(RW_INVALID_ARGUMENT,"motion validity requires motion");
            const auto& plan=runtime->plan();
            if(frame->width!=static_cast<uint32_t>(plan.valid_width) || frame->height!=static_cast<uint32_t>(plan.valid_height) ||
               !frame->color_hwc || reinterpret_cast<uintptr_t>(frame->color_hwc)%alignof(float) ||
               frame->color_elements!=multiply(multiply(frame->width,frame->height),3))
                return reject(RW_INVALID_ARGUMENT,"input must match exact F32 HWC plan dimensions and element count");
            if(has_previous && !session_history && previous_elements!=frame->color_elements)
                return reject(RW_INVALID_ARGUMENT,"previous colour element count differs from current frame");
            if(has_motion && motion_elements!=multiply(multiply(frame->width,frame->height),2))
                return reject(RW_INVALID_ARGUMENT,"motion element count differs from current frame geometry");
            if(has_motion_validity && motion_validity_elements!=multiply(frame->width,frame->height))
                return reject(RW_INVALID_ARGUMENT,"motion validity element count differs from current frame geometry");
            if(runtime->closing || runtime->failed || runtime->domain_failed() || !session->accepting) return reject(RW_NOT_READY,"runtime or session is not accepting");
            if(frame->epoch!=session->epoch || frame->expected_predecessor!=session->predecessor)
                return reject(RW_STALE,"session epoch or predecessor mismatch");
            if(frame->frame_id==RW_NR_NO_PREDECESSOR ||
               (session->predecessor!=RW_NR_NO_PREDECESSOR && frame->frame_id<=session->predecessor))
                return reject(RW_INVALID_ARGUMENT,"duplicate or backwards frame id");
            if(!rw::detail::request_has_capacity(runtime->counters.inflight,runtime->max_inflight,session->pending))
                return reject(RW_RESOURCE_EXHAUSTED,"bounded session/queue capacity exhausted");
            const uint64_t input=multiply(frame->color_elements,sizeof(float));
            const uint64_t previous_bytes=has_previous
                ? multiply(session_history ? frame->color_elements : previous_elements,sizeof(float)) : 0;
            const uint64_t motion_bytes=has_motion ? multiply(motion_elements,sizeof(float)) : 0;
            const uint64_t motion_validity_bytes=has_motion_validity ? multiply(motion_validity_elements,sizeof(float)) : 0;
            const uint64_t padded=multiply(multiply(multiply(plan.padded_width,plan.padded_height),3),sizeof(float));
            try {
                reservation=std::make_shared<Reservation>(runtime->host,
                    cuda_input?multiply(input,2):
                    add(multiply(input,2),add(multiply(padded,2),add(multiply(previous_bytes,2),add(multiply(motion_bytes,2),multiply(motion_validity_bytes,2))))));
                job=std::make_shared<Job>(); job->session=session; job->reservation=reservation; job->info=info;
            }
            catch(const StatusError& e) { return reject(e.status,e.what()); }
            catch(const std::bad_alloc&) { return reject(RW_RESOURCE_EXHAUSTED,"SDK job metadata allocation failed"); }
            session->pending=true; ++runtime->counters.inflight;
        }
        // Input has not reached the GPU yet. Any copy/validation failure safely
        // releases only this reservation and is counted as a failed admission.
        std::unique_ptr<rw_nr_job> result;
        try {
            const bool phase_detail=runtime->input_phase_detail;
            std::array<uint64_t,5> phase{};
#if defined(__linux__)
            rusage before_usage{},after_usage{};
            const bool before_usage_valid=phase_detail && getrusage(RUSAGE_THREAD,&before_usage)==0;
#endif
            if(phase_detail)phase[0]=now_ns();
            if(cuda_input) {
                job->cuda_input=std::make_shared<CudaInputLease>();
                job->cuda_input->input=*cuda_input;job->cuda_input->color=frame->color_hwc;
            } else job->input.assign(frame->color_hwc,frame->color_hwc+frame->color_elements);
            if(phase_detail)phase[1]=now_ns();
            job->has_previous=has_previous;
            if(has_previous && !session_history) job->previous.assign(previous_color,previous_color+previous_elements);
            job->use_session_history=session_history;
            if(session_history && has_previous) {
                std::lock_guard lock(runtime->mutex);
                job->previous_history=job->session->history;
                job->previous_gpu_history=job->session->gpu_history;
            }
            if(phase_detail)phase[2]=now_ns();
            job->has_motion=has_motion;
            if(has_motion && !cuda_input) job->motion.assign(motion,motion+motion_elements);
            job->has_motion_validity=has_motion_validity;
            if(has_motion_validity && !cuda_input) job->motion_validity.assign(motion_validity,motion_validity+motion_validity_elements);
            if(phase_detail)phase[3]=now_ns();
            if(!cuda_input && (!valid_sdr(job->input) ||
               (has_previous && !job->previous_history && !valid_sdr(job->previous)) ||
               (has_motion && !valid_finite(job->motion))))
                throw StatusError(RW_INVALID_ARGUMENT,"input contains non-finite or out-of-range SDR code values");
            if(!cuda_input && has_motion_validity && !valid_sdr(job->motion_validity))
                throw StatusError(RW_INVALID_ARGUMENT,"motion validity contains non-finite or out-of-range values");
            if(phase_detail) {
                phase[4]=now_ns();
#if defined(__linux__)
                const bool after_usage_valid=getrusage(RUSAGE_THREAD,&after_usage)==0;
#endif
                std::lock_guard lock(runtime->mutex);++runtime->input_phase_samples;
                for(size_t i=0;i<4;++i) {
                    const auto span=phase[i+1]-phase[i];runtime->input_phase_totals[i]+=span;
                    runtime->input_phase_max[i]=std::max(runtime->input_phase_max[i],span);
                }
#if defined(__linux__)
                if(before_usage_valid && after_usage_valid) {
                    ++runtime->input_fault_samples;runtime->input_minor_faults+=after_usage.ru_minflt-before_usage.ru_minflt;
                }
#endif
            }
            result=std::make_unique<rw_nr_job>(); result->owner=runtime; result->state=job;
        } catch(const std::bad_alloc&) {
            std::lock_guard lock(runtime->mutex); session->pending=false; --runtime->counters.inflight;
            runtime->record_rejection(info,RW_RESOURCE_EXHAUSTED,"input ownership allocation failed before GPU submission");
            runtime->finished.notify_all();
            throw;
        } catch(...) {
            std::lock_guard lock(runtime->mutex); session->pending=false; --runtime->counters.inflight;
            runtime->record_rejection(info,RW_INVALID_ARGUMENT,"SDR validation failed before GPU submission");
            runtime->finished.notify_all();
            throw;
        }
        {
            std::lock_guard lock(runtime->mutex);
            if(runtime->closing || runtime->failed || runtime->domain_failed() || !session->accepting || session->epoch!=frame->epoch ||
               session->predecessor!=frame->expected_predecessor) {
                session->pending=false; --runtime->counters.inflight;
                runtime->record_rejection(info,RW_STALE,"session changed during input ownership transfer");
                runtime->finished.notify_all();
                return RW_STALE;
            }
            job->info.admitted_host_ns=now_ns();
            try { runtime->queue.push_back(job); }
            catch(...) { session->pending=false; --runtime->counters.inflight;
                runtime->record_rejection(info,RW_RESOURCE_EXHAUSTED,"bounded queue node allocation failed");
                runtime->finished.notify_all(); throw; }
            if(job->cuda_input)job->cuda_input->adopted=true;
            ++runtime->counters.admitted; runtime->available.notify_one();
        }
        *output=result.release(); return RW_OK;
    });
}
rw_status_t rw_nr_submit(rw_nr_session_t handle,const rw_nr_frame* frame,rw_nr_job_t* output) {
    return submit_impl(handle,frame,nullptr,0,nullptr,0,nullptr,0,false,false,output);
}
rw_status_t rw_nr_submit_cuda_history(rw_nr_session_t handle,const rw_nr_frame* frame,
    const rw_nr_cuda_input* input,rw_nr_job_t* output) {
    rw_nr_cuda_input missing{};missing.struct_size=sizeof(missing);missing.abi_version=RW_ABI_VERSION;
    if(!input)input=&missing;
    if(input->struct_size!=sizeof(*input) || input->abi_version!=RW_ABI_VERSION)
        return submit_impl(handle,frame,nullptr,0,nullptr,0,nullptr,0,true,true,output,input);
    return submit_impl(handle,frame,nullptr,0,input->motion_hwc2,input->motion_elements,
        input->motion_validity,input->motion_validity_elements,true,true,output,input);
}
rw_status_t rw_nr_submit_temporal_rgb(rw_nr_session_t handle,const rw_nr_frame* frame,
                                       const float* previous_color,uint64_t previous_elements,
                                       rw_nr_job_t* output) {
    return submit_impl(handle,frame,previous_color,previous_elements,nullptr,0,nullptr,0,true,false,output);
}
rw_status_t rw_nr_submit_temporal_rgb_motion(rw_nr_session_t handle,const rw_nr_frame* frame,
                                              const float* previous_color,uint64_t previous_elements,
                                              const float* motion,uint64_t motion_elements,
                                              rw_nr_job_t* output) {
    return submit_impl(handle,frame,previous_color,previous_elements,motion,motion_elements,nullptr,0,true,false,output);
}
rw_status_t rw_nr_submit_temporal_history_motion(rw_nr_session_t handle,const rw_nr_frame* frame,
                                                  const float* motion,uint64_t motion_elements,
                                                  rw_nr_job_t* output) {
    return submit_impl(handle,frame,nullptr,0,motion,motion_elements,nullptr,0,true,true,output);
}
rw_status_t rw_nr_submit_temporal_history_motion_validity(
    rw_nr_session_t handle,const rw_nr_frame* frame,const float* motion,uint64_t motion_elements,
    const float* motion_validity,uint64_t motion_validity_elements,rw_nr_job_t* output) {
    return submit_impl(handle,frame,nullptr,0,motion,motion_elements,motion_validity,motion_validity_elements,true,true,output);
}
rw_status_t rw_nr_job_wait(rw_nr_job_t handle,int64_t timeout) {
    return boundary([&]() -> rw_status_t {
        if(!handle || !valid_timeout(timeout)) return RW_INVALID_ARGUMENT;
        auto runtime=handle->owner; auto job=handle->state; std::unique_lock lock(runtime->mutex);
        if(timeout<0) job->changed.wait(lock,[&]{return job->done;});
        else if(!job->changed.wait_for(lock,std::chrono::milliseconds(timeout),[&]{return job->done;})) {
            ++runtime->counters.wait_timeouts;
            if(!job->info.first_wait_timeout_host_ns) job->info.first_wait_timeout_host_ns=now_ns();
            return RW_NOT_READY;
        }
        if(job->info.status!=RW_OK) set_error(job->info.error);
        return job->info.status;
    });
}
rw_status_t rw_nr_job_cancel(rw_nr_job_t handle) {
    return boundary([&]() -> rw_status_t {
        if(!handle) return RW_INVALID_ARGUMENT; std::lock_guard lock(handle->owner->mutex);
        if(handle->state->done) return RW_NOT_READY;
        handle->state->cancelled=true; handle->owner->available.notify_one(); return RW_OK;
    });
}
rw_status_t rw_nr_job_query(rw_nr_job_t handle,rw_nr_job_info* output) {
    return boundary([&]() -> rw_status_t {
        if(!handle || !output) return RW_INVALID_ARGUMENT;
        if(output->struct_size!=sizeof(*output) || output->abi_version!=RW_ABI_VERSION) return RW_ABI_MISMATCH;
        std::lock_guard lock(handle->owner->mutex); *output=handle->state->info; return RW_OK;
    });
}
rw_status_t rw_nr_job_get_result(rw_nr_job_t handle,rw_nr_result_t* output) {
    return boundary([&]() -> rw_status_t {
        if(!output) return RW_INVALID_ARGUMENT; *output=nullptr; if(!handle) return RW_INVALID_ARGUMENT;
        auto result=std::make_unique<rw_nr_result>();
        std::lock_guard lock(handle->owner->mutex);
        if(!handle->state->done) return RW_NOT_READY;
        if(handle->state->info.status!=RW_OK) return handle->state->info.status;
        if(!handle->state->info.physical_completion_confirmed || !handle->state->result) return RW_INTERNAL_ERROR;
        result->state=handle->state->result; *output=result.release(); return RW_OK;
    });
}
void rw_nr_job_release(rw_nr_job_t handle) { boundary([&]() -> rw_status_t { delete handle; return RW_OK; }); }
rw_status_t rw_nr_job_retain(rw_nr_job_t handle,rw_nr_job_t* output) {
    return boundary([&]() -> rw_status_t {
        if(!output) return RW_INVALID_ARGUMENT; *output=nullptr; if(!handle) return RW_INVALID_ARGUMENT;
        auto result=std::make_unique<rw_nr_job>(); result->owner=handle->owner; result->state=handle->state;
        *output=result.release(); return RW_OK;
    });
}
rw_status_t rw_nr_result_view_get(rw_nr_result_t handle,rw_nr_result_view* output) {
    return boundary([&]() -> rw_status_t {
        if(!handle || !output) return RW_INVALID_ARGUMENT;
        if(output->struct_size!=sizeof(*output) || output->abi_version!=RW_ABI_VERSION) return RW_ABI_MISMATCH;
        const auto& r=*handle->state; output->width=r.width; output->height=r.height;
        const auto color=r.color->view();
        output->color_hwc=color.data(); output->color_elements=color.size();
        output->session_id=r.session; output->frame_id=r.frame; output->epoch=r.epoch; return RW_OK;
    });
}
rw_status_t rw_nr_result_retain(rw_nr_result_t handle,rw_nr_result_t* output) {
    return boundary([&]() -> rw_status_t {
        if(!output) return RW_INVALID_ARGUMENT; *output=nullptr; if(!handle) return RW_INVALID_ARGUMENT;
        auto result=std::make_unique<rw_nr_result>(); result->state=handle->state; *output=result.release(); return RW_OK;
    });
}
void rw_nr_result_release(rw_nr_result_t handle) { boundary([&]() -> rw_status_t { delete handle; return RW_OK; }); }
} // extern C
