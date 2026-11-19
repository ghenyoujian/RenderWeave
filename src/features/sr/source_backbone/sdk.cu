#include "rw/rw_native_sr.h"
#include "reconstruction.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
namespace sr=rw::sr::source_backbone;
using Clock=std::chrono::steady_clock;
thread_local char error_text[768]{};

struct Failure {rw_status_t status; const char* message;};
void require(rw_status_t status,const char* message) {if(status!=RW_OK) throw Failure{status,message};}
void check(cudaError_t status) {
    if(status!=cudaSuccess)
        throw Failure{status==cudaErrorMemoryAllocation ? RW_RESOURCE_EXHAUSTED : RW_INTERNAL_ERROR,
                      cudaGetErrorString(status)};
}

template<class F> rw_status_t boundary(F function) noexcept {
    error_text[0]=0;
    try {return function();}
    catch(const Failure& e) {std::snprintf(error_text,sizeof(error_text),"%s",e.message);return e.status;}
    catch(const std::bad_alloc&) {return RW_RESOURCE_EXHAUSTED;}
    catch(const std::invalid_argument& e) {std::snprintf(error_text,sizeof(error_text),"%s",e.what());return RW_INVALID_ARGUMENT;}
    catch(const std::exception& e) {std::snprintf(error_text,sizeof(error_text),"%s",e.what());return RW_INTERNAL_ERROR;}
    catch(...) {return RW_INTERNAL_ERROR;}
}

std::size_t pixels(sr::Geometry g) {return std::size_t(g.width)*g.height;}
uint64_t now_ns() {return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}

std::vector<std::byte> read_exact(const char* path,std::size_t length) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=static_cast<std::streamoff>(length)) throw std::invalid_argument("Incorrect SR model asset size");
    std::vector<std::byte> bytes(length);
    file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(bytes.data()),bytes.size())) throw std::invalid_argument("SR model asset read failed");
    return bytes;
}

struct Buffer {
    void* address{};
    Buffer()=default;
    Buffer(const Buffer&)=delete;
    ~Buffer() {if(address) cudaFree(address);}
    void allocate(std::size_t bytes) {check(cudaMalloc(&address,bytes));}
    template<class T> T* data() const {return static_cast<T*>(address);}
    void abandon() {address=nullptr;}
};

struct Model {
    explicit Model(int device):ordinal(device) {}
    int ordinal;
    sr::Weights layout;
    Buffer gpu_weights,gpu_filter;
    ~Model() {cudaSetDevice(ordinal);}
};

// Both asset decoders verify the fixed original SHA before reusing an upload.
std::shared_ptr<Model> model_for(int ordinal,std::span<const std::byte> weights,
                                std::span<const std::byte> filter_bytes) {
    auto decoded=sr::decode_original(weights);
    auto filter=sr::decode_reconstruction_filter(filter_bytes);
    static std::mutex mutex;
    static std::vector<std::weak_ptr<Model>> devices;
    std::lock_guard lock(mutex);
    if(devices.size()<=std::size_t(ordinal)) devices.resize(std::size_t(ordinal)+1);
    auto& entry=devices[ordinal];
    if(auto current=entry.lock()) return current;
    auto model=std::make_shared<Model>(ordinal);
    model->layout=std::move(decoded);
    model->gpu_weights.allocate(sr::packed_bytes);
    model->gpu_filter.allocate(filter.size()*2);
    check(cudaMemcpy(model->gpu_weights.address,model->layout.elements.data(),sr::packed_bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(model->gpu_filter.address,filter.data(),filter.size()*2,cudaMemcpyHostToDevice));
    entry=model;
    return model;
}

struct History {
    Buffer colors,luma,hidden;
    void allocate(const sr::SurfaceGeometry& geometry) {
        colors.allocate(pixels(geometry.history)*8);
        luma.allocate(pixels(geometry.luma)*2);
        hidden.allocate(pixels(geometry.neural)*8);
    }
    void abandon() {colors.abandon();luma.abandon();hidden.abandon();}
};

struct Session {
    explicit Session(int device):ordinal(device) {}
    int ordinal;
    rw_state_guard_t guard{};
    std::array<History,2> history;
    unsigned read_index=0,phase=0;
    bool inflight=false;
    ~Session() {cudaSetDevice(ordinal);rw_state_guard_release(guard);}
};

struct Job {
    std::shared_ptr<Session> session;
    std::size_t slot=0;
    uint64_t ticket=0;
    rw_sr_job_info info{};
    std::array<double,5> stage_ms{};
    bool stage_timing_valid=false;
    bool cuda_io=false;
    bool released=false;
};

struct Slot {
    Buffer input,prepared,field,scratch,rgba,invalid,history_binding;
    void* host{};
    cudaStream_t stream{};
    cudaEvent_t start{},completion{};
    std::array<cudaEvent_t,4> stage_ends{};
    std::shared_ptr<Job> job;
    std::array<cudaGraphExec_t,8> recurring_graphs{};

    void clear_graphs() {
        for(auto& graph:recurring_graphs) {
            if(graph) cudaGraphExecDestroy(graph);
            graph=nullptr;
        }
    }

    ~Slot() {
        clear_graphs();
        if(start) cudaEventDestroy(start);
        if(completion) cudaEventDestroy(completion);
        for(auto event:stage_ends) if(event) cudaEventDestroy(event);
        if(stream) cudaStreamDestroy(stream);
        if(host) cudaFreeHost(host);
    }
    void abandon() {
        input.abandon();prepared.abandon();field.abandon();scratch.abandon();rgba.abandon();invalid.abandon();
        history_binding.abandon();
        host=nullptr;stream=nullptr;start=nullptr;completion=nullptr;
        stage_ends.fill(nullptr);
        recurring_graphs.fill(nullptr);
    }
};

__global__ void validate_result(const float* rgba,std::size_t rgba_count,
                                const __half* colors,std::size_t color_count,const __half* luma,
                                std::size_t luma_count,const __half* hidden,std::size_t hidden_count,int* bad) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if ((i<rgba_count && !isfinite(rgba[i])) ||
        (i<color_count && !isfinite(__half2float(colors[i]))) ||
        (i<luma_count && !isfinite(__half2float(luma[i]))) ||
        (i<hidden_count && !isfinite(__half2float(hidden[i])))) atomicOr(bad,1);
}

__global__ void snapshot_input(const float* color,const float* motion,const float* depth,
                               float* destination,std::size_t low,int* bad) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto rgb_size=3*low, motion_end=5*low;
    if(i>=(motion ? 6*low : rgb_size)) return;
    const bool in_motion=i>=rgb_size && i<motion_end;
    const float value=i<rgb_size ? color[i] : (in_motion ? motion[i-rgb_size] : depth[i-motion_end]);
    const auto bits=__float_as_uint(value);
    const bool valid=in_motion ? (bits&0x7f800000u)!=0x7f800000u
                              : bits<=0x3f800000u || bits==0x80000000u;
    // Invalid inputs never reach the model as NaNs or invalid coordinates.
    // The same flag suppresses publication of the resulting slot and history.
    destination[i]=valid ? value : 0.f;
    if(!valid) atomicOr(bad,2);
}

__global__ void bind_history(sr::HistoryBinding* destination,sr::HistoryBinding binding) {
    *destination=binding;
}

struct Runtime {
    Runtime(const rw_sr_runtime_desc& desc,std::span<const std::byte> weights={},
            std::span<const std::byte> filter={}):ordinal(desc.device_ordinal),width(desc.width),height(desc.height),
        max_sessions(desc.max_sessions),geometry(sr::reconstruction_geometry(width,height)),
        low(std::size_t(width)*height),out(pixels(geometry.output)*4),
        use_graph(desc.struct_size>=sizeof(desc) && (desc.options&RW_SR_CUDA_GRAPH)),
        backbone_options{desc.struct_size>=sizeof(desc) && (desc.options&(RW_SR_TILED_BACKBONE|RW_SR_REUSE_MATRIX_ROWS)),
                         desc.struct_size>=sizeof(desc) && (desc.options&(RW_SR_PARALLEL_REDUCE|RW_SR_ORIGINAL_FP16_BACKBONE)),
                         desc.struct_size>=sizeof(desc) && (desc.options&RW_SR_REUSE_MATRIX_ROWS),
                         desc.struct_size>=sizeof(desc) && (desc.options&RW_SR_ORIGINAL_FP16_BACKBONE)},
        profile_stages(desc.struct_size>=sizeof(desc) && (desc.options&RW_SR_STAGE_TIMINGS)),
        cuda_io_only(desc.struct_size>=sizeof(desc) && (desc.options&RW_SR_CUDA_IO_ONLY)) {
        for(unsigned phase=0;phase<8;++phase) scratch=std::max(scratch,sr::workspace_bytes(geometry.neural,phase));
        slot_bytes=low*24+pixels(geometry.neural)*112+scratch+out*4+4+
            (use_graph ? sizeof(sr::HistoryBinding) : 0);
        const auto history_bytes=pixels(geometry.history)*8+pixels(geometry.luma)*2+pixels(geometry.neural)*8;
        session_bytes=history_bytes*2;
        const rw_memory_plan plan{sizeof(plan),RW_ABI_VERSION,desc.device_budget_bytes,
            sr::packed_bytes+sr::filter_elements*2,slot_bytes,desc.execution_slots,session_bytes,
            desc.max_sessions,0,0,0};
        uint64_t required=0;
        require(rw_memory_plan_required(&plan,&required),"SR slot and history allocation exceeds the device budget");
        activate();
        model=weights.empty()
            ? model_for(ordinal,read_exact(desc.weights_path,sr::packed_bytes),
                        read_exact(desc.filter_path,sr::filter_elements*2))
            : model_for(ordinal,weights,filter);
        const auto completion_flags=desc.struct_size>=sizeof(desc) && (desc.options&RW_SR_SPIN_WAIT)
            ? cudaEventDefault : cudaEventBlockingSync;
        for(uint32_t i=0;i<desc.execution_slots;++i) {
            auto slot=std::make_unique<Slot>();
            slot->input.allocate(low*24);
            slot->prepared.allocate(pixels(geometry.neural)*32);
            slot->field.allocate(pixels(geometry.neural)*80);
            slot->scratch.allocate(scratch);
            slot->rgba.allocate(out*4);
            slot->invalid.allocate(4);
            if(use_graph) slot->history_binding.allocate(sizeof(sr::HistoryBinding));
            check(cudaMallocHost(&slot->host,staging_bytes()));
            check(cudaStreamCreateWithFlags(&slot->stream,cudaStreamNonBlocking));
            check(cudaEventCreate(&slot->start));
            check(cudaEventCreateWithFlags(&slot->completion,completion_flags));
            if(profile_stages) for(auto& event:slot->stage_ends) check(cudaEventCreate(&event));
            slots.push_back(std::move(slot));
        }
    }

    int ordinal,width,height;
    uint32_t max_sessions;
    sr::SurfaceGeometry geometry;
    std::size_t low,out,scratch=0,slot_bytes=0,session_bytes=0;
    std::mutex mutex;
    std::shared_ptr<Model> model;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<std::weak_ptr<Session>> sessions;
    const bool use_graph;
    const sr::BackboneOptions backbone_options;
    const bool profile_stages;
    const bool cuda_io_only;
    bool accepting=true;

    void activate() const {check(cudaSetDevice(ordinal));}
    float* output(Slot& slot) const {return static_cast<float*>(slot.host)+low*6;}
    std::size_t staging_bytes() const {return cuda_io_only ? 4 : low*24+out*4+4;}
    void* validation(Slot& slot) const {return static_cast<char*>(slot.host)+staging_bytes()-4;}

    void enqueue(Slot& slot,const sr::PrepareInput& input,const History& previous,
                 const sr::ReconstructionOutput& next) {
        auto submit_kernels=[&](const sr::HistoryBinding* binding=nullptr) {
            return sr::enqueue_frame(input,previous.luma.data<uint16_t>(),slot.job->session->phase,model->layout,
                model->gpu_weights.data<uint16_t>(),model->gpu_filter.data<uint16_t>(),slot.prepared.data<uint16_t>(),
                slot.field.data<uint16_t>(),next,slot.scratch.address,scratch,slot.stream,backbone_options,
                profile_stages ? slot.stage_ends.data()+1 : nullptr,binding);
        };
        if(!use_graph || input.reset) {
            check(submit_kernels());
            return;
        }

        // The binding is written on the same stream as the graph. Slot
        // retirement prevents it from being updated while kernels can read it.
        auto* binding=slot.history_binding.data<sr::HistoryBinding>();
        const sr::HistoryBinding pointers{input.previous_ycocg,previous.luma.data<uint16_t>(),
                                         input.previous_hidden,next.ycocg,next.luma,next.hidden};
        bind_history<<<1,1,0,slot.stream>>>(binding,pointers);
        check(cudaGetLastError());
        auto& executable=slot.recurring_graphs[slot.job->session->phase];
        if(!executable) {
            cudaGraph_t graph=nullptr;
            check(cudaStreamBeginCapture(slot.stream,cudaStreamCaptureModeThreadLocal));
            const auto launched=submit_kernels(binding);
            const auto captured=cudaStreamEndCapture(slot.stream,&graph);
            const auto instantiated=launched==cudaSuccess && captured==cudaSuccess
                ? cudaGraphInstantiateWithFlags(&executable,graph,0) : cudaSuccess;
            if(graph) cudaGraphDestroy(graph);
            check(launched);
            check(captured);
            check(instantiated);
        }
        check(cudaGraphLaunch(executable,slot.stream));
    }

    void retire(Slot& slot,rw_status_t error=RW_OK) {
        auto& job=*slot.job;
        auto& session=*job.session;
        int bad=1;
        if(error==RW_OK) {
            std::memcpy(&bad,validation(slot),sizeof(bad));
            float stream_ms=0;
            if(cudaEventElapsedTime(&stream_ms,slot.start,slot.completion)==cudaSuccess)
                job.info.device_stream_ms=stream_ms;
            if(profile_stages) {
                const std::array events{slot.start,slot.stage_ends[0],slot.stage_ends[1],
                                        slot.stage_ends[2],slot.stage_ends[3],slot.completion};
                job.stage_timing_valid=true;
                for(std::size_t i=0;i<job.stage_ms.size();++i) {
                    float duration=0;
                    if(cudaEventElapsedTime(&duration,events[i],events[i+1])!=cudaSuccess) {
                        job.stage_timing_valid=false;
                        break;
                    }
                    job.stage_ms[i]=duration;
                }
            }
        }
        const bool valid=bad==0;
        const auto publication=rw_state_guard_complete(session.guard,job.ticket,1,valid);
        if(publication==RW_OK) {
            session.read_index=1-session.read_index;
            session.phase=(session.phase+1)%8;
        }
        session.inflight=false;
        job.info.status=error==RW_OK ? publication : error;
        if((bad&2) && job.info.status==RW_OUTPUT_INVALID) job.info.status=RW_INVALID_ARGUMENT;
        job.info.physical_complete=1;
        job.info.observed_complete_ns=now_ns();
        if(job.released) slot.job.reset();
    }

    void poll(Slot& slot) {
        if(!slot.job || slot.job->info.physical_complete || !slot.job->info.device_enqueued_ns) return;
        const auto status=cudaEventQuery(slot.completion);
        if(status==cudaErrorNotReady) return;
        if(status!=cudaSuccess) {accepting=false; check(status);}
        retire(slot);
    }

    ~Runtime() {
        const bool active=cudaSetDevice(ordinal)==cudaSuccess;
        bool unknown=false;
        for(auto& slot:slots) {
            if(slot->job && !slot->job->info.physical_complete) {
                auto& job=*slot->job;
                rw_state_guard_cancel(job.session->guard,job.ticket);
                if(active && cudaStreamSynchronize(slot->stream)==cudaSuccess)
                    retire(*slot,job.info.device_enqueued_ns ? RW_OK : RW_INTERNAL_ERROR);
                else {unknown=true;slot->job->session->history[0].abandon();slot->job->session->history[1].abandon();slot->abandon();}
            }
        }
        if(unknown) {
            // A device failure that prevents draining can leave immutable model
            // addresses live on the device. Preserve their shared owner.
            auto* retained=new(std::nothrow) std::shared_ptr<Model>(model);
            if(!retained) std::terminate();
        }
    }
};

bool correct_header(uint32_t size,uint32_t version,std::size_t expected) {
    return size>=expected && version==RW_ABI_VERSION;
}

bool check_input(const float* data,uint64_t actual,std::size_t expected,bool normalized) {
    if(!data || actual!=expected) return false;
    // A reduction without per-sample branches lets the host vectorize both
    // passes. Positive IEEE float bits sort numerically. Keep signed zero
    // legal for the normalized colour and depth planes.
    uint32_t invalid=0;
    if(normalized) {
        for(std::size_t i=0;i<expected;++i) {
            const auto bits=std::bit_cast<uint32_t>(data[i]);
            invalid|=(bits>0x3f800000u)&(bits!=0x80000000u);
        }
    } else {
        for(std::size_t i=0;i<expected;++i) {
            const auto bits=std::bit_cast<uint32_t>(data[i]);
            invalid|=(bits&0x7f800000u)==0x7f800000u;
        }
    }
    return invalid==0;
}
} // namespace

struct rw_sr_runtime {std::shared_ptr<Runtime> state;};
struct rw_sr_session {std::shared_ptr<Runtime> owner;std::shared_ptr<Session> state;};
struct rw_sr_job {std::shared_ptr<Runtime> owner;std::shared_ptr<Job> state;};

namespace {
void validate_runtime_desc(const rw_sr_runtime_desc* desc) {
    require(desc ? RW_OK : RW_INVALID_ARGUMENT,"SR runtime descriptor is required");
    require(correct_header(desc->struct_size,desc->abi_version,offsetof(rw_sr_runtime_desc,options)) ? RW_OK : RW_ABI_MISMATCH,
            "SR runtime descriptor ABI mismatch");
    constexpr auto eager_profile=RW_SR_CUDA_GRAPH|RW_SR_STAGE_TIMINGS;
    if(desc->struct_size>=sizeof(*desc) && (desc->options&eager_profile)==eager_profile)
        throw std::invalid_argument("SR stage timings require eager kernel dispatch");
    if(desc->device_ordinal<0 || !desc->width || desc->width>4096 || !desc->height || desc->height>4096 ||
       !desc->execution_slots || !desc->max_sessions)
        throw Failure{RW_INVALID_ARGUMENT,"invalid SR geometry, device or session counts"};
}
}

extern "C" {
const char* rw_sr_last_error() {return error_text;}

rw_status_t rw_sr_runtime_create(const rw_sr_runtime_desc* desc,rw_sr_runtime_t* out) {
    return boundary([&]() -> rw_status_t {
        if(!out) return RW_INVALID_ARGUMENT;
        *out=nullptr;
        validate_runtime_desc(desc);
        if(!desc->weights_path || !desc->filter_path) return RW_INVALID_ARGUMENT;
        *out=new rw_sr_runtime{std::make_shared<Runtime>(*desc)};
        return RW_OK;
    });
}

rw_status_t rw_sr_runtime_create_from_memory(const rw_sr_runtime_desc* desc,const void* weights,
    uint64_t weights_bytes,const void* filter,uint64_t filter_bytes,rw_sr_runtime_t* out) {
    return boundary([&]() -> rw_status_t {
        if(!out) return RW_INVALID_ARGUMENT;
        *out=nullptr;
        validate_runtime_desc(desc);
        if(!weights || weights_bytes!=sr::packed_bytes || !filter || filter_bytes!=sr::filter_elements*2)
            return RW_INVALID_ARGUMENT;
        *out=new rw_sr_runtime{std::make_shared<Runtime>(*desc,
            std::span(static_cast<const std::byte*>(weights),std::size_t(weights_bytes)),
            std::span(static_cast<const std::byte*>(filter),std::size_t(filter_bytes)))};
        return RW_OK;
    });
}

void rw_sr_runtime_release(rw_sr_runtime_t handle) {
    if(!handle) return;
    {std::lock_guard lock(handle->state->mutex);handle->state->accepting=false;}
    delete handle;
}

rw_status_t rw_sr_runtime_get_memory_info(rw_sr_runtime_t handle,rw_sr_memory_info* out) {
    return boundary([&]() -> rw_status_t {
        if(!handle || !out) return RW_INVALID_ARGUMENT;
        if(!correct_header(out->struct_size,out->abi_version,sizeof(*out))) return RW_ABI_MISMATCH;
        auto& runtime=*handle->state;
        std::lock_guard lock(runtime.mutex);
        const auto sessions=std::count_if(runtime.sessions.begin(),runtime.sessions.end(),
                                          [](const auto& session){return !session.expired();});
        const uint64_t model=sr::packed_bytes+sr::filter_elements*2;
        const uint64_t slots=runtime.slots.size()*runtime.slot_bytes;
        const uint64_t histories=sessions*runtime.session_bytes;
        *out={sizeof(*out),RW_ABI_VERSION,model,slots,histories,model+slots+histories,
              runtime.slots.size()*runtime.staging_bytes()};
        return RW_OK;
    });
}

rw_status_t rw_sr_session_create(rw_sr_runtime_t handle,uint64_t epoch,rw_sr_session_t* out) {
    return boundary([&]() -> rw_status_t {
        if(!out) return RW_INVALID_ARGUMENT;
        *out=nullptr;
        if(!handle || !epoch) return RW_INVALID_ARGUMENT;
        auto& runtime=*handle->state;
        std::lock_guard lock(runtime.mutex);
        if(!runtime.accepting) return RW_NOT_READY;
        runtime.activate();
        for(auto& slot:runtime.slots) runtime.poll(*slot);
        std::erase_if(runtime.sessions,[](const auto& ref){return ref.expired();});
        if(runtime.sessions.size()>=runtime.max_sessions) return RW_RESOURCE_EXHAUSTED;
        auto session=std::make_shared<Session>(runtime.ordinal);
        require(rw_state_guard_create(epoch,0,&session->guard),"SR session transaction creation failed");
        for(auto& history:session->history) history.allocate(runtime.geometry);
        runtime.sessions.push_back(session);
        *out=new rw_sr_session{handle->state,std::move(session)};
        return RW_OK;
    });
}

rw_status_t rw_sr_session_reset(rw_sr_session_t handle,uint64_t epoch) {
    return boundary([&]() -> rw_status_t {
        if(!handle) return RW_INVALID_ARGUMENT;
        std::lock_guard lock(handle->owner->mutex);
        const auto status=rw_state_guard_reset(handle->state->guard,epoch,0);
        if(status==RW_OK) handle->state->phase=0;
        return status;
    });
}

void rw_sr_session_release(rw_sr_session_t handle) {delete handle;}

static bool on_device(const float* data,uint64_t count,std::size_t expected,int device) {
    if(!data || count!=expected || reinterpret_cast<std::uintptr_t>(data)%alignof(float)) return false;
    cudaPointerAttributes attrs{};
    if(cudaPointerGetAttributes(&attrs,data)!=cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attrs.type==cudaMemoryTypeDevice && attrs.device==device;
}

static rw_status_t submit_frame(rw_sr_session_t handle,const rw_sr_frame* frame,
                                const rw_sr_cuda_input* cuda,rw_sr_job_t* out) {
    return boundary([&]() -> rw_status_t {
        const auto started=now_ns();
        if(!out) return RW_INVALID_ARGUMENT;
        *out=nullptr;
        if(!handle || !frame) return RW_INVALID_ARGUMENT;
        if(!correct_header(frame->struct_size,frame->abi_version,sizeof(*frame))) return RW_ABI_MISMATCH;
        if(cuda && !correct_header(cuda->struct_size,cuda->abi_version,sizeof(*cuda))) return RW_ABI_MISMATCH;
        auto& runtime=*handle->owner;
        auto& session=*handle->state;
        const bool reset=frame->expected_predecessor==0;
        const bool missing_guides=!frame->motion && !frame->motion_elements && !frame->depth && !frame->depth_elements;
        if(!cuda && runtime.cuda_io_only) return RW_UNSUPPORTED;
        if(!cuda && (!check_input(frame->color,frame->color_elements,runtime.low*3,true) ||
           (!(reset && missing_guides) && (!check_input(frame->motion,frame->motion_elements,runtime.low*2,false) ||
            !check_input(frame->depth,frame->depth_elements,runtime.low,true))))) return RW_INVALID_ARGUMENT;
        std::lock_guard lock(runtime.mutex);
        if(!runtime.accepting) return RW_NOT_READY;
        runtime.activate();
        if(cuda && (!on_device(frame->color,frame->color_elements,runtime.low*3,runtime.ordinal) ||
           (!(reset && missing_guides) && (!on_device(frame->motion,frame->motion_elements,runtime.low*2,runtime.ordinal) ||
            !on_device(frame->depth,frame->depth_elements,runtime.low,runtime.ordinal))))) return RW_INVALID_ARGUMENT;
        for(auto& slot:runtime.slots) runtime.poll(*slot);
        if(session.inflight) return RW_NOT_READY;
        const auto found=std::find_if(runtime.slots.begin(),runtime.slots.end(),[](const auto& slot){return !slot->job;});
        if(found==runtime.slots.end()) return RW_NOT_READY;
        auto& slot=**found;
        auto job=std::make_shared<Job>();
        job->session=handle->state;
        job->slot=found-runtime.slots.begin();
        job->cuda_io=cuda!=nullptr;
        job->info={sizeof(rw_sr_job_info),RW_ABI_VERSION,RW_NOT_READY,0,frame->frame_id,frame->epoch,started,0,0,0};
        auto result=std::make_unique<rw_sr_job>(rw_sr_job{handle->owner,job});
        const auto admitted=rw_state_guard_begin(session.guard,frame->epoch,frame->expected_predecessor,frame->frame_id,&job->ticket);
        if(admitted!=RW_OK) return admitted;
        session.inflight=true;
        slot.job=job;
        // The slot keeps the input snapshot and output at graph-stable addresses.
        try {
            auto* device=slot.input.data<float>();
            if(cuda) {
                if(cuda->ready_event)
                    check(cudaStreamWaitEvent(slot.stream,static_cast<cudaEvent_t>(cuda->ready_event),0));
                check(cudaEventRecord(slot.start,slot.stream));
                check(cudaMemsetAsync(slot.invalid.address,0,4,slot.stream));
                snapshot_input<<<(runtime.low*(missing_guides ? 3 : 6)+255)/256,256,0,slot.stream>>>(
                    frame->color,frame->motion,frame->depth,device,runtime.low,slot.invalid.data<int>());
                check(cudaGetLastError());
            } else {
                auto* host=static_cast<float*>(slot.host);
                std::copy_n(frame->color,runtime.low*3,host);
                if(!missing_guides) {
                    std::copy_n(frame->motion,runtime.low*2,host+runtime.low*3);
                    std::copy_n(frame->depth,runtime.low,host+runtime.low*5);
                }
                check(cudaEventRecord(slot.start,slot.stream));
                check(cudaMemcpyAsync(device,host,runtime.low*(missing_guides ? 12 : 24),cudaMemcpyHostToDevice,slot.stream));
            }
            if(runtime.profile_stages) check(cudaEventRecord(slot.stage_ends[0],slot.stream));
            auto& previous=session.history[session.read_index];
            auto& next=session.history[1-session.read_index];
            const sr::PrepareInput input{runtime.width,runtime.height,device,
                missing_guides ? nullptr : device+runtime.low*3,missing_guides ? nullptr : device+runtime.low*5,
                previous.colors.data<uint16_t>(),previous.hidden.data<uint16_t>(),0,0,2,2,0,0,1,reset};
            const sr::ReconstructionOutput output{slot.rgba.data<float>(),next.colors.data<uint16_t>(),
                next.luma.data<uint16_t>(),next.hidden.data<uint16_t>()};
            runtime.enqueue(slot,input,previous,output);
            const auto colors=pixels(runtime.geometry.history)*4,luma=pixels(runtime.geometry.luma),hidden=pixels(runtime.geometry.neural)*4;
            if(!cuda) check(cudaMemsetAsync(slot.invalid.address,0,4,slot.stream));
            validate_result<<<(std::max({runtime.out,colors,luma,hidden})+255)/256,256,0,slot.stream>>>(
                slot.rgba.data<float>(),runtime.out,
                next.colors.data<__half>(),colors,next.luma.data<__half>(),luma,next.hidden.data<__half>(),hidden,slot.invalid.data<int>());
            check(cudaGetLastError());
            if(!cuda)
                check(cudaMemcpyAsync(runtime.output(slot),slot.rgba.address,runtime.out*4,cudaMemcpyDeviceToHost,slot.stream));
            check(cudaMemcpyAsync(runtime.validation(slot),slot.invalid.address,4,cudaMemcpyDeviceToHost,slot.stream));
            check(cudaEventRecord(slot.completion,slot.stream));
            job->info.device_enqueued_ns=now_ns();
        } catch(...) {
            job->released=true;
            if(cudaStreamSynchronize(slot.stream)==cudaSuccess) runtime.retire(slot,RW_INTERNAL_ERROR);
            else {
                runtime.accepting=false;
                if(cuda) throw Failure{RW_INTERNAL_ERROR,"CUDA submission failed before input access could be drained"};
            }
            throw;
        }
        *out=result.release();
        return RW_OK;
    });
}

rw_status_t rw_sr_submit(rw_sr_session_t handle,const rw_sr_frame* frame,rw_sr_job_t* out) {
    return submit_frame(handle,frame,nullptr,out);
}

rw_status_t rw_sr_submit_cuda(rw_sr_session_t handle,const rw_sr_frame* frame,
                             const rw_sr_cuda_input* cuda,rw_sr_job_t* out) {
    if(!cuda) {
        if(out) *out=nullptr;
        return RW_INVALID_ARGUMENT;
    }
    return submit_frame(handle,frame,cuda,out);
}

static rw_status_t wait_frame(rw_sr_job_t handle,int64_t timeout_ms,float* rgba,uint64_t count,rw_sr_cuda_view* view) {
    return boundary([&]() -> rw_status_t {
        if(!handle || (!rgba && !view) || timeout_ms< -1) return RW_INVALID_ARGUMENT;
        auto& runtime=*handle->owner;
        if(handle->state->cuda_io!=(view!=nullptr)) return RW_UNSUPPORTED;
        if(view && !correct_header(view->struct_size,view->abi_version,sizeof(*view))) return RW_ABI_MISMATCH;
        if(!view && count<runtime.out) return RW_BUFFER_TOO_SMALL;
        if(view) {view->rgba=nullptr;view->rgba_elements=0;}
        auto& slot=*runtime.slots[handle->state->slot];
        const auto start=Clock::now();
        for(;;) {
            {
                std::lock_guard lock(runtime.mutex);
                runtime.activate();
                runtime.poll(slot);
                if(handle->state->info.physical_complete) {
                    if(handle->state->info.status==RW_OK) {
                        if(view) {view->rgba=slot.rgba.data<float>();view->rgba_elements=runtime.out;}
                        else std::copy_n(runtime.output(slot),runtime.out,rgba);
                    }
                    return handle->state->info.status;
                }
            }
            if(timeout_ms<0) {
                const auto status=cudaEventSynchronize(slot.completion);
                if(status!=cudaSuccess) {
                    std::lock_guard lock(runtime.mutex);
                    runtime.accepting=false;
                    check(status);
                }
                continue;
            }
            if(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-start).count()>=timeout_ms)
                return RW_NOT_READY;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
}

rw_status_t rw_sr_job_wait(rw_sr_job_t handle,int64_t timeout_ms,float* rgba,uint64_t count) {
    return wait_frame(handle,timeout_ms,rgba,count,nullptr);
}

rw_status_t rw_sr_job_wait_cuda(rw_sr_job_t handle,int64_t timeout_ms,rw_sr_cuda_view* view) {
    return wait_frame(handle,timeout_ms,nullptr,0,view);
}

rw_status_t rw_sr_job_get_info(rw_sr_job_t handle,rw_sr_job_info* out) {
    return boundary([&]() -> rw_status_t {
        if(!handle || !out) return RW_INVALID_ARGUMENT;
        if(!correct_header(out->struct_size,out->abi_version,sizeof(*out))) return RW_ABI_MISMATCH;
        auto& runtime=*handle->owner;
        std::lock_guard lock(runtime.mutex);
        runtime.activate();
        runtime.poll(*runtime.slots[handle->state->slot]);
        *out=handle->state->info;
        return RW_OK;
    });
}

rw_status_t rw_sr_job_get_stage_timings(rw_sr_job_t handle,rw_sr_stage_timings* out) {
    return boundary([&]() -> rw_status_t {
        if(!handle || !out) return RW_INVALID_ARGUMENT;
        if(!correct_header(out->struct_size,out->abi_version,sizeof(*out))) return RW_ABI_MISMATCH;
        auto& runtime=*handle->owner;
        if(!runtime.profile_stages) return RW_UNSUPPORTED;
        std::lock_guard lock(runtime.mutex);
        runtime.activate();
        runtime.poll(*runtime.slots[handle->state->slot]);
        if(!handle->state->info.physical_complete) return RW_NOT_READY;
        if(!handle->state->stage_timing_valid) return RW_INTERNAL_ERROR;
        const auto& values=handle->state->stage_ms;
        *out={sizeof(*out),RW_ABI_VERSION,values[0],values[1],values[2],values[3],values[4]};
        return RW_OK;
    });
}

rw_status_t rw_sr_job_cancel(rw_sr_job_t handle) {
    return boundary([&]() -> rw_status_t {
        if(!handle) return RW_INVALID_ARGUMENT;
        std::lock_guard lock(handle->owner->mutex);
        if(handle->state->info.physical_complete) return RW_NOT_READY;
        return rw_state_guard_cancel(handle->state->session->guard,handle->state->ticket);
    });
}

void rw_sr_job_release(rw_sr_job_t handle) {
    if(!handle) return;
    {
        std::lock_guard lock(handle->owner->mutex);
        auto& job=*handle->state;
        auto& slot=*handle->owner->slots[job.slot];
        job.released=true;
        if(job.info.physical_complete) slot.job.reset();
        else rw_state_guard_cancel(job.session->guard,job.ticket);
    }
    delete handle;
}
} // extern "C"
