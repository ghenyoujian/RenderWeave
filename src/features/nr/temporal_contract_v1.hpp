#pragma once
// RW067 CPU contract/model only. No native temporal execution or public ABI.
#include "../../core/pts_order.hpp"
#include <openssl/evp.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace rw::nr::temporal_contract_v1 {
// Revision2 adds host sequence identity/epoch checks; this private source
// path/namespace remains stable for the existing RW067 diagnostic harness.
inline constexpr uint32_t contract_version=2;
inline constexpr bool native_temporal_execution_available=false;
using Digest=std::array<uint8_t,32>;
inline Digest sha256(std::span<const uint8_t> bytes){Digest out{};unsigned count=0;
    if(EVP_Digest(bytes.data(),bytes.size(),out.data(),&count,EVP_sha256(),nullptr)!=1||count!=32)
        throw std::runtime_error("contract SHA failed");return out;}
inline bool known(const Digest& d){for(auto b:d)if(b)return true;return false;}
enum class Code:uint32_t { contract_valid=0,version,scope,identity,presence,shape,owner,integrity,
    missing_color,missing_motion,missing_validity,invalid_validity,unmapped_guide,unfrozen_color,
    nonfinite,range,control,pts,history,rng,unknown_resource,overflow,capacity,busy,stale,
    not_ready,cancelled,invalid_output,reset_required,invalid_ticket,native_contract_unavailable };
enum class Presence:uint32_t { absent=0,present=1 };
enum class Format:uint32_t { unknown=0,f32_hwc4,f32_hwc2,f32_hw };
enum class Source:uint32_t { unknown=0,renderer_derived,original_capture,synthetic_control };
enum class Color:uint32_t { unknown=0,controlled_srgb_code_bt709_sdr_full };
enum class Motion:uint32_t { unknown=0,backward_pixels_y_down_no_jitter,backward_uv,forward_pixels };
enum class Depth:uint32_t { unknown=0,device_0_1_noninverted,camera_z_expected };
enum class JitterUnits:uint32_t { unknown=0,input_pixels };
enum class Timeline:uint32_t { unknown=0,generated_MIP_camera,original_capture,synthetic_control };
enum class Scene:uint32_t { unknown=0,bicycle,bonsai,counter,flowers,garden,kitchen,room,stump,treehill };
struct Pts { bool present=false;int64_t ticks=0;int32_t numerator=0,denominator=0; };
inline bool valid_pts(Pts p){return p.present&&p.numerator>0&&p.denominator>0;}
inline bool later(Pts a,Pts b){
    return rw::detail::later_pts(a.ticks,a.numerator,a.denominator,b.ticks,b.numerator,b.denominator);
}
inline bool mul(uint64_t a,uint64_t b,uint64_t& out){if(b&&a>UINT64_MAX/b)return false;out=a*b;return true;}
inline bool add(uint64_t a,uint64_t b,uint64_t& out){if(a>UINT64_MAX-b)return false;out=a+b;return true;}
struct Lease { uint64_t session_id=0,owner_id=0;std::optional<uint64_t> bytes; };
using Owner=std::shared_ptr<const Lease>;
inline bool canonical_owner(const Owner& a,const Owner& b){return a&&b&&a.get()==b.get()&&
    !a.owner_before(b)&&!b.owner_before(a);}
struct Plane {
    Presence presence=Presence::absent;Format format=Format::unknown;Source source=Source::unknown;
    uint32_t width=0,height=0;Owner owner;
    std::shared_ptr<const std::vector<uint8_t>> payload,validity;
    std::optional<Digest> payload_sha256,validity_sha256;
};
struct ModelIdentity { Digest original_dll{},weights{},rng_algorithm{}; };
inline bool operator==(const ModelIdentity& a,const ModelIdentity& b){
    return a.original_dll==b.original_dll&&a.weights==b.weights&&a.rng_algorithm==b.rng_algorithm;
}
struct History {
    Presence presence=Presence::absent;Owner owner;uint64_t frame=0,epoch=0;
    // Null means genuinely unknown, never an arbitrary native codec/descriptor.
    std::optional<Digest> codec,sampler;
};
struct NativeBindings {
    std::optional<int32_t> active_selector;std::optional<bool> semantic_enabled;
    std::optional<Digest> preprocessing,geometry,history_codec,sampler,controls;
};
struct Frame {
    uint32_t version=contract_version;std::string dataset="mip360",scene;
    // Host-defined sequence identity, not an NVIDIA-provided native field.
    // A stable nonzero digest identifies the input timeline within one epoch;
    // it cannot itself detect an unreported scene cut or seek.
    Digest sequence_identity{};
    uint64_t session_id=0,frame_id=0,epoch=0;std::optional<uint64_t> expected_predecessor;
    Pts pts;Timeline timeline=Timeline::unknown;bool reset=false;ModelIdentity identity;
    Plane color,depth,motion;Color color_profile=Color::unknown;
    Motion motion_profile=Motion::unknown;Depth depth_profile=Depth::unknown;
    bool jitter_present=false,exposure_present=false;
    JitterUnits jitter_units=JitterUnits::unknown;
    uint32_t jitter_x_bits=0,jitter_y_bits=0,exposure_scale_bits=0;
    bool jitter_explicitly_disabled=false,exposure_explicit_no_transform=false;
    History history;uint64_t rng_counter_before=0;NativeBindings declared_native;
};
inline Scene scene_id(const std::string& name){
    if(name=="bicycle")return Scene::bicycle;if(name=="bonsai")return Scene::bonsai;
    if(name=="counter")return Scene::counter;if(name=="flowers")return Scene::flowers;
    if(name=="garden")return Scene::garden;if(name=="kitchen")return Scene::kitchen;
    if(name=="room")return Scene::room;if(name=="stump")return Scene::stump;
    if(name=="treehill")return Scene::treehill;return Scene::unknown;
}
inline bool scene_known(const std::string& name){return scene_id(name)!=Scene::unknown;}
inline bool finite_bits(uint32_t word){return (word&0x7f800000u)!=0x7f800000u;}
inline uint32_t word_at(const std::vector<uint8_t>& bytes,size_t at){
    return uint32_t(bytes[at])|(uint32_t(bytes[at+1])<<8)|(uint32_t(bytes[at+2])<<16)|(uint32_t(bytes[at+3])<<24);
}
inline Code plane_contract(const Plane& p,uint64_t session,uint32_t width,uint32_t height,Format format,unsigned channels){
    if(p.presence==Presence::absent){
        return p.payload||p.validity||p.owner||p.payload_sha256||p.validity_sha256?Code::presence:Code::contract_valid;
    }
    if(p.presence!=Presence::present)return Code::presence;
    if(p.width!=width||p.height!=height||p.format!=format||!width||!height)return Code::shape;
    uint64_t elements=0,bytes=0;if(!mul(width,height,elements)||!mul(elements,channels*4,bytes))return Code::overflow;
    if(!p.owner||p.owner->session_id!=session||!p.owner->owner_id)return Code::owner;
    if(!p.owner->bytes)return Code::unknown_resource;
    uint64_t group_bytes=0;if(!add(bytes,elements,group_bytes))return Code::overflow;
    if(*p.owner->bytes<group_bytes||!p.payload||p.payload->size()!=bytes)return Code::shape;
    if(p.source!=Source::renderer_derived&&p.source!=Source::original_capture&&p.source!=Source::synthetic_control)return Code::identity;
    if(!p.payload_sha256||sha256(*p.payload)!=*p.payload_sha256)return Code::integrity;
    if(!p.validity||!p.validity_sha256)return Code::missing_validity;
    if(p.validity->size()!=elements||sha256(*p.validity)!=*p.validity_sha256)return Code::integrity;
    for(uint8_t valid:*p.validity)if(valid!=1)return Code::invalid_validity;
    for(size_t at=0;at<p.payload->size();at+=4)if(!finite_bits(word_at(*p.payload,at)))return Code::nonfinite;
    return Code::contract_valid;
}
inline Code validate_prepared_input(const Frame& f){
    if(f.version!=contract_version)return Code::version;
    if(f.dataset!="mip360"||!scene_known(f.scene))return Code::scope;
    if(!f.session_id||!known(f.sequence_identity)||!known(f.identity.original_dll)||!known(f.identity.weights)||!known(f.identity.rng_algorithm))return Code::identity;
    if(!valid_pts(f.pts)||(f.timeline!=Timeline::generated_MIP_camera&&f.timeline!=Timeline::original_capture&&f.timeline!=Timeline::synthetic_control))return Code::pts;
    if(!f.jitter_present||!f.exposure_present||!f.jitter_explicitly_disabled||!f.exposure_explicit_no_transform||
       f.jitter_units!=JitterUnits::input_pixels||f.jitter_x_bits||f.jitter_y_bits||f.exposure_scale_bits!=0x3f800000u)return Code::control;
    if(f.color.presence!=Presence::present)return Code::missing_color;
    if(f.color_profile!=Color::controlled_srgb_code_bt709_sdr_full)return Code::unfrozen_color;
    const uint32_t w=f.color.width,h=f.color.height;
    auto code=plane_contract(f.color,f.session_id,w,h,Format::f32_hwc4,4);if(code!=Code::contract_valid)return code;
    for(size_t at=0;at<f.color.payload->size();at+=4){const uint32_t word=word_at(*f.color.payload,at);
        if((word&0x7fffffffu)>0x3f800000u||((word>>31)&&(word&0x7fffffffu)))return Code::range;
        if((at/4)%4==3&&word!=0x3f800000u)return Code::range;}
    code=plane_contract(f.depth,f.session_id,w,h,Format::f32_hw,1);if(code!=Code::contract_valid)return code;
    if(f.depth.presence==Presence::present){if(f.depth_profile!=Depth::device_0_1_noninverted)return Code::unmapped_guide;
        for(size_t at=0;at<f.depth.payload->size();at+=4){const uint32_t word=word_at(*f.depth.payload,at);
            if((word&0x7fffffffu)>0x3f800000u||((word>>31)&&(word&0x7fffffffu)))return Code::range;}}
    code=plane_contract(f.motion,f.session_id,w,h,Format::f32_hwc2,2);if(code!=Code::contract_valid)return code;
    if(f.motion.presence==Presence::present&&f.motion_profile!=Motion::backward_pixels_y_down_no_jitter)return Code::unmapped_guide;
    if(f.reset){if(f.expected_predecessor||f.history.presence!=Presence::absent||f.history.owner||f.rng_counter_before)return Code::history;}
    else {
        if(f.motion.presence!=Presence::present)return Code::missing_motion;
        if(!f.expected_predecessor||f.history.presence!=Presence::present||!f.history.owner)return Code::history;
        if(f.history.owner->session_id!=f.session_id||!f.history.owner->owner_id||f.history.epoch!=f.epoch||f.history.frame!=*f.expected_predecessor)return Code::history;
        if(!f.history.owner->bytes)return Code::unknown_resource;
        if(!*f.history.owner->bytes)return Code::history;
        if(f.frame_id<=*f.expected_predecessor)return Code::history;
    }
    // Validates a versioned, controlled CPU envelope; no trusted native profile
    // exists in this revision, irrespective of caller-provided hashes/flags.
    return Code::contract_valid;
}
inline Code native_admission(const Frame&){return Code::native_contract_unavailable;}

struct Budget {
    uint64_t bytes=0,slots=0,sessions=0;
    std::optional<uint64_t> weights,slot_workspace,committed_history,provisional_history,imports,graphs,reserve;
};
struct BudgetResult { Code code;uint64_t known_total=0; };
inline BudgetResult resource_admission(Budget b){
    if(!b.weights||!b.slot_workspace||!b.committed_history||!b.provisional_history||!b.imports||!b.graphs||!b.reserve)return {Code::unknown_resource,0};
    if(!b.slots||!b.sessions)return {Code::capacity,0};
    uint64_t total=*b.weights,per_slot=0,part=0;
    if(!add(*b.slot_workspace,*b.provisional_history,per_slot)||!mul(per_slot,b.slots,part)||!add(total,part,total)||
       !mul(*b.committed_history,b.sessions,part)||!add(total,part,total)||!add(total,*b.imports,total)||
       !add(total,*b.graphs,total)||!add(total,*b.reserve,total))return {Code::overflow,0};
    return {total<=b.bytes?Code::contract_valid:Code::capacity,total};
}

// CPU ownership simulator. IDs/leases are not CUDA handles or completion proofs.
class SlotLedger {
    friend class SessionModel;
    struct Use{uint64_t session=0,ticket=0;std::array<Owner,4> readers{};Owner writer;};
    std::array<Use,8> uses_{};mutable std::mutex mutex_;
    static bool same_owner(const Owner& a,const Owner& b){return a&&b&&
        (a==b||(a->session_id==b->session_id&&a->owner_id==b->owner_id));}
    bool claim(size_t slot,uint64_t session,uint64_t ticket,const std::array<Owner,4>& readers,Owner writer){std::lock_guard lock(mutex_);
        if(slot>=uses_.size()||uses_[slot].ticket)return false;
        for(const auto& use:uses_)if(use.ticket){
            // Old epochs retain the same read/write claims until physical
            // completion. Immutable read/read sharing is safe; every write
            // conflict, in either direction, must remain unavailable.
            if(same_owner(writer,use.writer))return false;
            for(const auto& old_read:use.readers)if(same_owner(writer,old_read))return false;
            for(const auto& new_read:readers)if(same_owner(new_read,use.writer))return false;
        }
        uses_[slot]={session,ticket,readers,std::move(writer)};return true;}
    bool release(size_t slot,uint64_t session,uint64_t ticket){std::lock_guard lock(mutex_);
        if(slot>=uses_.size()||uses_[slot].session!=session||uses_[slot].ticket!=ticket)return false;uses_[slot]={};return true;}
public:
    SlotLedger()=default;
    bool busy(size_t slot)const{std::lock_guard lock(mutex_);return slot<uses_.size()&&uses_[slot].ticket;}
};
struct Snapshot { uint64_t session=0,epoch=0,generation=0,counter=0,issued_total=0;uint32_t width=0,height=0;std::optional<uint64_t> predecessor;Pts pts;
    Scene scene=Scene::unknown;Timeline timeline=Timeline::unknown;Digest sequence_identity{};
    History history;bool reset_required=false;size_t pending=0; };
class SessionModel {
    struct Pending {uint64_t ticket,epoch,generation,frame,counter;size_t slot;std::optional<uint64_t> predecessor;
        Pts pts;Frame retained_input;History candidate;bool issued=false,cancelled=false,timed_out=false;};
    struct RetainedGroup {std::array<std::optional<Pending>,8> entries;std::shared_ptr<RetainedGroup> quarantine_self;};
    Snapshot state_;ModelIdentity identity_;SlotLedger& slots_;uint64_t next_=1;
    std::shared_ptr<RetainedGroup> group_=std::make_shared<RetainedGroup>();mutable std::mutex mutex_;
    Pending* find(uint64_t ticket){for(auto& p:group_->entries)if(p&&p->ticket==ticket)return &*p;return nullptr;}
public:
    SessionModel(uint64_t session,uint64_t epoch,ModelIdentity identity,SlotLedger& slots):identity_(identity),slots_(slots){
        if(!session)throw std::invalid_argument("model session ID required");state_.session=session;state_.epoch=epoch;state_.generation=1;}
    SessionModel(const SessionModel&)=delete;SessionModel& operator=(const SessionModel&)=delete;
    ~SessionModel()noexcept{if(state_.pending)group_->quarantine_self=group_;}
    Snapshot snapshot()const{std::lock_guard lock(mutex_);return state_;}
    Code begin(Frame f,size_t slot,History candidate,uint64_t& ticket){std::lock_guard lock(mutex_);ticket=0;
        const auto input=validate_prepared_input(f);if(input!=Code::contract_valid)return input;
        if(f.session_id!=state_.session||!(f.identity==identity_))return Code::identity;
        if(state_.reset_required)return Code::reset_required;
        if(f.epoch!=state_.epoch||f.expected_predecessor!=state_.predecessor)return Code::stale;
        if(f.reset!=!state_.predecessor||f.rng_counter_before!=state_.counter)return Code::rng;
        if(state_.predecessor&&(f.color.width!=state_.width||f.color.height!=state_.height))return Code::reset_required;
        if(state_.predecessor&&(scene_id(f.scene)!=state_.scene||f.timeline!=state_.timeline||
                                f.sequence_identity!=state_.sequence_identity))return Code::reset_required;
        if(state_.predecessor&&!later(f.pts,state_.pts))return Code::pts;
        if(state_.predecessor&&(!canonical_owner(f.history.owner,state_.history.owner)||f.history.codec!=state_.history.codec||f.history.sampler!=state_.history.sampler))return Code::history;
        // Keep the actual committed control block, never a caller's same-get()
        // shared_ptr with a different/non-owning deleter and ownership group.
        if(state_.predecessor)f.history=state_.history;
        if(candidate.presence!=Presence::present||!candidate.owner||candidate.owner->session_id!=state_.session||
           !candidate.owner->owner_id||candidate.epoch!=f.epoch||candidate.frame!=f.frame_id)return Code::owner;
        if(!candidate.owner->bytes)return Code::unknown_resource;
        if(!*candidate.owner->bytes)return Code::history;
        for(const Owner* owner:{&f.color.owner,&f.depth.owner,&f.motion.owner,&f.history.owner})
            if(*owner&&candidate.owner->owner_id==(*owner)->owner_id)return Code::history;
        if(state_.predecessor&&(candidate.codec!=state_.history.codec||candidate.sampler!=state_.history.sampler))return Code::history;
        if(state_.counter==UINT64_MAX||next_==UINT64_MAX)return Code::overflow;
        for(const auto& p:group_->entries)if(p&&p->epoch==state_.epoch)return Code::busy;
        for(auto& p:group_->entries)if(!p){
            const std::array<Owner,4> readers={f.color.owner,f.depth.owner,f.motion.owner,f.history.owner};
            if(!slots_.claim(slot,state_.session,next_,readers,candidate.owner))return Code::busy;
            // All input/history/next-history owners are installed before issue.
            p=Pending{next_,f.epoch,state_.generation,f.frame_id,state_.counter+1,slot,f.expected_predecessor,f.pts,std::move(f),std::move(candidate)};
            ticket=next_++;++state_.pending;return Code::contract_valid;
        }return Code::capacity;
    }
    Code issue(uint64_t ticket){std::lock_guard lock(mutex_);auto* p=find(ticket);if(!p)return Code::invalid_ticket;
        if(p->epoch!=state_.epoch||p->generation!=state_.generation||p->predecessor!=state_.predecessor)return Code::stale;
        if(p->issued||p->cancelled)return Code::invalid_ticket;if(state_.issued_total==UINT64_MAX)return Code::overflow;
        p->issued=true;++state_.issued_total;return Code::contract_valid;}
    Code observe_timeout(uint64_t ticket){std::lock_guard lock(mutex_);auto* p=find(ticket);if(!p)return Code::invalid_ticket;p->timed_out=true;return Code::not_ready;}
    Code cancel(uint64_t ticket){std::lock_guard lock(mutex_);auto* p=find(ticket);if(!p)return Code::invalid_ticket;p->cancelled=true;return Code::contract_valid;}
    Code reset(uint64_t epoch){std::lock_guard lock(mutex_);if(epoch<=state_.epoch)return Code::stale;if(state_.generation==UINT64_MAX)return Code::overflow;
        ++state_.generation;state_.epoch=epoch;state_.counter=0;state_.width=state_.height=0;state_.predecessor.reset();state_.history={};state_.pts={};
        state_.scene=Scene::unknown;state_.timeline=Timeline::unknown;state_.sequence_identity={};state_.reset_required=false;
        // Old pending input/history/slot leases intentionally remain installed.
        return Code::contract_valid;}
    Code complete_cpu_observation(uint64_t ticket,bool physically_confirmed,bool output_valid){std::lock_guard lock(mutex_);
        auto* p=find(ticket);if(!p)return Code::invalid_ticket;if(!physically_confirmed)return Code::not_ready;
        Code result=Code::contract_valid;
        const bool current=p->epoch==state_.epoch&&p->generation==state_.generation&&p->predecessor==state_.predecessor;
        if(!current)result=Code::stale;else if(p->cancelled)result=Code::cancelled;
        else if(!output_valid||!p->issued)result=Code::invalid_output;
        if(current&&result!=Code::contract_valid&&p->issued)state_.reset_required=true;
        if(result==Code::contract_valid){state_.history=p->candidate;state_.counter=p->counter;
            state_.predecessor=p->frame;state_.pts=p->pts;state_.width=p->retained_input.color.width;state_.height=p->retained_input.color.height;
            state_.scene=scene_id(p->retained_input.scene);state_.timeline=p->retained_input.timeline;state_.sequence_identity=p->retained_input.sequence_identity;}
        if(!slots_.release(p->slot,state_.session,p->ticket))throw std::logic_error("CPU slot owner mismatch");
        for(auto& entry:group_->entries)if(entry&&entry->ticket==ticket){entry.reset();break;}--state_.pending;return result;
    }
    // Abandoning a pending CPU model retains the bounded group by a self-cycle;
    // it cannot return the slot/owners for reuse. This is deliberate quarantine,
    // not actual CUDA cancellation or a recovery qualification.
};
} // namespace rw::nr::temporal_contract_v1
