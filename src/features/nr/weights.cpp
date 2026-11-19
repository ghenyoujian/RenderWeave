#include "weights.hpp"
#include "native_graph.hpp"
#include "../../backends/cuda/native_nr/lowering_recipe.hpp"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

namespace rw::native_nr {
namespace {
using Json=nlohmann::json;
namespace fs=std::filesystem;
constexpr const char* original_weights="4f236d8a58a48653cde064d4a526d194cd1d9f97c65b26bba3e2e49e598ce353";
constexpr const char* original_dll="2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f";
constexpr const char* original_resource="836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4";
constexpr const char* original_source="3beef7935a8ef976b47ab78b906e75600c52d2de";
void require(bool condition,const char* message) { if(!condition)throw std::invalid_argument(message); }
bool valid_sha(const std::string& text) {
    return text.size()==64 && std::all_of(text.begin(),text.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
std::string hex(std::span<const uint8_t> bytes) {
    constexpr char characters[]="0123456789abcdef";std::string result;
    for(auto byte:bytes){result+=characters[byte>>4];result+=characters[byte&15];}return result;
}
std::string text_sha(const std::string& text) {
    std::array<uint8_t,32> output{};unsigned count=0;
    if(EVP_Digest(text.data(),text.size(),output.data(),&count,EVP_sha256(),nullptr)!=1 || count!=32)
        throw std::runtime_error("model registration SHA256 failed");
    return hex(output);
}
std::string file_sha(const fs::path& path) {
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    if(!digest || EVP_DigestInit_ex(digest.get(),EVP_sha256(),nullptr)!=1)
        throw std::runtime_error("model registration SHA256 init failed");
    std::ifstream input(path,std::ios::binary);require(bool(input),"cannot read registered model material");
    std::array<char,65536> bytes{};
    while(input) {
        input.read(bytes.data(),bytes.size());
        if(input.gcount() && EVP_DigestUpdate(digest.get(),bytes.data(),input.gcount())!=1)
            throw std::runtime_error("model registration SHA256 update failed");
    }
    require(input.eof(),"registered model material read failed");
    std::array<uint8_t,32> output{};unsigned count=0;
    if(EVP_DigestFinal_ex(digest.get(),output.data(),&count)!=1 || count!=32)
        throw std::runtime_error("model registration SHA256 finish failed");
    return hex(output);
}
bool within(const fs::path& path,const fs::path& root) {
    const auto relative=path.lexically_relative(root);
    return !relative.empty() && !relative.is_absolute() && *relative.begin()!="..";
}
fs::path checked_path(const fs::path& path,const char* declared_root) {
    require(path.is_absolute(),"registered model paths must be absolute");
    const auto resolved=fs::canonical(path);require(fs::is_regular_file(resolved),"registered model material is not a file");
    if(const char* root=std::getenv(declared_root);root && *root) {
        const auto canonical=fs::canonical(root);
        require(canonical==fs::path(root) && within(resolved,canonical),"registered model path escapes its declared root");
    }
    return resolved;
}
Json json_file(const fs::path& path,uint64_t maximum) {
    require(fs::file_size(path)<=maximum,"registered model metadata exceeds parser bound");
    std::vector<std::set<std::string>> keys;
    auto callback=[&](int depth,Json::parse_event_t event,Json& value) {
        require(depth<=48,"registered metadata nesting exceeds bound");
        if(event==Json::parse_event_t::object_start)keys.emplace_back();
        else if(event==Json::parse_event_t::object_end)keys.pop_back();
        else if(event==Json::parse_event_t::key)
            require(!keys.empty() && keys.back().insert(value.get<std::string>()).second,"duplicate registered metadata key");
        return true;
    };
    std::ifstream input(path);return Json::parse(input,callback);
}
std::mutex& registry_mutex(){static auto* mutex=new std::mutex;return *mutex;}
std::map<std::string,std::weak_ptr<const VerifiedModelRegistration>>& registrations(){
    static auto* values=new std::map<std::string,std::weak_ptr<const VerifiedModelRegistration>>;return *values;
}
}

struct VerifiedModelRegistration::Impl {
    ModelFingerprint fingerprint;
    fs::path weights;
    std::string network="ht";
    uint64_t source_weights=0,source_slot=0;
};
VerifiedModelRegistration::VerifiedModelRegistration(std::shared_ptr<const Impl> value):implementation_(std::move(value)){}
const ModelFingerprint& VerifiedModelRegistration::fingerprint()const noexcept{return implementation_->fingerprint;}
const fs::path& VerifiedModelRegistration::weights_path()const noexcept{return implementation_->weights;}
const std::string& VerifiedModelRegistration::network()const noexcept{return implementation_->network;}
uint64_t VerifiedModelRegistration::source_weight_bound_bytes()const noexcept{return implementation_->source_weights;}
uint64_t VerifiedModelRegistration::source_single_slot_bound_bytes()const noexcept{return implementation_->source_slot;}

RegisteredModelOwner ModelRegistry::register_original_ht(const ModelRegistrationRequest& request) {
    require(valid_sha(request.weights_sha256) && valid_sha(request.parameter_manifest_sha256) && valid_sha(request.plan_sha256),
        "original model registration requires full lowercase SHA256");
    const auto current_recipe=lowering_recipe_json("ht").at("recipe_fingerprint_sha256").get<std::string>();
    require(request.weights_sha256==original_weights && request.lowering_recipe_sha256==current_recipe,
        "shared weights admit only frozen original HT and the current lowering recipe");
    const auto weights=checked_path(request.weights_path,"PROJECT_DOWNLOADED_WEIGHTS_ROOT");
    const auto manifest=checked_path(request.parameter_manifest_path,"PROJECT_STORAGE_ROOT");
    const auto plan_path=checked_path(request.plan_path,"PROJECT_STORAGE_ROOT");
    require(file_sha(weights)==request.weights_sha256 && file_sha(manifest)==request.parameter_manifest_sha256 &&
        file_sha(plan_path)==request.plan_sha256,"actual registered model or exact-plan file SHA256 differs");
    const auto receipt=json_file(manifest,8*1024*1024),document=json_file(plan_path,16*1024*1024);
    require(receipt.at("artifact_kind")=="native_nr_parameter_inventory" && receipt.at("variant")=="ht" &&
        receipt.at("dll_sha256")==original_dll && receipt.at("resource_sha256")==original_resource &&
        receipt.at("parameters_sha256")==original_weights && fs::canonical(receipt.at("parameters_path").get<std::string>())==weights &&
        receipt.at("authenticode").at("status")=="verified_nvidia" && receipt.at("extractor").at("commit")==original_source &&
        receipt.at("extractor").at("inverse_parameter_packing")=="byte_exact" &&
        receipt.at("parameter_updates_allowed")==false && receipt.at("inference_only")==true,
        "trusted original NVIDIA HT parameter receipt is incomplete");
    require(document.at("parameter_inventory").at("sha256")==original_weights,
        "exact source plan is not bound to original HT material");
    const auto parsed=SourcePlan::load(plan_path);
    require(parsed.network=="ht","shared model registration is HT only");
    require(document.at("state").at("mode")=="stateless_reset_each_frame" &&
        document.at("state").at("history_implemented")==false,
        "shared source registry cannot silently admit a temporal/history plan");
    const auto& geometry=document.at("geometry");
    const Json actual_graph={{"network",parsed.network},{"geometry",{{"width",geometry.at("width")},
        {"height",geometry.at("height")},{"pad_x",geometry.at("pad_x")},{"pad_y",geometry.at("pad_y")}}},
        {"nodes",document.at("nodes")}};
    // Match tools/graph/nr_semantic_plan.py: sorted keys, compact separators,
    // ensure_ascii=True. A well-formed claimed hash is not topology evidence.
    require(text_sha(actual_graph.dump(-1,' ',true))==parsed.semantic_graph_sha256,
        "declared semantic graph fingerprint differs from actual node/geometry content");
    const auto& resources=document.at("resources");
    require(resources.at("W_registered_f32_module_bytes").is_number_unsigned() || resources.at("W_registered_f32_module_bytes").is_number_integer(),
        "source weight bound is not integral");
    const auto source_weights=resources.at("W_registered_f32_module_bytes").get<uint64_t>();
    require(source_weights==583020328 && source_weights<=parsed.modelled_memory_bound,"source W budget is not the registered HT bound");
    auto value=std::make_shared<VerifiedModelRegistration::Impl>();
    value->weights=weights;value->source_weights=source_weights;value->source_slot=parsed.modelled_memory_bound;
    auto& fingerprint=value->fingerprint;
    fingerprint.original_file_sha256=original_weights;fingerprint.original_resource_sha256=original_resource;
    fingerprint.topology_sha256=parsed.semantic_graph_sha256;
    // A plan's runtime/job/provenance timestamps are evidence, not geometry.
    // Canonicalize exact semantic geometry/state rather than duplicate W for
    // byte-different receipts of the same original graph and native plan.
    fingerprint.exact_plan_sha256=text_sha(Json{{"network",parsed.network},{"input",document.at("input")},
        {"geometry",document.at("geometry")},{"state",document.at("state")},
        {"topology",parsed.semantic_graph_sha256},{"source",original_source}}.dump());
    fingerprint.parameter_manifest_sha256=request.parameter_manifest_sha256;
    fingerprint.lowering_recipe_sha256=request.lowering_recipe_sha256;
    fingerprint.quantization_precision_layout_sha256=text_sha(
        "original_ht_fp8_e4m3fn_decode_exact_to_f32;original_f16_exact_to_f32;f32_bits_preserved;"
        "FE_TONEAREST_derived;FP32_PEDANTIC_noTF32;post_RGB_Sgemm_32MiB_v2;"
        "post_compose_residual_gain_1/4;stateless_reset_no_history");
    const auto key=text_sha(Json{{"parameters",fingerprint.original_file_sha256},{"resource",fingerprint.original_resource_sha256},
        {"topology",fingerprint.topology_sha256},{"exact_plan",fingerprint.exact_plan_sha256},
        {"precision_layout",fingerprint.quantization_precision_layout_sha256},{"lowering",fingerprint.lowering_recipe_sha256}}.dump());
    std::lock_guard lock(registry_mutex());auto& registry=registrations();
    if(const auto found=registry.find(key);found!=registry.end())if(auto owner=found->second.lock()){
        require(owner->source_weight_bound_bytes()==source_weights && owner->source_single_slot_bound_bytes()==parsed.modelled_memory_bound,
            "same semantic graph has conflicting source resource contracts");
        return owner;
    }
    for(auto it=registry.begin();it!=registry.end();)if(it->second.expired())it=registry.erase(it);else ++it;
    require(registry.size()<64,"live original-model registration limit exceeded");
    RegisteredModelOwner owner(new VerifiedModelRegistration(std::move(value)));
    registry.insert_or_assign(key,owner);return owner;
}
bool ModelRegistry::is_registered_owner(const RegisteredModelOwner& candidate)noexcept{
    if(!candidate)return false;
    try{std::lock_guard lock(registry_mutex());for(const auto& [key,weak]:registrations()){
        (void)key;if(auto actual=weak.lock())if(actual.get()==candidate.get() && !actual.owner_before(candidate) && !candidate.owner_before(actual))return true;
    }}catch(...){}return false;
}
} // namespace rw::native_nr
