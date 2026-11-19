#pragma once
#include "native_nr.hpp"
#include <nlohmann/json.hpp>
#include <limits>
#include <stdexcept>
#include <string_view>
#if defined(RW_NR_PORTABLE_CUBLAS_PROFILE)
#include <openssl/sha.h>
#include <array>
#endif

namespace rw::native_nr {
inline bool uses_post_projection_v2(std::string_view network) { return network=="ht"; }
inline constexpr int nr_required_cublas_version() {
#if defined(RW_NR_PORTABLE_CUBLAS_PROFILE)
    return CUBLAS_VERSION;
#else
    return 130002;
#endif
}
inline uint64_t lowering_memory_bound(uint64_t source_bound,bool prepare_post) {
    const uint64_t extra=prepare_post?post_projection_workspace_bytes:0;
    if(source_bound>std::numeric_limits<uint64_t>::max()-extra)
        throw std::overflow_error("native post32 lowering allocation bound overflows");
    return source_bound+extra;
}
inline nlohmann::json lowering_recipe_json(std::string_view network) {
    if(!uses_post_projection_v2(network)) return {{"recipe_id","mauri870_3beef793_small_legacy_fp32_v1"},
        {"recipe_version",1},{"post32_enabled",false},{"additional_device_workspace_bytes_per_slot",0},
        {"support_status","experimental_unqualified"}};
    nlohmann::json profile={{"recipe_id","mauri870_3beef793_ht_h200_fp32_post32_quarter_v3"},{"recipe_version",3},
        {"recipe_file","profiles/lowering/nr_ht_source_f32_post32_quarter_v3.json"},
        {"recipe_fingerprint_sha256","228729592f97c3c866edf3c10deefdbe56845f5cbc2cb44b2594cb1e661fd00f"},
        {"post_residual_scale",post_residual_scale},
        {"source_postprocessing_sha256","95b602aeef03eca86d27f6e62abcd7e7c90acc33d887294945ee8c84da28188a"},
        {"generic_gemm_api","cublasGemmEx/CUBLAS_COMPUTE_32F_PEDANTIC"},{"generic_workspace_bytes",4194304},
        {"post_projection_api","cublasSgemm"},{"post_projection_math_mode","CUBLAS_PEDANTIC_MATH"},
        {"post_projection_workspace_bytes",post_projection_workspace_bytes},{"post32_enabled",true},
        {"additional_device_workspace_bytes_per_slot",post_projection_workspace_bytes},
        {"owner","Context/execution slot"},{"prepare_policy","before graph submission"},
        {"earlier_gemm_configuration_changed",false},{"precision_or_model_changed",false},
        {"required_cublas_version",nr_required_cublas_version()},{"support_status","experimental_unqualified"}};
#if defined(RW_NR_PORTABLE_CUBLAS_PROFILE)
    profile["recipe_id"]="mauri870_3beef793_ht_cuda_fp32_post32_quarter_v3";
    profile["source_recipe_fingerprint_sha256"]=profile.at("recipe_fingerprint_sha256");
    profile.erase("recipe_fingerprint_sha256");
    const auto recipe=profile.dump();
    std::array<unsigned char,32> hash{};
    if(!SHA256(reinterpret_cast<const unsigned char*>(recipe.data()),recipe.size(),hash.data()))
        throw std::runtime_error("portable NR lowering recipe SHA256 failed");
    std::string fingerprint;
    for(const auto byte:hash) {
        constexpr char hex[]="0123456789abcdef";
        fingerprint+=hex[byte>>4];
        fingerprint+=hex[byte&15];
    }
    profile["recipe_fingerprint_sha256"]=std::move(fingerprint);
#endif
    return profile;
}
inline nlohmann::json post_projection_json(const Context& context) {
    const auto info=context.post_projection_info();
    return {{"preparation_attempted",info.preparation_attempted},{"prepared",info.prepared},
        {"initialization_failed",info.initialization_failed},{"admitted_workspace_bytes",info.admitted_workspace_bytes},
        {"allocated_workspace_bytes",info.allocated_workspace_bytes},{"successful_preparations",info.successful_preparations}};
}
} // namespace rw::native_nr
