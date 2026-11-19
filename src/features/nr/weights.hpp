#pragma once
// RW040 production model registration. No CUDA owner/cache back-references.
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace rw::native_nr {

struct ModelRegistrationRequest {
    std::filesystem::path weights_path, parameter_manifest_path, plan_path;
    std::string weights_sha256, parameter_manifest_sha256, plan_sha256;
    std::string lowering_recipe_sha256;
};
struct ModelFingerprint {
    std::string original_file_sha256, original_resource_sha256;
    std::string topology_sha256, exact_plan_sha256, parameter_manifest_sha256;
    std::string quantization_precision_layout_sha256, lowering_recipe_sha256;
    bool operator==(const ModelFingerprint&) const = default;
};

class ModelRegistry;
class VerifiedModelRegistration final {
    struct Impl;
    std::shared_ptr<const Impl> implementation_;
    explicit VerifiedModelRegistration(std::shared_ptr<const Impl>);
    friend class ModelRegistry;
public:
    const ModelFingerprint& fingerprint() const noexcept;
    const std::filesystem::path& weights_path() const noexcept;
    const std::string& network() const noexcept;
    // Source-owned W bound is planning evidence, not measured allocation.
    uint64_t source_weight_bound_bytes() const noexcept;
    uint64_t source_single_slot_bound_bytes() const noexcept;
};
using RegisteredModelOwner=std::shared_ptr<const VerifiedModelRegistration>;

class ModelRegistry final {
public:
    // Verifies actual files + trusted R02 NVIDIA receipt before canonicalizing.
    // Same fingerprint returns the same live shared_ptr control block. A source
    // receipt is CPU-only: it owns neither Weights caches nor a device pool.
    static RegisteredModelOwner register_original_ht(const ModelRegistrationRequest&);
    static bool is_registered_owner(const RegisteredModelOwner&) noexcept;
};

} // namespace rw::native_nr
