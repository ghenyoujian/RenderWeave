#pragma once

// Internal native CUDA inference interface. No tensor framework/provider is used.
// This first path reproduces the pinned stateless source model; native vendor
// equivalence and temporal behavior require their own evidence.
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "shared_weights.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace rw::native_nr {

struct DeviceState;
struct DeviceStorage;
inline constexpr uint64_t post_projection_workspace_bytes = 32ull * 1024 * 1024;
// Locked 3beef793 pre_post.compose; same-input public vendor comparison RW651.
inline constexpr float post_residual_scale = 0.25f;

struct Tensor {
    std::shared_ptr<DeviceStorage> storage;
    std::vector<int64_t> shape;
    float* data() const;
    size_t numel() const;
    bool empty() const { return !storage; }
    Tensor reshape(std::vector<int64_t> dimensions) const;
};

class Context {
    std::shared_ptr<DeviceState> state_;
    std::vector<Tensor>* capture_allocations_ = nullptr;
    std::vector<float> download_staged(const Tensor& tensor, const std::shared_ptr<float>& destination);
public:
    struct PostProjectionBlasInfo {
        bool preparation_attempted = false, prepared = false, initialization_failed = false;
        uint64_t admitted_workspace_bytes = 0, allocated_workspace_bytes = 0, successful_preparations = 0;
        const void* workspace_address = nullptr;
        const void* main_workspace_address = nullptr;
    };
    struct RetirementInfo {
        bool closed = false, attempted = false, succeeded = false, failed = false;
    };
    struct NodeTimingInfo {
        // Failed means this timing ownership domain became unsafe, including a
        // Context failure outside the timer. The first CUDA status stays latched.
        bool preparation_attempted = false, prepared = false, active = false, failed = false;
        // A non-null create output is registered even on an uncertain failure.
        // The remaining counters count successful API operations only.
        uint64_t created_events = 0, destroyed_events = 0, begin_records = 0, end_records = 0, elapsed_reads = 0;
        uint64_t active_run = 0;
        int active_node = -1;
        int last_cuda_error = 0;
    };
    explicit Context(uint64_t memory_budget_bytes, int device_ordinal = 0,
                     uint64_t host_staging_budget_bytes = 256ull * 1024 * 1024);
    Context(std::shared_ptr<SharedDeviceOwner> owner, uint64_t slot_id,
            uint64_t slot_memory_budget_bytes,
            uint64_t host_staging_budget_bytes = 256ull * 1024 * 1024);
    // Activations stay private to this exact Context. Only a sealed parameter
    // view may cross slots, under the bound model/device and active read pin.
    void bind_slot_parameters(std::unique_ptr<SlotParameterBinding>);
    void begin_weight_read(WeightReadStamp);
    void finish_weight_read();
    void validate_parameter(const Parameter&) const;
    std::shared_ptr<SharedDeviceOwner> shared_owner() const;
    uint64_t slot_id() const;
    bool has_active_weight_read() const;
    // Final owner release, after joined workers, retired Graph/weight-cache
    // aliases, no activation owners and no active read. Failure retains the
    // ownership domain; a destructor return is never a cleanup-success proof.
    bool retire_owned_resources() noexcept;
    RetirementInfo retirement_info() const;
    cudaStream_t stream() const;
    cublasHandle_t blas() const;
    void* blas_workspace() const;
    size_t blas_workspace_bytes() const;
    // Explicit bounded startup preparation; never selected by blas() or by
    // existing kernels. The getter performs no creation/allocation and rejects
    // use until preparation has completed successfully.
    void prepare_post_projection_blas();
    cublasHandle_t post_projection_blas() const;
    PostProjectionBlasInfo post_projection_info() const;
    // Optional detail instrumentation. Default Context construction creates no
    // node events. Prepare explicitly before graph submission; one owned pair
    // is reused only after a matching finish confirms physical stream drain.
    void prepare_node_timing();
    void begin_node_timing(uint64_t run_id, int node);
    float finish_node_timing(uint64_t run_id, int node);
    // Diagnostic POD remains readable after failure/checked retirement, with no
    // device selection or CUDA API call. A false active flag is not a job proof.
    NodeTimingInfo node_timing_info() const;
    Tensor empty(std::vector<int64_t> shape);
    // Captured kernels retain every temporary address until the slot retires
    // its executable graph. Preparation runs before entering capture.
    void begin_capture(std::vector<Tensor>& allocations);
    cudaGraph_t end_capture();
    bool capture_active() const { return capture_allocations_ != nullptr; }
    // Enqueue a set of host-to-device uploads on the context stream and drain
    // that stream once.  The single-upload API remains the compatibility path;
    // graph runners use this batched form to avoid one synchronize per input.
    struct UploadRequest {
        std::vector<int64_t> shape;
        std::span<const float> values;
    };
    std::vector<Tensor> upload_batch(std::span<const UploadRequest> requests);
    // Caller owns every source until this stream drains, including on error.
    std::vector<Tensor> upload_batch_borrowed(std::span<const UploadRequest> requests);
    Tensor upload(std::vector<int64_t> shape, std::span<const float> values);
    std::vector<float> download(const Tensor& tensor);
    // The tensor-sized pinned allocation remains owned by the transfer
    // record until the stream drains, including on a failed DMA call.
    void download_into_pinned(const Tensor& tensor, const std::shared_ptr<float>& destination);
    void synchronize();
    uint64_t peak_live_bytes() const;
    uint64_t live_bytes() const;
    uint64_t budget_bytes() const;
    uint64_t host_staging_live_bytes() const;
    uint64_t host_staging_peak_bytes() const;
    uint64_t host_staging_budget_bytes() const;
    uint64_t host_staging_quarantined_bytes() const;
    // True only when RW_NR_STREAM_ORDERED_MEMORY=1 explicitly opts the
    // experimental payload allocator into cudaMallocAsync/cudaFreeAsync.
    bool stream_ordered_memory() const;
    // True only when RW_NR_GPU_FAST_MATH=1 explicitly opts GEMM kernels into
    // TF32 tensor-core execution. The default keeps the pedantic FP32 path.
    bool fast_math() const;
    // More aggressive optional candidate: permit cuBLAS to down-convert FP32
    // operands to FP16/TF32. It is never enabled by the default profile.
    bool half_math() const;
    bool tensor_cache_enabled() const;
    uint64_t tensor_cached_bytes() const;
    bool deferred_tensor_cache() const;
    bool explicit_half_gemm() const;
    bool explicit_fp8_gemm() const;
    const void* half_parameter(const Parameter&) const;
    void* half_activation(size_t elements) const;
    const void* fp8_parameter(const Parameter&) const;
    void* fp8_activation(size_t elements) const;
    cublasLtHandle_t blas_lt() const;
    void* batched_pointer_workspace(size_t pointer_count) const;
    const void* identity() const;
    bool owns(const Tensor&) const;
};

// Internal conversion used only by the opt-in explicit-half GEMM candidate.
// The default FP32 path never calls this entry point.
void convert_f32_to_f16(cudaStream_t, const float*, void*, size_t);
void convert_f32_to_fp8(cudaStream_t, const float*, void*, size_t);
void convert_f16_to_f32(cudaStream_t, const void*, float*, size_t);

struct PreparedParameterSet {
    // Immutable aliases of already registered weight tensors, never activations.
    std::vector<Parameter> tensors;
};
struct PreparationStatistics {
    bool enabled = false;
    uint64_t lookups = 0, hits = 0, misses = 0, bypasses = 0;
    uint64_t completed_preparations = 0, publications = 0, duplicate_publications = 0, rejections = 0;
    uint64_t entries = 0, entry_limit = 2048;
    uint64_t host_payload_bytes = 0, host_container_capacity_bytes = 0;
    uint64_t host_allocator_allowance_bytes = 0, host_accounted_bytes = 0, host_accounted_peak_bytes = 0;
    uint64_t host_byte_limit = 8ull * 1024 * 1024;
    // The prepared registry is allowed only to alias registered GPU weights.
    uint64_t additional_device_allocation_bytes = 0;
};

class Weights {
    struct Impl;
    std::shared_ptr<Impl> implementation_;
    void attach_shared_pool(std::weak_ptr<DeviceWeightPool>,RegisteredModelOwner);
    friend class DeviceWeightPool;
public:
    explicit Weights(const std::filesystem::path& safetensors);
    std::string network() const;
    std::span<const uint8_t> raw(const std::string& key) const;
    std::span<const uint8_t> raw(int block, int layer, const std::string& parameter) const;
    std::vector<float> host_f16(int block, int layer, const std::string& parameter) const;
    std::vector<float> host_f32(int block, int layer, const std::string& parameter) const;
    std::vector<float> host_e4m3(int block, int layer, const std::string& parameter) const;
    Parameter f16(Context&, int block, int layer, const std::string&, std::vector<int64_t> shape);
    Parameter f32(Context&, int block, int layer, const std::string&, std::vector<int64_t> shape);
    Parameter e4m3(Context&, int block, int layer, const std::string&, std::vector<int64_t> shape);
    Parameter derived(Context&, const std::string& key, std::vector<int64_t> shape,
                   std::span<const float> values);
    // Frozen at construction from RW_NR_PREPARED_CACHE=0/1 (default 1).
    // A hit validates Context, configuration, Tensor shape/presence and backing
    // storage identity. First construction retains the original operation order.
    std::shared_ptr<const PreparedParameterSet> find_prepared(Context&, const std::string& key,
        std::span<const int64_t> configuration);
    void remember_prepared(Context&, const std::string& key, std::span<const int64_t> configuration,
        std::span<const Parameter> tensors);
    PreparationStatistics preparation_statistics() const;
    void clear_device_cache_for_retirement();
};

// GEMM uses FP32 operands/accumulators by default; RW_NR_GPU_FAST_MATH=1 opts
// into TF32 tensor-core execution. FP8 values are decoded exactly. Explicit
// publication kernels preserve each source rounding boundary.
Tensor linear_last(Context&, const Tensor& input, const Parameter& weight);
Tensor linear_nchw(Context&, const Tensor& input, const Parameter& weight);
Tensor narrow(Context&, const Tensor& input);               // f32 -> f16 -> f32
Tensor round8(Context&, const Tensor& input);               // RNE E4M3FN satfinite
Tensor store(Context&, const Tensor& input);                // f16 then E4M3FN
void store_into(Context&, const Tensor& input, Tensor& output, size_t offset); // same publication into an owned slice
Tensor activate(Context&, const Tensor& input);             // cubic SiLU then E4M3FN
// Cropped HWC RGB followed by one validity flag (0=valid, 1=invalid).
Tensor output_hwc(Context&, const Tensor& input, int height, int width);
Tensor input_nchw(Context&, const Tensor& hwc, int padded_height, int padded_width);
// Append CUDA producer input validity to the HWC delivery's final flag.
// 2=invalid producer input, 1=invalid output, 0=valid. No host input scan.
void validate_hwc_inputs(Context&, const Tensor& color, const Tensor* motion,
                         const Tensor* validity, Tensor& delivery);

struct SwinParameters {
    int channels = 0, heads = 0;
    Parameter expand, mid, contract, qkv, out_projection;
    Parameter exponent_bias, head_scale, mlp_scale, attention_scale;
};
SwinParameters load_swin(Context&, Weights&, int block, int layer, int channels, int heads);

struct SwinOutput {
    Tensor wide;           // raw f16 output, source swin_core(...)
    Tensor accumulator;    // raw f32 pre-f16 attention projection, for pooling
};
SwinOutput swin_body(Context&, const SwinParameters&, const Tensor& input,
                     int shift_rows, int shift_columns);
Tensor pool2x2(Context&, const Tensor& input);  // ordered f16 operations, then E4M3FN
Tensor window_attention(Context&, const Tensor& qkv_windows,
                        const Parameter& head_scale, const Parameter& exponent_bias, int heads);

struct LayerSpec {
    int index = 0, block = 0, layer = 0;
    std::string kind, variant;
    int width = 0, height = 0, out_width = 0, out_height = 0;
    int channels = 0, heads = 0, input_channels = 0, output_channels = 0;
    int shift_rows = 0, shift_columns = 0;
    int skip_source = -1;
};
struct LayerOutput { Tensor main, skip, next_history; };

struct Controls {
    float intensity = 1.0f, local_tone = 1.0f, local_structure = 1.0f, skin_structure = -1.0f;
    int style = 0;
    bool auto_mask = true;
    // Legacy diagnostic mode injects the pinned counter-noise lanes.  The
    // next-path experiment can derive those lanes from RGB without changing
    // the default frozen replay; real motion/depth/history still require the
    // producer packet path.
    bool synthetic_noise = true;
    // ABI2 diagnostic path: when present, the second colour triplet in the
    // source feature layout is populated from the previous completed frame.
    // The first ABI remains noise/current-colour and is unchanged.
    bool previous_color = false;
    // ABI2 motion-compensated temporal path. Motion is current->previous in
    // pixel units and is consumed by the native feature kernel before the
    // previous colour triplet is formed.
    bool motion_warp = false;
    bool motion_validity = false;
    uint32_t seed = 0;
};
struct PreBlockDiagnostics { Tensor features, lifted; };
struct PostBlockDiagnostics { Tensor blend, wide, residual; };
Tensor gaussian_noise_f32(Context&, int height, int width, uint32_t seed);
Tensor input_features(Context&, const Tensor& padded_image, const Controls&,
                      const Tensor* previous_image = nullptr,
                      const Tensor* motion_image = nullptr,
                      const Tensor* motion_validity_image = nullptr,
                      const Tensor* reprojected = nullptr);
LayerOutput run_pre(Context&, Weights&, const LayerSpec&, const Tensor& padded_image,
                    const Controls&, const Tensor* explicit_features = nullptr,
                    const Tensor* previous_image = nullptr,
                    const Tensor* motion_image = nullptr,
                    const Tensor* motion_validity_image = nullptr,
                    PreBlockDiagnostics* diagnostics = nullptr, const Tensor* reprojected = nullptr);
LayerOutput run_post(Context&, Weights&, const LayerSpec&, const Tensor& input,
                     const LayerOutput* skip, const Tensor& padded_image, const Controls&,
                     PostBlockDiagnostics* diagnostics = nullptr, const Tensor* reprojected = nullptr,
                     int temporal_width = 0, int temporal_height = 0);
Tensor reproject_sdr_history(Context&, const Tensor& previous, const Tensor& motion,
                            const Tensor* validity, int valid_width, int valid_height);

// Families each own their parameter interpretation and pointwise kernels.
LayerOutput run_swin(Context&, Weights&, const LayerSpec&, const Tensor& input, const LayerOutput* skip);
LayerOutput run_split(Context&, Weights&, const LayerSpec&, const LayerOutput& input, const LayerOutput* skip);
LayerOutput run_vit(Context&, Weights&, const LayerSpec&, const Tensor& input, const LayerOutput* skip);
LayerOutput run_decoder(Context&, Weights&, const LayerSpec&, const Tensor& input, const LayerOutput* skip);

void cuda_check(cudaError_t, const char* operation);
void blas_check(cublasStatus_t, const char* operation);
} // namespace rw::native_nr
