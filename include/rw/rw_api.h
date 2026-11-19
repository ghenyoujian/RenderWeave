#ifndef RW_API_H
#define RW_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
# define RW_EXPORT __declspec(dllexport)
# define RW_IMPORT __declspec(dllimport)
#else
# define RW_EXPORT __attribute__((visibility("default")))
# define RW_IMPORT RW_EXPORT
#endif
#if defined(RW_BUILDING_LIBRARY)
# define RW_API RW_EXPORT
#else
# define RW_API RW_IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RW_ABI_VERSION 1u
#define RW_MAX_TENSOR_RANK 8u
#define RW_SEMANTIC_BIT(semantic) (UINT64_C(1) << (semantic))

typedef int32_t rw_status_t;
enum {
    RW_OK = 0, RW_INVALID_ARGUMENT = 1, RW_ABI_MISMATCH = 2,
    RW_UNSUPPORTED = 3, RW_RESOURCE_EXHAUSTED = 4, RW_OVERFLOW = 5,
    RW_OUT_OF_BOUNDS = 6, RW_NOT_READY = 7, RW_STALE = 8,
    RW_CANCELLED = 9, RW_OUTPUT_INVALID = 10, RW_INTERNAL_ERROR = 11,
    RW_BUFFER_TOO_SMALL = 12
};
enum { RW_DEVICE_CPU = 1, RW_DEVICE_CUDA = 2, RW_DEVICE_ASCEND = 3 };
enum { RW_DTYPE_U8 = 1, RW_DTYPE_F16 = 2, RW_DTYPE_F32 = 3, RW_DTYPE_E4M3FN = 4 };
enum { RW_ACCESS_READ = 1, RW_ACCESS_WRITE = 2, RW_ACCESS_READ_WRITE = 3 };
enum {
    RW_RESOURCE_COLOR = 1, RW_RESOURCE_DEPTH = 2, RW_RESOURCE_MOTION = 3,
    RW_RESOURCE_HISTORY = 4, RW_RESOURCE_VALIDITY = 5,
    RW_RESOURCE_EXPOSURE = 6, RW_RESOURCE_JITTER = 7, RW_RESOURCE_CAMERA = 8
};
enum { RW_VALIDITY_ABSENT = 0, RW_VALIDITY_VALID = 1, RW_VALIDITY_INVALID = 2 };
enum {
    RW_PROVENANCE_CAPTURED = 1, RW_PROVENANCE_RENDERER_DERIVED = 2,
    RW_PROVENANCE_ESTIMATED = 3, RW_PROVENANCE_USER = 4
};
enum { RW_COLOR_SRGB_CODE = 1, RW_COLOR_LINEAR_REC709 = 2, RW_COLOR_LINEAR_REC2020 = 3 };
enum { RW_DYNAMIC_RANGE_SDR = 1, RW_DYNAMIC_RANGE_HDR = 2 };
enum { RW_DEPTH_VIEW_Z = 1, RW_DEPTH_EUCLIDEAN = 2 };
enum { RW_MOTION_CURRENT_TO_PREVIOUS = 1, RW_MOTION_PREVIOUS_TO_CURRENT = 2 };
enum { RW_MOTION_PIXELS = 1, RW_MOTION_UV = 2, RW_MOTION_NDC = 3 };
enum { RW_JITTER_INCLUDED = 1, RW_JITTER_EXCLUDED = 2 };

typedef struct rw_context* rw_context_t;
typedef struct rw_event* rw_event_t;
typedef struct rw_state_guard* rw_state_guard_t;

typedef struct rw_context_desc {
    uint32_t struct_size, abi_version, device_kind;
    int32_t device_ordinal;
    uint64_t memory_budget_bytes;
    uint32_t max_sessions, max_inflight, execution_slots, flags;
    const char* config_json;
} rw_context_desc;

typedef struct rw_tensor_view {
    uint32_t struct_size, dtype, device_kind;
    int32_t device_ordinal;
    uint32_t ndim, access_mode;
    int64_t shape[RW_MAX_TENSOR_RANK];
    int64_t strides_bytes[RW_MAX_TENSOR_RANK];
    void* base;
    uint64_t byte_offset, allocation_bytes;
    void* owner_token;
    rw_event_t ready_event;
} rw_tensor_view;

typedef struct rw_resource_metadata {
    uint32_t struct_size, color_space, dynamic_range, depth_definition;
    uint32_t motion_direction, motion_units, jitter_included, reserved;
} rw_resource_metadata;

typedef struct rw_resource {
    uint32_t struct_size, resource_semantic, validity, provenance;
    rw_tensor_view tensor;
    const char* metadata_json;
    rw_resource_metadata metadata;
} rw_resource;

typedef struct rw_frame_desc {
    uint32_t struct_size, flags;
    uint64_t frame_id, sequence_id, scene_epoch;
    int64_t pts;
    int32_t time_base_num, time_base_den;
    uint64_t deadline_monotonic_ns;
    const rw_resource* resources;
    uint32_t resource_count, reserved;
    const char* controls_json;
} rw_frame_desc;

/* This first input contract accepts interleaved HWC color, HWC2 motion,
 * floating F16/F32 HW depth, and U8 HW validity masks at matching dimensions.
 * It validates metadata and memory bounds; it does not certify any model input. */
typedef struct rw_frame_contract {
    uint32_t struct_size, abi_version, device_kind;
    int32_t device_ordinal;
    uint64_t required_semantics;
    uint32_t color_space, dynamic_range, temporal, require_ready_event;
    uint32_t motion_direction, motion_units, jitter_included, reserved;
} rw_frame_contract;

typedef struct rw_memory_plan {
    uint32_t struct_size, abi_version;
    uint64_t budget_bytes, shared_weights_bytes;
    uint64_t bytes_per_slot, slot_count, bytes_per_session, session_count;
    uint64_t import_output_bytes, graph_handle_bytes, reserve_quarantine_bytes;
} rw_memory_plan;

typedef struct rw_state_snapshot {
    uint32_t struct_size, abi_version;
    uint64_t epoch, generation, predecessor, pending_count;
} rw_state_snapshot;

RW_API uint32_t rw_abi_version(void);
RW_API const char* rw_status_string(rw_status_t status);
RW_API rw_status_t rw_context_create(const rw_context_desc*, rw_context_t*);
RW_API rw_status_t rw_context_query(rw_context_t, const char* key, char* json, size_t* bytes);
RW_API void rw_context_release(rw_context_t);
RW_API rw_status_t rw_validate_tensor(const rw_tensor_view*, uint64_t* required_allocation_bytes);
RW_API rw_status_t rw_validate_frame(const rw_frame_desc*, const rw_frame_contract*);
RW_API rw_status_t rw_memory_plan_required(const rw_memory_plan*, uint64_t* required_bytes);

/* E4M3FN, round-to-nearest-even, saturate finite/infinite values at +/-448.
 * Signed zero and NaN sign are preserved; NaN payload becomes mantissa 0b111. */
RW_API float rw_e4m3_decode(uint8_t code);
RW_API uint8_t rw_e4m3_encode(float value);
RW_API rw_status_t rw_e4m3_decode_array(const uint8_t*, float*, size_t count);
RW_API rw_status_t rw_e4m3_encode_array(const float*, uint8_t*, size_t count);

/* CPU arithmetic reference only. Row-major A[M,K], B[K,N]. Residual is
 * seeded into the accumulator before products. Output is unquantized FP32. */
RW_API rw_status_t rw_reference_gemm_e4m3(
    const uint8_t* a, size_t a_count, const uint8_t* b, size_t b_count,
    const float* residual, size_t residual_count, float* output, size_t output_count,
    uint32_t m, uint32_t n, uint32_t k);

/* Explicit general sampling primitive, NOT the NR temporal reconstruction filter.
 * Coordinates address pixel centers at integers and clamp to the valid image.
 * Each axis is limited to 2^24 pixels, the exact integer domain of float32. */
RW_API rw_status_t rw_reference_bilinear_clamp_f32(
    const float* image, size_t image_count, uint32_t width, uint32_t height,
    float x, float y, float* output);
RW_API rw_status_t rw_mirror_index(int64_t coordinate, uint32_t extent, uint32_t* output);

/* Control-plane transaction guard. complete(..., physical_completion=0, ...)
 * never retires a ticket. The device backend must prove physical completion
 * with its own event before passing 1. Timeout/cancel is a logical flag only. */
RW_API rw_status_t rw_state_guard_create(uint64_t epoch, uint64_t predecessor, rw_state_guard_t*);
RW_API rw_status_t rw_state_guard_snapshot(rw_state_guard_t, rw_state_snapshot*);
RW_API rw_status_t rw_state_guard_begin(rw_state_guard_t, uint64_t epoch,
    uint64_t expected_predecessor, uint64_t frame_id, uint64_t* ticket);
RW_API rw_status_t rw_state_guard_cancel(rw_state_guard_t, uint64_t ticket);
RW_API rw_status_t rw_state_guard_reset(rw_state_guard_t, uint64_t new_epoch, uint64_t predecessor);
RW_API rw_status_t rw_state_guard_complete(rw_state_guard_t, uint64_t ticket,
    uint32_t physical_completion, uint32_t output_valid);
/* release refuses pending physical work, so a caller cannot silently free it. */
RW_API rw_status_t rw_state_guard_release(rw_state_guard_t);

#ifdef __cplusplus
}
#endif
#endif
