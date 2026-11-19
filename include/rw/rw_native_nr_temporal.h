#ifndef RW_NATIVE_NR_TEMPORAL_H
#define RW_NATIVE_NR_TEMPORAL_H

/* ABI2 CPU temporal contract only.
 *
 * This header deliberately does not enable native temporal execution. It
 * provides a versioned request/state envelope and a fail-closed preflight so
 * callers can validate guide, timeline and history identity before a future
 * native graph is wired. ABI1 rw_native_nr.h remains unchanged.
 */
#include "rw_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RW_NR_TEMPORAL_ABI_VERSION 2u
#define RW_NR_TEMPORAL_NO_PREDECESSOR UINT64_MAX
#define RW_NR_TEMPORAL_NATIVE_EXECUTION_AVAILABLE 0u

typedef struct rw_nr_temporal_session* rw_nr_temporal_session_t;

enum {
    RW_NR_TEMPORAL_CODE_VALID = 0,
    RW_NR_TEMPORAL_CODE_ABI_MISMATCH = 1,
    RW_NR_TEMPORAL_CODE_INVALID_ARGUMENT = 2,
    RW_NR_TEMPORAL_CODE_MISSING_COLOR = 3,
    RW_NR_TEMPORAL_CODE_MISSING_MOTION = 4,
    RW_NR_TEMPORAL_CODE_MISSING_HISTORY = 5,
    RW_NR_TEMPORAL_CODE_HISTORY = 6,
    RW_NR_TEMPORAL_CODE_PTS = 7,
    RW_NR_TEMPORAL_CODE_RESET = 8,
    RW_NR_TEMPORAL_CODE_STALE = 9,
    RW_NR_TEMPORAL_CODE_OWNER = 10,
    RW_NR_TEMPORAL_CODE_RESOURCE = 11,
    RW_NR_TEMPORAL_CODE_SHAPE = 12,
    RW_NR_TEMPORAL_CODE_CONTROL = 13,
    RW_NR_TEMPORAL_CODE_NATIVE_UNAVAILABLE = 14
};

/* The wrapper adds explicit owner identity to rw_resource. owner_id is an
 * opaque lease identity, not a CUDA pointer; a future backend must retain the
 * lease until physical completion. */
typedef struct rw_nr_temporal_resource {
    uint32_t struct_size, abi_version;
    rw_resource resource;
    uint64_t owner_id, owner_session_id, owner_epoch, owner_frame_id;
} rw_nr_temporal_resource;

typedef struct rw_nr_temporal_frame {
    uint32_t struct_size, abi_version, flags, reset;
    uint64_t session_id, sequence_id, frame_id, epoch, expected_predecessor;
    int64_t pts;
    int32_t time_base_num, time_base_den;
    uint32_t scene_epoch, device_kind;
    int32_t device_ordinal;
    uint64_t required_semantics;
    const rw_nr_temporal_resource* resources;
    uint32_t resource_count, reserved;
    uint32_t motion_direction, motion_units, jitter_included, jitter_explicitly_disabled;
    uint32_t jitter_x_bits, jitter_y_bits, exposure_scale_bits, exposure_explicit_no_transform;
    uint8_t sequence_identity[32];
    uint8_t model_fingerprint[32];
    uint8_t rng_algorithm[32];
    uint8_t history_codec_sha256[32];
    uint8_t sampler_sha256[32];
} rw_nr_temporal_frame;

typedef struct rw_nr_temporal_state {
    uint32_t struct_size, abi_version, valid, reserved;
    uint64_t session_id, sequence_id, epoch, predecessor;
    int64_t pts;
    int32_t time_base_num, time_base_den;
    uint32_t scene_epoch, reserved2;
    uint64_t history_owner_id, history_epoch, history_frame;
    uint8_t sequence_identity[32];
    uint8_t model_fingerprint[32];
    uint8_t rng_algorithm[32];
    uint8_t history_codec_sha256[32];
    uint8_t sampler_sha256[32];
} rw_nr_temporal_state;

typedef struct rw_nr_temporal_preflight_result {
    uint32_t struct_size, abi_version, code, native_execution_available;
    uint64_t required_semantics, present_semantics;
    uint32_t history_commit_allowed, reserved;
} rw_nr_temporal_preflight_result;

typedef struct rw_nr_temporal_memory_plan {
    uint32_t struct_size, abi_version;
    uint64_t budget_bytes, shared_weights_bytes;
    uint64_t bytes_per_slot, slot_count;
    uint64_t committed_history_bytes, provisional_history_bytes, session_count;
    uint64_t import_output_bytes, graph_handle_bytes, reserve_quarantine_bytes;
} rw_nr_temporal_memory_plan;

/* Returns RW_OK only when the CPU contract is internally valid. A valid
 * request still reports native_execution_available=0; this function never
 * schedules CUDA work or claims model qualification. */
RW_API rw_status_t rw_nr_temporal_preflight(
    const rw_nr_temporal_frame*, const rw_nr_temporal_state*,
    rw_nr_temporal_preflight_result*);

/* Commit simulation for the control plane. No state is changed unless the
 * request is current, output is valid and physical completion is confirmed.
 * A stale/late result leaves `after` untouched. */
RW_API rw_status_t rw_nr_temporal_commit(
    const rw_nr_temporal_state*, const rw_nr_temporal_frame*,
    uint32_t physical_completion, uint32_t output_valid,
    uint64_t next_history_owner_id, rw_nr_temporal_state* after);

/* Exact checked arithmetic for shared weights, slots, session histories and
 * quarantine reserve. It is independent of CUDA and model execution. */
RW_API rw_status_t rw_nr_temporal_memory_required(
    const rw_nr_temporal_memory_plan*, uint64_t* required_bytes);

/* CPU control-plane session. It owns epoch/predecessor/history state only;
 * it never launches CUDA and always keeps native temporal unavailable. */
RW_API rw_status_t rw_nr_temporal_session_create(
    uint64_t session_id, uint64_t epoch, rw_nr_temporal_session_t*);
RW_API rw_status_t rw_nr_temporal_session_snapshot(
    rw_nr_temporal_session_t, rw_nr_temporal_state*);
RW_API rw_status_t rw_nr_temporal_session_reset(
    rw_nr_temporal_session_t, uint64_t epoch);
RW_API rw_status_t rw_nr_temporal_session_preflight(
    rw_nr_temporal_session_t, const rw_nr_temporal_frame*,
    rw_nr_temporal_preflight_result*);
RW_API rw_status_t rw_nr_temporal_session_commit(
    rw_nr_temporal_session_t, const rw_nr_temporal_frame*,
    uint32_t physical_completion, uint32_t output_valid,
    uint64_t next_history_owner_id, rw_nr_temporal_state*);
RW_API void rw_nr_temporal_session_release(rw_nr_temporal_session_t);

#ifdef __cplusplus
}
#endif
#endif
