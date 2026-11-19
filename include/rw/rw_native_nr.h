#ifndef RW_NATIVE_NR_H
#define RW_NATIVE_NR_H
#include "rw_api.h"

#if defined(RW_BUILDING_NR)
# define RW_NR_API RW_EXPORT
#else
# define RW_NR_API RW_IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Experimental fixed-HT, SDR sRGB, CPU/CUDA input and explicit RGB history.
 * A runtime owns one CUDA execution slot by default;
 * explicit two/eight-slot experiments use one shared immutable model and
 * independent Context/Graph/stream/workspace/event sets.
 * Sessions share immutable parameters. A second runtime on the same physical
 * device is refused until checked GPU retirement succeeds; CPU metadata child
 * handles may remain queryable after that retirement. */
#define RW_NR_EXPERIMENTAL 1u
#define RW_NR_TWO_SLOTS_EXPERIMENTAL 2u
#define RW_NR_EXECUTION_SLOT_POLICY_VERSION 1u
/* ABI1 additive opt-in; mutually exclusive with TWO_SLOTS. Neither bit means
 * the default one slot. Unknown bits and TWO|EIGHT are rejected. */
#define RW_NR_EIGHT_SLOTS_EXPERIMENTAL 8u
/* Opt-in node timing and memory observations. Without this bit, mandatory
 * request phases, counters, physical-completion guards and bounded traces stay
 * enabled, while optional per-node measurements are not collected. */
#define RW_NR_NODE_DETAIL 4u
#define RW_NR_STATELESS_RESET 1u
#define RW_NR_NO_PREDECESSOR UINT64_MAX
/* ABI2 experimental temporal colour-history input. The current frame keeps
 * the ABI1 layout; rw_nr_submit_temporal_rgb adds a retained previous HWC
 * colour plane for the second colour feature triplet. Motion/depth/history
 * resource ownership remains a separate temporal contract and is not inferred
 * by this colour-only probe. */
#define RW_NR_TEMPORAL_RGB_HISTORY 16u

typedef struct rw_nr_runtime* rw_nr_runtime_t;
typedef struct rw_nr_session* rw_nr_session_t;
typedef struct rw_nr_job* rw_nr_job_t;
typedef struct rw_nr_result* rw_nr_result_t;

enum { RW_NR_JOB_QUEUED=1, RW_NR_JOB_RUNNING=2, RW_NR_JOB_COMPLETED=3,
       RW_NR_JOB_CANCELLED=4, RW_NR_JOB_STALE=5, RW_NR_JOB_FAILED=6 };

typedef struct rw_nr_runtime_desc {
    uint32_t struct_size, abi_version, flags;
    int32_t device_ordinal;
    uint32_t max_sessions, max_inflight, trace_capacity, reserved;
    uint64_t device_budget_bytes, host_budget_bytes;
    const char* weights_path;
    const char* weights_sha256;
    const char* parameter_manifest_path;
    const char* parameter_manifest_sha256;
    const char* plan_path;
    const char* plan_sha256;
} rw_nr_runtime_desc;

typedef struct rw_nr_session_desc {
    uint32_t struct_size, abi_version, flags, seed;
    uint64_t epoch, predecessor;
} rw_nr_session_desc;

typedef struct rw_nr_session_state {
    uint32_t struct_size, abi_version, accepting, pending;
    uint64_t session_id, epoch, predecessor;
} rw_nr_session_state;

typedef struct rw_nr_frame {
    uint32_t struct_size, abi_version, flags, reserved;
    uint64_t frame_id, epoch, expected_predecessor;
    uint32_t width, height;
    const float* color_hwc;
    uint64_t color_elements;
} rw_nr_frame;

/* CUDA producer input, used with the existing frame's device color_hwc.
 * All planes are contiguous F32 HWC on this runtime's CUDA device. A null
 * ready_event means the producer has already completed its writes; otherwise
 * pass its recorded cudaEvent_t as an opaque pointer. On successful submit,
 * the SDK borrows all pointers/event until release(release_context), called
 * once after confirmed physical completion, or queued cancellation. On a
 * rejected submit ownership stays with the caller and release is not called.
 * Unconfirmed device work retains the lease. The callback runs on an SDK CPU
 * thread and must not throw. CPU output remains rw_nr_result_view. */
typedef struct rw_nr_cuda_input {
    uint32_t struct_size, abi_version;
    const float* motion_hwc2;
    uint64_t motion_elements;
    const float* motion_validity;
    uint64_t motion_validity_elements;
    void* ready_event;
    void (*release)(void* release_context);
    void* release_context;
} rw_nr_cuda_input;

typedef struct rw_nr_job_info {
    uint32_t struct_size, abi_version, state, physical_completion_confirmed;
    int32_t status;
    uint32_t gpu_submitted;
    uint64_t session_id, frame_id, epoch, expected_predecessor;
    uint64_t submit_host_ns, admitted_host_ns, device_execution_host_ns;
    uint64_t complete_host_ns, output_available_host_ns, first_wait_timeout_host_ns;
    /* ABI1 retains the numeric field (zero before collection). The bounded
     * runtime JSON trace supplies device_stream_span_collected and JSON null
     * for uncollected spans, including rejected/early-failed requests. */
    double device_stream_span_ms;
    char error[512];
} rw_nr_job_info;

typedef struct rw_nr_result_view {
    uint32_t struct_size, abi_version, width, height;
    const float* color_hwc;
    uint64_t color_elements, session_id, frame_id, epoch;
} rw_nr_result_view;

/* All exceptions stop at this C boundary. last_error is thread-local and remains
 * valid until the next SDK call on that thread. JSON queries use the size-query
 * convention of rw_context_query; the size includes the terminating zero. */
RW_NR_API const char* rw_nr_last_error(void);
RW_NR_API rw_status_t rw_nr_runtime_create(const rw_nr_runtime_desc*, rw_nr_runtime_t*);
RW_NR_API rw_status_t rw_nr_runtime_query(rw_nr_runtime_t, char* json, size_t* bytes);
/* close stops admission and drains accepted work. timeout_ms=-1 waits; a timeout
 * does not release, cancel or reuse device resources. release consumes the user
 * handle and reports success only after checked Context/pool/device retirement.
 * Unconfirmed work or cleanup retains its complete owner group and device claim
 * in quarantine until process teardown. Child handles remain valid for queries. */
RW_NR_API rw_status_t rw_nr_runtime_close(rw_nr_runtime_t, int64_t timeout_ms);
RW_NR_API rw_status_t rw_nr_runtime_release(rw_nr_runtime_t);
RW_NR_API rw_status_t rw_nr_session_create(rw_nr_runtime_t, const rw_nr_session_desc*, rw_nr_session_t*);
RW_NR_API rw_status_t rw_nr_session_snapshot(rw_nr_session_t, rw_nr_session_state*);
RW_NR_API rw_status_t rw_nr_session_reset(rw_nr_session_t, uint64_t new_epoch, uint64_t predecessor);
RW_NR_API void rw_nr_session_release(rw_nr_session_t);
/* The F32 HWC CPU input is copied before submit returns. The SDK owns that copy
 * until physical completion; the caller may release its original source buffer
 * as soon as submit returns.
 * One outstanding request per session is supported in this initial profile.
 * Every failed admission increments rejected and retains a bounded trace. */
RW_NR_API rw_status_t rw_nr_submit(rw_nr_session_t, const rw_nr_frame*, rw_nr_job_t*);
RW_NR_API rw_status_t rw_nr_submit_temporal_rgb(rw_nr_session_t, const rw_nr_frame*,
                                              const float* previous_color_hwc,
                                              uint64_t previous_color_elements,
                                              rw_nr_job_t*);
/* ABI2 additive temporal path: previous colour is sampled at the supplied
 * current->previous pixel motion before it enters the native feature lane.
 * Motion is HWC2 F32 pixels, one vector per current pixel. */
RW_NR_API rw_status_t rw_nr_submit_temporal_rgb_motion(rw_nr_session_t, const rw_nr_frame*,
                                                     const float* previous_color_hwc,
                                                     uint64_t previous_color_elements,
                                                     const float* motion_hwc2,
                                                     uint64_t motion_elements,
                                                     rw_nr_job_t*);
/* Native session-owned history path. After a successful physical output the
 * SDK retains that output and the next non-reset frame supplies only motion;
 * the retained output is sampled as the previous-color feature lane. */
RW_NR_API rw_status_t rw_nr_submit_temporal_history_motion(rw_nr_session_t, const rw_nr_frame*,
                                                          const float* motion_hwc2,
                                                          uint64_t motion_elements,
                                                          rw_nr_job_t*);
/* Native history path with an explicit HWC1 float validity plane. Invalid
 * motion pixels fall back to the current color when forming the history lane. */
RW_NR_API rw_status_t rw_nr_submit_temporal_history_motion_validity(
    rw_nr_session_t, const rw_nr_frame*, const float* motion_hwc2,
    uint64_t motion_elements, const float* motion_validity,
    uint64_t motion_validity_elements, rw_nr_job_t*);
/* Session-owned previous output, CUDA current input. Range/finite checks run
 * on the device; invalid input completes with RW_INVALID_ARGUMENT and does
 * not advance the predecessor/history. The frame uses TEMPORAL_RGB_HISTORY. */
RW_NR_API rw_status_t rw_nr_submit_cuda_history(rw_nr_session_t, const rw_nr_frame*,
                                             const rw_nr_cuda_input*, rw_nr_job_t*);
/* A wait timeout reports NOT_READY and only records an observation. Cancel is
 * logical intent: running device work drains before its input/slot is retired. */
RW_NR_API rw_status_t rw_nr_job_wait(rw_nr_job_t, int64_t timeout_ms);
RW_NR_API rw_status_t rw_nr_job_cancel(rw_nr_job_t);
RW_NR_API rw_status_t rw_nr_job_query(rw_nr_job_t, rw_nr_job_info*);
RW_NR_API rw_status_t rw_nr_job_get_result(rw_nr_job_t, rw_nr_result_t*);
/* Independent user handle for concurrent calls; dropping either handle never
 * cancels its work. Bindings retain a handle before a blocking GIL-free wait. */
RW_NR_API rw_status_t rw_nr_job_retain(rw_nr_job_t, rw_nr_job_t*);
RW_NR_API void rw_nr_job_release(rw_nr_job_t);
RW_NR_API rw_status_t rw_nr_result_view_get(rw_nr_result_t, rw_nr_result_view*);
RW_NR_API rw_status_t rw_nr_result_retain(rw_nr_result_t, rw_nr_result_t*);
RW_NR_API void rw_nr_result_release(rw_nr_result_t);

#ifdef __cplusplus
}
#endif
#endif
