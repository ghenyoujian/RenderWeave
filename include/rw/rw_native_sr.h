#ifndef RW_NATIVE_SR_H
#define RW_NATIVE_SR_H

#include "rw_api.h"

#if defined(RW_BUILDING_SR)
# define RW_SR_API RW_EXPORT
#else
# define RW_SR_API RW_IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Experimental CUDA K/LDR/2x profile: zero jitter, unit exposure.
 * Use increasing nonzero frame IDs and predecessor 0 for the first frame. */
typedef struct rw_sr_runtime* rw_sr_runtime_t;
typedef struct rw_sr_session* rw_sr_session_t;
typedef struct rw_sr_job* rw_sr_job_t;

#define RW_SR_CUDA_GRAPH 1u
#define RW_SR_TILED_BACKBONE 2u
#define RW_SR_STAGE_TIMINGS 4u
#define RW_SR_PARALLEL_REDUCE 8u
#define RW_SR_REUSE_MATRIX_ROWS 16u
#define RW_SR_SPIN_WAIT 32u
#define RW_SR_CUDA_IO_ONLY 64u
/* Select the original FP16 accumulation order for the experimental SR backbone. */
#define RW_SR_ORIGINAL_FP16_BACKBONE 128u

typedef struct rw_sr_runtime_desc {
    uint32_t struct_size, abi_version;
    int32_t device_ordinal;
    uint32_t width, height, execution_slots, max_sessions;
    uint64_t device_budget_bytes;
    const char* weights_path;
    const char* filter_path;
    /* Optional ABI 1 extension. Capture the recurrent kernels per slot. */
    uint32_t options;
} rw_sr_runtime_desc;

/* Contiguous host float32 RGB [0,1], current-to-previous input-pixel motion,
 * and forward depth [0,1]. Motion and depth may both be absent at reset.
 * Arrays may be reused after submit. */
typedef struct rw_sr_frame {
    uint32_t struct_size, abi_version;
    uint64_t frame_id, epoch, expected_predecessor;
    const float* color;
    uint64_t color_elements;
    const float* motion;
    uint64_t motion_elements;
    const float* depth;
    uint64_t depth_elements;
} rw_sr_frame;

/* Use the same frame layout with device pointers. The producer can supply
 * a recorded cudaEvent_t, or null for data already ready on the device.
 * Keep the planes and event alive and unchanged until physical completion.
 * After cancel, use wait and job info to confirm physical completion before
 * releasing the input or the job handle.
 * On RW_INTERNAL_ERROR with unconfirmed completion, retain the allocations
 * until their CUDA access has been independently drained. */
typedef struct rw_sr_cuda_input {
    uint32_t struct_size, abi_version;
    void* ready_event;
} rw_sr_cuda_input;

typedef struct rw_sr_cuda_view {
    uint32_t struct_size, abi_version;
    const float* rgba;
    uint64_t rgba_elements;
} rw_sr_cuda_view;

typedef struct rw_sr_job_info {
    uint32_t struct_size, abi_version;
    rw_status_t status;
    uint32_t physical_complete;
    uint64_t frame_id, epoch, submit_ns, device_enqueued_ns, observed_complete_ns;
    /* CUDA-event interval from input transfer or snapshot to output readiness,
     * including the validation result transfer. Zero before physical completion. */
    double device_stream_ms;
} rw_sr_job_info;

typedef struct rw_sr_memory_info {
    uint32_t struct_size, abi_version;
    uint64_t shared_model_device_bytes, slots_device_bytes, live_sessions_device_bytes;
    uint64_t total_device_bytes, pinned_host_bytes;
} rw_sr_memory_info;

typedef struct rw_sr_stage_timings {
    uint32_t struct_size, abi_version;
    double upload_ms, prepare_ms, backbone_ms, reconstruct_ms, validate_and_readback_ms;
} rw_sr_stage_timings;

typedef struct rw_sr_display_info {
    uint32_t struct_size, abi_version;
    float min_rgba, max_rgba;
    uint64_t clamped_rgb_values;
} rw_sr_display_info;

RW_SR_API rw_status_t rw_sr_runtime_create(const rw_sr_runtime_desc*,rw_sr_runtime_t*);
/* Read-only asset bytes are consumed during creation. The same original
 * weight/filter identities are required as in the file-backed entry. */
RW_SR_API rw_status_t rw_sr_runtime_create_from_memory(const rw_sr_runtime_desc*,
    const void* weights,uint64_t weights_bytes,const void* filter,uint64_t filter_bytes,rw_sr_runtime_t*);
RW_SR_API const char* rw_sr_last_error(void);
RW_SR_API rw_status_t rw_sr_runtime_get_memory_info(rw_sr_runtime_t,rw_sr_memory_info*);
RW_SR_API void rw_sr_runtime_release(rw_sr_runtime_t);
RW_SR_API rw_status_t rw_sr_session_create(rw_sr_runtime_t,uint64_t epoch,rw_sr_session_t*);
RW_SR_API rw_status_t rw_sr_session_reset(rw_sr_session_t,uint64_t new_epoch);
RW_SR_API void rw_sr_session_release(rw_sr_session_t);
RW_SR_API rw_status_t rw_sr_submit(rw_sr_session_t,const rw_sr_frame*,rw_sr_job_t*);
RW_SR_API rw_status_t rw_sr_submit_cuda(rw_sr_session_t,const rw_sr_frame*,const rw_sr_cuda_input*,rw_sr_job_t*);
/* RW_NOT_READY is a wait timeout; the job and device work remain live. A
 * successful wait copies (2*height)*(2*width)*4 float32 values to caller memory. */
RW_SR_API rw_status_t rw_sr_job_wait(rw_sr_job_t,int64_t timeout_ms,float* rgba,uint64_t elements);
/* A successful CUDA wait publishes a slot-owned RGBA view on the same device.
 * Hold the job until every consumer of the view has completed. */
RW_SR_API rw_status_t rw_sr_job_wait_cuda(rw_sr_job_t,int64_t timeout_ms,rw_sr_cuda_view*);
/* Convert contiguous host FP32 RGBA to packed RGB8 for SDR display.
 * Multiply saturated RGB by 255 in FP32 and round to nearest, ties up.
 * Input and output must not overlap. The output is valid on RW_OK. */
RW_SR_API rw_status_t rw_sr_display_rgb8(const float* rgba,uint64_t rgba_elements,
    uint8_t* rgb,uint64_t rgb_bytes,rw_sr_display_info*);
RW_SR_API rw_status_t rw_sr_job_get_info(rw_sr_job_t,rw_sr_job_info*);
/* Available after completion when RW_SR_STAGE_TIMINGS is enabled. */
RW_SR_API rw_status_t rw_sr_job_get_stage_timings(rw_sr_job_t,rw_sr_stage_timings*);
RW_SR_API rw_status_t rw_sr_job_cancel(rw_sr_job_t);
/* Dropping an unfinished job requests cancellation. Release the job to make
 * its execution slot available after completion. */
RW_SR_API void rw_sr_job_release(rw_sr_job_t);

#ifdef __cplusplus
}
#endif
#endif
