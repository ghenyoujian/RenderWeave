#ifndef RW_SR_POSTPROCESS_H
#define RW_SR_POSTPROCESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RW_SR_POSTPROCESS_API_VERSION 1u

typedef enum rw_sr_postprocess_status {
  RW_SR_POSTPROCESS_ACCEPTED = 0,
  RW_SR_POSTPROCESS_MISSING_SESSION = 1,
  RW_SR_POSTPROCESS_MISSING_EPOCH = 2,
  RW_SR_POSTPROCESS_INVALID_GEOMETRY = 3,
  RW_SR_POSTPROCESS_NONFINITE_COORDINATE = 4,
  RW_SR_POSTPROCESS_MISSING_HISTORY = 5,
  RW_SR_POSTPROCESS_MISSING_HISTORY_RESOURCE = 6,
  RW_SR_POSTPROCESS_MISSING_HISTORY_EVENTS = 7,
  RW_SR_POSTPROCESS_MISSING_HISTORY_CODEC = 8,
  RW_SR_POSTPROCESS_MISSING_LAUNCH_JOIN = 9,
  RW_SR_POSTPROCESS_HISTORY_OWNER_MISMATCH = 10,
  RW_SR_POSTPROCESS_STALE_PREDECESSOR = 11,
  RW_SR_POSTPROCESS_MISSING_OUTPUT_OWNER = 12,
  RW_SR_POSTPROCESS_ALIASED_OWNER = 13
} rw_sr_postprocess_status;

typedef struct rw_sr_history_binding {
  uint32_t struct_size;
  uint32_t valid;
  uint32_t authenticated;
  uint32_t reserved;
  uint64_t resource_id;
  uint64_t owner_id;
  uint64_t session_id;
  uint64_t epoch;
  uint64_t frame;
  uint64_t codec_fingerprint;
  uint64_t ready_event;
  uint64_t completion_event;
} rw_sr_history_binding;

typedef struct rw_sr_launch_binding {
  uint32_t struct_size;
  uint32_t authenticated;
  uint32_t reserved0;
  uint32_t reserved1;
  uint64_t correlation;
  uint64_t graph_generation;
  uint64_t module_id;
  uint64_t function_id;
} rw_sr_launch_binding;

typedef struct rw_sr_catmull_rom_submission {
  uint32_t struct_size;
  uint32_t output_valid;
  uint64_t session_id;
  uint64_t epoch;
  uint64_t frame_id;
  uint64_t expected_predecessor;
  uint64_t output_owner;
  uint32_t width;
  uint32_t height;
  float x;
  float y;
  rw_sr_history_binding history;
  rw_sr_launch_binding launch;
} rw_sr_catmull_rom_submission;

/* CPU-only admission. It never touches CUDA and never commits history. */
static inline rw_sr_postprocess_status rw_sr_postprocess_admit_catmull_rom(
    const rw_sr_catmull_rom_submission* request) {
  if (!request || request->struct_size < sizeof(*request)) return RW_SR_POSTPROCESS_INVALID_GEOMETRY;
  if (!request->session_id) return RW_SR_POSTPROCESS_MISSING_SESSION;
  if (!request->epoch) return RW_SR_POSTPROCESS_MISSING_EPOCH;
  if (!request->width || !request->height) return RW_SR_POSTPROCESS_INVALID_GEOMETRY;
  if (!(request->x == request->x) || !(request->y == request->y))
    return RW_SR_POSTPROCESS_NONFINITE_COORDINATE;
  const rw_sr_history_binding* history = &request->history;
  if (history->struct_size < sizeof(*history) || !history->valid || !history->owner_id)
    return RW_SR_POSTPROCESS_MISSING_HISTORY;
  if (!history->resource_id || !history->authenticated)
    return RW_SR_POSTPROCESS_MISSING_HISTORY_RESOURCE;
  if (!history->ready_event || !history->completion_event)
    return RW_SR_POSTPROCESS_MISSING_HISTORY_EVENTS;
  if (!history->codec_fingerprint)
    return RW_SR_POSTPROCESS_MISSING_HISTORY_CODEC;
  if (history->session_id != request->session_id || history->epoch != request->epoch)
    return RW_SR_POSTPROCESS_HISTORY_OWNER_MISMATCH;
  if (!request->expected_predecessor || history->frame != request->expected_predecessor ||
      request->frame_id <= request->expected_predecessor)
    return RW_SR_POSTPROCESS_STALE_PREDECESSOR;
  const rw_sr_launch_binding* launch = &request->launch;
  if (launch->struct_size < sizeof(*launch) || !launch->authenticated ||
      !launch->correlation || !launch->graph_generation)
    return RW_SR_POSTPROCESS_MISSING_LAUNCH_JOIN;
  if (!request->output_valid || !request->output_owner)
    return RW_SR_POSTPROCESS_MISSING_OUTPUT_OWNER;
  if (request->output_owner == history->owner_id)
    return RW_SR_POSTPROCESS_ALIASED_OWNER;
  return RW_SR_POSTPROCESS_ACCEPTED;
}

#ifdef __cplusplus
}
#endif
#endif
