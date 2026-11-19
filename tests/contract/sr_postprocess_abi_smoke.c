#include "rw/sr/postprocess.h"
#include <assert.h>
#include <string.h>

int main(void) {
  rw_sr_catmull_rom_submission request;
  memset(&request, 0, sizeof(request));
  request.struct_size = sizeof(request);
  request.session_id = 7;
  request.epoch = 3;
  request.frame_id = 8;
  request.expected_predecessor = 7;
  request.output_owner = 200;
  request.output_valid = 1;
  request.width = request.height = 64;
  request.x = 4.5f;
  request.y = 7.25f;
  request.history.struct_size = sizeof(request.history);
  request.history.valid = 1;
  request.history.authenticated = 1;
  request.history.resource_id = 300;
  request.history.owner_id = 100;
  request.history.session_id = 7;
  request.history.epoch = 3;
  request.history.frame = 7;
  request.history.codec_fingerprint = 500;
  request.history.ready_event = 600;
  request.history.completion_event = 601;
  request.launch.struct_size = sizeof(request.launch);
  request.launch.authenticated = 1;
  request.launch.correlation = 400;
  request.launch.graph_generation = 2;
  assert(rw_sr_postprocess_admit_catmull_rom(&request) == RW_SR_POSTPROCESS_ACCEPTED);
  request.history.ready_event = 0;
  assert(rw_sr_postprocess_admit_catmull_rom(&request) == RW_SR_POSTPROCESS_MISSING_HISTORY_EVENTS);
  return 0;
}
