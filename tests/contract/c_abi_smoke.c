#include "rw/rw_api.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    rw_context_desc descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = RW_ABI_VERSION;
    descriptor.device_kind = RW_DEVICE_CPU;
    descriptor.max_sessions = descriptor.max_inflight = descriptor.execution_slots = 1;
    rw_context_t context = NULL;
    if (rw_context_create(&descriptor, &context) != RW_OK) return 1;
    size_t count = 0;
    if (rw_context_query(context, "capabilities", NULL, &count) != RW_OK || count == 0) return 2;
    char output[1024];
    count = sizeof(output);
    if (rw_context_query(context, "capabilities", output, &count) != RW_OK) return 3;
    if (strstr(output, "\"native_model_execution_available\":false") == NULL) return 4;
    if (rw_e4m3_encode(448.0f) != 0x7e || rw_e4m3_decode(0x38) != 1.0f) return 5;
    rw_context_release(context);
    puts("{\"scope\":\"c_abi_smoke\",\"passed\":true}");
    return 0;
}
