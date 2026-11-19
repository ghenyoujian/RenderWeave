#ifndef RW_GSPLAT_H
#define RW_GSPLAT_H
#include "rw_api.h"

#if defined(RW_BUILDING_GSPLAT)
# define RW_GSPLAT_API RW_EXPORT
#else
# define RW_GSPLAT_API RW_IMPORT
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* Host adapter only: gsplat RGB+expected-camera-Z and alpha -> SDR color,
 * depth/alpha history and derived current-to-previous pixel motion/validity.
 * Caller owns every device allocation and supplies a CUDA stream. This call
 * only enqueues work; record a producer event after it for the NR submission.
 * Camera intrinsics are [fx,fy,cx,cy], transforms row-major 3x4 doubles. */
typedef struct rw_gsplat_cuda_frame {
    uint32_t struct_size, abi_version, width, height;
    const float* rgb_depth_hwc4;
    const float* alpha_hwc1;
    const float* previous_depth_hwc1;
    const float* previous_alpha_hwc1;
    float* color_hwc3;
    float* depth_hwc1;
    float* retained_alpha_hwc1;
    float* motion_hwc2;
    float* motion_validity_hwc1;
    double intrinsics[4], previous_intrinsics[4], current_to_previous[12];
    float alpha_min, absolute_depth_tolerance, relative_depth_tolerance;
    uint32_t reserved;
} rw_gsplat_cuda_frame;

RW_GSPLAT_API rw_status_t rw_gsplat_prepare_cuda(const rw_gsplat_cuda_frame*, void* cuda_stream);
RW_GSPLAT_API const char* rw_gsplat_last_error(void);
#ifdef __cplusplus
}
#endif
#endif
