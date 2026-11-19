#include "rw/rw_gsplat.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace rw::gsplat {
thread_local char last_error[512]{};
__global__ void prepare(rw_gsplat_cuda_frame f) {
    const size_t count=size_t(f.width)*f.height;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x) {
        for(int c=0;c<3;++c) {
            const float value=f.rgb_depth_hwc4[4*i+c];
            // Preserve NaN for the SDK's device input check; finite HDR values
            // use this adapter's explicit SDR clipping policy.
            f.color_hwc3[3*i+c]=!isfinite(value)?value:fminf(fmaxf(value,0.0f),1.0f);
        }
        const float z=f.rgb_depth_hwc4[4*i+3],alpha=f.alpha_hwc1[i];
        f.depth_hwc1[i]=z;f.retained_alpha_hwc1[i]=alpha;
        float dx=0,dy=0;bool valid=false;
        if(f.previous_depth_hwc1 && isfinite(z) && z>0 && isfinite(alpha) && alpha>=f.alpha_min) {
            const double x=double(i%f.width)+0.5,y=double(i/f.width)+0.5;
            const double cx=(x-f.intrinsics[2])/f.intrinsics[0]*double(z);
            const double cy=(y-f.intrinsics[3])/f.intrinsics[1]*double(z);
            const double* t=f.current_to_previous;
            const double px=((cx*t[0]+cy*t[1])+double(z)*t[2])+t[3];
            const double py=((cx*t[4]+cy*t[5])+double(z)*t[6])+t[7];
            const double pz=((cx*t[8]+cy*t[9])+double(z)*t[10])+t[11];
            if(isfinite(px) && isfinite(py) && isfinite(pz) && pz>0) {
                const double u=f.previous_intrinsics[0]*px/pz+f.previous_intrinsics[2];
                const double v=f.previous_intrinsics[1]*py/pz+f.previous_intrinsics[3];
                if(u>=0 && u<double(f.width) && v>=0 && v<double(f.height)) {
                    const size_t previous=size_t(v)*f.width+size_t(u);
                    const float old_z=f.previous_depth_hwc1[previous],old_alpha=f.previous_alpha_hwc1[previous];
                    const float tolerance=__fadd_rn(f.absolute_depth_tolerance,
                        __fmul_rn(f.relative_depth_tolerance,fabsf(old_z)));
                    valid=isfinite(old_z) && old_z>0 && isfinite(old_alpha) && old_alpha>=f.alpha_min &&
                          fabs(pz-double(old_z))<=double(tolerance);
                    if(valid) {
                        // Preserve the existing CPU normalized-f32 guide's
                        // publication, then its explicit pixel-unit conversion.
                        dx=__fmul_rn(float(u/double(f.width)-x/double(f.width)),float(f.width));
                        dy=__fmul_rn(float(v/double(f.height)-y/double(f.height)),float(f.height));
                    }
                }
            }
        }
        f.motion_hwc2[2*i]=dx;f.motion_hwc2[2*i+1]=dy;
        f.motion_validity_hwc1[i]=valid?1.0f:0.0f;
    }
}
}

extern "C" {
const char* rw_gsplat_last_error() {return rw::gsplat::last_error;}
rw_status_t rw_gsplat_prepare_cuda(const rw_gsplat_cuda_frame* frame,void* stream) {
    auto* error=rw::gsplat::last_error;error[0]=0;
    if(!frame || frame->struct_size!=sizeof(*frame) || frame->abi_version!=RW_ABI_VERSION)return RW_ABI_MISMATCH;
    const auto& f=*frame;
    if(!f.width || !f.height || f.reserved || !f.rgb_depth_hwc4 || !f.alpha_hwc1 ||
       !f.color_hwc3 || !f.depth_hwc1 || !f.retained_alpha_hwc1 || !f.motion_hwc2 || !f.motion_validity_hwc1 ||
       bool(f.previous_depth_hwc1)!=bool(f.previous_alpha_hwc1) ||
       !(f.intrinsics[0]>0) || !(f.intrinsics[1]>0))return RW_INVALID_ARGUMENT;
    const size_t count=size_t(f.width)*f.height;
    rw::gsplat::prepare<<<unsigned((count+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(f);
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess) {std::snprintf(error,512,"gsplat CUDA guide enqueue: %s",cudaGetErrorString(status));return RW_INTERNAL_ERROR;}
    return RW_OK;
}
}
