#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace rw::fg::mask {

struct MaskOutputGeometry {
    unsigned input_width, input_height;
    unsigned output_width, output_height;
    unsigned left, right, top, bottom;
};

struct MaskOutputGuides {
    // Unnormalized point-sampled float textures. Zero interpolation disables
    // the first adjustment. The optional override textures are supplied as a pair.
    cudaTextureObject_t interpolation = 0;
    cudaTextureObject_t first_override = 0;
    cudaTextureObject_t second_override = 0;
};

struct MaskOutputSurfaces {
    cudaSurfaceObject_t first;   // Resource 0x14, float2.
    cudaSurfaceObject_t second;  // Resource 0x15, float2.
    cudaSurfaceObject_t blend;   // Resource 0x06, float.
};

__device__ inline float adjust_mask_logit(float value, float weight) {
    float delta, adjusted;
    asm("sub.ftz.f32 %0, -1000.0, %1;" : "=f"(delta) : "f"(value));
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(adjusted) : "f"(weight), "f"(delta), "f"(value));
    return adjusted;
}

__device__ inline float mask_output_blend(float left, float right) {
    float weight;
    asm("{ .reg .f32 d, e, numerator; .reg .pred p;"
        "sub.ftz.f32 d, %1, %2;"
        "abs.f32 d, d;"
        "mul.ftz.f32 d, d, 0fBFB8AA3B;"
        "ex2.approx.ftz.f32 e, d;"
        "setp.gt.ftz.f32 p, %1, %2;"
        "selp.f32 numerator, 1.0, e, p;"
        "add.ftz.f32 d, e, 1.0;"
        "rcp.approx.ftz.f32 d, d;"
        "mul.ftz.f32 %0, numerator, d; }"
        : "=f"(weight) : "f"(left), "f"(right));
    return weight;
}

static __global__ void mask_output_kernel(const __half* input, MaskOutputGeometry geometry,
                                          MaskOutputGuides guides, MaskOutputSurfaces output) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= geometry.output_width || y >= geometry.output_height) return;

    const unsigned source_x = min(x / 2, geometry.input_width - geometry.left - geometry.right - 1) + geometry.left;
    const unsigned source_y = min(y / 2, geometry.input_height - geometry.top - geometry.bottom - 1) + geometry.top;
    const unsigned phase = 2 * (y % 2) + x % 2;
    // The original block transposes packed half pairs through shared memory.
    // A direct gather selects the same four values in their surface order.
    const __half* pixel = input + (size_t(source_y) * geometry.input_width + source_x) * 16 + phase;
    float2 first = make_float2(__half2float(pixel[0]), __half2float(pixel[4]));
    float2 second = make_float2(__half2float(pixel[8]), __half2float(pixel[12]));
    const float tx = float(x) + 0.5f;
    const float ty = float(y) + 0.5f;
    if (guides.interpolation) {
        const float guide = tex2D<float>(guides.interpolation, tx, ty);
        first.x = adjust_mask_logit(first.x, guide);
        second.x = adjust_mask_logit(second.x, guide);
    }
    const float blend = mask_output_blend(second.x, second.y);
    if (guides.first_override) {
        if (tex2D<float>(guides.first_override, tx, ty) > 0.99f) {
            first.x -= 1000.0f;
            first.y += 1000.0f;
        }
        if (tex2D<float>(guides.second_override, tx, ty) > 0.99f) {
            second.x -= 1000.0f;
            second.y += 1000.0f;
        }
    }
    surf2Dwrite(blend, output.blend, x * sizeof(float), y);
    surf2Dwrite(first, output.first, x * sizeof(float2), y);
    surf2Dwrite(second, output.second, x * sizeof(float2), y);
}

inline cudaError_t launch_mask_output(const __half* input, MaskOutputGeometry geometry,
                                      MaskOutputGuides guides, MaskOutputSurfaces output,
                                      cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid((geometry.output_width + block.x - 1) / block.x,
                    (geometry.output_height + block.y - 1) / block.y);
    mask_output_kernel<<<grid, block, 0, stream>>>(input, geometry, guides, output);
    return cudaGetLastError();
}

}
