#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace rw::fg::mask {

struct MaskInputGeometry {
    unsigned color_width, color_height;
    unsigned mask_width, mask_height;
    unsigned motion_width, motion_height;
    int mask_origin_x, mask_origin_y;
};

struct MaskInputTextures {
    // Resource 0x03 and 0x0f both have an integer-addressed point read and
    // a separate normalized linear read of the same underlying color plane.
    cudaTextureObject_t first_color_point, first_color_linear;
    cudaTextureObject_t second_color_point, second_color_linear;
    cudaTextureObject_t guide_13, guide_07, guide_05, guide_10, guide_06;
    cudaTextureObject_t motion_uv_28, motion_pixels_29, guide_2a;
};

struct MaskInputOutput {
    uint32_t* packed;                      // Resource 0x17, NHWC48.
    cudaSurfaceObject_t product_guide;     // Resource 0x16, float.
    cudaSurfaceObject_t color_difference;  // Resource 0x12, float2.
};

__device__ inline float input_reciprocal(float value) {
    float result;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ inline float input_luma(float4 color) {
    float result;
    asm("{ .reg .f32 r;"
        "mul.ftz.f32 r, %2, 0.587;"
        "fma.rn.ftz.f32 r, %1, 0.299, r;"
        "fma.rn.ftz.f32 r, %3, 0.114, r;"
        "add.ftz.sat.f32 %0, r, 0.0; }"
        : "=f"(result) : "f"(color.x), "f"(color.y), "f"(color.z));
    return result;
}

__device__ inline float input_color_difference(float4 sampled, float4 color) {
    float result;
    asm("{ .reg .f32 r,g,b;"
        "sub.ftz.f32 r, %1, %4;"
        "sub.ftz.f32 g, %2, %5;"
        "sub.ftz.f32 b, %3, %6;"
        "abs.f32 r, r; abs.f32 g, g; abs.f32 b, b;"
        "add.ftz.f32 r, r, g;"
        "add.ftz.f32 r, b, r;"
        "mul.ftz.sat.f32 %0, r, 0f3EAAAAAB; }"
        : "=f"(result) : "f"(sampled.x), "f"(sampled.y), "f"(sampled.z),
          "f"(color.x), "f"(color.y), "f"(color.z));
    return result;
}

static __global__ void mask_input_kernel(MaskInputTextures t, MaskInputGeometry g, MaskInputOutput out) {
    __shared__ __half tile[8 * 8][40];
    const unsigned phase = (threadIdx.y % 2) * 2 + threadIdx.x % 2;
    const unsigned local_pixel = (threadIdx.y / 2) * 8 + threadIdx.x / 2;
    const int x = int(blockIdx.x * 16 + threadIdx.x) - 2 * g.mask_origin_x;
    const int y = int(blockIdx.y * 16 + threadIdx.y) - 2 * g.mask_origin_y;
    float values[10]{};
    const unsigned mask_x = blockIdx.x * 8 + threadIdx.x / 2;
    const unsigned mask_y = blockIdx.y * 8 + threadIdx.y / 2;
    if (mask_x < g.mask_width && mask_y < g.mask_height && x >= 0 && y >= 0 &&
        unsigned(x) < g.color_width && unsigned(y) < g.color_height) {
        const float px = float(x) + 0.5f, py = float(y) + 0.5f;
        const float u = __saturatef(px * input_reciprocal(float(g.color_width)));
        const float v = __saturatef(py * input_reciprocal(float(g.color_height)));
        const float4 first = tex2D<float4>(t.first_color_point, px, py);
        const float4 second = tex2D<float4>(t.second_color_point, px, py);
        const float2 uv = tex2D<float2>(t.motion_uv_28, u, v);
        const float2 motion = tex2D<float2>(t.motion_pixels_29, u, v);
        float2 first_uv, second_uv;
        asm("add.ftz.f32 %0, %1, %2;" : "=f"(first_uv.x) : "f"(u), "f"(uv.x));
        asm("add.ftz.f32 %0, %1, %2;" : "=f"(first_uv.y) : "f"(v), "f"(uv.y));
        const float motion_x_scale = input_reciprocal(float(g.motion_width));
        const float motion_y_scale = input_reciprocal(float(g.motion_height));
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(second_uv.x) : "f"(motion.x), "f"(motion_x_scale), "f"(u));
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(second_uv.y) : "f"(motion.y), "f"(motion_y_scale), "f"(v));
        const float4 warp_first = tex2D<float4>(t.first_color_linear, first_uv.x, first_uv.y);
        const float4 warp_second = tex2D<float4>(t.second_color_linear, second_uv.x, second_uv.y);
        const float scalar13 = tex2D<float>(t.guide_13, u, v);
        const float scalar2a = tex2D<float>(t.guide_2a, u, v);
        const float scalar07 = tex2D<float>(t.guide_07, u, v);
        float product;
        asm("{ .reg .f32 pair; mul.ftz.f32 pair, %1, %2; fma.rn.ftz.sat.f32 %0, %3, pair, 0.0; }"
            : "=f"(product) : "f"(scalar2a), "f"(scalar07), "f"(scalar13));
        const float first_error = input_color_difference(warp_first, second);
        const float second_error = input_color_difference(warp_second, first);
        const float scalar06 = tex2D<float>(t.guide_06, px, py);
        surf2Dwrite(product, out.product_guide, x * sizeof(float), y);
        surf2Dwrite(make_float2(first_error, second_error), out.color_difference, x * sizeof(float2), y);
        values[0] = tex2D<float>(t.guide_10, px, py);
        values[1] = input_luma(first);
        values[2] = second_error;
        values[3] = tex2D<float>(t.guide_05, px, py);
        values[4] = scalar2a;
        values[5] = input_luma(second);
        values[6] = first_error;
        values[7] = scalar13;
        values[8] = scalar06;
        asm("sub.ftz.f32 %0, 1.0, %1;" : "=f"(values[9]) : "f"(scalar06));
    }
    #pragma unroll
    for (unsigned feature = 0; feature < 10; ++feature)
        tile[local_pixel][4 * feature + phase] = __float2half_rn(values[feature]);
    __syncthreads();

    const unsigned lane = threadIdx.y * 16 + threadIdx.x;
    #pragma unroll
    for (unsigned index = lane; index < 8 * 8 * 24; index += 16 * 16) {
        const unsigned pixel = index / 24, pair = index % 24;
        const unsigned dx = blockIdx.x * 8 + pixel % 8, dy = blockIdx.y * 8 + pixel / 8;
        if (dx < g.mask_width && dy < g.mask_height) {
            uint32_t packed = 0;
            if (pair < 20) {
                packed = __half_as_ushort(tile[pixel][2 * pair]);
                packed |= uint32_t(__half_as_ushort(tile[pixel][2 * pair + 1])) << 16;
            }
            out.packed[(size_t(dy) * g.mask_width + dx) * 24 + pair] = packed;
        }
    }
}

inline cudaError_t launch_mask_input(MaskInputTextures textures, MaskInputGeometry geometry,
                                     MaskInputOutput output, cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid((geometry.mask_width + 7) / 8, (geometry.mask_height + 7) / 8);
    mask_input_kernel<<<grid, block, 0, stream>>>(textures, geometry, output);
    return cudaGetLastError();
}

}
