// SPDX-License-Identifier: Apache-2.0
// Based on the preparation equations in MLX-DLSS DLSSSuperResolution.swift.
#include "prepare.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace rw::sr::source_backbone {
namespace {

__device__ int clamped(int value,int extent) { return min(max(value,0),extent-1); }

__device__ int nearest(float value,int extent) {
    const float p=fminf(fmaxf(value,0.f),static_cast<float>(extent-1));
    return static_cast<int>(truncf(p+0.5f));
}

__device__ int pixel(int x,int y,int width,int height) {
    return clamped(y,height)*width+clamped(x,width);
}

__device__ float3 ycocg(const PrepareInput& input,int x,int y) {
    const auto* rgb=input.color_rgb+pixel(x,y,input.width,input.height)*3;
    const float r=fmaxf(rgb[0],0.f),g=fmaxf(rgb[1],0.f),b=fmaxf(rgb[2],0.f);
    return make_float3(fmaf(b,.25f,fmaf(r,.25f,g*.5f)),(r-b)*0.5f,g*0.5f-r*0.25f-b*0.25f);
}

__device__ float2 flow(const PrepareInput& input,int x,int y) {
    const float center=input.depth[pixel(x,y,input.width,input.height)];
    constexpr int dx[]={-1,1,-1,1},dy[]={-1,-1,1,1};
    float values[4],closest=center;
    int offset_x=0,offset_y=0;
    for (int i=0;i<4;++i) {
        values[i]=input.depth[pixel(x+dx[i],y+dy[i],input.width,input.height)];
        if (values[i]<closest) {closest=values[i];offset_x=dx[i];offset_y=dy[i];}
    }
    const bool edge=center>center*.001f+(values[0]+values[3])*.5f ||
                    center>center*.001f+(values[1]+values[2])*.5f;
    const auto* motion=input.motion_xy+2*pixel(x+(edge ? offset_x : 0),y+(edge ? offset_y : 0),
                                              input.width,input.height);
    return make_float2((motion[0]+input.motion_bias_x)*input.motion_scale_x,
                       (motion[1]+input.motion_bias_y)*input.motion_scale_y);
}

__device__ void linear(const std::uint16_t* data,int width,int height,float x,float y,
                       float* output,int channels) {
    const auto* texture=reinterpret_cast<const __half*>(data);
    x-=0.5f;y-=0.5f;
    const int ix=static_cast<int>(floorf(x)),iy=static_cast<int>(floorf(y));
    const float tx=rintf((x-ix)*256.f)/256.f,ty=rintf((y-iy)*256.f)/256.f;
    const int indices[]={pixel(ix,iy,width,height),pixel(ix+1,iy,width,height),
                         pixel(ix,iy+1,width,height),pixel(ix+1,iy+1,width,height)};
    for (int c=0;c<channels;++c) {
        const float a=__half2float(texture[indices[0]*4+c]),b=__half2float(texture[indices[1]*4+c]);
        const float d=__half2float(texture[indices[2]*4+c]),e=__half2float(texture[indices[3]*4+c]);
        const float top=a+(b-a)*tx,bottom=d+(e-d)*tx;
        output[c]=top+(bottom-top)*ty;
    }
}

__device__ float compress(float x,float exposure) {
    // The K/LDR preparation kernel rounds these intermediates before converting to FP16.
    const float divided=__fdividef(x,fmaxf(1.f-x,.2f));
    const float z=divided*exposure;
    const float denominator=__fadd_rn(z,1.f);
    float reciprocal;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(denominator));
    const float y=__saturatef(z*fmaxf(reciprocal,.0001f));
    if (y<=.0031308f) return 12.92f*y;
    const float exponent=__log2f(y)*(1.f/2.4f);
    float raised;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(raised) : "f"(exponent));
    return fmaf(raised,1.055f,-.055f);
}

__device__ void neighborhood(const PrepareInput& input,int cx,int cy,float* mean,float* spread) {
    float minimum[]={INFINITY,INFINITY,INFINITY},maximum[]={-INFINITY,-INFINITY,-INFINITY};
    for (int y=-1;y<=1;++y) for (int x=-1;x<=1;++x) {
        const auto sample=ycocg(input,cx+x,cy+y);
        const float lanes[]={sample.x,sample.y,sample.z};
        for (int ch=0;ch<3;++ch) {
            mean[ch]+=lanes[ch];
            const float expanded=lanes[ch]*(1.f+3.f*sample.x);
            minimum[ch]=fminf(minimum[ch],expanded);maximum[ch]=fmaxf(maximum[ch],expanded);
        }
    }
    for (int ch=0;ch<3;++ch) {mean[ch]/=9.f;spread[ch]=maximum[ch]-minimum[ch];}
}

template<bool bound_history>
__global__ void prepare_field(PrepareInput input,Geometry geometry,__half* output,const HistoryBinding* history) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if (id>=geometry.width*geometry.height) return;
    if constexpr(bound_history) {
        input.previous_ycocg=history->previous_ycocg;
        input.previous_hidden=history->previous_hidden;
    }
    const int nx=id%geometry.width, ny=id/geometry.width;
    const int out_w=2*input.width,out_h=2*input.height;
    const int hist_w=(out_w+15)/16*16,hist_h=(out_h+15)/16*16;
    const auto low_x=[&](float x) {return nearest(x*.5f-.5f+input.jitter_x,input.width);};
    const auto low_y=[&](float y) {return nearest(y*.5f-.5f+input.jitter_y,input.height);};
    float features[4][4]{};
    constexpr int chosen[]={1,3,0,2};
    for (int q=0;q<4;++q) {
        const int sx=nx*4+(q%2)*2,sy=ny*4+(q/2)*2;
        const int pick=chosen[q];
        const float current=ycocg(input,low_x(sx+pick%2+.5f),low_y(sy+pick/2+.5f)).x;
        auto* lane=output+id*16+q*4;
        lane[0]=lane[1]=__float2half_rn(compress(current,input.exposure));
        lane[3]=__float2half_rn(0.f);
        if (input.reset) continue;

        float2 offsets[4],coords[4];
        bool valid=true;
        for (int j=0;j<4;++j) {
            const float x=sx+j%2+.5f,y=sy+j/2+.5f;
            offsets[j]=flow(input,low_x(x),low_y(y));
            const auto p=make_float2(x+offsets[j].x,y+offsets[j].y);
            coords[j]=p;
            valid=valid && p.x>=0.f && p.x<=out_w && p.y>=0.f && p.y<=out_h;
        }
        if (!valid) continue;

        float mean[3]{},spread[3];
        neighborhood(input,low_x(sx+1.f),low_y(sy+1.f),mean,spread);
        float rejection=0.f;
        for (int j=0;j<4;++j) {
            float old[3],outside[3];
            linear(input.previous_ycocg,hist_w,hist_h,coords[j].x,coords[j].y,old,3);
            if (j==pick) lane[1]=__float2half_rn(compress(old[0],input.exposure));
            for (int ch=0;ch<3;++ch) {
                const float distance=fabsf(mean[ch]*(1.f+3.f*mean[0])-old[ch]*(1.f+3.f*old[0]));
                outside[ch]=fmaxf(distance-.5f*spread[ch],0.f)/(1.f+8.f*spread[ch]);
            }
            rejection+=fminf(2.f*outside[0]+.5f*outside[1]+.5f*outside[2],1.f);
        }
        lane[3]=__float2half_rn(rejection*.25f);
        linear(input.previous_hidden,geometry.width,geometry.height,
               ((nx+.5f)*4.f+offsets[pick].x)*.25f,((ny+.5f)*4.f+offsets[pick].y)*.25f,
               features[q],4);
    }
    for (int q=0;q<4;++q)
        output[id*16+q*4+2]=__float2half_rn(((features[0][q]+features[3][q])+
                                              (features[1][q]+features[2][q]))*.25f);
}

} // namespace

cudaError_t prepare(const PrepareInput& input,std::uint16_t* prepared,cudaStream_t stream,
                    const HistoryBinding* history) {
    if (!input.color_rgb || !prepared || input.width<1 || input.width>4096 ||
        input.height<1 || input.height>4096 || !std::isfinite(input.exposure) || input.exposure<=0.f ||
        !std::isfinite(input.jitter_x) || !std::isfinite(input.jitter_y) ||
        (!input.reset && (!input.motion_xy || !input.depth || !input.previous_ycocg || !input.previous_hidden ||
          !std::isfinite(input.motion_scale_x) || !std::isfinite(input.motion_scale_y) ||
          !std::isfinite(input.motion_bias_x) || !std::isfinite(input.motion_bias_y))))
        return cudaErrorInvalidValue;
    const auto shape=network_geometry(input.width,input.height);
    const auto blocks=(shape.width*shape.height+255)/256;
    auto* target=reinterpret_cast<__half*>(prepared);
    if(history) {
        auto bound=input;
        bound.previous_ycocg=nullptr;
        bound.previous_hidden=nullptr;
        prepare_field<true><<<blocks,256,0,stream>>>(bound,shape,target,history);
    } else {
        prepare_field<false><<<blocks,256,0,stream>>>(input,shape,target,nullptr);
    }
    return cudaGetLastError();
}

} // namespace rw::sr::source_backbone
