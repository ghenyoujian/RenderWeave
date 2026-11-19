// Mathematical adaptation of the MIT-licensed equations identified in
// third_party/overclockers_dlssnr/NOTICE.md. No upstream runtime is linked.
#include "temporal_probe_ops.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rw::native_nr::temporal_probe {
namespace {
void image(Context& context,const Tensor& value,int channels) {
    value.numel();
    if(!context.owns(value) || value.shape.size()!=4 || value.shape[0]!=1 ||
       value.shape[1]!=channels || value.shape[2]>64 || value.shape[3]>64)
        throw std::invalid_argument("temporal math probe tensor identity/shape exceeds small profile");
}
void finite(Context& context,const Tensor& value) {
    // This bounded probe uses the existing owned DMA/fence path. It is not a
    // production validation/performance implementation.
    const auto values=context.download(value);
    for(float v:values) if(!std::isfinite(v))
        throw std::invalid_argument("non-finite temporal math probe input");
}
__device__ float add(float a,float b) { return __fadd_rn(a,b); }
__device__ float sub(float a,float b) { return __fsub_rn(a,b); }
__device__ float mul(float a,float b) { return __fmul_rn(a,b); }
__device__ float div(float a,float b) { return __fdiv_rn(a,b); }
__device__ float clamp(float a,float lo,float hi) { return fminf(fmaxf(a,lo),hi); }

struct Axis { float w0,w1,w3,wc,p0,pm,p2; };
__device__ Axis axis(float position,int extent) {
    const float c=add(floorf(sub(position,0.5f)),0.5f);
    const float t=clamp(sub(position,c),0.0f,1.0f);
    const float t2=mul(t,t),t3=mul(t2,t);
    const float w0=add(mul(add(t,t3),-0.5f),t2);
    const float w1=add(sub(mul(1.5f,t3),mul(2.5f,t2)),1.0f);
    const float w3=mul(0.5f,sub(t3,t2));
    const float w2=sub(sub(sub(1.0f,w0),w1),w3),wc=add(w1,w2);
    const float hi=sub(static_cast<float>(extent),0.5f);
    return {w0,w1,w3,wc,clamp(sub(c,1.0f),0.5f,hi),
        clamp(add(c,div(w2,wc)),0.5f,hi),clamp(add(c,2.0f),0.5f,hi)};
}
__device__ float bilinear(const float* data,int width,int height,float x,float y) {
    const float ix=floorf(sub(x,0.5f)),iy=floorf(sub(y,0.5f));
    const float tx=sub(x,add(ix,0.5f)),ty=sub(y,add(iy,0.5f));
    const int x0=max(0,min(static_cast<int>(ix),width-1));
    const int y0=max(0,min(static_cast<int>(iy),height-1));
    const int x1=max(0,min(static_cast<int>(ix)+1,width-1));
    const int y1=max(0,min(static_cast<int>(iy)+1,height-1));
    const float a=mul(mul(sub(1.0f,tx),sub(1.0f,ty)),data[y0*width+x0]);
    const float b=mul(mul(tx,sub(1.0f,ty)),data[y0*width+x1]);
    const float c=mul(mul(sub(1.0f,tx),ty),data[y1*width+x0]);
    const float d=mul(mul(tx,ty),data[y1*width+x1]);
    return add(add(add(a,b),c),d);
}
__global__ void sample_kernel(const float* history,const float* positions,float* result,
    int width,int height,size_t pixels) {
    const size_t pixel=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pixel>=pixels) return;
    const Axis x=axis(positions[pixel],width),y=axis(positions[pixels+pixel],height);
    const float k0=mul(x.w0,y.wc),k1=mul(y.w0,x.wc),k2=mul(x.wc,y.wc),
        k3=mul(y.w3,x.wc),k4=mul(x.w3,y.wc);
    const float denominator=add(add(add(add(k0,k1),k2),k3),k4);
    for(int channel=0;channel<3;++channel) {
        const float* h=history+static_cast<size_t>(channel)*width*height;
        float sum=mul(k0,bilinear(h,width,height,x.p0,y.pm));
        sum=add(sum,mul(k1,bilinear(h,width,height,x.pm,y.p0)));
        sum=add(sum,mul(k2,bilinear(h,width,height,x.pm,y.pm)));
        sum=add(sum,mul(k3,bilinear(h,width,height,x.pm,y.p2)));
        sum=add(sum,mul(k4,bilinear(h,width,height,x.p2,y.pm)));
        result[static_cast<size_t>(channel)*pixels+pixel]=div(sum,denominator);
    }
}
__global__ void position_kernel(const float* motion,float* positions,float* indices,
    int width,int height,int motion_width,int motion_height,int history_width,int history_height,
    MotionRegion region) {
    const size_t pixels=static_cast<size_t>(width)*height;
    const size_t pixel=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pixel>=pixels) return;
    const int x=static_cast<int>(pixel%width),y=static_cast<int>(pixel/width);
    const float u=div(add(static_cast<float>(x),0.5f),static_cast<float>(width));
    const float v=div(add(static_cast<float>(y),0.5f),static_cast<float>(height));
    // Preserve the readable source's region -> allocation map, not output-size
    // normalization. This probe accepts interior nearest-sampled motion only.
    const float mu=add(mul(u,div(static_cast<float>(region.width),static_cast<float>(motion_width))),
        div(static_cast<float>(region.x),static_cast<float>(motion_width)));
    const float mv=add(mul(v,div(static_cast<float>(region.height),static_cast<float>(motion_height))),
        div(static_cast<float>(region.y),static_cast<float>(motion_height)));
    const int mx=max(region.x,min(static_cast<int>(floorf(mul(mu,static_cast<float>(motion_width)))),region.x+region.width-1));
    const int my=max(region.y,min(static_cast<int>(floorf(mul(mv,static_cast<float>(motion_height)))),region.y+region.height-1));
    const size_t offset=static_cast<size_t>(my)*motion_width+mx;
    const float dx=mul(motion[offset],div(region.native_scale_x,static_cast<float>(region.width)));
    const float dy=mul(motion[static_cast<size_t>(motion_width)*motion_height+offset],
        div(region.native_scale_y,static_cast<float>(region.height)));
    positions[pixel]=mul(add(u,dx),static_cast<float>(history_width));
    positions[pixels+pixel]=mul(add(v,dy),static_cast<float>(history_height));
    indices[pixel]=static_cast<float>(mx); indices[pixels+pixel]=static_cast<float>(my);
}
__global__ void blend_kernel(const float* current,const float* residual,const float* logit,
    const float* history,float* temporal,float* display,float* gates,size_t pixels,
    float scale,float intensity,bool usable) {
    const size_t pixel=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pixel>=pixels) return;
    const float z=logit[pixel];
    // Stable mathematical sigmoid, explicitly not the unresolved vendor exp.
    const float e=expf(z>=0.0f?-z:z);
    const float sigmoid=z>=0.0f?div(1.0f,add(1.0f,e)):div(e,add(1.0f,e));
    const float gate=usable?clamp(mul(sigmoid,clamp(scale,0.0f,1.0f)),0.0f,1.0f):0.0f;
    gates[pixel]=gate;
    for(int channel=0;channel<3;++channel) {
        const size_t i=static_cast<size_t>(channel)*pixels+pixel;
        const float neural=clamp(add(current[i],mul(residual[i],0.25f)),0.0f,1.0f);
        const float value=add(mul(sub(1.0f,gate),neural),mul(gate,history[i]));
        temporal[i]=value;  // No display clamp or native history-codec claim.
        display[i]=clamp(add(current[i],mul(intensity,sub(value,current[i]))),0.0f,1.0f);
    }
}
}

Tensor sample_5tap_math(Context& context,const Tensor& history,const Tensor& positions) {
    image(context,history,3); image(context,positions,2);
    finite(context,history); finite(context,positions);
    auto output=context.empty({1,3,positions.shape[2],positions.shape[3]});
    const size_t pixels=static_cast<size_t>(positions.shape[2])*positions.shape[3];
    sample_kernel<<<static_cast<unsigned>((pixels+127)/128),128,0,context.stream()>>>(
        history.data(),positions.data(),output.data(),static_cast<int>(history.shape[3]),
        static_cast<int>(history.shape[2]),pixels);
    cuda_check(cudaGetLastError(),"temporal math five-tap software-linear sampling");
    return output;
}
WarpResult warp_math(Context& context,const Tensor& history,const Tensor& motion,
    int output_width,int output_height,const MotionRegion& region) {
    if(!region.present) throw std::invalid_argument("temporal math motion guide is missing, not zero");
    if(!region.current_to_previous) throw std::invalid_argument("explicit current-to-previous motion required");
    image(context,history,3); image(context,motion,2);
    if(output_width<=0 || output_height<=0 || output_width>64 || output_height>64 ||
       region.x<0 || region.y<0 || region.width<=0 || region.height<=0 ||
       region.x>motion.shape[3]-region.width || region.y>motion.shape[2]-region.height ||
       !std::isfinite(region.native_scale_x) || !std::isfinite(region.native_scale_y))
        throw std::invalid_argument("temporal math region/scale/dimensions invalid");
    finite(context,motion);
    WarpResult result;
    result.positions=context.empty({1,2,output_height,output_width});
    result.motion_indices=context.empty({1,2,output_height,output_width});
    const size_t pixels=static_cast<size_t>(output_width)*output_height;
    position_kernel<<<static_cast<unsigned>((pixels+127)/128),128,0,context.stream()>>>(
        motion.data(),result.positions.data(),result.motion_indices.data(),output_width,output_height,
        static_cast<int>(motion.shape[3]),static_cast<int>(motion.shape[2]),
        static_cast<int>(history.shape[3]),static_cast<int>(history.shape[2]),region);
    cuda_check(cudaGetLastError(),"temporal math motion region coordinate map");
    result.sampled=sample_5tap_math(context,history,result.positions);
    return result;
}
BlendResult blend_math(Context& context,const Tensor& current,const Tensor& residual,
    const Tensor& logit,const Tensor& history,float scale,float intensity,bool usable) {
    image(context,current,3); image(context,residual,3); image(context,history,3); image(context,logit,1);
    if(current.shape!=residual.shape || current.shape!=history.shape ||
       logit.shape[2]!=current.shape[2] || logit.shape[3]!=current.shape[3] ||
       !std::isfinite(scale) || !std::isfinite(intensity) || intensity<0.0f || intensity>1.0f)
        throw std::invalid_argument("temporal math blend geometry/controls invalid");
    finite(context,current); finite(context,residual); finite(context,history); finite(context,logit);
    BlendResult result{context.empty(current.shape),context.empty(current.shape),context.empty(logit.shape)};
    const size_t pixels=static_cast<size_t>(current.shape[2])*current.shape[3];
    blend_kernel<<<static_cast<unsigned>((pixels+127)/128),128,0,context.stream()>>>(
        current.data(),residual.data(),logit.data(),history.data(),result.temporal.data(),
        result.display.data(),result.gate.data(),pixels,scale,intensity,usable);
    cuda_check(cudaGetLastError(),"temporal math provisional history and display split");
    return result;
}
} // namespace rw::native_nr::temporal_probe
