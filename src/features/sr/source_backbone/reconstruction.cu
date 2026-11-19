// SPDX-License-Identifier: Apache-2.0
#include "reconstruction.hpp"
#include "reconstruction_parameters.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace rw::sr::source_backbone {
namespace {
using uchar=unsigned char;
using ushort=unsigned short;
using uint=unsigned int;
using ulong=unsigned long long;
using half=__half;

struct int2 {int x,y; __device__ int2(int a,int b):x(a),y(b) {} };
struct float2 {
    float x,y;
    __device__ float2(float a,float b):x(a),y(b) {}
};
struct float4 {
    float x,y,z,w;
    __device__ explicit float4(float a=0):x(a),y(a),z(a),w(a) {}
    __device__ float4(float a,float b,float c,float d):x(a),y(b),z(c),w(d) {}
};
struct uint4 {
    uint x,y,z,w;
    __device__ uint4(uint a,uint b,uint c,uint d):x(a),y(b),z(c),w(d) {}
};
struct bool2 {bool x,y;};
struct alignas(4) half2 {
    half x,y;
    __device__ explicit half2(float value=0):x(__float2half_rn(value)),y(x) {}
    __device__ half2(half a,half b):x(a),y(b) {}
};

template<class T,class U> __device__ T as_type(U bits) {
    static_assert(sizeof(T)==sizeof(U));
    T result;
    memcpy(&result,&bits,sizeof(T));
    return result;
}

__device__ half2 operator+(half2 a,half2 b) {return {__hadd(a.x,b.x),__hadd(a.y,b.y)};}
__device__ half2 operator-(half2 a,half2 b) {return {__hsub(a.x,b.x),__hsub(a.y,b.y)};}
__device__ half2 operator*(half2 a,half2 b) {return {__hmul(a.x,b.x),__hmul(a.y,b.y)};}
__device__ bool2 operator>(half2 a,half2 b) {return {__hgt(a.x,b.x),__hgt(a.y,b.y)};}
__device__ float clamp(float value,float lo,float hi) {return fminf(hi,fmaxf(lo,value));}
__device__ half clamp(half value,half lo,half hi) {return __hmin(hi,__hmax(lo,value));}
__device__ half2 clamp(half2 a,half2 lo,half2 hi) {
    return {__hmin(hi.x,__hmax(lo.x,a.x)),__hmin(hi.y,__hmax(lo.y,a.y))};
}
__device__ float min(float a,float b) {return fminf(a,b);}
__device__ int min(int a,int b) {return a<b ? a : b;}
__device__ float max(float a,float b) {return fmaxf(a,b);}
__device__ int max(int a,int b) {return a>b ? a : b;}
__device__ float fma(float a,float b,float c) {return __fmaf_rn(a,b,c);}
__device__ half fma(half a,half b,half c) {return __hfma(a,b,c);}
__device__ half2 fma(half2 a,half2 b,half2 c) {return {__hfma(a.x,b.x,c.x),__hfma(a.y,b.y,c.y)};}
__device__ float abs(float a) {return fabsf(a);}
__device__ half2 abs(half2 a) {return {__habs(a.x),__habs(a.y)};}
__device__ half2 exp2(half2 a) {
    uint result;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(result) : "r"(as_type<uint>(a)));
    return as_type<half2>(result);
}
__device__ uint shuffle_xor(uint value,ushort delta) {
    return __shfl_xor_sync(__activemask(),value,delta);
}
__device__ float source_div(float a,float b) {
    float result;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(a),"f"(b));
    return result;
}
__device__ float source_rcp(float a) {
    float result;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(a));
    return result;
}
__device__ float source_rsqrt(float a) {
    float result;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(a));
    return result;
}
__device__ uint bitInsert(uint bits,uint base,uint start,uint count) {
    uint result;
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(result) : "r"(bits),"r"(base),"r"(start),"r"(count));
    return result;
}

template<class T> __device__ T load(const uchar* bytes) {
    T value;
    memcpy(&value,bytes,sizeof(T));
    return value;
}

template<class T> __device__ T param_read(const uchar* params,ulong offset) {
    return offset<=328-sizeof(T) ? load<T>(params+offset) : T(0);
}

template<class T> __device__ T global(ulong address,const ushort* net,const ushort* filter,const uint* dims) {
    constexpr ulong field_base=1ull<<32, filter_base=2ull<<32;
    const ulong field_bytes=ulong(dims[4])*dims[5]*80;
    if (address>=field_base && address-field_base<=field_bytes-sizeof(T))
        return load<T>(reinterpret_cast<const uchar*>(net)+address-field_base);
    if (address>=filter_base && address-filter_base<=65536-sizeof(T))
        return load<T>(reinterpret_cast<const uchar*>(filter)+address-filter_base);
    return T(0);
}
__device__ ushort global16(ulong a,const ushort* n,const ushort* f,const uint* d) {return global<ushort>(a,n,f,d);}
__device__ uint global32(ulong a,const ushort* n,const ushort* f,const uint* d) {return global<uint>(a,n,f,d);}

template<class T> __device__ uchar* shared_address(ulong a,uchar* color,uchar* depth) {
    if (a>=0x10000000ull && a-0x10000000ull<=4608-sizeof(T)) return color+a-0x10000000ull;
    if (a>=0x10010000ull && a-0x10010000ull<=2304-sizeof(T)) return depth+a-0x10010000ull;
    return nullptr;
}
template<class T> __device__ T shared(ulong a,uchar* c,uchar* d) {
    const auto* p=shared_address<T>(a,c,d);
    return p ? load<T>(p) : T(0);
}
__device__ ushort shared16(ulong a,uchar* c,uchar* d) {return shared<ushort>(a,c,d);}
__device__ uint shared32(ulong a,uchar* c,uchar* d) {return shared<uint>(a,c,d);}
__device__ void store32(ulong a,uint value,uchar* c,uchar* d) {
    if (auto* p=shared_address<uint>(a,c,d)) memcpy(p,&value,sizeof(value));
}

__device__ int index(int x,int y,int width,int height) {
    return min(max(y,0),height-1)*width+min(max(x,0),width-1);
}

__device__ float4 old_point(const ushort* pixels,int x,int y,int width,int height,int channels) {
    if (!pixels) return float4(0);
    const half* p=reinterpret_cast<const half*>(pixels)+index(x,y,width,height)*channels;
    return channels==4 ? float4(__half2float(p[0]),__half2float(p[1]),__half2float(p[2]),__half2float(p[3]))
                       : float4(__half2float(p[0]),0,0,0);
}
__device__ float4 mix(float4 a,float4 b,float t) {
    return {a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,a.z+(b.z-a.z)*t,a.w+(b.w-a.w)*t};
}
__device__ float4 old_lookup(const ushort* data,float2 p,int width,int height,int channels,
                              bool integer,bool gather) {
    if (integer && !gather)
        return old_point(data,int(clamp(p.x,0.f,float(width-1))),int(clamp(p.y,0.f,float(height-1))),
                         width,height,channels);
    if (!integer) {p.x*=width;p.y*=height;}
    p.x=clamp(p.x-.5f,-1.f,float(width));p.y=clamp(p.y-.5f,-1.f,float(height));
    const int x=int(floorf(p.x)),y=int(floorf(p.y));
    const auto a=old_point(data,x,y,width,height,channels), b=old_point(data,x+1,y,width,height,channels);
    const auto c=old_point(data,x,y+1,width,height,channels), d=old_point(data,x+1,y+1,width,height,channels);
    if (gather) return {c.x,d.x,b.x,a.x};
    const float fx=rintf((p.x-x)*256.f)/256.f,fy=rintf((p.y-y)*256.f)/256.f;
    return mix(mix(a,b,fx),mix(c,d,fx),fy);
}

__device__ float4 current_point(ulong handle,const float* color,const float* motion,const float* depth,int i) {
    switch (handle) {
    case 1:return {color[i*3],color[i*3+1],color[i*3+2],1.f};
    case 4:return motion ? float4(motion[i*2],motion[i*2+1],0.f,0.f) : float4(0);
    case 5:return float4(depth ? depth[i] : 0.f);
    default:return float4(0);
    }
}
__device__ float4 textureRead(ulong handle,float2 coordinate,bool integer,bool gather,const float* color,
                              const ushort* previousColor,const ushort* previousLuma,const float* motion,
                              const float* depth,const uint* dims) {
    if (handle==2) return old_lookup(previousColor,coordinate,dims[6],dims[7],4,integer,gather);
    if (handle==3) return old_lookup(previousLuma,coordinate,2*dims[6],2*dims[7],1,integer,gather);
    const int width=dims[0],height=dims[1];
    if (gather) {
        const int x=int(floorf(clamp(coordinate.x-.5f,-1.f,float(width))));
        const int y=int(floorf(clamp(coordinate.y-.5f,-1.f,float(height))));
        return {current_point(handle,color,motion,depth,index(x,y+1,width,height)).x,
                current_point(handle,color,motion,depth,index(x+1,y+1,width,height)).x,
                current_point(handle,color,motion,depth,index(x+1,y,width,height)).x,
                current_point(handle,color,motion,depth,index(x,y,width,height)).x};
    }
    return current_point(handle,color,motion,depth,index(int(clamp(coordinate.x,0.f,float(width-1))),
                         int(clamp(coordinate.y,0.f,float(height-1))),width,height));
}

__device__ void half_surface(ushort* data,int width,int height,int channels,int2 p,uint4 v,bool bytes) {
    if (p.y<0 || p.y>=height || p.x<0) return;
    if (bytes) {
        const auto pitch=width*channels*2;
        if (p.x<=pitch-4) memcpy(reinterpret_cast<uchar*>(data)+p.y*pitch+p.x,&v.x,4);
    } else if (p.x<width && channels==4) {
        auto* item=reinterpret_cast<half*>(data)+4*(p.y*width+p.x);
        // The formatted PTX surface write converts FP32 to FP16 toward zero.
        item[0]=__float2half_rz(as_type<float>(v.x));item[1]=__float2half_rz(as_type<float>(v.y));
        item[2]=__float2half_rz(as_type<float>(v.z));item[3]=__float2half_rz(as_type<float>(v.w));
    }
}

__device__ void surfaceWrite(ulong handle,int2 p,uint4 v,bool bytes,float* rgb,ushort* history,
                              ushort* luma,ushort* hidden,const uint* dims) {
    if (handle==7) half_surface(history,dims[6],dims[7],4,p,v,bytes);
    else if (handle==8) half_surface(luma,2*dims[6],2*dims[7],1,p,v,bytes);
    else if (handle==9) half_surface(hidden,dims[4],dims[5],4,p,v,bytes);
    else if (handle==6 && p.y>=0 && p.y<int(dims[3]) && p.x>=0) {
        const int width=dims[2];
        if (bytes && p.x<=width*16-4) memcpy(reinterpret_cast<uchar*>(rgb)+p.y*width*16+p.x,&v.x,4);
        else if (!bytes && p.x<width) memcpy(rgb+4*(p.y*width+p.x),&v,16);
    }
}

// All three history planes have CUDA-allocation alignment and dimensions
// aligned to at least sixteen pixels, so their byte lengths divide by 16.
__global__ void clear_bound_history(const HistoryBinding* history,std::size_t color_blocks,
                                    std::size_t luma_blocks,std::size_t hidden_blocks) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto zero=::make_uint4(0,0,0,0);
    if(i<color_blocks) reinterpret_cast<::uint4*>(history->next_ycocg)[i]=zero;
    if(i<luma_blocks) reinterpret_cast<::uint4*>(history->next_luma)[i]=zero;
    if(i<hidden_blocks) reinterpret_cast<::uint4*>(history->next_hidden)[i]=zero;
}

template<bool bound_history>
__global__ __launch_bounds__(256) void reconstruct_k(detail::ReconstructionParameters p,const ushort* net,const ushort* filter,
    const float* color,const ushort* previousColor,const ushort* previousLuma,const float* motion,const float* depth,
    float* rgb,ushort* history,ushort* luma,ushort* hidden,const HistoryBinding* binding) {
    if constexpr(bound_history) {
        previousColor=binding->previous_ycocg;
        previousLuma=binding->previous_luma;
        history=binding->next_ycocg;
        luma=binding->next_luma;
        hidden=binding->next_hidden;
    }
    const uchar* params=p.block;
    const uint* dims=p.dimensions;
#include "rw_sr_k_reconstruction_body.inc"
}

std::size_t pixels(Geometry g) {return std::size_t(g.width)*g.height;}

bool valid_frame(const PrepareInput& input,const ushort* previous_luma,const ushort* filter,ReconstructionOutput out) {
    return input.width>=1 && input.width<=4096 && input.height>=1 && input.height<=4096 && input.color_rgb &&
        out.rgba && out.ycocg && out.luma && out.hidden && filter && input.exposure==1.f &&
        input.jitter_x==0.f && input.jitter_y==0.f && (input.reset || (input.motion_xy && input.depth &&
        input.previous_ycocg && previous_luma && input.previous_hidden && input.motion_scale_x==2.f &&
        input.motion_scale_y==2.f && input.motion_bias_x==0.f && input.motion_bias_y==0.f &&
        input.previous_ycocg!=out.ycocg && previous_luma!=out.luma && input.previous_hidden!=out.hidden));
}
} // namespace

SurfaceGeometry reconstruction_geometry(int width,int height) {
    const auto neural=network_geometry(width,height);
    const Geometry output{2*width,2*height};
    const Geometry history{(output.width+15)/16*16,(output.height+15)/16*16};
    return {output,history,{2*history.width,2*history.height},neural};
}

std::vector<std::uint16_t> decode_reconstruction_filter(std::span<const std::byte> raw) {
    constexpr std::array<uchar,32> expected={0x88,0xaa,0x97,0xaa,0x1e,0x2c,0xb9,0xe2,0xe7,0xc9,0x32,0x47,0x6b,0x24,0x79,0x3a,
        0x4f,0xb9,0x98,0xa9,0x7c,0x76,0xc3,0xb3,0x51,0x21,0x22,0x24,0xbc,0x00,0xab,0x3c};
    std::array<uchar,32> digest{};
    if (raw.size()!=filter_elements*2 || !SHA256(reinterpret_cast<const uchar*>(raw.data()),raw.size(),digest.data()) || digest!=expected)
        throw std::invalid_argument("SR K reconstruction filter differs from the registered 2x coefficients");
    std::vector<std::uint16_t> result(filter_elements);
    for (std::size_t i=0;i<result.size();++i)
        result[i]=std::to_integer<ushort>(raw[i*2]) | (std::to_integer<ushort>(raw[i*2+1])<<8);
    return result;
}

cudaError_t reconstruct(const PrepareInput& input,const std::uint16_t* previous_luma,
    const std::uint16_t* field,const std::uint16_t* filter,ReconstructionOutput out,cudaStream_t stream,
    const HistoryBinding* history) {
    if (!field || !valid_frame(input,previous_luma,filter,out)) return cudaErrorInvalidValue;
    const auto g=reconstruction_geometry(input.width,input.height);
    if (const auto status=cudaMemsetAsync(out.rgba,0,pixels(g.output)*16,stream);status!=cudaSuccess) return status;
    const dim3 grid(g.history.width/16,g.history.height/16), block(16,16);
    const auto params=detail::reconstruction_parameters(input.width,input.height,input.reset);
    if(history) {
        const auto color_blocks=pixels(g.history)/2, luma_blocks=pixels(g.luma)/8, hidden_blocks=pixels(g.neural)/2;
        const auto blocks=(std::max({color_blocks,luma_blocks,hidden_blocks})+255)/256;
        clear_bound_history<<<blocks,256,0,stream>>>(history,color_blocks,luma_blocks,hidden_blocks);
        if(const auto status=cudaGetLastError();status!=cudaSuccess) return status;
        reconstruct_k<true><<<grid,block,0,stream>>>(params,field,filter,input.color_rgb,nullptr,nullptr,
            input.motion_xy,input.depth,out.rgba,nullptr,nullptr,nullptr,history);
    } else {
        for (const auto& plane : std::array<std::pair<void*,std::size_t>,3>{{
                {out.ycocg,pixels(g.history)*8},{out.luma,pixels(g.luma)*2},{out.hidden,pixels(g.neural)*8}}})
            if (const auto status=cudaMemsetAsync(plane.first,0,plane.second,stream);status!=cudaSuccess) return status;
        reconstruct_k<false><<<grid,block,0,stream>>>(params,field,filter,input.color_rgb,
            input.reset ? nullptr : input.previous_ycocg,input.reset ? nullptr : previous_luma,
            input.motion_xy,input.depth,out.rgba,out.ycocg,out.luma,out.hidden,nullptr);
    }
    return cudaGetLastError();
}

cudaError_t enqueue_frame(const PrepareInput& input,const std::uint16_t* previous_luma,unsigned phase,
    const Weights& layout,const std::uint16_t* device_weights,const std::uint16_t* device_filter,
    std::uint16_t* prepared,std::uint16_t* field,ReconstructionOutput output,void* workspace,
    std::size_t bytes,cudaStream_t stream,BackboneOptions backbone_options,const cudaEvent_t* stage_ends,
    const HistoryBinding* history) {
    if ((input.reset && phase!=0) || !valid_frame(input,previous_luma,device_filter,output))
        return cudaErrorInvalidValue;
    const auto geometry=network_geometry(input.width,input.height);
    if (const auto status=prepare(input,prepared,stream,history);status!=cudaSuccess) return status;
    if (stage_ends) {
        if (const auto status=cudaEventRecord(stage_ends[0],stream);status!=cudaSuccess) return status;
    }
    if (const auto status=enqueue(geometry,phase,layout,device_weights,prepared,field,workspace,bytes,stream,backbone_options);
        status!=cudaSuccess) return status;
    if (stage_ends) {
        if (const auto status=cudaEventRecord(stage_ends[1],stream);status!=cudaSuccess) return status;
    }
    if (const auto status=reconstruct(input,previous_luma,field,device_filter,output,stream,history);
        status!=cudaSuccess) return status;
    return stage_ends ? cudaEventRecord(stage_ends[2],stream) : cudaSuccess;
}
} // namespace rw::sr::source_backbone
