// SPDX-License-Identifier: Apache-2.0
// CUDA translation of MLX-DLSS's K/LDR/2x neural backbone equations.
#include "shape.hpp"
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <stdexcept>

namespace rw::sr::source_backbone {
namespace {
using Half = __half;
constexpr unsigned threads=256;

// All allocations are half arrays. The first region holds two live image
// planes and five encoder skips. Each stage reuses the remaining region.
struct Arena {
    std::size_t cursor=0;
    std::size_t take(std::size_t count) {
        const auto offset=cursor;
        cursor+=count;
        return offset;
    }
};
struct StageScratch {
    std::size_t gathered,x,norm,attention,v,q,k,probabilities,values,ff_input,ff,folded,projected;
};

StageScratch scratch(Arena& arena,const Stage& shape,int stage) {
    const auto tokens=shape.windows()*64;
    const int c=channels[stage];
    const auto gathered=stage>=6 ? tokens/4*channels[stage-1] : tokens*(stage==0 ? 16 : c);
    const auto projected=stage<5 ? tokens/4*merge_channels(stage) : tokens*(stage==10 ? 48 : c);
    const auto attention_dynamic=stage!=0 && stage!=1 && stage!=10;
    return {arena.take(gathered),arena.take(tokens*c),arena.take(tokens*c),arena.take(tokens*c),
        arena.take(tokens*32),arena.take(attention_dynamic ? tokens*32 : 0),
        arena.take(attention_dynamic ? tokens*32 : 0),arena.take(attention_dynamic ? tokens*64 : 0),
        arena.take(tokens*32),arena.take(tokens*c),arena.take(tokens*4*c),
        arena.take(stage<5 ? tokens*c : 0),arena.take(projected)};
}

struct Persistent {
    std::size_t image[2],skip[5],end;
};
Persistent persistent(Geometry field,const std::array<Stage,stage_count>& geometries) {
    Arena arena;
    // All internal field shapes have at most 64 channels per full-resolution
    // pixel: finer stages have at most 64, coarser ones have 1/4 as many pixels.
    const auto capacity=std::size_t(field.width)*field.height*64;
    Persistent result{};
    for (auto& image:result.image) image=arena.take(capacity);
    for (int i=0;i<5;++i)
        result.skip[i]=arena.take(std::size_t(geometries[i].width)*geometries[i].height*channels[i]);
    result.end=arena.cursor;
    return result;
}

__device__ float f(Half x) {return __half2float(x);}
__device__ Half h(float x) {return __float2half_rn(x);}

__global__ void gather(const Half* image,Half* tiles,Stage shape,int c,int scale,std::size_t count) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=count) return;
    const int side=8/scale;
    const auto row=index/c;
    const int window=row/(side*side), pixel=row%(side*side);
    const int w=shape.width/scale, height=shape.height/scale;
    const int xx=abs((window%shape.grid_x*8-shape.x)/scale+pixel%side);
    const int yy=abs((window/shape.grid_x*8-shape.y)/scale+pixel/side);
    const int x=min(xx,2*(w-1)-xx), y=min(yy,2*(height-1)-yy);
    tiles[index]=image[(std::size_t(y)*w+x)*c+index%c];
}

// Reference reduction. Retire each eight-product sum to FP16 in K order.
// Weights are row-major, except a transposed, window-local K operand for QK.
__global__ void matrix(const Half* a,const Half* b,const Half* bias,Half* result,
    std::size_t rows,int inner,int cols,int bias_rows,bool rhs_transposed) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=rows*cols) return;
    const auto row=index/cols;
    const int column=index%cols;
    const auto ai=row*inner;
    float accumulator=bias ? f(bias[(bias_rows ? row%bias_rows : row)*cols+column]) : 0.f;
    for (int k=0;k<inner;k+=8) {
        float partial=0.f;
        #pragma unroll
        for (int i=0;i<8;++i) {
            const auto rhs=rhs_transposed ? ((row/64)*64+column)*inner+(k+i)
                                          : std::size_t(k+i)*cols+column;
            partial=__fmaf_rn(f(a[ai+k+i]),f(b[rhs]),partial);
        }
        accumulator=f(h(accumulator+partial));
    }
    result[index]=h(accumulator);
}

__global__ void window_values(const Half* probabilities,const Half* values,Half* output,
                              std::size_t rows,bool fixed) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=rows*32) return;
    const auto row=index/32;
    const auto p=(fixed ? row%64 : row)*64;
    const auto v=(row/64)*64*32+index%32;
    float sum=0.f;
    for (int k=0;k<64;k+=8) {
        float partial=0.f;
        #pragma unroll
        for (int i=0;i<8;++i) partial=__fmaf_rn(f(probabilities[p+k+i]),f(values[v+(k+i)*32]),partial);
        sum=f(h(sum+partial));
    }
    output[index]=h(sum);
}

enum class Product { projection, query_key, window_values, fixed_window_values };

// FP16 WMMA contraction for the original-backbone differential probe.
template<Product product>
__global__ void backbone_half_wmma(const Half* left,const Half* right,const Half* bias,Half* output,
                                   std::size_t rows,int inner,int columns,int bias_rows) {
    using namespace nvcuda::wmma;
    constexpr int side=16, area=side*side, warps=4;
    struct alignas(32) Tile { Half left[area], right[area], result[area]; };
    __shared__ Tile storage[warps];
    const int warp=threadIdx.x/32, lane=threadIdx.x%32;
    const auto tile=std::size_t(blockIdx.x)*warps+warp;
    const int tiles_across=columns/side;
    const auto row=tile/tiles_across*side;
    if(row>=rows) return;
    const int column=tile%tiles_across*side;
    auto& memory=storage[warp];
    fragment<accumulator,side,side,side,Half> value;
    if(bias) {
        for(int i=lane;i<area;i+=32) {
            const auto r=row+i/side;
            memory.result[i]=bias[(bias_rows ? r%bias_rows : r)*columns+column+i%side];
        }
        __syncwarp();
        load_matrix_sync(value,memory.result,side,mem_row_major);
    } else fill_fragment(value,__float2half_rn(0.f));
    for(int start=0;start<inner;start+=side) {
        for(int i=lane;i<area;i+=32) {
            const auto r=row+i/side;
            constexpr bool fixed=product==Product::fixed_window_values;
            memory.left[i]=left[(fixed ? r%64 : r)*inner+start+i%side];
            const auto k=start+i/side;
            const auto col=column+i%side;
            std::size_t source;
            if constexpr(product==Product::query_key)
                source=((row/64)*64+col)*inner+k;
            else if constexpr(fixed || product==Product::window_values)
                source=((row/64)*inner+k)*columns+col;
            else
                source=k*columns+col;
            memory.right[(i%side)*side+i/side]=right[source];
        }
        __syncwarp();
        fragment<matrix_a,side,side,side,Half,row_major> a;
        fragment<matrix_b,side,side,side,Half,col_major> b;
        load_matrix_sync(a,memory.left,side);
        load_matrix_sync(b,memory.right,side);
        mma_sync(value,a,b,value);
        __syncwarp();
    }
    store_matrix_sync(memory.result,value,side,mem_row_major);
    __syncwarp();
    for(int i=lane;i<area;i+=32)
        output[(row+i/side)*columns+column+i%side]=memory.result[i];
}

template<Product product>
void launch_backbone_wmma(const Half* left,const Half* right,const Half* bias,Half* output,
                          std::size_t rows,int inner,int columns,int bias_rows,cudaStream_t stream) {
    const auto tiles=(rows/16)*(columns/16);
    backbone_half_wmma<product><<<(tiles+3)/4,128,0,stream>>>(left,right,bias,output,rows,inner,columns,bias_rows);
}
constexpr int tile_columns=32, tile_rows=8, tile_inner=32;

template<Product product,int rows_per_thread>
__global__ void tiled_product(const Half* left,const Half* right,const Half* bias,Half* output,
                              std::size_t rows,int inner,int columns,int bias_rows) {
    constexpr int block_rows=tile_rows*rows_per_thread;
    __shared__ float a[block_rows][tile_inner];
    __shared__ float b[tile_inner][tile_columns+1];
    const int lane=threadIdx.x%tile_columns, local_row=threadIdx.x/tile_columns;
    const int tile_count=(columns+tile_columns-1)/tile_columns;
    const std::size_t row_base=std::size_t(blockIdx.x/tile_count)*block_rows;
    const std::size_t first_row=row_base+local_row, window=row_base/64;
    const int column_base=blockIdx.x%tile_count*tile_columns, column=column_base+lane;
    float sum[rows_per_thread];
    #pragma unroll
    for(int r=0;r<rows_per_thread;++r) {
        const auto row=first_row+r*tile_rows;
        sum[r]=row<rows && column<columns && bias
            ? f(bias[(bias_rows ? row%bias_rows : row)*columns+column]) : 0.f;
    }
    for(int start=0;start<inner;start+=tile_inner) {
        #pragma unroll
        for(int r=0;r<rows_per_thread;++r) {
            const auto row=first_row+r*tile_rows;
            const auto left_row=product==Product::fixed_window_values ? row%64 : row;
            a[local_row+r*tile_rows][lane]=row<rows && start+lane<inner ? f(left[left_row*inner+start+lane]) : 0.f;
        }
        for(int index=threadIdx.x;index<tile_inner*tile_columns;index+=threads) {
            constexpr bool transposed=product==Product::query_key;
            const int depth=transposed ? index%tile_inner : index/tile_columns;
            const int col=transposed ? index/tile_inner : index%tile_columns;
            const int k=start+depth, out=column_base+col;
            std::size_t offset;
            if constexpr(transposed) offset=(window*64+out)*inner+k;
            else if constexpr(product==Product::projection) offset=std::size_t(k)*columns+out;
            else offset=(window*64+k)*columns+out;
            b[depth][col]=k<inner && out<columns ? f(right[offset]) : 0.f;
        }
        __syncthreads();
        const int count=min(tile_inner,inner-start);
        for(int k=0;k<count;k+=8) {
            float partial[rows_per_thread]{};
            #pragma unroll
            for(int i=0;i<8;++i) {
                const float right_value=b[k+i][lane];
                #pragma unroll
                for(int r=0;r<rows_per_thread;++r)
                    partial[r]=__fmaf_rn(a[local_row+r*tile_rows][k+i],right_value,partial[r]);
            }
            #pragma unroll
            for(int r=0;r<rows_per_thread;++r) sum[r]=f(h(sum[r]+partial[r]));
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<rows_per_thread;++r) {
        const auto row=first_row+r*tile_rows;
        if(row<rows && column<columns) output[row*columns+column]=h(sum[r]);
    }
}

template<Product product>
void launch_tiled_product(const Half* left,const Half* right,const Half* bias,Half* out,
                          std::size_t rows,int inner,int columns,int bias_rows,bool row_reuse,cudaStream_t stream) {
    const auto columns_tiles=(columns+tile_columns-1)/tile_columns;
    if(row_reuse) {
        const auto grid=columns_tiles*((rows+2*tile_rows-1)/(2*tile_rows));
        tiled_product<product,2><<<grid,threads,0,stream>>>(left,right,bias,out,rows,inner,columns,bias_rows);
    } else {
        const auto grid=columns_tiles*((rows+tile_rows-1)/tile_rows);
        tiled_product<product,1><<<grid,threads,0,stream>>>(left,right,bias,out,rows,inner,columns,bias_rows);
    }
}

__device__ float reduce_half(const Half* x,int n,bool square) {
    float r[16];
    for (int i=0;i<16;++i) {
        r[i]=0.f;
        for (int g=0;g<n;g+=32) {
            float a=f(x[g+i]), b=f(x[g+i+16]);
            if (square) {a=f(h(a*a)); b=f(h(b*b));}
            const float pair=f(h(a+b));
            r[i]=g==0 ? pair : f(h(r[i]+pair));
        }
    }
    for (int i=0;i<8;++i) r[i]=f(h(r[i]+r[i+8]));
    const float a=f(h(f(h(r[0]+r[2]))+f(h(r[4]+r[6]))));
    const float b=f(h(f(h(r[1]+r[3]))+f(h(r[5]+r[7]))));
    return f(h(a+b));
}

__global__ void normalize(const Half* x,const Half* gamma,Half* y,std::size_t rows,int c) {
    const auto row=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row>=rows) return;
    const auto* values=x+row*c;
    const float sum=reduce_half(values,c,true);
    const float inverse=sum>0 ? f(h(1.f/sqrtf(sum))) : 0.f;
    for (int i=0;i<c;++i) y[row*c+i]=h(f(values[i])*f(h(inverse*f(gamma[i]))));
}

// One half warp per row. The XOR stages reproduce the scalar tree's
// (r0+r2)+(r4+r6) and (r1+r3)+(r5+r7) after the stride-eight merge.
__device__ float reduce_halfwarp(float part) {
    const unsigned lanes=0xffffu<<(threadIdx.x&16);
    const float p8=f(h(part+__shfl_xor_sync(lanes,part,8,16)));
    const float p2=f(h(p8+__shfl_xor_sync(lanes,p8,2,16)));
    const float p4=f(h(p2+__shfl_xor_sync(lanes,p2,4,16)));
    return f(h(__shfl_sync(lanes,p4,0,16)+__shfl_sync(lanes,p4,1,16)));
}

__device__ float norm_pair(const Half* values,int lane) {
    const float a=f(values[lane]), b=f(values[lane+16]);
    return f(h(f(h(a*a))+f(h(b*b))));
}

template<bool balanced_wide>
__global__ void normalize_parallel(const Half* x,const Half* gamma,Half* y,std::size_t rows,int c) {
    const auto row=(std::size_t(blockIdx.x)*blockDim.x+threadIdx.x)/16;
    if(row>=rows) return;
    const int lane=threadIdx.x%16;
    const auto* values=x+row*c;
    float part;
    if constexpr(balanced_wide) {
        // The 128/160-channel PTX balances the first four groups, then adds the fifth.
        const float first=f(h(norm_pair(values,lane)+norm_pair(values,lane+32)));
        const float second=f(h(norm_pair(values,lane+64)+norm_pair(values,lane+96)));
        part=f(h(first+second));
        if(c==160) part=f(h(part+norm_pair(values,lane+128)));
    } else {
        part=norm_pair(values,lane);
        for(int group=32;group<c;group+=32)
            part=f(h(part+norm_pair(values,lane+group)));
    }
    const float sum=reduce_halfwarp(part);
    const float inverse=sum>0 ? f(h(1.f/sqrtf(sum))) : 0.f;
    for(int i=lane;i<c;i+=16) y[row*c+i]=h(f(values[i])*f(h(inverse*f(gamma[i]))));
}

template<bool original_exponent = false>
__device__ Half probability_value(Half point) {
    const float t=fminf(fmaxf(f(point),-21.640625f),21.640625f);
    const float factor=f(h(__fmaf_rn(f(h(t*t)),-0.0007119178771972656f,1.f)));
    const auto exponent=h(__fmaf_rn(t,factor,-5.05078125f));
    if constexpr(original_exponent) {
        const unsigned bits=__half_as_ushort(exponent);
        const unsigned packed=bits | (bits << 16);
        unsigned result;
        asm("ex2.approx.f16x2 %0, %1;" : "=r"(result) : "r"(packed));
        return __ushort_as_half(static_cast<unsigned short>(result));
    }
    return h(exp2f(f(exponent)));
}

__global__ void probability(Half* x,std::size_t rows) {
    const auto row=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row>=rows) return;
    auto* values=x+row*64;
    for (int i=0;i<64;++i) values[i]=probability_value(values[i]);
    const float reciprocal=f(h(1.f/reduce_half(values,64,false)));
    for (int i=0;i<64;++i) values[i]=h(f(values[i])*reciprocal);
}

template<bool original_probability>
__global__ void probability_parallel(Half* x,std::size_t rows) {
    const auto row=(std::size_t(blockIdx.x)*blockDim.x+threadIdx.x)/16;
    if(row>=rows) return;
    const int lane=threadIdx.x%16;
    auto* values=x+row*64;
    const float a=f(probability_value<original_probability>(values[lane]));
    const float b=f(probability_value<original_probability>(values[lane+16]));
    const float c=f(probability_value<original_probability>(values[lane+32]));
    const float d=f(probability_value<original_probability>(values[lane+48]));
    const float sum=reduce_halfwarp(f(h(f(h(a+b))+f(h(c+d)))));
    float inverse;
    if constexpr(original_probability)
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(sum));
    else inverse=1.f/sum;
    const float reciprocal=f(h(inverse));
    values[lane]=h(a*reciprocal);
    values[lane+16]=h(b*reciprocal);
    values[lane+32]=h(c*reciprocal);
    values[lane+48]=h(d*reciprocal);
}

enum class Operation { add, scale, bias, relu, activate };
__global__ void element(Half* out,const Half* x,const Half* b,std::size_t count,int c,Operation op) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    float value=x ? f(x[i]) : 0.f;
    switch (op) {
    case Operation::add: value+=f(b[i]); break;
    case Operation::scale: value*=f(b[i%c]); break;
    case Operation::bias: value=(x ? value+f(b[i%c]) : f(b[i%c])); break;
    case Operation::relu: value=fmaxf(value,0.f); break;
    case Operation::activate: {
        const float t=fminf(fmaxf(value,-2.f),2.f);
        const float a=f(h(__uint_as_float(0x3ed306eb))), b0=f(h(__uint_as_float(0x3da60dd6)));
        const float slope=f(h(a-b0*fabsf(t)));
        value*=f(h(__fmaf_rn(t,slope,0.5f)));
        break;
    }
    }
    out[i]=h(value);
}

__global__ void activate_fused_half(__half2* values,std::size_t pairs) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=pairs) return;
    const auto a=__float2half2_rn(__uint_as_float(0x3ed306eb));
    const auto b=__float2half2_rn(__uint_as_float(0x3da60dd6));
    const auto value=values[index];
    const auto clamped=__hmin2(__hmax2(value,__float2half2_rn(-2.f)),__float2half2_rn(2.f));
    // ptxas contracts both multiply/add pairs in the original SM86 executable.
    const auto slope=__hfma2(__hneg2(b),__habs2(clamped),a);
    const auto factor=__hfma2(clamped,slope,__float2half2_rn(.5f));
    values[index]=__hmul2_rn(value,factor);
}

__global__ void rearrange2(const Half* input,Half* output,std::size_t count,int c,bool fold) {
    const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count) return;
    const int p=(i/c)%64, y=p/8, x=p%8;
    const int packed=(((y/2)*4+x/2)*4+(y%2)*2+x%2);
    const auto j=(i/(64*c)*64+packed)*c+i%c;
    if (fold) output[j]=input[i];
    else output[i]=input[j];
}

__global__ void scatter(const Half* tiles,Half* image,Stage s,int c,int stride,int scale,
                         int groups,std::size_t count) {
    const auto index=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index>=count) return;
    const int width=s.width/scale, side=8/scale;
    const auto pixel=index/c;
    const auto x=pixel%width+s.x/scale, y=pixel/width+s.y/scale;
    const auto window=y/side*s.grid_x+x/side;
    const auto token=(window*side+y%side)*side+x%side;
    const int channel=index%c;
    const int src_channel=groups>1 ? channel/(c/groups)*(stride/groups)+channel%(c/groups) : channel;
    image[index]=tiles[token*stride+src_channel];
}

unsigned blocks(std::size_t count) {return static_cast<unsigned>((count+threads-1)/threads);}
bool valid(Geometry g) {
    return g.width>=256 && g.height>=256 && g.width<=2048 && g.height<=2048 &&
           g.width%32==0 && g.height%32==0;
}
} // namespace

Geometry network_geometry(int width,int height) {
    if (width<=0 || height<=0 || width>4096 || height>4096)
        throw std::invalid_argument("SR source geometry requires input dimensions in [1,4096]");
    return {std::max(256,(width*2+127)/128*32),std::max(256,(height*2+127)/128*32)};
}

std::size_t workspace_bytes(Geometry g,unsigned phase) {
    if (!valid(g) || phase>=8) throw std::invalid_argument("Invalid SR neural field or phase");
    const auto stages=stage_geometry(g,phase);
    const auto live=persistent(g,stages);
    std::size_t end=live.end;
    for (int i=0;i<stage_count;++i) {
        Arena arena{live.end};
        scratch(arena,stages[i],i);
        end=std::max(end,arena.cursor);
    }
    return end*sizeof(Half);
}

static cudaError_t enqueue_stages(Geometry g,unsigned phase,const Weights& layout,const std::uint16_t* device_weights,
                                  const std::uint16_t* prepared,std::uint16_t* reconstruction,void* workspace,
                                  std::size_t bytes,cudaStream_t stream,BackboneOptions options,
                                  FirstEncoderOutput* first_encoder,BackboneProbeArithmetic arithmetic,
                                  StageObserver observe,void* user) {
    if (!valid(g) || phase>=8 || !device_weights || !prepared || (!reconstruction && !first_encoder) || !workspace ||
        layout.elements.size()!=packed_bytes/2 || bytes<workspace_bytes(g,phase)) return cudaErrorInvalidValue;
    const auto geometry=stage_geometry(g,phase);
    const auto live=persistent(g,geometry);
    auto* base=static_cast<Half*>(workspace);
    const auto* weights=reinterpret_cast<const Half*>(device_weights);
    const Half* input=reinterpret_cast<const Half*>(prepared);
    const int count=first_encoder ? 1 : stage_count;
    for (int i=0;i<count;++i) {
        const auto s=geometry[i];
        const auto& w=layout.stages[i];
        Arena arena{live.end};
        const auto a=scratch(arena,s,i);
        const int c=channels[i], h=heads[i];
        const auto rows=s.windows()*64;
        const bool tensor_core=(arithmetic.tensor_core_stages & (1u << i))!=0;
        const auto launch_mm=[&](const Half* left,const Half* right,const Half* bias,Half* output,
                                 std::size_t count,int k,int n,int bias_rows,bool transposed=false) {
            if(tensor_core) {
                if(transposed)
                    launch_backbone_wmma<Product::query_key>(left,right,bias,output,count,k,n,bias_rows,stream);
                else
                    launch_backbone_wmma<Product::projection>(left,right,bias,output,count,k,n,bias_rows,stream);
                return;
            }
            if(!options.tiled) {
                matrix<<<blocks(count*n),threads,0,stream>>>(left,right,bias,output,count,k,n,bias_rows,transposed);
                return;
            }
            if(transposed)
                launch_tiled_product<Product::query_key>(left,right,bias,output,count,k,n,bias_rows,options.reuse_matrix_rows,stream);
            else
                launch_tiled_product<Product::projection>(left,right,bias,output,count,k,n,bias_rows,options.reuse_matrix_rows,stream);
        };
        const auto pointwise=[&](Half* out,const Half* left,const Half* right,std::size_t size,int channels,Operation op) {
            element<<<blocks(size),threads,0,stream>>>(out,left,right,size,channels,op);
        };
        const auto store=[&](const Half* tiles,Half* image,int channels,int stride,int scale,int groups) {
            const std::size_t quantity=std::size_t(s.width/scale)*(s.height/scale)*channels;
            scatter<<<blocks(quantity),threads,0,stream>>>(tiles,image,s,channels,stride,scale,groups,quantity);
        };
        auto* x=base+a.x;
        auto* norm=base+a.norm;
        auto* y=base+a.attention;
        if (i>=6) {
            const int low_channels=channels[i-1];
            const auto count=rows/4*low_channels;
            gather<<<blocks(count),threads,0,stream>>>(input,base+a.gathered,s,low_channels,2,count);
            launch_mm(base+a.gathered,weights+w.embedding,weights+w.embedding_bias,base+a.projected,rows/4,low_channels,4*c,1);
            rearrange2<<<blocks(rows*c),threads,0,stream>>>(base+a.projected,x,rows*c,c,false);
            gather<<<blocks(rows*c),threads,0,stream>>>(base+live.skip[10-i],norm,s,c,1,rows*c);
            pointwise(x,x,norm,rows*c,c,Operation::add);
        } else if (i==0) {
            gather<<<blocks(rows*16),threads,0,stream>>>(input,base+a.gathered,s,16,1,rows*16);
            launch_mm(base+a.gathered,weights+w.embedding,weights+w.embedding_bias,x,rows,16,c,1);
            pointwise(x,x,nullptr,rows*c,c,Operation::relu);
        } else gather<<<blocks(rows*c),threads,0,stream>>>(input,x,s,c,1,rows*c);
        if(arithmetic.balanced_wide_norm && c>=128)
            normalize_parallel<true><<<blocks(rows*16),threads,0,stream>>>(x,weights+w.norm,norm,rows,c);
        else if(options.parallel_reduce)
            normalize_parallel<false><<<blocks(rows*16),threads,0,stream>>>(x,weights+w.norm,norm,rows,c);
        else
            normalize<<<blocks(rows),threads,0,stream>>>(x,weights+w.norm,norm,rows,c);
        const bool early=i==0 || i==10, fixed=early || i==1;
        pointwise(y,early ? x : nullptr,weights+w.attention_bias,rows*c,c,Operation::bias);
        for (int j=0;j<h;++j) {
            const auto& head=w.heads[j];
            const Half* p=weights+head.position;
            launch_mm(norm,weights+head.value,nullptr,base+a.v,rows,c,32,0);
            if (!fixed) {
                launch_mm(norm,weights+head.query,nullptr,base+a.q,rows,c,32,0);
                launch_mm(norm,weights+head.key,nullptr,base+a.k,rows,c,32,0);
                auto* calculated=base+a.probabilities;
                launch_mm(base+a.q,base+a.k,p,calculated,rows,32,64,64,true);
                if(arithmetic.half_attention_probability)
                    probability_parallel<true><<<blocks(rows*16),threads,0,stream>>>(calculated,rows);
                else if(options.parallel_reduce)
                    probability_parallel<false><<<blocks(rows*16),threads,0,stream>>>(calculated,rows);
                else
                    probability<<<blocks(rows),threads,0,stream>>>(calculated,rows);
                p=calculated;
            }
            if(tensor_core) {
                if(fixed)
                    launch_backbone_wmma<Product::fixed_window_values>(p,base+a.v,nullptr,base+a.values,rows,64,32,0,stream);
                else
                    launch_backbone_wmma<Product::window_values>(p,base+a.v,nullptr,base+a.values,rows,64,32,0,stream);
            }
            else if(!options.tiled) window_values<<<blocks(rows*32),threads,0,stream>>>(p,base+a.v,base+a.values,rows,fixed);
            else {
                if(fixed)
                    launch_tiled_product<Product::fixed_window_values>(p,base+a.v,nullptr,base+a.values,rows,64,32,0,options.reuse_matrix_rows,stream);
                else
                    launch_tiled_product<Product::window_values>(p,base+a.v,nullptr,base+a.values,rows,64,32,0,options.reuse_matrix_rows,stream);
            }
            launch_mm(base+a.values,weights+head.projection,y,y,rows,32,c,0);
        }
        if (!early) pointwise(y,y,x,rows*c,c,Operation::add);
        pointwise(base+a.ff_input,y,weights+w.ff_scale,rows*c,c,Operation::scale);
        launch_mm(base+a.ff_input,weights+w.ff_up,weights+w.ff_bias,base+a.ff,rows,c,4*c,1);
        if(arithmetic.fused_half_activation)
            activate_fused_half<<<blocks(rows*2*c),threads,0,stream>>>(reinterpret_cast<__half2*>(base+a.ff),rows*2*c);
        else
            pointwise(base+a.ff,base+a.ff,nullptr,rows*4*c,4*c,Operation::activate);
        pointwise(y,y,weights+w.ff_output_bias,rows*c,c,Operation::bias);
        launch_mm(base+a.ff,weights+w.ff_down,y,y,rows,4*c,c,0);
        auto* next=base+live.image[i%2];
        if (i<5) {
            store(y,base+live.skip[i],c,c,1,1);
            rearrange2<<<blocks(rows*c),threads,0,stream>>>(y,base+a.folded,rows*c,c,true);
            const int merged=merge_channels(i);
            launch_mm(base+a.folded,weights+w.merge,weights+w.merge_bias,base+a.projected,rows/4,4*c,merged,1);
            store(base+a.projected,next,channels[i+1],merged,2,h);
        } else if (i==10) {
            launch_mm(y,weights+w.output,weights+w.output_bias,base+a.projected,rows,32,48,1);
            store(base+a.projected,reinterpret_cast<Half*>(reconstruction),40,48,1,1);
        } else store(y,next,c,c,1,1);
        input=next;
        if (const auto status=cudaGetLastError();status!=cudaSuccess) return status;
        if(observe) {
            auto* output=i==10 ? reconstruction : reinterpret_cast<std::uint16_t*>(next);
            auto* skip=i<5 ? reinterpret_cast<std::uint16_t*>(base+live.skip[i]) : nullptr;
            if(const auto status=observe(user,i,output,skip,stream);status!=cudaSuccess) return status;
        }
    }
    if (first_encoder) {
        first_encoder->skip=reinterpret_cast<std::uint16_t*>(base+live.skip[0]);
        first_encoder->downsampled=reinterpret_cast<std::uint16_t*>(base+live.image[0]);
    }
    return cudaSuccess;
}

cudaError_t enqueue(Geometry g,unsigned phase,const Weights& layout,const std::uint16_t* device_weights,
                    const std::uint16_t* prepared,std::uint16_t* reconstruction,void* workspace,
                    std::size_t bytes,cudaStream_t stream,BackboneOptions options) {
    constexpr BackboneProbeArithmetic original_fp16{(1u << stage_count)-1u,true,true,true};
    const auto arithmetic=options.original_fp16 ? original_fp16 : BackboneProbeArithmetic{};
    return enqueue_stages(g,phase,layout,device_weights,prepared,reconstruction,workspace,bytes,stream,options,
                          nullptr,arithmetic,nullptr,nullptr);
}

cudaError_t enqueue_first_encoder(Geometry g,unsigned phase,const Weights& layout,const std::uint16_t* device_weights,
                                  const std::uint16_t* prepared,void* workspace,std::size_t bytes,
                                  cudaStream_t stream,FirstEncoderOutput& output,BackboneOptions options,bool tensor_core) {
    output={};
    return enqueue_stages(g,phase,layout,device_weights,prepared,nullptr,workspace,bytes,stream,options,
                          &output,{tensor_core ? 1u : 0u},nullptr,nullptr);
}

cudaError_t enqueue_backbone_probe(Geometry g,unsigned phase,const Weights& layout,const std::uint16_t* device_weights,
                                   const std::uint16_t* prepared,std::uint16_t* reconstruction,void* workspace,
                                   std::size_t bytes,cudaStream_t stream,BackboneOptions options,BackboneProbeArithmetic arithmetic,
                                   StageObserver observe,void* user) {
    return enqueue_stages(g,phase,layout,device_weights,prepared,reconstruction,workspace,bytes,stream,options,
                          nullptr,arithmetic,observe,user);
}

} // namespace rw::sr::source_backbone
