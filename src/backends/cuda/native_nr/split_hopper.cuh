// SPDX-License-Identifier: MIT
#include "hopper_mma.cuh"

__device__ float split_probability_reciprocal(const float* probability) {
    float halves[2];
    #pragma unroll
    for(int half=0;half<2;++half) {
        float column[8],pair[4];
        #pragma unroll
        for(int c=0;c<8;++c) {
            float sum=__fadd_rn(probability[half*8+c],probability[16+half*8+c]);
            sum=__fadd_rn(sum,probability[32+half*8+c]);
            column[c]=__fadd_rn(sum,probability[48+half*8+c]);
        }
        #pragma unroll
        for(int p=0;p<4;++p)pair[p]=__fadd_rn(column[2*p],column[2*p+1]);
        halves[half]=__fadd_rn(__fadd_rn(pair[0],pair[1]),__fadd_rn(pair[2],pair[3]));
    }
    const float total=__fadd_rn(halves[0],halves[1]);
    return __fdiv_rn(1.0f,isnan(total) ? total : fmaxf(total,6.103515625e-05f));
}

// Split has its own half normalization, half exponential affine and FP32
// denominator tree. Only its QK/PV multiplies use the shared Hopper primitive.
__global__ __launch_bounds__(128,2) void split_attention_hopper(
    const float* raw_input,const float* scale,const float* bias,float* output) {
    const size_t window=blockIdx.x/heads;
    const int head=blockIdx.x%heads,thread=threadIdx.x;
    const int row_base=(thread/32)*16+(thread%32)/4,column_pair=(thread%4)*2;
    __shared__ __align__(16) unsigned char queries[2048],keys[2048],values[2][1024];
    __shared__ __align__(16) unsigned char probability[2][2048];
    union Work {
        float raw[64][97];
        struct {float scores[64][65],reciprocal[64];} attention;
    };
    __shared__ Work work;
    for(int index=thread;index<64*96;index+=128) {
        const int row=index/96,dimension=index%96;
        work.raw[row][dimension]=raw_input[(window*64ull+row)*(3*channels)+head*96+dimension];
    }
    __syncthreads();
    if(thread<64) {
        const float* raw=work.raw[thread];
        const __half qi=inverse_norm32(raw),ki=inverse_norm32(raw+32),temperature=__float2half_rn(scale[head]);
        #pragma unroll
        for(int dimension=0;dimension<32;++dimension) {
            const __half q=__hmul(__hmul(__float2half_rn(raw[dimension]),qi),temperature);
            const __half k=__hmul(__float2half_rn(raw[32+dimension]),ki);
            queries[hopper_e4_offset(thread,dimension)]=hopper_encode(__half2float(q));
            keys[hopper_e4_offset(thread,dimension)]=hopper_encode(__half2float(k));
            values[thread/32][hopper_e4_offset(dimension,thread%32)]=hopper_encode(half_value(raw[64+dimension]));
        }
    }
    hopper_publish_shared();
    #pragma unroll
    for(int column=0;column<64;column+=32) {
        float accumulator[16]={};
        hopper_e4_mma(accumulator,queries,keys+(column/8)*256);
        #pragma unroll
        for(int tile=0;tile<4;++tile)
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int component=0;component<2;++component) {
                    const int query=row_base+half*8,key=column+tile*8+column_pair+component;
                    work.attention.scores[query][key]=exponential(__fadd_rn(
                        accumulator[tile*4+half*2+component],bias[(head*64+query)*64+key]));
                }
    }
    __syncthreads();
    if(thread<64)work.attention.reciprocal[thread]=split_probability_reciprocal(work.attention.scores[thread]);
    __syncthreads();
    for(int index=thread;index<4096;index+=128) {
        const int row=index/64,key=index%64;
        probability[key/32][hopper_e4_offset(row,key%32)]=hopper_encode(half_value(
            __fmul_rn(work.attention.scores[row][key],work.attention.reciprocal[row])));
    }
    hopper_publish_shared();
    float accumulator[16]={};
    #pragma unroll
    for(int k=0;k<2;++k)hopper_e4_mma(accumulator,probability[k],values[k]);
    #pragma unroll
    for(int tile=0;tile<4;++tile)
        #pragma unroll
        for(int half=0;half<2;++half)
            #pragma unroll
            for(int component=0;component<2;++component)
                output[(window*64ull+row_base+half*8)*channels+head*32+tile*8+column_pair+component]=
                    quant8(half_value(accumulator[tile*4+half*2+component]));
}
