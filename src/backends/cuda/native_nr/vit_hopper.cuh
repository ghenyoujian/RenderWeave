// SPDX-License-Identifier: MIT
#include "hopper_mma.cuh"

// Sixty-four queries share one K/V tile. The complete original token axis,
// 64-key denominator publication and final padding correction are retained.
template<int MmaColumns=8>
__global__ __launch_bounds__(128, 2) void global_attention_hopper(
    const float* qkv, float* output, int channels, int tokens, int padded_tokens) {
    const int thread=threadIdx.x, head=blockIdx.y, first_query=blockIdx.x*64;
    const int row_base=(thread/32)*16+(thread%32)/4, column_pair=(thread%4)*2;
    const size_t plane=size_t(channels)*tokens;
    __shared__ __align__(16) unsigned char queries[2048],keys[2048],values[2][1024];
    __shared__ __align__(16) unsigned char probability[2][2048];
    __shared__ float scores[64][65],denominator[64];
    float numerator[32/MmaColumns][MmaColumns/2]={};
    for(int index=thread;index<2048;index+=128) {
        const int row=index/32,dimension=index%32,query=first_query+row;
        const size_t source=size_t(head*32+dimension)*tokens+query;
        queries[hopper_e4_offset(row,dimension)]=hopper_encode(query<tokens ? qkv[source] : 0.0f);
    }
    for(int group=0;group<padded_tokens;group+=64) {
        for(int index=thread;index<2048;index+=128) {
            const int member=index/32,dimension=index%32,key=group+member;
            const size_t source=size_t(head*32+dimension)*tokens+key;
            keys[hopper_e4_offset(member,dimension)]=hopper_encode(key<tokens ? qkv[plane+source] : 0.0f);
            values[member/32][hopper_e4_offset(dimension,member%32)]=hopper_encode(key<tokens ? qkv[2*plane+source] : 0.0f);
        }
        hopper_publish_shared();
        #pragma unroll
        for(int column=0;column<64;column+=MmaColumns) {
            float accumulator[MmaColumns/2]={};
            hopper_e4_mma(accumulator,queries,keys+(column/8)*256);
            #pragma unroll
            for(int tile=0;tile<MmaColumns/8;++tile)
                #pragma unroll
                for(int half=0;half<2;++half)
                #pragma unroll
                for(int component=0;component<2;++component)
                    scores[row_base+half*8][column+tile*8+column_pair+component]=
                        approximate_exponential(accumulator[tile*4+half*2+component]);
        }
        __syncthreads();
        if(thread<64) {
            const float sum=group_denominator(scores[thread]);
            denominator[thread]=group==0 ? sum : half_round(__fadd_rn(denominator[thread],sum));
        }
        for(int index=thread;index<4096;index+=128) {
            const int row=index/64,key=index%64;
            probability[key/32][hopper_e4_offset(row,key%32)]=hopper_encode(scores[row][key]);
        }
        hopper_publish_shared();
        #pragma unroll
        for(int column=0;column<32/MmaColumns;++column)
            #pragma unroll
            for(int k=0;k<2;++k)
                hopper_e4_mma(numerator[column],probability[k],values[k]+column*(MmaColumns/8)*256);
        __syncthreads();
    }
    #pragma unroll
    for(int half=0;half<2;++half) {
        const int row=row_base+half*8,query=first_query+row;
        const float padding=half_round((padded_tokens-tokens)*0.083984375f);
        const float sum=half_round(__fadd_rn(denominator[row],-padding));
        const float bounded=isnan(sum) ? sum : fmaxf(sum,6.198883056640625e-05f);
        const float inverse=half_round(__fdiv_rn(1.0f,bounded));
        #pragma unroll
        for(int column=0;column<32/MmaColumns;++column)
            #pragma unroll
            for(int tile=0;tile<MmaColumns/8;++tile)
                #pragma unroll
                for(int component=0;component<2;++component)
                if(query<tokens) {
                    const int dimension=column*MmaColumns+tile*8+column_pair+component;
                    output[size_t(head*32+dimension)*tokens+query]=fp8_round(half_round(
                        __fmul_rn(half_round(numerator[column][tile*4+half*2+component]),inverse)));
                }
    }
}
