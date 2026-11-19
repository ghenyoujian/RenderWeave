// SPDX-License-Identifier: MIT
// One block per window/head exposes the source expert axis to the GPU.
// Later contract/residual/QKV GEMMs remain on the existing cuBLAS path.
template<int K>
__device__ __forceinline__ void expert_front_weight_tile(const float* source,unsigned char* target,int row_begin) {
    for(int index=threadIdx.x;index<32*K;index+=128) {
        const int row=index/K,k=index%K;
        target[(k/32)*1024+hopper_e4_offset(row,k%32)]=hopper_encode(source[(row_begin+row)*K+k]);
    }
}
template<int Channels>
__global__ __launch_bounds__(128,2) void fused_swin_expert_front_hopper(
    const float* input,const float* expand,const float* middle,float* output,
    int height,int width,int grid_columns,int front_rows,int front_columns) {
    constexpr int heads=Channels/32;
    const size_t window=blockIdx.x/heads;const int head=blockIdx.x%heads,thread=threadIdx.x;
    const int row_base=(thread/32)*16+(thread%32)/4,column_pair=(thread%4)*2;
    __shared__ __align__(16) unsigned char state[heads][2048],hidden[4][2048];
    extern __shared__ __align__(16) unsigned char weights[];
    for(int index=thread;index<64*Channels;index+=128) {
        const int token=index%64,channel=index/64;
        const int row=int(window/grid_columns)*8+(token/32)*4+(token/4)%4-front_rows;
        const int column=int(window%grid_columns)*8+((token/16)%2)*4+token%4-front_columns;
        const float value=row>=0 && row<height && column>=0 && column<width
            ? input[(size_t(channel)*height+row)*width+column] : 0.0f;
        state[channel/32][hopper_e4_offset(token,channel%32)]=hopper_encode(value);
    }
    for(int column=0;column<128;column+=32) {
        expert_front_weight_tile<Channels>(expand,weights,head*128+column);
        hopper_publish_shared();
        float accumulator[16]={};
        #pragma unroll
        for(int k=0;k<heads;++k)hopper_e4_mma(accumulator,state[k],weights+k*1024);
        #pragma unroll
        for(int tile=0;tile<4;++tile)
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int component=0;component<2;++component) {
                    const int row=row_base+half*8,channel=column+tile*8+column_pair+component;
                    const float value=accumulator[tile*4+half*2+component],t=clamp_preserving_nan(value,-4.0f,4.0f);
                    const float inner=__fmaf_rn(fabsf(t),-0.055908203125f,0.447265625f);
                    hidden[channel/32][hopper_e4_offset(row,channel%32)]=
                        hopper_encode(__fmul_rn(value,__fmaf_rn(t,inner,0.89453125f)));
                }
        __syncthreads();
    }
    expert_front_weight_tile<128>(middle+head*4096,weights,0);
    hopper_publish_shared();
    float accumulator[16]={};
    #pragma unroll
    for(int k=0;k<4;++k)hopper_e4_mma(accumulator,hidden[k],weights+k*1024);
    #pragma unroll
    for(int tile=0;tile<4;++tile)
        #pragma unroll
        for(int half=0;half<2;++half)
            #pragma unroll
            for(int component=0;component<2;++component) {
                const int token=row_base+half*8,channel=tile*8+column_pair+component;
                output[(window*64+token)*Channels+head*32+channel]=fp8_round(half_round(accumulator[tile*4+half*2+component]));
            }
}
