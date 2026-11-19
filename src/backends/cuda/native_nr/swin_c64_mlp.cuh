// SPDX-License-Identifier: MIT
// C64 has two 128->32 expert middle projections. Keep both intermediate
// half/E4 publications, then publish the quantized residual used by attention.
__global__ __launch_bounds__(128,2) void fused_swin_c64_mlp_qkv_hopper(
    const float* input,const float* expand,const float* middle,const float* contract,
    const float* qkv,const float* mlp_scale,float* quantized_output,float* qkv_output,
    int height,int width,int grid_columns,int front_rows,int front_columns,bool batched_mma) {
    const size_t window=blockIdx.x;
    const int thread=threadIdx.x,row_base=(thread/32)*16+(thread%32)/4,column_pair=(thread%4)*2;
    __shared__ __align__(16) unsigned char state[2][2048],hidden[8][2048],weights[16384];
    for(int index=thread;index<4096;index+=128) {
        const int token=index%64,channel=index/64;
        const int row=int(window/grid_columns)*8+(token/32)*4+(token/4)%4-front_rows;
        const int column=int(window%grid_columns)*8+((token/16)%2)*4+token%4-front_columns;
        const float value=row>=0 && row<height && column>=0 && column<width
            ? input[(size_t(channel)*height+row)*width+column] : 0.0f;
        state[channel/32][hopper_e4_offset(token,channel%32)]=hopper_encode(value);
    }
    for(int index=thread;index<16384;index+=128) {
        const int row=index/64,k=index%64;
        weights[(k/32)*8192+hopper_e4_offset(row,k%32)]=hopper_encode(expand[index]);
    }
    hopper_publish_shared();
    for(int column=0;column<256;column+=32) {
        float accumulator[16]={};
        hopper_e4_mma_tiles<2,2048,8192>(accumulator,state[0],weights+(column/8)*256,batched_mma);
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
    }
    __syncthreads();
    for(int index=thread;index<8192;index+=128) {
        const int row=index/128,k=index%128,head=row/32;
        weights[head*4096+(k/32)*1024+hopper_e4_offset(row%32,k%32)]=hopper_encode(middle[index]);
    }
    hopper_publish_shared();
    #pragma unroll
    for(int head=0;head<2;++head) {
        float accumulator[16]={};
        hopper_e4_mma_tiles<4,2048,1024>(accumulator,hidden[head*4],weights+head*4096,batched_mma);
        #pragma unroll
        for(int tile=0;tile<4;++tile)
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int component=0;component<2;++component) {
                    const int row=row_base+half*8,channel=tile*8+column_pair+component;
                    state[head][hopper_e4_offset(row,channel)]=hopper_encode(half_round(accumulator[tile*4+half*2+component]));
                }
    }
    __syncthreads();
    for(int index=thread;index<4096;index+=128) {
        const int row=index/64,k=index%64;
        weights[(k/32)*2048+hopper_e4_offset(row,k%32)]=hopper_encode(contract[index]);
    }
    hopper_publish_shared();
    #pragma unroll
    for(int column=0;column<64;column+=32) {
        float accumulator[16]={};
        hopper_e4_mma_tiles<2,2048,2048>(accumulator,state[0],weights+(column/8)*256,batched_mma);
        #pragma unroll
        for(int tile=0;tile<4;++tile)
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int component=0;component<2;++component) {
                    const int token=row_base+half*8,channel=column+tile*8+column_pair+component;
                    const int row=int(window/grid_columns)*8+(token/32)*4+(token/4)%4-front_rows;
                    const int x=int(window%grid_columns)*8+((token/16)%2)*4+token%4-front_columns;
                    const float residual=row>=0 && row<height && x>=0 && x<width
                        ? input[(size_t(channel)*height+row)*width+x] : 0.0f;
                    const float value=__fadd_rn(__fmul_rn(residual,mlp_scale[channel]),accumulator[tile*4+half*2+component]);
                    __nv_fp8_e4m3 published;published.__x=hopper_encode(half_round(value));
                    hidden[channel/32][hopper_e4_offset(token,channel%32)]=published.__x;
                    quantized_output[(window*64+token)*64+channel]=static_cast<float>(published);
                }
    }
    __syncthreads();
    for(int index=thread;index<12288;index+=128) {
        const int row=index/64,k=index%64;
        weights[(k/32)*6144+hopper_e4_offset(row,k%32)]=hopper_encode(qkv[index]);
    }
    hopper_publish_shared();
    for(int column=0;column<192;column+=32) {
        float accumulator[16]={};
        hopper_e4_mma_tiles<2,2048,6144>(accumulator,hidden[0],weights+(column/8)*256,batched_mma);
        #pragma unroll
        for(int tile=0;tile<4;++tile)
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int component=0;component<2;++component)
                    qkv_output[(window*64+row_base+half*8)*192+column+tile*8+column_pair+component]=accumulator[tile*4+half*2+component];
    }
}
