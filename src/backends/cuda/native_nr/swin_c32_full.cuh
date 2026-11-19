// SPDX-License-Identifier: MIT
// Full C32 window block. Every existing publication stays in place; only
// intermediate storage moves from device-global tensors into shared memory.
__global__ __launch_bounds__(128,2) void fused_swin_c32_full_hopper(
    const float* input,const float* expand,const float* contract,const float* qkv,
    const float* projection,const float* mlp_scale,const float* attention_scale,
    const float* head_scale,const float* exponent_bias,float* raw_output,float* wide_output,
    int height,int width,int grid_columns,int front_rows,int front_columns,bool batched_mma) {
    const size_t window=blockIdx.x;
    const int thread=threadIdx.x,row_base=(thread/32)*16+(thread%32)/4,column_pair=(thread%4)*2;
    union Math {
        struct {unsigned char state[2048],hidden[4][2048],weights[4096];} mlp;
        struct {unsigned char queries[2048],keys[2048],values[2][1024],probability[2][2048];} attention;
        struct {unsigned char a[2][2048],b[2][1024];} projection;
    };
    union Work {
        float qkv[64][97];
        struct {float scores[64][65],reciprocal[64];} attention;
    };
    __shared__ __align__(16) Math math;
    __shared__ Work work;
    __shared__ float residual[64][33];
    for(int index=thread;index<2048;index+=128) {
        const int token=index%64,channel=index/64;
        const int row=int(window/grid_columns)*8+(token/32)*4+(token/4)%4-front_rows;
        const int column=int(window%grid_columns)*8+((token/16)%2)*4+token%4-front_columns;
        const float value=row>=0 && row<height && column>=0 && column<width
            ? input[(size_t(channel)*height+row)*width+column] : 0.0f;
        residual[token][channel]=value;
        math.mlp.state[hopper_e4_offset(token,channel)]=hopper_encode(value);
    }
    for(int index=thread;index<4096;index+=128)
        math.mlp.weights[hopper_e4_offset(index/32,index%32)]=hopper_encode(expand[index]);
    hopper_publish_shared();
    for(int column=0;column<128;column+=32) {
        float accumulator[16]={};
        hopper_e4_mma(accumulator,math.mlp.state,math.mlp.weights+(column/8)*256);
        #pragma unroll
        for(int tile=0;tile<4;++tile)
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int component=0;component<2;++component) {
                    const int row=row_base+half*8,channel=column+tile*8+column_pair+component;
                    const float value=accumulator[tile*4+half*2+component],t=clamp_preserving_nan(value,-4.0f,4.0f);
                    const float inner=__fmaf_rn(fabsf(t),-0.055908203125f,0.447265625f);
                    math.mlp.hidden[channel/32][hopper_e4_offset(row,channel%32)]=
                        hopper_encode(__fmul_rn(value,__fmaf_rn(t,inner,0.89453125f)));
                }
    }
    __syncthreads();
    for(int index=thread;index<4096;index+=128) {
        const int column=index/128,k=index%128;
        math.mlp.weights[(k/32)*1024+hopper_e4_offset(column,k%32)]=hopper_encode(contract[index]);
    }
    hopper_publish_shared();
    float contracted[16]={};
    hopper_e4_mma_tiles<4,2048,1024>(contracted,math.mlp.hidden[0],math.mlp.weights,batched_mma);
    #pragma unroll
    for(int tile=0;tile<4;++tile)
        #pragma unroll
        for(int half=0;half<2;++half)
            #pragma unroll
            for(int component=0;component<2;++component) {
                const int row=row_base+half*8,channel=tile*8+column_pair+component;
                const float value=__fadd_rn(__fmul_rn(residual[row][channel],mlp_scale[channel]),contracted[tile*4+half*2+component]);
                residual[row][channel]=value;
                math.mlp.state[hopper_e4_offset(row,channel)]=hopper_encode(half_round(value));
            }
    __syncthreads();
    for(int index=thread;index<3072;index+=128)
        math.mlp.weights[hopper_e4_offset(index/32,index%32)]=hopper_encode(qkv[index]);
    hopper_publish_shared();
    for(int column=0;column<96;column+=32) {
        float accumulator[16]={};
        hopper_e4_mma(accumulator,math.mlp.state,math.mlp.weights+(column/8)*256);
        #pragma unroll
        for(int tile=0;tile<4;++tile)
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int component=0;component<2;++component)
                    work.qkv[row_base+half*8][column+tile*8+column_pair+component]=accumulator[tile*4+half*2+component];
    }
    __syncthreads();
    if(thread<64) {
        const float* raw=work.qkv[thread];
        const float qs=__fmul_rn(norm_scale(raw),head_scale[0]),ks=norm_scale(raw+32);
        #pragma unroll
        for(int dimension=0;dimension<32;++dimension) {
            math.attention.queries[hopper_e4_offset(thread,dimension)]=hopper_encode(__fmul_rn(raw[dimension],qs));
            math.attention.keys[hopper_e4_offset(thread,dimension)]=hopper_encode(__fmul_rn(raw[32+dimension],ks));
            math.attention.values[thread/32][hopper_e4_offset(dimension,thread%32)]=hopper_encode(half_round(raw[64+dimension]));
        }
    }
    hopper_publish_shared();
    #pragma unroll
    for(int column=0;column<64;column+=32) {
        float accumulator[16]={};
        hopper_e4_mma(accumulator,math.attention.queries,math.attention.keys+(column/8)*256);
        #pragma unroll
        for(int tile=0;tile<4;++tile)
            #pragma unroll
            for(int half=0;half<2;++half)
                #pragma unroll
                for(int component=0;component<2;++component) {
                    const int query=row_base+half*8,key=column+tile*8+column_pair+component;
                    const float ramp=clamp_preserving_nan(half_round(__fmaf_rn(accumulator[tile*4+half*2+component],
                        kExpSlope,exponent_bias[query*64+key])),kExpLow,kExpHigh);
                    const unsigned short bits=__half_as_ushort(__float2half_rn(ramp));
                    work.attention.scores[query][key]=__half2float(__ushort_as_half(static_cast<unsigned short>((bits&0x3ffu)<<5u)));
                }
    }
    __syncthreads();
    if(thread<64)work.attention.reciprocal[thread]=attention_probability_reciprocal(work.attention.scores[thread]);
    __syncthreads();
    for(int index=thread;index<4096;index+=128) {
        const int row=index/64,key=index%64;
        math.attention.probability[key/32][hopper_e4_offset(row,key%32)]=hopper_encode(half_round(
            __fmul_rn(work.attention.scores[row][key],work.attention.reciprocal[row])));
    }
    hopper_publish_shared();
    float attention[16]={};
    hopper_e4_mma_tiles<2,2048,1024>(attention,math.attention.probability[0],math.attention.values[0],batched_mma);
    __syncthreads();
    // The selected FAST_16F projection already narrows its FP32 input to half.
    // Preserve that boundary rather than adding an E4 quantization here.
    #pragma unroll
    for(int tile=0;tile<4;++tile)
        #pragma unroll
        for(int half=0;half<2;++half)
            #pragma unroll
            for(int component=0;component<2;++component) {
                const int row=row_base+half*8,dimension=tile*8+column_pair+component;
                *reinterpret_cast<__half*>(math.projection.a[dimension/16]+hopper_e4_offset(row,2*(dimension%16)))=
                    __float2half_rn(attention[tile*4+half*2+component]);
            }
    for(int index=thread;index<1024;index+=128) {
        const int row=index/32,k=index%32;
        *reinterpret_cast<__half*>(math.projection.b[k/16]+hopper_e4_offset(row,2*(k%16)))=__float2half_rn(projection[index]);
    }
    hopper_publish_shared();
    float projected[16]={};
    #pragma unroll
    for(int k=0;k<2;++k)hopper_f16_mma(projected,math.projection.a[k],math.projection.b[k]);
    #pragma unroll
    for(int tile=0;tile<4;++tile)
        #pragma unroll
        for(int half=0;half<2;++half)
            #pragma unroll
            for(int component=0;component<2;++component) {
                const int token=row_base+half*8,channel=tile*8+column_pair+component;
                const int row=int(window/grid_columns)*8+(token/32)*4+(token/4)%4-front_rows;
                const int column=int(window%grid_columns)*8+((token/16)%2)*4+token%4-front_columns;
                if(row>=0 && row<height && column>=0 && column<width) {
                    const float value=__fadd_rn(__fmul_rn(residual[token][channel],attention_scale[channel]),
                        projected[tile*4+half*2+component]);
                    const size_t destination=(size_t(channel)*height+row)*width+column;
                    raw_output[destination]=value;wide_output[destination]=half_round(value);
                }
            }
}
