// Native CUDA arithmetic port of dlssnr/layers/decoder.py at
// mauri870/dlssnr-pytorch 3beef7935a8ef976b47ab78b906e75600c52d2de.
// Copyright (c) 2026 Mauri de Souza Meneguzzo. MIT License.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "native_nr.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <climits>
#include <stdexcept>

namespace rw::native_nr {
namespace {
__device__ float half_value(float value) { return __half2float(__float2half_rn(value)); }
__device__ float quant8(float value) {
    __nv_fp8_e4m3 encoded;
    encoded.__x=__nv_cvt_float_to_fp8(value,__NV_SATFINITE,__NV_E4M3);
    return static_cast<float>(encoded);
}
__global__ void decoder_blend(const float* projected,const float* residual,
    const float* gain,float* output,int low_h,int low_w,int high_h,int high_w,size_t count) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const size_t high_pixels=size_t(high_h)*high_w;
    const size_t channel=index/high_pixels,pixel=index%high_pixels;
    const int row=static_cast<int>(pixel/high_w),column=static_cast<int>(pixel%high_w);
    const float up=projected[(channel*low_h+row/2)*low_w+column/2];
    const float gated=half_value(__fmul_rn(half_value(residual[index]),gain[channel]));
    output[index]=quant8(half_value(__fadd_rn(up,gated)));
}
}
LayerOutput run_decoder(Context& context,Weights& weights,const LayerSpec& spec,
    const Tensor& input,const LayerOutput* skip) {
    if(spec.kind!="DecInputUpsample")throw std::invalid_argument("unknown decoder kind: "+spec.kind);
    if (!context.owns(input) || !skip || !context.owns(skip->main))
        throw std::invalid_argument("decoder tensor belongs to another native context");
    if(input.empty()||input.shape.size()!=4||input.shape[0]!=1||input.shape[1]!=1024||
       !skip||skip->main.empty()||skip->main.shape.size()!=4||skip->main.shape[0]!=1||skip->main.shape[1]!=512)
        throw std::invalid_argument("decoder requires [1,1024,H,W] and [1,512,H,W] residual");
    input.numel(); skip->main.numel();
    const auto& residual=skip->main;
    if(input.shape[2]<=0||input.shape[3]<=0||residual.shape[2]<=0||residual.shape[3]<=0||
       input.shape[2]>INT_MAX||input.shape[3]>INT_MAX||residual.shape[2]>INT_MAX||residual.shape[3]>INT_MAX||
       residual.shape[2]>2*input.shape[2]||residual.shape[3]>2*input.shape[3])
        throw std::invalid_argument("decoder replication/crop geometry mismatch");
    const Parameter matrix=weights.e4m3(context,spec.block,spec.layer,"weight",{512,1024});
    const Parameter gain=weights.f16(context,spec.block,spec.layer,"skip_weight",{512});
    context.validate_parameter(gain);
    const Tensor projected=narrow(context,linear_nchw(context,input,matrix));
    Tensor output=context.empty(residual.shape);
    const size_t grid=(output.numel()-1)/256+1;
    if(grid>UINT_MAX)throw std::invalid_argument("decoder CUDA launch too large");
    decoder_blend<<<static_cast<unsigned>(grid),256,0,context.stream()>>>(projected.data(),residual.data(),
        gain.data(),output.data(),static_cast<int>(input.shape[2]),static_cast<int>(input.shape[3]),
        static_cast<int>(residual.shape[2]),static_cast<int>(residual.shape[3]),output.numel());
    cuda_check(cudaGetLastError(),"decoder nearest replication, gated half sum and FP8 publication");
    return {output,{}};
}
} // namespace rw::native_nr
