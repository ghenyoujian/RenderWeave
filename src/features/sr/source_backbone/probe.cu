// SPDX-License-Identifier: Apache-2.0
#include "shape.hpp"
#include "prepare.hpp"
#ifdef RW_SR_K_RECONSTRUCTION
#include "reconstruction.hpp"
#endif
#include <openssl/sha.h>
#include <cuda_fp16.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
namespace sr=rw::sr::source_backbone;
using Json=nlohmann::json;

void check(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

std::string digest(const void* bytes,std::size_t size) {
    unsigned char hash[32];
    SHA256(static_cast<const unsigned char*>(bytes),size,hash);
    std::string output;
    constexpr char hex[]="0123456789abcdef";
    for (const auto byte:hash) {output+=hex[byte>>4]; output+=hex[byte&15];}
    return output;
}

std::vector<std::byte> contents(const char* path,std::size_t expected) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file || file.tellg()!=static_cast<std::streamoff>(expected))
        throw std::invalid_argument("SR parameter file size changed");
    std::vector<std::byte> data(expected);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()),data.size()))
        throw std::runtime_error("SR parameter read failed");
    return data;
}

std::size_t verify(const sr::Weights& weights,const std::vector<std::byte>& decoded) {
    if (digest(decoded.data(),decoded.size())!="48146dee81bc74a1416e509f1a94ad606c857a1e8bb40073b28f4d7a7ff4f120")
        throw std::invalid_argument("SR decoded reference SHA-256 changed");
    const auto* data=reinterpret_cast<const unsigned char*>(decoded.data());
    std::size_t header_bytes=0;
    for (unsigned i=0;i<8;++i) header_bytes|=std::size_t(data[i])<<(8*i);
    const auto names=Json::parse(data+8,data+8+header_bytes);
    const unsigned char* payload=data+8+header_bytes;
    std::size_t count=0;
    for (int i=0;i<sr::stage_count;++i) {
        const auto& w=weights.stages[i];
        const int c=sr::channels[i],heads=sr::heads[i];
        const std::string prefix=std::to_string(i+1)+".";
        const auto tensor=[&](const std::string& name,std::size_t offset,std::size_t elements) {
            const auto& entry=names.at(prefix+name);
            const auto span=entry.at("data_offsets").get<std::array<std::size_t,2>>();
            if (entry.at("dtype")!="F16" || span[1]-span[0]!=elements*2 ||
                digest(payload+span[0],elements*2)!=digest(weights.elements.data()+offset,elements*2))
                throw std::runtime_error("Original SR tensor layout differs at "+prefix+name);
            ++count;
        };
        if (!i || i>=6) {
            const int in=i ? sr::channels[i-1] : 16, out=i ? 4*c : c;
            tensor("embedding",w.embedding,in*out); tensor("embeddingBias",w.embedding_bias,out);
        }
        tensor("norm",w.norm,c); tensor("attentionBias",w.attention_bias,c);
        for (int j=0;j<heads;++j) {
            const std::string suffix=std::to_string(j);
            const auto& h=w.heads[j];
            tensor("q"+suffix,h.query,c*32); tensor("k"+suffix,h.key,c*32);
            tensor("v"+suffix,h.value,c*32); tensor("position"+suffix,h.position,64*64);
            tensor("projection"+suffix,h.projection,32*c);
        }
        tensor("ffScale",w.ff_scale,c); tensor("ffOutputBias",w.ff_output_bias,c);
        tensor("ffUp",w.ff_up,4*c*c); tensor("ffBias",w.ff_bias,4*c); tensor("ffDown",w.ff_down,4*c*c);
        if (i<5) {
            const int merged=sr::merge_channels(i);
            tensor("merge",w.merge,4*c*merged); tensor("mergeBias",w.merge_bias,merged);
        } else if (i==10) {
            tensor("output",w.output,32*48); tensor("outputBias",w.output_bias,48);
        }
    }
    return count;
}

#ifdef RW_SR_K_RECONSTRUCTION
Json frame_control(const char* filter_path,sr::PrepareInput frame,const sr::Weights& weights,
                   const std::uint16_t* w,std::uint16_t* prepared,std::uint16_t* field,
                   void* scratch,std::size_t workspace,cudaStream_t stream,cudaEvent_t begin,cudaEvent_t end) {
    const auto raw=contents(filter_path,sr::filter_elements*2);
    const auto taps=sr::decode_reconstruction_filter(raw);
    std::uint16_t* filter=nullptr;
    check(cudaMalloc(&filter,raw.size()));
    check(cudaMemcpy(filter,taps.data(),raw.size(),cudaMemcpyHostToDevice));
    const auto g=sr::reconstruction_geometry(frame.width,frame.height);
    const auto pixels=[](sr::Geometry size){return std::size_t(size.width)*size.height;};
    std::vector<float> rgba(pixels(g.output)*4), motion(std::size_t(frame.width)*frame.height*2,0.f),
                       depth(std::size_t(frame.width)*frame.height,.5f);
    for (std::size_t i=0;i<motion.size();i+=2) motion[i]=-.25f;
    float *device_motion=nullptr,*device_depth=nullptr;
    check(cudaMalloc(&device_motion,motion.size()*sizeof(float)));
    check(cudaMalloc(&device_depth,depth.size()*sizeof(float)));
    check(cudaMemcpy(device_motion,motion.data(),motion.size()*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_depth,depth.data(),depth.size()*sizeof(float),cudaMemcpyHostToDevice));
    frame.motion_xy=device_motion;frame.depth=device_depth;frame.motion_scale_x=frame.motion_scale_y=2.f;
    std::array<sr::ReconstructionOutput,2> slots{};
    for (auto& surfaces:slots) {
        check(cudaMalloc(&surfaces.rgba,rgba.size()*sizeof(float)));
        check(cudaMalloc(&surfaces.ycocg,pixels(g.history)*8));
        check(cudaMalloc(&surfaces.luma,pixels(g.luma)*2));
        check(cudaMalloc(&surfaces.hidden,pixels(g.neural)*8));
    }
    Json result=Json::array();
    for (unsigned i=0;i<3;++i) {
        auto& next=slots[i%2];
        const auto& previous=slots[1-i%2];
        frame.reset=i!=1;
        frame.previous_ycocg=frame.reset ? nullptr : previous.ycocg;
        frame.previous_hidden=frame.reset ? nullptr : previous.hidden;
        check(cudaEventRecord(begin,stream));
        check(sr::enqueue_frame(frame,frame.reset ? nullptr : previous.luma,frame.reset ? 0u : 1u,
              weights,w,filter,prepared,field,next,scratch,workspace,stream));
        check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
        float ms=0.f;
        check(cudaEventElapsedTime(&ms,begin,end));
        check(cudaMemcpy(rgba.data(),next.rgba,rgba.size()*sizeof(float),cudaMemcpyDeviceToHost));
        if (std::any_of(rgba.begin(),rgba.end(),[](float value){return !std::isfinite(value);}))
            throw std::runtime_error("Nonfinite reconstructed SR output");
        const auto [minimum,maximum]=std::minmax_element(rgba.begin(),rgba.end());
        if (*minimum==*maximum) throw std::runtime_error("Constant reconstructed SR output");
        result.push_back({{"reset",frame.reset},{"device_ms",ms},{"min",*minimum},{"max",*maximum},
                          {"rgba_sha256",digest(rgba.data(),rgba.size()*sizeof(float))}});
    }
    if (result[0]["rgba_sha256"]!=result[2]["rgba_sha256"] || result[0]["rgba_sha256"]==result[1]["rgba_sha256"])
        throw std::runtime_error("SR reconstruction reset and feedback controls did not distinguish the frames");
    for (auto& surfaces:slots) {
        check(cudaFree(surfaces.rgba));check(cudaFree(surfaces.ycocg));check(cudaFree(surfaces.luma));check(cudaFree(surfaces.hidden));
    }
    check(cudaFree(device_motion));check(cudaFree(device_depth));check(cudaFree(filter));
    return {{"output_geometry",{g.output.width,g.output.height}},{"frames",result},
            {"filter_sha256",digest(raw.data(),raw.size())}};
}
#endif
} // namespace

int main(int argc,char** argv) {
    if (argc!=4 && argc!=5) {
        std::cerr<<"usage: rw-sr-source-backbone-probe PACKED_K DECODED_K {weights|cuda} [FILTER_K]\n";
        return 2;
    }
    try {
        const auto packed=contents(argv[1],sr::packed_bytes);
        const auto weights=sr::decode_original(packed);
        const auto decoded=contents(argv[2],3567296);
        const auto tensor_count=verify(weights,decoded);
        const bool gpu=std::string_view(argv[3])=="cuda";
        if (!gpu && std::string_view(argv[3])!="weights") return 2;
#ifndef RW_SR_K_RECONSTRUCTION
        if (argc==5) throw std::invalid_argument("Build the SR K reconstruction target to use the filter");
#endif
        constexpr int width=32,height=24;
        const auto g=sr::network_geometry(width,height);
        std::size_t workspace=0;
        for (unsigned phase=0;phase<8;++phase) workspace=std::max(workspace,sr::workspace_bytes(g,phase));
        Json report={{"model","dlss-sr-310.7.0-k-ldr-2x"},{"original_sha256",digest(packed.data(),packed.size())},
            {"decoded_bit_exact_tensors",tensor_count},{"gpu_executed",gpu},{"workspace_bytes",workspace},
            {"neural_field",{g.width,g.height}},{"scope","SDR prepare, 11 neural layers, and 40-channel reconstruction input"}};
        if (gpu) {
            int count=0;
            check(cudaGetDeviceCount(&count));
            if (count!=1) throw std::runtime_error("SR probe requires one visible H200");
            cudaDeviceProp device{};
            check(cudaGetDeviceProperties(&device,0));
            if (std::string_view(device.name).find("H200")==std::string_view::npos)
                throw std::runtime_error("SR probe is registered for H200");
            const auto pixels=std::size_t(g.width)*g.height;
            std::vector<__half> input(pixels*sr::input_channels),output(pixels*sr::reconstruction_channels);
            std::vector<float> color(width*height*3);
            for (std::size_t i=0;i<color.size();++i)
                color[i]=static_cast<float>((i*17+i/43)%91)/128.f;
            std::uint16_t *w=nullptr,*in=nullptr,*out=nullptr;
            float* rgb=nullptr;
            void* scratch=nullptr;
            check(cudaMalloc(&w,sr::packed_bytes)); check(cudaMalloc(&in,input.size()*2));
            check(cudaMalloc(&out,output.size()*2)); check(cudaMalloc(&scratch,workspace));
            check(cudaMalloc(&rgb,color.size()*sizeof(float)));
            check(cudaMemcpy(w,weights.elements.data(),sr::packed_bytes,cudaMemcpyHostToDevice));
            check(cudaMemcpy(rgb,color.data(),color.size()*sizeof(float),cudaMemcpyHostToDevice));
            cudaStream_t stream{};
            cudaEvent_t begin{},end{};
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
            sr::PrepareInput frame{};
            frame.width=width;frame.height=height;frame.color_rgb=rgb;frame.exposure=1.f;frame.reset=true;
            check(cudaEventRecord(begin,stream));
            check(sr::prepare(frame,in,stream));
            check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
            float prepare_ms=0.f;
            check(cudaEventElapsedTime(&prepare_ms,begin,end));
            check(cudaMemcpy(input.data(),in,input.size()*2,cudaMemcpyDeviceToHost));
            for (std::size_t i=0;i<input.size();i+=4)
                if (!std::isfinite(__half2float(input[i])) || __half_as_ushort(input[i])!=__half_as_ushort(input[i+1]) ||
                    __half_as_ushort(input[i+2])!=0 || __half_as_ushort(input[i+3])!=0)
                    throw std::runtime_error("SR reset field did not reproduce current luma and empty feedback");
            report["prepare_reset"]={{"device_ms",prepare_ms},{"output_sha256",digest(input.data(),input.size()*2)}};
            report["phases"]=Json::array();
            for (const unsigned phase:{0u,1u,0u}) {
                check(cudaEventRecord(begin,stream));
                check(sr::enqueue(g,phase,weights,w,in,out,scratch,workspace,stream));
                check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
                float ms=0.f;
                check(cudaEventElapsedTime(&ms,begin,end));
                check(cudaMemcpy(output.data(),out,output.size()*2,cudaMemcpyDeviceToHost));
                float minimum=INFINITY,maximum=-INFINITY;
                for (const auto half:output) {
                    const float value=__half2float(half);
                    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite SR backbone result");
                    minimum=std::min(minimum,value); maximum=std::max(maximum,value);
                }
                report["phases"].push_back({{"phase",phase},{"device_ms",ms},{"min",minimum},{"max",maximum},
                     {"output_sha256",digest(output.data(),output.size()*2)}});
            }
            if (report["phases"][0]["output_sha256"]!=report["phases"][2]["output_sha256"] ||
                report["phases"][0]["output_sha256"]==report["phases"][1]["output_sha256"])
                throw std::runtime_error("SR backbone phase or replay output differs from its control");
#ifdef RW_SR_K_RECONSTRUCTION
            if (argc==5) report["reconstruction"]=frame_control(argv[4],frame,weights,w,in,out,scratch,workspace,stream,begin,end);
#endif
            check(cudaEventDestroy(end)); check(cudaEventDestroy(begin)); check(cudaStreamDestroy(stream));
            check(cudaFree(rgb)); check(cudaFree(scratch)); check(cudaFree(out)); check(cudaFree(in)); check(cudaFree(w));
            report["device"]=device.name;
        }
        std::cout<<report.dump(2)<<'\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
}
