#include "../../src/features/nr/native_graph.hpp"
#include "../../src/backends/cuda/native_nr/preparation_json.hpp"
#include "../../src/backends/cuda/native_nr/lowering_recipe.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace fs=std::filesystem;
using namespace rw::native_nr;
using json=nlohmann::json;
#ifndef RW_NATIVE_BUILD_COMMIT
#error "Build native NR using the frozen CMake target with RW_NATIVE_BUILD_COMMIT"
#endif
namespace {
uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
std::string env(const char* name) {
    const char* result=std::getenv(name);
    if(!result || !*result) throw std::runtime_error(std::string("missing execution environment: ")+name);
    return result;
}
bool within(const fs::path& candidate,const fs::path& root) {
    const auto relative=candidate.lexically_relative(root);
    return !relative.empty() && *relative.begin()!=".." && !relative.is_absolute();
}
fs::path input_path(const std::string& text,const fs::path& root) {
    const auto path=fs::canonical(text);
    if(!within(path,root) || !fs::is_regular_file(path)) throw std::invalid_argument("input path escapes its declared root");
    return path;
}
std::vector<float> read_f32(const fs::path& path,size_t count) {
    static_assert(std::endian::native==std::endian::little,"raw fixture format requires little-endian host");
    if(count>SIZE_MAX/sizeof(float) || fs::file_size(path)!=count*sizeof(float))
        throw std::invalid_argument("raw float32 fixture shape/size mismatch");
    std::vector<float> values(count);
    std::ifstream stream(path,std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()),static_cast<std::streamsize>(count*sizeof(float)));
    if(!stream || !std::all_of(values.begin(),values.end(),[](float value){return std::isfinite(value);}))
        throw std::invalid_argument("truncated or non-finite float32 fixture");
    return values;
}
json write_tensor(Context& context,const Tensor& tensor,const fs::path& path) {
    if(fs::exists(path)) throw std::runtime_error("refuse overwrite of existing tensor evidence");
    const auto values=context.download(tensor);
    if(values.empty() || !std::all_of(values.begin(),values.end(),[](float value){return std::isfinite(value);}))
        throw std::runtime_error("non-finite/empty device output cannot be successful");
    std::ofstream stream(path,std::ios::binary);
    if(!stream) throw std::runtime_error("cannot create output tensor evidence");
    stream.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(float)));
    stream.close();
    if(!stream) throw std::runtime_error("output tensor write failed");
    const auto bounds=std::minmax_element(values.begin(),values.end());
    return {{"path",path.string()},{"shape",tensor.shape},{"dtype","float32_le"},
            {"bytes",values.size()*sizeof(float)},{"min",*bounds.first},{"max",*bounds.second}};
}
void write_json(const fs::path& path,const json& value) {
    const auto temporary=fs::path(path.string()+".partial");
    if(fs::exists(path) || fs::exists(temporary)) throw std::runtime_error("evidence output already exists");
    std::ofstream stream(temporary);
    stream<<value.dump(2)<<'\n';
    stream.close();
    if(!stream) throw std::runtime_error("evidence JSON write failed");
    fs::rename(temporary,path);
}
std::map<std::string,std::string> arguments(int argc,char** argv) {
    std::map<std::string,std::string> result;
    const std::set<std::string> supported={"--plan","--weights","--input","--output-dir","--run-id",
        "--budget-mib","--stop-after","--capture-nodes","--capture-preprocessing","--features",
        "--capture-max-mib","--max-pixels","--capture-postprocessing","--node-detail"};
    for(int i=1;i<argc;i+=2) {
        if(i+1>=argc || !supported.count(argv[i]) || !result.emplace(argv[i],argv[i+1]).second)
            throw std::invalid_argument("expected unique --key value command arguments");
    }
    for(const auto* key:{"--plan","--weights","--input","--output-dir","--run-id"})
        if(!result.count(key)) throw std::invalid_argument(std::string("missing argument: ")+key);
    return result;
}
uint64_t number(const std::map<std::string,std::string>& args,const char* key,uint64_t fallback) {
    const auto found=args.find(key);
    if(found==args.end()) return fallback;
    if(!std::regex_match(found->second,std::regex("[0-9]+"))) throw std::invalid_argument("invalid unsigned numeric argument");
    return std::stoull(found->second);
}
}

int main(int argc,char** argv) {
    json report={{"project_id","renderweave"},{"task_id","RW-20261101-018"},
        {"implementation_kind","native_self"},{"support_status","experimental_unqualified"},
        {"execution_backend","own_cpp_cuda_cublas_no_tensor_framework"},
        {"input_profile","pinned_3beef793_stateless_source_graph"},{"native_vendor_equivalence_verified",false},
        {"native_temporal_qualification",false},{"production_qualified",false},{"status","FAILED"},
        {"arrived_host_ns",now_ns()}};
    fs::path output;
    std::unique_ptr<Graph> graph;
    try {
        const auto args=arguments(argc,argv);
        const std::string job=env("SLURM_JOB_ID");
        if(!std::regex_match(job,std::regex("[0-9]+")) || env("PROJECT_EXECUTION_ROLE")!="h200_release")
            throw std::runtime_error("published Slurm release required");
        const fs::path project=env("PROJECT_STORAGE_ROOT"),source=env("PROJECT_SOURCE_ROOT");
        if(fs::canonical(project)!=project || fs::canonical(source)!=source ||
            source.parent_path()!=project/"releases" ||
            !std::regex_match(source.filename().string(),std::regex("[0-9a-f]{40,64}")))
            throw std::runtime_error("unexpected project/release identity");
        if(source.filename().string()!=RW_NATIVE_BUILD_COMMIT)
            throw std::runtime_error("native binary was compiled from a different release commit");
        report["compiled_source_commit"]=RW_NATIVE_BUILD_COMMIT;
        const fs::path public_root=env("PROJECT_DATASETS_ROOT");
        if(fs::canonical(public_root)!=public_root) throw std::runtime_error("public dataset root changed realpath");
        auto output_candidate=fs::absolute(args.at("--output-dir")).lexically_normal();
        if(!within(output_candidate,project/"runs") ||
            !within(fs::weakly_canonical(output_candidate),project/"runs"))
            throw std::invalid_argument("output must resolve inside project runs before directory creation");
        if(fs::exists(output_candidate)) throw std::runtime_error("choose a fresh native run output directory");
        fs::create_directories(output_candidate);
        output=fs::canonical(output_candidate);
        if(!within(output,project/"runs")) throw std::invalid_argument("output resolves outside project runs");
        report.update({{"run_id",args.at("--run-id")},{"source_commit",source.filename().string()},
            {"slurm_job_id",job},{"cuda_visible_devices",env("CUDA_VISIBLE_DEVICES")},
            {"output_dir",output.string()},{"weights_path",input_path(args.at("--weights"),public_root/"weights").string()},
            {"plan_path",input_path(args.at("--plan"),project).string()},
            {"input_path",input_path(args.at("--input"),project/"runs").string()}});
        auto plan=SourcePlan::load(report.at("plan_path").get<std::string>());
        const uint64_t pixels=static_cast<uint64_t>(plan.padded_height)*plan.padded_width;
        if(pixels>number(args,"--max-pixels",262144)) throw std::runtime_error("input exceeds explicitly admitted native smoke profile");
        const uint64_t mib=number(args,"--budget-mib",16384);
        if(mib==0 || mib>131072) throw std::invalid_argument("native budget must be in 1..131072 MiB");
        const uint64_t budget=mib*1024*1024;
        const uint64_t stop=number(args,"--stop-after",plan.layers.size()-1);
        if(stop>=plan.layers.size()) throw std::invalid_argument("stop node exceeds source graph");
        const bool prepare_post=uses_post_projection_v2(plan.network) && stop==plan.layers.size()-1;
        const auto native_bound=lowering_memory_bound(plan.modelled_memory_bound,prepare_post);
        if(native_bound>budget) throw std::runtime_error("source allocation envelope plus post-only workspace exceeds device budget");
        report.update({{"network",plan.network},{"semantic_graph_sha256",plan.semantic_graph_sha256},
            {"valid_size",{plan.valid_width,plan.valid_height}},{"padded_size",{plan.padded_width,plan.padded_height}},
            {"modelled_memory_bound_bytes",plan.modelled_memory_bound},{"budget_bytes",budget},
            {"source_modelled_memory_bound_bytes",plan.modelled_memory_bound},{"native_modelled_memory_bound_bytes",native_bound},
            {"native_lowering_extra_workspace_bytes",prepare_post?post_projection_workspace_bytes:0},
            {"lowering_recipe",lowering_recipe_json(plan.network)},
            {"controls",{{"intensity",1.0},{"style",0},{"local_tone",1.0},{"local_structure",1.0},
                {"skin_structure",-1.0},{"auto_mask",true},{"seed",0}}}});
        Context context(budget);
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties,0),"query native execution GPU identity");
        int driver_version=0,runtime_version=0,blas_version=0;
        cuda_check(cudaDriverGetVersion(&driver_version),"query CUDA driver version");
        cuda_check(cudaRuntimeGetVersion(&runtime_version),"query CUDA runtime version");
        blas_check(cublasGetVersion(context.blas(),&blas_version),"query cuBLAS version");
        if(prepare_post) {
            if(blas_version!=nr_required_cublas_version())
                throw std::runtime_error("loaded cuBLAS version differs from the compiled NR lowering recipe");
            context.prepare_post_projection_blas();
        }
        const auto node_detail=number(args,"--node-detail",0);
        if(node_detail>1) throw std::invalid_argument("--node-detail must be 0 or 1");
        if(node_detail) context.prepare_node_timing();
        report["observability"]={{"mode",node_detail?"node_detail":"summary"},
            {"node_detail_enabled",bool(node_detail)},{"node_detail_collected",false},
            {"mandatory_physical_barriers_enabled",true},{"overhead_qualified",false},
            {"pre_post_capture_callbacks_included_in_node_span",true}};
        constexpr const char* hex="0123456789abcdef";
        std::string uuid="GPU-";
        for(int i=0;i<16;++i) {
            if(i==4 || i==6 || i==8 || i==10) uuid+='-';
            const auto byte=static_cast<unsigned char>(properties.uuid.bytes[i]);
            uuid+=hex[byte>>4u]; uuid+=hex[byte&15u];
        }
        report["environment"]={{"device_name",properties.name},{"device_ordinal",0},
            {"gpu_uuid",uuid},{"compute_capability",{properties.major,properties.minor}},
            {"cuda_driver_version",driver_version},{"cuda_runtime_version",runtime_version},
            {"cublas_version",blas_version},{"matrix_compute","FP32_PEDANTIC_no_TF32"}};
        Weights weights(report.at("weights_path").get<std::string>());
        const auto host=read_f32(report.at("input_path").get<std::string>(),static_cast<size_t>(pixels)*3);
        auto image=context.upload({1,3,plan.padded_height,plan.padded_width},host);
        RunOptions options;
        options.collect_node_metrics=bool(node_detail);
        options.stop_after=static_cast<int>(stop);
        options.capture_preprocessing=number(args,"--capture-preprocessing",0)!=0;
        options.capture_postprocessing=number(args,"--capture-postprocessing",0)!=0;
        if(const auto found=args.find("--capture-nodes");found!=args.end()) {
            std::stringstream list(found->second);
            for(std::string item;std::getline(list,item,',');) {
                if(!std::regex_match(item,std::regex("[0-9]+"))) throw std::invalid_argument("invalid capture node");
                options.capture_nodes.insert(std::stoi(item));
            }
        }
        Tensor features;
        if(const auto found=args.find("--features");found!=args.end()) {
            const auto path=input_path(found->second,project/"runs");
            const auto values=read_f32(path,static_cast<size_t>(pixels)*16);
            features=context.upload({1,16,plan.padded_height,plan.padded_width},values);
            options.explicit_features=&features;
            report["feature_input"]={{"source","explicit_differential_fixture"},{"path",path.string()},
                {"native_rng_executed",false}};
        } else report["feature_input"]={{"source","native_cpp_counter_rng"},{"native_rng_executed",true},
            {"numpy_libm_bit_parity_verified",false}};
        report["admitted_host_ns"]=now_ns();
        graph=std::make_unique<Graph>(std::move(plan));
        const uint64_t capture_mib=number(args,"--capture-max-mib",128);
        if(capture_mib>4096) throw std::invalid_argument("capture budget exceeds explicit 4096 MiB diagnostic limit");
        const uint64_t capture_limit=capture_mib*1024*1024;
        uint64_t captured=0;
        report["captures"]=json::array();
        const auto capture=[&](int node,int port,const Tensor& tensor) {
            const uint64_t bytes=tensor.numel()*sizeof(float);
            if(bytes>capture_limit-captured) throw std::runtime_error("diagnostic tensor capture exceeds independent budget");
            captured+=bytes;
            const std::string name=node==-1 ? "pre_features" : node==-2 ? "pre_lifted" :
                node==-3 ? "post_blend" : node==-4 ? "post_wide" : node==-5 ? "post_residual" :
                "node_"+std::to_string(node)+"_port_"+std::to_string(port);
            auto record=write_tensor(context,tensor,output/(name+".f32"));
            record.update({{"node",node},{"port",port}});
            report["captures"].push_back(std::move(record));
        };
        report["device_submission_host_ns"]=now_ns();
        auto result=graph->execute(context,weights,image,options,capture);
        context.synchronize();
        report["physical_completion_host_ns"]=now_ns();
        report["output"]=write_tensor(context,result.main,output/"output.f32");
        if(options.stop_after==static_cast<int>(graph->plan().layers.size())-1) {
            if(report["output"]["min"].get<float>()<0 || report["output"]["max"].get<float>()>1)
                throw std::runtime_error("composed source NR output is outside SDR contract");
        }
        report["output_available_host_ns"]=now_ns();
        report["status"]="OUTPUT_VALID_EXPERIMENTAL";
        report["executed_layer_count"]=options.stop_after+1;
        report["complete_stateless_source_graph_executed"]=options.stop_after==static_cast<int>(graph->plan().layers.size())-1;
        report["owned_peak_bytes"]=context.peak_live_bytes();
        report["owned_live_after_output_bytes"]=context.live_bytes();
        report["immutable_preparation"]=preparation_json(weights.preparation_statistics());
        report["post_projection_workspace"]=post_projection_json(context);
        report["memory_counter_scope"]="owned buffers plus private cuBLAS workspace; driver/library reserve separate";
        report["timing_phase"]="cold_compatible_eager";
        report["timing_contains_diagnostics"]=!options.capture_nodes.empty() || options.capture_preprocessing || options.capture_postprocessing;
        // Optional event retirement must be observed before this diagnostic
        // can report success. Context's destructor is only a Q fallback.
        result={};features={};image={};
        weights.clear_device_cache_for_retirement();
        const bool retired=context.retire_owned_resources();
        const auto timer=context.node_timing_info();
        report["node_timing"]={{"preparation_attempted",timer.preparation_attempted},{"prepared",timer.prepared},
            {"active",timer.active},{"failed",timer.failed},{"created_events",timer.created_events},
            {"destroyed_events",timer.destroyed_events},{"begin_records",timer.begin_records},
            {"end_records",timer.end_records},{"elapsed_reads",timer.elapsed_reads},
            {"last_cuda_error",timer.last_cuda_error}};
        report["resource_retirement"]={{"succeeded",retired},{"owned_live_bytes_after_retirement",context.live_bytes()}};
        if(!retired) throw std::runtime_error("diagnostic Context retirement failed; owned resources remain quarantined");
    } catch(const std::exception& error) {
        report["status"]="FAILED"; report["error"]=error.what(); report["failed_host_ns"]=now_ns();
    }
    if(graph) {
        report["observability"]["node_detail_collected"]=!graph->measurements().empty();
        // Keep the legacy list shape for offline readers. Availability is
        // explicit above; individual unmeasured spans are JSON null.
        report["node_measurements"]=json::array();
        for(const auto& node:graph->measurements()) report["node_measurements"].push_back({
            {"index",node.index},{"kind",graph->plan().layers.at(static_cast<size_t>(node.index)).kind},
            {"status",node_measurement_state_name(node.status)},
            {"dispatch_start_ns",node.dispatch_start_ns},{"dispatch_end_ns",node.dispatch_end_ns},
            {"device_span_collected",node.device_span_collected},
            {"device_stream_span_ms",node.device_span_collected?json(node.device_stream_span_ms):json()},
            {"live_owned_bytes",node.live_owned_bytes},
            {"peak_owned_bytes",node.peak_owned_bytes}});
    }
    try { if(!output.empty()) write_json(output/"native_execution.json",report); }
    catch(const std::exception& error) {
        std::cerr<<"evidence delivery failed: "<<error.what()<<'\n';
        report["status"]="FAILED_EVIDENCE_DELIVERY";
    }
    std::cout<<report.dump()<<'\n';
    return report["status"]=="OUTPUT_VALID_EXPERIMENTAL" ? 0 : 1;
}
