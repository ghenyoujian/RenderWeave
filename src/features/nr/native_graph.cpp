#include "native_graph.hpp"
#include "../../backends/cuda/native_nr/lowering_recipe.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace rw::native_nr {
namespace {
constexpr const char* pinned_source="3beef7935a8ef976b47ab78b906e75600c52d2de";
const std::set<std::string> kinds={"PreBlock","PostBlock","Swin1H","Swin2H","Swin4H","Swin8H",
    "SplitFfwd","SplitFfwdProj","SplitQKVAttn","SplitProj","SplitProjPool","SplitFinalHead",
    "VitFfnExpand","VitFfnContract","VitQKV","VitAttention","VitProjection","DecInputUpsample"};
uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
int bounded_int(const nlohmann::json& object,const char* key,int low,int high) {
    const auto& field=object.at(key);
    if(!field.is_number_integer()) throw std::invalid_argument(std::string("nonintegral plan field: ")+key);
    const int64_t value=field.get<int64_t>();
    if(value<low || value>high) throw std::invalid_argument(std::string("plan field outside bound: ")+key);
    return static_cast<int>(value);
}
std::string exception_text(const std::exception_ptr& failure) {
    try { std::rethrow_exception(failure); }
    catch(const std::exception& error) { return error.what(); }
    catch(...) { return "non-standard node failure"; }
}
}

SourcePlan SourcePlan::load(const std::filesystem::path& path) {
    if(std::filesystem::file_size(path)>16*1024*1024) throw std::invalid_argument("unbounded semantic plan");
    std::ifstream stream(path);
    if(!stream) throw std::runtime_error("cannot open source semantic plan");
    nlohmann::json document;
    stream>>document;
    if(document.at("project_id")!="renderweave" ||
       document.at("artifact_kind")!="source_semantic_graph_and_preallocation_plan" ||
       document.at("source").at("commit")!=pinned_source ||
       document.at("input").at("mode")!="stateless_first_frame")
        throw std::invalid_argument("unrecognized semantic plan source/input identity");
    if(document.at("geometry").at("allow_generic_bucketing")!=false ||
       document.at("geometry").at("allow_unverified_tiling")!=false ||
       document.at("unknown_layer_kind_count")!=0)
        throw std::invalid_argument("unknown nodes or altered geometry forbidden");
    SourcePlan plan;
    plan.network=document.at("network").get<std::string>();
    if(plan.network!="ht" && plan.network!="small") throw std::invalid_argument("unknown original model variant");
    plan.source_commit=pinned_source;
    plan.semantic_graph_sha256=document.at("semantic_graph_sha256").get<std::string>();
    if(plan.semantic_graph_sha256.size()!=64 ||
       !std::all_of(plan.semantic_graph_sha256.begin(),plan.semantic_graph_sha256.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))
        throw std::invalid_argument("invalid graph fingerprint");
    plan.valid_width=bounded_int(document.at("input"),"width",33,16384);
    plan.valid_height=bounded_int(document.at("input"),"height",33,16384);
    plan.padded_width=bounded_int(document.at("geometry"),"width",33,32768);
    plan.padded_height=bounded_int(document.at("geometry"),"height",33,32768);
    if(plan.padded_width<plan.valid_width || plan.padded_height<plan.valid_height)
        throw std::invalid_argument("padded image does not contain valid image");
    plan.modelled_memory_bound=document.at("resources").at("modelled_device_allocation_bound_bytes").get<uint64_t>();
    const auto& nodes=document.at("nodes");
    const size_t expected=plan.network=="ht" ? 152 : 76;
    if(!nodes.is_array() || nodes.size()!=expected || document.at("layer_count")!=expected)
        throw std::invalid_argument("frozen source layer count mismatch");
    std::map<std::pair<int,int>,int> index;
    for(size_t i=0;i<nodes.size();++i) {
        const auto& node=nodes.at(i);
        const auto& source=node.at("spec");
        LayerSpec spec;
        spec.index=bounded_int(source,"index",0,static_cast<int>(expected)-1);
        if(spec.index!=static_cast<int>(i) || node.at("index")!=i) throw std::invalid_argument("reordered graph indices");
        spec.block=bounded_int(source,"block",0,70);
        spec.layer=bounded_int(source,"layer",0,4);
        spec.kind=source.at("kind").get<std::string>();
        if(!kinds.count(spec.kind)) throw std::invalid_argument("unknown original node: "+spec.kind);
        spec.variant=source.at("variant").get<std::string>();
        const std::set<std::string> variants={"","inpview","ds","upsample","outview","proj_pool","final_head"};
        if(!variants.count(spec.variant)) throw std::invalid_argument("unknown original layer variant");
        // Historical Python w/out_w are ROWS, h/out_h are COLUMNS.
        spec.height=bounded_int(source,"w",1,32768);
        spec.width=bounded_int(source,"h",1,32768);
        spec.out_height=bounded_int(source,"out_w",1,32768);
        spec.out_width=bounded_int(source,"out_h",1,32768);
        spec.channels=bounded_int(source,"channels",1,4096);
        spec.heads=bounded_int(source,"heads",0,32);
        spec.input_channels=bounded_int(source,"c_in",1,4096);
        spec.output_channels=bounded_int(source,"c_out",1,4096);
        // The pinned Swin wrapper calls _to_windows(..., shift_y, shift_x),
        // and Split likewise uses top=-shift_y*4, left=-shift_x*4. Both source
        // families therefore map physical ROWS from y and COLUMNS from x.
        spec.shift_rows=bounded_int(source,"shift_y",-1,0);
        spec.shift_columns=bounded_int(source,"shift_x",-1,0);
        if(!index.emplace(std::pair{spec.block,spec.layer},spec.index).second)
            throw std::invalid_argument("duplicate semantic block/layer identity");
        const auto& outputs=node.at("outputs");
        if(!outputs.is_array() || outputs.empty() || outputs.size()>2)
            throw std::invalid_argument("invalid published tensor ports");
        std::vector<std::vector<int64_t>> output_shapes;
        for(const auto& output:outputs) {
            const auto& dimensions=output.at("shape");
            if(!dimensions.is_array() || dimensions.size()!=4)
                throw std::invalid_argument("source output must have NCHW shape");
            std::vector<int64_t> shape=dimensions.get<std::vector<int64_t>>();
            if(shape[0]!=1 || shape[1]<1 || shape[1]>4096 || shape[2]<1 || shape[2]>32768 || shape[3]<1 || shape[3]>32768)
                throw std::invalid_argument("unbounded published tensor shape");
            output_shapes.push_back(std::move(shape));
        }
        plan.output_shapes.push_back(std::move(output_shapes));
        plan.layers.push_back(std::move(spec));
    }
    if(plan.layers.front().kind!="PreBlock" || plan.layers.back().kind!="PostBlock")
        throw std::invalid_argument("source graph endpoints changed");
    plan.last_use.resize(expected);
    for(size_t i=0;i<expected;++i) plan.last_use[i]=static_cast<int>(i);
    for(size_t i=0;i<expected;++i) {
        auto& spec=plan.layers[i];
        const auto& source=nodes.at(i).at("spec");
        auto resolve=[&](int block,int layer) {
            const auto found=index.find({block,layer});
            if(found==index.end() || found->second>=static_cast<int>(i))
                throw std::invalid_argument("missing or noncausal graph predecessor");
            return found->second;
        };
        if(spec.kind=="SplitProj" || spec.kind=="SplitProjPool") spec.skip_source=resolve(spec.block,1);
        else if(spec.kind=="SplitFinalHead") spec.skip_source=resolve(spec.block,3);
        else if(spec.kind=="VitFfnContract") spec.skip_source=resolve(spec.block-1,4);
        else if(spec.kind=="VitProjection") spec.skip_source=resolve(spec.block,1);
        else {
            const int block=bounded_int(source,"skip_block",-1,70);
            if(block>=0) spec.skip_source=resolve(block,spec.kind=="DecInputUpsample" ? 3 : 0);
        }
        if(i) plan.last_use[i-1]=std::max(plan.last_use[i-1],static_cast<int>(i));
        if(spec.skip_source>=0) plan.last_use[spec.skip_source]=std::max(plan.last_use[spec.skip_source],static_cast<int>(i));
    }
    return plan;
}

Graph::Graph(SourcePlan plan):plan_(std::move(plan)) {
    if(plan_.layers.empty() || plan_.layers.size()>max_node_measurements)
        throw std::invalid_argument("graph exceeds fixed node observation bound");
}
LayerOutput Graph::execute(Context& context,Weights& weights,const Tensor& image,
    const RunOptions& options,const Capture& capture) {
    // Reset before any guard may throw. A new failed invocation must never
    // expose a previous request's successful observations.
    measurement_count_=0;
    execution_summary_={};
    if(execution_serial_==std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("graph execution observation serial exhausted");
    execution_summary_.execution_serial=++execution_serial_;
    execution_summary_.graph_entered=true;
    execution_summary_.detail_requested=options.collect_node_metrics;
    if(weights.network()!=plan_.network) throw std::invalid_argument("weight variant differs from graph");
    if(!context.owns(image) || image.shape!=std::vector<int64_t>({1,3,plan_.padded_height,plan_.padded_width}))
        throw std::invalid_argument("image does not match exact source geometry/context");
    const int stop=options.stop_after<0 ? static_cast<int>(plan_.layers.size())-1 : options.stop_after;
    if(stop<0 || stop>=static_cast<int>(plan_.layers.size())) throw std::invalid_argument("invalid stop node");
    const bool needs_post=uses_post_projection_v2(plan_.network) && stop==static_cast<int>(plan_.layers.size())-1;
    const auto bound=lowering_memory_bound(plan_.modelled_memory_bound,needs_post);
    const uint64_t temporal_extra=options.sdr_temporal_head
        ? (options.previous_image ? 20ull : 4ull)*(image.numel()/3) +
          12ull*plan_.valid_width*plan_.valid_height + 512 : 0;
    if(bound>context.budget_bytes() || temporal_extra>context.budget_bytes()-bound)
        throw std::runtime_error("source analytical bound plus post-only workspace exceeds declared device budget");
    if(needs_post && !context.post_projection_info().prepared)
        throw std::runtime_error("full HT graph requires persistent post workspace prepared before submission");
    if(options.capture_nodes.size()>4) throw std::invalid_argument("select at most four diagnostic nodes");
    for(int node:options.capture_nodes) if(node<0 || node>stop) throw std::invalid_argument("capture node not executed");
    if(options.collect_node_metrics && !context.node_timing_info().prepared)
        throw std::runtime_error("detail node timing requires explicit Context startup preparation");
    if(options.sdr_temporal_head && (!needs_post || options.explicit_features ||
        options.controls.style!=0 || options.controls.intensity!=1.f ||
        (options.previous_image && !options.motion_image)))
        throw std::invalid_argument("SDR temporal head requires full HT, default style/intensity and motion for history");
    Tensor reprojected;
    std::map<int,LayerOutput> outputs;
    for(int i=0;i<=stop;++i) {
        const auto& spec=plan_.layers.at(i);
        execution_summary_.active_node=i;
        ++execution_summary_.started_nodes;
        NodeMeasurement* measurement=nullptr;
        if(options.collect_node_metrics) {
            measurement=&measurements_.at(measurement_count_++);
            *measurement={};
            measurement->index=i;
            measurement->dispatch_start_ns=now_ns();
            measurement->live_owned_bytes=context.live_bytes();
            measurement->peak_owned_bytes=context.peak_live_bytes();
        }
        try {
            if(options.node_boundary)options.node_boundary(i,true);
            if(options.collect_node_metrics)
                context.begin_node_timing(execution_summary_.execution_serial,i);
            const LayerOutput* skip=spec.skip_source<0 ? nullptr : &outputs.at(spec.skip_source);
            LayerOutput result;
            if(spec.kind=="PreBlock") {
                if(options.sdr_temporal_head && options.previous_image)
                    reprojected=reproject_sdr_history(context,*options.previous_image,*options.motion_image,
                        options.motion_validity_image,plan_.valid_width,plan_.valid_height);
                PreBlockDiagnostics diagnostic;
                result=run_pre(context,weights,spec,image,options.controls,options.explicit_features,
                    options.previous_image,options.motion_image,options.motion_validity_image,
                    options.capture_preprocessing ? &diagnostic : nullptr,
                    reprojected.empty() ? nullptr : &reprojected);
                if(options.capture_preprocessing && capture) {
                    capture(-1,0,diagnostic.features); capture(-2,0,diagnostic.lifted);
                }
            } else {
                const auto& previous=outputs.at(i-1);
                if(spec.kind=="PostBlock") {
                    PostBlockDiagnostics diagnostic;
                    result=run_post(context,weights,spec,previous.main,skip,image,options.controls,
                        options.capture_postprocessing ? &diagnostic : nullptr,
                        reprojected.empty() ? nullptr : &reprojected,
                        options.sdr_temporal_head ? plan_.valid_width : 0,
                        options.sdr_temporal_head ? plan_.valid_height : 0);
                    if(options.capture_postprocessing && capture) {
                        capture(-3,0,diagnostic.blend); capture(-4,0,diagnostic.wide); capture(-5,0,diagnostic.residual);
                    }
                }
                else if(spec.kind=="Swin1H" || spec.kind=="Swin2H" || spec.kind=="Swin4H" || spec.kind=="Swin8H")
                    result=run_swin(context,weights,spec,previous.main,skip);
                else if(spec.kind.rfind("Split",0)==0) result=run_split(context,weights,spec,previous,skip);
                else if(spec.kind.rfind("Vit",0)==0) result=run_vit(context,weights,spec,previous.main,skip);
                else if(spec.kind=="DecInputUpsample") result=run_decoder(context,weights,spec,previous.main,skip);
                else throw std::logic_error("unlowered semantic node");
            }
            if(result.main.empty()) throw std::runtime_error("node returned no real tensor");
            const auto& expected=plan_.output_shapes.at(i);
            if(result.main.shape!=expected.at(0) ||
               (expected.size()==2 && (result.skip.empty() || result.skip.shape!=expected.at(1))) ||
               (expected.size()==1 && !result.skip.empty()))
                throw std::runtime_error("lowered output differs from frozen semantic shape/port contract");
            if(options.collect_node_metrics) {
                measurement->device_stream_span_ms=context.finish_node_timing(execution_summary_.execution_serial,i);
                measurement->device_span_collected=true;
                measurement->dispatch_end_ns=now_ns();
                measurement->status=NodeMeasurementState::physically_complete;
                measurement->live_owned_bytes=context.live_bytes();
                measurement->peak_owned_bytes=context.peak_live_bytes();
            } else {
                // The default allocator keeps the original per-node physical
                // barrier because last-use retirement frees with cudaFree.
                // With explicit stream-ordered payload memory, cudaFreeAsync
                // sequences those frees on this stream, so one frame-end drain
                // is sufficient and removes up to 152 synchronizations/frame.
                if(!context.stream_ordered_memory() && !context.deferred_tensor_cache()) context.synchronize();
            }
            ++execution_summary_.physically_completed_nodes;
            outputs.emplace(i,std::move(result));
            if(options.capture_nodes.count(i) && capture) {
                capture(i,0,outputs.at(i).main);
                if(!outputs.at(i).skip.empty()) capture(i,1,outputs.at(i).skip);
            }
            for(auto found=outputs.begin();found!=outputs.end();) {
                if(found->first!=i && plan_.last_use.at(found->first)<=i) found=outputs.erase(found);
                else ++found;
            }
            if(options.node_boundary)options.node_boundary(i,false);
            execution_summary_.active_node=-1;
        } catch(...) {
            const auto failure=std::current_exception();
            execution_summary_.failed_node=i;
            if(measurement) {
                measurement->status=NodeMeasurementState::failed;
                measurement->dispatch_end_ns=now_ns();
                const auto timer=context.node_timing_info();
                if(timer.active && !timer.failed) {
                    try {
                        // A partial node can still own queued accesses. Drain
                        // that work before reusing its timer; failures latch Q.
                        measurement->device_stream_span_ms=context.finish_node_timing(execution_summary_.execution_serial,i);
                        measurement->device_span_collected=true;
                    } catch(const std::exception& timer_error) {
                        throw std::runtime_error(exception_text(failure)+"; node timing drain failed: "+timer_error.what());
                    } catch(...) {
                        throw std::runtime_error(exception_text(failure)+"; node timing drain failed with unknown error");
                    }
                }
            }
            std::rethrow_exception(failure);
        }
    }
    if(!options.defer_completion && !context.stream_ordered_memory() && !context.capture_active()) context.synchronize();
    execution_summary_.completed=true;
    return outputs.at(stop);
}
void Graph::note_replay() {
    measurement_count_=0;
    execution_summary_={};
    if(execution_serial_==std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("graph execution observation serial exhausted");
    execution_summary_.execution_serial=++execution_serial_;
    execution_summary_.graph_entered=true;
    execution_summary_.started_nodes=plan_.layers.size();
    execution_summary_.physically_completed_nodes=plan_.layers.size();
    execution_summary_.completed=true;
}
} // namespace rw::native_nr
