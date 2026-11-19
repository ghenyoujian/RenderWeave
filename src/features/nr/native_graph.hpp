#pragma once
#include "../../backends/cuda/native_nr/native_nr.hpp"
#include "node_metrics.hpp"
#include <array>
#include <functional>
#include <set>
#include <span>

namespace rw::native_nr {
struct SourcePlan {
    std::string network, source_commit, semantic_graph_sha256;
    int valid_width=0,valid_height=0,padded_width=0,padded_height=0;
    uint64_t modelled_memory_bound=0;
    std::vector<LayerSpec> layers;
    std::vector<std::vector<std::vector<int64_t>>> output_shapes;
    std::vector<int> last_use;
    static SourcePlan load(const std::filesystem::path&);
};
struct RunOptions {
    Controls controls;
    int stop_after=-1;
    std::set<int> capture_nodes;
    bool capture_preprocessing=false;
    bool capture_postprocessing=false;
    // Optional observation only. Shape, ownership, physical completion and
    // per-request correctness guards are independent of this switch.
    bool collect_node_metrics=false;
    bool defer_completion=false; // SDK download drains the same stream.
    bool sdr_temporal_head=false;
    const Tensor* explicit_features=nullptr;
    const Tensor* previous_image=nullptr;
    const Tensor* motion_image=nullptr;
    const Tensor* motion_validity_image=nullptr;
    std::function<void(int,bool)> node_boundary;
};
using Capture = std::function<void(int node,int port,const Tensor&)>;
class Graph {
    SourcePlan plan_;
    std::array<NodeMeasurement,max_node_measurements> measurements_{};
    uint32_t measurement_count_=0;
    uint64_t execution_serial_=0;
    GraphExecutionSummary execution_summary_{};
public:
    explicit Graph(SourcePlan plan);
    const SourcePlan& plan() const { return plan_; }
    std::span<const NodeMeasurement> measurements() const { return {measurements_.data(),measurement_count_}; }
    const GraphExecutionSummary& execution_summary() const { return execution_summary_; }
    LayerOutput execute(Context&,Weights&,const Tensor& padded_image,const RunOptions&,const Capture&);
    void note_replay();
};
} // namespace rw::native_nr
