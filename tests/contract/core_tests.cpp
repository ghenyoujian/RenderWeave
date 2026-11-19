#include "rw/rw.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {
uint64_t checks = 0;
#define REQUIRE(expression) do { ++checks; if (!(expression)) { \
    std::cerr << __FILE__ << ':' << __LINE__ << ": " #expression " failed\n"; std::exit(1); } } while (false)

// Independent finite codebook, expressed as exact integer multiples of 2^-9.
// No production decoder/encoder is called to construct expected values.
constexpr std::array<uint32_t, 127> positive_units = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,
    64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,
    256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,
    1024,1152,1280,1408,1536,1664,1792,1920,2048,2304,2560,2816,3072,3328,3584,3840,
    4096,4608,5120,5632,6144,6656,7168,7680,8192,9216,10240,11264,12288,13312,14336,15360,
    16384,18432,20480,22528,24576,26624,28672,30720,32768,36864,40960,45056,49152,53248,57344,61440,
    65536,73728,81920,90112,98304,106496,114688,122880,131072,147456,163840,180224,196608,212992,229376
};
uint8_t oracle_encode(float input) {
    const unsigned sign = std::signbit(input) ? 128 : 0;
    if (std::isnan(input)) return static_cast<uint8_t>(sign | 127);
    const double scaled = std::min(std::fabs(static_cast<double>(input)), 448.0) * 512.0;
    unsigned closest = 0;
    double distance = std::numeric_limits<double>::infinity();
    for (unsigned code = 0; code < positive_units.size(); ++code) {
        const double candidate = std::abs(scaled - positive_units[code]);
        if (candidate < distance || (candidate == distance && (code & 1u) == 0)) {
            closest = code; distance = candidate;
        }
    }
    return static_cast<uint8_t>(sign | closest);
}

void numeric_tests() {
    for (unsigned code = 0; code < 256; ++code) {
        const unsigned absolute = code & 127u;
        const auto decoded = rw_e4m3_decode(static_cast<uint8_t>(code));
        if (absolute == 127u) REQUIRE(std::isnan(decoded));
        else {
            const float expected = std::copysign(positive_units[absolute] / 512.0f, code & 128u ? -1.0f : 1.0f);
            REQUIRE(std::bit_cast<uint32_t>(decoded) == std::bit_cast<uint32_t>(expected));
            REQUIRE(rw_e4m3_encode(expected) == code);
        }
    }
    for (unsigned low = 0; low + 1 < positive_units.size(); ++low) {
        const float midpoint = (positive_units[low] + positive_units[low + 1]) / 1024.0f;
        const uint8_t even = static_cast<uint8_t>((low & 1u) ? low + 1 : low);
        for (float sign : {1.0f, -1.0f}) {
            REQUIRE(rw_e4m3_encode(sign * midpoint) == (even | (sign < 0 ? 128u : 0u)));
            for (float value : {std::nextafter(midpoint, -std::numeric_limits<float>::infinity()),
                                std::nextafter(midpoint, std::numeric_limits<float>::infinity())})
                REQUIRE(rw_e4m3_encode(sign * value) == oracle_encode(sign * value));
        }
    }
    for (float value : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min()})
        REQUIRE(rw_e4m3_encode(value) == oracle_encode(value));
    uint32_t state = 0x29d37815;
    for (unsigned i = 0; i < 12000; ++i) {
        state = state * 1664525u + 1013904223u;
        const float value = std::bit_cast<float>(state);
        REQUIRE(rw_e4m3_encode(value) == oracle_encode(value));
    }
    uint8_t code = 0;
    REQUIRE(rw_e4m3_decode_array(nullptr, nullptr, 0) == RW_OK);
    REQUIRE(rw_e4m3_decode_array(&code, nullptr, 1) == RW_INVALID_ARGUMENT);
    std::array<float, 8> overlapping{};
    REQUIRE(rw_e4m3_encode_array(overlapping.data(), reinterpret_cast<uint8_t*>(overlapping.data()), 8) == RW_INVALID_ARGUMENT);

    // A non-tile-aligned exact integer GEMM: independent scalar integer products
    // exercise index order, seeded residuals, repeated rows and alternating signs.
    constexpr uint32_t m = 3, n = 5, k = 7;
    std::array<uint8_t, m*k> a{};
    std::array<uint8_t, k*n> b{};
    std::array<float, m*n> residual{}, output{};
    std::array<int, m*k> ia{};
    std::array<int, k*n> ib{};
    for (unsigned i = 0; i < ia.size(); ++i) { ia[i] = int(i % k) - 3; a[i] = rw_e4m3_encode(float(ia[i])); }
    for (unsigned i = 0; i < ib.size(); ++i) { ib[i] = int(i % 5) - 2; b[i] = rw_e4m3_encode(float(ib[i])); }
    for (unsigned i = 0; i < residual.size(); ++i) residual[i] = float(i - 7.0f);
    REQUIRE(rw_reference_gemm_e4m3(a.data(),a.size(),b.data(),b.size(),residual.data(),residual.size(),output.data(),output.size(),m,n,k) == RW_OK);
    for (unsigned row = 0; row < m; ++row) for (unsigned column = 0; column < n; ++column) {
        int expected = 0;
        for (unsigned inner = 0; inner < k; ++inner) expected += ia[row*k+inner] * ib[inner*n+column];
        REQUIRE(output[row*n+column] == residual[row*n+column] + expected);
    }
    REQUIRE(rw_reference_gemm_e4m3(a.data(),a.size()-1,b.data(),b.size(),nullptr,0,output.data(),output.size(),m,n,k) == RW_OUT_OF_BOUNDS);
    REQUIRE(rw_reference_gemm_e4m3(a.data(),a.size(),b.data(),b.size(),output.data(),output.size(),output.data(),output.size(),m,n,k) == RW_INVALID_ARGUMENT);

    const std::array<float, 6> plane = {0,2,4,3,5,7};
    for (const auto point : std::array<std::array<float, 2>, 7>{{{0,0},{2,1},{0.5f,0.5f},{1.25f,0.75f},{-10,0.5f},{9,0.5f},{0.5f,9}}}) {
        float actual = -1;
        REQUIRE(rw_reference_bilinear_clamp_f32(plane.data(),plane.size(),3,2,point[0],point[1],&actual) == RW_OK);
        const float expected = 2*std::clamp(point[0],0.0f,2.0f)+3*std::clamp(point[1],0.0f,1.0f);
        REQUIRE(actual == expected);
    }
    float sample;
    REQUIRE(rw_reference_bilinear_clamp_f32(plane.data(),5,3,2,0,0,&sample) == RW_OUT_OF_BOUNDS);
    REQUIRE(rw_reference_bilinear_clamp_f32(plane.data(),6,3,2,std::numeric_limits<float>::quiet_NaN(),0,&sample) == RW_INVALID_ARGUMENT);
    REQUIRE(rw_reference_bilinear_clamp_f32(plane.data(),std::numeric_limits<size_t>::max(),
        std::numeric_limits<uint32_t>::max(),1,std::numeric_limits<float>::max(),0,&sample) == RW_UNSUPPORTED);
    REQUIRE(rw_reference_bilinear_clamp_f32(plane.data(),std::numeric_limits<size_t>::max(),
        1,std::numeric_limits<uint32_t>::max(),0,std::numeric_limits<float>::max(),&sample) == RW_UNSUPPORTED);
    const std::array<uint32_t, 13> mirrored = {0,1,2,3,2,1,0,1,2,3,2,1,0};
    for (int64_t i = -6; i <= 6; ++i) {
        uint32_t index;
        REQUIRE(rw_mirror_index(i,4,&index) == RW_OK);
        REQUIRE(index == mirrored[static_cast<size_t>(i+6)]);
    }
    uint32_t index;
    REQUIRE(rw_mirror_index(std::numeric_limits<int64_t>::min(),1,&index) == RW_OK && index == 0);
}

rw_tensor_view tensor(float* memory, uint64_t count, int64_t height, int64_t width, int64_t channels) {
    rw_tensor_view view{};
    view.struct_size = sizeof(view); view.dtype = RW_DTYPE_F32;
    view.device_kind = RW_DEVICE_CPU; view.ndim = 3; view.access_mode = RW_ACCESS_READ;
    view.shape[0]=height; view.shape[1]=width; view.shape[2]=channels;
    view.strides_bytes[2]=4; view.strides_bytes[1]=channels*4; view.strides_bytes[0]=width*channels*4;
    view.base=memory; view.allocation_bytes=count*4;
    return view;
}

void contract_tests() {
    rw::Runtime runtime;
    REQUIRE(runtime.capabilities().find("\"native_model_execution_available\":false") != std::string::npos);
    size_t size = 1;
    char buffer[1];
    REQUIRE(rw_context_query(runtime.get(),"capabilities",buffer,&size) == RW_BUFFER_TOO_SMALL && size > 1);
    REQUIRE(rw_context_query(runtime.get(),"unknown",nullptr,&size) == RW_UNSUPPORTED);
    rw_context_desc cuda{};
    cuda.struct_size=sizeof(cuda); cuda.abi_version=RW_ABI_VERSION; cuda.device_kind=RW_DEVICE_CUDA;
    cuda.max_sessions=cuda.max_inflight=cuda.execution_slots=1;
    rw_context_t rejected = nullptr;
    REQUIRE(rw_context_create(&cuda,&rejected) == RW_UNSUPPORTED && rejected == nullptr);

    std::array<float, 18> color_data{};
    std::array<float, 12> motion_data{};
    auto color = tensor(color_data.data(),color_data.size(),2,3,3);
    uint64_t required;
    REQUIRE(rw_validate_tensor(&color,&required) == RW_OK && required == sizeof(color_data));
    auto changed = color;
    changed.allocation_bytes-=1;
    REQUIRE(rw_validate_tensor(&changed,&required) == RW_OUT_OF_BOUNDS);
    changed=color; changed.strides_bytes[0]=-36;
    REQUIRE(rw_validate_tensor(&changed,&required) == RW_INVALID_ARGUMENT);
    changed=color; changed.strides_bytes[0]=4;
    REQUIRE(rw_validate_tensor(&changed,&required) == RW_INVALID_ARGUMENT);
    changed=color; changed.ndim=9;
    REQUIRE(rw_validate_tensor(&changed,&required) == RW_INVALID_ARGUMENT);
    changed=color; changed.shape[0]=std::numeric_limits<int64_t>::max();
    REQUIRE(rw_validate_tensor(&changed,&required) == RW_OVERFLOW);
    changed=color; changed.struct_size=4;
    REQUIRE(rw_validate_tensor(&changed,&required) == RW_ABI_MISMATCH);
    changed=color; changed.byte_offset=4;
    REQUIRE(rw_validate_tensor(&changed,&required) == RW_OUT_OF_BOUNDS);

    std::array<rw_resource, 2> resources{};
    for (auto& resource : resources) {
        resource.struct_size=sizeof(resource); resource.validity=RW_VALIDITY_VALID;
        resource.provenance=RW_PROVENANCE_RENDERER_DERIVED;
        resource.metadata.struct_size=sizeof(resource.metadata);
    }
    resources[0].resource_semantic=RW_RESOURCE_COLOR; resources[0].tensor=color;
    resources[0].metadata.color_space=RW_COLOR_SRGB_CODE;
    resources[0].metadata.dynamic_range=RW_DYNAMIC_RANGE_SDR;
    resources[1].resource_semantic=RW_RESOURCE_MOTION;
    resources[1].tensor=tensor(motion_data.data(),motion_data.size(),2,3,2);
    resources[1].metadata.motion_direction=RW_MOTION_CURRENT_TO_PREVIOUS;
    resources[1].metadata.motion_units=RW_MOTION_PIXELS;
    resources[1].metadata.jitter_included=RW_JITTER_EXCLUDED;
    rw_frame_desc frame{};
    frame.struct_size=sizeof(frame); frame.frame_id=1; frame.time_base_num=1; frame.time_base_den=30;
    frame.resources=resources.data(); frame.resource_count=resources.size();
    rw_frame_contract contract{};
    contract.struct_size=sizeof(contract); contract.abi_version=RW_ABI_VERSION;
    contract.device_kind=RW_DEVICE_CPU; contract.temporal=1;
    contract.color_space=RW_COLOR_SRGB_CODE; contract.dynamic_range=RW_DYNAMIC_RANGE_SDR;
    contract.required_semantics=RW_SEMANTIC_BIT(RW_RESOURCE_COLOR)|RW_SEMANTIC_BIT(RW_RESOURCE_MOTION);
    contract.motion_direction=RW_MOTION_CURRENT_TO_PREVIOUS;
    contract.motion_units=RW_MOTION_PIXELS; contract.jitter_included=RW_JITTER_EXCLUDED;
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_OK); // zero-valued valid motion is present
    resources[1].validity=RW_VALIDITY_ABSENT;
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);
    resources[1].validity=RW_VALIDITY_VALID;
    resources[1].metadata.motion_direction=RW_MOTION_PREVIOUS_TO_CURRENT;
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);
    resources[1].metadata.motion_direction=RW_MOTION_CURRENT_TO_PREVIOUS;
    resources[0].metadata.color_space=0;
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);
    resources[0].metadata.color_space=RW_COLOR_SRGB_CODE;
    frame.time_base_den=0;
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);
    frame.time_base_den=30;
    contract.require_ready_event=1;
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);
    contract.require_ready_event=0;
    resources[1].tensor.shape[0]=1;
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);
    resources[1].tensor.shape[0]=2;

    std::array<float, 6> depth_data{};
    std::array<float, 102> invalid_depth_channels{};
    std::array<uint8_t, 6> mask_data{};
    std::array<uint8_t, 102> invalid_mask_channels{};
    std::array<rw_resource, 3> guided = {resources[0],resources[1],{}};
    auto& guide = guided[2];
    guide.struct_size=sizeof(guide); guide.validity=RW_VALIDITY_VALID;
    guide.provenance=RW_PROVENANCE_RENDERER_DERIVED;
    guide.metadata.struct_size=sizeof(guide.metadata);
    guide.metadata.depth_definition=RW_DEPTH_VIEW_Z;
    guide.resource_semantic=RW_RESOURCE_DEPTH;
    guide.tensor=tensor(depth_data.data(),depth_data.size(),2,3,1);
    guide.tensor.ndim=2;
    frame.resources=guided.data(); frame.resource_count=guided.size();
    contract.required_semantics |= RW_SEMANTIC_BIT(RW_RESOURCE_DEPTH);
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_OK);
    guide.tensor=tensor(invalid_depth_channels.data(),invalid_depth_channels.size(),2,3,17);
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);
    guide.tensor=tensor(depth_data.data(),depth_data.size(),2,3,1);
    guide.tensor.ndim=2; guide.tensor.dtype=RW_DTYPE_U8;
    guide.tensor.strides_bytes[0]=3; guide.tensor.strides_bytes[1]=1;
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);

    contract.required_semantics &= ~RW_SEMANTIC_BIT(RW_RESOURCE_DEPTH);
    contract.required_semantics |= RW_SEMANTIC_BIT(RW_RESOURCE_VALIDITY);
    guide.resource_semantic=RW_RESOURCE_VALIDITY;
    guide.tensor.base=mask_data.data(); guide.tensor.allocation_bytes=mask_data.size();
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_OK);
    guide.tensor=tensor(depth_data.data(),depth_data.size(),2,3,1);
    guide.tensor.ndim=2;
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);
    guide.tensor.dtype=RW_DTYPE_U8; guide.tensor.ndim=3;
    guide.tensor.shape[2]=17;
    guide.tensor.strides_bytes[0]=51; guide.tensor.strides_bytes[1]=17; guide.tensor.strides_bytes[2]=1;
    guide.tensor.base=invalid_mask_channels.data(); guide.tensor.allocation_bytes=invalid_mask_channels.size();
    REQUIRE(rw_validate_frame(&frame,&contract) == RW_INVALID_ARGUMENT);

    rw_memory_plan plan{};
    plan.struct_size=sizeof(plan); plan.abi_version=RW_ABI_VERSION;
    plan.budget_bytes=1000; plan.shared_weights_bytes=100;
    plan.bytes_per_slot=10; plan.slot_count=8; plan.bytes_per_session=20; plan.session_count=32;
    plan.import_output_bytes=80; plan.graph_handle_bytes=30; plan.reserve_quarantine_bytes=70;
    REQUIRE(rw_memory_plan_required(&plan,&required) == RW_OK && required == 1000);
    plan.budget_bytes=999;
    REQUIRE(rw_memory_plan_required(&plan,&required) == RW_RESOURCE_EXHAUSTED && required == 1000);
    plan.bytes_per_slot=std::numeric_limits<uint64_t>::max();
    REQUIRE(rw_memory_plan_required(&plan,&required) == RW_OVERFLOW);
}

rw_state_snapshot snapshot(rw_state_guard_t guard) {
    rw_state_snapshot state{};
    state.struct_size=sizeof(state); state.abi_version=RW_ABI_VERSION;
    REQUIRE(rw_state_guard_snapshot(guard,&state) == RW_OK);
    return state;
}
void transaction_tests() {
    rw_state_guard_t a=nullptr, b=nullptr;
    REQUIRE(rw_state_guard_create(1,0,&a) == RW_OK);
    REQUIRE(rw_state_guard_create(1,0,&b) == RW_OK);
    uint64_t a1,b1,late;
    REQUIRE(rw_state_guard_begin(a,1,0,1,&a1) == RW_OK);
    REQUIRE(rw_state_guard_begin(b,1,0,1,&b1) == RW_OK);
    REQUIRE(rw_state_guard_complete(a,a1,0,1) == RW_NOT_READY);
    REQUIRE(snapshot(a).pending_count == 1 && snapshot(a).predecessor == 0);
    REQUIRE(rw_state_guard_release(a) == RW_NOT_READY);
    REQUIRE(rw_state_guard_cancel(a,a1) == RW_OK);
    REQUIRE(snapshot(a).pending_count == 1); // cancel does not release device-owned metadata
    REQUIRE(rw_state_guard_complete(a,a1,1,1) == RW_CANCELLED);
    REQUIRE(snapshot(a).predecessor == 0);
    REQUIRE(rw_state_guard_complete(b,b1,1,1) == RW_OK);
    REQUIRE(snapshot(b).predecessor == 1);
    REQUIRE(rw_state_guard_begin(a,1,0,2,&late) == RW_OK);
    REQUIRE(rw_state_guard_reset(a,2,0) == RW_OK);
    REQUIRE(snapshot(a).pending_count == 1);
    REQUIRE(rw_state_guard_complete(a,late,1,1) == RW_STALE);
    REQUIRE(snapshot(a).predecessor == 0 && snapshot(a).epoch == 2);
    uint64_t invalid;
    REQUIRE(rw_state_guard_begin(a,2,0,3,&invalid) == RW_OK);
    REQUIRE(rw_state_guard_complete(a,invalid,1,0) == RW_OUTPUT_INVALID);
    REQUIRE(snapshot(a).predecessor == 0);
    uint64_t competing;
    REQUIRE(rw_state_guard_begin(a,2,0,3,&a1) == RW_OK);
    REQUIRE(rw_state_guard_begin(a,2,0,4,&competing) == RW_OK);
    REQUIRE(rw_state_guard_complete(a,a1,1,1) == RW_OK);
    REQUIRE(rw_state_guard_complete(a,competing,1,1) == RW_STALE);
    REQUIRE(snapshot(a).predecessor == 3 && snapshot(b).predecessor == 1);
    REQUIRE(rw_state_guard_reset(a,2,0) == RW_INVALID_ARGUMENT);
    REQUIRE(rw_state_guard_release(a) == RW_OK);
    REQUIRE(rw_state_guard_release(b) == RW_OK);
}
}

int main() {
    numeric_tests(); contract_tests(); transaction_tests();
    std::cout << "{\"project_id\":\"renderweave\",\"scope\":\"cpu_contract_and_arithmetic_reference\","
              << "\"checks\":" << checks << ",\"passed\":true,\"native_model_executed\":false}\n";
    return 0;
}
