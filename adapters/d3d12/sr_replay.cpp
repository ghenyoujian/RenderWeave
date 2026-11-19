#include "sr_bridge.hpp"

#include <bcrypt.h>
#include <cuda_runtime_api.h>
#include <dxgi1_4.h>
#include <fcntl.h>
#include <io.h>
#include <psapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {
using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;
constexpr uint32_t width = 960, height = 540, pixels = width * height;
constexpr uint32_t output_pitch = 2 * width * 4 * sizeof(float);
constexpr uint64_t input_bytes = pixels * 6 * sizeof(float), output_bytes = pixels * 16 * sizeof(float);
constexpr int warmup = 12;

void check(HRESULT status, const char* operation) {
    if (FAILED(status)) throw std::system_error(status, std::system_category(), operation);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

double ms(Clock::duration duration) { return std::chrono::duration<double, std::milli>(duration).count(); }

double cpu_seconds() {
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user))
        throw std::runtime_error("GetProcessTimes failed");
    auto ticks = [](FILETIME time) { return (uint64_t(time.dwHighDateTime) << 32) | time.dwLowDateTime; };
    return double(ticks(kernel) + ticks(user)) * 1e-7;
}

std::string hash(std::span<const float> output) {
    std::array<unsigned char, 32> digest{};
    if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
                   reinterpret_cast<PUCHAR>(const_cast<float*>(output.data())),
                   static_cast<ULONG>(output.size_bytes()), digest.data(), static_cast<ULONG>(digest.size())) < 0)
        throw std::runtime_error("BCryptHash failed");
    std::ostringstream hex;
    for (auto byte : digest) hex << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return hex.str();
}

ComPtr<ID3D12Device> graphics_device(ComPtr<IDXGIAdapter3>& adapter) {
    cudaDeviceProp cuda{};
    check(cudaGetDeviceProperties(&cuda, 0));
    LUID luid{};
    static_assert(sizeof(luid) == sizeof(cuda.luid));
    std::memcpy(&luid, cuda.luid, sizeof(luid));
    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "Create DXGI factory");
    check(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)), "Find CUDA adapter by LUID");
    ComPtr<ID3D12Device> device;
    check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)), "Create D3D12 device");
    return device;
}

D3D12_RESOURCE_DESC buffer_desc(uint64_t bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

ComPtr<ID3D12Resource> resource(ID3D12Device& device, D3D12_RESOURCE_DESC desc,
                               D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES initial) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    ComPtr<ID3D12Resource> object;
    check(device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, initial,
          nullptr, IID_PPV_ARGS(&object)), "Create replay resource");
    return object;
}

struct Commands {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;

    template<class Record>
    Commands(ID3D12Device& device, Record record) {
        check(device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
              "Create command allocator");
        check(device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                       IID_PPV_ARGS(&list)), "Create command list");
        record(*list.Get());
        check(list->Close(), "Close command list");
    }
};

class Queue {
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> completed_;
    HANDLE event_ = nullptr;
    uint64_t serial_ = 0;

public:
    explicit Queue(ID3D12Device& device) {
        D3D12_COMMAND_QUEUE_DESC desc{};
        check(device.CreateCommandQueue(&desc, IID_PPV_ARGS(&queue_)), "Create graphics queue");
        check(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completed_)), "Create graphics completion fence");
        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_) throw std::runtime_error("CreateEventW failed");
    }

    ~Queue() { CloseHandle(event_); }
    ID3D12CommandQueue& get() { return *queue_.Get(); }

    void execute(const Commands& commands) {
        ID3D12CommandList* list[] = {commands.list.Get()};
        queue_->ExecuteCommandLists(1, list);
    }

    void finish() {
        ++serial_;
        check(queue_->Signal(completed_.Get(), serial_), "Signal graphics completion");
        check(completed_->SetEventOnCompletion(serial_, event_), "Wait for graphics completion");
        if (WaitForSingleObject(event_, INFINITE) != WAIT_OBJECT_0 || completed_->GetCompletedValue() != serial_)
            throw std::runtime_error("Graphics completion was lost");
    }
};

class Replay {
    rw::d3d12::SrBridge bridge_;
    Queue queue_;
    ComPtr<ID3D12Resource> upload_, source_, texture_, readback_;
    D3D12_TEXTURE_COPY_LOCATION shared_source_{}, destination_{}, readback_target_{};
    std::unique_ptr<Commands> upload_commands_, produce_commands_, consume_commands_, readback_commands_;
    std::vector<float> translated_ = std::vector<float>(6 * pixels);
    std::vector<float> result_ = std::vector<float>(16 * pixels);

    void upload_frame(std::span<const float> rgb, int shift, int previous_shift) {
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                const auto offset = y * width + x;
                const auto from = y * width + std::clamp(int(x) - shift, 0, int(width) - 1);
                std::copy_n(rgb.data() + 3 * from, 3, translated_.data() + 3 * offset);
                translated_[3 * pixels + 2 * offset] = float(previous_shift - shift);
                translated_[5 * pixels + offset] = .5f;
            }
        }
        D3D12_RANGE empty{0, 0};
        void* mapped = nullptr;
        check(upload_->Map(0, &empty, &mapped), "Map input upload buffer");
        std::memcpy(mapped, translated_.data(), input_bytes);
        upload_->Unmap(0, nullptr);
        queue_.execute(*upload_commands_);
        queue_.finish();
    }

    void read_graphics_output() {
        queue_.execute(*readback_commands_);
        queue_.finish();
        D3D12_RANGE range{0, output_bytes}, empty{0, 0};
        void* mapped = nullptr;
        check(readback_->Map(0, &range, &mapped), "Map D3D12 output readback");
        std::memcpy(result_.data(), mapped, output_bytes);
        readback_->Unmap(0, &empty);
    }

public:
    explicit Replay(ID3D12Device& device) : bridge_(device, width, height), queue_(device) {
        upload_ = resource(device, buffer_desc(input_bytes), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        source_ = resource(device, buffer_desc(input_bytes), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        readback_ = resource(device, buffer_desc(output_bytes), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = 2 * width;
        texture.Height = 2 * height;
        texture.DepthOrArraySize = texture.MipLevels = 1;
        texture.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        texture.SampleDesc.Count = 1;
        texture_ = resource(device, texture, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        shared_source_.pResource = bridge_.output();
        shared_source_.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        shared_source_.PlacedFootprint.Footprint = {texture.Format, 2 * width, 2 * height, 1, output_pitch};
        static_assert(output_pitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT == 0);
        destination_.pResource = texture_.Get();
        destination_.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        readback_target_ = shared_source_;
        readback_target_.pResource = readback_.Get();

        upload_commands_ = std::make_unique<Commands>(device, [&](auto& list) {
            list.CopyBufferRegion(source_.Get(), 0, upload_.Get(), 0, input_bytes);
        });
        produce_commands_ = std::make_unique<Commands>(device, [&](auto& list) {
            list.CopyBufferRegion(bridge_.input(), 0, source_.Get(), 0, input_bytes);
        });
        consume_commands_ = std::make_unique<Commands>(device, [&](auto& list) {
            list.CopyTextureRegion(&destination_, 0, 0, 0, &shared_source_, nullptr);
        });
        readback_commands_ = std::make_unique<Commands>(device, [&](auto& list) {
            // Keep the persistent texture in COPY_DEST between replay frames.
            const D3D12_RESOURCE_BARRIER barrier{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                D3D12_RESOURCE_BARRIER_FLAG_NONE, {D3D12_RESOURCE_TRANSITION_BARRIER{texture_.Get(),
                    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE}}};
            list.ResourceBarrier(1, &barrier);
            list.CopyTextureRegion(&readback_target_, 0, 0, 0, &destination_, nullptr);
            auto restore = barrier;
            std::swap(restore.Transition.StateBefore, restore.Transition.StateAfter);
            list.ResourceBarrier(1, &restore);
        });
    }

    int cuda_device() const { return bridge_.cuda_device(); }
    uint64_t shared_bytes() const { return bridge_.allocation_bytes(); }

    std::vector<std::string> run(rw::sr::Session& session, IDXGIAdapter3& adapter,
                                  std::span<const float> rgb, int frames, bool interop, std::ostream& out) {
        upload_frame(rgb, 0, 0);
        if (!interop) { queue_.execute(*produce_commands_); queue_.finish(); }
        std::vector<double> ready_ms, consumer_ms, delivery_ms, stream_ms;
        std::vector<std::string> hashes;
        Clock::time_point end;
        double cpu_end = 0;
        auto deliver = [&](uint64_t frame, bool timed, bool capture) {
            const auto begin = Clock::now();
            if (interop) queue_.execute(*produce_commands_);
            auto* ready_event = interop ? bridge_.producer_event(queue_.get()) : nullptr;
            auto input = bridge_.frame(frame, 1, frame - 1);
            if (frame == 1) {
                input.motion = input.depth = nullptr;
                input.motion_elements = input.depth_elements = 0;
            }
            auto job = session.submit_cuda(input, ready_event);
            const auto view = job.wait_cuda().value();
            const auto ready = Clock::now();
            const auto stream = timed ? job.info().device_stream_ms : 0.0;
            if (interop) {
                bridge_.publish(queue_.get(), std::move(job), view);
                queue_.execute(*consume_commands_);
                queue_.finish();
                bridge_.retire();
            }
            const auto delivered = Clock::now();
            if (timed) {
                ready_ms.push_back(ms(ready - begin));
                consumer_ms.push_back(ms(delivered - ready));
                delivery_ms.push_back(ms(delivered - begin));
                stream_ms.push_back(stream);
            }
            if (capture) {
                if (timed) { end = delivered; cpu_end = cpu_seconds(); }
                if (interop) read_graphics_output();
                else check(cudaMemcpy(result_.data(), view.rgba, output_bytes, cudaMemcpyDeviceToHost));
                hashes.push_back(hash(result_));
            }
        };

        for (int frame = 1; frame <= warmup; ++frame) deliver(frame, false, false);
        const auto cpu_start = cpu_seconds();
        const auto start = Clock::now();
        for (int frame = warmup + 1; frame <= warmup + frames; ++frame)
            deliver(frame, true, frame == warmup + frames);
        const auto elapsed = ms(end - start) / 1000;
        const auto cpu_cores = (cpu_end - cpu_start) / elapsed;
        PROCESS_MEMORY_COUNTERS memory{};
        if (!GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)))
            throw std::runtime_error("GetProcessMemoryInfo failed");
        DXGI_QUERY_VIDEO_MEMORY_INFO local{}, non_local{};
        check(adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local), "Query local video memory");
        check(adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local), "Query non-local video memory");

        uint64_t frame = warmup + frames + 1;
        int previous_shift = 0;
        for (int shift : {1, 2, 3, 4, 3, 2, 1, 0}) {
            upload_frame(rgb, shift, previous_shift);
            if (!interop) { queue_.execute(*produce_commands_); queue_.finish(); }
            deliver(frame++, false, true);
            previous_shift = shift;
        }

        auto mean = [frames](const auto& values) { return std::accumulate(values.begin(), values.end(), 0.0) / frames; };
        std::sort(delivery_ms.begin(), delivery_ms.end());
        const auto p99 = delivery_ms[static_cast<std::size_t>(std::ceil(.99 * frames)) - 1];
        out << "{\"d3d12\":" << (interop ? "true" : "false") << ",\"frames\":" << frames
            << ",\"fps\":" << frames / elapsed << ",\"elapsed_seconds\":" << elapsed
            << ",\"mean_ready_ms\":" << mean(ready_ms) << ",\"mean_consumer_ms\":" << mean(consumer_ms)
            << ",\"mean_delivery_ms\":" << mean(delivery_ms) << ",\"p99_delivery_ms\":" << p99
            << ",\"mean_sdk_stream_ms\":" << mean(stream_ms) << ",\"process_cpu_cores\":" << cpu_cores
            << ",\"working_set_bytes\":" << memory.WorkingSetSize
            << ",\"dxgi_process_local_bytes\":" << local.CurrentUsage
            << ",\"dxgi_process_non_local_bytes\":" << non_local.CurrentUsage
            << ",\"output_sha256\":[";
        for (std::size_t i = 0; i < hashes.size(); ++i) out << (i ? ",\"" : "\"") << hashes[i] << '"';
        out << "]}";
        return hashes;
    }
};

template<class T>
void read(std::span<T> data) {
    if (!std::cin.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size_bytes())))
        throw std::runtime_error("Unexpected end of replay input");
}

}

int main(int argc, char** argv) try {
    const int frames = argc == 2 ? std::stoi(argv[1]) : 600;
    if (frames < 1) throw std::invalid_argument("Specify a positive frame count");
    _setmode(_fileno(stdin), _O_BINARY);
    std::array<uint64_t, 2> sizes{};
    read(std::span<uint64_t>(sizes));
    std::vector<std::byte> weights(sizes[0]), filter(sizes[1]);
    std::vector<float> rgb(3 * pixels);
    read(std::span(weights));
    read(std::span(filter));
    read(std::span(rgb));

    ComPtr<IDXGIAdapter3> adapter;
    auto graphics = graphics_device(adapter);
    Replay replay(*graphics.Get());
    const rw_sr_runtime_desc desc{sizeof(rw_sr_runtime_desc), RW_ABI_VERSION, replay.cuda_device(), width, height,
        1, 1, 768ull << 20, nullptr, nullptr,
        RW_SR_CUDA_GRAPH | RW_SR_TILED_BACKBONE | RW_SR_PARALLEL_REDUCE | RW_SR_REUSE_MATRIX_ROWS | RW_SR_CUDA_IO_ONLY};
    rw::sr::Runtime runtime(desc, weights, filter);

    std::cout << std::setprecision(12);
    // Both profiles use separate sessions and the very same compiled runtime.
    std::ostringstream profile;
    std::vector<std::string> cuda, d3d12;
    auto run = [&](bool interop) {
        auto session = runtime.session(1);
        return replay.run(session, *adapter.Get(), rgb, frames, interop, profile);
    };
    cuda = run(false);
    profile << ',';
    d3d12 = run(true);
    const auto luid = graphics->GetAdapterLuid();
    const bool equal = cuda == d3d12;
    std::cout << "{\"status\":\"" << (equal ? "PASS" : "DIFFERENT_OUTPUT") << "\",\"cuda_ordinal\":" << replay.cuda_device()
              << ",\"d3d12_luid\":[" << luid.HighPart << ',' << luid.LowPart << ']'
              << ",\"shared_buffer_allocation_bytes\":" << replay.shared_bytes()
              << ",\"profiles\":[" << profile.str() << "]}\n";
    return equal ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
}
