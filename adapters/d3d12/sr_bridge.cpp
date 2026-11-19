#include "sr_bridge.hpp"

#include <cuda_runtime_api.h>
#include <wrl/client.h>

#include <cstring>
#include <limits>
#include <optional>
#include <system_error>

using Microsoft::WRL::ComPtr;

namespace rw::d3d12 {
namespace {

void check(HRESULT status, const char* operation) {
    if (FAILED(status)) throw std::system_error(status, std::system_category(), operation);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int cuda_ordinal(ID3D12Device& device) {
    const auto expected = device.GetAdapterLuid();
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (device.GetNodeCount() == 1) {
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, ordinal));
            if (properties.luidDeviceNodeMask == 1 &&
                std::memcmp(properties.luid, &expected, sizeof(expected)) == 0) return ordinal;
        }
    }
    throw std::runtime_error("No CUDA device matches the single-node Direct3D adapter LUID");
}

struct SharedHandle {
    HANDLE value = nullptr;
    ~SharedHandle() { if (value) CloseHandle(value); }
};

struct Buffer {
    ComPtr<ID3D12Resource> resource;
    cudaExternalMemory_t memory = nullptr;
    void* pointer = nullptr;
    uint64_t allocation_bytes = 0;

    void create(ID3D12Device& device, uint64_t bytes, D3D12_RESOURCE_FLAGS flags) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask = heap.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;
        check(device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource)), "Create shared buffer");
        allocation_bytes = device.GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;

        SharedHandle handle;
        check(device.CreateSharedHandle(resource.Get(), nullptr, GENERIC_ALL, nullptr, &handle.value), "Share buffer");
        cudaExternalMemoryHandleDesc external{};
        external.type = cudaExternalMemoryHandleTypeD3D12Resource;
        external.handle.win32.handle = handle.value;
        external.size = allocation_bytes;
        external.flags = cudaExternalMemoryDedicated;
        check(cudaImportExternalMemory(&memory, &external));
        cudaExternalMemoryBufferDesc mapping{};
        mapping.size = bytes;
        check(cudaExternalMemoryGetMappedBuffer(&pointer, memory, &mapping));
    }

    ~Buffer() {
        if (pointer) cudaFree(pointer);
        if (memory) cudaDestroyExternalMemory(memory);
    }
};

struct Fence {
    ComPtr<ID3D12Fence> object;
    cudaExternalSemaphore_t semaphore = nullptr;

    void create(ID3D12Device& device) {
        check(device.CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&object)), "Create shared fence");
        SharedHandle handle;
        check(device.CreateSharedHandle(object.Get(), nullptr, GENERIC_ALL, nullptr, &handle.value), "Share fence");
        cudaExternalSemaphoreHandleDesc external{};
        external.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        external.handle.win32.handle = handle.value;
        check(cudaImportExternalSemaphore(&semaphore, &external));
    }

    ~Fence() { if (semaphore) cudaDestroyExternalSemaphore(semaphore); }
};

}

struct SrBridge::State {
    int ordinal;
    uint64_t pixels;
    uint64_t producer_value = 0, consumer_value = 0;
    Buffer input, output;
    Fence producer, consumer;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
    std::optional<rw::sr::Job> output_job;

    ~State() {
        output_job.reset();
        if (ready) cudaEventDestroy(ready);
        if (stream) cudaStreamDestroy(stream);
    }
};

SrBridge::SrBridge(ID3D12Device& device, uint32_t width, uint32_t height) : state_(new State{}) {
    if (!width || width > 4096 || !height || height > 4096) throw std::invalid_argument("Invalid SR input size");
    auto& state = *state_;
    state.ordinal = cuda_ordinal(device);
    state.pixels = uint64_t(width) * height;
    check(cudaSetDevice(state.ordinal));
    state.input.create(device, state.pixels * 6 * sizeof(float), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    state.output.create(device, state.pixels * 16 * sizeof(float), D3D12_RESOURCE_FLAG_NONE);
    state.producer.create(device);
    state.consumer.create(device);
    check(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&state.ready, cudaEventDisableTiming));
}

SrBridge::~SrBridge() {
    if (cudaSetDevice(state_->ordinal) != cudaSuccess || cudaStreamSynchronize(state_->stream) != cudaSuccess)
        (void)state_.release();
}

int SrBridge::cuda_device() const { return state_->ordinal; }
ID3D12Resource* SrBridge::input() const { return state_->input.resource.Get(); }
ID3D12Resource* SrBridge::output() const { return state_->output.resource.Get(); }
uint64_t SrBridge::allocation_bytes() const { return state_->input.allocation_bytes + state_->output.allocation_bytes; }

rw_sr_frame SrBridge::frame(uint64_t id, uint64_t epoch, uint64_t predecessor) const {
    const auto* planes = static_cast<const float*>(state_->input.pointer);
    const auto pixels = state_->pixels;
    return {sizeof(rw_sr_frame), RW_ABI_VERSION, id, epoch, predecessor,
            planes, 3*pixels, planes + 3*pixels, 2*pixels, planes + 5*pixels, pixels};
}

void* SrBridge::producer_event(ID3D12CommandQueue& queue) {
    auto& state = *state_;
    check(cudaSetDevice(state.ordinal));
    const auto value = ++state.producer_value;
    check(queue.Signal(state.producer.object.Get(), value), "Signal D3D12 producer");
    cudaExternalSemaphoreWaitParams wait{};
    wait.params.fence.value = value;
    check(cudaWaitExternalSemaphoresAsync(&state.producer.semaphore, &wait, 1, state.stream));
    check(cudaEventRecord(state.ready, state.stream));
    return state.ready;
}

void SrBridge::publish(ID3D12CommandQueue& queue, rw::sr::Job job, rw_sr_cuda_view view) {
    auto& state = *state_;
    if (state.output_job) throw std::logic_error("Previous graphics output has not retired");
    if (view.rgba_elements != state.pixels * 16) throw std::invalid_argument("Unexpected SR output size");
    check(cudaSetDevice(state.ordinal));
    state.output_job.emplace(std::move(job));
    check(cudaMemcpyAsync(state.output.pointer, view.rgba, view.rgba_elements * sizeof(float),
                          cudaMemcpyDeviceToDevice, state.stream));
    cudaExternalSemaphoreSignalParams signal{};
    signal.params.fence.value = ++state.consumer_value;
    check(cudaSignalExternalSemaphoresAsync(&state.consumer.semaphore, &signal, 1, state.stream));
    check(queue.Wait(state.consumer.object.Get(), state.consumer_value), "Wait for CUDA consumer fence");
}

void SrBridge::retire() {
    const auto completed = state_->consumer.object->GetCompletedValue();
    if (completed < state_->consumer_value || completed == std::numeric_limits<uint64_t>::max())
        throw std::logic_error("CUDA output copy has not completed");
    state_->output_job.reset();
}

}
