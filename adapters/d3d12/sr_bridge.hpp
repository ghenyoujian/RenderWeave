#pragma once

#include <windows.h>
#include <d3d12.h>
#include <rw/rw_sr.hpp>

namespace rw::d3d12 {

// One shared input/output pair for one serial SR frame chain. The calling
// renderer owns resource transitions and retires the pair after graphics use.
class SrBridge {
public:
    SrBridge(ID3D12Device& device, uint32_t width, uint32_t height);
    ~SrBridge();
    SrBridge(const SrBridge&) = delete;
    SrBridge& operator=(const SrBridge&) = delete;

    int cuda_device() const;
    ID3D12Resource* input() const;
    ID3D12Resource* output() const;
    uint64_t allocation_bytes() const;

    // Packed contiguous float32 planes are RGB, current-to-previous XY
    // motion in input pixels, and forward depth. Color and depth use [0,1].
    rw_sr_frame frame(uint64_t id, uint64_t epoch, uint64_t predecessor) const;

    // Call after submitting the producer commands, with the buffer available
    // for CUDA access. Pass the returned cudaEvent_t to Session::submit_cuda.
    void* producer_event(ID3D12CommandQueue& queue);

    // Enqueue the copy into the shared output and a graphics queue wait for
    // that copy. The view must belong to the supplied completed job.
    void publish(ID3D12CommandQueue& queue, rw::sr::Job job, rw_sr_cuda_view view);

    // Called after graphics finishes consuming output(). The job and the
    // shared pair become available for the next frame.
    void retire();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}
