"""Measured original-HT H200 execution settings; apply before runtime creation."""
import os

H200_FAST_ENV = {
    "RW_NR_BORROW_PYTHON_INPUT": "1",
    "RW_NR_CAPTURE_NODE_DETAIL": "0",
    "RW_NR_CUBLAS_WORKSPACE_MB": "4",
    "RW_NR_CUDA_GRAPH": "1",
    "RW_NR_DEFERRED_TENSOR_CACHE": "1",
    "RW_NR_GPU_FAST_MATH": "0",
    "RW_NR_GPU_HALF_MATH": "1",
    "RW_NR_GPU_HISTORY": "1",
    "RW_NR_GPU_INPUT_LAYOUT": "1",
    "RW_NR_GPU_OUTPUT_LAYOUT": "1",
    "RW_NR_HOPPER_COALESCED_INPUT": "1",
    "RW_NR_HOPPER_WIDE_MMA": "1",
    "RW_NR_INPUT_PHASE_DETAIL": "0",
    "RW_NR_PIXEL_POST_MIX": "1",
    "RW_NR_PORTABLE_INPUT_SCAN": "0",
    "RW_NR_PREPARED_CACHE": "1",
    "RW_NR_SPLIT_TENSOR_CORE": "1",
    "RW_NR_STREAM_ORDERED_MEMORY": "1",
    "RW_NR_SWIN_C32_FULL_BLOCK": "1",
    "RW_NR_SWIN_C32_FUSED_MLP_QKV": "1",
    "RW_NR_SWIN_C64_FUSED_MLP": "1",
    "RW_NR_SWIN_DIRECT_RESIDUAL": "1",
    "RW_NR_SWIN_EXPERT_FRONT": "0",
    "RW_NR_SWIN_FUSED_ALL_ATTN": "1",
    "RW_NR_SWIN_FUSED_ALL_ATTN_64": "1",
    "RW_NR_SWIN_FUSED_ATTN": "1",
    "RW_NR_SWIN_FUSED_LAYOUT": "1",
    "RW_NR_SWIN_FUSED_NORM": "1",
    "RW_NR_SWIN_MID_STRIDED_BATCH": "1",
    "RW_NR_SWIN_POINTER_BATCH": "1",
    "RW_NR_SWIN_TENSOR_CORE": "1",
    "RW_NR_SWIN_TILED_LAYOUT": "1",
    "RW_NR_TENSOR_CACHE": "1",
    "RW_NR_VIT_TENSOR_CORE": "1",
    "RW_NR_VIT_TILED_ATTN": "1",
    "RW_NR_VIT_TILED_ATTN_MULTI": "1"
}

def configure_h200_fast():
    """Enable the single-H200 CUDA input/CPU output profile measured in RW649."""
    os.environ.update(H200_FAST_ENV)
    return dict(H200_FAST_ENV)
