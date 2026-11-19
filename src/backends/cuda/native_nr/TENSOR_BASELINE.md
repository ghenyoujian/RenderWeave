# Native NR tensor and parameter baseline

This code owns CUDA allocations and a nonblocking stream, calls cuBLAS directly,
and implements explicit half/FP8 publications as CUDA kernels. It does not use
LibTorch, a Python tensor provider, a vendor NR execution binary, TF32, or INT4.
The pinned source arithmetic is the stateless `3beef793` reference contract;
agreement with original NVIDIA output remains a separate qualification.

Each Context is one execution slot. FP32 tensor allocations and a private 4 MiB
cuBLAS workspace are checked against its byte budget. CUDA context/library
persistent allocations are outside `live_bytes()` and must also be measured and
reserved by the enclosing device profile before production admission. This
counter alone is not a complete device memory upper bound.

Tensor storage retains its creating device state. The baseline synchronizes its
stream before freeing retired storage, and leaves unconfirmable allocations
quarantined and charged against budget. This is intentionally a correctness
baseline with visible synchronization cost, not a graph/multislot performance
claim. Cross-context operands are rejected.

Weights are mapped read-only from original raw-U8 safetensors records. Typed
views decode finite E4M3 and FP16 values exactly into FP32. Decoded and derived
immutable layouts are cached within each Weights object, keyed by retained
Context identity, semantic key, dtype and shape. Reusing a derived key with
different float bytes is rejected. Originals are never modified or copied to
private disk. The enclosing registration workflow must bind verified source
and parameter hashes before construction.
