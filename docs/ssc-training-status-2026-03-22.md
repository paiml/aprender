# SSC training infrastructure status (snapshot 2026-03-22, STALE)

Moved out of CLAUDE.md in the 2026-10-16 context trim. Re-verify before acting.

This block has not been re-measured in ~5 months and one of its premises no longer
holds: it points at "trueno 0.4.36" as an external crate, but since APR-MONO trueno is
in-tree as `crates/aprender-compute` and has no independent version to wait on.

- **SSC canary eval**: 90% accuracy, SHIP gate PASS — classifier ready to ship
- **entrenar cuBLAS integration**: GEMM parity verified between CPU and GPU paths
- **Blackwell (GB10) training**: Blocked by JIT pre-warming bug in custom PTX kernels. Must use fused NF4 kernel path until trueno 0.4.36 ships with pre-compiled kernels
- **apr-cli inference NOT affected**: `apr run` / `apr serve` use cuBLAS (GPU) or trueno SIMD (CPU) — pre-compiled, no custom PTX involved
- **Trained model (LoRA adapter)**: Architecture-independent safetensors — works on any GPU or CPU via standard PEFT loading
- **Key tickets**: trueno#200 (Blackwell JIT), trueno#203 (pre-compiled kernels), entrenar#300 (cuBLAS backward)
