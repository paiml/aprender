# R5 kernel-key census (static, wt-4565 @ origin/main aca6f2d7f6, 2026-09-27 ~14:20Z)
Method: Explore agent, reading only (nothing was built or run). The 4 pivotal claims were re-checked by hand [V]. Everything else is agent-reported [A].
Key = (qtype, op, layout, precision). Paths are relative to crates/.

| op × qtype | CUDA (C0,C5) | wgpu (C1–C3) | x86 SIMD (C0 CPU) | aarch64 (C4) |
|---|---|---|---|---|
| GEMV Q4_K | gemv_dispatch.rs:46, ~20 variants | WGSL basic_ops.rs:555 (M=1) | fused_k.rs:193 AVX2, fused_q4k.rs:338 VNNI | **scalar** fused_k.rs:60 |
| GEMV Q5_K | device.rs:93 | host dequant→F32 (wa:312) | fused_q5k_q6k.rs:388 | scalar |
| GEMV Q6_K | weight.rs:77 | host dequant→F32 (wa:311) | fused_q5k_q6k.rs:118 AVX2 | scalar :15 |
| GEMV Q8_0 | weight.rs:480 | **refused** wa:339 → CPU [V] | fused_q8_0_q8_0.rs:19 AVX2 | scalar :156 |
| GEMV Q4_0 | weight.rs:626 | **refused** wa:339 → CPU [V] | fused_q4_0_q8_0.rs:342, q4_0.rs:200 VNNI | scalar :197 |
| GEMV F32 | weight.rs:692 | WGSL GEMV_SHADER | fused_matmul default [inferred] | same |
| attention (decode) | incremental_attention.rs:164 | CPU scalar (wf:1131, readback wf:850) | attention_gqa.rs:139 | scalar |
| rmsnorm | layer_norm_gpu.rs:149 | WGSL wf:115 | gguf/ops.rs:39 | scalar [inferred] |
| rope | fused_ffn.rs:360, rope_indirect.rs:8 | CPU, theta **hardcoded 1e6** (wf:943) [V] | q/rope.rs:62 AVX2/512 | scalar |
| qk-norm (Qwen3) | layer_norm_gpu.rs:303 | **absent** | scalar gguf/ops.rs:379 | scalar |
| gated-delta (Qwen3.5) | gdn_ops.rs:445 (+conv1d, l2norm, gates) | **absent** (forward_qwen35.rs:1351 no wgpu arm) | scalar | scalar |
| lm-head + argmax/sampling | reduces.rs:144 | CPU (gg:340; sampling gg:68) | AVX2 softmax | scalar |

## Findings
- **F-R5-1 [V]:** on aarch64, `detect_simd_backend()` returns `SimdBackend::Neon` (quantize/simd_backend.rs:41-44). No code dispatches on `Neon`: the only uses are Display and tests. Every quantized kernel is cfg(x86_64)-gated, so C4 runs scalar Q4K/Q6K while a trace prints "NEON". That is a verification-rule-2 label hazard. The only real NEON code is f32 (aprender-compute backends/neon). R3 is therefore a from-scratch kernel, not a port.
- **F-R5-2 [V]:** wgpu has 2 real GEMVs (Q4_K, F32). Q5K/Q6K/F16 are widened on the host. Q4_0/Q8_0 are refused at wa:339 but pass the GPU whitelist (gguf/dtype.rs:78) [A], so the run silently falls back to CPU. A wgpu cell for those qtypes measures the CPU. The R1 harness must refuse a cell whose trace does not show the wgpu forward.
- **F-R5-3 [V]:** wgpu decode measures cosine 0.955 on intel, gx10 and mini (gguf_gpu_generate.rs:104-110). That is below the 0.995 E1 leg floor. Candidate causes: rope_theta hardcoded to 1e6 (wf:943) [V], head_dim = hidden/heads (gg:185) [A], no qk-norm [A]. This supports H1. R2 scope = QK-norm + theta from metadata + head_dim + Q6K/Q8_0 WGSL + gated-delta.
- **F-R5-4 [A]:** wgpu cells are hybrids. Attention, RoPE, LM head and argmax run on the CPU, so the E2 wgpu speed ratio measures a mostly-CPU pipeline.
- **F-R5-5 [A] (H5 CONFIRMED statically):** no Rust kernel registry has both a backend and a qtype dimension. `cuda/kernel_type.rs:5 KernelType` has 115 variants, is CUDA-only, and bakes qtype/layout into variant names. `tuner/types.rs:62` is CUDA-only. `trueno::registry` lists backends, not kernels. So E5 needs KREG-001 to carry `backend` as a key field (K4, sent to kreg).
