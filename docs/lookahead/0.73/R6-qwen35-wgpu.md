# Row 6: Qwen3.5 on wgpu (a wgpu backend in `Qwen35Session`)

Read at origin/main 4098007133. `S:` = `crates/aprender-serve/src/gguf/inference/forward/qwen35_session.rs`,
`Q:` = `.../forward/forward_qwen35.rs`, `K:` = `crates/aprender-gpu/src/kernels/gdn/`,
`W:` = `crates/aprender-compute/src/backends/gpu/device/linalg/wgsl_forward.rs`. No build.
The landing-map cites for row 6 (inference_result.rs:365-385, server.rs:172-180,
run_entry.rs:331 `reconcile_accelerator`) hold at 4098007133.

## Today
- The route is a pure function of two bits: `qwen35_route(no_gpu, cuda_backend)` (Q:1835)
  returns `Gpu` (CUDA) or `Cpu`. There is no wgpu input, so a wgpu build always lands on
  `Cpu(NoCudaBackend)` and prints "this binary has no CUDA backend" (Q:1870).
- `Backend` (S:223) and `Checkpoint` (S:216) have a `cuda`-gated `Gpu` arm and a `Cpu` arm.
  Every device method is `cuda`-gated: `try_forward` (S:451), `try_batched_prefill`
  (S:480), `validate_gpu_once` (S:525, the F2 hybrid guard), `forward_one` (S:550).
- `WgslForwardPass::forward_layer` (W:819) is one dense layer: RMSNorm, QKV, host
  attention, FFN, keyed by a `layer_prefix`. Qwen3.5 has two layer kinds (Gated DeltaNet
  and gated full attention) and neither fits it.
- The device kernels exist for CUDA only: K:mod.rs lists six Gated DeltaNet kernels
  (conv1d+SiLU, per-head L2 norm, gates, delta-rule recurrence, gated RMSNorm, sigmoid
  gate) and three for the 256-wide gated attention (q/gate split, partial NeoX RoPE,
  decode attention 256). Each is specified by a CPU function in Q: and tested against a
  verbatim port of it. No WGSL counterpart exists (no gated-delta or conv1d shader under
  aprender-compute).

## Finding R6-1: a third route, not a cfg flip
`qwen35_route` must take the wgpu capability (feature + a usable adapter, after R13's
software-adapter refusal) and return `Wgpu`; the notice at Q:1870 must name the backend
actually missing. `Backend::Wgpu` and `Checkpoint::Wgpu` join the enums. A forced
`--backend wgpu` that lands on the CPU stays exit 14 (run) and a refusal (serve
server.rs:178), unchanged.

## Finding R6-2: the F2 guard is CUDA-only, and would become a fifth probe copy
`validate_gpu_once` (S:525) compares the device forward with the CPU before the device
may serve. A wgpu arm needs the same guard. Writing it a third time repeats R10-3: use
R10's shared probe (`wgpu_parity_probe`) with the Qwen3.5 forwards plugged in, so the
guard, leg A and E1 measure one function.

## Finding R6-3: a separate forward, not `forward_layer`
Qwen3.5 on wgpu is its own pass type (the CUDA side is its own `Qwen35CudaModel` too),
reusing W:'s RMSNorm and GEMV pipelines and adding nine WGSL kernels ported from K:,
each tested against the same CPU function the CUDA kernel was. Not a branch inside
`forward_layer`: that would put a layer-kind switch in the dense path P4 is repairing.

## Finding R6-4: the CUDA tolerances do not carry over
K:mod.rs says its kernels use `ex2.approx`/`lg2.approx` (~2 ulp) and sizes its test
tolerances for that. WGSL `exp`/`log` precision is set by the platform (Metal fast
math, Vulkan drivers) [U]. Each WGSL kernel's tolerance against the fp32 CPU function
is measured per host (C1, C5b), not copied from K:.

## Finding R6-5: the recurrent state must stay on the device
Per DeltaNet layer the state is `num_v_heads * head_v_dim * head_k_dim` floats
(16*128*128*4 B = 1 MiB on 0.8B) plus the conv window. A host round-trip per layer per
token would repeat R12's readback cost at a larger size. State lives in device buffers
from load; `Checkpoint::Wgpu` is a device-to-device copy, and a download only when the
session checkpoints to the host.

## Finding R6-6: the RoPE theta of the attention layers [U]
The CUDA port carries partial NeoX RoPE with the model's theta. A wgpu port that reuses
W:'s RoPE meets R12-1 (`rope_theta = 1e6`, #4832). If Qwen3.5's `rope.freq_base` is not
1e6 [U, read it from the GGUF header], reuse without #4832 gives a cosine drop that the
guard will read as "wgpu inaccurate". #4832 lands before this row.

## Order
P4 (WGSL GEMV) -> row 19 (Q5_K GEMV, if the E1 Qwen3.5 file carries Q5_K [U]) -> #4832 -> R10
shared probe -> this row: route + enums, then kernels in K:mod.rs order, each with its
CPU parity test, then the session wiring and the F2 guard. `forward_one` returns logits
(S:550) and the session samples on the host, so row 17 does not gate this row; device
argmax and attention residency are row 12.

## Falsifiers
| id | claim | how |
|---|---|---|
| F6-1 | a wgpu build with an adapter routes Qwen3.5 to wgpu | `qwen35_route` unit table over (no_gpu, cuda, wgpu) has a `Wgpu` row; the notice names the missing backend |
| F6-2 | each WGSL kernel equals its CPU function | per-kernel test against the Q: function, tolerance recorded per host |
| F6-3 | the guard is the shared probe | a planted error in the wgpu DeltaNet output fails the guard, falls back loudly, exits 14 when forced |
| F6-4 | state stays resident | per-token host<->device byte count is O(hidden), not O(layers * state) |
| F6-5 | theta comes from the model | a GGUF with a changed `rope.freq_base` changes wgpu and CPU logits the same way |
| F6-6 | E1 WGPU leg runs on Qwen3.5 | `apr run --backend wgpu` on the E1 Qwen3.5 file on C1 and C5b exits 0 with `used_gpu` and an adapter block |

## Open
- RQ-8 (above P5 or not) is unchanged by this read.
- Batched prefill on wgpu (the CUDA side has `try_batched_prefill`, S:480): v1 can
  prefill per token if leg A (R10, Qwen3.5 arm) measures the same path [U].
