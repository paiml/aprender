# M-GPU-MOE-1.4 LIVE bisection — gx10 Blackwell GB10 — 2026-05-06

## Outcome

**FALSIFY-MOE-SUB-003 DISCHARGED.** First NaN-emitting GPU stage pinpointed to **layer 6, `moe_ffn_out`**.

## Setup

- **Host**: gx10 (Blackwell GB10, sm_120, driver 590.48.01, CUDA 13.0)
- **Repo**: aprender main @ `01732c555` (M82 cascade — falsifier hygiene amendment)
- **Model**: Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf (18 GB) at `/home/noah/.cache/pacha/models/2b88b180a790988f.gguf` (symlinked from `/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`)
- **Harness**: `cargo test -p aprender-serve --features cuda --release --test qwen3_moe_gpu_per_stage_diff falsify_moe_sub_002_cpu_gpu_traced_per_stage_diff -- --include-ignored --nocapture` (M80 PR #1524)
- **Wall time**: 23.18s (build cached + load + 48 layers × 2 forward paths)

## Per-layer cos-sim summary

```
layer | moe_router (cos / verdict)        | moe_ffn_out (cos / verdict)
------|-----------------------------------|-----------------------------------
L00   | 1.000000 MATCH                    | 0.999987 MATCH
L01   | 0.999998 MATCH                    | 0.999904 MATCH
L02   | 0.999998 MATCH                    | 0.999953 MATCH
L03   | 0.999975 MATCH                    | 0.999876 MATCH
L04   | 0.999994 MATCH                    | 0.999861 MATCH
L05   | 0.999993 MATCH                    | 0.999896 MATCH
L06   | 0.999986 MATCH                    |             NanGpu  ← FIRST NaN
L07   | 0.818903 DIVERGE                  |             NanGpu  ← NaN poisons router
L08–L47 | DIVERGE (cos 0.77 – 0.997)     |             NanGpu  ← downstream poisoning
```

## Bisection summary (harness output)

```
first DIVERGE on moe_router  : Some(7)    ← downstream of L6 NaN poison
first DIVERGE on moe_ffn_out : None       ← all DIVERGE-ish; first NaN at L6 wipes the metric
first NaN_GPU on moe_router  : None       ← router stays finite (CPU dot product F32)
first NaN_GPU on moe_ffn_out : Some(6)    ← ROOT CAUSE LOCATION
```

## Decision logic match

The harness's decision tree (printed at end of run):

> If first_NaN_GPU(moe_ffn_out) > 0 and earlier layers MATCH: bug is layer-N specific (rare).

This case fires:
- L0–L5 MATCH on moe_ffn_out (cos > 0.99986) — kernels work for layers 0-5
- L6 first NaN — something layer-6-specific triggers the overflow

## Implications for M-GPU-MOE-1.4 fix scope

Bug surface narrows to:
- `crates/aprender-serve/src/gguf/cuda/moe_ffn_forward_layer_cuda.rs` (per-layer GPU helper)
- `crates/aprender-serve/src/gguf/cuda/expert_swiglu_cuda.rs` (per-expert SwiGLU)
- `CudaExecutor::q4k_matvec` / `q6k_gemv` (custom PTX kernels in trueno)

The bug is NOT in the routing logic (router stays finite at L6). It's in the per-expert FFN computation at layer 6 specifically. Hypotheses:
1. **Numerical overflow in expert SwiGLU at L6**: layer 6's intermediate activations have a distribution that causes silu(gate) * up to overflow F16/F32 accumulator
2. **Expert weight distribution at L6**: layer 6's experts have weights that, when combined with CPU-traced layer-5 output, produce activations large enough to trigger overflow
3. **Q4K dequant accumulator at L6**: a specific Q4K block at layer 6 has a scale value that causes overflow during dequant + matmul fusion

## Architectural portability finding

This bisection ran on Blackwell GB10 (sm_120). The original M-GPU-MOE-1.3 NaN bug (PR #1493 diagnostic) was characterized on Ada RTX 4090 (sm_89). The fact that **both architectures produce NaN at the same layer** indicates:

- Bug is **algorithmic / numerical**, NOT kernel codegen
- A single fix at the bisected stage discharges both arch-specific manifestations
- Trueno custom PTX kernels (`q4k_matvec`, `q6k_gemv`) compile and run on sm_120 — trueno#200 JIT pre-warming bug did NOT block this dispatch

This is a stronger signal than expected — we initially assumed the gx10 run might fail to reproduce or hit JIT issues. It worked first-shot.

## Discharge status

| Falsifier | Pre-M82 status | Post-bisection status |
|-----------|----------------|------------------------|
| FALSIFY-MOE-SUB-001 | DISCHARGED (M82) | DISCHARGED |
| FALSIFY-MOE-SUB-002 | ALGORITHM_LEVEL_DISCHARGED (M82) | **DISCHARGED** (live `--include-ignored` ran clean on gx10) |
| FALSIFY-MOE-SUB-003 | PROPOSED | **DISCHARGED** (this run; first NaN-emitting stage = L6 moe_ffn_out) |
| FALSIFY-MOE-SUB-004 | PROPOSED | unchanged (pending M-GPU-MOE-1.4 fix PR citing L6 moe_ffn_out) |

## Next steps

1. Author contract amendment: `trace-moe-gpu-sub-stages-v1` v1.4.0 → v1.5.0 records SUB-002 + SUB-003 DISCHARGED with this evidence pointer.
2. Author contract amendment: `qwen3-moe-forward-gpu-v1` records M-GPU-MOE-1.4 bisection result.
3. M-GPU-MOE-1.4 fix PR: investigate layer-6-specific overflow in `moe_ffn_forward_layer_cuda` / `expert_swiglu_cuda`. Cites this evidence dir.
4. Companion record: M83 cross-references aprender contract amendments + M82 falsifier hygiene chain.
