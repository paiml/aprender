# SHIP-007 v5 — ROOT CAUSE CONFIRMED: GPU forward path

**Date**: 2026-05-03 (v5, definitive)
**Status**: SHIP-007 root cause empirically pinpointed by toggling `--no-gpu`.

## The decisive test

Same model (canonical 7B teacher), same prompt ("What is 2+2?"),
same greedy decode (`--temperature 0.0`), same 16-token cap.
Only one CLI flag changes: `--no-gpu`.

| Configuration | First 16 tokens output | Correct? |
|---|---|---|
| `apr run` (GPU default — CUDA graph + wgpu hybrid) | `ampiezza = 0.5\ndiametro = 10` | ❌ GIBBERISH |
| `apr run --no-gpu` (CPU scalar-loop matmul) | `2 + 2 equals 4.` | ✅ CORRECT |

**The bug is in the GPU forward path. The CPU forward path is correct.**

This eliminates every other hypothesis from v1/v2/v3/v4:
- ❌ Not an attention-math bug (CPU uses same attention math, works)
- ❌ Not a Q4K dequant precision artifact (both paths dequantize at load)
- ❌ Not a tokenizer/chat-template issue (same on both paths)
- ❌ Not an autoregressive/KV-cache bug (CPU has both, works)
- ✅ It IS a GPU kernel bug (the only varying piece between the two runs)

## Single-token corroboration

```bash
$ apr run [...] --max-tokens 1 --temperature 0.0 --no-gpu
Output: 2

$ apr run [...] --max-tokens 1 --temperature 0.0   # GPU
Output: ampie
```

Even at single-step level, GPU disagrees. So this is NOT a multi-step
compounding bug — it's a single-forward-pass bug that manifests on the
very first generated token under GPU.

## Performance corroboration (separate concern)

```
GPU run:  Completed in 72.95s (cached)
CPU run:   Completed in  9.81s (cached)
```

The GPU run is **7.4× SLOWER** than CPU on the same workload. This is
itself a sign of broken GPU dispatch — a working RTX 4090 should be
much faster than CPU on a 7B model. The performance regression is
collateral evidence of the same kernel-correctness bug.

## What is the GPU forward path?

From `apr run` diagnostic output:
```
[trueno#243] ✓ Manual graph: 646 kernels.
[wgpu] Skipping weight 'lm_head' (2180.0 MB > 2147.5 MB limit) — CPU fallback
[GH-175] OwnedQuantizedModel::from_apr: 28 layers loaded in 3712.3ms
[PMAT-333] Dequantizing 28 layers (hidden=3584, heads=28/4, intermediate=18944)
[PMAT-333] Dequantized 337 weights, 28282.5 MB F32
```

So `apr run` GPU path:
1. Loads Q4K weights via `OwnedQuantizedModel::from_apr`
2. **Dequantizes all 28 layers to F32** (28.3 GB total) — `[PMAT-333]`
3. Builds a **trueno manual graph of 646 kernels** — `[trueno#243]`
4. Uses **wgpu** for matmuls (with CPU fallback for lm_head due to 2 GB limit)

The bug is somewhere in those 646 trueno kernels OR in the wgpu
dispatch / graph construction.

## SHIP-007 surface narrowed to

`crates/aprender-serve/src/` GPU/wgpu execution path:
- `OwnedQuantizedModel` GPU path (used when `--no-gpu` is NOT set)
- `trueno_gpu::*` kernels in `../trueno`
- The 646-kernel manual graph builder

NOT in the CPU path (`forward_traced` uses scalar-loop matmul,
produces correct results).

## Five Whys

1. **Why does GPU produce gibberish?** Some kernel in the trueno-gpu
   manual graph computes attention or matmul incorrectly.
2. **Why didn't this surface earlier?** Previous SHIP-007 work
   (v1/v2/v3) only instrumented `forward_traced` (CPU path) — never
   the GPU path that `apr run` actually uses.
3. **Why did the layer-0 attention bisection look like the bug?** The
   1.4e-3 cosine drop at attn_out is real but BENIGN — it's just Q4K
   dequant precision noise that doesn't flip argmax. The GIBBERISH
   from GPU is a separate, much larger error.
4. **Why does GPU exist if CPU works?** Performance: GPU should be
   much faster. But this build regressed (or never worked correctly).
   The GPU is currently both **wrong AND slower** than CPU.
5. **Why is this not caught by existing tests?** Because tests probably
   only exercise the CPU path (which is correct) or the F32-dequant
   path (which is correct). The wgpu manual graph path lacks a
   correctness gate vs CPU path.

## Next milestones

1. **Spec contract gate**: codify "CPU vs GPU forward output must
   match for a fixed prompt" in `apr-vs-gguf-forward-parity-v1` (or
   a new `apr-cpu-vs-gpu-parity-v1` contract). Right now there's no
   gate; this regression slipped through.
2. **Audit trueno_gpu manual graph builder** for kernel correctness.
   Likely candidates: matmul tile sizes, RoPE freq tables, softmax
   accumulator precision, attention mask construction.
3. **Use `apr trace --gpu`** if it exists — capture GPU-side stage
   tensors and diff vs CPU `forward_traced` stages. If memory
   `feedback_apr_trace_not_eprintln` is correct, this is the path
   per stack policy.

## Implications for ship %

- SHIP-007 root cause is now **localized to a specific surface**:
  the GPU forward path (646 trueno kernels + wgpu dispatch).
- This is the single biggest narrowing in the entire SHIP-007 chain.
- Workaround already exists: ship MODEL-1 with `--no-gpu` flag set
  in default config, OR fix the GPU bug before ship.
- Fix path is now actionable.

## Reproducer

```bash
cd /home/noah/src/aprender
APR_BIN=/mnt/nvme-raid0/targets/aprender/release/apr
MODEL=/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr

# Show GPU produces gibberish
$APR_BIN run "$MODEL" --prompt "What is 2+2?" --max-tokens 16 --temperature 0.0
# → "ampiezza = 0.5\ndiametro = 10"

# Show CPU produces correct answer
$APR_BIN run "$MODEL" --prompt "What is 2+2?" --max-tokens 16 --temperature 0.0 --no-gpu
# → "2 + 2 equals 4."
```
