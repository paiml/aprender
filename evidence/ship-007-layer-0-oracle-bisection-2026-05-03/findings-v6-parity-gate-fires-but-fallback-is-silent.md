# SHIP-007 v6 — parity_gate IS already firing on .apr GPU init; the gap is silent fallback

**Date**: 2026-05-03 (v6, refines v1.0.0 of `apr-cpu-vs-gpu-output-parity-v1`)
**Status**: Empirical correction of the contract's PARTIAL_ALGORITHM_LEVEL
algorithm_evidence. The v1.0.0 contract's claim that "no gate runs for the
trueno-graph .apr load path" was wrong. Live `apr` shows the gate IS wired and
IS rejecting the canonical 7B teacher; the user-visible problem is that the
rejection is verbose-only and the wgpu fallback then ships its own gibberish.

## What v1.0.0 of the contract claimed (incorrect)

> Existing parity_gate covers only the `gguf::cuda::OwnedQuantizedModelCuda`
> path used by `apr parity` and `apr run --force-gpu`; the default `.apr` load
> path (trueno manual graph, 646 kernels) has NO gate and produces gibberish
> silently.

## What live `apr` actually does today

The path for default `apr run <model.apr>`:

1. `apr-cli::commands::run_entry::run(no_gpu=false)`
2. `realizar::run_inference(config)` → `run_apr_inference(config, prepared)`
3. `try_apr_cuda_inference(config, ...)` (gguf_gpu_generate.rs:522)
4. `load_apr_cuda_model(model_path, verbose)` (gguf_gpu_generate.rs:461) calls
   `OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048)` at line 487
5. `with_max_seq_len` (mod.rs:285) ends in `preload_and_verify()` (mod.rs:434)
6. `preload_and_verify` runs `parity_gate(&mut self)` at mod.rs:268-279

So the gate IS exercised. The contract was wrong.

## Live smoke result on canonical 7B teacher (2026-05-03)

```bash
APR_BIN=/mnt/nvme-raid0/targets/aprender/release/apr   # apr 0.31.2 (8d1d9feb1)
MODEL=/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr

# Without --verbose: silent gibberish
$APR_BIN run "$MODEL" --prompt "What is 2+2?" --max-tokens 8 --temperature 0.0
# → "ampiezza = 10\nampie"     (NO [PARITY-GATE] log line)

# With --verbose: the gate's failure surfaces
$APR_BIN -v run "$MODEL" --prompt "What is 2+2?" --max-tokens 4 --temperature 0.0
# Stderr (filtered):
#   [GH-181] Workspace reinit failed (non-fatal): GPU memory allocation failed:
#     CUDA driver error: CUDA_ERROR_ILLEGAL_ADDRESS (code: 700)
#   Backend: CPU (GPU unavailable: Inference error: PARITY-GATE: GPU forward
#     failed: Operation 'forward_gpu_resident' not supported:
#     forward_all_layers_gpu_to_logits_graphed failed: GPU memory allocation
#     failed: CUDA driver error: CUDA_ERROR_ILLEGAL_ADDRESS (code: 700))
#   Backend: wgpu (Vulkan)
#   [wgpu] Skipping weight 'lm_head' (2180.0 MB > 2147.5 MB limit) — CPU fallback
```

So the gate fires, errors out (its OWN GPU forward dies with ILLEGAL_ADDRESS),
and `load_apr_cuda_model` returns None. `try_apr_cuda_inference` falls through
to `try_apr_wgpu_inference` (gguf_gpu_generate.rs:266). wgpu loads the same
model fresh and produces "ampiezza = 10\nampie" — its own gibberish, unrelated
to the CUDA failure.

## Why the user sees silent gibberish

`gguf_gpu_generate.rs:487-489` (pre-fix):

```rust
let cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048).map_err(|e| {
    if verbose { eprintln!("Backend: CPU (GPU unavailable: {})", e); }
}).ok()?;
```

The `if verbose` gate means non-verbose runs swallow the error. Combined with
the wgpu path producing gibberish, the user sees no signal that GPU was
rejected — they just see garbage output.

## The fix landed in this PR

`gguf_gpu_generate.rs:487-494` now logs unconditionally with the contract tag:

```rust
let cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048).map_err(|e| {
    eprintln!("[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected, attempting fallback: {}", e);
}).ok()?;
```

So default-mode `apr run` on the canonical teacher will now stderr-emit:

```
[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected, attempting fallback:
  Inference error: PARITY-GATE: GPU forward failed: ...CUDA_ERROR_ILLEGAL_ADDRESS...
```

This is contract option (a): clear PARITY-GATE FAILED-class message, no
silence. Output may still be wgpu-gibberish until a follow-up adds a parity
gate to the wgpu path (FALSIFY-CPU-GPU-005, deferred), but the user now has
the diagnostic signal to know `--no-gpu` is the working path.

## Five Whys

1. **Why was the contract wrong about gate wiring?** The contract author
   inferred the gap from the comment at apr_inference.rs:46 ("Both AprF32ToGpuAdapter
   and forward_token_apr_q4k produce garbage on GPU") plus knowing that the
   gguf::cuda gate exists. They missed that `try_apr_cuda_inference` already
   funnels the .apr path through `OwnedQuantizedModelCuda::with_max_seq_len`,
   which inherits the gate.
2. **Why didn't the gate stop the gibberish then?** The gate does stop the
   CUDA path (returns Err). The fallback path (wgpu) has its own SHIP-007 bug
   that produces independent gibberish.
3. **Why was the fallback silent?** The `Backend: CPU (GPU unavailable: ...)`
   eprintln was wrapped in `if verbose`. The `verbose` parameter mirrors the
   global `--verbose` flag, which is off by default.
4. **Why was the eprintln verbose-gated?** The pattern was inherited from
   adjacent map_err handlers for non-fatal load errors (MappedAprModel /
   OwnedQuantizedModel). For those, verbose-only is appropriate. For
   GPU-init-failure → backend-fallback it is not — it changes which kernel
   path serves the user's tokens.
5. **Why didn't tests catch this?** `apr parity --assert` exits non-zero on
   cosine < 0.99, but `apr run` has no equivalent assertion. The
   `validate_gpu_first_token` call in `try_apr_cuda_inference` is also
   verbose-gated (`[GH-480] F2 validation FAILED ...`).

## Implications for ship %

- MODEL-1 ship is unblocked via `--no-gpu` (already working, 9.81s on RTX 4090
  via CPU FP16). v6 doesn't change MODEL-1 status.
- The parity contract status flips from "gate not wired" → "gate wired AND
  failure is now user-visible". This is a strict improvement to jidoka
  (stop-the-line) hygiene.
- Follow-up FALSIFY-CPU-GPU-005 (deferred): wgpu also needs a parity gate, so
  that fall-through-to-wgpu doesn't ship gibberish either. Tracked but not in
  this PR.
- Root-cause GPU kernel audit (v5's "audit trueno_gpu manual graph builder")
  is independent of this fix and proceeds on its own track.

## Reproducer

```bash
cd /home/noah/src/aprender
APR_BIN=/mnt/nvme-raid0/targets/aprender/release/apr     # rebuild after this PR lands
MODEL=/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr

# After the fix, this MUST emit the contract tag on stderr (without -v):
$APR_BIN run "$MODEL" --prompt "What is 2+2?" --max-tokens 4 --temperature 0.0 2>&1 \
  | grep "apr-cpu-vs-gpu-output-parity-v1"
# expected: [apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected, attempting fallback: ...
```
