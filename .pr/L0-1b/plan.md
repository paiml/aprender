# L0-1b — P0/P1 (2026-09-06): the per-op table, then bisection by controls, then the fix

## P0 — what the tree can already do (cited)
- Per-position CPU-vs-GPU logits: `crates/aprender-serve/src/gguf/parity.rs::check_parity(tokens)` → `ParityResult` per position (`apr parity --json`, `commands/parity_03.rs`). Logits only; no per-op view.
- Stage dump plan: `inference_trace/save_tensor_plan.rs::SaveTensorPlan::should_save(stage, layer)` with `SaveTensorStage` covering every dense stage (Embedding, AttnNorm, QkvMatmul, QkvBias, Q/KPostRope, AttnScores, AttnSoftmax, Attention, AttnOut, PostAttnResidual, FfnNorm, FfnGate, FfnUp, FfnSilu, FfnSwigl, FfnOut, PostFfnResidual, FinalNorm, LmHead). `gpu_stage_dump.rs::maybe_dump_host_buffer` writes host buffers under `APR_GPU_STAGE_DUMP`.
- GPU dump sites: `gguf/cuda/uses.rs:141` (Embedding) and `:180` (LmHead) only; the graphed all-layers forward (`forward_all_layers_gpu_to_logits_graphed`) exposes no per-layer host buffer. The CPU dense forward has NO `should_save` sites (only the MoE traced forward does).
- Controls that exist: `executor.force_high_precision_ffn()` (unfused FFN); `gpu_profile.rs:238` `auto_q4k` → `Mwv` (fused gate+up OFF by default — measured fact to re-state in the table, not a lane verdict); FP8 cuBLASLt path ([PMAT-082], env/feature? — find its switch); fused attention switch; graph construction ([trueno#243], the graphed vs non-graphed forward).
- Known-good pair 7B@lambda min 0.9986 ×5; known-bad 1.5B@lambda 0.9508 ×5 at position 0 (token 785); deterministic.

## Step 0 — the instrument (`apr parity <model> --per-op`)
Design: a non-graphed, layer-by-layer GPU forward that mirrors the CPU forward stage by stage for the SAME token at the SAME position, dumping (or holding in memory) every `SaveTensorStage` on both sides, over ≥ 64 positions; per (stage, layer) compute cosine and max_abs across positions; print the table {op, layer, cosine_min, max_abs, positions} and name the FIRST (stage, layer) whose cosine falls under the threshold — exit 0 with the table even on failure; the admission gate is bypassed INTERNALLY (the model is constructed without the gate for this command only), never via SKIP_PARITY_GATE.
Work: (a) CPU dense forward gains `should_save` points per stage (mirror the MoE traced forward's pattern in `forward/core.rs`/`attention.rs`/`ffn_block.rs`); (b) GPU: a `forward_single_traced` that runs layer by layer through the existing per-layer kernels with host read-backs per stage (slow, correctness-only); (c) `scripts/parity_per_op.py` compares the two dump trees → the table; (d) `apr parity --per-op` wraps (a)+(b)+(c). RED test: on the 1.5B the table names a first diverging op (≠ LmHead alone); on the 7B every op ≥ threshold.
Hosts: lambda (excepted) for development; gx10 through fleet-verify.

## Step 1 — bisection by controls (`docs/audits/l0-1b-arms.md`)
| arm | control | expected if the mechanism | measured cosine (1.5B pos 0) |
|---|---|---|---|
| baseline | default config | 0.9508 | |
| fused FFN off | `force_high_precision_ffn()` | — (already off by default per gpu_profile.rs:238; re-measure) | |
| fused attention off | (switch to find) | | |
| FP8 cuBLASLt off | [PMAT-082] switch | | |
| graph construction off | non-graphed forward ([trueno#243]) | | |
The arm that flips cosine names the mechanism; the 1.5B/7B asymmetry is answered by the per-op table.

## Step 2 — the fix
Revert → 1.5B RED, 7B GREEN; m=1 stream byte-identical on every other manifest model (CF-4).

## Andon
Two sessions without a named op → STOP with the table.
