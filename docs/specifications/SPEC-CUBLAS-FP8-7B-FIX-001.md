# SPEC-CUBLAS-FP8-7B-FIX-001 — Root-cause + fix the cuBLAS FP8 7B Q4K gibberish

| Field | Value |
|---|---|
| Status | **PROPOSED** |
| Owner | (assigned at kickoff) |
| Created | 2026-05-22 |
| Last updated | 2026-05-22 |
| Tracks | [#1864](https://github.com/paiml/aprender/issues/1864) |
| Blocks | v0.35.0 release tag + crates.io publish cascade |
| Estimate | 2-5 days of focused work (6-stage falsifier cascade) |

## Why this is an epic, not a one-PR fix

The 2026-05-22 dogfood session surfaced `<|im_start|>` gibberish from `apr qa` Golden Output on `qwen2.5-coder-7b-instruct-q4_k_m.gguf` via the cuBLAS FP8 path. Initial investigation (1.5 hours, this date) attempted:

1. **`git bisect run`** with `apr qa` Golden Output as the oracle — identified commit `8bd4ce5a` (monorepo consolidation, 17,830 files / 13.7M insertions). Re-running the oracle on v0.31.2 (the "good" baseline from the bisect) **also produced the same gibberish** — the bisect was invalid because the test signal was non-deterministic.
2. **Layer-by-layer trace** via `cargo run --example layer_by_layer_trace --release -p aprender-serve --features cuda` — showed Layer 0 Q/K inputs already differ between CPU and cuBLAS GPU. Logit correlation 0.987 (high); linear fit `GPU ≈ 0.96 × CPU + 0.12`. Top residuals concentrated at vocab positions 15-22.
3. **Targeted-file reverts** — `cublas.rs` math mode (PEDANTIC ↔ DEFAULT ↔ TF32), `cublas_prefill/attention.rs`, `weights.rs`, `flash_decoding_graphed.rs`, `rms_norm.rs` (backward, training-only). None of these alone restored Golden Output.
4. **Dependency archaeology** — `trueno-gpu v0.4.36` was bit-identical at v0.31.2 and 8bd4ce5a (same crates.io checksum). `aprender-compute` re-export shim points are also consistent. The cuBLAS code itself didn't change.

**Conclusion**: this is not a simple regression. The cuBLAS FP8 path for `hidden_dim=3584` (Qwen2.5-Coder-7B) has likely been broken across multiple releases — including v0.34.0 currently on crates.io — and only surfaced now because:

- The wgpu path silently fell back to CPU (without #1876's multi-step gate, it shipped its own different gibberish)
- The cuBLAS path's CUDA context poisoning intermittently masks the symptom across runs
- `apr qa` Golden Output gate isn't wired into CI for the 7B teacher (per the 5-whys for #1864 — "provable contracts only verify what's *executed*")

## Falsifier cascade (per `feedback_falsifier_cascade_decomposes_magnitude`)

Six PRs, ~50-200 LOC each, each one a falsifiable hypothesis test. Each landing is a §N spec amendment with empirical evidence.

### Stage A — Deterministic reproducer

**Hypothesis**: The gibberish is intermittent; bisection requires it to be deterministic first.

**Falsifier**: `cargo run --example cublas_fp8_7b_reproducer --release -p aprender-serve --features cuda` produces bit-identical output on 5 consecutive runs (either all-gibberish or all-correct).

**Deliverable**:
- `crates/aprender-serve/examples/cublas_fp8_7b_reproducer.rs` — minimal harness: load 7B Q4K, run single forward step on cuBLAS, dump logits to JSON, compare against a golden reference.
- Environment audit: identify all sources of non-determinism (CUDA stream order, JIT cache warmup, FP8 weight cache LRU). Document and pin them.
- Contract: `contracts/cublas-fp8-7b-determinism-v1.yaml` with `FALSIFY-CUBLAS-FP8-DET-001` (5 consecutive runs bit-identical).

**Out-of-scope**: actually fixing the gibberish. This stage only makes the bug visible.

### Stage B — Per-layer parity instrumentation

**Hypothesis**: The existing `GPU_DEBUG_ALL_LAYERS=1` only dumps Layer 0 input/output. To find the first divergent layer we need every layer's `(rms_norm_out, q, k, v, attn_out, ffn_out, residual)` checksummed.

**Falsifier**: `APR_PER_LAYER_PARITY_DUMP=1 cublas_fp8_7b_reproducer` writes one JSON per layer with `(layer_idx, stage, cpu_checksum, gpu_checksum, cosine, max_abs_diff)`. The 28 outputs make divergence point trivially greppable.

**Deliverable**:
- New env-var-gated instrumentation in `crates/aprender-serve/src/cuda/executor/layers/cublas_prefill/` and the matching CPU forward in `crates/aprender-serve/src/gguf/runtime.rs`.
- Per-layer dump emits to `$APR_PER_LAYER_PARITY_DIR` (default `./trace-tensors/<run_id>/`).
- Contract: extend `apr-cpu-vs-gpu-output-parity-v1` with `per_layer_parity_dump` equation.

**Output**: a single sentence — "First layer where cosine drops below 0.99: layer K, stage S".

### Stage C — Embed lookup parity

**Hypothesis**: The Layer 0 K/Q inputs already differ (per the 2026-05-22 trace), so the divergence may originate in the embed lookup itself (token_id → embedding vector). 7B has `embed_tokens.weight = [152064, 3584]` in Q4_K which dequantizes to F32 = 2180 MB — at the wgpu max-binding limit but well under cuBLAS limits.

**Falsifier**: For `token_id = 791` (the example's probe), `cpu_embed[..16] == gpu_embed[..16]` bit-exactly (or within FP8 quantization error if the path uses FP8 embed).

**Deliverable**:
- Add a sub-target to the Stage A reproducer: `--check-embed` mode that compares the first 16 elements of the embedding lookup on both backends.
- If embed differs → fix here (likely a Q4_K dequant offset or layout issue).
- If embed matches → cascade continues to Stage D.

### Stage D — Pre-attention RMSNorm parity

**Hypothesis**: After embed, the first operation is RMSNorm. With CPU `1e-6` and GPU `1e-5` (the PMAT-698n bug class from prior cascades), eps-mismatch could explain Q/K divergence.

**Falsifier**: `cpu_rms_norm(embed[token_id]) == gpu_rms_norm(embed[token_id])` within 1 ULP for all 3584 hidden dims.

**Deliverable**:
- Test in `crates/aprender-serve/tests/cublas_fp8_rms_norm_parity.rs`.
- Verify both backends use the same `rms_norm_eps` from the GGUF metadata.
- If RMSNorm differs → fix the eps/algorithm.
- If RMSNorm matches → cascade continues to Stage E.

### Stage E — Q/K/V FP8 matmul parity

**Hypothesis**: The cuBLASLt FP8 GEMM at shape `(3584, batch, 3584)` produces wrong results for Qwen2's specific weight layout. This is the most likely candidate (per the layer-0 trace showing Q/K divergence after RMSNorm + projection).

**Falsifier**: For a fixed `rms_normed_input`, `Q_cpu = X @ Wq` (F32 matmul) and `Q_cublaslt_fp8 = X @ Wq` (FP8 matmul) agree within 1% relative error across all 3584 output dims.

**Deliverable**:
- Test in `crates/aprender-serve/tests/cublas_fp8_qkv_parity.rs`.
- Compare against `cublasLtMatmul` reference invocation with known inputs.
- Likely root cause space: FP8 scale calibration, accumulator precision, algorithm selection, JIT-compile cache miss reusing wrong kernel.

### Stage F — Root-cause fix + contract amendment

**Hypothesis**: Whatever Stages C/D/E surface as the divergence source has a known fix in either:
- `aprender-gpu/src/kernels/quantize/fp8/` (FP8 quantization)
- `aprender-serve/src/cuda/executor/layers/cublas_prefill/` (call-site setup)
- `trueno-gpu` (upstream cuBLAS driver — needs vendor PR if so)

**Falsifier**: `apr qa /home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf` reports `✓ PASS Golden Output` on **5 consecutive runs** post-fix (per Stage A's determinism contract).

**Deliverable**:
- The actual fix (commit + tests).
- Contract amendment to `apr-cpu-vs-gpu-output-parity-v1` (probably v1.7.0 → v1.8.0):
  - new equation `cublas_fp8_7b_correctness`
  - new falsifier `FALSIFY-CPU-GPU-007` LIVE-DISCHARGED with 5-run evidence
- Release v0.35.0 once F lands.

## Why v0.35.0 holds

Per `feedback_release_only_after_bug_hunt`: don't ship a release with a major-bug-just-found, even if the bug pre-existed. Users running 7B Q4K on cuBLAS today get gibberish from `apr serve` HTTP and `apr code` (which uses serve). That's load-bearing functionality that needs a real fix, not a known-issue note.

v0.34.0 already has this bug; v0.35.0 holding doesn't make things worse. The 7 PRs (#1867, #1868, #1870, #1872, #1873, #1875, #1876, #1878) continue to merge into main as individual fixes — they're net positive regardless of when v0.35.0 cuts.

## Open questions for the kickoff

1. Does the bug reproduce on Blackwell GB10 (sm_121) too, or is it sm_89-specific?
2. Does the bug reproduce on Qwen2.5-7B variants with a *different* hidden_dim (e.g., a fictional 3712-dim variant)? — would test whether `hidden=3584` is the trigger or whether it's Qwen2 7B specifically.
3. Does the bug reproduce on `apr finetune` (training) too? — would test whether cuBLAS FP8 training is also broken, not just inference.
4. What's the relationship to closed bugs #374 and #559 (both fixed for sm_121, this reproducer is sm_89)?

## Out of scope

- 30B-MoE inference (#1583, separate epic)
- wgpu kernel-level fix (covered by #1864 sub-issue once #1876 lands)
- Apollo/jetson cross-platform validation (separate hardware-availability work)

## References

- [#1864](https://github.com/paiml/aprender/issues/1864) — tracking issue
- PR [#1876](https://github.com/paiml/aprender/pull/1876) — wgpu side fix (multi-step parity gate)
- `contracts/apr-cpu-vs-gpu-output-parity-v1.yaml` — current parity contract
- `crates/aprender-serve/examples/layer_by_layer_trace.rs` — existing diagnostic
- `memory/feedback_falsifier_cascade_decomposes_magnitude.md` — 6-stage cascade pattern
- `memory/feedback_release_only_after_bug_hunt.md` — release-hold justification
- `memory/feedback_test_methodology_can_fake_bugs.md` — why the initial bisect was invalid
