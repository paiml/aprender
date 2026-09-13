# L0-1b — P3 quorum (2026-09-06, agy 1.1.27, three families, review-only)

Delegate: paiml-agy-delegate (opus) on an exported tree of 9cb834956; lanes ran in `--sandbox`, writes=false.
Conversations: 9c765895-7196-43c5-b4c6-5872a80794dc · 2226e648-c466-4ab3-afb6-f63dda1c2497 · cfb22d4f-d0cf-41d1-bf33-4cb698dacf1f.

| lane | model (family) | turns | claim 1 (instrument faithful) | claim 2 (CPU Q8_K is the wrong side) | claim 3 (fix direction) | verdict |
|---|---|---|---|---|---|---|
| 1 | gemini-3.1-pro-high (Google) | 1 | SOUND | SOUND | per-32 rescale | implement-with-changes: rename — one scale per 32 is Q8_0, not Q8_K (types.rs:48) |
| 2 | claude-sonnet-4-6 (Anthropic; opus-4-6 returned 503) | 1 | SOUND w/ caveats | SOUND w/ caveats | per-32 rescale | implement-with-changes: quantize_into → 8 sub-scales; fused_q4k_q8k_dot + every SIMD variant; bsum_precompute; rename; re-run acceptance |
| 3 | gpt-oss-120b-medium (OpenAI-class) | 2 | SOUND | SOUND | per-32 rescale | implement-as-written — DISCOUNTED: no alternatives, no falsifier, cited a non-existent receipt path |

**Consensus.** 3/3 endorse fixing the CPU Q4_K × Q8 activation path rather than an f32 fallback or moving the gate's
reference; 0/3 do-not-implement. The delegate re-verified the pivotal call chain in the tree
(ffn_block.rs → fused_matmul_into.rs:388-411 → matmul_fused.rs:288 → parallel_k.rs:190/225/288 → quantize_activations_q8k_into
mod.rs:213 `as_chunks::<256>`): the mechanism is LIVE on this model.

**Dissent (lane 2, upheld).** "per-32 matches the GPU's numerics" is falsified by the probe table itself: per-32 gives −1116.3
(2.25 % from the f64 truth −1142.0) while the GPU tap is −1136.5 (0.48 %). The GPU's activation format is NOT established by
this PR; the fix is justified as "closer to the f64 truth", never as "matching the GPU". → docs/audits/l0-1b-arms.md reworded.

**Delegate findings (folded).**
1. The shipped JSON lacked the `lm_head`/`final_norm` rows (first-run artefact) → regenerated with the current binary
   (283 rows; 1.5B lm_head 0.950827, 7B 0.998607) in `.pr/L0-1b/step0/`.
2. A zero-code CPU-side arm exists: `DIRECT_FP32_GEMV=1` (parallel_k.rs:258, f32 activations in
   `fused_q4k_parallel_matvec_into` only) → RUN (arm A7): **1.5B min cosine 0.950827 → 0.999896, 0 positions < 0.98, 0 argmax
   mismatches; per-op names NO op (layer 26 ≥ 0.9992)**. The strongest falsifier of claim 2, on the shipped binary.
3. Arms-doc probe path corrected (`.pr/L0-1b/step0/probe26.py`).
4. Open: the probe models the kernel (dequant → f64) rather than calling `fused_q4k_q8k_dot`; A7 makes the point end to end,
   the direct-kernel call remains a worthwhile unit test for step 2.

**Orchestrator fold on claim 3 (measured after the quorum; recorded as the decided recommendation with the quorum's dissent).**

| candidate | neuron 2908 error vs f64 | cost | blast radius |
|---|---|---|---|
| current per-256 Q8_K | −12.75 % | — | — |
| per-32 scales (quorum) | −2.25 % | ≈ 0 (8 scales/256) | scalar + AVX2 + AVX-512 VNNI + bsums + horizontal + fused_q4k + Q5K/Q6K twins; rename |
| f32 activations (A7 / DIRECT_FP32_GEMV) | ≈ 0 (0.999896 end to end) | −17 % CPU decode (PMAT-305 basis: 25.4 vs 30.8 tok/s) | one switch |
| outlier split (zero |x| > τ·rms before Q8_K, add the ≤ 3 outlier terms exactly) | −0.06 % (τ ∈ [4, 8]) | ≈ 0 when no outlier; ≤ 2× when split | quantizer + drivers; a threshold with basis |
| two-pass residual Q8_K | ≈ per-32 | 2× always | none | 

Criterion measurements for the split (this tree, 78 positions): global max|x|/rms does NOT separate (ffn_swigl median 78);
per-256-block max/rms saturates at 16.0 on ordinary blocks too; per-block max/second-largest is the candidate (crushed block
at layer 26 = 20.1) — its distribution is in the arms doc.

**Decided:** step 2 implements the CPU-side accuracy fix; the exact mechanism is chosen by the max/second-largest table:
clean separation → outlier split with a measured threshold (basis = that table); else per-32 scales as the quorum decided,
with lane 2's five changes. Revert falsifier either way: `.pr/L0-1b/accept.sh` legs A5/A6 (1.5B names layer 26 again with
lm_head ≈ 0.9508; 7B clean) — POP-F-003.
