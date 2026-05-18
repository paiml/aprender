# Audit: Q4_K shape-swap impact on v0.33.0-and-earlier Q4_K artifacts

**Document ID:** AUDIT-Q4K-SHAPE-001
**Version:** 1.1.0 (empirically falsified the original concern; benign for 256-divisible-both-dims case)
**Status:** Live — closes PMAT-690 P3-C-prep defect 4 (task #110)
**Parent:** [SPEC-HF-PUBLISH-001](../aprender-train/model-hf-publish-pipeline-spec.md), [ship-model-2-spec.md §84](../aprender-train/ship-model-2-spec.md)
**Trigger:** PR #1771 (v0.34.0, 2026-05-18) fixed `quantize_q4_k_matrix` to receive APR-native shape `[rows=out, cols=in=K]` instead of the swapped `[K, out]`. The swap had been in production since the function was first called; we need to know whether shipped artifacts produced under the pre-fix path are usable.

## TL;DR

**Already-shipped Q4_K artifacts produced before v0.34.0 are BIT-EQUIVALENT to a hypothetical post-v0.34.0 re-export** whenever both weight-tensor dims are 256-divisible. No re-export needed for correctness. Confirmed by the in-tree falsification test `audit_q4k_shape_swap_byte_identical_when_both_dims_divisible` (passes 2026-05-18) which proves the pre-fix `quantize_q4_k_matrix(data, [a, b])` and post-fix `quantize_q4_k_matrix(data, [b, a])` produce **byte-identical** output on the 256-divisible-both-dims case.

The defect-3 fix only matters when the inner dim K is NOT 256-divisible (e.g., Qwen2 0.5B `hidden=896`) — and that case is independently caught by the defect-2 K-divisibility fallback which forces F32 instead of quantizing. So:
- **Qwen2 0.5B** (hidden=896, NOT divisible): defect-2 F32 fallback → no Q4_K applied → defect-3 irrelevant.
- **Qwen2 1.5B** (hidden=1536, divisible): defect-3 byte-identical → pre-fix artifact = post-fix artifact.
- **Qwen2 7B** (hidden=3584, divisible): defect-3 byte-identical → pre-fix artifact = post-fix artifact.

This audit was triggered by an a-priori concern that the pre-fix swap might silently degrade inference quality on 1.5B/7B shipped artifacts. The empirical evidence falsified that concern.

## Mechanism

`quantize_q4_k_matrix(data, shape)` interprets `shape[0]` as `rows` and `shape[1]` as `cols`, then iterates:

```rust
for row_idx in 0..rows {
    let mut padded_row = vec![0.0f32; padded_cols];  // padded_cols = ceil(cols/256) * 256
    if row_end <= data.len() {
        padded_row[..cols].copy_from_slice(&data[row_idx * cols .. (row_idx+1) * cols]);
    }
    let row_q4k = quantize_q4_k(&padded_row);        // splits into 256-element super-blocks
    result.extend_from_slice(&row_q4k);
}
```

Each iteration:
1. Reads `cols` contiguous elements from `data` (offset `row_idx * cols`).
2. Pads to next 256-multiple if needed.
3. Quantizes via `quantize_q4_k` which produces one super-block per 256 elements of the padded slice.

### Case A: both dims 256-divisible (Qwen2 1.5B/7B)

`cols % 256 == 0` → no padding. The function consumes data in linear order, chunking into `cols / 256` super-blocks per iteration. **The data is consumed in the same linear order with the same 256-aligned chunking regardless of how rows/cols are partitioned.** Specifically:

- Call A: `[rows=256, cols=512]`, 256 iterations each producing 2 super-blocks → 512 super-blocks total, in offset order `0, 256, 512, 768, …, 130816`.
- Call B: `[rows=512, cols=256]` (the swap), 512 iterations each producing 1 super-block → 512 super-blocks total, in offset order `0, 256, 512, 768, …, 130816`.

The super-block at offset `i*256` reads the same 256 elements in both calls. The output bytes are byte-identical.

**Verified empirically** by the in-tree test `audit_q4k_shape_swap_byte_identical_when_both_dims_divisible` with a heterogeneous-per-row matrix (so any layout-sensitive divergence would be amplified). `assert_eq!(correct_bytes, buggy_bytes)` passes.

### Case B: K (inner dim) NOT 256-divisible (Qwen2 0.5B)

For Qwen2 0.5B `ffn_down.weight` with APR shape `[hidden=896, intermediate=4864]`:

- Pre-fix call: shape swapped to `[4864, 896]`. The function iterates rows=4864, cols=896. `cols % 256 = 128 ≠ 0` → pads to 1024. Each iteration produces 4 super-blocks (the 4th being half-padding). Total output: `4864 × 4 × 144 = 2,801,664 bytes`.
- llama.cpp expectation: `ne[0]=4864, ne[1]=896` → super-blocks = `(4864 × 896) / 256 = 17,024`. Bytes = `17,024 × 144 = 2,451,456`.
- **Excess: 350,208 bytes**, causing the offset drift that PR #1771 surfaced.

This is the case PR #1771's defect 2 fix forecloses by falling back to F32 when `shape[1] % 256 != 0`. With that fallback in place, the swap is moot — no Q4_K bytes are produced for the affected tensor.

## Why MODEL-1's 4.4pp HumanEval gap vs upstream is NOT this bug

Pre-finding hypothesis: `paiml/qwen2.5-coder-7b-apache-q4k-v1` at AC-SHIP1-005 = 86.59% vs upstream `Qwen/Qwen2.5-Coder-7B-Instruct` Q4_K_M ≈ 91% could mean the shape-swap was degrading our quantization.

Post-finding: the audit proves the bytes are identical to a hypothetical post-fix re-export, so the gap CANNOT be attributable to defect 3. The actual contributors are:
- **Q4_K vs Q4_K_M (mixed precision)**: llama.cpp's `_M` variant leaves attention output projections + the FFN down projection at Q6_K for sensitivity. Aprender's Q4_K is pure Q4_K throughout. This is the largest plausible contributor.
- **Different cumulative rounding paths** from different f32 → Q4_K implementations (aprender's `trueno_quant` vs llama.cpp's native quantizer). Small but non-zero.

If MODEL-1 quality matters more than v1 immutability, the path is to:
1. Add Q4_K_M (mixed precision) support to `apr export` — leave sensitive tensors at Q6_K.
2. Re-export `paiml/qwen2.5-coder-7b-apache-q4k-v1` as v1.1.0 with Q4_K_M.
3. Re-run HumanEval; expect to recover most of the 4.4pp gap.

This is a feature-add, not a bug fix. Out of scope for this audit.

## Action items (closed)

| ID | Task | Status |
|----|------|--------|
| Q4K-AUDIT-001 | Run the in-tree falsification test and pin the numerical result | **DONE** — `audit_q4k_shape_swap_byte_identical_when_both_dims_divisible` passes |
| Q4K-AUDIT-002 | Re-export `paiml/qwen2.5-coder-7b-apache-q4k-v1` post-v0.34.0 | **WITHDRAWN** — empirically unnecessary; bytes are identical |
| Q4K-AUDIT-003 | CHANGELOG entry warning external users of pre-v0.34.0 Q4_K artifacts | **REVISED** — v0.34.0 CHANGELOG should clarify the bug only affected `K % 256 != 0` tensors; 256-divisible artifacts are unaffected |

## Action items (open)

| ID | Task | Owner | Status |
|----|------|-------|--------|
| Q4K-AUDIT-004 | (Optional) Add Q4_K_M (mixed precision) export support to recover the ~4.4pp HumanEval gap on MODEL-1 | apr-export team | proposed (not blocking) |

## v0.34.0 CHANGELOG correction

The v0.34.0 CHANGELOG implies the shape-swap fix changed published byte layouts for all Qwen2 variants. This audit refutes that: the byte layouts only differ when `K % 256 != 0` (which the defect-2 fix forces to F32 fallback anyway). A small clarification PR is warranted but is non-urgent.

## References

- In-tree falsifier: `crates/aprender-core/src/format/converter/gguf_export_config.rs::q4k_divisibility_tests::audit_q4k_shape_swap_byte_identical_when_both_dims_divisible`
- PR #1771 (v0.34.0) — the original fix
- `trueno_quant::quantize_q4_k_matrix` — the function under audit
- [SPEC-HF-PUBLISH-001 §"HF API gotchas" rule 6](../aprender-train/model-hf-publish-pipeline-spec.md) — captures "Q4_K shape must be APR-native" as a load-bearing rule for future publishes; this audit clarifies that the rule prevents a future regression rather than fixing existing artifacts.
- `feedback_predict_then_verify_closes_cascade.md` (2026-05-12) — memory rule for falsification-first audits; this doc follows that pattern.

## Changelog

- **1.1.0 (2026-05-18)** — Empirically falsified the v1.0.0 concern. The shape-swap bug is benign on 256-divisible-both-dims tensors (Qwen2 1.5B/7B). The defect-2 K-divisibility fallback already handles the non-divisible case (Qwen2 0.5B). No re-export needed for any already-shipped Q4_K artifact. Closes task #110.
- **1.0.0 (2026-05-18)** — Initial publish with a-priori concern that pre-v0.34.0 Q4_K artifacts might be silently degraded. Math established but not yet empirically verified; turned out to be wrong.
