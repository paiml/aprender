# Inference-stack adversarial audit (2026-06-14)

Systematic adversarial bug-hunt over aprender's three highest-stakes, load-bearing
subsystems. Each hunt: parallel finders (one per area, read-only Explore agents) →
every finding pipelined into a **skeptic verifier** that defaults to *refute* (most
"bugs" in 25k-test code are false positives) → only adversarially-confirmed bugs kept.
Confirmed bugs were root-caused, fixed, contracted, and mutation-verified.

## Coverage + results

| Subsystem | Areas hunted | Findings | Confirmed real | Outcome |
|-----------|-------------|----------|----------------|---------|
| **realizar inference hot path** | decode loop, KV cache, sampling, dequant kernels, attention/GQA | several | **2** | **FIXED** — PMAT-749 (GQA serve crash) + PMAT-750 (truncation fail-closed) |
| **format-conversion / quant-layout** (LAYOUT-001/002) | GGUF↔APR transpose, layout-contract gating, SafeTensors↔APR, dtype/quant round-trip | 7 | **0** | CLEAN — all 7 refuted; the "100+ historical fixes" guards hold |
| **training path** | AdamW/SGD update, LR scheduler, cross-entropy + grad-accumulation, autograd backward | 3 | **0** | CLEAN — math verified vs canonical formulas |

## The two confirmed inference bugs (both shipped)
- **PMAT-749** (`apr serve` GQA crash): `adaptive_attention_with_cache` routed *all*
  models to MHA-only cache kernels that stride the KV cache by `q_dim`; GQA models
  (TinyLlama/Llama-2-3/Mistral/Qwen2) have a `[seq, kv_dim]` cache, so at
  `head ≥ num_kv_heads` the slice ran past `kv_dim` → index-OOB panic at ~64 tokens.
  Fixed: route GQA to `attention_with_cache_gqa`. Mutation-verified.
- **PMAT-750** (truncated-model garbage): `from_ref_with_dims` silently zeroed a tensor
  whose bytes ran past the file, so a corrupt/incomplete GGUF loaded with a dead weight
  and produced garbage on `apr run`/`apr serve`. Fixed: `is_truncated()` +
  load-time `validate_quantized_tensors()` → fail closed at load.

## Conclusion
The two clean subsystems (conversion/layout, training) were each filtered by adversarial
skeptic verification — the false positives the finders raised did not survive scrutiny,
which is the expected result for mature, heavily-guarded code. The inference hot path —
the most complex and recently-evolved subsystem — held the real defects, now fixed. The
load-bearing inference/training/conversion stack is adversarially audited as of this date.
