# M32d Step 5 — per-head Q/K RMSNorm fix evidence

**Date:** 2026-05-01
**Host:** lambda-vector RTX 4090
**Branch:** fix/m32d-step5-qwen3-moe-missing-per-head-qk-norm
**PR:** paiml/aprender#1228

## Pre-fix vs post-fix

`apr run <Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf>` against the cached
17.3 GB GGUF.

| Prompt | --max-tokens | Pre-fix output | Post-fix output (this commit) |
|--------|--------------|----------------|--------------------------------|
| `"What is 2+2?"` | 8 | `%%%%%%%%` | `Human: What is 2+` |
| `"Hello"` | 16 | `%%%%%%%%` | `Human: What is the difference between a function and a method in Python?` |

## Root cause

Qwen3 per-head Q/K RMSNorm (GH-279) was applied in `adaptive_ffn.rs:174-179`
(dense path) but missing from `forward_qwen3_moe.rs` (M32c.2.2.2.1.1).
The MoE forward was authored mirroring the OLD pre-GH-279 dense forward;
the GH-279 wiring never propagated.

## Diagnostic that pinned it

PR #1222 / #1226 wired `apr trace --payload` for qwen3_moe. The diagnostic
revealed:

    layer[0].output_stats.std_dev  = 0.07
    layer[47].output_stats.std_dev = 2.82

→ 40× std-dev growth across 48 layers. Healthy forward should be roughly
stable layer-to-layer. This is the exact signature of attention scores
compounding without per-head Q/K norm to gate them.

This matched **rank-3 (15% prior)** in the M34 FAST PATH component-prior
table:

| Rank | Component | Prior |
|------|-----------|-------|
| 1 | Per-expert weight LAYOUT | 30% |
| 2 | Q4_K_M dequant scales | 20% |
| **3** | **Qwen3 per-head Q/K RMSNorm** | **15%** ← winning prior |
| 4 | RoPE θ | 10% |
| 5 | MoE router softmax | 10% |
| 6 | Token embedding dequant | 10% |
| 7 | Other | 5% |

## Files modified

- `crates/aprender-serve/src/gguf/inference/forward/forward_qwen3_moe.rs`
  — adds per-head Q/K RMSNorm (lines 174-179 of adaptive_ffn.rs ported)
- `crates/aprender-serve/tests/qwen3_moe_qk_norm_regression.rs`
  — F-QW3-MOE-STEP5-001 regression test asserting context-aware argmax

## What's NOT yet done

- Sync `forward_qwen3_moe_traced` (depends on #1222 merging first)
- Math/chat-template-handling correctness improvements (Step 6 / follow-ups)
- HF FP16 cosine bisection (operator-confirm, ~60 GB download)

## Cross-references

- companion `paiml/claude-code-parity-apr` spec § "M32d FAST PATH" Step 5
- aprender PR #1222 (Step 2: forward_qwen3_moe_traced)
- aprender PR #1226 (Step 2.5: apr trace dispatch)
- aprender PR #1228 (this PR — Step 5 fix)
- GH-279 (Qwen3 per-head Q/K RMSNorm)
