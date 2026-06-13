# Pillar-3 LoRA merge forward-equivalence beat — measured (2026-06-14)

**Claim:** apr's `MergeEngine::merge` folds a LoRA adapter into the base weight
faithfully — the merged weights produce a forward pass numerically equivalent to
applying the LoRA factors unmerged. apr contract-gates this; PEFT/Unsloth
`merge_and_unload` ships no forward-equivalence guarantee.

## Why this completes the P3 "replace Unsloth" story
Unsloth's value is fast QLoRA fine-tuning → merge → export. apr's correctness beats now
cover both stages: NF4 quantization ≡ bitsandbytes (PMAT-745) and LoRA merge
forward-equivalence (this, PMAT-747). Both are contract + (NF4) Lean-gated.

## Method (self-contained, not tautological)
Fold `A:[4,2], B:[2,3], scale=alpha/rank=2.0` into `W_base:[3,4]` via
`MergeEngine::merge`, then forward a fixed `x:[2,4]` two ways:
1. **merged:** `y = x @ W_merged^T`
2. **factored (independent path):** `y = x @ W_base^T + scale·(x @ A @ B)`

A transpose/indexing bug in `merge` would make these diverge — the reference is
computed from the A,B factors via `x@A@B`, never from `W_merged − W_base`.

| metric | value |
|--------|-------|
| **max \|y_merged − y_factored\|** | **1.49e-8** (essentially bit-exact) |

## CI-gated form
`crates/aprender-train-lora/src/merge.rs::tests::beat_lora_merge_forward_equivalence`
(threshold 1e-4) — deterministic, CPU, no external deps. Contract:
`contracts/apr-lora-merge-equivalence-beat-v1.yaml`. Wired into ci.yml.
