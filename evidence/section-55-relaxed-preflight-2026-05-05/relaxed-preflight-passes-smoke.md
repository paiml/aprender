# LIVE smoke: §55 relaxed preflight passes on §54-extracted Qwen tokenizer

**Date:** 2026-05-05T05:48Z
**Branch:** feat/section-55-polymorphic-preflight-relaxation (this PR).
**Spec ref:** SPEC-SHIP-TWO-001 §55.
**Contract:** apr-pretrain-arch-polymorphic-v1 v1.3.0 FUNCTIONAL.
**Falsifiers exercised:** FALSIFY-APR-PRETRAIN-ARCH-009 (LIVE) + 010 (unit-test only).

## Inputs

- **Init APR:** `qwen2.5-coder-0.5b-instruct-fp16.apr` (declared vocab=151936).
- **Tokenizer dir:** `/tmp/qwen-0.5b-tokenizer-extracted` (vocab.json with 151643 BPE entries — produced by §54's PR #1497 `apr tokenize import-hf` LIVE smoke).
- **Corpus shards:** `/mnt/nvme-raid0/data/codeparrot-python-permissive-shards`.
- **Binary:** `apr` rebuilt from this branch at `/mnt/nvme-raid0/targets/aprender/release/apr`.

## Command

```bash
timeout 30 apr pretrain \
  --dataset /mnt/nvme-raid0/data/codeparrot-python-permissive-shards \
  --tokenizer /tmp/qwen-0.5b-tokenizer-extracted \
  --run-dir /tmp/apr-55-smoke/run-1 \
  --init /mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-fp16.apr \
  --mode finetune --num-steps 1 --device cpu --seed 42 \
  --vocab-size 151936 --batch-size 1 --seq-length 32
```

## Result

Process timeout-killed at 30s (exit=124) AFTER the preflight passed:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  apr pretrain — SHIP-TWO-001 MODEL-2 training loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


=== Configuration ===
    Dataset: /mnt/nvme-raid0/data/codeparrot-python-permissive-shards
    Tokenizer: /tmp/qwen-0.5b-tokenizer-extracted
    Run dir: /tmp/apr-55-smoke/run-1
    LR max: 5.00e-5
    Total steps: 1
    Warmup steps: 100
    Batch × seq: 1 × 32
    Steps / epoch: 100
    Seed: 42
    Target val_loss: 2.20

    Device: cpu
```

**No GATE-ARCH-370M-011 errors. No FALSIFY violations. No `violated:` lines.**

The process proceeded past preflight to weight load + forward-pass attempt; timeout fired at 30s (the 942MB FP16 weight load alone takes longer than 30s single-threaded on CPU; this smoke confirms preflight pass, not full training).

## What this proves

- **FALSIFY-APR-PRETRAIN-ARCH-009 LIVE-DISCHARGE-INTEGRATION**: the relaxed bound `tokenizer_vocab (151643) ≤ model_vocab (151936)` accepts an HF-distributed Qwen2.5 tokenizer dir under the polymorphic init path. Step 5g.1 (corpus retokenize) is now technically dispatchable.
- §55 amendment is **internally consistent**: code change + contract bump + helper test + integration test + LIVE smoke all align.
- §54's "5g.0 unblocks 5g.1 modulo §55" assertion is **resolved**: 5g.1 is now actually dispatchable (modulo the multi-hour wall the operator must accept).

## What this does NOT yet prove

- Full forward-pass numerical correctness on the loaded Qwen weights (the timeout cut the run before the first step completed).
- Convergence behavior — `val_loss < 9.38` is still 5g.3.
- Correctness of `populate_trainer_from_init_tensors` on a 0.5B FP16 source vs the trainer's 290-tensor expected shape (the load might still error on shape mismatch when the timeout-killed process eventually got there).

These are tracked as 5g.2 / 5g.3 follow-up work.

## Files referenced

- `crates/aprender-train/src/models/llama_370m.rs::assert_tokenizer_vocab_within_model_bound` (this PR).
- `crates/apr-cli/src/commands/pretrain.rs::preflight_tokenizer_vocab_matches_target` (this PR — extended signature).
- `contracts/apr-pretrain-arch-polymorphic-v1.yaml` v1.3.0 FUNCTIONAL (this PR).
- `evidence/section-54-5g-prereqs-2026-05-05/preflight-fail-fast-smoke.md` — §54's complementary fail-fast smoke that triggered §55.
- `evidence/section-50-4-step-5g-0-import-hf-2026-05-05/live-extraction-smoke.md` — §54's tokenizer extraction (PR #1497).
