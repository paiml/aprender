# 5g.2 LIVE Re-Dispatch — H1 Eval-Batch Divergence Evidence (2026-05-09)

## Dispatch

```
apr pretrain \
  --dataset /mnt/nvme-raid0/data/codeparrot-python-permissive-shards-qwen \
  --tokenizer /tmp/qwen-0.5b-tokenizer-extracted \
  --run-dir /mnt/nvme-raid0/runs/5g-2-redispatch-with-bias-fix-2026-05-09 \
  --mode finetune --num-steps 500 \
  --batch-size 4 --seq-length 512 \
  --device cuda \
  --init /mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-fp16.apr \
  --target-val-loss 0.001  # disable convergence-based early-stop
```

Host: lambda-vector RTX 4090. Build: `apr-cli` from
`feat/populate-tensor-coverage-falsifier` branch (rebased onto main
post-PR #1577 5f.5 wireup, with PR #1579 populate-coverage fix
applied locally).

## Context

This is the FOLLOW-UP to the §59 evidence (PR #1578) that recorded
val_loss=0.0008 from a 500-step run with the pre-fix code. PR #1579
(populate-coverage falsifier + fix) addressed H2 (Q/K/V biases were
silently dropped). The hypothesis was: "with H2 fixed, val_loss
should land in 1.5-2.5 (industry-plausible for Qwen 0.5B on Python)."

That hypothesis was FALSIFIED by this re-run.

## Result (raw — see metadata files)

Epoch | train_loss | val_loss   | train_ppl | val_ppl  | grad_norm
------|-----------|-----------|-----------|----------|----------
0     | 1.20      | 0.00081   | 3.33      | 1.0008   | 14.81
1     | 0.0014    | 0.00077   | 1.0014    | 1.0008   | 0.77
2     | 0.0019    | 0.00075   | 1.0019    | 1.0008   | 1.07

Run early-stopped at epoch 2 (3 × 100 steps = 300 steps; target
val_loss=0.001 exceeded by val_loss=0.00075).

## The smoking gun for H1

**At epoch 0**, the model has been trained for exactly 100 steps
(204,800 tokens). Train_loss across those 100 batches = **1.20**
(perplexity 3.33 — empirically PLAUSIBLE for Qwen 0.5B fine-tuning
on Python code). But val_loss on the held-out 16 batches at the
same model state = **0.00081** (perplexity 1.0008 — physically
IMPOSSIBLE for a non-degenerate LM).

**1500× train/eval discrepancy at the same model state.** Same
kernel (`fused_cross_entropy_cuda` from
`crates/aprender-train/src/autograd/cuda_optim.rs`). Same scaling
(`scale = 1.0 / seq_len`). Same forward path (`gpu_forward`
→ logits in `gpu_training.logits_buf`). Different batches, but
both drawn from the same Python corpus shards.

A model that produces train_loss=1.20 cannot produce val_loss=0.00081
unless one of:
1. Eval reads a different model state than train.
2. Eval reads a different (degenerate) batch.
3. Eval's loss computation has a bug.

The held-out batches were captured with `iter.next()` BEFORE
training — they are valid `LMBatch` objects from the same shard
files. So #2 is unlikely. The model state is shared via
`SharedCudaTrainer = Rc<RefCell<CudaTransformerTrainer>>` between
`CudaRealStepFn` and `CudaRealValFn`. So #1 is also unlikely.

The remaining hypothesis is #3: **a bug in the eval-side loss
computation that doesn't affect the train-side measurement** even
though they share the kernel. Possible mechanisms (each its own
investigation):

- A) `gpu_training.logits_buf` state contamination: train_batch's
  `fused_cross_entropy_cuda` writes gradients **in-place** into
  `logits_buf` (KAIZEN-052). If `eval_batch`'s subsequent
  `gpu_forward` doesn't fully overwrite the gradients (e.g., a
  GEMM that's add instead of set, or a partial write), eval would
  read a mix of fresh logits and stale gradients.
- B) Stream synchronization: although CUDA serializes kernels on
  the same stream, there could be a host-side ordering bug where
  the host reads `loss_partials` before the kernel finishes
  writing them. The `stream.synchronize()` at line 805 of
  `cuda_optim.rs` should prevent this, but if there's a kernel
  failure that leaves loss_partials at zero and the failure isn't
  surfaced, the result would be ~0.
- C) Held-out batch label corruption: `LMBatch::from_sequences`
  produces a shared-layout batch when sequences are uniform; the
  shared-layout `get_target` returns
  `tokens[batch_idx * stride + 1 .. + seq_len]`. If the held-out
  batches happen to land on a buffer where `tokens[batch_idx *
  stride + 1]` is the same as `tokens[batch_idx * stride]` (a
  pathological structure), softmax would assign probability 1
  trivially — but this would be visible in the data and is hard
  to hit by accident on real Python code.

## Falsifier discharges

| ID | Rule | Result |
|----|------|--------|
| FALSIFY-APR-PRETRAIN-INIT-FINETUNE-001 | exit 0 | ✅ DISCHARGED |
| FALSIFY-APR-PRETRAIN-INIT-FINETUNE-002 | wall ≤ 3600s | ✅ DISCHARGED (~40s) |
| FALSIFY-APR-PRETRAIN-INIT-FINETUNE-003 | step-0 train_loss ≤ 8.35 | ✅ DISCHARGED (1.20) |
| FALSIFY-APR-PRETRAIN-INIT-FINETUNE-004 | checkpoint with valid magic | ✅ DISCHARGED |
| FALSIFY-APR-PRETRAIN-INIT-FINETUNE-005 | val_loss < 9.38 | **NUMERICALLY-PASSED-METHODOLOGY-SUSPECT** (0.00075 numerically passes; H1 eval bug means the number is not honest) |
| FALSIFY-APR-PRETRAIN-INIT-FINETUNE-006 | no CUDA errors | ✅ DISCHARGED |

## Hypothesis status update

| Hypothesis | Pre-this-evidence | Post-this-evidence |
|------------|-------------------|--------------------|
| H2 — populate gap (Q/K/V biases dropped) | OPEN | **DISCHARGED** by PR #1579 (train_loss now 1.20 vs prior 0.0019; structurally complete model) |
| H1 — eval_batch degenerate | OPEN (suspected) | **CONFIRMED OPEN** by 1500× train/eval disagreement at same model state |

**H2 was a real defect with a real fix** (the `train_loss = 1.20`
vs prior `0.0019` shift confirms the structural change). But H2
was NOT the root cause of the val_loss anomaly — H1 is. The two
defects had compounding effects on the val_loss number; only fixing
both would make the number trustworthy.

## SHIP-TWO impact

- **MODEL-1 ship %**: unchanged at 91% (this is MODEL-2 work).
- **MODEL-2 ship %**: **unchanged at 57%** until H1 is also
  resolved AND a 500-step re-dispatch produces a numerically-
  plausible val_loss in the 1.5-2.5 range.
- **§50.4 cascade**: COMPLETE per PR #1577.
- **5g.2 dispatch**: OPERATOR-RUNNABLE end-to-end on RTX 4090
  (PR #1577) with structurally-complete model (PR #1579) but
  the **honest 5g.3 verdict** remains gated on H1 resolution.

## Next steps (out of scope — own falsifier-discharge cascades)

Tracked as PMAT-CODE-PRETRAIN-EVAL-METHODOLOGY-001:

1. Author a unit test that calls `CudaTransformerTrainer::eval_batch`
   on a fresh-init trainer with a synthetic batch and asserts loss
   in [0.5, ln(vocab_size) × 1.1]. Random-init Qwen 0.5B should
   produce loss ≈ ln(151936) = 11.93. If the test produces ~0,
   H1 is confirmed at the unit-test level.
2. Bisect the three sub-hypotheses (A logits_buf contamination,
   B stream sync, C held-out label corruption) with targeted
   instrumentation.
3. Fix root cause; re-dispatch 5g.2 to obtain honest 5g.3 verdict.

## Files

- `dispatch.txt` — full apr pretrain stdout/stderr from the run
- `epoch-{000,001,002}.metadata.json` — per-epoch loss + grad_norm
- (this file) — H1/H2 hypothesis decomposition + methodology audit
