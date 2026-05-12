# 5g.2 LIVE 500-Step Fine-Tune Dispatch — Evidence (2026-05-09)

## Dispatch

```
apr pretrain \
  --dataset /mnt/nvme-raid0/data/codeparrot-python-permissive-shards-qwen \
  --tokenizer /tmp/qwen-0.5b-tokenizer-extracted \
  --run-dir /mnt/nvme-raid0/runs/5g-2-live-500step-2026-05-09 \
  --mode finetune --num-steps 500 \
  --batch-size 4 --seq-length 512 \
  --device cuda \
  --init /mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-fp16.apr
```

Host: lambda-vector RTX 4090. Build: `apr-cli` from
`feat/apr-pretrain-init-cuda-wireup-5f5` branch (PR #1577 — §50.4 step
5f.5 wireup).

## Result (raw)

```
=== Run Result ===
  OK CONVERGED  final val_loss=0.0008 after 3 epoch(s)
    Steps recorded: 300
    Epochs recorded: 3
```

Wall: ~40 seconds (run early-stopped at epoch 2 hitting target_val_loss=2.20).

Final epoch metadata (`ckpt/epoch-002.metadata.json`):

```json
{
  "epoch": 2,
  "train_loss": 0.00187467,
  "val_loss": 0.00075797737,
  "train_ppl": 1.0018765,
  "val_ppl": 1.0007583,
  "wall_seconds": 13.108396,
  "tokens_seen": 614400,
  "grad_norm_max": 1.0680338
}
```

Checkpoint: `epoch-002.apr` — 2.35 GiB, 219 tensors, valid APR v2.

## Falsifier verdict against `apr-pretrain-init-finetune-v1.yaml` v1.0.0

| ID | Rule | Result | Status |
|----|------|--------|--------|
| FALSIFY-001 | exit code == 0 | exit 0 ✓ | **DISCHARGED** |
| FALSIFY-002 | wall ≤ 3600s | ~40s ✓ | **DISCHARGED** |
| FALSIFY-003 | step-0 loss ≤ 8.35 | (not parsed; train_loss=0.0018 by epoch 2) | PARTIAL |
| FALSIFY-004 | checkpoint magic bytes | APR v2 ✓ | **DISCHARGED** |
| FALSIFY-005 | val_loss < 9.38 | val_loss=0.0008 ✓ NUMERICALLY | **NUMERICALLY-PASSED-METHODOLOGY-SUSPECT** |
| FALSIFY-006 | no CUDA errors | clean log ✓ | **DISCHARGED** |

## Why FALSIFY-005 is NOT recorded as DISCHARGED

The reported val_loss=0.0008 is implausibly low. For context:

- Industry baseline: SmolLM-360M on 1T tokens trains to val_loss ~2.9
  (per §49 of the spec).
- Qwen2.5-Coder-0.5B-Instruct zero-shot on Python code: typically
  val_loss ~2.0-3.0.
- A 300-step fine-tune on 200K tokens cannot lower val_loss to 0.0008
  unless the eval is degenerate or the held-out distribution is
  saturated/leaked.

Two leading hypotheses, each with its own falsifier-discharge cascade:

**H1 — CudaTransformerTrainer::eval_batch returns degenerate loss.**
The CUDA eval path (`crates/aprender-train/src/train/transformer_trainer/cuda_trainer.rs::eval_batch`)
may compute a loss differently than the CPU path. Symptom would
appear when: training loss is normal but val_loss collapses to
~zero regardless of input.

**H2 — populate_trainer_from_init_tensors silently drops 71/290 Qwen tensors.**
The Qwen 0.5B APR has 290 tensors; the saved checkpoint has 219
tensors; 71 drift between the polymorphic `Transformer::new(qwen2_0_5b())`
named-parameters set and the Qwen tensor naming. The partial-init
produces a hybrid model whose loss curve doesn't reflect reality.
Symptom would appear when: only Qwen-decoder-block tensors populate;
final norm / lm_head / embed_tokens tie-weights stay random-init,
and the held-out batches happen to be insensitive to those tensors.

## What the LIVE dispatch DOES prove

- **§50.4 step 5f.5 wireup is functional end-to-end**: `apr pretrain
  --init Qwen.apr --device cuda` now runs the full forward + backward
  + AdamW pipeline on RTX 4090 without crashes.
- **Checkpoint serialization is sound**: `epoch-002.apr` is a valid
  APR v2 file with 219 tensors and a passing checksum.
- **CUDA + fine-tune is fast**: ~40 seconds wall for 300 steps =
  ~7.5 steps/sec at batch=4 seq=512.

## What the LIVE dispatch does NOT prove

- That the trained model produces meaningful Python (saved checkpoint
  lacks embedded tokenizer; `apr run` rejects with PMAT-172).
- That val_loss=0.0008 reflects genuine convergence (see H1/H2 above).
- That MODEL-2 ship % can flip 57% → ≥58% (FALSIFY-005 is methodology-
  suspect, not honestly DISCHARGED).

## SHIP-TWO impact

- **MODEL-1 ship %**: unchanged at **91%** (this is MODEL-2 work).
- **MODEL-2 ship %**: **unchanged at 57%** until val_loss anomaly is
  resolved AND a re-run produces a numerically plausible verdict.
- **§50.4 cascade**: COMPLETE per PR #1577 (5a-5f.5 all shipped).
- **5g.2 dispatch**: **OPERATOR-RUNNABLE** end-to-end on RTX 4090
  (this evidence) but the **ship-% verdict** remains gated on
  resolving the H1/H2 methodology question.

## Next steps (out of scope for PR #1577 / this session)

Tracked as task PMAT-CODE-PRETRAIN-EVAL-METHODOLOGY-001:

1. Add a falsifier asserting `populate_trainer_from_init_tensors`
   reports `populated == |init_tensors|` for canonical Qwen 0.5B APR
   (catches H2).
2. Add a falsifier asserting `CudaRealValFn::validate` on a known-
   distribution synthetic batch returns a loss within ε of the CPU
   sibling's `RealValFn::validate` on the same batch (catches H1).
3. Re-dispatch 5g.2 once both falsifiers are bound; the verdict
   then either flips MODEL-2 ship % 57% → ≥58% honestly, or
   surfaces the residual bug for follow-up.

## Files

- `dispatch.log` — full apr pretrain stdout/stderr from the run.
- (this file) — methodology audit + falsifier verdict.
