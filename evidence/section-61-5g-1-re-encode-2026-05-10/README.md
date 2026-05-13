# §61 — 5g.1 Re-Encode + 5g.2 Honest Dispatch (2026-05-10)

## TL;DR

- **5g.1 re-encode SUCCESS**: 1.24 B Python tokens, **0% unk**, 7.42 bits
  entropy on shard-0 first 32K tokens (was 99.99% unk / 0.001 bits in
  the broken §60 corpus).
- **5g.2 LIVE dispatch (500-step) ABORTED**: GATE-TRAIN-005 fired at
  epoch 0 with val_loss = **11.55** (> 10.0 threshold).
- **5g.2 1-step diagnostic ABORTED**: val_loss = **19.80** (> ln(vocab)
  = 17.21 — *worse than uniform-over-vocab*).
- **NEW defect surface H4**: Qwen 0.5B init weights load (PR #1579's
  populate-coverage fix is in main) but produce sub-random predictions.
  Tracked as PMAT-CODE-PRETRAIN-INIT-LOAD-003.

## What worked (PR #1598's encoder fix landed)

`apr tokenize encode-corpus` with the upfront format detection
(this branch is on a main that includes #1598-equivalent local
build) processed the full codeparrot Python corpus correctly:

```
$ apr tokenize encode-corpus \
    --corpus /mnt/.../python-permissive.jsonl \
    --tokenizer /tmp/qwen-0.5b-tokenizer-extracted \
    --output /mnt/.../codeparrot-python-permissive-shards-qwen-v2 \
    --shard-tokens 10000000

  Shards: 126
  Total tokens: 1241.7M
  Total documents: 405.9K
  Manifest: /mnt/.../manifest.json
```

Entropy audit on shard-0 first 32K tokens:

| Metric | §60 broken corpus | §61 fixed corpus |
|--------|-------------------|------------------|
| Distinct tokens | 2 | **3324** |
| Shannon entropy | 0.001 bits | **7.415 bits** |
| Unk ratio | 99.99% | **0.00%** |
| Top tokens | 128244 (`<unk>`), 128247 (`</s>`) | 220 (Ġ-prefix), 198 (\n), 284, 11, 364 |

The corpus is now real Python tokenization. Ship-ready data input.

## What broke (5g.2 LIVE)

```
$ apr pretrain --mode finetune --num-steps 500 \
    --device cuda --init <Qwen 0.5B> \
    --dataset <5g.1-v2 corpus> \
    --target-val-loss 0.001  # disable convergence early-stop

  === Run Result ===
  FAIL ABORTED  DIVERGENCE at epoch 0: val_loss 11.552676 is non-finite or > 10.0
    Steps recorded: 100
    Epochs recorded: 0
```

Diagnostic 1-step run (to see step-0 baseline):

```
$ apr pretrain --mode finetune --num-steps 1 --steps-per-epoch 1 \
    --device cuda --init <Qwen 0.5B> --target-val-loss 0.0001 ...

  FAIL ABORTED  DIVERGENCE at epoch 0: val_loss 19.80251 > 10.0
    Steps recorded: 1
```

## The smoking gun for H4

`val_loss = 19.80` at step 1 is **higher than ln(vocab) = 17.21**.
This means the model assigns **less than uniform probability** to
the true tokens — *worse than random init*.

Reference baselines:

| Model state | Expected val_loss |
|-------------|-------------------|
| Uniform-over-vocab (151643) | ln(151643) = 17.21 |
| Random-init Qwen 0.5B | ≈ 17.21 |
| Qwen 0.5B zero-shot on Python | ≈ 1.5–3.0 |
| Trained-and-converged | ≈ 1.0–2.0 |

Observed at step 1: **19.80**. The init pipeline produces a model
that's **anti-aligned** with the held-out distribution.

## H4 candidate hypotheses

H4A — **Tied weights bug**: `tie_word_embeddings: true` on Qwen 0.5B
means lm_head shares with embed_tokens. If populate writes
embed_tokens but doesn't propagate to lm_head (or writes lm_head
separately to a random-init buffer), forward predictions are
random while embeddings are correct.

H4B — **Layout mismatch**: GGUF/APR layout contracts (`tensor-layout-v1`)
mandate row-major. If the init APR's lm_head is column-major (from
some HF→APR conversion path), the matmul produces wrong logits.

H4C — **Norm scale**: RMSNorm weight loaded but `rms_norm_eps` config
mismatch produces wrong activations after layer 0, cascading to
final logits.

H4D — **Residual stream init**: some block's residual contributes
zero (uninitialized buffer) → forward output is broken.

## Status against `apr-pretrain-init-finetune-v1.yaml`

| Falsifier | Status (post-§61) |
|-----------|-------------------|
| FALSIFY-001 (exit 0) | RED (diverge-abort returns Err) |
| FALSIFY-002 (wall ≤ 3600s) | DISCHARGED (~10s wall to abort) |
| FALSIFY-003 (step-0 ≤ 8.35) | RED (step-1 = 19.80) |
| FALSIFY-004 (checkpoint magic) | NOT-EVALUATED (run-dir cleaned on abort) |
| FALSIFY-005 (val_loss < 9.38) | RED-WITH-METHODOLOGICALLY-HONEST (was NUMERICALLY-PASSED-METHODOLOGY-SUSPECT pre-§61) |
| FALSIFY-006 (no CUDA errors) | DISCHARGED (clean abort, no kernel error) |

The status flip from FALSIFY-005 = NUMERICALLY-PASSED to RED is itself
**progress**: the contract now reports the real defect surface
instead of the data-bug-induced fake pass. The honest verdict
unblocks the H4 investigation.

## SHIP-TWO impact

- **MODEL-1 ship %**: unchanged at 91% (this is MODEL-2 work)
- **MODEL-2 ship %**: unchanged at 57% — but the path forward is now
  correctly diagnosed. Pre-§61: stuck on data bug masking the real
  defect. Post-§61: data is fixed; H4 (init load → sub-random
  predictions) is the binding gate.

## Out-of-scope follow-ups

PMAT-CODE-PRETRAIN-INIT-LOAD-003 (H4 cascade):
1. Bisect H4A (tied weights): print lm_head and embed_tokens after
   populate; assert they share the underlying tensor data.
2. Bisect H4B (layout): inspect Qwen APR's lm_head shape + first
   few values; compare to what aprender::Transformer expects.
3. Bisect H4C (norm scale): dump RMSNorm weights pre/post populate;
   verify they're not all-1.0 (would mean populate didn't write
   them).
4. Bisect H4D (residual): forward-pass a single token through
   each layer; verify outputs are non-degenerate.
5. Fix root cause; re-dispatch 5g.2 LIVE; flip MODEL-2 ship %
   57% → ≥58%.

## Files

- `dispatch.txt` (5g.1 re-encode log)
- (companion) `evidence/section-61-5g-2-honest-2026-05-10/dispatch.txt`
   (5g.2 dispatch log)
- This README — H4 hypothesis decomposition + audit trail
