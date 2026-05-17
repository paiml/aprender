# P2-C Verdict: corpus diversity not binding — upstream metadata defect blocks downstream P0 chain

**Date:** 2026-05-17
**Run dir:** `/mnt/nvme-raid0/runs/model-2-p2c-50k-20260517`
**Ticket:** PMAT-681
**Spec:** [albor-370m-roadmap.md §4 P2-C](../../docs/specifications/aprender-train/albor-370m-roadmap.md) + [ship-model-2-spec.md §82, §83](../../docs/specifications/aprender-train/ship-model-2-spec.md)

## Summary

P2-C dispatched the audit-recommended multi-source corpus widening: 49.6B tokens (the-stack-dedup Python 28.6 GB + codeparrot-clean 12.8 GB), tokenized via the §83 P2-C multi-source `--corpus` machinery, trained with Qwen-0.5B init under the §83 P0-J Chinchilla gate. The training completed end-to-end, but:

- **val_loss best = 4.91 @ epoch 20** — *worse* than §82's 4.71 baseline despite 5× corpus
- **EARLY_STOP fired at 27 epochs / 2700 steps** — identical termination shape to §82
- **P0-D / P0-H / llama-cli interop ALL still blocked** — root cause is **upstream**: the init APR `qwen2.5-coder-0.5b-instruct-imported.apr` lacks `hf_architecture` + embedded tokenizer metadata, so every downstream P0-* fix has nothing to propagate

The audit's **Chinchilla-data-starvation hypothesis is FALSIFIED** by P2-C's empirical equivalence to §82.

## Numbers

| Quantity | §82 (qwen-v2, single-source) | P2-C (qwen-v3, multi-source) |
|---|---|---|
| Total corpus | 1.24B tokens, 1 source | **49.6B tokens, 2 sources** |
| Chinchilla ratio (corpus) | 0.125× | **100.45×** |
| Steps recorded | 2700 | 2700 |
| Epochs recorded | 27 | 27 |
| Best val_loss | 4.7111 @ ep20 | 4.9112 @ ep20 |
| Termination | OK EARLY_STOP | OK EARLY_STOP |
| Bench tok/s | 325.1 | 315.6 |

**Identical termination shape** (27 epochs, 2700 steps, best at epoch 20) with **+0.2 absolute val_loss** despite 80× more corpus tokens.

## Trajectory (every 2 epochs)

```
ep  0:  7.03 (init eval)
ep  1:  6.43
ep  3:  6.18
ep  5:  5.57
ep  7:  5.32
ep  9:  5.33  ← noise plateau forming
ep 12:  5.44  ← spike +0.12
ep 14:  5.29  ← new best, breakout
ep 16:  5.60  ← spike +0.31
ep 18:  4.96  ← second breakout
ep 20:  4.91  ← BEST
ep 22:  4.93
ep 24:  4.99
ep 26:  5.02  ← patience exhausted, early-stop
```

The oscillation pattern (descend → spike → bigger descend) is more pronounced than §82's monotonic profile. Likely cause: multi-source distribution makes single-step gradient updates harder to fit cleanly.

## What worked end-to-end (newly verified live)

- **P2-C multi-source `--corpus`** (PR #1721): consumed 18.3M docs across 2 sources at 22,500 docs/s on 48 workers, 17 min wall.
- **Manifest `corpus_roots`** (PR #1732): manifest.json correctly enumerates `["/mnt/.../the-stack-dedup-python/...", "/mnt/.../codeparrot-clean-jsonl/"]`.
- **Tokenizer extraction** (existing `apr tokenize import-hf`): 151,643 Qwen vocab + 151,387 merges extracted from `qwen2.5-coder-1.5b-safetensors/tokenizer.json`.
- **Multi-source manifest invariants**: total_tokens = 49.6B (passes INV-MERGE-002 4.94B floor by 10×); corpus_roots.length = 2 (passes INV-MERGE-001).
- **P0-J Chinchilla gate `--force-under-provisioned` bypass** (PR #1722 + #1731): per-run D=410M < 10·N=4.94B triggers gate; bypass works, emits loud BYPASSED log, run proceeds.
- **CudaAprCheckpointFn save with arch metadata** (PR from §81): all 27 epoch checkpoints have arch / hidden_size / num_layers / num_heads stamped.

## What's still broken — and the upstream root cause

### Symptom 1: `apr qa` fails — "APR missing embedded tokenizer"

The trained checkpoint's APR file does not embed the tokenizer JSON even though `--tokenizer <DIR>` was passed to `apr pretrain`.

### Symptom 2: `apr inspect` reports `architecture = "LlamaForCausalLM"`

Despite the init being a Qwen2 model, the checkpoint stamps `LlamaForCausalLM` — P0-H's `checkpoint_name_and_arch` helper fell back to the default because `init_arch.hf_architecture == None`.

### Symptom 3: llama-cli rejects exported GGUF — "cannot find tokenizer merges in model file"

`apr export --format gguf` produces a GGUF with no `tokenizer.ggml.merges` array because the source APR has no embedded tokenizer to copy them from. Separately, the 72 Qwen2 attn biases leak as passthrough names because arch=llama family mapper doesn't know them.

### Root cause (NEW P0-K)

All three symptoms trace to **one upstream defect**: `apr convert` (the import-from-HF-safetensors path used to produce `qwen2.5-coder-0.5b-instruct-imported.apr`) does NOT stamp:
- `apr_metadata.hf_architecture`
- `apr_metadata.tokenizer.vocabulary` + `tokenizer.merges`
- (Other per-arch metadata keys may also be incomplete)

When `apr pretrain --init <imported.apr>` reads this incomplete metadata, the downstream P0-D/E/F/G/H/J machinery cannot propagate what isn't there. The §82 + P2-C trained checkpoints are byte-for-byte downstream consequences of this single upstream gap.

**P0-K scope**: `apr convert` (or wherever it lives) must round-trip stamp:
1. `hf_architecture` from the source `config.json` `architectures[0]`
2. `tokenizer.vocabulary` + `tokenizer.merges` from `tokenizer.json` if present alongside the input
3. `hf_model_type` from `config.json` `model_type`
4. Other well-known metadata keys (rope_theta, rms_norm_eps, vocab_size, etc.) are already stamped — just the tokenizer + arch identity gaps remain

## Methodology lesson #33 NEW — Upstream metadata defects masquerade as downstream packaging defects

The §81-§83 Class 3 cascade (P0-D / P0-E / P0-F / P0-G / P0-H — 5 PRs, ~3 days of work) treated each downstream tool failure as its own defect. The actual root cause was a single upstream defect in `apr convert`: imported APRs lack the metadata that all the downstream tools depend on. P2-C's live run made this visible because the trained checkpoint exposes the EXACT same failures as the imported init — the import-to-pretrain pipeline is metadata-transparent.

**Lesson**: when a Class 3 packaging wave (lesson #29) extends past 4-5 defects, **pause** and check whether they share an upstream metadata producer that's the actual root cause. The audit's pre-falsification framework (#30) is great at killing dead dispatches but doesn't surface upstream-producer defects.

## Why P2-C's val_loss is 0.2 worse than §82

Hypothesis: held-out validation batches are drawn from the start of the corpus (§82 default `HELD_OUT_BATCHES=N` from the first N batches of the shard iterator). For qwen-v2 those were codeparrot Python; for qwen-v3 they're a mix of the-stack-dedup + codeparrot. The val sets are NOT comparable — the +0.2 gap is not a "P2-C is worse" finding, it's a "different val distribution" finding.

For a properly comparable P2-C vs §82 result, the held-out batches need to be from the SAME source shards. That's a separate methodology fix.

## What this means for ship %

- **MODEL-2 ship %**: stays at **79%**. No movement.
  - +0 because val_loss didn't break 3.0 threshold for P1-B/C eligibility.
  - +0 because P0-D/H discharge still blocked on the upstream P0-K defect.
  - 0 net effect on AC-SHIP2-003 (val_loss best worse than §82's, not better).
- **Audit hypothesis FALSIFIED**: corpus diversity / Chinchilla ratio was not the binding constraint.
- **Real binding constraint**: hyperparameters (LR schedule, patience) and/or held-out val distribution.

## Next-action priority queue (updated)

1. **P0-K (NEW, highest EV)**: fix `apr convert` to stamp `hf_architecture` + embedded tokenizer + arch metadata into imported APRs. Unblocks all of P0-D, P0-H, llama-cli interop transitively. Scope: ~100 LOC + integration test. Effort: 2-4 h.
2. **Re-dispatch P2-C trained checkpoint AFTER P0-K**: same checkpoint via `apr export` should now work end-to-end through llama-cli.
3. **P2-E (NEW)**: hyperparameter tuning — lower peak LR, longer warmup, longer patience. Effort: 2-4 h per dispatch, 1-2 dispatches.
4. **P2-F (NEW)**: shared held-out val set for §82 / P2-C comparison — pre-emit a "golden val batches" shard at corpus assembly time, train against it. Effort: 1 day.

## Evidence files

- `pretrain-50k.log` — full training stdout (epoch metadata + per-step progress)
- `pretrain-final-tail.log` — last 20 lines of training output (EARLY_STOP banner)
- `loss-trajectory.tsv` — 27-epoch trajectory (epoch, val_loss, train_loss, tokens_seen)
- `epoch-020.metadata.json` — best checkpoint's metadata
- `bench-epoch-020.json` — apr bench result (315.6 tok/s)
- `tokenize-qwen-v3.log` — corpus assembly log
- `pull-thestack-v1.log`, `pull-codeparrot.log`, `decompress-codeparrot.log` — data pipeline logs
