# PMAT-697 — Albor Orphan Contract Triage (first pass)

**Date:** 2026-04-23
**Auditor:** PAIML Engineering via spec audit branch
**Input:** 52 filenames from `albor-orphans.txt` (albor contracts with no
monorepo counterpart)
**Method:** Read each albor contract title/description, classify by topic,
assign provisional disposition per §3 row #9 monorepo-single-source
policy.

## Summary

| Category              | Count | Provisional disposition                                                     |
|-----------------------|-------|-----------------------------------------------------------------------------|
| **training-kernel**   | 15    | PROMOTE into `contracts/entrenar/` — already the monorepo home for training |
| **backend-kernel**    |  8    | PROMOTE into `contracts/` or `contracts/entrenar/` — decide per kernel       |
| **checkpoint**        |  5    | PROMOTE; merge with `contracts/apr-provenance-v1.yaml` where overlapping    |
| **data-pipeline**     |  5    | MERGE-INTO-EXISTING — monorepo has `tokenizer-bpe-v1`, `dataset-thestack-python-v1`, `pretokenize-bin-v1`; diff for missing invariants |
| **eval**              |  6    | PROMOTE as new `contracts/eval/` family — expand on `eval-harness-humaneval-v1` |
| **architecture**      |  5    | PROMOTE; check overlap with `contracts/tensor-layout-v1.yaml` + `layout_contract.rs` |
| **hpo / budgeting**   |  5    | PROMOTE as new `contracts/training/hpo/` family                              |
| **parity**            |  1    | MERGE-INTO-EXISTING — monorepo has parity infrastructure; add invariants    |
| **other**             |  2    | Case-by-case (gguf-openai-completions, wire-protocol-v2-kernel)             |

**Totals:** PROMOTE: 44 | MERGE-INTO-EXISTING: 6 | CASE-BY-CASE: 2 = 52 ✓

## Detailed classification

### Category 1 — training-kernel (15) → PROMOTE into contracts/entrenar/

Core training-loop primitives. Monorepo `contracts/entrenar/` already
hosts GPU training backend contracts (see `gpu-training-backend-v1.yaml`,
`cuda-graph-training-step-v1.yaml`); these slot in naturally.

1. `fresh-training-step-v1.yaml`
2. `deterministic-training-kernel-v1.yaml`
3. `gradient-accumulation-kernel-v1.yaml`
4. `gpu-gradient-accumulation-v1.yaml`
5. `interleaved-optimizer-v1.yaml`
6. `tied-weight-optimizer-v1.yaml`
7. `ddp-pretrain-kernel-v1.yaml`
8. `ring-allreduce-kernel-v1.yaml`
9. `cosine-lr-schedule-v1.yaml`
10. `gradient-clipping-v1.yaml`
11. `training-config-kernel-v1.yaml`
12. `training-gpu-kernel-v1.yaml`
13. `training-memory-kernel-v1.yaml`
14. `training-step-budget-v1.yaml`
15. `knowledge-distillation-kernel-v1.yaml`

### Category 2 — backend-kernel (8) → PROMOTE into contracts/ or contracts/entrenar/

Compute-backend primitives. Disposition per kernel:

1. `cublas-gemm-v1.yaml` — `contracts/entrenar/` (NVIDIA training-path GEMM)
2. `cuda-inference-v1.yaml` — `contracts/` root (inference path — realizar)
3. `q4k-gpu-gemv-v1.yaml` — `contracts/` root
4. `wgpu-q4k-gemv-v1.yaml` — `contracts/` root
5. `fused-kernels-v1.yaml` — split into per-kernel sub-contracts OR pin under compute/
6. `model-merging-kernel-v1.yaml` — `contracts/entrenar/` (TIES/DARE/SLERP)
7. `pruning-kernel-v1.yaml` — `contracts/entrenar/` (QAT/PTQ-adjacent)
8. `safetensors-to-q4k-v1.yaml` — `contracts/` root (format conversion)

### Category 3 — checkpoint (5) → PROMOTE; merge with apr-provenance-v1 where overlap

1. `checkpoint-resume-v1.yaml` — NEW under `contracts/` root
2. `checkpoint-tokenizer-v1.yaml` — MERGE with `tokenizer-bpe-v1.yaml`
3. `checkpoint-inference-bridge-v1.yaml` — NEW under `contracts/`
4. `gpu-optimizer-checkpoint-v1.yaml` — NEW under `contracts/entrenar/`
5. `gpu-weight-pool-v1.yaml` — NEW under `contracts/entrenar/`

### Category 4 — data-pipeline (5) → MERGE-INTO-EXISTING

Monorepo already has `tokenizer-bpe-v1.yaml`, `dataset-thestack-python-v1.yaml`,
`pretokenize-bin-v1.yaml`. Diff each albor contract's invariants into the
monorepo counterpart:

1. `data-quality-filtering-v1.yaml` → merge into `dataset-thestack-python-v1.yaml`
2. `data-resume-position-v1.yaml` → merge into `pretokenize-bin-v1.yaml`
3. `data-shard-kernel-v1.yaml` → merge into `pretokenize-bin-v1.yaml`
4. `streaming-reader-v1.yaml` → merge into `pretokenize-bin-v1.yaml`
5. `bpe-tokenizer-kernel-v1.yaml` → merge into `tokenizer-bpe-v1.yaml`

### Category 5 — eval (6) → PROMOTE as new contracts/eval/ family

Monorepo has `eval-harness-humaneval-v1.yaml` but no broader eval family.
This batch justifies creating one:

1. `eval-humaneval-v1.yaml` — compare/reconcile with monorepo's
   `eval-harness-humaneval-v1.yaml`; one wins
2. `eval-mbpp-v1.yaml` — NEW
3. `auto-eval-scheduling-v1.yaml` — NEW
4. `multi-sample-passk-v1.yaml` — NEW
5. `v28-humaneval-eval-v1.yaml` — likely RETIRE (experiment-specific)
6. `teacher-completions-pipeline-v1.yaml` — NEW (distillation-adjacent)

### Category 6 — architecture (5) → PROMOTE; check layout contract overlap

1. `causal-attention-mask-v1.yaml` — NEW under `contracts/`
2. `residual-connection-v1.yaml` — NEW under `contracts/`
3. `weight-initialization-v1.yaml` — NEW under `contracts/entrenar/`
4. `lm-head-layout-parity-v1.yaml` — merge with `tensor-layout-v1.yaml`
5. `scaling-law-prediction-v1.yaml` — NEW under `contracts/`

### Category 7 — hpo / budgeting (5) → PROMOTE as new contracts/training/hpo/

1. `hyperparameter-tuning-v1.yaml` — C-HPO-001 per albor README
2. `batch-size-scaling-v1.yaml`
3. `memory-profiling-v1.yaml`
4. `resource-budget-v1.yaml`
5. `q4k-inference-sync-budget-v1.yaml`

### Category 8 — parity (1) → MERGE-INTO-EXISTING

1. `backward-parity-v1.yaml` — merge with `contracts/entrenar/apr-training-parity-v1.yaml`
   (monorepo already has training-parity infrastructure)

### Category 9 — other (2) → CASE-BY-CASE

1. `gguf-openai-completions-v1.yaml` — serving-side OpenAI-compat endpoint
   spec; belongs in `contracts/realizar/` or monorepo root; NEW
2. `wire-protocol-v2-kernel.yaml` — name suggests in-flight protocol work;
   read before deciding promote vs retire

## Next actions

1. **Fan out PMAT-697 into per-category sub-tickets** (one per category, 9
   sub-tickets), each owning the per-file decisions:
   - PMAT-698 Category 1 training-kernel (15 contracts)
   - PMAT-699 Category 2 backend-kernel (8)
   - PMAT-700 Category 3 checkpoint (5)
   - PMAT-701 Category 4 data-pipeline merges (5)
   - PMAT-702 Category 5 eval family (6)
   - PMAT-703 Category 6 architecture (5)
   - PMAT-704 Category 7 hpo / budgeting (5)
   - PMAT-705 Category 8 backward parity merge (1)
   - PMAT-706 Category 9 other (2)
2. Each sub-ticket lands a PR per-category that (a) copies the albor
   contract into the monorepo under the right path, (b) validates via
   `pv validate`, (c) adds `promoted_from: albor@be23737` metadata, (d)
   flags albor-side cleanup under the paired PMAT-691 hygiene ticket.
3. After all 9 sub-tickets close, run `pv lint contracts/` on the whole
   monorepo to confirm the promoted-in contracts don't duplicate or
   conflict internally.
4. Once the orphan count is zero (modulo RETIRE decisions), enable
   **PMAT-693 v2.0** (`apr audit-ship-two --include-albor`) to prevent
   future re-divergence.

## Caveats

- This is a first-pass classification from filenames + titles. A
  per-file read may reclassify some contracts (e.g., `fused-kernels-v1`
  may turn out to be a single-contract umbrella rather than a split-
  candidate). Each sub-ticket's first step is the per-file read.
- The RETIRE disposition is rare (only `v28-humaneval-eval-v1` is
  flagged as experiment-specific). RETIRE decisions are irreversible
  (the contract's invariants get dropped); the sub-ticket owner MUST
  document why retirement is safe.
- "MERGE-INTO-EXISTING" means the albor contract's invariants get
  absorbed into the monorepo contract with a version bump and a
  `promoted_from:` audit-trail entry in the metadata — the albor file
  ultimately deletes.
