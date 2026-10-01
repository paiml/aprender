# Aprender Native SetFit Classification

## What This Is

This milestone extends the existing Aprender pure-Rust machine-learning framework with a
production-grade SetFit-style text-classification pipeline. It is aimed at developers and
business teams that need accurate, fast, auditable classifiers from small labeled datasets,
especially for social-media messages where operating a large generative model is unnecessarily
expensive.

The delivered path covers the complete lifecycle: deterministic few-shot data selection,
contrastive sentence-encoder fine-tuning, multiclass linear-head training, APR serialization,
CLI evaluation, native inference and serving, and reproducible comparison with Aprender's
existing 9B LoRA classification path.

## Core Value

A small labeled dataset can produce an accurate, fast, reproducible classifier that trains and
runs entirely through Aprender's native Rust and APR lifecycle.

## Requirements

### Validated

- ✓ Aprender provides native Rust training, evaluation, inference, serving, data, compute,
  GPU, and executable-contract crates — existing
- ✓ `apr finetune --task classify` provides a larger-model LoRA classification baseline —
  existing
- ✓ Aprender provides BERT-family loading and sentence-embedding inference paths — existing
- ✓ APR provides a checksummed, metadata-rich model container with native loading and serving
  integration — existing
- ✓ Classification evaluation already reports multiclass metrics and can express TweetEval's
  official `F_avg` over the `against` and `favor` classes — benchmark foundation implemented and
  verified in the current worktree

### Active

- [ ] Train a sentence encoder with a real differentiable SetFit-style contrastive objective;
  encoder parameters must demonstrably receive gradients and update
- [ ] Generate deterministic, class-aware positive and negative sentence pairs without
  materializing a quadratic Cartesian product
- [ ] Support batched tokenization, attention masks, masked pooling, normalization, and reusable
  tokenizer state for practical training throughput
- [ ] Train one fallible, regularized multiclass linear classifier head that also handles binary
  classification as the two-class case
- [ ] Expose a cohesive Rust API and `apr` CLI workflow for training, evaluating, predicting with,
  inspecting, and benchmarking SetFit classifiers
- [ ] Serialize the encoder, tokenizer identity, pooling/normalization policy, head, label map,
  training configuration, and provenance as one loadable APR artifact
- [ ] Reload the APR artifact through the production inference and serving paths while preserving
  embeddings, logits, probabilities, and labels within documented tolerances
- [ ] Compare SetFit with the 9B LoRA baseline across balanced 8, 16, 32, and 64 examples per
  class using identical sampled IDs and ten deterministic seeds
- [ ] Report classification quality, training time, inference throughput/latency, peak memory,
  model size, and calibration with machine-readable benchmark output
- [ ] Enforce test-split isolation, source provenance, feature compatibility, parameter updates,
  and artifact round trips through executable contracts and automated tests
- [ ] Support native CPU training/inference first and integrate optional existing accelerator
  backends without making GPU hardware a requirement

### Out of Scope

- Centroid-only classification presented as SetFit — genuine contrastive encoder tuning plus a
  learned classifier head is the defining capability
- Python-based training or serving in the production path — Python may be used only as a numerical
  reference during verification
- Replacing the existing 9B LoRA classifier — it remains a supported baseline and an option for
  tasks that benefit from large-model capacity
- Multilabel, hierarchical, token-level, or generative classification in the first milestone —
  v1 is single-label binary and multiclass sentence classification
- Universal compatibility with every Hugging Face sentence-transformer architecture — the first
  release supports a contracted encoder family and expands through verified adapters
- Hyperparameter tuning on TweetEval's merged SetFit compatibility test split — canonical
  validation and test isolation is mandatory for comparative claims
- Vendoring TweetEval tweet text — dataset content is acquired on demand from a pinned source

## Context

Aprender is an established Rust monorepo with separate crates for core ML algorithms, training,
data, compute, GPU execution, APR format/lifecycle, CLI, contracts, and serving. The architecture
map identifies the appropriate integration points in `crates/aprender-core/`,
`crates/aprender-train/`, `crates/aprender-data/`, `crates/apr-cli/`,
`crates/aprender-serve/`, and `contracts/`.

The codebase already has sentence-embedding inference, trainable transformer components,
contrastive metric utilities, linear probes, and classification evaluation, but they do not yet
form a valid SetFit training path. The most important technical risks are two competing encoder
and autograd stacks, graph-detaching embedding/pooling operations, scalar-only contrastive losses,
batch-size-one sentence inference, and fragmented binary/multiclass classifier heads.

TweetEval abortion stance is the reference benchmark. Its fixed labels are `none`, `against`, and
`favor`; the primary metric is `F_avg = (F1_against + F1_favor) / 2`. The canonical split contains
587 training, 66 validation, and 280 test examples. Few-shot experiments use balanced 8, 16, 32,
and 64 examples per class over seeds 13, 17, 23, 29, 31, 37, 41, 43, 47, and 53. The current
worktree already contains the data-conversion, provenance, metric, documentation, and benchmark
contract foundation for this protocol.

## Constraints

- **Tech stack**: Production training and inference remain pure Rust and integrate with Aprender's
  existing tensor, compute, training, CLI, APR, and serving boundaries
- **Algorithm fidelity**: SetFit means supervised contrastive sentence-encoder fine-tuning followed
  by a learned classification head; a frozen embedding probe is a baseline, not the algorithm
- **Correctness**: Tests must prove named encoder parameters receive finite non-zero gradients and
  change after an optimizer step
- **Benchmark integrity**: Canonical validation selects models and hyperparameters; canonical test
  is evaluated only after selection is fixed
- **Reproducibility**: Dataset revision, sampled IDs, seeds, tokenizer identity, encoder source,
  configuration, and content hashes must be recorded
- **Artifact integrity**: The evaluated model and the served APR artifact must be the same model,
  verified by a train-save-load-predict round trip
- **Performance**: Pair generation must be bounded; tokenization and encoder execution must batch;
  benchmarks must include time, memory, throughput, and artifact size rather than quality alone
- **Compatibility**: CPU inference and training must work without accelerator features; optional GPU
  paths must be feature-gated and tested as an explicit support matrix
- **Data licensing**: Tweet text is never committed; users obtain it from the pinned upstream source
- **Repository safety**: Existing public APIs, APR format boundaries, crate publishability, MSRV
  declarations, and feature combinations must remain verifiable

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build the full production vertical slice in the initial roadmap | The business value depends on deployment and reproducibility, not a training-only demonstration | — Pending |
| Implement genuine SetFit rather than centroid classification | Contrastive encoder adaptation is the source of SetFit's expected few-shot advantage | — Pending |
| Use TweetEval abortion stance as the reference dataset | It is an original SetFit evaluation dataset with a small, imbalanced social-media classification task | — Pending |
| Preserve canonical train/validation/test isolation | The two-split compatibility profile merges validation and test and is unsuitable for model selection | — Pending |
| Compare against Aprender's 9B LoRA classifier | The comparison quantifies the accuracy/resource tradeoff against the existing option | — Pending |
| Package encoder and head as one APR artifact | Production inference must consume exactly what training and evaluation produced | — Pending |
| Make CPU the mandatory baseline and GPU optional | Few-shot business datasets should be usable without specialized hardware | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-07 after initialization*
