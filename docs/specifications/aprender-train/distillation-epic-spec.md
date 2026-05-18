# Specification: Distillation Epic — paiml/albor-370m-v2

**Document ID:** SPEC-DISTILL-001
**Version:** 1.0.0
**Status:** Live — opens the distillation track that picks up where MODEL-2's §88 stack-existence-proof ship left off
**Parent:** [ship-model-2-spec.md §84.5](./ship-model-2-spec.md)
**Triggers:** PMAT-683 (P2-D: True distillation from MODEL-1), PMAT-684 (P1-B: HumanEval pass@1)
**First applied:** TBD — see Phase 1 below

## Purpose

MODEL-2's v1 ship (`paiml/albor-370m-v1`, val_loss=4.6227, 2026-05-18) proved the pure-Rust Sovereign AI Stack runs end-to-end. The model is honestly framed as a stack-existence-proof — it produces gibberish at val_perplexity≈102 and is explicitly NOT a production code-completion model.

The distillation epic ships **MODEL-2 v2** — `paiml/albor-370m-v2` (or v1.1.0) — a ~370M-param student distilled from the **MODEL-1 teacher** (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) that actually hits a usable HumanEval pass@1. Target: pass@1 ≥ 25% (loose first-target; competitive 0.5B is ~30-40%; the upstream 7B teacher is at ~91%).

The epic's strategic value: it converts the stack from "runs end-to-end" → "produces models worth using". That's the difference between an existence proof and a product.

## Current state (2026-05-18)

The distillation infrastructure is **scaffolded but stubbed**:

| Component | File | State |
|---|---|---|
| CLI: `apr distill ...` | `crates/apr-cli/src/commands/distill.rs:1119` | Stub — writes a metadata-only "pending_download" manifest |
| Pipeline orchestrator | `crates/aprender-train-distill/src/pipeline.rs:115-160` | Stub — uses **synthetic logits derived from weight tensor slices** instead of actual forward passes through teacher + student |
| KD loss | `crates/aprender-train/src/distill/loss.rs:34` (`DistillationLoss`) | **Real** — KD loss = α · CE(student, hard_target) + (1-α) · T² · KL(softmax(student/T), softmax(teacher/T)) |
| HF-pipeline KD variant | `crates/aprender-train/src/hf_pipeline/distillation/loss.rs:24` | **Real** — used for distillation under the hf_pipeline path |
| Student forward+backward | `crates/aprender-train/src/train/transformer_trainer/cuda_trainer.rs:1256` (`forward_logits`), `:2378` (`forward_backward_batch`) | **Real** — `CudaTransformerTrainer` proven via MODEL-2 §82 P2-A 5000-step run |
| Teacher inference | `realizar` (aprender-serve) — `apr run` path | **Real** — proven via MODEL-1 SHIP-005 HumanEval=86.59% |
| GGUF / SafeTensors weight loading | `aprender-core::format::converter` | **Real** — v0.34.0 layout + Q4_K paths shipped |

**The gap**: pipeline.rs's `train()` method constructs `teacher_logits` from `build_synthetic_logits(&teacher_weights, batch_size, num_classes=32)` — a flat 32-class slice of weight bytes. It NEVER calls realizar to run the teacher, NEVER calls `CudaTransformerTrainer.forward_backward_batch` on the student. The loss math is exercised on synthetic data; the gradient updates do nothing meaningful.

Closing this gap is the epic.

## Phased plan

### Phase 1 — Teacher logits service via realizar (PMAT-683 P1, ~3 days)

**Goal**: `apr distill` invokes realizar to compute teacher logits over the training corpus and caches them to disk.

Why a separate phase: teacher forward over a 7B model is expensive (~3-5s/batch on RTX 4090 for batch=8 seq=512). Recomputing per training step is wasteful. The HF community standard (DistilBERT, MiniLM, Distil-Qwen) caches teacher logits once per corpus and replays them across training epochs. We do the same.

**Deliverables:**
1. `apr distill prepare --teacher <path> --dataset <bin-shards> --out <cache-dir>` — new subcommand. Iterates batches, calls `realizar::Model::forward_logits(input_ids) -> Vec<f32>`, writes per-batch `.logits.bin` files (top-K + indices to reduce storage; K=64 is standard).
2. New module `aprender-train-distill/src/teacher_cache.rs` — defines on-disk format (header magic `APRLOG\0`, per-batch records `(batch_idx, seq_len, top_k_logits[K], top_k_indices[K])`).
3. Integration test: prepare 100-batch slice, verify cache file size + read-back matches realizar's online compute within float-roundoff.

**Effort estimate**: 16-24 hours engineering + 4-8h compute (1B-token corpus prep takes ~6h on RTX 4090 for a 7B teacher at batch=8).

**Falsifier**: F-DISTILL-PREP-001 — `apr distill prepare` produces a cache where, for any random batch, calling `realizar` online produces the same top-K within cosine sim ≥ 0.999.

### Phase 2 — Student forward+backward wired to KD loss (PMAT-683 P2, ~2 days)

**Goal**: Replace pipeline.rs's `build_synthetic_logits(...)` with a real `CudaTransformerTrainer.forward_backward_batch_with_kd(...)` call.

**Deliverables:**
1. Extend `CudaTransformerTrainer` (`crates/aprender-train/src/train/transformer_trainer/cuda_trainer.rs`) with `forward_backward_kd_batch(batch: &LMBatch, teacher_logits: &TeacherLogitsBatch) -> f32`. The trainer already has `forward_backward_batch`; the KD variant computes the combined loss = α·CE + (1-α)·T²·KL via `DistillationLoss` instead of CE alone, then back-props.
2. Rewrite `aprender-train-distill/src/pipeline.rs::train()` to (a) load teacher cache, (b) iterate corpus batches alongside cached teacher logits, (c) call `forward_backward_kd_batch`, (d) optimizer step.
3. Unit test: 10-step run on a 3-layer toy student + 4-layer toy teacher, verify loss strictly decreases.

**Effort estimate**: 16-24 hours engineering. No compute beyond the unit test.

**Falsifier**: F-DISTILL-KD-001 — student loss after 100 steps < student loss at step 0 (sanity); F-DISTILL-KD-002 — student logits cosine sim to teacher logits increases monotonically over the first 1000 steps (modulo noise).

### Phase 3 — End-to-end smoke (~1 day + 4h compute)

**Goal**: Run `apr distill` for 500 steps on a 10K-batch subset of the qwen-v3 corpus, with the MODEL-1 teacher cache. Verify it doesn't crash, val_loss decreases, output passes `apr qa`.

**Deliverables**:
1. `apr distill --teacher paiml/qwen2.5-coder-7b-apache-q4k-v1 --init Qwen/Qwen2.5-Coder-0.5B-Instruct --dataset qwen-v3-10k --steps 500 --out runs/distill-smoke/` — end-to-end command.
2. Smoke evidence in `evidence/distill-smoke-<date>/` — launch.log, per-step metrics, final val_loss.
3. Spec amendment to ship-model-2-spec.md §85 documenting the smoke.

**Effort estimate**: 8h engineering + 4h gx10 compute.

**Falsifier**: F-DISTILL-SMOKE-001 — val_loss at step 500 < val_loss at step 0.

### Phase 4 — Distillation training run for the v2 ship (~10 days compute, mostly unattended)

**Goal**: Long-running distillation training to produce the v2 student checkpoint.

**Hyperparameters** (Chinchilla-adjacent, tuned from §82 P2-A):
- Init: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (same as v1)
- Teacher: `paiml/qwen2.5-coder-7b-apache-q4k-v1` (Q4_K, our distilled teacher)
- Corpus: qwen-v3 (1.24B tokens, the §77 5g.1 corpus) — full pass for the first time
- Steps: ~50K (≈ 1.6B tokens consumed at batch=16 seq=512 = 8192 tokens/step)
- LR: 1.5e-5 → 5e-7 cosine
- T (KD temperature): 4.0 (DistilBERT default; tune to 2.0 if loss curve plateaus early)
- α (CE weight): 0.3 (per Hinton et al. 2015 — KD signal dominates when teacher is reliable)
- Wall time on RTX 4090: 50K steps × ~1.5 s/step (student forward+backward + teacher cache read) ≈ 21 hours active training. With teacher cache prep + checkpointing overhead ≈ 30 hours total.

**Deliverables**:
1. Best checkpoint `runs/albor-370m-v2/ckpt/epoch-N.apr`.
2. Per-epoch eval against `apr eval humaneval` (sample 20 problems mid-run, full 164 at end).
3. Spec amendment §86 (re-using the deprecated §86 slot) documenting the v2 run.

**Effort estimate**: 4h dispatch + setup + 30h compute (unattended on gx10 or lambda-labs).

**Falsifier**: F-DISTILL-V2-001 — best val_loss < 3.0 (would unblock PMAT-684 HumanEval per the audit Rec #3); F-DISTILL-V2-002 — HumanEval pass@1 ≥ 15% (loose first-target).

### Phase 5 — HumanEval discharge (PMAT-684, ~1 day)

**Goal**: Run full 164-problem HumanEval on the v2 best checkpoint, verify pass@1 against the target.

**Deliverables**:
1. `apr eval humaneval --model runs/albor-370m-v2/ckpt/epoch-N.apr --samples 164 --temperature 0.2 --top-p 0.95 --gpu cuda` — run.
2. Evidence at `evidence/distill-v2-humaneval-<date>/`.
3. Discharge AC-SHIP2-005 in ship-model-2-spec.md.

**Effort estimate**: 5-8h gx10 compute (sequential through 164 problems, 16 samples each for pass@1@k).

**Falsifier**: F-HUMANEVAL-V2-001 — pass@1 ≥ 15% (acceptance threshold; under is "the corpus didn't transfer; need to widen corpus per P2-C lineage"); F-HUMANEVAL-V2-002 — pass@1 ≥ 25% (loose ship-goal threshold).

### Phase 6 — Publish v2 (~3h)

**Goal**: Re-stamp the best checkpoint with provenance + tokenizer (per SPEC-HF-PUBLISH-001) and ship to `paiml/albor-370m-v2` (or `paiml/albor-370m-v1.1.0`).

**Deliverables**: Follows SPEC-HF-PUBLISH-001 end-to-end — `apr stamp --tokenizer` → `apr export --quantize int4` → `apr publish`. With the v0.34.0+1783 (defect 6) binary, everything is autodetected from the staging directory. Three-path verification: `apr run` + HF Transformers + llama-cli.

**Effort estimate**: 3h staging + 1h compute (Q4_K export + upload).

## Total effort + timeline

| Phase | Engineering | Compute | Calendar |
|---|---|---|---|
| 1: Teacher logits cache | 16-24h | 4-8h | 3 days |
| 2: KD-wired forward+backward | 16-24h | <1h | 2 days |
| 3: E2E smoke (500 steps) | 8h | 4h | 1 day |
| 4: V2 training run | 4h | 30h (unattended) | 2-3 days wall |
| 5: HumanEval discharge | 4h | 5-8h | 1 day |
| 6: Publish v2 | 3h | 1h | 0.5 day |
| **Total** | **~70h** | **~45h** | **~10 days** |

This matches PMAT-683's "16-40h. Δship +10. P=25%" original estimate when scaled to the realistic full epic; PMAT-683 was authored before §82 P2-A landed the trainer infrastructure that Phase 2 reuses.

## Risk register

| Risk | Mitigation |
|---|---|
| Teacher logits cache is too large (~100 GB for full qwen-v3) | Top-K=64 sparsification reduces by ~25× → ~4 GB. Stream from disk per batch. |
| Student forward+backward + KD has unmeasured CUDA bugs that only surface at scale | Phase 3 smoke catches anything that surfaces in the first 500 steps. Phase 4 has per-epoch sanity (val_loss monotone + checkpoint preserved). |
| HumanEval pass@1 falls below 15% at end of Phase 4 → ship is blocked | Two-path fallback: (a) widen corpus per the P2-C lineage (4× corpus), retrain Phase 4 from scratch; (b) drop KD temperature from 4.0 → 2.0 to sharpen teacher distribution. Both add ~1 week each. |
| KD loss is mathematically wrong or numerically unstable (e.g., logsumexp drift, FP16 overflow on KL) | Phase 2 unit test compares against a Python reference (PyTorch nn.KLDivLoss) within 1e-4 absolute error. |
| Realizar inference of the 7B teacher mid-training is too slow (~5s/batch) — burns 50K × 5s ≈ 70h of teacher time on top of training | Phase 1 cache amortizes this once. The cache prep itself takes ~6h, but reads at <100ms/batch during training. |

## Open architectural decisions

1. **Top-K cache vs full logits**: top-K=64 trades fidelity for storage. KD literature (DistilBERT, Distil-Qwen) shows pass@1 deltas from K=32 to K=full are within noise for code corpora. We start with K=64; if Phase 3 smoke shows loss-curve degradation vs a comparison full-logits batch, bump K.
2. **Online teacher vs cached**: cached (decided). Online would let temperature/α be tuned mid-training without re-prep cost; cached requires re-prep per α/T change. The hyperparameter sweep cost is < single cache regen, so cache wins.
3. **Distillation algorithm variant**: vanilla KD (Hinton 2015), NOT MiniLM (which adds attention-distillation losses) or TinyBERT (intermediate-layer matching). MODEL-1 is a Q4_K teacher — its FP16 intermediate activations aren't recoverable post-quantization. Vanilla KD on logits-only is the practical match.
4. **Tokenizer**: same Qwen2 vocab (151,936 tokens) as v1 — teacher + student share vocab, no token-alignment loss. This is the strongest argument for using `Qwen2.5-Coder-0.5B-Instruct` as init: matched tokenizer + matched arch family.

## Acceptance criteria

| ID | Criterion | Phase | Status |
|---|---|---|---|
| AC-DISTILL-001 | `apr distill prepare` end-to-end cache write + read-back parity ≥ 0.999 cos sim | Phase 1 | planned |
| AC-DISTILL-002 | `apr distill` runs 500 steps without crash; val_loss monotone | Phase 3 | planned |
| AC-DISTILL-003 | Phase 4 best val_loss < 3.0 | Phase 4 | planned |
| AC-DISTILL-004 | Phase 5 HumanEval pass@1 ≥ 15% (acceptance); aspire to ≥ 25% (ship-goal) | Phase 5 | planned |
| AC-DISTILL-005 | Phase 6 publish at `paiml/albor-370m-v2` (or v1.1.0) per SPEC-HF-PUBLISH-001 with all 3 usage paths verified | Phase 6 | planned |

## Cross-references

- Parent: [ship-model-2-spec.md §84.5](./ship-model-2-spec.md) — "What's next for MODEL-2"
- Predecessor: SPEC-SHIP-MODEL-2 §35 (`apr distill` stub identification, 2026-04-28), §82 (P2-A trainer infrastructure proven, 2026-05-15)
- Companion: [SPEC-HF-PUBLISH-001](./model-hf-publish-pipeline-spec.md) — used for Phase 6 publish
- Roadmap: [docs/roadmaps/roadmap.yaml](../../roadmaps/roadmap.yaml) PMAT-683 + PMAT-684
- Teacher: [`paiml/qwen2.5-coder-7b-apache-q4k-v1`](https://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1) (MODEL-1, HumanEval=86.59%)
- Init: [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) (Apache-2.0, same arch + vocab as student)
- Audit: [audits/q4k-shape-swap-impact.md](../audits/q4k-shape-swap-impact.md) — confirms teacher's Q4_K artifact is bit-correct (no re-export needed)

## Changelog

- **1.0.0 (2026-05-18)** — Initial publish. Scopes the 6-phase plan opening the distillation track that picks up MODEL-2 v2 from where the §88 stack-existence-proof ship left off.
