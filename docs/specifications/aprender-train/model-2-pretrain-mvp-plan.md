---
spec_id: SPEC-MODEL-2-PRETRAIN-MVP
version: 1.0.0
status: DRAFT
tracks: task #111 (MODEL-2 pretrain real-corpus MVP)
created: 2026-04-18
---

# SPEC-MODEL-2-PRETRAIN-MVP: Real-corpus + real-checkpoint MVP

## Motivation

Task #105 landed `apr pretrain` with GATE-TRAIN-005 / -007 / -008 wired, but the
CLI guard `if !synthetic { return Err(...) }` (`apr-cli/src/commands/pretrain.rs:59-66`)
rejects every invocation that is not a synthetic decreasing-loss drive. The real
pretraining machinery is absent: no forward pass, no corpus reader, no checkpoint
writes.

This spec is the MVP path from "gates verified in isolation" to "one real 370M
forward+backward step + one real `.apr` checkpoint on disk".

## Non-goals (explicitly deferred)

- Async H2D down-weight streaming (task #24)
- Full `apr-corpus-ingest run`: HF pull, license-detector, PII scrub, MinHash-LSH
  dedup, deterministic train/val split, provenance manifest
- Mixed-precision `GradScaler` tuning
- Distributed / tensor-parallel / ZeRO (the crates exist but are not wired)
- Convergence to val_loss ≤ 2.20 (compute-budget ticket, not correctness)
- Checkpoint-resume round-trip — MVP is **write-only**
- Real `nvml` GPU util telemetry — MVP keeps the seeded jitter
- `apr qa` post-hoc validators (GATE-TRAIN-001 evidence, GATE-TRAIN-002 scan)

## Architectural invariants (do not rewrite)

1. `Llama370MConfig::{HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, INTERMEDIATE,
   VOCAB_SIZE, RMS_NORM_EPS}` are the source-of-truth model dims. Instantiate via
   `aprender_train::transformer::Transformer::new(&cfg)` with those dims — do
   **not** write a new `Llama370M` struct.
2. `PretrainLoop` is model-agnostic. Its gate calls
   (`check_non_divergence`, `check_numerical_stability`, `validate_finite`) fire
   automatically on any `StepFn` / `ValFn`. No gate re-wiring.
3. `AdamW` (`optim/adamw.rs`) is the optimizer; scheduler is
   `WarmupCosineDecayLR`. Both are drop-in.
4. `AprWriter` (`format/apr/mod.rs:412`) writes the checkpoint format; `save_apr`
   (`io/save.rs:161`) serializes a `Model`.

## MVP edit list (7 items)

### 1. Real `StepFn` / `ValFn` implementors
- File: `crates/aprender-train/src/train/pretrain_real.rs` (**new**)
- `struct RealStepFn { trainer: TransformerTrainer, batches: Box<dyn Iterator<Item = LMBatch>> }`
  - `step()` pulls one `LMBatch`, `trainer.model.forward(input_ids)`,
    `CausalLMLoss::forward`, backward, `clip_grad_norm`, `AdamW::step`.
  - Returns `(loss_f32, grad_norm_f32)`.
- `struct RealValFn { held_out: Vec<LMBatch>, model_ref: &Model }`
  - Forward-only + loss; returns mean val_loss.

### 2. Build Transformer at 370M dims
- File: `crates/aprender-train/src/train/pretrain_real.rs`
- `TransformerConfig` field-for-field from `Llama370MConfig::*`.
- `debug_assert_eq!` on `Transformer.parameters().iter().map(|t| t.len()).sum()` within
  the INV-ARCH-370M-001 band.

### 3. Minimal shard reader
- File: `crates/aprender-train/src/train/shard_reader.rs` (**new**)
- Reads file of LE u32 tokens.
- Yields fixed-length `seq_length+1` sequences.
- Wraps in `LMBatch::from_sequences`.
- No MinHash, no license filter, no PII scrub — those belong to
  `apr-corpus-ingest run`.

### 4. Swap checkpoint format to APR
- File: `crates/aprender-train/src/train/transformer_trainer/trainer.rs:519`
- Current: hardcodes `ModelFormat::SafeTensors`.
- Change: accept `ModelFormat` param (or add `save_apr(path: &Path)`).
- Wire new per-epoch hook in `PretrainLoop::run_epoch` that, after
  `check_non_divergence` passes (`pretrain.rs:534`), calls
  `trainer.save_apr(&artifact.checkpoint_path, ...)` and writes `artifact.metadata`
  as JSON to `artifact.metadata_path`.

### 5. Update `apr pretrain` CLI
- File: `crates/apr-cli/src/commands/pretrain.rs:59-66`
- Remove the `if !synthetic { return Err(...) }` guard.
- Branch on `synthetic`: when false, build `RealStepFn` + `RealValFn` from
  `dataset`, `tokenizer`, and the Transformer config.
- Keep synthetic path intact — the gate reproducibility tests depend on it.

### 6. Real optimizer-state sha256
- File: `crates/aprender-train/src/train/pretrain.rs:615`
- Replace `fake_optimizer_sha` with sha256 over `AdamW` `m` / `v` / `t` buffers.
- Discharges INV-TRAIN-003.

### 7. Per-epoch checkpoint hook
- Files: `pretrain.rs:534` (post-gate-pass), `io/save.rs`
- After `check_non_divergence` passes, write `.apr` to `{run_dir}/ckpt/epoch-{N:03d}.apr`
  per `contracts/training-loop-pretrain-v1.yaml`.
- Write `metadata.json` sidecar.

## Critical Files (five nodes, one spec)

| File | Role |
|------|------|
| `crates/aprender-train/src/train/pretrain.rs` | gate machinery (do not touch gate fns) |
| `crates/apr-cli/src/commands/pretrain.rs` | CLI guard removal |
| `crates/aprender-train/src/train/transformer_trainer/trainer.rs` | SafeTensors → APR swap |
| `crates/aprender-train/src/transformer/model.rs` | Transformer::new consumer |
| `crates/aprender-train/src/io/save.rs` | `save_apr` already exists |

## Acceptance (binary pass/fail)

- AC-111-001: `apr pretrain --no-synthetic ...` runs end-to-end without panic
- AC-111-002: One real `.apr` file exists at `{run_dir}/ckpt/epoch-000.apr` after 1 epoch
- AC-111-003: `metadata.json` sidecar present; optimizer-state-sha matches AdamW buffers
- AC-111-004: `check_non_divergence` still fires (fail path: inject NaN → exit non-zero)
- AC-111-005: Synthetic path `apr pretrain --synthetic` STILL PASSES (regression guard on #105)

## Host assignment

- **Impl/test**: lambda-labs (this host) — 4090, any scale
- **Smoke test at 8GB**: yoga (4060 Laptop) — 370M fits
- **Parity check**: gx10 aarch64 — separate compute path if desired

## Source

Implementation plan produced 2026-04-18 by Plan agent
`afd391d1eb1395d30` against commit `9209383da` (main, post-#882-merge).
