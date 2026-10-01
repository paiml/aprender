---
phase: 05-benchmark-and-claims-gate
plan: 06
subsystem: testing
tags: [lora, setfit, selection-manifest, apr, classify, benchmark, clap, autograd]

requires:
  - phase: 02-deterministic-pair-and-data-protocol
    provides: "SelectionManifest / Selection::replay / PreparedDataset<Canonical> — the manifest→Selection door this plan plumbs into `apr finetune`"
  - phase: 04-apr-artifact-and-production-parity
    provides: "commands/eval/setfit.rs's three-call attested ingest, copied verbatim as the D-10 door"
provides:
  - "`--selection-manifest`, `--seed`, `--val-split`, `--early-stopping-patience` on `apr finetune --task classify`"
  - "`run_classify_core` — one shared classify implementation for the CLI flag path and 05-09's bench caller"
  - "`TrainResult.epochs_completed` — the field a LoRA row attests `epochs_completed == epochs_requested` from"
  - "val_split 0.0 and early_stopping_patience 0 as real, supported values (A6 resolved by measurement)"
  - "`ClassifyPipeline::load_adapter` — the strict, never-partial adapter reload route"
  - "`ClassifyPipeline::predict_proba_tokenized` — ordered K-length probability vector"
  - "`Transformer::save_apr` — base-model persistence, and `base_model_bytes` for 05-09"
  - "`ClassifyPipeline::from_model` — attach a head to an APR-loaded base without a sibling tokenizer"
affects: [05-09 bench run, 05-10 claims gate, 05-11 the 40 remote 9B cells]

actuals:
  tokens: 30500
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Read-everything-then-install: a loader validates every tensor before mutating any, so a refusal leaves the object untouched"
    - "Self-spawning fresh-process test: re-exec `current_exe` with a marker env var, reap `ExitStatus` from `Command::output`, never read a status through a pipe"
    - "Two-sided control: assert the artifact CHANGES the output as well as that the reload reproduces it, so the fidelity assertion cannot be tautological"
    - "Encode a pre-existing defect as an asserting test rather than a comment, so a future fix fails loudly"

key-files:
  created:
    - crates/aprender-train/src/finetune/classify_reload_tests.rs
    - crates/apr-cli/src/commands/finetune_selection_tests.rs
    - .planning/phases/05-benchmark-and-claims-gate/deferred-items.md
  modified:
    - crates/apr-cli/src/model_ops_commands.rs
    - crates/apr-cli/src/dispatch.rs
    - crates/apr-cli/src/commands/finetune.rs
    - crates/aprender-train/src/finetune/classify_trainer.rs
    - crates/aprender-train/src/finetune/classify_pipeline/mod.rs
    - crates/aprender-train/src/finetune/classify_pipeline/training.rs
    - crates/aprender-train/src/transformer/model.rs

key-decisions:
  - "A6 is FALSE as recorded: the trainer refused val_split 0.0 in three independent places. All three patched surgically; a negative split is still refused, so the bound is guarded where it is actually invalid."
  - "patience 0 now means DISABLED. Read literally the old code made it 'stop after one epoch' — the opposite of the intent."
  - "With validation disabled no `best/` checkpoint is written. That directory, not save_every, is the real model-selection surface."
  - "The new flags travel as one `ClassifyOverrides` struct, not four more positional parameters on a 30-argument signature."
  - "`load_adapter` reads and shape-checks every tensor before installing any; a refusal leaves the pipeline untouched. A partial adapter classifies confidently and is not the model that was trained."
  - "The preflight stamps non-zero LoRA weights before writing, because CPU training leaves them at init — otherwise the round trip would be satisfied by a loader that read only the classifier head."

patterns-established:
  - "Falsify the assertion, do not just watch it pass: the fidelity check was mutated (delete the child's load_adapter) and confirmed RED at exactly the control gap"
  - "A spawned test leg must prove it RAN — the parent requires `1 passed` in the child's output after an earlier `--exact` filter selected nothing and exited 0"

requirements-completed: [EVAL-02, EVAL-05]

coverage:
  - id: D1
    description: "`apr finetune --task classify --selection-manifest` resolves rows through the Phase 2 manifest→Selection door and refuses on hash mismatch (D-10)"
    requirement: "EVAL-02"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/finetune_selection_tests.rs#attested::a_valid_manifest_resolves_the_selected_rows_in_selection_order"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/finetune_selection_tests.rs#attested::a_doctored_manifest_byte_refuses_on_the_envelope_digest"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/finetune_selection_tests.rs#attested::a_manifest_from_another_preparation_refuses_against_this_data"
        status: pass
    human_judgment: false
  - id: D2
    description: "seed / val_split / early_stopping_patience are controllable per cell, with defaults byte-compatible with the pre-D-10 hardcode block"
    requirement: "EVAL-02"
    verification:
      - kind: unit
        ref: "crates/apr-cli/src/commands/finetune_selection_tests.rs#absent_flags_resolve_to_the_pre_d10_hardcode_block"
        status: pass
      - kind: unit
        ref: "crates/apr-cli/src/commands/finetune_selection_tests.rs#each_flag_overrides_exactly_its_own_field"
        status: pass
    human_judgment: false
  - id: D3
    description: "ClassifyTrainer tolerates val_split 0.0 with early stopping disabled, trains every requested epoch, and exposes epochs_completed (A6 probed, not assumed)"
    requirement: "EVAL-02"
    verification:
      - kind: unit
        ref: "crates/aprender-train/src/finetune/classify_trainer_tests.rs#a6_probe_val_split_zero_and_patience_zero_trains_every_requested_epoch"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/finetune/classify_trainer_tests.rs#a6_validation_disabled_writes_no_best_checkpoint"
        status: pass
    human_judgment: false
  - id: D4
    description: "The LoRA save → fresh-process reload → ordered probability-vector route is demonstrated end-to-end, with a two-sided control proving the adapter is applied"
    requirement: "EVAL-05"
    verification:
      - kind: integration
        ref: "crates/aprender-train/src/finetune/classify_reload_tests.rs#lora_adapter_reload_yields_ordered_probability_vector"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/finetune/classify_reload_tests.rs#classify_reload_installs_the_lora_tensors_not_only_the_head"
        status: pass
    human_judgment: false
  - id: D5
    description: "CPU classify training moves the classification head and NOT the LoRA adapters (pre-existing defect, surfaced and encoded, not fixed)"
    verification:
      - kind: unit
        ref: "crates/aprender-train/src/finetune/classify_reload_tests.rs#cpu_training_moves_the_head_but_not_the_lora_adapters_pre_existing"
        status: pass
    human_judgment: true
    rationale: "The test proves the CURRENT behaviour. Whether the 9B GPU cells (D-09, lambda-vector) are affected is a different code path this CPU test cannot answer, and the decision to run the LoRA matrix anyway is a human call about what the baseline means."

duration: 88min
completed: 2026-08-16
status: complete
---

# Phase 5 Plan 06: LoRA Selection-Manifest Input and Reload Tracer Summary

**`apr finetune --task classify` now reads its rows through the Phase 2 manifest→`Selection` door with seed/split/early-stopping under per-cell control, and the LoRA save → fresh-process reload → ordered probability-vector route is demonstrated end-to-end — which is also how we found that CPU classify training never touches the LoRA adapters.**

## Performance

- **Duration:** 88 min
- **Started:** 2026-08-17T03:42:44Z
- **Completed:** 2026-08-17T05:10:49Z
- **Tasks:** 3
- **Files modified:** 12 (3 created, 9 modified)

## Accomplishments

- **D-10 is structural, not a promise.** `--selection-manifest` runs the same three calls `commands/eval/setfit.rs:227-237` makes — `read_attested_canonical`, `read_selection_manifest`, `Selection::replay` — and restricts training to the replayed rows *in selection order*. Both methods now read one artifact; EVAL-02's identical-sampled-ID guarantee no longer depends on an exporter being correct.
- **The three hardcodes are gone as literals** and exist as named constants used only as the flag-else fallback. `resolve_training_config` is a pure function, tested against the *literal* values the old block contained (42 / 0.2 / 5 / 10) rather than against the constants — a test quoting the constants would pass no matter what they became.
- **A6 was false, in three independent places.** Measured, not assumed. See "Issues Encountered".
- **The LoRA reload route exists now — it did not before.** Probe finding recorded verbatim below.
- **The tracer found a defect the benchmark needs to know about**: on CPU, classify training moves the classification head and nothing else.

## Task Commits

1. **Task 1: CLI args + manifest→Selection ingest + shared core entry** — `2eff9fa9e` (feat)
2. **Task 2: A6 probe, epochs_completed exposure, refusal tests** — `83840b625` (fix)
3. **Task 3: LoRA reload preflight (tracer)** — `9fc474f5a` (feat)

## The Task 3 probe finding, stated literally

**No adapter load route existed.** Recorded as the plan required:

- `ClassifyPipeline::from_apr` loads the base transformer and then calls `build_lora_layers`, producing **fresh** adapters. The trained ones are never read.
- The only reader of the classify adapter format was `ClassifyTrainer::resume_from_apr_checkpoint`, unusable here for three reasons: it needs a corpus to construct a trainer, it verifies the checkpoint's `data_hash` against that corpus, and it installs LoRA tensors through `if let Ok(..)` — a **silent partial load by construction**.
- `instruct_pipeline`'s `crate::lora::load_adapter_peft` + `inject_adapter_weights` reads PEFT `adapter_model.safetensors`, which the classify writer does not produce, carries no classifier head, and `continue`s past every unmatched tensor.

So `ClassifyPipeline::load_adapter` was added, along with `predict_proba_tokenized`, `Transformer::save_apr` and `ClassifyPipeline::from_model`.

### The probability-vector entry, exact signature

```rust
pub fn predict_proba_tokenized(&mut self, token_ids: &[u32]) -> Vec<f32>
```

Returns `num_classes` softmax probabilities in the trainer's class-index order. It shares a new `head_logits` helper with `forward_only`, so there is one forward pass in the file rather than two that can drift (OPS-03).

### Measured numbers

| Quantity | Measured | Bound |
|---|---|---|
| `base_model_bytes` (`base.apr`) | 783,236 | — |
| `artifact_bytes` (`model.adapter.apr`) | 10,180 | — |
| `deployable_total_bytes` | 793,416 | — |
| Fresh-process vs in-process, max elementwise \|diff\| | **0.000000000** | < 1e-4 |
| With-adapter vs without-adapter, max elementwise \|diff\| | **0.194929659** | > 1e-3 |

The artifact set the reload consumed: `base.apr` (written by `Transformer::save_apr`) and `model.adapter.apr` (the production artifact `ClassifyTrainer::save_checkpoint` already writes — the preflight reloads that, not a special one).

Fidelity is exactly 0.0 because APR stores f32 without quantization and both legs run identical ops on the same machine.

### The fidelity assertion was falsified, not merely watched

Deleting the child's `load_adapter` call turns it **RED at 0.19492966** — exactly the control gap, as predicted. Reverted after confirming.

An earlier version of the child leg passed `--exact` a bare function name, which selected **no test**; the child exited 0 having run nothing. The marker-line assertion caught it. The filter is now derived from `module_path!`, and the parent additionally requires `1 passed` in the child's output (CR-02 vacuity floor).

## Files Created/Modified

- `crates/apr-cli/src/model_ops_commands.rs` — four new `Finetune` args with refusal semantics in the doc comments
- `crates/apr-cli/src/dispatch.rs` — builds `ClassifyOverrides` and threads it through
- `crates/apr-cli/src/commands/finetune.rs` — `ClassifyOverrides`, `ClassifyRun`, `ClassifySelection`, `run_classify_core`, `resolve_training_config`, `resolve_classify_selection`, `resolve_selected_samples`
- `crates/apr-cli/src/commands/finetune_selection_tests.rs` — 12 tests (flag validation, defaults preservation, row resolution, attested end-to-end refusals)
- `crates/aprender-train/src/finetune/classify_trainer.rs` — A6 patch + `TrainResult.epochs_completed`
- `crates/aprender-train/src/finetune/classify_pipeline/mod.rs` — `from_model`, `load_adapter`, `predict_proba_tokenized`, `softmax_stable`
- `crates/aprender-train/src/finetune/classify_pipeline/training.rs` — `head_logits` extracted from `forward_only`
- `crates/aprender-train/src/transformer/model.rs` — `Transformer::save_apr`
- `crates/aprender-train/src/finetune/classify_reload_tests.rs` — the 7-test preflight
- `.planning/phases/05-benchmark-and-claims-gate/deferred-items.md` — out-of-scope failures

## Decisions Made

- **One `ClassifyOverrides` struct, not four more positional parameters.** `finetune::run` already carried 30 arguments; 05-09's bench caller constructs the struct directly rather than re-deriving defaults.
- **`save_every` moves to `epochs` when early stopping is disabled**, but the load-bearing guarantee is that **no `best/` checkpoint is written when validation is off**. That directory is the actual model-selection surface; feeding it a constant 0.0 placeholder loss would have selected epoch 0 in every cell.
- **`resolve_selected_samples` takes plain slices, not a `Selection`.** `Selection` has no public constructor by design, so a function taking one cannot be unit-tested. Taking `&[(&str, usize)]` makes the missing-id refusal directly testable while keeping the production call identical.
- **The preflight stamps non-zero LoRA weights before writing.** Forced by the finding below: with adapters left at initialization (B = 0), a round trip over an untouched adapter is satisfied by a loader that reads only the classifier head. Stamping makes the probability vectors depend on the LoRA tensors, so the fidelity assertion can only pass if they were reloaded too.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `ClassifyTrainer` rejected `val_split 0.0` in three places**
- **Found during:** Task 2 (A6 probe)
- **Issue:** A6 assumed the trainer tolerated `val_split 0.0`. It did not. (a) `new()` refused `val_split <= 0.0`. (b) `split_dataset` forced `val_count >= 1`, so a 0.0 request still held one example out of every cell. (c) `epochs_without_improvement >= patience` is true at the end of epoch 0 when patience is 0 — so "disabled" meant "stop after one epoch".
- **Fix:** Bound moved to `[0.0, 0.5]` (negative still refused); `split_dataset` returns `(all, empty)` at ratio ≤ 0; early stopping gated on `patience > 0`; validation pass and `best/` checkpoint skipped when there are no validation rows.
- **Files modified:** `crates/aprender-train/src/finetune/classify_trainer.rs`
- **Verification:** 4 A6 tests + a best-checkpoint suppression test; 330 `classify_trainer` tests green.
- **Committed in:** `83840b625`

**2. [Rule 2 - Missing Critical] Two pre-existing tests asserted the OLD val_split contract**
- **Found during:** Task 2
- **Issue:** `test_ssc026_invalid_val_split_zero` and `test_trainer_new_val_split_zero` asserted that 0.0 is an error — the behaviour deliberately changed.
- **Fix:** Repointed to the new contract and **paired each with a negative-split refusal**, so the bound stays guarded at the value that is actually invalid. The deliberate change is recorded in-file at both sites.
- **Files modified:** `crates/aprender-train/src/finetune/classify_trainer_tests.rs`
- **Committed in:** `83840b625`

**3. [Rule 3 - Blocking] `Transformer` had no way to persist a base model**
- **Found during:** Task 3
- **Issue:** A fresh-process reload requires the base to BE a file, and `Transformer::new` produces weights that live only in that process. `ClassifyPipeline::from_apr` additionally demands a sibling `tokenizer.json`, which a byte-level toy does not have.
- **Fix:** Added `Transformer::save_apr` (the exact inverse of `from_apr`; optional tensors written only when present) and `ClassifyPipeline::from_model` (`new`'s body, so the CUDA/wgpu/NF4 init ladder has one implementation).
- **Files modified:** `crates/aprender-train/src/transformer/model.rs`, `crates/aprender-train/src/finetune/classify_pipeline/mod.rs`
- **Verification:** Both exercised by all 7 reload tests; `base.apr` round-trips to bit-exact probabilities.
- **Committed in:** `9fc474f5a`

---

**Total deviations:** 3 auto-fixed (1 bug, 1 missing critical, 1 blocking)
**Impact on plan:** All three were required to execute the plan as written — deviation 1 *is* the A6 measurement the plan asked for. No scope creep; `save_apr` is additive and needed again by 05-09 for `base_model_bytes`.

## Issues Encountered

### FINDING (HIGH, pre-existing, NOT fixed): CPU classify training never touches the LoRA adapters

The tracer's LoRA-specific control failed with `max |diff| A=0, B=0`. Diagnosed rather than guessed:

1. A direct in-memory probe showed the LoRA tensors are byte-identical before and after training, while the classification head moves. So the loader was not at fault.
2. The mechanism is a **graph cut in `ClassificationHead::mean_pool`** (`finetune/classification.rs:149): it returns `Tensor::from_vec(pooled, requires_grad)` — a *new* tensor carrying no backward op back to `hidden_states`. The backward started at `logits` terminates at the pooled vector and never enters the transformer.

Consequences:
- `forward_hidden_with_lora` **does** apply the adapters in the forward pass, so they are not inert by design — they simply never receive gradient. And because LoRA B is zero-initialized, their forward contribution after CPU training is exactly zero. On CPU, `apr finetune --task classify` is a **linear probe on a frozen encoder**, not a LoRA fine-tune.
- Not fixed here: repairing it means a differentiable pooling plus a CPU transformer backward — architectural, several times this plan's size, and owned by whoever owns the CPU training path.
- Encoded as an **asserting** test, `cpu_training_moves_the_head_but_not_the_lora_adapters_pre_existing`, with a positive half (the head must move) so it cannot pass by everything being frozen. A future fix fails it loudly and says exactly what changed.

**Scope for Phase 5:** the 9B cells run on the lambda-vector GPU (D-09), where `gpu_training` is `Some` and `backward_gpu_blocks` / `backward_nf4_gpu_blocks` run an explicit transformer backward. **Whether that path trains adapters is a different question this CPU test does not answer and must not be read as answering.** Flagged for 05-09/05-11 below.

### Out of scope: 24 pre-existing failures in the full `aprender-train` lib suite

`gpu::ledger` (12), `gpu::guard` (8), `gpu::wait` (1), `prune::snapshot_tests` (3). Reproduced in isolation (so not a parallel-execution interaction with the new tests), in modules this plan does not touch, failing on VRAM-reservation figures on a host with no reservable GPU ledger state. Logged to `deferred-items.md`; needs its own ticket.

### Self-inflicted: a careless `git checkout --` discarded Task 1's `finetune.rs` work

Run to remove a temporary mutation probe. Only that one file was affected (the other two Task-1 files were already correct on disk) and the edits were re-applied from context. Recorded because the lesson generalizes: **never `git checkout --` a file holding uncommitted task work** — remove the probe with a targeted edit instead.

## Threat Flags

None. The plan's `<threat_model>` covers every surface this plan touched; T-05-06-01 (manifest tampering), T-05-06-02 (unlocked model selection), T-05-06-04 (which rows trained) and T-05-06-05 (silently-unapplied adapter) each have a passing test named in `coverage:` above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

**Ready:**
- 05-09's bench caller has `run_classify_core` + `ClassifyOverrides` and does not need to re-derive any default.
- The row fields exist: manifest `semantic_hash`, `epochs_completed`, `base_model_bytes` / `artifact_bytes` / `deployable_total_bytes`.
- The reload → probability-vector route is proven, so a LoRA `QualityBlock` can be computed from reloaded artifacts rather than in-memory final-epoch state.

**Blocker to resolve BEFORE the 40 remote 9B cells (05-11):**
Confirm that the **GPU** training path actually updates the LoRA adapters. This plan proves it does not on CPU. If the GPU path shares the `mean_pool` graph cut, the "9B LoRA baseline" would be a linear probe on a frozen 9B encoder — a defensible baseline, but not the one the milestone claims to compare against, and the framing in every published number would be wrong. The cheapest check is this file's `cpu_training_moves_the_head_but_not_the_lora_adapters_pre_existing` re-run under `--features cuda` on lambda-vector: if it FAILS there, the GPU path trains adapters and the matrix is safe to schedule.

## Self-Check: PASSED

Files verified present:
- `crates/aprender-train/src/finetune/classify_reload_tests.rs` — FOUND
- `crates/apr-cli/src/commands/finetune_selection_tests.rs` — FOUND
- `.planning/phases/05-benchmark-and-claims-gate/deferred-items.md` — FOUND

Commits verified in `git log`: `2eff9fa9e`, `83840b625`, `9fc474f5a` — all FOUND.

Verification commands, last run:
- `cargo test -p aprender-train --lib classify_trainer` → 330 passed, exit 0
- `cargo test -p aprender-train --lib classify_reload` → 7 passed, exit 0
- `cargo test -p aprender-train --lib finetune` → 1260 passed, exit 0
- `cargo test -p apr-cli --lib --features setfit finetune` → 81 passed, exit 0
- `cargo clippy -p aprender-train --lib` / `-p apr-cli --features setfit --lib` → no findings in changed files

---
*Phase: 05-benchmark-and-claims-gate*
*Completed: 2026-08-16*
