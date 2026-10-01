# Phase 5: Benchmark and Claims Gate - Research

> **RECONCILED 2026-09-08 against `05-CONTEXT.md` D-19 — the 9B LoRA arm is DESCOPED.**
> Every claim below about the 40 LoRA cells, the lambda-vector GPU host, or a
> SetFit-versus-LoRA comparison is **superseded**: the host is unreachable and aprender
> cannot run the Qwen3.5-9B hybrid architecture. EVAL-02/EVAL-04 and the Phase 5 goal were
> amended accordingly; the arm is deferred as `D-ITEM-05-15`. Findings about the SetFit half,
> the numerics substrate, the row schema and the claims gate are UNAFFECTED and still hold.
> This file was reconciled in place rather than regenerated — the descope narrows scope, it
> does not invalidate the surviving research.


**Researched:** 2026-08-16
**Domain:** SetFit-vs-LoRA benchmark claims layer (calibration unblock, paired statistics, selection-safe row/manifest gate) — pure in-repo Rust + pinned Python fixture env
**Confidence:** HIGH (all load-bearing claims verified by reading the shipped code and contracts in this session; the few [ASSUMED] items are listed in the Assumptions Log)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### F-10 Calibration Unblock

- **D-01: The calibration run + contract edit is Phase 5's wave-1 gating plan.** Calibrate on the
  production `all-MiniLM-L6-v2`, land the contract edit, and prove one user-produced
  `setfit-apr-v1` exists (the `apr setfit train → inspect → eval → predict` chain that 04-15's
  spawned ladder showed stopping at rung 4) before any benchmark-cell plan depends on it.
  Everything downstream is blocked on this; front-loading surfaces failure earliest.

- **D-02: The new regime entry covers the full benchmark envelope, boundary-measured.** The entry
  lists all 10 contracted seeds and all 4 cell labels (`s{8,16,32,64}` at the production
  epochs/batch), so no benchmark cell can hit `UncalibratedRegime` mid-matrix. Thresholds are
  measured on the envelope's boundary cells (e.g. s8 and s64 across a seed subset) plus a
  documented margin — the contract records which cells were MEASURED and which are COVERED by
  margin. Measuring all 40 cells (double the training compute) and minimal-then-extend (repeated
  contract edits, mid-benchmark failures possible) were both rejected.

- **D-03: Fresh per-class ε and scale floors are derived from the production measurements, as a
  SECOND regime entry.** The Phase 3 fixture entry stays byte-untouched, so `pv diff` shows a
  purely additive edit and Phase 1–3 gates keep their exact meaning. Reusing fixture ε was
  rejected as the exact non-transfer D-10(c) warns about: sparse embedding-table relative deltas
  shrink as vocabulary grows from the 97-row fixture closure to the full ~30k vocab. The
  derivation follows Phase 3's margin methodology.

- **D-04: The contract edit lands only after a human checkpoint.** The calibration plan marks the
  contract-edit task `autonomous: false`: the executor prepares the edit, runs `pv diff` (two
  filesystem paths — materialize the old revision with `git show` first), presents the measured
  thresholds and the diff, and waits for approval before committing. This matches the 04-11
  CI-patch precedent and honours STATE.md's "open by explicit human ruling".

#### Statistical Claims Protocol

- **D-05: Headline aggregation is mean ± sample (n−1) std with min/max, per cell and per shot
  level.** Matches the SetFit paper's reporting convention so numbers are directly comparable to
  published results; exactly recomputable from rows with closed-form arithmetic (EVAL-04). This
  closes the pre-recorded STATE.md blocker "Choose validation-only calibration and uncertainty
  estimators before collecting benchmark results" together with D-06/D-07.

- **D-06: Uncertainty and deltas are closed-form paired t statistics.** Per shot level: per-seed
  paired deltas (SetFit − LoRA evaluated on the same sampled-ID hash), the mean delta, and a
  Student-t 95% CI over the 10 paired differences. No RNG anywhere in the claims path — EVAL-04's
  "exactly recompute" is bit-level, and a bootstrap would put a resampling stream inside it.
  Stats functions get scipy/sklearn reference fixtures per the `glm_tests.rs:280` house precedent.

- **D-07: Calibration diagnostics are top-label ECE + multiclass Brier, canonical validation
  only.** The existing contract-bound `expected_calibration_error` / `brier_score`
  (`aprender-core/src/calibration.rs`) are binary-only (`&[bool]` labels); this phase extends the
  surface with the standard multiclass pair, bound to explicit ordered labels. Per-class OvR
  suites were rejected as fixture-verification burden the claims don't need.

- **D-08: The report is estimation-first — no significance verdicts.** It states point estimates,
  dispersion, and paired CIs; it never prints "significantly better/worse". This is the PF-007
  guard (few-shot seed sensitivity hiding behind a binary verdict). p-values may sit in the
  machine-readable detail, never in claim language.

#### LoRA Baseline Execution

- **D-09: LoRA cells run on the lambda-vector GPU host (pre-authorized); SetFit cells run CPU.**
  **[SUPERSEDED 2026-09-08 by D-19 — the LoRA half is descoped; only the CPU SetFit half remains.]**
  Every row records its true backend/hardware identity read from execution (Ph4 D-12 — never
  echoed from config). Resource comparisons are presented as as-deployed method costs — which is
  the milestone's actual question (cheap CPU SetFit vs GPU-hungry 9B LoRA) — and never framed as
  same-hardware kernel comparisons. Same-host runs and harness-only deferral were both rejected.

- **D-10: `apr finetune --task classify` gains a `--selection-manifest` input** that resolves rows
  through the same Phase 2 manifest→`Selection` path SetFit uses, refusing to start if the
  manifest hash does not verify. EVAL-02's identical-sampled-ID guarantee becomes structural: both
  methods read one artifact, and each row records the manifest hash it consumed. A wrapper that
  pre-materializes subset files was rejected — identity would then depend on an exporter being
  correct rather than on a shared code path.

- **D-12: Rows are method-tagged, not null-padded.** One versioned row type with a shared
  mandatory core (dataset/model revisions, sampled-ID hash, selection-lock reference,
  backend/hardware identity, quality metrics, bounded resource metrics) plus a method-specific
  evidence block: SetFit rows carry encoder-update evidence + APR artifact hash + lock; LoRA rows
  carry adapter/base-model hashes + training provenance. The completeness gate pairs cells on the
  shared core; a missing required block invalidates the row. Nothing is faked to look
  SetFit-shaped (SAFE-03's spirit), and Phase 4's nullable-path allowlist discipline is preserved.

#### Claims Surface and Gate

- **D-13: The CLI surface is `apr setfit bench run` and `apr setfit bench report`.** `bench run`
  executes one method/shot/seed cell and emits one row; `bench report` recomputes all statistics
  from the stored rows and fails closed on any missing, unmatched, hash-failing, or
  post-test-selected cell. This extends Ph4 D-05's namespace ("an obvious home for future
  subcommands"); Ph4 D-06's generic-only rule covered predict/eval/inspect and does not apply to
  method-comparison logic, which has no generic analog.

- **D-14: Row storage is row-per-file plus a hashed run manifest.** Each cell writes one
  schema-versioned `deny_unknown_fields` JSON row file into a benchmark directory; a manifest
  lists all 80 expected cells with each row's content hash. `bench report` refuses to aggregate
  unless every expected cell is present, hash-verified, and pair-matched on sampled-ID hash.
  Completeness is defined by the manifest, not the directory listing — selective omission is
  structurally visible (EVAL-04). Follows the Phase 2 manifest house style.

- **D-11: `apr qa` stays a generative-model gate — out of scope, with a deferred ticket.**
  Phase 4's audit proved (with a plain-BERT control) that `apr qa` exits 5 on ANY encoder-only
  APR; SetFit is not the cause. Phase 5 documents that scope at the refusal site (a clearer error
  naming `apr validate --quality` / `apr eval` is fair game) and defers the capability extension.

- **D-15: The gate's honesty is contracted.** A new Phase 5 contract (working name
  `setfit-benchmark-claims-v1.yaml`, one-contract-per-phase per Ph1 D-23) owns the row schema,
  completeness rule, pairing rule, and stats equations. In-band negatives — a doctored row set
  with a missing cell, a trimmed cell, a mismatched sampled-ID hash, an edited row failing its
  content hash, and a post-test-selected lock — must each be REFUSED in every `cargo test`
  (Ph1 D-24 / Ph2 D-25 / Ph3 D-08 / Ph4 D-13 discipline).

#### Carried Forward (not re-litigated)

- Shots {8,16,32,64} × ten contracted seeds; the selected-ID manifest is a materialized artifact
  (Ph2 D-08); canonical validation selects, canonical test only after the selection lock (Ph2
  D-16/D-19, Ph3 D-14); the compatibility profile cannot construct a selection run (Ph2 D-19).
- The selection lock is hash-committing with typestate token minting from the verified run object;
  lock-then-tune-then-test INVALIDATES (Ph3 D-14). `apr eval` exercises
  `create_selection_lock → mint_test_token → CanonicalTestAccess::grant` (Ph4 D-16) — this is
  EVAL-04's post-test-selection mechanism; Phase 5 consumes it, never reimplements it.
- The evidence record in every SetFit row is Ph3 D-12's summary + table hash, exactly as the APR
  carries it.
- Only a closed, production-reloaded, parity-verified F32 APR reaches benchmarking (Ph4);
  benchmark predictions come from the RELOADED production artifact, per success criterion 5.
- Frozen probes/centroids run as differently-named baseline types that can never claim SetFit
  (Ph3 D-11, SAFE-03 — proven at compile time).
- Backend identity is read from execution (`ExecutionBackend::identity`), never echoed from
  config (Ph4 D-12). Note the open binding-row defect: `contracts/aprender/binding.yaml` names
  `aprender::setfit::classify::backend_identity`, which does not exist — Phase 5 row work should
  resolve or explicitly re-record that row, not silently inherit it (OPS-04/OPS-06 clause).
  *(Research note: this carried-forward item is STALE — see Finding F-R7 below; 04-19 already
  resolved the row and the registry is 15/15 `implemented` with zero BIND findings.)*
- Contract-resident constants committed before comparison (Ph1 D-14); fixture SHA-256 manifests
  (Ph1 D-13); `pv` only, never bash/yq/python; `pv diff` takes two filesystem paths.
- In-band negatives in every `cargo test`; `cargo-mutants` scoped to new code — but heavy
  mutation baselines are a standalone compute ticket per the Phase 3/4 rulings, not a Phase 5
  must-have.
- Tier wiring: fast gates in tier2, heavy/contract gates in tier3/tier4; a feature-gated surface
  must have a tier and CI job that compiles AND runs it (CR-01 lesson, SAFE-02).
- `unwrap()` banned, `unsafe_code = "forbid"`, typed errors on all fallible paths;
  `CARGO_INCREMENTAL=0` on this host (recurring ~25 GB incremental-cache ENOSPC).
- Branch/PR policy: phases 2–4 ride `gsd/phase-2-contract-gate`; no PR opened yet — opening it is
  the human's call (02-01 policy). The planner must state Phase 5's branch base explicitly.
- GSD state-handler discipline: diff STATE.md/ROADMAP.md after EVERY handler call and revert any
  completion claim the verifier has not earned (6 occurrences of the false-completion defect).

### Claude's Discretion

- **Hyperparameter policy (explicit "you decide"):** whether both methods run frozen published
  defaults across all 40 cells, or get an equal validation-only search budget. Research should
  settle it from what `entrenar::finetune::classify_pipeline::ClassifyConfig`'s defaults actually
  are and what the SetFit reference recipe pins. Default lean: frozen defaults for both (no
  selection to leak, matches the SetFit paper protocol); any tuning must be validation-only with
  selection-lock discipline on BOTH sides, which the LoRA path currently lacks.
- Resource-measurement boundaries: cold/warm latency definitions, warmup/batch boundaries, peak
  memory mechanism — must be contracted and "consistently bounded" (success criterion 3), exact
  protocol is planning's.
- The benchmark directory location, row/manifest file naming, and report output formats (human +
  `--json`).
- How the 40-cell runs are orchestrated (Make targets, a driver subcommand, per-cell resume) and
  how GPU-host rows travel back into the local benchmark directory.
- The exact multiclass Brier/ECE formulations and their reference-fixture sources (sklearn/scipy
  versions), following the Phase 1 hash-locked `uv` environment pattern.
- Where the stats functions live (`entrenar::eval` vs `aprender-core::metrics`) — follow the
  "first consumer, not owner" siting rule.
- Whether `bench run` shells out to existing commands (`apr setfit train`, `apr eval`,
  `apr finetune`) or calls library APIs directly — subject to OPS-03's one-implementation rule.

### Deferred Ideas (OUT OF SCOPE)

- **Extend `apr qa` with an encoder/classifier capability path** (Phase 4 audit item 7; D-11) —
  its own ticket; the refusal-site error message improvement is in scope, the capability is not.
- **Bootstrap/nonparametric robustness columns in the report** — additive later; the closed-form
  paired t is the contracted primary (D-06).
- **Validation-tuned hyperparameter comparison under equal budgets** — only if research (under
  the D-Claude's-discretion item) finds frozen defaults indefensible; requires LoRA-side
  selection-lock plumbing that does not exist today.
- **Per-class one-vs-rest calibration suites** — rejected for v1 (D-07); revisit if a stance
  class shows headline miscalibration.
- **The standing Phase 4 open items that are NOT Phase 5 work:** the mutation-score compute
  ticket, WR-01 (`O_CREAT|O_EXCL` in `atomic_write`), SAFE-02's "in CI" execution (blocked on
  the human-opened PR), the `aprender-contrastive-data` → `apr-cli` publish cascade, and the
  repo-wide deferred defects (D-ITEM-01..04).
- **MCP exposure of benchmark results / models** — future milestone, per the Phase 2 strategic
  frame.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-01 | Evaluate ordered single-label predictions with TweetEval's official `F_avg`, per-class metrics, three-class macro-F1, MCC, confusion matrix, validation-only calibration diagnostics | `MultiClassMetrics`/`ConfusionMatrix`/`f1_avg_for_classes` shipped and sklearn-parity tested (Finding F-R10); `matthews_corrcoef` shipped; binary ECE/Brier contract-bound in `aprender-core/src/calibration.rs`; multiclass top-label ECE + Brier reference formulas + an unfixtured in-repo Brier/ECE precedent in `ClassifyEvalReport` (F-R11); fixture env has sklearn 1.9.0 + scipy 1.18.0 locked |
| EVAL-02 | Run all 4 shots × 10 seeds for both SetFit and 9B LoRA with identical sampled-ID hash per cell | Phase 2 manifest→`Selection::replay` path measured end-to-end (F-R13); `apr finetune --task classify` dispatch entry + `ClassifyConfig`/`TrainingConfig` fields inventoried, incl. the hardcoded `seed: 42`/`val_split: 0.2` that D-10's wiring MUST override (F-R12); 40-cell orchestration + resume options (Pattern 5) |
| EVAL-03 | One machine-readable row per method/shot/seed with revisions, lock, artifact hash, evidence, backend/hardware, quality + resource metrics | `EvalRow` in `commands/eval/setfit.rs` is already labeled "the machine-readable row Phase 5 consumes (EVAL-03 shape)" (F-R8); Ph3 evidence summary + table hash fields located; `ExecutionBackend::identity` + LoRA GPU identity mechanisms located; resource-measurement options audited (Pattern 6) |
| EVAL-04 | Exactly recompute means/dispersion/uncertainty/paired deltas from all 40 stored comparison cells; missing/omitted cells invalidate | Closed-form formulas + frozen t-critical-constant recommendation (Pattern 4); row-per-file + hashed manifest house style measured in `data_contrastive.rs`/`manifest.rs` (F-R13); lock-record consumption mechanics for post-test-selection invalidation (F-R9); sealed-credential constraint for the LoRA side surfaced (F-R14) |
| EVAL-05 | Compare training time, cold/warm latency, throughput with batch/warmup boundaries, peak memory, artifact size, calibration, quality from reloaded production artifacts | `latency_ms` in `ClassifyResponse` (bit-asserted, excluded from `PartialEq`); `apr bench` warmup/iteration precedent; peak-memory mechanism options under `unsafe_code = "forbid"` (Pattern 6); `epoch_time_ms`/`samples_per_sec` in the LoRA train result; artifact size from bytes written |
</phase_requirements>

## Summary

Phase 5 is almost entirely an **in-repo integration phase**: every substrate it needs — metrics,
manifests, locks, the trusted evaluator, the atomic-write house style, a paired t-test, even a
multiclass Brier/ECE precedent — already exists in the tree. No new external Rust dependency is
needed, and the Python reference-fixture environment (scipy 1.18.0, scikit-learn 1.9.0) is already
hash-locked in `scripts/setfit_fixtures/uv.lock`. The research risk is therefore not "what library
do we use" but "which of the many nearby implementations is the contracted one, and where do the
shipped constants fight the plan."

The keystone (D-01..D-04) is mechanically precise and this research measured all of it:
`CALIBRATED_REGIMES` is a compile-time constant at
`crates/aprender-train/src/train/setfit/thresholds.rs:61`, parsed from
`contracts/setfit-train-lifecycle-v1.yaml` (v2.0.0) `equations.calibration_regime.calibrated_regimes`
by the `include_str!`-pinned test `thresholds_match_the_contract` — which **asserts the calibrated
set has exactly ONE entry and is string-equal to the contract's list**, so the deliberate edit is a
three-place synchronized change (contract + Rust constant + test), not a YAML append. Bigger:
the per-class ε table is currently **global** (one `frozen_thresholds` map serves the single
regime), so D-03's "fresh ε as a second regime entry" requires restructuring `Thresholds` to
per-regime tables — a real code change the planner must scope, not a data edit. The calibration
measurement itself has an exact house pattern to copy: the `#[ignore]`d in-crate test
`calibration_matrix_epsilon_basis` (evidence.rs:1593) runs `run_tuning` +
`UpdateEvidence::from_tune_output` **without** the gate (the regime check lives in
`validate_evidence`, at judgement time), so measuring the production encoder needs no gate
widening. The production checkout already exists on this host at
`~/.cache/aprender/minilm-l6-v2-1110a243/` and `SetFitMiniLm::from_pretrained_dir` imports it today.

Two traps found in this session deserve top billing. First, `architecture_fingerprint()` hardcodes
the `minilm-slice-` prefix (encoder.rs:303-311) regardless of whether the encoder is the slice or
the full production model — so the production regime id the code will actually render is
`minilm-slice-h384-l6-a12-i1536-v30522@1110a243|…`, NOT the `minilm-full-…@production` string a
Phase 3 negative test invented. The regime entry must be **copied from a measured run's own
rendered id**, never hand-derived. Second, the LoRA path's `run_classify` hardcodes `seed: 42`
(explicitly NOT a contracted seed — the tweet-eval contract pins "42 is NOT a contracted seed"),
`val_split: 0.2` and early stopping with best-epoch checkpoint selection — an internal random
validation split that is both a reproducibility hole and an uncontrolled model-selection channel.
D-10's wiring must make these explicit per-cell inputs, and the frozen-defaults policy should
disable early stopping (early stopping IS selection).

**Primary recommendation:** Wave 1 = production calibration probe (time one s8 cell first) → full
boundary matrix → three-place contract edit at a human checkpoint → re-run the 04-15 ladder to
completion. Then build `apr setfit bench` on the existing `EvalRow`/lock/manifest machinery, site
multiclass ECE/Brier in `aprender-core::calibration` beside the binary pair, and derive paired-t
CIs from the existing `aprender-core::stats::hypothesis::ttest_rel` shape with a contract-resident
`t_{0.975,df=9}` critical constant (no inverse-CDF needed, no RNG anywhere).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Calibration measurement (production ε basis) | `aprender-train` (in-crate `#[ignore]`d harness) | — | `run_tuning` is `pub(crate)`; the Phase 3 matrix is the exact precedent; measurement must not require widening the gate |
| Regime gate + thresholds | `aprender-train::train::setfit::thresholds` + `contracts/setfit-train-lifecycle-v1.yaml` | `pv` (validation) | The constant and the contract are two halves of one check, pinned by `thresholds_match_the_contract` |
| Multiclass top-label ECE + Brier | `aprender-core::calibration` | fixture env (`scripts/setfit_fixtures`) | Binary pair lives there, contract-bound (`calibration-v1`); "first consumer, not owner" — general capability beside its binary siblings |
| Paired-t statistics + mean±std aggregation | `aprender-core::stats::hypothesis` (extend) or `aprender-train::eval` | contract constants | `ttest_rel` already exists in core stats with scipy-oracle tests; the CI helper is a small extension of the owner module |
| Quality metrics per cell (F_avg, MCC, confusion) | `aprender-train::eval::classification` + `aprender-core::metrics::agreement` | — | Already shipped and sklearn-parity tested; EVAL-01 is assembly |
| Per-cell execution (`bench run`) | `apr-cli` (`commands/` adapter) | libraries above | Filesystem adapter per house rule: no semantics in the CLI; drives library doors |
| Row schema + run manifest + completeness gate | claims contract + a library home (see Open Q3) | `apr-cli` adapter | D-15: the contract owns the schema/rules; adapters only read/write files |
| Statistics recomputation (`bench report`) | `apr-cli` adapter calling library stats | — | Fails closed on manifest/hash/pairing/lock violations before any aggregation |
| SetFit cell training | `aprender-train` lifecycle via `apr setfit train` | `aprender-core` (encoder/head) | Shipped; unblocked by wave 1 |
| LoRA cell training | `entrenar::finetune` (`ClassifyTrainer`) via `apr finetune --task classify` | lambda-vector GPU host | Shipped; needs `--selection-manifest` + explicit `TrainingConfig` control (D-10) |
| Canonical-test gating | `aprender-train::train::setfit::lock` (SetFit side) | claims contract (LoRA side — design needed, F-R14) | Lock/token/grant is credential-sealed to SetFit; LoRA-side discipline must be designed, not inherited |
| GPU-host row transport | orchestration scripts / Make targets | hashed manifest verification on ingest | Rows are hash-committed files; transport must not break verification (D-14) |

## Project Constraints (from CLAUDE.md)

- **Debugging:** use `apr` diagnostic tools first; ALWAYS pin the binary via `. scripts/apr_bin.sh` — never bare `apr`, never a hardcoded path.
- **Code search:** `pmat query`, never grep/glob for discovery (research used targeted greps for exact-symbol location only).
- **Contracts:** `pv` only — never bash/yq/python reimplementations. `pv diff` takes TWO FILESYSTEM PATHS (materialize old revisions with `git show`). If `pv validate` rejects a contract shape, restructure or extend the schema — never work around it.
- **Realizar-first:** inference/serving through realizar; the SetFit row is the documented exception (core owns the verified classify path; serve owns transport only).
- **Verification discipline:** never read `$?` through a pipe; never label runs by intent (prove the mechanism engaged — cite trace lines/hashes); one failing input is an anecdote (vary before naming a cause); guards must scan the surface where the DECISION is made; guard regexes ship must-match/must-not-match case tables; check for shadowed artifacts when a fix seems ineffective.
- **Quality:** `unwrap()` banned (`.clippy.toml` disallowed-methods), `unsafe_code = "forbid"`, complexity ≤10/fn, SATD 0, coverage floor 88%.
- **Coverage + contracts co-evolve** — coverage without contract density improvement is rejected.
- **Git:** `main` protected; feature branches + PR; CI (`ci / gate` + `workspace-test`) must pass. Phases 2–4 ride `gsd/phase-2-contract-gate` (current branch); opening the PR is the human's call.
- **Escalations:** compute > 1 hr on non-lambda-vector hosts, CI workflow edits, publish cascade, force-pushes — human checkpoint first. lambda-vector compute is pre-authorized.
- **Shell:** bashrs-lint scripts; sourced libraries must be option-neutral; `set -euo pipefail` in executables.
- **Publishing safety:** root-anchored ignore patterns; run both CB-510 check scripts after ignore/exclude changes (note: both pass vacuously on macOS — D-ITEM-01).
- **justfile note:** user-global CLAUDE.md prefers justfile, but this repo's convention is the Makefile (no justfile exists; tiered gates, `$(CONTRACTS)`, and 30+ setfit targets live there). Follow the repo.

## Standard Stack

This phase introduces **no new external dependencies**. Everything is in-tree or already locked.

### Core (in-repo)
| Asset | Location | Purpose | Status |
|-------|----------|---------|--------|
| `Thresholds` / `CALIBRATED_REGIMES` / `RegimeCoordinates` | `crates/aprender-train/src/train/setfit/thresholds.rs` | The regime gate D-01..D-04 edit | Verified this session |
| `calibration_matrix_epsilon_basis` | `crates/aprender-train/src/train/setfit/evidence.rs:1593` | THE ε-derivation harness pattern to replicate for production | `#[ignore]`d test; run with `cargo test -p aprender-train --lib --features setfit calibration_matrix -- --ignored --nocapture` |
| `SetFitMiniLm::from_pretrained_dir` | `crates/aprender-core/src/setfit/mod.rs:298` | Production encoder import (ENC-01) | Works today; checkout present on host |
| `MultiClassMetrics`, `ConfusionMatrix`, `f1_average_for_classes` | `crates/aprender-train/src/eval/classification/` | EVAL-01 quality metrics | sklearn-parity tested; `official_f_avg` contract-bound (02-08) |
| `matthews_corrcoef` | `crates/aprender-core/src/metrics/agreement.rs:63` | MCC | sklearn-parity tested |
| `expected_calibration_error`, `brier_score` (binary) | `crates/aprender-core/src/calibration.rs:140,379` | The surface D-07 extends alongside | Contract-bound (`calibration-v1`) |
| `ttest_rel`, `ttest_1samp` | `crates/aprender-core/src/stats/hypothesis.rs:191,81` | Paired t statistic + p-value (incomplete-beta exact) | scipy-oracle tested (`tests_hypothesis_contract.rs`); f32; NO YAML contract yet; no CI helper |
| `SelectionManifest`, `Selection::replay`, `AccessLedger` | `crates/aprender-contrastive-data/src/` | D-10's manifest→Selection door; the pairing key | Production-proven at CLI tier |
| `SelectionLock`, `mint_test_token`, `CanonicalTestAccess::grant` | `crates/aprender-train/src/train/setfit/lock.rs` | EVAL-04 post-test-selection mechanism (SetFit side) | Credential-SEALED — see F-R14 |
| `evaluate_validation_from_artifact` | `crates/aprender-train/src/train/setfit/apr_evaluate.rs:185` | Trusted evaluator (validation, scalar metric) | Needs a per-row-predictions sibling for EVAL-01 (F-R8) |
| `EvalRow` + `LockRow` + `CandidateRow` | `crates/apr-cli/src/commands/eval/setfit.rs:112-171` | The EVAL-03 row seed | Serialize-only today; explicitly written for Phase 5 |
| `ClassifyConfig` / `TrainingConfig` / `ClassifyTrainer` / `run_classify` | `entrenar::finetune::classify_pipeline`, `classify_trainer.rs`; `crates/apr-cli/src/commands/finetune.rs:1563` | LoRA baseline (D-09/D-10) | Hardcoded seed 42 / val_split 0.2 / early stopping — must be overridden (F-R12) |
| `ClassifyEvalReport` | `crates/aprender-train/src/finetune/classify_eval_report.rs` | Existing multiclass ECE/Brier/MCC/kappa precedent | UNfixtured, contains bootstrap CIs — do NOT consume in claims path (F-R11) |
| `atomic_write` / `atomic_write_with` house style | `crates/apr-cli/src/commands/data_contrastive.rs`, `setfit_train.rs` | D-14 row/manifest writes | One rename site, temp-in-destination-dir, fill-closure form for streams |
| `ExecutionBackend::identity` | `crates/aprender-core/src/setfit/encoder.rs:227-235` | Row backend identity (SetFit) | Grammar `<device>:<implementation>:<kernel>`, v1 `cpu:setfit-core:autograd-trueno-matmul`; binding row resolved by 04-19 |
| `SetfitCommands` + `dispatch_setfit_command` | `crates/apr-cli/src/setfit_commands.rs:17`, `src/dispatch_analysis.rs:832` | D-13 CLI wiring point | `#[cfg(feature = "setfit")]`; currently `Train`-only |

### Supporting (already locked, no installs)
| Tool | Version | Purpose | Verified |
|------|---------|---------|----------|
| `pv` (in-tree) | 0.63.0 | Contract validate/diff/audit | `target/release/pv --version` ran this session; Makefile invokes via `PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --` |
| fixture env (`scripts/setfit_fixtures/`) | uv-locked | scipy/sklearn reference fixtures (D-06/D-07) | `scipy 1.18.0` present in `uv.lock` (transitive via `scikit-learn 1.9.0`); pins: setfit 1.1.3, sentence-transformers 5.7.0, torch 2.13.0 |
| `sysinfo` | 0.32 (workspace dep) | Peak-memory option under `unsafe_code = "forbid"` | Declared at root `Cargo.toml:266` |
| `criterion` | 0.7 (dev-dep) | NOT recommended for cells (statistical resampling ≠ contracted single-run rows); fine for micro-bench sanity only | Root `Cargo.toml:160` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Contract-resident `t_{0.975,9}` constant | Implementing Student-t inverse CDF in Rust | df is FIXED at 9 by the contracted design; a frozen constant (Ph1 D-14 pattern) verified against a scipy fixture is exactly recomputable and removes a numerical-code surface. Implementing the inverse CDF buys generality nobody contracted |
| In-crate `#[ignore]`d production calibration test | A new `apr` calibration subcommand | `run_tuning` and the evidence internals are `pub(crate)`; the Phase 3 harness is the reviewed precedent; a CLI surface would widen the API for a one-time measurement |
| `/proc/self/status` VmHWM (Linux) for peak memory | `getrusage` via libc | libc requires `unsafe` in our code — forbidden workspace-wide. VmHWM is a text read, is the true high-water mark, and lambda-vector is Linux. macOS fallback: `sysinfo` (unsafe stays inside the dependency) or run all cells on Linux |
| Extending `ttest_rel` (f32→f64 + CI) in core stats | New stats module in `entrenar::eval` | Core stats already owns the t machinery with scipy-oracle tests and the incomplete-beta p-value; "first consumer, not owner" says extend the owner. `entrenar::eval` would duplicate |

**Installation:** none. **Version verification:** not applicable — no registry installs; in-tree tool versions recorded above from direct invocation this session.

## Package Legitimacy Audit

**This phase installs no external packages.** All Rust work is in-tree (workspace crates); all
Python reference-fixture work uses the existing hash-locked `scripts/setfit_fixtures/` uv
environment, where `scipy 1.18.0` is already resolved in `uv.lock` as a transitive dependency of
the pinned `scikit-learn 1.9.0` — no `pyproject.toml` edit is required for the scipy fixtures.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| — | — | — | — | — | not run (no installs) | N/A |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

If planning later decides a new Python package is needed for a fixture generator (it should not
be), it must be added to the pinned `pyproject.toml` with an exact `==` pin and go through the
Package Legitimacy Gate before the uv re-lock.

## F-10 Unblock: Measured Mechanics (D-01..D-04)

Everything in this section was read from the shipped code/contract this session — it is the
answer to the planner's question 1.

### F-R1: Where the gate lives and what the edit actually touches

Three places must change **together**, or `thresholds_match_the_contract` goes red:

1. **Contract:** `contracts/setfit-train-lifecycle-v1.yaml` (v2.0.0),
   `equations.calibration_regime.calibrated_regimes` (line ~281) — currently the single fixture
   entry `minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4`. The
   invariant text two lines below says "THE CALIBRATED SET CONTAINS EXACTLY ONE FINGERPRINT" —
   that prose must be amended too (the entry itself stays byte-identical per D-03).
2. **Rust constant:** `CALIBRATED_REGIMES` at `thresholds.rs:61` — gains the second entry.
3. **The pinned test:** `thresholds_match_the_contract` (`thresholds.rs:394`) asserts
   `contracted_regimes.len() == 1` ("exactly ONE calibrated fingerprint; a second would mean
   numbers measured on one architecture are being applied to another") **and** asserts the Rust
   list is string-EQUAL to the contract list (the CR-04 anti-widening equality). The length
   assertion changes to 2; the equality assertion is the protection and must stay an equality
   over the full 2-entry list.

The contract also contains an explicit in-file clause (calibration_regime invariants, ~line 328):
"CONSEQUENCE FOR PHASE 5 … Per D-10(c) that is a deliberate contract edit which `pv diff` flags.
It is not a code change, not a threshold relaxation, and not something a Phase 5 executor may do
inline" — and clause N-02: "An argument is not a measurement." The edit is expected and
pre-authorized in shape; the human checkpoint (D-04) is about the measured numbers.

### F-R2: The rendered production regime id is NOT what prior documents guessed

`BertSentenceEncoder::architecture_fingerprint()` (`encoder.rs:303-311`) renders
`format!("minilm-slice-h{}-l{}-a{}-i{}-v{}", …)` — the `minilm-slice-` prefix is **hardcoded**
regardless of `remap.is_some()`. The revision tag is `SLICE_SOURCE_REVISION = "1110a243"`
(`aprender-train/.../mod.rs`, appended crate-side, documented in-code as "a REVISION TAG, not a
weights hash … a KNOWN LIMIT"). So a production run will render:

```
minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=<seed>|cells=s<shots>e<epochs>b<batch>
```

The `minilm-full-h384-l6-a12-i1536-v30522@production` string in thresholds.rs's negative test is
an INVENTED string, not a rendering. Note `@1110a243` is truthful for production — it is the
pinned HF revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` both the slice and the full model
come from. **Planner decision:** either accept the misleading `minilm-slice-` prefix on the full
model (dimensions disambiguate; zero code change) or amend `architecture_fingerprint` to render
`minilm-full-` when `remap.is_none()` (the fixture rendering is unchanged since the fixture HAS a
remap; but this touches a Phase 1 surface and every place the string appears). Either way, the
contract entry must be **copied from a measured run's own `calibration_regime_id`**, byte-for-byte
— never hand-composed.

### F-R3: The regime id grammar and cell labels

From `calibration_regime.formula` + `cell_label()`/`shots_component()` in
`aprender-train/.../mod.rs`: cell label = `s{shots}e{epochs}b{batch}`, shots read from the
SELECTION (uniform branch), epochs/batch from the resolved config. Membership is component-wise
subset coverage (`RegimeCoordinates::covers`), never string equality. **Consequence:** the
production epochs/batch must be FROZEN before the calibration, because they are part of every
cell label the entry enumerates. D-02's entry shape:
`…@1110a243|seeds=13,17,23,29,31,37,41,43,47,53|cells=s16e{E}b{B},s32e{E}b{B},s64e{E}b{B},s8e{E}b{B}`
(sets, order irrelevant — `regime_parse_reads_sets_not_ordered_lists`).

### F-R4: Per-regime thresholds do not exist yet — D-03 requires code restructuring

`Thresholds::frozen()` holds ONE global class table (`embedding 1.1e-5, layer_norm_weight 2.9e-6,
layer_norm_bias 1.1e-5, projection_weight 1.8e-5, projection_bias 8.3e-6, attention_key_bias
None/ungated`) + one `embedding_delta_floor 2.7e-5` + the regimes list. The contract mirrors this:
`equations.evidence_gate.frozen_thresholds` is a single map. D-03 demands FRESH ε for the
production regime while the fixture entry keeps its exact meaning — so the gate must select the
class table **by regime**. This is a real design task: extend the contract with an additive
per-regime thresholds block (fixture block byte-untouched), restructure `Thresholds` to map
regime→table, update `validate_evidence`'s threshold lookup, and extend
`thresholds_match_the_contract` to parse both blocks. The `gradient_free_parameters` analysis
(attention key bias, dL/db_k = 0 by softmax shift-invariance) is architecture-independent physics
but its MEASUREMENT is fixture-scale — the contract's own "RE-DERIVATION, NOT RELAXATION" clause
requires re-checking the class on the production encoder (FALSIFY-STL-002 pattern).

### F-R5: The measurement methodology to replicate (Phase 3's ε derivation, verbatim mechanics)

From the contract's evidence_gate DERIVATION invariant + `calibration_matrix_epsilon_basis`:

- **Conditions per cell:** real (`encoder_lr` 2e-5), null control (1e-30 → bit-identical zero
  deltas), near-null control (1e-8 → the REAL lower bound; 2000× below reference but AdamW still
  writes back a different f32).
- **Window per class:** `10 × worst_control_delta ≤ ε ≤ best_real_delta / 10`, across ALL
  measured cells; frozen value = upper edge rounded DOWN to two significant figures.
- **Noise floor check:** every ε must clear `(f32::EPSILON/2) × init_norm / max(denom, s_class)`
  per row (the `rounding_noise_floor` helper) by a wide margin (49×–315× on the fixture).
- **Support fraction:** embedding sparse denominator restricted to touched rows (fixture support
  0.32–0.35; production expected similar in ABSOLUTE rows — this is the invariance argument that
  motivates but does not replace the measurement, contract N-02).
- **Recorded per class:** real min/median/max, ctrl max, near-null max + moved flags, noise
  floor, support fraction, binding parameter name (so a too-narrow margin names what to widen).
- The harness passes the regime id into `UpdateEvidence::from_tune_output` and PRINTS it —
  that printed id is what the contract entry copies (F-R2).

**Gate-independence (critical enabler):** the `UncalibratedRegime` refusal lives in
`validate_evidence` (`tune.rs:1139`, order-of-checks item 1) — i.e., at judgement time.
`run_tuning` + `UpdateEvidence::from_tune_output` carry no regime gate, so the production matrix
runs on today's code with zero relaxation. `run_tuning` is `pub(crate)`; the harness therefore
lives in-crate as a second `#[ignore]`d test beside the fixture one (same file or sibling),
parameterized by `APRENDER_MINILM_DIR` like `full_weight_parity.rs:54` does.

### F-R6: What happens today, end to end, when you point the CLI at production

`apr setfit train --model-dir ~/.cache/aprender/minilm-l6-v2-1110a243 …` →
`SetFitMiniLm::from_pretrained_dir` → `MiniLmImport::open` validates `config.json`,
`modules.json`, `1_Pooling/config.json`, `tokenizer.json` (SHA-256 against the pin), then the
weights — all present in the cached checkout (verified on this host: `full_model.apr` 86.7 MB +
`model.safetensors` + `full_manifest.json`). Import succeeds with `vocab_remap: None`, so
`probe_unicode` (canonical id 5915) IS computable — the artifact-probe blocker is slice-only.
The run then **tunes fully and is refused at the evidence gate** with `UncalibratedRegime`
naming both the observed id and the calibrated set. (04-15's rung-4 exit 6 is a different,
earlier refusal: the spawned ladder points `--model-dir` at the slice fixture dir, which is not a
pretrained checkout.) So the ONLY thing between today and a user-produced `setfit-apr-v1` on the
production encoder is the calibrated-regime entry + its thresholds — exactly as F-10 states.

### F-R7: Stale carried-forward item — `backend_identity` binding is RESOLVED

`contracts/aprender/binding.yaml` (read this session, lines 1349-1371): "FIFTEEN of fifteen are
`implemented`. `backend_identity` was the last entry still `pending`; 04-19 performed the
semantic correction … the scoped audit now reports zero BIND findings." The row now names
`ExecutionBackend::identity` with a compile-witnessed resolution guard. CONTEXT.md's
carried-forward bullet (and REQUIREMENTS.md audit item 5) predate 04-19. **Planner action:** row
work should USE `ExecutionBackend::identity` (SetFit rows) and the pipeline's executed GPU
identity (`pipeline.gpu_name()`/`gpu_total_memory()`, LoRA rows) — no binding repair needed;
do not re-open 04-19's work.

### F-10 flip blast radius (wave-1 checklist input)

Files carrying F-10-conditional claims/refusal assertions (located via grep this session):
`aprender-train/tests/setfit_apr_lifecycle.rs`, `aprender-train/src/train/setfit/verify.rs`,
`apr_evaluate_tests.rs`, `apr-cli/tests/setfit_cli_lifecycle.rs`, `apr-cli/tests/setfit_parity.rs`,
`apr-cli/src/commands/{setfit_train.rs, predict_tests.rs, eval/setfit_tests.rs}`. The
slice-encoder refusal tests (e.g. `apr_codec.rs`'s `probe_unicode` refusal) stay green — the
slice still cannot compute the probe; only PROSE claims of the form "no user-reachable path
produces a setfit-apr-v1" become false and must be re-worded. The blocked-rung tests are
documented to "panic with restore instructions the day the refusal stops happening" — their
refusals are slice-dir-shaped and survive, but wave 1 must add the POSITIVE production-chain
rungs (env-gated on the checkout, like `full_weight_parity`) and re-audit every file above.

## Architecture Patterns

### System Architecture Diagram

```
                          ┌────────────────────────────────────────────────┐
                          │ WAVE 1 — F-10 unblock                          │
  ~/.cache/aprender/      │  in-crate #[ignore]d production calibration     │
  minilm-l6-v2-1110a243 ──┼─► run_tuning ×(cells×seeds×3 conditions)       │
  (production checkout)   │      └─► UpdateEvidence stats ─► ε window      │
                          │  measured regime id + ε table                  │
                          │      └─► CONTRACT EDIT (human checkpoint,      │
                          │          pv diff old vs new) ─► CALIBRATED_    │
                          │          REGIMES + per-regime Thresholds       │
                          └───────────────┬────────────────────────────────┘
                                          │ unblocks
        ┌─────────────────────────────────▼──────────────────────────────────────┐
        │ PER CELL (method × shot × seed)                                        │
        │                                                                        │
        │  apr data select (seed s, shots n) ─► selection-manifest.json          │
        │        │  (ONE artifact, hash-verified — the EVAL-02 pairing key)      │
        │        ├────────────────────────────┬───────────────────────────────┐  │
        │  SetFit cell (CPU)                  │  LoRA cell (lambda-vector GPU)│  │
        │  apr setfit train ─► model.apr      │  apr finetune --task classify │  │
        │  apr eval --split validation        │    --selection-manifest ...   │  │
        │    --lock-out L (COMMITS selection) │  (explicit TrainingConfig:    │  │
        │  apr eval --split test              │   contracted seed, no early   │  │
        │    --selection-lock L (grant-gated) │   stop, val policy per plan)  │  │
        │        │                            │  evaluate checkpoint on test  │  │
        │        ▼                            ▼        (post-lock discipline) │  │
        │   bench run: full prediction vectors ─► metrics (F_avg, MCC,        │  │
        │   confusion, top-label ECE, mc-Brier) + resource measurements       │  │
        │        ▼                                                            │  │
        │   ONE schema-versioned JSON row file (atomic write)                 │  │
        └──────────────┬─────────────────────────────────────────────────────┘  │
                       ▼                                              (GPU rows │
        benchmark directory + RUN MANIFEST (80 expected cells,        transported│
        content hash per row) ◄───────────────────────────────────────as files, │
                       │                                              re-hash-  │
                       ▼                                              verified) │
        bench report: manifest-complete? every hash green? every pair matched   │
        on sampled-ID hash? no post-test-selected lock? ── any NO ─► REFUSE     │
                       │ all YES                                                │
                       ▼                                                        │
        closed-form stats: per-cell/per-shot mean ± (n−1) std, min/max,         │
        per-seed paired deltas, t 95% CI (frozen t_{0.975,9}) ─► human + --json │
        └───────────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure (new files)

```
crates/aprender-train/src/train/setfit/
├── thresholds.rs                  # EDIT: per-regime tables + second entry + test update
├── evidence.rs                    # EDIT: production calibration harness (#[ignore]d, env-gated)
├── bench_row.rs (or similar)      # NEW: the D-12 row type + content-hash + manifest types
crates/aprender-core/src/
├── calibration.rs                 # EDIT: multiclass top-label ECE + multiclass Brier beside binary
├── stats/hypothesis.rs            # EDIT: paired-CI helper (f64) beside ttest_rel
crates/apr-cli/src/
├── setfit_commands.rs             # EDIT: Bench { Run {...}, Report {...} } variants
├── commands/setfit_bench.rs       # NEW: filesystem adapter (data_contrastive.rs house style)
├── commands/finetune.rs           # EDIT: --selection-manifest + explicit TrainingConfig control
contracts/
├── setfit-benchmark-claims-v1.yaml # NEW: row schema, completeness/pairing rules, stats equations
├── setfit-train-lifecycle-v1.yaml  # EDIT: additive regime entry + per-regime thresholds (D-04 gate)
scripts/setfit_fixtures/
├── generate_fixtures.py (or new gen_stats_fixtures.py)  # scipy/sklearn stats fixtures
Makefile                            # EDIT: append new contract to $(CONTRACTS); bench gates; tiers
```

### Pattern 1: The deliberate contract edit (D-04 worked example, house style)

The tweet-eval contract's own metadata header is the model: record in-file the `pv diff` command,
its suggested bump, and the reasoning. Procedure:
```bash
git show HEAD:contracts/setfit-train-lifecycle-v1.yaml > /tmp/setfit-train-lifecycle-old.yaml
target/release/pv diff /tmp/setfit-train-lifecycle-old.yaml contracts/setfit-train-lifecycle-v1.yaml
```
Note: the "EXACTLY ONE FINGERPRINT" invariant prose changes, so `pv diff` may suggest more than a
patch bump — take its suggestion (house rule: the bump is pv's call, not a judgement). The
contract-edit task is `autonomous: false`; the executor presents measured thresholds + diff and waits.

### Pattern 2: Multiclass calibration metrics (D-07) — site, formulas, fixtures

Site in `aprender-core::calibration` beside the binary pair, contract-bound the same way
(`#[provable_contracts_macros::contract("calibration-v1", …)]` or the new claims contract —
planner's call; the binary pair binds to `calibration-v1`).

- **Top-label ECE** (ordered labels, K probabilities per row):
  `conf_i = max_k p_ik`, `pred_i = argmax_k p_ik`, bin by `conf_i` into `n_bins` equal-width bins
  over [0,1]; `ECE = Σ_b (n_b/N)·|acc_b − conf_b|`. This matches both the binary implementation's
  binning style (`calibration.rs:147` bin indexing) and `ClassifyEvalReport`'s 10-bin
  `calibration_bins` semantics.
- **Multiclass Brier** (original Brier 1950 definition, matches `compute_brier_score` at
  `classify_eval_report.rs:316`): `BS = (1/N) Σ_i Σ_k (p_ik − y_ik)²` with one-hot `y`.
- **Fixtures:** the pinned uv env (sklearn 1.9.0 / scipy 1.18.0). sklearn has no multiclass-ECE
  API [ASSUMED — verify in env]; generate fixtures with a small numpy reference implementation
  cross-checked THREE ways per the Phase 2 fixture discipline (measurement, closed form, contract
  literal). For Brier, an independent identity check: `multiclass BS = Σ_k brier_score_loss(y==k,
  p_k)` using sklearn's binary `brier_score_loss` per class [ASSUMED — verify identity numerically
  in the generator]. Record fixture SHA-256s in a manifest (Ph1 D-13).
- **OPS-03 note:** `ClassifyEvalReport.ece/brier_score` is a second, unfixtured implementation.
  The claims path must call the new contract-bound core functions. Recommend (non-blocking) a
  follow-up making `classify_eval_report` delegate; at minimum the claims contract should name the
  core functions as THE implementation for rows.

### Pattern 3: Paired statistics (D-05/D-06) — exact shapes

Per shot level `n ∈ {8,16,32,64}`, over the 10 contracted seeds:
- Per cell (method, n, seed): the row's metric values (F_avg primary per EVAL-01).
- Headline per (method, n): `mean = x̄`, `std = √(Σ(x_i−x̄)²/(n−1))` with n=10, plus min/max.
- Paired per n: `d_s = SetFit_s − LoRA_s` for each seed s (pair validity = identical sampled-ID
  hash, enforced by the gate before any arithmetic); `d̄`, `s_d` (n−1), and
  `CI95 = d̄ ± t_{0.975,df=9} · s_d/√10`.
- **`t_{0.975,9} = 2.2621571628…` [ASSUMED — freeze as a contract-resident constant and verify
  against `scipy.stats.t.ppf(0.975, 9)` in the fixture generator before committing]**. df is
  fixed by the contracted design, so no inverse-CDF implementation is needed; this keeps the
  claims path pure closed-form arithmetic (no RNG, no iteration).
- p-values (machine-readable detail only, per D-08): `aprender-core::stats::hypothesis` already
  computes the exact two-tailed t p-value via the incomplete-beta relation
  (`t_distribution_pvalue`, `hypothesis.rs:360`) with scipy-oracle tests. It is f32; the claims
  path should use f64 mirrors with scipy fixtures (glm_tests.rs:280 precedent: record the
  RED/wrong value in the test comment). Note `tests_hypothesis_contract.rs` says "no YAML
  contract for hypothesis testing yet" — binding the new claims equations closes that for the
  functions the report uses.

### Pattern 4: Row + manifest storage (D-14) — copy the Phase 2 doors

- Row files: `serde` struct with `#[serde(deny_unknown_fields)]`, schema_version field, written
  via the `atomic_write` house pattern (temp in destination dir, `write_all`, `sync_all`, one
  `rename` site, no-clobber unless `--force`; WR-01's rename race is a known, human-ruled
  out-of-scope residue — do not fix it here, do not widen it).
- Content hash: SHA-256 over the row's canonical bytes; the manifest lists all 80 expected cells
  (method × shot × seed) with hashes. Follow `SelectionManifest`'s design: `from_bytes` verifies
  the digest BEFORE returning, so a forged manifest is unrepresentable downstream.
- Bounded reads before parse (16 MiB-cap house pattern from 04-18; stat first, refuse over-cap).
- In-band negatives (D-15): commit a doctored row set (missing cell, trimmed cell, mismatched
  sampled-ID hash, bit-flipped row, post-test-selected lock) as fixtures; every `cargo test` must
  show each REFUSED. Gates must be incapable of vacuous pass (CR-02: a filter that matches zero
  tests exits 0 — pin suite non-emptiness).

### Pattern 5: 40-cell orchestration + GPU-host transport

- House precedent for remote GPU dispatch: `scripts/dispatch-*.sh` (ssh to host, pull checkout,
  build, run, artifacts return as files). lambda-vector is the RTX 4090 host
  (`dispatch-distill-phase-3-gx10.sh:28`); no `Host lambda-vector` stanza exists in this
  machine's `~/.ssh/config` — reachability is an OPEN environment item (see Environment
  Availability).
- Per-cell resume falls out of D-14 naturally: a cell is DONE iff its row file exists and
  hash-verifies against the manifest expectation; the driver (Make target or `bench run` loop)
  skips verified cells. Recommend the manifest be created FIRST (the expectation set is
  declared before any cell runs — that is also what makes selective omission visible).
- Transport: rows are self-verifying files; `scp`/`rsync` back into the benchmark directory and
  re-hash on ingest. Never regenerate or rewrite a row in transit; `bench report` re-verifies
  every hash anyway.
- SetFit cells are CPU (D-09) — host not pinned by the decision. Running them on lambda-vector's
  CPU would make peak-memory measurement uniform (Linux VmHWM) and keep the macOS dev box free;
  rows record true host identity either way. Planner's call; see Pattern 6.

### Pattern 6: Resource measurement (EVAL-05) under `unsafe_code = "forbid"`

Nothing in-repo measures peak RSS today (`aprender-simulate`'s `peak_memory_bytes` is an
always-`None` placeholder). Options, in preference order:
1. **Linux `/proc/self/status` VmHWM** — plain text read, true high-water mark, no unsafe. Works
   for both cell types if cells run on Linux hosts (LoRA already must; SetFit could).
2. **`sysinfo` 0.32** (already a workspace dep) — cross-platform; unsafe stays inside the
   dependency; process-level RSS polling is a SAMPLED maximum, not a true HWM — if used, the
   contract must say "sampled at ≥X Hz" (bounded, per success criterion 3).
3. Spawned-cell design: the driver spawns `apr …` per cell and reads the child's VmHWM at exit
   (Linux) — clean isolation, one process per row, matches the spawned-tier evidence style.
- **Latency:** `ClassifyResponse.latency_ms` exists (bit-asserted in parity tests, excluded from
  `PartialEq` by contract). Contract cold vs warm explicitly: cold = first classify in a fresh
  process after artifact load; warm = steady-state after a contracted warmup count (the
  `apr bench --warmup/--iterations` precedent at `commands/bench.rs`). Record batch size at
  measurement.
- **Training time:** LoRA path already emits `epoch_time_ms`/`samples_per_sec`/`total_time_ms`
  (contract `apr-finetune-metrics-v1.yaml` — note its `tokens_per_sec` multiplies by a hardcoded
  512; do NOT propagate that number into rows). SetFit: wall-clock the train invocation in the
  driver; check what the train completion report already carries and extend at the adapter if
  needed.
- **Artifact size:** bytes of the written APR / adapter checkpoint (both known at write time).

### Anti-Patterns to Avoid

- **Hand-composing the regime id string** — copy the measured `calibration_regime_id` (F-R2).
- **Consuming `ClassifyEvalReport`'s bootstrap CIs** in any claims output (RNG in claims path,
  violates D-06). The bootstrap fields exist and are tempting; the claims contract should
  explicitly exclude them.
- **Letting the LoRA CLI defaults ride** — `seed: 42` (contract-excluded), `val_split: 0.2`
  (uncontracted internal split), `early_stopping_patience: 10` + best-epoch checkpointing
  (uncontrolled model selection). All are `pub` fields on `TrainingConfig` — set them explicitly
  per cell.
- **A second evaluator** — extend `apr_evaluate`/`evaluate.rs` for per-row predictions rather
  than classifying test rows in the CLI adapter (OPS-03; the adapter is a filesystem shim).
- **Directory-listing completeness** — the manifest defines the expectation set (D-14).
- **Averaging across hosts as if one** — rows carry host identity; the report never pools
  resource metrics across backends (D-09 framing).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Selection replay / pairing key | A subset exporter or row re-reader | `SelectionManifest::from_bytes` + `Selection::replay` | Digest-verified before return; the identical-sampled-ID guarantee is structural (D-10's rationale) |
| Test-split gating (SetFit) | Any new gate | `create_selection_lock → mint_test_token → CanonicalTestAccess::grant` | Ph3/Ph4 machinery, 73+45 tests, compile-time negatives; Phase 5 is a consumer (CONTEXT: "never reimplements") |
| F_avg / macro-F1 / confusion | New metric code | `MultiClassMetrics::from_predictions` + `f1_avg_for_classes(&[1,2])` | Contract-bound (`official_f_avg` → `f1_average_for_classes`), sklearn-parity tested, returns `None` on bad class indices instead of mislabeling |
| MCC | New formula | `aprender_core::metrics::agreement::matthews_corrcoef` | sklearn-parity tested |
| Paired t statistic / p-value | New t machinery | `stats::hypothesis::ttest_rel` shape (+ f64/CI extension) | Exact incomplete-beta p-value, scipy-oracle tests already exist |
| t 95% critical value | Inverse CDF implementation | Contract-resident `t_{0.975,9}` constant + scipy fixture | df fixed at 9 by design; Ph1 D-14 pattern |
| Atomic row writes | New file plumbing | `atomic_write`/`atomic_write_with` house pattern | One rename site, fault-injection seams, no-clobber semantics already reviewed twice |
| Content hashing | Anything but SHA-256 via `sha2` | `sha2::Sha256` (already used everywhere) | V6: never hand-roll crypto; consistency with every other digest in the tree |
| Contract validation | bash/yq/python scripts | `pv validate/lint/diff/audit` | CLAUDE.md hard rule; scripts are muda and get rejected |
| YAML config parsing for cells | New config formats | `SetFitTrainConfig` (.toml/.json, deserialization-is-validation) + `ClassifyConfig` | Twelve-knob validation exists; file-first reproducibility is the house rule |

**Key insight:** every "build" in this phase is assembly and gating of existing verified doors.
The only genuinely new algorithms are multiclass ECE/Brier (≈30 lines each, fixture-verified) and
the paired-CI arithmetic (≈10 lines + one frozen constant).

## Common Pitfalls

### Pitfall 1: The three-place regime edit done as a YAML append
**What goes wrong:** adding the second entry to the contract alone (or the constant alone) turns
`thresholds_match_the_contract` red — or worse, editing the test's `len == 1` to `>= 1` and the
equality to a subset would re-open CR-04's widening hole.
**How to avoid:** one commit touches contract + `CALIBRATED_REGIMES` + test together; the
equality-over-full-list assertion survives with `len == 2`. The commit is the D-04 checkpoint.
**Warning signs:** any plan task that edits the contract without naming `thresholds.rs`.

### Pitfall 2: Unknown production tuning cost blows the compute budget
**What goes wrong:** the boundary matrix is ~18 full `run_tuning` passes (2 boundary cells × 3
seeds × 3 conditions) on a 6-layer/h384/30522-vocab encoder through the pure-Rust autograd CPU
path — wall-clock is UNMEASURED and could be hours. CLAUDE.md requires a human check-in for >1 hr
compute on non-lambda-vector hosts.
**How to avoid:** wave 1 starts with a single timed s8 cell (one condition) and extrapolates
before committing to the matrix; report the projection at the D-04 checkpoint. s64 cells are
~8× the selected rows of s8.
**Warning signs:** a plan that schedules the full matrix without a timing probe task.

### Pitfall 3: LoRA hardcoded internals silently poison EVAL-02/EVAL-04
**What goes wrong:** `run_classify` builds `TrainingConfig { val_split: 0.2, seed: 42,
early_stopping_patience: 10, … }` (finetune.rs:1675-1685). Seed 42 is explicitly NOT contracted
(`tweet-eval` contract line 204: "42 is NOT a contracted seed"); the 0.2 split re-partitions the
selected rows randomly; best-epoch checkpointing on that split is model selection with no lock.
**How to avoid:** D-10's wiring passes contracted seed, an explicit validation policy, and
disables early stopping under frozen defaults; the row records what was actually run.
**Warning signs:** LoRA rows whose training provenance does not name the seed/val policy; any
epoch count in results ≠ requested epochs (early stop fired).

### Pitfall 4: The claims path inherits RNG through a convenience call
**What goes wrong:** `ClassifyEvalReport` computes bootstrap CIs on construction; consuming the
report wholesale puts a resampling stream inside "exactly recompute".
**How to avoid:** rows carry only the closed-form fields; the claims contract enumerates the row
schema with `deny_unknown_fields` so a bootstrap field cannot ride along.

### Pitfall 5: New contract validated by nothing
**What goes wrong:** `$(CONTRACTS)` in the Makefile is an explicit list — a new
`setfit-benchmark-claims-v1.yaml` is validated by NO gate until appended (Phase 2 decision, logged
in STATE.md).
**How to avoid:** the plan that creates the contract also appends it to `$(CONTRACTS)` and to the
phase's scoped `contract-audit-phase5` target, and proves the gate can fail (induce one mutation).

### Pitfall 6: Vacuous bench gates (CR-02 recurrence)
**What goes wrong:** libtest exits 0 on a zero-match filter; a renamed test leaves a green gate
running nothing.
**How to avoid:** every new Make filter asserts non-zero matched tests (the `setfit-*-tests`
targets' floor pattern); mutate once to observe red before trusting.

### Pitfall 7: Host/toolchain sharp edges (all measured previously, all still live)
- `cargo check --workspace` cannot exit 0 on Darwin — use `--exclude aprender-profile`.
- `make tier2` is RED on arm64 (24 pre-existing clippy errors in arch-gated SIMD, none in
  phase-owned crates) — don't burn time "fixing" the phase for them.
- `CARGO_INCREMENTAL=0` (two ENOSPC stops at ~25 GB incremental cache).
- `rtk` hook rewrites `git status --porcelain` — porcelain-emptiness assertions must run through
  `rtk proxy`.
- Pin the apr binary (`. scripts/apr_bin.sh`) for every spawned-tier measurement.
- `cargo package -p apr-cli` is KNOWN-RED until the human-run publish cascade (pre-release skill
  reads this correctly; don't report it as a Phase 5 regression).

### Pitfall 8: PF-007 / PF-008 (the domain pitfalls the phase exists to prevent)
Seed sensitivity hidden behind verdicts; wrong headline metric (accuracy or unlabeled F1 instead
of explicit `F_avg = (F1_against + F1_favor)/2`); unequal comparisons (different IDs, hardware
framing, timing definitions); omitted per-class/calibration/resource columns. The row schema and
D-08's estimation-first rule are the mitigations — the claims contract should quote both PF ids.

## Code Examples

All from the tree this session (paths absolute from repo root).

### The regime constant and membership (the thing wave 1 edits)
```rust
// crates/aprender-train/src/train/setfit/thresholds.rs:61
pub(crate) const CALIBRATED_REGIMES: &[&str] =
    &["minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4"];

// Membership: parse both sides, calibrated entry must COVER the run (subset semantics)
pub(crate) fn is_calibrated(&self, regime_id: &str) -> bool { /* thresholds.rs:330 */ }

// The run renders its own id — architecture from the encoder, seed from config, cell from selection:
// crates/aprender-train/src/train/setfit/mod.rs (calibration_regime_id + cell_label)
let architecture = format!("{}@{SLICE_SOURCE_REVISION}", encoder.architecture_fingerprint());
// encoder.rs:303 — NOTE the hardcoded prefix, even for the full model:
format!("minilm-slice-h{}-l{}-a{}-i{}-v{}", hidden, layers, heads, intermediate, vocab)
```

### The calibration harness invocation (Phase 3 precedent to replicate for production)
```bash
# Fixture matrix (existing):
cargo test -p aprender-train --lib --features setfit calibration_matrix -- --ignored --nocapture
# Production harness (new, same shape): gate on the cached checkout like full_weight_parity.rs:
#   std::env::var("APRENDER_MINILM_DIR") … default ~/.cache/aprender/minilm-l6-v2-1110a243
```

### The gate order (why measurement needs no gate widening)
```rust
// crates/aprender-train/src/train/setfit/tune.rs:1139 (validate_evidence)
// (1) FAIL CLOSED OUTSIDE THE CALIBRATED REGIME — before any comparison.
if !thresholds.is_calibrated(&evidence.calibration_regime_id) {
    return Err(SetFitTrainError::UncalibratedRegime { observed, calibrated });
}
// run_tuning + UpdateEvidence::from_tune_output carry NO regime check — measurement is legal today.
```

### The two-invocation lock workflow bench cells drive (Ph4 D-16)
```bash
apr eval M.apr --task classify --data D --selection S --split validation --lock-out L
apr eval M.apr --task classify --data D --selection S --split test       --selection-lock L
# eval/setfit.rs: EvalRow is "The machine-readable row Phase 5 consumes (EVAL-03 shape)";
# EVAL_METRIC is const Accuracy; ValidationEvaluation is a SCALAR —
# per-row predictions for F_avg/MCC/calibration need a library-side sibling door.
```

### The LoRA fields D-10 must control (all pub — settable via library call)
```rust
// crates/aprender-train/src/finetune/classify_trainer.rs:26 TrainingConfig
pub epochs: usize, pub val_split: f32, pub save_every: usize,
pub early_stopping_patience: usize, pub checkpoint_dir: PathBuf, pub seed: u64, /* … */

// crates/apr-cli/src/commands/finetune.rs:1675 — what the CLI hardcodes today:
let training_config = TrainingConfig { epochs, val_split: 0.2, save_every: 5,
    early_stopping_patience: 10, checkpoint_dir, seed: 42, log_interval: 1, .. };

// ClassifyConfig::default(): lora_rank 16, lora_alpha 16.0, lr 1e-4, epochs 3,
// max_seq_len 512, batch_size 32, accumulation_steps 1, grad_clip Some(1.0)
// CLI mapping: lora_alpha = rank as f32 (finetune.rs:1486). 9B config:
// model_config.rs:158  "9B" | "qwen3.5-9b" → TransformerConfig::qwen3_5_9b
```

### Existing paired t (extend, don't replace)
```rust
// crates/aprender-core/src/stats/hypothesis.rs:191
pub fn ttest_rel(sample1: &[f32], sample2: &[f32]) -> Result<TTestResult> // diffs → ttest_1samp
// p-value: exact I_x(df/2, 1/2), x = df/(df+t²)  (hypothesis.rs:360, scipy-oracle tested)
// Missing for D-06: f64 precision + the CI half-width (frozen t_{0.975,9} · s_d/√n)
```

### Multiclass Brier reference already in-tree (formula source; NOT the claims implementation)
```rust
// crates/aprender-train/src/finetune/classify_eval_report.rs:316
// BS = mean_i Σ_k (p_ik − y_ik)²   — matches the standard multiclass Brier the new
// contract-bound aprender-core::calibration function must implement with fixtures.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `backend_identity` binding row `pending`, naming a nonexistent symbol | 15/15 `implemented`, row names `ExecutionBackend::identity`, zero BIND findings | 04-19 (wave 11) | CONTEXT.md carried-forward bullet is stale; do not re-fix (F-R7) |
| "bashrs is not installed on this host" (REQUIREMENTS.md Phase 4 note) | bashrs 6.66.3 on PATH; 04-22 ran it for real over Makefile + scripts | 04-22 | bashrs gates are runnable in Phase 5 plans |
| `04-VERIFICATION.md` stale `gaps_found` | reconciled `gaps_acknowledged`; UAT 12/12 at `b3f816c25` | 2026-08-16 | Phase 5 builds on the UAT evidence baseline; Phase 4 still awaits `/gsd:secure-phase 04` |
| No user-reachable `setfit-apr-v1` (F-10) | unchanged — Phase 5 wave 1 is the fix | — | The keystone |

**Deprecated/outdated:** nothing else relevant; the phase consumes in-tree surfaces at HEAD.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `t_{0.975,df=9} = 2.2621571628` (from training knowledge) | Pattern 3 | Wrong CI half-widths in the headline claims — MUST be frozen only after a `scipy.stats.t.ppf(0.975, 9)` fixture confirms it |
| A2 | SetFit reference recipe defaults: 1 contrastive epoch, batch 16, body lr 2e-5, pair-iteration R=20, logistic head (paper + setfit 1.1.3 defaults) | Hyperparameter-policy discretion | Wrong "frozen defaults" for the SetFit side. Verify from the PINNED env, not the web: `cd scripts/setfit_fixtures && uv run python -c "from setfit import TrainingArguments; print(TrainingArguments())"` |
| A3 | sklearn 1.9.0 has no multiclass-ECE API; `brier_score_loss` is binary-only | Pattern 2 | Fixture generator design changes if wrong (would simplify, not break); verify in env before writing the generator |
| A4 | The OvR identity `multiclass BS = Σ_k binary BS_k` for one-hot labels | Pattern 2 | Cross-check in the generator; it is an algebraic identity but must be confirmed numerically against the implementation's edge handling |
| A5 | **REFUTED 2026-09-08 (D-19)** — lambda-vector is NOT reachable (12 candidates, 12 failures) AND the architecture is unimplemented, so the weights are moot. Arm descoped. Original assumption: lambda-vector is reachable and holds/can pull the qwen3.5-9b base weights | Pattern 5, Environment | The 40 LoRA cells cannot run; the phase's D-09 half stalls. No ssh config entry found on THIS host — must be confirmed with the human before planning locks the orchestration |
| A6 | `ClassifyTrainer` tolerates `val_split: 0.0` / effectively-disabled early stopping | Pitfall 3 | If it divides by zero or requires val batches, D-10 wiring needs a trainer-side patch; probe with a 5-minute test before the LoRA wave |
| A7 | The SetFit train completion report carries (or can trivially carry) wall-clock timing | Pattern 6 | Minor: the driver can wall-clock the invocation regardless |

## Open Questions (RESOLVED — every question has a plan-adopted resolution or a mandated checkpoint; pointers below)

1. **lambda-vector access + 9B base-model provenance** (A5)
   - What we know: pre-authorized compute; RTX 4090 per dispatch-script comments; dispatch-*.sh
     house pattern exists; `--model-size 9B → TransformerConfig::qwen3_5_9b`.
   - What's unclear: SSH reachability from this macOS host (no config entry found); where the 9B
     base weights live and their pinned hash (rows must record base-model hashes, D-12).
   - Recommendation: resolve at the first human checkpoint; plan the LoRA wave behind an explicit
     environment-verification task.
   - **RESOLVED →** fronted as the blocking human checkpoint 05-11.T1 (automation-first ssh
     probes; base-weight SHA-256 recorded before any cell runs — the LoRA wave's hard precondition).
2. **Production epochs/batch for the cell labels** (feeds D-02's entry and F-R3)
   - What we know: labels are `s{shots}e{E}b{B}`; fixture cells used e1b4/e2b8; SetFit reference
     recipe leans 1 epoch / batch 16 (A2).
   - Recommendation: freeze from the pinned setfit 1.1.3 defaults (A2 verification) in the
     calibration plan, BEFORE the matrix runs — they are baked into the contract entry.
   - **RESOLVED →** 05-01.T1 freezes E/B from the pinned env before the matrix; the frozen values
     are baked into 05-03's contract entry.
3. **LoRA-side lock semantics** (F-R14, the one real design gap)
   - What we know: `create_selection_lock`/`mint_test_token`/`grant` are generic over a SEALED
     `SetFitCredential` (exactly two implementors, counted by
     `credential_seal_is_a_private_supertrait`); `PreparedDataset::test()` is `pub`, so LoRA test
     reads are mechanically possible; under frozen defaults with early stopping disabled, the
     LoRA side performs NO selection, so there is nothing to lock — but EVAL-04's row core names
     a "selection-lock reference" for every row.
   - Options: (a) widen the seal with a LoRA credential (deliberate edit to a counted Phase 4
     invariant — high ceremony); (b) claims-contract-level rule: LoRA rows carry a no-selection
     attestation + the manifest hash, and `bench report` enforces "no LoRA cell may reference
     more than one trained candidate per (shot, seed)"; (c) reuse the SetFit cell's lock as the
     cell's pairing witness. Recommendation: (b) — it matches "frozen defaults ⇒ no selection"
     and avoids touching the seal; the contract states it in-band with a doctored-row negative.
   - **RESOLVED →** option (b) adopted: 05-05's claims contract owns the LoRA no-selection
     attestation rule (seal NOT widened); 05-10's bench gate enforces one-trained-candidate-per-cell.
4. **Where the row/manifest TYPES live** — `aprender-train` (beside lock/evidence, feature
   `setfit`) vs a small new module in `apr-cli`. House rule ("no semantics in the CLI",
   `SelectionManifest` lives in the data crate) argues for the library. Planner's call; the
   contract owns the schema either way.
   - **RESOLVED →** the library: `aprender_train::train::setfit::bench_row` (feature `setfit`),
     built in 05-05.T2; the CLI stays a filesystem shim.
5. **`bench run` shells out vs library calls** — the spawned-process design gives per-cell
   isolation, true child peak-RSS, and matches the spawned-tier evidence style; library calls
   give richer typed errors. Hybrid recommendation: `bench run` calls libraries for evaluation
   (needs prediction vectors) but the 40-cell DRIVER spawns `bench run` per cell (isolation +
   resume + measurement). OPS-03 is satisfied either way as long as evaluation goes through the
   one library door.
   - **RESOLVED →** the hybrid, as recommended: 05-09's `bench run` calls library doors for
     evaluation; the 40-cell driver (run_bench_cells.sh) spawns one `bench run` process per cell.
6. **SetFit CPU host choice** (macOS dev box vs lambda-vector CPU) — affects peak-memory
   mechanism uniformity (Pattern 6). Recommend deciding in the same checkpoint as Q1.
   - **RESOLVED →** decided at the same 05-11.T1 checkpoint as Q1 (default recommendation:
     macOS local, mechanism declared per row); 05-12 consumes the decision.

## Environment Availability

| Dependency | Required By | Available | Version / Evidence | Fallback |
|------------|------------|-----------|--------------------|----------|
| cargo / rustc | everything | ✓ | cargo 1.93.0 | — |
| `pv` (in-tree) | contract gates | ✓ | 0.63.0 (`target/release/pv`; Makefile PV_BIN via cargo run) | rebuild via `cargo run -p aprender-contracts-cli --bin pv` |
| pmat | code search / quality gates | ✓ | 3.15.0 | — |
| uv + fixture env | scipy/sklearn fixtures (D-06/D-07) | ✓ | uv 0.9.5; `scripts/setfit_fixtures/uv.lock` resolves scipy 1.18.0, sklearn 1.9.0 | — |
| bashrs | script/Makefile lint gates | ✓ | 6.66.3 (REQUIREMENTS.md "not installed" note is stale) | — |
| Production MiniLM checkout | wave-1 calibration + all SetFit cells | ✓ | `~/.cache/aprender/minilm-l6-v2-1110a243/` verified this session: config.json, modules.json, 1_Pooling/, tokenizer.json, full_model.apr (86.7 MB), model.safetensors, full_manifest.json | `uv run python fetch_full_weights.py` regenerates |
| batuta | alternative model pull route | ✗ (not on PATH) | — | not needed — checkout exists; `fetch_full_weights.py` is the maintained route |
| TweetEval prepared dataset dir | all cells | ◐ | `apr data tweet-eval-stance` machinery is DATA-01-complete; a current attested directory must be (re)generated at bench setup (needs the pinned source acquisition once) | rung-1 of the 04-15 ladder already exercises it |
| lambda-vector (ssh) | 40 LoRA cells (D-09) | ✗ UNVERIFIED | no `~/.ssh/config` entry on this host; dispatch scripts reference a Linux home (`/home/noah`) | **blocking for the LoRA wave — resolve with human (Open Q1)** |
| qwen3.5-9b base weights | LoRA cells | ✗ UNVERIFIED | `--model-size 9B` config path exists in-code | same checkpoint as above |
| CI (`ci.yml` setfit legs) | phase gates "in CI" | ◐ | 16 legs applied at `57f7823ab`, NEVER EXECUTED (no PR opened — human's call) | local Make gates remain the executed evidence |

**Missing with no fallback:** lambda-vector access + 9B weights (LoRA wave blocker — human input).
**Missing with fallback:** batuta (not needed), TweetEval directory (regenerable).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (libtest) + trybuild (compile-fail negatives) + cargo-mutants (scoped to new code) |
| Config | workspace `Cargo.toml` lints; `.clippy.toml` disallowed-methods; Make targets are the gate registry |
| Quick run command | `cargo test -p aprender-train --lib --features setfit <filter>` (also `-p apr-cli`, `-p aprender-core`) |
| Full suite command | `make setfit-all-tests` (18 scoped suites with non-vacuity floors) + `make tier2` / `make tier3` |
| Contract gate | `target/release/pv validate contracts/<file>` per contract; scoped `make contract-audit-phase5` (to be created, `$(CONTRACTS)` appended) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | F_avg/per-class/macro-F1/MCC/confusion/calibration bound to ordered labels | unit + fixture-parity | `cargo test -p aprender-core --lib calibration` + `cargo test -p aprender-train --lib --features setfit eval::classification` | ◐ existing metrics tested; ❌ Wave 0: multiclass ECE/Brier fixtures + tests |
| EVAL-02 | 40+40 cells, identical sampled-ID hash per cell | integration (spawned) + unit on manifest resolution | `cargo test -p apr-cli --lib --features setfit finetune_selection_manifest` (new) + one spawned two-method cell in `setfit_cli_lifecycle`-style test | ❌ Wave 0 |
| EVAL-03 | One schema-versioned row per run, complete field set | unit (`deny_unknown_fields` round-trip + refusal negatives) | `cargo test -p aprender-train --lib --features setfit bench_row` (new) | ❌ Wave 0 |
| EVAL-04 | Exact recomputation; missing/omitted/unmatched/post-test-selected invalidates | unit + in-band doctored fixtures (D-15's five negatives) | `cargo test -p apr-cli --lib --features setfit setfit_bench_report` (new; suite must assert non-zero matches) | ❌ Wave 0 |
| EVAL-05 | Resource metrics with contracted boundaries from reloaded artifacts | unit on boundary logic + spawned measurement smoke | `cargo test -p apr-cli --lib --features setfit bench_resource` (new) | ❌ Wave 0 |
| F-10 (gating) | Production regime calibrated; user chain closes | `#[ignore]`d calibration harness + spawned production ladder (env-gated) | `cargo test -p aprender-train --lib --features setfit production_calibration -- --ignored --nocapture`; ladder via `cargo test -p apr-cli --test setfit_cli_lifecycle --features setfit` | ❌ Wave 0 (harness); ◐ ladder exists, gains positive rungs |

### Sampling Rate
- **Per task commit:** the owning scoped suite (`cargo test -p <crate> --lib --features setfit <module>`), < 30 s each.
- **Per wave merge:** `make setfit-all-tests` + `pv validate` over touched contracts + `make contract-audit-phase5`.
- **Phase gate:** full tier3 (with the known Darwin/arm64 caveats read correctly) + all in-band negatives green + `/gsd:verify-work`.

### Wave 0 Gaps
- [ ] Production calibration harness (`evidence.rs` sibling, env-gated, `#[ignore]`d) — F-10
- [ ] Per-regime `Thresholds` restructuring + updated `thresholds_match_the_contract` — F-10/D-03
- [ ] `contracts/setfit-benchmark-claims-v1.yaml` + `$(CONTRACTS)` append + scoped audit target — D-15
- [ ] Multiclass ECE/Brier in `aprender-core::calibration` + uv-env fixture generator + SHA-256 manifest — D-07
- [ ] f64 paired-stats helpers + frozen `t_{0.975,9}` constant + scipy fixtures — D-05/D-06
- [ ] Row type + run manifest + doctored-row negative fixtures — D-12/D-14
- [ ] `SetfitCommands::Bench` variants + `commands/setfit_bench.rs` adapter + Make/tier wiring (non-vacuous) — D-13
- [ ] `--selection-manifest` on finetune + explicit `TrainingConfig` control + refusal tests — D-10
- [ ] Per-row-predictions evaluator door in `aprender-train` (beside `evaluate_validation_from_artifact`) — EVAL-01

## Security Domain

ASVS Level 1 scope (`security_enforcement: true`, `security_block_on: high`). This is a local
CLI/file-artifact phase — no auth/session/network surface is added.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (no network surface added; serve untouched) |
| V3 Session Management | no | — |
| V4 Access Control | yes (data-access discipline) | The lock/token typestate for canonical test (SetFit); claims-contract rule for the LoRA side (Open Q3); sealed credential stays sealed unless deliberately widened |
| V5 Input Validation | yes | `deny_unknown_fields` + schema_version on every row/manifest; digest verified BEFORE parse (`SelectionManifest::from_bytes` pattern); stat-then-refuse byte caps before read (16 MiB house cap, 04-18); typed errors, no `unwrap()` |
| V6 Cryptography | yes | SHA-256 via `sha2` only (content hashes, manifest, pairing keys) — never hand-rolled |
| V10 Malicious-input handling | yes | In-band doctored fixtures (D-15's five negatives) prove refusal in every `cargo test` |
| V12 File handling | yes | Atomic writes, temp-in-destination, one rename site; note WR-01 (rename clobber race) and the 04-REVIEW symlink-following warning in the lock write path are OPEN by human ruling — Phase 5 must not silently rely on `atomic_write` for adversarial-clobber protection, and must not fix WR-01 in-phase |

### Known Threat Patterns for this phase

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Doctored/edited row file | Tampering | Content hash in run manifest; `bench report` re-verifies every row before aggregation |
| Selective cell omission | Repudiation/Tampering | Manifest-defined expectation set (80 cells) declared before runs; directory listing never trusted |
| Post-test selection laundering | Integrity of claims | Lock-record consumption (SetFit) + claims-contract LoRA rule; a post-test-selected lock is one of the five contracted negatives |
| Oversized row/manifest DoS | DoS | Bounded reads (stat first, refuse over cap) per the 04-18 pattern |
| Cross-host row forgery in transport | Spoofing/Tampering | Rows are hash-committed before transport; ingest re-verifies; provenance (host identity) inside the hashed bytes |
| Regime-gate loosening | Tampering (of the gate itself) | Three-place synchronized edit pinned by `thresholds_match_the_contract` equality assertion; D-04 human checkpoint |

## Sources

### Primary (HIGH confidence — read/executed in this session)
- `crates/aprender-train/src/train/setfit/thresholds.rs` — full read (constant, grammar, covers(), the pinned test and its len==1 + equality assertions)
- `contracts/setfit-train-lifecycle-v1.yaml` v2.0.0 — calibration_regime + evidence_gate equations, ε-derivation tables, N-02, Phase-5 consequence clause
- `crates/aprender-train/src/train/setfit/{evidence.rs (calibration matrix), tune.rs (validate_evidence), mod.rs (regime id/cell label), lock.rs, apr_evaluate.rs, evaluate.rs, config.rs, baseline.rs}`
- `crates/aprender-core/src/setfit/{mod.rs (from_pretrained_dir), encoder.rs (architecture_fingerprint, ExecutionBackend), import.rs (MiniLmImport::open), classify.rs}`
- `crates/aprender-core/src/{calibration.rs, metrics/agreement.rs, stats/hypothesis.rs, stats/tests_hypothesis_contract.rs}`
- `crates/aprender-train/src/eval/classification/metrics.rs`; `crates/aprender-train/src/finetune/{classify_pipeline/mod.rs, classify_trainer.rs, classify_eval_report.rs}`
- `crates/apr-cli/src/{commands/setfit_train.rs, commands/eval/setfit.rs, commands/finetune.rs, commands/data_contrastive.rs, commands/model_config.rs, commands/bench.rs, setfit_commands.rs, dispatch_analysis.rs}`; `crates/apr-cli/tests/setfit_cli_lifecycle.rs`
- `contracts/tweet-eval-stance-benchmark-v1.yaml` (seed set, 42-exclusion, metadata house style); `contracts/aprender/binding.yaml` (04-19 resolution)
- `scripts/setfit_fixtures/{README.md, pyproject.toml, uv.lock}`; `Makefile` (targets, PV_BIN, $(CONTRACTS) convention)
- Host probes: tool versions, `~/.cache/aprender/minilm-l6-v2-1110a243/` contents, `target/release/pv 0.63.0`
- `.planning/{REQUIREMENTS.md, STATE.md, ROADMAP.md, research/PITFALLS.md}`, `05-CONTEXT.md`

### Secondary (MEDIUM confidence)
- `scripts/dispatch-distill-phase-3-gx10.sh` header (remote-dispatch house pattern; lambda-vector = RTX 4090)

### Tertiary (LOW confidence, flagged for validation)
- Assumptions A1–A7 (training-knowledge items with in-env verification commands specified)

## Metadata

**Confidence breakdown:**
- F-10 mechanics: HIGH — every constant, test assertion, gate order and import path read directly; the two traps (fingerprint prefix, three-place edit) are measured, not inferred
- Standard stack / metrics surface: HIGH — signatures and contract bindings read in-file
- LoRA baseline behavior: HIGH for what the code does today; MEDIUM for `val_split: 0.0` tolerance (A6, needs a probe)
- Statistics protocol: HIGH for formulas/siting; the t-critical constant is [ASSUMED] until the scipy fixture freezes it
- Resource measurement: MEDIUM — mechanism options verified, but no in-repo precedent actually measures peak RSS; the chosen mechanism needs a spike
- Orchestration/lambda-vector: LOW on reachability (unverified from this host), HIGH on the file-transport design (hash verification makes transport trivial)

**Research date:** 2026-08-16
**Valid until:** ~2026-09-15 (stable in-repo domain; re-verify only if Phase 4 secure-phase or the publish cascade lands changes in the setfit modules)
