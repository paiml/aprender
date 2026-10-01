# Phase 5: Benchmark and Claims Gate - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-16
**Phase:** 5-benchmark-and-claims-gate
**Areas discussed:** F-10 calibration unblock, Statistical claims protocol, LoRA baseline execution, Claims surface & QA ruling

Pre-discussion gates (from the plan-phase invocation that routed here): the user chose to run
discuss-phase before planning, chose to plan without an AI-SPEC (the keyword hit was the substring
"eval" in "TweetEval"), and chose research-first for the plan-phase re-run.

---

## F-10 calibration unblock

| Option | Description | Selected |
|--------|-------------|----------|
| Wave 1, gating plan | First plan of Phase 5: calibrate production encoder, land the pv diff-flagged contract edit, prove one user-produced setfit-apr-v1 before harness work | ✓ |
| Parallel with harness build | Calibration alongside harness plans; only cell-execution plans gate on it | |
| Separate pre-phase | Split calibration into its own mini-phase (5a) | |

| Option | Description | Selected |
|--------|-------------|----------|
| Full envelope, boundary-measured | Entry lists all 10 seeds + all 4 cell labels; thresholds measured on boundary cells plus documented margin; contract records measured vs covered | ✓ |
| Full envelope, all 40 measured | Measure every seed×cell before freezing ε (~doubles training compute) | |
| Minimal, extend as needed | Calibrate 1–2 cells, extend later (repeated contract edits, mid-benchmark UncalibratedRegime possible) | |

| Option | Description | Selected |
|--------|-------------|----------|
| Fresh ε from production runs | New per-class ε + scale floors from production measurements as a SECOND regime entry; fixture entry byte-untouched | ✓ |
| Reuse fixture ε, add fingerprint only | Apply 97-row-vocab numbers to a 30k vocab — the non-transfer D-10(c) warns about | |
| You decide | Leave derivation policy to research/planning | |

| Option | Description | Selected |
|--------|-------------|----------|
| Human checkpoint in-plan | Contract-edit task `autonomous: false`; executor presents pv diff + measured thresholds and waits (04-11 precedent) | ✓ |
| Autonomous with pv diff evidence | Executor lands the edit itself with evidence in the SUMMARY | |

**User's choice:** Recommended option in all four questions.
**Notes:** None.

---

## Statistical claims protocol

| Option | Description | Selected |
|--------|-------------|----------|
| Mean ± sample std + min/max | SetFit-paper convention, closed-form recomputable | ✓ |
| Median + IQR | Robust but not paper-comparable | |
| Both families | Headline mean±std with median/IQR alongside | |

| Option | Description | Selected |
|--------|-------------|----------|
| Paired t-CI, closed form | Per-seed paired deltas on the same sampled-ID hash; Student-t 95% CI over 10 differences; no RNG in the claims path | ✓ |
| Seeded paired bootstrap | Percentile bootstrap with contracted recorded seed | |
| Both: t primary, bootstrap secondary | t headline plus bootstrap robustness column | |

| Option | Description | Selected |
|--------|-------------|----------|
| Top-label ECE + multiclass Brier | Standard pair, modest extension of the binary contract-bound functions, validation-only | ✓ |
| Classwise one-vs-rest suite | Per-class OvR ECE + per-class Brier plus aggregates | |
| Both | Top-label headline plus per-class detail | |

| Option | Description | Selected |
|--------|-------------|----------|
| Estimation-first, no verdicts | Point estimates, dispersion, paired CIs; no "significantly better" labels (PF-007 guard) | ✓ |
| CIs + explicit significance verdicts | Labels each delta significant/not at a contracted α | |

**User's choice:** Recommended option in all four questions.
**Notes:** Together these close the pre-recorded STATE.md blocker "Choose validation-only
calibration and uncertainty estimators before collecting benchmark results."

---

## LoRA baseline execution

| Option | Description | Selected |
|--------|-------------|----------|
| LoRA on GPU host, SetFit on CPU | lambda-vector (pre-authorized) for LoRA; true backend/hardware identity per row; as-deployed cost framing | ✓ |
| Both methods on one Linux host | Shared environment but adds a SetFit platform-parity question (all prior evidence is macOS/arm64) | |
| Harness first, runs deferred | Land harness/gate/stats; 40-cell compute as a separate ticket | |

| Option | Description | Selected |
|--------|-------------|----------|
| Manifest flag on finetune | `apr finetune --task classify --selection-manifest` through the shared Phase 2 manifest→Selection path; refuses on hash mismatch | ✓ |
| Pre-materialized subset files | Exporter writes per-cell dataset files; no finetune change but identity depends on the exporter | |
| You decide | Let research pick from finetune's input formats | |

| Option | Description | Selected |
|--------|-------------|----------|
| Frozen published defaults, both methods | One contracted config per method across all 40 cells; no selection to leak | |
| Validation-tuned, equal budget | Same validation-only search budget per method with locks on both sides | |
| You decide | Research settles it from the classify pipeline's actual defaults | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| Method-tagged row schema | Shared mandatory core + method-specific evidence block; gate pairs on the core; missing block invalidates | ✓ |
| Uniform schema, null-filled for LoRA | Flat schema with empty SetFit-only fields (cuts against the Phase 4 nullable-allowlist discipline) | |
| You decide | Planning shapes the schema from the gate's needs | |

**User's choice:** Recommended options, except hyperparameter policy = "You decide"
(Claude's discretion — research settles from the classify pipeline's actual defaults).
**Notes:** None.

---

## Claims surface & QA ruling

| Option | Description | Selected |
|--------|-------------|----------|
| apr setfit bench run/report | Namespaced cell runner + fail-closed report recompute; extends Ph4 D-05's namespace | ✓ |
| Extend generic apr bench | Comparison mode on the brick/syscall-profiling command (poor fit) | |
| Separate driver script + apr eval | Orchestration outside the contracted binary surface | |

| Option | Description | Selected |
|--------|-------------|----------|
| Row-per-file + hashed manifest | Schema-versioned JSON row per cell; manifest of all 80 expected cells with content hashes defines completeness | ✓ |
| Single append-only JSONL | Simpler but weaker against partial writes and selective trimming | |
| You decide | Planning picks the layout | |

| Option | Description | Selected |
|--------|-------------|----------|
| Out of scope + deferred ticket | apr qa stays generative-only; refusal-site error message improved; extension is its own ticket | ✓ |
| Extend apr qa this phase | Teach it an encoder/classifier capability path now | |
| You decide | Research sizes the extension | |

| Option | Description | Selected |
|--------|-------------|----------|
| Contract + in-band negatives | New claims contract owns row schema/completeness/pairing/stats; doctored row sets must be REFUSED in every cargo test | ✓ |
| Report-command tests only | Unit/integration tests without a contract | |

**User's choice:** Recommended option in all four questions.
**Notes:** None.

---

## Claude's Discretion

- Hyperparameter policy for the two methods (explicit "you decide" — the one non-recommended
  selection of the discussion); research settles from the classify pipeline defaults, leaning
  frozen-defaults-both-methods.
- Resource-measurement boundaries (cold/warm, warmup/batch, peak-memory mechanism).
- Benchmark directory location, file naming, report output formats.
- 40-cell orchestration and GPU-host row transport.
- Exact multiclass Brier/ECE formulations + reference-fixture sources.
- Stats function siting (`entrenar::eval` vs `aprender-core::metrics`).
- Whether `bench run` shells out to existing commands or calls library APIs.

## Deferred Ideas

- Extend `apr qa` with an encoder/classifier capability path (deferred ticket; refusal-site
  message fix stays in scope).
- Bootstrap robustness columns as an additive secondary to the closed-form paired t.
- Validation-tuned equal-budget comparison (only if frozen defaults prove indefensible).
- Per-class one-vs-rest calibration suites.
