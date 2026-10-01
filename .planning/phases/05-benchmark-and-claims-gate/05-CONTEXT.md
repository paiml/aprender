# Phase 5: Benchmark and Claims Gate - Context

**Gathered:** 2026-08-16
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers **the milestone's claims layer**: a user can audit and exactly recompute a
complete, selection-safe TweetEval abortion-stance comparison between the verified SetFit APR and
the existing 9B LoRA path across every contracted cell — shots `{8,16,32,64}` × seeds
`{13,17,23,29,31,37,41,43,47,53}`, both methods, 80 rows total — with official `F_avg`, per-class
metrics, macro-F1, MCC, confusion matrix, validation-only calibration diagnostics, resource
metrics, and a fail-closed completeness/pairing gate (EVAL-01..EVAL-05).

It also delivers **the F-10 unblock this milestone is gated on**: a calibration run on the
production `sentence-transformers/all-MiniLM-L6-v2` encoder and the deliberate, `pv diff`-flagged
edit to `contracts/setfit-train-lifecycle-v1.yaml` that adds a production regime entry (Ph3
D-10(c)). Until that lands, no user-reachable path produces a `setfit-apr-v1`, and every "a user
can" requirement blocked on F-10 in the Phase 4 audit (OPS-01, OPS-02, and the positive tiers of
TRN-07, APR-01..05, OPS-03..05) stays open. Phase 5's wave 1 is where they become closable.

**In scope:** the production-encoder calibration run + contract edit; multiclass calibration
metric extensions (top-label ECE, multiclass Brier); paired statistics (mean ± std, paired t-CIs);
the `apr setfit bench run` / `bench report` surface; the method-tagged row schema and hashed
run manifest; the `--selection-manifest` input on `apr finetune --task classify`; the 40 SetFit
cells (CPU) and 40 LoRA cells (lambda-vector GPU); the claims contract with in-band negatives.

**Not in scope (and why):**
- Extending `apr qa` to encoder-only APRs — declared out of scope by user decision here (see
  D-11); its own deferred ticket. `apr qa` is a generative-model gate; SetFit QA lives in
  `apr validate --quality`, `apr eval`, and this phase's bench gate.
- Any relaxation of the SetFit-identity gate other than the deliberate contract edit (D-01..D-04).
  Three Phase 4 executors declined to route around F-10; that ruling stands.
- New encoder families, alternative objectives, GPU SetFit training, quantized artifacts — v2.
- Re-litigating trainer, artifact, or serving semantics — settled in Phases 1–4.
- The Phase 4 leftovers that are explicitly NOT Phase 5 work: the mutation-score compute ticket,
  WR-01 (`O_CREAT|O_EXCL`), and SAFE-02's "in CI" execution (needs the PR the human hasn't opened).

**Phase 4 handoff state:** Phase 4 is EXECUTED and UAT-passed (12/12, `04-UAT.md` at `b3f816c25`)
but NOT closed — `/gsd:secure-phase 04` has not run. Its verification is reconciled as
`gaps_acknowledged` with F-10 deferred to this phase by explicit human ruling (STATE.md).

</domain>

<decisions>
## Implementation Decisions

### F-10 Calibration Unblock

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

- **D-03: Fresh per-class ε and scale floors are derived from the production measurements, as a SECOND regime entry.**
  The Phase 3 fixture entry stays byte-untouched, so `pv diff` shows a
  purely additive edit and Phase 1–3 gates keep their exact meaning. Reusing fixture ε was
  rejected as the exact non-transfer D-10(c) warns about: sparse embedding-table relative deltas
  shrink as vocabulary grows from the 97-row fixture closure to the full ~30k vocab. The
  derivation follows Phase 3's margin methodology.

- **D-04: The contract edit lands only after a human checkpoint.** The calibration plan marks the
  contract-edit task `autonomous: false`: the executor prepares the edit, runs `pv diff` (two
  filesystem paths — materialize the old revision with `git show` first), presents the measured
  thresholds and the diff, and waits for approval before committing. This matches the 04-11
  CI-patch precedent and honours STATE.md's "open by explicit human ruling".

### Statistical Claims Protocol

- **D-05: Headline aggregation is mean ± sample (n−1) std with min/max, per cell and per shot level.**
  Matches the SetFit paper's reporting convention so numbers are directly comparable to
  published results; exactly recomputable from rows with closed-form arithmetic (EVAL-04). This
  closes the pre-recorded STATE.md blocker "Choose validation-only calibration and uncertainty
  estimators before collecting benchmark results" together with D-06/D-07.

- **D-06: Uncertainty and deltas are closed-form paired t statistics.** Per shot level: per-seed
  paired deltas (SetFit − LoRA evaluated on the same sampled-ID hash), the mean delta, and a
  Student-t 95% CI over the 10 paired differences. No RNG anywhere in the claims path — EVAL-04's
  "exactly recompute" is bit-level, and a bootstrap would put a resampling stream inside it.
  Stats functions get scipy/sklearn reference fixtures per the `glm_tests.rs:280` house precedent.

- **D-07: Calibration diagnostics are top-label ECE + multiclass Brier, canonical validation only.**
  The existing contract-bound `expected_calibration_error` / `brier_score`
  (`aprender-core/src/calibration.rs`) are binary-only (`&[bool]` labels); this phase extends the
  surface with the standard multiclass pair, bound to explicit ordered labels. Per-class OvR
  suites were rejected as fixture-verification burden the claims don't need.

- **D-08: The report is estimation-first — no significance verdicts.** It states point estimates,
  dispersion, and paired CIs; it never prints "significantly better/worse". This is the PF-007
  guard (few-shot seed sensitivity hiding behind a binary verdict). p-values may sit in the
  machine-readable detail, never in claim language.

### LoRA Baseline Execution

- **D-09: LoRA cells run on the lambda-vector GPU host (pre-authorized); SetFit cells run CPU.**
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

### Claims Surface and Gate

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

### Carried Forward (not re-litigated)

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

### F-10 Calibration Unblock — Measured (supersedes parts of D-02/D-03)

Plan 05-01 was front-loaded specifically so F-10 would fail early if it was going to
(D-01). It did. These decisions are **measured, not deliberated**, and they supersede
parts of D-02/D-03 written before the evidence existed. Source of record:
`05-01-calibration-measurements.md`, `05-01-SUMMARY.md`, and the 12 persisted evidence
files under the plan's calibration store (each carrying a verified `evidence_sha256`).

- **D-16: The 10×/10× window rule does NOT survive production step counts; this supersedes D-03's premise.**
  At the production envelope (s64 = 1536 steps, quadratic
  budget `2·max(pos_cap, neg_cap)` with `neg_cap = 3n²`), five of six parameter classes
  have **no legal ε**: `embedding`, `layer_norm_bias`, `projection_weight`,
  `projection_bias`, `attention_key_bias`. Only `layer_norm_weight` survives. The
  collapse is not an artifact of mixing cells — it holds within s64 alone
  (`real/near-null = 39.7` against a rule requiring `> 100`). Cross-cell:
  `lower = 10 × 9.278e-9 = 9.278e-8` exceeds `upper = best_real/10 = 1.714e-8` by 5.41×.

  Cause is the **near-null (1e-8) leg**, not the control: the 1e-30 control writes back
  bit-identical weights at 1536 steps (max `relative_delta` 0.0, zero rows moved), so
  1e-30 underflows every parameter's ULP however often applied. With `worst_ctrl` pinned
  at zero, near-null alone sets every window's lower bound, and it grows 56–72× from 24
  to 1536 steps (299× for `attention_key_bias`).

  05-03 must therefore choose a replacement rule on this evidence. It must NOT freeze ε
  from the current table: the report's cross-cell basis is explicitly PROVISIONAL (3 of 6
  boundary cells; s64:31 and s64:53 unmeasured), and unmeasured cells can only narrow
  windows further, never widen them.

- **D-17: `attention_key_bias` has NO available justification for its current treatment.**
  Both options are now closed. "Ungated because it receives no gradient
  (`dL/db_k = 0` by softmax shift-invariance)" is refuted by measurement:
  `grad_norm_max = 8.084e-10` against a `9.051e-10` noise floor, with deltas scaling with
  the learning rate (~3600× delta ratio for a 2000× LR ratio, consistent across three s8
  seeds). In f32 the shift-invariance is only *approximate*, so a ~1e-10 residual gradient
  is the expected numerical consequence — not a bug, but not zero either. "Gated at a
  frozen ε" is unavailable because its window is empty.

  Its previously-tracked `eps/noise = 15.1` **must not be carried forward**: that column
  is `upper / noise_floor` and was computed regardless of whether `lower < upper`, so with
  `supports_margin = false` it divided an illegal ε by the noise floor. 05-01 fixed the
  reporting (now `n/a`) in both the production and fixture tables.

  **What is NOT in question:** separation. `ctrl_max = 0.000e0 < real_min` for all six
  classes in every measured cell. Real training remains cleanly distinguishable from no
  training. 05-03 must not over-read D-16/D-17 as "calibration is impossible" — what
  failed is ε-freezing under one specific rule.

- **D-18: The evidence gate is not fail-closed on window collapse — a defect to fix, not a fact to work around.**
  Separation is *asserted*; `supports_margin` is only
  *reported*. Verified, not inferred: the four-cell combine exited `rc=0` while printing
  `EMPTY` five times. A gate that reports success with an empty ε basis is theatre in
  exactly the surface F-10 exists to protect. Whatever rule D-16 settles on, the run must
  FAIL when a class has no legal ε.

- **Method facts worth reusing (measured, cheap to lose):** debug and release measure
  bit-identically (every relative delta, `delta_norm`, `init_norm`, `grad_norm_max` and
  noise floor agreeing to the last digit) while release is ~34.5× faster — the build
  profile is a price, not a degree of freedom. Cross-process determinism of persisted
  evidence is proven bit-identical (same sha256 across two fresh processes), which is what
  licenses combining per-condition passes. The architecture component
  `h384-l6-a12-i1536-v30522@1110a243` is byte-identical across `cells=s8e1b16` and
  `cells=s64e1b16` over a genuinely cross-half 12-pass combine — both halves measured the
  same 22M-param production encoder, not the 97-token fixture slice, which is what makes
  D-16 a result about step count rather than about two different models.

- **Open, deliberately untested:** the superlinear-growth **hypothesis** — 299× over a 64×
  step increase suggests the s8 near-null is underflow-dominated, while at 1536 steps
  (warmup complete, Adam normalising each step to ~the learning rate) far more updates
  survive rounding. It would be tested by measuring near-null at an intermediate step
  count. It has NOT been tested and must not be stated as a result.

- **D-19: The 9B LoRA comparison arm is not buildable in this phase; EVAL-02/EVAL-04 descope to a
  SetFit-only claim set.** *(recorded 2026-09-07, at plan 05-11's Task 1 checkpoint, before any
  compute was spent — no LoRA cell was ever run.)*

  Two independent findings, either of which alone blocks the arm.

  **(a) No GPU host exists to run it on.** `lambda-vector`/`gx10` — the host every
  `scripts/dispatch-*.sh` hardcodes (`GX10_HOST=gx10`, `GX10_USER=noah`,
  `/home/noah/src/aprender`) — is unreachable: 12 enumerated candidates, 12 failures, rc captured
  per attempt (8 × `Could not resolve hostname`, 4 × TCP timeout on the two RFC1918 addresses in
  `known_hosts`). Credentials are not the missing piece; `~/.ssh/config` holds one stanza (`rvsc`,
  an AWS EC2 box) and there is no tailscale, no lambda CLI, and nothing in `/etc/hosts`. Human
  ruling: the box belongs to the project's main maintainer and is not accessible to us. The AWS
  fallback was enumerated and refuted rather than assumed — all five instances in the account
  (eu-west-1, eu-west-2, us-east-1 ×3) are **stopped**, and none is a GPU instance; the largest is
  a `t4g.large`. `rvsc` itself is a `t3.small`, which is exactly right for what it did do (the
  Chronos x86_64 parity run, `06-UAT.md:36`) and cannot hold a 9B model.

  **(b) More decisively, aprender cannot run the Qwen3.5-9B architecture — so a GPU would not have
  helped.** The weights are NOT the blocker: `Qwen/Qwen3.5-9B` is public and ungated, revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 19.31 GB bf16 across 4 shards. The blocker is
  structural, three gaps, each independently fatal:

  | The checkpoint declares | `TransformerConfig` (`crates/aprender-train/src/transformer/config.rs`) has |
  |---|---|
  | `layer_types` — **24 `linear_attention` + 8 `full_attention`** over 32 layers, `full_attention_interval: 4` (measured from the live config, not inferred) | no `layer_types` field — 14 `pub` fields, uniform layers only |
  | `attn_output_gate: true` | no such field |
  | `Qwen3_5ForConditionalGeneration`, `image_token_id: 248056`, image+video preprocessor configs, every text hyperparameter nested under `text_config` | `qwen3_5_9b()` is a flat, text-only config |

  The only hybrid-forward artifact in the tree is
  `crates/aprender-contracts-staging/generated/qwen35-hybrid-forward-v1_scaffold.rs`, and
  `aprender-contracts-staging` **has no `Cargo.toml`** — CLAUDE.md names it as one of the two
  directories under `crates/` that are not crates. It never compiles. It is a scaffold.

  `contracts/qwen35-e2e-verification-v1.yaml` does not contradict this: all seven of its
  falsification tests are **analytical** — parameter count, FLOPs-per-token, quantized-memory
  ordering, roofline, obligation coverage, per-block shape preservation, layer composition. Not
  one loads a weight or compares against the reference implementation. It verifies the
  *description* of the architecture, not an implementation of it. Consistent with that, every
  `"9B"` occurrence in the workspace resolves to config, CLI dispatch, or `training_plan.rs`
  (cost *estimation*) — nothing that loads 9B weights.

  Corroborating scale evidence: 05-06 Task 3's reload preflight, the precondition licensing 9B
  compute, passed against a **783,236-byte** base model. The LoRA machinery is proven at fixture
  scale and has never met a real 9B checkpoint.

  **Confidence, stated honestly.** (b) rests on *structural absence* — fields that do not exist in
  a config struct — not on an observed loader failure. By this repo's own Verification Discipline
  rule 6 that is an inference from one line of evidence, not a measurement. The cheap falsifier
  needs no GPU and was offered before this amendment was written: point aprender's loader at the
  real `config.json` and see whether it accepts or rejects the checkpoint. **It was not run.** If
  it ever loads cleanly, D-19(b) is refuted and the arm becomes a pure host-access question again.
  (a) is measured and stands on its own regardless.

  **What this changes.** EVAL-02 and EVAL-04 drop the LoRA arm and the paired-delta clause; the
  Phase 5 goal and success criteria 2 and 4 follow. **The claims gate itself is untouched** —
  05-10's fail-closed `verify_run` and the "any missing, selectively omitted, unmatched or
  post-test-selected cell invalidates the report" behaviour are unchanged, and the *declared
  matrix* shrinks from 40 comparison cells (80 rows) to 40 SetFit rows.

  **CORRECTED 2026-09-08 — the six doctored negatives do NOT survive byte-for-byte.** This
  record originally said they did; that was written from the intent of the descope rather than
  from the code, and it is retracted. `EXPECTED_CELLS` is a product of the method list, and two
  of the six negatives target a second method's row — `BenchGateError::UnpairedSelection` carries
  a `lora_hash` field (`bench_gate.rs:449`), and the forged-candidate-ledger case likewise needs
  a counterpart row. Under a 40-cell single-method expectation neither is constructible from a
  production path. Disposition: both keep their shape and variant tag under a retained deferred
  scope with no production constructor; the remaining four are **re-mutated at the new scope**,
  per Verification Discipline rule 4 — the 80-cell RED proof does not transfer to a 40-cell gate.
  The gate's *strength* is therefore preserved by re-proof, not by assertion of sameness. The phase's thesis — *reject incomplete or unequal claims* — is preserved by
  refusing to publish a comparison we cannot substantiate, which is that thesis applied to
  ourselves.

  **What this does NOT license.** This is not a ruling that the 9B path is unnecessary, and it is
  not permission to compare SetFit against nothing and call it a win. The comparison is deferred
  as real work, not cancelled: see `D-ITEM-05-15`. No report produced under this amendment may
  state or imply a SetFit-versus-LoRA result.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase requirements and prior decisions
- `.planning/ROADMAP.md` § "Phase 5: Benchmark and Claims Gate" — goal, the five success
  criteria, and the blocker paragraph this phase's D-01..D-04 discharge.
- `.planning/REQUIREMENTS.md` — EVAL-01..EVAL-05 verbatim; the "Phase 4 closing audit" section
  (items 5 and 7 are Phase 5 inputs: the `backend_identity` binding row and the `apr qa` ruling);
  the F-10 blocker paragraph naming `CALIBRATED_REGIMES` and `probe_unicode`.
- `.planning/STATE.md` — the four items OPEN by human ruling (F-10 is item 1 and is THIS phase's
  work; items 2–4 are NOT); the Phase 5 pre-recorded blocker on calibration/uncertainty
  estimators (closed by D-05..D-07); handler-damage repair rules.
- `.planning/phases/04-apr-artifact-and-production-parity/04-CONTEXT.md` — Ph4 D-01..D-16.
  D-05/D-06 (CLI namespace rules D-13 extends), D-08 (shared response envelope), D-11 (probe
  replay), D-12 (backend identity), D-16 (`apr eval` lock workflow) are load-bearing here.
- `.planning/phases/03-faithful-two-stage-trainer-and-head/03-CONTEXT.md` — Ph3 D-10 (the regime
  mechanism D-02/D-03 calibrate), D-11 (baseline types Phase 5 may run), D-12 (evidence record in
  rows), D-14 (selection lock semantics EVAL-04 relies on).
- `.planning/phases/02-deterministic-pair-and-data-protocol/02-CONTEXT.md` — Ph2 D-08 (the
  selected-ID manifest D-10 plumbs into finetune), D-16/D-19 (split typestates), D-27 (exclusion
  semantics that keep all 40 cells alive).
- `.planning/phases/04-apr-artifact-and-production-parity/04-UAT.md` — the executed evidence
  baseline Phase 5 builds on (all 12 gates, at `b3f816c25`).
- `.planning/research/PITFALLS.md` — PF-007 (seed sensitivity → D-08's estimation-first rule),
  PF-008 (never attribute deviations to SetFit).

### Contracts (use `pv`, never bash/yq/python)
- `contracts/setfit-train-lifecycle-v1.yaml` — THE file D-01..D-04 edit. The `calibration_regime`
  equation (regime-id formula, component-wise membership, the single-fingerprint calibrated set)
  and OBLIG-STL-REGIME-FAIL-CLOSED are the exact semantics the new entry must satisfy. The edit
  is additive: a second regime entry, fixture entry byte-untouched.
- `contracts/tweet-eval-stance-benchmark-v1.yaml` (v2.0.0) — dataset identity, labels, split
  sizes, the contracted seed set, official `F_avg` invariant, provenance obligations. The claims
  contract references it rather than restating it.
- `contracts/contrastive-pair-protocol-v1.yaml` — selection/manifest semantics `--selection-manifest`
  consumes.
- `contracts/setfit-apr-v1.yaml` — artifact schema + load rules; benchmark cells consume
  artifacts under this contract.
- `contracts/setfit-encoder-conformance-v1.yaml` — Phase 1 tolerances the calibration run must
  not contradict.
- `contracts/linear-probe-classifier-v1.yaml` — the distinctly-named baseline path if planning
  includes a frozen-probe reference row.
- `contracts/calibration-v1` bindings in `crates/aprender-core/src/calibration.rs` — the existing
  binary ECE/Brier contract surface D-07 extends.
- `contracts/aprender/binding.yaml` — the pending `backend_identity` row (Phase 4 amendment 5)
  that row work should resolve or explicitly re-record.

### Code the phase builds on
- `crates/aprender-train/src/eval/classification/` — `MultiClassMetrics`, `ConfusionMatrix`,
  `f1_average_for_classes` (contract-bound official `F_avg`), sklearn-parity tests. EVAL-01's
  quality-metric substrate.
- `crates/aprender-core/src/metrics/agreement.rs` — `matthews_corrcoef` (sklearn-parity tested).
- `crates/aprender-core/src/calibration.rs` — binary `expected_calibration_error` /
  `brier_score`; D-07 extends alongside, not in place.
- `crates/aprender-train/src/train/setfit/` — `lock.rs`, `evaluate.rs` (trusted evaluator +
  token minting), `evidence.rs` (row evidence schema), `config.rs`, `verify.rs`,
  `apr_evaluate_tests.rs`. The gate machinery rows and the report consume.
- `crates/aprender-train/src/train/setfit/` regime code — `CALIBRATED_REGIMES` and the
  `UncalibratedRegime` path (the thresholds/contract parse pinned by
  `the_rung_numbering...`-style include_str tests; find via `pmat query "calibration regime"`).
- `crates/apr-cli/src/commands/setfit_train.rs`, `commands/eval/setfit.rs`,
  `commands/inspect_setfit.rs` — the spawned-tier lifecycle 04-15 proved up to rung 4; wave 1
  re-runs this ladder to completion after the contract edit.
- `crates/apr-cli/src/commands/finetune.rs` + `entrenar::finetune::classify_pipeline` — the 9B
  LoRA baseline D-09/D-10 drive; `run_classify` is the dispatch entry.
- `crates/apr-cli/tests/setfit_cli_lifecycle.rs` — the seven-process TRN-07 ladder; the pattern
  for spawned-tier bench evidence.
- `crates/aprender-contrastive-data/src/` — `select.rs` (`Selection`), `manifest.rs` (the
  manifest `--selection-manifest` verifies), `ledger.rs`.
- `crates/apr-cli/src/commands/data_contrastive.rs` — the CLI adapter style (attested ingest,
  atomic writes) `bench` follows.

### Project rules
- `CLAUDE.md` — Verification Discipline (rule 2 for backend identity, rule 6 "one failing input
  is an anecdote" — already applied to `apr qa` by the Phase 4 control), `pv` dogfooding, apr
  binary pinning (`scripts/apr_bin.sh`), tiered gates, branch protection, pre-authorized
  lambda-vector compute.
- `.claude/skills/pre-release/SKILL.md` — the standing KNOWN-RED (`cargo package -p apr-cli`
  until the human-run publish cascade) that any Phase 5 pre-release check must read correctly.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Quality metrics are substantially built:** `MultiClassMetrics::from_predictions`,
  `ConfusionMatrix`, contract-bound `f1_average_for_classes`, `matthews_corrcoef` — EVAL-01
  needs assembly + the multiclass calibration extension (D-07), not a metrics rewrite.
- **The selection-safety machinery is complete and production-proven at the CLI tier:**
  Phase 2 manifests, Phase 3 locks/tokens, Phase 4's `apr eval` workflow. EVAL-04's
  post-test-selection invalidation is a consumer of `lock.rs`/`evaluate.rs`, not new mechanism.
- **The spawned CLI ladder** (04-15) is the template for proving the unblocked lifecycle: rungs
  0–3 already pass; rung 4 (`setfit train` exit 6) is exactly what D-01's plan flips to exit 0.
- **`apr finetune --task classify`** exists end-to-end (`run_classify`,
  `ClassifyConfig`) — D-10 adds an input path, not a training path.
- **Phase 2's CLI adapter + manifest house style** (`data_contrastive.rs`, atomic writes,
  digest-verified manifests) maps directly onto D-14's row/manifest layout.

### Established Patterns
- **Deliberate contract edits with `pv diff` evidence** — materialize old revision via
  `git show`, diff two paths, record the suggested bump and the reasoning in-file (the
  tweet-eval contract's own header is the worked example D-04 follows).
- **Reference-fixture falsification for numerics** (`glm_tests.rs:280` scipy precedent; Phase 1's
  hash-locked `uv` env) — applies to the t-statistics, ECE, and Brier implementations.
- **In-band negatives in every `cargo test`** — D-15's doctored row sets are this phase's
  instance.
- **Gates that cannot go vacuous** — every new Make/CI filter must fail if it ran nothing
  (CR-02 lesson); applies to bench gate wiring.
- **"First consumer, not owner" siting** — stats/calibration extensions go where the general
  capability lives, with the benchmark as first caller.

### Integration Points
- `contracts/setfit-train-lifecycle-v1.yaml` gains the second regime entry (D-02/D-03) — the
  single highest-risk edit in the phase; the regime-parse tests (`include_str!`-pinned) must stay
  green and the fixture entry byte-identical.
- `apr-cli` gains the `setfit bench` subcommands (D-13) and finetune gains
  `--selection-manifest` (D-10).
- The claims contract (D-15) references tweet-eval-stance-benchmark-v1 and setfit-apr-v1 rather
  than editing them (Ph1 D-23 pattern).
- Rows consume: Ph3 D-12 evidence summaries, APR artifact hashes, lock records, backend identity
  from execution — all already emitted by Phase 3/4 surfaces.
- The lambda-vector GPU host enters the evidence chain for the first time: row provenance must
  make host heterogeneity explicit (D-09), and orchestration must move rows into the manifest'd
  benchmark directory without breaking hash verification.

</code_context>

<specifics>
## Specific Ideas

- **The unblock is the phase's keystone, and it is high-ceremony on purpose.** Three Phase 4
  executors declined to synthesize an APR-capable encoder that would have passed their acceptance
  criteria; the human ruling routed the fix here as a deliberate contract edit. D-04's
  `autonomous: false` checkpoint is the structural form of that ceremony.
- **"Exactly recompute" means no RNG in the claims path** — that single constraint drove D-06
  (closed-form t over seeded bootstrap). If a future reviewer wants bootstrap robustness, it is
  an additive secondary column, never a replacement for the closed-form primary.
- **The comparison's honesty lives in the framing:** as-deployed method costs on heterogeneous
  hardware, declared per-row — not a same-hardware kernel shootout (D-09). The report must say
  what was run where; it must never average across hosts as if they were one.
- **Completeness is defined by the manifest, not the directory** (D-14) — "selectively omitted
  cells invalidate the report" is only checkable against a pre-declared expectation set.
- **`apr qa`'s refusal is correct behavior, misdocumented** — the fix in scope is the error
  message; the capability is a ticket (D-11).

</specifics>

<deferred>
## Deferred Ideas

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

</deferred>

---

---

*Phase: 5-Benchmark and Claims Gate*
*Context gathered: 2026-08-16*
*Amended: 2026-08-17 with plan 05-01 execution evidence (D-16..D-18)*
*Amended: 2026-09-07 at plan 05-11's Task 1 checkpoint — the 9B LoRA arm descoped (D-19)*
