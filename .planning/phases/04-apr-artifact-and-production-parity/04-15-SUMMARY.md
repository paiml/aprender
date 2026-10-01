---
phase: 04-apr-artifact-and-production-parity
plan: 15
status: PARTIAL
subsystem: apr-cli
tags: [setfit, cli, spawned-process, ops-02, trn-07, d-01, d-04, a3, f-10, finding, integration-test]

# Dependency graph
requires:
  - phase: 04-06
    provides: "`apr setfit train` and its flag set; the offline `--model-dir` contract; the dry-run report shape"
  - phase: 04-07
    provides: "`apr predict` / `apr inspect` / `apr eval --task classify`; `read_setfit_tag`; and the IN-PROCESS TRN-07 test that assigns the SPAWNED citation to this plan"
  - phase: 04-12
    provides: "F-10 measured on three independent routes, and the refused shortcut this file refuses again"
provides:
  - "crates/apr-cli/tests/setfit_cli_lifecycle.rs — the ONLY spawned-binary evidence in this phase"
  - "The SPAWNED TRN-07 citation 04-07 assigned here, honestly scoped to the half that is reachable"
  - "The FIRST executed evidence for D-01's generic-tooling promise (research assumption A3)"
  - "A NEW FINDING with its control: `apr qa` does not apply to any encoder-only APR, and the cause is not the SetFit entries"
affects: [04-10, 04-11, phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A spawned-process test reads its verdict from a reaped ExitStatus and never wires one child into another; the source guard asserts zero `Stdio::from(` so the pipeline defect cannot be introduced later"
    - "One pinned-binary constant beats five copies of the env lookup: the guarantee is carried by `exactly one Command::new site, and it takes the constant`, which a high occurrence count would not give"
    - "Two invocations differing in ONE flag and producing TWO different refusals is what turns 'the gate fired' from a label into a measurement"
    - "A tagged-but-invalid container is not a manufactured artifact: nothing goes green because of it, and the binary itself asserts it does not load as a model"
    - "A surprising verdict gets a CONTROL before it gets a cause — the qa finding inverted from 'the U8 entry broke tooling' to 'qa is a generative-model gate' on one extra input"

key-files:
  created:
    - crates/apr-cli/tests/setfit_cli_lifecycle.rs
  modified: []

key-decisions:
  - "DID NOT manufacture a setfit-apr-v1 artifact to close the five-rung chain. `SetFitArtifactView`'s fields and `write_setfit_apr` are public, so it would have compiled and would have satisfied every acceptance criterion to the letter — while the model classified was never produced by the train rung. OPS-02 is a claim about the JOIN between the rungs."
  - "The blocked rung is a PASSING test asserting the typed refusal with a named exit code, and panics with eight lines of restore instructions the day F-10 closes. An #[ignore]d full-chain test would fail under --ignored, which is how 04-10 gates this phase."
  - "The TRN-07 spawned citation covers the NEGATIVE half of the durable-lock workflow and says so in its own header. That half is the one that actually guards the canonical test split, and it is fully reachable because `check_split_flags` runs at step (1), before the corpus and before the artifact."
  - "The Phase 2 fixture recipe is replicated rather than reused, because the writer that owns it is `#[cfg(test)] pub(crate)` and an integration test is out-of-crate. The replication is self-diagnosing: the counts are validated by the shipped command, not asserted here."
  - "`apr qa`'s nonzero verdict is recorded as a FINDING with a control, not suppressed and not blamed on the U8 entry. One failing input would have produced the wrong cause."

requirements-completed: []

# Metrics
duration: ~2h30m
completed: 2026-08-15
---

# Phase 04 Plan 15: The Spawned CLI Lifecycle — PARTIAL (F-10), and one new finding

**One file, three spawned tests plus a source guard, 1295 lines, all green.** The two review
findings this plan was created to close are both answered — one fully, one to the boundary
F-10 draws — and a third thing came out of it that nobody had measured: **`apr qa` does not
apply to a SetFit artifact, and the reason is not the reason it looks like.**

`crates/apr-cli/tests/setfit_cli_lifecycle.rs` is the only place in this phase where the
**shipped binary** is the subject. Everything else tests modules.

## Task Commits

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | the spawned OPS-02 ladder — 6 invocations, rungs 0-5 | `5883e2bc4` |
| 2 | the spawned TRN-07 lock gate (7 invocations) + D-01/A3 tooling (4 invocations) + source guards | `b8755fdc3` |

`git diff --name-only HEAD~2 HEAD` lists **exactly one file**. No `src/` file, no `Cargo.toml`,
no `Cargo.lock`, no shared orchestrator artifact, and nothing 04-09 owns. Zero deletions.

---

## FINDING 1 (NEW, this plan) — `apr qa` does not apply to an encoder-only APR, and the U8 entry is NOT why

This is the one nobody had run. It is also the one where the obvious reading is wrong.

```
$ apr qa <setfit-shaped container> --json
exit 5   error: Validation failed: APR missing embedded tokenizer

$ apr qa crates/aprender-core/tests/fixtures/setfit/slice_model.apr --json   [CONTROL]
exit 5   error: Validation failed: APR missing embedded tokenizer
```

The tempting conclusion from the first line alone is *"the U8 `tokenizer.blob` broke qa —
assumption A3 falsified."* **The control refutes it.** `slice_model.apr` is a plain Bert APR: no
SetFit tag, no `setfit.head.*` entries, no U8 blob. It gets the **identical exit code and the
identical message.**

The cause, read from the source rather than guessed
(`commands/output_verification.rs:505-527`): qa's gate calls
`AprV2Model::load_embedded_bpe_tokenizer()` and then `run_inference` with a token budget and a
top-k. **`apr qa` is a GENERATIVE-model gate.** It does not apply to any encoder-only APR,
classifier or not — and a SetFit artifact's tokenizer is a WordPiece blob at `tokenizer.blob`,
not a BPE blob at the key that gate reads.

**This is a real limit on D-01's "generic tooling keeps working unmodified", and it is narrower
and more actionable than the version without the control.** It is not "the SetFit schema breaks
qa"; it is "qa's gates are LLM gates". Both verdicts are asserted in the test, so the day either
changes the test turns red and hands its author the measurement instead of a stale sentence here.

**Owner: not this plan.** Making qa applicable to a classifier means either a classifier-shaped
gate set or an explicit "this artifact is not generative" verdict. That is a Phase 5 decision.

---

## FINDING 2 (executed, SURVIVED) — D-01 / assumption A3: the U8 pseudo-tensor is fine

04-RESEARCH flagged A3 — a U8 tensor entry surviving generic tooling — as assumed and never
tested. It is now tested. `apr tensors`, verbatim from the run:

```
[04-15] apr tensors row: │ setfit.head.bias   │ [3]    │ f32   │ 12 B │
[04-15] apr tensors row: │ setfit.head.weight │ [3, 8] │ f32   │ 96 B │
[04-15] apr tensors row: │ tokenizer.blob     │ [64]   │ u8    │ 64 B │
```

All three schema-owned entries are **named, shaped and correctly typed**. The U8 entry is
rendered as `u8` with its length — not omitted, not crashed on, not reported as a nonsense f32
statistic. `apr inspect` (human mode) exits 0 with non-empty output on the same file.

**The falsification attempt failed to falsify, which is the useful outcome and the one that had
never been run.** The mutation that removes the blob turns the assertion red (M2 below), so this
is a live check rather than a sentence.

---

## FINDING 3 (re-confirmed at a new tier) — F-10 stops OPS-02 at rung 4, spawned

04-12 measured F-10 through the library on three routes. This is the same wall observed through
the **binary**, which is the tier OPS-02 is written at:

```
$ apr --json setfit train --config train.toml --data benchmark \
      --selection benchmark/selection-manifest.json --model-dir <slice fixture> \
      --output model.apr
rc=6
error: Model load failed: --model-dir <...>/tests/fixtures/setfit:
  SetFitError::ImportIo(config.json: No such file or directory (os error 2)).
  Point --model-dir at a pinned all-MiniLM-L6-v2 checkout containing tokenizer.json and the
  encoder weights. This command NEVER downloads: obtain the pinned revision beforehand
  (for example with `batuta hf pull`) and pass the directory.
```

Reaped status, from the test's own transcript:
`ExitStatus(unix_wait_status(1536)) (code Some(6))`, elapsed `197.86ms`.

**`model.apr` does not exist afterwards** — asserted directly and again as a set difference over
the whole tempdir. Rung 5 (`apr inspect model.apr --json`) is run anyway and refuses, so the
chain's stop is **demonstrated** rather than described.

---

## The chain, rung by rung, with real exit codes

`setfit_cli_lifecycle_the_binary_walks_ops_02_to_the_rung_f_10_removes`

| rung | invocation | exit | what it proves |
|---|---|---|---|
| 0 | `--version` | **0** | the pinned child executes and produces output (rule 2 — the mechanism, before any conclusion is drawn from a later code) |
| 1 | `data tweet-eval-stance --output benchmark --source srctree` | **0** | writes `benchmark-manifest.json` |
| 2 | `data select --data benchmark --shots 8 --seed 13` | **0** | consumes rung 1's DIRECTORY, writes `selection-manifest.json` |
| 3 | `--json setfit train … --dry-run` | **0** | consumes rungs 1+2's FILES; reports the merged config and both provenance fingerprints |
| 4 | `--json setfit train …` | **6** | F-10's rung. Typed `ModelLoadFailed`, names `--model-dir`, `config.json` and `NEVER downloads`. Nothing written |
| 5 | `inspect model.apr --json` | nonzero | the artifact rung 4 did not write is not there |

**Rungs 1-3 are a genuine end-to-end CLI chain across three processes, each consuming the
previous process's output files.** That is the part of OPS-02 that is real today, and it had no
spawned evidence before this plan.

What rung 3's report is asserted to carry, individually and by name:

```
dry_run = true
provenance.dataset_fingerprint            f170e2ce04e06d3144c19eee06ad346500dac7b34065f321c36d28f09df8c018
provenance.validation_split_fingerprint   f16a46b91baf9e2aa616626c359a6be24a4bee30ebd8fb559abb0c727a6be53e
provenance.selection_root_seed            13     <- rung 2's seed, arriving via a FILE
resolved.resolved_device                  cpu
checks_skipped                            names --model-dir
```

The two fingerprints are asserted to be **different values** and 64 hex chars, so a renderer that
read one path twice cannot pass a presence check — the trap 04-07 closed in `apr inspect`, closed
again here at the spawned tier.

---

## The SPAWNED TRN-07 citation — what it covers, and what it does not

`setfit_cli_lifecycle_trn_07_the_test_split_gate_holds_across_processes`

04-07 shipped `apr_evaluate_the_lock_travels_between_two_invocations_as_a_file` and labelled it
**IN-PROCESS, file-mediated**, explicitly assigning the spawned citation here. This is it, and it
is scoped honestly in its own header rather than only in this summary.

**The POSITIVE half is unreachable and is not faked.** A validation process writing `lock.json`
and a separate test process being admitted by that FILE requires `create_selection_lock`, which
requires a credential, which requires `load_setfit_apr`, which requires bytes no shipped command
can produce (F-10). Rung 4 above is where that stops.

**The NEGATIVE half is fully reachable, and it is the half that actually guards the canonical
test split.** `commands::eval::setfit::run` checks the split flag set at step (1) — before the
Phase 2 ingest at step (2) and before the artifact reload at step (3). So a fresh process can be
observed refusing test access with nothing on disk but a tagged file.

| # | invocation (all against the tagged container) | exit | verdict |
|---|---|---|---|
| L1 | `--split test`, no `--selection-lock` | **5** | names `--selection-lock`, `--split validation` and `--lock-out`; and is **SILENT about the corpus path** |
| L2 | L1 **+ `--selection-lock`** | **5** | gets PAST the gate and **names the corpus** — the only delta is one flag |
| L3 | `--split test --lock-out L` | **5** | a test run may not write its own lock; **`L` does not exist afterwards** |
| L4 | `--split validation --selection-lock L` | **5** | the mirror: a validation run COMMITS, it does not consume |
| L5 | `--split test --candidate <file>` | **5** | the candidate set is the DECISION, fixed when the lock was written |
| L6 | `--split validation --lock-out L`, absent corpus | **5** | dies at the ingest; **`L` does not exist afterwards** |
| L7 | `--split train` | **5** | no third split; the refusal says *memorisation*, not "not in a list" |

**L1 vs L2 is the mechanism proof and it is the point of the pair.** On its own, L1's silence
about the corpus is indistinguishable from "this invocation is broken in a way that never reaches
anything". Two runs differing in exactly one flag, producing two *different* messages, is what
makes "the gate fired, at step (1), before the corpus" a measurement (CLAUDE.md rule 2). The test
also asserts the two messages are not equal.

**L3 and L6 are the durable-lock half that IS about a file:** a refused run must leave no lock on
disk, because a lock is the durable record of a decision and one written by a run that never took
a decision would later admit a test run on the strength of nothing.

---

## The tagged container: what it is, and the assertion that it is not an artifact

Both of the last two tests need a file carrying the SetFit tag, because `apr eval` and
`apr predict` route on `model_type` (D-04) and nothing else. The file is written by **core's
production `AprV2Writer`** and carries the three schema-owned entries.

**It is not a manufactured `setfit-apr-v1` artifact, and that is asserted by the binary rather
than promised in a comment:**

```
$ apr predict <tagged container> --text hola
rc=6   (ModelLoadFailed)
```

Exit **6** and not 4 is the load-bearing detail: 4 would mean the file was never recognised as a
classifier, which would make every gate assertion above vacuous. 6 means **the tag routed and the
loader then refused** — the container reaches `load_setfit_apr`'s full ladder and is thrown out by
it. Nothing in this file goes green because of that container; every test that touches it asserts
a refusal or a read-only rendering.

This is the distinction the phase's hard constraint is about. The forbidden shortcut is building
a *valid* artifact from a synthetic APR-capable encoder so a lifecycle goes green — a green light
reading "OPS-02 holds" about a model the train rung never produced. 04-12 identified and refused
it; this plan refuses it again and says so in the module header.

---

## Verification — status captured DIRECTLY, never through a pipe

Every command ran as `cmd > log 2>&1; echo "rc=$?"`. This mattered more than usual here: a
spawned-process test is the easiest place in this repository to read the wrong status, and the
test itself carries a source guard (below) that forbids the in-Rust version of the same defect.

```
$ cargo test -p apr-cli --features setfit --test setfit_cli_lifecycle -- --ignored lifecycle
rc=0     2 passed, 2 filtered out          (plan filter: `lifecycle`)

$ cargo test -p apr-cli --features setfit --test setfit_cli_lifecycle -- --ignored tooling
rc=0     1 passed, 3 filtered out          (plan filter: `tooling`)

$ cargo test -p apr-cli --features setfit --test setfit_cli_lifecycle
rc=0     1 passed, 3 ignored               (the source guard runs in the DEFAULT invocation)

$ <test binary> --ignored --test-threads 1          (raw libtest, unfiltered by the rtk hook)
rc=0     3 passed; 0 failed; 0 ignored; 1 filtered out; finished in 0.95s

$ cargo fmt -p apr-cli -- --check                                          rc=0
$ cargo check -p apr-cli --all-targets            (feature OFF)            rc=0
$ cargo clippy -p apr-cli --features setfit --test setfit_cli_lifecycle --no-deps
rc=0     0 diagnostics naming setfit_cli_lifecycle.rs
```

Counts are stated non-zero numbers against a stated filter (F-04): a filter matching zero tests
exits 0, so "rc=0" alone would prove nothing. `--no-deps` on the clippy leg is F-03's requirement.

**The clippy run's mechanism was PROVEN engaged, not assumed.** "Zero diagnostics naming my file"
is indistinguishable from "my file was never linted". Injecting
`let _clippy_probe = format!("{}", "x");` produced `useless use of format!` naming
`crates/apr-cli/tests/setfit_cli_lifecycle.rs:1200:25`; reverted.

**The feature-OFF leg is what proves the gating.** The file carries a module-level
`#![cfg(feature = "setfit")]`, so without the feature it compiles to an empty test binary — which
is why no `[[test]]` entry and no `Cargo.toml` change was needed, and why 04-09's ownership of
that manifest was never contended.

### Guards shown able to FAIL

Three behavioural mutations, each observed red **by name**, each reverted from a pristine copy
with the green baseline re-measured afterwards.

| # | Mutation | Result | Killed by |
| - | -------- | ------ | --------- |
| — | baseline (`--ignored lifecycle`) | 2 passed, rc=0 | — |
| M1 | rung 4 expects exit **5** instead of 6 | **1 FAILED, rc=101** | the ladder test — and its transcript printed the real `ExitStatus(unix_wait_status(1536)) (code Some(6))`, which is what proves a genuine child was reaped |
| M2 | `check_split_flags` no longer requires `--selection-lock` on `--split test` (**subject** mutation, in `eval/setfit.rs`) | **1 FAILED, rc=101** | `..._trn_07_...` — and the failure transcript shows the run falling through into the ingest and naming the corpus, i.e. exactly the ordering L1 asserts |
| M3 | the U8 `tokenizer.blob` entry is not written into the container | **1 FAILED, rc=101** | `..._tooling_...` — the printed table shows `2 tensors` and no blob row |
| — | all reverted | 2 / 1 passed, rc=0 | — |

M2 is the one that earned its keep: it is a mutation of the **subject**, not of the test, and it
falsifies the TRN-07 claim rather than a restatement of it. `git status --short` is clean after
the revert and `git diff --name-only HEAD~2 HEAD` lists only this plan's file, so nothing of that
mutation survived.

### apr-cli regression witness — DIFFED against 04-07's post-plan measurement, not eyeballed

```
$ cargo test -p apr-cli --features setfit --lib
rc=0     6751 passed, 15 ignored, 0 FAILED
```

**Identical to 04-07's recorded post-plan numbers** (`6751 passed, 15 ignored, 0 FAILED`). There is
no known-red set for this crate to diff — apr-cli has been fully green throughout the phase — so
the witness is the exact equality of the pass and ignore counts. Zero regressions, and the +0
delta is expected: this plan adds an integration target, and `--lib` does not run it.

No `src/` file was modified by this plan. The one temporary subject mutation (M2, in
`eval/setfit.rs`) was reverted and is provably absent: `git diff --name-only HEAD~2 HEAD` lists
only this plan's test file.

---

## Deviations from Plan

### 1. [Rule 3 — blocking] The five-rung chain cannot close; the file asserts the boundary instead

The plan's Task 1 prescribes five green invocations ending in a parsed `ClassifyResponse`. Rungs
4-8 have no reachable input (F-10). Rather than stop with nothing, the file proves rungs 0-3,
asserts rung 4's typed refusal with its exit code, and runs rung 5 anyway so the stop is
observable. The `F10_CLOSED` panic message spells out all five rungs to restore, in order, with
the flags. **The plan's HARD CONSTRAINT was honoured exactly**: one file created, no
`Cargo.toml` touched, no `src/` file in the diff.

### 2. [Deviation, argued] `CARGO_BIN_EXE_apr` appears ONCE, not the five times the criterion asks for

The criterion is `grep -c "CARGO_BIN_EXE_apr" >= 5`. Shipped: one `const APR_BIN`, used
everywhere, plus a source guard asserting **exactly one `Command::new(` site in the whole file and
that it takes that constant**, exactly one `env!` of the cargo variable, zero bare command-name
string literals, and zero `Stdio::from(` (the in-Rust pipeline that would make every status read
the wrong process's). Five copies of an environment lookup is five places a `PATH` lookup can
later be introduced without anyone noticing; one constant plus an enforced spawn-site count is the
stronger claim. The intent of the criterion — *the installed binary is the subject, never a PATH
lookup* — is met and enforced rather than counted.

### 3. [Rule 3 — blocking] The Phase 2 fixture recipe is replicated, not reused

The plan says to reuse 04-06 Task 3's recipe. That recipe is
`data_tweeteval::fixtures::write_canonical_fixture_tagged`, which is `#[cfg(test)] pub(crate)`;
an integration test is out-of-crate and cannot reach it. The three contracted count arrays and the
row-text tag are therefore replicated at the top of the file, **with the reason and the drift
behaviour documented on them**: the counts are not asserted here, they are handed to the shipped
`data tweet-eval-stance`, which validates them against its own constants. If they ever change,
rung 1 fails with the binary's own message naming the expected counts — a self-diagnosing failure
rather than a silent divergence.

### 4. [Rule 2 — missing critical] Every child is bounded, which `Command::output()` is not

The plan's acceptance criterion requires no unbounded wait. `Command::output()` waits forever, so
a hung rung would burn the whole CI job and report nothing about which one hung. The harness
spawns with piped stdio, drains both pipes on their own threads (an undrained pipe deadlocks
against exactly the verbose runs whose output matters most), polls `try_wait`, and on deadline
**kills the child first and then panics** — panicking with the child still running would orphan it
holding the pipes the readers are blocked on.

### 5. [Rule 2 — missing critical] The `apr qa` leg gained a CONTROL the plan did not specify

The plan asks to record qa's verdict and surface a nonzero exit as a finding. As specified, the
finding would have been recorded as *"qa refuses a SetFit artifact"*, whose natural reading is
that the SetFit entries are why — and that is false. One extra input (the committed plain-Bert
APR) inverted the diagnosis. Both verdicts are now asserted, so the finding cannot drift.

### 6. [Rule 2 — missing critical] The container is asserted NOT to be a model, by the binary

The plan did not ask for this. Without it, a reader has only this summary's word that the tagged
file is not a manufactured artifact. `apr predict` against it exits **6**, and the exit code is
asserted specifically (6, not merely nonzero) because 4 would mean the tag never routed and every
gate assertion in that test would be vacuous.

### 7. [Deviation, additive] A fourth test that is NOT `#[ignore]`d

`setfit_cli_the_binary_under_test_is_pinned_and_spawned_from_exactly_one_site` runs in the default
invocation. It costs one file read. Without it, `cargo test --test setfit_cli_lifecycle` with no
`--ignored` would report a vacuous green over a target containing nothing but ignored tests.

---

**Total deviations:** 2 blocking, 3 missing-critical, 2 argued/additive. No new package: no
`Cargo.toml` and no `Cargo.lock` change in the diff.

---

## Plan acceptance criteria — what was and was not met

| Criterion | Status |
|---|---|
| Task 1 passes under `-- --ignored lifecycle` | **MET** — 2 passed, rc=0 |
| five invocations all succeeding | **NOT MET** — rungs 0-3 succeed; rung 4 is F-10's and is asserted as a typed refusal |
| the missing-lock negative fails nonzero | **MET** — exit 5, message asserted, and silent about the corpus |
| `#![cfg(feature = "setfit")]` at the top, no Cargo.toml change | **MET** — both, and the feature-off `cargo check --all-targets` is rc=0 |
| `grep -c "CARGO_BIN_EXE_apr"` >= 5 | **NOT MET, deliberately** — see deviation 2; a stronger enforced property replaces it |
| no bare command-name string | **MET** — asserted in-test, with a needle split mid-token so the guard cannot match itself |
| no status read through a pipe | **MET** — asserted in-test (`Stdio::from(` count is 0) and by construction |
| the lock assertion checks the FILE between two processes | **MET for the negative half** — L3 and L6 assert the file's ABSENCE after a refused run; the positive half is F-10-blocked and is not faked |
| Task 2 passes under `-- --ignored tooling` | **MET** — 1 passed, rc=0 |
| `apr tensors` names all three schema-owned entries | **MET** — rows transcribed above |
| `apr qa`'s exit status and verdict recorded, nonzero surfaced as a finding | **MET** — with a control that changed the diagnosis |
| every invocation bounded | **MET** — deadline + kill, on all 18 (6 + 8 + 4) |
| `git diff --name-only` lists exactly one file | **MET** |

`requirements-completed: []`. **OPS-02 must NOT be checked off.** Its train leg cannot complete
on this host and its predict/eval legs have no artifact to run against. OPS-03's structural claim
is untouched by this plan.

---

## Known Stubs

**None.** No stub, no placeholder, no `#[ignore]`d test that would fail if run. The unreachable
rungs are absent rather than faked, and their absence is asserted by tests that turn red the day
it stops being justified.

## Threat Flags

None. No new network endpoint, no auth path, no schema at a trust boundary, no new package. The
file spawns the repository's own binary with absolute paths inside a `TempDir`, writes six text
fixtures and one container into that directory, and reads two committed fixtures.

Threat register, as covered:

| Threat ID | Covered as shipped |
| --- | --- |
| T-04-54 (passes unit-by-unit, fails end to end) | six spawned invocations consuming each other's FILES; rungs 1-3 green, rung 4 typed |
| T-04-55 (untested compatibility claim, A3) | executed against a real container; the U8 entry survives; the one nonzero verdict is recorded WITH a control |
| T-04-29 (stale/shadowed binary) | `env!("CARGO_BIN_EXE_apr")` only, asserted exactly once, with zero bare command-name literals |
| T-04-21 (canonical test leakage) | seven spawned refusals; the gate is proven to fire before the corpus is opened |
| T-04-SC (package installs) | zero new packages; no manifest or lock change |

---

## Notes for later plans and for the orchestrator

- **04-10 (gates).** This target has THREE legs, all requiring `--features setfit`:
  `--test setfit_cli_lifecycle` (default: **1** passed, 3 ignored — the source guard),
  `-- --ignored lifecycle` (**2**), `-- --ignored tooling` (**1**). libtest takes one positional
  filter, so the two ignored legs cannot be combined. The whole target runs in **under 1 second**
  after build. Your ran-something guard matters here: all three legs would exit 0 on a filter
  that matched nothing.
- **04-11 (requirements audit).** **Do not mark OPS-02 complete.** This plan is the closest
  evidence that exists and it stops at rung 4. The spawned TRN-07 citation is
  `setfit_cli_lifecycle_trn_07_the_test_split_gate_holds_across_processes` — cite it as the
  **negative half at the binary tier**, paired with 04-07's
  `apr_evaluate_the_lock_travels_between_two_invocations_as_a_file` (**in-process**) for the
  positive half. Neither is the full cross-process happy path; that needs F-10 closed.
- **Phase 5, two items.** (a) F-10 remains the single blocker for OPS-02's train leg; when it
  closes, this file's `F10_CLOSED` panic fires and hands its author the eight rungs to restore.
  (b) **`apr qa` needs a decision**: it is a generative-model gate and refuses every encoder-only
  APR. Either a classifier gate set, or an explicit "not a generative artifact" verdict — right
  now an operator following CLAUDE.md's "always start with `apr qa`" gets a message about a
  missing BPE tokenizer, which is true and unhelpful.
- **Anyone editing `commands/eval/setfit.rs`.** `check_split_flags`'s position — step (1), before
  the ingest — is now asserted from OUTSIDE the process, by the difference between two
  invocations. Moving it later will turn L1 red rather than silently weakening the gate.

## Self-Check: PASSED

File claimed, checked on disk:

```
FOUND: crates/apr-cli/tests/setfit_cli_lifecycle.rs   (1295 lines, 60.9 KB)
```

Commits claimed, checked in the log: `5883e2bc4`, `b8755fdc3`.

| Assertion | Criterion | Observed |
| --- | --- | --- |
| `git diff --name-only HEAD~2 HEAD` | exactly the plan's one file | **1** — `crates/apr-cli/tests/setfit_cli_lifecycle.rs` |
| `git diff --diff-filter=D --name-only HEAD~2 HEAD` | empty | **empty** — no deletions |
| `src/` files touched | 0 | **0** |
| `Cargo.toml` / `Cargo.lock` touched (04-09's file) | no | **not touched** |
| 04-09's `tests/setfit_parity.rs` / fixtures | not touched | **not touched** — they do not exist on this branch |
| `STATE.md` / `ROADMAP.md` / `REQUIREMENTS.md` | not touched | **not touched** — the orchestrator owns them |
| `git status --short` after each commit | empty | **empty** |
| scoped filters | green, counted | **2 / 1 / 1**, rc=0 each |
| guards shown able to fail | >= 1 behavioural | **3**, one of them a SUBJECT mutation, all reverted |
| clippy actually linted the file | mechanism proven | **injected lint fired at line 1200**, then reverted |
| `cargo test -p apr-cli --features setfit --lib` | equal to 04-07's post-plan counts | **6751 passed, 15 ignored, 0 FAILED** — identical |
| a fabricated artifact used to manufacture green | none | **none** — and `apr predict` exits 6 on the container, asserted |

---
*Phase: 04-apr-artifact-and-production-parity*
*Plan: 15 — PARTIAL. The spawned OPS-02 evidence exists and stops where F-10 stops it; A3 is executed and survives; `apr qa`'s inapplicability to encoder-only APRs is a new finding with a control.*
*Completed: 2026-08-15*
