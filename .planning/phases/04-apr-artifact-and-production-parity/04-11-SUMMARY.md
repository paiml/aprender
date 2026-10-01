---
phase: 04-apr-artifact-and-production-parity
plan: 11
status: PARTIAL
subsystem: ci-gates-and-audit
tags: [safe-02, trn-07, ops-01, ops-02, ci, mutation, requirements-audit, m12, m13, w-5, f-10, cr-01]

# Dependency graph
requires:
  - phase: 04-10
    provides: "the 23 named, floored Make targets whose commands the CI patch mirrors verbatim, and setfit-feature-matrix grown to four crates"
  - phase: 04-12
    provides: "F-10 measured on three independent routes — the OPS-01 blocker the audit refuses to route around"
  - phase: 04-15
    provides: "F-10 corroborated through the BINARY, and the SPAWNED TRN-07 citation"
  - phase: 04-07
    provides: "the IN-PROCESS TRN-07 citation, labelled by its author as two invocations, not two processes"
provides:
  - ".planning/phases/04-apr-artifact-and-production-parity/04-11-ci-setfit.patch — the proposed ci.yml extension as TEXT, git apply --check rc=0, ci.yml untouched"
  - "the target-coverage accounting: all 24 targets by inclusion, existing-leg coverage, or written exclusion"
  - "the closing requirements audit: no Phase 4 box checked, every partial named by half"
  - "a MEASURED correction to the plan's own mutation recipe (its -- --features setfit never reaches cargo)"
affects: [phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A proposed workflow change is a PATCH FILE under .planning/, never a working-tree edit — preparing the edit is already acting"
    - "Generate unified diffs with difflib against the real file; hand-authored hunk headers were wrong twice and git apply --check caught both"
    - "A forbidden-command scan must run over COMMAND lines only — a scan over all added lines turns red on its own documentation (the F-05 class)"
    - "cargo-mutants: features go through --features / --cargo-arg, NOT through trailing `-- args`, which reach libtest instead"

key-files:
  created:
    - .planning/phases/04-apr-artifact-and-production-parity/04-11-ci-setfit.patch
  modified:
    - .planning/REQUIREMENTS.md

key-decisions:
  - "ci.yml was NOT modified, staged or applied. `git status --porcelain -- .github/workflows/ci.yml` measured 0 bytes at every step, including after `git apply --check`."
  - "THREE exclusions, not the two the plan expected. The third (setfit-api-boundary) is surfaced loudly rather than absorbed, with the reason it is excluded AND the counter-argument for including it written into the patch itself."
  - "The `-- --ignored` setfit_train leg IS wired: that test calls commands::setfit_train::run in-process and spawns nothing, so the spawned-binary flake rationale does not cover it — the same W-5 reasoning that put setfit_apr_lifecycle back."
  - "NO Phase 4 requirement box was checked. Every one whose 'a user can' tier routes through a produced artifact stops at F-10, and SAFE-02 — the one F-10 does not block — is still not met because its CI half is an unapplied patch."
  - "Two frontmatter `requirements-completed` claims (04-13, 04-14) were refused rather than transcribed."

requirements-completed: []

# Metrics
duration: ~2h
completed: 2026-08-15
---

# Phase 4 Plan 11: The CI Patch, the Mutation Gate, and the Closing Audit — PARTIAL

**The CI change exists as inspectable text and the workflow file was never touched. The
requirements audit checked nothing.** That second sentence is the deliverable, not an apology:
every Phase 4 requirement whose "a user can" tier routes through a produced artifact stops at the
same wall, and the one requirement F-10 does not block is still not met because its CI half is a
patch awaiting a human.

## Task Commits

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | `04-11-ci-setfit.patch` — the proposed ci.yml extension, `git apply --check` rc=0, ci.yml clean | `eb0f8e1c5` |
| 3 | the closing requirements audit in `REQUIREMENTS.md` | `608aa954e` |
| — | this SUMMARY, committed before the mutation run returned so a stall could not lose it | `5037e0d90` |
| 2 | mutation gate — PARTIAL; findings as `D-04-11-A` / `D-04-11-B` in `deferred-items.md` | (final commit) |

Task 2 in the plan is the blocking human checkpoint on the patch. **It was not executed by me and
must not be**: applying the patch is the orchestrator's step after the human approves. Nothing in
this branch modifies `.github/workflows/`.

---

## Task 1 — the CI patch, and the ci.yml assertion at every step

`.github/workflows/ci.yml` is **provably untouched**. Measured, not asserted, after writing the
patch and again after the dry-run apply:

```
$ rtk proxy git status --porcelain -- .github/workflows/ci.yml
rc=0   bytes=0
$ rtk proxy git apply --check .planning/.../04-11-ci-setfit.patch
APPLY_CHECK_RC=0
```

`--check` is a dry run; it reports applicability and writes nothing. It was never run without it.

### The patch was GENERATED, not hand-authored — and that is a finding

Two hand-written attempts shipped **wrong hunk line counts** (`+286,93` and `+286,99` against an
actual 116 and 99). `git apply --check` rejected both with "patch does not apply", pointing at
line 286 — which read like a context mismatch and is not. A byte-exact comparison of the patch's
24 context lines against the file found **zero differences**, which is what proved the defect was
in the header arithmetic rather than the text.

The patch is now produced by `difflib.unified_diff` against the real `ci.yml`, with the three
anchors (step name, comment tail, `bash -c` block) **asserted** before substitution, so a moved
anchor fails loudly instead of patching the wrong place. The generator writes only to
`.planning/`; it reads `ci.yml` and never opens it for writing.

### Target-coverage accounting — all 24, none silent

04-10 created 23 targets and grew `setfit-feature-matrix` from two crates to four. Every one is
accounted for. (The four Phase-3 targets — `setfit-tests`, `setfit-repro-inproc`,
`setfit-repro-crossproc`, `setfit-repro-replay` — are not 04-10's and are out of this accounting.)

| Target | Disposition |
| ------ | ----------- |
| `setfit-apr-tests` | already covered — existing leg `core --lib setfit::` is its prefix superset |
| `setfit-classify-tests` | already covered — same leg |
| `setfit-bundle-tests` | already covered — existing leg `train --lib setfit::` |
| `setfit-config-tests` | already covered — same leg |
| `setfit-evaluate-tests` | already covered — same leg |
| `setfit-codec-tests` | already covered — same leg |
| `setfit-reload-tests` | already covered — same leg |
| `setfit-lock-tests` | already covered — same leg |
| `setfit-verify-tests` | already covered — same leg |
| `setfit-ui-tests` | already covered — existing leg `train --test ui` |
| `setfit-lifecycle-tests` | **ADDED** — `--test setfit_apr_lifecycle` (this is W-5; see below) |
| `setfit-cli-train-tests` | **ADDED** — both legs, default and `-- --ignored` |
| `setfit-cli-predict-tests` | **ADDED** |
| `setfit-cli-inspect-tests` | **ADDED** |
| `setfit-cli-eval-tests` | **ADDED** |
| `setfit-cli-io-tests` | **ADDED** |
| `setfit-cli-serve-tests` | **ADDED** — *omitted from the plan's own additions list; see deviation 1* |
| `setfit-serve-tests` | **ADDED** — scoped `--lib setfit`, never the whole crate (D-04-08-A) |
| `setfit-parity` | **ADDED** — `--test setfit_parity` |
| `setfit-feature-matrix` | **ADDED** — the six apr-cli / aprender-serve check cells, i.e. the two crates 04-10 grew it by |
| `setfit-all-tests` | aggregate with no command of its own; covered by its 18 members, all above |
| `setfit-serve-smoke` | **EXCLUDED 1** — spawns a real server on a loopback port; a CI loopback-spawn is a new flake surface |
| `setfit-cli-lifecycle` | **EXCLUDED 2** — spawns the real `apr` binary five times; same rationale |
| `setfit-api-boundary` | **EXCLUDED 3 — a DEVIATION, surfaced not absorbed; see deviation 2** |

16 new cargo legs are added. A seventeenth `+` line is the existing `--test ui` leg, whose only
change is its line terminator (`'` → `; \`) so the block can continue.

### W-5, closed and asserted

The plan's own previous draft lost `--test setfit_apr_lifecycle`. It is in, and the reason it was
easy to lose is now written into the patch: **the three existing CI legs are `--lib` or `--test
ui`, and `--lib` never reaches an integration target.** Before this patch,
`crates/aprender-train/tests/setfit_apr_lifecycle.rs` — OPS-01's only proof — and
`crates/apr-cli/tests/setfit_parity.rs` ran in **no CI job at all**, which is the CR-01 shape one
phase later. Source-asserted: both `--test setfit_apr_lifecycle` and `--test setfit_parity` appear
in the patch.

### Source assertions, all executed

| Assertion | Result |
| --- | --- |
| `git apply --check` on the patch | **rc=0** |
| `git status --porcelain -- .github/workflows/ci.yml` | **0 bytes**, before and after |
| every added cargo command appears VERBATIM in the Makefile | **17/17 found** |
| `\|` pipe characters in any added line | **0** |
| added commands sit inside the existing `set -e` block | yes |
| forbidden legs present as COMMANDS | **0** |
| exclusion rationale present for each of the three | 3/3 |

**The forbidden-leg scan ships a case table and it was EXECUTED** (CLAUDE.md rule 7), because a
scan that matches nothing is indistinguishable from a dead pattern:

| Case | Input | Expect | Got |
| ---- | ----- | ------ | --- |
| MUST-MATCH | `cargo test -p aprender-serve --lib` | flagged | flagged |
| MUST-MATCH | `cargo check -p apr-cli --no-default-features` | flagged | flagged |
| must-not | `cargo test -p aprender-serve --features setfit --lib setfit` | clean | clean |
| must-not | `cargo check -p apr-cli --all-targets` | clean | clean |
| MUST-MATCH | `cargo check -p apr-cli --no-default-features --features setfit` | flagged | flagged |

5/5. The last case matters: D-04-09-A is red for **both** apr-cli minimal cells, because `setfit`
does not imply `inference`, so the scan must flag the `--features setfit` form too.

**The scan is restricted to COMMAND lines, and that restriction is load-bearing.** The patch's
comment block deliberately contains both forbidden strings written exactly as they would be
invoked, so a future `grep` for either finds the warning. A scan over *all* added lines would turn
red on its own documentation — the F-05 defect class, one file over. Both facts are asserted:
the strings are absent from added commands and present in added comments.

---

## Task 2 — the mutation gate

Run **per crate** (M13). The plan's recipe needed two measured corrections before it could run at
all; both are recorded under Deviations.

### Scope and denominators — enumerated, not estimated

`cargo mutants --list` over the Phase 4 files, per crate:

| Crate | Files | Mutants |
| ----- | ----- | ------- |
| aprender-core | `setfit/artifact.rs` (311), `setfit/classify.rs` (69) | **380** |
| aprender-train | `bundle.rs` (188), `lock.rs` (81), `config.rs` (63), `apr_reload.rs` (14), `apr_codec.rs` (12) | **358** |
| aprender-serve | `api/setfit_handlers.rs` | **101** |
| apr-cli | `commands/setfit_train.rs` (25), `commands/predict.rs` (20), `setfit_io.rs` (6) | **51** |
| | | **890 total** |

A near-zero count is itself a red flag (T-04-35); these are the real numbers and the globs
resolved (paths relative to the workspace root).

### Result: PARTIAL, and stopped on a MEASURED projection rather than a guess

**There is no aggregate adjusted score, and I am not going to compute one from a partial run.**
Here is exactly what was measured.

| Crate | Mutants | Baseline | Run | Outcome |
| ----- | ------- | -------- | --- | ------- |
| aprender-serve | 101 | **ok** | `--shard 1/25` (4 mutants) | **4 / 4 caught**, 453 s |
| aprender-serve | 101 | **ok** | full | **INTERRUPTED at 4,094 s (68 min)**; 9 survivors reported, completed count unknown |
| aprender-core | 380 | not reached | — | not run |
| aprender-train | 358 | not reached | — | not run |
| apr-cli | 51 | not reached | — | not run |

cargo-mutants prints a line only for non-caught outcomes and a summary at the end. The full run was
killed before that summary, so **the number of mutants completed is unknown** and no ratio can be
stated. What IS known is the nine survivors it named, and they are worth more than a score.

### Survivor triage — 7 equivalent by construction, 2 real

**Seven of nine mutate the file's own `#[cfg(test)]` module** — five `tests::<fn> -> ()` and two
inside `fixture::Filler::next`. A test suite cannot detect the deletion of one of its own tests, so
these are equivalent by construction, not coverage gaps. This is the **F-05 self-scan class at the
mutation tier**: a `-f <file>` glob mutates the file's tests along with its production code. Any
future run must exclude them or ~7 of every 9 survivors will be noise.

**Two are real production survivors:**

| Mutation | Source | Assessment |
| -------- | ------ | ---------- |
| `replace AppState::has_setfit_model -> bool with false` | `:70-72`, body `self.setfit_model.is_some()` | **Real gap.** Under the `setfit` filter — the same one `make setfit-serve-tests` uses — nothing pins this returning `true`. A readiness path that always reported "no classifier resident" would pass. |
| `replace > with == in setfit_classify_handler` | `:158`, `if request.texts.len() > MAX_BATCH_TEXTS` | **Real, and it should not have survived.** A co-located test is *named* `setfit_classify_refuses_a_batch_one_over_the_contract_bound` — `len == MAX+1` is the exact input distinguishing `>` from `==`. Either it does not exercise the branch its name claims, or it is not reached under this filter. |

The second is the finding worth having: a test whose NAME asserts boundary coverage, beside a
boundary mutant that lives. That is CLAUDE.md rule 2 — labelling by intent — one level down, at the
test name rather than at the run. **I did not diagnose it**: one failing input is an anecdote (rule
6), and diagnosing it means re-running that mutant alone.

Neither was fixed. `crates/aprender-serve/src/api/setfit_handlers.rs` is 04-08's file and is not in
this plan's `files_modified`; fixing it here would be scope creep into a file another plan owns.
Both are logged as **D-04-11-A** in `deferred-items.md` with the reproduction command.

### The projection, measured — why the run stopped

From the full run: 4,094 s elapsed without finishing 101 mutants gives an average **> 40.5 s per
mutant** including baseline — a LOWER bound, since it had not completed. `aprender-core` and
`aprender-train` carry far heavier builds and test suites than `aprender-serve`, so their real cost
is strictly worse.

```
890 mutants x >= 40.5 s  =  >= 36,000 s  =  >= 10.0 hours, plus four baselines
789 not yet attempted    =  >=  8.9 hours
```

The plan's own instruction covers this exactly: *"If wall-clock projects beyond a reasonable budget,
report the measured projection and stop for the human (the Phase 3 compute-budget precedent) rather
than shrinking scope silently."* That is what happened. Recorded as **D-04-11-B** with the four
corrections a future run needs.

**Disk was watched throughout** (the ENOSPC precedent): 91 GiB free at start, 88 GiB at the
deepest point of the tree-copy build, 91 GiB after cleanup. It never approached the 15 GiB floor,
so disk was not the constraint — wall clock was. `CARGO_INCREMENTAL=0` on every invocation.

---

## Task 3 — the closing requirements audit

**Nothing was checked.** `- [x]` count in `REQUIREMENTS.md` is **13**, unchanged from before this
plan: the Phase 2 DATA rows, the Phase 3 TRN rows and SAFE-03. Every Phase 4 requirement stays
`- [ ]` and now carries a written statement of which half landed and which did not.

### Why nothing was checked — one cause, everywhere

`CALIBRATED_REGIMES` holds exactly **one** entry, and its architecture component is compared for
**exact equality**. So the only trainable encoder is the phase-3 MiniLM slice, whose 97-row
vocabulary closure cannot compute `probe_unicode` (canonical id 5915). Measured by 04-12 on three
independent routes, corroborated through the binary by 04-15 (rung 4 exits 6, `model.apr` asserted
absent afterwards), and respected by 04-09. Consequence: no user-reachable path produces a
`setfit-apr-v1` artifact, and every requirement whose "a user can" tier needs one stops there.
Closing it is a **Phase 5** item — a calibration run plus a deliberate edit to
`contracts/setfit-train-lifecycle-v1.yaml` (D-10(c)).

Three separate executors (04-12, 04-15, 04-09) each declined to route around F-10 by synthesising
an APR-capable encoder. That shortcut compiles and would have satisfied their acceptance criteria
to the letter while testing a model training never produced. **This audit does not undo that.**

### Per-requirement verdicts

| Req | Verdict | What shipped | What is missing |
| --- | ------- | ------------ | --------------- |
| APR-01 | Partial | writer + storage map for every item; bundle field 20 `ProvenanceRecord` (6 fields, no `Option`, no float) | `into_artifact_bytes` yields **`setfit-serde-json-v1`**, not `setfit-apr-v1` — no producer (F-10) |
| APR-02 | Partial | 8-rung ladder matching the contract, pinned by a test that parses the contract's own `rungs:` via `include_str!` and was shown RED on a rename | accept path exercised only over an in-crate `#[cfg(test)]` fixture; F-14(5) pre-reservation still open |
| APR-03 | Partial | 04-05 `06b48cf32`, 7 tests through the real `verify_artifact` and the whole trusted policy | encoder+head **substituted**; never run over a model the train path produced |
| APR-04 | Partial | no-bypass proven OUT-OF-CRATE at COMPILE time (trybuild `d0b51815c`); sealed credential trait, 2 implementors counted | the positive clause across all five surfaces is fixture-only |
| APR-05 | Partial | full APR-05 render, offline, one renderer pinned to `SETFIT_ARTIFACT_DOC_FIELDS`; `apr tensors` names all 3 schema entries incl. the U8 blob | no user-produced artifact to inspect |
| OPS-01 | **Pending — blocked** | graph boundary closed by `setfit-api-boundary` with an executed 5-leg table incl. a MUST-MATCH control; lifecycle test 5/5 | load/embed/classify/inspect all unreachable (F-10) |
| OPS-02 | **Pending — blocked** | rungs 0-3 a genuine three-process CLI chain consuming each other's files | rung 4 exits 6; `model.apr` absent |
| OPS-03 | Partial | routing proven EXECUTED — `apr predict` exits **6, not 4**; non-reconstruction structural (`from_run_parts` is `pub(crate)`) | no prediction ever completes through core |
| OPS-04 | Partial | 3-reader parity gate, 20 tests, proven able to fail; `latency_ms` out of `PartialEq`, asserted separately BY BITS | fixture synthetic; **`backend_identity` binds to a symbol that does not exist** |
| OPS-05 | Partial | `classifier_artifact_sha256` + `classifier_verified`; the two WIRE values proven to name one artifact | "the same APR" presupposes an APR a user produced |
| OPS-06 | Partial | `resolved_device = cpu` asserted at the spawned tier; CPU-only throughout | the "misreporting the backend" clause rides the same unbound `backend_identity` |
| SAFE-01 | Partial | parity detector delivered and falsifiable; 18 floored suites; `contract-audit-phase4` rc=0, zero BIND-001 | one equation `pending` naming a nonexistent symbol; several clauses are Phase 1-3 |
| SAFE-02 | Partial | local 4x3 matrix, build AND run legs, two-sided graph negatives, 4 guards mutated red | **the CI half is this plan's UNAPPLIED patch** — "in CI" is not yet true |
| TRN-07 | Partial | negative half proven CROSS-PROCESS at the binary tier (7 spawned refusals) | positive half still IN-CRATE only; no `apr` surface creates a lock |

### TRN-07 — two citations, each with only the label it earned

- **CROSS-PROCESS:** `setfit_cli_lifecycle_trn_07_the_test_split_gate_holds_across_processes`
  (04-15). Seven spawned `apr` processes. L1 vs L2 differ in exactly one flag and produce two
  *different* refusals — that pair is what makes "the gate fired at step (1), before the corpus"
  a measurement rather than a label.
- **IN-PROCESS:** `apr_evaluate_the_lock_travels_between_two_invocations_as_a_file` (04-07). Two
  invocations inside **one** process, mediated by a lock FILE. Its own action text says "ACROSS
  TWO INVOCATIONS", not two processes.

Asserted mechanically: `CROSS-PROCESS` appears on the 04-15 citation and **not** on the 04-07 one.
Writing "cross-process" beside the 04-07 test would be labelling a run by intent (CLAUDE.md rule
2) in a file that outlives the phase.

**Why TRN-07 is still not checked**, against the plan's instruction to check it: 03-10 left it
unchecked for a specific stated reason — no out-of-crate caller and no `apr` surface reaches
`create_selection_lock -> mint_test_token -> CanonicalTestAccess::grant`. Phase 4 moved both
halves and did not close that. 04-16 landed `reload_verified_run_from_apr`, a genuine public
fresh-process door that *does* reach all three — but the only bytes able to drive it come from an
in-crate `#[cfg(all(test, feature = "setfit"))]` fixture, because F-10 blocks the artifact. So the
positive half is in-crate and the negative half is cross-process, and no user reaches a lock. See
deviation 3.

### Two frontmatter overclaims, refused

`04-13` asserts `requirements-completed: [APR-01, APR-05]`; `04-14` asserts `[OPS-02, TRN-07]`.
Both shipped real work. Neither delivers its requirement at the "a user can" tier, and 04-15 —
later, and the only plan that drives the shipped binary — states plainly that OPS-02 must not be
checked off. **No box was flipped on the strength of a frontmatter field.** This is the T-04-34
defect class, caught at the audit rather than inherited.

### The seven recorded amendments and limitations

All seven are written into `REQUIREMENTS.md` as a table so they outlive the phase:

1. **D-02(a) typed-key amendment** — `model_type` is the only typed key.
2. **D-01 name extension** — 6 of 21 HF templates have no canonical form (22 of 101 encoder
   tensors unnamed on the pinned model); reserved in-contract with non-collision enumerated.
3. **Nullable-path allowlist** — four paths over five walked sub-documents, deliberately; residual
   `evidence.epsilon_used`.
4. **`SetFitBundle` field 20** — six-field `ProvenanceRecord`.
5. **Backend-identity v1** — grammar shipped; the binding row names a nonexistent symbol and stays
   `pending`. **OPEN**, and it blocks a named clause of both OPS-04 and OPS-06.
6. **apr-cli run-leg limitation, in its CORRECTED form** — T-04-61 **does not arise** (04-09
   measured the dev-dep unnecessary and added none). The real measured weakness is that apr-cli's
   `--lib setfit` filter runs **13** tests with the feature OFF, so that leg proves the gated
   surface APPEARS, not that it is ABSENT.
7. **`apr qa` is a generative-model gate** — exit 5 on any encoder-only APR. The CONTROL (a plain
   Bert `slice_model.apr`, no SetFit tag, no U8 blob) produces the *identical* exit and message,
   so the SetFit entries are **not** the cause. One failing input would have produced the wrong
   diagnosis. **OPEN**, Phase 5.

---

## Deviations from Plan

### 1. [Rule 2 — missing critical] The plan's own additions list omitted `setfit-cli-serve-tests`

The plan enumerates the apr-cli lib suites to add as "setfit_train, predict, inspect, eval::setfit,
setfit_io". `setfit-cli-serve-tests` (`--lib serve`, 358 tests, 04-08's cited evidence) is also an
apr-cli lib suite and is not in that list. Leaving it out would have made the coverage obligation
false while the patch looked complete — the same W-5 shape the plan exists to close. **Added.**

### 2. [Deviation, surfaced] THREE exclusions, not two — `setfit-api-boundary`

The plan expects exactly two and says a third means "stop and surface it rather than shipping a
patch whose omissions are invisible". Here it is, surfaced.

`setfit-api-boundary` is not a cargo test target at all: it is a `cargo tree` graph gate written as
a Make `for` loop with `$$`-escaped shell variables and **single-quoted** `grep`/`tr` patterns. The
CI step's shape is `bash -c '...'` — also single-quoted. The loop cannot be pasted in without
rewriting its quoting, and rewriting is transcription, which is exactly the drift the step's own
comment exists to prevent.

**The counter-argument is written into the patch beside the exclusion, because it is real:** unlike
the two spawned targets, this one has genuine CI-only value. A Linux dependency closure can differ
from the macOS dev host through `cfg(target_os)` deps, so CI would exercise a graph `make` on this
box cannot. It is excluded on **quoting** grounds, not on value grounds, and the human should rule
on it at the checkpoint alongside the two spawned legs.

### 3. [Deviation, argued] TRN-07 is left UNCHECKED against the plan's instruction

The plan says to check TRN-07. The audit does not, for the reason given above: the specific gap
03-10 named — a user reaching the lock — is not closed, and the orchestrator's standing
instruction for this plan is that no box may be checked beyond what shipped. Both citations are
recorded with their correct labels and the traceability row reads
`Partial — negative half proven CROSS-PROCESS (04-15); positive half IN-CRATE only (F-10)`, which
is strictly more informative than a tick. If the phase owner disagrees, the change is one
character and every piece of evidence needed to make it is in the row.

### 4. [Rule 3 — blocking] The plan's mutation recipe does not pass features to cargo

The plan prescribes `cargo mutants ... -- --features setfit`. **Measured: it does not work.**
cargo-mutants passes trailing `--` arguments to the *test binary*, not to cargo; the build it
actually ran was

```
cargo test --no-run --verbose --package=aprender-serve@0.63.0
```

with **no `--features setfit`** — i.e. over code that is `#[cfg(feature = "setfit")]`-compiled out.
That is the F-04 vacuity class at the mutation tier. cargo-mutants has dedicated `--features` and
`--cargo-arg` flags; the corrected invocation uses them. Had this gone unnoticed the run would have
produced confident numbers about code that was never compiled.

### 5. [Rule 3 — blocking] cargo-mutants builds ALL test targets, so `aprender-serve` had no baseline

The first probe's baseline **FAILED** (rc=4). Cause, read from the log rather than guessed:

```
error[E0063]: missing field `query_pre_attn_scalar` in initializer of `GGUFConfig`
error: could not compile `aprender-serve` (test "gguf_coverage") due to 14 previous errors
```

That is the pre-existing integration-target drift D-04-08-A already recorded, reached because
`cargo mutants` builds every test target, not just the lib. Fixed with `--cargo-arg=--lib`, which
also keeps the run off the 51-red whole-crate suite. Baseline then reported `ok`. **Neither
forbidden leg was wired to get there.**

---

## Known gaps — stated, not papered over

- **`bashrs` is NOT installed on this host** (`command -v bashrs` → not found; genuinely absent,
  not shadowed). No bashrs check is claimed. Substitutes actually run: `git apply --check`, the
  byte-exact context comparison, the 17-command Makefile verbatim assertion, the pipe scan, and
  the executed 5-case forbidden-leg table.
- **The patch is not applied and must not be by me.** CLAUDE.md reserves `.github/workflows/*.yml`
  for human decision, and review M12 establishes that preparing the working-tree edit is already
  acting. SAFE-02 stays open until a human lands it.
- **The `-- --ignored` setfit_train leg's wall-clock was NOT measured.** It tunes a real encoder
  over the slice fixture. It is wired because it is in-process and the spawn rationale does not
  cover it, but if it proves slow in CI, that leg is the one to drop — never the OPS-01 one. This
  is written in the patch.

## Threat Flags

None. This plan adds no runtime surface: one patch file under `.planning/` and one documentation
edit. No endpoint, no auth path, no file-access pattern, no schema change, no package install
(T-04-SC: zero new packages).

Threat register as covered:

| Threat | Covered as shipped |
| ------ | ------------------ |
| T-04-33 (autonomous CI edit) | ci.yml measured 0-byte porcelain at every step; the proposal is a file under `.planning/` |
| T-04-34 (unearned checkboxes) | zero boxes flipped; two frontmatter overclaims refused by name |
| T-04-35 (mutation scoped to nothing) | per-crate denominators enumerated (380/358/101/51 = 890) before any run |
| T-04-63 (a tier-covered surface silently absent from CI) | 24/24 accounted; the third exclusion surfaced rather than absorbed |

## Notes for the orchestrator and Phase 5

- **The patch is NOT applied and must be presented to the human before it is.** Plan Task 2 is the
  blocking checkpoint. `git apply --check` rc=0; `git apply` has never been run.
  Present alongside the 24-row accounting table above, not just the diff — the accounting is where
  the third exclusion is visible.
- **Three decisions are owed at that checkpoint:** (a) promote `setfit-serve-smoke` to CI?
  (b) promote `setfit-cli-lifecycle`? (c) **`setfit-api-boundary` — the exclusion I could not make
  cleanly**; it has real CI-only value and was dropped on quoting grounds, so it deserves a ruling
  rather than inheritance.
- **Mutation gate is unfinished by measurement, not by omission** — D-04-11-B carries the numbers
  and the four corrections a re-run needs. It belongs in a nightly job, not in a plan.
- **Phase 5 inherits, unchanged:** F-10 (the single blocker for OPS-01/OPS-02 and the artifact
  half of APR-01..05), `backend_identity` binding a nonexistent symbol (blocks a named clause of
  OPS-04 and OPS-06), F-14(5) bounded-read pre-reservation, and `apr qa`'s inapplicability to
  encoder-only APRs.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: .planning/phases/04-apr-artifact-and-production-parity/04-11-ci-setfit.patch   8,133 B
FOUND: .planning/phases/04-apr-artifact-and-production-parity/04-11-SUMMARY.md
FOUND: .planning/REQUIREMENTS.md                                                     32,794 B
FOUND: .planning/phases/04-apr-artifact-and-production-parity/deferred-items.md      15,101 B
```

Commits claimed, checked in `git log`: `eb0f8e1c5`, `608aa954e`, `5037e0d90` — all present.

| Assertion | Criterion | Observed |
| --- | --- | --- |
| `.github/workflows/ci.yml` modified | never | **0-byte porcelain** at every step, incl. after `--check` |
| `git apply --check` on the patch | rc=0 | **rc=0** |
| `git apply` run without `--check` | never | **never run** |
| `STATE.md` / `ROADMAP.md` touched | no | **not touched** — the orchestrator owns them |
| added cargo commands verbatim in the Makefile | all | **17/17** |
| pipes in added lines | 0 | **0** |
| forbidden legs as added commands | 0 | **0**, with a 5/5 executed case table proving the scan is live |
| 04-10 targets accounted for | all | **24/24** (10 already covered, 10 added, 1 aggregate, 3 excluded) |
| Phase 4 requirement boxes checked | 0 | **0** — `- [x]` count is 13, unchanged |
| TRN-07 label separation | `CROSS-PROCESS` on 04-15 only | **asserted mechanically** — present on 04-15's citation, absent from 04-07's |
| mutation score claimed from a partial run | none | **none claimed** |
| `bashrs` reported as passing | never | **never** — it is not installed on this host |
