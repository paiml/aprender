---
phase: 04-apr-artifact-and-production-parity
plan: 20
subsystem: testing
tags: [setfit, selection-lock, cli, ordering-discipline, no-clobber, apr-eval]

# Dependency graph
requires:
  - phase: 04-apr-artifact-and-production-parity
    provides: "`apr eval --lock-out` (04-07/04-16), `setfit_train::atomic_write` as the crate's ONE writer (04-06/04-16 refactor), `write_tagged_decoy` and the spawned lifecycle harness (04-15)"
provides:
  - "`apr eval --lock-out <FILE>` refuses an already-occupied lock destination BEFORE the dataset is read, before any artifact is loaded, and before any candidate is evaluated"
  - "`setfit_train::refuse_existing_output` promoted to `pub(crate)` — the standalone no-clobber gate is now callable from the sibling command it was factored out to serve"
  - "`eval::setfit::refuse_existing_lock` — the SINGLE producer of the 'a lock is a COMMITMENT' wording, shared by the pre-flight and the write"
  - "Two in-process ordering tests with DIFFERENT later-stage failures, plus a spawned three-leg process-tier witness"
affects: [phase-05-production-encoder-calibration, 04-22-makefile-floors, WR-01-atomic-write-race]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Ordering as a SOURCE fact, not only a test fact: the pre-flight's line number is asserted to sit between `check_split_flags` and `read_attested_canonical`"
    - "Falsify the witness, don't label it: every ordering claim here was proven by deleting the mechanism and observing RED"
    - "One decision, one wording: a bespoke message re-states a shared gate's refusal via `map_err` instead of re-implementing the predicate"

key-files:
  created: []
  modified:
    - crates/apr-cli/src/commands/eval/setfit.rs
    - crates/apr-cli/src/commands/eval/setfit_tests.rs
    - crates/apr-cli/src/commands/setfit_train.rs
    - crates/apr-cli/tests/setfit_cli_lifecycle.rs

key-decisions:
  - "Test 4 drives `setfit_train::atomic_write` directly rather than `write_lock`, because no `SelectionLock` is constructible in apr-cli (F-10) and hand-forging canonical lock bytes would be a second implementation of the lock wire form inside a CLI test"
  - "Test 2's later-stage failure varies WITHIN step (2) (unparseable manifest) rather than at step (3), because reaching step (3) in-process needs a valid attested dataset whose only fixture builder is `#[cfg(test)]`-private to `data_contrastive` — outside this plan's file scope"
  - "`refuse_existing_lock` delegates the exists()/force DECISION to `refuse_existing_output` and replaces only the MESSAGE, so there is one gate and one wording"
  - "No Makefile edit: `make setfit-cli-lifecycle`'s `assert_tests_ran` floor is a MINIMUM (`-lt` at Makefile:1699), so the new spawned case is covered by the existing filter"

patterns-established:
  - "Pattern: a guard's SCOPE extension requires re-mutating in the new scope — the write-time check was re-proven load-bearing by induced deletion after the pre-flight moved earlier"
  - "Pattern: a spawned CLI witness must be falsified (mechanism removed -> RED) before it may be cited, because a passing spawned test proves nothing about which code path answered"

requirements-completed: [TRN-07]

# Metrics
duration: 58min
completed: 2026-08-16
---

# Phase 04 Plan 20: WR-10 — the no-clobber gate now precedes the sweep Summary

**`apr eval --lock-out` no longer runs a full multi-candidate sweep before refusing to overwrite the lock file it was asked to write; the standalone gate `setfit_train.rs:12-20` says exists to be called early is now actually called early, proven by two in-process ordering tests with different later-stage failures and a falsified three-leg spawned witness.**

## Performance

- **Duration:** 58 min
- **Started:** 2026-08-16T18:08:55Z
- **Completed:** 2026-08-16T19:07:02Z
- **Tasks:** 2 of 2
- **Files modified:** 4

## Accomplishments

- Closed **WR-10**. The refusal that used to arrive after a bounded corpus read, the eight-rung load ladder and a classify pass over the whole validation split *for every candidate* now arrives before any input is opened.
- Made the ordering a **source-level fact** as well as a test fact, and made the write-time check's survival a **mutation-proven** fact rather than an assumed one.
- Added the spawned process-tier witness the plan asked for — and **falsified it** rather than reporting a green run as evidence.

## Task Commits

1. **Task 1: the ordering proof (RED at HEAD)** — `e9fbfd3a0` (test)
2. **Task 2: call the standalone gate early** — `9949f982f` (fix)
3. **Task 2 (cont.): the spawned ordering witness** — `e3c9c1f00` (test)

## Files Created/Modified

- `crates/apr-cli/src/commands/setfit_train.rs` — `refuse_existing_output` promoted `fn` → `pub(crate) fn`; its doc now states the three ORDERED call sites, that the write-time one is not made redundant by the two pre-flights, and that WR-01 is still open. **No behaviour change to the writer.**
- `crates/apr-cli/src/commands/eval/setfit.rs` — new `refuse_existing_lock` (single producer of the COMMITMENT text, delegating the decision to `refuse_existing_output`); pre-flight inserted in `run` between `check_split_flags` and the Phase 2 ingest; `write_lock` now calls the shared helper; step-(1) and `write_lock` comment blocks rewritten to describe three ordered checks.
- `crates/apr-cli/src/commands/eval/setfit_tests.rs` — the four tests, plus two fixtures (`occupied_lock_destination`, `corpus_with_an_unparseable_manifest`).
- `crates/apr-cli/tests/setfit_cli_lifecycle.rs` — `setfit_cli_lifecycle_wr_10_the_lock_refusal_precedes_the_corpus_read`, three spawned legs.

---

## The HEAD RED, verbatim

All four tests compiled and **ran at HEAD** (`91eba6f1c` + Task 1's test-only commit) before any production edit. Status captured with `cmd > log 2>&1; rc=$?`, never through a pipe.

```
test result: FAILED. 18 passed; 2 failed; 0 ignored; 0 measured; 6755 filtered out
```

Exactly 2 failed, and they are Tests 1 and 2. 18 = the 16-test HEAD baseline (independently measured) plus Tests 3 and 4, which pass at HEAD as predicted.

**Test 1 — the pre-fix code produced this instead of the lock refusal:**

```
the refusal must carry the domain wording that says what replacing a lock costs; got:
Validation failed: /var/folders/3s/xftgktnj6qs681vbh0tg5hmc0000gn/T/.tmpLNBZUt/no-such-corpus/benchmark-manifest.json
not found. Prepare one with `apr data tweet-eval-stance --output <DIR>` (canonical profile),
then point --data at that directory.
```

**Test 2 — a DIFFERENT unrelated error, which is what makes the pair an ordering proof:**

```
the same bespoke refusal must win here too; got:
Validation failed: benchmark-manifest.json is not valid JSON: expected ident at line 1 column 2
```

Both are step-(2) diagnoses — true statements about the wrong problem, reached only after the read the gate was supposed to precede. One failing input would have been an anecdote (CLAUDE.md rule 6); two later-stage failures of different *kinds* (not-found vs parse) both losing to the pre-flight is what distinguishes "the check moved earlier" from "the check happened to beat one particular error".

## The GREEN, and the source ordering

```
cargo test -p apr-cli --features setfit --lib eval::setfit
test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured
```

16 at HEAD → 20. Ordering is a source fact, from
`grep -n 'check_split_flags(args)?;|refuse_existing_lock(destination, args.force)|read_attested_canonical(' crates/apr-cli/src/commands/eval/setfit.rs`:

| line | site |
|------|------|
| **222** | `check_split_flags(args)?;` |
| **224** | `refuse_existing_lock(destination, args.force)?;` ← the pre-flight |
| **229** | `data_contrastive::read_attested_canonical(data, &mut ledger)?` |

`222 < 224 < 229` — the required relation holds.

Single-producer and visibility counts:

| check | expected | measured |
|-------|----------|----------|
| `grep -c 'A selection lock is a COMMITMENT' eval/setfit.rs` | 1 | **1** |
| `grep -c 'pub(crate) fn refuse_existing_output' setfit_train.rs` | 1 | **1** |

## The induced deletion — the write-time check is still load-bearing

Deleting `refuse_existing_output(target, force)?;` from `atomic_write` turns Test 4 RED:

```
thread '...write_lock_still_refuses_a_destination_that_appeared_mid_run' panicked at
crates/apr-cli/src/commands/eval/setfit_tests.rs:464:10:
a file that appeared mid-run must still not be clobbered: ()

test result: FAILED. 0 passed; 1 failed
```

Reverted; `eval::setfit` back to 20 passed / 0 failed. This is CLAUDE.md rule 4 in miniature: the pre-flight extended the gate's SCOPE, so the write-time proof was re-run in the new arrangement rather than assumed to transfer.

## The spawned witness — RUN, and falsified

**Result: RAN and PASSED.** Not a NOT-RUN.

A naive live `"$APR"` probe cannot reach this code, and that was measured rather than assumed. `dispatch_analysis.rs:759` reads the SetFit tag *before* `eval::setfit::run`, so an untagged artifact is refused at `:787`. I re-enumerated every `*.apr` in the repository independently of the plan: **11 files, ZERO carrying the `setfit-apr-v1` schema string**; the nearest, `crates/aprender-core/tests/fixtures/setfit/slice_model.apr`, is `"model_type":"Bert"`. F-10 prevents producing a tagged one. So the case is built on `write_tagged_decoy` (`setfit_cli_lifecycle.rs:731`), spawned via `env!("CARGO_BIN_EXE_apr")`, verdict read off a reaped `ExitStatus`.

Three legs, each varying exactly one thing:

| leg | varies | verdict |
|-----|--------|---------|
| W1 | occupied `--lock-out` | exit **5**, COMMITMENT refusal, **silent about the absent corpus** |
| W2 | vacant `--lock-out` | exit 5, names the corpus — the mechanism proof |
| W3 | occupied + `--force` | exit 5, names the corpus, no COMMITMENT refusal, prior lock byte-identical |

**W1's verbatim output, printed from the run itself** (`-- --ignored lifecycle --nocapture`, the convention Test 3 established so a SUMMARY transcribes from a run rather than from a format string):

```
[04-20] W1 spawned refusal (exit 5): error: Validation failed:
/var/folders/3s/xftgktnj6qs681vbh0tg5hmc0000gn/T/.tmp5lV0pT/committed-lock.json already exists.
A selection lock is a COMMITMENT, so overwriting one is never implicit: pass --force if you
intend to replace the committed decision, and be aware that any test measurement taken under
the old lock no longer describes the selection this file records.
```

**A passing spawned test proves nothing about which code path answered, so it was falsified.** With the pre-flight deleted from `run`, W1 turns RED and the reaped child reports the pre-fix behaviour:

```
expected the run to name `A selection lock is a COMMITMENT`:
  argv:    ["eval", ".../tagged.apr", "--task", "classify", "--data", ".../no-such-corpus",
            "--selection", ".../no-such-selection.json", "--split", "validation",
            "--lock-out", ".../committed-lock.json"]
  status:  ExitStatus(unix_wait_status(1280)) (code Some(5))
  elapsed: 66.99075ms
  stderr:  error: Validation failed: .../no-such-corpus/benchmark-manifest.json not found.
           Prepare one with `apr data tweet-eval-stance --output <DIR>` (canonical profile),
           then point --data at that directory.
```

The mutation was reverted and the case re-run green.

**Which Make gate covers it** — read from the recipe, then measured. `make setfit-cli-lifecycle` drives `-- --ignored lifecycle`; the new test's name contains `lifecycle`, so it is selected, and that leg went **2 → 3 tests**. `assert_tests_ran` (Makefile:1697-1706) compares with `-lt`, i.e. it is a **floor**, so no Makefile edit was needed or made.

## Verification

Every status captured with `cmd > log 2>&1; rc=$?`.

| # | command | rc | result |
|---|---------|----|--------|
| 1 | `make setfit-cli-eval-tests` | **0** | 20 passed, 0 failed; floor 13 satisfied |
| 2 | `make setfit-cli-train-tests` | **0** | 15 passed / 0 failed (floor 13) + 1 `--ignored` leg (floor 1) |
| 3 | `cargo test -p apr-cli --features setfit --lib` | **0** | **6768 passed, 0 failed, 15 ignored** (floor 6750) |
| 4 | `make setfit-cli-lifecycle` | **0** | lifecycle leg 3 passed (floor 2), tooling leg 1 passed (floor 1) |
| 5 | `make setfit-feature-matrix` | **0** | `setfit-feature-matrix: PASSED`; apr-cli setfit delta 0 off → 79 on |
| 6 | `cargo test -p apr-cli --features setfit --test setfit_cli_lifecycle -- --ignored` | **0** | 4 passed, 0 failed |

Also: `rustfmt --check` on all four files rc=0; `cargo clippy -p apr-cli --features setfit --lib --tests` rc=0 with no finding in my files; no `unwrap()` added (`git diff 91eba6f1c..HEAD | grep '^+.*unwrap()'` empty).

**Negative-assertion count.** The plan's literal metric `grep -c 'assert!(!'` is **0 at HEAD and 0 now** — rustfmt reflows the single-line form into the repo's multi-line style, so that proxy cannot move and the repo's own `cargo fmt --check` gate wins. Measured with a regex that matches the style rustfmt actually produces (`^[[:space:]]+![A-Za-z_]`): **4 → 11 (+7)**, of which three are the message-negatives in Tests 1 and 2. Recording both numbers rather than the one that flatters (CLAUDE.md rule 7: a guard regex ships a case table).

## Deviations from Plan

### [Rule 3 - Blocking] Test 2's later-stage failure varies within step (2), not at step (3)

- **Found during:** Task 1
- **Issue:** The plan specified Test 2 use a *valid* `--data` so the run would fail at step (3) (`reload_artifact`). Reaching step (3) in-process requires a valid attested prepared dataset; the only fixture that writes one is `#[cfg(test)]`-private to `data_contrastive.rs`, which is **not** in this plan's `files_modified`. Promoting it would put a dataset fixture in a second place and widen the file scope into a sibling's territory.
- **Fix:** Test 2 uses an *existing* corpus directory whose `benchmark-manifest.json` is unparseable, producing a genuinely different diagnosis (`is not valid JSON`) from Test 1's (`not found`). The plan's underlying property — two different later-stage failures both losing to the pre-flight — is preserved and measured; only the *depth* of the second failure changed.
- **Consequence, stated plainly:** there is **no step-(3) ordering witness** in this plan. The two in-process witnesses are both step-(2); the spawned witness is also step-(2). WR-10's fix sits above all of step (2), so this is sufficient for the finding, but a future plan that wants an artifact-load-tier witness will need a reachable attested-dataset fixture.
- **Files:** `crates/apr-cli/src/commands/eval/setfit_tests.rs`

### [Rule 3 - Blocking] Test 4 drives `atomic_write`, not `write_lock`

- **Found during:** Task 1
- **Issue:** The plan said to drive `write_lock` directly. `write_lock` takes a `&SelectionLock`, and none is constructible in `apr-cli` — the tests module header already records why (`SelectionCandidate::from_evaluation` needs a `ValidationEvaluation` whose only producers take a verified model; F-10 blocks producing one). `SelectionLock::from_canonical_bytes` exists, but hand-forging canonical lock bytes in a CLI test would be a second implementation of the lock's wire form. Separately, the plan's own acceptance criterion — *deleting `refuse_existing_output` inside `atomic_write` must turn Test 4 RED* — is unsatisfiable through `write_lock`, whose own check fires first.
- **Fix:** Test 4 pins the property at the writer `write_lock` delegates to, plus a source anchor asserting the delegation still exists. Both halves of the criterion are then met, and the induced deletion does turn it RED (transcript above).
- **Files:** `crates/apr-cli/src/commands/eval/setfit_tests.rs`

### [Environment, not a code deviation] This ran in the SHARED main checkout, not an isolated worktree

- The orchestrator's brief described a git worktree with sibling agents isolated. In fact `.git` is a **directory** (`git worktree list` shows one entry) and all four wave-11 agents operated in the **same working tree** on `gsd/phase-2-contract-gate`.
- **Observed consequence:** partway through Task 2 my *uncommitted* edits to `eval/setfit.rs` and `setfit_train.rs` were silently reverted while files I never touched (`inspect.rs`, `classify.rs`) appeared modified. No stash was involved (`git stash list` empty). My Task 1 commit survived; the edits were re-applied and committed promptly.
- **Mitigations applied:** only my own paths were ever staged; a crate-wide `cargo fmt` was replaced with `rustfmt` on my four files after it dirtied a sibling's `serve/handlers.rs` (that file was reverted, untouched by me); two verification runs failed to compile on siblings' half-landed edits (`aprender-core/setfit/classify.rs` E0599, `setfit_tag_tests.rs` E0603) and were retried to green rather than "fixed".
- **Worth flagging to the orchestrator**: with concurrent agents in one tree, uncommitted work is not safe, and full-crate test counts are not attributable to a single plan. The `eval::setfit` figures (16 → 20) *are* attributable — measured twice, on files verified unchanged by others (`git diff e3c9c1f00 HEAD -- <my files>` empty).

No Rule 1, Rule 2 or Rule 4 deviations. No architectural change. No dependency added; no `Cargo.toml` touched (threat T-04-SC not triggered).

## Threat Model Disposition

| Threat ID | Disposition | Outcome |
|-----------|-------------|---------|
| T-04-78 (DoS: wasted sweep) | mitigate | **Mitigated.** Pre-flight sits above `read_attested_canonical`; proven at source (222 < 224 < 229), in-process (2 tests), and spawned (W1 silent about the corpus). |
| T-04-79 (Tampering: replacing a committed lock) | mitigate | **Mitigated.** Three ordered checks, all `--force`-gated. The third was proven not deleted, by induced deletion. |
| T-04-80 (check → `rename` window) | accept | **NOT closed, as planned.** See below. |
| T-04-81 (refusal message disclosure) | accept | Unchanged — names only the operator-supplied path and the `--force` remedy. |
| T-04-SC (dependency installs) | mitigate | Not triggered: no dependency added, no `Cargo.toml` in the diff. |

## What remains OPEN — stated plainly

- **WR-01 is OPEN. This plan does NOT make the write path race-free.** `fs::rename` inside `atomic_write` replaces its destination unconditionally, so a file created between the check and the rename is still destroyed without `--force`. This plan *narrows* the window by refusing earlier; closing it needs the destination taken with `O_CREAT|O_EXCL`. Both `refuse_existing_output`'s doc and `write_lock`'s comment now say so in the source, so a reader cannot infer otherwise.
- **F-10 is OPEN.** No `setfit-apr-v1` artifact can be produced on this host — re-measured here: 11 `*.apr` files in the repo, zero tagged. The production-encoder calibration that unblocks it is Phase 5 work per the ROADMAP's blocking note.
- **OPS-01 and OPS-02 are NOT MET.** This plan closes neither. OPS-02's "a user can complete the lifecycle" clause is blocked behind F-10.
- **TRN-07 stays UNCHECKED overall.** This plan changes *when* the lock record's creation is refused, which is the half that was reachable; its missing positive "a user can" tier still needs a user-produced artifact.
- **SC1, SC2, SC3 and the "over a produced artifact" halves of SC4/SC5** remain OPEN.
- **The per-crate cargo-mutants gate (04-11 must-have 4) and the SAFE-02 "in CI" clause** remain OPEN human_verification items — out of scope here.
- **IN-06 / D-04-14-B** (the `unwrap()` ban being lint-inert via `crates/apr-cli/src/lib.rs:9-16`) is untouched and out of scope; the ban was still honoured by discipline in every line added.
- **WR-03, WR-04, WR-06, IN-01, IN-02, IN-07** also live in `eval/setfit.rs` and were deliberately **not** touched.
- **`Makefile:1918`'s gate table is still stale** (records `15 / 0` for `eval::setfit`; actual is now 20). Not corrected here — 04-21 owns the Makefile this wave and 04-22 raises the `setfit-cli-eval-tests` floor in wave 12.
- **No step-(3) ordering witness exists** (see the first deviation).

## Known Stubs

None. Nothing added is a placeholder; every assertion added runs and was independently falsified.

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or schema change at a trust boundary. The one new filesystem interaction is a `Path::exists()` check on an operator-supplied path that the same command already opened later.

## Self-Check: PASSED

Files:

- `FOUND: crates/apr-cli/src/commands/eval/setfit.rs`
- `FOUND: crates/apr-cli/src/commands/eval/setfit_tests.rs`
- `FOUND: crates/apr-cli/src/commands/setfit_train.rs`
- `FOUND: crates/apr-cli/tests/setfit_cli_lifecycle.rs`

Commits (all verified ancestors of HEAD after sibling commits landed on top):

- `FOUND: e9fbfd3a0` test(04-20) — the RED
- `FOUND: 9949f982f` fix(04-20) — the pre-flight
- `FOUND: e3c9c1f00` test(04-20) — the spawned witness

Per the orchestrator's instruction, **STATE.md and ROADMAP.md were NOT modified** — the orchestrator owns those writes after the wave completes.
