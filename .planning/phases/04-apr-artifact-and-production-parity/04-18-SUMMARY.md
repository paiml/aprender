---
phase: 04-apr-artifact-and-production-parity
plan: 18
subsystem: cli-artifact-detection
tags: [setfit, apr, apr-cli, inspect, predict, eval, security, dos, wr-08, wr-09, t-04-70, t-04-71, apr-02, apr-05, ops-03]

# Dependency graph
requires:
  - phase: 04-08
    provides: "serve/handlers.rs:774 `.ok().flatten()` — the fall-through the new Err must not break"
  - phase: 04-06
    provides: "commands/predict.rs:114 — the `?`-propagating consumer that needed no edit"
  - phase: 04-17
    provides: "the consolidated single detector `setfit_tag::read_setfit_tag` this plan hardens"
provides:
  - "crate::setfit_tag::MAX_TAG_METADATA_BYTES — ONE absolute metadata cap, pub(crate), referenced by both readers and re-declared by neither"
  - "commands::inspect::read_metadata bounded by that cap BEFORE its vec![0u8; ..] allocation (T-04-70 closed)"
  - "MetadataInfo::metadata_over_cap_bytes — the over-cap fact DISCLOSED rather than silently defaulted"
  - "read_setfit_tag returns a typed CliError::InvalidFormat for an over-cap block instead of Ok(None) (T-04-71 closed)"
  - "four_consumers_agree_about_one_over_cap_file — an executed table over all four decision surfaces"
affects: [04-19, 04-20, 04-21, phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two bounds where neither subsumes the other: the stat'd file length is tighter for a small file, an absolute cap is tighter for a large one — and a sparse file satisfies the first for free"
    - "A fail-open `Ok(None)` in a SHARED detector is a fan-out defect: one value became three statements at three consumers, two of them untrue"
    - "`Err` at a shared door needs no consumer edit when the consumers already `?`-propagate — and the one that deliberately swallows keeps its disposition for free"
    - "A disclosure field with `skip_serializing_if` adds a diagnosis to the pathological case while leaving every legitimate output byte-identical and every call site untouched"
    - "A test that only reaches the reachable surfaces is theater (CLAUDE.md rule 5); the visibility seam that makes the DECISION surface reachable is part of the guard, and COMPILING is the reachability proof"

key-files:
  created: []
  modified:
    - crates/apr-cli/src/setfit_tag.rs
    - crates/apr-cli/src/setfit_tag_tests.rs
    - crates/apr-cli/src/commands/inspect.rs
    - crates/apr-cli/src/commands/inspect_tests.rs
    - crates/apr-cli/src/commands/predict_tests.rs
    - crates/apr-cli/src/dispatch_analysis.rs
    - crates/apr-cli/src/commands/construction.rs   # NOT in the plan's files_modified — see Deviations

decisions:
  - "The human-readable disclosure lives in a NEW function in inspect.rs, not in output_metadata_text — because that function is in inspect_output_json.rs, which an acceptance criterion requires to stay out of the diff"
  - "The over-cap message deliberately says NOTHING about SetFit: the block was never read, so that fact is genuinely unknown at the refusal point"
  - "The inspect.rs visibility seam is WIDER than the plan budgeted; the compiler, not preference, set its size"

metrics:
  duration: ~2h
  tasks: 3
  commits: 3
  completed: 2026-08-16
---

# Phase 4 Plan 18: Bound the Artifact-Detection Surface Summary

Closes **WR-09** (an unbounded attacker-controlled allocation in `apr inspect`) and **WR-08**
(a fail-open that made three tools state contradictory things about one file), by giving both
readers ONE absolute metadata cap and turning the over-cap case into a typed refusal that says
something true on every consumer.

## Commits

| Task | Commit | Subject |
|------|--------|---------|
| 1 | `c756c48ea` | `fix(04-18): WR-09 — bound apr inspect's metadata read by the shared absolute cap` |
| 2 | `323bbbc35` | `fix(04-18): WR-08 — an over-cap metadata block is a typed refusal, not a fail-open` |
| 3 | `15bea7b6d` | `test(04-18): the four-consumer agreement table, over ONE over-cap file` |

## Baselines, MEASURED BEFORE EDITING

All at the plan's base commit `91eba6f1c`, captured with `cmd > log 2>&1; rc=$?` — never through
a pipe. Counts read from libtest's own `test result:` line via the same `awk` the Makefile's
`assert_tests_ran` uses. (The `rtk` hook rewrites `cargo test` output into a summarised form that
`assert_tests_ran` would read as 0, so every measurement below used `rtk proxy cargo test`.)

| Measurement | Baseline | After | Floor |
|-------------|----------|-------|-------|
| `cargo test -p apr-cli --features setfit --lib` | **6756 passed, 0 failed, 15 ignored** | 6769 passed, 0 failed, 15 ignored | ≥6750 |
| `--lib inspect` | **120** | 123 | ≥123 (make floor 110) |
| `--lib setfit_tag` | **8** | 12 | strictly > baseline |
| `--lib predict` | **34** | 35 | ≥35 (make floor 30) |
| `--lib serve` | **361** | 361 | make floor 330 |
| `grep -c 'return Ok(None);' setfit_tag.rs` | **9** | 8 | exactly one lower |
| `grep -c 'const MAX_' inspect.rs` | 0 | 0 | must stay 0 |

The plan's stated 6756 was reproduced exactly. The +13 delta is 9 tests I added plus 4 that
arrived from a sibling plan's commit landing in the same checkout mid-run (see Environment below);
sibling drift is purely additive, so the floor claim is unaffected.

## RED-before-GREEN, with the exact failure lines

**Task 1, Test 1** — run against `read_metadata` with the field present but the cap hunk absent:

```
thread 'commands::inspect::inspect_tests::read_metadata_refuses_a_block_over_the_absolute_cap'
panicked at crates/apr-cli/src/commands/inspect_tests.rs:465:9:
assertion `left == right` failed: an over-cap block must be DISCLOSED, not silently defaulted
— the fact that it was refused is the whole diagnosis (WR-08's inspect half)
  left: None
 right: Some(16777217)
```

The two sibling controls (`..._admits_a_block_of_exactly_the_absolute_cap`,
`..._still_parses_a_legitimate_artifact_under_the_cap`) passed pre-fix, which is correct — they
are boundary and non-vacuity controls, not the subject.

**Task 2** — 4 failed / 1 passed against the `Ok(None)` fail-open. The `predict` failure captured
the defect verbatim, for a container that genuinely carries `model_type = "setfit"`:

```
panicked at crates/apr-cli/src/commands/predict_tests.rs:123:5:
predict must surface the CAP that fired, so the operator learns what to fix; got:
Invalid APR format: /var/.../over-cap.apr: not a SetFit classifier. `apr predict` supports
`setfit-apr-v1` classifiers in v1. ... run `apr inspect <FILE>` to see what it actually is.
```

The non-vacuity control (`..._keeps_both_in_bounds_dispositions_...`) passed pre-fix.

**Task 3, mutation control** — the Task 2 `Err` reverted to `Ok(None)`, run, then reverted (the
file was restored from a byte copy and verified identical to `323bbbc35`; `return Ok(None);` is
back to 8):

```
panicked at crates/apr-cli/src/setfit_tag_tests.rs:332:10:
the shared detector refuses a block it cannot cheaply read: None
```

The table fails at the FIRST surface, so on its own it does not exhibit *which* consumer then
lies. I therefore ran `predict_surfaces_the_cap_refusal` under the same mutation, which does:
it reproduced the `not a SetFit classifier … run apr inspect <FILE>` message above. Both halves
are recorded because the table alone would have understated the control.

## Verification — all six items RUN

| # | Check | Result |
|---|-------|--------|
| 1 | `cargo test -p apr-cli --features setfit --lib` | rc=0, **6769 passed, 0 failed, 15 ignored** |
| 2 | `make setfit-cli-inspect-tests` | rc=0, 123 passed (floor 110) |
| 3 | `make setfit-cli-predict-tests` | rc=0, 35 passed (floor 30) |
| 4 | `make setfit-cli-serve-tests` | rc=0, **361 passed** — unchanged from baseline, i.e. the `.ok().flatten()` fall-through survived |
| 5 | `make setfit-feature-matrix` | rc=0, **PASSED**; apr-cli SAFE-02 delta 18 off → 79 on (+61, needs ≥40) |
| 6 | Live binary probe, PINNED | **RUN** — see below |

### Live probe (item 6) — binary pinned and proven

`. scripts/apr_bin.sh` resolved `target/debug/apr`, reporting `apr 0.63.0 (15bea7b6d)`, matching
`git rev-parse --short HEAD` = `15bea7b6d` exactly. Not a bare `apr`, not a hardcoded path.

Fixture built per the plan's recipe, with the offsets **CONFIRMED by reading
`crates/apr-format/src/v2/header_impl.rs` first**, not taken from the plan: `AprV2Header::from_bytes`
reads `metadata_offset` as a LE `u64` at bytes 12..20 (line 79) and `metadata_size` as a LE `u32` at
bytes 20..24 (line 80). `golden_v2.apr` (1092 bytes) copied, bytes 20..24 overwritten with LE
`16777217`, file extended to 16777282 bytes so the pre-existing length check passes and the cap is
what fires. Measured `metadata_offset` = 64, as expected.

`"$APR" inspect` — **exit 0**, no 4 GiB allocation, and the refusal is disclosed:

```
  Metadata: NOT READ
    The header declares 16777217 bytes, over the 16777216 byte (16 MiB) cap.
    The block was NOT read, so no model type, no identity fields and no SetFit section
    could be derived from it.
```

`"$APR" predict … --text hi` — **exit 4**:

```
error: Invalid APR format: target/04-18/probe/over-cap.apr: the metadata block declares
16777217 bytes, over the 16777216 byte (16 MiB) cap this reader will read to identify a
container. The block was NOT read, so what this file is has not been determined.
```

Neither says "not a SetFit classifier". Both cite one cause.

**Controls, because one input is an anecdote (CLAUDE.md rule 6).** The same pinned binary on the
UNMODIFIED `golden_v2.apr`: `inspect` exit 0, `Checksum ✓ VALID`, `Model Type: linear_regression`,
and **zero** `Metadata: NOT READ` lines; `inspect --json` contains **zero** occurrences of
`metadata_over_cap_bytes`. The "every legitimate `--json` output is byte-identical" claim is
therefore executed, not asserted.

## Deviations from Plan

### [Rule 3 — Blocking] `commands/construction.rs` had to be edited; it is not in `files_modified`

Adding `metadata_over_cap_bytes` to `MetadataInfo` broke a second exhaustive struct literal at
`construction.rs:405` (a test fixture for `output_metadata_text`) with `E0063`. There is no way to
add a field without fixing every exhaustive literal. One line added, plus a comment; the literal was
kept exhaustive rather than elided behind `..Default::default()`, because exhaustiveness is exactly
what made the compiler point at the file. **Commit `c756c48ea`.**

### [Rule 3 — Blocking] The `inspect.rs` visibility seam is wider than the plan's "one keyword"

The plan specified promoting `read_metadata` to `pub(crate)` "and nothing else". That does not
compile. The compiler's verdict, taken as the specification:

```
error[E0603]: function `read_and_parse_header` is private
error[E0616]: field `metadata_over_cap_bytes` of struct `MetadataInfo` is private
error[E0616]: field `model_type` of struct `MetadataInfo` is private
error[E0616]: field `setfit_doc` of struct `MetadataInfo` is private
```

`read_metadata`'s signature names two module-private types, and the facts the table must assert
live in private fields. So `read_and_parse_header`, `HeaderData`, `MetadataInfo` and exactly three
of its fields are also `pub(crate)`. Everything else in the struct stays private. The seam was
grown by compiling, not by preference. **Commit `15bea7b6d`.**

**`dispatch_analysis.rs` was held to the plan's strict bound**: its entire diff is the one
declaration line plus a two-line comment. The flag table, the refusal messages and every call site
are untouched — verified by reading the diff, which is reproduced in the commit body.

### [Rule 3 — Blocking] The disclosure renderer could not go where the plan pointed

The plan said to render the disclosure at "the metadata rendering site in `inspect.rs` … that
prints model type, name, author and the provenance keys". That function is `output_metadata_text`,
which lives in **`inspect_output_json.rs`** — a file a Task 3 acceptance criterion requires to be
ABSENT from the diff. The two instructions conflict. Resolved in favour of the hard criterion: a
new `output_metadata_over_cap_text` in `inspect.rs`, called from the non-`--json` branch of `run()`.
`git diff --name-only` does not list `inspect_output_json.rs`. **Commit `15bea7b6d`.**

### [Out of scope — logged, NOT fixed]

- `cargo fmt --check` fails on `crates/apr-cli/src/commands/serve/handlers.rs:1402`. This file is
  **not dirty in git**, so the violation is pre-existing at HEAD and unrelated to this plan. Left
  alone per the scope boundary.
- `cargo clippy -p apr-cli --all-targets` returned `error[E0599]: no method named 'identity' found
  for struct 'setfit::encoder::ExecutionBackend'` in **`aprender-core`** — a sibling plan's
  in-flight edit, not reachable from anything this plan changed. My own `cargo test -p apr-cli`
  runs were rc=0 throughout, so apr-cli itself compiles clean.

## Environment anomaly — this plan did NOT run in an isolated worktree

Recorded because it affects how the numbers above should be read, and because it silently destroyed
work once.

The orchestrator briefed this as a parallel executor in a git worktree. It was not: `.git` is a
**directory**, `git worktree list` showed a single entry, and HEAD was on `gsd/phase-2-contract-gate`
— the main checkout, shared concurrently with siblings 04-19, 04-20 and 04-21.

Consequences, all measured:

1. **Mid-flight clobber.** After Task 1's RED was observed, my edits to `setfit_tag.rs`,
   `inspect_tests.rs` and `construction.rs` were reverted to HEAD by a concurrent operation; only
   my most recent edit survived, leaving `inspect.rs` in a non-compiling state. I re-applied the
   four lost edits and added a **marker assertion immediately before every `git add`** for the rest
   of the plan. No commit was made from a clobbered tree.
2. **No contamination, verified.** Sibling commit `e9fbfd3a0` was inspected: it contains only
   `crates/apr-cli/src/commands/eval/setfit_tests.rs`, and `git show e9fbfd3a0 | grep -c
   metadata_over_cap_bytes` returned 0. My work did not leak into theirs, and only my own files
   were ever staged — never `git add .` or `-A`.
3. **HEAD moved during the plan**, from `91eba6f1c` to `15bea7b6d`, via sibling commits. Baselines
   were taken at `91eba6f1c` before any edit; sibling drift is additive, so the floors hold.

Per the destructive-git prohibition I ran no `git clean`, `git stash`, `git reset --hard` or
blanket restore at any point. The `worktree-agent-*` HEAD assertion was correctly skipped: it is
guarded by `[ -f .git ]`, and `.git` is a directory here.

## What this plan does NOT close — OPEN after this run

Stated plainly, per the plan's own success criteria:

- **This plan closes neither OPS-01 nor OPS-02.**
- **F-10** (production-encoder calibration) remains OPEN. With it, **OPS-01, OPS-02, SC1, SC2, SC3**
  and the "over a produced artifact" halves of **SC4/SC5** remain OPEN. The ROADMAP's explicit
  blocking note puts that work in Phase 5.
- The **per-crate `cargo-mutants` gate** (04-11 must-have 4, ≥10h) was NOT run and remains a
  `human_verification` item.
- The **SAFE-02 "in CI" clause** remains OPEN. `make setfit-feature-matrix` was run **locally** and
  passed; that is not the same as it running in CI, and `.github/workflows/*.yml` is out of scope.
- **IN-06** is out of scope and untouched.
- **WR-01, WR-03, WR-04, WR-05, WR-06** and **IN-01/02/04/05/07** from `04-REVIEW.md` remain open
  and untouched by this plan.
- **T-04-73** (the header checksum is computed at `inspect.rs` but nothing branches on it) is
  explicitly ACCEPTED here and untouched. The live probe above shows `Checksum ✗ INVALID` on the
  patched fixture while `inspect` still proceeds — the cap bounds resource use, it does not
  authenticate the header. That belongs to whoever closes the APR-02 residual I-01.

## Threat Flags

None. This plan adds no dependency (`crates/apr-cli/Cargo.toml` untouched, per T-04-SC), opens no
network endpoint, adds no auth path and changes no schema. It narrows an existing trust boundary.

## Known Stubs

None.

## Self-Check: PASSED

Files claimed as modified, all confirmed present on disk:

```
FOUND: crates/apr-cli/src/setfit_tag.rs
FOUND: crates/apr-cli/src/setfit_tag_tests.rs
FOUND: crates/apr-cli/src/commands/inspect.rs
FOUND: crates/apr-cli/src/commands/inspect_tests.rs
FOUND: crates/apr-cli/src/commands/predict_tests.rs
FOUND: crates/apr-cli/src/dispatch_analysis.rs
FOUND: crates/apr-cli/src/commands/construction.rs
```

Commits claimed, all confirmed in `git log`:

```
FOUND: c756c48ea
FOUND: 323bbbc35
FOUND: 15bea7b6d
```

Per the orchestrator's instruction, **STATE.md and ROADMAP.md were NOT modified** by this plan.
