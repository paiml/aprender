---
phase: 02-deterministic-pair-and-data-protocol
plan: 01
subsystem: data-protocol
tags: [d06-baseline, contracts, tweet-eval, pv, makefile-gates, provenance]
requires: []
provides:
  - "gsd/phase-2-d06-baseline branch + PR #1 (stacked baseline, twelve files, hash-attested)"
  - "gsd/phase-2-contract-gate branch — the ONE working branch for Phase 2 waves 2-6"
  - "tracked D-06 baseline: data_tweeteval.rs, tweet-eval contract, docs example, apr eval F_avg"
  - "contracts/tweet-eval-stance-benchmark-v1.yaml v1.1.0, pv-valid, tier3-reachable"
  - "OBLIG-TWEET-EVAL-{COUNTS,REVISION-HONESTY,HASH-FROM-PARSED-BYTES,SEED-SET,LABEL-BOUNDS}"
  - "FALSIFY-TWEET-EVAL-008 + label_index proptest (runnable backing for KANI-TWEET-EVAL-001)"
  - "corrected repo-wide pv diff usage in CLAUDE.md"
affects:
  - "all Phase 2 plans 02-02..02-09 (branch/PR policy, baseline to diff against)"
  - "plan 02-09 (OBLIG-TWEET-EVAL-SEED-SET is what its required --seed CLI discharges)"
  - "any future contract author (the $(CONTRACTS) list is not a glob)"
tech-stack:
  added: []
  patterns:
    - "pre-staging SHA-256 record + post-commit blob digest verification for as-is landings"
    - "explicit $(CONTRACTS) wiring as the only path from a contract file to a tier"
    - "declared-not-executed Kani harnesses backed by identically bounded proptests"
key-files:
  created:
    - .planning/phases/02-deterministic-pair-and-data-protocol/02-01-SUMMARY.md
  modified:
    - contracts/tweet-eval-stance-benchmark-v1.yaml
    - Makefile
    - CLAUDE.md
    - crates/apr-cli/src/commands/data_tweeteval.rs
decisions:
  - "Phase 2 ships as exactly two PRs; waves 2-6 all commit to gsd/phase-2-contract-gate"
  - "Baseline PR stacked on the Phase 1 branch, not main, and auto-merge deliberately NOT armed"
  - "Contract bumped 1.1.0 (minor) per pv diff's own suggestion — purely additive"
  - "make contract-audit left red: already red at HEAD from Phase 1's contract, and in no tier"
metrics:
  duration: ~1h20m
  tasks: 2
  files: 14
  completed: 2026-08-08
---

# Phase 2 Plan 01: D-06 Baseline and Contract Gate Summary

Landed the 813-line TweetEval baseline as one hash-attested standalone commit and its own
stacked PR, then turned `tweet-eval-stance-benchmark-v1.yaml` from an unvalidated file in a
directory into a contract that tier3 actually runs.

## Branch and PR policy for the REST of Phase 2 (stated once, here)

**Plans 02-02 through 02-09 inherit this and do not restate it.**

`.planning/config.json` sets `git.branching_strategy: "none"`, so GSD is not managing branches
for this phase and the policy is this plan's to set.

- All Phase 2 work after the baseline commit rides on **ONE working branch,
  `gsd/phase-2-contract-gate`**, created by Task 2 **from the baseline commit**
  `7eb1a67faea0b913e7e42eb2d369daf123a49810`.
- Every plan in waves 2-6 **commits to that branch**. They do **NOT** create their own branches
  and do **NOT** open their own PRs.
- The phase ships as **exactly two PRs**: (1) the baseline PR
  ([#1](https://github.com/guyernest/aprender/pull/1)), and (2) one Phase 2 PR opened from
  `gsd/phase-2-contract-gate` after wave 6, stacked on the same base while Phase 1 remains
  unlanded.
- Rebasing the Phase 2 branch onto `main` after Phase 1 lands is a **post-phase action**, not a
  plan step.

HEAD is left on `gsd/phase-2-contract-gate` at `4ea35e1ee`, which is where waves 2-6 fork from.

## What Was Built

### Task 1 — the D-06 baseline, landed as-is and attested by digest

Commit `7eb1a67faea0b913e7e42eb2d369daf123a49810` on branch `gsd/phase-2-d06-baseline`.
Exactly twelve files, 1347 insertions, 9 deletions. Zero content edits.

The interesting part is not the commit, it is the proof that the committed bytes are the bytes
that were tested. `git show --stat` showing an untracked file as ADDED proves only that it is
new. So: SHA-256 of all twelve files was recorded **before** staging
(`/tmp/d06-baseline.sha256`), and after committing, each blob was hashed **straight out of the
git object store** (`git cat-file blob HEAD:<path>`) and compared. All twelve matched.
`cargo fmt` was run in `--check` mode only and never permitted to write — a format write here
would have contradicted "as-is" and invalidated the digests.

PR [#1](https://github.com/guyernest/aprender/pull/1) — base
`gsd/phase-1-differentiable-minilm-conformance`, head `gsd/phase-2-d06-baseline`, exactly twelve
files, **auto-merge deliberately not armed** (merging into an unlanded Phase 1 branch would
reorder the stack).

### Task 2 — the contract gate

Commit `4ea35e1ee40aa19fef9905090ac0f96adb612acf` on `gsd/phase-2-contract-gate`, branched from
the baseline commit precisely so the baseline PR's diff can never grow. Verified after the fact:
`git diff --name-only gsd/phase-1-...^{}...7eb1a67fa` is still exactly twelve files, and
`gh pr view 1 --json files` still returns 12.

- **The contract is pv-valid at v1.1.0.** Five `proof_obligations` and one `kani_harnesses`
  entry close PROVABILITY-001 ×2. The restructure is **purely additive** — a diff of the
  `falsification_tests` block old-vs-new has **zero removed or changed lines**; the seven
  original `FALSIFY-TWEET-EVAL-00x` entries survive verbatim and `FALSIFY-TWEET-EVAL-008` is
  appended.
- **The Makefile wiring is the whole point.** `$(CONTRACTS)` is an explicit hardcoded list, not
  a glob over `contracts/*.yaml`. The contract had been sitting in the tree failing
  `pv validate` and no gate noticed, because nothing referenced it. `make contract-validate` now
  runs 43/43 contracts green and reaches the tweet-eval entry; tier3 calls that target.
- **`OBLIG-TWEET-EVAL-SEED-SET`** pins the ten contracted seeds to the value of
  `BENCHMARK_SEEDS` at `data_tweeteval.rs:45`, and states that 42 is not a contracted seed — a
  tool defaulting `--seed` to 42 samples outside the contract while appearing to honour it.
  This is the obligation plan 02-09's required-`--seed` design discharges.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] The plan's own mandated test command does not exist**

- **Found during:** Task 2, writing the Kani honesty prose.
- **Issue:** The plan required the harness to name
  `cargo test -p apr-cli --lib data_tweeteval label_index` as its runnable backing. That command
  is rejected: `error: unexpected argument 'label_index' found`. `cargo test` accepts only one
  positional TESTNAME. Shipping it would have repeated, inside the contract, the exact defect
  this plan exists to fix in CLAUDE.md — a document naming an invocation that cannot run.
- **Fix:** Verified both working forms empirically, then used
  `cargo test -p apr-cli --lib label_index` (1 passed) in the contract, with a note recording
  why the two-filter form needs `--` (`cargo test -p apr-cli --lib -- data_tweeteval
  label_index`, 9 passed / 1 ignored).
- **Commit:** `4ea35e1ee`

**2. [Rule 2 - Missing functionality] The harness needed a falsification test, not just prose**

- **Found during:** Task 2.
- **Issue:** `must_haves.truths[5]` requires each declared harness be backed by a proptest
  **named in its falsification test**. Naming it only in `property:` prose would not satisfy
  that, and the contract's own machinery for runnable evidence is the
  `test`/`test_harness`/`expected_output` triple.
- **Fix:** Added `FALSIFY-TWEET-EVAL-008` with that triple, and updated `qa_gate.pass_criteria`
  from "all seven" to "all eight". Also added a `falsification:` clause naming the mutation that
  must turn it RED (move `counts[label] += 1` above the bound check), so the test is shown to
  discriminate ordering rather than merely assert a range.
- **Commit:** `4ea35e1ee`

**3. [Rule 3 - Blocking] The PR base branch did not exist on the remote**

- **Found during:** Task 1, STEP 6.
- **Issue:** `origin` is the fork `guyernest/aprender` and had exactly **one** branch, `main`.
  `gsd/phase-1-differentiable-minilm-conformance` had never been pushed, so
  `gh pr create --base` had nothing to target.
- **Fix:** Pushed the Phase 1 branch to origin as a new ref (non-destructive; no force, no
  overwrite) and then pushed the baseline branch. The documented fallback was **not** taken —
  the environment could host the PR, it just needed the base ref to exist.
- **Commit:** n/a (git ref operations)

### Scope additions beyond the declared `<files>` list

**4. `crates/apr-cli/src/commands/data_tweeteval.rs` modified in Task 2.** Task 2's frontmatter
lists only the contract, Makefile and CLAUDE.md, but its action text explicitly says "Add that
proptest if it does not already exist." No `label_index` test existed, so one was added to the
existing `#[cfg(test)] mod tests`. The proptest asserts an **ordering** property, not just a
range: `load_split` does `counts[label] += 1` immediately after resolving `label_text`, so if
the bound check did not strictly precede the increment, an out-of-range label would panic with
an index-out-of-bounds instead of returning a typed error. proptest treats a panic as a failure,
so the test distinguishes "rejected properly" from "crashed". Bound 4, identical to
KANI-TWEET-EVAL-001.

## Measurement Corrections (CLAUDE.md Verification Discipline)

Three checks in this plan reported the wrong answer for mechanical reasons, and each was caught
only by asking how the result was produced. Recording them because later plans run the same
assertions.

1. **`git status --porcelain` is rewritten by the rtk hook.** On a clean path it prints the
   literal string `ok`, not empty output. Task 1's acceptance criterion is
   `[ ! -s /tmp/d06-status.log ]` — under the hook that check **can never pass**, and it briefly
   reported the dependency crates as DIRTY when they were clean. Every porcelain-emptiness
   assertion in this plan was re-run through **`rtk proxy git status --porcelain`**, which
   yields true raw output (verified against both a known-clean and a known-dirty path before
   being relied on). **Plans 02-02..02-09 must use `rtk proxy` for any git-porcelain emptiness
   check.**
2. **`wc -l < file` returns 0 in this shell**, while `wc -l file` returns the real count. This
   made a 12-line digest file look empty and a 569-line log look like 1 line. Counting was moved
   to `awk 'END{print NR}'`.
3. **rtk also summarizes `make` output.** The first `make contract-validate` log showed 16
   "Contract is valid." lines for 43 contracts. Re-running under `rtk proxy make` gave the true
   43/43. The `rc=0` was correct throughout, but the *evidence* was truncated — a log that is
   silently abridged is a poor thing to cite.

A fourth artifact: a `while read` loop lost `PATH` in its command substitution, so `shasum` and
`awk` were "command not found" and every blob comparison reported FAIL with an empty `got=`.
That was a broken mechanism, not a mismatch. Re-run from a script with absolute tool paths, all
twelve matched.

## Known Red, Pre-Existing, NOT Caused By This Plan

- **`cargo clippy -- -D warnings` without `--no-deps`** is red on this tree. All 22 errors are
  in `crates/aprender-compute` and `crates/aprender-zram-core`; both are byte-identical to HEAD
  (porcelain empty over them), so the diagnostics necessarily pre-date this plan. `--no-deps` is
  clean for both baseline crates, before and after.
- **`cargo clippy -p apr-cli --no-deps --all-targets`** is red from `json!`-macro expansions
  tripping the disallowed-`unwrap` lint in four unmodified `crates/apr-cli/tests/
  falsification_crux_*.rs` files plus `nf4_classifier.rs`. **Zero diagnostics reference
  `data_tweeteval.rs`.**
- **`cargo check --workspace --all-targets`** exits 101 solely from `aprender-profile`'s
  deliberate `compile_error!("renacer requires Linux (ptrace syscall tracing)")` on this Darwin
  host (pre-measured by the orchestrator).

## Repo-Wide Gaps Surfaced, Not Fixed

**1. No Kani harness in this repository has ever been executed.** `cargo-kani` is not installed
and there are zero `#[kani::proof]` harnesses anywhere in `crates/`. This is not specific to
this plan's contract — **Phase 1's `setfit-encoder-conformance-v1.yaml` declares four harnesses
(KANI-SETFIT-ENC-001..004) with the same gap and no such disclaimer**. KANI-TWEET-EVAL-001's
`property:` prose now states in words that it is DECLARED and NOT YET EXECUTED and names its
runnable stand-in, and the `qa_gate` says a harness whose named proptest is not green is a
FAILED gate rather than a pending one. Installing and wiring Kani is a toolchain addition
outside this phase.

**2. `make contract-audit` is already red at HEAD and was left red.** `pv audit` reports
BIND-001 for this contract's `official_f_avg` equation — and for **all ten** of Phase 1's setfit
equations, which have zero entries in `contracts/aprender/binding.yaml`. Since
`setfit-encoder-conformance-v1.yaml` was the last `$(CONTRACTS)` entry, `contract-audit` already
exited non-zero before this plan; appending the tweet-eval contract does not newly break it.
`contract-audit` is reachable only from `contract-check`, which **no tier calls** — the
Makefile's own comment at line 239 records this. Adding a single binding entry would not turn
the target green while ten siblings remain unbound, and a *wrong* binding claim is worse than an
absent one, so this is surfaced rather than papered over.

## Evidence

All statuses captured directly, never read through a pipe (CLAUDE.md rule 1).

| Check | Result |
|---|---|
| `shasum -a 256 -c /tmp/d06-baseline.sha256` | exit 0, 12/12 OK |
| committed blob digests via `git cat-file blob HEAD:<path>` | 12/12 match pre-staging record |
| `rtk proxy git status --porcelain` over the twelve | empty |
| `git rev-list --count gsd/phase-1-...\..HEAD` at baseline | 1 |
| `git diff --name-only gsd/phase-1-...\...<baseline>` | exactly 12 files |
| `gh pr view 1 --json files \| length` | 12 |
| `git check-ignore -v docs/examples/tweet-eval-stance.md` | exit 1 (not ignored, CB-510) |
| `.serena/` after the commit | still untracked |
| `cargo fmt -p apr-cli --check` / `-p aprender-train --check` | exit 0 / exit 0 (no write run) |
| `cargo clippy -p apr-cli --no-deps -- -D warnings` | exit 0 (before and after) |
| `cargo clippy -p aprender-train --no-deps -- -D warnings` | exit 0 |
| `cargo test -p apr-cli --lib data_tweeteval` | exit 0 — 9 passed, 1 ignored |
| `cargo test -p apr-cli --lib label_index` | exit 0 — 1 passed, `test result: ok` |
| `cargo test -p apr-cli --lib eval` | exit 0 — 207 passed |
| `cargo test -p aprender-train --lib classification` | exit 0 — 176 passed |
| `pv validate contracts/tweet-eval-stance-benchmark-v1.yaml` **before** | exit 1 — PROVABILITY-001 ×2 |
| `pv validate contracts/tweet-eval-stance-benchmark-v1.yaml` **after** | exit 0, 0 errors 0 warnings |
| `pv status` | v1.1.0, 5 obligations, 8 falsification tests, 1 kani harness |
| `pv diff /tmp/tweet-eval-old.yaml <new>` | "Suggested bump: minor" → applied 1.1.0 |
| `make contract-validate` | exit 0, 43/43 valid, tweet-eval reached |
| falsification block diff old→new | 0 removed/changed lines, +FALSIFY-TWEET-EVAL-008 |
| `pv diff <contract> HEAD~3` (the CLAUDE.md form) | exit 1 — "Failed to read contract file" |
| `grep -c "pv diff contracts/apr-mcp-server-v1.yaml HEAD~3" CLAUDE.md` | 0 |

## Threat Model Dispositions

| Threat | Disposition |
|---|---|
| T-02-01 tampering with dataset ingestion | mitigated — OBLIG-TWEET-EVAL-COUNTS + OBLIG-TWEET-EVAL-HASH-FROM-PARSED-BYTES now pv-checked obligations |
| T-02-02 spoofed provenance | mitigated — FALSIFY-TWEET-EVAL-006 verbatim + OBLIG-TWEET-EVAL-REVISION-HONESTY added |
| T-02-03 tweet-text licensing | mitigated — no tweet text committed; JSONL is user output |
| T-02-33 baseline not byte-identical to what was tested | mitigated — pre-staging digest + committed-blob digest + empty porcelain; no `cargo fmt` write |
| T-02-34 unearned verification claims | mitigated — declared-not-executed stated in-contract, backed by FALSIFY-TWEET-EVAL-008 |
| T-02-SC package tampering | n/a — zero packages installed |

## Tools Used

`pv` for every contract operation (validate, diff, status, audit) — no bash/yq/python
workaround. `pmat query` for locating the `ObligationType` enum. `rtk proxy` for raw git output.
Targeted `Read` on files whose exact line numbers the plan cited.

## Self-Check: PASSED

- `contracts/tweet-eval-stance-benchmark-v1.yaml` — FOUND
- `Makefile` — FOUND, contains `contracts/tweet-eval-stance-benchmark-v1.yaml`
- `CLAUDE.md` — FOUND, contains `git show`
- `crates/apr-cli/src/commands/data_tweeteval.rs` — FOUND, contains `revision_verified`
- commit `7eb1a67faea0b913e7e42eb2d369daf123a49810` — FOUND
- commit `4ea35e1ee40aa19fef9905090ac0f96adb612acf` — FOUND
- PR #1 — OPEN, base `gsd/phase-1-differentiable-minilm-conformance`, 12 files
