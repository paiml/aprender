---
phase: 04-apr-artifact-and-production-parity
plan: 10
subsystem: build-gates
tags: [safe-01, safe-02, ops-01, make, tiers, feature-matrix, contract-binding, cr-01, cr-02, m5, w-6, w-7, d-04-08-a, d-04-09-a, d-04-10-a]
requires:
  - "04-09 (setfit-parity target + the corrected SAFE-02 leg)"
  - "04-12 (setfit_apr_lifecycle integration target)"
  - "04-15 (setfit_cli_lifecycle spawned target and its #[ignore] tiering)"
  - "04-01 (assert_tests_ran, PHASE4_CONTRACTS, contract-audit-phase4)"
provides:
  - "20 named, floored Make targets covering every Phase 4 test surface"
  - "setfit-all-tests aggregate (18 suites, each independently guarded)"
  - "setfit-api-boundary — OPS-01 graph gate with an EXECUTED must-match/must-not-match case table"
  - "assert_tests_absent — the inverse of assert_tests_ran, for gating-by-absence"
  - "setfit-feature-matrix grown from 2 crates to 4, build AND run legs"
  - "tier3 wiring for all five Phase 4 gates"
  - "14 setfit-apr-v1 binding statuses flipped pending -> implemented"
affects:
  - "04-11 (the ci.yml half — must stay command-identical to these recipes)"
tech-stack:
  added: []
  patterns:
    - "one positional filter per cargo test invocation (libtest accepts one)"
    - "cmd > log 2>&1; rc=$? — status never read through a pipe"
    - "assert_tests_ran / assert_tests_absent used in PAIRS, never alone"
    - "two-sided cargo tree negatives with a live-marker control"
key-files:
  created: []
  modified:
    - "Makefile (+433 lines: 20 targets, extended matrix, tier3 wiring, assert_tests_absent)"
    - "contracts/aprender/binding.yaml (14 status flips + provenance notes)"
    - ".planning/phases/04-apr-artifact-and-production-parity/deferred-items.md (D-04-10-A)"
decisions:
  - "Wired NEITHER red-on-arrival leg: no whole-crate aprender-serve test leg (D-04-08-A), no apr-cli --no-default-features check leg (D-04-09-A)"
  - "No whole-crate aprender-train --lib leg either — every train leg is scoped under setfit::, which is disjoint from both known-red modules, so no name-diff apparatus is needed"
  - "backend_identity stays `pending`: its claimed function does not exist on this branch"
  - "Binding statuses live in contracts/aprender/binding.yaml, NOT contracts/setfit-apr-v1.yaml (the plan's files_modified is wrong); the contract file is byte-identical to 04-01's"
  - "aprender-train --all-features MEASURED clean and therefore wired; the other three crates measured red for alsa/wgpu/trueno-viz reasons and are not"
metrics:
  tasks: 2
  commits: 5
  targets_added: 20
  suites_guarded: 18
  falsifications_performed: 9
  duration: "~4h"
  completed: 2026-08-15
---

# Phase 4 Plan 10: Local Make Gates for the Whole Phase 4 Surface — Summary

Twenty named Make targets that compile, RUN and floor every Phase 4 test surface; a SAFE-02
feature matrix spanning four crates and three CPU profiles with build AND run legs; five gates
wired into tier3; and 14 contract binding statuses flipped after each was resolved against
shipped source. Nine guards were shown able to fail, reverted, and re-measured green.

## What shipped

| Commit | What |
| ------ | ---- |
| `c26e74274` | 18 scoped targets + `setfit-api-boundary` + `setfit-all-tests`, one filter per invocation, each floored |
| `41d1e805d` | two suites the all-38-tasks cross-check found uncovered (`setfit-verify-tests`, `setfit-cli-serve-tests`) |
| `af466836e` | SAFE-02 matrix over 4 crates x 3 profiles, tier3 wiring, `assert_tests_absent`, 14 binding flips |
| `af76f1065` | D-04-10-A recorded in `deferred-items.md` |
| (this file) | SUMMARY |

## Target inventory — every floor MEASURED, none guessed

Measured on `5887d301c` with the status captured directly off cargo (`cmd > log 2>&1; rc=$?`),
never through a pipe. Measurement ran under `rtk proxy` so the log holds libtest's raw
`test result:` line — the rtk hook's summarised form does not contain it and `assert_tests_ran`
would read 0 from a hook-rewritten log. Make recipes are not hook-rewritten, so they see the
raw form; this was confirmed by running `make setfit-evaluate-tests` and reading
`target/setfit-evaluate-tests.log` (1 raw `test result:` line, parsed to 14).

| Target | Invocation | passed / failed | floor |
| ------ | ---------- | --------------- | ----- |
| `setfit-apr-tests` | core `--lib setfit::artifact::` | 86 / 0 | 80 |
| `setfit-classify-tests` | core `--lib setfit::classify::` | 50 / 0 | 45 |
| `setfit-bundle-tests` | train `--lib setfit::bundle` | 33 / 0 | 30 |
| `setfit-config-tests` | train `--lib setfit::config` | 40 / 0 | 36 |
| `setfit-evaluate-tests` | train `--lib setfit::evaluate` | 14 / 0 | 12 |
| `setfit-codec-tests` | train `--lib setfit::apr_codec::` | 17 / 0 | 15 |
| `setfit-reload-tests` | train `--lib setfit::apr_reload::` | 17 / 0 | 15 |
| `setfit-lock-tests` | train `--lib setfit::lock` | 36 / 0 | 32 |
| `setfit-verify-tests` | train `--lib setfit::verify` | 18 / 0 | 16 |
| `setfit-lifecycle-tests` | train `--test setfit_apr_lifecycle` | 5 / 0 | 5 |
| `setfit-ui-tests` | train `--test ui` (trybuild) | 1 / 0 | 1 |
| `setfit-cli-train-tests` | cli `--lib setfit_train` | 15 / 0 (1 ign) | 13 |
| `setfit-cli-train-tests` (2nd leg) | `... -- --ignored` | 1 / 0 | 1 |
| `setfit-cli-predict-tests` | cli `--lib predict` | 34 / 0 | 30 |
| `setfit-cli-inspect-tests` | cli `--lib inspect` | 120 / 0 | 110 |
| `setfit-cli-eval-tests` | cli `--lib eval::setfit` | 15 / 0 | 13 |
| `setfit-cli-io-tests` | cli `--lib setfit_io` | 5 / 0 | 5 |
| `setfit-cli-serve-tests` | cli `--lib serve` | 358 / 0 | 330 |
| `setfit-serve-tests` | serve `--lib setfit` (SCOPED) | 10 / 0 | 9 |
| `setfit-parity` | cli `--test setfit_parity` | 20 / 0 (1 ign) | 18 |
| `setfit-serve-smoke` | `... -- --ignored spawned_serve_smoke` | 1 / 0 | 1 |
| `setfit-cli-lifecycle` | `... -- --ignored lifecycle` | 2 / 0 | 2 |
| `setfit-cli-lifecycle` (2nd leg) | `... -- --ignored tooling` | 1 / 0 | 1 |
| `setfit-api-boundary` | `cargo tree` x5 | n/a — explicit count compare | n/a |

`make setfit-<each>` re-run after wiring: **all 18 test targets rc=0**, plus
`setfit-api-boundary` rc=0 and `setfit-feature-matrix` rc=0.

## The all-38-tasks cross-check — RUN, and it found two gaps

The plan makes this an acceptance criterion. It was executed, not asserted.

**Task count first**, because "38 tasks" is itself a claim: `grep -c '^<task '` across all
seventeen `04-*-PLAN.md` files sums to exactly **38** (3+2+3+3+2+3+3+3+3+2+3+1+2+2+2+1+0).

Every `cargo test` invocation inside an `<automated>` block was extracted and de-duplicated —
**53 distinct lines**. Cross-checked against the target list:

- Sub-filters of a covered prefix — `setfit::artifact::determinism`, `::ladder`, `::probe`,
  `setfit::classify::backend`, `::envelope`, `setfit::apr_codec::round_trip`,
  `--test setfit_parity golden` — are all subsumed by their parent-prefix target. Covered.
- **`cargo test -p aprender-train --features setfit --lib setfit::verify` (04-13) — NO TARGET.**
- **`cargo test -p apr-cli --features setfit --lib serve` (04-08) — NO TARGET.**

Both were closed rather than merely recorded (commit `41d1e805d`). 04-13's is the one that
mattered: its acceptance criterion is that a *compile* failure there is the signature of a
missed `verify_tests.rs` call site, and that claim was resting on the unguarded broad
`-p aprender-train --lib setfit::` leg inside `setfit-tests`.

W-6's two suites (`setfit::config`, `setfit::evaluate`) were in the plan's own target list and
are wired; both are prerequisites of `setfit-all-tests`.

**Source assertion, run:** every one of the 18 suite names appears >= 2x in the Makefile (own
target + aggregate prerequisite + PHONY + log filename; measured 8-10 occurrences each).

## M5 — no recipe passes two positional filters

63 `cargo test` lines exist in the Makefile. A pattern-based scan reports **0** candidates.

**Zero hits is also what a dead pattern reports** (CLAUDE.md rule 7), so the pattern ships a
case table and the table was EXECUTED:

| Case | Line | Expect | Got |
| ---- | ---- | ------ | --- |
| MUST-MATCH | `--lib setfit::artifact:: setfit::classify::` | MATCH | MATCH |
| MUST-MATCH | `--lib setfit::config setfit::evaluate` | MATCH | MATCH |
| must-not | `--lib setfit::artifact::` | NOMATCH | NOMATCH |
| must-not | `--lib setfit_train -- --ignored` | NOMATCH | NOMATCH |
| must-not | `--test setfit_parity -- --ignored spawned_serve_smoke` | NOMATCH | NOMATCH |
| must-not | `--test ui` | NOMATCH | NOMATCH |
| must-not | `--test setfit_repro --features setfit in_process` (pre-existing) | NOMATCH | NOMATCH |

7/7. The two MUST-MATCH cases are the exact shape the previous plan draft shipped.

**Pipe-before-status scan:** zero `| tee` and zero `pipe-then-capture-rc` occurrences in any
added recipe. Two pre-existing `| tee` hits remain at Makefile:2376 and :2398 inside `publish`
— that is #2360's own known defect, untouched by this plan and out of its scope.

## Guards shown able to FAIL — 9 mutations, each observed red, reverted, re-measured green

A Make target never proven able to fire is theatre. Each mutation was applied, the target run,
the red observed **with its specific message**, then reverted; `git status --short` was clean
after every revert and the green state re-measured.

| # | Mutation | Target | Result |
| - | -------- | ------ | ------ |
| — | baseline | `setfit-evaluate-tests` | rc=0, 14 passed |
| **A** | floor 12 -> 999 | `setfit-evaluate-tests` | **rc=2** — "reported 14 test(s) passed, expected at least 999" |
| **B** | filter `setfit::evaluate` -> `setfit::evaluete` | `setfit-evaluate-tests` | **rc=2** — cargo exited **0** with "0 passed"; only `assert_tests_ran` turned it red. This IS the CR-02 class, observed |
| **C** | control pattern `apr-cli` -> `apr-cIi` | `setfit-api-boundary` | **rc=2** — "the MUST-MATCH control read 0 ... the four absence legs above were passing for the wrong reason" |
| **D** | first absence leg `aprender-core` -> `apr-cli` | `setfit-api-boundary` | **rc=2** — "FAIL (OPS-01): apr-cli depends on apr-cli (1 node(s))" |
| **E** | self-name check `^$$crate` -> `^ZZ$$crate` | `setfit-api-boundary` | **rc=2** — "the tree ... does not contain aprender-core itself. A tree that resolved nothing satisfies an absence check vacuously" |
| **F** | core OFF leg given `--features setfit` | `setfit-feature-matrix` | **rc=2** — "ran 240 test(s) with the setfit feature OFF, expected 0" |
| **G** | apr-cli delta threshold 40 -> 400 | `setfit-feature-matrix` | **rc=2** — "apr-cli's setfit delta is 55 (13 off, 68 on), expected at least 400" |
| **H1** | apr-cli absence marker `tokenizers` -> `serde` | `setfit-feature-matrix` | **rc=2** — "tokenizers leaked into a DEFAULT apr-cli build" |
| **H2** | serve presence marker `tokenizers` -> `tokenizerZ` | `setfit-feature-matrix` | **rc=2** — "--features setfit did NOT pull tokenizers into aprender-serve; the absence check above is vacuous" |
| — | all reverted, re-measured | matrix + boundary + evaluate | **rc=0**, identical counts (240 / 311 / 13->68 / 10) |

F, G, H1 and H2 are the Task-2 guards and were re-mutated **in their new scope** rather than
inheriting Task 1's transcript (CLAUDE.md rule 4). B and F are a matched pair: B proves the
gate catches a filter that selected nothing, F proves the inverse gate catches a filter that
selected something it should not have.

## SAFE-02 feature matrix — 4 crates x 3 CPU profiles, MEASURED per cell

| crate | (a) `--no-default-features` | (b) ndf + setfit | (c) default + setfit |
| ----- | --------------------------- | ---------------- | -------------------- |
| aprender-core | rc=0 | rc=0 | rc=0 |
| aprender-train | rc=0 | rc=0 | rc=0 |
| apr-cli | **rc=101 NOT WIRED** | **rc=101 NOT WIRED** | rc=0 |
| aprender-serve | rc=0 | rc=0 | rc=0 |

Both apr-cli minimal cells fail with the SAME four errors — `inference`-gated code that is not
`cfg`-gated at `src/commands/explain.rs:231,344`, `src/commands/diff_05_aprt_stage.rs:100`,
`src/lib.rs:63`. Setting `--features setfit` does not help because `setfit` does not imply
`inference`. **D-04-09-A. Not wired.** The green gating leg is
`cargo check -p apr-cli --all-targets` with default features, rc=0.

**RUN legs** (checking is not testing — CR-01):

| leg | feature OFF | feature ON | guard |
| --- | ----------- | ---------- | ----- |
| aprender-core `--lib setfit::` | **0** | 240 | `assert_tests_absent` + `assert_tests_ran` (floor 230) |
| aprender-train `--lib setfit::` | **0** | 311 | same pair (floor 300) |
| apr-cli `--lib setfit` | **13** | 68 | floors 10 / 60 **plus** an explicit `on - off >= 40` delta |
| aprender-serve `--lib setfit` | *(build red)* | 10 | floor 9, profile (c) only |

**`--all-features` was MEASURED for all four crates, not transcribed:**

| crate | rc | first error |
| ----- | -- | ----------- |
| aprender-core | 101 | `alsa-sys` build script (no ALSA headers on macOS) |
| **aprender-train** | **0** | **CLEAN — so it IS wired, as leg (d)** |
| apr-cli | 101 | `entrenar::finetune::wgpu_pipeline::WgpuInstructPipeline` |
| aprender-serve | 101 | `trueno_viz::plots::Histogram::dimensions` (API drift) |

The aprender-train result **contradicts** the older in-file comment claiming no `--all-features`
leg is buildable on a CPU host. It was re-measured and it is. The recipe records that if the
train leg ever goes red for a `cuda`/`wasm` reason, that is NOT a SAFE-02 signal — scope it out
rather than silencing the whole matrix.

**Graph negatives, two-sided.** `aprender-contrastive-data` is NOT usable as the apr-cli marker
(it is already in the DEFAULT tree via `training`, a default feature). Measured: `tokenizers`
is 0 in the default apr-cli tree and 1 with setfit, so it is the marker that discriminates.
Same for aprender-serve (0 -> 1). Both sides asserted, so absence cannot pass on a dead marker.

## The dev-dependency note, as shipped in the recipe

The plan required this comment. **Its premise is false on this tree**, and the recipe says so
rather than transcribing a mitigation for a threat that does not arise:

> THE dev-dependency RULE, recorded here because a reader of this recipe is who needs it
> (04-09-SUMMARY deviation 1). Cargo does not permit an OPTIONAL dev-dependency, so a dev-dep
> on a package that is ALSO a normal dependency is unconditional, and its features UNIFY with
> the normal dependency for any build that includes test targets. Such an entry silently
> weakens the corresponding `cargo test --no-default-features` run leg: the crate under test
> would still be built WITH the feature the leg is trying to prove absent.
> **IT DOES NOT APPLY ON THIS TREE.** 04-09's plan required a `realizar` dev-dep with
> `features = ["setfit"]`; 04-09 MEASURED it unnecessary ... so threat **T-04-61 does not
> arise**. The rule stays written down because the next person reaching for such a dev-dep
> needs it; the apr-cli run leg's weakness above has a different, measured cause.

That different, measured cause is the one W-7 actually needs recorded: apr-cli's `--lib setfit`
filter runs **13** tests with the feature OFF, because apr-cli carries setfit-NAMED tests
(argument parsing, error strings) that are not behind the feature. **The apr-cli run leg
therefore proves the gated surface APPEARS, not that it is ABSENT** — which is why it is
asserted as a delta and why SAFE-02's apr-cli gating evidence is the CHECK leg plus the
`tokenizers` graph negative. Source assertion run: the block contains `dev-dep` (4x),
`no-default-features` (22x), `all-targets` (4x), `T-04-61` (1x).

## Tier placement

`make -n tier3` expands to 612 lines and reaches all six gates:

| line | gate |
| ---- | ---- |
| 159 | `setfit-feature-matrix` (pre-existing; grown 2 -> 4 crates here) |
| 306 | `setfit-all-tests` |
| 469 | `setfit-parity` |
| 482 | `setfit-serve-smoke` |
| 491 | `setfit-cli-lifecycle` |
| 507 | `setfit-api-boundary` |

The two `#[ignore]`d gates are why this matters: an ignored test runs in NO default invocation
— not `cargo test --all`, not tier2, not CI's nextest run. Left unwired they would reintroduce
CR-01 inside the plans that were closing it. tier2 was **not** modified; the fast scoped suites
run under tier3's `setfit-all-tests` alongside the existing `setfit-tests`.

## Contract binding statuses

**The plan's `files_modified` names the wrong file.** Binding statuses live in
`contracts/aprender/binding.yaml`, not `contracts/setfit-apr-v1.yaml`. See Deviation 1.

`contracts/setfit-apr-v1.yaml` is **byte-identical to 04-01's version** (`cmp` against
`git show 488e307d5:...` — identical), so the required `pv diff` is:

```
$ pv diff /tmp/.../setfit-apr-v1-04-01.yaml contracts/setfit-apr-v1.yaml
Contracts are identical.                                        rc=0
$ pv validate contracts/setfit-apr-v1.yaml
0 error(s), 0 warning(s) / Contract is valid.                   rc=0
```

The binding edit is **status-only**, asserted mechanically rather than claimed:

| field | changed lines in the diff |
| ----- | ------------------------- |
| `contract` | 0 |
| `equation` | 0 |
| `module_path` | 0 |
| `function` | 0 |
| `status` | 28 (14 `- pending`, 14 `+ implemented`) |

**Every pair was resolved against shipped source before its status moved**, through the
dogfooded pmat index. That check found a real discrepancy:

| equation | claimed | resolved |
| -------- | ------- | -------- |
| `artifact_storage_map` | `aprender::setfit::artifact::write_setfit_apr` | `crates/aprender-core/src/setfit/artifact.rs` |
| `artifact_doc_schema` | `...::SetFitArtifactDoc` | artifact.rs |
| `nullable_path_allowlist` | `...::first_unallowed_null_path` | artifact.rs |
| `doc_bundle_bijection` | `entrenar::train::setfit::apr_codec::deserialize` | `crates/aprender-train/src/train/setfit/apr_codec.rs` |
| `canonical_tensor_names` | `...::canonical_name_for_hf` | artifact.rs |
| `architecture_derived_tensor_set` | `...::expected_tensor_names` | artifact.rs |
| `artifact_size_bounds` | `...::load_setfit_apr` | artifact.rs |
| `bounded_read` | `...::read_setfit_apr_bytes_bounded` | artifact.rs |
| `probe_policy` | `...::rung8_replay_probes` | artifact.rs |
| `probe_and_parity_tolerances` | `...::within` | artifact.rs |
| `classify_response_schema` | `aprender::setfit::classify::ClassifyResponse` | classify.rs |
| **`backend_identity`** | **`aprender::setfit::classify::backend_identity`** | **DOES NOT EXIST — stays `pending`** |
| `selection_lock_lifecycle` | `entrenar::train::setfit::lock::mint_test_token` | lock.rs |
| `d02_typed_key_amendment` | `...::write_setfit_apr` | artifact.rs |
| `load_validation_ladder` | `...::load_setfit_apr` | artifact.rs |

`backend_identity` was **not** flipped. The identity comes from `ExecutionBackend::identity` in
the encoder; there is no such function in `classify`. Flipping it would record a claim nothing
can check — the exact failure class the scoped audits exist to catch. Correcting the `function`
column is a *semantic* registry edit, not a binding-status flip, so it is out of this plan's
declared scope; the orchestrator has it as an open wave-6 item.

```
$ make contract-audit-phase4                                     rc=0
Total equations: 15 / Bound equations: 15 / Implemented: 14 / Not implemented: 1
[WARN] BIND-004: Equation 'backend_identity' ... is pending implementation
Phase 4 binding audit: 1 contract(s) audited, every equation is bound
```

Zero BIND-001. The one BIND-004 is the honest state.

## Deviations from Plan

### 1. [Rule 1 — the plan's `files_modified` names the wrong file]

The plan declares `contracts/setfit-apr-v1.yaml` and instructs "flip `status: pending` to
bound/implemented in `contracts/setfit-apr-v1.yaml`". **That file contains no `status` field
for any equation.** The binding registry is `contracts/aprender/binding.yaml` — it is what
`$(BINDING)` points to and what `pv audit --binding` reads. The edit was made there.
`contracts/setfit-apr-v1.yaml` was not touched at all and is byte-identical to 04-01's version,
so the required `pv diff` reports "Contracts are identical" — which is the *strongest* possible
form of "only binding-status lines changed". No wave-9 plan contends for `binding.yaml`
(04-10 is the sole plan in the wave), so the file-ownership model is intact.

### 2. [Rule 1 — the plan's dev-dep instruction restates a premise 04-09 already falsified]

The plan requires the matrix recipe to record 04-09's "unconditional parity dev-dependency"
in that plan's own wording. **04-09 added no such dev-dependency** — it measured one
unnecessary, and the orchestrator recorded "T-04-61 does not arise". Transcribing the plan's
paragraph would have shipped a comment describing a manifest that does not exist. The recipe
instead records the RULE (which is genuinely useful), states plainly that it does not apply
here, and gives the measured reason the apr-cli run leg *is* weak (13 ungated setfit-named
tests). See "The dev-dependency note, as shipped" above.

### 3. [Rule 1 — the plan names a red leg as SAFE-02's gating evidence]

The plan (via 04-09's) points at `cargo check -p apr-cli --no-default-features`. Measured
rc=101, and rc=101 with the pre-Phase-4 manifest restored. **Not wired.** Substituted with
`cargo check -p apr-cli --all-targets` (default features), rc=0. Both apr-cli minimal cells
(a) and (b) are affected, not just (a) — measured, and both produce the identical four errors.

### 4. [Rule 2 — two suites the plan's target list omitted]

The all-38-tasks cross-check is an acceptance criterion that says "record any suite with no
target in the SUMMARY". Two were found. Recording them and leaving them uncovered would have
left the plan's own must-have ("every suite a Phase 4 plan cites as its own evidence has a
guarded target") false, so both were closed: `setfit-verify-tests` (04-13) and
`setfit-cli-serve-tests` (04-08).

### 5. [Rule 3 — `aprender-serve`'s minimal TEST build is red]

New finding, filed as **D-04-10-A** in `deferred-items.md`. The CHECK cells at profiles (a) and
(b) are rc=0; the TEST target at the same profiles is rc=101 because `#[cfg(test)]` code
imports `crate::gguf::OwnedQuantizedModelCached`, `crate::gpu` and `crate::api` GPU types
unconditionally. Pre-existing and unrelated to setfit — identical first errors with the feature
off and on. The matrix therefore runs aprender-serve at profile (c) only and says so in the
recipe. **Only findable from a run leg**; a build-only matrix reports this crate clean.

### 6. [Deviation, argued — no whole-crate `aprender-train --lib` leg]

The success criteria contemplate one with a known-red NAME diff. None was wired. Every train
leg is scoped under `setfit::`, and the 24 known-red tests are `gpu::` (21) and
`prune::snapshot_tests` (3) — **disjoint from `setfit::`**. Each scoped train leg measured
0 failed. The Makefile records this so a future reader does not add a broad leg without the
name-diff apparatus. This is strictly safer than wiring a leg that needs a baseline file to
stay correct.

## Known gaps — stated, not papered over

**`bashrs` is NOT installed on this host, so `bashrs make lint Makefile` was NOT run and must
not be claimed as passing.** `command -v bashrs` -> not found; `~/.cargo/bin` contains no
`bashrs` (checked for the shadowing case in CLAUDE.md rule 8 — it is genuinely absent, not
shadowed). Installing it is a package-manager install, outside this plan's declared files and
outside the executor's auto-fix scope. The Makefile's own `lint-scripts` target already guards
with `command -v bashrs`. Substitutes actually run, and what each covers:

| substitute | covers |
| ---------- | ------ |
| `make -n <target>` for every new target, and `rtk proxy make -n tier3` (612 lines) | recipe parses, variables expand, sub-makes are reached |
| the 7-case must-match/must-not-match table above | the M5 two-positional-filter scan is live |
| pipe-before-status scan (`\| *tee` and pipe-then-`rc=$$?`) | CLAUDE.md rule 1 in every added recipe |
| 9 executed mutations | every new guard can actually fire |

**OPS-01 and OPS-02 remain UNMET behind F-10** (a Phase 5 item). `setfit-api-boundary` closes
only the *mechanical* half of OPS-01 — the graph property that no library crate depends on
apr-cli. It does not, and must not be read to, assert the end-to-end
train->save->load->embed->classify->inspect chain works; it does not on this host. Nothing in
`REQUIREMENTS.md` was flipped.

**04-11 must stay command-identical.** Every recipe line here is the exact command the ci.yml
half needs. The two forbidden legs (`-p aprender-serve --lib` whole-crate,
`-p apr-cli --no-default-features`) must not appear there either.

## A new measurement hazard, worth recording

`make -n tier3 > log 2>&1` produced a **51-line log with ZERO matches** for the new gates. The
gates were wired correctly; the log was wrong. The rtk hook rewrote `make` and truncated its
output **before** the redirect, so the log file itself ended in
`... (562 lines truncated)`. `rtk proxy make -n tier3` gives the real 612 lines and all six
matches.

**A redirect does not protect you from the hook.** This is a close cousin of CLAUDE.md rule 1
(reading `$?` through a pipe) and it bit twice in this plan — the same mechanism is why every
count measurement above ran under `rtk proxy`, after the first `cargo test` log came back with
zero `test result:` lines. Any assertion made against a hook-visible command's redirected
output is suspect; route it through `rtk proxy`.

## Threat Flags

None. This plan adds no runtime surface — no endpoints, no auth paths, no file access patterns,
no schema changes. The `binding.yaml` edit is metadata, and the contract file is untouched.

## Self-Check: PASSED

Files:
- FOUND: `Makefile`
- FOUND: `contracts/aprender/binding.yaml`
- FOUND: `.planning/phases/04-apr-artifact-and-production-parity/deferred-items.md`
- FOUND: `.planning/phases/04-apr-artifact-and-production-parity/04-10-SUMMARY.md`

Commits: `c26e74274`, `41d1e805d`, `af466836e`, `af76f1065` — all present in `git log`.

Final state: `git status --short` clean; `make setfit-feature-matrix` rc=0 with counts
identical to the pre-mutation baseline (240 / 311 / 13->68 / 10); `make setfit-api-boundary`
rc=0 with its five-leg case table intact; `make contract-audit-phase4` rc=0.
