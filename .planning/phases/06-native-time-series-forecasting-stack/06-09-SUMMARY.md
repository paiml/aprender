---
phase: 06-native-time-series-forecasting-stack
plan: 09
subsystem: build
tags: [makefile, contracts, binding, pv, readme, claude-md, drift-gate, ci, deferred-items, closing-sweep]

requires:
  - phase: 06-02
    provides: "the monorepo_invariants binary register (allowed_bins / deployment_unit_bins) this plan's landed delta extends with FALSIFY-MONO-011"
  - phase: 06-03
    provides: "prophet-parity-v1.yaml and the fit/predict ladder its 12 binding rows name"
  - phase: 06-04
    provides: "neuralprophet-parity-v1.yaml, np.rs and the `neuralprophet` arm its 9 binding rows name"
  - phase: 06-05
    provides: "chronos-bolt-parity-v1.yaml, the arch-keyed provisional bar, and the bare-token `D-13 decision:` line D-ITEM-06-08 quotes"
  - phase: 06-06
    provides: "forecast-tool-boundary-v1.yaml and the router pool whose gate this plan corrects"
  - phase: 06-07
    provides: "aprender-mcp-chronos, named for the first time in CLAUDE.md's Realizar-first table by this plan"
  - phase: 06-08
    provides: "the bare-token `CI decision:` line, 06-ci-chronos-step.patch, 06-EVIDENCE.md, and the five open items this plan closes or carries"
provides:
  - "Makefile: $(CONTRACTS) +4, PHASE6_CONTRACTS, and `contract-audit-phase6` as a BLOCKING tier3 gate that does what no earlier phase audit does — RESOLVES every Phase 6 binding row to a definition site in the file its module_path names"
  - "contracts/aprender/binding.yaml: 55 rows, one per Phase 6 equation, every `signature:` extracted from source at authoring time"
  - "README.md + CLAUDE.md: re-derived counts (86 crates, 1790 contracts) and the D-07 Realizar-first exception row/paragraph — both CI-wired drift gates green for the first time this phase"
  - "deferred-items.md: the phase close-out register D-ITEM-06-01..09, including D-ITEM-06-08 (the D-13 memory-clause override) and D-ITEM-06-03 (the D-18 CI gap under its three names)"
  - "The uncommitted working-tree delta LANDED with per-file provenance, so 06-EVIDENCE.md's numbers are now reproducible from real commits"
  - "https://github.com/paiml/aprender/issues/3034 — the SmoothL1Loss core ticket"
affects: [phase-verification, ship-gates, milestone-close]

actuals:
  tokens: 30283
  tasks: 3
  commits: 5
plan_head_before: 18758190ba853665e39aec9b02d0c362e83f4aa7
# MEASURED, not narrated (#3968). `git rev-list --count 18758190b..HEAD` = 5 at
# SUMMARY-write time (1d170f383, 1acf5ead1, 68ff7a3c2, f291cfd34, dbcdb6ea1). The SUMMARY
# commit and the STATE/ROADMAP commit that follow make it 7 at the boundary this line is
# committed on; the BOUNDARY is named because a count of its own commit cannot be measured
# before it exists. Re-derive with `git rev-list --count 18758190b..<the docs(06-09) SUMMARY commit>`.
# `tokens` is estimateTokens scale: 121 133 chars over `git diff 18758190b..HEAD` / 4.
# The plan estimated 60 000 against 30 283 actual — a 2.0x over-estimate, and its
# `confidence: low` was right about the uncertainty and wrong about the direction, exactly
# as 06-08's was. Two consecutive 2-3.5x over-estimates on this phase's closing plans is a
# pattern worth carrying into the next estimate, not an accident.

tech-stack:
  added: []
  patterns:
    - "A binding registry proves COMPLETENESS, never LINKAGE: `pv audit` matches contract filename + equation and trusts `status`, so a gate that wants linkage must resolve `function` to a definition site in the file `module_path` names — file-scoped, not name-scoped"
    - "Generate binding rows, do not transcribe them: the generator reads each `signature:` off the source and exits non-zero on an unresolvable row, so an unresolvable row cannot reach the file"
    - "A deferral entry names its SOURCE line, so the next milestone starts from evidence rather than from a claim"
    - "Investigate provenance BEFORE committing found work: attribute every hunk to the plan that produced it, verify it contradicts no landed measurement, and put the attribution in the commit message"
    - "One obligation carrying three names (a deferred item, a review finding, a ledger entry) is cross-referenced rather than tracked three times"

key-files:
  created:
    - .planning/phases/06-native-time-series-forecasting-stack/06-09-SUMMARY.md
  modified:
    - Makefile
    - contracts/aprender/binding.yaml
    - contracts/forecast-tool-boundary-v1.yaml
    - README.md
    - CLAUDE.md
    - crates/aprender-serve/CLAUDE.md
    - crates/aprender-forecast/examples/mase_rolling_origin.rs
    - crates/aprender-forecast/src/forecast.rs
    - crates/aprender-forecast/src/types.rs
    - crates/aprender-forecast/src/np.rs
    - crates/aprender-forecast/src/prophet.rs
    - crates/aprender-forecast/src/chronos.rs
    - crates/aprender-forecast/src/safetensors.rs
    - crates/aprender-forecast/src/test_support.rs
    - crates/aprender-forecast/build.rs
    - crates/aprender-forecast/README.md
    - crates/aprender-mcp-forecast/src/lib.rs
    - crates/aprender-mcp-forecast/src/main.rs
    - crates/aprender-core/tests/monorepo_invariants.rs
    - .planning/phases/06-native-time-series-forecasting-stack/deferred-items.md

key-decisions:
  - "measure-x86-first APPLIED means NOT editing ci.yml: `.github/` is provably untouched in both the working tree and 18758190b..HEAD, the patch stays preserved and re-verified with `git apply --check`, and D-18 clause 2 becomes a NAMED CI gap carrying three cross-referenced names rather than a silent drop"
  - "The plan's proposed binding row `counted_skip_when_unarmed -> build_rs_cfg_emission` was NOT taken: no such function exists, so it would have tripped this phase's own RESOLVE- gate on its first run. Bound to `aprender_forecast::build` / `main` (build.rs), which IS the mechanism"
  - "`verify_source_functions` was READ and rejected for this scope on three measured grounds, not assumed unsuitable: it matches the bare lowercased name across all 77 crates (so a wrong-FILE row resolves), it has no caller, and it cannot express a justfile row"
  - "The D-07 row/paragraph was MERGED with the earlier form already in the working tree rather than duplicated or discarded: the single F8 row names aprender-mcp-chronos, which neither prior row did"
  - "FALSIFY-BOUNDARY-011 was corrected but NOT promoted: the recipe exists and passes at 5.162x on aarch64, so the 'does not exist' wording was false, but it stays out of pass_criteria because CI is X64/debug and one arch's number is not a claim the guarantee can make everywhere"

requirements-completed: [SC5]

coverage:
  - deliverable: "The four Phase 6 contracts are validated by the Makefile loop and audited as a blocking tier3 gate (D-15)"
    verification:
      - kind: command
        ref: "make contract-validate"
        status: pass
      - kind: command
        ref: "make contract-audit-phase6"
        status: pass
    human_judgment: false
  - deliverable: "Every Phase 6 equation has a binding row that names REAL code — 55 rows resolved to definition sites, proven by an observed negative control (REVIEW-06-U3)"
    verification:
      - kind: command
        ref: "make contract-audit-phase6 (resolved 55 Phase 6 binding rows; RESOLVE- lines = 0)"
        status: pass
      - kind: command
        ref: "negative control: function -> this_function_does_not_exist => rc=2 with one RESOLVE- line, reverted => rc=0"
        status: pass
    human_judgment: false
  - deliverable: "README counts re-derived and both CI-wired drift gates green"
    verification:
      - kind: test
        ref: "crates/aprender-core/tests/readme_contract.rs (15 passed, 0 failed)"
        status: pass
      - kind: test
        ref: "crates/aprender-core/tests/monorepo_invariants.rs (11 passed, 0 failed)"
        status: pass
    human_judgment: false
  - deliverable: "The D-07 Realizar-first exception row and paragraph, with every cited path present"
    verification:
      - kind: test
        ref: "crates/aprender-core/tests/readme_contract.rs#FALSIFY-DOCS-CLAUDE-001"
        status: pass
    human_judgment: false
  - deliverable: "The closing sweep on the three new crates: clippy -D warnings, fmt, workspace check, CI's nextest lib leg with counted skips"
    verification:
      - kind: command
        ref: "cargo clippy -p aprender-forecast -p aprender-mcp-forecast -p aprender-mcp-chronos --all-targets --no-deps -- -D warnings"
        status: pass
      - kind: command
        ref: "cargo fmt --all -- --check"
        status: pass
      - kind: command
        ref: "cargo check --workspace --exclude aprender-profile"
        status: pass
      - kind: command
        ref: "cargo nextest run --profile ci -p aprender-forecast -p aprender-mcp-forecast -p aprender-mcp-chronos --lib (119 passed, 11 skipped)"
        status: pass
    human_judgment: false
  - deliverable: "The 06-08 CI decision applied exactly, and every deferral recorded with its source"
    verification:
      - kind: command
        ref: "git diff --name-only 18758190b..HEAD -- .github and git diff --name-only HEAD -- .github, both EMPTY under measure-x86-first"
        status: pass
      - kind: command
        ref: "git apply --check .planning/phases/06-native-time-series-forecasting-stack/06-ci-chronos-step.patch"
        status: pass
      - kind: command
        ref: "deferred-items.md carries D-ITEM-06-01..09, D-13 AMENDED, the amend-memory-clause token, and every named string"
        status: pass
    human_judgment: false
  - deliverable: "The pre-existing uncommitted working-tree delta landed with per-file provenance"
    verification:
      - kind: command
        ref: "cargo nextest run --profile ci -p aprender-forecast -p aprender-mcp-forecast -p aprender-mcp-chronos --lib on the delta (119 passed, 11 skipped)"
        status: pass
    human_judgment: true
    rationale: "The TESTS on the delta are automated and green, but the ATTRIBUTION of each hunk to the plan that produced it is a reading of intent. A human should confirm the per-file attribution in commits 1d170f383 and 1acf5ead1 matches what they remember writing — in particular the root CLAUDE.md deletions that were DROPPED rather than moved (CI/CD list, Modules, pmat Cross-Project Search / Output Formats / Quick Reference, the SSC snapshot)."

duration: 25 min
completed: 2026-09-06
status: complete
---

# Phase 06 Plan 09: Closing Sweep — Contracts Wired, Counts Re-Derived, Deferrals Recorded Summary

**The four Phase 6 contracts are now validated by the Makefile, bound by 55 registry rows, and — for the first time in this repo — RESOLVED to real definition sites by a gate that was observed failing under an invented function name; both CI-wired drift gates are green; and the working-tree delta that made 06-EVIDENCE.md irreproducible has landed with per-file provenance.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-09-06T23:23:01Z
- **Completed:** 2026-09-06T23:48:24Z
- **Tasks:** 3 (plus one investigate-then-commit pre-step directed by the human)
- **Commits:** 5 (measured: `git rev-list --count 18758190b..HEAD`)
- **Files modified:** 20

## Accomplishments

### Pre-step (human-directed): the uncommitted delta, investigated then landed

18 paths were dirty at plan start and predated both 06-07 and 06-08. The human's
instruction was **investigate provenance, THEN commit** — not commit blind. Every hunk
was read before anything was staged, attributed to the plan that plausibly produced it,
and the attribution written into the commit message. Two commits:

- **`1d170f383`** — 13 source/contract/test files. A review-hardening pass over 06-03..06-07,
  concentrated in exactly the crates 06-08 measured: D-11 cost bounds (`MAX_SPAN_DAYS` — ten
  points a millennium apart bought a 3.6M-row imputed daily grid; `MAX_HOLIDAY_WINDOW` /
  `_COLUMNS` / `_DATES` — two `i64` fields were an unbounded design-column multiplier),
  cross-model option refusals (a well-formed key aimed at the wrong arm was silently
  DROPPED), a finite-`cap` check, the `normal_quantile` tail clamp (the widest accepted
  `interval_width` returned NaN, which serde writes as JSON `null`), four safetensors panics
  on input the loader documents itself as refusing, the `{lvl:.1}` quantile-key collapse (the
  surviving "0.9" series was q95 mislabelled as q90), the `MAX_POOL` ceiling
  (`--pool 18446744073709551615` panicked in `Vec::with_capacity`), a per-process contract
  parse (~87 re-parses of 30–50 KB YAML for Prophet alone), and **FALSIFY-MONO-011** — the
  deployment-unit register's defining property `publish = false` lived only in prose, so
  flipping one of those crates publishable would have shipped a non-`apr` binary to
  crates.io with the gate fully green.
- **`1acf5ead1`** — CLAUDE.md + crates/aprender-serve/CLAUDE.md. Two distinct changes: the
  Phase 6 D-03/D-07 Realizar-first material, and a context-slimming pass that MOVED
  "Realizar Inference Tracing" and "FFN Gate+Up Kernel Fusion" verbatim into the crate-local
  CLAUDE.md (which auto-loads only under that crate) and DROPPED five stale root sections,
  including one self-labelled "snapshot 2026-03-22 — STALE".

Nothing contradicted a measurement a landed SUMMARY claims. Verified BEFORE committing:
nextest 119 passed / 11 skipped, `monorepo_invariants` rc=0, and `readme_contract`
14 passed / 1 failed — the single failure being the stale crate count, which Task 2 then
fixed. `06-EVIDENCE.md`'s numbers are now reproducible from real commits rather than from
`adc8a560a` plus the source-only sha256 `bd42a46dda0f0c…`.

The three `.pv/` index artifacts were deliberately NOT committed: they are `pv`'s local
cache, rewritten by any read-only `pv validate`, and committing one produces a spurious
diff for the next author (D-ITEM-06-06-a).

### Task 1 — the four contracts wired, 55 equations bound, and the gate that resolves them

`$(CONTRACTS)` +4, so `make contract-validate` reaches them; `PHASE6_CONTRACTS`;
`contract-audit-phase6` mirrored line-for-line from `contract-audit-phase5` (the `set +e`,
the `status=$?` on its own line, the `audited` counter, the `Total equations:` presence
check, the BIND- line count, the empty-list guard) and wired into tier3 five lines after
the Phase 5 call.

**Four entries where Phases 4 and 5 have one.** Not a departure from Ph1 D-23, which counts
contracts EDITED (zero here). Three ports with three different oracles plus one boundary
shared by two servers; one merged contract would be an equation set no single falsification
run can evaluate.

**The extra step (REVIEW-06-U3).** `pv audit` matches filename + equation and trusts
`status`. `Makefile:2130-2190` records the measurement in the repo's own words: plan 05-10
set `function: this_symbol_does_not_exist_anywhere` and `pv audit` reported rc=0, zero BIND-
lines. So "zero BIND- findings" is REGISTRY completeness, not implementation linkage.
`contract-audit-phase6` therefore also resolves every Phase 6 row to a definition site in
the file its `module_path` names.

**Route taken, and why not `verify_source_functions`** (the plan asked for this to be stated).
It is `pub` at `crates/aprender-contracts/src/build_helper.rs:183-258` and was READ, not
assumed unsuitable. Three disqualifications:

1. It matches the BARE, lowercased name against every `pub fn` found anywhere under
   `crates/` — 77 crates. Phase 6 binds `predict`, `forecast`, `validate`, `new`, `load`,
   `main`; every one exists in some unrelated crate, so a row naming the WRONG FILE would
   resolve. That is precisely the defect REVIEW-06-U3 names.
2. It has no caller — no binary or example invokes it, so wiring it means adding one purely
   to be invoked by a Makefile.
3. It cannot express a `justfile` row, and two Phase 6 equations bind to recipes.

The equivalent was implemented in the target instead, FILE-scoped rather than name-scoped,
which is strictly stronger for this purpose.

**The negative control, verbatim as the plan requires.** Invented function name used:
`this_function_does_not_exist`, on the `single_row_routing_dot8` row (was `proj`).

```
$ make contract-audit-phase6
RESOLVE- chronos-bolt-parity-v1.yaml single_row_routing_dot8 aprender_forecast::bolt::this_function_does_not_exist (no definition site in crates/aprender-forecast/src/bolt.rs)
FAIL: 1 Phase 6 binding row(s) name a function with no definition site
rc=2
```

In that SAME run `pv audit` printed **"No binding gaps found"** for all four contracts
(4 occurrences in the log) — which is the whole point: the audit alone cannot see this.
Post-revert: **rc=0**, `resolved 55 Phase 6 binding rows to definition sites`,
`4 contract(s) audited, zero BIND- findings`.

**Binding rows: 55 = 55.** Derived both sides: 12 (prophet-parity) + 9
(neuralprophet-parity) + 18 (chronos-bolt-parity) + 16 (forecast-tool-boundary) = 55
equations, and `grep -c '^- contract: (…)-v1.yaml'` = 55 rows. All `status: implemented`
(0 rows otherwise). Every `signature:` was EXTRACTED FROM SOURCE by a generator that exits
non-zero on an unresolvable row, so no signature is transcribed from memory. Rows present
for `quantiles_abs_f32_nonaarch64`, `weights_dual_layout_documented`,
`single_row_routing_dot8`, `weights_verified_on_every_gate_run`, `pool_speedup_host_gated`,
`date_shape_exactly_ten_ascii_bytes`; **0 rows** for `weights_transposed_only`.

### Task 2 — counts re-derived, D-07 landed, both drift gates green

Derived on 2026-09-06, pasted from the command output:

| Fact | Command | Value |
|---|---|---|
| Workspace crates (N) | `cargo metadata --no-deps --format-version 1 \| python3 -c "import json,sys;print(len(json.load(sys.stdin)['packages']))"` | **86** |
| Provable contracts (M) | `find contracts -name '*.yaml' \| wc -l` | **1790** |

README's crate row moved 85 → 86; the contracts row already read 1790 and was left alone —
a count that already equals its command needs no edit. CLAUDE.md's derive-the-numbers table
got both Sample cells refreshed with the date inline (the column header carries one date and
these two rows are now later than it).

A **third cell in the same table was wrong** and was corrected, because leaving a stale
explanation beside a freshly derived number is exactly the drift that section warns about.
"Dirs under `crates/`" read 82 with the note "4 excluded, `aprender-contracts-staging` has no
manifest". Re-derived: 92 dirs, and the arithmetic no longer closed. Enumerated by diffing
`ls crates/` against `cargo metadata`'s package set: **5** are `exclude`d (`aprender-present`,
`aprender-test`, `aprender-train-canary`, `aprender-viz-ttop`, `facades`) and **2** have no
manifest (`aprender-contracts-staging`, `deploy`). 92 − 7 = the 85 members under `crates/`,
plus the root facade = 86.

**D-07 was MERGED, not duplicated.** The row and paragraph already existed in an earlier form
inside the delta this plan landed. The two table rows became the ONE RESEARCH-F8 row, which
is strictly more information — it names `crates/aprender-mcp-chronos`, which neither prior row
did, so the second thin server was previously undocumented in that table. The paragraph kept
its substance and took the F8 opening, gaining the exact parameter count (8.65M, was "~9M")
and the stated parity number (9.5e-7 against `chronos-forecasting` 2.3.1). Both paragraphs the
delta added — the SafeTensors carve-out and the transport-only boundary — are kept.

Every backticked path added or moved was `test -e`'d before committing.

**Both drift gates green** (statuses captured directly, never through a pipe):

```
cargo test -p aprender-core --test readme_contract      -> rc=0, 15 passed, 0 failed
cargo test -p aprender-core --test monorepo_invariants  -> rc=0, 11 passed, 0 failed
```

`readme_contract` was 14 passed / 1 failed before this commit, and **FALSIFY-README-005 was
the last red row**: 06-01 logged three, 06-06 closed two, this closed the third.

### Task 3 — CI decision applied, deferrals recorded, ticket filed, sweep closed

**CI decision, read before anything else.** `grep -Ec '^CI decision: (…)$'` on 06-08's
SUMMARY printed exactly **1**. Quoted verbatim as the acceptance criterion requires:

```
CI decision: measure-x86-first
CI decision (verbatim): CI decision: measure-x86-first
CI mount: <none given — 0 such lines>
```

Under `measure-x86-first`, **applying the decision means NOT editing ci.yml**. `.github/` is
provably untouched: `git diff --name-only 18758190b..HEAD -- .github` and
`git diff --name-only HEAD -- .github` are BOTH EMPTY. The patch is preserved and re-verified
(`git apply --check` rc=0). The plan's `files_modified` lists `.github/workflows/ci.yml`
because it predates the decision; the decision supersedes it. Recorded as a deliberate
deviation, not a missed task.

**D-18 clause 2 is a NAMED CI gap under three names for one obligation** — D-ITEM-06-03,
REVIEW-06-02, windows-ledger entry #4 — cross-referenced in the register so closing one
closes all three. Also appended to `.planning/WINDOWS.md` as an `unrun-verify` entry.

**deferred-items.md close-out register, D-ITEM-06-01..09.** Added as a NEW section rather
than by renumbering: the found-in-plan scheme (`D-ITEM-06-03-a` = plan 3's first finding) and
the deferral-topic scheme (`D-ITEM-06-03` = topic 3) collide in text, and the section says so,
because renaming a record other documents already cite is worse than an explained collision.
Every entry names its source. D-ITEM-06-08 carries the literal `D-13 AMENDED` marker, the bare
token `amend-memory-clause` read from 06-05's SUMMARY, both conflicting sentences quoted, the
9.5367e-7 / 1.0e-6 evidence, the ~69 MB residency cost with the explicit note that SC4's
"< 30 MB" is BINARY size and a different quantity, the three places the amendment lives, and
the governance line naming 06-05 Task 2's blocking human checkpoint as the ratifier.

**SmoothL1Loss ticket FILED (no auth gate — `gh auth status` showed a live token):**
**https://github.com/paiml/aprender/issues/3034**, with the `loss.rs:170-195` quote showing
`Tensor::new(&loss_data, …)` ending the graph, the `weighted_huber` workaround location, the
op-composition fix, and the per-loss connectivity gate that matters more than the one fix.

**Closing sweep**, each status on its own line:

```
cargo clippy … --all-targets --no-deps -- -D warnings           -> rc=0
cargo fmt --all -- --check                                       -> rc=0
cargo check --workspace --exclude aprender-profile               -> rc=0
cargo nextest run --profile ci -p aprender-forecast -p aprender-mcp-forecast \
    -p aprender-mcp-chronos --lib                                -> rc=0
    Summary [  38.483s] 119 tests run: 119 passed, 11 skipped
make contract-validate                                           -> rc=0
make contract-audit-phase6                                       -> rc=0
```

**11 skipped is the D-18 counted skip** (unarmed, `CHRONOS_MODEL_DIR` unset). ZERO skipped
would have been the failure — a weight-dependent test passing vacuously.

## SC1/SC4/SC5 host-gated evidence

`.planning/phases/06-native-time-series-forecasting-stack/06-EVIDENCE.md` (06-08) is the
host-gated evidence file for SC1/SC4/SC5: every bar there was MEASURED on aarch64 release
with its command, host, profile, N and log path, and its §7 lists what the numbers do NOT
establish. This plan's contribution is that those numbers are now reproducible from real
commits (`1d170f383`, `1acf5ead1`) rather than from `adc8a560a` plus a recorded delta digest.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] The plan's `counted_skip_when_unarmed` binding row would have failed the plan's own gate**
- **Found during:** Task 1
- **Issue:** The plan text proposed `function: build_rs_cfg_emission`. No such function exists
  anywhere in the tree, so that row would have produced a `RESOLVE-` line on the resolver's
  very first run — the plan's own acceptance criterion requires zero.
- **Fix:** Bound to `aprender_forecast::build` / `main` — `crates/aprender-forecast/build.rs`,
  which IS the mechanism the equation describes (it emits `cargo:rustc-cfg=chronos_weights`
  only when `CHRONOS_MODEL_DIR` holds a `model.safetensors`). The row's `notes` record the
  substitution and why.
- **Files modified:** `contracts/aprender/binding.yaml`
- **Verification:** `make contract-audit-phase6` rc=0, 55 rows resolved
- **Commit:** `68ff7a3c2`

**2. [Rule 1 - Bug] `cargo fmt --all -- --check` was RED on Phase 6's own committed code**
- **Found during:** Task 3 (and observed at plan start, before any edit)
- **Issue:** `crates/aprender-forecast/examples/mase_rolling_origin.rs`, committed by 06-08
  and unmodified in the working tree, had three purely cosmetic rustfmt hunks. SC5 requires
  that gate green.
- **Fix:** Reformatted the file. In scope: it is a Phase 6 artifact, not a pre-existing
  unrelated failure.
- **Files modified:** `crates/aprender-forecast/examples/mase_rolling_origin.rs`
- **Verification:** `cargo fmt --all -- --check` rc=0
- **Commit:** `dbcdb6ea1`

**3. [Rule 1 - Bug] `contracts/forecast-tool-boundary-v1.yaml` made a FALSE statement about the tree**
- **Found during:** Task 3 (carrying 06-08's deferred item 1)
- **Issue:** The contract said twice that `just forecast-pool-ratio` "is NOT implemented" /
  "does not exist". True when written; FALSE from 06-08 onward — the recipe exists and passes
  on aarch64 release (5.150 / 5.162 / 5.146, best 5.162x against a 2.0 bar, 06-EVIDENCE.md §3).
  A contract that misdescribes the tree is a defect.
- **Fix:** Corrected to the measured truth WITHOUT promoting the gate. `FALSIFY-BOUNDARY-011`
  stays out of `pass_criteria`, now for the honest reason: every CI job here is X64 and builds
  debug, so one arch's number is not a claim the guarantee can make everywhere.
- **Files modified:** `contracts/forecast-tool-boundary-v1.yaml`
- **Verification:** `pv validate` rc=0 (0 errors, 0 warnings); `make contract-audit-phase6` rc=0
- **Commit:** `dbcdb6ea1`

**4. [Rule 1 - Bug] CLAUDE.md's "Dirs under `crates/`" sample and its explanation were both wrong**
- **Found during:** Task 2
- **Issue:** 82 dirs with "4 excluded + 1 without a manifest". Re-derived: 92 dirs, and
  92 − 5 ≠ 85.
- **Fix:** Enumerated the real breakdown (5 `exclude`d, 2 without manifests) so the arithmetic
  closes: 92 − 7 = 85 + facade = 86.
- **Files modified:** `CLAUDE.md`
- **Verification:** `cargo test -p aprender-core --test readme_contract` rc=0
- **Commit:** `f291cfd34`

### Deliberate deviations from the plan text

**5. `.github/workflows/ci.yml` was NOT edited, and that IS the CI decision applied.**
The plan's `files_modified` lists it because the plan predates 06-08's recorded
`measure-x86-first`. Under that token the correct action is to touch nothing under
`.github/`, preserve both hunks, and record the gating work as an open item — which is what
D-ITEM-06-03 does, under all three of its names. Not a missed task.

**6. SC5's clippy clause was measured with `--no-deps`, and engagement was PROVEN.**
SC5 reads, verbatim in ROADMAP.md, "`cargo clippy -- -D warnings` on the new crates" — already
scoped to them. Without `--no-deps` the trailing `-- -D warnings` becomes `CLIPPY_ARGS` and
reaches path dependencies: rc=101 with 18 errors, **all in `crates/aprender-compute` and none
in the three crates being linted**. Those 18 are pre-existing, arch-conditional, and tracked
(D-ITEM-06-01-b, D-ITEM-06-02-a, D-ITEM-06-09); fixing them is out of scope and was not done.
Engagement of the `--no-deps` form was proven rather than assumed (CLAUDE.md Verification
rule 2): a `clippy::needless_bool` inserted into `crates/aprender-mcp-chronos/src/lib.rs`
turned it RED (rc=101, "this if-then-else expression returns a bool literal", "could not
compile `aprender-mcp-chronos`"); reverted → rc=0.

**7. The D-07 row/paragraph was merged with the earlier form rather than inserted alongside it.**
Inserting the F8 row while leaving the two prior rows would have put three overlapping
forecast rows in one table. The merge preserves every fact from both and adds
`crates/aprender-mcp-chronos`, which neither prior row named.

**Total deviations:** 4 auto-fixed (1 missing-critical, 3 bugs) + 3 deliberate, all recorded.
**Impact:** the auto-fixes turned two SC5 gates from red to green and removed a false claim
from a shipped contract; the deliberate ones are the human's recorded decisions applied.

## Authentication Gates

None. `gh auth status` showed a live token for `guyernest`, so the SmoothL1Loss ticket was
filed directly (issue #3034) rather than falling back to recording its text.

## Known Stubs

None introduced by this plan.

## Threat Flags

None. The plan's threat register (T-06-16/17/18) is fully mitigated: T-06-17 by
`PHASE6_CONTRACTS` + the blocking audit with its empty-list and `Total equations:` guards,
T-06-18 by counts pasted from the derivation commands and re-derived in the verify, T-06-16
by not editing ci.yml at all under the recorded decision.

## Issues Encountered

**One carried forward, unchanged in substance:** D-18 clause 2 has no CI leg, and
`quantiles_abs_f32_nonaarch64` remains PROVISIONAL and UNMEASURED at 5.0e-6. This is not new
and is not a failure of this plan — it is the explicit consequence of the human's
`measure-x86-first` decision. It is recorded in three cross-referenced places (D-ITEM-06-03,
D-ITEM-06-04, `.planning/WINDOWS.md`), each naming the single command that closes it.

**One pre-existing, restated so it is not mistaken for an SC5 failure:** `make tier1` /
`make tier2` cannot pass on an aarch64 macOS dev box, because of 18 `aprender-compute`
findings plus one in `aprender-core`'s own lib. Out of scope for SC5 by the criterion's own
wording. Three of the 18 are arch-conditional and therefore invisible to CI, which is X64 —
the same shape as CLAUDE.md #2370.

## Next Phase Readiness

Phase 6's last plan is complete and SC5 is met on its own terms. Every gate this plan owns is
green, every contract is validated and bound, both drift gates pass, and every deferral is
recorded with its source. Ready for `/gsd-verify-work 06`.

The one thing a verifier should NOT read as green: `FALSIFY-BOUNDARY-011` and D-18 clause 2
are host-gated and unarmed by decision, and both say so in their own words.

## Self-Check: PASSED

- `.planning/phases/06-native-time-series-forecasting-stack/06-09-SUMMARY.md` — FOUND (this file)
- Commits in `18758190b..HEAD` — 5 FOUND: `1d170f383`, `1acf5ead1`, `68ff7a3c2`, `f291cfd34`, `dbcdb6ea1`
- `make contract-validate` — rc=0
- `make contract-audit-phase6` — rc=0, 55 rows resolved, 0 BIND-, 0 RESOLVE-
- `cargo test -p aprender-core --test readme_contract` — rc=0, 15 passed / 0 failed
- `cargo test -p aprender-core --test monorepo_invariants` — rc=0, 11 passed / 0 failed
- `cargo nextest run --profile ci` on the three crates — rc=0, 119 passed / 11 skipped
- `cargo fmt --all -- --check` — rc=0; `cargo check --workspace --exclude aprender-profile` — rc=0
- `.github` diff under `measure-x86-first` — EMPTY in both the working tree and `18758190b..HEAD`
