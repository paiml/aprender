---
phase: 04-apr-artifact-and-production-parity
plan: 19
subsystem: testing
tags: [contracts, binding-registry, pv, setfit, backend-identity, falsification, tdd]

# Dependency graph
requires:
  - phase: 04-apr-artifact-and-production-parity
    provides: "plan 04-04's ExecutionBackend + classify path; plan 04-10's verified binding flips, which found this discrepancy and deliberately left it open"
provides:
  - "contracts/aprender/binding.yaml backend_identity now names the shipped symbol: aprender::setfit::encoder / ExecutionBackend::identity, signature recorded, status implemented"
  - "a compile-witnessed resolution guard (6 tests in setfit::classify::backend) that makes a binding row falsifiable rather than merely asserted"
  - "make contract-audit-phase4 at 15/15 implemented with zero BIND findings"
  - "a measured, on-registry record that pv verify-bindings is blind in this monorepo layout (0/124)"
affects: [phase-05, contract-registry-maintenance, setfit-serving]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "typed function-item coercion as a rustc-witnessed binding check"
    - "include_str! of the binding registry from a test, so a missing registry is a compile error"
    - "row-parser returning Option so non-vacuity is its own executed assertion"

key-files:
  created: []
  modified:
    - crates/aprender-core/src/setfit/classify.rs
    - contracts/aprender/binding.yaml

key-decisions:
  - "function column is TYPE-QUALIFIED (ExecutionBackend::identity), because identity is an inherent method and a bare name would be as unresolvable as the ghost it replaced"
  - "pv verify-bindings is run and recorded but is NOT the gate — it reports 0/124 and ghosts symbols that demonstrably exist"
  - "the pv diff rejection of the registry is recorded verbatim with a control run, not worked around with a shell script"
  - "the measured blindness of pv verify-bindings is recorded ON the registry row rather than in deferred-items.md, which a wave-1 sibling owns"

patterns-established:
  - "Check before flip: the guard was committed RED (daf6ce868) before the status moved (b94c5a2d4), so the commit order itself is the ordering proof"
  - "A falsification probe must be compile-neutral when the assertion under test is textual, or the probe tests the compiler rather than the guard"

requirements-completed: []  # OPS-04, OPS-06, SAFE-01 remain UNCHECKED — see Open Items

# Metrics
duration: 47min
completed: 2026-08-16
---

# Phase 04 Plan 19: backend_identity Binding Correction Summary

**The last Phase 4 binding row that named a nonexistent symbol now names `ExecutionBackend::identity`, and the status flip is earned by a guard rustc witnesses for the symbol and `include_str!` witnesses for the row — `pv audit` goes 14/15 with a BIND-004 warning to 15/15 with none.**

## Performance

- **Duration:** ~47 min
- **Started:** 2026-08-16T17:50Z (approx, first baseline measurement)
- **Completed:** 2026-08-16T18:37Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Corrected the `backend_identity` row: `module_path: aprender::setfit::encoder`, `function: ExecutionBackend::identity`, `signature: 'fn identity(&self) -> String'`, `status: implemented`. The ghost path `aprender::setfit::classify::backend_identity` is now absent from `contracts/` entirely.
- Built the check **before** the flip. Six tests in `setfit::classify::backend`, committed RED, then turned green by the row edit.
- Closed the last BIND-004 in Phase 4: `make contract-audit-phase4` reports 15/15 bound and implemented, zero `BIND-` lines.
- Moved both stale header claims in the same commit as the row, so the registry no longer contradicts the rows it introduces.
- Recorded, on the registry itself, that `pv verify-bindings` is blind in this layout — with the arithmetic that proves it.

## Task Commits

1. **Task 1: the resolution guard, RED before the flip** — `daf6ce868` (test)
2. **Task 2: correct the row and the two stale header claims** — `b94c5a2d4` (fix)

## Files Created/Modified

- `crates/aprender-core/src/setfit/classify.rs` (+209) — extends the existing `#[cfg(test)] mod backend` with a `BINDING` `include_str!` constant, a `ROW_SIGNATURE` constant, an `Option`-returning `binding_row()` parser, and Tests 1-6.
- `contracts/aprender/binding.yaml` (+38 / -19) — the corrected row plus the two header-claim rewrites.

## The measured numbers

### `pv audit contracts/setfit-apr-v1.yaml --binding contracts/aprender/binding.yaml`

Statuses captured with `cmd > log 2>&1; rc=$?`, never through a pipe.

| Counter | BEFORE (at `91eba6f1c`) | AFTER |
|---|---|---|
| Total equations | 15 | 15 |
| Bound equations | 15 | 15 |
| **Implemented** | **14** | **15** |
| Partial | 0 | 0 |
| **Not implemented** | **1** | **0** |
| Obligations total | 18 | 18 |
| Obligations covered | 252 | 270 |
| `BIND-` lines | 1 (`[WARN] BIND-004: Equation 'backend_identity' ... is pending implementation`) | **0** (`No binding gaps found`) |
| rc | 0 | 0 |

`make contract-audit-phase4` — rc=0, `Phase 4 binding audit: 1 contract(s) audited, every equation is bound`.

### `pv validate contracts/setfit-apr-v1.yaml`

rc=0, `0 error(s), 0 warning(s)`, `Contract is valid.` — before and after. The contract is byte-unchanged (`git diff --name-only -- contracts/setfit-apr-v1.yaml` is empty), and `pv` itself corroborates it: `pv diff` of HEAD's copy against the working copy returns rc=0 `Contracts are identical.`

### `pv diff` on the registry — recorded, not worked around

Given two materialised filesystem paths (never a git revision):

- old: `…/scratchpad/binding-04-19-old.yaml` (1520 lines, from `git show HEAD:contracts/aprender/binding.yaml`)
- new: `…/scratchpad/binding-04-19-new.yaml` (1539 lines, the working copy)

Verbatim output, rc=1:

```
error: Failed to parse YAML: missing field `metadata`
```

**Control run**, because an error on an edited file invites the wrong conclusion: `pv diff OLD OLD` — the *unmodified* registry against itself — fails **identically** (rc=1, same message). So this is a typed rejection of the file **kind** (a binding registry is not a `KernelContract` and has no `metadata` block), not a reaction to this edit. Second control: `pv diff` on two real contracts returns rc=0, so the tool works when given its kind. No shell workaround was written.

### `pv verify-bindings` — reported honestly, and NOT this plan's evidence

| | BEFORE | AFTER |
|---|---|---|
| `aprender: N/124 binding functions verified in source` | **0/124** | **0/124** |
| ghost count | 124 | 124 |
| rc | 1 | 1 |

The ghost-list delta is one line: `backend_identity` leaves, and the visible 20-entry window shifts by one (`dispatch_inspection_commands` pulled in). **The count stays exactly 124**, which is the point — the replacement name `ExecutionBackend::identity` is ghosted too, and it demonstrably exists at `encoder.rs:235`. The tool resolved neither name. It also ghosts `softmax` (`binding.yaml:7`), `gelu` (`:22`) and `within` (`:1438`); with 0/124 verified, *every* row is a ghost by construction. This is a repo-wide tooling limitation, not a Phase 4 defect, and it is why the compile-witness test is the gate.

### Test counts

| Filter | BEFORE | AFTER | Δ |
|---|---|---|---|
| `setfit::classify::backend` | 7 | **13** | +6 |
| `setfit::classify::` | 50 | **56** | +6 |
| `aprender-core --features setfit --lib` | 14423 (0 failed, 2 ignored) | **14429** (0 failed, 2 ignored) | +6 |

`make setfit-classify-tests` rc=0, 56 passed, `assert_tests_ran` floor of 45 satisfied.

## The induced-mutation REDs

Each mutation was applied, observed, and reverted from a pristine scratch copy (never `git checkout -- .`). The registry's natural RED came first and is stronger than a planted one.

**RED-0 (natural, un-planted — Task 1's commit `daf6ce868`).** The guard was committed against the registry as it stood, rc=101, 10 passed / 3 failed:

- test 2: `the backend_identity row does not carry "\n  module_path: aprender::setfit::encoder\n". …`
- test 3: `the registry still names "aprender::setfit::classify::backend_identity", which exists at no point on this branch`
- test 6: `the row's module_path must end in the `encoder` segment this test reads`

**Probe A — renamed the equation key to `backend_identity_x`.** rc=101. Test 4 RED with its own message:

> `no `equation: backend_identity` row was found in contracts/aprender/binding.yaml. A parser that silently matched NOTHING would make every pin in this module vacuously true`

**Probe C — decoy row carrying `module_path: aprender::setfit::classify` + `function: backend_identity`.** rc=101, **12 passed / 1 failed** — only the targeted test fired:

> `the registry still carries a bare "\n  function: backend_identity\n" column; an inherent method needs its type to resolve`

Note which arm fired: the decoy never spelled the full dotted ghost path, so the *bare-column* arm caught it while the path arm stayed quiet. The two arms are independent, as intended.

**Probe D — compile-neutral `impl  ExecutionBackend {` (DOUBLE SPACE).** rc=101, **11 passed / 2 failed**. Test 6 RED with its own assertion:

> `encoder.rs does not declare `impl ExecutionBackend {`, so the row's `encoder` segment names a module the symbol does not live in`

Proven compile-neutral rather than assumed: `running 13 tests` appears in the log, **zero** `error[E….]` rustc diagnostics, and the only `error:` line is cargo's `test failed, to rerun pass …`. So Test 6's own assertion fired — this probe did not accidentally test Test 1. The sibling guard `execution_backend_has_no_public_constructor_or_setter` also went red, as expected, since it keys on the same marker.

**Probe B — renamed the method to `identity_x`.** rc=101, `error[E0599]: no method named 'identity' found for struct 'ExecutionBackend' in the current scope`, **zero tests ran** — a build failure, as required.

**Probe B2 — added because Probe B does not prove what it appears to.** Probe B's E0599 was raised at `classify.rs:726`, a **pre-existing production call site** inside `VerifiedSetFitModel::classify`. The plain-lib build aborts there before the test-cfg build is attempted, so Test 1's line is never compiled and the probe credits code that already existed. Reporting Test 1 as "a real compile witness" on that evidence would be exactly the CLAUDE.md rule-2 error of labelling a run by intent.

So a sharper mutation was run: change only the **receiver**, `pub fn identity(&self)` → `pub fn identity(self)`. `ExecutionBackend` is `Copy`, so every shipped call site still compiles. Result: **zero E0599**, and two `error[E0308]` — at `classify.rs:1523` (Test 1's coercion: *expected fn pointer `for<'a> fn(&'a ExecutionBackend) -> String`, found fn item*) and `classify.rs:1631` (Test 5's UFCS call). Both are lines this plan added. **A signature change invisible to every pre-existing caller is caught only by the new guard** — which is Test 1's actual, non-redundant contribution.

## The diff, enumerated

`git diff --stat -- contracts/` lists `contracts/aprender/binding.yaml` and nothing else. **4 hunks:**

| Hunk | Old range | What |
|---|---|---|
| 1 | 1349-1350 | header claim (a): "FOURTEEN of fifteen … alone is still `pending`" → fifteen of fifteen, naming 04-19 |
| 2 | 1362-1363 | the dotted ghost path inside the 04-10 paragraph |
| 3 | 1365-1367 | header claim (c): "out of 04-10's scope and the entry stays `pending` … BIND-004" → past tense + what 04-19 did |
| 4 | 1449-1460 | the row: `module_path`, `function`, `signature` (added), `status`, `notes` |

Structural row fields in the whole diff: `-module_path/-function/-status` and `+module_path/+function/+signature/+status` — all four from the `backend_identity` row. **Zero `equation:` lines appear in the diff**, so no other row moved.

String-level criteria, all met: `FOURTEEN of fifteen` → 0; ``the entry stays `pending` `` → 0; ghost path in `binding.yaml` → 0; ghost path across `contracts/` → 0; `^  function: identity$` → 0; `^  function: backend_identity$` → 0; `function: ExecutionBackend::identity` → **1**; `include_str!("../../../../contracts/aprender/binding.yaml")` in `classify.rs` → **1**.

## Decisions Made

- **The `function` column is type-qualified.** `identity` is an inherent method; a bare `identity` would not resolve. The registry's own precedent for a type in this column is `SetFitArtifactDoc` and `ClassifyResponse`.
- **The blindness of `pv verify-bindings` is recorded on the row.** It is the registry documenting the blind spot in its own auditor, and `deferred-items.md` belongs to a wave-1 sibling.
- **Test 6 reads `encoder.rs` directly.** Tests 1-3 compare `module_path` against a literal inside `classify.rs`, which is not the module that literal names; reading the file the last path segment names is the strongest tie available without a proc-macro.
- **The 04-10 paragraph explaining why the correction was declined is kept.** It is still true and is the reason this needed its own plan.

## Deviations from Plan

### 1. [Rule 3 — Blocking] Hunk 2 touches lines 1362-1363, outside the plan's enumerated header range

- **Found during:** Task 2.
- **Issue:** The plan enumerated the header edits as lines 1349-1350 and 1366-1367. But the dotted ghost path `aprender::setfit::classify::backend_identity` also appeared at line 1363, inside the 04-10 paragraph. The plan's own must-have ("the ghost path appears nowhere in the registry"), its Test 3 specification, and its acceptance criterion (`grep -c … returns 0`) each independently require zero occurrences file-wide, so leaving :1363 would have failed the plan against itself.
- **Fix:** Rewrote 1362-1367 as one contiguous paragraph that states the discrepancy without spelling the dotted path ("named a free function `backend_identity` in the `aprender::setfit::classify` module"), keeping the reasoning the plan asked to preserve.
- **Verification:** `grep -c` for the ghost path returns 0 in `binding.yaml` and across `contracts/`; Test 3 green; no other row's fields moved.
- **Committed in:** `b94c5a2d4`.

### 2. [Rule 1 — Bug] Probe B does not test what the plan assumed; added Probe B2

- **Found during:** Task 1's acceptance probes.
- **Issue:** The plan's probe for Test 1 (rename the method, expect a compile error) produces an E0599 at a pre-existing production call site, aborting the build before Test 1 is ever compiled. The probe passes while proving nothing about the new guard.
- **Fix:** Added Probe B2, a receiver-only change (`&self` → `self`) that is invisible to every `Copy`-mediated call site and fails only at Test 1's typed coercion (E0308 at `classify.rs:1523`).
- **Verification:** Zero E0599 in Probe B2; both E0308s point at lines this plan added.
- **Committed in:** no code change — this is measurement, recorded here.

### 3. [Rule 3 — Blocking] `cargo fmt -p aprender-core` reformatted six `apr-cli` files; reverted

- **Found during:** Task 1.
- **Issue:** Formatting the crate also rewrote files under `crates/apr-cli/`, outside this plan's `files_modified` and plausibly owned by a parallel sibling.
- **Fix:** Reverted each of the six individually with `git checkout -- <file>` (never a blanket reset). Only `classify.rs` remained modified.
- **Note:** Those `apr-cli` files are unformatted at HEAD — a pre-existing condition, out of this plan's scope, deliberately not fixed.

**Total deviations:** 3 (1 required by the plan's own must-have, 1 measurement correction, 1 scope protection). **No scope creep** — the shipped diff is exactly the two planned files.

## Issues Encountered

**This agent was not isolated in a git worktree.** It ran in the main checkout (`.git` is a directory) on the shared branch `gsd/phase-2-contract-gate`, concurrently with siblings 04-18 and 04-20, whose commits (`323bbbc35`, `e3c9c1f00`, `9949f982f`, `e9fbfd3a0`) are interleaved with mine in `git log`, and whose uncommitted edits to `crates/apr-cli/` and `crates/aprender-serve/` appeared in `git status` mid-run. HEAD was already at the expected base `91eba6f1c`, so the prescribed `git reset --hard` was a no-op and was not executed.

Mitigation, and why the result is still clean: every commit staged its files individually by path, never `git add .`/`-A`. Both commits are verified single-file (`git show --stat`): `daf6ce868` → `classify.rs` only; `b94c5a2d4` → `binding.yaml` only. All probe reverts restored from pristine scratch copies rather than git, so no sibling work was touched. `cargo test -p aprender-core` does not compile `apr-cli` or `aprender-serve`, so sibling churn could not contaminate any measurement above. **The orchestrator should be aware that `files_modified` disjointness was the only isolation actually in force.**

## Open Items — what this plan does NOT close

Stated plainly, per the plan's success criteria:

- **This closes ONE clause each of SC4 and SC5** — the registry no longer names a nonexistent `backend_identity` symbol, and cannot silently misreport it. The **other half of each** — running parity over a user-**produced** artifact — remains blocked behind **F-10** and is Phase 5 work.
- **OPS-04, OPS-06 and SAFE-01 remain UNCHECKED.** This plan removes their named blocker; it does not deliver the produced artifact they also require. `requirements-completed` is deliberately empty.
- **F-10 remains OPEN.**
- **The ≥10h per-crate `cargo-mutants` gate (04-11 must-have 4) remains OPEN.**
- **The SAFE-02 "in CI" clause remains OPEN.**
- **OPS-01 and OPS-02 stay NOT MET.** This plan closes neither.
- **`pv verify-bindings` is blind repo-wide** (0/124 verified, ghosting symbols that exist). Recorded on the registry row; a real fix is its own ticket, out of scope here.

## Next Phase Readiness

Phase 4's binding registry is internally consistent: no row names a symbol that does not exist, no header claim contradicts the rows beneath it, and the one row that was unfalsifiable now has a guard that fails four different ways. `make contract-audit-phase4` is green and blocking in tier3. The produced-artifact parity work (F-10) is the gating item for OPS-04 / OPS-06 / SAFE-01 and carries into Phase 5.

## Self-Check: PASSED

- Files exist: `04-19-SUMMARY.md`, `crates/aprender-core/src/setfit/classify.rs`, `contracts/aprender/binding.yaml`.
- Commits exist: `daf6ce868` (test), `b94c5a2d4` (fix), `9d8cd054a` (docs).
- Each commit is single-file, verified by `git show --stat`.
- `.planning/STATE.md` and `.planning/ROADMAP.md` untouched — zero occurrences in this plan's commit range, clean working tree. The orchestrator owns those writes.
- No file deletions in any commit.

---
*Phase: 04-apr-artifact-and-production-parity*
*Completed: 2026-08-16*
