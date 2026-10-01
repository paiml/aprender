---
phase: 06-native-time-series-forecasting-stack
plan: 02
subsystem: infra
tags: [ci-gates, monorepo-invariants, readme-contract, falsify-mono-011, ratchet, mcp, policy]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-01 landed crates/aprender-mcp-forecast, whose [[bin]] turned FALSIFY-MONO-011's four pre-existing violations into five"
  - phase: 04-setfit-mcp
    provides: "the four thin SetFit MCP deployment crates that were already violating FALSIFY-MONO-011 before Phase 6 began"
provides:
  - "A recorded human decision closing RESEARCH Open Question 1: thin MCP deployment units are a SECOND, separately-ratcheted category in FALSIFY-MONO-011"
  - "crates/aprender-core/tests/monorepo_invariants.rs: deployment_unit_bins (6 names), DEPLOYMENT_UNIT_BASELINE = 6 shrink-only assert, and its own stale-entry check — with allowed_bins and ALLOWLIST_BASELINE = 27 byte-identical"
  - "monorepo_invariants GREEN for the first time this phase: 11 passed; 0 failed"
  - "A forward-looking, non-firing registration for aprender-mcp-chronos, so plan 06-07 creates that crate without re-opening the human gate"
  - "crates/aprender-mcp-setfit-lambda/README.md and crates/aprender-contrastive-data/README.md (new); crates/aprender-mcp-setfit/README.md gains the monorepo link"
  - "readme_contract down to ONE failure (FALSIFY-README-007, the contract count), owned by plan 06-09"
affects: [06-07, 06-08, 06-09]

actuals:
  tokens: 3764
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "A shrink-only ratchet whose policy sentence stops describing its members is SPLIT into a second register, not silently widened — the sentence is the artifact, the constant is only its enforcement"
    - "Every ratchet edit is proven with a two-sided control: a mutation that must turn it RED, then a byte-identical revert that must turn it GREEN. A ratchet never seen to fail is not a proven ratchet"
    - "A forward-looking allowlist entry is only safe where an existence guard makes it inert; verify the guard by reading the predicate, not by assuming it"

key-files:
  created:
    - crates/aprender-mcp-setfit-lambda/README.md
    - crates/aprender-contrastive-data/README.md
    - .planning/phases/06-native-time-series-forecasting-stack/06-02-SUMMARY.md
  modified:
    - crates/aprender-core/tests/monorepo_invariants.rs
    - crates/aprender-mcp-setfit/README.md
    - .planning/phases/06-native-time-series-forecasting-stack/deferred-items.md
    - .planning/STATE.md

key-decisions:
  - "FALSIFY-MONO-011 gains a SECOND, separately-ratcheted category for thin MCP deployment units (human answer: `deployment-unit-class`); allowed_bins and ALLOWLIST_BASELINE = 27 stay byte-identical so `migration debt` keeps meaning migration debt"
  - "Policy sentence for the new register, verbatim as the human approved it: publish = false thin MCP servers whose capability IS a protocol surface; adding one requires a CONTEXT decision, never a same-PR edit"
  - "aprender-mcp-chronos is registered BEFORE the crate exists; the stale-entry check's `crates_dir.join(c).exists()` guard was read (line 379), not assumed, and the directory was confirmed ABSENT"
  - "The three README fixes state no count and no version number — the count rows are re-derived once, in plan 06-09 (CLAUDE.md `Derive the numbers`)"

patterns-established:
  - "Split-the-register: when a ratchet's category stops describing a class of member, add a parallel register with its own baseline, its own message and its own rot check rather than raising the original baseline"
  - "Prove the pre-existing: an out-of-scope failure is logged only after an untouched-sibling control reproduces it"

requirements-completed: []

coverage:
  - id: D1
    description: "FALSIFY-MONO-011 is GREEN: the four pre-existing SetFit deployment crates and aprender-mcp-forecast are accounted for by name, and no other invariant in the file regressed"
    requirement: SC5
    verification:
      - kind: integration
        ref: "cargo test -p aprender-core --test monorepo_invariants (11 passed; 0 failed; rc=0)"
        status: pass
      - kind: other
        ref: "baseline before the edit: FAILED, 10 passed / 1 failed, violations [aprender-mcp-setfit, aprender-mcp-setfit-lambda, aprender-mcp-setfit-train, aprender-setfit-train-lambda, aprender-mcp-forecast] — /tmp/p06-02-baseline.log"
        status: pass
    human_judgment: false
  - id: D2
    description: "The new deployment-unit ratchet still bites — growth is rejected with the shrink-only message, and the revert restores green"
    requirement: SC5
    verification:
      - kind: other
        ref: "control mutation (bogus 7th entry): rc=101, panic at monorepo_invariants.rs:407 — 'the thin-MCP deployment-unit register grew to 7 (baseline 6) … It is shrink-only' — /tmp/p06-02-t2-control-red.log"
        status: pass
      - kind: other
        ref: "revert: `cmp` byte-identical to the pre-mutation file, then 11 passed; 0 failed (rc=0) — /tmp/p06-02-t2.log"
        status: pass
    human_judgment: false
  - id: D3
    description: "The migration-debt register is untouched: allowed_bins and ALLOWLIST_BASELINE = 27 are byte-identical to HEAD~"
    verification:
      - kind: other
        ref: "git diff -U0 on the file removes exactly ONE line (the violations condition); `const ALLOWLIST_BASELINE: usize = 27;` present at :390"
        status: pass
    human_judgment: false
  - id: D4
    description: "FALSIFY-README-CRATE-001 and -002 are closed: both missing crate READMEs exist and all three files carry the paiml/aprender link"
    requirement: SC5
    verification:
      - kind: integration
        ref: "cargo test -p aprender-core --test readme_contract --no-fail-fast: 15 test lines ran, 14 passed / 1 failed, no FALSIFY-README-CRATE-00[12] in the output — /tmp/p06-02-t3.log"
        status: pass
      - kind: other
        ref: "grep -c 'paiml/aprender' on all three READMEs returns 1 each; git check-ignore exits 1 on both new files"
        status: pass
    human_judgment: false
  - id: D5
    description: "The two new crate READMEs describe their crates accurately (no fabricated command, no count, no version number)"
    verification:
      - kind: other
        ref: "every claim traced to source before writing: build.rs (the two reads of APRENDER_SETFIT_MODEL), src/lib.rs (stateless() rationale), src/main.rs (loopback proxy), Makefile:808 (contrastive-data-boundary), allowed-deps.txt, extended_commands.rs:874 + dispatch_analysis.rs:433-456 (apr data select / pairs)"
        status: pass
    human_judgment: true
    rationale: "Prose accuracy is not asserted by any test — readme_contract only checks existence and the monorepo substring. Every factual claim was traced to a source line this session, but a human should skim both new READMEs before the phase ships (threat T-06-09, Repudiation: README claims)."

duration: 20 min
completed: 2026-09-06
status: complete
---

# Phase 06 Plan 02: Contract-Gate Reachability Summary

**FALSIFY-MONO-011 now carries two registers instead of one — migration debt at 27, untouched, and a new shrink-only thin-MCP deployment-unit register at 6 — turning `monorepo_invariants` green for the first time this phase, with the ratchet's teeth proven by a mutation that made it fail.**

## Performance

- **Duration:** ~20 min (start approximate; first commit `572326eab` at 2026-09-06T04:42:00Z, last at 04:49:51Z)
- **Tasks:** 3 (1 human decision + 2 auto)
- **Files created/modified:** 4 by the tasks (1 test file, 3 READMEs) + STATE.md + deferred-items.md

## The human decision (Task 1) — recorded verbatim

The plan's Task 1 was a `gate="blocking-human"` `checkpoint:decision`. A previous executor
reached it and stopped without touching a file. The human's answer:

> **`deployment-unit-class`**

selected with **no wording changes** to the policy sentence. Scoped as the plan wrote it:

- `allowed_bins` and `ALLOWLIST_BASELINE = 27` stay **byte-identical**.
- A new `deployment_unit_bins` set of six names: `aprender-mcp-setfit`,
  `aprender-mcp-setfit-lambda`, `aprender-mcp-setfit-train`, `aprender-setfit-train-lambda`,
  `aprender-mcp-forecast`, `aprender-mcp-chronos`.
- Its own `DEPLOYMENT_UNIT_BASELINE = 6` shrink-only assert and its own stale-entry check.
- Policy sentence, verbatim: *"publish = false thin MCP servers whose capability IS a protocol
  surface; adding one requires a CONTEXT decision, never a same-PR edit."*

**Rationale as recorded.** "Migration debt" keeps meaning migration debt. The existing ratchet's
claim — that every entry is a capability reachable only by its own binary and not yet through
`apr <subcommand>` — stays literally true, because CONTEXT explicitly defers `apr forecast`, so
these six are not awaiting migration at all. They have no `apr` destination to await: the
capability *is* the MCP protocol surface, and the binary *is* the deployment unit (a pmcp.run
process or a Lambda `bootstrap`). Folding them into `allowed_bins` (option `single-list-33`)
would have made the ratchet's own sentence false for six of its thirty-three entries.

The three rejected options and why: **`single-list-33`** blurs debt with deployment unit;
**`phase6-only-29`** leaves the gate RED on the SetFit four, so SC5's "every gate is green" would
be unreachable in this phase; **`halt`** contradicts D-06 and the CONTEXT deferral of
`apr forecast`.

RESEARCH Open Question 1 is closed. The decision was written to STATE.md and committed
(`572326eab`) **before** any edit to `monorepo_invariants.rs`, which is what the plan required.

## Measured, not quoted: the gate state at plan start

The plan and RESEARCH both described **four** FALSIFY-MONO-011 violations. Re-measured on this
branch before touching anything (`/tmp/p06-02-baseline.log`, rc=101):

```
FALSIFY-MONO-011: Unauthorized [[bin]] sections found in: ["aprender-mcp-setfit",
"aprender-mcp-setfit-lambda", "aprender-mcp-setfit-train", "aprender-setfit-train-lambda",
"aprender-mcp-forecast"]
test result: FAILED. 10 passed; 1 failed
```

**Five**, not four — plan 06-01 landed `aprender-mcp-forecast` in between. This is exactly the
class of drift CLAUDE.md's "derive the numbers" rule exists for, and it is why the baseline was
re-run rather than read off RESEARCH.

## The two-sided ratchet control (the plan's central proof obligation)

A ratchet that has never been seen to fail is not a proven ratchet. Both sides were run, and both
observations are recorded here as the plan demanded.

**RED — a bogus seventh entry (`"aprender-mcp-bogus-control"`) added to `deployment_unit_bins`:**

```
CONTROL rc=101
thread 'test_no_unauthorized_binaries' panicked at crates/aprender-core/tests/monorepo_invariants.rs:407:5:
FALSIFY-MONO-011: the thin-MCP deployment-unit register grew to 7 (baseline 6). It is
shrink-only: a new publish = false MCP server binary requires a recorded CONTEXT decision
(a `gate="blocking-human"` checkpoint), never a same-PR edit to this list.
test result: FAILED. 10 passed; 1 failed
```

The message contains `shrink-only`, and the failing assertion is the **new** baseline
(`grew to 7 (baseline 6)`), not the pre-existing one — so the control proves the code this plan
added, not code that was already there.

**GREEN — revert:**

```
mutation reverted
BYTE-IDENTICAL to pre-mutation state      # cmp against /tmp/p06-02-mono-good.rs
REVERT rc=0
test result: ok. 11 passed; 0 failed
```

The revert was verified with `cmp` against a copy taken before the mutation, so "reverted" is a
measured byte-identity claim rather than an intention.

## Verifying the forward-looking `aprender-mcp-chronos` entry

Registering a crate that does not exist is only safe if the stale-entry check cannot fire on it.
The plan said to verify this by reading, not assuming. Read (`monorepo_invariants.rs:379`):

```rust
.filter(|c| !with_bins.iter().any(|b| b == *c) && crates_dir.join(c).exists())
```

The `&& crates_dir.join(c).exists()` conjunct means a **missing directory is not stale**. And
measured: `test -d crates/aprender-mcp-chronos` → **ABSENT**. The new stale check mirrors the
predicate exactly and carries a comment saying the existence guard is load-bearing for this entry,
so a future reader deleting it as "redundant" would find out at plan 06-07's expense rather than
silently.

## Task Commits

1. **Task 1: record the FALSIFY-MONO-011 decision** — `572326eab` (docs)
2. **Task 2: the deployment-unit register + two-sided control** — `be5f4f974` (test)
3. **Task 3: three crate-README drift fixes** — `db8c9c2d0` (docs)
4. *(out-of-scope finding logged)* — `dd3e4f8af` (docs)

## Files Created/Modified

| File | What changed |
|---|---|
| `crates/aprender-core/tests/monorepo_invariants.rs` | +61/−1. A policy comment block, `deployment_unit_bins` (6 names, one comment each), the violations condition widened to accept either register, `DEPLOYMENT_UNIT_BASELINE = 6` with its own shrink-only assert, and `stale_units` — a second rot check. `allowed_bins` and `ALLOWLIST_BASELINE = 27` untouched. |
| `crates/aprender-mcp-setfit-lambda/README.md` | **New.** The Lambda custom-runtime wrapper: why the binary must be named `bootstrap`, the loopback-proxy pattern, why `stateless()` is the only serverless-viable config, and a Build section explaining that `APRENDER_SETFIT_MODEL` is read at *two different times* (build → `include_bytes!`; run → path fallback). |
| `crates/aprender-contrastive-data/README.md` | **New.** The Cargo description, the D-04 bytes boundary and *why* (object storage behind a serverless consumer), the positive dependency allowlist enforced by `make contrastive-data-boundary`, Philox determinism, and the `apr data select` / `apr data pairs` entry points. |
| `crates/aprender-mcp-setfit/README.md` | +2. The monorepo link sentence after the opening paragraph; the existing `paiml/rust-mcp-sdk` link and every run instruction untouched. |
| `.planning/phases/…/deferred-items.md` | +26. D-ITEM-06-02-a (below). |
| `.planning/STATE.md` | The Task 1 decision, plus position/metrics/session. |

## Verification results

| Gate | Before this plan | After |
|---|---|---|
| `cargo test -p aprender-core --test monorepo_invariants` | **FAILED** — 10 passed, 1 failed, 5 unauthorized binaries | **ok. 11 passed; 0 failed** (rc=0) |
| `cargo test -p aprender-core --test readme_contract --no-fail-fast` | 4 failed at RESEARCH time; 2 failed at this plan's start | **14 passed; 1 failed** — only `FALSIFY-README-007` |
| `cargo fmt --all -- --check` | — | rc=0 |
| Task 2 acceptance criteria | — | 5/5 |
| Task 3 acceptance criteria | — | 3/4 (one criterion was stale — see deviation 1) |
| Deletion check on all four commits | — | no file deleted by any commit |

**Drift-failure accounting, measured.** Four failing test rows existed on this branch when this
plan started: `FALSIFY-MONO-011`, `FALSIFY-README-CRATE-001`, `FALSIFY-README-CRATE-002`,
`FALSIFY-README-007`. This plan closed **three**. The one remaining is the contract count
(`**1786** provable contracts` vs the README's `**1778**`), owned by plan 06-09, which re-derives
every count once after all crates and contracts exist.

## Decisions Made

Beyond the human decision above:

- **Task 1 was committed as a STATE.md decision record**, not left as prose in this SUMMARY alone.
  The plan gave Task 1 no file artifact, but "each task committed individually" and "recorded
  before any edit" are both satisfiable by writing the decision to STATE.md's Decisions section
  first. That commit (`572326eab`) is therefore the audit trail proving the ordering, not just an
  assertion about it.
- **The new register got its own message text rather than reusing the migration-debt wording.**
  Its failure names the CONTEXT-decision requirement, so an engineer who trips it is told the
  actual remedy (a `gate="blocking-human"` checkpoint) instead of being told to migrate a
  capability to an `apr` subcommand that does not and will not exist for it.
- **No `apr data select` / `apr data pairs` claim was written on trust.** `contracts/apr-cli-commands-v1.yaml`
  has no `contrastive` entry and the registry grep came back empty, which looked like the plan was
  asking for a fabricated command. Traced instead to the code: `extended_commands.rs:874` defines
  the `Data` subcommand and `dispatch_analysis.rs:433-456` dispatches `DataCommands::Select` and
  `::Pairs`. Both commands are real, so the sentence stayed.

## Deviations from Plan

### 1. [Rule 1 — Stale criterion] Task 3's acceptance criterion expected `FALSIFY-README-005` to still be RED; it is GREEN

- **Found during:** Task 3 verification.
- **Issue:** The criterion reads: *"`/tmp/p06-02-t3.log` contains `FALSIFY-README-005` and
  `FALSIFY-README-007` (the count rows, still red — proof the run reached them)"*, and the plan's
  `must_haves` say *"the two COUNT failures stay red"*. Measured: `FALSIFY-README-005` does **not**
  appear, because `test_readme_crate_count_matches_workspace` now **passes** —
  plan 06-01 corrected the README's workspace-crate count as its own Rule 3 deviation (06-01
  SUMMARY, deviation 4). The plan was authored against the pre-06-01 tree.
- **Fix:** None needed in code. The criterion's stated *purpose* — "proof the run reached them" —
  is satisfied by a different, stronger observation: the run executed **15 test lines** and
  reported `14 passed; 1 failed`, so `README-005` was reached and passed rather than skipped. The
  criterion is recorded here as satisfied-in-purpose, not silently dropped.
- **Net effect:** better than planned. One count row remains red, not two.
- **Verification:** `grep 'test_readme_crate_count_matches_workspace' /tmp/p06-02-t3.log` →
  `... ok`; `grep -c FALSIFY-README-CRATE-00[12]` → 0.

### 2. [Rule 3 — Blocking, harness] `rtk` strips the `test result:` line the plan's `<verify>` greps for

- **Found during:** every test invocation in this plan.
- **Issue:** The `rtk` Bash hook rewrites `cargo test` and filters its output, so the plan's
  `grep -E 'test result: ok\. [1-9][0-9]* passed; 0 failed'` can never match — the same harness
  artifact 06-01 hit and documented.
- **Fix:** ran every verification through `rtk proxy cargo test …`, which emits raw output. The
  assertion the criterion encodes was then checked against real text.
- **Not a code defect.** Recorded so the next executor does not re-diagnose it.

---

**Total deviations:** 2 — 1 stale acceptance criterion (resolved in favour of the criterion's
purpose, with the measurement recorded), 1 harness workaround.
**Impact on plan:** No scope creep. The decided option was implemented verbatim; no file outside
the plan's `files_modified` list was touched by a task commit.

## Issues Encountered

- **`cargo clippy -p aprender-core --test monorepo_invariants --no-deps -- -D warnings` fails —
  and `--no-deps` does not rescue it.** The error is `unreachable_code` in
  `crates/aprender-core/src/demo/reliable/performance.rs:126`, inside the primary package's own
  lib, which the test target links.
  **Proved pre-existing rather than assumed:** the identical command against the *untouched*
  sibling target `--test readme_contract` fails with the same single error (rc=101), and this plan
  modified no file under `crates/aprender-core/src/`. Out of scope; logged as **D-ITEM-06-02-a**
  in `deferred-items.md`. It also refines 06-01's D-ITEM-06-01-b, whose diagnosis ("`-D warnings`
  reaches path dependencies") turns out to be only one of two causes.
- **No git pre-commit hook is installed** in this checkout (`ls .git/hooks/` shows only samples),
  so the clippy condition above did not block any commit. Worth knowing before someone assumes a
  green commit implies a green lint.

## Known Stubs

None. This plan added no code path — one test file gained a second register and three READMEs were
written. No hardcoded empty value, placeholder string or unwired component was introduced.

## Threat Flags

None. `T-06-08` (Elevation of Privilege — the allowlist as a policy surface) was mitigated as the
threat model specified: the edit followed a `gate="blocking-human"` decision, the two-sided control
proves the ratchet still fails on growth, and **both** stale-entry checks are retained. `T-06-09`
(Repudiation — README claims) was mitigated by writing no count and no version number, and by
tracing every factual claim to a source line before writing it. `T-06-SC` (Tampering — cargo
installs): this plan installed nothing and added no dependency; `Cargo.lock` is untouched.

## User Setup Required

None — no external service configuration.

## Next Phase Readiness

- **`monorepo_invariants` is green and stays green as Phase 6 grows.** Plan 06-07 can create
  `crates/aprender-mcp-chronos` without re-opening the human gate: the name is already registered,
  and the stale check will simply start finding the directory once it exists.
- **`readme_contract` has exactly one failure left**, `FALSIFY-README-007`, and it belongs to plan
  06-09 by design — every count is re-derived once, after all crates and contracts land.
- **SC5 ("every gate is green") is now reachable.** Nothing in this plan added to the drift
  failures; three of the four measured on this branch are closed. SC5 was **not** marked complete
  because sibling plans in this phase also declare it and have not finished
  (`requirements ready-ids` → `0/1 ready`), which is the correct shared-ID behaviour.
- **One caution for whoever adds the next MCP server:** `DEPLOYMENT_UNIT_BASELINE = 6` is
  deliberately hostile to a same-PR edit. That is the point of the decision, not an oversight.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-06*

## Self-Check: PASSED

- `crates/aprender-mcp-setfit-lambda/README.md` — FOUND
- `crates/aprender-contrastive-data/README.md` — FOUND
- `crates/aprender-core/tests/monorepo_invariants.rs` — FOUND
- `crates/aprender-mcp-setfit/README.md` — FOUND
- Commits `572326eab`, `be5f4f974`, `db8c9c2d0`, `dd3e4f8af` — all present in `git log --all`
