---
status: partial
ticket: PMAT-976
row: C0-2
issue: 2891
epic: 2873
model: claude-sonnet-5
tokens_used: "[U] — not instrumented this session"
wall_clock_s: "[U] — not instrumented this session"
turns: "[U] — not instrumented this session"
---
# impl-PMAT-976 — C0-2 · sovereign-ci.yml pinned by sha; CB-2100 reachability NOT closed (finding filed)

## Identity
ticket PMAT-976 · kind code · branch `agent/C0-2` (worktree) · base `origin/main` @ `3792afa3d` · quorum `teamwork` (per DAG row).

## What this PR does
1. Pins `.github/workflows/ci.yml`'s `ci` job `uses:` reference from
   `paiml/.github/.github/workflows/sovereign-ci.yml@main` to
   `@4453399ee3794714800ff8db316ea7e1d3705a00` — content-diffed identical to
   `@main` at pin time (`gh api` content comparison, both directions).
2. Adds `scripts/check_ci_reusable_workflow_pinned.sh` (14-row self-test, both
   polarities) and wires it into `guard-runner-labels` (self-test then
   live-check, existing convention), which is itself in `gate.needs`.
3. Extends `contracts/apr-required-checks-v1.yaml` (the contract the DAG names
   for this row) with equation `c0_2_reusable_workflow_pinned_by_sha`, proof
   obligation RC-OB-008, falsification test RC-F-008.
4. Files #3029: a finding that pinning by sha does **not** make CB-2100 pass,
   contrary to the ticket's own registered mutation premise. Linked from #2891.

## Acceptance — A_i (from the DAG row / issue #2891), both re-run at HEAD

| A_i | command | result |
|---|---|---|
| A1 | `pmat comply check \| grep CB-2100` shows ✓, naming the nine rules under a required context | **NOT MET.** Still `✗`: `9 severity=error rule(s) unreachable … required context \`ci / gate\` resolves into \`…sovereign-ci.yml@4453399ee…\`, whose steps are not readable from this repository`. Verified against pmat 3.39.0 source (`services/gate_effect/resolve.rs::local_reusable_path`): any external `uses:` ref — branch, tag, or sha alike — is `Resolution::Opaque`. See #3029 for the full finding and why closing it today would red every PR on six untracked debts. |
| A2 | `.github/workflows/ci.yml` references `…sovereign-ci.yml@<sha>`, not `@main` | **MET.** `grep -n "sovereign-ci.yml@" .github/workflows/ci.yml` → `@4453399ee3794714800ff8db316ea7e1d3705a00`. |

Because A1 is not met, this row's DONE-IF condition is not satisfied. Status is
`partial`, not `complete`; the DAG's derived status (G-11: status comes from
receipts, not written by row PRs) should read this row as open pending #3029.

## Verification (this session, re-run directly — not delegated)
| check | result |
|---|---|
| `bash scripts/check_ci_reusable_workflow_pinned.sh --self-test` | rc 0, 14/14 rows, both polarities |
| `bash scripts/check_ci_reusable_workflow_pinned.sh` (live, HEAD) | rc 0, PASS |
| same script against `git show HEAD~1:.github/workflows/ci.yml` (RED commit, unpinned) | rc 1, FAIL naming `@main` |
| same script against `git show origin/main:.github/workflows/ci.yml` | rc 1, FAIL naming `@main` — reproduces the mutation directly on the real tree, not only in synthetic fixtures |
| `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` | OK |
| `bashrs lint scripts/check_ci_reusable_workflow_pinned.sh` | 0 errors, 8 warnings / 34 info — parity with sibling guards (e.g. `check_no_ghsa_banned_crates.sh`: 0 errors, 10 warnings / 42 info) |
| `bash -c 'source scripts/pv_bin.sh; "$PV" validate contracts/apr-required-checks-v1.yaml'` | 0 errors, 0 warnings — valid |
| `bash -c 'source scripts/pv_bin.sh; "$PV" lint contracts/ --diff origin/main'` | Diff-aware: 1 contract changed (`apr-required-checks-v1`); Gate 4 verify 26/26 refs found (0 missing); overall 0 errors, PASS |
| `bash scripts/check_contract_test_binding.sh` | rc 0, PASS (baseline unchanged, 27 pre-existing dangling refs elsewhere, none new) |
| `bash scripts/check_row_pr_write_set.sh --base origin/main --head HEAD --branch agent/C0-2` | PASS — 3 changed paths, no shared file (DAG/roadmap/spec/README) touched |
| `cargo fmt --all -- --check` | exit 0 |

## Mutation (RED, then the remedy) — commit-level, not just self-test
Commit 1 (`test(C0-2): RED …`) adds only the guard script, before the pin: the
live check against that commit's `ci.yml` fails, naming `@main`. Commit 2
(`fix(C0-2): pin …`) pins the ref and wires the guard: the live check passes.
Both states verified directly (table above), not asserted. The registered
mutation in the contract (RC-F-008: revert the `uses:` line to `@main`) is
identical to reverting commit 2 — verified by running the guard against
`origin/main`'s actual (still-`@main`) `ci.yml`, which is the real pre-fix
state, not a synthetic stand-in.

CI-level RED→GREEN (push the mutant, observe the PR's own required checks turn
red, revert, observe green, record both run ids) is **not yet done** — pending
this PR opening and its first CI run. Will be added to the PR body before
auto-merge is armed, per driver protocol.

## Quorum
Dispatched `paiml-agy-delegate` (lane=teamwork, width=1) with the full finding
and proposed plan (pin now, file #3029, receipt partial, continue to C0-4).
Both the primary run and a mechanism-check retry returned PROCEED/PASS with no
dissent, but the delegate's own receipt flags this as a **weak** quorum: agy
1.1.27's headless `-p` mode did not engage true `/teamwork-preview` fan-out
(2 turns, ~27s, zero child agents, every finding `grounding=asserted`, no repo
files read by the lane itself). Recorded as `feedback_teamwork_lane_does_not_fan_out.md`
in the delegate's own agent-memory for future sessions. The load-bearing
evidence in this receipt is the orchestrator's own direct source-reading and
empirical reproduction (tables above), not the quorum lane.

## Gaps / follow-up
- #3029: the real remedy for A1 — either (a) ratchet-baseline CB-040/081/400/
  1305/1308/1650 (new DAG rows) so a bare `pmat comply check` exits 0, then wire
  it into a required job, or (b) cross-repo manifest tooling (`paiml/.github` +
  pmat). Not scoped to this PR.
- CI-level mutation run ids: added to the PR body once the PR's first CI run
  completes (see above).

## Verdict
PARTIAL (`status: partial`). A2 lands with real, independent value (supply-chain
provenance). A1 remains open; #3029 is the tracked remedy. C0-2 is not DONE.
