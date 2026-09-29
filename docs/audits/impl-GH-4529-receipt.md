---
status: shadow-landed-on-branch
ticket: PMAT-4529
github_issue: 4529
part: 1 of 2 (shadow; the ruled join C is part 2)
kind: code
model: claude-opus-5-5
---

# #4529 QM-11 tier router — part 1: the SHADOW (aprender-52, 2026-09-29)

Branch `52/4529-qm11-tier-router`, stacked on `60/qm10-on-qm09`. Branch-only; lands after 0.70.

## What part 1 does

`ci/sections.yml` workspace-test-build gains ONE step, after "Stage the tier this job decided" and before the
sigma-build upload: `scripts/ci/tier_route_step.sh` prints `qm11-shadow: today=<tier> router=<T0|selective|T3>
packages=N causes=… owners=…` (or `router=abstained why=…`) and writes `sigma-build/qm11-route.json`.
It gates nothing: `continue-on-error: true`, `timeout-minutes: 5`, a fault is a `::warning::`, and no workflow
or section reads the file. `scripts/check_qm11_shadow.sh` (wired in guard-tree, run plain and `--self-test`)
turns each of those properties into a check.

Keep-or-tighten: every existing gate is unchanged (no step condition, output or fan-in input moved); one guard
is added. Red/green proof is the mutant table below: each way the shadow could start to gate is RED.

## Red/green — measured at 739d823ba1

| check | result | host |
|---|---|---|
| `tier_router.py --self-test` | 19/19 rows, 6/6 mutants RED | lambda (python only) |
| `tier_route_step.sh --self-test` | 14/14 rows (incl. hostile base ref: named abstention, PWNED not created) | lambda |
| `check_qm11_shadow.sh --self-test` | 14/14 mutants RED (8 wiring, 4 step-script, 2 curl-bound) | lambda |
| `check_qm11_shadow.sh` | PASS | lambda |
| `fat_driver.py self-test` | 49/49 | lambda |
| `check_guards_are_wired.sh` | PASS (ratcheted) | lambda |
| `bashrs lint` (2 new scripts) | 0 errors | lambda |
| `pv validate contracts/ci-tier-router-v1.yaml` | 0 errors | lambda |
| `pv lint contracts --gate sigma` | Pass, formal_prose 1464 (= baseline) | lambda |
| `pv extract contracts --check` | rc 0 | lambda |
| `cargo test -p aprender-contracts --lib` | 1721 passed, 0 failed, 5 ignored | framework16 |
| `cargo test -p aprender-contracts-cli --tests` | 21/21 binaries ok, 0 failed | framework16 |

At e21e13d57d the two contract suites were RED (1 each): the sigma gate, `formal_prose 1464 -> 1470`, caused by
this branch's seven prose `formal:` entries. Fixed in 739d823ba1 by symbolic formals; head vs head re-run GREEN.

not_measured: `scripts/guard_tree.sh --no-cargo` on lambda ran 132 checks, 2 RED — `check_baseline_ratchets` and
`check_complexity_ratchet`, both a tool_version mismatch of the local host (pmat 3.42.0 vs 3.41.1, bashrs 7.4.2 vs
7.4.1), not a property of the diff. intel was at load 168–236, so it was not used.

## Quorum

| round | commit | sonnet-5 | haiku-4.5 | agy gemini-3.1-pro-high |
|---|---|---|---|---|
| 1 | e21e13d57d | REQUEST_CHANGES | APPROVE | REQUEST_CHANGES (fcf3d43b-952d-408b-a49e-c603fda9c9e3) |
| 2 | 739d823ba1 | APPROVE | APPROVE | APPROVE (76691ae1-9c30-4eaf-896d-18b53043ad0e) |

Round-1 findings, all dispositioned and verified by every lane in round 2:
F1 BLOCKER base-ref shell injection (fixed: `$QM11_BASE` as data) · F2 BLOCKER unbounded curl (fixed: curl
bounds + step timeout, both guarded) · F3 MAJOR `--owners` never passed (fixed: dep-info owners, named in the
line) · F4 MAJOR vacuous closure row (fixed: app -> core fixture) · F5 MINOR uncommitted mutants (fixed) ·
F6/F7 MINOR (fixed) · F8 NIT docker `-v` with a colon — NOT fixed: the paths are fixed runner paths.
Round-2 NIT (sonnet), not fixed: dep-info owners assume the compile step and the shadow share the `/workspace`
mount convention; if that changes the line degrades to a NAMED `owners=absent:…`, never a wrong gate.

No opus lane: the author is opus.

## Ruling

aprender-1b, 08:32Z: "take C (B's selective + today's quick set; keep-or-tighten, pre-approved). T0 narrowing
goes to the operator as a separate ruling, not in this PR." Part 2 implements C; T0 is out of scope.
