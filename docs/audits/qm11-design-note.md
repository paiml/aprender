# QM-11 (#4529) tier router — how it plugs into today's tiers (aprender-52, 2026-09-29)

Branch `52/4529-qm11-tier-router`, stacked on `60/qm10-on-qm09` (QM-09 #4527 + QM-10 #4528). The router is
aprender-a2's (`scripts/ci/tier_router.py`, self-test 19/19 rows, 5/5 mutants RED on this stack).

## What exists today (ci/sections.yml `ws_tier` → scripts/ci_test_tier.sh)

| tier | when (PR) | runs |
|---|---|---|
| none | every path under docs/roadmaps/ or docs/audits/ | nothing |
| quick | any other PR | `--lib --tests` of touched + direct reverse deps (≤ cap 3; over cap: touched only + one `cargo check --workspace`), plus the tree-reader targets |
| full | root Cargo.toml / Cargo.lock / toolchain | QM-09 archive: every `--lib` test, 8 shards, plus the explicit integration commands |

`full ⊉ quick`: quick runs every `--tests` target of the touched crates; full runs only the targets listed in
`ci/explicit-test-commands.d`. So "tighter" is not a total order over these tiers.

## Measured: R' against today's selection

Last 80 first-parent commits on main; crates owned by autodiscovery (src/, tests/, …), `cargo metadata` of this
tree (81 members). Script: the router's own `Workspace.closure`.

| set | median | p90 | max |
|---|---|---|---|
| touched crates | 3 | 6 | 32 |
| today (touched + direct rdeps) | 12 | 22 | 46 |
| R' (reverse closure, Thm 15) | 44.5 | 48 | 55 |

34 of 80 touched crate code. 29/34 are over today's cap (so today runs touched-only + a workspace check);
29/34 have |R'| ≥ 40. For a code PR, "selective" is half to nearly all of the workspace.

## Options

- **A. Join, keep-or-tighten only.** final = today ∪ router. Code PRs with R' over cap go to full, plus today's
  `--tests`. Pre-approved (gates kept or tightened) — but CI gets SLOWER for ~29/34 code PRs, and the docs win
  is only the existing `none`. No latency target is met.
- **B. The spec as written (§QM-11).** T0 = no tests when D ∩ I = ∅ (INV-BASE held); selective = QM-09 archive
  + nextest `package(R')` filter + the explicit integration commands; T3 = full. This NARROWS today in two
  places: T0 drops the tree-reader run (justified by F, the nightly strace), and selective drops the
  non-explicit `--tests` targets of touched crates. Narrowing = a gate loosening → needs an operator ruling.
- **C. B's selective plus today's quick integration set** (touched + direct-rdep `--tests`, tree readers) —
  keep-or-tighten; T0 still needs a ruling. Cost: one archive build + R' lib runs + today's quick compile.

Common to all three and built first on this branch: the route is computed ONCE per run (build job), shipped to
the 8 shards inside `ws-lib-archive`, so every shard applies the same decision (workspace-test already refuses
shards whose tier disagrees) and the nightly artifact is fetched once, not 9× (GH-1).

Recommendation: C now (pre-approved); T0 (the docs ≤ 2 min target) waits for the ruling.
