# Autonomous run — 2026-09-15

Operator away. Route = self orchestrator; every decision goes to the quorum, and
anything it cannot decide stops the line for that item and is written here as a
decision request while the other lanes continue. §7 report per interval.

---

## Interval — 13:20Z

```
scope: clean-room ROOT-CAUSED and fixed (#3305/#3307) · was the top blocker
queue: 11 PRs unstranded · 4 resolved out of CONFLICTING · 7 updated onto main
pins:  unchanged
```

### What moved

**The clean-room hard gate — root cause, not a ticket.** It had been red 8/8
since 2026-09-08 and **v0.66.0 and v0.67.0 both shipped over it.** The gate was
right the whole time.

`gates-lib.sh` stage A0 strips path-only dependencies to simulate `cargo
publish`. `cargo publish` omits a dev-dependency carrying only `path` — that
omission is deliberate, and is what makes a dev-dep *cycle* legal — but the
crate's `#[cfg(test)]` code is published anyway. `crates/aprender-compute/src/`
named three stripped crates across two files, which is exactly the 11 error
sites in run `34915682258`:

| File | Stripped dep | Sites |
|---|---|---|
| `contract_tests_linalg.rs` | `trueno_sparse`, `trueno_solve` | 10 |
| `simulation/mod.rs` | `simular` | 1 |

Two of the four aliases were **already declared in the root
`[workspace.dependencies]` with a version**; aprender-compute redeclared them
locally and bypassed the table. `trueno-sparse` and `trueno-solve` were in no
workspace table at all. All three siblings are published at 0.67.0 and none
depends back on aprender-compute, so inheritance creates no cycle.
`aprender`/aprender-core stays path-only — that one *is* a cycle, and no `src/`
code touches it.

Five whys terminates at: **nothing asserts that a path dependency on a
publishable member carries a version.** In-tree builds resolve the path, so
`ci / gate`, `workspace-test` and every nightly are green on the same commit;
the only instrument that can see it runs downstream of publish, and had been red
long enough that its signal never reached a PR. So the guard shipped in the same
PR, not as a follow-up.

Falsified through the gate's own `drop_sourceless_dev_deps`, not a
re-implementation: **4 aliases stripped before, 4 kept after.**

The guard found **27 more sites across five other crates**, latent only because
clean-room compiles `aprender-compute` alone. Shrink-only baseline keyed by
`(manifest, alias)` — deliberately not `file:line`, which drifts inside a PR.
Drain is #3306; two of the five are genuine publish cycles whose test code has
to leave `src/`.

**The PR backlog was stranded, and not for the reason anyone thought.** Eight
PRs went red inside a few minutes. The common failure was
`check_silicon_coverage.sh`, which #3299 had already fixed on main — 18 of 34
open PRs were simply cut before `ab7619ff5` and read `PROMOTABLE 1`. The same
shape as the 2026-09-14 lesson: a main-level fix leaves every PR stale.

Seven of those could not even be updated. The measured conflict surface:

| File | PRs |
|---|---|
| `docs/roadmaps/roadmap.yaml` | 4 |
| `.github/workflows/ci.yml` | 3 |
| `docs/audits/impl-estimates.jsonl` | 2 |
| `lib.rs`, `Makefile`, `tree_reader_tests.txt` | 1 each |

`impl-estimates.jsonl` should never conflict — it is strictly append-only, and
#3256 declared `docs/audits/*.jsonl merge=union` weeks ago. **It conflicted
anyway: a `.gitattributes` merge-driver declaration is retroactively inert.**
Git reads merge attributes from the side being merged *into*, so a branch cut
before the rule landed never sees it. `agent/T-2` is the measured case — its
`.gitattributes` lists only the roadmap rule, `768e740b1` is not an ancestor,
and the merge conflicts while `git merge-file --union` on the same three stages
returns rc=0, 50 rows, 0 markers.

Fixed in `ci_resolve_dirty.sh`, which already registered the roadmap driver per
invocation: bind `union` the same way through `core.attributesFile`, the lowest
-priority attribute source, so it applies only where the branch has no rule of
its own and nothing shared is written. #3271's merge, which had failed minutes
earlier, then went rc=0 with all three files auto-merged.

### Shipped this interval

| Item | State |
|---|---|
| #3307 — clean-room fix + guard + baseline (closes #3305) | open, auto-merge armed, updated onto main |
| #3309 — union binds on older branches (closes #3308) | open, auto-merge armed |
| #3305, #3306, #3308 | filed with measurements |
| #3297 — missing `ont-delta:` line | fixed, verified against the guard's own vocabulary |
| #3005, #3271, #3278, #3281 | pushed out of CONFLICTING |
| #3205 #3238 #3246 #3249 #3259 #3265 #3270 | updated onto main |

### Instruments added

| | |
|---|---|
| `scripts/check_pathonly_devdeps_unused_in_src.sh` | case table 4 must-match / 7 must-not-match; red on the pre-fix manifests AND on a stale baseline row; 3.1 s bare; 0 bashrs errors |
| `ci_resolve_dirty.sh` row 9 | both polarities — must conflict without the override, merge with it. Mutation-verified: blanking the override turns it red (8/9), restoring it green (9/9) |

### Still open

- **#3134, #3245, #3248** — genuine `.github/workflows/ci.yml` content conflicts.
  Not the mechanical class; they need their authors' intent, so they are not
  being force-resolved.
- **#3041** — deliberately not updated. Operator ruled it gets split, not
  resolved; updating it would burn CI on a branch that is going to be taken apart.
- **#3306** — 27 baselined sites. Three crates version cleanly; `aprender-core`'s
  `entrenar` (19 sites) and `renacer` (3) are publish cycles whose test code must
  move out of `src/`, and `ci.yml` runs only an explicit list of `--test` targets,
  so a moved file is dark until it is added to that line.
- **Lane B** — contract rows undischarged; PR-1 (wire `contract_tests` into
  `ci.yml`) not yet opened.

### Not claimed

- The A1 half of #3189 — "the clean-room gate cannot pass on a release commit at
  all" — remains **unreproduced**. #3305 explains all 11 observed errors, and
  none of them is release-shaped. Whether a release commit fails for an
  additional reason is only settled by `make clean-room-p1` at a release-shaped
  commit (`CARGO_BUILD_JOBS=2`, doctests `--test-threads=4`). Until #3307 lands
  and the gate runs green once, that question is open.
- `pack:` stays `[U]` — §5 P0·Instrument has not written ledger records.

---

## Interval — 13:45Z

```
scope: Lane B PR-1 and PR-2 both open · 355 dark tests wired · 19 obligations named
queue: BEHIND 0 · DIRTY 1 (#3041, split by ruling) · everything else on CI
pins:  unchanged
```

### Lane B, both ordered items

**PR-1 — #3312, `contract_tests` wired into the gate.** The premise checked out
and then some. `crates/aprender-core/tests/contracts/` holds **68 modules, 355
`#[test]` functions, 66 `proptest!` blocks**, and `ci.yml:570` — the one line
naming explicit `--test` targets — lists 38 of them and not this one. CI runs
`--lib` across the workspace, which never reaches an integration target under
`tests/`.

The reason it could stay dark for so long is that **the hand entry point was
broken too**:

```
$ cargo test --test contract_tests --no-run
error: no test target named `contract_tests` in default-run packages
help: available test in `aprender-core` package: contract_tests
```

`make contract-test` ran exactly that, with no `-p`. Nobody could run the suite
by hand either, so nobody noticed it was not running in CI.

The suite is healthy — `300 passed; 0 failed; 55 ignored` — so wiring it does
not red-line main. RED evidence through the exact command ci.yml now carries:
mutating softmax's sum-to-one contract to sum-to-two gives
`299 passed; 1 failed`, `exit=101`.

**The 55 ignored stay ignored, and that is the correct call.** Each carries a
reason and an EMPTY body — placeholders for obligations whose code does not
exist ("Gated Delta Net not implemented"). Deleting the attribute would make all
55 pass *vacuously*, which is strictly worse than skipping and is the same
enforcement-theater class the PR closes. Reported as
`300 evaluated, 0 failed, 55 blocked-on-unimplemented`.

**PR-2 — #3316, the 19 obligations get names.** 14 of 19 were anonymous, and
four kani harnesses pointed at ids that do not exist (`QHF-INV-001`,
`QHF-INV-002`, `QE2E-INV-001`, `QE2E-ORD-001`). `pv validate` said
`0 error(s), 0 warning(s)` for all three contracts; `pv proof-status` counted
`9 kani` against `7 obligations` without noticing two of the nine attach to
nothing.

The 21 that *did* resolve were worse than they looked — they matched the
obligation's `property` **prose**. A proof bound to a sentence unbinds itself
the moment anyone rewords the sentence. All 25 now cite an id.

One judgement call is stated rather than buried: `KANI-QHF-001` asserts "Shape
preservation through hybrid block", the conjunction of three per-sublayer
invariants that no single obligation states. Pointed at `QHF-INV-001` —
understating what it verifies, overstating nothing. **Decision request:** if a
block-level obligation was intended, it needs writing.

### The class is repo-wide, and that reframes #3091

Scanning all 876 contracts carrying obligations or harnesses:

| | count | share |
|---|---|---|
| ≥1 **anonymous** obligation | **824** | 94% |
| ≥1 **dangling** kani reference | **287** | 33% |
| dangling references in total | **586** | — |

"0/17 bound" was never a Qwen problem. It is the visible corner of a repo-wide
one, and `pv validate` reports `0 error(s)` for every instance. Filed as #3314
with the guard, the `pv` fix, and the ratchet; #3315/#3316 is the 0.68 slice
only, per the ruling that 0.68 does not widen its scope.

### The merge path, continued

Three PRs could not be updated, all conflicting on the **same single line**:
`ci.yml:570`, a ~4000-character string holding all 38 `--test` targets. Every PR
adding a target must edit that one line, so two such PRs always collide — the
serial fraction #3297 removed from `roadmap.yaml`, still present here. It
compounds: a new test file is dark until added to that line, so the file every
new target must touch is also the one most likely to conflict.

Every collision was append-vs-append, resolved as the union with main's order
authoritative. A conflict whose resolution is always the same is a merge driver
nobody wrote. Filed as #3313 with the fix: make it a sequence, one target per
line.

### Backlog

| | start of run | now |
|---|---|---|
| DIRTY (cannot update) | 7 | **1** — #3041 only, split by ruling |
| BEHIND | 18 | **0** |
| open non-draft | 34 | 34 |

### Shipped this interval

| Item | State |
|---|---|
| #3312 — 355 contract tests wired + broken Makefile target (closes #3311) | open, armed |
| #3316 — 19 obligations named, 25 harnesses cite ids (closes #3315) | open, armed |
| #3311, #3313, #3314, #3315 | filed with measurements |
| #3134, #3245, #3248 | ci.yml union, pushed out of CONFLICTING |
| 9 further PRs | updated onto main |

### Still open

- **#3041** — deliberately untouched. Split, not resolved.
- **#3314** — 824 anonymous / 586 dangling. Needs a scope ruling: baseline all
  287 contracts now, or drain by area.
- **`crates/aprender-contracts-staging/contracts/`** — holds stale copies of two
  0.68 contracts; its `gated-delta-net-v1` already differs from `contracts/` by
  88 lines at `origin/main`. Not a mirror anyone maintains, and a hazard if
  someone "reconciles" the wrong direction. Deletion candidate, not actioned.
- **A1 (#3189)** — still unreproduced, and cannot be settled until #3307 lands
  and clean-room runs green once.

---

## Interval — 14:45Z

```
scope: GDN 0/5 discharged — 5 obligations, 5 tests, ALL 5 #[ignore]d, empty bodies
       #3303 CPU parity ref filed 12:14Z, not started · #3114 still draft
       andon 2h06m elapsed at last report, 69h53m to §8 at 2026-09-18T12:19:39Z
queue: BEHIND 0 · DIRTY 1 (#3041, split by ruling) · 7 PRs open and armed
pins:  unchanged
```

`pv proof-status` reports `gated-delta-net-v1` at **L4 — "5 tests, 7 kani, 4 lean
proved"** — for a contract whose every executable test is an empty `#[ignore]`d
stub asserting nothing. pmat#1369's denominator problem surfacing as a *level*,
not just a count. Nothing is discharged; the number says otherwise.

### gx10 — reclaimed, root-caused, ticketed

73 GB freed (`df avail` **8 GB → 84 GB**, now 167 GB). Free space was at 8 GB
when the reclaim began, having fallen from 65 GB while I measured it, so jobs
were hitting ENOSPC live.

None of the three suspected causes held: the timer is active and hourly, the
service runs `User=root`, and the path is right (same inode, `66306:34610110`).
The sweep had **never run** — 428 consecutive `SKIPPED` in seven days, every one
logged `ok:`. `sweep_shared_registry_src()` returns early whenever any job is
live, which is correct for the genuinely shared registry it was written for
(infra#509) and absolute on a CI host when pointed at a *per-PR* tree that is
attributable. paiml/infra#612, now four parts: split by tree shape, budget cap,
**N consecutive skips exits non-zero**, and a per-run ledger record so this
appears in `build-report` instead of at 8 GB free.

### Contract obligations — named (#3320)

3,612 anonymous obligations across 823 contracts, by a committed, idempotent
generator (`scripts/lib/obligation_ids.py`). The 52 already-named files are
skipped whole — measured 0 modified. Ownership resolved across the whole corpus
so the id space is global from day one: **0 ids claimed by more than one file.**

Two bugs, both caught by assertions rather than review, both the same shape — a
rule that silently did nothing:

- 83 of 823 contracts **indent** their list items. The first cut matched only
  column 0, reported those files as named, and changed nothing; `--check` found
  410 obligations still anonymous.
- collision detection counted only *computed* prefixes, so it could not see that
  an untouched file already owned one, and minted `AL-BND-001` in two files.

The ruling's order is inverted deliberately: `validate_contracts` walks every
contract and is wired at `ci.yml:570`, so arming `pv` first red-lines main on
3,612 rows. Naming lands first; `pv` strictness is the next PR and is this one's
RED evidence.

### Two guards caught my own work

- **#3307** — the new baseline was unclassified and declared no instrument.
  `check_baseline_ratchets.sh` and `check_tool_versions.sh` both refused it.
  Fixed: `# tool_version=none` with its derivation, and registered as kind `set`.
- **Five PRs** — `check_pr_closes_issue.sh` refuses a `Refs #N` with no
  `no-close:` reason. That is the guard working exactly as intended; every body
  now carries one, verified by running the guard against each.

Neither was visible to me before CI said so, which is the argument for both.
