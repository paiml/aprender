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
