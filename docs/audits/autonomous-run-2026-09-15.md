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

---

## Interval — 16:05Z

```
scope: GDN 5/5 discharged, mutation-verified (#3322) · 305/0/50
       #3303 BLOCKED — llama.cpp at pin 39173bcac has 0 lines of `qwen35` in src/ (16 of
       `qwen3next`); apr refuses rc=8 on both backends. Neither side can produce a logit.
       Needs a decider on B3 (bump / second pin / re-convert) and B5 (known-bad model).
       andon ~66h to §8 at 2026-09-18T12:19:39Z
queue: BEHIND 0 · DIRTY 1 (#3041, by ruling) · 11 PRs open and armed · 0 merged since 11:09Z
pins:  unchanged
```

### P0·Instrument — two rules recorded, per ruling

**Sampling window ≥ p95 `ci / gate` duration.** A 3-minute window read a
healthy fleet as stalled: 0 completions inside it, because no aprender CI run
finishes in 3 minutes. Measured properly — 17 runs in 30 min, 29 in 60 — the
same fleet was draining at ~29/hr with 17 intel workers at load 62. Any
throughput sample shorter than one run's p95 measures nothing but its own
window.

**"main moved → every PR stale → burst" is the cost of BEHIND=0 without group
batching.** Main advanced at 12:49Z; every open PR went stale; bringing 18 of
them back to BEHIND=0 plus 9 new PRs produced ~50 queued runs and 4.5 h with
nothing merging — not a stall, the price of the discipline paid all at once.
Group size 8 in the ruleset after PR3 is what removes that burst: one batch
absorbs the update instead of eighteen runs paying for it separately.

### Ruling 1 — the ids are now a fact, not a text edit (#3327)

`ProofObligation` had no `id` field and no `deny_unknown_fields`. Every id
#3320 wrote was dropped on parse — decoration, by this repo's own definition.
`pub id: Option<String>` plus `pv validate --check-ids`, run with one binary
against both trees:

| | `main` (un-named) | #3320 (named) |
|---|---|---|
| obligations | 3764 | 3764 |
| **with id** | **161** | **3750** |
| referenced | 34 | 117 |

**Reconciled, not rounded.** The generator says 3,773; pv says 3,750. The 23
are five files under `contracts/entrenar/kaizen/`, which pv's walker excludes
by directory name at any depth. Replicating that walk gives `3764 / 3750`
exactly. Round-trip hazard checked and cleared: `pv unlock` writes through
`serde_yaml::Value`, so unknown keys survive — and the PR body says so, so
nobody "improves" it into a typed round trip. Two tests lock the field in,
mutation-verified (rename the serde key → red).

### #3320 — the census gate caught a real consequence of my naming

`real_corpus_census_names_every_baselined_stem` went 48 → 55. Eight pairs
`contracts/X.yaml` ↔ `contracts/aprender/X.yaml` were byte-identical at main —
one logical contract in two places — and the generator treated each copy as a
separate claimant, giving the second a path-keyed suffix. Identical pairs became
divergent pairs, differing only in ids. The gate counted them exactly as it
should. Byte-identical candidates are now one claimant; 48 == 48 again. What
"globally unique" means after this, stated precisely: an id names one *distinct*
contract — 38 ids appear in two files, and in every case the files are
byte-identical.

### Ruling 3 — #3326

The C14 presence glob is case-insensitive **and** a mismatch fails loudly: a
case-insensitive hit returns rc=3, the caller prints `NAME-MISMATCH` and sets
rc=1, because a registry disagreeing with its artifact is a defect and measuring
it silently is how it sat. The resolution was extracted into a function so the
case table could reach it at all; the fixture is deliberately mixed-case, since
a row in the registry's own casing cannot see the defect. 12/12 rows,
mutation-verified.

### Two things that were not this run's fault, and one that was

- **#3328** — `aprender-gpu`'s test binary **SIGSEGVs** on gx10 on identical
  Rust: `gpu-quick` passed on #3307 at 14:19Z and crashed at 15:13Z with only
  shell scripts changed between. Not charged to the PR; ticketed with the
  schedule that would name the dying test.
- `guard-tree` on #3312/#3316 was the frozen-payload class from the previous
  interval; the post-reopen run on #3316 is green.
- **#3307 lacked an `ont-delta:` line** — it was the first PR I opened, before
  the guard taught me the rule on #3297. Added, reopened for a fresh payload.

### Open

| | |
|---|---|
| #3307 | reopened; gpu-quick and guard-tree re-running on the corrected body |
| #3327 | armed; its `--check-ids` numbers are the evidence #3320 was waiting for |
| #3318 | held behind #3307, arms immediately after |
| `ci.yml:570` → file | held behind #3312 |
| pv strictness | held behind #3320 + #3327 |
| #3303 B3 / B5 | **need a decider** — the comparator baseline does not exist at the pin |

## Interval — 13:20Z (2026-09-16)

### The queue was blocked for 2 h 14 m by one PR, and every PR page said green

`main` produced **zero merges from 10:48:34Z to 13:02Z** with 8 entries queued.
Every `merge_group` build ejected on the same two required contexts —
`workspace-test` (step "Integration tests") and `gate` (fan-in, log reads
`workspace-test failed: failure`).

Root cause: **#3320** edits `contracts/setfit-encoder-conformance-v1.yaml`
(`+28 -14`) without regenerating the tolerances pinned to its hash.

```
contract @ #3320 head 67359fead : fdfaeb5ce8fbd53d93e946169d5dca9dde352eeed06aced038ee20338b46f14d
CONTRACT_SHA256 in tolerances_generated.rs : 16a6591788a6c693ad3d08845a20267e31d4a86ee663310a943c841d9e7b2b93
```

Fixed on its branch (`67359fead..8f58592a9`): regenerated, agree-test passes,
and the diff is **four lines — the sha256 comment and the constant only**. No
tolerance VALUE changed, so no conformance bound was loosened to unblock a queue.

### Why nobody could see it from the PR

The "Integration tests" step is gated `if: steps.tier.outputs.tier == 'full'`.
PRs run the quick tier; `merge_group` runs `full`. So this target is **dark on
PRs and armed only inside the queue** — #3320 read `workspace-test: SUCCESS` on
its own page and ejected three times (07:46:53, 09:05:20, 12:25:10, all
`github-merge-queue[bot]`), taking unrelated PRs batched behind it each time.
A merge_group branch is main *plus the entries ahead of it*, which is why
`pr-3302-0cd4d940…` died too. Filed as #3362. The durable half is the
asymmetry, not this contract: either the target runs on PRs touching
`contracts/**`, or the quick-tier filter treats a contract edit as relevant.

### Two wrong diagnoses I published and withdrew

- **"main is RED"** — refuted by measurement: the contract hashes identically at
  `origin/main` and in the constant, and the exact CI fragment command passes
  locally at `4b00761a3`. Regenerating on main would have baked a false constant.
- **"#3320 does not touch the contract"** — wrong because `gh pr view --json
  files` **caps at 100 entries** and #3320 changes **824** files.
  `gh api .../pulls/3320/files --paginate` found it immediately. Same class as
  a `head -14` that hid `REAPER_RUN_BUDGET_SEC` on yoga and a `head -1` that
  read the wrong cascade script. A truncated list reads exactly like a complete
  one; prove the universe before reporting ABSENT.

### The 0.68 series is NOT a stack — plain rebase, never `--onto`

Patch-id comparison: PR1 #3354 has 2 patches, PR2 #3355 has 8, **2 shared by
patch-id**, both from base `d83592af8`. PR2 carries *copies* of PR1's commits,
it does not contain PR1, so `rebase --onto <PR1-tip>` would be wrong. All three
rebased with a plain `git rebase origin/main`:

| PR | was | now | note |
|---|---|---|---|
| #3354 | `815e511df` | unchanged | fully green, queue position 7 |
| #3355 | `bf796cc5c` | `47f9ddc91` | `guard-tree` PMAT-3318 violation was pure staleness; rebase alone fixed it |
| #3356 | `839b56137` | `56d54336c` | 511 files / 185,679 insertions preserved |

PR3's local worktree held a *diverged* branch — 31 ahead, 33 behind, only 26
patches shared. Its 5 "local-only" commits were **already-merged main commits**;
the remote's 6 were real content. Force-pushing local would have destroyed six
commits. Remote was authoritative; git then dropped three of those six itself as
"patch contents already upstream", and two more were superseded drafts (main's
`LOAD.md` **corrects** `b97af81a3`: `-no-cnv` was never accepted, "it did not at
the old pin either").

**PR2 and PR3 are deliberately NOT armed** — they hold copies of PR1's patches,
so letting either merge first would leave PR1 empty.

### Rulings closed

- **`REAPER_RUN_BUDGET_SEC`** — was declared in all three units on infra main
  (gx10:98, yoga:62, intel:93) but **absent from every loaded unit**: the
  on-disk file had changed without a `daemon-reload`, so the reaper was taking
  420 from the script default — exactly what the ruling forbade. `daemon-reload`
  on gx10, yoga and intel; all three now report `REAPER_RUN_BUDGET_SEC=420`
  in the loaded environment (count 0 → 1 on each).
- **T-3/T-4 chain** — verified complete, no work needed. infra#621 landed the
  `ref` input (`clean-room.yml` `workflow_dispatch.inputs.ref`, "Commit sha
  (full 40 chars) or tag"), `assert-tested-ref.sh` runs at `:282` (self-test)
  and `:291` (the assert), `tested_sha` is stamped at `:339`; and
  `cascade-publish.sh:321` refuses to start unless clean-room is green on
  exactly the tag's commit, fail-closed, "neither → REFUSE".
- **ms 0.68** — 2 issues (#3091, #3208) + 4 PRs = 6, matching the revised §6.

### `pack:` — the 0.8 rule is a false positive unless the cap is right

Raw busy/online reads yoga 3/5 and gx10 4/6, which trips §1's 0.8 rule and would
raise a §8 stop two wakeups running. `yoga-gpu` and `gx10-blackwell` are
GPU-reserved **host** runners that cannot take `X64,clean-room` work; excluding
them gives yoga 3/4 and gx10 4/5, and both boxes were demonstrably serving
aprender (`guard-tree=yoga-build2`, `guard-cargo=gx10-build`,
`vendored-schemas=intel-clean-room-2`). Occupancy cap = only runners that CAN
take the job class. Ledger record in #3363.

Also corrected: `workspace-test` is **not** X64-pinned. Main reads
`runs-on: [self-hosted, Linux, clean-room]` (PMAT-3138/#3138) and gx10-pool1/2
served it successfully at 11:38/11:43. The three groups landing on intel is
scheduling chance, and it costs ~3× (gx10 10.6 min vs intel 34.5 for step 1).

### Shipped this interval

| | |
|---|---|
| #3320 | **fixed by me** — regenerated tolerances, pushed; this is what unblocked the queue |
| #3360 | H1 SIMD speedup assertion no longer fails a debug-build required check; release still enforces; mutation proof RED |
| #3361 | P0·Unwedge — superseded-head + aged-with-idle-capacity, capacity host-side, `cancel-in-progress` narrowed to `pull_request` |
| #3363 | pack ledger record + the cap correction above |
| #3362 | the merge_group-only visibility asymmetry |
| #3359 | wall-clock assertions in a required check |

### Still open

| | |
|---|---|
| #3354 | green, queue position 7; PR2 arms only after it merges |
| #3355 / #3356 | rebased, CI running, **intentionally unarmed** |
| #3114 | needs the series + #3360 before it can leave draft |
| #3320 | `workspace-test` pending; **I will re-arm it** — I asked its author to hold |
| #3361 | its own CI run has been `pending` with **0 jobs** for 11 min — the exact mechanism it fixes. Not hand-cancelled. |

### Not claimed

- `gpu-quick` on #3361 exits 101 with **no** `test result:` line and no `FAILED`
  — the binary aborted rather than reporting. It is SUCCESS on #3360 and #3302,
  so it is not broad. This looks like the same fault as **#3328**
  (`aprender-gpu`'s test binary SIGSEGVs on identical Rust), already recorded in
  this log. Not charged to #3361, not asserted as the same bug without the
  dying test's name.
- Three runs have sat `queued` with **0 jobs since 2026-09-13** (ages 4571,
  4575, 4583 min); every other pending run carries 13–18 jobs. All three are
  superseded (`8633da345` vs #3200's `7b452797f`; `d3571663f` vs #3202's
  `9b4b0595e`; a `push` run whose head is an ancestor of main). Left alone —
  operator ruling is no hand-cancels; they are rule-1's first real targets.

## Interval — 14:05Z (2026-09-16)

### What moved

- **#3278 unwedged.** Sat at merge-queue position 1 with a `roadmap.yaml` entry and no `entries/PMAT-3231.yaml`; guaranteed to eject every batch behind it. GitHub refuses pushes to a queued PR's branch ("protected branch hook declined"), so the fix waited for the queue's own ejection at 13:52Z, then `roadmap_fragments.py adopt PMAT-3231` + `make roadmap-aggregate`, guard PASS on committed refs, pushed `eb883a2f2`, re-armed; re-entered at position 8. Finding: `roadmap_trim.py` run after `adopt` re-introduced DRIFT against the aggregator — on fragment branches the aggregator is the last writer.
- **#3341 ruled and measured.** First bisect used `apr chat`; step 2 showed CPU `apr chat` never enters the MoE path for any qwen3_moe GGUF (Q4_K_M fails byte-identically, rc=0). Corrected on the issue; re-measured with `apr run`: Q4_0 → `UnsupportedOperation('moe_expert_matvec', qtype 2)` at v0.66.0 and v0.67.0, Q4_K_M control generates. Verdict NEVER WORKED → 0.69.0. Mechanism: `matvec_for_qtype` has Q4_K/Q6_K arms only; step 3 (load-time contract) dispatched in slot 3 on `PMAT-3341-load-time-contract`. Side defects filed: #3367 (apr chat architecture-blind + rc=0), #3368 (placeholder qtype=0 message misdirects).
- **#3320 re-armed** after its workspace-test went green on `8f58592a9`.

### Claims (other sessions)

- **apex-ca** (APEX-001, `~/src/apex`): **#3259** (APEX-2b) and **#3273** (APEX-2a). Will push #3259 only after #3270 leaves the queue, #3273 only after #3259 merges; will not touch #3270's branch while queued.

### Still open

- Series: #3354 position 6 (groups for positions 1–3 rebuilt 13:54Z, queued for runners); #3355/#3356 held; #3114 undraft after PR3 + #3360.
- #3361 (unwedge rule 1) green, unarmed; rule 2 pending. #3363/#3364 green, held so they do not queue ahead of PR2.

## Interval — 19:15Z (2026-09-16)

### What moved

- **Critical path**: PR1 #3354 merged 16:59Z; #3360, #3278, #3320, #3361, #3281, #3238 merged between 15:56Z and 17:55Z (eight merges, main `2fb79ff0b`). PR2 #3355 rebased onto main at 18:49Z (two copied PR1 patches dropped by the rebase) and **ARMED 19:03Z** through `arm_pr_automerge.sh` once guard-tree was green on `e8e1157b5`. PR3 #3356 waits DIRTY for PR2's merge.
- **STOPPED 15:23Z–15:29Z (§8)**: a PMAT-544 fragment was inserted above the shebang of `~/.claude/hooks/subagent-lock.sh`; `/bin/sh -c` ran it under dash, `set -o pipefail` exited 2, and the harness read DENY for Bash/Edit/Write/Agent/SendMessage/Workflow in all ten sessions. Operator repaired by hand. Ruling: `~/.claude/hooks` is shared state — never edited in place; versioned + forjar install with a `/bin/sh -c` self-test. Recorded in memory.
- **Repo corruption after the usage-limit stop**: 12 empty loose objects in the shared `.git` plus an `ORIG_HEAD` naming a lost commit; every fetch failed. Removed, fsck clean, fetch restored.
- **Self-inflicted, caught by the fragment guard**: after the squash churn ejected every open PR DIRTY, I resolved `roadmap.yaml` by running the aggregator over the CONFLICTED file, which keeps the `<<<<<<<` marker; pushed one on #3396, staged one on #3366. The guard reported it as a phantom `CHANGED PMAT-3347`. Fixed by regenerating from `origin/main`'s copy; lesson appended to the squash-merge memory.
- **#3366 (Alfredo, 0.68)**: branch-owned reds fixed (8 pipe-into-`grep -q`, bare `apr` in the help heredoc), fragment added, advertised URL moved to the raw GitHub path (paiml.com/apr/install.sh is 404), pr-review v2 receipt for `d1d510bba` signed + guard-ACCEPTed (17 advisory findings), merged from main (not rebased, so the reviewed sha stays in history), re-armed 19:09Z.
- **Unwedge**: #3361 (rules 1+2, oracle collector fixed off `pgrep -f`) merged 17:33Z; #3403 (rule 3 orphan-group cancel + arming precondition helper) armed 19:07Z and is queue position 1.
- **#3341**: NEVER WORKED on `apr run` at v0.66/v0.67 → 0.69.0; load-time contract on `PMAT-3341-load-time-contract` (verified, PR pending); #3367/#3368/#3369 filed; #3396 (chat exit code) armed 19:11Z.
- **MINI**: probe workflow PR #3404 (never dispatched); infra#645 (runner prerequisites via forjar); aprender#3402 (0.69 macos-arm64 leg + install.sh Darwin). mini-m4: `self-hosted,macOS,ARM64,apple-silicon,m4,mini`, 16 GiB, 0 of 12 PR jobs eligible today (all carry `Linux`).
- Spec §4 T-2/T-3 rows amended in place for the installer.

### Still open

- PR2 → PR3 → #3114 undraft → #3091. Ruleset 17836320 still build 3 / merge 1 at 18:47Z.
- #3364 (this log) and #3363 held unarmed; #3341 contract PR to open after the series.

## Interval — 21:10Z (2026-09-16)

### What moved

- **PR2 #3355 merged 20:57Z** (main `f326f5c43`). PR3 #3356 rebased `--onto main 8341d6d16` (8 copied PR1/PR2 commits dropped, 18 own kept) and pushed; arms via `arm_pr_automerge.sh` when guard-tree is green.
- **intel hard-reset twice** (boots 18:45Z and 19:30Z; previous boot's journal ends mid-activity at 19:28:51Z, no shutdown sequence). The 19:30 reset killed 16 jobs across 6 runs at 19:38Z — PR2's group, #3396, #3364, main's post-merge run — while yoga jobs finished. Live readings afterwards: 67 °C, 16 G / 283 G used, load 62/32 (intended), 0 MCE, PL1 150 W. Cause unknown from here; not the reaper (timer fired 20:04Z). Re-ran the three PR/push runs; main green again (attempt 2). Groups rebuilt themselves.
- **#3366 (Alfredo, installer) merged 20:16Z** with the signed pr-review receipt for `d1d510bba` in tree.
- **#3396** was UNMERGEABLE in the queue after the squash churn; `dequeuePullRequest` (own PR, no run cancelled), aggregate rebuilt from main's copy, re-armed 20:31Z on `f7d3990f2`; queue position 1 at 21:09Z.
- **Merged this interval**: #3403 (rule 3 + arming helper) 19:28Z.

### Still open

- PR3 → #3114 undraft → #3091. #3405 (#3341 contract), #3404 (mini probe), #3364 (this log), #3363 open. Ruleset 17836320 still build 3 / merge 1 at 20:10Z.

## Interval — 22:30Z (2026-09-16)

### What moved

- **Series complete.** PR3 #3356 merged 22:23Z (main `0ac2aab8c`); PR1 16:59Z, PR2 20:57Z. **#3114 undrafted 22:15Z** on a fully green head (`71b913fdf`: merge from main, one evidence path made portable for `check_hardcoded_paths.sh`) and armed through `arm_pr_automerge.sh`; queue position 2 behind #3404 at 22:19Z. #3091 closes on its merge.
- **Site event, measured**: gx10 rebooted 19:29:51Z and intel 19:30:25Z — thirty seconds apart, so not per-box. gx10 came back with its declared NM profile (`fb7c4cce…`, autoconnect=yes) INACTIVE on `enP7s7`: a stray 10.42.0.15/24, no default route, no DNS. Its four ephemeral runner units (`github-runner-{ephemeral,build,pool1,pool3}`) crash-looped 797× on `curl: (6) Could not resolve host: api.github.com`; gx10 offered 1 listener (gx10-blackwell) instead of 6 and #3405's `gpu-quick` sat queued 2.5 h. `sudo -n nmcli con up <profile>` (§3.5 unclogging, restores declared state) → default route + DNS in 4 s; units and containers recovered on their own; gpu-quick ran green on gx10-eph 21:47Z. The forjar-side fix (autoconnect that does not autoconnect after a boot) is infra's. **mini is DOWN** (no route to host from the LAN; gx10 on the same LAN answers). yoga never rebooted (up since 09-12).
- **Unwedge rule 2 exercised for real**: the scan with a live capacity reading correctly declined the queued run (17/18 jobs complete, one GPU job waiting on a real pool outage) — a queue, not a wedge.
- #3405 (#3341 contract): guard-tree failed twice on `pp066_v16_defects.sh --v15-red` (passes locally, unchanged vs main — runner-environment), then on a stale aggregate (REMOVED PMAT-3365 = generated before two later merges) and PR-body `no-close:` lines; all fixed, rebased to main+2, armed 22:19Z. #3404 (mini probe): grep -q ratchet + PR-body no-close fixed, armed 22:10Z.

### Still open

- #3114 in queue; #3404, #3405 in/entering queue; #3364 (this log), #3363 held. Ruleset 17836320 still build 3 / merge 1 at 22:10Z.

## Interval — 23:10Z (2026-09-16) — T-0 cut

### What moved

- **#3114 merged 22:58Z** (main `e822e5adb`); #3091 and #3303 closed. The §1.6 critical path is complete 37 h before the andon.
- **Freeze (§4.1)**: #3208 (Q8_K activations + f16 KV parity) → 0.69.0 with `slipped_from: 0.68.0`; milestone 5 reads 0 open / 87 closed.
- **T-0 (§4)**: `rel-068-autopilot/prepare_bump.sh` — worktree at `e822e5adb`, `bump-version.sh 0.68.0` + `--check` green across every workspace incl. facades, CHANGELOG `[0.68.0]` from the 89 PRs merged since `v0.67.0` (Added 11 / Fixed 32 / Changed 46), summary paragraph: Qwen 3.5 on the CPU first, the one-line installer second. **Bump PR #3406**, armed through `arm_pr_automerge.sh` once guard-tree is green (loop in `ship.log`).
- **Autopilot launched** (`autopilot.sh 3406 wait close`, pid in `rel-068-autopilot/`): steps `wait deep dogfood tag cleanroom assets preflight cascade install hosts close`. Derived from the 0.67 driver with today's rulings folded in: T-1 as a local deep run (no `ci / deep` workflow exists on main; `--no-default-features` records the standing #3176 class inside aprender-distribute and is RED for anything outside it), T-3 dispatches `paiml/infra clean-room.yml -f repos=aprender -f ref=v0.68.0` (infra#621 ref input, verified present) and records the run id, T-2 gains `install.sh --version v0.68.0` receipts on intel (x86_64) and gx10 (aarch64) from the tag's raw URL, T-4 automated to close (operator 2026-09-13), ledger record written at close for a docs PR.
- Also merged: #3404 (mini probe workflow) 22:57Z. #3405 (#3341 contract) in queue.

### Still open

- #3406 → autopilot. #3405 in queue. #3364 (this log) and #3363 to arm after the release commit is fixed. mini down; gx10 network repaired 21:47Z (forjar item open in memory).

## Interval — 03:10Z (2026-09-17) — T-2 stop and restart

### What moved

- **T-0**: bump PR #3406 merged 00:54Z → release commit `49fe9155a`. **T-1 GO** 01:05Z (980 doctests, examples, `--no-default-features` all rc=0 — the standing #3176 class is gone on this tree).
- **T-2 NO-GO on `49fe9155a`** (`dogfood-pre-publish-49fe9155a-NOGO.log`): `check_perf041_marker.sh` RED — `evidence/perf041/lambda/marker.json` 7.3 d old vs `witness.max_age_days=7`. Root cause: the sanctioned producer (`cuda-nightly.yml`, gx10) has failed on every run since 09-12 with the standing Blackwell c=16 slot-invariance defect **#3096** (0.70; intra_agree_to=31 < 64 at c=16, c≤8 invariant); nothing refreshed the marker after 09-09. v0.67.0 (cut 09-13) passed on that same lambda marker while fresh.
- **Fix, precedented (0.67: post-bump fix PRs)**: lambda (RTX 4090, sm_89) PP-26 witness on the release commit with a `--features cuda` build proven by the release lane's bytes test (`libcuda.so`=1): c=1/4/8/16 all intra-invariant to 128, no frozen slot, exit 0. **PR #3407** (marker + witness + CHANGELOG *Known limitations* naming #3096) merged **02:41Z** → **release commit `27f070324`**. Autopilot for #3406 stopped by the orchestrator after the RED row (rc=143 recorded), relaunched `3407 wait close` (pid 1117699). **T-1 GO again 02:50Z**; T-2 running 02:50Z with all 7 declared gates OK.
- Side defect filed: **#3408** — `apr devices` prints `cuda unavailable reason=NotCompiled` on a CUDA build that serves on the GPU (0.69).
- Merged this interval: #3363 (pack ledger), #3364 (this log's earlier intervals) 03:00Z.

### Still open

- T-2 verdict → T-3 (tag, release, clean-room dispatched on `v0.68.0`, assets) → T-4 (cascade, install, host + installer receipts, close). Ledger record at close → docs PR.

## Interval — 05:30Z (2026-09-17) — second T-2 stop, three root causes

### What moved

- **T-2 NO-GO #2 on release commit `27f070324`** (03:15Z): three RED rows, none of them a PR-level required check —
  1. `pv-contracts`: `contracts/external-corpora.yaml` (ONT-1 census declaration, #3281) is not a KernelContract; `pv validate` had no kind for it → **#3410** `ArtifactKind::ExternalCorpora`, one struct shared with `pv census`, 1841 contracts 1→0 failing; merged 05:13Z.
  2. `pmat-comply` CB-200: 604 below B vs baseline 602 — measured real debt under the same pmat 3.40.2 (baseline commit reads 602); roster diff: `forward_qwen35.rs::from_model_and_layers` (C+, #3114) → A+ by extraction, plus two C++ `main`s of the vendored llama.cpp reference harness under `evidence/` → `evidence/**` excluded from `[tdg]` as reference harness, baseline **lowered** 602 → 601 with its paired `scripts/cb200_baseline.txt` → **#3411** (queued).
  3. `model-parity` C14: `apr parity` is architecture-blind (dense loop) and after #3325 reaches MoE/qwen35 files — "EMPTY data buffer qtype=0" is the tool defect #3367 named, not a model defect → **#3412** pre-load refusal (exit 12, raw arch tag, never the normalizer's fold), C14 reports `UNMEASURED-TOOL (#3367 / #3090)`, crashes still FAIL; measured on lambda with a CUDA build: 3 PASS, 3 UNMEASURED-TOOL, 0 FAIL (queued).
- Autopilot relaunched `3411 wait close` (pid 2009080) — #3411 enters the queue last, so its merge commit is the release commit.
- Memory: release-phase-only gates (perf041 age, `pv validate` over all contracts, CB-200 pair, C14) must run on `origin/main` BEFORE the bump — 0.67 ×3 and 0.68 ×2 NO-GOs after the cut are the same shape.

### Still open

- #3412 → #3411 → T-1 → T-2 → T-3 (clean-room on the tag) → T-4 → close → ledger record.

## Interval — 07:00Z (2026-09-17) — pass 3, one row; rulings applied

### What moved

- **T-2 pass 3 on `5d95ed54e` (06:18Z): NO-GO on ONE row** — `pmat-comply` CB-200 602 vs 601. Roster diff (pmat's own comply index, release tree vs the CB-200 fix tree): the single addition is `crates/apr-cli/src/commands/parity_03.rs::run`, taken from B to B- (cyc 11) by #3412's refusal call site. **#3415** extracts `refuse_unroutable` + `head_geometry` (both A+); `run` → B+ (cyc 7); the tree's sub-B count is 601 with both baseline lines untouched at 601 — **the 602nd site is removed, nothing restamped** (operator ruling 1, verified before merge). Every other T-2 row was green on pass 3: `pv-contracts` 1841/1841, `model-parity` reports `UNMEASURED-TOOL` for the MoE/qwen35 rows, perf041 marker fresh.
- **Operator rulings 2026-09-17, applied**: (2) §4 T-0 now says the T-2 preflight runs on `main` HEAD *before* the bump PR opens, the bump PR is refused at arm time without a GO receipt for its parent sha, and the release commit moves zero times — from 0.69; (3) §4 T-2 + §7: a nightly feeding a T-2 row that fails ≥2 consecutive runs auto-opens an issue on the next milestone and reports `NIGHTLY-RED` — mechanism ticket **#3416** (0.69); first entry: `cuda-nightly.yml` 5 consecutive fails → #3096 (stays 0.70); the ledger record now carries the PP-26 witness provenance (lambda sm_89) beside the red producer. **T-4 for this train is the operator's**: the driver is re-pointed to stop after the cascade dry-run receipt (`to-step=dryrun`, pid 2553564).
- Merged: #3410 05:13Z, #3412 05:44Z, #3411 05:44Z, #3414 06:23Z.

### Still open

- #3415 (queued, group on intel) → T-1 → T-2 pass 4 → T-3 (tag, clean-room dispatched on the tag with HEAD==tag asserted, `apr-*` assets, install.sh rows on intel + gx10) → T-4 dry-run receipt → STOP and report.

## Interval — 08:25Z (2026-09-17) — T-2 GO, T-3 stopped on infra, fixed

### What moved

- **T-2 pass 4 GO on `c91e065dd` (07:51Z)** — all rows green; `model-parity` reports `UNMEASURED-TOOL` for the MoE/qwen35 rows (#3367/#3090), `pmat-comply` CB-200 at the 601 baseline, `pv-contracts` 1841/1841, perf041 marker fresh (lambda sm_89).
- **T-3**: `v0.68.0` tagged at `c91e065dd` 07:51:24Z, GitHub release created, `binary-release.yml` run 35196635815 → **16/16 assets** (`check_release_assets.sh` rc=0). **Installer rows** (§4 T-2, by hand): intel rc=0, gx10 rc=0, `apr 0.68.0` from the tag's own `install.sh`.
- **T-3 clean-room RED, infra defect, driver stopped fail-closed** (07:54Z, run 35196636753): `clean-room.yml` fetched the tag by name into `FETCH_HEAD` and never materialized `refs/tags/v0.68.0`, so `assert-tested-ref.sh` — correct in itself — could not resolve the ref inside the clone while HEAD *was* the tag commit. First real exercise of infra#621 on an annotated tag; its self-test fixture had the tag locally and could not see the trap. Reproduced on aprender against GitHub on protocol v0 and v2 (neither leaves a tag ref); a local `file://` transport auto-follows, so the new fixture rows pin the production state with `--no-tags`. **infra#650** merged 08:17Z (gate: validate, arbiter, gate green). Driver resumed `3415 cleanroom dryrun`; **clean-room re-dispatched on `v0.68.0`, run 35198901863** (08:18Z).
- Operator ruling: T-4 is the operator's for this train — the driver stops after the `cascade-publish.sh --check` dry-run receipt.

### Still open

- clean-room (aprender) on `v0.68.0` → assets check → preflight → dry-run receipt → STOP and report.

## Interval — 10:00Z (2026-09-17) — clean-room through B1, B2 red, ruling: v0.68.0 GitHub-only, 0.68.1 to crates.io

### What moved

- **infra#650** (tag ref materialized) and **infra#652** (A1 overlays `[patch.crates-io]` for all 79 members at the tag; 24/24 rows, both mutation polarities; A1 on the real tag 101→0 in 7 s) merged 08:17Z / 08:53Z. Re-dispatched clean-room on `v0.68.0` (run 35202211338): **A0, A1, A2, B0, B1 PASSED — first time ever on a tag**; **B2 FAILED**: `aprender-core (lib test)` — 19 × `cannot find module or crate entrenar`. Mechanism: PMAT-955 keeps sibling dev-deps path-only (a versioned one would create a crates.io publish cycle), `cargo publish` deletes them, so the published crate's own lib tests do not compile. Identical at `v0.67.0`: standing, shipped over once. #3307's class.
- **Operator ruling (10:0xZ)**: `v0.68.0` is **GitHub-only** — tag and 16 binaries stand, `install.sh` verified on x86_64 + aarch64, no crates.io publish, one release-note line pointing at 0.68.1 (done). **0.68.1** is the crates.io release and carries the B2 fix. No dry-run on 0.68.0. T-2 preflight on main BEFORE the bump (§4 as amended), then cut 0.68.1, clean-room on its tag through B2 and past it, dry-run receipt, stop for the attended cascade.
- **B2 exposure fixed in one ticket — #3425**: measured with B2's own shape (A0 strip + A1 overlay, `cargo test --lib --workspace --no-run`): two crates, `aprender-core` (19 sites/14 tests, `entrenar`) and `aprender-test-showcase` (3 sites/100 tests, `jugar-probar`); tests moved out of `src/` to targets the published crate does not carry (contract_tests fragment 390; new fragment 410 for the showcase target); after: all 72 lib-test binaries build in published form, 0 failures. Three other path-only pairs triaged as not B2 exposure and recorded in the baseline.
- Learning filed (memory): a gate never green on its own target is not a gate — first-green proof on a real target before it may block (infra#621's assert and A1 both stopped this train on first real use).

### Still open

- #3425 → T-2 preflight on main (GO receipt for the parent sha) → bump PR 0.68.1 → T-1 → T-2 → T-3 (tag, clean-room on the tag, assets, installer rows) → dry-run receipt → STOP for the operator.

## 2026-09-17T15:30Z → 22:55Z — 0.68.1 train: T-3 clean-room chain → T-4 cascade → SHIPPED (session b59147ae)

```
APR-RELEASE-001 | did=TRAIN | train=v0.68.1 | verdict=SHIPPED
fanout:  shards 2 (B2-cpu intel, B2-gpu yoga) hosts intel,yoga | commit→tag 41 min | ratchet p95 [U] (1 record) | serial fallbacks 1 (whole chain serial; P0·Fan-out is 0.69's first PR)
nightly: NIGHTLY-RED cuda-nightly.yml (gx10 leg, red since 09-12) #3096 (0.70); witness for PP-26 = lambda sm_89
train:   step reached T-4 | skip reason none | attended min 0 (operator-authorized unattended, 2026-09-17)
build:   row none | PR aprender#3467 #3469 #3470, infra#654 #656 #658 | records added 1
gate:    p95 ci/gate [U] | max PRs/train [U] | queue p95 [U]
pack:    gx10 dark 15:31Z→22:50Z (no default route/DNS after the fleet reboot; 3 of 4 runner units crash-looping) — unclogged via the declared NM profile, infra#255 | verdict P0-UNDERUTILIZED for that window
scope:   §1.5 0.68 Qwen 3.5 merged yes | blocker none
queue:   open PRs [U] | WIP cap 13 | group size 3
pins:    lambda resolves pmat 3.40.2 vs baselines recorded under 3.40.1 (check_baseline_ratchets + complexity ratchet FAIL(instrument) locally; CI unaffected)
tree:    release commit 1661c7138 == tag v0.68.1; gate fixes on main 49c52cc05
stops:   (1) run 35238057841 void — fleet reboot; (2) B2 child-cargo overlay → infra#654; (3) B2 full exposure ×3 → infra#656; (4) b2-gpu unreachable from paiml/infra → infra#658 + aprender#3467; (5) preflight R6 shape-not-cycle → aprender#3469. All gate-side; tag never moved.
next:    0.69 — first PR is P0·Fan-out (rule 14); no scope work before it lands
```

Cascade: `publish_strict.sh`, 22:00:38Z → 22:41:36Z (2458 s; 0.67 attended was 08:01→08:43Z), 71 published + 3 facades already
live = 74/74, 11 crates retried on HTTP 429 only, zero other errors. Records: `docs/audits/release/v0.68.1/`,
`docs/build-ledger/2026-09-17/1661c7138-lambda-release-train-v0.68.1.json`.

Filed for 0.69: #3462 (TIERS is not a dependency order — 47 violations), #3464 #3465 #3466 (test hygiene the chain exposed),
#3468 (closed by #3469; the tag-independent preflight rows belong in the PR guard set), infra#255 (gx10 boot-time route assert).
Learning: every one of the six stops was a gate meeting its real target for the first time (rule 15). `--no-fail-fast` turned
three serial 1-hour discoveries into one.
