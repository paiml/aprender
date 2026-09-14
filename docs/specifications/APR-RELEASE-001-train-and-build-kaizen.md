# APR-RELEASE-001 — Release train + build kaizen

Spec for the `paiml-implement` harness (repo `paiml/paiml-implement` — not `pmat-implement`).
Drop at `~/src/aprender/docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md`.
One spec, one command, one merged PR or one shipped tag per session.

## §0 Run it

From `~/src/aprender`:

```
Implement docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md autonomously.
```

**Selector — every session does exactly one of these, first match wins:**

| # | If | Do |
|---|---|---|
| **0** | `yoga` or `gx10` is under-utilised (§1 packing rule) while intel has queue pressure | **P0, minutes not a session:** arm every green PR, route what can leave intel, reap disk (§5 P0·Pack, P0·Reap); record the `pack:` line; then continue to the first matching row below |
| 1 | ≥ 48 h since the last tag on `main` **and** no SKIPPED record for the current HEAD | run the train (§4) |
| 2 | else a §5 row whose *Done* test fails at HEAD | do the first such row, one PR |
| 3 | else the last train (shipped or skipped) has no triage record | do the triage pass (§6) |
| 4 | else | emit the §7 report, exit 0 |

Nothing in this spec asks a question. Running it ten times a day is safe.

## §1 Goal

Ship a tag every 48–72 h (`0.67 → 0.68 → …`) **on a clock, not on scope**, and keep
shrinking the wall-clock from *PR opened* to *tag published* so the clock stays cheap.
The objective is elapsed time and green trains. **Packing rule (operator, 2026-09-12, verbatim):** "these two boxes: yoga and gx10 should be always 80% full of PRs from aprender if ANY queue pressure on intel … not acceptable to have slow releases when boxes are idel". Utilisation is not the goal for its own sake; an idle GPU box next to an intel queue is lost release time and is a **P0 defect**, not a state to tolerate. Measure it every wakeup:

```
gh api --paginate orgs/paiml/actions/runners --jq '.runners[] | "\(.name) \(.status) \(.busy)"'   # busy/online per host prefix
intel pressure  = any aprender job queued, or a workspace-test running, on intel
under-utilised  = intel pressure AND (busy/online < 0.8 on yoga OR on gx10)
```

Coupling, one line:

```
max PRs per train  ≈  72 h  /  p95 `ci / gate` wall-clock      (upper bound)
```

One PR in CI at a time means gate latency *is* release throughput. Compute this on every
train. p95 is `[U]` until §5 P0 lands.

## §2 Ground truth — verify at HEAD before writing anything

| Fact | Source of truth | Mark |
|---|---|---|
| `intel` — clean-room runner, 8 concurrent, memory-bound, 3.6 TB NVMe | `infra/machines/intel/forjar.yaml` | `[V]` snapshot |
| `yoga` — CI runner, RTX 4060 8 GB, 32 GB RAM, 10G NIC needs `bolt.service` | `infra/machines/yoga/forjar.yaml` | `[V]` snapshot |
| `gx10` — aarch64 GB10, sm_121, 120 GB unified; **not** a documented general runner | `infra/machines/gx10/forjar.yaml` | `[V]` snapshot |
| `main` protected; required check literally named `ci / gate` | org ruleset | `[V]` |
| Last tag = `git describe --tags --abbrev=0` on `main`; next minor = that + 1 | git | `[V]` live |
| p95 `ci / gate`, tag→publish, cascade wall time (automated) | — | `[U]` unmeasured |
| `workspace-test` is pinned `runs-on: [self-hosted, X64, Linux, clean-room]` (#3104) — the long pole never lands on gx10; #3139 lifts the pin (795 s on gx10-pool3, 34693750990) | `.github/workflows/ci.yml` | `[V]` 2026-09-12 |
| gx10 and yoga: `/mnt/nvme-raid0 -> /home/noah/eph-work/intel-mirror`; per-PR target dirs under `targets/{aprender-ci,sovereign-ci-aprender}/<pr>`; pool runners are docker containers (`sovereign-gpu-runner:2.337.0`); no reaper existed until 2026-09-12 (gx10 hit 100 %, 146 GB reclaimed by hand, then declared in forjar) | `ssh gx10`, `ssh yoga`, `machines/{gx10,yoga}/forjar.yaml` | `[V]` 2026-09-12 |

Live `forjar.yaml` beats this table. Record the diff in the receipt and continue.

## §3 Hard rules

1. **The train is never delayed.** Cut can't go green in one attempt → train **SKIPPED**,
   reason recorded against that HEAD, no retry until `main` moves. Next eligible in 48 h.
2. **Two consecutive skips, or ≥ 72 h since the last tag with no runnable train → andon.**
   Stop, five-whys terminating in a mechanism, escalate. Do not cut a third.
3. **No invented numbers.** Every threshold cites its measurement command or carries `[U]`
   and is not a gate. No build target is set before P0 has ≥ 20 ledger records.
4. **Heijunka is bounded by the fleet, not by one.** WIP = what the fleet can build without
   starving a box: the merge queue builds 3 entries in parallel (ruleset 17836320, by design).
   **Never dequeue a PR because another is in CI**; dequeue only a group that is *known* RED
   (roadmap-additive guard, a red required check) and fix or trim it rather than park it. Under
   intel pressure, arm every green PR and prefer the ones whose jobs can land on `yoga`/`gx10`.
   Splitting *one* gate across hosts (§5 P2) is the other half of the same rule. "Parallel"
   inside a session still means ≤ 3 read-only subagents, never two sessions merging.
   *(Amended 2026-09-12 by operator ruling; the previous text said "one aprender PR in CI at a
   time" and this session dequeued four PRs on it — #3175's group was running its workspace-test
   on yoga at the time.)*
5. **No ad-hoc host CONFIG.** Durable host changes are `machines/<host>/forjar.yaml` →
   `forjar apply` → `make -C machines/<host> verify-systemd-units`. Repo edits are inert until
   deployed (2026-04-26 ENOSPC was exactly this). **SSH for measurement and for unclogging is
   expected** (`ssh gx10`, `ssh yoga`: `df`, `du`, `systemctl status`, `docker system df`,
   reclaiming a full disk that is blocking the queue right now). The reclaim and its forjar
   encoding are one piece of work: what was done by hand once is declared so the next time is
   automatic. *(Amended 2026-09-12: "you DO HAVE SSH (ssh gx10, ssh yoga)".)*
6. **Ledger is append-only, one file per run:** `docs/build-ledger/<YYYY-MM-DD>/<sha>-<host>-<job>.json`.
   Never a shared file (G-11 rebuild-storm class). The ledger *is* the project memory.
7. **Publishing is automated — the train publishes itself (operator ruling 2026-09-13:
   "T4 is never mine … releases are automated by release train … for ALL releases").**
   Pre-publish dogfood GO on the release commit, every release asset present and verified,
   publish preflight R1–R6 green ⟹ `scripts/cascade-drain.sh --target 0.N.0 --passes 30`
   runs from the detached checkout of the promoted tag, one crate per call, never
   `--allow-dirty`. No workflow runs `cargo publish`; the autopilot on the driver host does.
   There is **no** attended step; anything needing Noah is a §8 stop.
8. `pmat work add` from the driver session only. Stop the line on RED — no reruns-as-passes,
   no `--skip`, no waivers.

## §4 The train — each step has its own already-done test

| Step | Action | Skip if |
|---|---|---|
| **T-0 Cut** | cut sha = `main` HEAD; bump minor; `CHANGELOG` from merged PR titles since last tag | tag `v0.N.0` exists |
| **T-1 Deep** | `ci / deep` green on cut sha: full tests, doctests, `--no-default-features`, feature matrix (§4.1), GPU, every `cargo run --example` (§4.2) | green run recorded for this sha |
| **T-2 Dogfood** | `apr-dogfood` skill go/no-go receipt; `apr-cookbook` current; release notes generated | receipt exists for this sha |
| **T-3 Promote** | tag; clean-room on the tag; GitHub release with T-2 receipt attached | release `v0.N.0` exists |
| **T-4 Publish** | **automated** — the autopilot runs `cascade-drain.sh` after T-3's preflight; record cascade wall minutes; attended minutes are 0 by construction | all crates at `0.N.0` on crates.io |
| **T-5 Reconcile** | **hard gate (operator 2026-09-13: kaizen)** — the §6 reconcile predicates hold, receipt `docs/build-ledger/<date>/<sha>-reconcile.json` committed; the train has no `DONE` line without it | receipt exists for this sha and every predicate reads 0 |

Any step RED → SKIPPED, no partial promotion. Scope is assigned to trains after the fact:
0.67 contains whatever merged before the 0.67 cut, by definition.

### §4.1 Feature matrix — what T-1 means by it

One `cargo check -p <crate> --no-default-features --features <one feature>` per **(crate, feature)
pair**, not a powerset: a powerset is 2^n per crate (aprender-orchestrate alone declares 78 features)
and infeasible. The universe comes from `cargo metadata`, never a written list.

This clause was in T-1 from the start and was never built. Its first run, 2026-09-14 (#3262),
measured **100 of 430 pairs RED — every one unreachable from any crate's default set**. That is the
whole reason it can go dark: `cargo check --workspace` stays green over all of it, because feature
unification hands each crate whatever its siblings enabled.

Five shapes account for all 100; check them in this order:

1. **Definition gated, caller not.** Often the gate is over-broad — **un-gate it, do not spread it**.
2. **Declared in `Cargo.toml`, never applied in code.** `grep -rn 'feature = "X"' --include='*.rs'`
   returning nothing is the test.
3. **The implicit dep-feature used where the composite was meant.** An optional dep `foo` creates
   feature `foo`, which links foo and nothing else.
4. **A feature that supplies nothing its code needs** — usually a dependency removed for lock bloat
   with the gated code left behind. **Measure `Cargo.lock` before re-adding**: a version already
   locked by a sibling costs zero new packages.
5. **Struct drift** — an upstream type grows a field and the one initializer in the tree is never
   updated. Nothing else in the train watches for this; the matrix is what catches it.

**Known-red list.** A pair that cannot be made green in the same train is listed with a date and an
issue reference, and is reported, not fatal — a lane born red is a lane taught to be ignored. A
listed pair that **PASSES is fatal**: the list may only shrink, and it shrinks by deleting a line.
Re-derive the list against the tree immediately before the lane is undrafted; a dated list is
evidence of when it was true, not that it still is.

**A feature that cannot be built at all** — its dependency is deliberately absent, or circular —
emits ONE `compile_error!` naming the dependency and the exact edit that would enable it, with the
real gate moved to a private `__x-linked` feature so the refusal is the only diagnostic. Precedent
in tree: `aprender-simulate/z3-proofs`, `aprender-core/showcase-profile`, `aprender-zram-core/cuda`.

Fixing one cause exposes the next — the nextest-fail-fast class. Re-sweep after every fix; a
single-cause sweep is not a measurement.

### §4.2 Examples — run, not just build

T-1 says *every* `cargo run --example`. Building them is a different clause and much cheaper (83 s
for 980 targets vs ~2 h to run them). Measured 2026-09-14 on a random five with `-- --help` and a
60 s cap: **three ran past it**. They are not CLIs, they are compute demos that ignore argv, so
**timeout is a PASS**. The assertion T-1 actually owes is *every example starts and does not crash*.
Asserting a duration here would be a wall-clock assertion in a required check, which
`check_no_timing_in_required.sh` exists to forbid.

Both clauses carry a **vacuity floor** (measured 980 examples / 430 pairs; floors 800 / 400): a
discovery that finds almost nothing reports zero failures, which reads exactly like a pass.

## §5 Build rows — in order, one PR each, only when no train is due

**P0 · Pack** *(operator 2026-09-12; precedes everything while it fails)*. Every wakeup: sample
busy/online per host (§1), decide `intel pressure`, and act in minutes — arm every green PR, re-run
withdrawn CI, trim non-additive roadmap diffs, route arch-neutral work off intel, dispatch the
nightlies that produce T-1 evidence on the idle GPU boxes. Record the sample as a ledger record
`docs/build-ledger/<date>/<sha>-fleet-pack.json` `{at, intel_busy, intel_online, gx10_busy,
gx10_online, yoga_busy, yoga_online, intel_pressure, verdict}`.
*Done:* over the trailing 10 trains, in every sample with `intel_pressure=true`, median busy/online
≥ 0.8 on both `yoga` and `gx10`; the §7 `pack:` line is never `P0-UNDERUTILIZED` two wakeups in a row.
The structural lever is `workspace-test` leaving its `X64` pin (#3139: 82,085 tests in 795 s on gx10
against 1,000–6,000 s on intel); until it lands, `pack` is bounded by the short jobs.

**P0 · Reap** *(operator 2026-09-12: "automated disk clearing/reaping … managed by forjar and p0 if
it gets blocked")*. `ci-reaper` + `ci-disk-watch` (one script, per-host thresholds citing a
measurement) declared in `machines/{intel,gx10,yoga}/forjar.yaml`, applied, timers verified active;
`/mnt/nvme-raid0 → ~/eph-work/intel-mirror` declared on gx10/yoga so a reimage keeps the layout.
Admission at job start: `ci_self_hosted_preflight.sh` prints `disk_free_gb` (measurement first; the
gate threshold waits for 20 records, §3.3). *Done:* zero ENOSPC across 10 trains; a queue blocked on
disk is P0 and stops the current build row.

**P0 · Instrument.** One ledger record per gate job: `sha host job queue_wait_s exec_s
total_s peak_rss_mb free_disk_gb exit`. Add `make build-report`: p50/p95 `total_s` and
`queue_wait_s` per host, 10 slowest test targets, and the §1 PRs-per-train number.
*Done:* ≥ 20 records; `make build-report` runs on a clean checkout.

**P1 · Two lanes.** `ci / gate` (every PR) = compile + **fast set**. `ci / deep` (tags +
nightly) = everything else. Fast set is derived from the ledger, not opinion: rank tests by
failures-caught-per-second, keep the shortest prefix that caught every failure the full
suite caught; record the cut and the escape count. Runner: `cargo nextest` for the fast set
only if measured faster than `cargo test` on the same sha three times; record the delta
either way.
*Done:* one sha runs both lanes; a tag cannot publish with `ci / deep` red; escape count
over the next 10 merges is in the ledger.

**P2 · Shard across hosts.** Infra PR first (`paiml/infra`): runner labels for `yoga` and
`gx10` plus intel's disk preflight (`PREFLIGHT_PRESSURE_GB`, `PREFLIGHT_CRITICAL_GB`) via
forjar. Route by capability, then idleness:

| Work | Host |
|---|---|
| x86 CUDA build + CUDA unit tests | `yoga` |
| aarch64 / sm_121 (`-DGGML_CUDA_ARCHITECTURES=121`) | `gx10` |
| clean-room, CPU differential — reserved, never shared with a perf run | `intel` |

Admission control replaces "never idle": start a job only when free disk ≥ 1 × p95 run size
for that job; andon below 2 ×. Queue depth per host is a ledger field.
*Done:* one PR's gate runs on ≥ 2 hosts; `build-report` prints p95 queue-wait per host;
zero ENOSPC in 10 trains.

**P3 · Ratchet.** After 10 trains of post-P2 data, land as fail-closed checks: p95
`ci / gate` ≤ 0.70 × baseline `[A]`; tag→publish ≤ 0.70 × baseline `[A]`. Nightly coverage
job, low priority, report-only: target = measured baseline + 2 points per ratchet toward
95 %. Re-evaluate each train; ratchet downward only, ≤ 10 % per step.

### §5.1 The debt tax — paid on every row, scheduled by none

*(measured 2026-09-14; the largest single time sink of that day)* The pre-commit
gate refuses ANY commit touching a file that carries a function over cyclomatic 30 / cognitive 25,
and `--no-verify` is banned. So a one-line fix in a debt-carrying file costs the decomposition of
every offender in that file. Measured on one day's work: **11 pre-existing violations** paid down to
land small fixes — worst cases cognitive 91 (`execute_llm_score`), 73 (`help_producer_truth::resolve`),
61 (`execute_llm_load`). None were in code that day's changes wrote.

This is a real cost and it is not optional, so it is scheduled rather than absorbed silently:

- The decomposition lands in the SAME commit as the fix that triggered it, and the commit message
  names each offender with its before-number. A stack green only at the tip hides which change
  broke what.
- Extraction, not rewriting. Prefer moving a whole block to a named function over clever
  restructuring; a behaviour-preserving refactor is the only kind admissible here.
- When a match arm is extracted, re-match behind a `let … else` that refuses by name. Never panic,
  never `unwrap()`.
- Repeated identical blocks are the cheapest win — one helper taking the two things that differ
  collapsed nine sites in one file.
- **Record per train:** `debt: files_touched <n> | violations_paid <n> | worst_before <n>`. After 10
  trains, if `violations_paid` is not falling, the gate's thresholds or the tree's debt is the build
  row, not the fixes.

*Done:* no train spends more turns on debt paydown than on the change that required it, measured
over a trailing 5 trains `[U]`.

## §6 Triage — continuous (§6.1), three surfaces (§6.2), capacity-bounded (§6.3) — and the T-5 reconcile, a HARD gate

**Measured 2026-09-13 (0.67.0 train):** 308 open issues, 424 opened vs 124 closed in 30 d, 95 open
issues already cited by a commit on `main`, 48 merged PRs of which 9 carried a closing reference,
880 remote branches of which 847 have no PR and 832 are unmerged, 13 DIRTY PRs listed and never
decided. Filing was the work product and closing was nobody's; the milestone pass assigned
attributes, not decisions. **Operator ruling: reconciliation is part of every train (kaizen), not
a separate chore.** A train that ships code and leaves its own tickets, branches and dead PRs
behind is not done.

### §6.1 Cadence — every wakeup, not once per train

"Once per train" is what let 20 of 34 open PRs carry no milestone at all on 2026-09-14, every one
of them opened in the preceding two days. A train is 48–72 h; a PR opened an hour after the pass is
invisible for the rest of it. Triage runs on the **P0 · Pack** wakeup, beside the fleet sample —
same cadence, same receipt, and it is equally P0 (operator: "ticket, pull requests, branches that
are not triaged are P0").

The per-wakeup pass is mechanical and bounded: anything opened since the last sample gets a
milestone and a label; `BEHIND` PRs get `gh pr update-branch`; `DIRTY` PRs get their conflicting
files named in a comment. Nothing here is a judgement call. The once-per-train pass keeps only
what needs the whole window: the §6.3 capacity check and the T-5 reconcile.

### §6.2 Three surfaces — issues, PRs, and BRANCHES

Branches were the unwatched one. Measured 2026-09-14: **107 remote branches, 35 with an open PR,
72 without**. Of those 72 — 32 younger than 7 d, **27 in a 7–14 d band that no rule looks at**, 13
already R-3-eligible. R-3 archives a branch with no PR and a tip older than 14 d, so work that
stalls on day 8 is invisible for six more days and then deleted without ever having been seen.

| Surface | Per-wakeup test | Fix |
|---|---|---|
| Issue | opened since last sample → milestone + label | assign; `untriaged_issues = 0` |
| PR | open → milestone; `BEHIND` → update-branch; `DIRTY` → conflicting files named | `untriaged_prs = 0` |
| Branch | no open PR and tip older than **7 d** | open a PR (draft is fine) or archive it now — do not wait for R-3 to delete it at 14 d |

A branch with no PR is not work in progress; it is work nobody can see. Seven days is the point at
which it must become visible or become history.

### §6.3 Prioritisation — capacity, not preference

§4 says scope is assigned after the fact: "0.67 contains whatever merged before the 0.67 cut, by
definition." That is right for what a train *contains* and wrong as a plan for what it *promises*.
With no capacity rule, a milestone is a dumping ground with a date on it.

**Measured 2026-09-14**, closure = **6.1 issues/day** over the trailing 7 days:

| Milestone | Open | Due in | Needs | Verdict |
|---|---:|---:|---:|---|
| 0.68.0 | 280 | 1 d | ~46 d | **over by 45 d** |
| 0.69.0 | 53 | 4 d | ~9 d | over by 5 d |
| 0.70.0 | 15 | 7 d | ~2 d | fits |

A date that is 45 days of arithmetic away from its content is not a commitment; it is a label, and
every number computed from it is fiction.

**The rule, and it is arithmetic so it stays inside §6's no-judgement-calls design:**

```
capacity(M) = days_remaining(M) × closure_rate_p50(trailing 7 d)
```

- `open(M) > capacity(M)` → the milestone is **OVERCOMMITTED**. Report it on the `triage:` line
  every wakeup. It is not an error; it is a number that must be visible.
- At T-0 of each train, an overcommitted next-milestone **spills**, lowest priority first, until
  `open(M) ≤ capacity(M)`. Spill order is by label — `P0` never spills, then `P1`, then unlabelled,
  then by age, oldest kept. The operator sets priority by labelling; **the arithmetic sets the cut
  line**, so no train needs a judgement call about scope.
- A `P0` set that alone exceeds capacity is a **stop** (§8): the release is over-promised at the
  priority level the operator controls, and only the operator can resolve that.
- `closure_rate` is MEASURED, never assumed. First train with fewer than 7 days of data reports
  `[U]` and spills nothing.

**Arrival is the other half.** Closure of 6.1/day against a ledger that grew net +149 in ten days
means the cut line moves further out every train no matter how it is drawn. §6's arrival/closure
ratio (R-5) is the control on that; capacity only decides what a date is allowed to claim.

### T-5 reconcile predicates — every one must read 0 in the receipt

| Predicate | Derive with | Fix |
|---|---|---|
| R-1 fixed-but-open | open issues cited by a `main` commit since the previous tag whose subject/body uses `Closes/Fixes/Resolves #N` or is `fix(...)`: must be ∅ | close with the commit sha as receipt |
| R-2 closing-reference | merged PRs since the previous tag whose body cites an issue without a closing keyword: must be ∅ going forward (guard `scripts/check_pr_closes_issue.sh` on the PR; `Refs #N` is allowed only with `no-close:` and a reason) | the guard refuses the PR |
| R-3 dead branches | remote branches with no open PR and a tip older than 14 d: must be ∅ | archive to `refs/archive/<branch>` (objects kept, reversible), then delete the head |
| R-4 dirty PRs | PRs `DIRTY` for more than one train: must be ∅ | a verdict per PR in the train: rebase (union tool, or by hand) or close with the reason |
| R-5 ratio | `closure / arrival` over the train window, recorded; ratchet as below | — |

The receipt is `{sha, window:[prev_tag, tag], R1..R5: {count, list}, actions:{closed, archived, rebased, pr_closed}}`
and is committed to the ledger; the autopilot writes `DONE` only after `check_reconcile.sh <tag>` reads
every predicate 0. Falsifier: seed one fixed-but-open issue (a closed test issue reopened) — the check
must go RED.


Queues stabilise when closure ≥ arrival; age falls when WIP is capped. That is the whole
mechanism.

- Every issue opened since the last train: labelled and sized. **`untriaged = 0`** at the end
  of every pass — hard target today, needs no baseline.
- **Counted per surface, never as one number.** Measured 2026-09-14: issues were 319/320 triaged
  while PRs were 20 of 34 with no milestone at all. A single `untriaged` figure reported the clean
  surface and hid the breached one. The pass ends only when `untriaged_issues = 0` AND
  `untriaged_prs = 0`.
- **Triage is not disposal.** Over the trailing 10 days the issue ledger grew net **+149** while
  being ~100 % triaged, and **312 of 320 open issues were opened by the agent itself**. Classifying
  a finding does not close it. If `arrival > closure` for 3 consecutive trains, filing new findings
  as issues stops being free: the next train's build row is closure, not features.
- **Mechanical PR states are the pass's job, not a reviewer's**: `BEHIND` → `gh pr update-branch`
  (measured: 14 → 4 in one pass), `DIRTY` → named on the PR with the conflicting files.
- PR lifecycle, deterministic: no activity for 2 trains → label `stale` + comment; `stale`
  and no activity for 1 more train → closed. (~9 days, no discretion.)
- Record per train: `arrival closure open_prs age_p95 untriaged`.
- Ratchet: if `closure ≥ arrival` holds over any trailing 5 trains, cut the stale label from
  2 trains to 1. Total open count gets a target only after 10 trains `[U]`.

## §7 Session report — exactly this, in the PR body or the release

```
APR-RELEASE-001 | did=<TRAIN|BUILD|TRIAGE|NOOP> | train=v0.<N>.0 | verdict=<SHIPPED|SKIPPED|MERGED|NOOP>
train:   step reached <T-0..T-4> | skip reason <none|…> | cascade wall min <n|[U]> | attended min 0
build:   row <P0..P3|none> | PR <url|none> | records added <n>
gate:    p95 ci/gate <min|[U]> | max PRs/train <n|[U]> | queue p95 intel <s> yoga <s> gx10 <s>
pack:    intel <busy>/<online> | gx10 <busy>/<online> | yoga <busy>/<online> | intel-pressure <yes|no> | verdict <OK|P0-UNDERUTILIZED>
triage:  arrival <n> | closure <n> | open PRs <n> (age p95 <d>) | untriaged issues <n> prs <n>
branches: total <n> | no-PR <n> | no-PR >7d <n> | archived this pass <n>
capacity: milestone <M> open <n> | closure/day <x|[U]> | capacity <n> | verdict <FITS|OVERCOMMITTED by <n>d> | spilled <n>
matrix:  pairs <n> | red <n> | known-red <n> | stale-known-red <n> | examples started <n>/<n>
debt:    files touched <n> | violations paid <n> | worst before <n>
quorum:  rounds <n> | width <n> | verdicts <a/b/c> | overridden <yes|no>
reconcile: R1 fixed-open <n> | R2 no-close <n> | R3 dead-branches <n> | R4 dirty>1train <n> | R5 closure/arrival <x> | receipt <path|MISSING>
stops:   <none|list>
next:    train eligible at <timestamp>
```

## §8 Stop conditions — stop and report, do not work around

- Two consecutive SKIPPED trains, or ≥ 72 h with no runnable train → andon.
- `ci / deep` red on a tag → no publish.
- Cascade dry-run non-zero → stop before any real publish.
- Measured max PRs/train < 10 `[A]` → stop cutting trains; finish P1/P2 first.
- `yoga` or `gx10` has no runner unit in live `forjar.yaml` and the infra PR is unmerged → stop at P2.
- Fewer than 20 ledger records → stop at P0.
- `yoga` or `gx10` under-utilised (§1) while intel has queue pressure, two wakeups in a row → P0:
  stop the current build row, pack first (arm, route, reap), report the `pack:` line.
- A runner box at or below `REAPER_CRITICAL_GB` free, or any job dead on ENOSPC → P0: reclaim
  over SSH now, then land the forjar change that makes it automatic, before anything else.
- A host DECLARED a full-time build host sits at 0 % occupancy for a whole session while any
  queue has pressure → P0: that is a routing or job-class defect, not spare capacity. (`mini`
  measured at 0.0 % all of 2026-09-14 while declared full-time in #3205 — its ceiling is the
  macOS-capable job classes, not capacity.)
- A `ci / deep` known-red list names a pair that now PASSES → stop: the list is stale and the lane
  is asserting something that is no longer true (§4.1).
- A design fork appears that §6 cannot decide without judgement → §10, not a coin flip and not a
  question to the operator.
- The `P0`-labelled set of the next milestone alone exceeds its capacity (§6.3) → stop: the release
  is over-promised at the one priority level the operator controls, and only the operator can cut it.
- Any step would need an invented threshold, a second concurrent aprender PR in CI,
  `--allow-dirty`, or SSH into a host → stop.

## §9 Budget and the one risk to measure first

K̂ = 14 sessions (10 trains to `0.76` + P0, P1, infra, P2) · K = 18 · andon at 16 sessions
or 2 consecutive skips `[A]`.

The 82-crate cascade is automated (ruling 2026-09-13) and its wall time is unmeasured.
Record cascade wall minutes at T-4 on the `0.67` train. Above ~20 min `[A]` it delays the
post-publish receipts and the cascade — not the build — becomes the next kaizen target.

## §10 Decision procedure — for the forks §6 refuses to make

§6 is deliberately "no judgement calls". Design forks still occur — four ways to fix one broken
feature declaration, restore-vs-delete a dead GPU path — and they are not triage. They go here.

**Fan out through agy, not through more Claude subagents.** Lanes cost agy credits; orchestrator
turns are the scarce budget. `lane-group.sh run --out-dir D --width 3 --scratch -- agy-lane.sh
--mode plan --repo-root <toplevel> --prompt …`. Width 3 is the default; 10 is the cap.

**The brief carries the measurements, not the question.** Name the files, the line numbers, the
acceptance command, and every option including the ones you dislike. A lane that has to go and
measure the premise will measure it differently from the next lane, and the vote becomes noise.

**Plant the trap.** Include one question whose obvious answer is wrong — a rename that looks
mechanical but is not. A quorum that misses it has told you how much its other answers are worth.
(2026-09-14: `Lz4WarpShuffleKernel` → `Lz4WarpCompressKernel` reads as a rename and is a
literal-only encoder swapped for a real match-finder. All three lanes caught it.)

**A verdict is a claim until re-run.** The orchestrator re-executes the acceptance command itself
and records both columns. A lane reporting "all 9 selections green, 84 tests pass" is evidence of
what the lane believes.

**A premise error voids the vote, and the fix is another round — not the orchestrator's judgement.**
Measured on #3179: round 1 voted 2/3 to make a dead path compile. Three facts were then verified by
reading the tree (a stub decompress kernel, a launched entry point that does not exist, a host-memory
API that never existed here). Round 2 with those facts overturned round 1 unanimously. **Re-run the
quorum with the corrected premise; do not silently substitute your own conclusion.**

**Overriding the majority is allowed exactly once and must be recorded.** Only on a fact no lane in
that round had, stated in the commit and the PR, with the losing option's own argument quoted. On
#3179 round 2 split 2/1 to delete; the minority option shipped because the crate's default build is
green and the module in question is live public API — both deleting lanes had argued from "a dark,
unused feature". When the fact is not decisive, the majority wins and your discomfort is not
evidence.

**Prefer the reversible option when the vote is close.** A `compile_error!` can be deleted later; a
deleted module and its public types cannot be recovered from a reviewer's memory.

**Record per train:** `quorum: rounds <n> | width <n> | verdicts <a/b/c> | overridden <yes|no>`.

