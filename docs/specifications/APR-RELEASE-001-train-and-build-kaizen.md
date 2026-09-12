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
| p95 `ci / gate`, tag→publish, attended cascade time | — | `[U]` unmeasured |
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
7. **Publishing is unchanged and attended.** Clean-room is the hard gate, named first. Then
   `scripts/publish_cascade.sh` from a detached checkout of the promoted tag — dry-run
   receipt, one crate per call, stop on first non-zero, never `--allow-dirty`. No workflow
   runs `cargo publish`. This is the **only** attended step; anything else needing Noah is
   a §8 stop.
8. `pmat work add` from the driver session only. Stop the line on RED — no reruns-as-passes,
   no `--skip`, no waivers.

## §4 The train — each step has its own already-done test

| Step | Action | Skip if |
|---|---|---|
| **T-0 Cut** | cut sha = `main` HEAD; bump minor; `CHANGELOG` from merged PR titles since last tag | tag `v0.N.0` exists |
| **T-1 Deep** | `ci / deep` green on cut sha: full tests, doctests, `--no-default-features`, feature matrix, GPU, every `cargo run --example` | green run recorded for this sha |
| **T-2 Dogfood** | `apr-dogfood` skill go/no-go receipt; `apr-cookbook` current; release notes generated | receipt exists for this sha |
| **T-3 Promote** | tag; clean-room on the tag; GitHub release with T-2 receipt attached | release `v0.N.0` exists |
| **T-4 Publish** | **attended** — cascade dry-run receipt, then Noah runs the cascade; record attended minutes | all crates at `0.N.0` on crates.io |

Any step RED → SKIPPED, no partial promotion. Scope is assigned to trains after the fact:
0.67 contains whatever merged before the 0.67 cut, by definition.

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

## §6 Triage pass — once per train, no judgement calls

Queues stabilise when closure ≥ arrival; age falls when WIP is capped. That is the whole
mechanism.

- Every issue opened since the last train: labelled and sized. **`untriaged = 0`** at the end
  of every pass — hard target today, needs no baseline.
- PR lifecycle, deterministic: no activity for 2 trains → label `stale` + comment; `stale`
  and no activity for 1 more train → closed. (~9 days, no discretion.)
- Record per train: `arrival closure open_prs age_p95 untriaged`.
- Ratchet: if `closure ≥ arrival` holds over any trailing 5 trains, cut the stale label from
  2 trains to 1. Total open count gets a target only after 10 trains `[U]`.

## §7 Session report — exactly this, in the PR body or the release

```
APR-RELEASE-001 | did=<TRAIN|BUILD|TRIAGE|NOOP> | train=v0.<N>.0 | verdict=<SHIPPED|SKIPPED|MERGED|NOOP>
train:   step reached <T-0..T-4> | skip reason <none|…> | attended min <n|[U]>
build:   row <P0..P3|none> | PR <url|none> | records added <n>
gate:    p95 ci/gate <min|[U]> | max PRs/train <n|[U]> | queue p95 intel <s> yoga <s> gx10 <s>
pack:    intel <busy>/<online> | gx10 <busy>/<online> | yoga <busy>/<online> | intel-pressure <yes|no> | verdict <OK|P0-UNDERUTILIZED>
triage:  arrival <n> | closure <n> | open PRs <n> (age p95 <d>) | untriaged <n>
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
- Any step would need an invented threshold, a second concurrent aprender PR in CI,
  `--allow-dirty`, or SSH into a host → stop.

## §9 Budget and the one risk to measure first

K̂ = 14 sessions (10 trains to `0.76` + P0, P1, infra, P2) · K = 18 · andon at 16 sessions
or 2 consecutive skips `[A]`.

The 82-crate cascade is the only attended step and is unmeasured. Record attended minutes
at T-4 on the `0.67` train. Above ~20 min `[A]`, a 48–72 h takt costs ~4 h/month of
babysitting and the cascade — not the build — becomes the next kaizen target.
