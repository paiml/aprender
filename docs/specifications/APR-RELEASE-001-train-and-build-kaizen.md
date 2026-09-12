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
| 1 | ≥ 48 h since the last tag on `main` **and** no SKIPPED record for the current HEAD | run the train (§4) |
| 2 | else a §5 row whose *Done* test fails at HEAD | do the first such row, one PR |
| 3 | else the last train (shipped or skipped) has no triage record | do the triage pass (§6) |
| 4 | else | emit the §7 report, exit 0 |

Nothing in this spec asks a question. Running it ten times a day is safe.

## §1 Goal

Ship a tag every 48–72 h (`0.67 → 0.68 → …`) **on a clock, not on scope**, and keep
shrinking the wall-clock from *PR opened* to *tag published* so the clock stays cheap.
The objective is elapsed time and green trains — never machine utilisation.

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

Live `forjar.yaml` beats this table. Record the diff in the receipt and continue.

## §3 Hard rules

1. **The train is never delayed.** Cut can't go green in one attempt → train **SKIPPED**,
   reason recorded against that HEAD, no retry until `main` moves. Next eligible in 48 h.
2. **Two consecutive skips, or ≥ 72 h since the last tag with no runnable train → andon.**
   Stop, five-whys terminating in a mechanism, escalate. Do not cut a third.
3. **No invented numbers.** Every threshold cites its measurement command or carries `[U]`
   and is not a gate. No build target is set before P0 has ≥ 20 ledger records.
4. **Heijunka holds.** One aprender PR in CI at a time. Speed comes from splitting *one*
   gate across hosts, never from two PRs. "Parallel" means ≤ 3 read-only subagents inside a
   session (dogfood-skill style), never two sessions merging.
5. **No ad-hoc SSH.** Host changes are `machines/<host>/forjar.yaml` → `forjar apply` →
   `make -C machines/<host> verify-systemd-units`. Repo edits are inert until deployed
   (2026-04-26 ENOSPC was exactly this).
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
- Any step would need an invented threshold, a second concurrent aprender PR in CI,
  `--allow-dirty`, or SSH into a host → stop.

## §9 Budget and the one risk to measure first

K̂ = 14 sessions (10 trains to `0.76` + P0, P1, infra, P2) · K = 18 · andon at 16 sessions
or 2 consecutive skips `[A]`.

The 82-crate cascade is the only attended step and is unmeasured. Record attended minutes
at T-4 on the `0.67` train. Above ~20 min `[A]`, a 48–72 h takt costs ~4 h/month of
babysitting and the cascade — not the build — becomes the next kaizen target.
