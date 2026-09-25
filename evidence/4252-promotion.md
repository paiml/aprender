# #4252: can the apr 4B lane be promoted to a forced 4th quorum lane? (interim receipt)

Owner: aprender-60. Measured 2026-09-25, all times UTC. Cell: gx10 loopback :18253, Qwen3.5-4B-Q4_K_M, apr v0.69.3
aarch64 cuda release asset, served by `apr_dogfood_lane_4252.py watch` (pid 3854882).

**Standing answer: not yet.** Under the operator ruling of 16:32Z the lane is SHADOW: its presence is mandatory, it
carries zero weight, and any apr row (Verdict, Refused or NotRun with a reason) meets the seat. Nothing here changes
the rail.

Method, per the cop's amendment: the planted-defect and known-good part is PRM-001's (epic #4354). It uses the sealed
corpus `docs/audits/review-corpus/corpus-v1.jsonl` and the `rex` harness, and the numbers go to prometheus
(aprender-84) as input rows, not as a separate verdict here. This file keeps only the live-quorum shadow tally.

## (2) Availability: measured

Source: the watcher's own log, `gx10:/mnt/nvme-raid0/tmp/aprfc4252/v0693/watch.log` (62 lines, 22:26Z 09-24 to
16:38Z 09-25).

| window | served (START→YIELD) | availability (upper bound) | starts | yields |
|---|---|---|---|---|
| 18.2 h | 11.1 h | **60.8%** | 31 | 31 |

The figure is an upper bound because START→YIELD includes model load time, before `/health` answers.

Yield causes:

| cause | yields |
|---|---|
| free disk on `/` below the 60G floor (`/` is 93% full, 64G free) | 13 |
| `/tmp/apr-gpu.lock` held | 12 |
| foreign GPU process (release tests, `apr.cur`, `~/.cargo/bin/apr`) | 9 |

The watcher has been yielded since 15:22Z because the 0.69.5 release test's `apr run` holds the GPU. At 16:37Z
`/health` did not answer.

**This alone blocks promotion.** A forced lane that can answer at most ~61% of the time would NotRun roughly 2 rounds
in 5. On a shared GPU host, yielding is correct behaviour. A forced lane would need a dedicated device or a CPU tier,
and PMAT-392 measured the CPU tier at >300 s per call on this host.

## (1) Agreement with the 3-lane outcome: not measurable yet

- **Forward:** the installed `~/.claude/skills/paiml-implement/config.json` has no `quorum.advisory_lane`, so real
  rounds do not invoke the lane. The ledger holds 2 rows. Forward shadowing needs that key in the installed config.
  That is a rail config change, and this task says "no rail change", so it waits for the cop's ruling.
- **Retroactive:** 103 `brief.md` files exist on this host. Only 6 are real 3-lane rounds with a recoverable verdict,
  and all 6 are PASS. The rest are fixtures, plant dirs, or rounds with fewer than 3 lanes. So history cannot supply
  ≥30 rounds, and it cannot measure agreement on FAIL at all.

## (2b) p95 wall-clock per cell: see "Executor change" below

It is now measured on :8091. The earlier estimate from the watcher's bench (TTFT ~26 s, prefill ~32 tok/s) is
superseded.

## (3) Planted / known-good: routed to PRM-001

`rex review --split dev` rows for this cell, once prometheus names the corpus items dir. The sealed Test split is not
touched. PRM-001 itself is parked on infra#1088.

## Executor change (18:0xZ, cop ruling)

The GPU gets one executor: infra#1088's `apr-review-serve` on gx10 :8091. It runs the same binary (apr 0.69.3,
sha256 `2f274f47…`) and the same weights (`00fe7986…`) under `flock /tmp/apr-gpu.lock`. The :18253 watcher above
(pid 3854882) is retired: it was stopped by its recorded PID, and no child serve or port was left behind. The
availability figure in (2) describes the retired watcher, not :8091.

First Dev-split run against :8091: **3 rows, all Fail**, `rex review: …/v1/chat/completions: Network Error:
Unexpected EOF`. systemd stopped `apr-review-serve` at 18:00:51Z, mid-run; the journal shows about 20 start/stop
events in 4 h. At the time of the retirement check the lock was held by other consumers (a receipt script, a run
script and a cargo build), not by :8091. So the single executor has an availability problem of its own, and it
belongs to infra#1088's measurement. A rerun waits for `/health` and uses a fresh output dir, so the 3 EOF rows are
not mixed in. The rows go to prometheus (aprender-84) as shadow, non-prereg.

**Dev split, complete (19:02–19:09Z).** The executor yields to GPU-lock waiters by design, so each retry was a
whole fresh run in its own dir. Attempts 1–4 were cut at 11, 9, 8 and 0 rows. Attempt 5 finished 45/45. Output
dir `/mnt/nvme-raid0/rex-shadow/4252-gx10-4b-dev-reviewserve-r3/attempt-5/` (host lambda). `rex score`, unsigned
(exploratory), corpus review-corpus-v1@787d2026256cc08b:

| measure | k/n | 95% CI |
|---|---|---|
| parse rate | 45/45 | 0.92–1.00 |
| recall (planted P+R) | 27/30 | 0.74–0.97 |
| precision | 27/30 | 0.74–0.97 |
| false refute (known-good G) | 3/15 | 0.07–0.45 |

## (2b) p95 wall-clock per cell: measured on Dev

On 44 warm rows (the first row, which is cold, is excluded): p50 6.2 s, **p95 19.4 s**, max 34.6 s. The largest
brief is ~30k tokens, with prefill at 1.2–2.2k tok/s on :8091. That is well inside the lane's `timeout_s: 120`. The
earlier 26 s TTFT estimate came from the retired watcher's bench and no longer applies.

**What still blocks promotion is availability, not speed.** Serving windows between yields ran 1–7 min in this
hour, and only 1 of 5 attempts ran a full 45-item pass.

## Re-measure

On rc.2, once #4443 and #4313 land.

## (2c) :8091 availability after infra#1154 (19:13–20:13Z)

With infra#1154 the executor yields only to priority-lock waiters and drains on SIGINT. The probe hit gx10 loopback
`/health` every 15 s for 60 min (240 probes, `lambda:/mnt/nvme-raid0/rex-shadow/4252-avail-8091-1913/probe.tsv`).

| window | state | length |
|---|---|---|
| 19:13:36–19:27:06Z | up | 13.5 min (open at the start) |
| 19:27:06–19:35:36Z | down | 8.5 min |
| 19:35:36–19:55:51Z | up | 20.3 min |
| 19:55:51–20:06:06Z | down | 10.3 min |
| 20:06:06–20:13:21Z | up | ≥7.5 min (open at the end) |

- **Availability 165/240 = 68.8%**, against ≤60.8% for the retired watcher.
- Both stops are `gpu-priority` yields in the yield log (19:27:00Z, 19:55:43Z). There were no `gpu-lock-waiter`
  stops in the hour.
- The executor stopped within 1 s, and `/health` answered about 17 s after each restart.
- Every closed serving window was long enough for one 45-item Dev pass (7.3 min at 19:02Z), where last hour's
  windows (1–7 min) were not.
- Promotion to a forced lane is still blocked: it would NotRun about 3 rounds in 10.
