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

## (2b) p95 wall-clock per cell: pending

This is measured when the lane serves again. The earlier bench on this cell (`run_bench.log`, an 839-token prompt):
TTFT 25.9–26.7 s, prefill ~32 tok/s. A 16–24 KB brief is ~4–6k tokens, so expect ≥2 min of prefill alone, against the
lane's `timeout_s: 120`.

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

## Re-measure

On rc.2, once #4443 and #4313 land.
