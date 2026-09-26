# APR-OBS-001: apr dogfood observability and the directed-improvement loop

**Spec id:** `APR-OBS-001` · **Rows:** `OBS-00..OBS-12` (the traffic cop mints one `pmat` ticket per row; single-minter rule applies)
**Target repo:** `paiml/aprender` (`docs/specifications/APR-OBS-001-dogfood-observability.md`); infra rows are filed, not done (§0.7)
**Runner:** the aprender traffic cop (`aprender-traffic-cop-prompt.md`).
**Launch:** from `~/src/aprender`, run `Implement docs/specifications/APR-OBS-001-dogfood-observability.md autonomously.`
**Supersedes:** `apr-dogfood-observability-improvement-plan.md` (cop, 2026-09-26) — its P0–P6 map onto rows here.
**Related:** SRV-TIM-001 (serve timings), infra#1057 (per-binary perf ledger), infra#1088 (gx10 shadow lane), PRM-001 (Prometheus; §5.1 perf ratchet, §5.2 review ledger), APR-PERF-GATE-001 v2.2 (§4.2 artifact identity, §4.3 workload), ARBITER-001 / ARB-SELF-001 (self-improvement consumer), #3598 (0.71 "Verbs Are Fast"), #3596 (load + TTFT reporting).
**Status:** spec, not implemented. Dated 2026-09-26.

**Provenance marks:** `[V]` verified at the cited sha/time · `[C]` computed · `[A]` asserted · `[U]` unverified/unmeasured · `[X]` third-party.

---

## §0 Operating assumptions

1. **Purpose.** apr runs many times a day and records almost nothing. This spec makes every real apr invocation leave an identity-bearing timing row, makes regressions turn RED without anyone looking, and makes the *cause* of a slow run one flag away.
2. **Why the order matters (design basis).** A self-improving loop needs four things, in this order (Zhang, Yuan & Zhang, arXiv 2607.04277v2, §3.5.2): a **self-model** (a record of its own behaviour), **self-simulation** (replaying chosen inputs), **self-evaluation** (comparing expected against actual), and **directed modification** (changing *where* the evaluation points). Change that is not directed by self-knowledge is the degenerative regime (ibid. §A.8 Remark). Mapping:

   | Requirement | Row |
   |---|---|
   | Self-model | OBS-01 (lane rows), OBS-05 (nightly rows) |
   | Self-simulation | OBS-05 (fixed workload replay) |
   | Self-evaluation | OBS-06, OBS-07 (RED rules, fixed baseline) |
   | Directed modification | OBS-03 (prefill/decode split), OBS-09 (per-layer trace), OBS-11 (trace per perf PR) |

   No row may consume speed signals for self-improvement before the rows above it are GREEN (OBS-10).
3. **The evaluator is outside what it evaluates.** Ledger writers, RED rules and the baseline live in forjar-declared units and CI. Neither apr nor arbiter holds write access to them (ibid. Corollary A1; stack doctrine "producer is never the gate").
4. **Self-reports are cross-checked.** Server-reported timings (OBS-03) are the system describing itself; they are admissible only where they agree with an independent client measurement (OBS-04).
5. **The release train wins.** Every job here yields when the fleet-wide `train-active` flag is set on its pool, and holds `/tmp/apr-gpu.lock` around the binary only (never around cargo). Nothing here takes a session or CUDA host from an active release cut.
6. **Genchi genbutsu.** Every §1 value is re-measured at HEAD in Phase 0. A row whose premise is false at HEAD is `premise-falsified` and skipped.
7. **Cross-repo work is filed, not done.** Forjar declarations, timers and dashboards in `paiml/infra` are filed as issues with checkable acceptance criteria; dependent rows are `NotRun{NoDeclaredExecutor}` until they land.
8. **The operator has no pre-steps.** Anything only Noah can do is a §8 STOP.

---

## §1 Ground truth (baseline, frozen 2026-09-26; never quote as current)

| # | Fact | Value | Mark | Source of truth |
|---|---|---|---|---|
| G1 | Review-lane logging | `apr60-shadow-51725` (lambda, CPU, v0.69.3, Qwen3.5-4B Q4_K_M) records nothing per request; journal holds only the startup route list | [V] cop, 2026-09-26 | lane unit journal |
| G2 | gx10 shadow serve lane (:8091) | stopped, held | [V] | infra#1088 |
| G3 | Scheduled model runs | `cuda-nightly.yml` (03:30) and `qwen-story-daily.yml` (06:17): last run of each **cancelled** | [V] | GitHub Actions run history |
| G4 | Per-binary perf ledger | no data file exists; only #1057 work so far is agent-trace retention | [V] | infra#1057 |
| G5 | Serve timings | `timings` null on every path except CUDA chat | [V] | SRV-TIM-001 branch (2e) |
| G6 | Serve tracing | `X-Trace-Level` returns a single wall-clock total; `inference_trace` never runs in serve | [V] | aprender-serve source (`pmat query "X-Trace-Level"`) |
| G7 | `apr profile` on Qwen 3.5 GPU | roofline + per-op bricks, fixed on a branch | [V] | branch 9f1e456e0 (60) |
| G8 | Speed gap | apr serve ≈ 4.4× slower than llama.cpp end to end on Qwen 3.5 | [V] n=1 experiment | fb, PRM-S1 v2, 200 prompts, gx10 |
| G9 | Oracle pin | llama.cpp `d1d3c3396` | [V] 2026-09-20; [U] at HEAD | `scripts/llama_pin.toml` `build_commit` |
| G10 | Disk path in draft plan | `/mnt/nvme-raid0/...` is intel's directory on `/`; the lane runs on lambda | [V] | project guide §4.1 note |
| G11 | Stale-binary incident | a bare `apr` resolved to a 26-day-old build | [V] | cop plan §5 |

---

## §2 Design

### §2.1 Row identity block (shared by every ledger in this spec)

Every row, in every ledger, carries this block. A row missing any field is **inadmissible** and counts as a missing row.

| Field | Meaning | Source |
|---|---|---|
| `schema` | ledger schema id + version | constant |
| `ts` | UTC RFC 3339 | clock |
| `host` | forjar machine name (not `$(hostname)`: lambda is `noah-Lambda-Vector`) | forjar fact |
| `apr_version`, `apr_tag` | release tag | `scripts/apr_bin.sh` / fleet pin |
| `crate_tarball_sha256` | cross-host identity | APR-PERF-GATE-001 §4.2.2 |
| `binary_sha256` | host-local anti-substitution | sha256 of resolved binary |
| `build_identity` | `rustc -vV`, triple, **features read from the binary**, `uname -a`, accelerator + driver | APR-PERF-GATE-001 §4.2.2 |
| `model_id`, `model_sha256` | weights identity | sha256 of the GGUF |
| `backend` | `cpu` / `cuda` / `wgpu` / `metal` | request + proof (§2.5) |
| `gpu_proof` | trace line or used-GPU probe; `null` only when `backend=cpu` | §2.5 |
| `request_id` | UUIDv7; joins OBS-01 ↔ OBS-03 rows | client-generated, sent as header |

Cross-host comparison or ratio with mismatched identity is **FATAL** (refused). Within-host history with mismatch is annotated.

### §2.2 Ledgers

| Ledger | Schema | Writer | Cadence | Path |
|---|---|---|---|---|
| Lane | `apr-lane-row-v1` | review-lane client | every lane call | per-host, forjar-declared `fleet_perf_dir` `[U]`; synced to almacén |
| Serve | `apr-serve-timing-v1` | `apr serve` structured log | every request | journal → per-host file, same dir |
| Nightly | `apr-perf-ledger-v1` | forjar timer (OBS-05) | nightly per host × backend | same dir |
| Trace | `apr-trace-v1` | `apr serve` on `X-Trace-Level` | on demand + one per nightly | same dir, referenced by path + sha |

Append discipline: one JSON object per line, `O_APPEND`, line length ≤ 4096 bytes (atomic append); longer payloads go in a side file referenced by path + sha256. No ledger is ever rewritten in place.

### §2.3 Lane row (`apr-lane-row-v1`) — beyond §2.1

`prompt_n, completion_n, prompt_sha256, wall_ms, ttft_ms, ttft_reason, prefill_ms_per_tok, decode_ms_per_tok, ok, error, load1, cold`

- `ttft_ms` is measured only when the client streams. Otherwise `ttft_ms: null`, `ttft_reason: "non-streaming"`. **Never** copy `wall_ms` into `ttft_ms`.
- `cold: true` for the first request after unit start.
- Lane rows are **operational data, not benchmark data**: they feed failure detection and per-token p50/p95. No apr/llama ratio is ever computed from lane rows.

### §2.4 Workload for the nightly (OBS-05)

Reuse APR-PERF-GATE-001 §4.3 shape; do not invent a second prompt set.

| Field | Value | Mark |
|---|---|---|
| model | Qwen3.5-4B Q4_K_M, sha pinned | [A] (#3558 selection rule may change it; a change resets baselines) |
| prompt set | fixed corpus, `prompt_tokens = 512 ± 8`, ≥ 8 prompts | [A] |
| generation | `max_tokens = 128`, greedy, `seed = 0`, ignore-EOS | [A] |
| context | 4096 | [A] |
| repetitions | 5 per prompt; report median + MAD | [A] |
| reported | load ms, TTFT, pp512 tok/s, tg128 tok/s, wall | #3596 |
| comparator | llama.cpp at `scripts/llama_pin.toml` `build_commit`, same GGUF, same host, same run | [X] |

### §2.5 GPU proof

A row claiming `backend ∈ {cuda, wgpu, metal}` must carry `gpu_proof`: a trace line naming the device kernel path or a used-GPU probe sampled during the run. `CUDA_VISIBLE_DEVICES`, a flag, or a feature list is not proof. A GPU claim without proof is recorded as `backend_unproven` and excluded from GPU series.

---

## §3 Contracts and falsifiers

Each contract ships in the same PR as its row; each falsifier is planted and shown RED→GREEN in the PR body.

| Contract | Row | Falsifiers (each must turn RED) |
|---|---|---|
| `apr-lane-row-v1` | OBS-01 | client with the write step removed; row missing `model_sha256`; `ttft_ms == wall_ms` on a non-streaming call |
| `apr-nightly-liveness-v1` | OBS-02 | a workflow cancelled 2 nights in a row reported GREEN |
| `apr-serve-timing-v1` (SRV-TIM-001 F1–F5) | OBS-03 | F1 null timings on any backend; F2 token counts ≠ usage; F3 prefill + decode > wall; F4 planted `PhaseTimings::default()`; F5 log line ≠ response timings |
| `apr-selfreport-agreement-v1` | OBS-04 | a planted +20% skew in server timings passes |
| `apr-perf-ledger-v1` | OBS-05 | missing host row; empty ledger; missing version/sha; GPU row without `gpu_proof`; ratio computed across mismatched identity |
| `apr-perf-red-rules-v1` | OBS-06 | planted regression of 2× threshold on one night stays GREEN; planted noise-level wobble turns RED |
| `apr-perf-baseline-v1` | OBS-07 | planted 3%/week drift for 4 weeks stays GREEN against the release-tag baseline |
| `apr-trace-v1` | OBS-09 | trace returned with provenance `Measured` when the tracer did not run; per-layer sum > request wall |
| `apr-perf-pr-trace-v1` | OBS-11 | 0.71 perf PR merged without before/after trace |

---

## §4 RED rules (OBS-06, OBS-07)

Signal: **apr ÷ llama.cpp on the same host, same run** (host, heat and driver drift cancel). Raw apr ms are reported, never used for regression.

1. **Calibration (report-only).** Until 14 nightly rows exist per host × backend, every rule is report-only.
2. **Threshold** per host × backend × metric (load, TTFT, prefill, decode): `θ = max(5%, 3 × MAD/median)` over the 14-night calibration window `[C at calibration]`, recomputed every 30 nights, logged.
3. **Rolling RED:** ratio worse than the 7-night median by > θ on **2 consecutive nights**, or by > 2θ on one night.
4. **Baseline RED (OBS-07):** ratio worse than the ratio recorded at the **previous release tag** by > θ. The tag baseline is written once per tag by the release train (single writer) and never edited.
5. **Liveness RED:** a host × backend with no admissible row for the night; an empty ledger; a lane with calls but 0 rows in 24 h; a nightly workflow cancelled or failed 2 nights running (OBS-02, extends `NIGHTLY-RED`).
6. **RED output:** opens (or comments on) one aprender issue per (host, backend, metric) with both receipts and the latest trace (OBS-09) attached. Never a Slack-only alert.

**Blocking policy (operator ruling requested — §8 S-7 until given; default below):**
- Through 0.70: report-only.
- 0.71 final release: Baseline RED on decode or TTFT blocks the crates.io publish (0.71 = "Verbs Are Fast", #3598). Release candidates warn only.
- Every blocking rule needs a first-green run on a real tag before it can block (stack doctrine).

---

## §5 The directed-improvement loop

### §5.1 Admission (OBS-10)
A consumer (PRM-001 REX-10 perf ratchet, ARB-SELF-001 arbiter self-improvement, any autotuner) may act on speed signals only when **all** hold:
- OBS-01, OBS-03, OBS-04, OBS-05 GREEN;
- ≥ 14 consecutive nights of admissible nightly rows on the consumer's host × backend;
- the consumer holds **no write access** to any ledger, RED rule, baseline or this spec's contracts (checked by a CI job the consumer cannot modify).

### §5.2 Directed modification (OBS-09, OBS-11)
- `X-Trace-Level: step|layer` runs the existing `inference_trace` tracer for that request; response carries per-step/per-layer timings with provenance `Measured`. Off by default.
- The nightly (OBS-05) runs one traced request per host × backend; a RED night already has its per-layer table attached.
- **Every 0.71 performance PR** carries a before/after `apr-trace-v1` diff for the layers it claims to change, measured on the same host at the same identity. A perf PR without it does not merge.

### §5.3 Plateau rule
After 5 consecutive performance PRs whose traced target layer shows no improvement beyond θ, the owning epic stops and re-plans from the traces (not from intuition).

---

## §6 Hard rules

- **R-1 Pinned binaries only.** Resolve apr via `scripts/apr_bin.sh` or the fleet pin; never a bare `apr` on `PATH`; never aprender HEAD.
- **R-2 Empty is RED.** An empty ledger, a missing host row, a missing identity field: never a pass.
- **R-3 Varied input.** Never a single prompt; ≥ 8 per workload.
- **R-4 No `$?` through a pipe** in any script (`set -o pipefail`, or capture per stage). Shell passes `bashrs`.
- **R-5 No Python** in any file this spec creates. Harness and analysis are Rust (aprender workspace crate or xtask).
- **R-6 Declared executors only.** Timers, units, ledger paths and sync are forjar-declared (`forjar apply`); no ad-hoc SSH, no hand-installed apr.
- **R-7 GPU lock around the binary, never cargo; `train-active` yields.** The release always wins.
- **R-8 Retention.** Raw rows are kept forever in almacén; no roll-up replaces raw rows (≈ 11 MB/year at 100 rows/day × 300 B `[C]`).
- **R-9 No invented thresholds.** Every number cites its measurement command or is marked `[A]`/`[U]`.
- **R-10 Writer ≠ subject.** apr and arbiter never write the ledgers' RED rules, baselines or contracts.

---

## §7 Tickets (EV-ordered)

`K̂` in minutes `[A]`, recalibrated after the first three rows. **No row starts while a release cut holds the session or host it needs (§0.5).**

| EV | Row | Work | Contract | Done when (all must hold) | K̂ |
|---|---|---|---|---|---|
| 0 | **OBS-00** schemas | Commit this spec, the §2.1 identity block and the four ledger schemas as pv contracts; file the infra issue for per-host `fleet_perf_dir` + almacén sync | all §3 schemas | schemas lint; infra issue URL recorded with acceptance criteria | 60 |
| 1 | **OBS-01** lane rows (P0; owner 60) | Lane client appends `apr-lane-row-v1` per call, `request_id` header sent to serve | `apr-lane-row-v1` | rows = lane calls over 24 h, 0 gaps > 24 h; all three falsifiers RED→GREEN | 120 |
| 1 | **OBS-02** nightly liveness (P3) | Five-whys on the cancelled `cuda-nightly` / `qwen-story-daily`; fix the mechanism; extend `NIGHTLY-RED` to count cancelled runs | `apr-nightly-liveness-v1` | root cause written as a mechanism; 0 cancelled runs over the next 7 nights | 120 |
| 2 | **OBS-03** serve timings (P1; owner 2e; SRV-TIM-001) | Every backend fills `timings`; one structured log line per request; `/metrics` histograms for prefill, decode, TTFT | `apr-serve-timing-v1` | F1–F5 planted RED→GREEN on every backend; lands after the 0.70 publish | 240 |
| 2 | **OBS-04** self-report agreement | Join OBS-01 and OBS-03 rows by `request_id`; check server total vs client wall | `apr-selfreport-agreement-v1` | ≥ 99% of joined rows within max(5%, 50 ms) `[A]`; planted +20% skew RED | 90 |
| 3 | **OBS-05** nightly ledger (P2; owner infra-8d; closes infra#1057) | Forjar timer on gx10 (CUDA) and lambda (CPU; 4090 only if R-9 of PRM-001 does not apply to this lane) running pinned apr + pinned llama.cpp on §2.4 | `apr-perf-ledger-v1` | ≥ 13 of 14 nights with a row for every host × backend; all falsifiers RED→GREEN | 180 |
| 3 | **OBS-06** RED rules | §4 rules 1–3, 5, 6 as a CI job over the ledger | `apr-perf-red-rules-v1` | calibration log committed at night 14; planted regression RED; planted noise GREEN | 120 |
| 4 | **OBS-07** release-tag baseline | Release train writes the tag baseline at T-0 (single writer); §4 rule 4 | `apr-perf-baseline-v1` | baseline row exists for the first tag after OBS-05; planted slow drift RED | 90 |
| 5 | **OBS-08** one view (P6; owner infra-8d) | Page rendered from ledgers only: ratio over time per host × backend, lane p50/p95 per token, nightly status, latest trace link | — | generated from data only; 0 hand-edited values; filed as infra issue | 90 |
| 5 | **OBS-09** serve tracing (P5; **first row of the 0.71 serve epic**) | `X-Trace-Level: step\|layer` runs `inference_trace`; nightly attaches one traced request | `apr-trace-v1` | provenance `Measured` only when the tracer ran; per-layer sum ≤ wall; lands before the first 0.71 perf PR merges | 240 |
| 6 | **OBS-10** loop admission | CI gate encoding §5.1 for PRM REX-10 and ARB-SELF-001 consumers | extends §3 | a consumer with a planted write path to a RED rule is refused; a consumer before night 14 is refused | 90 |
| 6 | **OBS-11** perf-PR trace | 0.71 perf PRs must carry an `apr-trace-v1` before/after diff (§5.2) | `apr-perf-pr-trace-v1` | 100% of 0.71 perf PRs carry it; planted PR without it blocked | 60 |
| 7 | **OBS-12** hourly probe decision (P4) | After 14 nights, report whether any RED went undetected > 12 h between nightlies; operator rules | — | report committed; probe built only if the operator approves | 30 |

**K̂ = 1530 `[A]` · K = 1700 · andon at 1360.**

---

## §8 STOP conditions (stop, write the §9 report, do not work around)

- **S-1** A row would take a session, runner or CUDA host from an active release cut, or run while `train-active` is set on its pool.
- **S-2** An executing host resolves apr (or llama.cpp) to an undeclared or unpinned binary.
- **S-3** A ledger writer or RED rule would be placed where apr or arbiter can write it (R-10).
- **S-4** A ratio would be computed across mismatched identity (§2.1).
- **S-5** A harness hook refuses the session's writes for a reason other than a missing ticket.
- **S-6** Budget K reached, or andon crossed with more than one row incomplete.
- **S-7** A rule would become release-blocking without an operator ruling on §4 blocking policy.
- **S-8** Two consecutive non-passing quorums on the same PR.

---

## §9 Final report schema (one per session)

```yaml
spec: APR-OBS-001
session: {date_utc, host, tree: {head, origin_main, worktree}, model}
selected_row: OBS-NN
ticket: PMAT-NNNN
outcome: merged | stopped | already-done | premise-falsified
stop: {id: S-n, evidence: "..."}
baseline_remeasured: [{row: G#, value, mark, command}]
ledgers:
  - {ledger, host, backend, rows_24h, admissible_pct, last_row_ts, empty: bool}
red:
  calibration: {nights, per_series: [{host, backend, metric, median, mad, theta}]}
  events: [{host, backend, metric, rule, ratio, baseline, issue_url}]
selfreport_agreement: {joined_rows, within_tolerance_pct}
loop_admission: [{consumer, admitted: bool, reason}]
issues_filed: [{repo, url, purpose}]
scoreboard_moved: [{metric, before, after, command}]
budget: {k_hat_row, k_actual_row, cumulative, andon_crossed: bool}
next_row: OBS-NN
```

### §9.1 Scoreboard (targets and probes only; state is rendered, never written here)

| Metric | Target | Rendered by |
|---|---|---|
| Lane rows ÷ lane calls, 24 h | **1.0**, 0 gaps > 24 h | OBS-01 |
| Cancelled or failed model nightlies, trailing 7 nights | **0** | OBS-02 |
| Serve rows with non-null prefill + decode, every backend | **100%** | OBS-03 |
| Joined rows with server/client agreement | **≥ 99%** | OBS-04 |
| Nights with a row for every host × backend, trailing 14 | **≥ 13** | OBS-05 |
| Rows with any `unknown`/missing identity field | **0** | receipt lint |
| GPU rows without `gpu_proof` | **0** | OBS-05 |
| RED events without an issue carrying receipts + trace | **0** | OBS-06 |
| 0.71 perf PRs without before/after trace | **0** | OBS-11 |
| Consumers acting on speed signals without admission | **0** | OBS-10 |
| apr ÷ llama.cpp (TTFT, decode) on the primary host | measured nightly; never-up vs tag baseline from 0.71 | OBS-07 |
| Python lines in files this spec touches | **0** | `grep -rl python3` over the diff |
