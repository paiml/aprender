# #4252 3-way `apr serve` bench: Qwen3.5-4B-Q4_K_M (aprender-6c [3ada9a], 2026-09-24)

GGUF sha256 `00fe7986…11a4` (identical on lambda, intel and gx10). Prompts: `p_fr.txt` ("The capital of France is", 5 tok) and `p850.txt` (lqw p850, 839 tok). n=3, temp 0, max_tokens 64.
Method: non-streaming two-request (TTFT = wall of max_tokens=1; decode = 63/(T64−T1)), because `/v1/completions` stream:true is buffered (#4272). `used_gpu` is read per request and `--expect-gpu` voids CPU answers.

| tier | binary (v0.69.1 d8a6df53a) | backend proof | prompt | TTFT s | prefill tok/s | decode tok/s (range) |
|---|---|---|---|---|---|---|
| **gx10 CUDA** (GB10) | aarch64-cuda asset `49afbc26…fad8` | `Backend: GPU (CUDA, NVIDIA GB10…) [qwen35 hybrid forward, #3090]` + `gpu-layers: requested=all resolved=32 total=32 (backend=cuda)`; used_gpu true 12/12; server pid 443076 held GPU mem | 5 tok | 0.171 | 29.3 | **34.1** (33.9–34.2) |
| | | | 839 tok | **26.2** | **32.0** | 31.9 (21–61, noisy: T64−T1 ≈ 2 s window) |
| **lambda CPU** | x86_64-cpu asset `9518ef96…79f3` | /health compute_mode cpu; used_gpu false 6/6; `taskset -c 0-7` + CPUQuota 800% + nice 19; host load 40→210; **overlaps f5's GPU hold** | 5 tok | 3.11 | 1.6 | **1.20** (1.18–1.38) |
| | | | 839 tok | PENDING: >20 min/rep at load 210 | | |
| **intel wgpu** (2× W5700X) | local `--features wgpu` build `7b7e6018…3311` (needed the #4056 one-line fix) | `--list-devices`: wgpu compiled in | — | **RED**: serve refuses at load ("qwen35 resolved to 0 transformer layers"); `run --backend wgpu` refuses with R-0b. No Qwen3.5 wgpu forward → **#4271** | | |

Readings:
- gx10 prefill ≈ decode rate (32 vs 34 tok/s): v0.69.1 prefills one token at a time. The 839-token TTFT is 26 s. rc.1's batched prefill (0.74 s claimed) is the next column once infra-8d installs it (operator: dogfood every rc.N).
- Lambda CPU pinned to 8 cores under load is ~28× slower on decode than gx10. That is not usable as more than a yield fallback.
- The same binary was not possible across tiers: 3 hosts, 2 arches, and intel needed a wgpu build. Each row names its sha.

## Yield mutation proof (lane `scripts/apr_dogfood_lane_4252.py`, branch feat/4252-apr-dogfood-lane 0f83b9a7a)
1. The gx10 watcher starts serve `--backend cuda --gpu-layers all`. `ask` → **served_by gx10-cuda**: wall 1.9 s, server pid 563471 holds GPU (175 MiB), used_gpu probe true, trace `gpu-layers … resolved=32 (backend=cuda)`, answer "PASS".
2. PLANT: on gx10, `flock /tmp/apr-gpu.lock sleep 150` (pid 739221). The watcher logs `11:12:27Z YIELD ['gpu lock held (/tmp/apr-gpu.lock)']`. `kill -0 563471` → No such process. nvidia-smi compute-apps are empty.
3. `ask` during the hold: gx10 attempt `CURL_RC=7` (state serving:false, stop_reason lock) → **served_by lambda-cpu**, wall 360 s (queued behind the bench). This run predates the hard budget: 0f83b9a7a had none. Under the current default (120 s) the same wait is `unavailable`; see below, answer "PASS", 51 prompt tok.
4. Hold released → `11:14:57Z START` (the watcher resumes).
Negative control (accidental): while the used_gpu probe was malformed, the lane labelled a real CUDA answer `gx10-cpu-UNPROVEN-GPU`. The label fails closed.

## Filed / commented
These are separate issues, each cross-linked from a #4252 comment. The ticket's "filed under #4252" means that link.
#4271 (Qwen3.5 no wgpu path), #4272 (buffered completions stream), #4056 (wgpu feature doesn't compile; fix evidence), #4146 (qwen35 chat omits used_gpu), #4254 (effective-config backend_loaded [] on CUDA too).

## Hard budget (added in 92e3c9c97; the proofs below run on the current head)
The quorum-round-1 finding (sonnet) said the budget was only shown on `--force-lambda`. It is now tested on the yield path.
- `scripts/test_apr_dogfood_lane_4252.py <budget>`: gx10 raises (the yield, `CURL_RC=7`), and lambda is a local socket that accepts and never answers (the SIGSTOP case).
  - At the **default 120 s**: `{"rc": 2, "verdict": "unavailable", "attempts": ["gx10", "lambda"], "wall_s": 119.1, "result": "PASS"}`.
  - At 6 s: wall 5.0, PASS.
- MUTANT (scratch copy): `signal.alarm` removed and lambda given a fixed 30 s timeout in place of `remaining()` → `wall_s 30.0 … "result": "FAIL"`, rc 1. The test goes RED without the budget.
- Live: `ask --force-lambda --timeout 8` against the same hung-server shape → `unavailable`, rc 2, wall 7.0 s.

## Server identity (round-1 finding)
`pid_alive` now requires `serve` and the lane's own port (`18253`) as whole argv tokens, not the substring `serve`, so a recycled pid is not taken for the server.

## Round-2 fixes (sonnet lane B) and a live re-proof on the current code (2026-09-24)
- **Yield during model load**: start_server re-checks need signals each second while /health is pending.
- **Bounded probes**: `sh()` probes time out (15 s) as rc 124, which counts as a yield reason, so a hung nvidia-smi no longer freezes the watcher.
- **gx10-side state dir**: the ssh remote uses `$HOME/.local/state/…`.
- **Startup timeout**: a server with no /health after 300 s is stopped and the reason recorded.
- `scripts/test_apr_dogfood_lane_4252_watch.py` drives the real need_signals / start_server / stop_server / cmd_watch against a fake apr (loopback /health). **7/7 PASS:**
  - free → START
  - real flock → YIELD, pid gone
  - released → START
  - claim raised mid-load → `YIELD (during start)` in 3.0 s
  - hung nvidia-smi → `rc=124` reason in 2.0 s
  - foreign GPU pid → reason
  - all free → none
- MUTANTS (scratch copies):
  - need check in the start loop removed → `need during load` row FAIL (7.0 s, no yield)
  - probe timeout removed → `hung nvidia-smi` row FAIL (30 s, no reason)
- **Live, on v0.69.3-rc.1** (asset sha256 a4b3e456…8197, verified against the release's .sha256), watcher on script 62eee9cda:
  1. `ask` → **served_by gx10-cuda**, serve pid 1431039. cuBLAS trace for that request: `[qwen35] batched prefill: 51 tokens in 305 ms (167 tok/s, chunk 51 rows, attention cuBLAS f32, from position 0)`. utilization.gpu `0 … 10 11 94 94 96 96 96 0`. used_gpu probe true.
  2. PLANT on gx10: `flock /tmp/apr-gpu.lock sleep 100` → `11:53:23Z YIELD ['gpu lock held …']`. Pid 1431039 gone; compute-apps empty.
  3. `ask` during the hold (default 120 s budget) → gx10 `CURL_RC=7` → **served_by lambda-cpu**, rc 0, wall 22 s.
  4. Hold expired → `11:55:03Z START`; serve pid 1653471 serving.

## Round-3 fixes (quorum R3 on 8f46a2be1: sonnet-A PASS, sonnet-B FAIL, haiku PASS)

| Finding (sonnet-B / sonnet-A) | Fix | Mutant → row |
|---|---|---|
| `stop_server` race: pid dies between `pid_alive` and `killpg` → ProcessLookupError kills the watcher | `try/except ProcessLookupError` around the kill block | except→KeyError: watch row 8 "killpg on a vanished pid -> no crash" RED (`ProcessLookupError(3, 'No such process')`) |
| `--timeout 0` → `signal.alarm(0)` CANCELS the budget | `args.timeout = max(1, args.timeout)` | line → `pass`: ask test "timeout 0" hangs (rc 124 under `timeout 30`). The first version of this row used a hung socket and stayed GREEN under the mutant (urlopen's per-read timeout bounded it) — replaced by a trickle server that sends one byte / 0.3 s so only the alarm can end the ask |
| (sonnet-A, non-blocking) bench `decode_tok_s` fell back to 0.0 when tN<=t1 | `None` instead of a fabricated 0.0 | — |

Real code: ask test 2/2 PASS (budget 6 s → 5.0 s; timeout 0 → 1.0 s), watch test 8/8 PASS.

## Round-4 fixes (quorum R4 on 5d17cc35d: sonnet-A FAIL, sonnet-B FAIL, haiku PASS)

Both sonnet lanes found the same gap independently. The alarm is one-shot. If it fired inside the gx10 leg, the broad `except` swallowed the TimeoutError and the lambda leg then ran with no wall-clock bound. Also, gx10's ssh timeout (`timeout + 30` against a caller's `remaining() - 30`) expired at the exact instant of the alarm, so the race was the default outcome.

| Fix | Mutant → result |
|---|---|
| lambda is not tried when < 1 s of budget is left; it records `not tried: lane budget Ns spent on gx10` | guard → `if False`: row "alarm spent in the gx10 leg" hangs, and the test's 60 s guard exits rc 3 (RED) |
| ssh timeout is `timeout + 10`, so it ends 20 s before the deadline and a hung gx10 raises TimeoutExpired without spending the alarm | — (a bound tighter than the one that was racing) |
| `--timeout 0` row reworked: gx10 now hangs in Python, so the alarm alone ends it (the lambda-skip guard would otherwise make the old row pass under the mutant) | `max(1,…)` → `pass`: rc 3 (RED) |
| the test has a daemon 60 s guard, so a hang is reported as a FAIL, not a stuck run | — |

Real code: 3/3 rows PASS (budget 6 → 5.0 s; timeout 0 → 1.0 s; gx10-leg alarm → 6.0 s, lambda not tried).

## Round-5 fixes (quorum R5 on af759e9c6: sonnet-A PASS, haiku PASS, sonnet-B FAIL). Round 6 NOT yet run

| sonnet-B finding | Fix | Row |
|---|---|---|
| `health()`'s `except Exception` swallowed the alarm's TimeoutError, which left post_chat unbounded | the alarm raises `LaneBudgetSpent(BaseException)`, which no `except Exception` can catch; cmd_ask catches it once and records the legs it cut | "alarm fires inside health()" (real health vs the trickle server) PASS 6.0 s. Mutant `BaseException→Exception` → hangs, 60 s guard rc 3 (RED) |
| a yielded server is never reaped, so each yield cycle leaves a zombie | `reap_children()` (waitpid -1 WNOHANG) in stop_server and on every watch tick | watch row "yielded server reaped, not a zombie" PASS |
| a malformed 200 crashed cmd_ask with no receipt | extraction guarded; the result is verdict unavailable plus `malformed response: …` | "malformed 200 -> unavailable receipt" PASS |
| (sonnet-A, non-blocking) under `--force-lambda`, the <1 s skip was mislabelled "spent on gx10" | message is now "< 1 s of the Ns lane budget left" | — |

Real code: ask 5/5 PASS, watch 9/9 PASS.

## Landing on batch/0.70.0 (aprender-60, 2026-09-24)
- PR #4282 was parked out of the B2 fold (#4315) because `scripts/bench_serve_4252.py` fails PERF-009 `check_no_competing_harnesses.sh`: it starts nothing but COMPUTES tok/s on its own, which makes it a second definition of how the project measures itself. The lane does not import it, so it is **dropped** here, not allowlisted. The bench table above was produced by that client at `519c54187`–`f5c28b106` (feat/4252-apr-dogfood-lane); re-measure through `scripts/perf_gate.sh` / `apr test llm bench`, never by restoring the client.
- The lane code (`apr_dogfood_lane_4252.py`) and both test files are byte-identical to `f5c28b106` (quorum R5). Round 6 has still not run; the lane stays advisory (`counts=false`).
- The lambda CPU dogfood server (`apr-dogfood-4252.service`, port 18252) moved from v0.69.3-rc.1 to the **v0.69.3-rc.2** x86_64-cpu release asset: tag `v0.69.3-rc.2` → `15c3032fc`, tarball sha256 matches the release `.sha256`, binary sha256 prefix `faa473e3`. The binary prints `+no-git`, so its provenance is the asset hash, not `--version`. Smoke: `/v1/chat/completions` "2+2" → `4`.
