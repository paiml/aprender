# PMAT-3556 — receipt (PP-LLAMA-001 §12 row 2)

**Ticket:** PMAT-3556 (aprender#3556) — `perf002_prefill_path_probe.sh` + `decompose()` with mandatory refusals. **Kind:** code.
**Row:** PP-LLAMA-001 §12 row 2, `expires: 2026-09-19`, owner `serve`, "needs a gx10 window". One of the two ROOT rows whose expiry has `guard-tree` red on every aprender PR since midnight UTC 2026-09-20; the other five expired rows inherit from these two.
**Authority:** Noah's decision on the expiry, 2026-09-20: **discharge** — not amend the date, not retire the rows. Split agreed with aprender-bf: they take row 1 (the Blackwell guard, `perf041` wiring, PP-26 witness), this is row 2.
**Branch:** `PMAT-3556-perf002-prefill-probe` from `origin/main` at `338f1d49b`, in a worktree of the persistent aprender clone; `~/src/aprender` untouched.

## Why the row exists

§9 #1 sizes the defect from a SINGLE point: 16.75 s at 513 prompt tokens = 32.65 ms per prompt token, "fitted over 28 of 30 samples". **One point has no slope.** "Linear at 32.65 ms/token" and "a 16.75 s fixed cost independent of prompt length" fit that point equally well and imply opposite defects — the first is the serial per-token loop at `gpu_profile.rs:517-531`, the second means §9 #1 is scoped wrong and the lever is somewhere else. §10 registers the linearity as a prediction with `kill if: flat in prompt length, or no collapse`.

## Change, all of the diff

- `scripts/perf002_prefill_path_probe.sh` — the driver. Resolves the binary through `apr_bin.sh`, runs the decider's `--selftest` BEFORE measuring and refuses to report a verdict if it fails, sweeps both arms (a server restart per arm, because `BATCHED_PREFILL` is read at startup), writes a marker on EVERY exit path carrying host / cc / commit / binary sha256 / started_utc / slope / refused_rule.
- `scripts/perf002_prefill_path_probe.py` — the decider, and the half that can be proven without a GPU. `decompose()`, `verdict()`, `measure()`, `--selftest`.
- `docs/roadmaps/entries/PMAT-3556.yaml` + the regenerated `docs/roadmaps/roadmap.yaml`, and this receipt.

Nothing under `crates/aprender-serve/**` or `.github/workflows/cuda-nightly.yml` — row 1's, by agreement.

## The refusals

The row names four; there are five, because the fourth defect below needed one.

| rule | fires when | why it is not a defect claim |
|---|---|---|
| `too_few_distinct_x` | < 2 distinct prompt-token counts | no slope exists — this is the single-point shape §9 #1 was sized from |
| `negative_slope` | slope < 0 | prefill cannot get cheaper with more tokens; the workload or the harness moved |
| `r2_below_bound` | r² < 0.95 | the points are not a line, so `fixed + slope*n` is the wrong model |
| `bimodal` | two per-token costs separated by 3× the within-mode spread | REPORTS per-mode fits rather than rejecting; §9 #1's own sizing discarded 2 of 30 samples to get one line |
| `implausibly_fast` | any sample < 5 ms | guards the HARNESS: no prefill of 64+ tokens is that fast on any path |

Every one maps to exit 2, never to exit 1. A guard that names a code cause for a box it could not evaluate has fired repeatedly in this repo.

## Result — MECHANISM_CONFIRMED

gx10 (GB10, **cc 121**, so §9 #1 is in scope), commit `338f1d49`, binary sha256 `f874b0be…`, `qwen2.5-coder-1.5b-instruct-q4_k_m`, ladder 64/128/256/384/513 ×3, started 2026-09-20T08:38:47Z.

| arm | result |
|---|---|
| default | **linear at 9.042 ms/token**, r² **0.999947**, n=15, fixed 0.067 s. 513 tok → 4713 / 4717 / 4719 ms |
| `BATCHED_PREFILL=1` | **0.099 s** at ~512 tokens (median of 3). 513 tok → 123 / 99 / 98 ms |

**Both halves of §10's registered prediction survive.** "Flat in prompt length" is dead at r² 0.999947; the collapse is 48× and 3.5× better than the predicted 0.35 s. The slope reproduced across independent runs at 9.054 and 9.042 ms/token — 0.1% apart.

## The prefix cache — and why §9 #1's SIZE is deliberately not re-measured here

With a repeated prompt the default arm is **bimodal**, and both modes span every rung:

```
10 of 15 samples   flat, shapeless   0.045 ms/token   r2 0.3699
 5 of 15 samples   a clean line      8.970 ms/token   r2 1.0000
```

A prefix cache explains it exactly, including the r²: a hit's latency does not depend on prompt length. aprender-bf identified the mechanism — `paged_kv/mod_quantized_paged.rs:177 find_longest_prefix`, `scheduler/chunked_prefill.rs:366 record_prefix_cache_hit`, counter at `:187`.

Their proposed falsifier — bucket samples by `prefix_cache_hits` — is better than mine and **is not runnable**: the counter is exposed nowhere. `/metrics` carries `realizar_requests_*` and no prefix/cache counter; there is no `/stats` route; `scheduler.stats()` reaches no handler. Filed as **aprender#3553**.

So the cache is excluded by CONSTRUCTION (a unique nonce per sample) rather than by label, and the evidence is a controlled pair: removing prefix reuse removes exactly the flat population and leaves a line agreeing with the repeat arm's linear mode to within 1%. That is one variable across two runs, not a per-sample label, and this receipt says so rather than claiming the stronger thing.

**Consequence:** 32.65 ms/token against 9.04 measured today is 3.6×, but the original is a different model at a much earlier commit, and without the label any prefill figure is a blend of hits and real prefills in an unknown proportion — and the proportion moves the number. Re-sizing §9 #1's `[C]` figure waits for #3553.

## Four harness defects, each a confident WRONG ANSWER rather than an error

Found by running the probe; all four are in its header so the next person does not re-derive them.

1. **`time_starttransfer` times response HEADERS, not the first token.** A streaming server sends headers before prefilling anything. The first draft produced ~0.6 ms samples, r² ≈ 1, and the verdict `PREDICTION_KILLED / the default arm is FLAT`. None of the four spec'd refusals catches that — the data really is linear, positive-slope, unimodal, two distinct x. A false finding about serve code would have gone to the spec owner. Fixed by reading TTFT off the SSE stream, and by `implausibly_fast`, whose self-test fixture is the exact fooling data plus an anti-vacuity leg proving the floor is what catches it.
2. **CUDA warm-up on the first request per arm** — 0.76 s at 64 tokens against 0.13 s at 128, a negative slope. `decompose()` refused rather than reporting a collapse, which is the refusals working; the harness should not have handed it that shape. One warm-up per arm, discarded.
3. **A repeated prompt measures the cache** (above).
4. **The r² floor was justified for the DEFAULT arm and wrongly applied to the batched one.** A serial per-token loop is a sum of n identical steps and is nearly exactly linear; a batched prefill is a fixed overhead plus a small per-token term with real variance and has no reason to reach 0.95. The probe reported UNMEASURABLE about the clearest result in the run. The collapse is now read off the MEASUREMENT — which is what §10 actually predicts, a time at a length, not a line — with the fit still reported.

verification:
  cmd=python3 scripts/perf002_prefill_path_probe.py --selftest  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3556-receipt.md  sha256=0   # OK (8 cases): one per refusal, bimodal-emits-two-fits, the floor's anti-vacuity leg, the clean-sweep CONTROL, and the planted-slope recovery at 32.650 ms/token
  cmd=bash scripts/perf002_prefill_path_probe.sh (gx10, APR_BIN=worktree CUDA build)  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3556-receipt.md  sha256=0   # MECHANISM_CONFIRMED; marker cc=121 commit=338f1d49b sha256=f874b0be… slope=9.042 refused_rule=null
  cmd=the same probe with --repeat-prompt  claimed_exit=2  rerun_exit=2  log_path=docs/audits/impl-PMAT-3556-receipt.md  sha256=0   # UNMEASURABLE / bimodal, per-mode fits 0.045 ms/token r2 0.3699 (n=10) and 8.970 ms/token r2 1.0000 (n=5) — the controlled half of the cache pair
  cmd=curl -s http://127.0.0.1:PORT/metrics | grep -iE 'prefix|cache'  claimed_exit=1  rerun_exit=1  log_path=docs/audits/impl-PMAT-3556-receipt.md  sha256=0   # no output: the counter is unreachable, hence #3553
  cmd=bash -n scripts/perf002_prefill_path_probe.sh; bashrs lint scripts/perf002_prefill_path_probe.sh  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3556-receipt.md  sha256=0   # parses; 0 errors
  cmd=bash scripts/check_no_competing_harnesses.sh; check_hardcoded_paths.sh; check_append_only_ledgers.sh; check_reconcile.sh; check_roadmap_fragment_required.sh  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3556-receipt.md  sha256=0   # all OK (count=0 baseline=0 on the harness check)

## Deviations, named

- Executed direct; review is the quorum. `goal.sh set` not run.
- The gx10 run uses `APR_BIN` pointing at a CUDA build made in the probe's own worktree, because `apr_bin.sh` attributes a binary to the tree it was BUILT in and the forjar-managed `~/.cargo/bin/apr` — same commit, `0.68.2 (338f1d49)` — was built in the runner's `_work` checkout. The forjar-declared path was not touched.
- §9 #1's size is not re-measured (see above). This row confirms the MECHANISM only, which is what it asks for.

## Round 2 — quorum r1 found two true defects, both mine

`NOT AGREED: lane 1=PASS, lane 2=FAIL, lane 3=PASS`. The failing lane was right on both counts.

1. **The marker was NOT written on every exit path**, while the file's own header said it was. `cd "$ROOT" || exit 2` and `. scripts/apr_bin.sh || exit 1` both bail BEFORE `write_marker()` is defined — so the two most likely early failures (a moved checkout, an unattributable binary) produced silence, and a missing marker is supposed to mean "the lane did not run at all" rather than "it ran and could not speak". This is the same shape as every "a guard that reports what it did not measure" finding I cleared this week, in my own file, asserted in a comment I wrote.
   Fixed: `OUT`/`MARKER` and a python-free `bail_marker` are established first, before anything that can fail. Measured after: running the script from a non-repo directory exits 2 and writes `{"status": "UNMEASURABLE", "reason": "apr_bin.sh could not attribute an apr binary to this tree", ...}` where it previously wrote nothing.
2. **`UNMEASURED` vs `UNMEASURABLE`.** The header documents exit 2 as `UNMEASURABLE`; `write_marker` wrote `UNMEASURED`. The sibling `perf041` writes `UNMEASURED`, which is where the word came from, but this row's text says `UNMEASURABLE` and a marker a downstream checker reads must not use two vocabularies for one state. Now consistently `UNMEASURABLE`.

verification (round 2):
  cmd=cd /tmp/<scratch> && bash p.sh  claimed_exit=2  rerun_exit=2  log_path=docs/audits/impl-PMAT-3556-receipt.md  sha256=0   # marker written where the first draft wrote nothing; status UNMEASURABLE, reason names apr_bin.sh
  cmd=grep -c UNMEASURED scripts/perf002_prefill_path_probe.sh  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3556-receipt.md  sha256=0   # 0
  cmd=bash -n scripts/perf002_prefill_path_probe.sh; bashrs lint  claimed_exit=0  rerun_exit=0  log_path=docs/audits/impl-PMAT-3556-receipt.md  sha256=0   # parses; 0 errors
