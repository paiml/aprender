# EXT-28 — C2 speed arms (aprender#4410)

EXT-001 §10 C2: `apr serve` and each competitor arm serve the blessed model
(`ours-q4k.gguf`, sha256 `97784e90405056f4529564a811acb333d2303b09ce493a361d75b5797e1158af`,
Qwen3.5-4B Q4_K_M, text-only) and are measured by one client at m=1. Absolute values
only; the apr/llama.cpp comparison exists only inside the EXT-19 ledger ratchet (T28).

| File | What |
|------|------|
| `c2_cell.sh` | the harness: every arm started under `env -i`, pinned to CPUs 0-15 with 16 threads, all arms **resident**, measured iterations **interleaved** (the order rotates each round), one warmup, N=5, `max_tokens` 32, temperature 0 |
| `assemble_cell.sh` | one record dir → one `cell.json` receipt (`speed_arms::CellReceipt`) |
| `prompt.txt` | the 681-byte prompt every arm is sent |
| `ollama-blob.json` | why Ollama is refused (S-14): the `qwen3.5:4b` blob carries 393 vision tensors of 834 and its server loads a CLIP encoder; ours carries 0 |
| `cells/*.json` | the five REX cells this run had no host for, recorded `not_run` |
| `lambda-cpu-median/{r1,r2,r3,plant,r4}/` | the first five records (median statistic, superseded — kept, see below) on lambda-cpu (GPU hidden): per-arm JSON, `cell.json`, and `logs/` (server logs + the raw timestamped SSE streams the comparator `artifact_sha256` covers) |

## Arms

- **apr** 0.70.0 @ 99c6c61b4 (`plant`: the same source plus a 1 s sleep per decode step)
- **llama.cpp** d1d3c3396 — the reference arm
- **mistral.rs** v0.9.4 @ 4400935, `serve --cpu --format gguf`
- **Ollama** 0.34.4 — refused, see `ollama-blob.json`

## lambda-cpu-median: the first method, medians of 5 (shared host, load average 30–55)

| record | arm | load ms | TTFT ms | ITL ms | e2e ms | decode tok/s | peak RSS MiB |
|---|---|---|---|---|---|---|---|
| r1 | apr | 74341 | 69228 | 473.1 | 85725 | 1.8236 | 7957 |
| r1 | llama.cpp | 5569 | 575 | 205.2 | 6827 | 4.3048 | 4441 |
| r1 | mistral.rs | 62518 | 10547 | 348.1 | 24456 | 1.9897 | 8154 |
| r2 | apr | 245785 | 92680 | 737.5 | 118202 | 0.8541 | 7938 |
| r2 | llama.cpp | 15202 | 601 | 216.2 | 7962 | 3.9375 | 4442 |
| r2 | mistral.rs | 106605 | 18407 | 375.6 | 29590 | 2.4143 | 8132 |
| r3 | apr | 85015 | 74574 | 409.6 | 87061 | 1.8045 | 7938 |
| r3 | llama.cpp | 7414 | 509 | 217.1 | 7239 | 4.0207 | 4441 |
| r3 | mistral.rs | 94756 | 10644 | 354.9 | 24518 | 1.9462 | 8132 |
| plant | apr (plant) | 59969 | 74167 | 2061.3 | 135832 | 0.4471 | 7956 |
| plant | llama.cpp | 23437 | 661 | 205.3 | 7962 | 3.5613 | 4441 |
| plant | mistral.rs | 59412 | 22376 | 484.4 | 37676 | 1.7647 | 8120 |
| r4 | apr | 121185 | 92332 | 500.8 | 111128 | 1.3832 | 7939 |
| r4 | llama.cpp | 10892 | 506 | 215.1 | 7210 | 4.1237 | 4441 |
| r4 | mistral.rs | 87078 | 15285 | 393.9 | 29883 | 1.8495 | 8129 |

## FALSIFY-EXT-022: RED on this host

`cargo test -p apr-cli --lib falsify_ext_022_planted_sleep_red` judges r1–r3 as the
floor, then the plant and r4. The plant is RED, as it must be. **The unpatched
control r4 is also RED**, so the test fails: on this host the ratchet cannot tell
a 1 s-per-step regression from host load.

Per-iteration apr decode tok/s (unpatched, same binary sha `678647001a49…`) spans
0.53–2.22 within and across records; llama.cpp stays within 2.8–4.7. Interleaving
put every arm under the same load, but the load does not hit them alike: the
rayon-based engines (apr, mistral.rs) lose far more to contention than llama.cpp,
so the load does not cancel. An earlier sequential pass (arms run one after
another) failed the same way.

What would make it hold is a method decision, not a tolerance edit made after
seeing the data: a quiet-host rule for records, or a contention-robust statistic
chosen in advance (e.g. best-of-N per arm). It is escalated, not decided here.

## Pre-registered second method (cop ruling, 2026-09-26)

Registered in this commit, before any record under it exists:

- **N = 5** measured iterations per arm, pinned in `c2_cell.sh` (not an env knob) and
  in `speed_arms::PREREGISTERED_ITERATIONS`; one warmup; arms resident, rounds
  interleaved with the order rotated each round (unchanged).
- **Statistic: best-of-5 decode tok/s per arm** (`best_of_n_decode`, i.e. the minimum
  per-token wall time). Host load only ever slows an iteration, so the best sample is
  the one least touched by it. Every sample is kept in `decode_tok_s_iters`, and
  `check_record` recomputes the best rather than trusting the field.
- **Load is recorded:** the 1-minute load average at each iteration's start, in
  `loadavg_1m_iters`.
- **TOLERANCE unchanged** (EXT-19 ratchet, floor × 0.95).
- **Pass condition, both sides:** under this statistic the unpatched control r4 must be
  GREEN against the r1–r3 floor, and the 1 s/step plant must still be RED. If the
  control stays RED at load above 30, that is recorded as a contention finding, not
  tuned away.

The new records go to `lambda-cpu/{r1,r2,r3,plant,r4}/`.

## lambda-cpu: records under the pre-registered method

Best-of-5 decode tok/s per arm; `samples` is every iteration in run order,
`load` the 1-minute load average at each iteration's start. The other timings are
medians. Host shared with other sessions' builds and tests on CPUs 0-15.

| record | arm | load ms | TTFT ms | ITL ms | e2e ms | decode tok/s (best) | samples | load | peak RSS MiB |
|---|---|---|---|---|---|---|---|---|---|
| r1 | apr | 118303 | 70655 | 873.8 | 100523 | 2.4454 | 1.25 2.45 0.87 0.68 2.28 | 54 29 46 46 54 | 7928 |
| r1 | llama.cpp | 12428 | 972 | 281.2 | 8921 | 5.0519 | 2.72 5.05 3.27 3.15 4.73 | 55 35 46 47 36 | 4440 |
| r1 | mistral.rs | 48080 | 14618 | 376.9 | 29286 | 2.2799 | 1.46 2.28 1.84 1.61 2.03 | 46 31 35 51 44 | 8078 |
| r2 | apr | 55790 | 54715 | 351.2 | 65060 | 2.8955 | 2.51 2.48 1.76 2.51 2.9 | 40 31 35 39 35 | 7957 |
| r2 | llama.cpp | 6377 | 522 | 200 | 6475 | 4.5536 | 4.55 4.41 4.25 3.98 4.39 | 40 30 36 39 36 | 4440 |
| r2 | mistral.rs | 122249 | 10779 | 338 | 20951 | 2.7287 | 2.65 2.64 2.11 2.73 2.72 | 34 30 30 36 31 | 8290 |
| r3 | apr | 150565 | 189291 | 779.1 | 231325 | 2.3458 | 0.88 1.35 0.87 0.6 2.35 | 57 54 48 57 51 | 8046 |
| r3 | llama.cpp | 18292 | 1261 | 281.1 | 9110 | 4.9315 | 1.83 2.77 3.73 3.1 4.93 | 51 41 48 60 38 | 4441 |
| r3 | mistral.rs | 38618 | 22035 | 531.9 | 39909 | 2.6317 | 1.52 0.64 1.44 2.63 2.3 | 61 29 43 58 38 | 8061 |
| plant | apr | 132889 | 378800 | 3359.6 | 469080 | 0.5261 | 0.53 0.29 0.24 0.24 0.3 | 43 28 122 140 129 | 7926 |
| plant | llama.cpp | 16859 | 1356 | 373.4 | 12693 | 4.3129 | 4.31 2.36 2.2 2.28 2.58 | 43 26 116 142 63 | 4440 |
| plant | mistral.rs | 68056 | 51681 | 2417 | 125465 | 0.6848 | 0.68 0.37 0.32 0.37 0.59 | 38 78 33 146 65 | 8069 |
| r4 | apr | 294106 | 238300 | 1742.3 | 313036 | 0.7930 | 0.79 0.52 0.54 0.42 0.35 | 79 76 79 25 41 | 7903 |
| r4 | llama.cpp | 13853 | 1806 | 371.2 | 13012 | 2.9239 | 2.34 2.39 2.18 1.71 2.92 | 78 58 80 32 70 | 4440 |
| r4 | mistral.rs | 124547 | 51610 | 1904.9 | 108196 | 1.1993 | 0.45 0.49 0.44 0.32 1.2 | 23 42 62 76 85 | 7966 |

`plant` and `r4` were re-run (12:00–12:41Z): the first attempt found the mistral.rs
binary deleted by a disk cleanup during r3 and exited 127. It was rebuilt from the
same commit 4400935; the build is not bit-reproducible, so r4/plant carry engine
sha `f79c5b54…` where r1–r3 carry `799d80e7…`. mistral.rs is not in the ratchet ratio.

### FALSIFY-EXT-022 under best-of-5: control still RED — a contention finding

`cargo test -p apr-cli --lib falsify_ext_022_planted_sleep_red`: all five receipts
pass `check_cell` (statistic, N, samples and the recomputed best). The plant is RED
(ratio 0.122). **The unpatched control r4 is also RED**: ratio 0.271 against the
r1–r3 floor 0.484 (`Red { tag: "v0.70.0", ratio: 0.2712, floor: 0.4841 }`).

The plant and r4 ran while the host load average swung 23–146. In r4 no apr
iteration beat 0.79 tok/s (r1–r3 best 2.35–2.90). Best-of-N only protects a record
that has at least one uncontended iteration. Under sustained load, every sample is
contended, and llama.cpp still degrades less than apr (2.92 vs 5.05 best), so the
ratio falls. The 1-minute load average lags, so a low reading (r4 iteration 4: 25) does
not mean that iteration ran quiet. Per the ruling this is recorded, not tuned away:
TOLERANCE and the statistic are unchanged.

## Amendment: quiet-host rule (cop ruling, 2026-09-26, after the best-of-5 RED)

Registered in this commit, before any record under it exists. The statistic, N and
TOLERANCE are unchanged. The records above stay as evidence of why this amendment exists.

- **Every record runs inside a unit of the reserved slice.** It is started as
  `systemd-run --user --slice=speedledger.slice -p AllowedCPUs=$CPUS …`; the cores
  are reserved through infra-8d's slice, and every foreign slice is fenced off them.
- **The mechanism is proven, not declared.** `c2_cell.sh` refuses (exit 3) unless
  its own `Cpus_allowed_list` equals `$CPUS` and its cgroup lies under the slice.
  Both values are written into each arm's `conditions` (`cpus_allowed_list`,
  `isolation_unit`), and `check_record` refuses a record where they disagree.
- **The cpuset must be sibling-complete.** On lambda, CPU 24+k is the SMT sibling of
  core k, so a reserved set must hold both threads of each core. For example,
  `16-23,40-47` is 8 cores, 16 threads.
- **Preconditions, recorded in the receipt:**
  - Host `load1_at_start` is recorded, not gated. Host-wide load includes the
    fenced-off rest of the host.
  - The reserved cpuset's busy share is measured over 1 s, with every arm idle,
    before each measured iteration (`cpuset_busy_pct_iters`, from `/proc/stat`). It
    must be ≤ 10 % (`PREREGISTERED_MAX_CPUSET_BUSY_PCT`). A busy iteration turns the
    record RED; it is not retried or dropped.
- **Pass condition as before:** the unpatched control GREEN against the r1–r3 floor, and
  the 1 s/step plant RED.

At registration, the unfenced `16-23,40-47` measured about 75 % busy. The
precondition refuses that host state, as intended.
