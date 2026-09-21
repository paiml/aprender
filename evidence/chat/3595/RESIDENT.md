# #3595 — `apr chat` keeps the Qwen3.5 hybrid resident: per-turn time and memory, before and after

Measured by aprender-c7, 2026-09-21, on both GPU hosts, under the cop's GPU rule (every run
`flock`/`gpu-q`-serialized and `choom -n 1000`; rev 5 priority 1 for the late cells).

- **before**: `apr 0.69.0 (225b2a9ab)` — `/mnt/nvme-raid0/targets/ladder-069/release/apr`, the release this row fixes.
- **after**: `apr 0.69.0 (f38e7483d)` — the resident session + the done_when 3 refusal text. The head under
  review adds one commit on top (the per-thread CUDA context bind, `4d4031194` on main); it changes nothing a
  single-threaded `apr chat` does.
- **the run** (`chat_pair.sh`): a two-turn chat at greedy, `--gpu --temperature 0 --max-tokens 1024` —
  turn 1 *"My name is Zorblat. What is the capital of Peru? Answer in one word."*, turn 2 *"What is my name?
  Answer in one word."* Turn seconds are chat's own `[Xs]` line. Memory is sampled every 0.5 s from THAT
  `apr` process: VmRSS, and its CUDA allocation where `nvidia-smi` reports one per process (on GB10's unified
  memory it does not — the RSS column is the unified number there).

| host | size | turn s before | turn s after | `Backend:` lines before→after | fallbacks before→after | peak RSS MiB before→after | peak GPU MiB before→after | replies before | replies after | same replies |
|---|---|---|---|---|---|---|---|---|---|---|
| lambda | 0.8B | 1.5 1.3 | 0.2 0.3 | 2→1 | 0→0 | 2678→2682 | 1044→1116 | Santo Domingo / Zořblat | Santo Domingo / Zořblat | yes |
| lambda | 2B | 3.0 2.5 | 0.3 0.4 | 2→1 | 0→0 | 5786→5797 | 1928→1930 | Lima / Zorblat | Lima / Zorblat | yes |
| lambda | 4B | 7.0 19.2 | 2.8 15.2 | 2→1 | 0→0 | 10454→10475 | 3512→3580 | Lima / <think> | Lima / <think> | yes |
| lambda | 9B | 78.4 29.6 | 47.3 39.9 | 2→1 | 2→2 | 24079→24157 | 5746→5884 | Lima / Zorblat | Lima / Zorblat | yes |
| lambda | 27B | 40.7 39.6 | 10.5 16.7 | 2→1 | 0→0 | 57444→57444 | 16878→17014 | Lima / Zorblat | Lima / Zorblat | yes |
| gx10 | 0.8B | 2.2 2.2 | 0.4 0.6 | 2→1 | 0→0 | 2694→2716 | 175→175 | Santo Domingo / Zořblat | Santo Domingo / Zořblat | yes |
| gx10 | 2B | 4.8 4.9 | 1.5 0.9 | 2→1 | 0→0 | 5805→5827 | 175→175 | Lima / Zorblat | Lima / Zorblat | yes |
| gx10 | 4B | 15.3 48.5 | 8.1 43.9 | 2→1 | 0→0 | 10469→10495 | 175→175 | Lima / <think> | Lima / <think> | yes |
| gx10 | 9B | 123.1 57.6 | 10.0 10.0 | 2→1 | 2→0 | 31980→31482 | 175→175 | Lima / Zorblat | Lima / Zorblat | yes |
| gx10 | 27B | 130.9 98.3 | 33.5 51.9 | 2→1 | 0→0 | 67344→67438 | 175→175 | Lima / Zorblat | Lima / Zorblat | yes |

## What the table shows

- **One `Backend:` line per session, not one per turn** — the model is built, uploaded and F2-checked once.
  Before, every turn rebuilt it (`2` lines for 2 turns; on lambda 9B the GPU memory trace went
  1187 → 6936 → 1576 MiB twice).
- **Per-turn time falls most where the rebuild dominated** (small models: lambda 0.8B 1.5 s → 0.2 s, 2B
  3.0 s → 0.3 s; gx10 0.8B 2.2 s → 0.4 s). Where a turn is dominated by a long generation (4B turn 2 is a
  1024-token thinking loop) the saving is the rebuild's share and no more.
- **The replies are identical before and after** in every cell measured: the resident session changes
  where the model lives, not what it computes (the session's own tests assert token-identity with the
  one-shot `apr run` path, on the CPU and the GPU, on a reused state and on a reset one).
- **Peak memory is unchanged** (the session holds what one turn used to peak at — for the whole session
  instead of re-allocating it each turn).

## Measured, and NOT caused or fixed by this row (identical before and after)

- **0.8B answers "Santo Domingo" / "Zořblat"** — the same on `--no-gpu` (CPU control): the model at greedy.
- **4B turn 2 never answers**: a `<think>` repetition loop until the 1024-token budget, on both hosts —
  apr's ChatML keeps turn 1's reasoning in the history (Qwen's own template drops it). Routed to #3723/#3755
  (aprender-fd), which renders the model's own template.
- **9B: the F2 guard rejects the GPU path on turn 1's prompt** (`diverges from CPU at position 23 … cosine
  0.8338`, argmax equal) on lambda, before AND after; the session fails closed to the CPU (approved by the
  cop). It is NOT the #3759 RMSNorm-epsilon defect: aprender-37 measured the same position and cosine on the
  #3759 fix binary on both hosts; the per-op localization is aprender-37's (#3751 or a new P0). Also measured: F2 receipts thrash between builds of
  different receipt schema (#3748).
- **gx10 9B reproduces the same F2 rejection** (`position 23 … cosine 0.8390`, GB10 sm_121) in its before run.
  Its after run did NOT re-validate: turn 2 of the before run had written a receipt (65 positions passed) and
  the after run read it (`F2 guard: receipt matches … skipped`). So the gx10 9B after row is a GPU run the
  guard did not re-check; its turn times are not comparable to the before run's CPU-fallback turns, and the
  row says nothing about the divergence.

## done_when 3, end to end

Alfredo's file, `Qwen3.5-4B-UD-Q4_K_XL.gguf`, `apr chat --gpu` on the after binary — ONE route message:

```
warning: GPU (CUDA) qwen35 path rejected, falling back to CPU: the CUDA model would not build: Operation 'qwen35_cuda_upload' not supported: 'qwen35.blk.0.ssm_alpha.weight' is F16 (GGML type 1), which has no verified GPU GEMV kernel — the file is fine and runs on the CPU; a Q4_K_M build of this model keeps every projection in a GPU-eligible type
```

(no `[qwen35: … does not implement it yet (#3090)]` banner; `qwen35` appears on 1 stderr line(s) in total)
