# #3604 — F2 guard receipt: before/after on the 144-word row

**Box:** this dev box, `NVIDIA GeForce RTX 4090` (24564 MiB), `nvcc` at `/usr/bin/nvcc`.
**Binary:** `apr 0.68.2 (e6f77c98c)` built from this branch's working tree with
`CARGO_TARGET_DIR=<worktree>/target-pinned` — pinned deliberately: the first build
went to the shared `/mnt/nvme-raid0/targets/aprender` and produced a binary from a
*different* worktree (`00e5d2dde`, no `--revalidate`), which is exactly the
stale-binary class the fleet was warned about the same evening.
**Model:** `/home/noah/models/Qwen3.5-4B-Q4_K_M.gguf`, 2,740,937,888 bytes,
sha256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
**Prompt:** 145 words (the "history of computing" paragraph), `--max-tokens 1`.
**Receipt dir:** `APR_F2_RECEIPT_DIR=<scratch>/f2-receipts`, emptied before run A.
**GPU occupancy** (`nvidia-smi memory.used`) recorded immediately before every run.

## The number (`done_when` 6)

| run | page cache | GPU mem at start | wall | guard | receipt |
|---|---|---|---|---|---|
| A `--revalidate` | cold (first load) | 1639 MiB | 20.86 s | **9,752 ms**, 65 positions | written |
| B plain | warm | 1463 MiB | 7.35 s | **0** — skipped | read, sha256 1,173 ms |
| C plain | warm | 1467 MiB | 7.34 s | 0 | read, sha256 1,171 ms |
| **D `--revalidate`** | **warm** | 563 MiB | **17.18 s** | **9,747 ms**, 65 positions | rewritten |
| **E plain** | **warm** | 1639 MiB | **7.39 s** | **0** | read, sha256 1,173 ms |

**D→E is the honest before/after** (same page-cache state): **17.18 s → 7.39 s, −9.8 s
wall**. The guard itself goes **9,747 ms → 0**. The key costs **1,173 ms** every run —
the whole-file sha256 at ~2.3 GB/s — so the *net* saving attributable to the guard is
**~8.6 s**, and the receipt line prints the hash cost separately so it cannot hide in
either number. A→B overstates the win (−13.5 s) because A also paid a cold page cache;
it is recorded, not claimed.

The fresh-run guard cost (9.5–9.9 s across five fresh runs) matches the instrument
lane's 9,526–9,553 ms on lambda 4090 (#3598 row 1), so the "before" is the same
phenomenon, not a slower box.

## The falsifiers (`done_when` 3, 4), end to end in the real binary

Each planted receipt was perfect in two keys and wrong in the third; each run
**re-validated** (full 9.7 s guard, `[source=fresh]`) and **named the key**:

| run | plant | guard line |
|---|---|---|
| F | `model_sha256` → `bbbb…` | `validating on this run (receipt is for model bbbbbbbbbbbb…, this file is 00fe7986ff5f…)` |
| G | `apr_version` → `0.61.0` | `validating on this run (receipt written by apr 0.61.0, this is 0.68.2)` |
| H | `device` → `NVIDIA GB10` | `validating on this run (receipt written for NVIDIA GB10, this device is NVIDIA GeForce RTX 4090)` |
| I | file overwritten with `{not json` | `validating on this run (receipt unreadable (…: not a receipt: …))` |
| J | file deleted | `validating on this run (no receipt for this model)` |

Absence and unreadability both validate and are **distinguishable** from each other
(`done_when` 4). After J the receipt on disk reads `NVIDIA GeForce RTX 4090` again.

`--revalidate` on a perfect receipt (runs A, D) validates and rewrites, reason
`--revalidate` (`done_when` 2). `done_when` 1 is B/C/E.

## `done_when` 5 — partially, and stated

`[source=receipt]` / `[source=fresh]` is on the guard's stderr line today, alongside
`sha256 <ms>` and (fresh) `passed in <ms>`. The `apr run --json` field lands with
#3606's `StageTimings` (which is not on `main` yet): `F2Outcome { source, validate_ms,
sha256_ms, receipt_path }` is returned to the call site for exactly that purpose, so it
is one field on #3606's side, not a re-plumb.

## What was NOT measured, and why

- The CPU-reference vs GPU-probe split inside the 9.7 s — #3606 instruments it; out
  of scope here by the ticket's own list.
- A cheaper identity than the whole-file sha256 (1.17 s/run). A candidate is a
  follow-up with its own falsifier; the whole-file hash is the only identity under
  which planted receipt F can be relied on to re-validate.

Raw stderr for every run: `run_*.stderr.txt` in this directory; the receipt that run
D wrote: `receipt_example.json`.
