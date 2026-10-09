# examples-nightly reds — receipts (#4756)

Before: CI run 37169058144 (main @316dee2cd4). Of the 36 rows, 29 fail and 7 time out.

After: `examples-36.tsv`, one run of the unchanged `scripts/dogfood_examples.sh` on the
code tree of d7d611f0e1, the last code commit on this branch. The run used debug builds,
`--timeout-secs 180` and a `--filter` that matched exactly the 36 rows, on 2026-10-09 from
23:19 to 23:37Z. It is a self-hosted x86-64 box with 4 CPUs pinned. It has a GPU that wgpu
can use, and libcuda with no CUDA device.

| class | rows |
|---|---|
| pass | 9 |
| needs-args | 11 |
| needs-hardware | 10 |
| needs-data | 5 |
| needs-feature | 1 |
| fail | 0 |
| timeout | 0 |

The script's `# summary` line has no needs-feature counter, so it sums to 35. The table
counts column 3 of the 36 rows.

## performance_parity on a GPU

Review round 2 found that performance_parity's GPU path ran past 25 minutes in a debug
build (rc 124 at 1505 s) when wgpu found an adapter. Since d7d611f0e1, a debug build runs
GPU-001 alone (`performance_parity.rs:52`, `:101`, `:107`). A release build still runs all 31
GPU benchmarks.

`performance_parity-gpu.txt` is the debug binary from the same run, re-run once with its
output kept, because the script keeps no per-example output. The file records the commit
and the binary's sha256. It printed `GPU detected and available`, ran GPU-001 on the GPU
(15.95 GFLOPS) and the debug-build note, and exited rc 0 in 10 s. The script reported the
same row as pass in 11 s.

On a host with no adapter, the example runs the same 8 CPU benchmarks without GPU-001.
They are not gated on the GPU (`performance_parity.rs:66` onward), and all 8 ran in that
10 s.

## Not measured here

- **The 10 CUDA rows on the nightly runner.** In the failing run they got
  `CUDA driver not found`, the error aprender-gpu gives when it cannot load libcuda.
  test_gemv_correctness shows that text on its row. The other nine rows show only their
  first output line. No classifier pattern matched it. Since 51b6d69fe8, aprender-gpu says
  `CUDA driver not found (libcuda.so)`, which is how its spec already writes it, and the
  classifier's `libcuda\.so` pattern matches that. This box has libcuda and no device, so
  its CUDA rows stop at `cuInit` instead. This run does not exercise the no-libcuda path:
  that path's class was checked against the regex, not run.
- **parity_035 with no Ollama server.** Here a server runs without the model, and the row
  is needs-data (`Model not found: phi2:2.7b ...`). With no server, the example prints
  `Ollama server not found at http://localhost:11434 ...`. Checked against the
  classifier's own regexes, that line is needs-data and not needs-args or needs-hardware.
  It was not run.
- **The issue's done-when** is the nightly itself: the same 36 rows with fail 0 and
  timeout 0. examples-nightly stays disabled until this lands on main. Then it is
  re-enabled for one green night, and that run is the receipt of record.
