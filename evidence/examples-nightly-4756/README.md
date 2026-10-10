# examples-nightly reds — receipts (#4756)

Before: CI run 37169058144 (main @316dee2cd4). Of the 36 rows, 29 fail and 7 time out.

After: `examples-36.tsv`, one run of the unchanged `scripts/dogfood_examples.sh` on
01107666b7, the last code commit on this branch, on 2026-10-10 from 00:30 to 00:34Z.

- Debug builds, `--timeout-secs 180`, and a `--filter` that matched exactly the 36 rows. The
  job stops before the run unless the filter matches 36 example targets.
- A self-hosted x86-64 box, 4 CPUs pinned, nice 19. It has a GPU that wgpu can use.
- libcuda hidden. The run was in a private mount namespace with `/dev/null` bind-mounted over
  the host's libcuda, which then read 0 bytes. The CUDA rows took the path of a host with no
  driver, the path whose `CUDA driver not found` the failing nightly's rows show.

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

A pass is a smoke test. The classifier reads only the exit status and the output, so pass
means the example ran to the end and exited 0. It checks no figure the example prints.

## Build time and the timeout

The run reused the build directory of the run before it, so its `secs` are mostly run time.
`examples-36-cold-4a79227e9f.tsv` is that run before, on 4a79227e9f from an empty build
directory: the same 36 classes, exit codes and cites. Its `secs` are build plus run, up to
293 s (agent_demo) and 235 s (publish_shell_safety). The script bounds only the run
(`scripts/dogfood_examples.sh:226`, "Only the RUN is bounded."), so a cold build cannot time
a row out.

`run-stage-secs.txt` times the run stage alone (`cargo run -q`, stdout to a file) for the six
rows with the highest `secs` in the cold run, after the 36-row run on the same tree. None ran longer than 15 s.

## What a debug build or a non-terminal run does differently

A release build run from a terminal does what it did before. Only a debug build, or a run
whose stdout or stdin is not a terminal, takes the shorter path:

- **pipeline_tui.** Its correctness check generated `GenerationConfig::default()`'s 100
  tokens (`crates/aprender-serve/src/generate/mod.rs:154`) through the unoptimized forward
  pass. On 4a79227e9f that took a 141 s run stage, out of the 180 s bound. A debug build now
  generates one token (`pipeline_tui.rs:19`, `:458`), and the run stage is 5 s. A release
  build still generates 100.
- **Bare runs with no arguments.** qa_verify, llama2-train and test_mac_worker keep their
  default only in a release build at a terminal (`qa_verify.rs:394`, `llama2/train.rs:236`,
  `test_mac_worker.rs:16`). Otherwise they print their usage and the row is needs-args.
- **Smaller debug workloads.** bug_hunter_demo (`:318`), design_by_contract (`:74`) and
  performance_parity (`:52`, `:373`) shrink only when `cfg!(debug_assertions)` is set.
- **Non-terminal runs.** api_server (`:103`), buggy_server (`:21`), brick_computer (`:698`)
  and calculator_tui (`:22`) take a bounded or one-shot path only when they are not run at a
  terminal, as each cited line tests it.

Release builds of the 36 rows were not run here. That their workloads are unchanged rests
on the gate lines cited above.

## A bug fix for every build

performance_parity's Q4_0 and Q8_0 benchmarks built 20×32 = 640 and 36×32 = 1152 bytes. The
dequantizers take 18-byte and 34-byte blocks and return an error for a length that is not a
multiple (`crates/aprender-serve/src/quantize/dequant.rs:49`, `:111`). 640 and 1152 are not
multiples, so `.expect` panicked in debug and release builds alike, and the failing nightly
row was fail. The sizes are now 18×32 and 34×32 (`performance_parity.rs:514`, `:522`).

## performance_parity on a GPU

Review round 2 found that performance_parity's GPU path ran past 25 minutes in a debug build
(rc 124 at 1505 s) when wgpu found an adapter. A debug build now runs GPU-001 alone
(`performance_parity.rs:52`, `:101`, `:107`). A release build still runs all 31 GPU
benchmarks.

`performance_parity-gpu.txt` is the debug binary from the 36-row run, run once more with its
output kept, because the script keeps no output per example. The file records the commit and
the binary's sha256. Outside the namespace, it printed `GPU detected and available`, ran
GPU-001 on the GPU (2.55 GFLOPS on the 4 pinned CPUs at nice 19) and printed the debug-build
note. It exited rc 0 in 23 s. The script reported the same row as pass in 14 s.

On a host with no adapter, the example runs the same 8 CPU benchmarks without GPU-001. They
are not gated on the GPU (`performance_parity.rs:66` onward).

## Measured since review round 3

- **The 10 CUDA rows with no libcuda.** All 10 are needs-hardware. aprender-gpu reports
  `CUDA driver not found (libcuda.so)` (`crates/aprender-gpu/src/driver/context.rs:48`), and
  the classifier's `libcuda\.so` pattern matches it. In run 37169058144 these rows got
  `CUDA driver not found` with no `(libcuda.so)`. test_gemv_correctness showed that text on
  its row, and the other nine showed only their first output line. No pattern matched, so all
  ten were fail.
- **parity_035 with no Ollama server.** `parity_035-no-server.txt` is the same tree in a
  private network namespace with loopback up and nothing listening. The example printed
  `Ollama server not found at http://localhost:11434 ...` and exited rc 1 in 1 s. The
  classifier's `not found at ` pattern makes that needs-data. In the 36-row run a server was
  up without the model, and the row is needs-data on `Model not found`.

## Not measured here

The issue's done-when is the nightly itself: the same 36 rows with fail 0 and timeout 0.
examples-nightly stays disabled until this lands on main. Then it is re-enabled for one green
night, and that run is the receipt of record.
