# examples-nightly reds — receipts (#4756)

Before: CI run 37169058144 (main @316dee2cd4). Of the 36 rows, 29 fail and 7 time out.

After: `examples-36.tsv`, one run of the unchanged `scripts/dogfood_examples.sh` on
73f34268e6, the last code commit on this branch, in a job on 2026-10-10 from 01:23:54Z to
01:41:20Z.

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

The job before this one deleted the build directory when it finished (00:34:45Z), so this
run built every example from an empty build directory. Each row's `secs` is build plus run,
up to 137 s (agent_demo) and 136 s (publish_shell_safety). The script bounds only the run
(`scripts/dogfood_examples.sh:226`, "Only the RUN is bounded."), so a cold build cannot time
a row out. `examples-36-cold-4a79227e9f.tsv` is an earlier cold run, on 4a79227e9f: the same
36 classes, exit codes and cites, with `secs` up to 293 s (agent_demo).

`run-stage-secs.txt` times the run stage alone (`cargo run -q`, stdout to a file) after the
36-row run on the same tree. It covers the six rows with the highest `secs` in the
4a79227e9f cold run, plus bug_hunter_demo and design_by_contract. None ran longer than 10 s.

## What a debug build or a non-terminal run does differently

A release build run from a terminal does what it did before. Only a debug build, or a run
whose stdout or stdin is not a terminal, takes the shorter path:

- **pipeline_tui.** Its correctness check generated `GenerationConfig::default()`'s 100
  tokens (`crates/aprender-serve/src/generate/mod.rs:154`) through the unoptimized forward
  pass. On 4a79227e9f the row's `secs` in the cold run, build included, were 137, and its run
  stage timed alone afterwards on the same tree took 141 s (`run-stage-secs-4a79227e9f.txt`),
  against the 180 s bound. The aprender-serve library had been built for an earlier row, so
  the row's build was the example alone. The two were timed separately on a shared box, so
  the run stage alone can take longer than the whole row did, and here it did by 4 s. Both
  put the row within 43 s of the bound. A debug build now generates one token
  (`pipeline_tui.rs:19`, `:458`), and on 73f34268e6 its run stage alone took 3 s
  (`run-stage-secs.txt`). A release build still generates 100.
- **Bare runs with no arguments.** qa_verify, llama2-train and test_mac_worker keep their
  default only in a release build at a terminal (`qa_verify.rs:394`, `llama2/train.rs:236`,
  `test_mac_worker.rs:16`). Otherwise they print their usage and the row is needs-args.
- **Smaller debug workloads.** bug_hunter_demo (`:318`), design_by_contract (`:74`) and
  performance_parity (`:52`, `:373`) shrink only when `cfg!(debug_assertions)` is set. The
  two bug-hunter examples then scan `crates/aprender-orchestrate/src/bug_hunter`, 34 tracked
  files, instead of the whole tree. `bug-hunter-scan.txt` keeps their debug output: the scan
  path they printed and its result, 0 findings. Both scans in that capture were cache hits,
  so it shows the path and the result, not a fresh scan.
- **Non-terminal runs.** api_server (`:103`), buggy_server (`:21`), brick_computer (`:698`)
  and calculator_tui (`:22`) take a bounded or one-shot path only when they are not run at a
  terminal, as each cited line tests it.

Release builds of the 36 rows were not run here. That their workloads are unchanged rests
on the gate lines cited above.

## Bug fixes for every build

These change debug and release builds alike. Each one lets a run that stopped on an error
reach its end. None of them shortens a run.

- **performance_parity.** Its Q4_0 and Q8_0 benchmarks built 20×32 = 640 and 36×32 = 1152
  bytes. The dequantizers take 18-byte and 34-byte blocks and return an error for a length
  that is not a multiple (`crates/aprender-serve/src/quantize/dequant.rs:49`, `:111`). 640
  and 1152 are not multiples, so `.expect` panicked, and the failing nightly row was fail. The
  sizes are now 18×32 and 34×32 (`performance_parity.rs:514`, `:522`).
- **agent_demo.** Its fallback driver's context window was 4096 tokens. The runtime builds the
  window with the manifest's `max_tokens` as the output reserve
  (`crates/aprender-orchestrate/src/agent/runtime.rs:340`), the default manifest's
  `max_tokens` is 4096 (`agent/manifest.rs:110`), and the input budget is the window minus
  the reserve (`serve/context.rs:32`). That left 0 input tokens, so every run stopped on
  `ContextOverflow`. The window is now 32 768 (`agent_demo.rs:399`).
- **parity_035.** With a server up and the model not pulled, `/api/generate` answers with an
  error object that the response decoder cannot read, and the run failed on the decode. The
  example now asks `/api/show` first. Only a 404 there is reported as `Model not found` (exit
  1). A failed request or any other status is returned as an error, a failure of the server
  (`parity_035_m4_verification.rs`, the block after the TCP probe).

## Every other change, by kind

- **A usage line and exit 2 for a missing required argument**, where the example panicked or
  returned an error: check_layer4, constrain_mask_overhead, qwen35_parity, ssc_eval,
  ssc_preflight. rex, qwen35_prefill_parity and think_ab already printed a usage line and now
  start it with `Usage`, the word the classifier's needs-args pattern matches
  (`^(Usage|...)`).
- **A message naming what is missing**: qa_chat and qa_serve (`Model not found: pass --model
  PATH ...`), publish_shell_safety (`[MISSING] <name> not found at <path>`), finetune_real
  (`No tokenizer found in HF cache`), profile_cuda_trainer (`requires the 'cuda' feature`),
  aprender-gpu's driver loader (`CUDA driver not found (libcuda.so)`), and parity_035's
  `Ollama server not found at ...` when nothing listens on the port.
- **Optional arguments.** `qa_verify -- --all` and `test_mac_worker <host:port>` run the full
  workload from any build, `api_server -- --serve` keeps serving with no terminal, and
  buggy_server takes an iteration count as its bound. The book pages for qa_verify now show
  `--all`.
- **The gated shorter paths** listed under "What a debug build or a non-terminal run does
  differently".

## performance_parity on a GPU

Review round 2 found that performance_parity's GPU path ran past 25 minutes in a debug build
(rc 124 at 1505 s) when wgpu found an adapter. A debug build now runs GPU-001 alone
(`performance_parity.rs:52`, `:101`, `:107`). A release build still runs all 31 GPU
benchmarks.

`performance_parity-gpu.txt` is the debug binary from the 36-row run, run once more with its
output kept, because the script keeps no output per example. The file records the commit and
the binary's sha256. Outside the namespace, it printed `GPU detected and available`, ran
GPU-001 on the GPU (6.34 GFLOPS on the 4 pinned CPUs at nice 19) and printed the debug-build
note. It exited rc 0 in 13 s. The script reported the same row as pass in 12 s, build
included.

It also printed `Overall: Some benchmarks failed (5/9, 56%)` and exited 0. Its `fn main()`
returns nothing and never calls `process::exit`, on this branch and on main, so it exits 0
whatever its benchmarks score. That is the smoke-test limit stated above.

On a host with no adapter, the example runs the same 8 CPU benchmarks without GPU-001. They
are not gated on the GPU (`performance_parity.rs:66` onward).

## Measured since review round 3

- **The 10 CUDA rows with no libcuda.** All 10 are needs-hardware. aprender-gpu reports
  `CUDA driver not found (libcuda.so)` (`crates/aprender-gpu/src/driver/context.rs:48`), and
  the classifier's `libcuda\.so` pattern matches it. In run 37169058144 all ten were fail, so
  no pattern matched any of their output. test_gemv_correctness's row shows `CUDA driver not
  found` with no `(libcuda.so)`. The other nine rows show only their first output line, so
  that log does not show what they printed after it.
- **parity_035 with no Ollama server.** `parity_035-no-server.txt` is the same tree in a
  private network namespace with loopback up and nothing listening. The example printed
  `Ollama server not found at http://localhost:11434 ...` and exited rc 1 in 1 s. The
  classifier's `not found at ` pattern makes that needs-data. In the 36-row run a server was
  up without the model, and the row is needs-data on `Model not found`.
- **parity_035 with a server that answers 500.** `parity_035-500.txt` is the same tree in a
  private network namespace, with a listener on port 11434 that reads each request and
  answers `HTTP/1.1 500 Internal Server Error`. The example printed `Error: "Ollama /api/show
  for phi2:2.7b answered 500 Internal Server Error"` and exited rc 1. The unchanged script,
  run on that one row, classed it fail (`summary pass=0 fail=1`, exit 1). The listener's log
  shows, for each of the two runs, the TCP probe's connection with no request and then
  `POST /api/show`. A broken server is
  now a failure, not needs-data.

## Not measured here

The issue's done-when is the nightly itself: the same 36 rows with fail 0 and timeout 0.
examples-nightly stays disabled until this lands on main. Then it is re-enabled for one green
night, and that run is the receipt of record.
