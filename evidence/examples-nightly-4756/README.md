# examples-nightly reds — receipts (#4756)

Before: CI run 37169058144 (main @316dee2cd4). Of the 36 rows, 29 fail and 7 time out.

After: `examples-36.tsv`, one run of the unchanged `scripts/dogfood_examples.sh` on
22954be7ab, the last code commit on this branch, in a job on 2026-10-10 from 09:15:58Z to
09:41:01Z. `qa_verify-sections.txt` and `bug-hunter-scan.txt` were taken on 73f34268e6. The
three code commits after it, fa3364688c, 819e4132dc and 22954be7ab, change only
`parity_035_m4_verification.rs`.

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

Each row's `secs` is build plus run. agent_demo's row took 165 s, and its run stage timed
alone afterwards on the same tree took 1 s (`run-stage-secs.txt`), so nearly all of that row
was build. The script bounds only the run (`scripts/dogfood_examples.sh:226`, "Only the RUN
is bounded."), so a long build cannot time a row out. `examples-36-cold-4a79227e9f.tsv` is
an earlier cold run, on 4a79227e9f: the same 36 classes, exit codes and cites, with `secs`
up to 293 s (agent_demo).

`run-stage-secs.txt` times the run stage alone (`cargo run -q`, stdout to a file) after the
36-row run on the same tree. It covers the six rows with the highest `secs` in the
4a79227e9f cold run, plus bug_hunter_demo and design_by_contract. None ran longer than 9 s.

## What a debug build or a non-terminal run does differently

A release build run from a terminal keeps its workload: nothing it ran before is skipped or
cut down. Its result can still change, through the two bug fixes under "Bug fixes for every
build", which apply to every build. Only a debug build, or a run without a terminal as each
example's own gate tests it, takes a shorter path. Each item below names its gate:

- **pipeline_tui.** Its correctness check generated `GenerationConfig::default()`'s 100
  tokens (`crates/aprender-serve/src/generate/mod.rs:154`) through the unoptimized forward
  pass. On 4a79227e9f the row's `secs` in the cold run, build included, were 137, and its run
  stage timed alone afterwards on the same tree took 141 s (`run-stage-secs-4a79227e9f.txt`),
  against the 180 s bound. The aprender-serve library had been built for an earlier row, so
  the row's build was the example alone. The two were timed separately on a shared box, so
  the run stage alone can take longer than the whole row did, and here it did by 4 s. Both
  put the row within 43 s of the bound. A debug build now generates one token
  (`pipeline_tui.rs:19`, `:458`), and on 22954be7ab its run stage alone took 3 s
  (`run-stage-secs.txt`). A release build still generates 100. The same debug-only flag,
  `REDUCED` (`:19`), also cuts the other stages: 1 iteration each (`:23`), the Medium model
  skipped (`:132`), sequence lengths 1, 5 and 10 instead of up to 50 (`:164`). The run prints
  a line that says so (`:551`).
- **Bare runs with no arguments.** qa_verify, llama2-train and test_mac_worker keep their
  default only in a release build whose stdout is a terminal (`qa_verify.rs:394`,
  `llama2/train.rs:236`, `test_mac_worker.rs:16`). Otherwise they print their usage and
  the row is needs-args. Given its argument, each runs its full workload from any build. None
  of the three defaults can reach a pass in the nightly, which starts every example from the
  checkout root (`.github/workflows/examples-nightly.yml:51-55` sets no working directory):
  - *qa_verify.* Its two sections run cargo commands as child processes with no working
    directory set (`qa_verify.rs:121-123`), so from the checkout root they run against the root
    facade package. `qa_verify-sections.txt` runs each section in a debug build with no
    terminal, after building the example alone in an empty build directory. Section 2, run
    first, took 325 s, and 81 s run again with its nested builds present. Section 1 ran
    third, after section 2 had built its nested test target, and took 283 s. Every run
    exited rc 1 with gates reported FAIL. Section 1's test-count gate (`fn test_count`,
    `qa_verify.rs:161`, called at `:425`) printed `Only 0 tests` (the message at `:172`;
    `qa_verify-sections.txt:78`).
    The example does not print its nested cargo output, so the file does not show why those
    gates failed. A bare run that ran a section would be a fail even when it finished inside
    the bound. Those gate failures were measured in a debug build only, and are #5026, not this
    ticket. The file is the job's output with its colour escape codes removed afterwards, as
    its first line says. On main the row was a timeout.
  - *llama2-train.* Its default config, `examples/llama2/configs/124m.toml`, is relative to
    the crate. `git ls-files` finds it tracked at
    `crates/aprender-train/examples/llama2/configs/124m.toml` and nothing at the default path
    from the checkout root, so from there the default does not resolve. Where it does
    resolve, a bare run starts a full training run. On main the row was fail.
  - *test_mac_worker.* Its default is one fixed LAN address, `192.168.50.100:9000`. The nightly
    job checks out the tree, runs the script's self-test and the script, and uploads the
    ledger (`examples-nightly.yml:43-63`). Nothing in it starts a worker at that address.
    Whether the runner can reach that address was not measured. On main the row was fail;
    the log shows only its first line,
    `Connecting to Mac Pro worker at 192.168.50.100:9000...`.

  A quorum asked whether needs-args is the right class for these three bare runs answered
  yes, 3 of 3 counted lanes, on the condition that this README says why for each one.
- **Smaller debug workloads.** bug_hunter_demo (`:318`), design_by_contract (`:74`) and
  performance_parity (`:52`, `:373`) shrink only when `cfg!(debug_assertions)` is set. The
  two bug-hunter examples then scan `crates/aprender-orchestrate/src/bug_hunter`, 34 tracked
  files, instead of the whole tree. `bug-hunter-scan.txt` keeps their debug output: the scan
  path they printed and its result, 0 findings. Both scans in that capture were cache hits,
  so it shows the path and the result, not a fresh scan. The debug pass is a smoke test
  that the scan path runs, not a measure of contract coverage in the workspace; a release
  build scans the whole tree as before.
- **Non-terminal runs.** Four examples take a bounded or one-shot path when their gate finds
  no terminal, and the gates differ. api_server (`:103`) takes it only when neither stdin nor
  stdout is a terminal. buggy_server (`:21`) and brick_computer (`:698`) take it when stdout
  is not a terminal. calculator_tui (`:22`) takes it when either stdin or stdout is not one.
  Run with both at a terminal, all four keep their full path.
- **parity_035's Ollama probe.** It runs only in a debug build or when stdout is not a
  terminal (`parity_035_m4_verification.rs:265`), and can stop the run before the benchmark.
  A release build whose stdout is a terminal goes straight to the benchmark. See "parity_035's
  probe" below.

Of the 36 rows, only parity_035 was also run from a release build here ("Measured since
review round 3"). That the other workloads are unchanged in a release build rests on the gate
lines cited above.

## Bug fixes for every build

These change debug and release builds alike. Each one lets a run that stopped on an error
reach its end. Neither shortens a run.

- **performance_parity.** Its Q4_0 and Q8_0 benchmarks built 20×32 = 640 and 36×32 = 1152
  bytes. The dequantizers take 18-byte and 34-byte blocks and return an error for a length
  that is not a multiple (`crates/aprender-serve/src/quantize/dequant.rs:49`, `:111`). 640
  and 1152 are not multiples, so `.expect` panicked, and the failing nightly row was fail. The
  sizes are now 18×32 and 34×32 (`performance_parity.rs:514`, `:522`).
- **agent_demo.** Demo 10's primary driver, `FailingPrimary` (`agent_demo.rs:385`), reported
  a context window of 4096 tokens. The demo wraps it in a `RoutingDriver` with the default
  strategy, which reports the primary's window
  (`crates/aprender-orchestrate/src/agent/driver/router.rs:238`). The runtime builds the
  window with the manifest's `max_tokens` as the output reserve
  (`crates/aprender-orchestrate/src/agent/runtime.rs:340`), the default manifest's
  `max_tokens` is 4096 (`agent/manifest.rs:110`), and the input budget is the window minus
  the reserve (`serve/context.rs:32`). That left 0 input tokens, and a prompt that does not fit is
  refused as `AgentError::ContextOverflow`, which the runtime maps from the context
  manager's `ExceedsLimit` (`agent/runtime_helpers.rs:31-34`). The window is now 32 768
  (`agent_demo.rs:399`).

## parity_035's probe: debug builds and runs without a terminal

This is not a bug fix and does not let a run reach its end. It changes how a missing server
or a missing model is reported, and only in a debug build or a run whose stdout is not a
terminal (`parity_035_m4_verification.rs:265`); a release build at a terminal skips it and
runs as before. The failing nightly's log shows only the row's first line (the banner), so it
does not show why that run failed. Before the benchmark the probe makes two checks
(`fn probe_ollama`, `parity_035_m4_verification.rs:208`):

- It tries a TCP connection to port 11434 on every address `localhost` resolves to, the name
  the benchmark asks for. If none connects, it prints `Ollama server not found at
  http://localhost:11434 ...` and exits 1.
- It asks the server's `/api/show` for phi2:2.7b. Ollama answers that for a model it does
  not have with a 404 whose JSON body is `{"error":"model 'phi2:2.7b' not found"}`, and for
  a route it does not have with a 404 whose body is the plain text `404 page not found`
  (`ollama-404-bodies.txt`). Only the first is reported as `Model not found` (exit 1): a 404
  whose JSON `error` names phi2:2.7b and says `not found`.

Any other answer from `/api/show`, or a failed request, falls through to the benchmark,
which runs or fails as it did before the probe. So a server whose `/api/show` fails while its
`/api/generate` works still runs the benchmark, and a server that is broken or is not Ollama
fails inside the benchmark, not as a missing model. When the server has the model, the
benchmark runs as before. No run here had a server with the model, so that path rests on the
code.

## Every other change, by kind

- **A usage line and exit 2 for a missing required argument**, where the example panicked or
  returned an error: check_layer4, constrain_mask_overhead, qwen35_parity, ssc_eval,
  ssc_preflight. rex, qwen35_prefill_parity and think_ab already printed a usage line and now
  start it with `Usage`, the word the classifier's needs-args pattern matches
  (`^(Usage|...)`).
- **A message naming what is missing**: qa_chat and qa_serve (`Model not found: pass --model
  PATH ...`), publish_shell_safety (`[MISSING] <name> not found at <path>`), finetune_real
  (`No tokenizer found in HF cache`; the run still ends in the existing `.expect` on the
  tokenizer load, `finetune_real.rs:2016-2017`, rc 101, and the row is classed by that
  message), profile_cuda_trainer (`requires the 'cuda' feature`),
  aprender-gpu's driver loader (`CUDA driver not found (libcuda.so)`), and parity_035's
  `Ollama server not found at ...` when nothing listens on the port (from its probe, so only
  in a debug build or a run without a terminal).
- **Optional arguments.** `qa_verify -- --all` and `test_mac_worker <host:port>` run the full
  workload from any build, `api_server -- --serve` keeps serving with no terminal, and
  buggy_server takes an iteration count as its bound. The book pages for qa_verify now show
  `--all`.
- **The gated shorter paths** listed under "What a debug build or a non-terminal run does
  differently".

## performance_parity on a GPU

Review round 2 found that performance_parity's GPU path ran past 25 minutes in a debug build
(rc 124 at 1505 s) when wgpu found an adapter. A debug build now runs GPU-001 alone
(`performance_parity.rs:52`, `:101`, `:107`). A release build runs all 31 GPU benchmarks.
On main no build reached them: `bench_quantization_formats` runs before them, after the
five IMP benchmarks (`performance_parity.rs:67-83`, `:88`), and panicked on the buffer
sizes fixed above (`:514`, `:522`).

`performance_parity-gpu.txt` is the debug binary from the 36-row run, run once more with its
output kept, because the script keeps no output per example. The file records the commit and
the binary's sha256. Outside the namespace, it printed `GPU detected and available`, ran
GPU-001 on the GPU and printed the debug-build note. GPU-001's row reads 15.82 GFLOPS
(`performance_parity-gpu.txt:47`); the process ran pinned to 4 CPUs at nice 19. It exited
rc 0 in 10 s (`:66`). The script reported the same row as pass in 14 s, build included
(`examples-36.tsv`).

It also printed `Overall: Some benchmarks failed (5/9, 56%)` and exited 0. Its `fn main()`
returns nothing and never calls `process::exit`, on this branch and on main, so it exits 0
whatever its benchmarks score. That is the smoke-test limit stated above.

On a host with no adapter, the example runs the same 8 CPU benchmarks without GPU-001. They
are not gated on the GPU (`performance_parity.rs:66` onward). That path was not run here.
Only `performance_parity-gpu.txt` kept this example's output, and that run found an
adapter. The 36-row run and the cold run on 4a79227e9f kept no output per example, so
whether those runs found an adapter is not recorded.

## Measured since review round 3

- **The 10 CUDA rows with no libcuda.** All 10 are needs-hardware. aprender-gpu reports
  `CUDA driver not found (libcuda.so)` (`crates/aprender-gpu/src/driver/context.rs:48`), and
  the classifier's `libcuda\.so` pattern matches it. In run 37169058144 all ten were fail, so
  no pattern matched any of their output. test_gemv_correctness's row shows `CUDA driver not
  found` with no `(libcuda.so)`. The other nine rows show only their first output line, so
  that log does not show what they printed after it.
- **parity_035 with no Ollama server.** `parity_035-no-server.txt` is the same tree in a
  private network namespace with loopback up and nothing listening. The example printed
  `Ollama server not found at http://localhost:11434 ...` and exited rc 1 in under a second
  (`secs=0`). The classifier's `not found at ` pattern makes that needs-data. In the 36-row run a server was
  up without the model, and the row is needs-data on `Model not found`.
- **parity_035 with a server that answers 500.** `parity_035-500.txt` is the same tree in a
  private network namespace, with a listener on port 11434 that reads each request and
  answers `HTTP/1.1 500 Internal Server Error`. A 500 from `/api/show` is not the
  model-missing 404, so the example went on to the benchmark. That failed on the first
  measured response it could not decode (`parity_035_m4_verification.rs:82`, a line the
  branch does not change; the branch's `use` line moved it down one): it printed `Error: reqwest::Error { kind: Decode, source:
  Error("EOF while parsing a value", ...) }` and exited rc 1. The unchanged script, run on
  that one row, classed it fail (`summary pass=0 fail=1`, exit 1). The listener's log shows,
  for each of the two runs, the TCP probe's connection with no request, then `POST
  /api/show`, then three `POST /api/generate`: the two warmup requests and the first measured
  one. A broken server is a failure, not needs-data.
- **parity_035 with a server that answers a plain-text 404.** `parity_035-404.txt` is the same
  setup with a listener that answers `HTTP/1.1 404 Not Found`, Content-Type text/plain and the
  body `404 page not found`, which is what Ollama sends for a route it does not have
  (`ollama-404-bodies.txt`). That is not the JSON 404 naming the model, so the example went
  on to the benchmark, which failed the same way: `Error: reqwest::Error { kind: Decode, ...
  }`, whose source says the body parsed as the integer 404, not as an `OllamaResponse`, and
  rc 1. The unchanged script, run on that one row, classed it fail (`summary pass=0 fail=1`,
  exit 1). The listener's log shows the same requests as in the 500 case. A 404 that does not
  say the model is missing is a failure, not needs-data.
- **parity_035 from a release build, at a terminal and not.** `parity_035-release.txt` is three
  runs of a release build, each in a private network namespace with loopback up and nothing
  listening on port 11434. On 22954be7ab with stdout a terminal (a pty from `script -qec`), the
  example skipped the probe and went straight to the benchmark: it printed `[1/3] Benchmarking
  Ollama phi2:2.7b...`, then `Error: reqwest::Error { kind: Request, ... }` on a refused
  connection, and exited rc 1. On the same build with stdout to a file, it probed and printed
  `Ollama server not found at http://localhost:11434 ...`, rc 1. On the merge-base, e37e6dec56,
  built and run the same way at a terminal, the output was the same as the first run: `diff`
  of the two terminal outputs, carriage returns removed, exited 0 with no lines.

## Not measured here

The issue's done-when is the nightly itself: the same 36 rows with fail 0 and timeout 0.
examples-nightly is disabled in the repository's Actions settings, not in its file, which
keeps its schedule: the Actions API gave its state as `disabled_manually`, last changed
2026-10-07T09:35:50Z, when read at 2026-10-10T05:35Z. It stays disabled until this lands on
main. Then it is re-enabled for one green night, and that run is the receipt of record.

Clippy on these examples is not measured here: the job that made `examples-36.tsv` runs no
clippy step.
