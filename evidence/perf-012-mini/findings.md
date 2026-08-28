# PERF-012 — mini c=8 / c=16: measured, and the Metal arm re-decided

**Ticket:** PERF-012 (epic APR-PERF-GATE-001, paiml/aprender#2706)
**Spec:** `docs/specifications/APR-PERF-GATE-001-v2.2.md` (committed `1ad451aa7`)
**Host:** `mini` — Apple M4 (`Mac16,10`), 10 cores, **16 GB unified**, macOS 26.5.2 (build 25F84), `arm64`
**Commit under test:** `de8fbc407` (`origin/main`, "feat(perf): PERF-008/010/016 …" #2710)
**Date:** 2026-08-28

**Outcome: BOTH.** Bands c=8 and c=16 are **MEASURED** on mini for the
comparator-free arms (A and C). The **Metal** attribute of those cells is
re-decided as **NOT_APPLICABLE with a concrete, mechanical reason**: the Metal
serving path is unreachable from the `apr` CLI in *every* buildable
configuration of `de8fbc407`. Details in §4. §9 goes further and *builds* the
candidate fix on the host, which turns "what would unblock it" from a
suggestion into a measured result — including the finding that the Metal lane,
once unblocked, is **0.36× the CPU path** and still cannot emit a conformant
receipt.

---

## 1. Model situation on mini

**The recorded state was correct: mini had zero usable models.**

Everything matching `*.gguf` / `*.safetensors` / `*.apr` under `$HOME` was a
dogfood *fuzz fixture*, not a model — largest 6,528 bytes, smallest 1 byte:

| Path | Size |
|---|---|
| `~/dogfood-mac/tiny-llama.gguf` | 6,528 B |
| `~/dogfood-mac/junk.gguf` | 4,096 B |
| `~/dogfood-mac/truncated.gguf` | 3,000 B |
| `~/dogfood-mac/tiny.gguf` | 1,028 B |
| `~/aprdog/casetest/model.gguf` | 1 B |

`~/.cache/apr/` was an empty directory. `~/.cache/huggingface` did not exist.

**What I added, and how.** `apr pull` was not needed — mini reaches
`huggingface.co` (HTTP 200 in 0.16 s), but the canonical model already existed
on the lambda box, and copying it makes the two hosts provably use the *same
bytes*, which a fresh download does not. Both files were transferred over
`scp` and verified by digest **on both ends**:

| Model | Size | sha256 (lambda == mini) |
|---|---|---|
| `qwen2.5-coder-7b-instruct-q4_k_m.gguf` → `~/models_perf012_7b_q4km.gguf` | 4,683,073,536 B | `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c` |
| `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` → `~/models_perf012_0.5b.gguf` | 491,400,064 B | `1d9614638d18024d0fbb36575a15f1302a3adf044df10345688ec4f6e1c4ff32` |

The 7B is the canonical `APR-BENCH-RFC-001` §3.1 W1 model named by the spec.
The 0.5B is a **second subject**, not a substitute — see §3.3.

> **Unproven / stated plainly:** the 0.5B file is named `q4_k_m` but
> `apr inspect` reports its quantization as **Q5_0** (`general.file_type = 15`),
> and its `general.finetune` is `Instruct-AWQ`. The filename is wrong about its
> own contents. It is used here only as a *second subject for the concurrency
> shape*, never as a Q4_K_M datapoint.

---

## 2. Which binary I ran, and the proof

`~/.cargo/bin/apr` on mini was a stale released `0.64.0 (v0.64.0+no-git)` from
Aug 24 with **no embedded git SHA** — unattributable to any commit, so unusable
as a measurement subject. mini's checkout was frozen at `45a2ce230` (PMAT-920
era, months old).

I synced the checkout to `origin/main` and built there:

```
$ cd ~/src/aprender && git checkout -f de8fbc407 && git rev-parse --short HEAD
de8fbc407
$ cargo build --release --bin apr          # default features
    Finished `release` profile [optimized] target(s) in 1m 24s
```

Provenance proven by the repo's own fail-closed resolver, not by my assertion:

```
$ bash scripts/apr_bin.sh
/Users/noahgift/src/aprender/target/release/apr
$ echo $?            # captured directly, NOT through a pipe
0
$ ./target/release/apr --version
apr 0.64.0 (de8fbc407)
$ git rev-parse --short HEAD
de8fbc407
```

`scripts/apr_bin.sh` returns non-zero unless the binary's build-time-embedded
SHA equals `HEAD`; it returned 0. The measurement driver
(`perf012_bands.sh`) sources it (`. scripts/apr_bin.sh || exit 1`) and uses
`$APR`, so no bare `apr` and no hardcoded path was ever invoked.

---

## 3. The measurement

### 3.1 Protocol — declared, not implicit

Taken verbatim from `scripts/llama_pin.toml` `[protocol.http]` on the commit
under test, so these numbers sit in the same series as the lambda numbers that
file records:

| Knob | Value | Source |
|---|---|---|
| client | `apr test llm bench`, external HTTP, closed-loop | §4.4.1 |
| bands | 1, 4, 8, 16 | `http_concurrency_bands` |
| profile | `medium` (128 in / 128 out) | `http_profile` |
| warmup | 15 s, discarded | `http_warmup_secs` |
| duration | 30 s per run | `http_duration_secs` |
| runs (replicates) | 3 | `http_runs` |
| cooldown | 10 s | `http_cooldown_secs` |
| streaming | on (so TTFT and ITL are real) | `http_stream` |
| context | 4096 | `apr_serve_command` |

Server: `apr serve run <model> --port <p> --context-length 4096` — **no
accelerator flag at all**, because on this host every accelerator flag is
either refused or unbuildable (§4).

---

## 4. Was Metal engaged? **No — and it cannot be, on this commit**

This is the part of the ticket with a hard answer. Four independent lines of
evidence, none of which is "a flag I passed".

### 4.1 The server says `cpu` about itself

The driver reads the compute class from the **server's own log** using the same
predicate as `scripts/parity_host_receipt.sh:apr_class_from_log` — never from
the flag handed to it, because "`apr serve run --gpu` on the published binary
prints no CUDA banner and holds zero VRAM" is the whole of #2696.

```
compute_class_from_server_log=cpu
```

`server-7b.log` contains no `Metal`, no `wgpu`, no `CUDA` banner. It prints
`Model ready: 28 layers, vocab_size=152064, hidden_dim=3584` and nothing about
an accelerator.

### 4.2 The product itself refuses the Metal path (exit 9)

```
$ ./target/release/apr serve run ~/models_perf012_0.5b.gguf --backend wgpu --port 8433
$ echo $?     # captured directly, not through a pipe
9
error: Feature not enabled: --backend wgpu was requested, but this build has no
GPU backend compiled in, so the server would have run on CPU without telling you.
```

This is the landed jidoka behaviour working correctly. It is also the proof: on
the binary that produced every number below, there is no GPU backend.

### 4.3 Behavioural delta: the GPU is idle while the CPU is saturated

Sampled with `ioreg -r -d 1 -w 0 -c AGXAccelerator` (no sudo needed) **during**
a 7B inference run, alongside `apr`'s CPU share, 10 samples 3 s apart:

| | median | min | max |
|---|---|---|---|
| GPU `Device Utilization %` | **7** | 6 | 9 |
| `apr serve` CPU % (of 1000% = 10 cores) | **983** | 949 | 988 |

apr is pinning ~9.8 of mini's 10 cores while the M4 GPU sits at desktop-idle
level. This is a CPU/NEON measurement.

### 4.4 The mechanical reason — and exactly what would unblock it

**`cargo install aprender --features wgpu`, the remedy the error message in
§4.2 tells the user to run, does not compile.**

On `de8fbc407`, `crates/apr-cli/Cargo.toml:92` reads:

```toml
wgpu = ["inference"]
```

The feature enables **nothing GPU-related**. It does not enable `trueno/gpu`,
`entrenar/gpu`, or `realizar/gpu`. But turning it on *does* switch on apr-cli's
own `#[cfg(feature = "wgpu")]` blocks — in `commands/serve/handlers.rs`
(lines 11, 26, 43, 89, 296) and `commands/finetune.rs` — and those blocks
reference items gated behind *other crates'* `gpu` features that nothing turned
on. The build fails:

```
error[E0432]: unresolved import `entrenar::finetune::wgpu_pipeline::WgpuInstructPipeline`
   --> crates/apr-cli/src/commands/finetune.rs:585:9
   note: found an item that was configured out
         crates/aprender-train/src/finetune/wgpu_pipeline.rs:51:12
         the item is gated behind the `gpu` feature
error[E0433]: failed to resolve: could not find `wgpu_training` in `autograd`
   --> crates/apr-cli/src/commands/finetune.rs:658:39
   note: the item is gated behind the `gpu` feature
error: could not compile `apr-cli` (lib) due to 2 previous errors
```

Reproduced **twice on mini at `de8fbc407`**, by two different entry points:

1. `cargo build --release --bin apr --features wgpu` (root facade → `apr-cli/wgpu`)
2. `cargo check -p apr-cli --features wgpu --lib` (the crate directly)

So on this commit the CLI's Metal entry point has exactly two states and both
are dead ends:

| Build | `--backend wgpu` |
|---|---|
| default features | refused, **exit 9** — "no GPU backend compiled in" |
| `--features wgpu` | **does not compile** |

**What would unblock it.** The wgpu/Metal machinery is *already in the default
binary* — `crates/aprender-serve/Cargo.toml` has `default = ["server", "cli",
"gpu"]` with `gpu = ["trueno/gpu"]`, which is why
`realizar::gpu::adapters::wgpu_adapter::dequant_model_weights` (handlers.rs:367)
compiles in a default build at all. The missing piece is one line. Restoring
apr-cli's `wgpu` feature to actually enable its dependencies —

```toml
wgpu = ["inference", "trueno/gpu", "entrenar?/gpu", "realizar/gpu"]
```

— is the minimal candidate fix. **I subsequently built and tested it on mini;
see §9**, where it compiles (rc=0) and, together with #2638's second hunk,
brings up a working Metal server. Note `realizar/gpu` turns out to be
unnecessary: `crates/aprender-serve/Cargo.toml` already has
`default = ["server", "cli", "gpu"]`, which is why
`realizar::gpu::adapters::wgpu_adapter` resolves even in a default build.

What has no coverage is the configuration itself: **no CI workflow builds `apr`
with `--features wgpu`** (`.github/workflows/gpu-vulkan.yml` exercises the
compute crate, not the CLI), which is why a manifest line that makes the
documented install command fail could sit on `main` unnoticed.

### 4.5 Two spec/matrix corrections this forces

1. **`scripts/perf-matrix.yaml:29-32` declares `mini.compute_class: metal`.**
   On `de8fbc407` mini cannot reach a metal compute class from the CLI at all.
   The declaration is aspirational. Any mini receipt produced today is `cpu`
   class, and under §4.9/PARITY the comparator for a cpu-class apr must be
   llama.cpp `-ngl 0`, **not** Metal llama.cpp — otherwise it is the
   cross-class comparison the harness explicitly refuses to make.

2. **The PERF-012 cell does not exist in the machine-readable matrix.** The
   spec's §4.7.1 shows a `{host: mini, band: 16, status: UNMEASURED, ticket:
   PERF-012}` cell and §4.9's host table says `c=8,16 UNMEASURED`, but
   `scripts/perf-matrix.yaml`'s `cell_exceptions:` contains **one** entry, for
   `gx10`/`vllm`. mini appears only as workload-level
   `{W1: UNMEASURED, W2: UNMEASURED, expires: 2026-09-25}`. The band-level
   status this ticket was opened to resolve was never written down where the
   verdict job reads it.

> **Also unproven, stated plainly:** the spec's §4.7.1 recorded reason for the
> mini cells — *"16 GB unified memory — OOM at c=16 with Q4_K_M"* — is **not
> what happened**. See §5.3. No OOM occurred.

### 4.6 The fix already exists, and has been stalled for five days

Searching for when `wgpu` lost its dependencies turned up a branch, not a
regression:

```
$ git log --oneline -S'wgpu = ["inference", "trueno/gpu"' --all -- crates/apr-cli/Cargo.toml
1073105ca fix(compute,cli): a self-deadlock on main, a feature that could not be
           installed, and a GPU lane that could not go green
$ git merge-base --is-ancestor 1073105ca origin/main && echo YES || echo "NO - not on main"
NO - not on main
```

`1073105ca` is the head of **PR #2638**, branch `feat/wgpu-feature-surface`,
authored **2026-08-23**. Its diff against `origin/main` is exactly the candidate
fix in §4.4:

```diff
-wgpu = ["inference"]
+wgpu = ["inference", "trueno/gpu", "entrenar?/gpu"]
-full = ["inference", "cuda", "cuda-batch", "visualization", ...]
+full = ["inference", "cuda", "cuda-batch", "wgpu", "visualization", ...]
```

Its own commit subject names this defect — *"a feature that could not be
installed"*. **State as of 2026-08-28: `OPEN`, `mergeable: CONFLICTING`,
`mergeStateStatus: DIRTY`, last updated 2026-08-23, 2 of 12 checks failing
(`gate`, `workspace-test`).**

So PERF-012's Metal blocker is not undiagnosed — it is diagnosed, fixed on a
branch, and stuck. **Landing #2638 is the single action that would let mini
produce a Metal receipt.**

---

## 5. Results — subject 1: the canonical 7B Q4_K_M (W1 model)

`qwen2.5-coder-7b-instruct-q4_k_m.gguf`, sha `509287f7…`, compute class `cpu`.

| c | runs | requests | ok | failed | agg tok/s (med) | agg min–max | decode tok/s (med) | decode min–max | TTFT med | n (requests) |
|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| 1 | 3 | 3 | 3 | 0 | **1.21** | 1.21–1.21 | **2.38** | 2.37–2.38 | 52.5 s | 3 |
| 4 | 3 | 12 | 0 | **12** | — | — | — | — | — | 0 |
| 8 | 3 | 24 | 0 | **24** | — | — | — | — | — | 0 |
| 16 | 3 | 48 | 0 | **48** | — | — | — | — | — | 0 |

### 5.1 c=4, c=8 and c=16 are UNMEASURABLE on the canonical model — and why

Not "slow". **Zero completed requests.** Every run at every band c≥4 ended at
*exactly* the protocol's hard per-request timeout:

```
  c=1   run1 total=1   ok=1   failed=0   elapsed=105.93s
  c=1   run2 total=1   ok=1   failed=0   elapsed=105.94s
  c=1   run3 total=1   ok=1   failed=0   elapsed=105.93s
  c=4   run1 total=4   ok=0   failed=4   elapsed=120.00s
  c=4   run2 total=4   ok=0   failed=4   elapsed=120.00s
  c=4   run3 total=4   ok=0   failed=4   elapsed=120.00s
  c=8   run1 total=8   ok=0   failed=8   elapsed=120.00s
  …
  c=16  run3 total=16  ok=0   failed=16  elapsed=120.01s
```

`120.00 s` is §4.4.3's *"Timeout = 120 s hard per request."* A **single**
request on this host already takes **105.93 s** — 88 % of the entire timeout
budget — so there is no headroom for a second concurrent request, let alone
fifteen.

The harness refused to launder this into a number, which is the gate working
exactly as designed:

```
error: Validation failed: run 1 had 4 failed request(s); run 2 had 4 failed
request(s); run 3 had 4 failed request(s) — a throughput averaged over the
requests that survived is not a measurement of this runtime, it is a
measurement of its survivors
```

Under §4.4.3 / §4.7.3 `timeouts != 0` is **fatal to this host's ratio**, so the
correct receipt value for these three cells on W1 is a **FAIL of Arm C**, not a
throughput.

### 5.2 The magnitude, stated plainly

1.21 tok/s aggregate and 2.38 tok/s decode on a 7B Q4_K_M, with a **52.5 s**
time to first token for a 128-token prompt (a prefill rate of 1.95 tok/s — apr
is prefilling at essentially its decode rate, i.e. not batching the prompt).

This is a **CPU/NEON** number on an M4 with no accelerator, taken with the
pinned `medium` profile. It is reported because it is what the host produces
today, not as a comparator ratio — no llama.cpp is installed on mini, so
**Arm B was not attempted** (§7).

### 5.3 The recorded reason for these cells is FALSIFIED

Spec §4.7.1 records the mini cells as:

```json
{ "host":"mini", "band":16, "comparator":"llamacpp", "status":"UNMEASURED",
  "reason":"16 GB unified memory — OOM at c=16 with Q4_K_M", … "ticket":"PERF-012" }
```

**No OOM occurred, at any band, at any point.** Measured across the entire
sweep including all three c=16 runs:

| Quantity | Value |
|---|---|
| Peak RSS of `apr serve` | **10,972,016 KB = 10.47 GB** of 16 GB (65 %) |
| `vm.swapusage` | **total = 0.00M, used = 0.00M** — macOS never even allocated a swap file |
| jetsam / `memorystatus` events naming `apr` (60 min window) | **0** |
| Server liveness after c=16 | alive, `/health` → `200` |

The server was never under memory pressure sufficient to page, let alone be
killed. **The blocker is latency against a fixed per-request timeout, not
memory.** The reason string should be replaced.

---

## 6. Results — subject 2: a smaller model, so the bands can actually complete

`qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` (**actually Q5_0**, see §1), compute
class `cpu`. Run twice, on two different workload profiles, to separate "apr
serialises" from "that one model/workload is slow".

### 6.1 `medium` profile (128 in / 128 out) — the pinned protocol

| c | runs | requests | ok | failed | agg tok/s med | agg min–max | decode tok/s med | decode min–max | TTFT p50 | latency p50 | ITL p50 |
|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 3 | 3 | 3 | 0 | **3.70** | 3.69–3.72 | **7.29** | 7.27–7.31 | 17.1 s | 34.4 s | 136.7 ms |
| 4 | 3 | 12 | 12 | 0 | **8.85** | 8.84–8.86 | **4.36** | 4.36–4.38 | 28.7 s | 57.8 s | 229.4 ms |
| 8 | 3 | 24 | 24 | 0 | **10.03** | 10.02–10.03 | **2.48** | 2.48–2.48 | 50.7 s | 101.9 s | 403.1 ms |
| 16 | 3 | 48 | 0 | **48** | — | — | — | — | — | *120.00 s timeout* | — |

**c=16 timed out here too** — and it was *predicted before it ran*, by the
latency column: 34.4 → 57.8 → 101.9 s, i.e. p50 latency **doubles** from c=4 to
c=8. One more doubling puts c=16 at ~190 s, comfortably past the 120 s hard
timeout. It duly failed 48/48 at exactly `120.00 s`. Peak RSS **2.09 GB**,
swap **0.00M** — again, not memory.

### 6.2 `short` profile (32 in / 32 out) — the band set that completes

Same model, same server, same binary; only the declared `--profile` knob
changes, which shortens each request enough to fit c=16 inside the timeout.

| c | runs | requests | ok | failed | agg tok/s med | agg min–max | decode tok/s med | decode min–max | TTFT p50 | n (requests) |
|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
| 1 | 3 | 12 | 12 | 0 | **3.84** | 3.84–3.85 | **7.32** | 7.31–7.33 | 4.1 s | 12 |
| 4 | 3 | 36 | 36 | 0 | **9.18** | 9.17–9.20 | **4.38** | 4.38–4.39 | 6.8 s | 36 |
| **8** | 3 | **48** | 48 | **0** | **10.41** | 10.41–10.42 | **2.49** | 2.48–2.50 | 12.1 s | **48** |
| **16** | 3 | **48** | 48 | **0** | **10.49** | 10.49–10.50 | **1.26** | 1.26–1.27 | 23.9 s | **48** |

**This is the PERF-012 deliverable: c=8 and c=16, measured, n=48 each, zero
failures, dispersion reported.** Dispersion is not a formality here — it is
astonishingly tight, ±0.01 tok/s across three independent replicates at every
band, which is itself worth recording: whatever limits this host, it is not
noisy.

### 6.3 Arm A — scaling efficiency (comparator-free, the arm that gates today)

`scaling_efficiency(c) = (agg(c)/agg(1))/c`, and its reciprocal-flavoured twin
`serialization_index(c) = c·agg(1)/agg(c)` (1.0 = perfect batching, c = total
serialization):

| c | agg tok/s | speedup vs c=1 | **scaling_efficiency** | **serialization_index** |
|---:|---:|---:|---:|---:|
| 1 | 3.84 | 1.000× | 1.000 | 1.00 |
| 4 | 9.18 | 2.391× | 0.598 | 1.67 |
| 8 | 10.41 | 2.712× | **0.339** | **2.95** |
| 16 | 10.49 | 2.734× | **0.171** | **5.85** |

**The headline: aggregate throughput moves from 10.41 to 10.49 tok/s — +0.8 % —
while the offered load doubles from 8 to 16 concurrent streams.** Per-request
decode halves on each doubling (4.38 → 2.49 → 1.26) and TTFT doubles
(6.8 → 12.1 → 23.9 s). The machine is saturated at c≈8 and every request added
past that point is pure latency with no throughput in return.

### 6.4 The two profiles agree to three decimal places

This is the "one input is an anecdote" check, and it came back unusually clean.
Two different workload shapes, run on different servers on different ports:

| c | scaling_efficiency (medium) | scaling_efficiency (short) | serialization_index (medium) | serialization_index (short) |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.00 | 1.00 |
| 4 | 0.598 | 0.598 | 1.67 | 1.67 |
| 8 | 0.339 | 0.339 | 2.95 | 2.95 |

Identical. The scaling curve is a property of the runtime on this host, not of
the prompt shape — so the c=16 short-profile figures are a sound reading of the
c=16 *shape*, even though the c=16 *medium* cell itself timed out.

### 6.5 What this does NOT say

apr on mini's CPU path is **not** the flat-aggregate exclusive-writer-lock
signature recorded on lambda (agg pinned at ~110 tok/s across all bands while
per-user decode *rose* to 1.554×). Here aggregate **rises 2.7×** from c=1 to
c=8 and per-user decode **falls**. Those are different shapes and should not be
conflated:

- lambda's PERF-000 probe (`scripts/perf000_serialization_probe.sh`) drives the
  server with **`--gpu`**. mini has no GPU path at all (§4).
- So this measurement says nothing about whether the PERF-000 writer lock
  exists. It measures a **different code path** — GGUF CPU/NEON — which
  saturates rather than serialises outright.

**Unproven, stated plainly:** I did not isolate *why* aggregate plateaus at
~10.5 tok/s. Rayon thread contention, memory bandwidth, and a partial lock are
all consistent with this data and I did not falsify between them.

---

## 7. Arm B (comparator) was NOT attempted, deliberately

No llama.cpp exists on mini, and `scripts/llama_pin.toml` still declares
`build_commit = "39173bcac"` pinned *"from the binary actually installed on
lambda … CUDA-enabled"*. Building a Metal llama.cpp on mini and comparing it to
a **cpu-class** apr would be exactly the cross-class comparison
`scripts/parity_host_receipt.sh` refuses to make in its own header:

> *"It will not compare across classes. A cpu-class apr is measured against
> llama.cpp `-ngl 0`, never against a CUDA comparator. Cross-class is how a
> 0.099x artifact defect reads as a kernel defect."*

Per §4.7.2, Arms A, C, D and E need no comparator, so the mini cells are
gated by Arm A today regardless. **Arm B on mini should stay `UNMEASURED` with
this ticket as owner** until either (a) #2638 lands and apr gets a real metal
class, at which point the comparator is Metal llama.cpp; or (b) someone
deliberately decides to run the cpu-vs-`-ngl 0` lane, which needs a
`build_flags_mini` CPU variant that `llama_pin.toml` does not currently declare
(it declares only `cmake -B build -DGGML_METAL=ON`).

Also blocking a Metal comparator today: **mini has no `cmake`**
(`which cmake` → not found), so the pinned `build_flags_mini` cannot be executed
on the host as written.

---

## 8. What I recommend the matrix records

### 8.1 The cells

| Host | Workload | Band | Arm | Status | Value |
|---|---|---:|---|---|---|
| mini | W1 (`medium` 128/128) | 1 | A, C | **MEASURED** | agg 1.21 tok/s, decode 2.38 tok/s, n=3 (7B Q4_K_M) |
| mini | W1 | 4, 8, 16 | C | **FAIL** | 100 % request timeout at 120.00 s; 0/12, 0/24, 0/48 completed |
| mini | `short` 32/32 | 1, 4, 8, 16 | A, C | **MEASURED** | see §6.2/6.3; c=8 agg 10.41, c=16 agg 10.49, n=48 each |
| mini | any | any | B | **UNMEASURED** | no comparator on host; owner PERF-012; see §7 |
| mini | — | — | Metal | **NOT_APPLICABLE** | see §8.2 |

### 8.2 The Metal re-decision, in the schema's own vocabulary

```json
{ "host":"mini", "band":16, "arm":"metal", "status":"NOT_APPLICABLE",
  "permanent": false,
  "decided_by":"PERF-012",
  "decided_on":"2026-08-28",
  "reason":"apr has no reachable Metal path on de8fbc407. A default build refuses
            --backend wgpu with exit 9 ('no GPU backend compiled in'); a
            --features wgpu build does not compile, because
            crates/apr-cli/Cargo.toml:92 declares wgpu = [\"inference\"], which
            enables apr-cli's own #[cfg(feature=\"wgpu\")] blocks without enabling
            trueno/gpu or entrenar/gpu that those blocks reference (E0432/E0433 in
            commands/finetune.rs:585,658). Verified twice on mini at de8fbc407.",
  "unblocked_by":"paiml/aprender#2638 (branch feat/wgpu-feature-surface, OPEN,
                  CONFLICTING since 2026-08-23). VERIFIED on the host, §9: its
                  Cargo.toml hunk makes --features wgpu build (rc=0) and its
                  aprender-compute hunk clears a main-thread self-deadlock that
                  otherwise hangs WGPU device init on Metal. With both applied a
                  Metal server comes up and generates.",
  "still_blocked_after_2638":[
    "The WGPU path dequantizes the whole model to F32 before upload. The
     canonical W1 7B Q4_K_M produces 28,282.5 MB of F32 on a 16 GB host and
     drives it into swap at c=1. Needs native Q4_K GPU kernels, not a fix here.",
    "The WGPU server registers only POST /v1/chat/completions and GET /health
     with no SSE path, so ttft_ms and itl_ms -- required by 4.4.3, and
     http_stream = true in llama_pin.toml -- cannot be measured on this lane.",
    "Measured 0.36x the CPU path on the same model (4.94 -> 1.77 tok/s, n=6
     each), so even a working Metal lane is not currently the fast lane here."
  ] }
```

Note `permanent: false` — this is **not** a permanent property of macOS or of
the M4. The *build* half is a one-line manifest defect with an open PR. The
three items under `still_blocked_after_2638` are real engineering (GPU-side
Q4_K kernels; an SSE route on the WGPU server), and each is separately
falsifiable. Marking the cell permanently `NOT_APPLICABLE` would be wrong;
marking it unblocked by #2638 alone would also be wrong.

### 8.3 Corrections owed to the spec / matrix

1. **§4.7.1's reason string is false.** *"16 GB unified memory — OOM at c=16
   with Q4_K_M"* — no OOM, no swap, no jetsam; peak 10.47 GB of 16 GB (§5.3).
   Replace with the 120 s timeout reason.
2. **`scripts/perf-matrix.yaml:31` `compute_class: metal` is aspirational.** It
   should read `cpu` until #2638 lands, or the host will be compared against a
   Metal comparator and read as a kernel defect.
3. **The band-level mini cells this ticket exists for are not in
   `cell_exceptions:` at all** — only a workload-level `UNMEASURED` at
   `perf-matrix.yaml:105-107`. The verdict job cannot see what it is not told.
4. **`llama_pin.toml` has no `build_flags_mini` CPU variant**, and mini has no
   `cmake`, so Arm B on mini is un-runnable as declared today (§7).

---

## 9. Addendum — I tested whether #2638 actually unblocks Metal. It does, twice, and then it loses to the CPU

§8.2 claims #2638 would unblock Metal. Claiming it is not the same as showing
it, so I built it. **This section is about a patched tree, not about
`origin/main`, and nothing in it is a receipt for main.**

**Provenance of the probe.** Applied as *commits*, never a dirty tree, so
`apr --version` still names what it actually is and `scripts/apr_bin.sh` still
validates:

| Branch `perf012-wgpu-probe` (on mini) | |
|---|---|
| `94a13ab8a` | de8fbc407 + #2638's `crates/apr-cli/Cargo.toml` hunk |
| `71f6bad85` | + #2638's `aprender-compute/…/gpu/device/mod.rs`, `serve/handlers.rs`, `finetune.rs` hunks |

`./target/release/apr --version` → `apr 0.64.0 (71f6bad85)`; `git rev-parse
--short HEAD` → `71f6bad85`; `bash scripts/apr_bin.sh` → rc **0**.

### 9.1 Result 1 — the Cargo.toml hunk alone fixes the build

`cargo build --release --bin apr --features wgpu` at `94a13ab8a`: **rc=0**. So
§4.4's one-line diagnosis is correct and #2638's manifest change is sufficient
to make `cargo install aprender --features wgpu` compile on aarch64 macOS.

### 9.2 Result 2 — I reproduced #2638's "self-deadlock on main", live, on Metal

At `94a13ab8a` (Cargo.toml hunk only) the server got as far as
`Initializing WGPU device…` and then **hung at 0.0 % CPU indefinitely**
(>6 min). `sample(1)` on the live process:

```
2274 Thread_414175   DispatchQueue_1: com.apple.main-thread  (serial)
  + 2274 _pthread_mutex_firstfit_lock_slow  (in libsystem_pthread.dylib)
  +   2274 _pthread_mutex_firstfit_lock_wait (in libsystem_pthread.dylib)
```

Main thread blocked on a non-reentrant mutex. This is exactly the bug #2638's
other hunk describes — `shared_instance()`'s `OnceLock` initializer took
`DEVICE_INIT_LOCK` while its callers already held it. The PR found it on
**Linux/AMD-RADV**; this is independent confirmation on **Apple/Metal**, and it
means the deadlock is platform-independent, not a RADV quirk.

Applying the compute hunk (`71f6bad85`) cleared it immediately.

### 9.3 Result 3 — Metal then works, and is provably on the GPU

```
Backend: WGPU (Vulkan/Metal/WebGPU)
GPU weights ready: 337 tensors, 6174.9 MB F32
Initializing WGPU device...
WGPU device ready (Vulkan/Metal)
Uploaded 337 weights to GPU (6174.7 MB VRAM) in 2054.3ms
WGPU inference ready: 28 layers, vocab=151936, hidden=1536
WGPU inference server listening on http://127.0.0.1:8604
```

Mechanism proof, not a flag: `ioreg -c AGXAccelerator` `Alloc system memory`
went from **194,297,856 B (194 MB)** before load to **13,151,911,936 B
(13.15 GB)** after, and `Device Utilization %` read **38–49 %** during
generation against a 7–17 % idle band. Generation was coherent.

### 9.4 Result 4 — and Metal is 2.8× SLOWER than the CPU path

Controlled A/B: same binary (`71f6bad85`), same model
(`qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, sha `cc324af0…`), same prompt, same
`max_tokens=32`, `temperature=0`, one server at a time, n=6 each.

| Lane | median tok/s | min–max | median wall | n |
|---|---:|---|---:|---:|
| CPU (default path) | **4.94** | 4.908–4.958 | 4.25 s | 6 |
| Metal (`--backend wgpu`) | **1.77** | 1.741–1.778 | 11.85 s | 6 |

**Metal ratio vs CPU: 0.36×.** Dispersion on both sides is <1 %, so this is not
a sampling artifact. Landing #2638 makes the Metal lane *exist*; it does not
make it *worth using* on this host as implemented.

### 9.5 Result 5 — the Metal lane cannot produce a conformant receipt anyway

Two independent blockers, both structural:

1. **F32 dequantization.** The WGPU path expands the whole quantized model to
   F32 before upload. The **canonical W1 7B Q4_K_M produced 28,282.5 MB
   (28.3 GB) of F32 on a 16 GB host** — I watched macOS open a 16 GB swap file
   and reach 15.3 GB used before I killed it. So the canonical model can never
   run on mini's Metal lane at *any* concurrency, including c=1. **This, not KV
   cache at c=16, is the real memory story** — and it is a much better reason
   than the one §4.7.1 records.
2. **No streaming.** The WGPU server registers exactly two routes —
   `POST /v1/chat/completions` and `GET /health` (`handlers.rs:711-712`) — with
   no SSE path, against ~35 routes on the CPU server. The pinned protocol sets
   `http_stream = true` and the gate requires `ttft_ms` and `itl_ms` (§4.4.3).
   **Those metrics are unobtainable on this lane as built.**

Also: `dequant_tensor_public` supports only Q4_K, Q6_K, Q5_K, F32, F16 and APR
q4/q8. The 0.5B subject is **Q5_0** and was rejected outright with
`Unsupported quantization type 6 for WGPU dequant`.

### 9.6 What I did NOT do

- **No Metal band sweep.** At 1.77 tok/s a 32-token request takes ~11.9 s, so
  c=8 would need ~145 s against a 120 s hard timeout. c=1 and possibly c=4
  would have completed; c=8/c=16 could not. Since the lane also cannot emit
  TTFT/ITL (§9.5) it cannot produce a conformant receipt, so a partial sweep
  would have been numbers without a receipt — the thing this epic exists to
  stop.
- **I did not verify #2638 passes CI.** It is `CONFLICTING` with 2 failing
  checks; I tested its *content* on one host, not its mergeability.
- **One GPU-utilization sample is unexplained.** A single `ioreg` read taken
  just after the CPU A/B loop returned 49 %, where the 10-sample series during
  7B CPU inference read 6–9 %. I did not chase it, and I am **not** relying on
  it. The load-bearing GPU evidence is the 194 MB → 13.15 GB allocation delta
  and the server's own banner, both of which are unambiguous.

---

## 10. Reproduction

```bash
# on mini, from ~/src/aprender at de8fbc407
. scripts/apr_bin.sh || exit 1          # refuses a binary that is not HEAD
~/perf012_bands.sh <model.gguf> <tag> <port> ~/perf012_out 1 4 8 16
```

Artifacts on mini under `~/perf012_out/`: `<tag>-c<N>.json` (full harness
report incl. `request_details`), `<tag>-c<N>.rc` (exit status captured
directly, never through a pipe), `server-<tag>.log`, `class-<tag>.txt`,
`rss-<tag>.txt` (2 s RSS series with `BAND_START`/`BAND_END` markers).
Probe branch `perf012-wgpu-probe` (`94a13ab8a`, `71f6bad85`) is preserved;
the working checkout was restored to `de8fbc407` and all servers stopped.

## 11. Summary of every claim's status

| Claim | Evidence | Status |
|---|---|---|
| mini had zero usable models | file listing, max 6,528 B | **verified** |
| Models added by digest-verified `scp` | sha256 equal on both hosts | **verified** |
| Binary built from `de8fbc407` | `apr_bin.sh` rc=0, version string == HEAD | **verified** |
| c=8 / c=16 measured (0.5B, short) | agg 10.41 / 10.49 tok/s, n=48 each, 0 failures | **verified** |
| Aggregate flat c=8→c=16 (+0.8 %) | 10.41 → 10.49 | **verified** |
| Canonical 7B unmeasurable at c≥4 | 0/84 completed, all at exactly 120.00 s | **verified** |
| No OOM at any band on the CPU path | peak 10.47 GB/16 GB, swap 0.00M, 0 jetsam | **verified** |
| Metal not engaged on `origin/main` | exit 9, class=cpu, GPU idle, build fails | **verified** |
| Cause is `wgpu = ["inference"]` | E0432/E0433, reproduced twice on mini | **verified** |
| #2638 fixes the build | `cargo build --features wgpu` rc=0 at `94a13ab8a` | **verified** |
| #2638's deadlock is real on Metal too | `sample(1)` stack, cleared by the compute hunk | **verified** |
| Metal is 0.36× the CPU path here | n=6 vs n=6, dispersion <1 % | **verified** |
| 7B Q4_K_M needs 28.3 GB F32 for Metal | server's own line + 15.3 GB swap observed | **verified** |
| Why aggregate plateaus at ~10.5 tok/s | not isolated | **UNPROVEN** |
| Whether #2638 passes CI | not tested | **UNPROVEN** |
| The single 49 % GPU sample | unexplained | **UNPROVEN** |
| Arm B (comparator) on mini | not attempted, by design (§7) | **not attempted** |
