# Performance parity with llama.cpp — inference

**Status:** DRAFT for review · **Date:** 2026-09-01 · **Tree:** `origin/main` `b7bfcafa1`
**Supersedes:** epic #2706 (APR-PERF-GATE-001) and **thirteen** documents across four repositories (§11).
**Scope:** the *only* specification governing inference performance of models in this project.

Every number below carries its provenance. Nothing here is a target derived from a wish;
where a value is unmeasured this document says `UNMEASURED` and names who owes it.

---

## §1 What this document is, and the one thing it is for

`apr serve` must be as fast as `llama.cpp` at serving the same model on the same hardware,
and must be able to **prove it**. This document defines what "as fast" means, how it is
measured, what may be claimed, and what blocks a merge or a release.

It replaces a landscape of thirteen overlapping specifications (§11) totalling roughly
14,800 lines across `aprender`, `realizar`, `qwen-coder-deploy` and `trueno`. That landscape
is the reason two of them could each be "the parity spec" and disagree.

**One spec. One harness. One receipt. One comparator.**

---

## §2 Measured ground truth

This section is data, not policy. Every row is a measurement someone actually took, with the
host, model, date and comparator pin that produced it. Nothing in §4–§7 may contradict it.

### §2.1 aprender, 7B on RTX 4090

`scripts/llama_pin.toml`, measured **2026-08-25**, host `lambda` (RTX 4090, sm_89), model
`qwen2.5-coder-7b-instruct-q4_k_m.gguf`, comparator `llama.cpp` pinned `39173bcac`, 15s warmup /
30s window / 3 runs, streaming:

| band | llama agg | apr agg | **agg ratio** | llama dec | apr dec | **dec ratio** |
|---|---|---|---|---|---|---|
| c=1  | 168.9  | 90.2  | **0.534×** | 171.5 | 100.7 | **0.587×** |
| c=4  | 484.7  | 111.9 | **0.231×** | 123.3 | 113.8 | 0.923× |
| c=8  | 650.5  | 109.6 | **0.169×** | 83.0  | 112.2 | 1.352× |
| c=16 | 1120.8 | 108.4 | **0.097×** | 71.2  | 110.6 | 1.554× |

### §2.2 The two readings of this table, and why only one is honest

**apr's aggregate throughput is flat at ~110 tok/s at every concurrency.** It does not batch;
it serialises. Per-user decode *rises* with concurrency (1.554× at c=16) purely because each
request gets the whole GPU in turn while llama.cpp shares it sixteen ways.

> A gate that reports only per-user decode would call 0.097× aggregate a PASS.
> — `scripts/llama_pin.toml`, in its own words

**Therefore both metrics are gated, on every band (I-4).** This is the single most important
rule in this document and it is derived from a measurement, not from taste.

### §2.3 The gap scales with model size

`qwen-coder-deploy/docs/specifications/gpu-performance-spec.md`, measured **2026-03-12**, host
`yoga` (RTX 4060 Laptop, 1900MHz), model **Qwen2.5-Coder-1.5B** Q4_K_M, `probador llm load`,
60s streaming isolated:

| band | runtime | decode tok/s | aggregate tok/s | TTFT p50 | ITL p50 |
|---|---|---|---|---|---|
| c=1 | llama.cpp | **161.7** | — | 10.2 ms | 6.2 ms |
| c=1 | realizr (now `aprender-serve`) | 148.5 → **0.92×** | — | 14.0 ms | 6.7 ms |
| c=1 | ollama | 164.6 | — | 69.8 ms | 6.1 ms |
| c=4 | llama.cpp | 89.2 | **348.7** | 26.0 ms | 11.2 ms |
| c=4 | realizr | 65.6 | 259.7 → **0.745×** | 39.8 ms | 15.3 ms |
| c=4 | vLLM | 150.4 | **594.8** | 25.3 ms | 6.7 ms |

**1.5B on a 4060 is 0.92× single-stream. 7B on a 4090 is 0.587×.** Two models, two hosts, one
direction: the deficit grows with model size. **Any scope decision that concedes single-stream
decode as "already at parity" is reading the small-model number.** `[U]` — the two runs differ
in model, host, date and harness, so this is a *hypothesis with two supporting points*, not an
established scaling law. §12 owes the controlled measurement.

### §2.4 The one root cause ever actually isolated

`realizar/docs/specifications/decoder-throughput-specification-llama-mistral-phi-qwen.md`:

| | before | after | factor |
|---|---|---|---|
| GEMV latency (1×4096×4096) | 4.41 ms | 0.023 ms | **192×** |

> **Non-coalesced global memory reads in M=1 GEMV during token generation**, reducing effective
> memory bandwidth by 68×. The initial implementation prioritised algorithmic simplicity over
> memory access patterns. **This is the root cause.**

Decode is `M=1` GEMV bound. This is the only five-whys in the corpus that terminated in a
mechanism and a fix, and it is the prior most likely to explain §2.1's c=1 deficit.

### §2.5 Ollama is not the comparator, and the reason is instructive

Ollama posts the **best** c=1 decode (164.6, +2% on llama.cpp) and the **worst** TTFT
(69.8 ms, 7× llama.cpp) — exclusive GPU access with a serial HTTP layer. That is apr's current
shape too (§2.2). **llama.cpp is the comparator because it is the one that batches.**

---

## §3 Metrics — defined once, here

Adapted from `qwen-coder-deploy/docs/specifications/benchmarking-v2.md`, which retired
`tok/s = total_tokens / wall_time` for conflating prefill with decode and rewarding verbosity.

| metric | definition | what it is about |
|---|---|---|
| **TTFT** | request → first token | prefill + queue wait; perceived responsiveness |
| **TPOT** | `(E2E − TTFT) / (output_tokens − 1)`, request-weighted | per-token decode |
| **ITL** | as TPOT, token-weighted | per-token decode, long replies dominate |
| **E2E** | `TTFT + TPOT × (tokens − 1)` | total request time |
| **decode tok/s** | per-request decode rate | what one user feels |
| **aggregate tok/s** | Σ tokens across all in-flight requests ÷ window | what the server delivers |
| **scaling_efficiency(c)** | `(agg(c) / agg(1)) / c` | whether concurrency buys anything |

`agg` and `decode` are **not interchangeable** and §2.2 is why.

---

## §4 What parity means

**P-1 · Parity is a paired measurement.** The target for any metric is the comparator's value
**from the same run, on the same host, against the same model**. No literal ratio ever enters
this document as a threshold. `0.415`, `1.43×`, `2.93×` are data; none is a target.

**P-2 · Parity is per-band.** A claim of parity names its band. "apr reaches parity" without a
band is not a claim, it is a slogan.

**P-3 · Both metrics, every band (§2.2).** `agg_ratio ≥ 1.0` **and** `dec_ratio ≥ 1.0` at
c ∈ {1, 4, 8, 16}. Either alone certifies a serialising server.

**P-4 · Latency does not regress to buy throughput.** TTFT and ITL bounds are `UNMEASURED`
pending §12.1's noise floor and are **REPORTING** until then. They are not zero, they are
unmeasured, and the difference is recorded rather than defaulted.

---

## §5 Protocol

| element | value | source |
|---|---|---|
| workload | fixed 512-token prompt / 128-token generation (W1) | v2.2 §4.3.2 |
| bands | c ∈ {1, 4, 8, 16} | `llama_pin.toml` |
| replicates | N = 3 per cell | v2.2 |
| warmup / window / cooldown | 15 s / 30 s / 10 s | `llama_pin.toml` |
| streaming | required | TTFT is unmeasurable without it |
| client | **one binary drives both servers** | v2.2 I-15 |
| comparator | `llama.cpp` server, commit pinned in the receipt | — |
| tokenization | declared, no default | v2.2 I-13 |

**One client, both servers.** `llama-bench` is not admissible: it is a different client with a
different request shape, and a ratio between two harnesses measures the harnesses.

---

## §6 Invariants

I-1 … I-14 are carried forward from APR-PERF-GATE-001 v2.2, which is the asset worth keeping
from that epic. I-15 … I-18 are new and each is derived from a defect found in this project.

| # | invariant | mutation that must turn it RED |
|---|---|---|
| **I-1** | Expected cell set enumerated from committed `perf-matrix.yaml`; the verdict job asserts every cell present | delete one cell's receipt |
| **I-2** | `provenance.compute_class` is the dispatch path **taken**, read from the running process — never the hardware present. `gpu_layers_resolved` is read from the loader, never inferred from the request | report `cuda` on a CPU-only build |
| **I-3** | No `ratio` is representable without a `baseline` object that itself passes every receipt rule | emit a ratio with a bare scalar baseline |
| **I-4** | **Both `agg_ratio` and `dec_ratio` are gated on every band.** A receipt carrying one without the other is schema-fatal | emit a decode-only receipt at c=16 |
| **I-5** | `timeouts > 0` on any band is fatal to that host's ratio | inject one timeout |
| **I-6** | No wall-clock ratio is a **merge**-phase check | promote a ratio arm to the required set |
| **I-7** | Raw samples retained on every cell; summary-only receipts rejected | strip the samples array |
| **I-8** | Comparator `http_concurrency` **equals** the band's `c` | pin the comparator at 1, run band 16 |
| **I-9** | A cell, once run, is spent; it may not be re-run to green | re-run a failed cell and publish the second |
| **I-10** | No request is issued at or after window close; pre-close requests are drained; `drain_ms` recorded | issue one request after close and count its tokens |
| **I-11** | `tokenization.method` has no default; absence is schema-fatal | omit the block |
| **I-12** | No comparator ratio is published outside a receipt | print a ratio to stdout |
| **I-13** | `max_in_flight` is reported by the **server**, never inferred by the harness | have the harness compute it |
| **I-14** | Auto-fit never modifies an explicitly-set argument | set `--gpu-layers 12`, have auto-fit raise it |
| **I-15** | **No boolean accelerator flag.** Every accelerator request is a quantity or a device list, and each has a reported resolution | add `--gpu` with no `gpu_layers_resolved` |
| **I-16** | **`receipt.provenance.compute_class` must equal `perf-matrix.yaml[host].compute_class`, and the binary must be able to reach that class.** A declared class no build can produce is schema-fatal | declare `metal` where no Metal path exists (#2841) |
| **I-17** | **A parity claim names its band.** A band-less ratio is schema-fatal | publish "apr is at parity" with no band |
| **I-18** | **The measuring binary is built from a commit that is an ancestor of `HEAD`, and its sha256 is in the receipt** | measure with a `+no-git` build |

### §6.1 I-15 and I-16 are live defects today, not hypotheticals

```
$ git show origin/main:scripts/llama_pin.toml | grep apr_serve_command
apr_serve_command = "apr serve run {model} --gpu --port {port} ..."
```

`--gpu` is the boolean flag I-15 forbids, in the file that drives every parity measurement
this project takes. **PERF-021 was meant to retire it and the harness still uses it.**

I-16 is #2841: `perf-matrix.yaml` declares host `mini` as `compute_class: metal`, and `apr`
has no Metal inference path — `aprender-serve` pins `aprender-gpu` to `features = ["cuda"]`,
nothing enables `aprender-gpu/metal`, its `Backend` trait has no compute method, and there is
no Q4_K kernel in any form. A `mini` cell run today records a CPU run as `metal`, permanently
under I-9.

---

## §7 The gate

**Merge phase** — integrity only. No wall-clock ratio (I-6), because a timing check on a shared
runner produces red PRs that are indistinguishable from real regressions, and the team routes
around them.

- receipt schema valid; every invariant in §6 that is checkable statically
- **no timing assertion of any kind**

**Release phase** — the parity claim.

- P-3: both ratios, every band, against the same-run comparator
- every cell in `perf-matrix.yaml` present or explicitly `UNMEASURED` with owner and expiry

**Three-valued cell status, and `Skip` is not a pass:**

| status | meaning | requires |
|---|---|---|
| `MEASURED` | a conformant receipt exists | receipt path + commit + binary sha256 |
| `UNMEASURED` | temporary; **counted against the denominator** | `owner`, `expires` |
| `NOT_APPLICABLE` | permanent; excluded from the denominator | `reason`, `decided_by`, `date` |

---

## §8 Scope — one gated cell, everything else REPORTING

The predecessor epic declared eight cells across four hosts and filled two in five days, and
neither was ratcheted. This document gates **one** cell and reports the rest.

| cell | host | model | status |
|---|---|---|---|
| **REFERENCE (gated)** | `lambda` RTX 4090, CUDA | Qwen2.5-Coder-7B Q4_K_M | the parity claim is made here |
| gx10 (GB10, aarch64) | CUDA | same | REPORTING |
| intel (`mac-server`) | CPU / wgpu | same | REPORTING |
| mini (M4) | — | same | **blocked on #2841** — no backend to measure |

**A second host may be promoted to gating only after the reference cell reaches parity.**
Breadth before depth is what produced eight empty cells.

---

## §9 What is known to be wrong, in priority order

| # | finding | evidence | status |
|---|---|---|---|
| **1** | **apr does not batch.** Aggregate is flat at ~110 tok/s from c=1 to c=16 | §2.1 | the whole aggregate gap |
| **2** | **Single-stream decode is behind**, 0.587× at 7B | §2.1 | not conceded — §2.3 |
| **3** | M=1 GEMV memory coalescing | §2.4, 192× on a prior codebase | likely mechanism for #2 |
| **4** | The harness uses a boolean `--gpu` | §6.1 | violates I-15 |
| **5** | `mini` declares a backend that does not exist | #2841 | violates I-16 |

**#1 and #2 are different subsystems** — scheduler and kernels. They are worked one at a time,
because concurrent work on both confounds every run.

---

## §10 What this document does not do

- **It does not set a latency threshold.** TTFT/ITL bounds are `UNMEASURED` (P-4).
- **It does not define an attribution identity.** A per-phase wall-clock decomposition was
  proposed and is **not adopted here**: under concurrency one request's queue wait is another's
  kernel time on the same device, and a receipt covers 540 requests across 4 bands, so a
  receipt-level identity over-counts. If it returns it must state per-request or per-run, and
  the per-run form must be over device time, not wall clock.
- **It does not declare an iteration budget.** A declared budget with a retrospective five-whys
  is not a control; nothing in it can decline work.
- **It does not gate a non-CUDA backend** (§8).
- **It does not cover training throughput, LAPACK-bound solvers, or datacenter-scale serving.**
  These are `NOT_APPLICABLE` with `decided_by` recorded, not silently dropped.

---

## §11 Archived by this document

Thirteen documents, ~14,800 lines, four repositories. Each moves to `docs/archive/` in its own
repository with a one-line pointer here. **Nothing is deleted**; superseded material that is
still readable as current is the condition this document exists to end.

| repo | document | lines |
|---|---|---|
| aprender | `APR-PERF-GATE-001-v2.2.md` | 1265 |
| aprender | `APR-PERF-GATE-001-RESTART.md` | 77 |
| aprender | `APR-PERF-GATE-001-status-review.md` | 251 (retained as the post-mortem) |
| qwen-coder-deploy | `gpu-performance-spec.md` | 5130 |
| qwen-coder-deploy | `perf-parity-spec.md` | 748 |
| qwen-coder-deploy | `benchmarking-v2.md` | 527 |
| realizar | `benchmarking-with-common-models-common-serving-spec.md` | 1628 |
| realizar | `benchmark-model-runners-spec.md` | 1021 |
| realizar | `llama-cpp-style-performance-spec.md` | 803 |
| realizar | `qwen-performance-improve.md` | 715 |
| realizar | `deterministic-reproducible-cargo-bench.md` | 541 |
| realizar | `decoder-throughput-specification-llama-mistral-phi-qwen.md` | 445 |
| realizar | `qwen-showcase-throughput-improve.md` | 344 |
| realizar | `performance-parity-ollama-llamacpp-gpu-inference-llms.md` | 54 (already SUPERSEDED) |
| trueno | `CUDA-parity-spec.md` | 1497 |

---

## §12 Unmeasured, owed, and named

| # | item | owner | why it matters |
|---|---|---|---|
| **12.1** | **Noise floor σ per metric per host.** No minimum detectable effect is published, so no run can distinguish a real move from noise | perf-gate | every threshold in §4 rests on it; P-4 stays REPORTING until it lands |
| 12.2 | §2.3's model-size scaling, controlled (same host, same harness, 1.5B vs 7B) | perf-gate | decides whether single-stream may ever be conceded |
| 12.3 | Whether llama.cpp can be driven at c>1 by our client without patching it | perf-gate | I-8 is unenforceable otherwise |
| 12.4 | Instrumentation overhead of any per-phase timing | perf-gate | unpriced; §10 declines the identity partly for this reason |
| 12.5 | `mini`'s backend decision | #2841 | I-16 blocks the cell until resolved |

**§12.1 is step zero.** Not because a threshold needs it — every rule in §4 is a paired
comparison against the same run, so no literal is required — but because without σ, a run that
moved nothing and a run that moved everything are the same artifact.
