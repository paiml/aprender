# Performance parity with llama.cpp — inference

**Status:** REVIEWED DRAFT — quorum + `agy /teamwork`, four rules changed by review (§11.1) · **Date:** 2026-09-01 · **Tree:** `origin/main` `b7bfcafa1`
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

> ## ⚠ §2.1 IS WITHDRAWN. It measured a build with batching compiled out.
>
> `evidence/parity-http/findings.json` records the run as
> **`apr_build: "0.64.0 (53062e7f3), --features cuda"`, dated 2026-08-24.**
> At that commit `with_cuda_batch_tx` sits behind `#[cfg(feature = "cuda-batch")]`, and
> `crates/apr-cli/Cargo.toml:90` declares **`cuda-batch = ["cuda"]`** — the implication runs
> the *wrong way*, so `--features cuda` compiled continuous batching **out**. Every request
> took the path `cuda_chat_backend.rs:149` labels `// Fallback: direct RwLock path (serialized)`.
>
> It was fixed by **`a18b1aced`, 2026-08-25 20:21 — 27 hours after the measurement** — in a
> commit titled *"continuous batching was never compiled into any build a user is told to make."*
>
> **The counter-measurement is committed in this tree.** `evidence/perf-gate-001-w1-lambda/`,
> taken 2026-09-01 on `745fa8588`, same host, same model, logs
> `CONTINUOUS BATCHING: max_batch=11` and aggregates **99.9 → 191.6 → 353.9 → 449.9**. It scales.
>
> | band | llama | apr @53062e7f3 (batching OFF) | apr @main | ratio then | ratio now |
> |---|---|---|---|---|---|
> | c=1 | 168.9 | 90.2 | 99.9 | 0.534 | 0.591 |
> | c=4 | 484.7 | 111.9 | 191.6 | 0.231 | 0.395 |
> | c=8 | 650.5 | 109.6 | 353.9 | 0.168 | 0.544 |
> | c=16 | 1120.8 | 108.4 | 449.9 | **0.097** | **0.401** |
>
> **The gap is indicatively ~2.5×, not 10.34×.** `[U]` — the two artifacts run *different
> workloads* (§2.1 is 128/128 single-prompt via `http_profile = "medium"`; the perf-gate
> receipts are W1, 512/128 over 1622 prompts), so the right-hand column is indicative and not a
> ratio. **No conclusion in this document may rest on §2.1 until a paired re-measure exists.**
>
> This is the failure the whole document exists to prevent — a number that did not prove the
> mechanism engaged — and it was found by a planning agent reading the build flags, not by any
> gate here. §6.2 gains that as its own finding.

### §2.1 aprender, 7B on RTX 4090 — WITHDRAWN, retained for the record

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

**apr's aggregate was flat at ~110 tok/s at every concurrency in the withdrawn run** — because
batching was compiled out of that binary. **On `main` it scales** (99.9 → 449.9). The reasoning
below about decode-vs-aggregate remains valid as a *rule*; the numbers that motivated it do not. Per-user decode *rises* with concurrency (1.554× at c=16) purely because each
request gets the whole GPU in turn while llama.cpp shares it sixteen ways.

> A gate that reports only per-user decode would call 0.097× aggregate a PASS.
> — `scripts/llama_pin.toml`, in its own words

**Therefore both metrics are always REPORTED, and neither is gated alone (I-4).**

**But they are not both gated at `≥ 1.0` on every band, and the reason is the fix itself.**
apr's 1.554× decode at c=16 exists *because* it serialises. When continuous batching lands,
aggregate rises and per-user decode necessarily **falls** toward the comparator's shared-GPU
figure (llama.cpp is at 71.2 tok/s there). A rule demanding `dec_ratio ≥ 1.0` on every band
would reject the very PR that fixes §9's defect #1 — it demands apr *dominate* on two metrics
that trade against each other, which is a beat, not parity.

So the gate is asymmetric, and §7 states it:

| band | gated on | why |
|---|---|---|
| c=1 | `dec_ratio` vs comparator | no batching is possible; decode is the whole story |
| c ∈ {4,8,16} | `agg_ratio` vs comparator | what the server delivers, and what batching improves |
| every band | **both reported**; a receipt carrying one alone is schema-fatal | §2.2 |

**Decode at c>1 is REPORTING until aggregate parity is reached — it is not gated at all**, and
the reason is that the obvious alternative is a trap. A non-regression floor against apr's *own
previous release* looks safe and is not:

```
apr decode at c=16 today (serialising) : 110.6 tok/s
llama.cpp decode at c=16 (batched)     :  71.2 tok/s
after batching lands, apr decode -> ~71.2  =  a 35.6% DROP
```

A non-regression floor rejects that, so the "fixed" rule rejects the batching PR exactly as the
comparator-ratio rule did — **the same defect in a new spelling**, which is this repository's
most-repeated failure (#2707 shipped #2696 again under a new name).

So: at c>1 decode is **recorded and never gated** until `agg_ratio ≥ 1.0`. Once batching has
landed, a non-regression floor is established **from the post-batching baseline**, where it
protects against silent collapse without forbidding the transition that has already happened.

### §2.3 Cross-model data, and what it does NOT establish

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

**1.5B on a 4060 is 0.92× single-stream. 7B on a 4090 is 0.587×.** `[U]` — two runs differing
in model, host, date *and* harness, with the bottleneck plausibly shifting between them (a 4060
Laptop is bandwidth-starved where a 4090 is not). **This is not a scaling law, and no rule in
this document rests on it.** It is recorded because it is the only cross-model data that exists;
§12.2 owes the controlled measurement.

**The policy it was first written to justify — that single-stream decode is not conceded —
rests instead on §2.1 alone: 0.587× on the reference cell.** That measurement is sufficient,
and it is not confounded.

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
| **I-4** | **Both `agg_ratio` and `dec_ratio` are REPORTED on every band; a receipt carrying one alone is schema-fatal.** Which is *gated* is asymmetric and set by §7 — never both at `≥1.0` on every band, because under batching the two trade against each other (§2.2) | emit a decode-only receipt at c=16 |
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

### §6.1a Three findings from the final review round

**(i) The counter-measurement that withdrew §2.1 is itself under-provenanced.**
`cuda_batch_scheduler.rs:38` reads `max_batch` from the **`CUDA_MAX_BATCH` env var, default 4**.
The 2026-09-01 run logged `max_batch=11`, and that value appears **nowhere** — not in the
receipt, not in `invocation.txt`, not in `perf-matrix.yaml` or `llama_pin.toml`. It survives only
in `server-startup.txt`. Batching demonstrably engaged and aggregate demonstrably scaled, so the
withdrawal of §2.1 stands — but **that artifact is not reproducible from its own receipt**, and it
may not become a baseline until it is. At least ten env vars change the decode path
(`CUDA_MAX_BATCH`, `CUDA_BATCH_WINDOW_MS`, `ITERATION_SCHEDULER`, `GRAPH_DISPATCH`,
`BATCHED_GRAPH`, `FP8_DECODE`, `CUBLAS_GEMM_THRESHOLD`, `APR_STREAM_NONBLOCKING`,
`STAGGERED_PREFILL`, `APR_FORCE_BATCHED_PATH`) and **none is recorded in any receipt**. I-2 must
extend to the scheduler configuration, or every receipt is a run of an unnamed configuration.

**(ii) "Uncap `max_batch = 4`" fixes a limit that is not the limit.** Both execution plans
proposed it. The active cap in the measured run was 11, set by an env var. Editing the Rust
default would not have moved it, and forcing 16 risks OOM if 11 was chosen for KV-cache headroom
on a 24 GB card.

**(iii) The residual gap is probably not a kernel problem at all.** Above `m ≥ 4` the batched
path routes to **cuBLAS GEMM** — NVIDIA's own kernels. If the aggregate deficit persists there,
it is host-side: admission, tokenization, transport, or KV-cache latency. **Scoping kernel work
before profiling would misdiagnose the bottleneck**, and §9 #3 is already discharged (ten
coalesced/DP4A GEMV variants ship in `crates/aprender-gpu/src/kernels/quantize/q4k/coalesced/`,
and the batched path calls no GEMV).

### §6.2 The gate in §7 is not implementable today, and this is the blocking item

Assimilated from the quorum that reviewed the rejected v3.0 plan; both verified independently
against `origin/main`.

**(a) The producer cannot emit a comparator ratio, by design.**
`crates/apr-cli/src/commands/test_llm_band.rs:231`:

```rust
fn comparator_status(args: &BandArgs<'_>) -> ComparatorStatus {
    ComparatorStatus::Unmeasured { .. }      // unconditional
}
```

guarded by a unit test at `:717` named **"The producer must never be able to emit a comparator
ratio."** Its doc comment is explicit: *"There is no CLI path to a measured ratio, and that is
the point."* That was the right call under the old epic — a synthesised ratio is the fabrication
it existed to remove — but it means **§7's gate has no producer**.

**(b) `decode_tok_per_sec` is absent from every band of every committed receipt.**

```
$ python3 -c "...evidence/perf-gate-001-w1-lambda/receipt.r1.json..."
[(1,'UNMEASURED','<ABSENT>'), (4,'UNMEASURED','<ABSENT>'),
 (8,'UNMEASURED','<ABSENT>'), (16,'UNMEASURED','<ABSENT>')]
```

Both cells, both hosts, all replicates, all four bands. `token_times_s` is `[]` on every sample
row, so per-token decode is not merely unaggregated — it is **not captured**.

**Consequence:** the §7 gate is a specification of a check nothing can currently feed. **The
comparator lane and decode capture are the first deliverable**, before any optimisation work,
and they need no matrix run. The numbers in §2.1 come from `llama_pin.toml`'s standalone
harness, not from this producer — which is why they exist at all.

**(c) `perf-matrix.yaml` already encodes the rule §2.2 forbids.** `scripts/perf-matrix.yaml`
declares Arm **B2 floor = 1.00 on every band** and `perf_gate.sh:246` implements it. That is
exactly the "decode ≥ 1.0 everywhere" rule that would reject the continuous-batching PR (§2.2).
**§7 supersedes it**, and the matrix must be edited in the same PR that lands this spec's gate,
or the two definitions disagree silently.

**(d) The measurement client omits `seed` and `ignore_eos` from the wire.**
`crates/aprender-test-lib/src/llm/client.rs:436-448` rebuilds the request body with
`seed: None, ignore_eos: None`, and both fields are `#[serde(skip_serializing_if)]`, so they are
**omitted rather than nulled**. Every number in §2.1 was therefore taken with unseeded sampling
and no output-length pin, on both lanes. It does not invalidate an aggregate-throughput
comparison, but it bounds what any *determinism* or *per-token* claim from this harness can mean.

---

## §7 The gate

**Merge phase** — integrity, plus one *deterministic* speed check. No comparator ratio and no
HTTP timing (I-6): a wall-clock ratio on a shared runner produces red PRs indistinguishable
from real regressions, and the team routes around them.

- receipt schema valid; every §6 invariant checkable statically
- **kernel microbenchmark gate (§7.1)** — deterministic, in-process, no socket
- **no HTTP timing assertion of any kind**

**Release phase** — the parity claim, asymmetric per §2.2:

| band | gated | against |
|---|---|---|
| c=1 | `dec_ratio ≥ 1.0` | the comparator, same run — **REPORTING until decode σ is measured (§12.1b)** |
| c ∈ {4,8,16} | `agg_ratio ≥ 1.0` | the comparator, same run |
| c ∈ {4,8,16} | `dec_ratio` | **REPORTING** until `agg_ratio ≥ 1.0`; a non-regression floor is set from the **post-batching** baseline thereafter (§2.2) |

Plus: every cell in `perf-matrix.yaml` present, or `UNMEASURED` with owner and expiry.

### §7.1 The kernel microbenchmark gate — and why it is the cheap one

§2.4 already names the mechanism behind single-stream decode: **M=1 GEMV memory coalescing**,
192× on a prior codebase. If the bottleneck is a kernel, gating pull requests on a three-minute
HTTP client-server harness over a socket is the wrong instrument at the wrong phase — it is
slow, it is noisy, and its noise floor is unmeasured (§12.1).

**Every PR is gated on an in-process, deterministic kernel benchmark** — M=1 GEMV at the
reference model's shapes, plus the dequant-matmul path — asserted against a committed
ratchet-down baseline, with no network, no server and no comparator. The E2E harness is a
**release** instrument only.

This is the one mechanism that makes this specification cheaper to run than its predecessor,
whose measurement instrument cost more than its measurements. `[U]` — the microbenchmark's own
variance is unmeasured, and it inherits §12.1's obligation before it may ratchet.

### §7.2 No-regression on every functional cell, not just the gated one

§8 gates *parity* on one cell. It does **not** exempt the others from regression: a change that
doubles CUDA throughput and breaks CPU, wgpu or aarch64 must not merge green. Every cell that
*functions* carries a correctness-and-non-regression check even while its parity status is
REPORTING. Gating parity narrowly and regression broadly are different decisions, and conflating
them is how a project optimises one GPU and discovers portability failures at release.

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
| **1** | ~~apr does not batch~~ — **WITHDRAWN.** It batches on `main` (`max_batch=11`); the claim came from a build with the feature compiled out | §2.1 banner | the whole premise, retracted |
| **2** | **Single-stream decode is behind** — 0.591× on main's indicative figure | §2.1 | still real, and now the *largest* known gap |
| **3** | ~~M=1 GEMV coalescing~~ — **DISCHARGED.** `crates/aprender-gpu/src/kernels/quantize/q4k/coalesced/` ships ten coalesced/DP4A GEMV variants; the 192× non-coalesced defect does not exist here. Above m≥4 the batched path routes to cuBLAS GEMM and does not call a GEMV at all | tree | the kernel lever has **no named live mechanism** |
| **4** | The harness uses a boolean `--gpu` | §6.1 | violates I-15 |
| **5** | `mini` declares a backend that does not exist | #2841 | violates I-16 |

**#1 and #2 are different subsystems** — scheduler and kernels. They are worked one at a time,
because concurrent work on both confounds every run.

---

## §10 What this document does not do

- **It does not set a latency threshold.** TTFT/ITL bounds are `UNMEASURED` (P-4).
- **It does not define a *summing* attribution identity, and that refusal is now measured.**
  A proposal required every receipt to satisfy
  `wall_clock == t_prefill + t_decode_kernel + … + t_residual`, with a non-summing decomposition
  schema-fatal. Run against the six committed W1 receipts, the sum of per-request samples over
  each band's own span is **exactly the concurrency**:

  | band | Σ samples_ms | span_ms | ratio |
  |---|---|---|---|
  | c=1  | 60,422.3  | 60,422.3 | **1.000** |
  | c=4  | 245,806.9 | 61,451.9 | **4.000** |
  | c=8  | 486,072.2 | 60,759.7 | **8.000** |
  | c=16 | 994,186.0 | 64,017.9 | **15.530** |

  Whole-receipt ratio **7.24** on lambda (r1/r2/r3: 7.24 / 7.22 / 7.23) and **3.36–3.54** on
  gx10. Concurrent requests overlap, so a receipt-level sum over-counts by ×c *by construction*.

  **What is adopted instead:** per-phase **averages** (Σ ÷ n, which is what the ×c cancels to)
  and **device utilization**, neither of which requires a closed identity. Attribution is not
  refused — *summing* is. A future amendment may add averaged phase timings, bounded by the
  comparator's own vocabulary: llama.cpp's `result_timings` exposes `prompt_*` and `predict_*`
  and nothing else, so any **paired** decomposition is a two-term one, not an eight-term one.
- **It does not declare an iteration budget.** A declared budget with a retrospective five-whys
  is not a control; nothing in it can decline work.
- **It does not claim the two levers are causally independent.** #2844 factors the 10.34× gap
  into batching (6.07×) and kernel (1.70×), and `A × B = 10.34×` closes exactly — but that is an
  **identity by construction**, not evidence. The causal reading requires per-token efficiency
  under batching to equal per-token efficiency at M=1, and **it may not**: the kernel deficit was
  measured on an M=1 GEMV, and a batched path runs M=B, which is different code with a different
  arithmetic intensity. **The M=1 coalescing fix may not transfer to the batched regime at all.**
  Step 2's exit therefore measures the *batched* kernel's efficiency, and Step 3 is scoped to
  whichever kernel the batched decode path actually uses — not assumed to be the M=1 one.
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

## §11.1 How this document was reviewed, and what review changed

Reviewed before leaving DRAFT by an agent quorum and by `agy /teamwork` (cross-vendor).
**Four of its rules were wrong and were changed by review, not by the author:**

| # | what review found | what changed |
|---|---|---|
| 1 | **I-4 as first written outlawed its own fix.** Gating `dec_ratio ≥ 1.0` on every band demands apr dominate two metrics that trade against each other; the continuous-batching PR that fixes §9 #1 would necessarily lower per-user decode and be rejected | §2.2 and §7 made **asymmetric** — decode vs comparator at c=1, aggregate vs comparator at c>1, decode as non-regression against apr's own prior release |
| 2 | **The model-size scaling claim was doing policy work on two confounded points** | Demoted to `[U]`, load-bearing on nothing; the refusal to concede single-stream now rests on §2.1's 0.587× alone |
| 3 | **Single-cell gating invites silent rot on the other backends** — a PR doubling CUDA and breaking Metal merges green | §7.2 added: parity gated narrowly, **non-regression gated broadly** |
| 4 | **Refusing attribution outright was over-correction.** The over-count applies to *summing* overlapping spans, not to averaging or utilization | §10 rewritten: summing refused *with the measured ×c proof*; averages and utilization adopted |

Review also supplied the argument for **§7.1**, the kernel microbenchmark gate — the one
mechanism that makes this document cheaper to run than its predecessor. Its verdict on the
first draft was **do not leave DRAFT**; §7.1, §7.2 and the I-4 correction are the response.

The quorum's 37 verified objections against the rejected v3.0 plan supplied §10's measurement
and the comparator-vocabulary bound.

## §12 Unmeasured, owed, and named

| # | item | owner | why it matters |
|---|---|---|---|
| **12.1** | **σ for `aggregate_tok_per_sec` is MEASURED — see §12.1a.** What remains unmeasured is σ for *decode* (not captured at all, §6.2b) and for the comparator lane (no producer, §6.2a) | perf-gate | the noise floor is no longer step zero; the comparator lane is |
| 12.2 | §2.3's model-size scaling, controlled (same host, same harness, 1.5B vs 7B) | perf-gate | decides whether single-stream may ever be conceded |
| 12.3 | Whether llama.cpp can be driven at c>1 by our client without patching it | perf-gate | I-8 is unenforceable otherwise |
| 12.4 | Instrumentation overhead of any per-phase timing | perf-gate | unpriced; §10 declines the identity partly for this reason |
| 12.5 | `mini`'s backend decision | #2841 | I-16 blocks the cell until resolved |

### §12.1b Decode has no measured noise floor, so it is not gated yet

§12.1a measures σ for **aggregate** only. `decode_tok_per_sec` is not captured at all (§6.2b),
so its variance over the socket is unknown. **A gate on a metric with no measured noise floor is
a flaky gate**, and this project has already paid for that class. Decode gating at c=1 is
therefore **REPORTING** until capture exists and σ is computed from N=3 — which costs no matrix
run beyond the one Step 0 already needs.

### §12.1a The measured noise floor

Recomputed from the six committed receipts (`N = 3`, two hosts), which are exactly I-9's protocol:

| host | band | mean agg tok/s | sd | MDE (k=2, n=3) |
|---|---|---|---|---|
| lambda | c=1  | 100.643 | 0.635 | 0.73% |
| lambda | c=4  | 191.663 | 0.029 | **0.02%** |
| lambda | c=8  | 353.336 | 0.505 | 0.17% |
| lambda | c=16 | 450.405 | 0.467 | 0.12% |
| gx10 | c=1  | 6.203 | 0.005 | 0.09% |
| gx10 | c=4  | 39.039 | 0.016 | 0.05% |
| gx10 | c=8  | 76.432 | **14.011** | **21.17%** |
| gx10 | c=16 | 162.647 | 2.576 | 1.83% |

**lambda is quiet enough to gate** — a 6× lever is four orders of magnitude above its MDE.
**gx10 c=8 is not**: 21% MDE, consistent with the device-wide stall recorded in #2833 (8 of 774
requests at 41s against an 8.65s median). A cell whose MDE exceeds the effect it must detect is
`UNMEASURED`, not a passing cell.

**§12.1's remainder is step zero.** Not because a threshold needs it — every rule in §4 is a paired
comparison against the same run, so no literal is required — but because without σ, a run that
moved nothing and a run that moved everything are the same artifact.
