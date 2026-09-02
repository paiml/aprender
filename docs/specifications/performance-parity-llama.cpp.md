# Performance parity with llama.cpp — inference

**Status:** REVIEWED DRAFT — quorum + `agy /teamwork`, an external audit, an independent cross-vendor re-review, a three-role adversarial team round, then a four-lens re-grounding **against the tree rather than the document**; **rules changed by review in all five rounds**, the last three of which found that earlier rounds' own fixes had introduced a scheduling deadlock (§12), three wrong IDs in the appendix that documents ID stability, a misstated decomposition (§10), a band no device could pass (PP-24), a roofline rule that fired on correct batching (PP-23), a broad gate improved by making the server slower (§7.2), and a ten-merge sequence that satisfies every rule here and ships no speed (§12.0) — **and that the central premise of the §12 chain was false: the comparator-ratio producer exists and runs today** (§6.2a) (post-mortem: [`docs/postmortems/perf-parity-review-2026-09.md`](../postmortems/perf-parity-review-2026-09.md)) · **Date:** 2026-09-02 · **Tree:** `origin/main` `b7bfcafa1`
**Supersedes:** epic #2706 (APR-PERF-GATE-001) and **fifteen** documents across four repositories (§11).
**Scope:** the *only* specification governing inference performance of models in this project.

Every number below carries its provenance. Nothing here is a target derived from a wish;
where a value is unmeasured this document says `UNMEASURED` and names who owes it.

---

## §1 What this document is, and the one thing it is for

`apr serve` must be as fast as `llama.cpp` at serving the same model on the same hardware,
and must be able to **prove it**. This document defines what "as fast" means, how it is
measured, what may be claimed, and what blocks a merge or a release.

It replaces a landscape of fifteen overlapping specifications (§11) totalling roughly
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
> **Provenance correction.** An earlier draft of this banner attributed the batching repair to
> commit `a18b1aced`. `git merge-base --is-ancestor a18b1aced origin/main` returns **1** — it is
> not on main, and never was; it sat on an unmerged branch. The repair reached main on
> **2026-08-27**, folded into the epic-landing PR **#2705**. PP-18 and PP-21 exist to require
> exactly this check of a receipt, and the banner that withdraws a measurement *on
> build-provenance grounds* cited build provenance that does not check out.
>
> **The counter-measurement is committed in this tree** (figures below are `receipt.r1.json`; the
> §12.1a means differ — 353.336 and 450.405). `evidence/perf-gate-001-w1-lambda/`,
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
below about decode-vs-aggregate remains valid as a *rule*; the numbers that motivated it do not.

**In that run** per-user decode *rose* with concurrency (1.554× at c=16) while aggregate stayed
flat — because each request got the whole GPU in turn while llama.cpp shared it sixteen ways.
That is a property of a **serialising** server, and `main` is no longer one.

> A gate that reports only per-user decode would call 0.097× aggregate a PASS.
> — `scripts/llama_pin.toml`, in its own words

**The rule survives the withdrawal because it rests on a mechanism, not on that run.** Whenever
a server shares one device across `c` requests, per-user decode falls and aggregate rises; when
it serialises, the reverse. The two metrics therefore move in opposite directions under the
change that matters most here, and a receipt carrying one of them cannot be read. **No
measurement is required for that to be true**, which is why PP-4 stands while §2.1 does not.

**Therefore a receipt must always CARRY both metrics on every band (PP-4).** PP-4 is a rule about
receipt *completeness*, not about verdict formation: it forbids a receipt that records only one
metric, because that is how 0.097× aggregate gets reported as a decode PASS. **The verdict is
formed by §7**, which gates one metric per band and reports the other. Both rules can be
satisfied at once, and neither stands in for the other.

**But they are not both gated at `≥ 1.0` on every band, and the reason is the fix itself.**
apr's 1.554× decode at c=16 in the withdrawn run existed *because that build serialised*.
Continuous batching has since landed on `main` (§2.1 banner). The rule below is about the
**general mechanism**, not a pending event: whenever a server shares a GPU across c requests
rather than serialising them, aggregate rises and per-user decode necessarily **falls** toward the comparator's shared-GPU
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

**1.5B on a 4060 is 0.92× single-stream. 7B on a 4090 is 0.587× — both WITHDRAWN figures** (§2.1), retained only to show what the cross-model comparison was. `[U]` — two runs differing
in model, host, date *and* harness, with the bottleneck plausibly shifting between them (a 4060
Laptop is bandwidth-starved where a 4090 is not). **This is not a scaling law, and no rule in
this document rests on it.** It is recorded because it is the only cross-model data that exists;
§12.2 owes the controlled measurement.

**The policy it was first written to justify — that single-stream decode is not conceded — may
NOT rest on §2.1, which is withdrawn.** It rests instead on the *conservative default*: no
conformant paired measurement of single-stream decode exists, and a cell is not conceded on the
strength of a measurement nobody has taken. The indicative figure on `main` is **0.591×**, which
is a reason to look, not a basis to concede. §12.2 owes the measurement that would settle it.

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
(69.8 ms, 7× llama.cpp) — exclusive GPU access with a serial HTTP layer. **That was apr's shape
in the withdrawn run, and is not its shape on `main`**, which batches (§2.1 banner). The point
stands as a choice of comparator: **llama.cpp is the comparator because it batches, so a parity
claim against it is a claim about a server doing the same work.**

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

**P-3 · Both metrics on every band, but the gate is asymmetric (§2.2, §7).** A receipt carries
`agg_ratio` *and* `dec_ratio` at c ∈ {1,4,8,16} (PP-4). **Parity is claimed on the gated metric
for that band** — `dec_ratio` at c=1, `agg_ratio` at c>1 — never on both at once. Demanding both
`≥ 1.0` everywhere is a *beat*, not parity, and it would reject the continuous-batching work
outright, because sharing a GPU sixteen ways necessarily lowers per-user decode.

**P-5 · The verdict is a bootstrap bound, and `1.0` is a definition rather than a threshold.**
P-1 forbids literal thresholds, and an earlier draft then wrote `agg_ratio ≥ 1.0` as a trigger —
a contradiction an audit caught. **`1.0` is what parity *means*: the comparator's own same-run
value.** It is not a number chosen from the range of plausible numbers, which is what P-1 bans.

But a point estimate on `n = 3` compared against 1.0 is a coin flip at true parity, so the rule
is stated on the interval, not the point:

**`ε` requires `n ≥ 5`, and that requirement is stated HERE rather than buried in §12.1a.** Every
row of §12.1a is `n = 3`; §12.1a closes with "no σ-dependent status may change on `n < 5`"; a
PASS/REPORTING decision is a σ-dependent status change. So as first written **P-5 had no legal
`ε` on the day §7 armed** — the rule and its own precondition were in different sections and
disagreed. `ε` is computed from the receipt in hand at `n ≥ 5`; §12.1a's `n = 3` table is a
historical noise-floor estimate and may not supply it, the more so because PP-4 classifies the
receipts behind it as historical records that may not be used as a baseline.

> **PASS** iff the **lower bound of the one-sided 95% paired bootstrap interval** on the ratio
> is `≥ 1.0 − ε`, where `ε` is **the receipt's own measured MDE at `n ≥ 5`** (§12.1a) — a ratchet, not a
> literal. Otherwise **REPORTING**, never FAIL, until the comparator lane exists (§6.2).

So the gate reads σ instead of ignoring it, and every constant in the rule is either definitional
(`1.0`) or measured (`ε`).

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
| client | **one binary drives both servers** | v2.2 PP-15 |
| comparator | `llama.cpp` server, commit pinned in the receipt | — |
| tokenization | declared, no default | v2.2 PP-13 |

**One client, both servers.** `llama-bench` is not admissible: it is a different client with a
different request shape, and a ratio between two harnesses measures the harnesses.

---

### §5.1 W1's sampler is pinned, and the tokens are counted server-side

§6.1a records that the client omits `seed` and `ignore_eos` from the wire. That is not only a
determinism problem: **a 128-token generation that stops at token 40 makes `aggregate tok/s` a
function of the sampler rather than the server.**

W1 therefore pins, on both lanes: `temperature = 0`, `seed` fixed and recorded, `ignore_eos =
true`, `n_predict = 128`. Tokens are counted from the **server's** `usage.completion_tokens`,
never from SSE chunk arrivals, which may carry more than one token. **`completion_tokens == 128`
on every retained sample, or the sample is fatal to the cell** — a poka-yoke, so the sampler
*cannot* shorten the workload.

### §5.2 The comparator's configuration is pinned, not just its commit

Pinning a SHA does not pin a configuration. PP-8 requires the comparator's concurrency to equal
the band's `c`, and the obvious fix — appending `-np {c}` — makes the comparator **worse**:
`llama-server` divides `-c` across `-np` slots, so `-np 16 -c 4096` leaves 256 tokens per slot
and W1's 512 + 128 no longer fits. **Verified at the pinned commit, and it is worse than that:**
`llama.cpp@39173bcac src/llama-context.cpp:174-178` computes
`n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256)` when `kv_unified` is off, and
`tools/server/server-context.cpp:2155-2172` **rejects** an over-long request with
`ERROR_TYPE_EXCEED_CONTEXT_SIZE` and releases the slot — it does not truncate. And it breaks at
**`-np 8`**, not 16: `4096 / 8 = 512 < 512 + 128`.

**A single pinned argv and a per-band concurrency are not compatible, and PP-8 needs both.**
§6.1b records that `parity_host_receipt.sh` starts the comparator *once, outside* the band loop,
so its slot count is fixed while apr's concurrency runs 1 → 16. Pinning one argv harder does not
fix that — it guarantees it. **The pin is a template, and the comparator is relaunched once per
band**, with `{c}` substituted into `-np` and `-c` together; one server process per band, not
one per cell. A comparator held at one slot count across four bands makes three of the four
ratios a comparison between different configurations, which PP-22 would refuse if the mismatch
were declared and does not catch because it is not.

So the contract is: the **full argv** lives in `llama_pin.toml` as a per-band template with
`-np {c}` and `-c` scaled by `c`;
`n_ctx_slot ≥ 640` is asserted; every default that moved in llama.cpp during 2026 is pinned
explicitly (`-fa`, `-b`/`-ub`, `--cache-type-k`/`-v`, `-cb`, slot-save off); and a `GET /props`
snapshot from the running server is stored in the receipt — **the comparator's own PP-2**.

### §5.3 Latency is not measured under W1 (W3)

W1 is closed-loop, homogeneous and starts every client together, so each round's requests finish
within a decode step of one another and all `c` prefills collide at every round boundary.
Aggregate is unaffected; **TTFT and ITL distributions under W1 are artifacts of the harness, not
of the server.** Gating latency from them would gate a convoy.

**Latency is measured under W3 — open-loop, Poisson arrivals — and nowhere else.** Until W3
exists, P-4 stays REPORTING for that reason, not merely for want of a noise floor. v2.2's W2
ragged mixture is re-adopted as the aggregate secondary.

## §6 Invariants

**The IDs are namespaced `PP-nn` and are NOT v2.2's `I-nn`.** An earlier draft renumbered v2.2's
invariants in place, so `I-4` meant *raw samples retained* in one document and *both metrics* in
the other, while 42 `roadmap.yaml` references pointed at neither unambiguously. **Appendix A maps
every v2.2 `I-nn` to its disposition here**, and four v2.2 controls this document had silently
dropped are re-adopted below as PP-19…PP-22.

| # | invariant | mutation that must turn it RED |
|---|---|---|
| **PP-1** | Expected cell set enumerated from committed `perf-matrix.yaml`; the verdict job asserts every cell present | delete one cell's receipt |
| **PP-2** | `provenance.compute_class` is the dispatch path **taken**, read from the running process — never the hardware present. `gpu_layers_resolved` is read from the loader, never inferred from the request. **And the receipt records the scheduler configuration actually in force** — every environment variable that changes the decode path (`CUDA_MAX_BATCH`, `CUDA_BATCH_WINDOW_MS`, `ITERATION_SCHEDULER`, `GRAPH_DISPATCH`, `BATCHED_GRAPH`, `FP8_DECODE`, `CUBLAS_GEMM_THRESHOLD`, `APR_STREAM_NONBLOCKING`, `STAGGERED_PREFILL`, `APR_FORCE_BATCHED_PATH`), read from the process, absent-or-default stated explicitly. A receipt without it is a run of an unnamed configuration (§6.1a i) | report `cuda` on a CPU-only build; separately, omit the scheduler block on a run where `CUDA_MAX_BATCH` was set |
| **PP-3** | No `ratio` is representable without a `baseline` object that itself passes every receipt rule | emit a ratio with a bare scalar baseline |
| **PP-4** | **Both `agg_ratio` and `dec_ratio` are REPORTED on every band; a receipt carrying one alone is schema-fatal — for receipts produced from this spec forward.** Receipts predating it (every one in `evidence/` today, §6.2b) are **historical records, not conformant receipts**: they may be cited as evidence of what a run did, and may not be used as a baseline or to support a parity claim. Which is *gated* is asymmetric and set by §7 — never both at `≥1.0` on every band, because under batching the two trade against each other (§2.2) | emit a decode-only receipt at c=16 |
| **PP-5** | `timeouts > 0` on any band is fatal to that host's ratio | inject one timeout |
| **PP-6** | **No comparator wall-clock ratio is a merge-phase check.** The bar is on *comparator* ratios — the class that produces red PRs indistinguishable from real regressions on a shared runner. A deterministic in-process microbenchmark against a committed self-baseline (§7.1) is not a comparator ratio and is not barred; it is also not a parity claim | promote an Arm-B ratio to the required set |
| **PP-7** | Raw samples retained on every cell; summary-only receipts rejected | strip the samples array |
| **PP-8** | Comparator `http_concurrency` **equals** the band's `c` | pin the comparator at 1, run band 16 |
| **PP-9** | A cell, once run **at a given commit**, is spent **at that commit**; it may not be re-run to green there. A **different** commit is a different cell run and starts a new ledger row — without that qualifier a single non-conformant run would make the cell unreachable forever, which is a state with no legal move rather than a discipline. Spent cells are recorded in [`evidence/parity/LEDGER.md`](../../evidence/parity/LEDGER.md), append-only, keyed on **(host, workload, model, quantization, commit)** | re-run a failed cell **at the same commit** and publish the second; separately, re-run at a *later* commit and require that it is accepted as a new row |
| **PP-10** | No request is issued at or after window close; pre-close requests are drained; `drain_ms` recorded | issue one request after close and count its tokens |
| **PP-11** | `tokenization.method` has no default; absence is schema-fatal | omit the block |
| **PP-12** | No comparator ratio is published outside a receipt, **and no `[X]` vendor-spec figure is published as a claim.** A vendor bandwidth or TFLOP number informs design and may appear tagged `[X]`; it may not appear as a measured value or feed a published ratio | print a ratio to stdout; separately, publish a vendor bandwidth figure untagged |
| **PP-13** | `max_in_flight` is reported by the **server**, never inferred by the harness | have the harness compute it |
| **PP-14** | Auto-fit never modifies an explicitly-set argument | set `--gpu-layers 12`, have auto-fit raise it |
| **PP-15** | **No boolean accelerator flag.** Every accelerator request is a quantity or a device list, and each has a reported resolution | add `--gpu` with no `gpu_layers_resolved` |
| **PP-16** | **`receipt.provenance.compute_class` must equal `perf-matrix.yaml[host].compute_class`, and the binary must be able to reach that class.** A declared class no build can produce is schema-fatal | declare `metal` where no Metal path exists (#2841) |
| **PP-17** | **A parity claim names its band.** A band-less ratio is schema-fatal | publish "apr is at parity" with no band |
| **PP-18** | **The measuring binary is built from a commit that is an ancestor of `HEAD`, and its sha256 is in the receipt** | measure with a `+no-git` build |
| **PP-25** | **One client binary drives both lanes, and its sha256 is in the receipt.** Re-adopted from v2.2 I-15, which an earlier draft demoted to §5 prose — a rule that lives only in prose is not an invariant and has no mutation | drive the subject with the harness client and the comparator with `curl`; the receipt must refuse |

### §6.0a Why PP-19…PP-22 are re-adoptions, not additions

An external audit found this document had claimed to carry v2.2's invariants forward while
**dropping four of them and reusing their IDs for different rules**. v2.2's I-7 (isolation),
I-8 (pin expiry), I-10 (receipt-commit binding) and I-11 (join-key refusal) each shipped with a
named mutation, and each vanished without a `decided_by`. They are restored above.

**The isolation one is not hypothetical.** §12.1a records gx10 c=8 with a 21.17% MDE, traced in
#2833 to a device-wide stall — which is exactly the failure v2.2's I-7 existed to prevent.

### §6.1 PP-15 and PP-16 are live defects today, not hypotheticals

```
$ git show origin/main:scripts/llama_pin.toml | grep apr_serve_command
apr_serve_command = "apr serve run {model} --gpu --port {port} ..."
```

`--gpu` is the boolean flag PP-15 forbids, in the file that drives every parity measurement
this project takes. **PERF-021 was meant to retire it and the harness still uses it.**

PP-16 is #2841: `perf-matrix.yaml` declares host `mini` as `compute_class: metal`, and `apr`
has no Metal inference path — `aprender-serve` pins `aprender-gpu` to `features = ["cuda"]`,
nothing enables `aprender-gpu/metal`, its `Backend` trait has no compute method, and there is
no Q4_K kernel in any form. A `mini` cell run today records a CPU run as `metal`, permanently
under PP-9.

| **PP-19** | **One global CI concurrency group for the perf gate, `cancel-in-progress: false`, shared with any job contending the same host.** Re-adopted from v2.2 I-7, which this document had dropped | run two perf jobs on one host concurrently; the second must queue, not start |
| **PP-20** | **The comparator pin carries an expiry and annotates when stale.** Re-adopted from v2.2 I-8 | set the pin's expiry in the past; every ratio it produces must be annotated stale |
| **PP-21** | **The receipt's signature is valid AND `receipt.commit ⊇ commit-under-test`.** Re-adopted from v2.2 I-10; PP-18's ancestor rule is the weaker half of this and does not replace it | sign a receipt for a commit the PR does not contain |
| **PP-22** | **A join-key mismatch — host, workload, band, quantization, `tokenization` — refuses the ratio rather than computing one.** Re-adopted from v2.2 I-11 | join a c=4 subject against a c=16 comparator; the ratio must be refused, not produced |
| **PP-23** | **`roofline_tok_per_sec` is recorded, computed from a MEASURED device bandwidth and the model's byte size, and compared ONLY against PER-SEQUENCE decode — never against aggregate** (§6.1b′). A **decode** rate above the ceiling is schema-fatal (a harness bug). No threshold is attached: a reading far below the ceiling is annotated with its ratio and reported, and `SUSPECT_DISPATCH` is raised by a **named mechanism from a profile** (§12.8), never by a number in this table | record a *decode* rate above the ceiling; separately, record gx10's c=8 **aggregate** of 84.417 tok/s against a ~58 tok/s ceiling and require that it does NOT fire |
| **PP-24** | **`server.max_in_flight ≥ c` on BOTH lanes, server-reported.** A transient mismatch is `UNMEASURED` reason `admission_capped`, naming which server capped and by how much; a *deliberate, server-reported* capacity limit is `NOT_APPLICABLE` with `decided_by` — never a band the device cannot hold expiring into `FAIL` (§6.1c) | run c=16 against a subject admitting 11; the band must refuse, not average — and a subject whose KV budget reports a ceiling of 11 must yield NOT_APPLICABLE, not a permanent UNMEASURED |

### §6.1a Three findings from the final review round

**(i) The counter-measurement that withdrew §2.1 is itself under-provenanced — but not in the way
an earlier draft said.** That draft asserted `max_batch=11` "came from the `CUDA_MAX_BATCH` env
var" and "appears **nowhere**" outside `server-startup.txt`. Both halves are wrong, and a review
caught it. The value is **auto-sized in Rust from free VRAM at load**, and `findings.json` — in
the same evidence directory — records it explicitly: *"The server auto-sized continuous batching
to max_batch=11 at load time. Pre-flight on the same host got max_batch=12; **the value falls out
of free VRAM**, which varies with what the desktop session is displaying."*

**The obligation survives; its subject changes.** The unrecorded quantity is not an env var — it
is the **free VRAM at load** that the value is derived from, which no receipt carries and which
varies with what else the machine is doing. That is a *harder* reproducibility problem than a
missing env var, not a softer one: a receipt could pin an env var, and it cannot pin a desktop
session. §12.6's endpoint must therefore report the derived `max_batch` **and its inputs**. Batching demonstrably engaged and aggregate demonstrably scaled, so the
withdrawal of §2.1 stands — but **that artifact is not reproducible from its own receipt**, and it
may not become a baseline until it is. At least ten env vars change the decode path
(`CUDA_MAX_BATCH`, `CUDA_BATCH_WINDOW_MS`, `ITERATION_SCHEDULER`, `GRAPH_DISPATCH`,
`BATCHED_GRAPH`, `FP8_DECODE`, `CUBLAS_GEMM_THRESHOLD`, `APR_STREAM_NONBLOCKING`,
`STAGGERED_PREFILL`, `APR_FORCE_BATCHED_PATH`) and **none is recorded in any receipt**. PP-2 must
extend to the scheduler configuration, or every receipt is a run of an unnamed configuration.

**(ii) "Uncap `max_batch = 4`" fixes a limit that is not the limit.** Both execution plans
proposed it. The active cap in the measured run was 11, set by an env var. Editing the Rust
default would not have moved it, and forcing 16 risks OOM if 11 was chosen for KV-cache headroom
on a 24 GB card.

**(iii) The residual gap is probably not a kernel problem at all.** Above `m ≥ 4` the batched
path routes to **cuBLAS GEMM** — NVIDIA's own kernels. If the aggregate deficit persists there,
it is host-side: admission, tokenization, transport, or KV-cache latency. **Scoping kernel work
before profiling would misdiagnose the bottleneck**. §9 #3b's `nsys` profile (#2697) has since
answered it with numbers: `cuLaunchKernel` is 0.7% of CUDA API time and synchronous copies plus
device allocations are 93.5%, so the prefill fixed cost is host-side. (An earlier draft added
that §9 #4 was discharged — it is not; see #4.) The coalescing claim itself stands (ten
coalesced/DP4A GEMV variants ship in `crates/aprender-gpu/src/kernels/quantize/q4k/coalesced/`,
and the batched path calls no GEMV).

### §6.1b PP-8 is violated by construction, not by omission

`scripts/llama_pin.toml:256` starts the comparator as

```
{llama_server} -m {model} --port {port} -ngl 999 -c {context_length} -t {threads} --no-warmup
```

with **no `-np` / `--parallel`**, and `scripts/parity_host_receipt.sh` starts that server
**once, outside the `for c in $BANDS` loop**. So the comparator's slot count is fixed for the
whole sweep while apr's concurrency varies 1 → 16. PP-8 requires the comparator's
`http_concurrency` to equal the band's `c`; today it cannot, because nothing varies it.

This is worse than the boolean-`--gpu` violation above: `--gpu` resolves to a defensible value,
whereas a comparator pinned at one slot count across four bands makes three of the four ratios a
comparison between different configurations. **§12.3 owes the check of whether llama.cpp can be
driven per-band by our client at all**, and until that is known PP-8 is aspirational.

**Correction from a later review: PP-8 is already met, and the live violation is PP-24.**
§6.1c defines PP-8 as the *client-side offered load*, and `parity_host_receipt.sh:108-170` does
vary `--concurrency "$c"` per band **on both lanes**. What is fixed across the sweep is the
comparator's **server slot count** — PP-24's surface, not PP-8's. Calling PP-8 "aspirational" and
hanging §12.3 on it mis-named the defect: the real one is that llama.cpp auto-selects 4 slots and
therefore *admits* 4 at c=8 and c=16, which `llama_pin.toml:129-165` keeps deliberately and with
an argued rationale. §12.3 is that decision, not this feasibility question.

### §6.1b′ PP-23 was stated against the wrong metric, and it fired on correct behaviour

`bandwidth ÷ bytes-per-token` is a **per-sequence decode** ceiling. Under continuous batching
`N` sequences share **one** weight read per decode step, so aggregate throughput scales with the
batch and legitimately reaches ~`N ×` that ceiling — which is the entire point of batching. An
earlier draft wrote PP-23 as "above roofline is schema-fatal (a harness bug)" without naming a
metric, and since decode is not captured (§6.2b) **aggregate is the only rate it could apply to**.

Against this document's own committed receipts:

| host | band | aggregate | ceiling | ratio | PP-23 as first written |
|---|---|---|---|---|---|
| gx10 | c=1 | 6.199 | ~58 | 0.11 | (below — the §9 #1 finding) |
| gx10 | c=8 | 84.417 | ~58 | **1.45** | **schema-fatal** |
| gx10 | c=16 | 163.995 | ~58 | **2.81** | **schema-fatal** |

So the rule declared a correctly-batching server a harness bug on the two bands that carry the
parity claim. It is now stated on decode only — which means **it has no applicable input today**,
and that is the honest position rather than one it can apply wrongly.

**The 25% literal is deleted with it.** Its defence was that no verdict turns on it, and that
does not survive: blocking a cell from `MEASURED` makes it `UNMEASURED`, §12 turns `UNMEASURED`
into `FAIL` at expiry, and a literal that fails a release is a threshold whatever it is called.
Gating *admissibility* is strictly stronger than gating a verdict, because it decides what may be
counted at all. Both of its terms were unmeasured besides — the denominator a `[X]` vendor
bandwidth PP-12 forbids publishing, the numerator aggregate rather than decode. §9 #1 stands
because at c=1 exactly one sequence is in flight, so aggregate there *is* a per-sequence rate; it
does not generalise one band to the right.

### §6.1c PP-24 — equal admission, and why an unequal band is not a slow band

The three concurrency rules act on three different surfaces and are easy to read as one. PP-22
checks the **declared** band — the join key both receipts carry. PP-8 checks the **offered load**:
the comparator's client-side `http_concurrency` must equal the band's `c`. PP-24 checks the
**served capacity**: how many of those requests the server actually admits at once. A run can
satisfy the first two and violate the third: a server accepts a `-np 16` configuration and still
admits only 11 requests at once. The withdrawn W1 run did exactly that — the subject's own log
printed `max_batch=11` while the band was driven at c=16.

That is not "apr is slower at c=16". Five of the sixteen requests were **queued outside the
measured window's concurrency**, so the band measured a c=11 server against a c=16 comparator
and divided. The result is a join-key violation dressed as a ratio, which is why PP-22 and
PP-24 are separate rules: PP-22 catches a *declared* mismatch, PP-24 catches an *effective* one.

**The remedy is refusal, not correction** — and, in one case below, refusal of the *question*.
A band whose lanes admit unequally is `UNMEASURED` with reason `admission_capped`, naming which
server capped and by how much. Rescaling the
subject's aggregate by 16/11 would be arithmetic on a number the harness never measured;
re-running at c=11 would be a different band. Both are answers to a question the run cannot
answer, and §7's three-valued status exists precisely so that "we do not know" is sayable.

**A band the hardware refuses is `NOT_APPLICABLE`, never a permanent `UNMEASURED`.** As first
written this rule was unpassable: if `apr serve` caps admission at 11 because the KV budget on a
24 GB card cannot hold sixteen sequences, then c=16 is `admission_capped` forever, the expiry
rule in §12 turns that into `FAIL`, and the cell blocks every release with no legal move
available — the exact dead state §7's three-valued status exists to avoid.

So the two cases are separated by **who decided**:

| the cap is | status | what it means |
|---|---|---|
| a *transient* mismatch — one lane misconfigured, a stale pin, a flag not forwarded | `UNMEASURED`, reason `admission_capped` | fix the configuration and re-run; the band is still owed |
| a *deliberate, server-reported* capacity limit — the KV budget for this model on this device does not reach `c` | `NOT_APPLICABLE`, with `decided_by` and the reported budget | the band is not owed on this cell, and never expires into `FAIL` |

**The band ladder is therefore derived, not declared.** A cell's bands are `{c ∈ ladder : c ≤
min(admission capacity of both lanes)}`, read from the two servers rather than typed into
`perf-matrix.yaml`, and a cell that admits 11 is measured at c ∈ {1, 4, 8} and reports its
ceiling. A ladder that demands a concurrency the device cannot hold is not a stricter
specification; it is one that cannot be satisfied, and this document has already shipped two of
those.

Its producer is §12.6's server-reported effective-config endpoint — the same one PP-2 needs —
which is why 12.6 sits at order 3 in §12's chain and PP-24 cannot be armed before it.

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

**And the blocker is the TYPE, not the function body.** `crates/aprender-test-lib/src/perf_gate/drain.rs:142`
declares `ComparatorStatus` with exactly two variants — `NotApplicable` and `Unmeasured` — and
`wire_token()` at `:163` maps only those two. There is **no `Measured` variant anywhere in the
tree**, so a comparator ratio is not *representable* in this receipt schema, never mind
unproduced.

**But "§7's gate has no producer" — the consequence an earlier draft drew from this — is FALSE,
and it was the sentence the whole §12 chain was built on.** A comparator-ratio producer exists
in-tree and runs today. `scripts/parity_host_receipt.sh` drives **both** servers with the same
`$APR` client and varies `--concurrency` per band **on both lanes**; `scripts/lib/perf_receipt.py`
emits `agg_ratio` and `decode_ratio` in `perf_gate.sh`'s own schema. Run against the committed
artifacts it reproduces §2.1's withdrawn table to four decimals:

```
$ python3 scripts/lib/perf_receipt.py --from-bands evidence/parity-http/bands \
      --subject apr --comparator llamacpp --derive-only
  c   agg_tok_s  scaling_eff   agg_ratio  decode_ratio   cmp completed
  1       90.19       1.0000      0.5341        0.5873           77/80
  4      111.87       0.3101      0.2308        0.9231         220/223
  8      109.63       0.1520      0.1685        1.3525         296/302
 16      108.37       0.0751      0.0967        1.5540         512/522
```

**What is missing is the JOIN, not the lane.** The comparator lane is bolted to the *legacy*
`apr test llm bench` (duration-terminated, no drain phase, errors collapsed into `failed`, no
`tokenization` block), so a receipt built from it fails Arm C on `timeouts`, `tokenization.method`
and `drain_ms`. The conformant `--band` producer passes merge (`gate-merge-r1.txt`) and has no
comparator. **Neither producer alone can clear the gate, and both halves already exist.** Order 1
is putting the comparator into the conformant producer — a `Measured { baseline }` variant, the
paired derivation, the wire mapping, a second `LlmClient` inside the band loop — which is
plumbing between two working things, not a lane to build.

**(b) `decode_tok_per_sec` is absent from every band of every committed receipt.**

```
$ python3 -c "...evidence/perf-gate-001-w1-lambda/receipt.r1.json..."
[(1,'UNMEASURED','<ABSENT>'), (4,'UNMEASURED','<ABSENT>'),
 (8,'UNMEASURED','<ABSENT>'), (16,'UNMEASURED','<ABSENT>')]
```

Both cells, both hosts, all replicates, all four bands. `token_times_s` is `[]` on every sample
row, so per-token decode is not merely unaggregated — it is **not captured**.

**And "not captured at all" is the wrong cause and overstates the gap.** The receipt names the
real one ten times in `unproduced_fields`: *"the transport did not stream, so the client never
observed a first-token instant"*, *"…so there are no per-token arrival times to pool"*.

**The capture path is complete and unit-tested.** `llm/client.rs:429-533` collects per-token
arrival times, `perf_gate/drain.rs:114` computes per-request decode, `:330` medians it,
`receipt.rs:591-603` emits it when present, and `test_llm_band.rs:402-406` prints a NOTE saying
decode is omitted *because* `--stream` was absent. `llama_pin.toml:257` already carries `--stream`
in its harness command; `evidence/perf-gate-001-w1-lambda/invocation.txt` contains it **zero
times**. So this is a **flag on an existing code path**, not a deliverable — and §5's
`streaming \| required` row was violated by the run this whole document is built on, with
`perf_gate.sh` returning `VERDICT PASS` and `findings.json` containing the string `stream` zero
times.

**Consequence:** the §7 gate is a specification of a check nothing can currently feed. **The
comparator lane and decode capture are the first deliverable**, before any optimisation work,
and they need no matrix run. The numbers in §2.1 come from `llama_pin.toml`'s standalone
harness, not from this producer — which is why they exist at all.

**(c) `perf-matrix.yaml` already encodes the rule §2.2 forbids — and a second one this document
had not noticed.** `scripts/perf-matrix.yaml:56-63` declares Arm **B2 floor = 1.00 on every
band** and `perf_gate.sh:246` implements it: exactly the "decode ≥ 1.0 everywhere" rule that
would reject the continuous-batching PR (§2.2). B2 also carries
`inherited_from: docs/specifications/perf-parity-spec.md` — **one of the fifteen documents §11
archives**, so the live gate's authority is an archived spec.

**And there are THREE encodings, not two.** `scripts/lib/parity_block.py:23-24` sets
`STRETCH = 1.50` and `CEILING = 1.50` — *"a ratio above this is likelier a measurement error
than a win"* — and applies the floor and ceiling itself. So the withdrawn run's **1.554 decode
ratio at c=16** would be recorded `FAIL` by the ceiling, a number §7 explicitly designates
`REPORTED`. A promise that "the gate PR moves both in one commit" covers two files when there
are three, and the third can fail a receipt for being *too fast*.

**Arm B1 is the one an earlier draft missed.** `:45-55` sets `floor: 0.80` with
`threshold_class: policy` on `agg_ratio(c) = agg(c) ÷ comparator_agg(c)` — a literal ratio
threshold on the **comparator ratio**, which is precisely and only what P-1 bans. Naming B2 and
not B1 read as a complete inventory of the live P-1 violations and was not one.

**§7 supersedes both — but not yet.** Until the comparator lane exists, **§7 is DESIGNED, NOT
ARMED**: it gates nothing, `perf-matrix.yaml` remains the live rule, and the two do not silently
disagree because only one of them runs.

**The gate lands in REPORTING mode first, then the matrix flips in a separate one-line commit.**
An earlier draft said "the gate PR moves both in one commit or neither", which manufactured the
highest-risk pull request in the project and put it at order zero: one commit adding a
`Measured` variant to a public enum, the PP-3 baseline object, the comparator lane, one client
driving both lanes (PP-25), the streaming fix, a seeded 10,000-resample paired bootstrap,
`perf_gate.sh`'s Arm B verdict logic in both directions, and `perf-matrix.yaml` — all behind the
required `workspace-test`. The invariant that mattered was *never two live disagreeing rules*,
and REPORTING-then-flip preserves it exactly: at every instant only one of them produces a
verdict.

**(d) The measurement client omits `seed` and `ignore_eos` from the wire.**
`crates/aprender-test-lib/src/llm/client.rs:436-448` rebuilds the request body with
`seed: None, ignore_eos: None`, and both fields are `#[serde(skip_serializing_if)]`, so they are
**omitted rather than nulled**. Every number in §2.1 was therefore taken with unseeded sampling
and no output-length pin, on both lanes. It does not invalidate an aggregate-throughput
comparison, but it bounds what any *determinism* or *per-token* claim from this harness can mean.

---

## §7 The gate

**Merge phase** — integrity, plus one *deterministic* speed check. No comparator ratio and no
HTTP timing (PP-6): a wall-clock ratio on a shared runner produces red PRs indistinguishable
from real regressions, and the team routes around them.

- receipt schema valid; every §6 invariant checkable statically
- **no HTTP timing assertion of any kind**
- **§7.1's kernel microbenchmark — `DESIGNED, NOT ARMED`, and listed here as the shape the merge
  phase will take, not as something it does today.** An earlier draft listed it among what the
  merge phase gates while §7.1 said "nothing is gated by §7.1 today", so §7 asserted a gate its
  own subsection denied. Of these three bullets exactly two are live rules; `gate-merge-r1.txt`
  runs Arms A–E and no microbenchmark step

**Release phase** — the parity claim, asymmetric per §2.2:

| band | metric | status | against |
|---|---|---|---|
| c=1 | `dec_ratio` | **gated** — REPORTING until decode σ exists (§12.1) | the comparator, same run |
| c=1 | `agg_ratio` | **REPORTED** — at c=1 the two still differ (§2.1's withdrawn run read 0.534 vs 0.587, because aggregate includes prefill and queue time while decode does not). Gating both here would re-create the two-metric trap §2.2 removes, so decode is the gated one and aggregate is recorded beside it | — |
| c ∈ {4,8,16} | `agg_ratio` | **gated** | the comparator, same run |
| c ∈ {4,8,16} | `dec_ratio` | **REPORTED** until `agg_ratio ≥ 1.0` is first achieved; a non-regression floor is set from **that same receipt** thereafter | apr's own post-batching baseline |

**The trigger is `agg_ratio ≥ 1.0`, in both §2.2 and here** — not "when batching lands", which
is not an observable event. The receipt that first achieves it *is* the post-batching baseline.

**The c=1 decode gate is a REGRESSION gate, not a parity floor, and the review round that caught
this said the trap had merely moved.** The objection: continuous batching can add a fixed
per-request cost that lowers c=1 decode while c=16 aggregate reaches parity — so an asymmetric
gate with a c=1 *parity* floor would reject the batching PR at c=1 having been designed
precisely to stop rejecting it at c>1. That is the same defect one band to the left, and it
would be the third time.

So at c=1 the gated comparison is against **apr's own previous release, outside the receipt's
measured MDE** (§12.1a). The *parity* claim at c=1 stays REPORTING until §12.2 sizes single-
stream against a conformant comparator lane. A change may lower c=1 decode within the MDE and
merge; a change that lowers it beyond the MDE must say so and be argued, which is what a gate
is for. Nothing here may arm before §12.1 — a gate on a metric with no captured σ is a coin
flip, and this document has already shipped one of those.

Plus: every cell in `perf-matrix.yaml` present, or `UNMEASURED` with owner and expiry.

### §7.1 The kernel microbenchmark gate — and why it is the cheap one

**Its target kernel is named by a profile, never inherited.** §2.4's M=1 GEMV mechanism is
**partly discharged** (§9 #4): the tree does ship ten coalesced/DP4A Q4_K GEMV variants, and above
`m ≥ 4` the batched path routes to cuBLAS GEMM and calls no GEMV at all. A gate on `q4k_gemv`
would therefore protect a kernel the optimised path does not execute — the prior epic's failure
mode under a new name.

So the ordering is: **profile first, then gate what the profile names.** Until a profile exists,
§7.1 specifies a *shape*, not a target, and nothing is gated by it.

**Status: DESIGNED, NOT ARMED — but its target kernel is no longer unknown.** An earlier draft
said in one paragraph that §7.1 "specifies a shape, not a target, and nothing is gated by it"
and in the next that "every PR is gated on an in-process kernel benchmark — M=1 GEMV". Both
cannot be the rule. **Nothing is gated by §7.1 today**, and that stands.

What does not stand is the reason given for it. That draft said the target kernel was "an output
of the profile obligation in §12", on the strength of §9 #4's discharge — and §9 #4's discharge
was wrong. The decode path runs under CUDA graph capture, `use_cublas` is `&& !self.is_capturing`,
so **batched decode calls GEMV, not cuBLAS**, and `cublas_prefill/mod.rs:24-38` states the
deficiency in the team's own words: single warp at `M ≤ 8`, multi-warp specializations only at
`M = 16/32`, `TODO: Add M=4 multi-warp kernel`.

**§7.1's first benchmark is therefore batched Q4_K GEMV at `M ∈ {4, 8}`**, against the existing
`M = 16/32` multi-warp path as its own control. That needs no profile, no comparator and no
matrix run — it is `cargo bench` on one host — which makes it the cheapest arming path in this
document and the reason §12.8 is off the chain rather than behind three orders of plumbing.

When it is armed it will be in-process, deterministic, socket-free, and ratchet-down against a
committed self-baseline — which PP-6 permits because it is not a comparator ratio. `[U]` — its
own variance is unmeasured and inherits §12.1's `n ≥ 5` rule before it may ratchet.

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

| cell | host | model | parity status (§7 vocabulary) | **parity**-gated? |
|---|---|---|---|---|
| **REFERENCE** | `lambda` RTX 4090, CUDA | Qwen2.5-Coder-7B Q4_K_M | `UNMEASURED` — owner `perf-gate`, expires **2026-09-25** | **yes**, once a conformant receipt exists |
| gx10 (GB10, aarch64) | CUDA | same | `UNMEASURED` — owner `perf-gate`, expires **2026-09-25** | no |
| intel (`mac-server`) | CPU / wgpu | same | `UNMEASURED` — owner `perf-gate`, expires **2026-09-25** | no |
| mini (M4) | — | same | `UNMEASURED` — owner `perf-gate`, expires **2026-09-25**, **blocked on #2841** (PP-16) | no |

Every cell is `UNMEASURED` today; **none is `MEASURED`, and none is `NOT_APPLICABLE`.** "Gated"
is a separate column from status precisely so the two vocabularies do not blur — a cell can be
the gated one and still be UNMEASURED, which is exactly today's state.

**"Parity-gated" is not the only gate they carry.** Every cell in this table that *functions*
is also covered by §7.2's **non-regression** check, which is a different decision: parity is
gated narrowly on one cell, regression is gated broadly on all of them. A `no` in the column
above means "makes no parity claim", never "is unguarded".

**§7.2's arm is `scaling_efficiency(c)` PAIRED WITH A FLOOR ON `agg(1)` — never `scaling_efficiency`
alone.** §3 defines it, an earlier draft then used it nowhere, and re-adopting it bare was a
defect a review caught: it needs no comparator, which is why it was chosen, and that is also why
nothing notices when it is gamed.

`scaling_efficiency(c) = (agg(c) ÷ agg(1)) ÷ c` has `agg(1)` in its **denominator** and
`scripts/perf-matrix.yaml:38-44` ratchets it `up-only`. **Any change that makes single-stream
slower raises the score at every band.** Lambda's measured 0.2814 at c=16 becomes 0.563 by
halving `agg(1)` from 100.6 to 50 — the server strictly worse for every individual user, and the
ratchet records an improvement. The one speed metric this document gates broadly would have
rewarded regressing §9 #3, the gap §2.2 and §9 both refuse to concede.

So the arm is **both terms or neither**: `scaling_efficiency(c)` up-only, *and* `agg(1)`
non-regression against the same prior release outside that receipt's MDE. Neither is sufficient;
a PR must satisfy both. It still needs no comparator — a cell's scaling and its own single-stream
figure both come from one host, with no second server to configure.

**A second host may be promoted to parity-gating only after the reference cell reaches parity.**
Breadth before depth is what produced eight empty cells.

---

## §9 What is known to be wrong, in priority order

| # | finding | evidence | status |
|---|---|---|---|
| **1** | **gx10 is at most 10.6% of its own memory roofline.** The 6.203 tok/s is the **aggregate** figure at c=1 — decode is not captured at all (§6.2b), so this is the only number that exists. At c=1 aggregate carries prefill and queue time that decode does not, so true decode is *higher* and **10.6% is a floor on the decode ratio, not the decode ratio**. Even as a floor it is less than half of PP-23's 25%, and that a physical-plausibility check must be stated on the wrong metric is itself the argument for §12.1 being step zero. Measured against a ~58 tok/s ceiling (GB10: 4.68 GB of Q4_K weights read per token at ~273 GB/s `[X]` vendor bandwidth — the invariant requires a **measured** one before this may be published). This is the only *live, sized, comparator-free* finding in this table — it needs no second server and no ratio to be true, and a ~9× headroom on the aarch64 host is the same order as the 10.34× §2.1 claimed and withdrew — except this one is a single-host reading, not a ratio between two servers | #2846; PP-23 | **OPEN — `SUSPECT_DISPATCH`.** §12.8 owes the profile |
| **2** | ~~apr does not batch~~ — **WITHDRAWN.** It batches on `main` (`max_batch=11`); the claim came from a build with the feature compiled out | §2.1 banner | the whole premise, retracted |
| **3** | **Single-stream decode: 0.650× — MEASURED, and this document had it as `UNMEASURED`.** #2694, open, measured 2026-08-24 by `apr test llm bench` with **one client driving both servers, streaming, c=1**: decode **103.26** vs llama.cpp **158.90** tok/s, ITL p50 9.68 vs 6.29 ms, same GGUF, same 4090. Receipt `evidence/parity-http/findings.json`. An earlier draft wrote that single-stream was "neither sized nor conceded" and that §12.2 owed the measurement — while a receipted paired measurement of exactly the shape §12.2 describes sat open in the tracker | **#2694** | **OPEN and SIZED.** §12.2 now owes *conformance*, not existence |
| **3a** | **Prefill is 0.275× — also measured, also open.** #2693: **2,860** vs **10,399** tok/s, TTFT p50 **35.66** vs **9.81** ms, same run as #2694. Prefill is the larger single-stream gap by a wide margin and this document did not mention it | **#2693** | **OPEN.** §10's two-term decomposition exists to separate this from decode |
| **3b** | **The prefill cost is HOST-SIDE, and that is measured too.** #2697's `nsys` profile: `cuLaunchKernel` is **0.7%** of CUDA API time (2,650 calls, 5.5 ms) while `cuStreamSynchronize` (57.4%), `cuMemcpyHtoD_v2` (32.8%, 1,018 synchronous copies) and `cuMemAlloc_v2` (3.3%, 904 device allocs) are **93.5%**. §6.1a(iii) advanced this as a *hypothesis*; #2697 had already measured it | **#2697** | **OPEN.** This is §12.8's profile, already taken |
| **4** | **The kernel lever has a named live mechanism after all — RE-OPENED.** The 192× non-coalesced defect is genuinely absent (ten coalesced/DP4A variants ship). But *"above m≥4 the batched path routes to cuBLAS GEMM and calls no GEMV"* is **false on the decode path**: `use_cublas = m >= 4 && (Q4K\|Q6K) && CUBLAS_PREFILL != "0" && !self.is_capturing`, and decode runs **under CUDA graph capture** (`flash_decoding_graphed.rs`, `core.rs:94` "lazily created on first graph capture"). Under capture cuBLAS is never used and batched GEMV is. The deficiency is then named **in the tree, by the team**: *"Batched Q4K GEMV at M≤8 uses single warp (32 threads/block) — insufficient parallelism. Multi-warp specializations only exist for M=16/32. TODO: Add M=4 multi-warp kernel"* | `cublas_prefill/attention.rs:1089`; `cublas_prefill/mod.rs:24-38` | **OPEN, and it is §7.1's target kernel** |
| **5** | **P0 — batched CUDA decode emits garbage for every `m > 1`.** #2753: slots served from a batch emit a constant token to the cap, never emit a stop token, and always run to `max_tokens`, with `[PMAT-044] Batch m=3 done` in the log proving the batched path engaged. **PERF-001's 3.32× aggregate is therefore throughput of garbage tokens** | **#2753** (fix in flight: #2809) | **OPEN, P0.** See below — it bears on §2.1 |
| **6** | The harness uses a boolean `--gpu` | §6.1 | violates PP-15 |
| **7** | **`cuda-batch = ["cuda"]` — the implication still points the wrong way at `HEAD`,** so §2.1's banner reads as fixed and is not. `crates/apr-cli/Cargo.toml:90` is unchanged; the repair moved one consumer's `cfg` to `feature = "cuda"` while two live sites still branch on `cuda-batch`, and under plain `--features cuda` the `.apr` Q4K GPU serve path compiles to a hard-error stub | `crates/apr-cli/Cargo.toml:89-90`; `handlers_include_01.rs:95,:148` | **OPEN.** The defect that produced the §2.1 withdrawal is partially live |
| **8** | `mini` declares a backend that does not exist | #2841 | violates PP-16 |

**#5 bears on §2.1's withdrawal and does not undo it.** §2.1 was withdrawn because the measured
binary had batching compiled out. #2753 says the batching that replaced it *emits garbage above
`m = 1`*. Both can be true and both are: the withdrawal stands (the old number described a
serialising build), and the aggregate figure that replaced it is not yet a figure about **correct
output**. **No aggregate claim at `c > 1` may be published until #2753 closes** — a throughput
number over garbage tokens is not a throughput number, and this document would otherwise have
inherited the same defect one build later.

**These four issues were all open, all measured, and none appeared in a document claiming to be
the only spec governing inference performance and to enumerate what is known to be wrong.** That
is the §11 failure — a live figure outside the spec that contradicts it — occurring inside the
spec's own repository, and it is why §11's "the live overlap after this PR is one" is not yet true.

**#2 and #3 are different subsystems** — scheduler and kernels. They are worked one at a time,
because concurrent work on both confounds every run.

---

## §10 What this document does not do

- **It does not set a latency threshold.** TTFT/ITL bounds are `UNMEASURED` (P-4).
- **It does not define a *summing* attribution identity, and that refusal is now measured.**
  A proposal required every receipt to satisfy
  `wall_clock == t_prefill + t_decode_kernel + … + t_residual`, with a non-summing decomposition
  schema-fatal. Run against the six committed W1 receipts, the sum of per-request samples over
  each band's own span is **the concurrency, to within the tail** (c=16 reads 15.530, not 16.000, because the last requests finish inside the window):

  | band | Σ samples_ms | span_ms | ratio |
  |---|---|---|---|
  | c=1  | 60,422.3  | 60,422.3 | **1.000** |
  | c=4  | 245,806.9 | 61,451.9 | **4.000** |
  | c=8  | 486,072.2 | 60,759.7 | **8.000** |
  | c=16 | 994,186.0 | 64,017.9 | **15.530** |

  Whole-receipt ratio **7.24** on lambda (r1/r2/r3: 7.24 / 7.22 / 7.23) and **3.36–3.54** on
  gx10. Concurrent requests overlap, so a receipt-level sum over-counts by approximately ×c *by construction* — the shortfall at c=16 is requests completing before the window closes, not a defect in the reading.

  **What is adopted instead:** per-phase **averages** (Σ ÷ n, which is what the ×c cancels to)
  and **device utilization**, neither of which requires a closed identity. Attribution is not
  refused — *summing* is. **The two-term paired decomposition is adopted now, not deferred to a
  future amendment — and an earlier draft stated its meaning WRONG.** That draft said
  `agg_ratio ÷ dec_ratio` at c=1 *is* "the prefill-plus-overhead share, isolated". It is not.
  Expand it:

  ```
  agg_ratio ÷ dec_ratio = (apr_agg/llama_agg) ÷ (apr_dec/llama_dec)
                        = (apr_agg/apr_dec) ÷ (llama_agg/llama_dec)
  ```

  which is **the ratio of the two servers' own overhead fractions**, not either server's
  overhead. If both spend the same *share* of a request outside decode it returns 1.0 while
  saying nothing about how many milliseconds that share is. It answers "is apr's non-decode
  overhead a worse fraction of its own time than llama's is of its?" — a real paired question,
  and the one §9 #3 needs a direction from — but it is not an absolute.

  **The absolute is the per-lane term, and it needs no ratio at all**: `agg ÷ dec` on a single
  lane is that lane's own overhead fraction, available from each server's `result_timings`
  separately. Record both terms. The paired quotient is the comparison; the per-lane terms are
  the measurements, and a decomposition that only ever appears as a quotient cannot tell you
  which lane moved.
- **It does not declare an iteration budget.** A declared budget with a retrospective five-whys
  is not a control; nothing in it can decline work.
- **It does not claim the two levers are causally independent — and the arithmetic they rest on
  is withdrawn with §2.1.** #2844 factored a 10.34× gap into batching (6.07×) and kernel (1.70×).
  **That gap was measured on a build with continuous batching compiled out** (§2.1 banner), so
  both factors and the decomposition are withdrawn; the indicative gap on `main` is ~2.5×. Even
  had the number stood, the closure did not close: `6.07 × 1.70 = 10.32`, not 10.34. It was an
  **identity by construction**, not
  evidence, and its causal reading required per-token efficiency under batching to equal
  efficiency at M=1 — which the tree contradicts, since above `m ≥ 4` the batched path routes to
  cuBLAS GEMM and calls no GEMV at all. **No step in #2844 is scoped from these figures**; the
  first deliverable there is one honest paired measurement.
- **It does not gate a non-CUDA backend** (§8).
- **It does not cover training throughput, LAPACK-bound solvers, or datacenter-scale serving.**
  These are `NOT_APPLICABLE` with `decided_by` recorded, not silently dropped.

---

## §11 Archived by this document

**Fifteen** documents, ~14,800 lines, four repositories. *(Earlier drafts said thirteen; the archives were counted and it is 2 + 3 + 9 + 1 = 15. The number is corrected rather than the tables adjusted to fit it.)* **Nothing is deleted**; superseded material
still readable as current is the condition this document exists to end.

**Their status is not uniform, and the difference matters to this document's central claim.**
Checked 2026-09-01:

| repo | documents | state |
|---|---|---|
| `aprender` | 2 | **archived by this PR**, with 42 `roadmap.yaml` references repointed |
| `qwen-coder-deploy` | 3 | **archived and pushed** (`4fadc7c`) — the only sibling that was a live repo |
| `realizar` | 9 (8 substantive + one 8-line stub) | **the repository is ARCHIVED on GitHub** (read-only). Already superseded by the APR-MONO consolidation into `crates/aprender-serve` |
| `trueno` | 1 | **the repository is ARCHIVED on GitHub** (read-only). Consolidated into `crates/aprender-compute` |

So **ten of the fifteen were never live specs in a writable repository** — they sit in
read-only archives that the monorepo consolidation had already superseded. The genuinely live
overlap was **five documents in two repositories**, and after this PR it is one. Saying
"thirteen live specs" would have overstated the problem in the document written to stop
overstatement.

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

| # | item | owner | expires | why it matters |
|---|---|---|---|---|
| **12.1** | σ for `aggregate_tok_per_sec` is MEASURED (§12.1a). What remains unmeasured is σ for *decode* (not captured at all, §6.2b) and for the comparator lane (no producer, §6.2a) | perf-gate | **2026-10-02** | the comparator lane, not the noise floor, is step zero |
| 12.2 | **A CONFORMANT paired single-stream measurement — the measurement itself already exists.** #2694/#2693 measured decode 0.650× and prefill 0.275× on 2026-08-24, one client, both servers, streaming, c=1, receipted at `evidence/parity-http/findings.json`. What is owed is a receipt that clears §7 — the legacy lane cannot produce `timeouts`, `tokenization.method` or `drain_ms` — plus the model-size question (§2.3) | perf-gate | **2026-10-16** | until it exists, single-stream is neither sized nor conceded |
| 12.3 | **A DECISION, not a feasibility question: what *is* the comparator?** `llama_pin.toml:129-165` keeps `comparator_parallel = "default"` **on purpose**, pinning llama.cpp's auto value (the constant 4 at the pinned commit) and citing a measured falsifier — handicapping the comparator with `-b 1` manufactured a **2.39×** overstatement in apr's favour. So at c=8 and c=16 the comparator serves 4 slots *by design*, and PP-24 makes both bands permanently `admission_capped`. Half the gated bands can then never produce a verdict, and §6.1c forbids both available remedies. Somebody must choose: **llama.cpp as a user runs it** (llama_pin.toml's position, which loses PP-24 above c=4) or **llama.cpp configured to match the band** (PP-8/PP-24's position, which llama_pin.toml has already argued manufactures a flattering result) | **owner: this decision has none yet** | **2026-10-02** | this is a direct collision between an invariant and a documented, evidence-backed harness decision — §6.1b reads the missing `-np` as an omission to fix and §12.3 read it as a question to answer; it is neither |
| 12.4 | Instrumentation overhead of any per-phase timing | perf-gate | **2026-10-16** | unpriced; §10 declines the summing identity partly for this reason |
| 12.5 | `mini`'s backend decision | #2841 | **2026-09-25** | PP-16 blocks the cell until resolved |
| **12.6** | **PP-2's scheduler block has no producer — and the harness must NOT be the producer.** An earlier draft proposed adding ten env-var fields to the harness's `Provenance` struct: that is the harness *inferring* server state, exactly what PP-13 forbids for `max_in_flight`. The server exposes an effective-config endpoint reporting the resolved scheduler configuration plus v2.2 §4.4.9's counters (`admission_rejected`, `preempted_*`, `kv_blocks_*`, `backend_loaded[]`, `autofit_applied[]`), and the harness stores the response verbatim. `max_batch` then becomes a server-reported quantity derived from the KV budget, which discharges §6.1a(ii)'s OOM speculation with a number. `crates/aprender-test-lib/src/perf_gate/receipt.rs`'s `Provenance` struct carries binary, host, accelerator, model, quantization and feature_set — and no scheduler field | perf-gate | **2026-10-09** | every receipt is a run of an unnamed configuration until it does |
| **12.7** | **PP-18 has no automated check.** `grep -ir ancestor crates/aprender-test-lib/src/perf_gate/` returns nothing; the ancestor rule is enforced by hand | perf-gate | **2026-10-16** | a `+no-git` or off-branch build could produce a receipt nothing refuses |
| **12.8** | **The profile §7.1 is waiting on.** §7.1 is `DESIGNED, NOT ARMED` because no profile has named the kernel the microbenchmark would gate; §9 #1's `SUSPECT_DISPATCH` on gx10 is the obvious first subject | perf-gate | **2026-10-16** | a microbenchmark gating an unnamed kernel is a gate on a guess |
| **12.10** | **Five invariants have no producer and, until now, no owner** — PP-19 (there is no perf workflow at all, so no concurrency group), PP-20 (no `expiry` field in `llama_pin.toml`), PP-21 (no `signature` on the receipt; `gate-release-r1.txt` already prints `FAIL ArmC-sig UNSIGNED`), PP-23 (`roofline_tok_per_sec` occurs only in prose), PP-25 (no client sha distinct from the server binary's). Four are §6.0a's re-adoptions: restored with IDs and mutations, and nothing scheduled to build them | perf-gate | **2026-10-09** | §6.0a's own finding was that four controls vanished *without a `decided_by`*. Re-adopting them with no date, owner or producer re-creates it one level milder — and an invariant with no expiry cannot FAIL, which is the immunity §12's expiry rule exists to remove. **PP-19 first**: §12.1a's largest number, gx10's 21.17% MDE at c=8, was traced to a device-wide stall — exactly what I-7 existed to prevent |
| **12.11** | **Nothing in this document owes a tok/s figure.** Every other row here is an instrument obligation. A review constructed a fully legal ten-merge sequence that satisfies §12 end to end, breaks no rule, and moves no token per second — the identical outcome #2706 produced, reachable *through compliance*. The gap-closing work needs a row like any other: an owner, an expiry, and a number | **perf-gate + serve** | **2026-11-13** | a specification that cannot be violated by shipping nothing has not constrained anything. §9 #1 is the first subject because it is the only live sized finding |
| **12.9** | **§5.1's sampler pin is dropped on the STREAMING path only** — narrowed from an earlier draft that claimed both. `client.rs:436-448` rebuilds `stream_request` field-by-field with `seed: None, ignore_eos: None` and both are `skip_serializing_if`, so they are omitted; the **non**-streaming path at `:325` and `:369` uses struct-update (`..request.clone()`) and preserves them, and `prompts.rs:551-563` forwards `seed` from the W1 corpus. So the two committed W1 runs **did** carry a seed — and the moment §12.12 turns streaming on, they stop | perf-gate | **2026-10-02** | the fix must land with the `--stream` flag, or enabling streaming silently unpins the sampler |
| **12.12** | **§5's `streaming \| required` row was violated by the run this document is built on, and nothing noticed.** `receipt.r1.json`'s `unproduced_fields` says it ten times — *"the transport did not stream, so the client never observed a first-token instant"*, *"…so there are no per-token arrival times to pool"*. `perf_gate.sh` passed it (`gate-merge-r1.txt`: VERDICT PASS), `findings.json` contains the string `stream` zero times, and §6.2b attributed the missing decode to **capture** when the receipt attributes it to **transport** — a different cause with a different fix | perf-gate | **2026-09-25** | order 1 was scoped against the wrong cause. Streaming is the only §5 row enforced by nothing: bands by PP-1, replicates by a unit test, tokenization by PP-11, one client by PP-25, the pin by PP-20 — and streaming by no invariant, no mutation and no gate, while violating it produces a receipt that passes |

**The expiry has a stated consequence, and the obligations are sequenced.** An earlier draft
said expiry "is not decoration" and then did not say what happens on the date — while putting
seven obligations and four cells on the *same* date, which is a batch, not a flow.

**On expiry an `UNMEASURED` cell's verdict is `FAIL` at the next release gate**, and
`pmat comply` reports it. The obligations are a dependency chain, not a deadline:

**Nothing below blocks work that needs no comparator.** §12.7 is a `git merge-base --is-ancestor`
call, §12.8 is a profile of one host, §12.11 is engine work, and §12.10's PP-19 is a workflow
`concurrency:` key. None of the four depends on a second server, and an earlier draft put all
four behind three orders of comparator plumbing — including §12.8, which is the input to §7.1,
the one merge-phase speed gate this document gets.

| order | obligation | expires | unblocks |
|---|---|---|---|
| **0** | **pass `--stream`** (§12.12) — a flag, not a deliverable | **2026-09-25** | decode on every band; §12.1's decode σ; §5's `streaming` row |
| **0** | **`kv` block: `test_llm_band.rs:327` hardcodes `kv: None`** | **2026-09-25** | `FAIL ArmD instrumentation absent` — **blocks the release verdict today, with no comparator involved** |
| **0** | **receipt signing deployed** (PP-21) | **2026-09-25** | `FAIL ArmC-sig UNSIGNED` — **the second release blocker that has nothing to do with the comparator** |
| 1 | §12.6 server-reported scheduler config | **2026-10-02** | PP-2; PP-24; §6.1a(ii) — and the admissibility of every run below |
| 1 | §12.5 `mini` backend decision, #2841 | **2026-09-25** | PP-16; the `mini` cell |
| 2 | §6.2 the **JOIN**: comparator into the conformant producer | **2026-10-09** | every ratio. ~350 LOC across four Rust files; both halves exist |
| 2 | **PP-6 re-arm**: `arm_b_adoption` takes no `$phase` | **2026-10-09** | must land *with* the JOIN — see below |
| 2 | §12.3 the comparator-configuration **decision** | **2026-10-09** | PP-8; PP-24; §5.2's argv contract |
| 3 | §12.1 aggregate **and** decode σ at `n ≥ 5` | **2026-10-16** | P-5's ε; §7 may not arm before it |
| 3 | §12.10 the five unowned invariants — **PP-19 first** | **2026-10-16** | PP-19/20/23/25 |
| 4 | §12.2 a **conformant** single-stream receipt (the numbers exist: §9 #3, #3a) | **2026-10-23** | §7-clearing evidence for a gap already sized |
| 4 | §12.4 instrumentation overhead of per-phase timing | **2026-10-23** | §10's averaged attribution |
| — | §12.7 an automated PP-18 ancestor check | **2026-10-02** | *off the chain — a `git merge-base` call* |
| — | §12.8 batched Q4_K GEMV at `M ∈ {4,8}` (§7.1, §9 #4) | **2026-10-02** | *off the chain — `cargo bench`, one host* |
| — | §12.9 `seed`/`ignore_eos` on the **streaming** path | **2026-10-02** | *off the chain — but blocks nothing until order 0 lands* |
| — | §12.11 the tok/s figure | **2026-11-13** | *off the chain — this is the deliverable* |

**Order 0 is the two release blockers and a flag, none of which involve a comparator.**
`gate-release-r1.txt` on the reference cell prints `FAIL ArmC-sig UNSIGNED` and
`FAIL ArmD instrumentation absent`. **Land the comparator tomorrow and the release gate still
fails on both.** An earlier draft put the comparator at order 1 and the KV block at order 3, on
the strength of "§7's gate has no producer" — which §6.2a now retracts. The two things that
actually block a release verdict were scheduled behind the one thing that does not.

**PP-6 is armed by the very PR that discharges the JOIN, and must be fixed in it.**
`perf_gate.sh:392-398`: `run_gate` passes `$phase` to `arm_d_memory` and `arm_e_interference`
and **not** to `arm_b_adoption`, which is called unconditionally. That is dormant only because
every receipt says `comparator_status: UNMEASURED` and Arm B `continue`s. The moment a receipt
carries a comparator ratio, every PR gets a shared-runner comparator wall-clock ratio as a
**blocking merge check** — precisely the failure PP-6 exists to prevent, and the class the
postmortem records the team routing around.

**§12.6 moved to order 0 because PP-9 makes the alternative irreversible.** Order 1 produces
*spendable* cell runs; §12.6 produces the `max_in_flight` PP-24 needs to know whether such a run
was **admissible at all**. Scheduling the run-producer ahead of the admissibility-producer spends
the reference cell on runs nobody can later certify — and PP-9 forbids re-running them at that
commit. §6.1c already said "PP-24 cannot be armed before it"; the chain then ignored its own
sentence.

### §12.0 The failure this chain can still produce, and the row that stops it

An adversarial review was asked to construct a sequence of merges that spends months, breaks no
rule in this document, and lands zero speed. **It constructed one, in ten merges.** The load-
bearing observation is that the three rules which could force optimisation cannot: §7 is
`DESIGNED, NOT ARMED` and gated on a comparator that does not exist; §7.1 is `DESIGNED, NOT
ARMED` and explicitly not a parity claim; §7.2's non-regression is satisfied by changing nothing.
Every row of §12 as first drafted was an instrument obligation. **Not one owed a token per
second.**

That is #2706's outcome reached *through* compliance rather than around it — the postmortem's own
sentence, "twelve pull requests building the measuring instrument and zero speed", describes a
ten-row plan to build the measuring instrument. A specification that cannot be violated by
shipping nothing has not constrained anything.

Two changes answer it. **§12.11 owes a tok/s figure** with an owner and an expiry like any other
row, first subject §9 #1. And the derived-expiry rule below is bounded: it was written so a cell
inherits its blocker's date, which is correct — and it also means slipping order 0 slips every
cell automatically, a deadline-slip mechanism that is not merely legal but silent. **An
obligation's own expiry may be moved only by an amendment that records who moved it and why**,
which is the same `decided_by` discipline §6.0a exists to enforce one level up.

**The chain gates what may be CLAIMED, never what may be INVESTIGATED.** A review round split
on exactly this — one side calling the serialization mandatory because the §2.1 withdrawal is
what optimizing without a trusted instrument looks like, the other calling it a blocker that
pushes real speed months out. Both are right about different things, and the resolution is that
they are talking about different verbs.

Nothing in this chain blocks profiling, `nsys`, a microbenchmark, a kernel rewrite, or a fix
justified entirely by a **single-host** measurement — §9 #1 is comparator-free by construction
and needs no row of this table. What the chain gates is the *ratio*: no comparator claim, no
parity verdict, no published figure until order 1 exists. Work on the engine proceeds in
parallel from today; what it may not do is announce a ratio.

Order 1 is step zero for every ratio: no comparator claim can be made until the lane exists. Each row's
`expires` in the table above is this column, and the two must not be maintained separately —
an earlier draft did exactly that and the table said 2026-09-25 for obligations the chain did
not schedule until 2026-10-16.

**A cell's expiry is derived, never declared.** An audit found the previous shape guaranteed its
own failure: every cell in `perf-matrix.yaml` expired 2026-09-25 while the work required to
measure it was sequenced to 2026-10-16, so the FAIL rule above would have fired on every cell
weeks before the harness that could clear it existed. **A cell inherits the expiry of the latest
obligation blocking it**, and `pmat comply` computes that rather than reading a date someone
typed. A deadline that precedes its own prerequisite is not a control; it is a scheduled outage.

### §12.1a The measured noise floor — CORRECTED, and weaker than first stated

Recomputed from the six committed receipts (`N = 3`, two hosts). **An earlier draft of this
table used `MDE = 2·sd/√n` and was ~3× optimistic**; an external audit reproduced the error and
it is corrected here.

The `k = 2` normal quantile is wrong three times over for the decision §7 makes with it:
at `n = 3` the quantile is `t(0.975, df=2) = 4.30`, not 2; a **paired** comparator ratio has two
noisy lanes, so the paired difference carries a `√2`; and the ratio of two means needs a
delta-method or bootstrap interval, not a single-lane standard error.

| host | band | mean agg tok/s | sd | MDE as first stated | **MDE corrected** |
|---|---|---|---|---|---|
| lambda | c=1  | 100.643 | 0.635 | 0.73% | **2.22%** |
| lambda | c=4  | 191.663 | 0.029 | 0.02% | **0.05%** |
| lambda | c=8  | 353.336 | 0.505 | 0.17% | **0.51%** |
| lambda | c=16 | 450.405 | 0.467 | 0.12% | **0.35%** |
| gx10 | c=1  | 6.203 | 0.005 | 0.09% | **0.26%** |
| gx10 | c=4  | 39.039 | 0.016 | 0.05% | **0.13%** |
| gx10 | c=8  | 76.432 | **14.011** | 21.17% | **64%** |
| gx10 | c=16 | 162.647 | 2.576 | 1.83% | **5.6%** |

**And σ itself is barely estimated.** From `n = 3` the 95% CI on σ is **[0.52×, 6.3×]** of the
point estimate (χ², df=2). So the table cannot support "lambda is quiet enough to gate" as
stated: it supports *lambda's noise is small enough that a 6× effect is unambiguous*, and
nothing finer. An earlier draft said a 6× lever is "four orders of magnitude" above the MDE;
`500% / 0.73% = 686×`, which is **2.8 orders**, and against the corrected 2.22% it is 225×.

**`gx10 c=8` at 64% MDE cannot be gated on anything**, which is the same conclusion the
uncorrected table reached by a wrong route.

**Bootstrap is re-adopted, and it was v2.2's rule.** `APR-PERF-GATE-001-v2.2.md` §4.4.4
specified bootstrap percentile CIs — 10 000 resamples, seed 2026, resampling whole requests —
and this document had replaced it with the hand computation above. The hand computation is
retained only as the *sanity* figure; **the verdict statistic is the bootstrap interval** (P-5).

**No σ-dependent status may change on `n < 5`.** Three replicates size an effect; they do not
bound a variance.

**§12.1's remainder is step zero.** Not because a threshold needs it — every rule in §4 is a paired
comparison against the same run, so no literal is required — but because without σ, a run that
moved nothing and a run that moved everything are the same artifact.

---

## Appendix A — v2.2 `I-nn` → this document

Every invariant of `APR-PERF-GATE-001-v2.2.md` §4.10, and what became of it. A row with no
disposition is how a control disappears.

| v2.2 | content | disposition here |
|---|---|---|
| I-1 | expected cell set enumerated from the matrix | **PP-1**, unchanged |
| I-2 | `compute_class` is the path taken | **PP-2**, extended with the scheduler block |
| I-3 | no ratio without a conformant baseline | **PP-3**, unchanged |
| I-4 | raw samples retained | **PP-7** (the ID moved; the rule did not) |
| I-5 | `timeouts > 0` is fatal to the ratio | **PP-5**, unchanged |
| I-6 | no wall-clock ratio at merge | **PP-6**, narrowed to *comparator* ratios — §7.1's in-process microbenchmark is not one |
| I-7 | one global perf concurrency group | **PP-19**, re-adopted |
| I-8 | comparator pin expiry | **PP-20**, re-adopted |
| I-9 | a cell once run is spent | **PP-9**, unchanged |
| I-10 | signature valid ∧ receipt.commit ⊇ commit-under-test | **PP-21**, re-adopted |
| I-11 | join-key mismatch refuses the ratio | **PP-22**, re-adopted |
| I-12 | docs figures legal iff citing `evidence/`; `[X]` figures illegal | **PP-12**, and the `[X]` ban is restored: a vendor-spec figure informs design and may not be published as a claim |
| I-13 | `tokenization.method` has no default | **PP-11** |
| I-14 | drain at window close, `drain_ms` recorded | **PP-10** |
| I-15 | one client binary drives both servers | **PP-25**, re-adopted. An earlier draft demoted it to §5 prose and credited it to a `PP-25` that did not exist in §6 — a disposition column naming a nonexistent successor is the same defect as no disposition at all |
| I-16 | `max_in_flight` server-reported | **PP-13** |
| I-17 | auto-fit never modifies an explicit argument | **PP-14** |
| I-18 | no boolean accelerator flag | **PP-15** |

New in this document, with no v2.2 ancestor: **PP-4** (both metrics on every band), **PP-8**
(comparator *client* concurrency equals the band's `c`), **PP-16** (the declared `compute_class`
must exist as a code path), **PP-17** (a parity claim names its band), **PP-18** (the measuring
binary is built from an ancestor of `HEAD`), **PP-23** (roofline), **PP-24** (equal *server*
admission).

An earlier draft of this list mis-stated three of those IDs — it gave PP-17's content under
PP-16, PP-18's under PP-17, and omitted PP-18 — which is the identical defect the renumbering
in §6 exists to prevent, committed in the very appendix that documents it.
