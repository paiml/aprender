# APR-PERF-GATE-001 v2.2 — Serving Performance: Measurement, Gating, Dogfood, Migration

**Status:** DRAFT v2.2 — supersedes v2.1 (2026-08-25) · **Date:** 2026-08-25
**Repo:** `paiml/aprender` · fleet: `paiml/infra`
**Refs:** #2692 #2693 #2695 #2696 #2697 · PR #2682 · branch `feat/2692-apr-probar-llm`
**Consolidates:** `APR-BENCH-RFC-001`, `APR-QUALITY-001` v1.8 §0.7 J3, `perf-parity-spec.md` (superseded for release gating, §4.6.4)

**Review passes assimilated:** architectural review (Claude) · staff review "Approve with changes" · 11-person panel review · operator strategic input · external architectural + Lean review (Toyota Way analysis, 62 citations) · **live comparator-default investigation, 2026-08-25** — new in v2.2, and the only `[V]` evidence in this document.

> **Operator ruling (carried):** this is the most important gate in the project.
> *"If this is not addressed the entire project shouldn't exist."*

### What changed from v2.1 — the accelerator contract

v2.1 and everything before it treated defect #1 as a *distribution* problem: crates.io ships CPU-only. A live check of what the two comparators actually do, fetched 2026-08-25, shows that framing is incomplete and that the deeper defect is a **CLI contract** defect which would survive shipping CUDA tomorrow.

| Verified 2026-08-25 | Source | Mark |
|---|---|---|
| llama.cpp `-ngl` / `--n-gpu-layers` takes an integer, `auto`, or `all`; **default `auto`**. `--device` default auto. `--fit` **default `on`**, auto-fitting unset args to device memory. Documented default recipe is a bare `llama-server -m model.gguf` | `ggml-org/llama.cpp` `docs/multi-gpu.md` @ master | `[V]` |
| **llama.cpp exposes no boolean `--gpu` flag.** The request is a *quantity* and a *device list*; the loader reports the resolved layer split | ibid | `[V]` |
| **Auto-fit never overrides an explicitly-set argument** — it only touches parameters the user did not set | `ggml-org/llama.cpp` discussion #18049 | `[V]` |
| Ollama auto-detects Metal / CUDA / ROCm at startup, offloads what fits, and **silently falls back to CPU** for the rest | `ollama/ollama` issue #14258, open, filed 2026-02-14 | `[V]` |
| Ollama's own issue calls this arguably its single most common source of user confusion, with 500+ GitHub results across "GPU not detected", "fallback CPU", "very slow", "GPU not used" — most still open | ibid | `[V]` |

**Both comparators default to GPU.** So do we, going forward. But the finding that changes the spec is narrower and sharper:

| # | Finding | Consequence |
|---|---|---|
| **N4** | **A boolean accelerator flag has no observable resolution.** `--gpu` can be ignored and nothing in the output changes. `-ngl 999` cannot be ignored — the loader must state how many layers it placed. llama.cpp's design makes silent-ignore structurally hard; ours makes it easy | §4.10 I-2 extended; **PERF-021** retires the boolean |
| **N5** | **The rule aprender broke is not "default to GPU."** It is *automation never overrides an explicit user instruction.* The user typed `--gpu`; the system chose CPU anyway. This is a better why-5 than §7.5 had | §7.5 rewritten; **I-17** new |
| **N6** | **Ollama's silent fallback is defect #1, open, in a project with two orders of magnitude more users.** It is a documented failure mode with a long tail, not a design to copy. llama.cpp announces the split in log prose only. **Nobody reports resolved-vs-requested offload in structured output** | §3.3 — a second place we can be ahead, and it is cheap |
| **N7** | **Neither comparator distributes via a source-compiling package registry.** llama.cpp ships per-backend prebuilt binaries plus ggml dynamic backend loading; ollama ships one installer bundling per-backend runner libraries loaded at runtime. `cargo install aprender --features cuda` has **no precedent in the surveyed field** | §4.9.3 and O-1 reframed as a three-way choice |

N4 and N5 are the critical pair: they mean the jidoka fix already landed (exit 9) is **necessary and not sufficient**. Exit 9 fires when the backend is entirely absent. It does not fire when the backend is present, the user asked for full offload, and auto-fit quietly gave them eleven layers.

---

### What changed from v2.0

The external review did two things v2.0 could not do for itself: it supplied the **named prior art for the fix** (Orca iteration-level scheduling, OSDI 2022; PagedAttention, SOSP 2023), and — by describing what a *correct* serving engine does — it exposed **three failure classes this gate is structurally blind to.** All three become live the moment batching lands, which means they must exist *before* it lands, not after.

| # | Blindness | Why v2.0 could not see it | Section |
|---|---|---|---|
| **N1** | **Homogeneous workload.** Every band uses one fixed 512/128 shape. Prefill/decode interference — the failure chunked prefill exists to fix — is unreachable | A serialising server has no interference to measure, so the gap was invisible | §4.3.2, Arm E |
| **N2** | **No memory metric.** KV-cache fragmentation wastes 60–80% of VRAM under contiguous allocation; paging cuts it below 4%. A batching implementation without paging passes Arm A and regresses memory silently | apr allocates one request's cache at a time; there is nothing to fragment | §4.5 Arm D |
| **N3** | **Token counting undeclared.** SGLang's own benchmark-consolidation RFC records inconsistent tokenization and token counting across its harnesses. Our ratio is invalid if apr and the comparator count tokens differently | Both sides were measured by the same ad-hoc script, so the question never arose | §4.4.6 |

Plus a protocol defect (boundary effects, §4.4.7), a comparator-harness defect (§4.4.8), and **four factual errors in the external review itself**, recorded in §0.5.

---

**Restarting work on this?** Use
[`APR-PERF-GATE-001-RESTART.md`](./APR-PERF-GATE-001-RESTART.md) — a terse,
copy-pasteable prompt plus the two preconditions (`pmat serve` is not
persistent; gx10 must be verified before its numbers are trusted).

## §0.0 Nomenclature — how to refer to this document

Three registers. Use the one that fits the sentence; do not mix them in a commit subject.

| Register | Handle | Use it for |
|---|---|---|
| **Formal** | `APR-PERF-GATE-001` | commits, contracts, PR bodies, issue links, roadmap ids |
| **Artifact** | **the perf gate** | the machinery — `perf_gate.sh`, the arms, `perf-matrix.yaml` |
| **Principle** | **the receipt rule** | conversation, review, teaching it |

### Why not "the gate"

It is the shorthand that emerged on its own, and it is already ambiguous in this
repository: 72 `scripts/check_*.sh` guards, three CI contexts literally named
`gate`, three `gates` entries in `[package.metadata.dogfood]`, and the clean-room
release hard gate. "The gate" picks out nothing here.

### The receipt rule

> **A performance number is evidence only if something can prove how it was
> measured.**

Named for the principle rather than the mechanism, for three reasons:

1. **It survives revision.** `perf_gate.sh` will be rewritten and v2.2 will become
   v3; the rule will not change. A handle tied to the script would rot with it.
2. **It is already the vocabulary.** 33 files under `scripts/` and `docs/` use
   "receipt" in exactly this sense — `bench_receipt.py`,
   `check_parity_receipt.sh`, the `receipt.commit ⊇ commit-under-test` staleness
   arm. This promotes a load-bearing term rather than coining a competing one.
3. **It works as a verb.** *"Has that been receipted?"* — which is the question
   this document wants asked in review, phrased so it can be asked in four words.

### The test it lets you apply without reading the rest

| Claim | Verdict |
|---|---|
| `2.93× Ollama` in `book/`, from a hardcoded 291 tok/s baseline | **no receipt** |
| `expires:` declared on every cell, read by zero lines of code | **no receipt** |
| `apr gpu` printing a GPU id on a binary with zero CUDA symbols | **no receipt** |
| `serialization_index(2) = 2.85` on gx10, binary proven CUDA-linked | **receipted** |
| any `rc` read through a pipe (`cmd \| head; echo $?`) | **no receipt** |

The last row is not a joke: it is the most common way a receipt is forged by
accident, and it has happened repeatedly during this epic's own implementation.

---

## §0 Preflight

### 0.1 Grounding contract

| Mark | Meaning |
|---|---|
| `[V]` | Verified against a named artifact by a command reproducible from this document |
| `[C]` | Carried from a prior revision or a cited spec; **not re-verified in this pass** |
| `[U]` | Unverified — asserted, needs a command before it may be relied on |
| `[X]` | **External claim about a third-party system.** Design prior art only. **Never a target, never a claim of ours** |

`[X]` is new in v2.1 and exists because the external review introduced figures like *36.9× over FasterTransformer*, *23× over static batching*, *1.8× over vLLM*, and *13× on 200k-token prompts*. Those are other projects' published numbers about other projects' systems. Under §3.2 we publish no comparator ratios at all; importing someone else's would be the same defect as §1 defect #4 with a different provenance story. **A `[X]` figure may inform a design choice and may never appear in `README.md`, `book/`, or `docs/`** (I-12 extended).

**This revision was produced without read access to `origin/main`, to `feat/2692-apr-probar-llm`, or to any cited line number.** Every path, line number, branch state and count is `[C]`. §14 enumerates each with its promoting command. **No `[C]` item may be cited as evidence in a PR body.**

### 0.2 Sources of truth

| Question | Path |
|---|---|
| Host state / provisioning | `infra/machines/<host>/forjar.yaml`; `forjar apply -f machines/<host>/forjar.yaml` |
| Release hard gate | `infra/machines/clean-room/gates-lib.sh`, `gates/aprender.sh`; `make -C machines/clean-room clean-room-p1` |
| Branch protection | `infra/docs/specifications/sovereign-stack-protected-branch-strategy.md` — `main` protected, required contexts `ci / gate` + `workspace-test` |
| Canonical model / quant / comparator pin | `APR-BENCH-RFC-001` §3, §8; `aprender/benchmarks/canonical-ledger.yaml` |
| Noise floor / dispersion thresholds | BENCH-003 — **not yet run** |
| Receipt transport | `APR-QUALITY-001` v1.8 §0.7 J3 |
| Work items | `pmat work list`; `infra/docs/roadmaps/roadmap.yaml` |

### 0.3 Assimilation ledger — v2.0 passes (condensed)

Full text in v2.0 §0.3. Dispositions stand unchanged.

**Accepted, architectural review:** A1 clean-room Mode A as measurement target · A2/A3 the §3.2-vs-§8 non-sequitur and the 1.554/1.553 drift · A4 threshold epistemics · A5 declared-vs-resolved deletion claim · A6 mutation registry · A7 single verdict function · A8 signed receipt transport · A9 join-key enforcement · A10 sample-retention budget · A11 resolver contract · A12 comparator-free primary arm.

**Accepted, staff review:** S1 two-phase model · S2 all metrics all bands · **S3 measurement protocol** · S4 canonical workload · **S5 binary SHA is host-local only** · S6 expiry semantics · S7–S9 host-matrix rigour · S10 replicate wording · S11 branch split · S12 precedence rule · S13 ratchet mechanics.

**Accepted, panel:** P1 single-lock reading · P2 jidoka to main now · P3 whole-tool profile audit · P4 three-valued status · P5 qwen-story as a second subject · P6 CPU-only posture · P7 detector-vs-poka-yoke reclassification.

**Rejected (unchanged):** R1 runtime descriptor bridge · R2 abstention-to-conceal · R3 0.80-as-CI-blocker · R4 parity-as-dashboard-target.

### 0.4 Assimilation ledger — external architectural + Lean review

| # | Contribution | Disposition |
|---|---|---|
| X1 | **Orca (Yu et al., OSDI 2022) named as the prior art** for iteration-level scheduling: evict on EOS, inject waiting requests before every forward pass | §7.1 — the reference architecture for PERF-001. Removes "design continuous batching from scratch" from the estimate |
| X2 | **PagedAttention (vLLM, SOSP 2023):** contiguous KV allocation must reserve for `max_sequence_length` at arrival; 60–80% VRAM wasted; paging cuts waste below 4% `[X]` | **N2** — §4.5 **Arm D**, new. Must exist *before* PERF-001 lands |
| X3 | **Chunked prefill:** a long prefill injected into an active batch stalls every concurrent decode and spikes ITL | **N1** — §4.3.2 workload **W2**, §4.5 **Arm E**. Our fixed 512/128 shape cannot reach this |
| X4 | **Preemption is not free:** recompute costs GPU, swap costs hundreds of ms across PCIe on long context | §4.4.9 — preemption counters required in the receipt from day one, so the first batching PR's preemption strategy is visible rather than discovered later |
| X5 | **SGLang benchmark fragmentation (issue #9808):** `bench_one_batch` vs `bench_serving`, inconsistent tokenization and token counting | **N3** — §4.4.6. Also external corroboration for §9's deletion mandate: this is what happens when harnesses are deprecated instead of deleted |
| X6 | **TGI boundary effects:** a large request arriving just before benchmark termination skews RPS | §4.4.7 — drain semantics, new |
| X7 | **`llama-bench` does not separate PP from TG under concurrent load;** metrics intertwine | §4.4.8 — the comparator is `llama-server` driven by **our** client, never `llama-bench`. Otherwise the ratio compares two client implementations |
| X8 | **vLLM CI:** diff-aware pipeline generation, 100+ parallel jobs on heterogeneous hardware; `perf.vllm.ai` runs continuously **every 4 hours**, independent of PRs | §4.1 — validates two-phase + nightly. Diff-aware generation recorded as future prior art (§12), not built now |
| X9 | **vLLM gates a derived internal metric** — speculative-decode draft acceptance rate against historical baselines, not a competitor ratio | Independent corroboration of Arm A. Cited in §2.1 |
| X10 | **Triton Model Analyzer** sweeps batch/concurrency/instance configs against QoS constraints | §12 do-not-build — we gate a fixed band set; config search is a tuning tool, not a gate |
| X11 | **LMDeploy TurboMind:** persistent batching implemented in C++/CUDA to bypass Python dispatch overhead `[X]` | §7.1 — aprender is Rust and structurally in TurboMind's class, not vLLM's. The Python-dispatch overhead argument does not apply to us, which removes one hypothesis for the 0.075 |
| X12 | Lean mapping (Jidoka/Poka-yoke/Kaizen/Genchi Genbutsu/Heijunka) independently reconstructed and matched §6 | Accepted; §6 unchanged except where §0.5 corrects it |

### 0.4a Live comparator-default investigation (v2.2)

Not a review pass — a measurement, and the only `[V]` evidence in this document. Findings N4–N7 in the header; dispositions below.

| # | Contribution | Disposition |
|---|---|---|
| V1 | llama.cpp defaults to GPU (`-ngl auto`, `-fit on`) when the build carries the backend | §4.9.3 — adopt GPU-first default. Not controversial; both comparators do it |
| V2 | llama.cpp has **no boolean `--gpu`**; the request is a quantity + device list, and the resolution is reported | **PERF-021** — retire the boolean. §3.1 row 16 |
| V3 | Auto-fit is disabled for any argument the user set explicitly | **I-17** — the poka-yoke form of jidoka. §7.5 why-5 |
| V4 | Ollama's silent CPU fallback is an open issue with a 500+ downstream tail | §12 do-not-build. **Do not copy the comparator here** |
| V5 | Neither comparator ships via a source-compiling registry | O-1 reframed three ways; `--features cuda` marked as the option with no field precedent |

### 0.5 Corrections to the external review

The external review is the most useful input this document has received. It also states four things about the spec that are not true. Recording them is not pedantry: a review that misdescribes the artifact will be cited later as though it described it.

| # | Review says | Correct | Why it matters |
|---|---|---|---|
| **X-E1** | §1.4: clean-room ensures the measured artifact is *"byte-for-byte identical to the binary shipped to users."* | **False.** `cargo install` compiles from source on the host. Binaries are not bit-identical across hosts or toolchains — this is finding S5, and §4.2.2 splits identity into three fields precisely because binary SHA cannot bear that role | This is the exact belief that made v1.0 assign cross-host duty to `binary_sha256`. If it re-enters through a review, S5 is undone |
| **X-E2** | §3.1: Arm A replaces comparator ratios *"for merge-blocking Continuous Integration"* | **False.** Arm A is **release**-blocking. No wall-clock arm blocks a merge (§4.1, I-6). The review's own scorecard table has it right; the prose contradicts the table | Promoting Arm A to merge phase reintroduces the red-PR fatigue the two-phase model exists to prevent |
| **X-E3** | §1.3: the ratchet *"replaces static numerical limits"* | **Partially false.** The dispersion ceiling (1.50) was deleted. The B1 policy floor (0.80) is retained and is static **by design** — a product decision, legal precisely because it does not claim to be a measurement (§4.6.1) | Collapsing the four threshold classes back into one is how `10.0` happened |
| **X-E4** | §4.3: attributes the harness-deletion mandate to *"Invariant 4 and §9"* | I-4 is raw-sample retention. Deletion is §9 and the `check_no_competing_harnesses.sh` guard | Citation hygiene |

Also noted, without correction: the review describes exit 9 as *"stopping the CI line."* It stops the **product**, at user runtime, on the user's machine. That is a stronger form of jidoka than a CI stop and is the intended reading.

### 0.6 Corrections to v1.0 — self-reported (carried)

| # | v1.0 | Correct | Class |
|---|---|---|---|
| E1 | §4: harnesses "are **deleted**" | Staged on branch, unshipped; present on `origin/main` `[C]` | Declared-state-as-resolved-state |
| E2 | §8: c=8 FAILs at decode 1.352× under `ratio ∈ [0.80,1.50]` | 1.352 is inside the interval; c=8 fails on **aggregate** (0.169×) | Non-sequitur proof |
| E3 | c=16 decode 1.554 (§1) vs 1.553 (§8) | Canonical **1.554** `[C]` | Transcription drift |
| E4 | §4 cross-refs §6 for deletions | §7 | Cross-ref |
| E5 | Instrument defects as preamble prose | Promoted to case file CF-5 + tickets | Failure-archaeology discipline |

---

## §1 The four adoption killers

All `[C]`, measured 2026-08-25, quiet box, comparator llama.cpp pinned `39173bcac`.

| # | Defect | Evidence | Ticket |
|---|---|---|---|
| 1 | **`cargo install aprender` is CPU-only and says nothing.** `--gpu` accepted, runs on CPU. Idle RTX 4090 → 15.7 tok/s, 7.5 s TTFT vs llama.cpp's 158.9. **Two defects, not one:** a distribution defect (the artifact has no backend) *and* a contract defect (a boolean flag whose resolution is unobservable — N4/N5) | published `aprender-0.64.0/Cargo.toml`: `default = ["cli"]`, `cuda` opt-in | #2696, PERF-021 |
| 2 | **apr does not batch.** Aggregate flat ~110 tok/s at every concurrency — 0.097× at c=16 | §2 band sweep | PERF-001 |
| 3 | **`--batch` hangs** on four concurrent chat requests, advertised as "2X+ throughput" | probe timed out at 9m50s; harness reported 0.5 tok/s aggregate | PERF-002 |
| 4 | **The book publishes a number no harness produced** — "851.8 tok/s = 2.93× Ollama", Ollama never executed | `book/src/examples/showcase-benchmark.md:17,22`; `book/src/tools/apr-cli.md:1396,1493,1498` | PERF-010 |

Defects 2 and 3 are provisionally **one defect** (§7.1). Do not open two workstreams before PERF-000 answers it.

### 1.1 CF-5 — the instrument was lying too (permanent case file)

Kept permanently alongside CF-1…CF-4 in `aprender-pillar-vision-v2.md` §7.

`apr profile --granular` prints *"Large non-kernel overhead — investigate sampling sync (gpu_argmax D2H)"* — a **hardcoded string fired on a threshold** (`profile_print_hotspot.rs:155-162` `[C]`); the tool never inspects sampling. The attached 90.3% is arithmetically invalid: `kernel.rs:531-542` `[C]` subtracts a **16-token** kernel sum from a **32-token** wall time, and the "kernel" term is CPU `Instant` timing, not GPU time.

It was cited as evidence for a root cause during this investigation before being caught. **Neither the number nor the cause may be cited.**

**Doctrine lesson:** a diagnostic tool that emits a *causal* claim is making a claim and is bound by the same provenance rule as a published number. A hardcoded causal string is a fabricated baseline wearing a different hat.

**Scope (P3):** the two sites are instances, not the population. `apr profile` is audited whole (PERF-016).

---

## §2 Why one metric was not enough — and what replaces it

Measured on lambda (RTX 4090), comparator pinned `39173bcac`, quiet box. All `[C]`, and **none conformant to §4.4** (see §14).

| band | llama agg | apr agg | agg ratio | llama dec | apr dec | dec ratio |
|---|---|---|---|---|---|---|
| c=1 | 168.9 | 90.2 | 0.534× | 171.5 | 100.7 | 0.587× |
| c=4 | 484.7 | 111.9 | 0.231× | 123.3 | 113.8 | 0.923× |
| c=8 | 650.5 | 109.6 | 0.169× | 83.0 | 112.2 | **1.352×** |
| c=16 | 1120.8 | 108.4 | **0.097×** | 71.2 | 110.6 | **1.554×** |

apr's aggregate is flat because it **serialises**. Per-user decode *rises* to 1.554× purely because each request gets the whole GPU in turn while llama.cpp shares it sixteen ways.

**A gate reading only per-user decode scores c=16 a comfortable PASS while a sixteen-user deployment runs at a tenth of llama.cpp.** `scripts/llama_pin.toml` said `http_concurrency = 1` `[C]` — one line, and every parity number this project has published measured the worst band and called it the answer.

### 2.1 The derived metrics — comparator-free

```
batch_scaling(c)       = agg(c) / agg(1)
scaling_efficiency(c)  = batch_scaling(c) / c           # 1.0 = perfect batching
serialization_index(c) = decode_ratio(c) / agg_ratio(c) # → c when fully serialised
```

| c | apr `scaling_efficiency` | llama.cpp `scaling_efficiency` | apr `serialization_index` |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.10 |
| 4 | 0.310 | 0.718 | 3.996 |
| 8 | 0.152 | 0.481 | 8.00 |
| 16 | **0.075** | 0.415 | **16.02** |

`serialization_index(c) ≈ c` at every band — the signature of a server admitting exactly one request at a time, and the tightest single piece of evidence in this document.

**`scaling_efficiency` is the primary gate arm** (§4.5 Arm A):

1. It measures defect #2 **directly** — no ratio, no comparator, no pin.
2. It is producible on **all four hosts today**, independent of the vLLM aarch64 gap, the gx10 llama.cpp build, and the #2696 release decision.
3. It needs **no invented threshold** — baseline is today's measurement, ratchet-up-only.
4. It is the self-referential assert tract uses (`hwbench --assert`) and that **vLLM independently arrived at** for speculative decoding: `perf.vllm.ai` gates draft **acceptance rate** against historical baselines rather than a competitor ratio `[X]`. Two mature projects converged on internal-metric ratcheting from opposite directions.

Divergence between Arm A and Arm B indicts the **pin**, not the build.

### 2.2 What `scaling_efficiency` alone still cannot see

Arm A is necessary and **not sufficient**. It is a single scalar over a homogeneous workload, and three failure classes pass it cleanly:

| Passes Arm A | Fails the user | Covered by |
|---|---|---|
| Batching that reserves KV for `max_seq_len` at arrival, fragmenting 60–80% of VRAM and rejecting admissions while compute sits idle `[X]` | Concurrency ceiling far below the hardware's | **Arm D** §4.5 |
| Batching that admits a long prefill into an active batch and stalls every concurrent decode | p95 ITL spikes; interactive use unusable | **Arm E** §4.5 + workload **W2** §4.3.2 |
| Batching that hits the memory wall and preempts by **swapping** KV blocks across PCIe | Hundreds of ms added to tail latency `[X]` | preemption counters §4.4.9 |

Each becomes reachable the moment batching lands. **Therefore each must be instrumented before it lands** (§10.1), or the first batching PR ships three new invisible defects and the gate certifies them.

---

## §3 Competitive research (CRUX)

Quorum: llama.cpp, vLLM, Ollama, TGI, SGLang, Rust runtimes (candle, mistral.rs, tract), non-LLM perf-rigor projects (rustc-perf, ClickHouse, SQLite, Chromium). Extended in v2.1 with Orca, LMDeploy/TurboMind, Triton, and the tooling survey. All `[C]`/`[X]`.

### 3.1 Practices

| # | Practice | Who | Blocking there | v2.2 status |
|---|---|---|---|---|
| 1 | Concurrency sweep is the **unit** of measurement | vLLM `max_concurrency`, llama.cpp `-npl`, TGI (30 rates), SGLang, mistral.rs | SGLang: release | adopted §4.5 |
| 2 | Aggregate gated **jointly** with per-request latency | SGLang (`output_throughput` **and** `median_ttft_ms`), vLLM (TTFT **and** TPOT), mistral.rs | yes | adopted — **and this is why the fix cannot precede the gate** (§10.1) |
| 3 | `completed == requested` before reading throughput | SGLang | yes | Arm C |
| 4 | Hardware/config identity as a **join key** (25 properties) | llama.cpp `compare-llama-bench.py` | n/a | **fatal** for cross-host, §4.2.3 |
| 5 | Fail-closed when the fast path is unreachable | ollama `sched_test.go`, `ml/device_test.go`, GPU-less runner | every PR | exit 9 landed; device test PERF-005 |
| 6 | Self-referential dispatch assert | tract `hwbench --assert`; **vLLM draft-acceptance gating** | yes | **Arm A** |
| 7 | Threshold derived from the metric's own history | tract `bench-thresholds.toml`, k=3.0 × dispersion | advisory | blocked on BENCH-003 |
| 8 | Comparator pinned by SHA + literal flags, captured into the result | mistral.rs `capture_metadata.sh`; ClickHouse pins the baseline binary **and** checks out `tests/performance` at the baseline SHA | ClickHouse: yes | on branch, not main |
| **9** | **Iteration-level scheduling** — evict on EOS, inject before every forward pass | **Orca, OSDI 2022**; now baseline in vLLM, TGI, SGLang, LMDeploy | n/a (architecture) | **prior art for PERF-001**, §7.1 |
| **10** | **Paged KV cache** — fixed blocks + per-request block table | vLLM PagedAttention, SOSP 2023; LMDeploy TurboMind | n/a | **Arm D** measures the absence; PERF-018 |
| **11** | **Chunked prefill** — bound prefill work per iteration, interleave with decode | vLLM, TGI v3, SGLang | n/a | **Arm E** + workload W2; PERF-017 |
| **12** | **Continuous benchmarking decoupled from PRs** — `perf.vllm.ai` every 4 h | vLLM | dashboard, not gate | adopted as nightly cadence §4.1 |
| **13** | **Boundary-effect control** in the harness | TGI benchmarking tooling | n/a (tool) | §4.4.7 drain semantics |
| **14** | **Declared token-counting method** | SGLang RFC #9808 records its absence as a defect | open there | §4.4.6 — **we adopt what they are still fixing** |
| **15** | QoS-constrained config sweeps | Triton Model Analyzer (Optuna) | tuning tool | §12 do-not-build |
| **16** | **Accelerator request is a quantity + device list, never a boolean; the loader reports the resolved split** | llama.cpp `-ngl {N\|auto\|all}`, `-dev`, `--list-devices`; ollama `ollama ps` PROCESSOR column | n/a (CLI contract) | **absent — we ship a boolean `--gpu`.** PERF-021 |
| **17** | **Auto-fit touches only arguments the user did not set** | llama.cpp `--fit on` (default); explicitly setting `-ngl` disables it | n/a | **absent — this is defect #1's root cause.** I-17 |

### 3.2 The finding that reframes the problem

**On claim-drift, all twenty surveyed projects abstain.** llama.cpp publishes zero quantitative performance claims. vLLM says "state-of-the-art serving throughput" and gives no number. Ollama, candle, SGLang, TGI: nothing.

Nobody has a guard because **nobody makes the claim**. We make claims, so we need the guard — but *make fewer claims* is the cheaper half, and it is llama.cpp's actual strategy.

> **Rule.** A number in `README.md`, `book/` or `docs/` is legal **iff** it cites an `evidence/` receipt path. A comparator ratio is illegal regardless of citation. **A `[X]` figure — any third-party project's published number — is illegal regardless of citation** (v2.1 extension, I-12).

This keeps `scaling_efficiency(16) = 0.075` public. We are not hiding it (§0.3 R2).

### 3.3 Where we are ahead — do not regress

**No surveyed project installs from a package registry and measures the installed artifact.** `scripts/check_multiplatform_dogfood.sh` does, with its comparand read from protected `origin/main` so the matrix can grow and never shrink.

**Correction:** the `-lt 4` host floor is an invented literal like any other. The *mechanism* is ahead of the field; the constant is not. Replace with `floor = count(origin/main)`.

**Third place we can be ahead, and it is cheap (N6):** **nobody reports resolved-vs-requested offload in structured output.** llama.cpp announces the layer split in load-time log prose. Ollama buries it in debug logs and pays for it with an open issue and a 500+ result tail `[V]`. Emitting `gpu_layers_requested` / `gpu_layers_resolved` / `backend_loaded[]` in the banner, `/health` **and** the receipt (§4.4.9) is a few fields of work and closes the single most common support failure in the category. It is also andon in the literal sense: the machine raises its own lamp.

**Second place we are ahead, newly evidenced:** SGLang's own RFC #9808 documents harness fragmentation — competing scripts with divergent interfaces, tokenization and token counting — as a live, open problem in a top-tier project `[X]`. §9's delete-don't-deprecate mandate and §4.4.6's declared token-counting method are the countermeasure they have not yet shipped. This is the strongest external validation in the review set, and it is validation of the *unglamorous* half of the spec.

---

## §4 The gate, specified

**Entry point:** `scripts/perf_gate.sh --host <name> --phase {merge|release} --workload {W1|W2}`, host resolved from committed `perf-matrix.yaml`, runnable verbatim on a dev box.

### 4.1 Two-phase blocking model

| Phase | Where | What blocks | Rationale |
|---|---|---|---|
| **merge** | required context `ci / gate`, every PR | schema validity, receipt parse, `completed == requested`, `timeouts == 0`, claim-literal guard, competing-harness guard, resolver selftest, token-method declaration | Cheap, deterministic, **no wall-clock ratio**. Eleven wall-clock ratios have failed here; one blocked all nine open PRs |
| **release** | clean-room Mode A, at publish | Arms A, B, C (+D, E when promoted), cell completeness, receipt freshness + signature | Jidoka stops the line at the release, not at every commit |
| **nightly** | receipt-push hosts, cadence ≤ 24 h | nothing — REPORTING | vLLM runs `perf.vllm.ai` every 4 h decoupled from PRs `[X]`. Trend data is what makes BENCH-003 and history-derived thresholds possible at all |

`qwen-story` (correctness/determinism) runs at **merge**; the concurrency matrix runs at **release** and **nightly**. Two subjects, three cadences, **one receipt schema and one comparator-required rule**.

**Consequence, stated plainly:** landing this gate turns **zero** PRs red and makes **0.64.0 uncuttable** until Arm A clears. That is intended.

**X-E2 correction:** Arm A is release-blocking, not merge-blocking. Promoting it to merge phase reintroduces exactly the red-PR fatigue the model prevents (I-6).

### 4.2 What is measured

#### 4.2.1 The artifact

v1.0 measured `cargo install aprender --version X.Y.Z` — a version that exists only **after** publish. A gate whose subject is a published artifact is a yank trigger.

The target is the artifact produced by **clean-room Mode A**:

```
A0 strip path deps → A1 fresh lockfile → A2 cargo check → A3 cargo install → A4 verify installed binary
                                                                              └── perf lane attaches here
```

Never a workspace build. Never `target/release`. **Clean-room is the release hard gate; no `cargo publish` for any stack crate without it.** A post-publish confirmation run against the real crates.io artifact is retained as **REPORTING**; its only blocking authority is to trigger a yank.

> **Why stated emphatically.** On 2026-08-25 `~/.cargo/bin/apr` was silently replaced by a local build of main at 06:58, where the day before it was the genuine crates.io artifact `[C]`. A conclusion about "the published binary" drawn from a path would have been wrong.

#### 4.2.2 Artifact identity — three parts, three jobs

`cargo install` **compiles from source on the host.** `binary_sha256` therefore differs across hosts and toolchains for the same crate version and **cannot** bear a cross-host role. See **X-E1**: an external review restated the byte-identity belief, which is how this defect returns.

| Field | Scope | Job |
|---|---|---|
| `crate_tarball_sha256` | **cross-host**, from static.crates.io or the Mode A tarball | The identity that must agree before any two hosts' numbers are compared |
| `binary_sha256` | **host-local** | Anti-substitution fingerprint — detects the 06:58 replacement, nothing more |
| `build_identity` | host-local; compared cross-host as a **join key** | `rustc -vV`, target triple, enabled cargo features **read from the built binary**, `uname -a`, accelerator + driver version |

Features are read from the binary, never from `Cargo.toml`. Declared features are declared state.

#### 4.2.3 Join-key enforcement

| Operation | Identity mismatch |
|---|---|
| Cross-host aggregation; any comparator ratio | **FATAL** — refused, not warned |
| Within-host history, trend, CI | REPORTING — annotated |

#### 4.2.4 Resolver contract

`scripts/apr_bin.sh` and `scripts/llama_bin.sh` are **sourced**. A sourced script that calls `exit` terminates its caller.

- Resolvers **`return <code>`, never `exit`**.
- Callers check an exported variable, not `$?` of a `source`.
- **Selftest:** source each resolver in a subshell with the target binary absent; assert the *caller* survives to print a remedy. Merge phase.

### 4.3 Workload

#### 4.3.1 W1 — homogeneous (blocking today)

| Field | Value | Authority |
|---|---|---|
| model | `paiml/qwen2.5-coder-7b-apache-q4k-v1` | `APR-BENCH-RFC-001` §3.1 |
| file | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` | ibid |
| quant | **Q4_K_M** | ibid §8 — `IQ*`, `Q4_0_4_8/8_8`, plain `Q4_0` excluded as arch-specific by construction; `Q8_0` blows `mini` |
| model sha256 | REQUIRED; no cell without it | ibid §9 |
| prompt profile | fixed corpus, `prompt_tokens = 512 ± 8`, `crates/aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl` | this doc |
| generation | `max_tokens = 128`, greedy, `seed = 0`, ignore-EOS | this doc |
| context | 4096 | this doc |

Mirrors `APR-BENCH-RFC-001`'s `pp512`/`tg128` so the canonical benchmark and this gate cross-validate.

**Corpus format — JSONL, and only JSONL (PERF-039).** One JSON object per line;
no enclosing array, no YAML. An optional first line carrying a `_meta` key is
the corpus's own provenance header and is not a request.

```jsonl
{"_meta":{"corpus":"W1","provenance":"SYNTHETIC — seeded PRNG over a fixed word pool","token_count_verified":false}}
{"id":0,"prompt":"…","max_tokens":128,"temperature":0.0,"seed":0,"target_prompt_tokens":512}
```

`prompt` is required; `role`, `max_tokens`, `temperature`, `seed`, `ignore_eos`,
`id` and `target_prompt_tokens` are optional. **Unknown fields are rejected**, so
a record whose budget key is misspelled cannot load with a silently defaulted
budget. W1 and W2 share this schema and one loader
(`jugar_probar::llm::load_prompts_from_file`).

This paragraph exists because the format was, until PERF-039, specified in three
mutually incompatible places: this section named `.jsonl`, the only loader in the
tree parsed YAML with a top-level `prompts:` key, and `apr test llm bench --prompts`
advertised "a JSON array". The consequence was that `prompts-w2.jsonl` — the only
corpus committed to the repo — could not be read by the only loader in the repo,
and nothing noticed because nothing had tried. This document is the authority; the
two implementations were the drift.

**`target_prompt_tokens` is a target, not a measurement.** No tokenizer runs in
the generator, so `512 ± 8` is asserted by the harness against the model's own
tokenizer at measurement time and declared in the receipt's §4.4.6 `tokenization`
block. **§4.3.1 does not say whether the 512 is counted before or after the chat
template wrapper**; the corpus stores raw prompt text and the harness applies the
template, so the receipt must declare which side of that boundary its count was
taken on.

**Prompts are distinct `[U]`.** "Fixed corpus" pins the corpus; it does not
require one prompt repeated, and `[U]` marks distinctness as chosen here rather
than derived. N identical prompts would let a server with prefix caching serve
bands 2..c from cache, so Arm A's `agg(c)` would rise with `c` for a reason that
is not batching — a gate measuring its own cache. Corpus size is 256 `[U]`:
§4.4.2 consumes at most `8×16` sampled + `2×16` warmup = 160 requests at the
widest band of §4.5.

**ignore-EOS is a wire field with partial backend coverage (PERF-039).** `ignore_eos` had no representation on either side before PERF-039 — not in the
harness's `ChatRequest`, not in the server's `ChatCompletionRequest` — so the
§4.3.1 row above described a workload no client could request. It now exists on
both. The quantized GGUF chat backend that W1's Q4_K_M model runs on honours it;
the SafeTensors-CUDA, f32 APR-transformer and dense-registry backends stop on EOS
in a way no request field reaches and therefore **refuse the request with 501**
rather than serving it with EOS live. A silently dropped `ignore_eos` would let a
receipt record a pinned token budget it never received, so a cell measured on one
of those three backends is unrepresentable rather than wrong.

**`pp512` and `tg128` are never blended** — GB10 legitimately loses ~4× on decode while winning prefill, and a blended figure reports a correct machine as broken.

#### 4.3.2 W2 — ragged (N1, new in v2.1)

W1 is a single shape at every band. Real serving is variable-length, and two of the three most important properties of a correct batching engine are **only** observable under variance:

- Under **static batching**, the batch is padded to the longest sequence and the GPU stays locked until the longest finishes — a 10-token request and a 1000-token request in the same batch waste ~99% of one slot `[X]`. A homogeneous workload makes static and continuous batching **indistinguishable**.
- A **long prefill** injected into an active batch occupies the GPU for the whole forward pass and stalls every concurrent decode, spiking ITL `[X]`. With a uniform 512-token prompt there is no long prefill to inject.

| Field | Value |
|---|---|
| prompt lengths | mixture, `crates/aprender-serve/benchmarks/qwen-coder/prompts-w2.jsonl`: 40% at 128, 30% at 512, 20% at 2048, 10% at 8192 tokens |
| generation | mixture: 40% `max_tokens=16`, 40% `128`, 20% `512`; greedy, `seed = 0` |
| injector | at `t = window/2`, one request at the maximum prompt length is issued out-of-band; its arrival index is recorded |
| context | 16384 |

**Status at v2.1: REPORTING, non-blocking.** A serialising server has no batch for a long prefill to interfere with, so W2 measures nothing today and a blocking W2 would be permanently red — a gate people learn to walk past. W2 is recorded as `UNMEASURED`, `expires` = PERF-001 merge + 30 days, owner `@noah`. **W2 becomes blocking in the same PR that lands batching**, not later: that PR is the first moment its metrics are meaningful and the first moment it can regress.

The distribution weights above are `[U]` — chosen to span the range, not derived. They are a *shape* parameter, not a threshold, and are revised once W2 produces its first conformant receipt.

### 4.4 Measurement protocol

Without this, two conformant `perf_gate.sh` implementations produce incomparable receipts — which is no protocol at all wearing a schema. SGLang is currently living this failure (§3.1 row 14).

#### 4.4.1 Client model

**Closed-loop**, fixed concurrency `c`. Each of `c` workers issues a request, waits for completion, immediately issues the next. Chosen over open-loop because the failure being gated is *admission serialization*, which closed-loop exposes directly and open-loop masks behind queueing delay. Recorded as `client_model: closed_loop` so the choice is falsifiable rather than implicit.

The client is **external HTTP** on the same host, never in-process. An in-process client shares the runtime with the server under measurement.

#### 4.4.2 Warmup and sampling

| Parameter | Value | Basis |
|---|---|---|
| Warmup requests | `2 × c`, discarded, not written to the receipt | every worker completes ≥1 before sampling |
| Warmup gate | first sampled request begins after warmup completion **and** a 5 s quiesce | |
| Minimum sampled requests | `max(30, 8 × c)` | `[U]` — provisional, revised by BENCH-003 |
| Minimum wall-clock per band | 60 s | `[U]` — provisional, revised by BENCH-003 |
| Termination | whichever bound is satisfied **last**, then drain (§4.4.7) | prevents a fast host sampling too few and a slow host running unbounded |
| Replicates | `N = 3` full band runs per cell | I-9 |

The two `[U]` rows are *sample-size* parameters, not thresholds. An undersized `n` widens the CI and fails the gate rather than passing silently, so the failure direction is safe. Still provisional until BENCH-003.

#### 4.4.3 Metrics, defined

| Metric | Definition |
|---|---|
| `agg_tok_s` | (Σ generated tokens over **completed, non-truncated** sampled requests) ÷ (last completion − first request start). **Wall-clock, not the mean of per-request rates** |
| `decode_tok_s` | per request: (generated tokens − 1) ÷ (last token time − first token time). Median across sampled requests |
| `ttft_ms` | request start → first token byte at the client. p50 and p95 |
| `itl_ms` | inter-token latency, all gaps pooled across requests. p50 and p95 |
| `completed`, `requested`, `timeouts`, `truncated` | counts. Timeout = 120 s hard per request |

#### 4.4.4 Confidence intervals

Bootstrap **percentile** method, 10 000 resamples, seed `2026`, resampling **whole requests** — tokens within a request are not independent. BCa is not used: unnecessary at this dispersion, and its acceleration term is another undocumented degree of freedom. Seed in the receipt; interval reproducible from retained samples.

**REPORTING at v2.1.** Blocking only when BENCH-003 supplies the dispersion.

#### 4.4.5 Sample retention and its budget

Raw per-request samples are retained on every cell — a summary-only receipt cannot be resampled and is rejected (I-4).

- Gzipped JSONL inside the receipt directory.
- **Budget:** measure one full receipt, commit its size as `receipt_size_budget_bytes`, assert `≤ budget` in CI. No literal until measured `[U]`.
- Above budget: samples move to `almacen` with `sha256` + retrieval path in the receipt; the receipt stays in git.

BENCH-005's lesson: a large blob in git history is effectively permanent.

#### 4.4.6 Token counting must be declared (N3, new in v2.1)

SGLang RFC #9808 records inconsistent tokenization and token counting across its own harnesses as an open defect `[X]`. A throughput ratio between two servers that count tokens differently is not a ratio of anything.

Required in every receipt:

```yaml
tokenization:
  method: server_usage | client_tokenizer   # REQUIRED, no default
  tokenizer_sha256: "<REQUIRED when method = client_tokenizer>"
  counts_special_tokens: true | false        # REQUIRED
  counts_prompt_echo: true | false           # REQUIRED
```

- **`method` has no default.** Absence is schema-fatal, not a warning (poka-yoke, §6).
- **The canonical method is `client_tokenizer`** with the model's own tokenizer, applied identically to apr and to the comparator. Server-reported `usage` fields are two different implementations' opinions.
- **Any mismatch in the `tokenization` block between the measured lane and its comparator baseline is FATAL** — the ratio is refused, not annotated. This extends I-11 from build identity to counting semantics.

#### 4.4.7 Boundary effects and drain (X6, new in v2.1)

TGI's tooling controls for a large request arriving milliseconds before the benchmark terminates and skewing RPS `[X]`. v2.0's termination rule was open to exactly this.

- The measurement window closes at `T`. **No new request is issued at or after `T`.**
- All requests issued before `T` are **drained** to completion or timeout.
- `agg_tok_s`'s denominator is (last drained completion − first request start). Its numerator counts only requests that **started and completed** inside that span.
- A request that timed out during drain increments `timeouts`; one abandoned at drain deadline increments `truncated`.
- `drain_ms` is recorded. `drain_ms > 0.5 × window` is annotated `SUSPECT` — it means one request dominated the window, and the band should be re-run with a longer window.

#### 4.4.8 The comparator harness (X7, new in v2.1)

`llama-bench` does not separate prompt processing from token generation under concurrent load; the metrics intertwine `[X]`. It is a single-process local benchmark, not a concurrent-serving harness.

> **The comparator is `llama-server`, driven by the same client binary that drives `apr serve`.** `llama-bench` is never the comparator harness for Arm B.

A ratio produced by two different clients measures two client implementations. One client, both servers, same tokenization block, same workload file, same drain rule — or the ratio is refused (§4.2.3).

The comparator's build provenance is captured verbatim per `APR-BENCH-RFC-001` BENCH-001: single fleet-wide SHA, full `cmake` line, `-DGGML_CUDA_ARCHITECTURES=121` on gx10, `89` on lambda, `GGML_CUDA_FORCE_MMQ=1` on both CUDA hosts.

#### 4.4.9 Scheduler observability (X4, new in v2.1)

Required in every receipt from **v2.1 onward**, before batching exists, so the first batching PR's scheduler behaviour is visible in the diff rather than discovered in production:

| Field | Meaning | Today |
|---|---|---|
| `max_in_flight` | peak concurrent requests admitted by the server | expected **1** — this is defect #2 stated by the server itself |
| `admission_rejected` | requests refused for capacity | |
| `preempted_recompute` | sequences evicted with KV discarded, prefill re-run | |
| `preempted_swap` | sequences whose KV was moved to host DRAM | |
| `kv_blocks_total`, `kv_blocks_peak_used` | paged allocator state; `null` when allocation is contiguous | `null` |
| `kv_bytes_reserved`, `kv_bytes_used` | reserved vs actually populated | |
| `gpu_layers_requested` | what the user asked for, as parsed — `N`, `auto`, `all`, or `0` | |
| `gpu_layers_resolved` | what the loader actually placed, **read from the loader** | expected **0** on the published artifact |
| `gpu_layers_total` | the model's layer count, so `resolved/total` is legible | |
| `backend_loaded[]` | backends actually initialised, in load order (ollama's `load_backend:` line, structured) | expected `["cpu"]` |
| `autofit_applied[]` | which arguments auto-fit modified. **Must never contain an explicitly-set argument** (I-17) | |

`max_in_flight` deserves emphasis: **if the server reports it, defect #2 is self-diagnosing.** A field the server fills in is worth more than a ratio the harness computes, and it is the andon lamp for this failure class.

Preemption strategy matters because recompute costs GPU time while swap can cost hundreds of milliseconds across PCIe on long context `[X]`. A batching implementation that silently chooses swap will pass Arm A and destroy p95 ITL. These counters make that choice legible at review time.

**`gpu_layers_requested` vs `gpu_layers_resolved` is the same pattern applied one layer down (N4).** The jidoka fix already landed — exit 9 — fires only when the backend is *entirely absent*. It does not fire when the backend is present, the user asked for `all`, and auto-fit quietly placed eleven of thirty-two layers. That case is indistinguishable from success in every output aprender currently produces, and it is precisely ollama's open issue #14258 `[V]`. A boolean flag cannot express it; two integers can. **The gap between requested and resolved is the andon lamp, and it must be visible without a debug flag.**

### 4.5 Bands and arms

**All metrics at all four bands.** `agg(1)` is required — it is Arm A's denominator.

| Band | Client | Required |
|---|---|---|
| c=1 | 1 stream | `agg_tok_s`, `decode_tok_s`, p50/p95 `ttft_ms`, p50/p95 `itl_ms`, `completed`, `requested`, `timeouts`, `truncated`, `tokenization`, §4.4.9 block |
| c=4 / c=8 / c=16 | 4 / 8 / 16 concurrent | same |

A band missing any field is **schema-invalid** — unrepresentable, not merely rejected.

#### Arm A — scaling (comparator-free) · **PRIMARY** · release-blocking

```
scaling_efficiency(c) = (agg(c) / agg(1)) / c
```

| Rule | Value |
|---|---|
| Threshold | `≥ committed baseline`, per host, per band. **Ratchet-up-only** |
| Baseline today (lambda, W1) | c=4 `0.310`, c=8 `0.152`, c=16 `0.075` `[C]`, **non-conformant to §4.4** — must be re-measured before commit |
| Invented numerics | **zero** |
| Requires comparator | **no** |
| Mutation | cap the in-flight semaphore to 1 → red at c≥4 |

#### Arm B — adoption (comparator ratio) · release-blocking · degrades to `UNMEASURED`

| Sub-arm | Rule | Class |
|---|---|---|
| B1 aggregate | `agg_ratio(c) ≥ 0.80`, every band | **policy** — a product decision, not a measurement (§4.6.1) |
| B2 decode | `decode_ratio(c) ≥ 1.00`, every band | inherited from `perf-parity-spec.md` (§4.6.4) |
| B3 latency | p95 TTFT and p95 ITL jointly bounded | **blocked on BENCH-003** |

**B2's rising-decode trap:** decode ratio rising while aggregate falls is the serialization signature, not a win. When `serialization_index(c) > 1.5`, a B2 pass is annotated `SUSPECT` and cannot be cited as parity evidence. v1.0's 1.50 ceiling is **deleted** (§4.6.2); this cross-arm rule is what the ceiling was reaching for.

#### Arm D — memory efficiency (N2, new in v2.1) · **REPORTING at v2.1, blocking with PERF-001**

Contiguous KV allocation must reserve against `max_sequence_length` at request arrival; 60–80% of VRAM is wasted on tokens never generated, and admission stalls while compute idles. Paged allocation cuts waste below 4% `[X]`.

| Metric | Rule when promoted |
|---|---|
| `kv_utilization = kv_bytes_used / kv_bytes_reserved` | `≥ committed baseline`, ratchet-up-only |
| `admission_rejected` at `kv_utilization < 0.5` | `== 0` — refusing work while memory sits reserved-and-empty is the fragmentation signature |
| `preempted_swap` | `≤ committed baseline`, ratchet-**down**-only |

**Why this must exist before PERF-001, not after.** A batching implementation with contiguous allocation raises `scaling_efficiency` — Arm A goes green — while capping real-world concurrency far below the hardware and adding a new failure mode nothing measures. Arm A certifies it. That is the same shape as the c=1 decode gate certifying a serialising server, one layer up. **The gate must be able to see the fix's own failure modes before the fix lands.**

#### Arm E — prefill/decode interference (N1, new in v2.1) · **REPORTING at v2.1, blocking with PERF-001**

Measured on **W2** only.

| Metric | Rule when promoted |
|---|---|
| `itl_p95_ratio = p95_itl(W2) / p95_itl(W1)` at the same band | `≤ committed baseline`, ratchet-down-only |
| `injector_stall_ms` | p95 ITL of all *other* in-flight requests during the injected long prefill; ratchet-down-only |

This is the metric chunked prefill exists to move. Without it, a batching implementation that blocks the GPU on an 8192-token prefill passes every other arm.

#### Arm C — integrity · **merge**- and release-blocking

| Rule | Phase |
|---|---|
| `completed == requested` | both |
| `timeouts == 0` — fatal to that host's ratio | both |
| Receipt schema valid; every required field present | both |
| `tokenization.method` present; comparator block matches | both |
| Zero-token responses are failures, not fast requests | both |
| `drain_ms` present; `SUSPECT` annotation applied where due | both |
| Receipt signature valid; `receipt.commit ⊇ commit-under-test` | release |
| Every cell in `perf-matrix.yaml` present | release |

### 4.6 Thresholds and their epistemic status

| Class | Meaning | Source | Examples | Status |
|---|---|---|---|---|
| **Policy** | A deliberate product decision. Needs an author and a rationale, **not** a measurement | operator | B1 floor `0.80` | legal, labelled |
| **Inherited** | Set by a prior governing spec | `perf-parity-spec.md` | B2 `1.00` | legal, precedence recorded |
| **Dispersion** | A claim about noise | measurement only | v1.0's `1.50`; B3 bounds; CI blocking | **illegal until BENCH-003** |
| **Ratchet** | Today's measurement, monotone | the metric's own history | Arms A, D, E; claim-literal baseline; host floor | legal, no literal |
| **Shape** | Structure of the input, not a pass/fail bound | design | W2 length mixture; `N = 3`; sample-size floors | legal, marked `[U]`, revisable |

`Shape` is new in v2.1 so that W2's distribution weights are not mistaken for thresholds. **X-E3:** the ratchet did not replace all static limits — B1 is static by design.

#### 4.6.1 The 0.80 floor is policy, and says so

*Below 0.80× a user is better served by llama.cpp — that is the adoption question.* A product judgement, legal precisely **because** it does not pretend to be a measurement. v1.0's defect was not the number; it was presenting a policy number and a dispersion number in the same interval notation as though they had the same warrant.

#### 4.6.2 The 1.50 ceiling is deleted

*"Likelier a measurement error than a win"* is a **dispersion** claim over unmeasured dispersion. Deleted; its real job is done by `serialization_index`. Reintroduce as `k × σ` only after BENCH-003, with `k` and `σ` both cited.

#### 4.6.3 Ratchet mechanics

- Comparand read from **protected `origin/main`**, never from the PR's own tree.
- Arms A and D `kv_utilization`: **increase only**. Arm D `preempted_swap`, Arm E: **decrease only**. B1 floor: increases only, by recorded operator decision.
- Claim literals, competing harnesses, unwired guards: **decrease only** — with one admission, added by PERF-049 and named `set-aperture` in `scripts/lib_baseline_ratchet.sh`. A guard whose aperture WIDENS reveals claims that were already in the tree, and from the working tree alone that is the same diff as writing a fresh one, so the ratchet refused it and `check_no_claim_literals.sh` could not be widened at all. An entry may now be added iff (a) its `<path>:<line>` is **byte-identical at the comparand** — the branch neither wrote nor moved it — and (b) the **owning guard's own source changed** in the same diff. Every admitted coordinate is printed in the verdict row; everything else is refused exactly as before. This narrows, and does not remove, the rule that a finding is fixed rather than recorded.
- A PR moving a baseline the forbidden way fails at **merge** phase.

This is the answer to *"any state the author writes and the gate reads can be moved in the same commit."*

#### 4.6.4 Precedence over `perf-parity-spec.md`

> **For release gating of `apr serve`, APR-PERF-GATE-001 governs.** `perf-parity-spec.md`'s `decode ≥ 1.0×` is **adopted verbatim as Arm B2** and remains binding. Its TTFT ≤ 2× and ITL ≤ 1.5× bounds are **suspended** pending BENCH-003 and re-derived as Arm B3; the suspension is recorded in `perf-parity-spec.md` in the same PR that lands this document, **or the precedence clause is void.**

A precedence rule that leaves the superseded document unedited is two live specs, which is the condition it was written to end.

### 4.7 Cell status

#### 4.7.1 Three-valued

```json
{ "host":"gx10", "band":16, "comparator":"vllm", "status":"NOT_APPLICABLE",
  "reason":"vLLM publishes no aarch64/sm_121 wheel; source build unsupported",
  "permanent":true, "decided_by":"perf-matrix.yaml", "decided_on":"2026-08-25" }

{ "host":"mini", "band":16, "comparator":"llamacpp", "status":"UNMEASURED",
  "reason":"16 GB unified memory — OOM at c=16 with Q4_K_M",
  "permanent":false, "expires":"2026-11-25", "owner":"@noah", "ticket":"PERF-012" }
```

#### 4.7.2 Arm applicability

`NOT_APPLICABLE` / `UNMEASURED` apply to **Arm B only**. Arms A, C, D and E need no comparator, so a cell may be `NOT_APPLICABLE` for B and fully measured for the rest. **This is the structural payoff of §2.1** and it is what keeps gx10, intel and mini in the gate today rather than after the #2696 decision.

#### 4.7.3 Expiry semantics

| Condition | Effect |
|---|---|
| `NOT_APPLICABLE`, permanent, reason + `decided_by` present | Excluded from the verdict. Arms A/C/D/E still required |
| `NOT_APPLICABLE` missing reason or `decided_by` | **Schema-fatal.** Receipt rejected — not a FAIL, an invalid receipt |
| `UNMEASURED`, unexpired, owner + ticket present | Excluded from the verdict; host annotated `INCOMPLETE`; release **WARN** |
| `UNMEASURED`, **expired** | Fails **that band**; under min-of-bands the host verdict is FAIL. Not a ratio of `0.0` (a fabricated `0.0` pollutes history) and not schema-fatal |
| `UNMEASURED` missing owner, ticket or expiry | **Schema-fatal** |

A host with an expired `UNMEASURED` cell still reports its measured bands. The verdict is FAIL; the data is not discarded.

### 4.8 Verdict function

```
band_verdict(host, c)  = min(A, B1, B2, C [, D, E])   over applicable arms
host_verdict(host)     = min over bands
release_verdict        = min over hosts
```

**Min. Geomean is deleted from this document** — not demoted to reporting, deleted, because a reported geomean beside a min verdict is the number people will quote.

### 4.9 Hosts, comparators, transport

| Host | Silicon | Arms A/C/D/E | llama.cpp | ollama | vLLM | CI role |
|---|---|---|---|---|---|---|
| lambda-4090 | x86_64, sm_89, 24 GB | ✅ all bands | ✅ CUDA | ✅ | ✅ | **not a CI runner** — retired 2026-05-10, do-not-revive |
| gx10 | aarch64, GB10 sm_121, 120 GB unified | ✅ all bands | `UNMEASURED`, expires 2026-09-25, @noah, PERF-011 | `UNMEASURED` | `NOT_APPLICABLE` — no aarch64/sm_121 wheel; source build unsupported | receipt producer |
| intel | x86_64 Xeon W-3245, AVX-512 | ✅ all bands | ✅ CPU | ✅ | `NOT_APPLICABLE` — requires CUDA/ROCm; no supported CPU-only serving path at the pinned version | clean-room runner — **contended** |
| mini | arm64 macOS, M4, 16 GB unified | c=1,4 ✅; c=8,16 `UNMEASURED` | ✅ Metal | ✅ | `NOT_APPLICABLE` — no macOS/Metal support | selective |

gx10 requires `-DGGML_CUDA_ARCHITECTURES=121`; omitting it JITs from stale PTX — a large, **silent** loss that reads as "GB10 is just slow."

#### 4.9.1 Receipt transport — resolves `APR-BENCH-RFC-001` §7.3

Two of four hosts are not general CI runners and `lambda-labs` — the only fully-comparated host — is do-not-revive. The gate cannot be a job that runs *on* the hosts.

Per `APR-QUALITY-001` v1.8 §0.7 J3: **hosts push signed receipts** (forjar cron), and the blocking job runs anywhere and verifies **signature + freshness**. Cite J3; do not re-litigate.

The **staleness arm is what makes it a gate**. Without `receipt.commit ⊇ commit-under-test`, `evidence/` is a declared-state artifact.

#### 4.9.2 intel contention

`intel` is the clean-room CI runner (8 concurrent, memory-bound) and clean-room is the **release hard gate**. A perf run there contends with the gate that authorises publishing.

- Dedicated single-agent label, **not** one of `intel-clean-room-{1..16}`.
- A `paiml/infra` change: `machines/intel/forjar.yaml` + `forjar apply`, then `make -C machines/intel deploy-systemd-units` and `verify-systemd-units`. **Repo edits are inert until deployed** — the 2026-04-26 ENOSPC outage was a 7-day silent desync of exactly this kind.
- Admission jidoka per `APR-QUALITY-001` J2: refuse start when free space < 1× p95 run size; andon below 2×.
- Concurrency group **shared with clean-room** on intel (I-7).

#### 4.9.3 The accelerator contract, and release posture for #2696

Decoupled from the gate by Arm A — ship the gate on all four hosts regardless of how this resolves.

**Runtime contract (adopted, v2.2).** Both comparators default to GPU when the build carries a backend `[V]`. So do we, with three rules the comparators do not all follow:

1. **Default is `--gpu-layers auto`**, GPU-first, hybrid offload when the model does not fit. Matches llama.cpp `-ngl auto` + `-fit on`.
2. **The boolean `--gpu` is retired** (PERF-021). Ship `--gpu-layers {N|auto|all|0}`, `--device`, `--list-devices`. `--gpu` survives as a deprecated alias for `all` that hard-fails (exit 9) when no backend is loadable. *A boolean is a request with no observable resolution; a quantity forces the system to state what it resolved to.*
3. **An explicitly-set argument is never modified by auto-fit** (I-17). This is the llama.cpp rule verbatim and it is defect #1's root cause stated as a mechanism rather than a symptom.
4. **Never silently fall back.** Requested and resolved are both reported, in the banner, `/health`, and the receipt (§4.4.9). Ollama's silent fallback is an open issue with a 500+ result tail `[V]` — a documented failure mode, not a design to copy.

**Distribution (open, O-1).** Rules 1–4 change nothing if the artifact carries no backend. Note how the comparators actually solve this: llama.cpp ships **per-backend prebuilt release binaries** plus ggml dynamic backend loading; ollama ships **one installer bundling per-backend runner libraries** loaded at runtime (`libggml-cuda.so` vs `libggml-cpu-haswell.so` in its own logs). **Neither uses a source-compiling package registry as its primary channel** (N7).

| Option | Cost | Field precedent |
|---|---|---|
| **(a)** Runtime backend loading — thin `apr` + separately-shipped backend `.so`/`.dylib` | Large; cross-crate, touches dispatch | **Both comparators.** The honest long answer |
| **(b)** Prebuilt per-backend binaries via GitHub releases + installer; `cargo install` demoted to the CPU/dev path, README says so | Moderate; adds a release surface that interacts with the clean-room gate | llama.cpp |
| **(c)** `cargo install aprender --features cuda` only | Cheapest, honest, **poor conversion** — requires a local CUDA toolkit | **Nobody.** This is #2696's current implied answer |

**(c) is the only option with no precedent in the surveyed field.** It deserves a five-whys before being accepted as the destination rather than the stopgap. Whichever is chosen:

- **`README.md` states plainly that the crates.io artifact is CPU-only.** No hedging.
- Any documented `--features cuda` path is honest only once **clean-room Mode A passes with `--features cuda`** and Mode C (lambda-labs, `rust-cuda:1.89`) is green. Documenting an install command never exercised from a clean room is defect #1 in a different costume.
- Jidoka is landed and ships immediately (§10.1 step 1): `--gpu` on a build with no GPU backend exits 9 with a working remedy. **It is necessary and not sufficient** — see §4.4.9.

### 4.10 Invariants

| # | Invariant | Mutation → RED |
|---|---|---|
| I-1 | Expected cell set enumerated from committed `perf-matrix.yaml`; verdict job asserts every cell present | delete one cell's receipt |
| I-2 | `provenance.compute_class` is the dispatch path **taken**, read from the running process — never the hardware present. **`gpu_layers_resolved` is read from the loader and is never inferred from `gpu_layers_requested`** | report `cuda` on a CPU-only build; separately, report `resolved == requested` on a partial offload |
| I-3 | No `ratio` is representable without a `baseline` object that itself passes every receipt rule | emit a ratio with a bare scalar baseline |
| I-4 | Raw samples retained on every cell; summary-only receipts rejected | strip the samples array |
| I-5 | `timeouts > 0` on any band is fatal to that host's ratio | inject one timeout |
| I-6 | No wall-clock ratio is a **merge**-phase check | promote Arm A to `ci / gate`'s required set |
| I-7 | One global `concurrency: perf-gate`, `cancel-in-progress: false`, **shared with clean-room on intel** | launch two full runs; both must not proceed |
| I-8 | The comparator pin carries an expiry and annotates when stale | set expiry in the past |
| I-9 | **Replicates are declared in advance and all are retained.** `N = 3`; verdict is the median of all N; a failed replicate is never discarded or re-run to green. Re-running is legal only as a *new* cell with a new timestamp, both persisting | discard one replicate and re-run; the shortened set must be rejected |
| I-10 | Receipt signature valid **and** `receipt.commit ⊇ commit-under-test` | present a receipt one commit stale |
| I-11 | Cross-host aggregation and any comparator ratio refuse to execute on join-key mismatch — **including a `tokenization` block mismatch** | compare two lanes with differing `tokenization.method` |
| I-12 | A number in `README.md`/`book/`/`docs/` is legal iff it cites an `evidence/` path. **A comparator ratio is illegal regardless. A third-party `[X]` figure is illegal regardless** | add `"36.9× over FasterTransformer"` citing the Orca paper → still red |
| **I-13** | `tokenization.method` has **no default**; its absence is schema-fatal | omit the block; receipt must be unparseable, not defaulted |
| **I-14** | No request is issued at or after window close `T`; all pre-`T` requests are drained; `drain_ms` recorded | issue one request after `T` and count its tokens |
| **I-15** | The comparator is driven by the **same client binary** as apr, on the same workload file | measure the comparator with `llama-bench`; the ratio must be refused |
| **I-16** | `max_in_flight` is reported by the **server**, not inferred by the harness | have the harness compute it from request timestamps |
| **I-17** | **An explicitly-set device or layer argument is never modified by auto-fit.** `autofit_applied[]` and the explicit-argument set are disjoint | set `--gpu-layers 12`; have auto-fit raise it to 32 → red |
| **I-18** | No boolean accelerator flag exists in the CLI surface. Every accelerator request is a quantity or a device list, and every one has a reported resolution | add a `--gpu` boolean with no corresponding `gpu_layers_resolved` field |

---

## §5 Mutation registry

**A gate with no registered mutation is inadmissible.** v1.0's "fed it the real data, every band FAILs" is **replay**: red on known-bad input, not red for the right reason and not green on a no-op.

Every row requires: named mutation → **RED**; pre-fix **GREEN** as before-evidence in the PR body; discrimination check (no-op rebuild stays GREEN).

**The Status column is a CLOSED vocabulary (PERF-047), and `scripts/check_mutation_registry.sh` enforces it.** Prose statuses are how this table rotted: `**not written**` sat beside three guards that had shipped, carried case tables and were named twice each in `ci.yml`, and nothing compared the words to the tree. Every row now opens with exactly one of:

- **PROVEN** — the mutation named in this row was applied, the gate went RED, the row's discrimination case stayed GREEN, and the revert went GREEN. The Status cell carries the counts.
- **PARTIAL** — a mutation exists and discriminates, but it is **not the one this row names**, or it covers only part of the arm. The Status cell says which half is uncovered. A PARTIAL row is not admissible evidence for the half it does not cover.
- **UNCOVERED** — the file this row names ships a case table that a workflow runs, and **this rule is not in it**. The Status cell must quote the mutation that leaves the table green. This is the honest middle: a self-test exists and does not reach here. No row uses it today — cell completeness did, for the hour between finding the hole and closing it — and it exists so the next one has somewhere honest to sit instead of `**not written**`.
- **UNPROVEN** — no mutation turns this gate RED.

The rule is two-way, and both halves are the drift PERF-047 measured. A PROVEN, PARTIAL or UNCOVERED row must name, in backticks, a file that exists in the tree — a row claiming anything about a mutation must name something a person can run. An UNPROVEN row may **not** name a file whose self-test a workflow invokes: if a table exists and skips this rule, the status is UNCOVERED with the evidence, not silence. And UNCOVERED may not be used where no such table exists, or it becomes a softer spelling of UNPROVEN.

| Gate / control | File `[C]` | Mutation → RED | Discrimination | Status |
|---|---|---|---|---|
| Arm A scaling | `scripts/perf_gate.sh` | delete BOTH c=1 clauses (`1 not in bands` and `agg(1) missing or zero`) | `baseline_healthy` green | PARTIAL — 19 → 18/19, `band_c1_absent` breaks. Deleting either clause alone leaves 19/19: they catch the same receipt, which is defence in depth rather than a hole. The band-presence half is proven; the **`scaling_efficiency < floor` comparison and the UNMEASURED-`expires` branch are UNREACHED** — neutering either leaves 19/19, because all 8 cells are `UNMEASURED` with a future expiry. v2.2's "cap the in-flight semaphore to 1" is a product mutation nothing implements. |
| Arm B1 aggregate floor | `scripts/perf_gate.sh` | inject `agg_ratio = 0.79` | `0.80` green | PROVEN — `b1_aggregate_below_floor` / `b1_aggregate_at_floor`; neutering `ag<b1` gives 17/19 (it also breaks `serialization_shape_rejected`). `ci.yml:1011`. |
| Arm B2 decode parity | `scripts/perf_gate.sh` | inject `decode_ratio = 0.99` | `1.00` green | PROVEN — `b2_decode_below_floor` / `b2_decode_at_floor`; neutering `de<b2` gives 18/19. |
| **Arm D kv_utilization** | `scripts/perf_gate.sh` | omit `kv.bytes_used` / `bytes_reserved` at `--phase release` | present → green | PARTIAL — `armDE_absent_is_fatal_at_release` / `armDE_present_passes_release`; neutering **both** release exits gives 17/19. Arm D applies **no bound** until PERF-001, so "reserve `max_seq_len` per request" cannot turn it RED by construction. Only field presence can, and that is what is proven. |
| **Arm D swap ratchet** | `scripts/perf_gate.sh` | omit `kv.preempted_swap` at `--phase release` | present → green | PARTIAL — same two cases. No swap ratchet exists; the threshold arrives with PERF-001. |
| **Arm E interference** | `scripts/perf_gate.sh` | omit `itl` / `injector` on W2 at `--phase release` | W1 skips, W2 present → green | PARTIAL — `armE_absent_is_fatal_on_w2` / `armE_skipped_on_w1_with_d_present`; neutering **both** release exits gives 17/19. Same reporting-arm limit as Arm D. |
| Arm C integrity | `scripts/perf_gate.sh`, `scripts/lib/bench_receipt.py` | `completed = requested − 1` | equal green | PROVEN — `completed_lt_requested` / `baseline_healthy`; neutering the comparison gives 18/19. Enforced in `perf_gate.sh::arm_c_integrity`, **not** in `bench_receipt.py`, which returns rc=0 on the same receipt. |
| Band-completeness poka-yoke | `scripts/perf_gate.sh` | omit `aggregate_tok_per_sec` at c=1 | full band green | PARTIAL — measured directly: the gate goes RED (`FAIL ArmA agg(1) missing or zero`) and the full band PASSes. But **no self-test row covers it**, so a regression is invisible to CI, and it is a DETECTOR rather than a poka-yoke: `bench_receipt.py` returns rc=0 on the same receipt. §6's Poka-yoke row is corrected below. |
| **Tokenization poka-yoke** | `scripts/perf_gate.sh`, `scripts/lib/bench_receipt.py` | omit `tokenization.method` | present green | PROVEN — `tokenization_absent` / `baseline_healthy`; neutering the check gives 18/19. |
| **Drain rule** | `crates/aprender-test-lib/src/perf_gate/drain.rs`, `scripts/perf_gate.sh` | accept a request issued at or after `T` (`issued_ms >= window_ms` → `false`) | pre-`T` only green | PROVEN — `cargo test -p aprender-test-lib --lib perf_gate::` goes 34 → 33/34 on `a_request_issued_at_or_after_t_is_refused`, revert 34/34. `drain_ms` absence is separately covered by `drain_ms_absent` (neuter → 18/19). |
| Receipt table (23 cases) | `scripts/check_parity_receipt.sh` | per existing table | the 6 valid cases stay valid | PROVEN — 23 cases, 6 valid / 17 invalid, one-sidedness asserted in both directions. **Named by no workflow**: it is on `scripts/unwired_guards_baseline.txt` and runs only in the release dogfood sweep via `[package.metadata.dogfood]`. |
| Claim-literal guard | `scripts/check_no_claim_literals.sh` | add `"2.93× Ollama"` to `book/` | unrelated prose green | PROVEN (PERF-049, #2758) — rc=0 → **rc=1** naming `book/src/tools/apr-cli.md:1786`; unrelated prose in the same file rc=0; revert rc=0. **The mutation this cell names left the guard GREEN until PERF-049**: `RATIO_RE` matched ASCII `x` only, so the U+00D7 spelling — the one the book actually publishes — was unreadable, and this row was recorded as proof by a mutation that did not bite. 27 ratio case rows now assert both spellings; `ci.yml:969`, `:971`. |
| **`[X]` figure guard** | `scripts/check_no_claim_literals.sh` | add `"36.9×"` to `docs/` | unrelated prose green | PROVEN (PERF-049, #2758) — rc=1 for `36.9× over FasterTransformer`, `36.9x over FasterTransformer`, `23x over static batching` and `1.8x over vLLM`; a line carrying `3x3 matrix`, `2x2 grid`, `1024x1024` and `v1.8x` beside `llama`/`torch` stays rc=0. **One intervening word used to defeat the adjacency, and that is the spelling §0.1 above uses**, so the guard was blind to the exact form this document writes. |
| Fabricated-baseline guard | `scripts/check_no_fabricated_baselines.sh` | `${OLLAMA_BASELINE:-137}` in a **new** file | unrelated shell edit green | PROVEN — rc=1 naming the new file, rc=0 on the control; `OLLAMA_TPS=137` also rc=1, so it is shape recognition and not string equality against `291`. 52 case rows run in front of the verdict on **every** invocation, so the absent `--selftest` flag is a design choice, not a gap. |
| Competing-harness guard | `scripts/check_no_competing_harnesses.sh` | re-add `scripts/gpu_2x_benchmark.sh` (`git show 64cb68177^:…`, 171 lines, untracked) | unrelated script green | PROVEN — rc=1, `COMPETING scripts/gpu_2x_benchmark.sh`, count=1 > baseline=0; the control script rc=0. The mutation is landed **untracked**, so the tracked-∪-working-tree universe is exercised too. 9 case rows, `ci.yml:989`. |
| Claims-cite-receipts | `scripts/check_perf_claims_cite_receipts.sh` | uncited speed comparison in `docs/` | the same claim citing a real `evidence/` path green | PROVEN — rc=1 uncited, rc=0 cited, rc=1 on a dangling citation. 12 case rows, `ci.yml:985`. |
| **Comparator-harness guard** | `scripts/check_no_competing_harnesses.sh` | a script driving `llama-bench` and printing tok/s | `scripts/parity_host_receipt.sh`, the same-client producer, green | PROVEN — rc=1 on the probe, rc=0 on the tree. It lives in the competing-harness predicate (`starts_server` lists `llama-bench`), **not** in `perf_gate.sh` as v2.2 recorded. Residual: the same-client producer stays green because it computes no rate itself, not because it is allowlisted. |
| Resolver contract | `scripts/apr_bin.sh`, `scripts/llama_bin.sh` | replace `return 1` with `exit 1` | caller survives on `return` | UNPROVEN — nothing mutates it. `check_sourced_libs_option_neutral.sh` covers file-scope `set` and `check_apr_bin_resolution.sh` covers resolution ORDER; neither is this property. |
| Jidoka `--gpu` | `crates/apr-cli/src/commands/serve/mod.rs`, `contracts/accelerator-request-v1.yaml` | delete `ensure_accelerator_available(config)?;` from `run()` | CPU run without `--gpu` still succeeds | PROVEN — `cargo test -p apr-cli --lib commands::serve::` goes 252 → 251/252 on `the_guard_is_actually_wired_into_run`, revert 252/252. F-ACCEL-001..004 record the build-conditional halves. The call site is the gate: every unit test survives deleting it, which is why that one test reads the source. |
| **Explicit-wins (I-17)** | `crates/apr-cli/src/commands/serve/mod.rs`, `contracts/accelerator-request-v1.yaml` | `fits >= total_layers` → `true`, and `n <= fits` → `true` | `auto` still fitted → green | PROVEN — 252 → 251/252 on `an_explicit_request_that_does_not_fit_is_an_error`, `auto_resolves_to_what_fits` stays green, revert 252/252. F-ACCEL-005. |
| **Resolved-vs-requested (I-2)** | `crates/apr-cli/src/commands/serve/mod.rs` | emit `gpu_layers_resolved = gpu_layers_requested` on a partial offload | full offload, equal → green | UNPROVEN — `gpu_layers_resolved` appears nowhere but in `receipt.rs`'s `unproduced_fields` list. The field this mutation would corrupt is not emitted yet, so the mutation has nothing to act on. |
| **No-boolean-flag (I-18)** | `crates/apr-cli/src/commands/serve/mod.rs`, `contracts/accelerator-request-v1.yaml` | `wants_layers = false` | `--gpu-layers 0` still not an accelerator request → green | PROVEN — 252 → 251/252 on `gpu_layers_is_refused_on_a_build_with_no_accelerator`, revert 252/252. F-ACCEL-006. **I-18's wording claims more than is proven**: the boolean `--gpu` still exists as the deprecated spelling of `all`. What is enforced is that both spellings reach one refusal. |
| Ratchet direction | `scripts/lib_baseline_ratchet.sh`, `scripts/check_baseline_ratchets.sh` | append one entry cloned from a baseline's own last real entry; and, for `set-aperture`, record a line this branch WROTE | delete an entry → green; a reveal that predates the comparand → green | PROVEN (PERF-049, #2758) — 42 case rows, `ci.yml:472`. Six named mutations run: forcing the aperture-moved test true, neutering the byte-identity comparison, neutering the coordinate parse, and silencing the admitted-entry list each take the table from PASS to a named FAIL; a comment reflow stays green. Neutering the coordinate parse was GREEN on the first pass — `pre.md:x` is refused one branch later — so the row that only it catches (`pre.md:$`, sed's last-line address) was added and the mutation now bites. |
| **Cell completeness** | `scripts/perf_gate.sh` | `cell_completeness`'s `sys.exit(1)` → `sys.exit(0)` | a full band set → green | PROVEN — 19 → 18/19 on `cells_missing_bands_at_release`, control `cells_complete_at_release`. This row did not exist in v2.2 and the arm was exercised by nothing: every fixture carried all four bands, so the mutation left the table at 17/17. PERF-047 added both cases. The registry was incomplete, not merely stale. |
| Staleness arm | verdict job (§4.9.1) | receipt one commit stale | fresh receipt green | UNPROVEN — no verdict job exists in `.github/workflows/`, and `receipt.commit ⊇ commit-under-test` is unimplemented. |

**Coverage: 15 PROVEN, 7 PARTIAL, 3 UNPROVEN of 25** (v2.2 recorded "1 of 24"; v2.1 1 of 21, v2.0 1 of 14). The denominator grew by one because the registry was missing a row, not because the gate grew. Target **25 PROVEN** before this document leaves DRAFT.

**PERF-047 audit — the drift, measured in both directions (#2752, against `origin/main` at `31732f5db`, 2026-08-29).**

- **Recorded as proof, not proven: 2.** *Band-completeness poka-yoke* named `bench_receipt.py` as the enforcer and was marked `claimed, unverified`; `bench_receipt.py` returns rc=0 on a receipt whose c=1 band has no `aggregate_tok_per_sec`, and no self-test row covers the rule at all. *Claim-literal guard* was marked with a case-table count and named a mutation — `"2.93× Ollama"`, with U+00D7 — that leaves the guard GREEN. This is the dangerous direction: a row read as evidence for a gate that, on the input the row itself names, cannot fail.
- **Recorded as unproven, actually proven: 12**, plus 5 more where a discriminating mutation exists for a property adjacent to the one named (now PARTIAL). Three of the twelve — competing-harness, claims-cite-receipts, `[X]` figure — were `**not written**` beside guards that ship case tables and are invoked twice each in `ci.yml`. *Ratchet direction* was `**missing**` while `check_baseline_ratchets.sh --self-test` runs 31 rows at `ci.yml:472`. *Jidoka `--gpu`* was `mutation unrecorded` while the test that carries it names this registry row in its own doc comment.
- **Rows the registry did not have: 1** — cell completeness. It gates release and was exercised by nothing: mutating its `sys.exit(1)` left the case table at 17/17. PERF-047 added the row, the mutation case and its control, so it lands PROVEN rather than recorded.

The asymmetry matters. An understated row costs credit; an overstated one is a gate that cannot fail, wearing a citation. Both were present here, which is why the registry is now checked by `scripts/check_mutation_registry.sh` rather than by whoever reads it next.

One more case-table row was green for the wrong reason and is fixed here: `zero_token_response` built its fixture with a stray `}`, so it was RED on a JSON parse error rather than on a zero-token band — neutering the zero-token rule itself left the table at 17/17. The fixture now parses, and the same mutation gives 18/19. `ci.yml:1025` runs the registry guard and `:1027` its case table.

**On the fabricated-baseline guard.** A guard that knows the literal `291` proves string equality, not shape recognition; the mutation therefore uses a **different** default, and `${OLLAMA_BASELINE:-137}` and `OLLAMA_TPS=137` both go RED in a file the guard has never seen. v2.2 recorded two outstanding holes — the `"Using default Ollama baseline (318 tok/s from spec)"` branch and the `225.0 // Ollama parity` literals in `crates/aprender-serve/src/gguf/tests/parity*.rs` `[C]`. Both are now inside the shrink-only ledger `scripts/fabricated_baseline_rust_sites.txt` (73 lines, 36 Rust sites); the note is retained because ledgered is not fixed.

**What the registry guard still cannot check.** It reads the table against the tree: that a claiming row names a file that exists, that an UNPROVEN row is not sitting beside a self-test a workflow runs, and that the unproven set only ever shrinks against `origin/main`. It cannot read the Mutation cell and decide whether that English sentence describes something real — the `2.93×` row is exactly that failure, and it was caught by running the mutation, not by any parser. The counts in a PROVEN cell are a human claim; the standing rule is that a PR moving a row to PROVEN quotes the RED, the discrimination and the revert in its body.

---

## §6 Toyota Way mapping

| Concept | Here | Mechanism |
|---|---|---|
| **Jidoka** — the product stops itself | `--gpu` on a build with no GPU backend fails rather than running on CPU. It stops the **product on the user's machine**, not just CI | `serve/mod.rs::ensure_accelerator_available` — landed, exit 9 with a working remedy; **mutation unrecorded** |
| **Andon** — one visible signal | one `compute_class()` feeding banner, `/health`, `provenance.compute_class`; `max_in_flight`, the server declaring its own concurrency ceiling; **and `gpu_layers_requested` vs `_resolved`, the machine raising its own lamp when it gave the user less than they asked for** | PERF-006, PERF-021; §4.4.9 |
| **Poka-yoke** — unwriteable, not detected | a lane must carry `bands`; `tokenization.method` has no default; a ratio must derive from that band's own samples; a lane can never be greener than its worst band. **"A band must carry every metric" is NOT poka-yoke** — see the PERF-047 correction below | `scripts/lib/bench_receipt.py` `[C]`, `crates/aprender-test-lib/src/perf_gate/` `[C]` |
| **Genchi Genbutsu** — go and see | the gemba is the clean-room-installed artifact on four hosts; the receipt is emitted by the process that installed it. **Not byte-identical to the user's binary — see X-E1** | Mode A A4 + `check_multiplatform_dogfood.sh` |
| **Standardized work** | one entrypoint, one receipt schema, one verdict, one client for both servers — competing harnesses **deleted**. SGLang RFC #9808 is the counterfactual `[X]` | §9 — **deletion staged on branch, unshipped** (E1) |
| **Kaizen** — a ratchet that cannot slip | comparand from protected `origin/main`; A and D up-only, D-swap and E down-only, claim literals down-only | §4.6.3 |
| **Heijunka** — level the load | perf never contends with the release gate on `intel` | §4.9.2 |
| **Muda** — waste made visible | `kv_bytes_reserved − kv_bytes_used` is reserved-but-never-used VRAM: inventory, in the TPS sense, expressed in bytes | Arm D |

**P7 correction, retained.** `check_perf_claims_cite_receipts.sh` is a **detector** — it finds a bad number after someone wrote it. The poka-yoke is the receipt schema: a band missing a metric is *unrepresentable*. Both are needed; conflating them inflates how much of the problem is mistake-proofed. Detectors can be bypassed; unrepresentable states cannot.

**PERF-047 correction (#2752).** The row above used to claim "a band must carry every metric" as an unrepresentable state. Measured: a receipt whose `c=1` band has no `aggregate_tok_per_sec` is accepted by `scripts/lib/bench_receipt.py` with rc=0, and is caught downstream by `perf_gate.sh`'s Arm A — `FAIL ArmA agg(1) missing or zero`. That is a detector, not a poka-yoke, and the same conflation P7 corrects one paragraph up. What IS unrepresentable is on the **producer** side: `crates/aprender-test-lib/src/perf_gate/` derives `drain_ms` and the request counters from per-request terminal records and offers no constructor that accepts a `drain_ms` scalar. Poka-yoke lives where the number is made, not where it is read.

---

## §7 Five Whys

### 7.1 Defect #2 — why it serialises

v1.0 asked why nobody *noticed*. It never asked why it *serialises*. You cannot schedule a fix with no root cause and no scope estimate.

The §2 data constrains it hard: `apr agg(c) ≈ agg(1)` at every band, and `serialization_index(c) ≈ c` exactly. That is not a small batch window or a scheduler-tuning problem. It is **one lock held for the duration of a request**, admitting exactly one. The panel's Principal Systems Engineer and the external reviewer reached the same reading independently.

**One hypothesis is now eliminated.** LMDeploy's TurboMind exists partly to bypass Python-managed dispatch overhead present in Python-hosted engines `[X]`. aprender is Rust end-to-end and is structurally in TurboMind's class, not vLLM's. **Interpreter dispatch overhead cannot explain 0.075.** That removes a plausible and expensive wrong turn.

**And this reframes defect #3.** `--batch` **exists** and **hangs** at 9m50s. If a batch path is implemented, the fix is probably not "build continuous batching" — it is "the batch path deadlocks against the lock the serial path holds." Weeks versus days.

**Hypothesis, to be falsified before any batching work is scheduled (PERF-000):**

> Run `apr serve run` under 4 concurrent requests; capture a thread dump / `tokio-console` snapshot at t = 60 s. **If all four tasks are parked on one mutex (model handle or KV-cache arena), defects #2 and #3 are one defect and one ticket.**

Recording the prediction before the measurement is the point. If the dump shows four independent tasks progressing, this section is wrong and says so.

#### 7.1.1 Ordered workstreams, once PERF-000 answers

Three distinct problems, frequently conflated. The external review's chief contribution is separating them and naming the prior art for each:

| Order | Problem | Prior art | Ticket |
|---|---|---|---|
| 1 | **Admission serialization / deadlock.** One request at a time; `--batch` hangs | none needed — this is a lock bug | PERF-000 → PERF-002 |
| 2 | **Iteration-level scheduling.** Evict on EOS, inject a waiting request before every forward pass, no head-of-line blocking | **Orca, OSDI 2022 (Yu et al.)** — the canonical design | PERF-001 |
| 3 | **Paged KV cache.** Fixed blocks + per-request block table; on-demand allocation instead of `max_seq_len` reservation | **PagedAttention, SOSP 2023 (vLLM)** | PERF-018 |
| 4 | **Chunked prefill.** Bound prefill work per iteration; interleave with decode | vLLM, TGI v3, SGLang | PERF-017 |

Doing 2 without 3 raises `scaling_efficiency` while capping real concurrency on memory. Doing 2 and 3 without 4 fixes throughput and destroys interactive ITL. **Arms D and E exist so the gate can tell these apart** — without them, one green Arm A would certify all three states as identical.

### 7.2 Defect #2 — why nobody noticed

1. **Why 0.097× at c=16?** The published binary serves one request at a time.
2. **Why did nobody notice?** Every published number is measured at concurrency 1 (`scripts/llama_pin.toml`: `http_concurrency = 1`).
3. **Why has the one sweep harness never run?** `scripts/benchmark-matrix.sh` accepts `--batch-sizes 1,8,16,32`; a repo-wide search across `.github/`, `Makefile`, `*.toml` returns **zero hits** `[C]`.
4. **Why is an unwired harness invisible?** `unwired_guards_baseline.txt` enumerates *guards*, not measurement harnesses.
5. **Why does no schema require a claim to name the workload it is a claim about?** `PROVENANCE_REQUIRED` had no workload field.

### 7.3 Defect #4 — 2.93× Ollama from a harness that never ran Ollama

1. **Why does the book publish it?** The ratio was computed against a default baseline constant.
2. **Why is there a default baseline?** So the harness would run on hosts without Ollama installed.
3. **Why is unmeasured output indistinguishable from measured?** The ratio comes from a bare scalar with no provenance.
4. **Why did it reach the book?** `readme_contract.rs` is wired and covers `README.md`; the claim lives in `book/`.
5. **Why do claims and receipts live in disjoint universes?** No index maps a published number to the receipt that produced it.

### 7.4 One countermeasure, not two

v1.0 answered 7.2 with a `workload` schema field and 7.3 with a claims↔receipt index — two mechanisms for one root cause, and the 7.2 fix is the same *class* as the thing that failed.

**Consolidated:** one **claim↔receipt join**. The `workload` + `tokenization` objects are the receipt-side half; `check_perf_claims_cite_receipts.sh` is the claim-side half; I-12 is the rule. One mechanism to keep alive.

### 7.5 Defect #1 — why `--gpu` was silently ignored

1. **Why does `--gpu` run on CPU?** The flag parses; nothing asserts the backend it names is present.
2. **Why is there no assert?** Dispatch degrades gracefully by design, and graceful degradation was never distinguished from *requested* acceleration.
3. **Why was the degradation invisible?** `--gpu` is a **boolean**. A boolean request has no observable resolution: honoured and ignored produce byte-identical output. llama.cpp has no such flag — `-ngl` is a quantity and the loader must report how many layers it placed `[V]`.
4. **Why is a request allowed to have no resolution?** Because the system was designed to *decide* rather than to *report*. Automatic device selection was written as a policy engine, not as a negotiation with a stated outcome.
5. **Why was the user's explicit instruction overridable at all?** No rule said it was not. llama.cpp's auto-fit disables itself for any argument the user set explicitly `[V]`; aprender had no equivalent, so an explicit `--gpu` and an unset default were treated identically by the resolver.

**Root cause (v2.2, replacing v2.1's schema-gap reading):** *automation overrode an explicit user instruction, and the override was unobservable.* Both halves are required for the defect. Either alone is survivable — an unobservable *default* is merely quiet; an observable *override* is merely annoying.

**Countermeasures, in dependency order:**

| # | Countermeasure | Status |
|---|---|---|
| 1 | **Explicit-wins rule** — auto-fit never modifies a user-set argument (I-17) | **PERF-021**, not written |
| 2 | **Retire the boolean** — `--gpu-layers {N\|auto\|all\|0}` + `--device` + `--list-devices` (I-18) | **PERF-021**, not written |
| 3 | **Report the resolution** — `gpu_layers_requested` / `_resolved` / `_total`, `backend_loaded[]`, `autofit_applied[]` in banner, `/health`, receipt (§4.4.9, I-2) | PERF-006 |
| 4 | Exit 9 when the requested backend is entirely absent | **landed** — necessary, not sufficient |
| 5 | Measurement target = clean-room-installed artifact (§4.2.1) | step 4 |
| 6 | ollama-style **GPU-less runner** device test at merge phase | PERF-005 |

Countermeasure 4 is the one that landed first because it was the cheapest. It catches the *absent-backend* case only. Countermeasures 1–3 catch the *present-backend, partial-offload* case — which is ollama's open issue #14258 and, on a machine that does have CUDA, the far more common one `[V]`.

---

## §8 Repo migration — assimilate `qwen-coder-deploy`, then archive

`~/src/qwen-coder-deploy` (public, last pushed 2026-03-29) `[C]`: 5 runtimes × 5 hosts, forjar deploy/bench/**teardown** isolation, `docs/specifications/benchmarking-v2.md`, `perf-parity-spec.md`, 942 result JSONs.

We had **more of its code than its practice**: `crates/aprender-test-lib/src/llm/` already implements tail percentiles, jitter CV, drift and scoring anchors, with **zero callers** `[C]`.

| # | Decision | Authority |
|---|---|---|
| P1 | **No `git subtree`** — curated copy, source SHA per landed file. Subtreeing 1,214 files / 312 MB / 475 commits puts every dropped blob in every clone forever. History is preserved by the archived read-only repo, which is what archiving is *for* | deviation from APR-MONO Phase 2, flagged |
| P2 | Specs → `docs/specifications/aprender-serve/`, components → `.../sub/` | precedent `aprender-compute/sub/` |
| P3 | Contracts → flat `contracts/` | APR-MONO F5 |
| P4 | Evidence → `evidence/qwen-coder-showdown-2026-03-29/` + `findings.json`, **dated by source HEAD** | Convention 8 |
| P5 | Reproducible corpus → `crates/aprender-serve/benchmarks/qwen-coder/`. No root `benchmarks/` | Convention 9 |
| **P6** | **forjar host descriptors → `paiml/infra` `machines/<host>/`, NOT into aprender** | infra Policy 2 + APR-MONO Appendix B |

### 8.1 P6 — ratified, bridge rejected

Four review passes converged on `paiml/infra`. **Ratified, no longer open.**

The operator and the panel's QA lead both proposed a bridge — fetch or symlink descriptors from `paiml/infra` at gate runtime. **Rejected (R1).** A gate that reads mutable remote state is the declared-vs-resolved defect, and an external dependency in the gate path violates the clean-room hard gate.

Uniformity is delivered by the **interface**: one `perf-matrix.yaml`, one receipt schema, one `perf_gate.sh --host <name>`. Under J3 aprender never needs the descriptor — hosts push receipts; the gate verifies signature and freshness.

The narrow in-tree exception — *a recipe provisioning only one crate's own toolchain* — is **explicitly narrowed**: llama.cpp is a **fleet** comparator used by more than aprender and does not qualify.

### 8.2 Archive procedure

1. Land P1–P6; record source SHA in each landed file.
2. Old repo `README.md` → pointer to monorepo paths.
3. GitHub: **Archive** (read-only), do not delete — it is the history P1 relies on.
4. The 942 provenance-free receipts become `UNMEASURED`. **Do not backfill.**

---

## §9 What must be DELETED

Standardized work means the other ways *go*. Deprecation is not enough. SGLang RFC #9808 is the counterfactual: a top-tier project living with `bench_one_batch` vs `bench_serving` divergence in interface, tokenization and token counting `[X]`.

**Status, corrected (E1):** all three below still exist on `origin/main` `[C]`, deleted only on `feat/2692-apr-probar-llm` (PR #2682, "delete 1,086 lines that fabricate their comparator"). **Staged, not shipped.**

- `scripts/benchmark-2x-ollama.sh` — three default baselines (291/120/15)
- `scripts/gpu_2x_benchmark.sh`
- `scripts/benchmark-matrix.sh` — accepts `--batch-sizes 1,8,16,32`, zero references; fold into `apr test llm bench --concurrency`
- `scripts/bench.sh` — still present on the branch, **audit outstanding**

**Enforcement:** `scripts/check_no_competing_harnesses.sh`, shrink-only baseline committed at the **true count today**, merge-phase blocking. Without it, §6's standardized-work row is a claim about a branch.

**Fabrication sites** — outstanding: `"Using default Ollama baseline (318 tok/s from spec)"` and 15+ `225.0 // Ollama parity` literals in `crates/aprender-serve/src/gguf/tests/parity*.rs`.

**Rule going forward:** the comparator is a **required argument**. Its absence is an error, never a default. The rule now names a file (`check_no_fabricated_baselines.sh`) and a mutation (§5).

**Claims — delete, do not soften:** `book/src/examples/showcase-benchmark.md:17,22` · `book/src/tools/apr-cli.md:1396,1493,1498`.

**`qwen-story` — resolved.** Two subjects, not two methodologies. Retained, runs at **merge** phase, **shares the receipt schema and the comparator-required rule**. Separate workflows, one schema.

---

## §10 Sequencing

v1.0 listed "the batching fix itself" as bullet 7 of 9 under *Not started*, with no ticket, owner, EV rank or dependency edge. **The fix for the #1 adoption killer carried the same weight as a shell script.** That was the finding.

### 10.1 Order — gate first, and instrumentation before the fix

The panel argued *"fixing the `--batch` hang is P0 before we even look at gating — we can't gate performance we don't have."* Half accepted: #3 is P0. The sequencing conclusion is rejected, and v2.1 strengthens the reason.

**Gate-first is a precondition for accepting the fix at all.** Continuous batching trades per-request latency for aggregate throughput. Without the joint arm (§3.1 row 2), an implementation that lifts aggregate to 800 tok/s while pushing p95 TTFT to 4 s is **indistinguishable from success**.

**v2.1 adds the sharper form:** Arms D and E must exist *before* PERF-001, because a batching implementation without paged KV, or without chunked prefill, **passes Arm A**. The gate would certify the fix and hide its two successors. §7.1.1's four workstreams are four distinct states of the system, and a gate that cannot distinguish them will call all four green.

§4.1 dissolves the apparent conflict: no ratio is a per-PR check, so landing the gate turns **zero** PRs red. It makes **0.64.0 uncuttable**.

| Step | Item | Phase | Gate on main |
|---|---|---|---|
| 1 | Cherry-pick `ensure_accelerator_available` (jidoka, exit 9) — **isolated PR** | merge | green |
| 1b | **PERF-021 accelerator contract** — retire boolean `--gpu`; `--gpu-layers` + `--device` + `--list-devices`; explicit-wins (I-17); resolved-quantity reporting. Independent of everything below; the completion of step 1, which catches only the absent-backend case | merge | green |
| 2 | Receipt schema v2.2: all metrics, `workload`, `tokenization`, §4.4.9 scheduler block **including `gpu_layers_requested`/`_resolved`/`backend_loaded[]`/`autofit_applied[]`**, three-part identity, drain rule; Arm C; guards; resolver selftest | merge | green |
| 3 | **PERF-000** — falsify the single-lock hypothesis (§7.1) | — | green |
| 4 | `perf_gate.sh` + `perf-matrix.yaml`; Arms A/B release-blocking, **baselined at a §4.4-conformant re-measurement** | release | green; **0.64.0 NO-GO** |
| 5 | **Arms D and E + workload W2, REPORTING** — instrumentation before the fix | release (reporting) | green |
| 6 | PERF-002 / PERF-001 — the fix. **Arms D and E promoted to blocking in the same PR** | release | first release that can cut |
| 7 | PERF-018 paged KV, PERF-017 chunked prefill — each ratchets the arm that measures it | release | — |
| 8 | BENCH-003 → dispersion thresholds → Arm B3 + CI blocking | release | — |

Step 2 is what makes this honest: commit today's measured `scaling_efficiency` up-only. Main stays green, regression becomes impossible, the release stays blocked. Zero invented thresholds, zero red-PR fatigue, zero "gate everyone learns to walk past."

**The intent is to fail releases until batching lands.** The floor is not temporarily lowered. Lowering a floor to permit a release is the defect this document exists to prevent, and Arm A's ratchet makes it mechanically visible at merge phase.

### 10.2 Branch split

`feat/2692-apr-probar-llm` spans ≥5 issues and 8 landed items behind one required `ci / gate`. A single revert loses all eight.

| PR | Contents | Depends |
|---|---|---|
| 1 | `ensure_accelerator_available` only | — |
| 1b | PERF-021 accelerator contract (CLI surface + resolver + reporting) | 1 |
| 2 | `bench_receipt.py`, `check_parity_receipt.sh`, `check_no_claim_literals.sh`, resolver fixes + selftests | — |
| 3 | `apr test llm bench` harness + `llama_pin.toml` banded protocol + one-client rule | 2 |
| 4 | Deletions (#2682) + `check_no_competing_harnesses.sh` with baseline | 3 |
| 5 | `perf-matrix.yaml` + `perf_gate.sh` + staleness arm | 2, 4 |
| 6 | Arms D/E + W2, reporting-only | 5 |

PRs touching `.github/workflows/*` need a web-UI merge click (`gh` token lacks `workflow` scope). Anything ending in a publish runs `make -C machines/clean-room clean-room-p1` first — aprender is P0.

### 10.3 Hoshin targets

Every target is a zero, a one, a 100%, or a ratchet on a measured baseline. No target is an invented continuous threshold.

| # | Metric | Today | Target | Type | Ticket |
|---|---|---|---|---|---|
| H1 | Mutation-verified gates (§5) | 1 / 21 | **21 / 21** | one | PERF-004 |
| H2 | Invented thresholds in this spec | 0 (2 `[U]` sample-size, 1 `[U]` shape) | **0**, `[U]` count → 0 after BENCH-003 | zero | BENCH-003 |
| H3 | `scaling_efficiency(16)`, lambda, W1 | 0.075 `[C]` | re-measure §4.4-conformant, commit, **up-only** | ratchet | PERF-001 |
| H4 | Competing harnesses on `origin/main` | 3 (+1 audit) | **0** | zero | PERF-009 |
| H5 | Fabricated-baseline literals | 16+ | commit true count, **down-only** | ratchet | PERF-008 |
| H6 | Comparator + `[X]` numbers in published prose | ≥5 sites | **0** | zero | PERF-010 |
| H7 | Cells with valid signature + freshness | 0 | **100%** | one | PERF-007 |
| H8 | Defects #2/#3 root cause named | unknown | **1 named lock, or hypothesis falsified** | one | PERF-000 |
| H9 | `apr profile` false causal strings | ≥1 shipped | **0** | zero | PERF-014/016 |
| H10 | PRs turned red by the perf gate | n/a | **0** | zero | I-6 |
| H11 | Verdict functions in this spec | 1 (min) | **1** | one | §4.8 |
| **H12** | `max_in_flight` reported by the server | absent | **present, and > 1 after PERF-001** | one | PERF-006 |
| **H13** | `kv_utilization` (Arm D) | unmeasured | instrument, then **up-only** | instrument→ratchet | PERF-018 |
| **H14** | `itl_p95_ratio` W2/W1 (Arm E) | unmeasured | instrument, then **down-only** | instrument→ratchet | PERF-017 |
| **H15** | Arms that cannot distinguish §7.1.1's four states | 4 of 4 states conflated | **0** | zero | step 5 |
| **H17** | Boolean accelerator flags in the CLI surface | 1 (`--gpu`) | **0** | zero | PERF-021 |
| **H18** | Explicit user arguments silently overridden by auto-fit | unmeasured | **0**, gated by I-17 | zero | PERF-021 |
| **H19** | Offload resolution present in structured output (banner, `/health`, receipt) | 0 of 3 | **3 of 3** | one | PERF-006 |
| **H20** | Distribution channels with field precedent under consideration for O-1 | 1 of 3 evaluated | **3 of 3 costed, one chosen with recorded rationale** | one | O-1 |
| H16 | Days from gate-on-main to first cuttable release | ∞ | measure it — it **is** the batching estimate | instrument | — |

---

## §11 Ticket set

Definition of done, every ticket:

1. Merged to `main` via green `ci / gate`
2. A gate exists that would have caught the original gap
3. The named mutation applied and observed **RED**, with pre-fix **GREEN** as before-evidence in the PR body
4. A `pv` contract (invariants + falsification_tests + Kani where applicable) in the **same PR**
5. Discrimination confirmed — gate stays GREEN on a no-op change
6. Any doc claim the change invalidates updated in the same PR

IDs are placeholders; real IDs come from `pmat work add`. **File the ticket before the branch.**

| ID | Repo | Depends | Title | EV |
|---|---|---|---|---|
| PERF-003 | aprender | — | Cherry-pick jidoka exit 9 to `main`, isolated PR + mutation record | **1** |
| PERF-000 | aprender | — | Falsify single-lock hypothesis; determine whether #2 and #3 are one defect | **2** |
| PERF-004 | aprender | — | Receipt schema v2.1 (all metrics, `workload`, `tokenization`, scheduler block, drain, three-part identity, size budget) + mutation registry | 3 |
| PERF-014 | aprender | — | Delete hardcoded causal string, `profile_print_hotspot.rs:155-162` | 4 |
| PERF-015 | aprender | — | Fix invalid arithmetic, `kernel.rs:531-542` | 5 |
| PERF-005 | aprender | PERF-004 | ollama-style GPU-less device test, merge phase | 6 |
| PERF-009 | aprender | PERF-004 | `check_no_competing_harnesses.sh` + baseline; land deletions | 7 |
| PERF-008 | aprender | PERF-004 | Harden fabricated-baseline guard (shape not literal); commit true count | 8 |
| PERF-010 | aprender | PERF-004 | `check_perf_claims_cite_receipts.sh` + `[X]`-figure guard; delete comparator claims from `book/` | 9 |
| PERF-006 | aprender | PERF-004 | One `compute_class()` + `max_in_flight` → banner, `/health`, receipt (andon) | 10 |
| **PERF-019** | aprender | PERF-004 | One client, both servers; retire `llama-bench` from the comparator path (I-15) | 11 |
| PERF-007 | infra | PERF-004 | Signed receipt push via forjar cron; staleness arm | 12 |
| PERF-013 | infra | — | Dedicated single-agent `intel` label; `forjar apply` + deploy + verify | 13 |
| PERF-011 | infra | PERF-013 | Prove gx10 llama.cpp buildable (`-DGGML_CUDA_ARCHITECTURES=121`); expires 2026-09-25 | 14 |
| **PERF-020** | aprender | PERF-004 | Workload W2 corpus + injector; Arms D and E, REPORTING | 15 |
| BENCH-003 | aprender | PERF-004 | Characterize variance idle + loaded, n≥10; derive dispersion thresholds | 16 |
| PERF-002 | aprender | PERF-000 | `--batch` hang — bounded-time regression test + fix | 17 |
| PERF-001 | aprender | PERF-000, PERF-020 | Iteration-level scheduling (Orca design). **Promotes Arms D and E to blocking in the same PR** | 18 |
| PERF-018 | aprender | PERF-001 | Paged KV cache (PagedAttention design); ratchets Arm D | 19 |
| PERF-017 | aprender | PERF-001 | Chunked prefill; ratchets Arm E | 20 |
| PERF-012 | aprender | PERF-004 | mini c=8/c=16 — measure or re-decide `NOT_APPLICABLE`; expires 2026-11-25 | 21 |
| PERF-016 | aprender | PERF-014/015 | Audit `apr profile` whole for further fabricated causal claims | 22 |
| **PERF-021** | aprender | PERF-006 | **Accelerator contract.** Retire boolean `--gpu`; ship `--gpu-layers {N\|auto\|all\|0}`, `--device`, `--list-devices`; explicit-wins rule (I-17); no-boolean rule (I-18); resolved-quantity reporting. `--gpu` kept as a deprecated alias for `all` that exits 9 with no backend | **4** |
| **PERF-022** | aprender | PERF-021 | Five-whys and cost the three O-1 distribution options (runtime backend loading / prebuilt per-backend releases / `--features cuda`); record the choice and rationale in the ticket. **Escalation, not a default** | 12 |

---

## §12 Do-not-build

| Item | Why not |
|---|---|
| Runtime fetch/symlink of `forjar` descriptors from `paiml/infra` | R1 — mutable remote state in the gate path; violates clean-room |
| Any threshold before BENCH-003 measures the noise floor | `APR-BENCH-RFC-001` §12 by name; the April reaper calibration failed exactly this way |
| A geomean anywhere in this system | §4.8 — a reported geomean beside a min verdict is the number people quote |
| A "release target" or dashboard number that cannot fail a build | R4 — that is documentation |
| Re-running a failed perf assertion to green | I-9 |
| Lowering the Arm B1 floor to permit a release | §10.1 |
| Backfilling the 942 provenance-free receipts | §8.2 |
| Blending `pp512` and `tg128` | Reports a correct GB10 as broken |
| A second receipt schema for `qwen-story` | §9 |
| Publishing a comparator ratio, cited or not | I-12 |
| **Publishing or targeting any `[X]` figure** (36.9×, 23×, 1.8×, 13×) | I-12 — other projects' numbers about other projects' systems. Prior art for design, never a target and never a claim |
| **`llama-bench` as the Arm B comparator harness** | I-15 — does not separate PP from TG under concurrent load; different client from apr's |
| **QoS-constrained config sweeps (Triton Model Analyzer style)** | We gate a fixed band set. Config search is a tuning tool; adding a search space to a gate makes the gate's subject non-deterministic |
| **Diff-aware pipeline generation (vLLM/Buildkite style)** | Correct at vLLM's scale (100+ parallel jobs). At our merge-phase cost — schema checks only — it is machinery with no waste to remove. Revisit if merge phase exceeds 5 min |
| **`IQ*`, `W4A16`, INT8 KV cache** for the gated workload | Arch-specific or calibration-dependent by construction; destroys the cross-host comparison the gate exists for. Legitimate product features, illegitimate gate variables |
| **Blocking W2 / Arms D and E before batching lands** | Permanently red for a known reason is a gate people learn to walk past. REPORTING until step 6 |
| **Ollama-style silent CPU fallback** | An open issue in that project since 2026-02-14 with a 500+ result downstream tail `[V]`. It is a documented failure mode, not a design to copy — and it is our defect #1 |
| **Any new boolean accelerator or backend flag** | I-18. A boolean request has no observable resolution. Quantities and device lists only |
| **Treating `cargo install --features cuda` as the destination for O-1** without a five-whys | It is the only one of three options with no precedent in the surveyed field (N7). Legitimate as a stopgap, unexamined as an endpoint |
| **Auto-fit or any heuristic that modifies an explicitly-set argument** | I-17. This is defect #1's root cause; reintroducing it anywhere reintroduces the defect |

---

## §13 Open — escalations, not defaults

| # | Question | Blocked on | Owner |
|---|---|---|---|
| O-1 | **Distribution channel: (a) runtime backend loading, (b) prebuilt per-backend releases, or (c) `--features cuda` only?** Reframed in v2.2 — it was a binary, it is a three-way, and (c) has no field precedent (§4.9.3) | PERF-022 five-whys; clean-room Mode A with `--features cuda` | operator |
| **O-9** | Does `--gpu` map to a layer count internally today, or is it a pure boolean all the way down? Determines whether PERF-021 is a rename or a resolver rewrite | `pmat query --file crates/apr-cli/src/commands/serve/mod.rs --pattern 'gpu'` | @noah |
| **O-10** | Default when a GPU is present but the model does not fit: hybrid offload (both comparators) or refuse and require an explicit split? Hybrid is the field default and is also how ollama's most-reported failure arises | PERF-021 design | operator |
| O-2 | Sample-size parameters §4.4.2 (`max(30, 8×c)`, 60 s) | BENCH-003 | @noah |
| O-3 | Arm B3 latency bounds (p95 TTFT, p95 ITL) | BENCH-003 | @noah |
| O-4 | `receipt_size_budget_bytes` — git or `almacen`? | one measured receipt | @noah |
| O-5 | Is `scripts/bench.sh` a seventh harness or something else? | audit outstanding | @noah |
| **O-6** | W2 length/generation mixture weights (§4.3.2) — currently spanning, not derived | first W2 conformant receipt | @noah |
| **O-7** | Preemption policy: recompute or swap? Decided by PERF-001's author; Arm D makes the choice visible but does not decide it | PERF-001 design | @noah |
| **O-8** | Does `client_tokenizer` counting hold for the comparator without a tokenizer mismatch (llama.cpp GGUF vocab vs HF tokenizer)? If not, `server_usage` may be the only comparable method and I-11 needs an amendment | PERF-019 | @noah |

---

## §14 Unverified appendix

Nothing below may be cited as evidence until promoted to `[V]`.

| Claim | Command |
|---|---|
| Three harnesses present on `origin/main` | `git ls-tree origin/main -- scripts/ \| grep -E 'benchmark-2x-ollama\|gpu_2x_benchmark\|benchmark-matrix\|bench.sh'` |
| Eight items landed on `feat/2692-apr-probar-llm` | `git log --oneline origin/main..feat/2692-apr-probar-llm --stat` |
| `let threshold = 10.0;` at `bench.rs:255` | `pmat query --file crates/apr-cli/src/commands/bench.rs --pattern 'threshold'` |
| Hardcoded causal string, `profile_print_hotspot.rs:155-162` | `pmat query --file crates/apr-cli/src/commands/profile_print_hotspot.rs` |
| 16-vs-32-token arithmetic, `kernel.rs:531-542` | `pmat query --file crates/apr-cli/src/commands/kernel.rs` |
| `http_concurrency = 1` in `llama_pin.toml` | `git show origin/main:scripts/llama_pin.toml` |
| `benchmark-matrix.sh` zero callers | `pmat query --pattern 'benchmark-matrix' --scope .github,Makefile,'*.toml'` |
| `aprender-test-lib/src/llm/` zero callers | `pmat query --pattern 'llm load\|llm score' --scope scripts,.github,Makefile` |
| 15+ `225.0 // Ollama parity` literals | `pmat query --pattern '225\.0' --scope crates/aprender-serve/src/gguf/tests` |
| Published `0.64.0` is `default = ["cli"]`, cuda opt-in | fetch `aprender-0.64.0.crate` from static.crates.io; read `Cargo.toml` |
| **`max_in_flight` is currently 1** | instrument per §4.4.9 and read it from the server — **the cheapest confirmation of defect #2 in this document** |
| **`--gpu` is a pure boolean with no internal layer count** | `pmat query --file crates/apr-cli/src/commands/serve/mod.rs --pattern 'gpu'` — O-9. **Blocks PERF-021 scoping** |
| **aprender applies no auto-fit today** | same query; if an auto-fit path exists, I-17 has a live falsifier immediately |
| llama.cpp `-ngl auto`, `-dev auto`, `-fit on` defaults; no boolean `--gpu` | `ggml-org/llama.cpp` `docs/multi-gpu.md` @ master — **fetched 2026-08-25, `[V]`** |
| Auto-fit skips explicitly-set arguments | `ggml-org/llama.cpp` discussion #18049 — **`[V]`** |
| Ollama silent CPU fallback; 500+ result tail | `ollama/ollama` issue #14258, open — **`[V]`.** Re-check status before citing; an open issue can close |
| Every §1 / §2 measurement | re-run under §4.4; **v1.0's numbers predate this protocol and are not conformant** |
| §4.4.2 sample-size parameters | BENCH-003 |
| §4.3.2 W2 mixture weights | first W2 receipt |
| `receipt_size_budget_bytes` | measure one full receipt |
| lambda-labs retired 2026-05-10; intel is clean-room runner | `machines/lambda-labs/forjar.yaml`, `machines/intel/forjar.yaml`; `forjar drift` |
| All `[X]` figures (36.9×, 23×, 8×, 1.8×, 13×, 60–80%, <4%) | Third-party published claims. **Not verifiable by us and not to be verified — they are design context, not evidence.** Never cite outside this spec (I-12) |

**Note on the §1/§2 measurements.** Every number in §1 and §2 was measured *before* §4.4 existed — no declared warmup, no declared sample count, no declared client model, no declared token-counting method, no drain rule. They are the best evidence available and are sufficient to justify this document, but they are **not conformant receipts** and must be re-measured under §4.4 before any becomes a committed baseline. That includes `scaling_efficiency(16) = 0.075`. Committing it as the Arm A baseline (§10.1 step 4) requires a §4.4-conformant run first.

---

## §15 Amendment — parallel by default, across agents AND hosts

**Added 2026-08-27.** Ratified against epic paiml/aprender#2706. Every host fact below was measured the day it was written; none is carried.

### 15.1 The rule

> **Work is parallel by default, across BOTH axes: multiple agents, and multiple hosts. Serial single-host work is the exception and must say why.**

Two axes, two different reasons, and conflating them is how the second one gets dropped:

- **Agents → speed.** Independent lanes — assess, verify, design — have no data dependency. Running them in sequence is latency nobody chose.
- **Hosts → coverage.** This one is not about speed at all, and it is the one that changes verdicts. **A single-host measurement cannot distinguish "apr is slow" from "apr is slow on x86_64."**

### 15.2 The fleet, measured 2026-08-27

| Host | Silicon | Cores | Reachable | Role in a parallel run |
|---|---|---|---|---|
| `lambda` | x86_64, RTX 4090 sm_89, 24 GB | 48 | ✅ | CUDA discrete, comparator-complete |
| `gx10` | aarch64, GB10 sm_121, ~120 GB **unified** | 20 | ✅ | CUDA unified-memory — the only non-x86 CUDA |
| `intel` | x86_64 Xeon W-3245, AVX-512 | 32 | ✅ | CPU/AVX-512 — contended, see §4.9.2 |
| `mini` | arm64 macOS, M4, 16 GB unified | 10 | ✅ | Metal — the only Apple path |

`aarch64` + `x86_64`, discrete + unified memory, CUDA + Metal + AVX-512. **No two of these are substitutable**, which is the entire argument: a result from one is a result about one.

### 15.3 Why coverage is the load-bearing half — worked example

The `#2567` aarch64 Q4_K defect was invisible on x86_64 by construction: it lived in an aarch64-only code path. It was found, and then **its first measurement was wrong** — reported as 2.91×, corrected to 1.21×, because the first number was a bimodal median.

Both failures are host-shaped. The defect needed aarch64 to exist; the correction needed enough samples on aarch64 to be trustworthy. A parallel fleet run gets both for free; an x86-only run gets neither and reports green.

Fresh corroboration, same day: `serialization_index(2)` measures **2.45 on lambda (sm_89)** and **2.85 on gx10 (GB10)** against a postcondition of `< 2`. Same defect, different magnitude. One host would have given a number; two give a *shape* — and the shape is what tells you it is architectural rather than a sampling artifact.

### 15.4 What this obliges

1. **Any claim about apr's performance names its host, or it is not a claim.** Already implied by §4.4's receipt; stated here so the omission is a defect and not a style note.
2. **A cell is measured on the host it describes.** No extrapolation across silicon — `perf-matrix.yaml` is a matrix, not a column.
3. **A defect found on one host is checked on the others before it is characterised.** §14 already says one failing input is an anecdote; this extends it: one host is also an anecdote.
4. **Missing coverage is `UNMEASURED` with an owner and an expiry, never silence.** §4.7.3 already provides the mechanism, and as of the current tree `expires` is actually read — before that it was declared on every cell and evaluated by zero lines of code.
5. **Agent lanes carry their own verification.** A lane that reports a finding without a mutation that turns it RED has produced a claim, not a result.

### 15.5 Two corrections this amendment forces

**§4.9's host table is wrong about `lambda`.** It reads *"retired 2026-05-10, do-not-revive"* and *"not a CI runner"*. `lambda` is reachable, is the box most of this epic was built on, and produced the `F-BATCH-001` measurement the `batch-admission-v1` contract records. A spec that describes a live host as retired will send someone to reproduce a result on hardware the spec says does not exist. Corrected in the table above; §4.9 to be reconciled in the same pass.

**`gx10` was not usable when this amendment was written, and the spec did not know.** It had no default route, no DNS, a checkout frozen at 0.60.0, and — the part that matters — an installed `apr` with **zero CUDA symbols**. Any benchmark taken through it would have reported CPU numbers from a GB10 box. `forjar.yaml` already declared the fix and had never converged, because `nmcli connection show enP7s7` is ambiguous (two profiles share the name) so its completion_check could never pass.

The lesson generalises past the one host: **a fleet is only as parallel as its least-reachable member, and "unreachable" is indistinguishable from "slow" until someone checks.** Reachability and capability are therefore preconditions of a fleet run, not assumptions of one — and capability means the binary, not the hardware. `apr gpu` prints the same GPU id on a CUDA build and a CPU-only one.
