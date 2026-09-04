# PP-LLAMA-001 v3.1 — MASTER — Inference performance parity with `llama.cpp`

**Status:** VERIFIED, NOT ARMED (§0.2) · **Supersedes:** `docs/archive/perf-2026-09-02/performance-parity-llama.cpp.md` and fifteen documents in four repositories (§13) · **Governs:** inference performance of `apr serve` in `paiml/aprender`

| | |
|---|---|
| **Document status** | `VERIFIED, NOT ARMED` (§0.2) |
| **Date** | 2026-09-02 |
| **Tree** | `origin/main` `b7bfcafa1`; every file:line in this document re-verified at the branch head that lands it |
| **Supersedes** | `performance-parity-llama.cpp.md` (PR #2845, unmerged; archived by this PR at `docs/archive/perf-2026-09-02/performance-parity-llama.cpp.md`); epic #2706 (`APR-PERF-GATE-001` v2.2); fifteen documents in four repositories (§13) |
| **Scope** | the only specification governing inference performance of `apr serve` in `paiml/aprender` |
| **Companions** | `evidence/parity/LEDGER.md` (Appendix C) · `scripts/perf-matrix.yaml` (every threshold, cell, phase) · `scripts/llama_pin.toml` (comparator template) · `docs/postmortems/perf-parity-review-2026-09.md` (history) · `docs/specifications/PP-LLAMA-001-RATIONALE.md` (why each rule) · `docs/audits/parity-spec-audit-2026-09-02.md` (the audit this version answers, C-1…C-20, CO-1…CO-4) · `evidence/bandwidth/` (PP-23 inputs) · `evidence/parity/derived_expiries.json` (§12 expiries, generated) |
| **Owner** | `perf-gate` (instrument), `serve` (engine), `spec-owner` (decisions marked `decided_by`) |

---

## §0 Conventions

**§0.1 Three artifacts, three jobs.** This document states rules. `perf-matrix.yaml` declares every cell, phase, threshold and ratchet the gate reads — a number that a gate compares against lives there and nowhere else (PP-33). `LEDGER.md` records every spent cell and every measurement with its validity. The spec cites the other two by path; it never duplicates their content.

**§0.2 Document status.** `DRAFT` → `VERIFIED, NOT ARMED` (every claim checked against the tree; no gate reads this document) → `ARMED` (§7.5 conditions met). A document may not be `ARMED` while any invariant it gates on is `OPEN`.

**§0.3 Provenance marks.** `[V]` verified by a cited command · `[C]` calculated from `[V]` inputs · `[U]` unverified · `[A]` asserted · `[X]` third-party or vendor figure — may inform design, may never be published as a claim or feed a published ratio (PP-12).

**§0.4 Rule statuses.** `ARMED` — a check exists, its must-fire mutation turns it RED, its must-not-fire fixture leaves it GREEN, both reachable from the surface named in §6 and joined by `scripts/spec_conformance.sh` (PP-29). `OPEN` — no automated check; the producer is owed in §12. `NA` — not applicable; `decided_by` recorded. `RETIRED` — the ID is never reused.

**§0.5 Editing rule.** This document contains no sentence about its own history. A change is a row in Appendix D with the diff; the reason is a paragraph in `RATIONALE.md` keyed to the rule ID. A pull request that adds prose of the form "an earlier draft said…" to this file fails review.

**§0.6 Invariant IDs.** `PP-nn` are stable. A retired rule keeps its number and the status `RETIRED`. New rules take the next number. Appendix A maps `APR-PERF-GATE-001` v2.2 `I-nn` to `PP-nn`.

---

## §1 Purpose

`apr serve` must serve the same model, on the same hardware, at the same offered load, as fast as `llama-server`, and must prove it with a receipt that (a) witnesses that the tokens it counted were correct, (b) names the band and the mechanism engaged, (c) carries prefill, decode and aggregate on every band, (d) is reproducible from its own contents, and (e) is decided by a stated statistical rule on interleaved replicates.

A throughput number that fails (a) is not a throughput number. A gate that has never passed cannot fail a change (§7.2).

**One spec. One matrix. One ledger. One harness. One client. One comparator.**

---

## §2 Evidence

The ledger is authoritative; this section carries only what a rule rests on. Every figure below cites the `evidence/` path it came from on its own line or table row.

### §2.1 Validity of every measurement in the tree, by band

| artifact | commit | date | c=1 | c>1 | usable for |
|---|---|---|---|---|---|
| **1a** `evidence/parity-http/findings.json` — the c=1 pair | `53062e7f3` `[V]` via `evidence/parity-http/findings.json:14` | 2026-08-24 | `NONCONFORMANT-VALID`: paired, one client, streaming. Decode **0.650×** (subject 103.26 against comparator 158.90 `tok/s`), prefill **0.275×** (2,860 against 10,399 `tok/s`), TTFT p50 35.66 against 9.81 ms, VRAM 14,030 against 4,554 MiB — all `[V]` at `evidence/parity-http/findings.json:16-17` and `:29-115`. Source runs: `evidence/parity-http/apr-0.64.0-cuda.json` + `evidence/parity-http/llamacpp-39173bcac.json` (3 × 30 s, streaming, c=1). Replicate pair `evidence/parity-http/lambda-apr.json` + `evidence/parity-http/lambda-llamacpp.json` (2026-08-25) gives 0.646× and 0.272× on the same two metrics | `NA` — this artifact is c=1 only (`evidence/parity-http/findings.json:4`) | c=1 sizing (§9 #3, #4, #7) |
| **1b** `evidence/parity-http/bands/` — the four-band sweep | **UNKNOWN** `[U]` — no `bands/*.json` carries a commit, a binary sha or a compute class; upper bound `ce712eae0`, the commit that added them | 2026-08-25 (`evidence/parity-http/bands/apr-c1.json` `runs[0].timestamp`) | `NONCONFORMANT-VALID` (`evidence/parity-http/bands/apr-c1.json`) | `INVALID-BUILD`: batching compiled out (`cuda-batch = ["cuda"]`, §9 #8); the subject serialises (`evidence/parity-http/bands/apr-c16.json`) | the JOIN fixture only (§12 row 7). Its PP-9 key is **unspendable**: with no commit there is no key |
| **2** `evidence/perf-gate-001-w1-lambda/` | `745fa8588` | 2026-09-01 | `NONCONFORMANT-VALID`: no stream, no comparator; `agg(1)` 100.643 ± 0.635 at n=3 (`evidence/perf-gate-001-w1-lambda/receipt.r1.json`), and 28 of 55 retained samples stopped short of `n_predict` (`evidence/perf-gate-001-w1-lambda/samples.c1.r1.jsonl.gz`) | `INVALID-CORRECTNESS`: #2753 — batched decode emits a constant token for every `m>1`; 0 of 485 samples short at c>1 against 28 of 55 at c=1 (`evidence/perf-gate-001-w1-lambda/samples.c4.r1.jsonl.gz`) is that shape | noise floor at c=1 only; proof that batching *engages* (`max_batch=11`, `evidence/perf-gate-001-w1-lambda/server-startup.txt:27-28`), not that it is fast |
| **3** `evidence/perf-gate-001-w1-gx10/` | `745fa8588` | 2026-09-01 | `NONCONFORMANT-VALID`: serial prefill (§9 #1), 16.75 s fixed + 32.1 ms/token `[C]` — **fit over 28 of 30 samples; 2 samples per replicate completed 128 tokens in 4.21–4.24 s and are unexplained** (`evidence/perf-gate-001-w1-gx10/samples.c1.r1.jsonl.gz`). **Transport caveat:** the fixed cost is paid on the blocking path (`crates/aprender-serve/src/gguf/cuda/generate_2.rs:284-288`); the streaming c=1 gx10 run shows TTFT 34.27 ms (`evidence/parity-http/findings.json`, `FINAL_quiet_box_parity`) | `INVALID-CORRECTNESS`: `prefill_multi_prompt` runs unguarded on Blackwell (§9 #1a) at every `m>=3` (`evidence/perf-gate-001-w1-gx10/server-full.log.gz:104`) | §9 #1 mechanism; noise-floor caveat (c=8 stall, #2833, `evidence/perf-gate-001-w1-gx10/findings.json`) |

Nothing in this tree is a conformant parity receipt. No `c>1` aggregate figure from any of the four may be published, quoted, or used as a baseline (PP-26, PP-12).

The ratios `0.5341/0.2308/0.1685/0.0967` (aggregate) and `0.5873/0.9231/1.3525/1.5540` (decode) exist only as JOIN-fixture values (§12 row 7, `evidence/parity-http/bands/`). The series `0.591/0.395/0.544/0.401` is a **cross-run quotient** — a `745fa8588` subject (`evidence/perf-gate-001-w1-lambda/receipt.r1.json`) divided by a 2026-08-25 comparator (`evidence/parity-http/bands/`) — which is INVALID under P-1 and is never quoted.

### §2.2 Mechanism rule (comparator-independent)

A server that shares one device across `c` requests raises aggregate and lowers per-user decode; a server that serialises does the reverse. The two move in opposite directions under the change that matters most, so a receipt carrying one alone cannot be read (PP-4), and a gate demanding both `>= 1` on every band demands a beat, not parity (P-3). Prefill and decode at c=1 do not trade against each other; gating both at c=1 is not this trap (P-3).

### §2.3 What the tree says (premises falsified by reading it) — `[V]` by the command in each row

| premise | tree | verified_by |
|---|---|---|
| the gate has no comparator-ratio producer | `scripts/lib/perf_receipt.py` emits both ratio series today; the missing piece is the JOIN into the conformant producer | `python3 scripts/lib/perf_receipt.py --from-bands evidence/parity-http/bands --derive-only` (rc 0; prints the four aggregate and four decode ratios) |
| decode is not captured | capture path complete and unit-tested; `--stream` was not passed | `sed -n '429,533p' crates/aprender-test-lib/src/llm/client.rs`; `grep -n 'stream' evidence/perf-gate-001-w1-lambda/producer-stdout.txt` (line 6 declares the metrics UNDEFINED) |
| batched decode calls cuBLAS above `m>=4` | decode runs under graph capture, so `use_cublas = … && !self.is_capturing` is false; batched decode calls GEMV, single-warp at `M<=8` | `sed -n '995p;1022p' crates/aprender-serve/src/cuda/executor/layers/cublas_prefill/attention.rs`; `sed -n '29,34p' crates/aprender-serve/src/cuda/executor/layers/cublas_prefill/mod.rs` |
| gx10 is bandwidth-starved | the code defaults `cc >= 120` to a per-token serial prefill loop; the two-parameter fit matches to 2.6% | `sed -n '517,531p' crates/aprender-serve/src/cuda/gpu_profile.rs` (prints `select_prefill_path` and `SM12X_MIN_CC`); the fit inputs are `evidence/perf-gate-001-w1-gx10/samples.c1.r1.jsonl.gz` |
| single-stream is unmeasured | #2694 and #2693 are open, receipted and paired | `jq '.apr_over_llamacpp' evidence/parity-http/findings.json` |
| batching on `main` is fast | #2753 P0: `m>1` emits a constant token; the `perf041` probe exists and is wired to nothing | `grep -rn 'perf041' .github/workflows/` (empty at `b7bfcafa1`) |
| `max_batch=11` is an env var | auto-sized from free VRAM at load; the unrecorded input is free VRAM at the sizing instant | `sed -n '28,62p' crates/aprender-serve/src/cuda/executor/kv_cache_gpu_init.rs`; `sed -n '27,28p' evidence/perf-gate-001-w1-lambda/server-startup.txt` |

### §2.4 Bandwidth ceiling (design bound only)

`ceiling_dec_tok_per_sec = measured_device_bytes_per_sec / model_bytes` applies to **per-sequence decode** only; batched aggregate legitimately exceeds it. With `[X]` inputs the ceilings are ≈215 `tok/s` (RTX 4090) and ≈58 `tok/s` (GB10) `[C]`. No percentage of a ceiling is published until `scripts/measure_bandwidth.sh` commits a `[V]` bandwidth under `evidence/bandwidth/` (PP-23, §12 row 14).

---

## §3 Metrics

All token counts are the server's `usage.completion_tokens` / `prompt_tokens` (PP-28). All timings are the client's clock except where marked server.

| metric | definition | unit | source |
|---|---|---|---|
| `ttft` | request send → first content byte | ms | client |
| `tpot` | `(e2e − ttft) / (completion_tokens − 1)` | ms | client / server count |
| `dec` | per request `(completion_tokens − 1) / (e2e − ttft)`; band value = **median** over retained requests | tok/s | derived |
| `itl_p95` | 95th percentile of per-token inter-arrival intervals within the band | ms | client |
| `prefill` | `prompt_tokens / prefill_ms`, **server-reported** (`timings.prompt_per_second` on the comparator; `apr` equivalent, PP-2) | tok/s | server |
| `agg` | Σ `completion_tokens` of requests admitted in the window ÷ window span | tok/s | server count / client span |
| `scaling_efficiency` | `agg(c) / (c · agg(1))` | — | derived; **reported, never ratcheted** (PP-31) |
| `overhead_share` | per lane, `agg(1) / dec(1)` | — | derived; paired quotient reported beside both lane values |
| `vram.used_peak_bytes` (sampled), `vram.recorded_alloc_peak_bytes` (a recorded **lower bound**, never a peak), `kv.kv_per_slot_bytes`, `scheduler.slots_admitted` | server-reported by `GET /v1/effective-config` at load and after each band; there is no key named `vram_peak` | bytes, bytes, bytes, count | server (PP-2) |
| `stream_mode` | `live` \| `replayed`, declared by the server on the first SSE chunk | — | server (PP-27) |

`n_predict` is the generation budget. **On the wire it is carried as the OpenAI field `max_tokens`** — that is what the W1 corpus records, what the client sends and what both servers read; `n_predict` is reserved for the comparator's launch argument.

Ratios: `x_ratio(c) = x_apr(c) / x_llama(c)` from the same run (P-1). `agg` and `dec` are not interchangeable (§2.2).

---

## §4 Parity

**P-1 · Paired.** The target for any metric is the comparator's value from the same run, host, model, client binary, workload file and window. `1.0` is the definition of parity, not a threshold. No other literal enters this document as a bound; every bound is in `perf-matrix.yaml` with a `threshold_class` and an author (PP-33).

**P-2 · Named.** A parity claim names cell, band and metric. Otherwise it is schema-fatal (PP-17).

**P-3 · Asymmetric by band.** Gated at c=1: `dec_ratio` and `prefill_ratio`. Gated at c>1: `agg_ratio`. Everything else is REPORTED on every band (PP-4). Both `agg` and `dec` at `>= 1` on every band is refused as a rule (§2.2).

**P-4 · Correct before fast.** A band whose correctness witness (PP-26) is absent or failing is `INVALID-CORRECTNESS`: its throughput is not reported, not gated, and never a baseline.

**P-5 · Decision rule (§4.3).** A gated metric PASSES iff the one-sided 95% lower confidence bound of its ratio is `>= 1 − δ`, where `δ` is the declared non-inferiority margin in `perf-matrix.yaml` (`threshold_class: policy`, author named). `δ = 0` is parity. The MDE is reported as the cell's resolving power and never enters the verdict.

**P-6 · Seeded at achieved.** A parity gate on (cell, band, metric) is REPORTING until the first receipt that PASSES P-5 on it, and ARMED from that receipt on (§7.2). A self-regression ratchet (PP-31) is seeded at the value the build under test has already produced. No gate arms by date.

**P-7 · Must-not-fire.** Every gate lands with a must-fire mutation and a must-not-fire fixture in one commit (PP-29).

**P-8 · Latency.** `ttft` and `itl_p95` at c=1 are valid under W1 and REPORTED; at c>1 they are measured only under W3 (§5.1) and are REPORTING until W3 exists. No latency bound is set in this version.

### §4.3 Statistics

| unit | metrics | design | estimator | verdict statistic |
|---|---|---|---|---|
| **replicate** (window statistics) | `agg`, `prefill`, `vram.used_peak_bytes` | `n >= 5` **interleaved** paired replicates, A,B,A,B,… within one harness invocation, comparator and subject alternating | mean of per-replicate `ln(x_apr / x_llama)` | one-sided t lower bound, `df = n − 1`, exponentiated |
| **request** (per-request statistics) | `dec`, `ttft`, `itl_p95` | all retained requests of the band, both lanes | paired percentile bootstrap, 10 000 resamples, seed `2026`, resampling whole requests | 5th percentile of the bootstrap ratio distribution |

Interleaving is mandatory: thermal state, JIT/graph-capture warm state and free VRAM drift across a sweep, and alternation is the only design that cancels the drift. A receipt whose replicates were not interleaved is `NONCONFORMANT` (PP-9 key includes `interleaved: true`). `n = 3` sizes an effect and bounds no variance: no σ-dependent status changes at `n < 5`.

---

## §5 Protocol

### §5.1 Workloads

| id | shape | arrival | sampler (both lanes) | yields | status |
|---|---|---|---|---|---|
| **W1** | 512 prompt / 128 gen, fixed prompt corpus `crates/aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl` | closed-loop at concurrency `c` | `temperature 0`, `seed` recorded, `ignore_eos true`, `n_predict 128` carried as `max_tokens` | `agg`, `dec`, `prefill`; `ttft`/`itl` at c=1 | **primary** |
| W2 | ragged length mixture (`docs/archive/perf-2026-09-01/APR-PERF-GATE-001-v2.2.md` §4.3.2) | closed-loop | as W1 | `agg` secondary, KV pressure | REPORTING |
| W3 | W1 shapes | open-loop Poisson, λ per host so median in-flight ≈ c | as W1 | `ttft`, `itl_p95` at c>1 | owed (§12 row 16) |
| W4 | W1 prompt, `n_predict ∈ {32, 64, 128, 256}` | closed-loop c=1 | as W1 | the two-parameter `fixed_s + ms_per_token` decomposition | owed (§12 row 17) |

**The measurement window is dual-bound and normative.** A band ends when **both** bounds are satisfied: at least `max(30, 8·c)` retained samples **and** at least `window_ms` of wall clock. Before the window opens each worker issues **2 warmup requests** (`2·c` in total for the band), then the harness **quiesces for `quiesce_ms`**; between the two lanes of a replicate it **cools down for `cooldown_ms`**. Requests in flight at window close are drained, never counted as issued after close (PP-10), and `drain_ms` is recorded.

Every one of those quantities lives in `scripts/perf-matrix.yaml` under `protocol:` with a `threshold_class` and an author (PP-33) — `window_ms`, `warmup_requests_per_worker`, `quiesce_ms`, `cooldown_ms`, `n_predict`, `prompt_tokens`, `replicates_min`, `interleaved` and the `sampler` block. This document states the shape; the matrix states the numbers, and `crates/aprender-test-lib/src/perf_gate/protocol.rs` reads them from the matrix rather than from its own constants.

Streaming required (PP-27). `completion_tokens == n_predict` on every retained sample or the sample is fatal to the band (PP-28). Bands are **derived**: `ladder = {c ∈ {1,4,8,16} : c <= min(slots_admitted_apr, slots_admitted_llama)}` read from both servers (PP-24); a band above a server-reported deliberate ceiling is `NA` with the reported budget.

**Corpus note.** `ignore_eos` is a per-record and `_meta` field of `prompts-w1.jsonl`, emitted by `scripts/gen_prompts_w1.py`. Regenerating the corpus to add it **rotates the corpus sha256**, which is a component of the PP-22 join key and of the PP-9 cell key: receipts taken before the rotation stay keyed to the old digest and do not join to receipts taken after it. That is intended, and it is why the rotation happens once, in the same change as PP-28.

### §5.2 Subject configuration

`apr serve run {model} --gpu-layers all --port {port} --stream …` — a quantity, never a boolean (PP-15). Before the first request the harness stores `GET /v1/effective-config` verbatim (PP-2): resolved `max_batch` with its four inputs (free VRAM at load, `kv_per_slot`, reserve, clamp) and `source` (`computed` | `env`); resolved `GpuProfile` (`cc`, `q4k` variant, fp8 flags, `fused_gate_up`, graph enablement); the prefill path `run_prefill` will select; scheduler identity, `window_ms`, `max_in_flight`; `vram_peak`, `kv_bytes_reserved`, `kv_blocks_total`; `gpu_layers_{requested,resolved,total}`; `backend_loaded[]`; `autofit_applied[]`; UTC start timestamp (PP-30).

### §5.3 Comparator — decided

**The comparator is `llama-server` configured to serve the band** — relaunched once per band from the `scripts/llama_pin.toml` template with `-np {c}` and `-c {c · n_ctx_slot}`, `n_ctx_slot >= 640`, every 2026-mobile default pinned (`-fa`, `-b`, `-ub`, `--cache-type-k/-v`, `-cb`, slot-save off, `--seed`), `cmake` line and commit pinned with an expiry (PP-20). `GET /props` is stored per band before the first request; `props.n_ctx / props.total_slots >= 640` or the band is schema-fatal. A numeric `batch_size <= 1` is refused as a comparator configuration.

`decided_by: spec-owner, v3.0`. Rationale: parity is a claim about serving the same offered load; a comparator that queues 12 of 16 requests is not serving the band, and `-np c` is the documented way to serve `c` users. **Dissent recorded** in `RATIONALE.md` §5.3, moved verbatim from `scripts/llama_pin.toml`. The must-not-fire fixture for this decision is a `-b 1` run, which must be refused as a comparator configuration (the PP-22 join key carries `n_batch`). **First action** (§12 row 3): read `/props` on the pinned build at the withdrawn run's argv — its `total_slots` decides whether the "4 slots by design" premise was ever true; the withdrawn run's own lane shows 3.93 / 7.84 / 15.74 sequences in flight at c = 4/8/16 `[C]` from `evidence/parity-http/bands/`.

The CLI-differential lane (`apr_command` / `comparator_command` at `scripts/llama_pin.toml:183`) is RETIRED: `llama-bench` is never the comparator. A second lane, `llama-server` at its defaults, may be REPORTED beside the configured one; it is never the comparator either.

### §5.4 Isolation

One global CI concurrency group `perf-<host>`, `cancel-in-progress: false`, shared with any job contending the host (clean-room on `intel`) (PP-19). `nvidia-smi --query-compute-apps=pid,used_memory` before and after each band; any foreign PID is fatal to the band.

---

## §6 Invariants

One table. Columns: rule · must-fire (RED) · must-not-fire (GREEN) · status · producer · selftest names. The last column names the **surface** each case lives on: `pg` = `bash scripts/perf_gate.sh --list-selftests`, `sh:<script>` = that script's own case table, `rs:<crate>` = a `#[test] fn` under `crates/<crate>/src`. `scripts/spec_conformance.sh` joins this table to those surfaces, and a row is `ARMED` only when **both** cases exist by the exact names given (PP-29).

| id | rule | must-fire | must-not-fire | status | producer · selftest |
|---|---|---|---|---|---|
| PP-1 | Expected cell set is enumerated from `perf-matrix.yaml`; the verdict asserts every cell `MEASURED`, `UNMEASURED{owner,expires}` or `NA{decided_by}` | delete one receipt | a matrix with one `NA` cell | ARMED | `perf_gate.sh` · `cellset_missing` / `cellset_na_ok` (pg) |
| PP-2 | `provenance.compute_class` is the dispatch path taken, read from the process; `gpu_layers_resolved` from the loader; `provenance.server_config` is the §5.2 endpoint response verbatim, incl. memory fields | report `cuda` on a CPU build; omit `server_config` | a CPU cell reporting `cpu` | ARMED | `GET /v1/effective-config` (§12 row 6) · `config_missing` / `config_present` (pg); ArmD over the pre-v3 receipt of an UNMEASURED cell: `historical_unmeasured_armd_reports` (pg) |
| PP-3 | No `ratio` is representable without a `baseline` object that passes every receipt rule and shares `run_id` | bare scalar baseline; baseline from another run | a same-run baseline | ARMED | `receipt.rs`, `scripts/lib/bench_receipt.py` · `ratio_bare` / `ratio_paired` (pg + rs:aprender-test-lib `ratio_bare__a_scalar_ratio_is_unrepresentable`, `ratio_paired__a_same_run_baseline_joins`) |
| PP-4 | `agg`, `dec` and `prefill` present on every band; one absent is schema-fatal for receipts dated `>= v3.0`; earlier receipts are historical records, never baselines | decode-only receipt at c=16 | a v2-dated receipt cited as history | ARMED | §12 rows 4, 6 · `band_metric_absent` / `historical_cited` (pg) |
| PP-5 | `timeouts > 0` on any band is fatal to that band's ratio | inject one timeout | a band with 0 timeouts and 3 drained | ARMED | `perf_gate.sh` · `timeout_fatal` / `drain_ok` (pg) |
| PP-6 | No comparator wall-clock ratio is a merge-phase check; `run_gate` obeys the `phase:` each arm declares in `perf-matrix.yaml` | promote Arm B to `ci / gate`; run a ratio-carrying receipt at `--phase merge` | the same receipt at `--phase release` still fails L3 | ARMED | `scripts/perf_gate.sh` `run_gate` (`:938`) dispatching every arm through `run_phased` (`:80`) with `arm_phase` (`:63`) reading the matrix; `arm_a_self_regression` (`:557`) and `arm_l3_parity` (`:660`) run REPORT-only outside `phase: release` · `phase_guard_b_merge` / `phase_guard_b_release` (pg; `phase_guard_a_merge` beside them) |
| PP-7 | Raw per-request samples retained on every band; size budget committed after first measurement | strip `samples[]` | a receipt at budget | ARMED | `perf_gate.sh` · `samples_stripped` / `samples_ok` (pg) |
| PP-8 | Comparator **client** `http_concurrency` equals the band's `c` on both lanes | drive the comparator at 1, band 16 | both lanes at `c` | ARMED (case table) · the rule binds on a two-lane receipt, and the second lane arrives with §12 row 7 | `scripts/lib/bench_receipt.py` `validate_parity`, `scripts/lib/parity_block.py` · `client_conc_mismatch` / `client_conc_ok` (pg) |
| PP-9 | A cell is keyed `(host, workload, model, quant, commit, interleaved)`; once run at a key it is spent there and may not be re-run to green; a new commit is a new row | re-run at the same key and publish the second | re-run at a later commit accepted | ARMED | Appendix C ledger parser · `respend_same_key` / `respend_new_commit` (sh:scripts/spec_conformance.sh) |
| PP-10 | No request issued at or after window close; pre-close requests drained; `drain_ms` recorded | issue one request after close and count it | a drained band | ARMED | `perf_gate.sh` · `post_close_request` / `drain_recorded` (pg) |
| PP-11 | `tokenization.method` has no default; absence is schema-fatal | omit the block | block present | ARMED | `perf_gate.sh` · `tokenization_absent` / `tokenization_ok` (pg) |
| PP-12 | No comparator ratio outside a receipt; no `[X]` figure published as a claim; a number in `README.md`/`book/`/`docs/` is legal iff it cites an `evidence/` receipt path — the checker's universe includes `docs/specifications/` | a spec line carrying a comparator ratio with no receipt path | the same line followed by an `evidence/` path | ARMED | the `docs/specifications/` exclusion is deleted (§12 row 10); the surviving exclusions are `docs/specifications/archive/` and `docs/archive/` (`scripts/check_no_claim_literals.sh:1097-1098`), and the citation exemption applies to markdown surfaces only · `claim_unreceipted` / `claim_receipted` (sh:scripts/check_no_claim_literals.sh) |
| PP-13 | `max_in_flight`, `slots_admitted` and every §5.2 field are server-reported; a harness-computed or harness-declared value is schema-fatal | harness derives `max_in_flight` from timestamps; `feature_set` taken from `--server-feature` with no `server_config` | server-reported value | ARMED | §12 row 6 · `inferred_field` / `reported_field` (pg) |
| PP-14 | Auto-fit never modifies an explicitly-set argument; `autofit_applied[] ∩ explicit_args = ∅` | set `--gpu-layers 12`; auto-fit raises it | `--gpu-layers all` resolved to 28/28 | ARMED | `OffloadReport.pp14_holds` · `autofit_override` / `autofit_ok` (pg) |
| PP-15 | No boolean accelerator flag on the CLI surface or in any harness command | a boolean accelerator flag in `llama_pin.toml` | `--gpu-layers all` with `gpu_layers_resolved` | ARMED | `scripts/check_comparator_flags.sh` · `boolean_flag` / `quantity_flag` (sh:scripts/check_comparator_flags.sh) |
| PP-16 | `receipt.provenance.compute_class == perf-matrix.yaml[host].compute_class` and some build can reach that class | `mini: metal` (#2841) | `mini: NA{decided_by}` | ARMED | §12 row 5 · `class_unreachable` / `class_na` (pg) |
| PP-17 | A parity claim names cell, band and metric | a `ratios` object outside a band | a `ratios` object inside a band carrying `concurrency` | ARMED | `scripts/lib/bench_receipt.py` · `claim_bandless` / `claim_named` (pg) |
| PP-18 | The measuring binaries (subject, comparator, client) are built from commits that are ancestors of the commit under test; each sha256 in the receipt; `git merge-base --is-ancestor` asserted by the validator | a non-ancestor `provenance.subject.commit` | an ancestor build | ARMED | `perf_gate.sh` (`PERF_GATE_GIT_DIR` seam) · `ancestor_fail` / `ancestor_ok` (pg) |
| PP-19 | One global concurrency group per host, `cancel-in-progress: false`; no foreign compute PID during the window | launch two runs; inject a foreign process | one run, clean device | ARMED | `scripts/check_perf_concurrency_groups.sh` (the CI half) and `scripts/perf_isolation.sh` (the foreign-PID half) · `isolation_breach` / `isolation_ok` (sh:scripts/check_perf_concurrency_groups.sh + sh:scripts/perf_isolation.sh `foreign_pid_breach`, `foreign_pid_ok`) |
| PP-20 | The comparator pin carries commit, `cmake` line, template and expiry; a stale pin marks every ratio `COMPARATOR_STALE` and blocks `MEASURED` | expiry in the past | expiry in the future | ARMED | `scripts/check_llama_pin.sh` · `pin_stale` / `pin_fresh` (sh:scripts/check_llama_pin.sh) |
| PP-21 | Receipt signature valid and `receipt.commit ⊇ commit-under-test` | unsigned; stale-by-one | signed, current | ARMED | `scripts/perf_receipt_sign.sh` · `sig_missing` / `sig_ok` (pg); release over the pre-v3 receipt of an UNMEASURED cell: `historical_unmeasured_release_reports` / `historical_measured_release_fails`, `v3_unsigned_unmeasured_release_fails` (pg) |
| PP-22 | Join key — host, workload, band, model, quant, `tokenization`, window, replicate count, `interleaved`, `n_ctx_slot`, `kv_type`, `fa`, `n_batch`, `n_predict` — mismatch **refuses** the ratio | join c=4 against c=16; join 30 s against 60 s windows; join a `-b 1` lane | matching keys | ARMED | §12 row 7 · `join_mismatch` / `join_ok` (pg + rs:aprender-test-lib `join_mismatch__c4_against_c16_is_refused`, `join_ok__matching_keys_join`) |
| PP-23 | `roofline_tok_per_sec` from a `[V]` bandwidth and `stat -c %s` of the model; compared to **per-sequence decode only**; `dec` above the ceiling is schema-fatal; no lower threshold | a decode rate above the lambda ceiling | a gx10 c=8 **aggregate** above the ceiling, which is correct batching | ARMED | `evidence/bandwidth/` · `roofline_exceeded` / `roofline_aggregate_ok` (pg) |
| PP-24 | `slots_admitted >= c` on both lanes, server-reported, or the band is `UNMEASURED{admission_capped, lane, cap}`; a deliberate server-reported ceiling yields `NA{decided_by, budget}`; the ladder is derived (§5.1) | c=16 against a subject admitting 11 | subject reports a KV-budget ceiling of 11, so c=16 is `NA` | ARMED | §12 row 6 · `admission_unequal` / `admission_na` (pg) |
| PP-25 | One client binary drives both lanes; its sha256 in the receipt | comparator driven by a second binary | same binary both lanes | ARMED | §12 row 12 · `client_mismatch` / `client_ok` (pg) |
| **PP-26** | **Batch-invariance witness (v3.1).** For the witness prompt at `temperature 0`: **(a)** every slot of an `m=c` batch of identical prompts agrees with slot 0 to the declared point (`witness.min_agree_tokens`); **(b)** no slot repeats one token id for `witness.max_constant_run` steps (#2753's signature); **(c)** the `m=1` stream's agreement with the batch is RECORDED per band (`divergence_at`), not gated — measured on lambda 2026-09-02 (`evidence/perf041/lambda/witness.json`, `m1-vs-m4-three-prompts.txt`) it is the fp divergence between kernel families (`m=1`; `m=2,3`; `m=4,8,16`), each of which is batch-size invariant to the end; it becomes a gate when §12 row 22 puts a top-2 margin on the wire. The result is in every band's receipt; absent or failing (a)/(b) → band `INVALID-CORRECTNESS` | a constant-token stream at `m=3` (#2753); one slot of an `m=4` batch parting from the others | four slots identical to the end while all part from `m=1` at the third token (the lambda shape) | ARMED | §12 row 1 · `batch_invariance_fail` / `batch_invariance_ok` / `witness_intra_below_declared` (pg + sh:scripts/perf041_batched_parity_probe.py `witness_constant_token_m3`, `witness_intra_batch_disagree_m4`, `witness_identical_128_ok`, `witness_kernel_family_flip_recorded_ok`; sh:scripts/lib/perf_receipt.py `witness_attached_from_perf041`, `witness_absent_band_is_invalid_correctness` — the executor harness attaches the perf041 witness per band, since the bench reports carry none) |
| **PP-27** | Streaming required; the server declares `stream_mode` on the first chunk; the client independently computes `ttft/e2e` (≈1.0 on replay); disagreement sends `dec`/`ttft`/`itl` to `unproduced_fields` with a reason; `usage` on the terminal chunk on both lanes; chunk-count fallback is a hard refusal | a replayed SSE stream | a live stream with `usage` | ARMED | §12 row 0b · `stream_replayed` / `stream_live` (pg; `stream_absent` beside them) |
| **PP-28** | `temperature 0`, `seed`, `ignore_eos` and `max_tokens` on the wire for both lanes, streaming and not; `completion_tokens == n_predict` on every retained sample or the band is fatal | lambda `samples.c1.r1`: 28 of 55 samples at 24–127 generated tokens | 30 of 30 at 128 | ARMED | §12 row 0b · `sampler_unpinned` / `sampler_pinned` (pg) |
| **PP-29** | Every `ARMED` row of this table has both selftest cases named here present on the surface named here; `scripts/spec_conformance.sh` joins the table to those surfaces and runs in `ci / gate` | remove one case | the full table | ARMED | `scripts/spec_conformance.sh` · `conformance_missing` / `conformance_ok` (sh:scripts/spec_conformance.sh) |
| **PP-30** | `provenance.started_utc` and `clock_source` on every receipt | receipt without a timestamp | a timestamped receipt | ARMED | §12 row 6 · `timestamp_absent` / `timestamp_ok` (pg) |
| **PP-31** | Self-regression: per (cell, band) `agg(c)`, `dec(c)`, and at c=1 `prefill`, each ratchets **against the last `MEASURED` receipt on protected `origin/main`**, seeded at the value that receipt achieved; a decrease beyond that cell's LCB at `n >= 5` is a release FAIL for that cell; `scaling_efficiency` is never ratcheted | halve CPU `agg(4)` in a PR that doubles CUDA | raise `agg(1)` by 20% with `agg(16)` unchanged: `scaling_efficiency` falls and nothing fails | ARMED | §12 row 8 · `self_regress_fail` / `agg1_improve_ok` (pg) |
| **PP-32** | Engine-track `AbRecord` (§10) has no field able to hold a comparator, a second runtime name, or a parity verdict; carries `delta_kind ∈ {config, code}`, per-arm `(commit, sha256)`, interleaved arms, and both arms' effective-config responses diffed | add a `comparator` field; two arms not interleaved | a `code` delta with two shas | ARMED | §12 row 9 · `abrecord_comparator` / `abrecord_ok` (rs:aprender-test-lib `abrecord_comparator__a_comparator_field_does_not_parse`, `abrecord_ok__a_code_delta_with_two_shas_parses`) |
| **PP-33** | Every threshold, floor, ceiling, ratchet direction and phase the gate reads lives in `perf-matrix.yaml` with `threshold_class` and author; a numeric comparison in `perf_gate.sh`, `parity_block.py` or any lane script that is not read from the matrix is RED | a bare stretch constant in `parity_block.py` | all comparisons read from the matrix | ARMED | `scripts/check_thresholds_in_matrix.sh` · `threshold_outside_matrix` / `threshold_in_matrix` (sh:scripts/check_thresholds_in_matrix.sh) |

**RETIRED:** none in v3.0.

---

## §7 The gate

### §7.0 Layers

A number advances one layer at a time; a failure at any layer stops it there.

| layer | question | rules | phase | comparator |
|---|---|---|---|---|
| **L0 Correctness** | were the tokens right? | PP-26 (+ `perf041` nightly, missing marker = RED) | nightly + release | no |
| **L1 Integrity** | is the receipt trustworthy? | PP-1,3,4,5,7,8,9,10,11,13,15,16,17,18,20,21,22,25,27,28,30,33; PP-29 | **merge** (static) | no |
| **L2 Self-regression** | did this host get slower? | PP-31, PP-19, PP-23 | release | no |
| **L3 Parity** | is it as fast as the comparator? | P-5 on the gated metrics; PP-24 ladder | release | yes |
| Engine track | did the fix do what it predicted? | PP-32 (§10) | per PR, informational | no |

### §7.1 Merge phase — what `ci / gate` concretely is

`ci / gate` is the job `gate` (`grep -n '^  gate:' .github/workflows/ci.yml`; `:2241` at this commit), whose `needs:` list (`:2243`) includes `guard-runner-labels` (`:443`) and which fails if that job's result is not `success`. The required check on `main` is the bare name `gate`; **`guard-runner-labels` is therefore required transitively, and it is the job that runs every shell guard named in §6.** A guard that is not invoked by a `run:` line inside `guard-runner-labels` is not in the merge phase, whatever this document says about it.

At merge: receipt schema; every `ARMED` L1 rule, statically; `scripts/spec_conformance.sh` (PP-29); no HTTP timing assertion of any kind (PP-6). The kernel microbenchmark (§7.3) is `NOT ARMED` in v3.0.

### §7.2 Release phase and the arming rule

**What "release" concretely is.** A release is: `bash scripts/bump-version.sh <version>` (workspace-wide version bump), the pre-release gate set (`.claude/skills/pre-release`), an annotated tag, `bash scripts/cascade-drain.sh` for the crates.io publish cascade, and `gh release create` from the `CHANGELOG.md` entry. **The perf release gate is `bash scripts/perf_gate.sh --phase release` run over the receipt committed under `evidence/` for the cell**, and it is the only place a comparator ratio may decide anything.

| band | gated (L3) | reported |
|---|---|---|
| c=1 | `dec_ratio`, `prefill_ratio` | `agg_ratio`, `ttft`, `itl_p95`, `overhead_share` per lane |
| c ∈ ladder, c>1 | `agg_ratio` | `dec_ratio`, `prefill_ratio`, `scaling_efficiency`, `itl_p95` |

**Arming.** For every (cell, band, metric) the L3 gate is `REPORTING` until the first receipt that PASSES P-5; from that receipt it is `ARMED`, that receipt is recorded in `perf-matrix.yaml` as `armed_by`, and a later receipt that FAILS P-5 blocks release. Nothing in L3 arms by date. L2 arms on the first `MEASURED` receipt of the cell (PP-31 seeds there).

Plus: every cell in the matrix `MEASURED`, `UNMEASURED{owner, expires}` unexpired, or `NA{decided_by}`.

### §7.3 Kernel microbenchmark — `DESIGNED, NOT ARMED`

Target: batched Q4_K GEMV at `M ∈ {4, 8}` against the `M ∈ {16, 32}` multi-warp path as control (§9 #6). In-process, socket-free, `cargo bench`, ratchet-down against a committed self-baseline; PP-6 permits it because it is not a comparator ratio. Arms after §12 row 13; its variance obeys §4.3 (`n >= 5`) before it ratchets.

### §7.4 Status vocabulary

| status | meaning | requires | on derived expiry |
|---|---|---|---|
| `MEASURED` | conformant receipt, fresh pin, PP-26 passed on every band in the ladder | receipt path, commit, shas, `armed_by` where ARMED | — |
| `UNMEASURED` | temporary; in the denominator | `owner`, `expires` (derived, §12) | release `FAIL: INSTRUMENT` for the cell; `scripts/spec_conformance.sh` reports it |
| `NA` | permanent; out of the denominator | `reason`, `decided_by`, `date` | — |
| `INVALID-CORRECTNESS` | PP-26 absent or failed on this band | issue id | as `UNMEASURED`; can never be a baseline |
| `NONCONFORMANT-VALID` | historical record; cited, never a baseline | ledger row | — |
| `COMPARATOR_STALE` | PP-20 tripped | re-pin | as `UNMEASURED` |

`Skip` is not a status. `SUSPECT_DISPATCH` is not a status; a dispatch anomaly is a §9 finding with a mechanism. On the wire the legacy `comparator_status` field keeps its own token set (`UNMEASURED`, `NOT_APPLICABLE`, `MEASURED`); the vocabulary above lives in the per-band `status` field.

### §7.5 Arming conditions for this document

`ARMED` when, in one PR: PP-6, 26, 27, 28, 29, 33 are `ARMED`; PP-2, 13, 24, 30 are `ARMED` (the effective-config endpoint); the JOIN (§12 row 7) reproduces the fixture to four decimals; one `MEASURED` reference-cell receipt at `n >= 5` interleaved exists; `perf-matrix.yaml`'s legacy B1/B2 floors are replaced by P-5 rows with `δ` and author, in the same commit.

---

## §8 Scope

| cell | host | class | model | parity-gated (L3) | self-regression (L2) |
|---|---|---|---|---|---|
| **REFERENCE** | `lambda` RTX 4090 | cuda | Qwen2.5-Coder-7B-Instruct Q4_K_M | **yes**, per §7.2 arming | yes |
| shakedown | `gx10` GB10 (aarch64) | cuda (sm_121) | same | no; **first conformant shakedown cell** (§12 row 15) | yes, after §9 #1a |
| cpu | `intel` Mac Pro 2019 | cpu / wgpu | same | no | yes |
| deferred | `mini` M4 | `NA{#2841}` (§12 row 5) | same | no | after PP-16 |

No dates here. Every cell's `expires` is **derived by `scripts/spec_conformance.sh`**, which walks the §12 obligation DAG and writes `evidence/parity/derived_expiries.json`; a cell inherits the latest expiry among the rows that block it. A second host is promoted to L3 only after the reference cell is `MEASURED` with every ladder band `ARMED`.

---

## §9 Known defects, by priority

| # | defect | mechanism (file:line) | size `[V]` | lever | host | status |
|---|---|---|---|---|---|---|
| 1 | Blackwell prefill is a per-token serial loop | `crates/aprender-serve/src/cuda/gpu_profile.rs:517-531` (`select_prefill_path`: `cc >= SM12X_MIN_CC ⇒ Serial`; the predicate is numeric — datacenter Blackwell sm_100/103 and Thor sm_110 are below it unless `BATCHED_PREFILL` overrides), consumed by `run_prefill` in `generate_2.rs` | 16.75 s fixed at 513 prompt tokens = 32.65 ms per prompt token against 32.1 ms per decode step `[C]`, fitted over 28 of 30 samples (`evidence/perf-gate-001-w1-gx10/samples.c1.r1.jsonl.gz`). **Transport-dependent:** measured on the blocking `/v1/chat/completions` path only; `generate_1.rs` (streaming) carries no `cc >= 120` guard, and the streaming gx10 c=1 run in `evidence/parity-http/findings.json` shows TTFT 34.27 ms with no fixed cost | fix the KV-scatter root cause the comment at `generate_2.rs:261-283` names; `BATCHED_PREFILL=1` is a **measurement arm** (poisons KV, PMAT-810), not a fix | gx10 | OPEN, mechanism named |
| 1a | `prefill_multi_prompt` had no Blackwell guard; gx10 `m >= 2` ran the KV-corrupting path | guard landed (this PR): `gpu_profile.rs::multi_prompt_prefill_allowed` consulted at `generate_batched_streaming.rs:880` (`prefill_and_scatter`) and `:1045` (`recycle_slots_batch`); the refused batch takes the serial per-prompt fallback and prints `[PMAT-810] … refused` | gx10 `c>=4` receipts `INVALID-CORRECTNESS` (`evidence/perf-gate-001-w1-gx10/server-full.log.gz:104`) | guard landed; `perf041` must still classify gx10 `m >= 3` as a defect at the pre-guard commit, PASS after, defect again on revert — a gx10 transcript this host (sm_89) cannot produce | gx10 | guard LANDED (this PR); the on-device RED→GREEN→RED transcript is OPEN |
| 2 | Batched CUDA decode emits garbage for every `m > 1` | #2753; `[PMAT-044] Batch m=3 done` with constant token ids (`evidence/perf-gate-001-w1-gx10/server-full.log.gz:122-126`) | every `c > 1` figure in the tree | land #2809; wire `perf041` nightly, missing marker = RED (PP-26) | lambda | **OPEN, P0** |
| 3 | Prefill 0.275× at c=1, host-side | #2697 `nsys`: `cuStreamSynchronize` 57.4%, `cuMemcpyHtoD_v2` 32.8% over 1,018 sync copies, `cuMemAlloc_v2` 3.3% over 904 allocs, `cuLaunchKernel` **0.7%** | 2,860 against 10,399 `tok/s` (`evidence/parity-http/findings.json:29-30`, `:79-80`; ratio at `:122`) | pre-allocated pinned buffers, async copies, allocator reuse; prediction: CUDA-API share of copies and allocs falls from 93.5% to under 10%, prefill more than doubles | lambda | OPEN — **largest single-stream gap, gated at c=1 (P-3)** |
| 4 | Decode 0.650× at c=1 | #2694 | 103.26 against 158.90 `tok/s` (`evidence/parity-http/findings.json:21-22`, `:71-72`; ratio at `:121`); ITL 9.68 against 6.29 ms (`evidence/parity-http/findings.json:46`, `:96`) | profile after #3 (prefill fixes may move c=1 overhead); §7.3 GEMV is the named kernel lever | lambda | OPEN, sized |
| 5 | Batched decode path-switch penalty | #2771/#2753: eager launch against graph replay; per-token `model.write()`, recycle/join rescan, per-slot CPU embed, per-slot `try_send` | 2.93 → 5.01 ms per token (1.71×); `agg(2)` at 0.82× of one client | lift per-token host work out of the loop; prediction: at most 3.5 ms per token, break-even below c=2 | lambda | OPEN, sized, **largest lever on the gated cell** |
| 6 | Batched Q4_K GEMV single-warp at `M <= 8` | `cublas_prefill/mod.rs:29-34` names the missing M=4 multi-warp kernel; decode is under capture so cuBLAS is bypassed (`cublas_prefill/attention.rs:995`, `:1022`) | unsized | §7.3 bench, then the M=4 kernel | lambda | OPEN |
| 7 | VRAM 3.08× at c=1 | `evidence/parity-http/findings.json:16-17`: 14,030 against 4,554 MiB; **9,476 MiB** non-comparator `[C]`, roughly 20 KV slots of headroom at 448 MiB per slot (f32 KV, 4 B/element) consumed before a second request exists | determines `max_batch` (11 on this host) and therefore the ladder | account the 9.5 GB via the effective-config endpoint (`cuda.vram` breakdown: weights, prefill cache, KV per slot, `used_peak_bytes`); Arm D re-adopted REPORTING | lambda | OPEN, unscheduled until §12 row 6 |
| 8 | `cuda-batch = ["cuda"]` implication was inverted; two sites branched on `cuda-batch`; the `.apr` Q4K GPU serve path compiled to a stub under `--features cuda` whose error the caller **caught and silently downgraded** to the generic GPU path | was `crates/apr-cli/Cargo.toml:89-90`, `handlers_include_01.rs:95`/`:148`; now `handlers_include_01.rs` gates the path on `cuda` alone and the stub is deleted (this PR) | the §2.1 row 1b build defect at that artifact's commit | `INVALID-BUILD` is decided from `server_config.build_features_cli` returned by `GET /v1/effective-config` (PP-2), **never from a harness flag** — `provenance.feature_set` was copied from `--server-feature` and proved nothing about the build | all | **FIXED** (this PR); `cuda-batch = ["cuda"]` stays as a compatibility alias |
| 9 | Harness uses a boolean accelerator flag | three sites: `scripts/llama_pin.toml:255`, `scripts/parity_host_receipt.sh:185` and `:187`, `scripts/perf041_batched_parity_probe.sh:25` (plus `scripts/llama_pin.toml:183` in the lane §5.3 retires) | PP-15 | `--gpu-layers all`, and `--gpu-layers 0` where the lane exists to prove the CPU path | all | OPEN |
| 10 | `mini` declared a compute class with no reachable path | #2841; `crates/aprender-serve/Cargo.toml:52` pins `features = ["cuda"]` | PP-16 | `NA{decided_by: spec-owner, reason: no Metal inference path (#2841), date: 2026-09-02}` in `perf-matrix.yaml` | mini | **DONE** (§12 row 5) |

Scheduler (#2, #5) and kernel (#6) work are sequenced, never concurrent on one host, because concurrent changes confound every A/B (§10).

---

## §10 Engine track — predict, then verify

Engine work proceeds from today and needs no comparator, no matrix run and no §12 row. It may not announce a ratio (PP-12).

**Artifact:** `AbRecord` (`crates/aprender-test-lib/src/perf_gate/`), `serde(deny_unknown_fields)`, PP-32. Two arms, interleaved (A,B,A,B,…) in one harness invocation; `delta_kind: config` (one binary, one flag) or `code` (two binaries, two commits, both effective-configs diffed — any difference outside the declared delta is a hard error). The prediction is written into the probe script above the run, before it executes.

**Reconciliation.** Every `AbRecord` whose change later appears in a receipt has its interval compared with the receipt's. A non-overlap terminates in a named protocol delta (workload, band, sampler, stream, admitted concurrency, or a config diff) or in a retraction of the `AbRecord`. Two unexplained contradictions retire the engine track's self-baseline and force it onto W1-shaped A/B.

**Registered predictions (falsifiers in `RATIONALE.md` §10):**

| subject | prediction | kill if |
|---|---|---|
| §9 #1 probe (`scripts/perf002_prefill_path_probe.sh`, gx10) | prefill wall time linear at ≈32.6 ms per prompt token on the default arm; collapses to ≈0.35 s at 512 prompt tokens under `BATCHED_PREFILL=1` | flat in prompt length, or no collapse |
| §9 #5 fix (lambda) | batched decode at most 3.5 ms per token; `agg(2)` above one client | above 3.5 ms per token after the per-token host work is removed |
| §9 #3 fix (lambda) | copies and allocs under 10% of CUDA API time; prefill above 5,700 `tok/s` | the share stays above 30% |
| effective-config endpoint | `max_batch = 11` reconstructs from reported inputs (28 layers, 4 KV heads, head_dim 128, 4096 ctx, f32 KV at 4 B/element ⇒ `kv_per_slot` = 469,762,048 B = 448.0 MiB, 3.5 GB reserve, clamp [1, 32]) | it does not; then the KV budget is not the mechanism and §9 #7 is scoped wrong |
| JOIN fixture | reproduces `0.5341/0.2308/0.1685/0.0967` and `0.5873/0.9231/1.3525/1.5540` to four decimals with zero GPU. **The statistic is historical and must be named to reproduce:** aggregate = median over the 2 runs per lane per band of the run-level field `tokens_per_sec`; decode = median over the same 2 runs of `decode_tok_per_sec` — **not** §3's per-request `dec`, which the fixture cannot express because every request row carries the run's p50 inter-token latency (`evidence/parity-http/bands/apr-c1.json`) | any digit differs in the new Rust |

**Refusals.** No `HW_DP4A_Q4K` re-arm (correctness default, cosine 0.9186 against a 0.95 floor). No `FUSED_GATE_UP=1`: the flag is read at `crates/aprender-serve/src/cuda/gpu_profile.rs:134` and refused at `:274-293` (`detect_fused_gate_up`) whenever the Q4_K variant is not `HwDp4a`, because it would select a module the PTX preload path does not compile. No `BATCHED_PREFILL=1` as a fix (§9 #1). No re-run of either cell at `745fa8588` (PP-9). No percent-of-roofline figure (PP-12, PP-23).

---

## §11 Non-goals

No latency bound (P-8). No summing attribution identity (Σ samples / span = c by construction; per-phase averages and the per-lane `overhead_share` are used instead). No iteration budget. No causal independence between levers. No non-CUDA L3 gate (§8). No training, LAPACK or datacenter serving — `NA{decided_by}` in `perf-matrix.yaml`. No review-process rules — `PR-REVIEW-SKILL-002-v2.md` owns those.

---

## §12 Obligations — a DAG, not a deadline

Every row has an owner. `expires` is a date only on root rows; every other row inherits the latest expiry among its blockers, **computed by `scripts/spec_conformance.sh` and written to `evidence/parity/derived_expiries.json`**. An expiry moves only by an amendment recording who moved it and why (Appendix D). On expiry: an instrument row makes every cell it blocks `FAIL: INSTRUMENT`; a speed row makes the release notes carry `NO SPEED DELIVERED` with the row id. The chain gates what may be **claimed**; nothing in it gates what may be **investigated**.

| row | deliverable | discharges | owner | blocked_by | status / expires |
|---|---|---|---|---|---|
| **0a** | `run_gate` dispatches every arm through `run_phased`, which reads the arm's `phase:` from `perf-matrix.yaml` (`arm_phase`) and demotes FAIL to REPORT outside it — `arm_a_self_regression` and `arm_l3_parity` replace `arm_a_scaling`/`arm_b_adoption`; selftests `phase_guard_{b_merge,b_release,a_merge}` | PP-6 | perf-gate | — | **LANDED** (this PR) — `scripts/perf_gate.sh:63`, `:80`, `:938` |
| **0b** | `--stream` passed; `stream_request = ChatRequest { stream: Some(true), ..request.clone() }` on both the streaming and the blocking constructor; `ignore_eos` in `prompts-w1.jsonl` and its `_meta`; `short_of_n_predict` counted and fatal; `stream_mode` + client dual witness; `usage` on the terminal chunk | PP-27, PP-28 | perf-gate | — | **LANDED** (this PR) — `crates/aprender-test-lib/src/llm/client.rs` (`stream_request_keeps_seed_and_ignore_eos`, `blocking_request_keeps_seed_and_ignore_eos`), `crates/aprender-test-lib/src/perf_gate/metrics.rs`, `scripts/gen_prompts_w1.py`. **The blocker was the subject's own SSE terminal chunk**, which carried no `usage` object, so a streaming receipt had no server token count to obey PP-28 with; that is fixed here |
| **0c** | receipt signing; and the `kv` block that replaced `kv: None` | PP-21; the memory arm | perf-gate | — | **LANDED** (this PR), in two halves: *signing* — `scripts/perf_receipt_sign.sh` with `sig_missing` / `sig_ok`; *kv* — the block is no longer hand-built at `crates/apr-cli/src/commands/test_llm_band.rs:327` but read from `GET /v1/effective-config` (row 6), so it cannot disagree with the server |
| **0d** | the boolean accelerator flag replaced by a quantity in `scripts/llama_pin.toml` and every lane script (§9 #9) | PP-15 | perf-gate | — | **LANDED** (this PR) — `scripts/check_comparator_flags.sh` (`boolean_flag` / `quantity_flag`) |
| **0e** | `scripts/spec_conformance.sh` + `scripts/lib/spec_conformance.py` wired into `guard-runner-labels`; `scripts/check_mutation_registry.sh` and `scripts/lib/mutation_registry.py` retired in the same commit | PP-29 | perf-gate | — | **LANDED** (this PR) — `conformance_missing` / `conformance_ok` |
| **1** | land #2809; Blackwell guard on `prefill_multi_prompt`; `perf041` in `cuda-nightly.yml`, missing marker = RED; PP-26 witness in the receipt | PP-26; §9 #1a, #2 | serve | — | **OPEN.** The Blackwell guard (`gpu_profile.rs::multi_prompt_prefill_allowed`, consulted at `generate_batched_streaming.rs:880` and `:1045`) and the `perf041` witness wiring land in this PR; #2809 does not. **The first wired gx10 night is EXPECTED RED** — that is §9 #1a's before-state, and it is a finding to file, not a flake to rerun. lambda witness taken 2026-09-02 at `a765c86a5` under v3.1: PASS on every band, `m=1` agreement recorded at 3 tokens (`evidence/perf041/lambda/witness.json`); under the v3.0 rule the same run was DEFECT on every `c>1` band, which is what §7.1's `[U]` anticipated. Expires **2026-09-19** |
| 2 | `scripts/perf002_prefill_path_probe.sh` + `decompose()` with mandatory refusals (negative slope, `R²` bound, fewer than 2 distinct token counts, bimodal → per-mode) | §9 #1 mechanism at `HEAD` | serve | — (needs a gx10 window) | **OPEN**, 2026-09-19 |
| 3 | `/props` read on the pinned build at the withdrawn argv; `llama_pin.toml` becomes a per-band template with `pinned_on`/`pin_expiry`; a numeric `batch_size <= 1` refused; the CLI-differential lane retired | §5.3; PP-20 | spec-owner (decision) + perf-gate | — | **LANDED** (this PR) — `scripts/check_llama_pin.sh` (`pin_stale` / `pin_fresh`), `scripts/check_comparator_flags.sh` |
| 4 | decode and prefill captured on every band from the conformant producer | PP-4 | perf-gate | 0b | **LANDED** (producer side, this PR) — `band_metric_absent` / `historical_cited`; the first receipt that exercises it is row 15 |
| 5 | `mini` → `NA{decided_by, #2841}` in `perf-matrix.yaml` | PP-16 | perf-gate | — | **LANDED** (this PR) — `class_unreachable` / `class_na` |
| 6 | `GET /v1/effective-config` (§5.2) stored verbatim; `started_utc` and `clock_source`; `slots_admitted` on both lanes; derived ladder | PP-2, 13, 24, 30; §9 #7 accounting | serve | — | **LANDED** (this PR) — the route is unconditional and its JSON shape is identical on CPU and CUDA builds (`cuda: null` off CUDA); `config_missing` / `config_present`, `inferred_field` / `reported_field`, `admission_unequal` / `admission_na`, `timestamp_absent` / `timestamp_ok` |
| 7 | the JOIN: `ComparatorStatus::Measured { baseline }` + wire token; `--comparator-url`; a second `LlmClient` in the band loop; PP-22 refusal; the "always UNMEASURED" test at `crates/apr-cli/src/commands/test_llm_band.rs:719` inverted **because the spec reverses the rule** — a comparator lane is now representable; accepted against the zero-GPU fixture | PP-3 same-run, PP-22, PP-25 | perf-gate | 0a, 0b, 0e | **LANDED** (this PR) — `join_fixture.rs` (`join_ok`, `join_mismatch`), `ratio_bare` / `ratio_paired`, `client_mismatch` / `client_ok` |
| 8 | `perf-matrix.yaml`: the `scaling_efficiency` ratchet removed; per-band `agg(c)`, `dec(c)`, `prefill(1)` ratchets seeded at the last `MEASURED`; `δ` rows with an author replacing the legacy B1/B2 floors | PP-31, PP-33 (matrix half) | perf-gate | 7 | **LANDED** (this PR) — `self_regress_fail` / `agg1_improve_ok`, `threshold_outside_matrix` / `threshold_in_matrix` |
| 9 | `AbRecord` type | PP-32 | perf-gate | — | **LANDED** (this PR) — `abrecord_comparator` / `abrecord_ok` |
| 10 | `check_no_claim_literals.sh` universe widened to `docs/specifications/`; the live uncited figures receipted or baselined with a dated comment; `docs/benchmarking-gate-spec.md` archived; `parity_block.py` thresholds removed | PP-12, PP-33 | perf-gate | — | **LANDED** (this PR) — `claim_unreceipted` / `claim_receipted` |
| 11 | `git merge-base --is-ancestor` + three shas in the validator | PP-18 | perf-gate | — | **LANDED** (this PR) — `ancestor_fail` / `ancestor_ok` (the `PERF_GATE_GIT_DIR` seam makes the case table hermetic) |
| 12 | perf workflow with `concurrency: perf-<host>`; pin expiry field; client sha field | PP-19, 20, 25 | perf-gate | — | **LANDED** (this PR) in the half that exists: `scripts/check_perf_concurrency_groups.sh` (`isolation_breach` / `isolation_ok`), the pin expiry (row 3) and the client sha. **The CI half for `lambda` and `mini` is `NA{decided_by: infra#359 (lambda has no CI runner by policy), #2841 (mini has no reachable class), date: 2026-09-02}`** — on those hosts the producer of isolation is the harness itself: `scripts/parity_host_receipt.sh` takes an exclusive `flock -n` on the cell lock for the whole run (`:158-176`; a held lock is a REFUSAL, never a queue) and `scripts/perf_isolation.sh` writes the `nvidia-smi --query-compute-apps` record before and after every band (`foreign_pid_breach` / `foreign_pid_ok` in its case table) |
| 13 | `cargo bench` batched Q4_K GEMV at `M ∈ {4,8}` against `M ∈ {16,32}`; then the M=4 kernel | §7.3, §9 #6 | serve | 1 | **OPEN**, derived |
| 14 | `scripts/measure_bandwidth.sh` per host, committed | PP-23 | perf-gate | — | **LANDED** (this PR) — `evidence/bandwidth/lambda.json`; `roofline_exceeded` / `roofline_aggregate_ok`. Hosts other than `lambda` stay `UNMEASURED{owner: perf-gate}` until their file lands |
| 15 | **gx10 shakedown cell**, W1, `n >= 5` interleaved, at a commit containing 0b, 0c, 1, 6, 7 → a new LEDGER row | first conformant receipt | perf-gate | 0b, 0c, 1, 6, 7, 12 | **OPEN**, derived |
| 16 | W3 open-loop workload | P-8 | perf-gate | 0b | **OPEN**, 2026-10-23 |
| 17 | W4 token sweep | §9 #1 decomposition on lambda | perf-gate | 0b | **OPEN**, 2026-10-23 |
| **18** | **reference cell**, lambda, W1, `n >= 5` interleaved, both lanes, derived ladder, one commit; `armed_by` set for every band that PASSES; document → `ARMED` | §7.5; §9 #3, #4 sized conformantly | perf-gate | 15 clean at `--phase merge` | **OPEN**, derived |
| **19** | **speed**: §9 #5 (per-token host work) — registered prediction at most 3.5 ms per token | throughput on the gated cell | serve | 1, 6 | **OPEN**, derived (row 1) |
| **20** | **speed**: §9 #3 (prefill sync copies and allocs) — registered prediction under 10% API share | throughput on the gated cell, at the gated c=1 metric | serve | — | **OPEN**, **2026-10-23** |
| **21** | **speed**: §9 #1 KV-scatter root cause on Blackwell | gx10 c=1 fixed cost 16.75 s → under 1 s, conditional on row 2 showing the cost exists on the streaming transport | serve | 2 | **OPEN**, derived (row 2) |
| **22** | **instrument**: a top-2 logit margin per generated token on the wire (`logprobs` on the SSE delta, both engines), so the witness can classify an `m=1`↔`m=c` divergence as a near-tie flip (margin below a declared τ at the divergence index) or a defect; PP-26 (c) then becomes a gate | PP-26 (c); the residual that (a)+(b) cannot see — a whole batch that is coherent and identically wrong | serve | — | **OPEN.** Root row. Why it exists: `GET /v1/chat/completions` answers `logprobs: null` (`realize_handlers_completion_request.rs`), so the divergence lambda measured (`evidence/perf041/lambda/m1-vs-m4-three-prompts.txt`) can be read by a person but not classified by the witness. Expires **2026-10-15** |

Rows 19–21 are the deliverable. A version of this table with no speed row is a defect of the table.

**Kill criteria** (any one stops the plan at that step and is filed as a finding): row 2's probe shows prefill flat in prompt length; row 7 does not reproduce the fixture; row 19's prediction fails; row 6's `max_batch` does not reconstruct; row 1 cannot close #2753 — then the product claim narrows to c=1 and §12 says so; any of rows 0a–0e or 6 reaches its second week without a committed finding in `evidence/`; row 18's receipt PASSES P-5 at every ladder band — then there is no parity gap on the reference cell and the epic closes rather than manufactures one.

---

## §13 Superseded

Fifteen documents (~14,800 lines) in `aprender`, `qwen-coder-deploy` (archived at `4fadc7c`), `realizar` and `trueno` — the latter two repositories archived on GitHub `[V]` (`gh repo view --json isArchived`). Sixteen rows with local paths, `repo_state` and the corrected count are in `evidence/parity/LEDGER.md` §13; the sixteenth is `docs/archive/perf-2026-09-02/performance-parity-llama.cpp.md`, archived by this PR. All 42 `spec:` fields in `docs/roadmaps/roadmap.yaml` name this document. `docs/benchmarking-gate-spec.md` is archived to `docs/archive/perf-2026-09-02/` by §12 row 10.

---

## Appendix A — `APR-PERF-GATE-001` v2.2 `I-nn` → `PP-nn`

| I-1→PP-1 | I-2→PP-2 | I-3→PP-3 | I-4→PP-7 | I-5→PP-5 | I-6→PP-6 | I-7→PP-19 | I-8→PP-20 | I-9→PP-9 |
|---|---|---|---|---|---|---|---|---|
| **I-10→PP-21** | **I-11→PP-22** | **I-12→PP-12** | **I-13→PP-11** | **I-14→PP-10** | **I-15→PP-25** | **I-16→PP-13** | **I-17→PP-14** | **I-18→PP-15** |

No v2.2 ancestor: PP-4, 8, 16, 17, 18, 23, 24, 26–33. v2.2 Arm A → PP-31 (re-shaped); Arm D → `cuda.vram.*`/`kv.*` fields in PP-2, REPORTING; Arm E → `itl_p95`, REPORTING.

Roadmap cross-references (`docs/roadmaps/roadmap.yaml` notes carry the same map): PERF-004 → PP-2/PP-13/PP-30 · PERF-006 → PP-2/PP-13/PP-16 · PERF-007 → PP-21 · PERF-011 → §12 row 15 · PERF-012 → PP-16/PP-24 · PERF-017 → PP-4 (`prefill`) · PERF-018 → PP-24 · PERF-019 → PP-25/§5.3 · PERF-020 → §5.1 W2 · PERF-021 → PP-15 · PERF-023 → PP-24 · PERF-024 → PP-27/PP-28 · PERF-025 → PP-4/PP-7 · PERF-031 → PP-19 · PERF-033 → PP-20 · PERF-039 → PP-28 · BENCH-003 → §4.3.

## Appendix B — Receipt (normative fields)

`run_id` · `started_utc`, `clock_source` (PP-30) · `provenance{subject{commit,sha256,feature_set}, comparator{commit,cmake,sha256,pin_expiry,props}, client{commit,sha256}, host, compute_class, server_config(verbatim), model{path,sha256,bytes}, quantization}` (PP-2, 18, 20, 25) · `workload{id,window_ms,warmup_requests_per_worker,quiesce_ms,cooldown_ms,n_predict,sampler{temperature,seed,ignore_eos}}` (PP-28) · `tokenization{method}` (PP-11) · `ladder{declared,derived,slots_admitted{apr,llama}}` (PP-24) · per band: `c`, `stream_mode`, `stream_witness`, `witness{batch_invariance, divergence_at}` (PP-26, 27), `samples[]` (PP-7), `short_of_n_predict` (PP-28), `timeouts`, `drain_ms` (PP-5, 10), `agg`, `dec`, `prefill`, `ttft`, `itl_p95`, `scaling_efficiency`, `overhead_share` per lane, `roofline_tok_per_sec` (PP-23), `baseline{…same schema…}` (PP-3), `ratios{agg,dec,prefill}` each `{point, lcb95, method, n}` (P-5), `status` (§7.4) · `signature` (PP-21).

Appendix B names are logical. The wire keeps the v2.2 spellings (`aggregate_tok_per_sec`, `decode_tok_per_sec`, `prefill_tok_per_sec`, `ttft_p50_ms`/`ttft_p95_ms`, `itl_p50_ms`/`itl_p95_ms`) so today's readers keep working; `schema_version: 3` marks the additive shape.

## Appendix C — `LEDGER.md` row schema

`# · started_utc · host · class·accelerator · model·quant · workload · commit · interleaved · n_replicates · n_samples · receipts · conformance {RECORDED | CONFORMANT} · validity_by_band {c: NONCONFORMANT-VALID | INVALID-BUILD | INVALID-CORRECTNESS(issue) | MEASURED} · what_it_lacks[]`.

`n_replicates` is the number of replicates of §4.3 (recoverable from the `receipt.r{k}.json` filenames on pre-v3 cells); `n_samples` is the receipt's `n`, the count of retained request samples. The two were one ambiguous `n` column before, and §4.3's `n` is always the first of them.

Append-only. PP-9 binds on `RECORDED`. Rows entered under the pre-v3 eleven-column schema are kept verbatim under `## Superseded rows (schema v2)` and superseded by new rows rather than edited (audit CO-1, CO-2).

## Appendix D — Changelog

| version | date | change | reason | audit ref |
|---|---|---|---|---|
| 3.1 | 2026-09-03 | §7.2's release phase, run for the first time (0.65.0), FAILED over the pre-v3 lambda receipt on ArmC-sig (unsigned) and ArmD (no kv block) while the cell it belongs to is UNMEASURED and unexpired — the state §7.2's last line permits a release in. Both arms now REPORT for a historical receipt of an UNMEASURED cell and FAIL as before for any other input; four `pg` rows, both polarities | a release gate that cannot PASS in a state the spec allows is unsatisfiable, and an unsatisfiable gate gets bypassed for substance (#2696) | PMAT-744 |
| 3.1 | 2026-09-02 | PP-26 amended after the first lambda witness: the `m=1`↔`m=c` 64-token bar measured kernel-family fp divergence (three families, each batch-size invariant to the end, `evidence/perf041/lambda/`), not batching; (a) intra-batch invariance and (b) no frozen slot are gated, (c) the `m=1` agreement is recorded; §12 row 22 (margin instrument) added so (c) can become a gate; `witness.max_constant_run` added to the matrix. |
| 3.0 | 2026-09-02 | Complete rewrite: normative-only (§0.5); single invariant table with must-not-fire and selftest columns; layered gate L0–L3; first-PASS arming (P-6); non-inferiority verdict with two estimators and interleaved replicates (§4.3); `prefill` first-class and gated at c=1; comparator decided (§5.3); derived ladder and derived expiries; PP-26…PP-33 added; PP-31 replaces the `scaling_efficiency` arm; §12 as one DAG with speed rows; LEDGER validity-by-band | the predecessor stated its current rules as corrections of earlier wrong ones, so a reader had to reconstruct the rule from the correction | C-1…C-20, CO-1…CO-4 |
| 3.0 | 2026-09-02 | Line 3 gains a `**Status:**` line | every sibling spec in `docs/specifications/` carries one; `grep -lE '^\*\*Status' docs/specifications/*.md` did not find this document | — |
| 3.0 | 2026-09-02 | Header `Supersedes` names PR #2845 as unmerged and this PR as the archiver | #2845 is `state=OPEN, mergedAt=null`; stating a landing that has not happened is the class §0.5 exists to stop | — |
| 3.0 | 2026-09-02 | Header `Tree` drops the `02362ef8d` execution-plan citation | no `EXECUTION-PLAN-*` or `RECONCILIATION.md` file was ever added in any ref, and `02362ef8d` touches one file, this document's predecessor | — |
| 3.0 | 2026-09-02 | Header `Companions` adds `docs/audits/parity-spec-audit-2026-09-02.md`, `evidence/bandwidth/`, `evidence/parity/derived_expiries.json` | a companion cited by bare filename that is in no repository is `[U]` under §0.3 | CO-3 |
| 3.0 | 2026-09-02 | §2.1 row 1 split into 1a (the c=1 pair, `53062e7f3` `[V]`) and 1b (`bands/`, commit `[U]`) | the two artifacts have different dates, different protocols and different provenance; one row asserted `[V]` for both | C-15 |
| 3.0 | 2026-09-02 | §2.1 last paragraph rewritten: the `0.591/0.395/0.544/0.401` series is named a cross-run quotient and forbidden | it is a `745fa8588` subject over a 2026-08-25 comparator, i.e. a P-1 violation the document was repeating | C-15 |
| 3.0 | 2026-09-02 | §2.1 row 3 records that the fit excludes 2 of 30 samples per replicate, and that the fixed cost is transport-dependent | publishing a `[C]` fit whose exclusions are undocumented is the shape §0.3 exists to prevent | C-18 |
| 3.0 | 2026-09-02 | Every figure in §2 carries its `evidence/` path on the same line or table row | PP-12's own rule, applied to this document | C-20 |
| 3.0 | 2026-09-02 | §2.3 header changed to `[V]` by the command in each row, and a `verified_by` column added | the attribution rested on a plan that exists in no repository | CO-3 |
| 3.0 | 2026-09-02 | §2.3 and §9 #6 line citations corrected to `attention.rs:995`/`:1022` and `cublas_prefill/mod.rs:29-34`; §9 #1 to `generate_2.rs:284-288` | the cited lines had drifted | — |
| 3.0 | 2026-09-02 | §3 and §5.1 state that `n_predict` is carried on the wire as `max_tokens` | the corpus, the loader and both servers use `max_tokens`; `n_predict` names only the comparator's launch argument | C-12 |
| 3.0 | 2026-09-02 | §5.1 states the dual-bound window, the warmup requests, the quiesce and the cooldown as normative, all resolved from `perf-matrix.yaml`, and records the corpus-sha rotation | the protocol constants were prose in one place and literals in another | C-12, C-17 |
| 3.0 | 2026-09-02 | §6 selftest names carry a surface prefix, and stale line citations replaced | `spec_conformance.sh` joins on a surface as well as a name; several citations had drifted | C-20 |
| 3.0 | 2026-09-02 | §6 PP-14's producer named as `OffloadReport.pp14_holds`, PP-8's status made honest | a row that names no producer cannot be joined | C-20 |
| 3.0 | 2026-09-02 | §7.1 states what `ci / gate` concretely is | a rule that names a check nobody can locate is unenforceable | — |
| 3.0 | 2026-09-02 | §7.2 states what a release concretely is, and where the perf release gate runs | "release phase" named no command | — |
| 3.0 | 2026-09-02 | §8 and §12 derive expiries with `scripts/spec_conformance.sh` into `evidence/parity/derived_expiries.json` | the previous text named a tool that computes nothing of the kind | C-7 |
| 3.0 | 2026-09-02 | §9 #1 carries the transport; #8's lever moved to `server_config.build_features_cli`; #9 lists all three flag sites; #10 marked DONE | a lever keyed on a harness-declared field is theater | — |
| 3.0 | 2026-09-02 | §10's `FUSED_GATE_UP` refusal cites `gpu_profile.rs:266`; the JOIN-fixture prediction names its statistic | the previous citation could not be located in the tree; the fixture's statistic is not §3's `dec` | — |
| 3.0 | 2026-09-02 | §12 rows re-stated with LANDED/OPEN status and the discharging file or test named | an obligation table that does not say what discharged a row cannot be audited | C-8 |
| 3.0 | 2026-09-02 | Appendix C gains `n_replicates` and `n_samples` | one `n` column meant two different things | CO-2 |
| 3.0 | 2026-09-02 | Appendix E gains the landing checklist and the pre-push command list | the enforced procedure lived only in the guards | — |
| 2.x | 2026-09-01/02 | see `docs/postmortems/perf-parity-review-2026-09.md` | the history of the predecessor lives in the post-mortem, never in this file (§0.5) | B-1…B-14 (audit not archived; `[U]`) |
| 3.0.24 | 2026-09-02 | §12 rows 19 and 21: the literal expiries (2026-10-16, 2026-11-06) are replaced by `derived` — both rows are blocked by live rows (1 and 2), and §12's preamble allows a literal date on root rows only; `scripts/spec_conformance.sh` refuses the typed dates (D3). Their expiry is now the later of their blockers'. | C-7 |
| 3.0.25 | 2026-09-02 | Citations re-verified against the tree after the code landed: §3/§4.3/§9 #7/Appendix A name the endpoint's `vram.*` keys (no `vram_peak` key exists); §9 #8 marked FIXED; §9 #1/#1a, §2.3 and §12 row 1 cite `gpu_profile.rs::select_prefill_path` / `multi_prompt_prefill_allowed`; §9 #3/#4 cite the decode and prefill lines of `findings.json`; §6 PP-6 and §12 row 0a name `run_phased`/`arm_phase`/`arm_l3_parity`/`arm_a_self_regression`; §7.1 ci.yml anchors by grep; §10's `FUSED_GATE_UP` read and refusal lines; §6 PP-12's surviving exclusions; §12 row 12 cites the `flock` that now exists; §6 PP-19 gains the foreign-PID surface | review of this PR |

## Appendix E — Landing a rule, and what to run before pushing

**The landing checklist** — this is the procedure the tree's own guards enforce, in the order they enforce it:

1. **Arm.** Add `arm_x()` in `scripts/perf_gate.sh` (each arm prints PASS/FAIL/REPORT/SKIP and returns 0 or 1) and dispatch it in `run_gate`.
2. **Classify every receipt key the arm reads** in `scripts/perf-receipt-fields.yaml` under `fields:` with a class. `scripts/check_perf_receipt_fields_have_producers.sh` extracts the reads from `perf_gate.sh` mechanically and turns RED on an unclassified key or an unknown receiver variable; a new non-receipt variable goes under `non_receipt_receivers.perf_gate`.
3. **Matrix row.** Every number the arm compares against goes in `scripts/perf-matrix.yaml` with `threshold_class` and `author` (PP-33). No literal in the script.
4. **§6-named cases.** Add the must-fire and must-not-fire cases under the exact names in this document's §6, on the surface §6 names. `scripts/spec_conformance.sh` joins the two and fails on either half.
5. **`ci.yml` line.** A new script is invisible to the wiring meta-guard unless it has a `--selftest)` arm and a `run:` line inside the `guard-runner-labels` job (`.github/workflows/ci.yml:443`).
6. **PR body.** Paste the must-fire mutation turning the guard RED, and the fixture leaving it GREEN. A guard whose RED was never observed is not known to be able to fail.
7. **`pr-review` receipt** for the PR, as `PR-REVIEW-SKILL-002-v2.md` requires.

**Before pushing a PP-29-class branch, run exactly this:**

```
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings -A unused-variables
cargo deny check advisories licenses sources
bash scripts/check_guards_are_wired.sh
bash scripts/perf_gate.sh --selftest
bash scripts/spec_conformance.sh --selftest
bash scripts/spec_conformance.sh
bash scripts/check_no_timing_in_required.sh
bash scripts/check_shell_lint_ratchet.sh
bash scripts/check_perf_claims_cite_receipts.sh
bash scripts/check_no_claim_literals.sh
bash scripts/check_roadmap_completion_is_cited.sh
```

`make tier1`/`tier2`/`tier3` are narrower than CI: their clippy omits `--all-targets`, and the pre-push list in `CLAUDE.md` omits `licenses sources` and clippy entirely. Run the list above, not the tiers, when the change touches a guard.
