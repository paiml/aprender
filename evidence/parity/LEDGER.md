# Parity matrix-run ledger

**PP-9: a cell, once run, is spent; it may not be re-run to green.** That rule is only
enforceable against a written record of what has been spent. This file is that record.

Governing spec: [`docs/specifications/PP-LLAMA-001-MASTER.md`](../../docs/specifications/PP-LLAMA-001-MASTER.md).
Row schema: Appendix C of that document. Status vocabulary: §7.4.

**Append-only.** One row per *cell run* — a **(host, workload, model, quantization, commit,
interleaved)** tuple driven across its bands (PP-9). A row is added when the run is committed and
is never edited afterwards; a run found invalid is superseded by a **new** row that says why, and
the original stays where it is. Rows written under the pre-v3.0 eleven-column schema are kept
verbatim at the bottom of this file rather than rewritten.

## Conformance — two tiers (audit CO-2)

A single criterion that no row can meet is the same defect as a gate that has never fired, so
conformance is recorded in two tiers and **PP-9 binds on the lower one**:

- **`RECORDED`** — the row carries what the run actually produced: receipt directory,
  `receipt.commit`, host, class, model, quantization, workload, band set, and an explicit
  `what_it_lacks[]`. A `RECORDED` row spends its PP-9 key. It may be cited as history and may
  never be a baseline.
- **`CONFORMANT`** — every producer the master requires is present: `signature` (PP-21),
  `compute_class` and the whole §5.2 block **as reported by the server** (PP-2, PP-13), comparator
  pin and its expiry state (PP-20), `slots_admitted` on both lanes (PP-24), `started_utc` and
  `clock_source` (PP-30), the streaming witness (PP-27), the batch-invariance witness (PP-26),
  `short_of_n_predict` (PP-28), the client sha256 (PP-25), and `roofline_tok_per_sec` against
  per-sequence decode (PP-23). Only a `CONFORMANT` row may become a baseline or an `armed_by`
  receipt (§7.2).

Every row below is `RECORDED`. None is `CONFORMANT`, and each names what it lacks rather than
implying the absence away.

## Spent cells

| # | started_utc | host | class · accelerator | model · quant | workload | commit | interleaved | n_replicates | n_samples | receipts | conformance | validity_by_band | what_it_lacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0a | 2026-08-24T15:13:23Z (`llamacpp-39173bcac.json` `runs[0].timestamp`) | `lambda` | cuda · RTX 4090 | qwen2.5-coder-7b-instruct · q4_k_m | c=1 paired HTTP pair (pre-W1) | `53062e7f3` `[V]` ([`parity-http/findings.json`](../parity-http/findings.json) `apr_build`) | false (comparator lane 15:13–15:14Z, subject lane 15:15–15:17Z) | 3 runs per lane | 3 per lane | [`parity-http/apr-0.64.0-cuda.json`](../parity-http/apr-0.64.0-cuda.json), [`parity-http/llamacpp-39173bcac.json`](../parity-http/llamacpp-39173bcac.json), [`parity-http/findings.json`](../parity-http/findings.json) | RECORDED | `c=1: NONCONFORMANT-VALID` — paired, one client, streaming, comparator pinned at `39173bcac`; the replicate pair [`parity-http/lambda-apr.json`](../parity-http/lambda-apr.json) + [`parity-http/lambda-llamacpp.json`](../parity-http/lambda-llamacpp.json) (2026-08-25T07:27–07:31Z) reproduces the decode and prefill ratios | `[interleaving, n_replicates >= 5, receipt schema (these are harness JSON, not v3 receipts), signature, batch-invariance witness, slots_admitted, client sha256, started_utc in the artifact itself]` |
| 0b | 2026-08-25T07:43:43Z (`bands/llamacpp-c1.json` `runs[0].timestamp`) | `lambda` | cuda · RTX 4090 | qwen2.5-coder-7b-instruct · q4_k_m | band sweep c=1,4,8,16 (pre-W1) | **UNKNOWN** `[U]` — no provenance block in any `bands/*.json`; upper bound `ce712eae0` (the commit that added them); argv `[U]` | false (comparator lanes 07:43:43–07:47:43Z, subject lanes 07:48:41–07:54:09Z) | 2 runs per lane per band | 2 per lane per band | [`parity-http/bands/`](../parity-http/bands/) | RECORDED | `c=1: NONCONFORMANT-VALID`; `c>1: INVALID-BUILD` — the subject serialises; no band carries a correctness witness. The **PP-9 key is unspendable** because the commit is unknown, so this row records a fixture, not a spent cell | `[commit, binary sha256, argv, interleaving, n_replicates >= 5, equal windows per lane (subject c=16 spans 43.3/42.9 s against the comparator's 25.3/25.8 s), comparator yield at n_predict (every comparator request stopped at 112 completion tokens against `max_tokens` 128), signature, witnesses]` |
| 3 | UNKNOWN; upper bound 2026-09-01 (evidence commit). Producer wall-clock per `invocation.txt` only | `lambda` | cuda · RTX 4090 | qwen2.5-coder-7b-instruct · q4_k_m | W1 | `745fa8588` | false (single-lane sequential bands; no comparator lane existed) | 3 (`receipt.r1..r3.json`) | 540 (`receipt.n`, summed over bands) | [`perf-gate-001-w1-lambda/`](../perf-gate-001-w1-lambda/) | RECORDED | `c=1: NONCONFORMANT-VALID` — no stream (`producer-stdout.txt:6`), no comparator (`comparator_status: UNMEASURED` on every band), 28 of 55 retained samples short of `n_predict` (`samples.c1.r1.jsonl.gz`, minimum 24 generated tokens). `c=4,8,16: INVALID-CORRECTNESS(#2753)` — 0 of 485 samples short at `c>1` against 28 of 55 at c=1 (`samples.c4.r1.jsonl.gz`, `samples.c8.r1.jsonl.gz`, `samples.c16.r1.jsonl.gz`), the constant-token shape #2753 describes, in which no sequence ever emits EOS | `[stream, comparator lane, signature, started_utc, clock_source, kv/scheduler block, slots_admitted, client sha256, pin expiry, PP-26 batch-invariance witness, PP-28 short_of_n_predict counter]` |
| 4 | UNKNOWN; upper bound 2026-09-01 (evidence commit). `invocation.txt` records the producer launched 00:33:37Z | `gx10` | cuda · GB10 (aarch64, sm_121) | qwen2.5-coder-7b-instruct · q4_k_m | W1 | `745fa8588` | false (single-lane sequential bands) | 3 (`receipt.r1..r3.json`) | 258 (`receipt.n`, summed over bands) | [`perf-gate-001-w1-gx10/`](../perf-gate-001-w1-gx10/) | RECORDED | `c=1: NONCONFORMANT-VALID` — serial prefill on the blocking transport (`crates/aprender-serve/src/gguf/cuda/generate_2.rs:284-288`), 7 of 30 samples short of `n_predict`, and 2 of 30 samples per replicate bimodally fast (`samples.c1.r{1,2,3}.jsonl.gz`; 128 generated tokens in 4.21–4.24 s against 18.8–20.9 s for the other 28). `c=4,8,16: INVALID-CORRECTNESS(§9 #1a)` — `prefill_multi_prompt` engaged at every `m>=3` (`server-full.log.gz:104`, `:118`, `:130`) and decode steps carry slot-identical token ids (`server-full.log.gz:122-126`); 20 of 36, 29 of 64 and 51 of 128 samples short of `n_predict`, minimum 5 generated tokens | `[stream, comparator lane, signature, started_utc, clock_source, slots_admitted, client sha256, pin expiry, PP-26 witness, PP-28 counter, a suspect rule that can see bimodality]` |
| 5 | 2026-09-02T10:30:39Z and 2026-09-02T10:30:44Z (`started_utc` in each file) | `lambda` | cuda · RTX 4090 | qwen2.5-coder-7b-instruct · q4_k_m | comparator configuration probe (`GET /props`, no generation) | comparator build `39173bcac` `[V]` (`props.build_info`) | NA — a probe, not a paired run | 1 read per configuration | 0 (no requests generated) | [`props-39173bcac-template.json`](props-39173bcac-template.json), [`props-39173bcac-np16.json`](props-39173bcac-np16.json) | RECORDED | `NA` — this row spends no PP-9 key and carries no throughput. It is the discharge of master §12 row 3: at the template argv (`-c 4096`, no `-np`) the server reports `total_slots = 4` with `default_generation_settings.n_ctx = 4096`; under the decided argv (`-np 16 -c 16384`) it reports `total_slots = 16` with `n_ctx = 1024` per slot. The "4 slots by design" premise is true of the *template*, and the decided configuration changes the per-slot KV budget as well as the slot count | `[the /props body carries no kv_unified field, so that half of the llama_pin.toml dissent stays [U] from evidence; gx10 not probed]` |
| 6 | 2026-09-02T14:39:50Z (v3.0-rule run) and the `started_utc` of `witness.json` (v3.1 run) | `lambda` | cuda · RTX 4090 (sm_89) | qwen2.5-coder-1.5b-instruct · q4_k_m | PP-26 witness (`scripts/perf041_batched_parity_probe.sh`), ladder c=1,4,8,16, `CUDA_BATCH_WINDOW_MS=200` | `a765c86a5` `[V]` (`marker.json.commit`, binary sha256 in the marker) | NA — a witness, not a paired run | 1 | 4 bands × c slots + 2 references | [`perf041/lambda/`](../perf041/lambda/): `marker.json`, `witness.json` (v3.1), `witness-v3.0-rule.json`, `witness-variant-cublas-threshold-99.json`, `witness-variant-batched-prefill-0.json`, `m1-vs-m4-three-prompts.txt` (the server log is not committed: `*.log` is ignored) | RECORDED | c=1 `MEASURED` (reference self-agreement 100 of 128 with batched prefill, 128 with `BATCHED_PREFILL=0`); c=4,8,16 **PASS under v3.1** (every slot identical to every other to the end, no frozen slot); **DEFECT under the v3.0 rule** (`m=1` agreement 3 tokens on every band). Cross-m comparison on two prompts: three kernel families — `m=1`; `m=2,3`; `m=4,8,16` (`CUBLAS_GEMM_THRESHOLD` default 4) — each batch-size invariant to the end (agreement 128 / 104–107 within a family), parting from each other at 3–26 tokens with coherent text on both sides. `CUBLAS_GEMM_THRESHOLD=99` and `BATCHED_PREFILL=0` do not remove the parting. This row spends no PP-9 key and carries no throughput. | the top-2 margin that would classify the parting as near-tie or defect (master §12 row 22) |
| 7 | 2026-09-04T23:47:42Z (`bands[0].subject_effective_config.started_utc`) | `lambda` | cpu · RTX 4090 host, default-feature install resolves 0 GPU layers | qwen2.5-coder-1.5b-instruct · q4_k_m | dogfood parity protocol (`scripts/llama_pin.toml#protocol.http`, `scripts/parity_host_receipt.sh`), ladder c=1,4,8,16 — NOT W1/W2 | `8e1e9ad40` (published 0.65.2, `cargo install aprender --version 0.65.2 --locked`) | true | 5 | 250 subject / 691 comparator (`bands[].*.completed`) | [`dogfood/0.65.2/lambda.json`](../dogfood/0.65.2/lambda.json) `parity.lanes[cpu]`, run `e167c1ab` | RECORDED | {c1: MEASURED (c=1, no witness required) · c4: INVALID-CORRECTNESS(#2753/#2776) · c8: INVALID-CORRECTNESS(#2753/#2776) · c16: INVALID-CORRECTNESS(#2753/#2776)} | PP-26 witness on every c>1 band (none run) · PP-21 signature · PP-25 client sha256; c=1 verdict FAIL (decode 0.5945, prefill 0.0053) — a recorded FAIL, never a baseline for parity; self-regression seed at c=1 only (P-4, P-6) |
| 8 | 2026-09-05T01:36:58Z | `lambda` | cuda · RTX 4090 (sm_89), second install of the same crate with `--features cuda` (`/tmp/apr-0652-cuda`) | qwen2.5-coder-1.5b-instruct · q4_k_m | dogfood parity protocol, ladder c=1,4,8,16 — NOT W1/W2 | `8e1e9ad40` (published 0.65.2 + `--features cuda`) | true | 5 | 4425 subject / 5571 comparator | [`dogfood/0.65.2/lambda.json`](../dogfood/0.65.2/lambda.json) `parity.lanes[cuda]`, run `e52be880` | RECORDED | {c1: MEASURED (c=1, no witness required) · c4: INVALID-CORRECTNESS(#2753/#2776) · c8: INVALID-CORRECTNESS(#2753/#2776) · c16: INVALID-CORRECTNESS(#2753/#2776)} | PP-26 witness on every c>1 band (none run) · PP-21 signature · PP-25 client sha256; c=1 verdict FAIL (decode 0.6919, prefill 0.1777); the earlier "parity at c=16" is withdrawn — no witness, no number |
| 9 | 2026-09-04T23:54:26Z (`parity_attempt.started_utc` in the receipt, copied from `apr-cpu-c1.config.json` `server.started_utc`) | `gx10` | cpu · GB10 (aarch64, sm_121 host), default-feature install | qwen2.5-coder-1.5b-instruct · q4_k_m | dogfood parity protocol, ladder c=1,4,8,16 — NOT W1/W2 | `8e1e9ad40` (published 0.65.2) | true | 5 | 25 successful subject requests (`parity_attempt.bands[].subject.successful`) | [`dogfood/0.65.2/gx10.json`](../dogfood/0.65.2/gx10.json) `parity_attempt` (block REFUSED: zero-throughput band); kept work dir on the host `/tmp/tmp.LCmTlB8O8M` | RECORDED | {c1: MEASURED (c=1, no witness required) · c4: INVALID-CORRECTNESS(#2753/#2776) · c8: INVALID-CORRECTNESS(#2753/#2776) · c16: INVALID-CORRECTNESS(#2753/#2776)} | a parity block at all (refused, F12) · PP-26 witness on every c>1 band (none run) · PP-21 signature · PP-25 client sha256; c=8 and c=16: 0 successful requests in every replicate; c=1 subject/comparator medians are in the receipt and are a recorded FAIL-class fact, never a baseline |
| 10 | 2026-09-05T00:03:26Z (`parity_attempt.started_utc` in the receipt, copied from `apr-cpu-c1.config.json` `server.started_utc`) | `intel` | cpu · x86 CI runner box, under runner load | qwen2.5-coder-1.5b-instruct · q4_k_m | dogfood parity protocol, ladder c=1,4,8,16 — NOT W1/W2 | `8e1e9ad40` (published 0.65.2) | true | 5 | 155 successful subject requests (`parity_attempt.bands[].subject.successful`) | [`dogfood/0.65.2/intel.json`](../dogfood/0.65.2/intel.json) `parity_attempt` (block REFUSED: zero-throughput band); kept work dir on the host `/tmp/tmp.bdsZKmjHK3` | RECORDED | {c1: MEASURED (c=1, no witness required) · c4: INVALID-CORRECTNESS(#2753/#2776) · c8: INVALID-CORRECTNESS(#2753/#2776) · c16: INVALID-CORRECTNESS(#2753/#2776)} | a parity block at all (refused, F12) · PP-26 witness on every c>1 band (none run) · PP-21 signature · PP-25 client sha256; c=16: replicate 2 0/16 successful, replicate 1 9/16; c=1 subject/comparator medians are in the receipt and are a recorded FAIL-class fact, never a baseline |
| 11 | 2026-09-05T00:23:15Z (`parity_attempt.started_utc` in the receipt, copied from `apr-cpu-c1.config.json` `server.started_utc`) | `mini` | cpu · Apple M4 (aarch64-apple), default-feature install; Homebrew bash 5.3 + util-linux flock | qwen2.5-coder-1.5b-instruct · q4_k_m | dogfood parity protocol, ladder c=1,4,8,16 — NOT W1/W2 | `8e1e9ad40` (published 0.65.2) | true | 5 | 30 successful subject requests (`parity_attempt.bands[].subject.successful`) | [`dogfood/0.65.2/mini.json`](../dogfood/0.65.2/mini.json) `parity_attempt` (block REFUSED: zero-throughput band); kept work dir on the host `/var/folders/kq/w2p_fc85343bjq5gb5qg9md80000gn/T/tmp.7cm9Og5XrU` | RECORDED | {c1: MEASURED (c=1, no witness required) · c4: INVALID-CORRECTNESS(#2753/#2776) · c8: INVALID-CORRECTNESS(#2753/#2776) · c16: INVALID-CORRECTNESS(#2753/#2776)} | a parity block at all (refused, F12) · PP-26 witness on every c>1 band (none run) · PP-21 signature · PP-25 client sha256; c=8 and c=16: 0 successful requests in every replicate; c=1 subject/comparator medians are in the receipt and are a recorded FAIL-class fact, never a baseline |

### Row 0b — the withdrawn band table, moved verbatim from `scripts/llama_pin.toml:218-237`

Recorded here, not in the specification: it is the measurement, and the pin is a declaration the
gate reads rather than a place for a result. The two metrics tell opposite stories, which is the
whole of §2.2 of the master, and neither series may be quoted outside this row.

```toml
# BANDS, not a point. This was `http_concurrency = 1`, and that one line is why
# every parity number this project has published measures the WORST band and
# calls it the answer.
#
# Measured 2026-08-25, lambda RTX 4090, comparator pinned 39173bcac:
#
#   band   llama agg   apr agg    agg ratio    llama dec   apr dec   dec ratio
#   c=1        168.9      90.2       0.534x        171.5     100.7      0.587x
#   c=4        484.7     111.9       0.231x        123.3     113.8      0.923x
#   c=8        650.5     109.6       0.169x         83.0     112.2      1.352x
#   c=16      1120.8     108.4       0.097x         71.2     110.6      1.554x
#
# The two metrics tell OPPOSITE stories and only one of them is about adoption.
# apr's aggregate is FLAT at ~110 tok/s at every concurrency: it does not batch,
# it serialises. Per-user decode looks excellent at c=16 (1.554x) purely because
# each request gets the whole GPU in turn while llama.cpp shares it sixteen
# ways. A gate that reports only per-user decode would call 0.097x aggregate a
# PASS — which is the shape of every cannot-fail gate this epic exists to kill.
#
# So both are gated, on every band.
```

**Note on row 4's headline number.** 6.2034 tok/s at c=1 is the **mean over the three replicates'**
`bands[0].aggregate_tok_per_sec` (`receipt.r1.json` 6.198938, `receipt.r2.json` 6.203144,
`receipt.r3.json` 6.208150 under `evidence/perf-gate-001-w1-gx10/`) and it is an **end-to-end aggregate**, not
decode: `decode_tok_per_sec` is declared UNPRODUCED in `unproduced_fields` because `--stream` was
off. The superseded row 2 below called that figure "decode"; it is not.

**No row here is a parity measurement.** Under PP-9 neither host may be re-run at `745fa8588` to
obtain a different answer. The next run on either host requires a commit containing §12 rows 0b,
0c, 1, 6 and 7 of the master, and it starts a new row.

## §13 Superseded documents

The master's §13 points here. Sixteen rows: **fifteen superseded documents and one retained**
(`APR-PERF-GATE-001-status-review.md`, kept live in `docs/specifications/` as the effectiveness
review that justifies the supersession). The superseded spec's own prose counted `2 + 3 + 9 + 1`
against a table of `3 + 3 + 8 + 1`; the table is the record, and the count sentence is corrected
to match it rather than the other way round.

| # | repo | document | lines | repo_state | local_path |
|---|---|---|---|---|---|
| 1 | aprender | `APR-PERF-GATE-001-v2.2.md` | 1265 | archived-by-#2845 | `docs/archive/perf-2026-09-01/APR-PERF-GATE-001-v2.2.md` |
| 2 | aprender | `APR-PERF-GATE-001-RESTART.md` | 77 | archived-by-#2845 | `docs/archive/perf-2026-09-01/APR-PERF-GATE-001-RESTART.md` |
| 3 | aprender | `APR-PERF-GATE-001-status-review.md` | 251 | **RETAINED, not superseded** | `docs/specifications/APR-PERF-GATE-001-status-review.md` |
| 4 | qwen-coder-deploy | `gpu-performance-spec.md` | 5130 | archived-at-4fadc7c | not vendored |
| 5 | qwen-coder-deploy | `perf-parity-spec.md` | 748 | archived-at-4fadc7c | not vendored — `scripts/perf-matrix.yaml` still names this path in an `inherited_from` field that resolves nowhere (master §12 row 8) |
| 6 | qwen-coder-deploy | `benchmarking-v2.md` | 527 | archived-at-4fadc7c | not vendored |
| 7 | realizar | `benchmarking-with-common-models-common-serving-spec.md` | 1628 | repo-archived-on-GitHub | not vendored |
| 8 | realizar | `benchmark-model-runners-spec.md` | 1021 | repo-archived-on-GitHub | not vendored |
| 9 | realizar | `llama-cpp-style-performance-spec.md` | 803 | repo-archived-on-GitHub | not vendored |
| 10 | realizar | `qwen-performance-improve.md` | 715 | repo-archived-on-GitHub | not vendored |
| 11 | realizar | `deterministic-reproducible-cargo-bench.md` | 541 | repo-archived-on-GitHub | not vendored |
| 12 | realizar | `decoder-throughput-specification-llama-mistral-phi-qwen.md` | 445 | repo-archived-on-GitHub | not vendored |
| 13 | realizar | `qwen-showcase-throughput-improve.md` | 344 | repo-archived-on-GitHub | not vendored |
| 14 | realizar | `performance-parity-ollama-llamacpp-gpu-inference-llms.md` | 54 | repo-archived-on-GitHub (already SUPERSEDED there) | not vendored |
| 15 | trueno | `CUDA-parity-spec.md` | 1497 | repo-archived-on-GitHub | not vendored |
| 16 | aprender | `performance-parity-llama.cpp.md` | 1194 | **archived by the PP-LLAMA-001 pull request** (introduced by PR #2845, which never merged) | `docs/archive/perf-2026-09-02/performance-parity-llama.cpp.md` |

`realizar` and `trueno` are read-only on GitHub `[V]` (`gh repo view --json isArchived`) and were
already superseded by the APR-MONO consolidation into `crates/aprender-serve` and
`crates/aprender-compute`. Ten of the sixteen were therefore never live specifications in a
writable repository; the genuinely live overlap was five documents in two repositories, and after
this pull request it is one.

## Superseded rows (schema v2)

Kept verbatim, because the file is append-only and rows 1–2 were entered in the eleven-column
schema. **Both status cells are wrong** and are corrected by rows 3 and 4 above: row 1 names a
build defect (`53062e7f3`, row 0b) that is not this run's — `745fa8588` logged
`CONTINUOUS BATCHING: max_batch=11` (`evidence/perf-gate-001-w1-lambda/server-startup.txt:27-28`)
— and neither run carries a ratio to withdraw, since `comparator_status` is `UNMEASURED` on
every band. Row 2 uses `SUSPECT_DISPATCH`, which §7.4 abolishes, and labels an end-to-end
aggregate as decode.

| # | date | host | class · accelerator | model · quant | workload | bands | N | commit | receipts | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-01 | `lambda` | cuda · RTX 4090 | qwen2.5-coder-7b-instruct · q4_k_m | W1 | c=1,4,8,16 | 3 | `745fa8588` | [`perf-gate-001-w1-lambda/`](../perf-gate-001-w1-lambda/) | **SPENT — subject lane invalid.** Ratios withdrawn by §2.1: the subject binary was built with continuous batching compiled out. Retained as the counter-measurement of record; the comparator lane and the noise floor in §12.1a survive. |
| 2 | 2026-09-01 | `gx10` | cuda · GB10 (aarch64) | qwen2.5-coder-7b-instruct · q4_k_m | W1 | c=1,4,8,16 | 3 | `745fa8588` | [`perf-gate-001-w1-gx10/`](../perf-gate-001-w1-gx10/) | **SPENT — subject lane invalid** (same build defect). Additionally flags `SUSPECT_DISPATCH` under PP-23: 6.203 tok/s decode is 10.6% of this device's roofline (#2846). c=8 carries a 21.17% MDE traced to a device-wide stall (#2833) — the failure PP-19 exists to prevent. |
