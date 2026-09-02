# PERF-024 — §4.4 measurement protocol: what was missing, and where it now lives

Ticket: PERF-024, epic APR-PERF-GATE-001 (paiml/aprender#2706).
Spec: `docs/archive/perf-2026-09-01/APR-PERF-GATE-001-v2.2.md` §4.4.
Audited against `origin/main` at `de8fbc407`.

## The thing that already existed

`crates/aprender-test-lib/src/llm/loadtest.rs` — reached from `apr test llm bench`
(`crates/apr-cli/src/commands/test_llm.rs` → `apr_test::llm::benchmark::Benchmark`
→ `LoadTest::run`) — is the project's comparator client. It is genuinely
concurrent, genuinely external HTTP, and its aggregate is genuinely wall-clock:

| Claim | Verdict | Evidence on `origin/main` |
|---|---|---|
| No concurrency | **FALSE** | `loadtest.rs:568` `for worker_id in 0..self.config.concurrency`, each worker `while Instant::now() < deadline { send_one_request(..).await }` |
| `agg` is the mean of per-request rates | **FALSE** | `loadtest.rs:519-521` `measure_start.elapsed()`; `:834` `tokens_per_sec = total_tokens / elapsed_secs` — Σ tokens ÷ wall-clock |
| No TTFT/ITL | **FALSE** | `ttft_p50/p90/p95/p99_ms`, `itl_p50_ms`, `tpot_p*_ms` on `LoadTestResult` |

`crates/aprender-serve/src/http_client/benchmark_runner.rs` is **not** that client.
It is reachable only from `http_client/tests/*` and `benches/external_matrix.rs`,
and it is a byte-identical duplicate of `crates/aprender-serve/src/http_client/preflight.rs`
(`diff -q` rc=0, 370 lines each) of which only `benchmark_runner.rs` is `include!`d.
See "Separate defect" below.

## What was genuinely missing, and where it is now

Everything new is under `crates/aprender-test-lib/src/perf_gate/`, deliberately
**outside** the `llm` feature so its tests compile and run under the crate's
default features. `llm/band.rs` drives it through the *existing* request path.

| Clause | Requirement | Status | Code |
|---|---|---|---|
| §4.4.1 | closed-loop, `c` workers, issue → wait → reissue | pre-existing | `llm/loadtest.rs:568`; `llm/band.rs::worker_loop` |
| §4.4.1 | external HTTP, same host, never in-process | pre-existing | `llm/client.rs`, driven by `llm/band.rs::issue` |
| §4.4.1 | record `client_model: closed_loop` | **added** | `perf_gate/protocol.rs::ClientModel` |
| §4.4.2 | warmup `2 × c` requests, discarded | **added** | `perf_gate/protocol.rs::warmup_requests`; `llm/band.rs::warmup` |
| §4.4.2 | 5 s quiesce before the first sampled request | **added** | `perf_gate/protocol.rs::QUIESCE`; `band.rs` `sleep(band.quiesce)` |
| §4.4.2 | minimum sampled requests `max(30, 8 × c)` | **added** | `perf_gate/protocol.rs::min_sampled_requests` |
| §4.4.2 | minimum 60 s wall-clock per band | **added** | `perf_gate/protocol.rs::MIN_WALL_CLOCK` |
| §4.4.2 | termination = whichever bound is satisfied **last** | **added** | `perf_gate/window.rs::WindowController::try_admit` |
| §4.4.2 | `N = 3` replicates per cell | **added** | `perf_gate/protocol.rs::REPLICATES`; `llm/band.rs::run_cell` |
| §4.4.3 | `agg_tok_s` **wall-clock**, not the mean of rates | pre-existing, **restated exactly** | `perf_gate/metrics.rs::agg_tok_s` |
| §4.4.3 | `decode_tok_s` = median of per-request `(tok-1)/(last−first)` | **added** | `perf_gate/metrics.rs::RequestSample::decode_tok_s` |
| §4.4.3 | `ttft_ms` p50 **and p95** | **added** (p95 absent before) | `perf_gate/metrics.rs::BandMetrics` |
| §4.4.3 | `itl_ms` **pooled gaps**, p50/p95 | **added** | `perf_gate/metrics.rs::RequestSample::itl_gaps_ms` |
| §4.4.3 | `completed`/`requested`/`timeouts`/`truncated` | **added** | `perf_gate/protocol.rs::Outcome`, `metrics.rs::BandMetrics` |
| §4.4.3 | 120 s hard per-request timeout | pre-existing, **now classified** | `llm/client.rs:213`; `band.rs::worker_loop` `tokio::time::timeout` → `Outcome::Timeout` |
| §4.4.4 | bootstrap percentile, 10 000 resamples, seed 2026, **whole requests** | **added** | `perf_gate/bootstrap.rs::bootstrap_ci` |
| §4.4.5 | raw samples retained as gzipped JSONL | **added** | `perf_gate/samples.rs::write_samples_gz` |
| §4.4.6 | `tokenization` block, `method` with no default | **added** | `perf_gate/protocol.rs::Tokenization` |
| §4.4.7 | no new request at or after `T`; drain to completion | pre-existing, **formalised** | `perf_gate/window.rs::WindowController` |
| §4.4.7 | `drain_ms` recorded | **added** | `perf_gate/window.rs::WindowReport::drain_ms` |
| §4.4.7 | `drain_ms > 0.5 × window` → `SUSPECT` | **added** | `perf_gate/window.rs::WindowReport::suspect` |

### Pre-existing but not §4.4-exact — differences worth knowing

- `LoadTestResult::decode_tok_per_sec` is `1000 / itl_p50_ms`: the reciprocal of a
  median ITL, **not** the median of per-request decode rates. Different statistic.
  `BandMetrics::decode_tok_s` implements the §4.4.3 one.
- `LoadTestResult::itl_p50_ms` pools **one value per request** (that request's mean
  ITL). §4.4.3 pools **every gap**. `BandMetrics` implements the §4.4.3 one.
- `LoadTestResult::truncated_pct` counts `finish_reason == "length"`. §4.4.7's
  `truncated` means "abandoned at the drain deadline". Under W1 — `max_tokens=128`,
  EOS ignored — *every* request has `finish_reason == "length"`, so conflating the
  two would exclude the whole workload from `agg_tok_s`'s numerator and report zero
  throughput for a healthy server. `perf_gate::protocol::Outcome` documents the split.
- `LoadTest::run` is **unchanged**, and so is every byte of `llm/loadtest.rs`. It
  answers a different question and a lot of tooling reads its output; changing its
  termination rule underneath those readers would silently move every number they
  have recorded. `llm/band.rs` therefore drives `LlmClient` directly — the same
  HTTP client `send_one_request` drives, so §4.4.8's one-client rule still holds —
  and maps its responses into `RequestSample` rather than into `RequestRecord`,
  which carries no absolute timing and cannot distinguish a timeout from a
  transport error.
- **Non-streaming requests yield no `ttft_ms`, `itl_ms`, or `decode_tok_s`.** The
  offsets are left empty rather than synthesised: inventing evenly-spaced token
  times would make an unmeasured quantity look measured. A conformant band must
  run with `stream = true`.

## NOT implemented — stated plainly

- **§4.4.8 comparator harness.** One client already drives both servers
  (`scripts/parity_host_receipt.sh`). Nothing here verifies that, and the
  "tokenization blocks must match across lanes or the ratio is refused" rule is a
  receipt-validator job.
- **§4.4.9 scheduler observability.** `max_in_flight`, `admission_rejected`,
  `preempted_*`, `kv_blocks_*`, `gpu_layers_*`, `backend_loaded[]`,
  `autofit_applied[]` are **server**-reported. A client cannot synthesise them and a
  client-side guess is worse than a null. `WindowReport::client_peak_in_flight` is
  the client's own observation and is named so it cannot be mistaken for the
  server's `max_in_flight`.
- **Receipt emission.** Nothing here writes `receipt.json`.
  `scripts/lib/bench_receipt.py` is the single schema authority; serialising a
  `BandRun` into it is a separate ticket.
- **`receipt_size_budget_bytes`.** §4.4.5 forbids the literal until a full receipt
  is measured. `SamplesFile::exceeds_budget` takes the budget as an argument so the
  check arms the day the number exists, with no placeholder committed today.
- **CLI wiring.** `apr test llm bench` still calls `LoadTest::run`. Nothing invokes
  `run_band`/`run_cell` outside tests yet.
- **W1/W2 workload construction** (§4.3) and the `prompt_tokens = 512 ± 8` check.
  Note: `crates/aprender-serve/benchmarks/qwen-coder/` contains `prompts-w2.jsonl`
  but **no `prompts-w1.jsonl`**, which §4.3.1 names as W1's corpus.
- **Any measured baseline.** Every cell in `scripts/perf-matrix.yaml` is untouched
  and stays `UNMEASURED`.

## Verification

Every new decision was mutation-verified, since a gate nobody proved can fail is
this epic's recurring defect:

| Mutation | Tests turned RED |
|---|---|
| `agg_tok_s` returns `mean_of_rates` | 2 |
| termination `AND` → `OR` (either bound closes the window) | 5 |
| bootstrap seed → wall clock | 3 |
| `for id in 0..band.concurrency` → `0..1` | 4, including the real-HTTP proof |

Concurrency is proved from the **server** side: a loopback probe counts its own
concurrent connections. At `c = 8` it reports peak 8; at `c = 1`, peak 1. The same
16 requests take 770 ms at `c = 1` and 186 ms at `c = 8` (4.14x).

TTFT and ITL are proved end to end against an SSE probe emitting 6 tokens 25 ms
apart: measured `ttft_p50 = 26.7 ms`, `itl_p50 = 26.2 ms`, `decode = 38.2 tok/s`,
with five pooled gaps per request rather than one summary per request.

## Separate defect worth its own ticket

`crates/aprender-serve/src/http_client/preflight.rs` and
`crates/aprender-serve/src/http_client/benchmark_runner.rs` are byte-identical
370-line files; only the latter is `include!`d, so `preflight.rs` is dead. It reads
like the production benchmark client and is not one. That duplicate is what caused
this ticket to be filed on a false premise.
