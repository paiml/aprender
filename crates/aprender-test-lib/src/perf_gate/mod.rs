//! APR-PERF-GATE-001 v2.2 §4.4 — the measurement protocol, as code.
//!
//! # What this is, and what it is not
//!
//! This is **not a benchmarking harness.** The harness is
//! [`crate::llm::loadtest::LoadTest`], reached from `apr test llm bench`, and it
//! already drives `c` concurrent workers over external HTTP against both `apr
//! serve` and `llama-server`. A second harness would be a
//! `check_no_competing_harnesses.sh` violation and a PERF-009 regression.
//!
//! This module is the **protocol** the harness was missing: the termination
//! rule, the metric definitions, the confidence interval, and the sample
//! retention that turn a load test into a §4.4-conformant *measurement*. Every
//! decision lives here as a pure function so it can be tested in microseconds
//! and, crucially, so it is compiled under **default features**. `aprender-test-lib`'s
//! `llm` feature is not default and CI runs
//! `cargo nextest run --profile ci --workspace --lib` with no `--features`, so a
//! test placed inside `llm/` never executes in CI. The protocol had to sit
//! outside that gate to be gated at all.
//!
//! # §4.4 requirement → code, and what is NOT implemented
//!
//! | Clause | Requirement | Status | Where |
//! |---|---|---|---|
//! | §4.4.1 | closed-loop, `c` workers, issue → wait → immediately reissue | pre-existing | `llm/loadtest.rs` `run_phase_max` |
//! | §4.4.1 | external HTTP on the same host, never in-process | pre-existing | `llm/client.rs` (reqwest to a URL) |
//! | §4.4.1 | record `client_model: closed_loop` | **added** | [`protocol::ClientModel`] |
//! | §4.4.2 | warmup `2 × c` requests, discarded | **added** | [`protocol::warmup_requests`], [`window::WindowController::with_bounds`] |
//! | §4.4.2 | 5 s quiesce before the first sampled request | **added** | [`protocol::QUIESCE`], [`protocol::BandConfig::quiesce`] |
//! | §4.4.2 | minimum sampled requests `max(30, 8 × c)` | **added** | [`protocol::min_sampled_requests`] |
//! | §4.4.2 | minimum 60 s wall-clock per band | **added** | [`protocol::MIN_WALL_CLOCK`] |
//! | §4.4.2 | termination = whichever bound is satisfied **last** | **added** | [`window::WindowController::try_admit`] |
//! | §4.4.2 | `N = 3` full band replicates | **added** (constant + assertion) | [`protocol::REPLICATES`] |
//! | §4.4.3 | `agg_tok_s` **wall-clock**, not the mean of per-request rates | **added** | [`metrics::agg_tok_s`], with [`metrics::mean_of_rates`] beside it purely so a test asserts they differ |
//! | §4.4.3 | `decode_tok_s` = median of per-request `(tok-1)/(last-first)` | **added** | [`metrics::RequestSample::decode_tok_s`] |
//! | §4.4.3 | `ttft_ms` p50/p95 | **added** (p95 was absent) | [`metrics::BandMetrics`] |
//! | §4.4.3 | `itl_ms` **pooled** gaps, p50/p95 | **added** | [`metrics::RequestSample::itl_gaps_ms`] |
//! | §4.4.3 | counts `completed`/`requested`/`timeouts`/`truncated` | **added** | [`protocol::Outcome`], [`metrics::BandMetrics`] |
//! | §4.4.3 | 120 s hard per-request timeout | pre-existing | `llm/client.rs:213` |
//! | §4.4.4 | bootstrap percentile, 10 000 resamples, seed 2026, **whole requests** | **added** | [`bootstrap::bootstrap_ci`] |
//! | §4.4.5 | raw samples retained as gzipped JSONL | **added** | [`samples::write_samples_gz`] |
//! | §4.4.5 | `receipt_size_budget_bytes` | **deliberately unset** | [`samples::SamplesFile::exceeds_budget`] takes the budget as an argument; the spec forbids the literal until a full receipt is measured |
//! | §4.4.6 | `tokenization` block, `method` with no default | **added** | [`protocol::Tokenization`] |
//! | §4.4.7 | no new request at or after `T`; drain to completion | pre-existing + **formalised** | [`window::WindowController`] |
//! | §4.4.7 | `drain_ms` recorded | **added** | [`window::WindowReport::drain_ms`] |
//! | §4.4.7 | `drain_ms > 0.5 × window` annotated `SUSPECT` | **added** | [`window::WindowReport::suspect`] |
//!
//! ## NOT implemented by this module — stated plainly
//!
//! - **§4.4.8 comparator harness.** One client driving both `apr serve` and
//!   `llama-server` already exists (`scripts/parity_host_receipt.sh` runs the
//!   same `apr test llm bench` against both). Nothing here changes or verifies
//!   that, and the `tokenization`-block-must-match-across-lanes rule is a
//!   *receipt validator's* job, not this module's.
//! - **§4.4.9 scheduler observability.** `max_in_flight`, `admission_rejected`,
//!   `preempted_*`, `kv_blocks_*`, `gpu_layers_*`, `backend_loaded[]`,
//!   `autofit_applied[]` are all **server**-reported. A client cannot synthesise
//!   them, and a client-side guess would be worse than a null.
//!   [`window::WindowReport::client_peak_in_flight`] is the client's own
//!   observation and is named so it can never be mistaken for the server's
//!   `max_in_flight`.
//! - ~~**Receipt emission.**~~ Shipped by PERF-025 in [`receipt`].
//!   `scripts/lib/bench_receipt.py` remains the single schema authority; this
//!   module writes the file that authority validates, and does not redefine it.
//! - **W1/W2 workload construction** (§4.3), prompt corpora, and the
//!   `prompt_tokens = 512 ± 8` assertion. Note in passing that
//!   `crates/aprender-serve/benchmarks/qwen-coder/` contains `prompts-w2.jsonl`
//!   but **no `prompts-w1.jsonl`**, which §4.3.1 names as W1's corpus.
//! - **Running any band.** No baseline is measured or committed by this ticket.
//!   Every cell in `scripts/perf-matrix.yaml` stays `UNMEASURED`.

pub mod bootstrap;
pub mod metrics;
pub mod protocol;
pub mod receipt;
pub mod samples;
pub mod window;

pub use bootstrap::{bootstrap_agg_tok_s_ci, bootstrap_ci, BootstrapCi, SplitMix64, Statistic};
pub use metrics::{agg_tok_s, aggregate_terms, percentile, BandMetrics, RequestSample};
pub use protocol::{
    min_sampled_requests, warmup_requests, BandConfig, ClientModel, Outcome, Tokenization,
    TokenizationMethod, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, MIN_WALL_CLOCK, QUIESCE, REPLICATES,
    REQUEST_TIMEOUT,
};
pub use receipt::{
    assemble, build_band, sha256_file, write_receipt, BandReceipt, BootstrapBlock, CiBlock,
    ProtocolBlock, Provenance, Receipt, ReceiptMeta, Replicate, ReplicateReceipt, WrittenReceipt,
    COMPARATOR_UNMEASURED, COMPUTE_CLASSES, SPEC,
};
pub use samples::{read_samples_gz, write_samples_gz, SamplesFile};
pub use window::{WindowController, WindowReport};

#[cfg(test)]
mod conformance_tests {
    use super::*;

    /// §4.4.2 — `N = 3` full band runs per cell. The constant exists so a
    /// harness can assert against it rather than defaulting to 1 and calling a
    /// single run a measurement.
    #[test]
    fn replicates_is_three() {
        assert_eq!(REPLICATES, 3);
    }

    /// The four bands of §4.5, and the sample floor each one owes.
    #[test]
    fn every_declared_band_has_a_conformant_config() {
        for (c, want_samples) in [(1_usize, 30_usize), (4, 32), (8, 64), (16, 128)] {
            let cfg = BandConfig::conformant(c);
            assert!(
                cfg.is_conformant(),
                "c={c}: {:?}",
                cfg.conformance_violations()
            );
            assert_eq!(cfg.min_samples, want_samples, "c={c}");
            assert_eq!(cfg.warmup_requests, 2 * c, "c={c}");
            assert_eq!(cfg.client_model, ClientModel::ClosedLoop);
        }
    }

    /// End to end over the pure protocol: admit under the real termination rule,
    /// build samples, compute §4.4.3 metrics, and bound them with the §4.4.4 CI.
    #[test]
    fn protocol_composes_end_to_end() {
        let cfg = BandConfig::conformant(4);
        let mut w = WindowController::new(&cfg);
        let mut samples = Vec::new();
        // Four workers, each request taking 1 s, stepping the clock by 0.25 s.
        let mut now = 0.0_f64;
        while let Some(index) = w.try_admit(now) {
            let start = now;
            let end = now + 1.0;
            let drained = w.complete(end);
            samples.push(RequestSample {
                index,
                worker: index % 4,
                start_s: start,
                end_s: end,
                token_times_s: (1..=8).map(|k| start + f64::from(k) * 0.125).collect(),
                generated_tokens: 8,
                prompt_tokens: 512,
                outcome: Outcome::Completed,
                in_flight_at_start: 4,
                drained,
            });
            now += 0.25;
        }
        let report = w.report();
        // 60 s at one admission per 0.25 s is 240 requests, well past max(30, 32).
        assert!(report.requested >= cfg.min_samples, "{report:?}");
        assert!(report.window_ms >= 60_000.0, "{report:?}");

        let m = BandMetrics::from_samples(cfg.concurrency, &samples);
        assert_eq!(m.requested, samples.len());
        assert_eq!(m.completed, samples.len());
        assert_eq!(m.timeouts, 0);
        assert!(m.agg_tok_s > 0.0);

        let ci = bootstrap_agg_tok_s_ci(&samples, 0.95).expect("n >= 2");
        assert_eq!(ci.seed, BOOTSTRAP_SEED);
        assert_eq!(ci.resamples, BOOTSTRAP_RESAMPLES);
        assert!(ci.lower <= ci.point && ci.point <= ci.upper, "{ci:?}");
    }
}
