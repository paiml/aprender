//! PP-LLAMA-001 v3.0 — the measurement protocol and the receipt, as code.
//!
//! (The clause numbers below that read `§4.4.x` are `APR-PERF-GATE-001` v2.2's,
//! kept where the v2.2 text is still the normative one; the v3 rules are cited
//! by their `PP-nn` id.)
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
//! | §4.3 | `N = 5` **interleaved** full band replicates | **added** (constant + assertion) | [`protocol::REPLICATES`], [`replicate::log_ratio_lcb`] |
//! | §4.4.3 | `agg_tok_s` **wall-clock**, not the mean of per-request rates | **added** | [`metrics::agg_tok_s`], with [`metrics::mean_of_rates`] beside it purely so a test asserts they differ |
//! | §4.4.3 | `decode_tok_s` = median of per-request `(tok-1)/(last-first)` | **added** | [`metrics::RequestSample::decode_tok_s`] |
//! | §4.4.3 | `ttft_ms` p50/p95 | **added** (p95 was absent) | [`metrics::BandMetrics`] |
//! | §4.4.3 | `itl_ms` **pooled** gaps, p50/p95 | **added** | [`metrics::RequestSample::itl_gaps_ms`] |
//! | §4.4.3 | counts `completed`/`requested`/`timeouts`/`truncated` | **added** | [`drain::Outcome`], [`metrics::BandMetrics`] |
//! | §4.4.3 | 120 s hard per-request timeout | pre-existing | `llm/client.rs:213` |
//! | §4.4.4 | bootstrap percentile, 10 000 resamples, seed 2026, **whole requests** | **added** | [`bootstrap::bootstrap_ci`] |
//! | §4.4.5 | raw samples retained as gzipped JSONL | **added** | [`samples::write_samples_gz`] |
//! | §4.4.5 | `receipt_size_budget_bytes` | **deliberately unset** | [`samples::SamplesFile::exceeds_budget`] takes the budget as an argument; the spec forbids the literal until a full receipt is measured |
//! | §4.4.6 | `tokenization` block, `method` with no default | **added** | [`receipt::TokenizationBlock`] |
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
//! - **Receipt emission.** Nothing here writes `receipt.json`.
//!   `scripts/lib/bench_receipt.py` is the single schema authority and the
//!   serialiser that feeds it is a separate ticket.
//! - **W1/W2 workload construction** (§4.3) and the prompt corpora. Both
//!   corpora now exist under `crates/aprender-serve/benchmarks/qwen-coder/`
//!   (`prompts-w1.jsonl` landed with PERF-039; this note used to say it was
//!   missing). The `prompt_tokens = 512 ± 8` assertion is no longer absent
//!   either — PERF-056 put it in `llm::prompts::assert_prompt_tokens_in_band`,
//!   called from the band harness against the counts the SERVER reports,
//!   because that is the only place the model's own tokenizer is observable.
//! - **Running any band.** No baseline is measured or committed by this ticket.
//!   Every cell in `scripts/perf-matrix.yaml` stays `UNMEASURED`.
//!
//! # The §4.4.6 / §4.4.7 half — the receipt producers (PERF-026)
//!
//! ## The defect this closes
//!
//! `scripts/perf_gate.sh:42` fails any receipt whose `drain_ms` is absent:
//!
//! ```text
//! if r.get("drain_ms") is None:
//!     bad.append("drain_ms absent")
//! ```
//!
//! On `62d23d8d1`, `grep -rn "drain_ms" --include="*.rs" crates` returned
//! **rc=1, zero lines**. The only `drain_ms` anywhere in the repo were the
//! spec, this line of the gate, a matrix comment, and the gate's own hand-typed
//! selftest fixture `"drain_ms":12`. So Arm C was **green on a string literal
//! and red on every measurement that could ever be taken** — a gate that cannot
//! pass, which is the exact mirror of the cannot-fail gates §5 catalogues.
//!
//! The same held for §4.4.6's `tokenization` block, which the spec marks
//! REQUIRED with no default.
//!
//! ## What is here
//!
//! - [`drain`] — the §4.4.7 producer. `drain_ms`, the four request counters,
//!   and the `SUSPECT` annotation, all **derived from per-request terminal
//!   records**. There is no constructor that accepts a `drain_ms` scalar.
//! - [`receipt`] — the emitter. Turns those records plus a §4.4.6 tokenization
//!   declaration and §4.2 provenance into the JSON `scripts/perf_gate.sh` and
//!   `scripts/lib/bench_receipt.py` actually read.
//!
//! ## The definition that had to be right first (§4.4.7)
//!
//! > `drain_ms` is the length of the **drain phase**: the last settlement of any
//! > pre-`T` request, minus `T`, clamped at zero.
//!
//! and, load-bearing:
//!
//! > `truncated` counts requests **abandoned at the drain deadline**, never
//! > `finish_reason == "length"`.
//!
//! W1 generates with `max_tokens = 128` and **EOS ignored** (§4.3.1), so every
//! healthy W1 request carries `finish_reason == "length"`. §4.4.3 puts
//! `agg_tok_s`'s numerator over "completed, non-truncated" requests, so the
//! finish-reason reading empties the numerator and reports `0 tok/s` for a
//! working server. [`drain::Outcome::AbandonedAtDrain`] is the §4.4.7 sense and
//! is the only thing that increments `truncated`;
//! `drain::tests::max_tokens_truncation_is_not_drain_truncation` is the guard.
//!
//! ## What a client cannot produce, said rather than synthesised
//!
//! §4.4.9's scheduler block is **server**-reported: I-16 requires
//! `max_in_flight` come from the server rather than be inferred by the harness,
//! and I-2 requires `gpu_layers_resolved` be read from the loader. Arm D's `kv`
//! block is the same. Neither is emitted from client-side data. They are named
//! in the receipt's `unproduced_fields` with the reason, because a default that
//! invents provenance is worse than a missing field — it is indistinguishable
//! from a real answer.
//!
//! # Overlaps between the two halves, resolved here rather than left to drift
//!
//! `main`'s `perf_gate/mod.rs` named three and they are discharged in this
//! merge commit, not deferred:
//!
//! - **`Outcome`.** `protocol::Outcome` and [`drain::Outcome`] were the same
//!   four mutually-exclusive counters under two spellings. [`drain::Outcome`]
//!   survives; `protocol` re-exports it. Its `AbandonedAtDrain` is the §4.4.7
//!   sense of `truncated` said in a name that cannot be misread as
//!   `finish_reason == "length"` — which is the exact confusion the other
//!   spelling's own doc comment warned about while spelling the variant
//!   `Truncated`.
//! - **The §4.4.6 `tokenization` block.** `protocol::Tokenization` was a struct
//!   with an `Option<String>` digest and a `validate()`;
//!   [`receipt::TokenizationBlock`] is an enum in which "`client_tokenizer`
//!   with no digest" is *unrepresentable* rather than merely rejected, and it
//!   is the one actually wired into the emitter the gate reads. The enum
//!   survives; `protocol::Tokenization` is gone, and its one behaviour the enum
//!   lacked — [`receipt::TokenizationBlock::require_counter`], the poka-yoke
//!   against a declared method the transport cannot honour — moved onto it.
//! - **`percentile` and `DRAIN_SUSPECT_FRACTION`.** Byte-identical in
//!   `drain.rs` and in `metrics.rs`/`protocol.rs` respectively. The `drain`
//!   definitions survive; the others re-export. `protocol::REQUEST_TIMEOUT`
//!   (a `Duration`) and [`drain::REQUEST_TIMEOUT_MS`] (an `f64`) are two
//!   spellings the type system cannot merge, so
//!   `conformance_tests::the_two_request_timeout_spellings_agree` pins them
//!   together instead.

pub mod ab;
pub mod bootstrap;
pub mod drain;
pub mod join;
pub mod metrics;
pub mod protocol;
pub mod receipt;
pub mod replicate;
pub mod samples;
pub mod window;
pub mod witness;

#[cfg(test)]
mod join_fixture;

pub use ab::{AbRecord, AbReplicate, Arm, ArmId, ConfigDiff, DeltaKind};
pub use bootstrap::{
    bootstrap_agg_tok_s_ci, bootstrap_ci, itl_p95_ms, median_decode_tok_s, paired_ratio_lcb,
    ttft_p50_ms, BootstrapCi, SplitMix64, Statistic,
};
pub use drain::{
    percentile, AdmissionCap, BandContext, BandInput, BandStatus, ComparatorStatus, DerivedBand,
    Lane, LaneConfig, MeasuredJoin, Outcome, RequestOutcome, SampleRow, StreamMode, StreamVerdict,
    StreamWitness, StreamWitnessSource, DRAIN_SUSPECT_FRACTION, REQUEST_TIMEOUT_MS, SCHEMA_VERSION,
};
pub use join::{BandRatios, JoinKey, Ratio, RatioBound, RatioMethod};
pub use metrics::{agg_tok_s, aggregate_terms, BandMetrics, RequestSample};
pub use protocol::{
    min_sampled_requests, warmup_requests, BandConfig, ClientModel, ProtocolParams, ProtocolSource,
    Sampler, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, COOLDOWN, INTERLEAVED, MIN_WALL_CLOCK, N_PREDICT,
    QUIESCE, REPLICATES, REQUEST_TIMEOUT,
};
pub use receipt::{
    sha256_file, ClientIdentity, ComparatorIdentity, ComputeClass, KvBlock, Ladder, ModelIdentity,
    Provenance, Receipt, ReceiptBand, ReceiptInput, Roofline, RunId, SlotsAdmitted,
    SubjectIdentity, TokenCountingMethod, TokenizationBlock, Workload,
    CLOCK_SOURCE_SYSTEM_REALTIME, SERVER_ONLY_FIELDS, SPEC_ID,
};
pub use replicate::{log_ratio_lcb, t_lower_one_sided_95, ArmOrder, ReplicatePair, MIN_REPLICATES};
pub use samples::{read_samples_gz, write_samples_gz, SamplesFile};
pub use window::{WindowController, WindowReport};
pub use witness::{BatchInvariance, BatchInvarianceWitness};

#[cfg(test)]
mod conformance_tests {
    use super::*;

    /// §4.3 — `N = 5` **interleaved** full band runs per cell.
    ///
    /// This test used to read `replicates_is_three`, pinning the v2.2 value.
    /// v3 reverses that rule outright: "`n = 3` sizes an effect and bounds no
    /// variance: no σ-dependent status changes at `n < 5`". The test is
    /// inverted rather than deleted, so the must-not-fire side (a constant that
    /// a harness can assert against, instead of defaulting to 1 and calling a
    /// single run a measurement) survives the change.
    #[test]
    fn replicates_is_at_least_five() {
        assert!(REPLICATES >= 5, "REPLICATES={REPLICATES}");
        assert_eq!(REPLICATES, MIN_REPLICATES, "one floor, two spellings");
        assert!(INTERLEAVED, "§4.3 makes interleaving mandatory");
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

#[cfg(test)]
mod gate_conformance_tests {
    // The `<selftest-name>__<sentence>` spelling is load-bearing: PP-29's
    // `scripts/spec_conformance.sh` joins the §6 invariant table to the test
    // list on the prefix before the double underscore, so renaming these to
    // single-underscore snake case would silently unjoin the rows they arm.
    #![allow(non_snake_case)]
    //! Does a receipt this producer emits satisfy `scripts/perf_gate.sh`?
    //!
    //! # Why these tests do not run the gate
    //!
    //! The obvious test shells out to `bash scripts/perf_gate.sh --receipt …`.
    //! It cannot live here. `workspace-test` — the job that runs
    //! `cargo nextest run --workspace --lib` — executes inside
    //! `localhost:5000/sovereign-ci:stable`, and **that image has no python3 at
    //! all** (probed 2026-08-28: `ls /usr/bin | grep -c ^python` → `0`,
    //! `command -v python3` → nothing). `perf_gate.sh` is a bash wrapper around
    //! five embedded python programs, so a subprocess test would have gone RED
    //! on every PR for a reason having nothing to do with the receipt. The gate
    //! runs today only in the `ci` job (`ci.yml:981`, `--selftest`), whose
    //! environment is a different one.
    //!
    //! So the coupling here is **textual, against the gate's own source**: the
    //! set of fields the gate reads is extracted from `perf_gate.sh` at test
    //! time and checked against what the producer emits. That is stronger than a
    //! hardcoded key list — when the gate starts requiring a new field, this
    //! goes red until the field is either produced or explicitly classified as
    //! one a client cannot produce.
    //!
    //! The end-to-end run against the real gate was performed by hand for
    //! PERF-026 and is recorded in the ticket's mutation table. Wiring a caller
    //! for the gate's real (non-selftest) mode is a separate ticket.

    use super::*;
    use serde_json::Value;
    use std::path::{Path, PathBuf};

    /// Top-level fields the gate reads that a **client** cannot produce, each
    /// with the reason it is declared rather than invented.
    const RECEIPT_SCOPE_UNPRODUCIBLE: &[(&str, &str)] = &[
        (
            "signature",
            "§4.9.1's signature is applied by `scripts/perf_receipt_sign.sh` on the PRODUCING \
             HOST, with a key that lives only there (forjar-deployed). A renderer able to sign \
             its own output would be a renderer able to forge one, which is precisely the \
             property the arm exists to deny — so `render()` emits the payload and never the \
             attestation over it. The receipt is legal unsigned at merge phase and fails \
             ArmC-sig at release phase, which is the §4.5 Arm C table exactly (PERF-007).",
        ),
        (
            "kv",
            "Arm D's memory block is server-reported (§4.4.9). Supplied only via \
             KvBlock::from_server_report; absent otherwise.",
        ),
        (
            "itl",
            "Arm E's itl.p95_w1_ms / p95_w2_ms is a ratio ACROSS two workloads. A single \
             host x workload receipt cannot hold it (PERF-020).",
        ),
        (
            "injector",
            "Arm E's out-of-band injector is a W2 construct (§4.3.2) this producer does not \
             model (PERF-020).",
        ),
    ];

    /// Per-band fields the gate reads that this producer refuses to derive.
    const BAND_SCOPE_UNPRODUCIBLE: &[(&str, &str)] = &[
        (
            "agg_ratio",
            "The v2.2 BARE SCALAR. A ratio is representable only inside `ratios`, only beside a \
             `baseline` that itself passes every receipt rule, and only when the two lanes share \
             a run_id (PP-3) and a join key (PP-22). `perf_gate.sh` still reads the scalar for \
             historical v2 receipts; a v3 receipt never writes one.",
        ),
        ("decode_ratio", "As agg_ratio (PP-3, PP-22, PP-25)."),
        (
            "comparator_requested",
            "The COMPARATOR lane's request counter as a v2.2 top-level scalar. At v3 the whole \
             comparator band travels in `baseline`, counters included, so a reader takes \
             `band.baseline.requested` instead. `scripts/perf-receipt-fields.yaml` marks the \
             legacy field `required: conditional` and `perf_gate.sh:88` reads it defensively.",
        ),
        (
            "comparator_completed",
            "As comparator_requested; at v3 it is `band.baseline.completed`.",
        ),
    ];

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .canonicalize()
            .expect("workspace root is two levels above this crate")
    }

    fn gate_source() -> String {
        let path = repo_root().join("scripts").join("perf_gate.sh");
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
    }

    /// Every `<receiver>.get("<field>")` the gate performs, as (receiver, field).
    fn gate_field_reads(source: &str) -> Vec<(String, String)> {
        let re =
            regex::Regex::new(r#"([A-Za-z_][A-Za-z0-9_]*(?:\[[a-z]\])?)\.get\("([a-z0-9_]+)""#)
                .expect("static regex");
        re.captures_iter(source)
            .map(|c| (c[1].to_string(), c[2].to_string()))
            .collect()
    }

    fn band(concurrency: u32, requests: Vec<RequestOutcome>) -> BandInput {
        BandInput::new(
            concurrency,
            60_000.0,
            requests,
            ComparatorStatus::unmeasured(
                "perf-gate",
                "no same-client comparator lane on this cell yet (PP-25)",
            ),
        )
    }

    /// PP-26's witness for a band whose batched arm agreed with `m = 1`.
    fn passing_witness(m: u32) -> BatchInvarianceWitness {
        let tokens: Vec<u32> = (0..128).collect();
        BatchInvarianceWitness::compare(&tokens, &tokens, 64)
            .formed_at(m, "scripts/perf041_batched_parity_probe.py")
    }

    /// A §5.1-shaped band: `max(30, 8c)` requests in a 60 s window, streamed
    /// live with server-reported prefill, with the final request deliberately
    /// straddling `T` so `drain_ms` is non-zero and demonstrably came from the
    /// data rather than from a constant.
    fn synthetic_band(concurrency: u32) -> BandInput {
        let window_ms = 60_000.0;
        let n = 30.max(8 * concurrency as usize);
        let step = (window_ms - 1_000.0) / n as f64;
        let mut requests: Vec<RequestOutcome> = (0..n)
            .map(|i| {
                let issued = i as f64 * step;
                // Varying durations: a constant distribution is F12.
                let dur = 400.0 + (i % 7) as f64 * 13.0;
                let ttft = 40.0 + (i % 5) as f64;
                RequestOutcome::completed(issued, issued + dur, 128)
                    .streamed(
                        ttft,
                        (0..128)
                            .map(|k| issued + ttft + f64::from(k) * (dur - ttft) / 128.0)
                            .collect(),
                    )
                    .server_prefill(512, 30.0 + (i % 3) as f64)
                    .in_flight(concurrency)
            })
            .collect();
        let last = requests.last_mut().expect("n >= 30");
        last.issued_ms = window_ms - 100.0;
        last.settled_ms = window_ms + 250.0;
        last.ttft_ms = Some(40.0);
        last.token_times_ms = (0..128)
            .map(|k| window_ms - 60.0 + k as f64 * 310.0 / 128.0)
            .collect();
        let mut band = band(concurrency, requests)
            .n_predict(128)
            .stream_mode(StreamMode::Live)
            .lane(LaneConfig {
                n_ctx_slot: Some(1024),
                kv_type: Some("f16".to_string()),
                fa: Some(true),
                n_batch: Some(2048),
            });
        if concurrency > 1 {
            band = band.witness(passing_witness(concurrency));
        }
        band
    }

    fn provenance() -> Provenance {
        Provenance {
            binary_path: "/opt/clean-room/bin/apr".to_string(),
            binary_sha256: "a".repeat(64),
            resolution: "scripts/apr_bin.sh".to_string(),
            compute_class: ComputeClass::Cuda,
            host: "lambda".to_string(),
            accelerator: "rtx-4090".to_string(),
            model: "qwen2.5-coder-7b-apache-q4k-v1".to_string(),
            quantization: "Q4_K_M".to_string(),
            feature_set: vec!["inference".to_string(), "cuda".to_string()],
            started_utc: "2026-09-02T10:11:12.345Z".to_string(),
            clock_source: CLOCK_SOURCE_SYSTEM_REALTIME.to_string(),
            subject: SubjectIdentity {
                path: "/opt/clean-room/bin/apr".to_string(),
                sha256: "5".repeat(64),
                commit: "62d23d8d1".to_string(),
                feature_set: vec!["inference".to_string(), "cuda".to_string()],
            },
            client: ClientIdentity {
                path: "/opt/clean-room/bin/apr".to_string(),
                sha256: "a".repeat(64),
                commit: "62d23d8d1".to_string(),
                pid: 4242,
            },
            comparator: None,
            server_config: None,
            model_file: None,
        }
    }

    fn ladder() -> Ladder {
        Ladder::derive(
            &[1, 4, 8, 16],
            SlotsAdmitted {
                apr: Some(16),
                llama: Some(16),
            },
        )
    }

    fn run_id() -> RunId {
        RunId::derive("2026-09-02T10:11:12.345Z", "lambda", &"a".repeat(64), 4242)
    }

    fn receipt_input(kv: Option<KvBlock>) -> ReceiptInput {
        ReceiptInput {
            kv,
            ..ReceiptInput::new(
                run_id(),
                provenance(),
                TokenizationBlock::ClientTokenizer {
                    tokenizer_sha256: "b".repeat(64),
                    counts_special_tokens: false,
                    counts_prompt_echo: false,
                },
                Workload::W1,
                ProtocolParams::spec_fallback(),
                "62d23d8d1",
                ladder(),
                vec![
                    synthetic_band(1),
                    synthetic_band(4),
                    synthetic_band(8),
                    synthetic_band(16),
                ],
            )
        }
    }

    fn rendered() -> Value {
        receipt_input(None).render().expect("valid receipt")
    }

    fn unproducible(table: &[(&str, &str)], field: &str) -> bool {
        table.iter().any(|(name, _)| *name == field)
    }

    /// THE RATCHET. The universe is derived from the gate's own source, so a new
    /// required field cannot be added to `perf_gate.sh` without this test going
    /// red until it is either produced or explicitly classified.
    #[test]
    fn every_field_the_gate_reads_is_produced_or_declared_unproducible() {
        let receipt = rendered();
        let bands = receipt["bands"].as_array().expect("bands array");
        let mut checked = 0_usize;
        for (receiver, field) in gate_field_reads(&gate_source()) {
            match receiver.as_str() {
                "r" => {
                    checked += 1;
                    assert!(
                        receipt.get(&field).is_some()
                            || unproducible(RECEIPT_SCOPE_UNPRODUCIBLE, &field),
                        "perf_gate.sh reads receipt field `{field}` and the producer neither \
                         emits it nor declares it unproducible"
                    );
                }
                "b" | "bands[c]" => {
                    checked += 1;
                    let missing: Vec<u64> = bands
                        .iter()
                        .filter(|b| b.get(&field).is_none())
                        .filter_map(|b| b["concurrency"].as_u64())
                        .collect();
                    assert!(
                        missing.is_empty() || unproducible(BAND_SCOPE_UNPRODUCIBLE, &field),
                        "perf_gate.sh reads band field `{field}`, absent at bands {missing:?} \
                         and not declared unproducible"
                    );
                }
                // `bl`/`m` read perf-matrix.yaml, and `kv`/`itl`/`inj` read
                // inside blocks already classified above.
                _ => {}
            }
        }
        assert!(
            checked >= 10,
            "only {checked} field reads found in perf_gate.sh — the extractor stopped matching, \
             so this test is no longer checking anything"
        );
    }

    /// And the ratchet discriminates: a field the gate reads that the producer
    /// does not emit and has not classified must fail the check above.
    #[test]
    fn the_field_ratchet_rejects_an_unclassified_new_requirement() {
        let receipt = rendered();
        assert!(
            receipt.get("max_in_flight").is_none()
                && !unproducible(RECEIPT_SCOPE_UNPRODUCIBLE, "max_in_flight"),
            "control: max_in_flight is neither emitted nor classified, so were the gate to start \
             reading it the test above would go red"
        );
    }

    /// Every rule Arm C applies, as a property of what the producer emits.
    /// `perf_gate.sh` remains the authority; this states the guarantee.
    #[test]
    fn a_produced_receipt_satisfies_every_arm_c_rule() {
        let r = rendered();
        assert_eq!(
            r["requested"], r["completed"],
            "Arm C: completed == requested"
        );
        assert_eq!(
            r["timeouts"], 0,
            "I-5: a timeout is fatal to this host's ratio"
        );
        assert!(
            r["tokenization"]["method"]
                .as_str()
                .is_some_and(|m| !m.is_empty()),
            "I-13: tokenization.method has no default"
        );
        assert!(
            r["drain_ms"].as_f64().is_some(),
            "§4.4.7: drain_ms is recorded — the field with zero producers on 62d23d8d1"
        );
        for b in r["bands"].as_array().expect("bands") {
            assert_ne!(
                b["tokens_total"], 0,
                "Arm C: a zero-token response is a failure, not a fast request"
            );
        }
    }

    /// Arm A's denominator and the four declared bands of §4.5.
    #[test]
    fn a_produced_receipt_carries_every_declared_band_including_the_arm_a_denominator() {
        let r = rendered();
        let seen: Vec<u64> = r["bands"]
            .as_array()
            .expect("bands")
            .iter()
            .filter_map(|b| b["concurrency"].as_u64())
            .collect();
        assert_eq!(seen, vec![1, 4, 8, 16], "§4.5: all four bands");
        let base = r["bands"][0]["aggregate_tok_per_sec"]
            .as_f64()
            .expect("agg(1)");
        assert!(base > 0.0, "Arm A is undefined without a non-zero agg(1)");
    }

    /// `scripts/lib/bench_receipt.py`'s rules, which Arm C delegates to.
    #[test]
    fn a_produced_receipt_satisfies_the_receipt_validators_rules() {
        let r = rendered();
        let prov = &r["provenance"];
        for key in [
            "binary_path",
            "binary_sha256",
            "resolution",
            "compute_class",
        ] {
            assert!(prov.get(key).is_some(), "provenance.{key} is required");
        }
        let sha = prov["binary_sha256"].as_str().expect("digest");
        assert_eq!(sha.len(), 64);
        assert!(sha
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        let samples = r["samples_ms"].as_array().expect("samples_ms");
        assert!(
            !samples.is_empty(),
            "I-4: summary-only receipts are rejected"
        );
        assert_eq!(r["n"].as_u64(), Some(samples.len() as u64));
        let distinct: std::collections::BTreeSet<String> = samples
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        assert!(
            distinct.len() > 1,
            "F12: a constant sample set is the fabricated-measurement shape"
        );
    }

    /// The server-only block is NAMED, not silently omitted and not invented.
    #[test]
    fn the_receipt_names_what_a_client_cannot_produce() {
        let r = rendered();
        assert!(r.get("kv").is_none(), "no invented Arm D block");
        let notes = r["unproduced_fields"]
            .as_array()
            .expect("unproduced_fields")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(notes.contains("PP-13"), "{notes}");
        assert!(notes.contains("Arm D `kv` block"), "{notes}");
    }

    /// With the server's own figures supplied, the block appears — and only then.
    #[test]
    fn the_kv_block_appears_only_when_the_server_reported_one() {
        let r = receipt_input(Some(KvBlock::from_server_report(50, 100, Some(0), Some(0))))
            .render()
            .expect("valid receipt");
        assert_eq!(r["kv"]["bytes_used"], 50);
        assert_eq!(r["kv"]["bytes_reserved"], 100);
    }

    /// THE MEASURED-NOT-DEFAULTED PROOF, at receipt level: the same producer
    /// over two band sets that differ only in drain behaviour emits two
    /// different `drain_ms`. A defaulted field cannot do this.
    #[test]
    fn the_emitted_drain_ms_tracks_the_measurement() {
        let straggling = rendered();
        let mut input = receipt_input(None);
        for b in &mut input.bands {
            let window = b.window_ms;
            let last = b.requests.last_mut().expect("non-empty band");
            last.settled_ms = window - 10.0;
            last.ttft_ms = Some(20.0);
            last.token_times_ms = vec![window - 80.0, window - 20.0];
            last.generated_tokens = 2;
        }
        let quiet = input.render().expect("valid receipt");
        assert_eq!(straggling["drain_ms"], serde_json::json!(250.0));
        assert_eq!(quiet["drain_ms"], serde_json::json!(0.0));
        assert_eq!(straggling["bands"][0]["drain_ms"], serde_json::json!(250.0));
    }

    /// §4.4.7 at receipt level is the MAX, so one dominated band cannot be
    /// averaged away by three clean ones.
    #[test]
    fn the_receipt_drain_ms_is_the_worst_band_not_the_mean() {
        let mut input = receipt_input(None);
        for b in input.bands.iter_mut().take(3) {
            let window = b.window_ms;
            let last = b.requests.last_mut().expect("non-empty band");
            last.settled_ms = window - 10.0;
            last.ttft_ms = Some(20.0);
            last.token_times_ms = vec![window - 80.0, window - 20.0];
            last.generated_tokens = 2;
        }
        let r = input.render().expect("valid receipt");
        assert_eq!(r["bands"][0]["drain_ms"], serde_json::json!(0.0));
        assert_eq!(r["bands"][3]["drain_ms"], serde_json::json!(250.0));
        assert_eq!(r["drain_ms"], serde_json::json!(250.0));
    }

    /// v2.2 CODIFIED non-streaming as legal: this test asserted that a band
    /// which never streamed simply omitted `ttft_p95_ms` and named it, and
    /// passed. PP-27 REVERSES that rule — "streaming required … chunk-count
    /// fallback is a hard refusal" — so the test is inverted rather than
    /// deleted: omission is still legal at `schema_version` 2 (a historical
    /// receipt), and at 3 the same band is `NONCONFORMANT-VALID`.
    #[test]
    fn a_non_streaming_band_is_nonconformant_at_v3_and_legal_at_v2() {
        let plain = RequestOutcome::completed(0.0, 500.0, 128).with_prompt_tokens(512);
        let other = RequestOutcome::completed(600.0, 1_250.0, 128).with_prompt_tokens(512);
        let input = ReceiptInput {
            bands: vec![band(1, vec![plain, other]).n_predict(128)],
            ladder: Ladder::derive(
                &[1],
                SlotsAdmitted {
                    apr: Some(16),
                    llama: Some(16),
                },
            ),
            ..receipt_input(None)
        };

        let v3 = input.render().expect("the evidence still renders");
        assert!(
            v3["bands"][0].get("ttft_p95_ms").is_none(),
            "no invented p95"
        );
        assert_eq!(
            v3["bands"][0]["status"], "NONCONFORMANT-VALID",
            "PP-27: a band that never streamed is a record, not a measurement"
        );
        let notes = v3["unproduced_fields"].to_string();
        assert!(notes.contains("PP-27"), "{notes}");

        let historical = ReceiptInput {
            schema_version: 2,
            ..input
        };
        let v2 = historical.render().expect("a v2 receipt still renders");
        assert_ne!(
            v2["bands"][0]["status"], "NONCONFORMANT-VALID",
            "the v2.2 rule stands for a v2-dated receipt"
        );
        assert_eq!(v2["schema_version"], 2);
    }

    /// Provenance has no defaults: an empty `resolution` is refused outright.
    /// A `--resolution` that defaults to `scripts/apr_bin.sh` invents provenance.
    #[test]
    fn an_empty_resolution_is_refused_rather_than_defaulted() {
        let mut input = receipt_input(None);
        input.provenance.resolution = String::new();
        let err = input.render().expect_err("empty resolution");
        assert!(err.contains("no default"), "{err}");
    }

    /// PP-2 — a compute class the build cannot reach is a fabricated claim.
    /// Checked against the SUBJECT's feature set: the subject is the process
    /// that took the path, and until v3 the receipt carried only the client's.
    #[test]
    fn a_compute_class_the_build_cannot_reach_is_refused() {
        let mut input = receipt_input(None);
        input.provenance.subject.feature_set = vec!["inference".to_string()];
        let err = input.render().expect_err("cuda without the feature");
        assert!(err.contains("PP-2"), "{err}");
        assert!(err.contains("subject.feature_set"), "{err}");
    }

    /// I-13 — a `client_tokenizer` declaration without a digest is refused.
    #[test]
    fn a_client_tokenizer_without_a_digest_is_refused() {
        let input = ReceiptInput {
            tokenization: TokenizationBlock::ClientTokenizer {
                tokenizer_sha256: "not-a-digest".to_string(),
                counts_special_tokens: false,
                counts_prompt_echo: false,
            },
            ..receipt_input(None)
        };
        let err = input.render().expect_err("bad digest");
        assert!(err.contains("client_tokenizer"), "{err}");
    }

    /// A receipt with no bands is a vacuous pass, and is refused.
    #[test]
    fn a_receipt_with_no_bands_is_refused() {
        let input = ReceiptInput {
            bands: Vec::new(),
            ..receipt_input(None)
        };
        let err = input.render().expect_err("no bands");
        assert!(err.contains("vacuous"), "{err}");
    }

    /// The rendered receipt is valid JSON that round-trips.
    #[test]
    fn the_receipt_renders_to_parseable_json() {
        let text = receipt_input(None).render_string().expect("renders");
        let back: Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(back["workload"], "W1");
        assert_eq!(back["client_model"], "closed_loop");
        assert_eq!(back["spec"], SPEC_ID);
        assert_eq!(back["schema_version"], 3);
    }

    // -- PP-30 ---------------------------------------------------------------

    /// PP-30 must-not-fire: every receipt carries the clock it was written by.
    #[test]
    fn timestamp_ok__a_receipt_carries_its_start_instant_and_clock() {
        let r = rendered();
        assert_eq!(r["provenance"]["started_utc"], "2026-09-02T10:11:12.345Z");
        assert_eq!(
            r["provenance"]["clock_source"],
            CLOCK_SOURCE_SYSTEM_REALTIME
        );
        assert_eq!(r["run_id"], run_id().as_str());
        // …and the id is recomputable from the receipt's own contents (§1 (d)).
        let recomputed = RunId::derive(
            r["provenance"]["started_utc"]
                .as_str()
                .expect("started_utc"),
            r["provenance"]["host"].as_str().expect("host"),
            r["provenance"]["client"]["sha256"]
                .as_str()
                .expect("client sha"),
            4242,
        );
        assert_eq!(r["run_id"], recomputed.as_str());
    }

    /// PP-30 must-fire: a receipt whose timestamp is not a canonical UTC
    /// instant cannot be written at all.
    #[test]
    fn timestamp_absent__an_unparseable_start_instant_is_refused() {
        for bad in ["", "2026-09-02", "2026-09-02T10:11:12+00:00"] {
            let mut input = receipt_input(None);
            input.provenance.started_utc = bad.to_string();
            let err = input.render().expect_err("{bad} must be refused");
            assert!(err.contains("started_utc"), "{bad}: {err}");
        }
    }

    // -- PP-2 / PP-18 / PP-25 -------------------------------------------------

    /// The subject and the client are separate identities. Until v3 one
    /// `binary_sha256` carried both, filled from `current_exe()` — the CLIENT —
    /// so PP-18's ancestor check would have validated the wrong binary.
    #[test]
    fn subject_and_client_are_distinct_identities() {
        let r = rendered();
        let subject = r["provenance"]["subject"]["sha256"]
            .as_str()
            .expect("subject sha");
        let client = r["provenance"]["client"]["sha256"]
            .as_str()
            .expect("client sha");
        assert_ne!(subject, client, "two binaries, two digests");
        assert_eq!(
            client, r["provenance"]["binary_sha256"],
            "the legacy field is the client's, and the new one says so"
        );
        assert!(r["provenance"]["subject"]["commit"].is_string());
        assert!(r["provenance"]["client"]["commit"].is_string());
    }

    /// Every digest in the split provenance is checked, not just the legacy
    /// one. Until v3 there was one `binary_sha256`; four more identities now
    /// carry a digest and each is a place a typo could enter unnoticed.
    #[test]
    fn every_digest_in_the_split_provenance_is_checked() {
        for (name, mutate) in [
            (
                "binary_sha256",
                Box::new(|p: &mut Provenance| p.binary_sha256 = "nope".to_string())
                    as Box<dyn Fn(&mut Provenance)>,
            ),
            (
                "subject.sha256",
                Box::new(|p: &mut Provenance| p.subject.sha256 = "A".repeat(64)),
            ),
            (
                "client.sha256",
                Box::new(|p: &mut Provenance| p.client.sha256 = "z".repeat(64)),
            ),
            (
                "model_file.sha256",
                Box::new(|p: &mut Provenance| {
                    p.model_file = Some(ModelIdentity {
                        path: "/models/qwen.gguf".to_string(),
                        sha256: "short".to_string(),
                        bytes: 4_683_073_440,
                    });
                }),
            ),
            (
                "comparator.sha256",
                Box::new(|p: &mut Provenance| {
                    p.comparator = Some(ComparatorIdentity {
                        commit: "39173bcac".to_string(),
                        cmake: "cmake -DGGML_CUDA=ON".to_string(),
                        sha256: "0123".to_string(),
                        pin_expiry: "2026-12-01T00:00:00.000Z".to_string(),
                        props: serde_json::json!({}),
                    });
                }),
            ),
        ] {
            let mut input = receipt_input(None);
            mutate(&mut input.provenance);
            let err = input.render().expect_err("{name} must be checked");
            assert!(err.contains(name), "expected {name} in: {err}");
            assert!(err.contains("64 lowercase hex"), "{err}");
        }

        // …and a well-formed model file is accepted, so the check above is
        // about the digest and not about the block being present at all.
        let mut ok = receipt_input(None);
        ok.provenance.model_file = Some(ModelIdentity {
            path: "/models/qwen.gguf".to_string(),
            sha256: "9".repeat(64),
            bytes: 4_683_073_440,
        });
        let r = ok.render().expect("a well-formed model file renders");
        assert_eq!(r["provenance"]["model_file"]["bytes"], 4_683_073_440_u64);
    }

    /// PP-13 — `prefill` is the SERVER's number, and the receipt says so beside
    /// it. A `prefill_tok_per_sec` with no `prefill_source` would be
    /// indistinguishable from a harness estimate.
    #[test]
    fn prefill_is_emitted_with_its_source() {
        let r = rendered();
        for b in r["bands"].as_array().expect("bands") {
            assert!(b["prefill_tok_per_sec"].as_f64().expect("prefill") > 0.0);
            assert_eq!(b["prefill_source"], "server");
        }

        // Drop the server timings and BOTH keys go, together.
        let mut input = receipt_input(None);
        for band in &mut input.bands {
            for req in &mut band.requests {
                req.prefill_ms = None;
            }
        }
        let without = input.render().expect("the evidence still renders");
        let b = &without["bands"][0];
        assert!(b.get("prefill_tok_per_sec").is_none());
        assert!(b.get("prefill_source").is_none());
        assert_eq!(b["status"], "NONCONFORMANT-VALID");
        assert!(without["unproduced_fields"].to_string().contains("PP-13"));
    }

    /// PP-20 — a pin that expired before the run marks every band
    /// `COMPARATOR_STALE`.
    #[test]
    fn a_stale_comparator_pin_marks_every_band() {
        let mut input = receipt_input(None);
        input.provenance.comparator = Some(ComparatorIdentity {
            commit: "39173bcac".to_string(),
            cmake: "cmake -DGGML_CUDA=ON".to_string(),
            sha256: "e".repeat(64),
            pin_expiry: "2026-08-01T00:00:00.000Z".to_string(),
            props: serde_json::json!({"n_ctx": 4096, "total_slots": 4}),
        });
        assert!(input.provenance.comparator_is_stale());
        let r = input.render().expect("renders");
        for b in r["bands"].as_array().expect("bands") {
            assert_eq!(b["status"], "COMPARATOR_STALE", "{b}");
        }

        // A pin that is still fresh does not.
        let mut fresh = input;
        if let Some(c) = fresh.provenance.comparator.as_mut() {
            c.pin_expiry = "2026-12-01T00:00:00.000Z".to_string();
        }
        assert!(!fresh.provenance.comparator_is_stale());
        let r = fresh.render().expect("renders");
        assert_ne!(r["bands"][0]["status"], "COMPARATOR_STALE");

        // And the boundary is exclusive: a pin expiring at the instant the run
        // started has not expired before it.
        let mut at_the_instant = fresh;
        if let Some(c) = at_the_instant.provenance.comparator.as_mut() {
            c.pin_expiry = at_the_instant.provenance.started_utc.clone();
        }
        assert!(
            !at_the_instant.provenance.comparator_is_stale(),
            "expiry == started_utc is not `expiry < started_utc`"
        );
    }

    // -- PP-7 -----------------------------------------------------------------

    /// PP-7 must-not-fire: every band carries its own per-request rows, and the
    /// receipt links the gz side file the token times went to.
    #[test]
    fn every_band_carries_its_own_samples() {
        let r = rendered();
        for b in r["bands"].as_array().expect("bands") {
            let samples = b["samples"].as_array().expect("samples[]");
            assert_eq!(
                samples.len(),
                b["requested"].as_u64().expect("requested") as usize
            );
            for row in samples {
                for key in [
                    "index",
                    "issued_ms",
                    "settled_ms",
                    "outcome",
                    "generated_tokens",
                    "prompt_tokens",
                    "ttft_ms",
                    "in_flight_at_start",
                ] {
                    assert!(row.get(key).is_some(), "samples row lacks {key}: {row}");
                }
                assert!(
                    row.get("token_times_ms").is_none(),
                    "token times stay in the gz side file"
                );
            }
            assert!(
                b.get("samples_file").is_some(),
                "the key is present even when no file was written"
            );
        }

        // And when a file WAS written, the band names it by digest.
        let mut input = receipt_input(None);
        input.bands[0] = input.bands[0].clone().samples_file(SamplesFile {
            path: std::path::PathBuf::from("samples.c1.r1.jsonl.gz"),
            sha256: "f".repeat(64),
            bytes: 4_096,
            rows: 30,
        });
        let r = input.render().expect("renders");
        assert_eq!(
            r["bands"][0]["samples_file"]["sha256"],
            "f".repeat(64).as_str()
        );
    }

    // -- PP-24 ----------------------------------------------------------------

    /// PP-24 must-fire: a band above what the servers admitted measured a
    /// queue, not a server — unless it says which lane capped it.
    #[test]
    fn a_band_above_the_derived_ladder_is_refused_unless_capped() {
        let capped = Ladder::derive(
            &[1, 4, 8, 16],
            SlotsAdmitted {
                apr: Some(11),
                llama: Some(16),
            },
        );
        assert_eq!(capped.derived, vec![1, 4, 8]);
        let input = ReceiptInput {
            ladder: capped.clone(),
            ..receipt_input(None)
        };
        let err = input.render().expect_err("c=16 above an 11-slot subject");
        assert!(err.contains("PP-24"), "{err}");
        assert!(err.contains("c=16"), "{err}");

        // …and it renders once the band says which lane capped it.
        let mut allowed = ReceiptInput {
            ladder: capped,
            ..receipt_input(None)
        };
        allowed.bands[3].comparator = ComparatorStatus::Unmeasured {
            owner: "perf-gate".to_string(),
            reason: "the subject admitted 11 slots; c=16 was not run".to_string(),
            admission_capped: Some(AdmissionCap {
                lane: Lane::Apr,
                cap: 11,
            }),
        };
        let r = allowed.render().expect("an admission cap is an answer");
        assert_eq!(r["bands"][3]["comparator_admission_capped"]["cap"], 11);
        assert_eq!(r["bands"][3]["comparator_admission_capped"]["lane"], "apr");
        assert_eq!(Lane::Apr.wire_token(), "apr");
        assert_eq!(Lane::Llama.wire_token(), "llama");
    }

    /// PP-24 must-not-fire: a server-reported budget ceiling yields `NA`.
    #[test]
    fn a_server_reported_budget_ceiling_yields_na() {
        let mut input = ReceiptInput {
            ladder: Ladder::derive(
                &[1, 4, 8, 16],
                SlotsAdmitted {
                    apr: Some(11),
                    llama: Some(16),
                },
            ),
            ..receipt_input(None)
        };
        input.bands[3].comparator = ComparatorStatus::NotApplicable {
            decided_by: "spec-owner".to_string(),
            reason: "KV budget admits 11 slots by design".to_string(),
            budget: Some("kv_per_slot 469.8 MB, reserve 3.5 GB => 11 slots".to_string()),
        };
        let r = input.render().expect("a decided ceiling is an answer");
        assert_eq!(r["bands"][3]["status"], "NA");
        assert_eq!(r["bands"][3]["comparator_status"], "NOT_APPLICABLE");
        assert!(r["bands"][3]["comparator_budget"].is_string());
        assert_eq!(r["ladder"]["derived"], serde_json::json!([1, 4, 8]));
        assert_eq!(r["ladder"]["slots_admitted"]["apr"], 11);
    }

    // -- PP-31 / PP-23 --------------------------------------------------------

    /// PP-31 — `scaling_efficiency` is REPORTED. Raising `agg(1)` lowers it,
    /// and nothing fails: an up-only ratchet on this figure fails a build for
    /// getting faster, which is the defect PP-31 exists to remove.
    #[test]
    fn scaling_efficiency_is_reported_not_gated() {
        let base = rendered();
        let se = base["bands"][1]["scaling_efficiency"]
            .as_f64()
            .expect("c=4 reports one");
        assert!(se > 0.0, "{se}");
        assert!(
            base["bands"][0]["scaling_efficiency"].is_null(),
            "null at c=1, where it is 1 by construction"
        );
        assert!(
            base["bands"][0]["overhead_share"].as_f64().is_some(),
            "overhead_share is the c=1 figure"
        );
        assert!(
            base["bands"][1]["overhead_share"].is_null(),
            "…and only the c=1 figure"
        );

        // Make c=1 20% faster by shortening every c=1 request. Scaling
        // efficiency falls, and the receipt still renders.
        let mut input = receipt_input(None);
        for r in &mut input.bands[0].requests {
            let dur = r.settled_ms - r.issued_ms;
            r.settled_ms = r.issued_ms + dur / 1.2;
        }
        let faster = input.render().expect("a faster c=1 is not an error");
        let agg1_before = base["bands"][0]["aggregate_tok_per_sec"]
            .as_f64()
            .expect("agg(1)");
        let agg1_after = faster["bands"][0]["aggregate_tok_per_sec"]
            .as_f64()
            .expect("agg(1)");
        assert!(agg1_after > agg1_before, "{agg1_after} vs {agg1_before}");
        assert!(
            faster["bands"][1]["scaling_efficiency"]
                .as_f64()
                .expect("se")
                < se,
            "SE falls when agg(1) rises — and that is REPORTED, not a failure"
        );
    }

    /// PP-23 must-fire: a per-sequence decode above the memory-bandwidth
    /// ceiling is a wrong measurement, not a fast run.
    #[test]
    fn a_decode_above_the_roofline_is_refused() {
        let mut input = receipt_input(None);
        input.roofline = Some(Roofline {
            // 1 tok/s ceiling: a byte a second against a one-byte model.
            bandwidth_bytes_per_sec: 1.0,
            model_bytes: 1,
        });
        let err = input.render().expect_err("dec(1) is far above 1 tok/s");
        assert!(err.contains("PP-23"), "{err}");
        assert!(err.contains("c=1"), "{err}");

        // The comparison is exclusive: a decode exactly AT the ceiling is the
        // bound being reached, not exceeded. Without this the `>` could be a
        // `>=` and every at-the-ceiling receipt would be refused.
        let dec1 = rendered()["bands"][0]["decode_tok_per_sec"]
            .as_f64()
            .expect("dec(1)");
        let mut exact = receipt_input(None);
        exact.roofline = Some(Roofline {
            bandwidth_bytes_per_sec: dec1,
            model_bytes: 1,
        });
        let r = exact.render().expect("at the ceiling is not above it");
        assert_eq!(r["bands"][0]["roofline_tok_per_sec"], dec1);
    }

    /// PP-23 must-not-fire: the AGGREGATE is never compared. Over `c` sequences
    /// one weight read serves `c` tokens, so gx10's c=8 aggregate sitting ABOVE the
    /// per-sequence ceiling was correct (evidence/perf-gate-001-w1-gx10/receipt.r1.json
    /// carries the figures; they are not restated here, PP-12).
    #[test]
    fn an_aggregate_above_the_roofline_is_not_compared() {
        // A slow single stream — four tokens 120 ms apart, so `dec(1)` is
        // 8.3 tok/s and sits under the ceiling. The c=8 band's AGGREGATE is two
        // orders of magnitude above the same ceiling, and that is not a defect.
        let slow: Vec<RequestOutcome> = (0..30)
            .map(|i| {
                let issued = f64::from(i) * 1_900.0;
                RequestOutcome::completed(issued, issued + 400.0 + f64::from(i % 5), 4)
                    .streamed(
                        40.0,
                        (0..4)
                            .map(|k| issued + 40.0 + f64::from(k) * 120.0)
                            .collect(),
                    )
                    .server_prefill(512, 30.0)
            })
            .collect();
        let mut input = receipt_input(None);
        input.bands[0] = band(1, slow).n_predict(4).stream_mode(StreamMode::Live);
        input.roofline = Some(Roofline {
            bandwidth_bytes_per_sec: 10.0,
            model_bytes: 1,
        });
        let r = input
            .render()
            .expect("an aggregate above the ceiling is fine");
        let dec1 = r["bands"][0]["decode_tok_per_sec"]
            .as_f64()
            .expect("dec(1)");
        let agg8 = r["bands"][2]["aggregate_tok_per_sec"]
            .as_f64()
            .expect("agg(8)");
        assert!(dec1 <= 10.0, "dec(1)={dec1} must sit under the ceiling");
        assert!(
            agg8 > 10.0,
            "agg(8)={agg8} must sit ABOVE it — that is the case under test"
        );
        assert_eq!(r["bands"][2]["roofline_tok_per_sec"], 10.0);
    }

    // -- PP-3 -----------------------------------------------------------------

    /// PP-3 must-fire, at the wire: a v3 receipt never writes a bare scalar
    /// ratio anywhere. The type-level half is in `drain.rs` — there is no
    /// public constructor for `ComparatorStatus::Measured` other than the join.
    #[test]
    fn ratio_bare__a_scalar_ratio_is_unrepresentable() {
        let r = rendered();
        let text = r.to_string();
        for scalar in ["agg_ratio", "decode_ratio", "prefill_ratio"] {
            assert!(
                !text.contains(scalar),
                "a v3 receipt must not write `{scalar}` anywhere: {text}"
            );
        }
        for b in r["bands"].as_array().expect("bands") {
            assert!(b["ratios"].is_null(), "no comparator lane, no ratios");
            assert!(b["baseline"].is_null());
            assert_eq!(b["comparator_status"], "UNMEASURED");
        }
    }

    /// PP-3 at the wire: a joined band emits its baseline and its ratios, the
    /// baseline is rendered by the same code path minus its own baseline, and
    /// there is still no bare scalar anywhere.
    #[test]
    fn a_measured_band_renders_its_baseline_and_ratios() {
        let mut input = receipt_input(None);
        let subject = input.bands[0].clone();
        let comparator = synthetic_band(1);
        let key = input.join_key(&subject);
        let comparator_key = input.join_key(&comparator);
        let id = run_id();
        input.bands[0].comparator =
            BandInput::join_status(&subject, &comparator, &key, &comparator_key, (&id, &id))
                .expect("a same-run join");

        let r = input.render().expect("renders");
        let b = &r["bands"][0];
        assert_eq!(b["status"], "MEASURED");
        assert_eq!(b["comparator_status"], "MEASURED");
        assert_eq!(b["baseline"]["concurrency"], 1);
        assert!(b["baseline"]["aggregate_tok_per_sec"].as_f64().is_some());
        assert_eq!(b["baseline"]["run_id"], id.as_str());
        assert!(
            b["baseline"].get("baseline").is_none() && b["baseline"].get("ratios").is_none(),
            "a baseline is one comparator lane, not a chain of them"
        );
        assert_eq!(b["ratios"]["agg"]["method"], "replicate_t_lower");
        assert_eq!(b["ratios"]["dec"]["method"], "paired_percentile_bootstrap");
        assert!(
            (b["ratios"]["agg"]["point"].as_f64().expect("agg point") - 1.0).abs() < 1e-9,
            "identical lanes are parity"
        );
        assert!(
            !r.to_string().contains("agg_ratio"),
            "and still no bare scalar"
        );

        let parsed = Receipt::parse(&r.to_string()).expect("a joined receipt parses");
        parsed.validate().expect("and validates");
        let band = &parsed.bands[0];
        assert!(band.ratios.is_some() && band.baseline.is_some());
        assert_eq!(
            band.baseline.as_ref().expect("baseline").run_id.as_ref(),
            Some(&id)
        );
    }

    /// §3 / PP-31 — `scaling_efficiency` and `overhead_share` on a BASELINE are
    /// the comparator lane's, computed from the comparator's own `agg(1)` and
    /// `dec(1)`.
    ///
    /// MUST-FIRE: the renderer passed the SUBJECT's `RenderContext` to the
    /// baseline, so the baseline's scaling efficiency was
    /// `llama_agg(c) / (c · apr_agg(1))` — not a scaling efficiency of either
    /// lane, and a number that MOVES when the subject gets faster while the
    /// comparator does not change at all.
    #[test]
    fn a_baselines_per_lane_figures_come_from_the_comparator_lane() {
        // Two lanes that differ: the comparator completes about half as many
        // requests over the SAME window, so its `agg` is about half the
        // subject's at every band and a cross-lane denominator shows in the
        // digits. Per-request timings are untouched, so the two lanes still
        // join (PP-22) and `dec` is unaffected.
        let halved = |c: u32| -> BandInput {
            let full = synthetic_band(c);
            let last = full.requests.len() - 1;
            let requests = full
                .requests
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 0 || *i == last)
                .map(|(_, r)| r.clone())
                .collect();
            BandInput { requests, ..full }
        };
        let mut input = receipt_input(None);
        let id = run_id();
        for i in [0_usize, 1] {
            let subject = input.bands[i].clone();
            let comparator = halved(subject.concurrency);
            let key = input.join_key(&subject);
            let ckey = input.join_key(&comparator);
            input.bands[i].comparator =
                BandInput::join_status(&subject, &comparator, &key, &ckey, (&id, &id))
                    .expect("a same-run join");
        }
        let r = input.render().expect("renders");

        let subject_agg1 = r["bands"][0]["aggregate_tok_per_sec"]
            .as_f64()
            .expect("agg(1) subject");
        let baseline_agg1 = r["bands"][0]["baseline"]["aggregate_tok_per_sec"]
            .as_f64()
            .expect("agg(1) comparator");
        assert!(
            (subject_agg1 / baseline_agg1) > 1.5,
            "the fixture's two lanes must differ, or this proves nothing: {subject_agg1} vs \
             {baseline_agg1}"
        );

        let baseline_c4 = &r["bands"][1]["baseline"];
        let baseline_agg4 = baseline_c4["aggregate_tok_per_sec"]
            .as_f64()
            .expect("agg(4) comparator");
        let want = baseline_agg4 / (4.0 * baseline_agg1);
        let got = baseline_c4["scaling_efficiency"]
            .as_f64()
            .expect("the baseline reports one");
        assert!(
            (got - want).abs() < 1e-9,
            "the baseline's scaling_efficiency must divide by the COMPARATOR's agg(1): got {got}, \
             comparator-lane {want}, subject-lane {}",
            baseline_agg4 / (4.0 * subject_agg1)
        );

        // overhead_share is agg(1)/dec(1), also per lane.
        let baseline_dec1 = r["bands"][0]["baseline"]["decode_tok_per_sec"]
            .as_f64()
            .expect("dec(1) comparator");
        let overhead = r["bands"][0]["baseline"]["overhead_share"]
            .as_f64()
            .expect("the baseline reports one");
        assert!(
            (overhead - baseline_agg1 / baseline_dec1).abs() < 1e-9,
            "overhead_share on the baseline is the comparator's agg(1)/dec(1): {overhead}"
        );
        assert!(
            (overhead - subject_agg1 / baseline_dec1).abs() > 1e-6,
            "…and the subject's agg(1) must give a DIFFERENT answer, or the fixture is degenerate"
        );
    }

    /// §7.4 MUST-FIRE at the RENDER, where the inversion actually lived: a
    /// `c > 1` band with no witness, under an EXPIRED comparator pin, comes out
    /// of `render()` as `INVALID-CORRECTNESS`.
    ///
    /// `render` applied `marked_comparator_stale` to every band after deriving
    /// it, and that method ASSIGNED the status — so a band that reported no
    /// throughput at all, because nothing established its tokens were right,
    /// was relabelled as though its only defect were an out-of-date pin. §7.4
    /// asks "were the tokens right?" before anything else.
    #[test]
    fn a_stale_pin_does_not_relabel_an_unwitnessed_band() {
        let expired = ComparatorIdentity {
            commit: "39173bcac0123456789abcdef0123456789abcde".to_string(),
            cmake: "cmake -B build -DGGML_CUDA=ON".to_string(),
            sha256: "d".repeat(64),
            // Before `provenance().started_utc` (2026-09-02T10:11:12.345Z).
            pin_expiry: "2026-01-01T00:00:00.000Z".to_string(),
            props: Value::Null,
        };
        let stale_provenance = Provenance {
            comparator: Some(expired),
            ..provenance()
        };
        assert!(
            stale_provenance.comparator_is_stale(),
            "the fixture's pin must be expired"
        );

        let unwitnessed = BandInput {
            witness: None,
            ..synthetic_band(4)
        };
        let input = ReceiptInput {
            provenance: stale_provenance.clone(),
            bands: vec![synthetic_band(1), unwitnessed],
            ..receipt_input(None)
        };
        let r = input.render().expect("renders");
        assert_eq!(
            r["bands"][1]["status"], "INVALID-CORRECTNESS",
            "correctness first: a fresher pin would not give this band a throughput"
        );
        assert!(
            r["bands"][1].get("aggregate_tok_per_sec").is_none(),
            "and it still reports none: {}",
            r["bands"][1]
        );
        // REVERT -> the WITNESSED band beside it under the same stale pin IS
        // COMPARATOR_STALE, so the pin rule is live and this is about ordering.
        assert_eq!(r["bands"][0]["status"], "COMPARATOR_STALE");
    }

    /// PP-24 MUST-FIRE: `ladder.derived` is recomputed at render time, so a
    /// hand-written one that disagrees with `declared` + `slots_admitted` is
    /// refused rather than obeyed.
    ///
    /// `Ladder`'s fields are public and it travels through the producer as
    /// data. Nothing recomputed `derived`, so writing `[1, 4, 8, 16]` beside
    /// `slots_admitted: {apr: 4}` excused every band from the PP-24 check and
    /// put four bands on the wire that measured a queue.
    #[test]
    fn admission_unequal__a_hand_written_derived_ladder_is_refused() {
        let honest = Ladder::derive(
            &[1, 4, 8, 16],
            SlotsAdmitted {
                apr: Some(4),
                llama: Some(16),
            },
        );
        assert_eq!(honest.derived, vec![1, 4], "the fixture's real ceiling");

        let forged = Ladder {
            derived: vec![1, 4, 8, 16],
            ..honest.clone()
        };
        let input = ReceiptInput {
            ladder: forged,
            ..receipt_input(None)
        };
        let err = input
            .render()
            .expect_err("a ladder that disagrees with its own inputs must be refused");
        assert!(err.contains("PP-24"), "{err}");
        assert!(
            err.contains("[1, 4]"),
            "the error names the real ladder: {err}"
        );

        // MUST-NOT-FIRE: the honest ladder renders — after the bands above its
        // ceiling are removed, which is what the ladder is FOR.
        let narrowed = ReceiptInput {
            ladder: honest,
            bands: vec![synthetic_band(1), synthetic_band(4)],
            ..receipt_input(None)
        };
        narrowed.render().expect("the derived ladder renders");
    }

    /// PP-3 / §1(d) — `run_id` is recomputable from the receipt's own contents,
    /// and `Receipt::validate` recomputes it.
    ///
    /// MUST-FIRE both ways: a receipt whose id its own four inputs do not
    /// reproduce is refused; the four inputs are all on the wire, `pid`
    /// included — before it was carried, "derived rather than random, so it is
    /// reproducible from the receipt" was a comment no reader could check.
    #[test]
    fn a_run_id_its_own_provenance_does_not_reproduce_is_refused() {
        let text = receipt_input(None).render_string().expect("renders");
        let good = Receipt::parse(&text).expect("parses");
        good.validate().expect("its own id reproduces");
        assert_eq!(good.provenance.client.pid, 4242, "pid is on the wire");

        // Same run, one different pid: a different run_id, and the stated one
        // no longer reproduces.
        let mut value: Value = serde_json::from_str(&text).expect("json");
        value["provenance"]["client"]["pid"] = serde_json::json!(4243);
        let err = Receipt::parse(&value.to_string())
            .expect("still parses")
            .validate()
            .expect_err("the id no longer reproduces");
        assert!(err.contains("PP-3"), "{err}");
        assert!(err.contains("DERIVED"), "{err}");

        // And the same for the other three inputs, one at a time.
        for (pointer, replacement) in [
            ("/provenance/host", serde_json::json!("gx10")),
            (
                "/provenance/started_utc",
                serde_json::json!("2026-09-02T10:11:13.345Z"),
            ),
            (
                "/provenance/client/sha256",
                serde_json::json!("c".repeat(64)),
            ),
        ] {
            let mut v: Value = serde_json::from_str(&text).expect("json");
            *v.pointer_mut(pointer).expect("field exists") = replacement;
            let e = Receipt::parse(&v.to_string())
                .expect("parses")
                .validate()
                .expect_err("changing a run_id input must break the id");
            assert!(e.contains("run_id"), "{pointer}: {e}");
        }
    }

    /// Arm D — `admission_rejected` and `preempted_swap` are `null` when the
    /// server does not count them, never `0`, and the receipt names them.
    ///
    /// `apr serve` reports both as null: it has no KV-admission refusal path
    /// and no swap path, so there is no quantity for either to denote. While
    /// the block demanded four numbers the whole `kv` block was dropped and the
    /// two byte figures the server DID report went with it.
    #[test]
    fn a_partial_kv_block_keeps_the_figures_the_server_did_report() {
        let uncounted = KvBlock::from_server_report(50, 100, None, None);
        assert_eq!(
            uncounted.uncounted_fields(),
            vec!["kv.admission_rejected", "kv.preempted_swap"]
        );
        let r = receipt_input(Some(uncounted)).render().expect("renders");
        assert_eq!(r["kv"]["bytes_used"], 50, "the reported figures survive");
        assert_eq!(r["kv"]["bytes_reserved"], 100);
        assert!(
            r["kv"]["admission_rejected"].is_null(),
            "null, not 0: `not counted` and `counted none` are different facts"
        );
        let notes = r["unproduced_fields"].as_array().expect("array");
        assert!(
            notes.iter().any(|n| n
                .as_str()
                .is_some_and(|t| t.contains("kv.admission_rejected"))),
            "the missing counters are NAMED: {notes:?}"
        );

        // MUST-NOT-FIRE: a server that counts them names nothing.
        let complete = KvBlock::from_server_report(50, 100, Some(0), Some(3));
        assert!(complete.uncounted_fields().is_empty());
        let full = receipt_input(Some(complete)).render().expect("renders");
        assert_eq!(full["kv"]["admission_rejected"], 0);
        assert_eq!(full["kv"]["preempted_swap"], 3);
        assert!(
            !full["unproduced_fields"]
                .as_array()
                .expect("array")
                .iter()
                .any(|n| n
                    .as_str()
                    .is_some_and(|t| t.contains("kv.admission_rejected"))),
            "a complete block has nothing to name"
        );
    }

    /// PP-22 through the receipt: the key is built from the receipt's own
    /// provenance and protocol, so two receipts that disagree on either cannot
    /// join even when their bands look identical.
    #[test]
    fn join_ok__the_receipt_builds_its_own_join_key() {
        let input = receipt_input(None);
        let key = input.join_key(&input.bands[1]);
        assert_eq!(key.host, "lambda");
        assert_eq!(key.band, 4);
        assert_eq!(key.window_ms, 60_000);
        assert_eq!(key.replicates, 5);
        assert!(key.interleaved);
        assert_eq!(key.n_predict, 128);
        assert_eq!(key.n_batch, Some(2048));
        key.refuse_mismatch(&input.join_key(&input.bands[1]))
            .expect("a key joins itself");

        let elsewhere = ReceiptInput {
            provenance: Provenance {
                host: "gx10".to_string(),
                ..provenance()
            },
            ..receipt_input(None)
        };
        let err = key
            .refuse_mismatch(&elsewhere.join_key(&elsewhere.bands[1]))
            .expect_err("two hosts");
        assert!(err.contains("host: lambda != gx10"), "{err}");

        // …and the key really is on the wire, band by band.
        let r = input.render().expect("renders");
        assert_eq!(r["bands"][1]["join_key"]["band"], 4);
        assert_eq!(r["bands"][1]["join_key"]["window_ms"], 60_000);
        assert_eq!(r["bands"][1]["join_key"]["interleaved"], true);
    }

    // -- the typed reader ------------------------------------------------------

    /// The receipt parses back into its own type, and out again unchanged.
    #[test]
    fn a_receipt_round_trips_through_its_own_type() {
        let text = receipt_input(None).render_string().expect("renders");
        let parsed = Receipt::parse(&text).expect("parses");
        parsed.validate().expect("and satisfies its own L1 rules");
        assert_eq!(parsed.spec, SPEC_ID);
        assert_eq!(parsed.schema_version, 3);
        assert_eq!(parsed.run_id, run_id());
        assert_eq!(parsed.bands.len(), 4);
        assert_eq!(parsed.bands[0].status, "UNMEASURED");
        assert_eq!(parsed.protocol, ProtocolParams::spec_fallback());

        let again = serde_json::to_string(&parsed).expect("serialises");
        let twice = Receipt::parse(&again).expect("re-parses");
        assert_eq!(parsed, twice, "the type is its own fixed point");
    }

    /// `deny_unknown_fields`: a receipt carrying a key the schema does not know
    /// is refused rather than silently ignored. That is what stops a producer
    /// adding `agg_ratio` beside a band and a reader quietly dropping it.
    /// PP-21: `scripts/perf_receipt_sign.sh` appends a `signature` block on the
    /// measuring host AFTER rendering. The reader must accept it (opaquely —
    /// verification is receipt_sig.py's job), or every signed receipt fails to
    /// parse and the typed reader can only ever read unsigned evidence.
    #[test]
    fn a_signed_receipt_parses_and_an_unsigned_one_carries_no_signature_key() {
        let unsigned = rendered();
        assert!(
            unsigned.get("signature").is_none(),
            "the renderer must never sign"
        );
        let mut signed = unsigned.clone();
        signed.as_object_mut().expect("object").insert(
            "signature".to_string(),
            serde_json::json!({"alg": "hmac-sha256", "key_id": "lambda-1",
                "signed_at": "2026-09-02T00:00:00Z", "commit": "0".repeat(40),
                "host": "lambda", "body_sha256": "0".repeat(64), "value": "00"}),
        );
        let parsed = Receipt::parse(&signed.to_string()).expect("a signed receipt parses");
        assert_eq!(
            parsed.signature.as_ref().and_then(|s| s["key_id"].as_str()),
            Some("lambda-1")
        );
        let back = serde_json::to_value(&parsed).expect("serialises");
        assert_eq!(back["signature"]["alg"], "hmac-sha256");
        let plain = Receipt::parse(&unsigned.to_string()).expect("unsigned parses");
        assert!(plain.signature.is_none());
    }

    #[test]
    fn an_unknown_field_is_refused() {
        let mut value = rendered();
        value
            .as_object_mut()
            .expect("object")
            .insert("agg_ratio".to_string(), serde_json::json!(1.42));
        let err = Receipt::parse(&value.to_string()).expect_err("unknown top-level field");
        assert!(err.contains("agg_ratio"), "{err}");

        let mut banded = rendered();
        banded["bands"][0]
            .as_object_mut()
            .expect("object")
            .insert("agg_ratio".to_string(), serde_json::json!(1.42));
        assert!(
            Receipt::parse(&banded.to_string()).is_err(),
            "and the same inside a band"
        );
    }

    /// PP-3 / PP-17, read back: `ratios` without a `baseline` is refused by the
    /// reader as well as being unconstructible by the producer.
    #[test]
    fn claim_bandless__ratios_without_a_baseline_are_refused_by_the_reader() {
        let mut value = rendered();
        value["bands"][0].as_object_mut().expect("object").insert(
            "ratios".to_string(),
            serde_json::json!({
                "agg": {"point": 1.42, "lcb95": null, "method": "replicate_t_lower", "n": 1},
                "dec": null,
                "prefill": null
            }),
        );
        let parsed = Receipt::parse(&value.to_string()).expect("the shape parses");
        let err = parsed.validate().expect_err("but does not validate");
        assert!(err.contains("PP-3"), "{err}");
        assert!(err.contains("baseline"), "{err}");
    }

    /// PP-11, read back: the status vocabulary is closed.
    #[test]
    fn a_status_outside_the_vocabulary_is_refused_by_the_reader() {
        let mut value = rendered();
        value["bands"][0]["status"] = serde_json::json!("SKIP");
        let parsed = Receipt::parse(&value.to_string()).expect("a string parses");
        let err = parsed.validate().expect_err("SKIP is not a status");
        assert!(err.contains("§7.4"), "{err}");
    }

    /// The `protocol` block is on the wire, so a reader can tell a 60 s
    /// conformant band from a 5 s one without trusting the producer.
    #[test]
    fn the_protocol_block_is_on_the_wire() {
        let r = rendered();
        let p = &r["protocol"];
        assert_eq!(p["window_ms"], 60_000);
        assert_eq!(p["warmup_requests_per_worker"], 2);
        assert_eq!(p["quiesce_ms"], 5_000);
        assert_eq!(p["cooldown_ms"], 10_000);
        assert_eq!(p["n_predict"], 128);
        assert_eq!(p["replicates"], 5);
        assert_eq!(p["interleaved"], true);
        assert_eq!(p["sampler"]["temperature"], 0.0);
        assert_eq!(p["sampler"]["ignore_eos"], true);
        assert!(p["sampler"]["seed"].is_u64());
    }

    /// PP-28 at receipt level: the short count is summed over bands.
    #[test]
    fn sampler_pinned__a_conformant_receipt_reports_no_short_samples() {
        let r = rendered();
        assert_eq!(r["short_of_n_predict"], 0);
        for b in r["bands"].as_array().expect("bands") {
            assert_eq!(b["short_of_n_predict"], 0);
            assert_eq!(b["stream_mode"], "live");
            assert_eq!(b["stream_witness"]["verdict"], "live");
        }

        let mut input = receipt_input(None);
        input.bands[0].requests[3].generated_tokens = 67;
        input.bands[2].requests[5].generated_tokens = 120;
        let short = input.render().expect("the evidence still renders");
        assert_eq!(short["short_of_n_predict"], 2);
        assert_eq!(short["bands"][0]["short_of_n_predict"], 1);
        assert_eq!(short["bands"][0]["status"], "NONCONFORMANT-VALID");
        assert_eq!(short["bands"][1]["short_of_n_predict"], 0);
    }
}
