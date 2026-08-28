//! APR-PERF-GATE-001 v2.2 §4.4.6 / §4.4.7 — the receipt producers.
//!
//! # The defect this closes (PERF-026)
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
//! # What is here
//!
//! - [`drain`] — the §4.4.7 producer. `drain_ms`, the four request counters,
//!   and the `SUSPECT` annotation, all **derived from per-request terminal
//!   records**. There is no constructor that accepts a `drain_ms` scalar.
//! - [`receipt`] — the emitter. Turns those records plus a §4.4.6 tokenization
//!   declaration and §4.2 provenance into the JSON `scripts/perf_gate.sh` and
//!   `scripts/lib/bench_receipt.py` actually read.
//!
//! # The definition that had to be right first (§4.4.7)
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
//! # What a client cannot produce, said rather than synthesised
//!
//! §4.4.9's scheduler block is **server**-reported: I-16 requires
//! `max_in_flight` come from the server rather than be inferred by the harness,
//! and I-2 requires `gpu_layers_resolved` be read from the loader. Arm D's `kv`
//! block is the same. Neither is emitted from client-side data. They are named
//! in the receipt's `unproduced_fields` with the reason, because a default that
//! invents provenance is worse than a missing field — it is indistinguishable
//! from a real answer.
//!
//! # MERGE NOTE — PERF-024 on `feat/n1-band-cli`
//!
//! That branch adds `window.rs`, `protocol.rs`, `metrics.rs`, `samples.rs` and
//! `bootstrap.rs` to this same directory: the §4.4.1/§4.4.2 admission and
//! termination rule, the §4.4.3 metric set, and the §4.4.4 bootstrap. The two
//! halves compose — its `WindowController` decides *when* `T` falls and hands
//! over per-request records; [`drain::BandInput`] derives the §4.4.7 numbers
//! from those records and [`receipt::ReceiptInput`] writes them out, which
//! PERF-024's own docs list as explicitly out of its scope ("Receipt emission.
//! Nothing here writes `receipt.json`").
//!
//! Two overlaps must be resolved in the integration commit rather than left to
//! drift: `protocol::Tokenization` and [`receipt::TokenizationBlock`] are the
//! same §4.4.6 block, and `protocol::Outcome` and [`drain::Outcome`] are the
//! same four counters. Keep one of each.

pub mod drain;
pub mod receipt;

pub use drain::{
    percentile, BandInput, ComparatorStatus, DerivedBand, Outcome, RequestOutcome,
    DRAIN_SUSPECT_FRACTION, REQUEST_TIMEOUT_MS,
};
pub use receipt::{
    ComputeClass, KvBlock, Provenance, ReceiptInput, TokenCountingMethod, TokenizationBlock,
    Workload, SERVER_ONLY_FIELDS,
};

#[cfg(test)]
mod gate_conformance_tests {
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
            "A comparator ratio needs a baseline object that itself passes every receipt rule \
             (I-3) and a lane driven by the same client binary (I-15, PERF-019).",
        ),
        ("decode_ratio", "As agg_ratio (I-3, I-15, PERF-019)."),
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
        BandInput {
            concurrency,
            window_ms: 60_000.0,
            requests,
            comparator: ComparatorStatus::Unmeasured {
                owner: "perf-gate".to_string(),
                reason: "no same-client comparator lane on this cell yet (I-15, PERF-019)"
                    .to_string(),
            },
        }
    }

    /// A §4.4-shaped band: `max(30, 8c)` requests in a 60 s window, with the
    /// final request deliberately straddling `T` so `drain_ms` is non-zero and
    /// demonstrably came from the data rather than from a constant.
    fn synthetic_band(concurrency: u32) -> BandInput {
        let window_ms = 60_000.0;
        let n = 30.max(8 * concurrency as usize);
        let step = (window_ms - 1_000.0) / n as f64;
        let mut requests: Vec<RequestOutcome> = (0..n)
            .map(|i| {
                let issued = i as f64 * step;
                // Varying durations: a constant distribution is F12.
                let dur = 400.0 + (i % 7) as f64 * 13.0;
                RequestOutcome {
                    issued_ms: issued,
                    settled_ms: issued + dur,
                    outcome: Outcome::Completed,
                    generated_tokens: 128,
                    ttft_ms: Some(40.0 + (i % 5) as f64),
                    token_times_ms: (0..128)
                        .map(|k| issued + 40.0 + k as f64 * (dur - 40.0) / 128.0)
                        .collect(),
                }
            })
            .collect();
        let last = requests.last_mut().expect("n >= 30");
        last.issued_ms = window_ms - 100.0;
        last.settled_ms = window_ms + 250.0;
        last.token_times_ms = (0..128)
            .map(|k| window_ms - 60.0 + k as f64 * 310.0 / 128.0)
            .collect();
        band(concurrency, requests)
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
        }
    }

    fn receipt_input(kv: Option<KvBlock>) -> ReceiptInput {
        ReceiptInput {
            provenance: provenance(),
            tokenization: TokenizationBlock::ClientTokenizer {
                tokenizer_sha256: "b".repeat(64),
                counts_special_tokens: false,
                counts_prompt_echo: false,
            },
            workload: Workload::W1,
            commit: "62d23d8d1".to_string(),
            bands: vec![
                synthetic_band(1),
                synthetic_band(4),
                synthetic_band(8),
                synthetic_band(16),
            ],
            kv,
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
        assert!(notes.contains("I-16"), "{notes}");
        assert!(notes.contains("Arm D `kv` block"), "{notes}");
    }

    /// With the server's own figures supplied, the block appears — and only then.
    #[test]
    fn the_kv_block_appears_only_when_the_server_reported_one() {
        let r = receipt_input(Some(KvBlock::from_server_report(50, 100, 0, 0)))
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
            last.token_times_ms = vec![window - 80.0, window - 20.0];
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
            last.token_times_ms = vec![window - 80.0, window - 20.0];
        }
        let r = input.render().expect("valid receipt");
        assert_eq!(r["bands"][0]["drain_ms"], serde_json::json!(0.0));
        assert_eq!(r["bands"][3]["drain_ms"], serde_json::json!(250.0));
        assert_eq!(r["drain_ms"], serde_json::json!(250.0));
    }

    /// A band whose requests never streamed says so; nothing is filled in.
    #[test]
    fn a_non_streaming_band_omits_the_latency_fields_and_names_them() {
        let plain = RequestOutcome {
            issued_ms: 0.0,
            settled_ms: 500.0,
            outcome: Outcome::Completed,
            generated_tokens: 128,
            ttft_ms: None,
            token_times_ms: Vec::new(),
        };
        let other = RequestOutcome {
            issued_ms: 600.0,
            settled_ms: 1_250.0,
            outcome: Outcome::Completed,
            generated_tokens: 128,
            ttft_ms: None,
            token_times_ms: Vec::new(),
        };
        let input = ReceiptInput {
            bands: vec![band(1, vec![plain, other])],
            ..receipt_input(None)
        };
        let r = input.render().expect("valid receipt");
        assert!(
            r["bands"][0].get("ttft_p95_ms").is_none(),
            "no invented p95"
        );
        let notes = r["unproduced_fields"].to_string();
        assert!(notes.contains("did not stream"), "{notes}");
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

    /// I-2 — a compute class the build cannot reach is a fabricated claim.
    #[test]
    fn a_compute_class_the_build_cannot_reach_is_refused() {
        let mut input = receipt_input(None);
        input.provenance.feature_set = vec!["inference".to_string()];
        let err = input.render().expect_err("cuda without the feature");
        assert!(err.contains("I-2"), "{err}");
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
    }
}
