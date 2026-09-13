//! §10's registered prediction: **the JOIN fixture**.
//!
//! > | JOIN fixture | reproduces `0.5341/0.2308/0.1685/0.0967` and
//! > `0.5873/0.9231/1.3525/1.5540` to four decimals with zero GPU | any digit
//! > differs in the new Rust |
//!
//! `evidence/parity-http/bands/{apr,llamacpp}-c{1,4,8,16}.json` are the eight
//! band files of the withdrawn 2026-08-25 lambda run. They are the only paired
//! parity data in the tree, and the ratios above are the ones the withdrawn
//! claim was built on. Reproducing them here — from the committed bytes, with
//! no GPU, in the code path that will carry the next real run — is what turns
//! "the join is implemented" into a falsifiable statement.
//!
//! Two halves, and the second is the point:
//!
//! - [`join_fixture_reproduces_parity_http_bands`] takes the eight files and
//!   gets the eight published digits back.
//! - [`join_fixture_is_refused_in_strict_mode`] takes the same files and shows
//!   that the join **refuses** them: the lanes' concurrency keys must match
//!   (PP-22), and the comparator lane's own `completion_tokens` is 112 where
//!   the subject's is 128, so `short_of_n_predict > 0` on every comparator band
//!   (PP-28). The digits above are therefore a reproduction of a
//!   `NONCONFORMANT-VALID` record, not a parity result — which is exactly what
//!   §2.1 says about that run.

use serde_json::Value;
use std::path::PathBuf;

use super::drain::{percentile, BandInput, ComparatorStatus, Outcome, RequestOutcome, StreamMode};
use super::join::JoinKey;
use super::receipt::{TokenCountingMethod, Workload};

/// The four bands the withdrawn run covered.
const BANDS: [u32; 4] = [1, 4, 8, 16];

/// §10's registered `agg` digits, in band order.
const AGG_RATIOS: [&str; 4] = ["0.5341", "0.2308", "0.1685", "0.0967"];

/// §10's registered `dec` digits, in band order.
const DEC_RATIOS: [&str; 4] = ["0.5873", "0.9231", "1.3525", "1.5540"];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("evidence")
        .join("parity-http")
        .join("bands")
}

fn load(lane: &str, concurrency: u32) -> Value {
    let path = fixture_dir().join(format!("{lane}-c{concurrency}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

/// The median of one field across a lane file's `runs[]`, through the same
/// `percentile` the band derivation uses — not a second implementation of it.
fn median_run_field(file: &Value, field: &str) -> f64 {
    let mut values: Vec<f64> = file["runs"]
        .as_array()
        .expect("runs[]")
        .iter()
        .map(|r| {
            r[field]
                .as_f64()
                .unwrap_or_else(|| panic!("{field} in a run"))
        })
        .collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&values, 0.50).expect("at least one run")
}

/// Rebuild a band from a lane file's first run's `request_details`.
///
/// The fixture records per-request latency, ttft and token counts but no issue
/// offsets, so requests are laid out back to back inside a 60 s window at the
/// file's own concurrency — enough for the PP-22 and PP-28 checks, which is all
/// this half of the test needs.
fn band_from_fixture(lane: &str, concurrency: u32) -> BandInput {
    let file = load(lane, concurrency);
    let run = &file["runs"][0];
    let details = run["request_details"].as_array().expect("request_details");
    let window_ms = 60_000.0;
    let step = window_ms / (details.len() as f64 + 1.0);
    let requests: Vec<RequestOutcome> = details
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let issued = i as f64 * step;
            let latency = d["latency_ms"].as_f64().expect("latency_ms");
            let ttft = d["ttft_ms"].as_f64().expect("ttft_ms");
            let tokens = d["completion_tokens"].as_u64().expect("completion_tokens") as u32;
            let prompt = d["prompt_tokens"].as_u64().expect("prompt_tokens") as u32;
            let itl = d["itl_ms"].as_f64().expect("itl_ms");
            let times: Vec<f64> = (0..tokens)
                .map(|k| issued + ttft + f64::from(k) * itl)
                .collect();
            RequestOutcome::new(issued, issued + latency, Outcome::Completed, tokens)
                .streamed(ttft, times)
                .server_prefill(prompt, ttft)
                .in_flight(concurrency)
        })
        .collect();
    BandInput::new(
        concurrency,
        window_ms,
        requests,
        ComparatorStatus::unmeasured("perf-gate", "fixture replay, not a live lane"),
    )
    .n_predict(128)
    .stream_mode(StreamMode::Live)
}

fn key(concurrency: u32) -> JoinKey {
    JoinKey {
        host: "lambda".to_string(),
        workload: Workload::W1,
        band: concurrency,
        model: "qwen2.5-coder-7b-apache-q4k-v1".to_string(),
        quant: "Q4_K_M".to_string(),
        tokenization: TokenCountingMethod::ServerUsage,
        window_ms: 60_000,
        replicates: 2,
        interleaved: false,
        n_ctx_slot: Some(1024),
        kv_type: Some("f16".to_string()),
        fa: Some(true),
        n_batch: Some(2048),
        // Both lanes were ISSUED n_predict=128. That the comparator's own
        // samples came back at 112 is a PP-28 finding about the run, not a
        // difference in the key it was issued under.
        n_predict: 128,
    }
}

/// §10's registered prediction, discharged: the eight committed band files give
/// back the eight published digits, with zero GPU.
#[test]
fn join_fixture_reproduces_parity_http_bands() {
    let mut agg = Vec::new();
    let mut dec = Vec::new();
    for c in BANDS {
        let subject = load("apr", c);
        let comparator = load("llamacpp", c);
        agg.push(format!(
            "{:.4}",
            median_run_field(&subject, "tokens_per_sec")
                / median_run_field(&comparator, "tokens_per_sec")
        ));
        dec.push(format!(
            "{:.4}",
            median_run_field(&subject, "decode_tok_per_sec")
                / median_run_field(&comparator, "decode_tok_per_sec")
        ));
    }
    assert_eq!(agg, AGG_RATIOS, "agg ratios at c={BANDS:?}");
    assert_eq!(dec, DEC_RATIOS, "dec ratios at c={BANDS:?}");

    // §2.2's mechanism rule, visible in the digits: `dec` RISES to 1.55x while
    // `agg` FALLS to 0.097x. Both at or above 1 on every band is refused as a
    // rule, and this is the other half of that — the two are not
    // interchangeable, and reading either alone inverts the verdict.
    assert!(dec[3].parse::<f64>().expect("f64") > 1.0);
    assert!(agg[3].parse::<f64>().expect("f64") < 0.2);
}

/// …and the refusal half. The same fixture does NOT produce a parity result:
/// the join refuses a band-key mismatch, and the comparator lane's own samples
/// stopped 16 tokens short of the pin.
#[test]
fn join_fixture_is_refused_in_strict_mode() {
    // PP-22: feeding the c=4 subject against the c=16 comparator is refused on
    // the concurrency key, however plausible the two numbers look together.
    let err = key(4)
        .refuse_mismatch(&key(16))
        .expect_err("c=4 against c=16");
    assert!(err.contains("band: 4 != 16"), "{err}");
    assert!(err.contains("PP-22"), "{err}");

    // …and matching bands do join, so the refusal above is about the key and
    // not about the fixture being unjoinable in principle.
    key(4)
        .refuse_mismatch(&key(4))
        .expect("matching bands join");

    // PP-28: the comparator lane ran to 112 completion tokens, not 128. Every
    // retained comparator sample is therefore short of the pin, which makes the
    // band NONCONFORMANT-VALID — a record, never a baseline.
    for c in BANDS {
        let comparator = band_from_fixture("llamacpp", c)
            .derive()
            .expect("the evidence still renders");
        assert_eq!(
            comparator.short_of_n_predict, comparator.completed,
            "every comparator sample at c={c} is short of n_predict=128"
        );
        assert!(!comparator.baseline_eligible(), "c={c}");

        let subject = band_from_fixture("apr", c)
            .derive()
            .expect("the evidence still renders");
        assert_eq!(
            subject.short_of_n_predict, 0,
            "the subject lane DID run to 128 at c={c} — the two lanes generated \
             different amounts of work, which is why their quotient is not a ratio"
        );
    }
}
