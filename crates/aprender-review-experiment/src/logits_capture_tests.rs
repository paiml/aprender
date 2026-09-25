use serde_json::{json, Value};

use super::*;
use crate::sparse_logits::{decode, Mode, DEFAULT_K};

const SHA: &str = "70dffe5625e1849661303b82bb1fdf5254ab1d4560e2fe3eacc147e5d2588bba";

fn meta() -> Meta {
    Meta {
        model_sha256: SHA.into(),
        tokenizer_sha256: SHA.into(),
        apr_tag: "v0.70.0".into(),
        backend: "lambda-cpu".into(),
        temperature_of_record: 0.0,
    }
}

/// A deterministic vocab row bigger than 2^16, as the codec's tests use.
fn row(seed: u32, vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            f32::from((x >> 20) as u16) / 256.0
        })
        .collect()
}

/// #4026's `top_k_logprobs`, restated: f32 log-sum-exp, descending logit,
/// lower id on ties.
fn run_top(logits: &[f32], k: usize) -> Vec<Top> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse: f32 = logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32, x))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(k);
    ranked
        .into_iter()
        .map(|(token_id, logit)| Top {
            token_id,
            logit,
            logprob: logit - max - lse,
        })
        .collect()
}

fn rows(n: u32, vocab: usize) -> Vec<Vec<f32>> {
    (0..n).map(|s| row(s, vocab)).collect()
}

/// The `logprobs` value an `apr run --json --logprobs k` would print.
fn run_json(rows: &[Vec<f32>], k: usize) -> Value {
    let steps: Vec<Step> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let top = run_top(r, k);
            Step {
                step: i,
                chosen: top[i % k].token_id,
                top,
            }
        })
        .collect();
    serde_json::to_value(RunLogprobs {
        top_k: k,
        prompt_token_ids: vec![151_644, 872, 198],
        steps,
    })
    .expect("ser")
}

fn blob(v: &Value) -> Result<(Vec<u8>, String), String> {
    capture_blob(v, &meta())
}

/// FALSIFY-SLC-001: header and residual mass round-trip. The decoded header
/// is the run's, each token's `1 − M` is the mass outside the recorded top-k,
/// `logsumexp_full` is `logit − logprob`, and `logits_sha` is the blob's sha.
#[test]
fn falsify_slc_001_header_and_residual_mass_round_trip() {
    let rs = rows(6, 70_000);
    let v = run_json(&rs, usize::from(DEFAULT_K));
    let (bytes, sha) = blob(&v).expect("captures");
    assert_eq!(sha, crate::prereg::sha256_hex(&bytes));
    let (h, toks) = decode(&bytes).expect("decodes");
    assert_eq!(h.schema, crate::sparse_logits::SCHEME);
    assert_eq!(h.mode, Mode::Topk);
    assert_eq!(h.k, DEFAULT_K);
    assert_eq!(h.n_tokens, 6);
    assert_eq!(h.model_sha256, SHA);
    assert_eq!(h.backend, "lambda-cpu");
    assert_eq!(h.apr_tag, "v0.70.0");
    let run: RunLogprobs = serde_json::from_value(v).expect("de");
    for (t, s) in toks.iter().zip(&run.steps) {
        assert_eq!(t.token_id_sampled, s.chosen);
        assert_eq!(t.ids, s.top.iter().map(|x| x.token_id).collect::<Vec<_>>());
        let kept: f64 = s.top.iter().map(|x| f64::from(x.logprob).exp()).sum();
        assert!(
            (f64::from(t.residual_mass()) - (1.0 - kept)).abs() < 4e-3,
            "residual {} vs {}",
            t.residual_mass(),
            1.0 - kept
        );
        assert!(
            t.residual_mass() > 0.5,
            "a 70k vocab keeps most mass in the tail"
        );
        let lse = s.top[0].logit - s.top[0].logprob;
        assert!(
            (t.logsumexp_full - lse).abs() < 1e-4,
            "{} vs {lse}",
            t.logsumexp_full
        );
    }
    // Every recorded token of a full-vocab run: nothing is left in the tail.
    let (full, _) = blob(&run_json(&rows(2, 40), 40)).expect("captures");
    let (_, ft) = decode(&full).expect("decodes");
    assert!(ft.iter().all(|t| t.residual_mass() < 4e-3), "{ft:?}");
    // The sha names these bytes: another run, another sha.
    let (_, other) = blob(&run_json(&rows(7, 70_000), usize::from(DEFAULT_K))).expect("captures");
    assert_ne!(sha, other);
}

/// FALSIFY-SLC-002: the capture of a run's top-k is the codec's own
/// `from_full_logits` of the logits it came from.
#[test]
fn falsify_slc_002_capture_matches_the_full_logit_row() {
    let rs = rows(4, 70_000);
    let v = run_json(&rs, usize::from(DEFAULT_K));
    let run: RunLogprobs = serde_json::from_value(v).expect("de");
    let (_, toks) = capture(&run, &meta()).expect("captures");
    for ((t, r), s) in toks.iter().zip(&rs).zip(&run.steps) {
        let want = TokenLogits::from_full_logits(r, DEFAULT_K, s.chosen).expect("row");
        assert_eq!(t.ids, want.ids);
        assert!((t.logsumexp_full - want.logsumexp_full).abs() < 1e-3);
        for (a, b) in t.logprobs.iter().zip(&want.logprobs) {
            assert!((a.to_f32() - b.to_f32()).abs() < 2e-3, "{a} vs {b}");
        }
        assert!((t.topk_mass.to_f32() - want.topk_mass.to_f32()).abs() < 4e-3);
    }
}

fn refused(what: &str, edit: impl FnOnce(&mut Value)) {
    let mut v = run_json(&rows(3, 5_000), 8);
    edit(&mut v);
    assert!(blob(&v).is_err(), "{what} was captured: {v}");
}

/// FALSIFY-SLC-003: a row the codec could not honestly carry is refused,
/// never repaired.
#[test]
fn falsify_slc_003_a_bad_run_is_refused() {
    assert!(
        blob(&run_json(&rows(3, 5_000), 8)).is_ok(),
        "the control captures"
    );
    refused("null (no --logprobs)", |v| *v = Value::Null);
    refused("no steps", |v| v["steps"] = json!([]));
    refused("top_k 0", |v| v["top_k"] = json!(0));
    refused("top_k 256", |v| v["top_k"] = json!(256));
    refused("an unknown key", |v| v["extra"] = json!(1));
    refused("a missing key", |v| {
        v.as_object_mut().expect("obj").remove("prompt_token_ids");
    });
    refused("a null logit", |v| {
        v["steps"][0]["top"][2]["logit"] = Value::Null
    });
    refused("a step out of order", |v| v["steps"][1]["step"] = json!(2));
    refused("a short top list", |v| {
        v["steps"][2]["top"].as_array_mut().expect("arr").pop();
    });
    refused("a repeated id", |v| {
        v["steps"][0]["top"][3] = v["steps"][0]["top"][2].clone();
    });
    refused("an ascending pair", |v| {
        let a = v["steps"][1]["top"][0].clone();
        v["steps"][1]["top"][0] = v["steps"][1]["top"][1].clone();
        v["steps"][1]["top"][1] = a;
    });
    refused("two distributions in one step", |v| {
        let l = v["steps"][0]["top"][5]["logit"].as_f64().expect("f");
        v["steps"][0]["top"][5]["logit"] = json!(l + 0.01);
    });
    refused("a positive logprob", |v| {
        for t in v["steps"][0]["top"].as_array_mut().expect("arr") {
            let (l, p) = (
                t["logit"].as_f64().expect("f"),
                t["logprob"].as_f64().expect("f"),
            );
            t["logit"] = json!(l + 20.0);
            t["logprob"] = json!(p + 20.0);
        }
    });
    let mut m = meta();
    m.model_sha256 = "not-a-sha".into();
    assert!(capture_blob(&run_json(&rows(1, 100), 4), &m).is_err());
}
