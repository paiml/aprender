//! Laya parity against its own oracle on the tiny fixture (plan 08-02), bars from
//! `contracts/laya-parity-v1.yaml` read at test time.

use super::{LayaError, QType};
use crate::test_support::{
    f32_b64, f32_list, fixture_apr, fixture_task, load_laya, max_abs, oracle, string_list,
    tolerance, u32_list, usize_list, within,
};
use crate::{DecisionMethod, Task};

fn qtype(v: &serde_json::Value) -> QType {
    let i = usize::try_from(v.as_u64().expect("qtype")).expect("qtype fits usize");
    QType::from_index(i).expect("known qtype")
}

/// FALSIFY-LAYA-PARITY-001 / -003 on the tiny fixture: tiny checkpoint -> in-memory
/// `.apr` -> `Laya::from_parts` on core ModernBERT -> built rows exact, logits and
/// probabilities within the contract bars, argmax exact, the truncation and
/// injection rows reproduced, and the `many` marker-loss question refused at load.
#[test]
fn tiny_parity() {
    let logits_bar = tolerance("logits_abs");
    let probs_bar = tolerance("probs_abs");
    let o = oracle();
    let apr = fixture_apr();
    let laya = load_laya(&apr, fixture_task()).expect("the fixture task loads");
    let b = laya.builder();
    let rows = o["rows"].as_array().expect("oracle rows");
    assert!(!rows.is_empty(), "oracle has rows");

    let (mut worst_logit, mut worst_prob) = (0.0f64, 0.0f64);
    let (mut saw_truncated, mut saw_injection, mut saw_shrunk) = (false, false, false);
    let mut team_probs: Vec<(String, Vec<f32>)> = Vec::new();
    for (ri, r) in rows.iter().enumerate() {
        let state = r["state"].as_str().expect("state");
        let t = r["t"].as_str().expect("t");
        let options = string_list(&r["options"]);
        let built = b
            .build(state, t, r["ins"].as_str().expect("ins"), &options)
            .expect("row builds");
        assert_eq!(built.ids, u32_list(&r["ids"]), "row {ri}: ids exact");
        assert_eq!(
            built.markers,
            usize_list(&r["markers"]),
            "row {ri}: markers exact"
        );
        assert_eq!(built.tokens, built.ids.len());
        let want_trunc = r["truncated"].as_bool().expect("truncated");
        assert_eq!(built.truncated, want_trunc, "row {ri}: truncated flag");
        saw_truncated |= want_trunc;
        saw_injection |= state.contains("[SEP]") && state.contains("[MASK]");
        saw_shrunk |= r["options_shrunk"].as_bool().expect("options_shrunk");

        let q = qtype(&r["qtype"]);
        let mut blocks: Vec<(String, Vec<f32>)> = Vec::new();
        let z = laya
            .forward_row(&built.ids, &built.markers, q, |n, x| {
                blocks.push((n.to_string(), x.to_vec()));
            })
            .expect("forward");
        let ladder: Vec<String> = ["final", "head0", "head1"]
            .iter()
            .map(|n| {
                let got = &blocks.iter().find(|(bn, _)| bn == n).expect("tapped").1;
                format!("{n}={:.2e}", max_abs(got, &f32_b64(&r["ladder"][*n])))
            })
            .collect();
        let m = &blocks
            .iter()
            .find(|(bn, _)| bn == "m_opts")
            .expect("m_opts")
            .1;
        let dm = max_abs(m, &f32_b64(&r["m_opts"]));

        let dz = max_abs(&z, &f32_list(&r["logits"]));
        let temp = super::temperature::temperature_for(
            laya.agent_config(),
            q,
            usize::try_from(r["k"].as_u64().expect("k")).expect("k fits"),
        );
        assert_eq!(
            f64::from(temp),
            f64::from(r["temperature"].as_f64().expect("temperature") as f32),
            "row {ri}: bucket temperature"
        );
        let p = super::temperature::softmax_t(&z, temp);
        let dp = max_abs(&p, &f32_list(&r["probabilities"]));
        println!(
            "row {ri} ({}): tokens {} {} m_opts={dm:.2e} logits={dz:.2e} probs={dp:.2e} ARCH={}",
            r["qid"].as_str().unwrap_or("?"),
            built.tokens,
            ladder.join(" "),
            std::env::consts::ARCH
        );
        assert!(
            within(dz, logits_bar),
            "row {ri}: logits max|d| {dz} > {logits_bar}"
        );
        assert!(
            within(dp, probs_bar),
            "row {ri}: probs max|d| {dp} > {probs_bar}"
        );
        let want_arg = usize::try_from(r["argmax"].as_u64().expect("argmax")).expect("fits");
        assert_eq!(super::argmax(&p), want_arg, "row {ri}: argmax exact");
        worst_logit = worst_logit.max(dz);
        worst_prob = worst_prob.max(dp);
        if r["qid"] == "team" {
            team_probs.push((state.to_string(), p));
        }
    }
    assert!(saw_truncated, "the oracle exercises the over-window row");
    assert!(
        saw_injection,
        "the oracle exercises the [MASK]/[SEP] injection row"
    );
    assert!(saw_shrunk, "the oracle exercises the option shrink");
    println!(
        "tiny_parity: {} rows, max|d| logits {worst_logit:.3e} (bar {logits_bar:e}), \
         probs {worst_prob:.3e} (bar {probs_bar:e}), ARCH={}",
        rows.len(),
        std::env::consts::ARCH
    );

    // The task path: classify(texts) through the seam equals the per-row results.
    let texts: Vec<String> = team_probs.iter().map(|(s, _)| s.clone()).collect();
    let decisions = laya.classify(&texts).expect("classify");
    assert_eq!(decisions.len(), texts.len());
    for ((_, p), d) in team_probs.iter().zip(&decisions) {
        assert_eq!(
            &d.probabilities, p,
            "classify equals the per-row probabilities"
        );
        assert_eq!(d.label_index, super::argmax(p));
    }
    assert!(
        decisions.iter().any(|d| d.truncated),
        "the over-window team row is served truncated (D-12)"
    );

    // The `many` question: Laya's ids and SHORTER marker list reproduced, and the
    // 16-criteria task refused at load with the oracle's marker count.
    let ml = &o["marker_loss"];
    let built = b
        .build(
            ml["state"].as_str().expect("state"),
            ml["t"].as_str().expect("t"),
            ml["ins"].as_str().expect("ins"),
            &string_list(&ml["options"]),
        )
        .expect("many builds");
    let want_markers = usize_list(&ml["markers"]);
    assert_eq!(built.ids, u32_list(&ml["ids"]), "many: ids exact");
    assert_eq!(built.markers, want_markers, "many: markers exact");
    let criteria = usize::try_from(ml["criteria"].as_u64().expect("criteria")).expect("fits");
    assert!(
        want_markers.len() < criteria,
        "the oracle recorded a marker loss"
    );
    let many = Task::from_slice(
        &serde_json::to_vec(&ml["question"]).expect("serialise the many question"),
    )
    .expect("many task parses");
    assert_eq!(many.criteria().len(), criteria);
    let err = load_laya(&apr, many).expect_err("marker loss refused at load");
    assert_eq!(
        err,
        LayaError::MarkersLost {
            criteria,
            markers: want_markers.len()
        }
    );
}

/// Review finding (FALSIFY-DECIDE-APR-010, served half): the fixture task's options
/// tokenize far beyond head_max_len, yet it LOADS, keeps all 3 markers and builds
/// Laya's own ids on every shrink row.
#[test]
fn shrunk_task_is_served() {
    let o = oracle();
    let task = fixture_task();
    let laya = load_laya(&fixture_apr(), task.clone()).expect("a shrunk task loads");
    let b = laya.builder();
    let options = task.render_options();
    let raw: usize = options
        .iter()
        .map(|opt| {
            let empty = b
                .build("", "choice", "q", std::slice::from_ref(opt))
                .expect("one");
            empty.ids.len() - empty.markers[0] - 2
        })
        .sum();
    println!(
        "rendered options: {raw} tokens against head_max_len {}",
        b.head_max_len()
    );
    assert!(
        raw > b.head_max_len(),
        "the options really exceed head_max_len"
    );
    let rows: Vec<&serde_json::Value> = o["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .filter(|r| r["qid"] == "team" && r["options_shrunk"] == true)
        .collect();
    assert!(!rows.is_empty(), "oracle has shrink rows for the task");
    for r in rows {
        let prepared = laya
            .prepare(&[r["state"].as_str().expect("state").to_string()])
            .expect("prepare");
        assert_eq!(prepared[0].markers().len(), 3, "all markers kept");
        assert_eq!(
            prepared[0].ids(),
            u32_list(&r["ids"]).as_slice(),
            "Laya's ids"
        );
        assert_eq!(prepared[0].markers(), usize_list(&r["markers"]).as_slice());
    }
}

/// A JSON string literal for `s`.
fn quote(s: &str) -> String {
    serde_json::to_string(s).expect("a str serialises")
}

/// FALSIFY-DECIDE-APR-010, refused half: the 16-criteria task loses markers at
/// max_len 64 and is refused at load with the oracle's marker count.
#[test]
fn marker_loss_is_refused_at_load() {
    let ml = &oracle()["marker_loss"];
    let names = string_list(&ml["options"]);
    let criteria: Vec<String> = names
        .iter()
        .map(|o| {
            let (name, desc) = o.split_once(": ").expect("name: description");
            format!("{}:{}", quote(name), quote(desc))
        })
        .collect();
    let doc = format!(
        r#"{{"type":"choice","instructions":{},"criteria":{{{}}}}}"#,
        quote(ml["ins"].as_str().expect("ins")),
        criteria.join(",")
    );
    let task = Task::from_slice(doc.as_bytes()).expect("many parses");
    assert_eq!(
        task.render_options(),
        names,
        "document order reproduces Laya's options"
    );
    let err = load_laya(&fixture_apr(), task).expect_err("refused");
    assert_eq!(
        err,
        LayaError::MarkersLost {
            criteria: 16,
            markers: usize_list(&ml["markers"]).len()
        }
    );
}

/// A row whose marker count is not the task's is refused, never scored.
#[test]
fn foreign_rows_are_refused() {
    let laya = load_laya(&fixture_apr(), fixture_task()).expect("loads");
    let mut rows = laya.prepare(&["hello".to_string()]).expect("prepare");
    rows[0].markers.pop();
    let e = laya.classify_prepared(&rows).expect_err("refused");
    assert_eq!(
        e,
        crate::DecideError::Laya(LayaError::RowMarkerCount {
            expected: 3,
            observed: 2
        })
    );
    let ids = rows[0].ids().to_vec();
    let e = laya
        .forward_row(&ids, &[ids.len()], QType::Choice, |_, _| {})
        .expect_err("out of range");
    assert_eq!(
        e,
        crate::DecideError::Laya(LayaError::MarkerOutOfRange {
            marker: ids.len(),
            tokens: ids.len()
        })
    );
}

/// V9-a / T-08-26-03: a tokenizer.json that declares truncation (max_length 4) and fixed
/// padding builds EXACTLY the row the same tokenizer without those blocks builds, for a
/// state far longer than 4 tokens. HF transformers passes truncation / padding per call,
/// so a file's own block never silently cuts the state while the row reports its own
/// `truncated` flag.
#[test]
fn tokenizer_truncation_and_padding_are_disabled() {
    use super::builder::Builder;
    let plain = crate::test_support::read("checkpoint/tokenizer/tokenizer.json");
    let mut doc: serde_json::Value = serde_json::from_slice(&plain).expect("tokenizer JSON");
    assert!(
        doc["truncation"].is_null() && doc["padding"].is_null(),
        "the fixture tokenizer declares neither block"
    );
    // (serde_json's `json!` object form expands to `unwrap`, which this workspace bans.)
    doc["truncation"] = serde_json::from_str(
        r#"{"direction": "Right", "max_length": 4, "strategy": "LongestFirst", "stride": 0}"#,
    )
    .expect("truncation block");
    doc["padding"] = serde_json::from_str(
        r#"{"strategy": {"Fixed": 40}, "direction": "Right", "pad_to_multiple_of": null,
            "pad_id": 0, "pad_type_id": 0, "pad_token": "[PAD]"}"#,
    )
    .expect("padding block");
    let injected = serde_json::to_vec(&doc).expect("serialise the injected tokenizer");
    let (max_len, head_max_len) = (64, 32);
    let a = Builder::from_bytes(&plain, max_len, head_max_len).expect("plain tokenizer");
    let b = Builder::from_bytes(&injected, max_len, head_max_len).expect("injected tokenizer");
    let options = vec!["alpha option".to_string(), "beta option".to_string()];
    for state in [
        "a short state",
        "a state that is much longer than four tokens and still well inside the row",
        &"word ".repeat(200),
    ] {
        let want = a
            .build(state, "choice", "question", &options)
            .expect("plain row");
        let got = b
            .build(state, "choice", "question", &options)
            .expect("injected row");
        assert!(
            want.tokens > 4,
            "the state is longer than the injected max_length"
        );
        assert_eq!(
            got, want,
            "the tokenizer file's truncation / padding leaked into the row"
        );
    }
}

/// IN-01 (plan 08-31): `argmax`'s "NaN never wins" doc is true. Before the fix the
/// fold seeded index 0, so `[NaN, 0.9]` returned 0 (every `>` against NaN is false).
#[test]
fn argmax_nan_never_wins() {
    let nan = f32::NAN;
    assert_eq!(
        super::argmax(&[nan, 0.9_f32]),
        1,
        "a NaN at index 0 must not win"
    );
    assert_eq!(
        super::argmax(&[0.2_f32, nan, 0.9]),
        2,
        "a NaN in the middle is skipped"
    );
    assert_eq!(
        super::argmax(&[nan, nan, 0.1_f32, 0.7, nan]),
        3,
        "leading and trailing NaN"
    );
    assert_eq!(
        super::argmax(&[0.5_f32, 0.5, 0.1]),
        0,
        "ties keep the first index"
    );
    assert_eq!(
        super::argmax(&[f64::NAN, 0.4_f64, 0.4]),
        1,
        "ties after a NaN keep the first real index"
    );
    assert_eq!(
        super::argmax(&[nan, nan]),
        0,
        "all-NaN returns 0, as before"
    );
    assert_eq!(super::argmax::<f32>(&[]), 0, "empty returns 0, as before");
    assert_eq!(
        super::argmax(&[0.1_f32, 0.3, 0.2]),
        1,
        "the ordinary case is unchanged"
    );
}
