// `json!` expands to an `unwrap` of an infallible `to_value`.
#![allow(clippy::disallowed_methods)]

use super::*;

fn row(lane: &str, diff: &str, input: u64, output: u64) -> String {
    json!({"lane": lane, "diff_sha256": diff, "trace_status": "ok",
           "tokens": {"input": input, "output": output}})
    .to_string()
}

fn jsonl(rows: &[String]) -> String {
    rows.iter().map(|r| format!("{r}\n")).collect()
}

fn lane<'a>(p: &'a Profile, name: &str) -> &'a LaneProfile {
    p.lanes
        .iter()
        .find(|l| l.lane == name)
        .unwrap_or_else(|| panic!("lane {name} missing from {:?}", p.lanes))
}

/// 20 rows of one lane: inputs 1000, 2000, …, 20000 and outputs 10, 20, …, 200.
fn ramp(name: &str) -> Vec<String> {
    (1..=20u64)
        .map(|i| row(name, &format!("d{i}"), 1000 * i, 10 * i))
        .collect()
}

/// FALSIFY-WLP-001: quantiles are nearest rank on measured rows; a failed
/// capture, an unparseable line, a zero input and a missing output are
/// excluded and counted, and move nothing.
#[test]
fn falsify_wlp_001_quantiles_are_nearest_rank_on_measured_rows_only() {
    let clean = ramp("qwen");
    let p = profile(&jsonl(&clean), 1);
    let q = lane(&p, "qwen");
    assert_eq!(q.n, 20);
    // nearest rank: p50 = rank 10, p95 = rank 19
    assert_eq!((q.input_p50, q.input_p95), (10_000.0, 19_000.0));
    assert_eq!((q.output_p50, q.output_p95), (100.0, 190.0));
    assert_eq!(p.excluded, 0);

    let mut dirty = clean.clone();
    dirty.push(
        json!({"lane": "qwen", "diff_sha256": "x1", "trace_status": "capture_failed",
               "tokens": {"input": 1, "output": 1}})
        .to_string(),
    );
    dirty.push("{not json".into());
    dirty.push(row("qwen", "x2", 0, 0));
    dirty.push(json!({"lane": "qwen", "diff_sha256": "x3", "tokens": {"input": 5}}).to_string());
    dirty.push(json!({"lane": "qwen", "diff_sha256": "x4"}).to_string());
    let d = profile(&jsonl(&dirty), 1);
    assert_eq!(d.excluded, 5);
    assert_eq!(lane(&d, "qwen"), q, "an excluded row moved the quantiles");
    assert_eq!(d.units, p.units);
}

/// FALSIFY-WLP-002: §0.2 is decided per lane on lanes with min_n rows.
#[test]
fn falsify_wlp_002_claim_is_decided_per_lane_and_refused_when_thin() {
    // ramp: input p50 10000 ≥ 2000, p95 19000 ≤ 32000, output p95 190 < 10000
    let mut rows = ramp("qwen");
    rows.extend(ramp("sonnet-5"));
    let p = profile(&jsonl(&rows), 20);
    assert!(lane(&p, "qwen").in_band() && lane(&p, "qwen").prefill_heavy());
    assert_eq!(p.claim, Some(true));
    assert!(p.refusals.is_empty(), "{:?}", p.refusals);

    // a decode-heavy lane: output p95 ≥ input p50 falsifies, naming the lane
    let mut chatty = rows.clone();
    chatty.extend((1..=20u64).map(|i| row("gemini", &format!("d{i}"), 4000, 5000)));
    let c = profile(&jsonl(&chatty), 20);
    assert!(lane(&c, "gemini").in_band() && !lane(&c, "gemini").prefill_heavy());
    assert_eq!(c.claim, Some(false));
    assert!(
        c.refusals.iter().any(|r| r.contains("gemini")),
        "{:?}",
        c.refusals
    );
    // the edge: output p95 == input p50 is not prefill-heavy
    let edge = profile(&jsonl(&[row("e", "d", 3000, 3000)]), 1);
    assert!(!lane(&edge, "e").prefill_heavy());
    assert_eq!(edge.claim, Some(false));

    // out of band, both ends; the bounds are inclusive
    for (input, band) in [(1999, false), (2000, true), (32_000, true), (32_001, false)] {
        let b = profile(&jsonl(&[row("b", "d", input, 10)]), 1);
        assert_eq!(lane(&b, "b").in_band(), band, "input {input}");
        assert_eq!(b.claim, Some(band), "input {input}");
    }

    // a thin decode-heavy lane does not decide
    let mut thin = rows.clone();
    thin.push(row("haiku", "d1", 3000, 9000));
    let t = profile(&jsonl(&thin), 20);
    assert_eq!(lane(&t, "haiku").n, 1);
    assert_eq!(t.claim, Some(true));

    // no lane at min_n: undecided, refused, never confirmed
    let u = profile(&jsonl(&rows), 21);
    assert_eq!(u.claim, None);
    assert!(
        u.refusals.iter().any(|r| r.contains("min_n")),
        "{:?}",
        u.refusals
    );
    let e = profile("", 1);
    assert_eq!(e.claim, None);
    assert!(e.lanes.is_empty());
}

/// FALSIFY-WLP-003: strata weights are over distinct diffs at their largest
/// lane measurement.
#[test]
fn falsify_wlp_003_strata_weights_are_over_distinct_diffs() {
    let rows = [
        // one diff, three lanes: sized by its largest measurement, 8001 → 16k
        row("qwen", "a", 7000, 10),
        row("sonnet-5", "a", 8001, 10),
        row("gemini", "a", 7500, 10),
        // inclusive bounds
        row("qwen", "b", 2000, 10),
        row("qwen", "c", 2001, 10),
        row("qwen", "d", 32_000, 10),
        // over the top stratum: counted, not weighted
        row("qwen", "e", 40_000, 10),
        row("sonnet-5", "e", 1000, 10),
    ];
    let p = profile(&jsonl(&rows), 1);
    assert_eq!(p.units, 5);
    assert_eq!(p.over_32k, 1);
    assert_eq!(
        p.strata,
        vec![(2000, 0.25), (8000, 0.25), (16000, 0.25), (32000, 0.25)]
    );
    let sum: f64 = p.strata.iter().map(|s| s.1).sum();
    assert!((sum - 1.0).abs() < 1e-12);

    // a lane count does not weight a diff
    let mut many = rows.to_vec();
    many.extend((0..10).map(|i| row(&format!("l{i}"), "b", 2000, 10)));
    assert_eq!(profile(&jsonl(&many), 1).strata, p.strata);

    // nothing in range: no weights, not NaN
    let o = profile(&jsonl(&[row("q", "z", 50_000, 10)]), 1);
    assert_eq!(o.over_32k, 1);
    assert!(o.strata.iter().all(|s| s.1 == 0.0), "{:?}", o.strata);
}

/// FALSIFY-WLP-004: the rendered block matches the v3 §7 receipt template.
#[test]
fn falsify_wlp_004_receipt_block_matches_the_v3_template() {
    let p = profile(&jsonl(&ramp("qwen")), 20);
    let v = render(&p);
    assert_eq!(
        v["workload"]["input_tokens_p50_p95"]["qwen"],
        json!([10_000.0, 19_000.0])
    );
    assert_eq!(
        v["workload"]["output_tokens_p50_p95"]["qwen"],
        json!([100.0, 190.0])
    );
    assert_eq!(v["workload"]["claim_0_2"], json!(true));
    assert_eq!(v["workload"]["excluded"], json!(0));
    assert_eq!(v["workload"]["strata"]["16k"], json!(0.4));
    let u = render(&profile(&jsonl(&ramp("qwen")), 99));
    assert_eq!(u["workload"]["claim_0_2"], Value::Null);
}
