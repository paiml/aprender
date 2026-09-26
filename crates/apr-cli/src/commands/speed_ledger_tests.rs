//! EXT-19 (aprender#4401): ledger schema, G3 coverage, and the speed ratchet
//! (FALSIFY-EXT-022).

use super::*;
use serde_json::json;

const H: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn block() -> Value {
    json!({
        "command": ["llama-bench", "-m", "m.gguf"],
        "version": "d1d3c3396",
        "env_sha256": H,
        "artifact_sha256": H,
        "log_path": "/tmp/llama.cpp.log",
        "started_utc": "2026-09-26T00:00:00.000Z",
        "finished_utc": "2026-09-26T00:00:01.000Z"
    })
}

fn measured(tag: &str, cell: &str, apr: f64, llama: f64) -> Value {
    json!({
        "tag": tag, "cell": cell,
        "outcome": {"measured": {
            "apr_decode_tok_s": apr,
            "receipt_sha256": H,
            "arms": [{"arm": "llama.cpp", "decode_tok_s": llama, "comparator": block()}]
        }}
    })
}

fn not_run(tag: &str, cell: &str, reason: &str) -> Value {
    json!({"tag": tag, "cell": cell, "outcome": {"not_run": {"reason": reason}}})
}

fn tags(n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("v0.71.{i}")).collect()
}

/// One measured row per tag on cell `c`, apr at `ratios[i]` × a 100 tok/s arm.
fn cell_series(ratios: &[f64]) -> Vec<Row> {
    tags(ratios.len())
        .iter()
        .zip(ratios)
        .map(|(t, r)| parse_row(&measured(t, "c", 100.0 * r, 100.0)).expect("row"))
        .collect()
}

#[test]
fn rows_round_trip_and_never_carry_a_ratio() {
    for v in [measured("v1", "c", 90.0, 100.0), not_run("v1", "d", "NoDeclaredExecutor")] {
        let row = parse_row(&v).expect("valid row");
        let back = serde_json::to_value(&row).expect("serialize");
        assert_eq!(back, v);
        // T28: the ratio is the ratchet's input only; no row can hold one.
        assert!(!back.to_string().contains("ratio"), "{back}");
    }
}

#[test]
fn t28_a_stored_ratio_is_refused() {
    let mut v = measured("v1", "c", 90.0, 100.0);
    v["outcome"]["measured"]["ratio"] = json!(0.9);
    assert!(parse_row(&v).is_err());
    let mut v = measured("v1", "c", 90.0, 100.0);
    v["ratio"] = json!(0.9);
    assert!(parse_row(&v).is_err());
}

#[test]
fn a_measured_row_without_the_reference_arm_is_refused() {
    let mut v = measured("v1", "c", 90.0, 100.0);
    v["outcome"]["measured"]["arms"][0]["arm"] = json!("ollama");
    let e = parse_row(&v).expect_err("no llama.cpp arm");
    assert!(e.contains("llama.cpp"), "{e}");
}

/// FALSIFY-EXT-020 carries through: an arm whose comparator block is
/// incomplete or malformed cannot enter the ledger.
#[test]
fn an_arm_with_a_bad_comparator_block_is_refused() {
    let mut v = measured("v1", "c", 90.0, 100.0);
    v["outcome"]["measured"]["arms"][0]["comparator"]["version"] = json!("  ");
    let e = parse_row(&v).expect_err("blank version");
    assert!(e.contains("FALSIFY-EXT-020"), "{e}");
    let mut v = measured("v1", "c", 90.0, 100.0);
    v["outcome"]["measured"]["arms"][0]["comparator"]["env_sha256"] = json!("abc");
    assert!(parse_row(&v).is_err());
}

#[test]
fn bad_speeds_hashes_and_reasons_are_refused() {
    for (apr, llama) in [(0.0, 100.0), (-1.0, 100.0), (90.0, 0.0)] {
        assert!(parse_row(&measured("v1", "c", apr, llama)).is_err(), "{apr} {llama}");
    }
    let mut v = measured("v1", "c", 90.0, 100.0);
    v["outcome"]["measured"]["receipt_sha256"] = json!(H.to_uppercase());
    assert!(parse_row(&v).is_err());
    assert!(parse_row(&not_run("v1", "c", " ")).is_err());
    assert!(parse_row(&not_run("", "c", "why")).is_err());
    let mut v = measured("v1", "c", 90.0, 100.0);
    let arm = v["outcome"]["measured"]["arms"][0].clone();
    v["outcome"]["measured"]["arms"] = json!([arm.clone(), arm]);
    assert!(parse_row(&v).expect_err("dup arm").contains("twice"));
}

#[test]
fn parse_ledger_names_the_bad_line() {
    let good = measured("v1", "c", 90.0, 100.0).to_string();
    let bad = not_run("v2", "c", "").to_string();
    assert_eq!(parse_ledger(&format!("{good}\n\n{good}\n")).expect("ok").len(), 2);
    let e = parse_ledger(&format!("{good}\n\n{bad}\n")).expect_err("bad line");
    assert!(e.starts_with("line 3:"), "{e}");
    assert!(parse_ledger("{not json").expect_err("json").starts_with("line 1:"));
}

/// G3: every (tag, cell) has a row; `not_run` covers its pair; a duplicate
/// pair is refused rather than letting a later row replace a verdict.
#[test]
fn g3_coverage_lists_holes_and_refuses_duplicates() {
    let t = tags(2);
    let cells = vec!["c".to_string(), "d".to_string()];
    let rows = parse_ledger(&format!(
        "{}\n{}\n{}\n",
        measured(&t[0], "c", 90.0, 100.0),
        not_run(&t[0], "d", "Refused(removed_by=0.71)"),
        measured(&t[1], "c", 90.0, 100.0),
    ))
    .expect("ledger");
    assert_eq!(
        uncovered(&rows, &t, &cells).expect("unique"),
        vec![(t[1].clone(), "d".to_string())]
    );
    let mut dup = rows.clone();
    dup.push(rows[0].clone());
    assert!(uncovered(&dup, &t, &cells).expect_err("dup").contains("two ledger rows"));
}

#[test]
fn the_ratchet_is_unarmed_until_three_records_precede_the_judged_one() {
    for n in 0..=ARM_AFTER {
        let rows = cell_series(&vec![0.9; n]);
        assert_eq!(ratchet(&rows, &tags(n), "c"), Ratchet::Unarmed { records: n });
    }
    let rows = cell_series(&[0.9; 4]);
    assert!(matches!(ratchet(&rows, &tags(4), "c"), Ratchet::Green { .. }));
}

/// FALSIFY-EXT-022: a planted sleep in the decode loop widens the gap vs the
/// llama.cpp arm, and the armed ratchet turns RED.
#[test]
fn falsify_ext_022_a_planted_sleep_turns_the_ratchet_red() {
    let rows = cell_series(&[0.90, 0.91, 0.90, 0.72]);
    match ratchet(&rows, &tags(4), "c") {
        Ratchet::Red { tag, ratio, floor } => {
            assert_eq!(tag, "v0.71.4");
            assert!((ratio - 0.72).abs() < 1e-9 && (floor - 0.90).abs() < 1e-9);
        }
        other => panic!("planted sleep not caught: {other:?}"),
    }
}

#[test]
fn noise_inside_the_tolerance_stays_green() {
    // 0.90 floor × 0.95 = 0.855: 0.86 is inside, 0.85 is not.
    let green = cell_series(&[0.90, 0.90, 0.90, 0.86]);
    assert!(matches!(ratchet(&green, &tags(4), "c"), Ratchet::Green { .. }));
    let red = cell_series(&[0.90, 0.90, 0.90, 0.85]);
    assert!(matches!(ratchet(&red, &tags(4), "c"), Ratchet::Red { .. }));
}

/// One lucky run does not raise the floor: the median ignores it.
#[test]
fn a_single_spike_does_not_raise_the_floor() {
    let rows = cell_series(&[0.90, 1.50, 0.90, 0.88]);
    assert!(matches!(ratchet(&rows, &tags(4), "c"), Ratchet::Green { floor, .. } if (floor - 0.90).abs() < 1e-9));
}

/// Shrink-only: a sustained improvement raises the floor, and falling back to
/// the old level afterwards is RED — the gap may not widen again.
#[test]
fn a_sustained_improvement_ratchets_the_floor_up_for_good() {
    let rows = cell_series(&[0.80, 0.80, 0.80, 0.95, 0.95, 0.95, 0.80, 0.80]);
    match ratchet(&rows, &tags(8), "c") {
        Ratchet::Red { floor, .. } => assert!((floor - 0.95).abs() < 1e-9, "{floor}"),
        other => panic!("gap widened back and stayed green: {other:?}"),
    }
}

/// `not_run` rows are coverage, not records: they neither arm nor judge.
#[test]
fn not_run_rows_do_not_count_as_records() {
    let t = tags(5);
    let mut rows = cell_series(&[0.90, 0.90, 0.90]);
    rows.push(parse_row(&not_run(&t[3], "c", "runner down")).expect("row"));
    assert_eq!(ratchet(&rows, &t[..4], "c"), Ratchet::Unarmed { records: 3 });
    rows.push(parse_row(&measured(&t[4], "c", 60.0, 100.0)).expect("row"));
    assert!(matches!(ratchet(&rows, &t, "c"), Ratchet::Red { .. }));
    // Another cell's records never judge this one.
    assert_eq!(ratchet(&rows, &t, "other"), Ratchet::Unarmed { records: 0 });
}
