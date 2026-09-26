//! EXT-19 (aprender#4401): the release-phase speed gate over the ledger
//! (G3 coverage + FALSIFY-EXT-022), and T28 on its printed report.

use super::*;
use serde_json::json;

const H: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn row(tag: &str, cell: &str, apr: f64) -> String {
    json!({
        "tag": tag, "cell": cell,
        "outcome": {"measured": {
            "apr_decode_tok_s": apr,
            "receipt_sha256": H,
            "arms": [{"arm": "llama.cpp", "decode_tok_s": 100.0, "comparator": {
                "command": ["llama-bench", "-m", "m.gguf"],
                "version": "d1d3c3396",
                "env_sha256": H,
                "artifact_sha256": H,
                "log_path": "/tmp/llama.cpp.log",
                "started_utc": "2026-09-26T00:00:00.000Z",
                "finished_utc": "2026-09-26T00:00:01.000Z"
            }}]
        }}
    })
    .to_string()
}

fn not_run(tag: &str, cell: &str) -> String {
    json!({"tag": tag, "cell": cell, "outcome": {"not_run": {"reason": "runner down"}}}).to_string()
}

fn tags(n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("v0.71.{i}")).collect()
}

fn cells() -> Vec<String> {
    vec!["gx10-cuda".into(), "lambda-cpu".into()]
}

/// Two cells over `apr` tok/s per tag (the arm is 100 tok/s, so apr/100 is
/// the ratio); `lambda-cpu` is `not_run` on every tag.
fn ledger(apr: &[f64]) -> String {
    let t = tags(apr.len());
    t.iter()
        .zip(apr)
        .flat_map(|(t, a)| [row(t, "gx10-cuda", *a), not_run(t, "lambda-cpu")])
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_full_ledger_at_its_floor_passes() {
    let r = gate(&ledger(&[90.0, 91.0, 90.0, 89.0]), &tags(4), &cells()).expect("gate");
    assert!(r.holes.is_empty(), "{:?}", r.holes);
    assert!(r.passed(), "{}", r.render());
    assert!(r.render().ends_with("speed gate: PASS\n"));
}

/// The gate half of FALSIFY-EXT-022: a ledger whose newest tag drops well
/// under the floor (what a planted decode-loop sleep produces) fails the GATE,
/// not just the ratchet. The end-to-end half — the sleep measured against the
/// llama.cpp arm — is owed on EXT-28 (#4410).
#[test]
fn a_planted_slowdown_in_the_ledger_fails_the_gate() {
    let r = gate(&ledger(&[90.0, 91.0, 90.0, 72.0]), &tags(4), &cells()).expect("gate");
    assert!(!r.passed(), "planted sleep passed the gate:\n{}", r.render());
    let out = r.render();
    assert!(out.contains("RED      gx10-cuda @ v0.71.4"), "{out}");
    assert!(out.ends_with("speed gate: FAIL\n"), "{out}");
}

/// G3: a tag with no row for a cell is a hole, and a hole fails the gate
/// even when every measured cell is green.
#[test]
fn g3_a_missing_pair_fails_the_gate() {
    let mut l = ledger(&[90.0, 90.0, 90.0, 90.0]);
    l = l.replace(&not_run("v0.71.2", "lambda-cpu"), "");
    let r = gate(&l, &tags(4), &cells()).expect("gate");
    assert_eq!(r.holes, vec![("v0.71.2".to_string(), "lambda-cpu".to_string())]);
    assert!(!r.passed());
    let out = r.render();
    assert!(out.contains("G3 coverage: 7/8"), "{out}");
    assert!(out.contains("HOLE     lambda-cpu @ v0.71.2"), "{out}");
}

/// An unarmed cell has nothing to be held to; it neither fails nor hides a
/// hole elsewhere.
#[test]
fn an_unarmed_cell_passes_and_says_so() {
    let r = gate(&ledger(&[10.0, 90.0]), &tags(2), &cells()).expect("gate");
    assert!(r.passed(), "{}", r.render());
    assert!(r.render().contains("UNARMED  gx10-cuda: 2 measured"), "{}", r.render());
}

/// T28: the printed report never carries a comparator ratio or the floor —
/// not as a decimal, not as a percentage, not under the word `ratio`.
#[test]
fn t28_the_report_never_prints_a_ratio_or_floor() {
    for apr in [[90.0, 91.0, 90.0, 72.0], [90.0, 91.0, 90.0, 89.0]] {
        let r = gate(&ledger(&apr), &tags(4), &cells()).expect("gate");
        let out = r.render();
        for needle in ["0.9", "0.72", "0.89", "0.91", "72%", "90%", "89%", "ratio", "floor ="] {
            assert!(!out.contains(needle), "report leaks `{needle}`:\n{out}");
        }
    }
}

#[test]
fn a_bad_ledger_or_empty_scope_is_an_error_not_a_verdict() {
    assert!(gate("{not json", &tags(1), &cells()).is_err());
    let dup = format!("{}\n{}", row("v0.71.1", "gx10-cuda", 90.0), row("v0.71.1", "gx10-cuda", 90.0));
    assert!(gate(&dup, &tags(1), &cells()).expect_err("dup").contains("two ledger rows"));
    assert!(gate("", &[], &cells()).is_err());
    assert!(gate("", &tags(1), &[]).is_err());
}
