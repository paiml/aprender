//! Falsifiers for the OBS-12 probe-decision report. Each planted input names the verdict
//! it must produce; a generator that always answered "not indicated" fails several.

// `serde_json::json!` expands to an internal `unwrap`; the fixtures below are literals.
#![allow(clippy::disallowed_methods)]

use super::*;
use serde_json::json;

const HOST: &str = "lambda-vector";
const BACKEND: &str = "cuda";
const BIN_A: &str = "aaaa";
const BIN_B: &str = "bbbb";

fn night(day: u32) -> String {
    format!("2026-10-{day:02}")
}

fn ident(schema: &str, ts: &str, backend: &str, binary: &str) -> Value {
    json!({
        "schema": schema, "ts": ts, "host": HOST, "apr_version": "0.70.0",
        "apr_tag": "v0.70.0", "crate_tarball_sha256": "c0ffee", "binary_sha256": binary,
        "build_identity": "0.70.0+abc", "model_id": "qwen3.5-4b", "model_sha256": "5eed",
        "backend": backend, "gpu_proof": {"device": "RTX 4090"}, "request_id": format!("r-{ts}"),
    })
}

/// One nightly row per day at 02:00Z; `binary(day)` picks the binary.
fn ledger(days: std::ops::RangeInclusive<u32>, binary: impl Fn(u32) -> &'static str) -> String {
    days.map(|d| {
        ident(
            LEDGER_SCHEMA,
            &format!("{}T02:00:00Z", night(d)),
            BACKEND,
            binary(d),
        )
        .to_string()
    })
    .collect::<Vec<_>>()
    .join("\n")
}

/// One OBS-06 verdict per day; `reds` are (day, rule).
fn verdicts(days: std::ops::RangeInclusive<u32>, reds: &[(u32, &str)]) -> String {
    days.map(|d| {
        let red: Vec<Value> = reds
            .iter()
            .filter(|(rd, _)| *rd == d)
            .map(|(_, rule)| json!({"rule": rule, "metric": "decode", "detail": "", "host": HOST, "backend": BACKEND}))
            .collect();
        json!({
            "schema": RED_SCHEMA, "night": night(d),
            "verdict": if red.is_empty() { "GREEN" } else { "RED" },
            "series": [{"host": HOST, "backend": BACKEND, "status": "present"}],
            "red": red,
        })
        .to_string()
    })
    .collect::<Vec<_>>()
    .join("\n")
}

#[test]
fn empty_inputs_are_refused_never_read_as_no_red() {
    let r = build_report("", "", "");
    assert_eq!(r.status, Status::Refused);
    assert_eq!(r.recommendation, Recommendation::NoDecision);
    assert!(
        r.refusals.iter().any(|x| x.contains("R-2")),
        "{:?}",
        r.refusals
    );
}

#[test]
fn verdicts_without_ledger_rows_are_refused() {
    let r = build_report("", &verdicts(1..=14, &[]), "");
    assert_eq!(r.status, Status::Refused);
}

#[test]
fn thirteen_nights_is_insufficient() {
    let r = build_report(&ledger(1..=13, |_| BIN_A), &verdicts(1..=13, &[]), "");
    assert_eq!(r.status, Status::InsufficientNights);
    assert_eq!(r.recommendation, Recommendation::NoDecision);
    assert_eq!(r.series[0].nights, 13);
}

#[test]
fn fourteen_green_nights_is_not_indicated() {
    let r = build_report(&ledger(1..=14, |_| BIN_A), &verdicts(1..=14, &[]), "");
    assert_eq!(r.status, Status::Ready);
    assert_eq!(r.recommendation, Recommendation::ProbeNotIndicated);
    assert_eq!(r.window, Some((night(1), night(14))));
}

#[test]
fn a_red_without_lane_evidence_is_inconclusive_not_proven() {
    // The nightly cannot say where inside 24 h the regression began.
    let r = build_report(
        &ledger(1..=14, |_| BIN_A),
        &verdicts(1..=14, &[(10, "rolling_2theta")]),
        "",
    );
    assert_eq!(r.recommendation, Recommendation::Inconclusive);
    let e = &r.events[0];
    assert_eq!(e.onset_basis, OnsetBasis::FirstMeasured);
    assert!(e.proven_h.abs() < f64::EPSILON);
    assert_eq!(e.max_h, Some(24.0));
}

#[test]
fn binary_first_seen_20h_before_the_nightly_is_probe_indicated() {
    // New binary B from night 10; lane rows show it serving at 06:00Z the day before.
    let lane = ident(LANE_SCHEMA, "2026-10-09T06:00:00Z", BACKEND, BIN_B).to_string();
    let r = build_report(
        &ledger(1..=14, |d| if d >= 10 { BIN_B } else { BIN_A }),
        &verdicts(1..=14, &[(10, "rolling_2theta")]),
        &lane,
    );
    assert_eq!(r.status, Status::Ready);
    assert_eq!(r.recommendation, Recommendation::ProbeIndicated);
    let e = &r.events[0];
    assert!(e.binary_changed);
    assert_eq!(e.onset_basis, OnsetBasis::BinaryFirstSeen);
    assert!((e.proven_h - 20.0).abs() < 1e-9, "{}", e.proven_h);
}

#[test]
fn binary_first_seen_6h_before_is_not_proven_over_limit() {
    let lane = ident(LANE_SCHEMA, "2026-10-09T20:00:00Z", BACKEND, BIN_B).to_string();
    let r = build_report(
        &ledger(1..=14, |d| if d >= 10 { BIN_B } else { BIN_A }),
        &verdicts(1..=14, &[(10, "rolling_2theta")]),
        &lane,
    );
    assert_eq!(r.summary.proven_over_limit, 0);
    assert_eq!(r.recommendation, Recommendation::Inconclusive);
    assert!((r.events[0].proven_h - 6.0).abs() < 1e-9);
}

#[test]
fn two_consecutive_rule_delay_is_reported_not_counted() {
    // Bad from night 9, flagged on night 10: the 24 h rule delay is not a nightly gap.
    let lane = ident(LANE_SCHEMA, "2026-10-08T22:00:00Z", BACKEND, BIN_B).to_string();
    let r = build_report(
        &ledger(1..=14, |d| if d >= 9 { BIN_B } else { BIN_A }),
        &verdicts(1..=14, &[(10, "rolling_2consecutive")]),
        &lane,
    );
    let e = &r.events[0];
    assert!((e.rule_delay_h - 24.0).abs() < 1e-9);
    assert!((e.proven_h - 4.0).abs() < 1e-9, "{}", e.proven_h);
    assert_eq!(r.summary.proven_over_limit, 0);
}

#[test]
fn a_gpu_row_without_gpu_proof_is_excluded() {
    let mut rows: Vec<String> = ledger(1..=13, |_| BIN_A)
        .lines()
        .map(String::from)
        .collect();
    let mut unproven = ident(LEDGER_SCHEMA, "2026-10-14T02:00:00Z", BACKEND, BIN_A);
    unproven["gpu_proof"] = Value::Null;
    rows.push(unproven.to_string());
    let r = build_report(&rows.join("\n"), &verdicts(1..=14, &[]), "");
    assert_eq!(r.series[0].nights, 13);
    assert_eq!(r.status, Status::InsufficientNights);
    assert!(r
        .inadmissible
        .iter()
        .any(|x| x.contains("backend_unproven")));
}

#[test]
fn a_row_with_unknown_identity_is_excluded() {
    let mut row = ident(LEDGER_SCHEMA, "2026-10-01T02:00:00Z", BACKEND, BIN_A);
    row["apr_tag"] = json!("unknown");
    assert!(inadmissible_reason(&row, LEDGER_SCHEMA).is_some());
    row["apr_tag"] = json!("v0.70.0");
    assert!(inadmissible_reason(&row, LEDGER_SCHEMA).is_none());
}

#[test]
fn lane_rows_never_supply_a_night() {
    // 14 lane rows on 14 days are not 14 nightly rows (§2.3: lane rows are operational only).
    let lane: Vec<String> = (1..=14)
        .map(|d| {
            ident(
                LANE_SCHEMA,
                &format!("{}T12:00:00Z", night(d)),
                BACKEND,
                BIN_A,
            )
            .to_string()
        })
        .collect();
    let r = build_report(
        &ledger(1..=1, |_| BIN_A),
        &verdicts(1..=14, &[]),
        &lane.join("\n"),
    );
    assert_eq!(r.series[0].nights, 1);
    assert_eq!(r.status, Status::InsufficientNights);
}

#[test]
fn a_red_night_with_no_admissible_ledger_row_is_refused() {
    let r = build_report(
        &ledger(1..=14, |_| BIN_A),
        &verdicts(1..=15, &[(15, "rolling_2theta")]),
        "",
    );
    assert_eq!(r.status, Status::Refused);
}

#[test]
fn liveness_reds_are_listed_without_an_onset() {
    let r = build_report(
        &ledger(1..=14, |_| BIN_A),
        &verdicts(1..=15, &[(15, "liveness")]),
        "",
    );
    assert_eq!(r.liveness.len(), 1);
    assert_eq!(
        r.liveness[0].last_row_ts.as_deref(),
        Some("2026-10-14T02:00:00+00:00")
    );
    assert!(r.events.is_empty());
}

#[test]
fn markdown_carries_the_verdict() {
    let r = build_report(&ledger(1..=14, |_| BIN_A), &verdicts(1..=14, &[]), "");
    let md = r.to_markdown();
    assert!(md.contains("ProbeNotIndicated"));
    assert!(md.contains("| lambda-vector | cuda | 14 |"));
}
