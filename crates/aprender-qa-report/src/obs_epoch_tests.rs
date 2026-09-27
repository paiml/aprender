use super::*;
use serde_json::json;

fn row(night: &str) -> Value {
    json!({
        "schema": PERF_SCHEMA,
        "ts": format!("{night}T03:00:00Z"),
        "model_sha256": "m1",
        "comparator": {"name": "llama-bench", "build_commit": "b100", "binary_sha256": "lb", "flags": {"p": 512, "n": 128}},
        "build_identity": {"rustc": "rustc 1.93.0", "driver": "580.1", "features": ["cuda"]},
        "band": {"c": 1, "ctx": 4096, "workload_id": "W1"},
        "binary_sha256": "apr-1",
    })
}

fn night(d: u32) -> String {
    format!("2026-10-{d:02}")
}

fn date(s: &str) -> NaiveDate {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").expect("test date")
}

#[test]
fn theta_min_is_minus_ln_095() {
    assert!((theta_min() - 0.05129).abs() < 1e-5);
}

// FALSIFY-OBS-EPOCH-005: the apr binary, including a rustc bump, is the signal (C19).
#[test]
fn rustc_or_apr_binary_change_does_not_open_an_epoch() {
    let a = row(&night(1));
    let mut b = row(&night(2));
    b["build_identity"]["rustc"] = json!("rustc 1.94.0");
    b["build_identity"]["features"] = json!(["cuda", "fp8"]);
    b["binary_sha256"] = json!("apr-2");
    let epochs = segment(&[a, b]).expect("segment");
    assert_eq!(epochs.len(), 1, "{epochs:?}");
    assert_eq!(epochs[0].nights.len(), 2);
}

#[test]
fn model_or_workload_change_recalibrates_comparator_or_driver_bridges() {
    let base = row(&night(1));
    let cases: [(&str, fn(&mut Value), Change); 5] = [
        (
            "model",
            |r| r["model_sha256"] = json!("m2"),
            Change::Recalibrate(vec!["model"]),
        ),
        (
            "workload",
            |r| r["band"]["workload_id"] = json!("W5"),
            Change::Recalibrate(vec!["workload"]),
        ),
        (
            "commit",
            |r| r["comparator"]["build_commit"] = json!("b101"),
            Change::Bridge(vec!["comparator"]),
        ),
        (
            "flags",
            |r| r["comparator"]["flags"]["n"] = json!(256),
            Change::Bridge(vec!["comparator"]),
        ),
        (
            "driver",
            |r| r["build_identity"]["driver"] = json!("590.0"),
            Change::Bridge(vec!["driver"]),
        ),
    ];
    for (name, mutate, want) in cases {
        let mut next = row(&night(2));
        mutate(&mut next);
        let epochs = segment(&[base.clone(), next]).expect("segment");
        assert_eq!(epochs.len(), 2, "{name}");
        assert_eq!(epochs[1].opened_by, want, "{name}");
    }
    let mut both = row(&night(2));
    both["model_sha256"] = json!("m2");
    both["build_identity"]["driver"] = json!("590.0");
    assert_eq!(
        segment(&[base, both]).expect("segment")[1].opened_by,
        Change::Recalibrate(vec!["model", "driver"]),
        "a model change never bridges, even alongside a driver change"
    );
}

#[test]
fn flags_compare_canonically_not_by_key_order() {
    let a = row(&night(1));
    let mut b = row(&night(2));
    b["comparator"]["flags"] = serde_json::from_str(r#"{"n":128,"p":512}"#).expect("json");
    assert_eq!(segment(&[a, b]).expect("segment").len(), 1);
}

#[test]
fn segments_follow_night_order_not_line_order() {
    let mut late = row(&night(3));
    late["model_sha256"] = json!("m2");
    let epochs = segment(&[late, row(&night(1)), row(&night(2))]).expect("segment");
    assert_eq!(epochs.len(), 2);
    assert_eq!(epochs[0].nights, vec![night(1), night(2)]);
}

// FALSIFY-OBS-EPOCH-004: a backfill row is refused by a series, not skipped.
#[test]
fn backfill_row_is_refused_by_a_series() {
    let mut b = row(&night(2));
    b["schema"] = json!(BACKFILL_SCHEMA);
    assert_eq!(
        segment(&[row(&night(1)), b]),
        Err(SegmentError::BackfillInSeries(1))
    );
    let mut other = row(&night(2));
    other["schema"] = json!("apr-lane-row-v1");
    assert!(matches!(
        segment(&[other]),
        Err(SegmentError::WrongSchema(0, _))
    ));
}

#[test]
fn unkeyed_rows_are_refused() {
    for f in ["model_sha256", "comparator", "build_identity", "band", "ts"] {
        let mut r = row(&night(1));
        r.as_object_mut().expect("obj").remove(f);
        assert!(
            matches!(segment(&[r]), Err(SegmentError::Unkeyed(0, _))),
            "{f}"
        );
    }
    let mut r = row(&night(1));
    r["build_identity"]
        .as_object_mut()
        .expect("obj")
        .remove("driver");
    assert!(
        matches!(segment(&[r]), Err(SegmentError::Unkeyed(0, _))),
        "driver key absent"
    );
    let mut cpu = row(&night(1));
    cpu["build_identity"]["driver"] = Value::Null;
    assert!(
        segment(&[cpu]).is_ok(),
        "a null driver (cpu host) is a value, not a hole"
    );
}

// FALSIFY-OBS-EPOCH-001: a comparator/driver change needs exactly 3 bridge nights.
#[test]
fn pin_change_without_three_bridge_nights_is_refused() {
    let change = Change::Bridge(vec!["comparator"]);
    assert!(matches!(
        entry(&change, None),
        Err(EntryError::Unbridged(_, None))
    ));
    for n in [0, 1, 2, 4] {
        let v = vec![0.1; n];
        let b = bridge(&v, &v, 0.02);
        assert_eq!(b, Err(BridgeError::NightCount { old: n, new: n }));
        assert!(entry(&change, Some(b)).is_err(), "{n} nights admitted");
    }
    assert_eq!(
        bridge(&[0.1, 0.1, 0.1], &[0.1, 0.1], 0.02),
        Err(BridgeError::NightCount { old: 3, new: 2 })
    );
    assert_eq!(
        bridge(&[0.1, f64::NAN, 0.1], &[0.1; 3], 0.02),
        Err(BridgeError::NonFinite)
    );
    assert_eq!(
        entry(&Change::Recalibrate(vec!["model"]), None),
        Ok(Entry::Fresh)
    );
}

// FALSIFY-OBS-EPOCH-002: dispersion above σ̂_e is not bridge_ok.
#[test]
fn bridge_ok_iff_bridge_dispersion_within_sigma_e() {
    let old = [0.10, 0.10, 0.10];
    let new = [0.12, 0.14, 0.20]; // d = 0.02, 0.04, 0.10: med 0.04, MAD 0.02
    let sb = MAD_TO_SIGMA * 0.02;
    let b = bridge(&old, &new, sb - 1e-9).expect("bridge");
    assert!(!b.bridge_ok);
    assert!((b.delta_e - 0.04).abs() < 1e-12);
    assert!((b.sigma_bridge - sb).abs() < 1e-12);
    assert!(bridge(&old, &new, sb + 1e-9).expect("bridge").bridge_ok);
}

// FALSIFY-OBS-EPOCH-003 and the §10 E9 `prov_ge` property.
#[test]
fn provisional_theta_is_theta_prov_and_never_below_full_calibration() {
    assert_eq!(theta_prov(2, 0.01, 0.01), None);
    let factor3 = 1.0 + 2.0 * 1.166 / 3f64.sqrt();
    assert!(
        (factor3 - 2.346).abs() < 1e-3,
        "worked [C]: 2.35 at n_e = 3"
    );
    let want = theta_min().max(3.0 * 0.03 * factor3) + 3.0 * 1.2533 * 0.01 / 3f64.sqrt();
    assert!((theta_prov(3, 0.03, 0.01).expect("n_e=3") - want).abs() < 1e-12);
    for n in 3..=13 {
        for s in [0.0, 0.001, 0.01, 0.05, 0.2] {
            for sb in [0.0, 0.01, 0.1] {
                let p = theta_prov(n, s, sb).expect("n_e>=3");
                assert!(p >= theta_min().max(K * s), "n={n} s={s} sb={sb}");
            }
        }
    }
    let prev = theta_prov(3, 0.03, 0.01).expect("3");
    assert!(
        theta_prov(13, 0.03, 0.01).expect("13") < prev,
        "θ_prov shrinks as nights accrue"
    );
}

fn nights_with_spread(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.2 + 0.01 * ((i % 5) as f64 - 2.0))
        .collect()
}

// Positive: a failed bridge stays armed (provisional) and graduates at n_e = 14.
#[test]
fn failed_bridge_stays_armed_and_graduates_at_14() {
    let b = bridge(&[0.1, 0.1, 0.1], &[0.12, 0.14, 0.20], 0.001).expect("bridge");
    assert!(!b.bridge_ok);
    let e = entry(&Change::Bridge(vec!["driver"]), Some(Ok(b))).expect("entry");
    for n in 3..14 {
        let l = nights_with_spread(n);
        let want = theta_prov(n, sigma_hat(&l).expect("s"), b.sigma_bridge).expect("θ");
        assert_eq!(
            mode(e, &l, None),
            Mode::Provisional {
                n_e: n,
                theta: want
            },
            "n_e={n}"
        );
    }
    let l = nights_with_spread(14);
    let c = calibrate(&l).expect("14 nights");
    assert_eq!(mode(e, &l, None), Mode::Calibrated { theta: c.theta });
}

// Positive: a clean pin bump with a passing bridge carries θ and the baseline.
#[test]
fn clean_pin_bump_with_passing_bridge_carries_theta_and_baseline() {
    let b = bridge(&[0.10, 0.11, 0.12], &[0.15, 0.16, 0.17], 0.02).expect("bridge");
    assert!(b.bridge_ok);
    assert!((carried_baseline(0.3, &b) - 0.35).abs() < 1e-12);
    let e = entry(&Change::Bridge(vec!["comparator"]), Some(Ok(b))).expect("entry");
    assert_eq!(
        mode(e, &[0.15, 0.16, 0.17], Some(0.07)),
        Mode::Calibrated { theta: 0.07 }
    );
}

#[test]
fn fresh_epoch_is_report_only_until_14_nights() {
    assert_eq!(
        mode(Entry::Fresh, &nights_with_spread(13), None),
        Mode::ReportOnly { nights: 13 }
    );
    let l = nights_with_spread(14);
    assert!(matches!(
        mode(Entry::Fresh, &l, None),
        Mode::Calibrated { .. }
    ));
    let c = calibrate(&l).expect("14");
    assert!(c.theta >= theta_min() && (c.theta - theta_min().max(3.0 * c.sigma_hat)).abs() < 1e-12);
}

#[test]
fn median_and_sigma_hat() {
    assert_eq!(median(&[]), None);
    assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
    assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
    assert!((sigma_hat(&[1.0, 2.0, 3.0, 4.0, 100.0]).expect("s") - 1.4826).abs() < 1e-12);
}

// FALSIFY-OBS-EPOCH-006: no epoch change on a gated host inside T−14 of a final.
#[test]
fn epoch_change_inside_t_minus_14_of_a_final_is_frozen() {
    let finals = [date("2026-10-20"), date("2026-12-01")];
    for (c, want) in [
        ("2026-10-20", Some(0)),
        ("2026-10-13", Some(7)),
        ("2026-10-06", Some(14)),
    ] {
        let r = freeze_check(date(c), &finals, true);
        assert_eq!(r.err().map(|f| f.nights_before), want, "{c}");
    }
    for c in ["2026-10-05", "2026-10-21"] {
        assert!(freeze_check(date(c), &finals, true).is_ok(), "{c}");
    }
    assert!(
        freeze_check(date("2026-10-13"), &finals, false).is_ok(),
        "ungated host"
    );
    assert!(
        freeze_check(date("2026-10-13"), &[], true).is_ok(),
        "no final scheduled"
    );
}
