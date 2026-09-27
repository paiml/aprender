use super::*;
use serde_json::json;

const NIGHT: &str = "2026-09-27";

fn row(schema: &str, host: &str, backend: &str, ts: &str) -> Value {
    json!({
        "schema": schema, "ts": ts, "host": host,
        "apr_version": "0.70.0", "apr_tag": "v0.70.0-rc.1",
        "crate_tarball_sha256": "c1", "binary_sha256": "b1", "build_identity": "bi",
        "model_id": "qwen3.5-4b-q4_k_m", "model_sha256": "m1",
        "backend": backend, "request_id": format!("{host}-{backend}-{ts}"),
        "gpu_proof": if backend == "cpu" { Value::Null } else { json!("trace: cuda kernel") },
    })
}

fn perf(host: &str, backend: &str, ts: &str, decode: f64) -> Value {
    let mut r = row(PERF_SCHEMA, host, backend, ts);
    r["ratio"] = json!({"load": 1.5, "ttft": 1.25, "prefill": 0.875, "decode": decode});
    r
}

fn lane(host: &str, i: usize, pre: f64, dec: f64) -> Value {
    let mut r = row(
        LANE_SCHEMA,
        host,
        "cpu",
        &format!("2026-09-27T10:{i:02}:00Z"),
    );
    r["ok"] = json!(true);
    r["prefill_ms_per_tok"] = json!(pre);
    r["decode_ms_per_tok"] = json!(dec);
    r
}

fn trace(host: &str, ts: &str, path: &str) -> Value {
    let mut r = row(TRACE_SCHEMA, host, "cpu", ts);
    r["level"] = json!("layer");
    r["provenance"] = json!("Measured");
    r["payload"] = json!({"path": path, "sha256": format!("sha-of-{path}")});
    r
}

fn jsonl(rows: &[Value]) -> String {
    rows.iter().map(|r| format!("{r}\n")).collect()
}

fn full() -> Ledgers {
    Ledgers {
        perf: jsonl(&[
            perf("lambda", "cpu", "2026-09-26T03:00:00Z", 0.9),
            perf("lambda", "cpu", "2026-09-27T03:00:00Z", 0.95),
        ]),
        lane: jsonl(&[lane("lambda", 1, 2.0, 20.0), lane("lambda", 2, 4.0, 40.0)]),
        trace: jsonl(&[
            trace("lambda", "2026-09-26T04:00:00Z", "old.jsonl"),
            trace("lambda", "2026-09-27T04:00:00Z", "new.jsonl"),
        ]),
    }
}

fn declared() -> Vec<Series> {
    vec![Series::new("lambda", "cpu")]
}

#[test]
fn complete_ledgers_render_green() {
    let v = build(&full(), &declared(), NIGHT);
    assert!(v.is_green(), "{:?}", v.red);
    assert_eq!(v.status[&Series::new("lambda", "cpu")], NightStatus::Green);
}

// R-2: every empty ledger is its own RED line, never 0/0 passing.
#[test]
fn each_empty_ledger_is_red() {
    for (i, name) in ["nightly", "lane", "trace"].iter().enumerate() {
        let mut l = full();
        match i {
            0 => l.perf.clear(),
            1 => l.lane.clear(),
            _ => l.trace.clear(),
        }
        let v = build(&l, &declared(), NIGHT);
        assert!(
            v.red.iter().any(|r| r.starts_with(name)),
            "{name}: {:?}",
            v.red
        );
    }
}

#[test]
fn nothing_at_all_is_red() {
    let v = build(&Ledgers::default(), &[], NIGHT);
    assert!(!v.is_green());
    assert!(v.red.contains(&"no declared series".to_string()));
}

// The declaration, not the rows present, says what is expected.
#[test]
fn a_declared_series_with_no_row_tonight_is_red() {
    let mut d = declared();
    d.push(Series::new("gx10", "cuda"));
    let v = build(&full(), &d, NIGHT);
    assert!(
        v.red.iter().any(|r| r.starts_with("gx10/cuda")),
        "{:?}",
        v.red
    );
    assert_eq!(
        v.status[&Series::new("gx10", "cuda")],
        NightStatus::Missing {
            last_seen: None,
            unproven: 0
        }
    );
}

#[test]
fn yesterdays_row_does_not_make_tonight_green() {
    let v = build(&full(), &declared(), "2026-09-28");
    assert_eq!(
        v.status[&Series::new("lambda", "cpu")],
        NightStatus::Missing {
            last_seen: Some("2026-09-27".into()),
            unproven: 0
        }
    );
    assert!(!v.is_green());
}

// The contract's list, written out here: iterating the module's own `IDENTITY` would let a
// field dropped from it drop out of this test too (mutation M2b survived that way).
const CONTRACT_IDENTITY: [&str; 12] = [
    "schema",
    "ts",
    "host",
    "apr_version",
    "apr_tag",
    "crate_tarball_sha256",
    "binary_sha256",
    "build_identity",
    "model_id",
    "model_sha256",
    "backend",
    "request_id",
];

#[test]
fn a_row_missing_any_identity_field_counts_as_absent() {
    for f in CONTRACT_IDENTITY {
        for bad in [None, Some(Value::Null), Some(json!("unknown"))] {
            let mut r = perf("lambda", "cpu", "2026-09-27T03:00:00Z", 0.95);
            match &bad {
                None => {
                    r.as_object_mut().expect("test fixture").remove(f);
                }
                Some(b) => r[f] = b.clone(),
            }
            assert!(
                matches!(
                    admit(&r.to_string(), PERF_SCHEMA),
                    Admission::Inadmissible(_)
                ),
                "{f} = {bad:?} was admitted"
            );
            let mut l = full();
            l.perf = jsonl(&[r]);
            let v = build(&l, &declared(), NIGHT);
            assert!(
                !v.is_green(),
                "{f} = {bad:?}: page GREEN on an inadmissible row"
            );
            assert!(v.ratios.is_empty());
        }
    }
}

#[test]
fn gpu_proof_key_absent_wrong_schema_bad_backend_bad_ts_long_line_are_inadmissible() {
    let base = perf("lambda", "cpu", "2026-09-27T03:00:00Z", 0.95);
    let mut no_key = base.clone();
    no_key
        .as_object_mut()
        .expect("test fixture")
        .remove("gpu_proof");
    let mut schema = base.clone();
    schema["schema"] = json!(LANE_SCHEMA);
    let mut backend = base.clone();
    backend["backend"] = json!("tpu");
    let mut ts = base.clone();
    ts["ts"] = json!("last night");
    let mut long = base;
    long["pad"] = json!("x".repeat(MAX_LINE_BYTES));
    for r in [no_key, schema, backend, ts, long] {
        assert!(
            matches!(
                admit(&r.to_string(), PERF_SCHEMA),
                Admission::Inadmissible(_)
            ),
            "{r}"
        );
    }
    assert!(matches!(
        admit("not json", PERF_SCHEMA),
        Admission::Inadmissible(_)
    ));
}

// §2.5: a GPU row without proof is out of every series, and its series is not GREEN.
#[test]
fn unproven_gpu_row_is_excluded_and_its_series_red() {
    let mut r = perf("gx10", "cuda", "2026-09-27T03:00:00Z", 1.1);
    r["gpu_proof"] = Value::Null;
    let mut l = full();
    l.perf.push_str(&jsonl(&[r]));
    let mut d = declared();
    d.push(Series::new("gx10", "cuda"));
    let v = build(&l, &d, NIGHT);
    assert_eq!(v.counts[0].unproven, 1);
    assert!(!v.ratios.contains_key(&Series::new("gx10", "cuda")));
    assert_eq!(
        v.status[&Series::new("gx10", "cuda")],
        NightStatus::Missing {
            last_seen: None,
            unproven: 1
        }
    );
}

#[test]
fn ratios_are_the_rows_own_in_time_order_and_identity_change_is_marked() {
    let mut l = full();
    let mut swapped = perf("lambda", "cpu", "2026-09-25T03:00:00Z", 0.5);
    swapped["model_sha256"] = json!("m0");
    l.perf.push_str(&jsonl(&[swapped]));
    let v = build(&l, &declared(), NIGHT);
    let pts = &v.ratios[&Series::new("lambda", "cpu")];
    let decode: Vec<_> = pts.iter().map(|p| p.ratio[3]).collect();
    assert_eq!(decode, vec![Some(0.5), Some(0.9), Some(0.95)]);
    assert_eq!(
        pts.iter().map(|p| p.identity_changed).collect::<Vec<_>>(),
        vec![false, true, false]
    );
}

#[test]
fn undeclared_series_is_listed_not_counted() {
    let mut l = full();
    l.perf
        .push_str(&jsonl(&[perf("yoga", "cpu", "2026-09-27T03:00:00Z", 1.0)]));
    let v = build(&l, &declared(), NIGHT);
    assert_eq!(v.undeclared, vec![Series::new("yoga", "cpu")]);
    assert!(!v.status.contains_key(&Series::new("yoga", "cpu")));
}

#[test]
fn nearest_rank_percentiles() {
    let v: Vec<f64> = (1..=20).map(f64::from).collect();
    assert_eq!(percentile(&v, 50), Some(10.0));
    assert_eq!(percentile(&v, 95), Some(19.0));
    assert_eq!(percentile(&[7.0], 95), Some(7.0));
    assert_eq!(percentile(&[], 50), None);
}

#[test]
fn lane_percentiles_skip_failed_rows() {
    let mut l = full();
    let mut bad = lane("lambda", 3, 999.0, 999.0);
    bad["ok"] = json!(false);
    bad["error"] = json!("timeout");
    l.lane.push_str(&jsonl(&[bad]));
    let v = build(&l, &declared(), NIGHT);
    let lp = &v.lane[&Series::new("lambda", "cpu")];
    assert_eq!((lp.measured, lp.failed), (2, 1));
    assert_eq!(lp.prefill, Some((2.0, 4.0)));
    assert_eq!(lp.decode, Some((20.0, 40.0)));
}

#[test]
fn latest_trace_wins_regardless_of_line_order() {
    let mut l = full();
    l.trace = jsonl(&[
        trace("lambda", "2026-09-27T04:00:00Z", "new.jsonl"),
        trace("lambda", "2026-09-26T04:00:00Z", "old.jsonl"),
    ]);
    let v = build(&l, &declared(), NIGHT);
    assert_eq!(v.traces[&Series::new("lambda", "cpu")].path, "new.jsonl");
}

// "0 hand-edited values": every number in a table row of the page comes from the
// ledgers. Change each input value and the page must change with it.
#[test]
fn page_values_follow_the_data() {
    let page = to_markdown(&build(&full(), &declared(), NIGHT));
    for want in [
        "0.950",
        "1.500",
        "1.250",
        "0.875",
        "2.000 / 4.000",
        "20.000 / 40.000",
        "[new.jsonl](new.jsonl)",
        "`sha-of-new.jsonl`",
        "**GREEN**",
    ] {
        assert!(page.contains(want), "missing {want}\n{page}");
    }
    let mut l = full();
    l.perf = l.perf.replace("0.95", "0.625");
    let page2 = to_markdown(&build(&l, &declared(), NIGHT));
    assert!(page2.contains("0.625") && !page2.contains("0.950"));
}

#[test]
fn page_is_a_pure_function_of_its_inputs() {
    let a = to_markdown(&build(&full(), &declared(), NIGHT));
    let b = to_markdown(&build(&full(), &declared(), NIGHT));
    assert_eq!(a, b);
}

#[test]
fn red_page_says_red_and_why() {
    let page = to_markdown(&build(&Ledgers::default(), &declared(), NIGHT));
    assert!(page.contains("**RED**"));
    assert!(page.contains("- RED: nightly ledger has 0 admissible rows"));
    assert!(page.contains("RED — missing (last seen never; 0 backend_unproven)"));
}

#[test]
fn missing_file_reads_as_empty_ledger() {
    let d = tempfile::tempdir().expect("test fixture");
    let p = d.path().join("perf.jsonl");
    std::fs::write(&p, full().perf).expect("test fixture");
    let l = Ledgers::read(
        &p,
        &d.path().join("absent-lane.jsonl"),
        &d.path().join("absent-trace.jsonl"),
    )
    .expect("test fixture");
    assert!(l.lane.is_empty() && l.trace.is_empty() && !l.perf.is_empty());
    assert!(!build(&l, &declared(), NIGHT).is_green());
}
