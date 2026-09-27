use super::*;
use serde_json::json;

fn identity(schema: &str, id: &str) -> Value {
    json!({
        "schema": schema,
        "ts": "2026-09-27T07:00:00Z",
        "host": "lambda",
        "apr_version": "0.71.0-dev",
        "apr_tag": "v0.71.0-dev+475f09a6f1",
        "crate_tarball_sha256": "a".repeat(64),
        "binary_sha256": "b".repeat(64),
        "build_identity": "features=inference",
        "model_id": "qwen3.5-4b-q4k",
        "model_sha256": "c".repeat(64),
        "backend": "cpu",
        "request_id": id,
        "gpu_proof": null,
    })
}

fn lane_row(id: &str, wall_ms: f64) -> Value {
    let mut v = identity(LANE_SCHEMA, id);
    v["wall_ms"] = json!(wall_ms);
    v["ok"] = json!(true);
    v
}

fn serve_row(id: &str, total_ms: f64) -> Value {
    let mut v = identity(SERVE_SCHEMA, id);
    v["total_ms"] = json!(total_ms);
    v["prefill_ms"] = json!(total_ms * 0.2);
    v["decode_ms"] = json!(total_ms * 0.7);
    v
}

fn jsonl(rows: &[Value]) -> String {
    rows.iter().map(|r| format!("{r}\n")).collect()
}

fn id(i: usize) -> String {
    format!("0192f000-0000-7000-8000-{i:012}")
}

/// 200 pairs, wall 1000..=4980 ms, server total = wall minus 10..40 ms of transport.
fn clean_pairs() -> (Vec<Value>, Vec<Value>) {
    let (mut lane, mut serve) = (Vec::new(), Vec::new());
    for i in 0..200 {
        let wall = 1000.0 + 20.0 * i as f64;
        lane.push(lane_row(&id(i), wall));
        serve.push(serve_row(&id(i), wall - 10.0 - (i % 4) as f64 * 10.0));
    }
    (lane, serve)
}

#[test]
fn clean_ledgers_are_green() {
    let (lane, serve) = clean_pairs();
    let r = check_agreement(&jsonl(&lane), &jsonl(&serve));
    assert_eq!((r.joined, r.agreeing), (200, 200));
    assert!(r.is_green(), "{:?}", r.red_reasons());
}

/// FALSIFY-OBS-AGREE-001: a planted +20% skew in server timings must turn RED.
#[test]
fn planted_plus_20_percent_skew_is_red() {
    let (lane, mut serve) = clean_pairs();
    for s in &mut serve {
        let t = s["total_ms"].as_f64().unwrap();
        s["total_ms"] = json!(t * 1.2);
    }
    let r = check_agreement(&jsonl(&lane), &jsonl(&serve));
    assert_eq!(r.joined, 200);
    assert_eq!(r.disagreements.len(), 200);
    assert!(!r.is_green());
}

/// FALSIFY-OBS-AGREE-002: an empty join is RED (R-2), whether the ledgers are empty,
/// disjoint, or every row is inadmissible.
#[test]
fn empty_join_is_red() {
    assert!(!check_agreement("", "").is_green());

    let (lane, _) = clean_pairs();
    let other: Vec<Value> = (500..700).map(|i| serve_row(&id(i), 1000.0)).collect();
    let r = check_agreement(&jsonl(&lane), &jsonl(&other));
    assert_eq!((r.joined, r.unjoined_lane, r.unjoined_serve), (0, 200, 200));
    assert!(!r.is_green());

    let (lane, mut serve) = clean_pairs();
    for s in &mut serve {
        s["model_sha256"] = json!("unknown");
    }
    let r = check_agreement(&jsonl(&lane), &jsonl(&serve));
    assert_eq!((r.serve_rows, r.inadmissible_serve.len()), (0, 200));
    assert!(!r.is_green());
}

/// FALSIFY-OBS-AGREE-003: a pair across mismatched identity is refused, not compared.
#[test]
fn identity_mismatch_is_refused() {
    let (lane, mut serve) = clean_pairs();
    serve[7]["backend"] = json!("cuda");
    serve[7]["gpu_proof"] = json!("trace: kernel q4k_gemv sm_89");
    let r = check_agreement(&jsonl(&lane), &jsonl(&serve));
    assert_eq!(r.joined, 199);
    assert!(r.refusals.iter().any(|m| m.contains("backend")));
    assert!(!r.is_green());
}

#[test]
fn duplicate_request_id_is_red() {
    let (mut lane, serve) = clean_pairs();
    lane.push(lane_row(&id(3), 1060.0));
    let r = check_agreement(&jsonl(&lane), &jsonl(&serve));
    assert!(r.refusals.iter().any(|m| m.contains("appears twice")));
    assert!(!r.is_green());
}

#[test]
fn ninety_nine_percent_boundary() {
    let lane: Vec<Value> = (0..100).map(|i| lane_row(&id(i), 1000.0)).collect();
    let mut serve: Vec<Value> = (0..100).map(|i| serve_row(&id(i), 990.0)).collect();
    serve[0] = serve_row(&id(0), 2000.0);
    assert!(check_agreement(&jsonl(&lane), &jsonl(&serve)).is_green());
    serve[1] = serve_row(&id(1), 2000.0);
    let r = check_agreement(&jsonl(&lane), &jsonl(&serve));
    assert_eq!((r.joined, r.agreeing), (100, 98));
    assert!(!r.is_green());
}

#[test]
fn tolerance_is_max_of_five_percent_and_fifty_ms() {
    assert!((tolerance_ms(100.0) - 50.0).abs() < 1e-9);
    assert!((tolerance_ms(1000.0) - 50.0).abs() < 1e-9);
    assert!((tolerance_ms(2000.0) - 100.0).abs() < 1e-9);
}

/// Documents the contract's stated blind spot: below 250 ms of wall time the 50 ms floor
/// absorbs a +20% skew, so FALSIFY-OBS-AGREE-001 is only sensitive on longer requests.
#[test]
fn skew_below_the_floor_is_not_detectable() {
    let r = check_agreement(
        &jsonl(&[lane_row(&id(0), 200.0)]),
        &jsonl(&[serve_row(&id(0), 240.0)]),
    );
    assert!(r.is_green());
}

#[test]
fn inadmissible_rows_count_as_absent() {
    let mut no_proof = lane_row(&id(0), 1000.0);
    no_proof.as_object_mut().unwrap().remove("gpu_proof");
    let mut wrong_schema = lane_row(&id(1), 1000.0);
    wrong_schema["schema"] = json!(SERVE_SCHEMA);
    let mut no_wall = lane_row(&id(2), 1000.0);
    no_wall["wall_ms"] = json!(0);
    let text = format!("{}not json\n", jsonl(&[no_proof, wrong_schema, no_wall]));
    let r = check_agreement(&text, "");
    assert_eq!((r.lane_rows, r.inadmissible_lane.len()), (0, 4));
}
