use super::*;
use serde_json::json;

// The contract's list, written out: iterating the module's own RECEIPT_IDENTITY would let
// a field dropped from it drop out of this test too.
const CONTRACT_FIELDS: [&str; 11] = [
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

fn complete() -> Value {
    json!({
        "ts": "2026-09-01T03:00:00Z", "host": "lambda",
        "apr_version": "0.68.2", "apr_tag": "v0.68.2",
        "crate_tarball_sha256": "c1", "binary_sha256": "b1",
        "build_identity": {"rustc": "rustc 1.93.0", "driver": null},
        "model_id": "qwen", "model_sha256": "m1",
        "backend": "cpu", "request_id": "0190-0000", "gpu_proof": null,
        "samples_ms": [1.0, 2.0],
    })
}

#[test]
fn complete_receipt_imports_as_a_backfill_row_by_reference() {
    let line = import(&complete(), "evidence/perf-x/r.json", "abc").expect("import");
    let v: Value = serde_json::from_str(&line).expect("json");
    assert_eq!(v["schema"], BACKFILL_SCHEMA);
    assert_eq!(
        v["source"],
        json!({"path": "evidence/perf-x/r.json", "sha256": "abc"})
    );
    assert!(
        v.get("samples_ms").is_none(),
        "payload is referenced, not copied"
    );
    assert_eq!(
        line,
        canonical(&v),
        "canonical JSON: sorted keys, no whitespace"
    );
    assert!(line.len() <= MAX_LINE_BYTES);
}

#[test]
fn every_missing_identity_field_is_refused_and_named() {
    for f in CONTRACT_FIELDS {
        for bad in [
            None,
            Some(Value::Null),
            Some(json!("unknown")),
            Some(json!("")),
        ] {
            let mut r = complete();
            match &bad {
                None => {
                    r.as_object_mut().expect("obj").remove(f);
                }
                Some(b) => r[f] = b.clone(),
            }
            let err = import(&r, "p", "s").expect_err(f);
            assert!(err.iter().any(|e| e == f), "{f} = {bad:?}: {err:?}");
        }
    }
}

#[test]
fn gpu_proof_is_required_and_proven_off_cpu() {
    let mut r = complete();
    r.as_object_mut().expect("obj").remove("gpu_proof");
    assert_eq!(import(&r, "p", "s"), Err(vec!["gpu_proof".to_string()]));
    let mut cuda = complete();
    cuda["backend"] = json!("cuda");
    assert!(import(&cuda, "p", "s").is_err(), "cuda with null gpu_proof");
    cuda["gpu_proof"] = json!("trace: sm_89 kernel");
    assert!(import(&cuda, "p", "s").is_ok());
}

#[test]
fn oversize_row_is_refused() {
    let mut r = complete();
    r["build_identity"] = json!({"uname": "x".repeat(MAX_LINE_BYTES)});
    assert!(import(&r, "p", "s").expect_err("oversize")[0].contains("4096"));
}

#[test]
fn not_an_object_is_refused() {
    assert_eq!(missing(&json!([1])), vec!["not a JSON object".to_string()]);
}

#[test]
fn import_tree_reads_only_evidence_perf_dirs() {
    let d = tempfile::tempdir().expect("tmp");
    let perf = d.path().join("evidence/perf-001/sub");
    std::fs::create_dir_all(&perf).expect("mkdir");
    std::fs::create_dir_all(d.path().join("evidence/other")).expect("mkdir");
    std::fs::write(perf.join("good.json"), complete().to_string()).expect("w");
    std::fs::write(perf.join("bad.json"), "{\"host\": \"lambda\"}").expect("w");
    std::fs::write(perf.join("garbage.json"), "not json").expect("w");
    std::fs::write(perf.join("notes.txt"), "{}").expect("w");
    std::fs::write(
        d.path().join("evidence/other/x.json"),
        complete().to_string(),
    )
    .expect("w");
    let got = import_tree(d.path()).expect("import");
    assert_eq!(got.scanned, 3);
    assert_eq!(got.rows.len(), 1);
    let refused: Vec<&str> = got.refused.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(
        refused,
        vec![
            "evidence/perf-001/sub/bad.json",
            "evidence/perf-001/sub/garbage.json"
        ]
    );
    let v: Value = serde_json::from_str(&got.rows[0]).expect("json");
    let bytes = std::fs::read(perf.join("good.json")).expect("r");
    assert_eq!(
        v["source"]["sha256"],
        format!("{:x}", Sha256::digest(&bytes))
    );
}

// Measured on the repository's own evidence: the importer runs over real receipts
// (non-vacuous) and anything it emits is a complete-identity backfill row.
#[test]
fn repository_evidence_imports_complete_identity_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let got = import_tree(&root).expect("import");
    assert!(
        got.scanned > 0,
        "no evidence/perf* receipts found: the run proves nothing"
    );
    assert_eq!(got.scanned, got.rows.len() + got.refused.len());
    for line in &got.rows {
        let v: Value = serde_json::from_str(line).expect("json");
        assert_eq!(v["schema"], BACKFILL_SCHEMA);
        assert!(missing(&v).is_empty(), "{line}");
    }
    assert!(got.refused.iter().all(|(_, why)| !why.is_empty()));
    eprintln!(
        "backfill: scanned {} imported {} refused {}",
        got.scanned,
        got.rows.len(),
        got.refused.len()
    );
}
