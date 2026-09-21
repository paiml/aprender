//! PMAT-3577 — the extractor's case table. Every case is a state that either happened in this tree or is
//! one edit away from happening, and each says which.

use super::*;

fn record(kind: &str, extra: serde_json::Value) -> serde_json::Value {
    let mut v = serde_json::json!({
        "schema": SCHEMA,
        "cell": {"model": "m", "file": "./m.gguf", "quant": "Q4_K_M"},
        "host": "gx10-a5b5",
        "backend": "cuda",
        "apr_version": "0.65.2",
        "generated_at": "2026-09-08",
        "comparator": {"kind": kind, "reason": "no oracle arm exists for this cell"},
        "partially_receipted": true,
        "threshold_source": "evidence/parity/thresholds.yaml",
        "unmeasured": ["ORACLE ARM: not measured."],
        "result": {"positions": 78, "parity": true},
        "raw": {"model": "./m.gguf", "metrics": []}
    });
    if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
        for (k, val) in e {
            o.insert(k.clone(), val.clone());
        }
    }
    v
}

fn emit_one(v: &serde_json::Value) -> (Graph, ParityStats) {
    let mut g = Graph::new();
    let mut stats = ParityStats::default();
    emit(&mut g, Path::new("."), "r.json", v, &mut stats);
    (g, stats)
}

fn objects(g: &Graph, prop: &str) -> Vec<String> {
    g.objects(&iri("parity-receipt", "r.json"), &parity(prop))
        .iter()
        .filter_map(|t| t.as_literal().map(|l| l.0.to_string()))
        .collect()
}

#[test]
fn a_self_compared_record_types_as_the_self_subclass_and_carries_every_named_property() {
    let (g, _) = emit_one(&record("self", serde_json::json!({})));
    let node = iri("parity-receipt", "r.json");
    assert_eq!(
        g.objects(&node, RDF_TYPE)[0].as_iri(),
        Some(parity("SelfComparedReceipt").as_str())
    );
    assert_eq!(objects(&g, "host"), ["gx10-a5b5"]);
    assert_eq!(objects(&g, "backend"), ["cuda"]);
    assert_eq!(objects(&g, "aprVersion"), ["0.65.2"]);
    assert_eq!(objects(&g, "generatedAt"), ["2026-09-08"]);
    assert_eq!(objects(&g, "partiallyReceipted"), ["true"]);
    assert_eq!(objects(&g, "unmeasured").len(), 1);
}

#[test]
fn a_non_self_comparator_types_as_the_oracle_subclass() {
    let v = record(
        "llama_cpp",
        serde_json::json!({"comparator": {"kind": "llama_cpp", "comparator_sha": "39173bcac"}}),
    );
    let (g, _) = emit_one(&v);
    assert_eq!(
        g.objects(&iri("parity-receipt", "r.json"), RDF_TYPE)[0].as_iri(),
        Some(parity("OracleComparedReceipt").as_str())
    );
    let c = iri("parity-comparator", "r.json");
    assert_eq!(
        g.objects(&c, &parity("comparatorSha"))[0]
            .as_literal()
            .map(|l| l.0),
        Some("39173bcac")
    );
}

#[test]
fn model_sha256_is_absent_when_the_record_carries_none_and_absent_is_legal() {
    // ONT-4c1's rule, and the reason the shape gives model_sha256 a pattern and no minCount: six of the
    // seven back-filled records never recorded a model hash, and hashing the file on the host TODAY would
    // attach a claim about a different world to a receipt about 2026-09-06.
    let (g, _) = emit_one(&record("self", serde_json::json!({})));
    assert!(objects(&g, "modelSha256").is_empty());
    let with = record(
        "self",
        serde_json::json!({"cell": {"model_sha256": "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"}}),
    );
    let (g2, _) = emit_one(&with);
    assert_eq!(objects(&g2, "modelSha256").len(), 1);
}

#[test]
fn a_threshold_source_that_names_no_file_is_materialised_for_a_shape_to_refuse() {
    let v = record(
        "self",
        serde_json::json!({"threshold_source": "evidence/parity/does-not-exist.yaml"}),
    );
    let (g, stats) = emit_one(&v);
    assert_eq!(stats.threshold_source_missing, 1);
    assert_eq!(
        objects(&g, "thresholdSourceMissing"),
        ["evidence/parity/does-not-exist.yaml"]
    );
}

#[test]
fn the_plant_removes_the_comparator_edge_every_run() {
    assert!(positive_control(&record("self", serde_json::json!({}))));
}

#[test]
fn an_unmigrated_legacy_record_is_refused_by_name_and_never_skipped() {
    // The exact layout the seven records carried before #3577: no schema, metrics[] at the top level.
    let dir = tempdir("unmigrated");
    let f = dir.join(EVIDENCE_DIR).join("l0-1/lambda");
    std::fs::create_dir_all(&f).expect("mkdir");
    std::fs::write(
        f.join("legacy.json"),
        r#"{"model":"./m.gguf","tokens":78,"passed":78,"failed":0,"parity":true,"metrics":[]}"#,
    )
    .expect("write");
    let mut g = Graph::new();
    let stats = extract(&dir, &mut g);
    assert_eq!(stats.records, 0);
    assert_eq!(stats.skipped, 0, "a legacy record must not be SKIPPED");
    assert_eq!(stats.errors.len(), 1);
    assert!(
        stats.errors[0].what.contains("UNMIGRATED"),
        "{:?}",
        stats.errors[0]
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_document_that_is_not_a_parity_record_is_skipped_and_counted() {
    let dir = tempdir("other");
    let f = dir.join(EVIDENCE_DIR);
    std::fs::create_dir_all(&f).expect("mkdir");
    std::fs::write(f.join("props-abc.json"), r#"{"seed": 1, "cases": []}"#).expect("write");
    let mut g = Graph::new();
    let stats = extract(&dir, &mut g);
    assert_eq!(
        (stats.records, stats.skipped, stats.errors.len()),
        (0, 1, 0)
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_denominator_pins_the_count_and_a_mismatch_is_reported_in_both_directions() {
    let dir = tempdir("denominator");
    let f = dir.join(EVIDENCE_DIR);
    std::fs::create_dir_all(&f).expect("mkdir");
    std::fs::write(
        f.join("r.json"),
        serde_json::to_string(&record("self", serde_json::json!({}))).expect("json"),
    )
    .expect("write");

    // Committed 2, found 1 — a record was deleted or the extractor stopped seeing it.
    std::fs::write(dir.join(EXPECTED_FILE), "# count\n2\n").expect("write");
    let mut g = Graph::new();
    let stats = extract(&dir, &mut g);
    assert_eq!(stats.wrong_corpus(), Some((2, 1)));

    // Committed 1, found 1 — the state the gate requires.
    std::fs::write(dir.join(EXPECTED_FILE), "1\n").expect("write");
    let mut g = Graph::new();
    let stats = extract(&dir, &mut g);
    assert_eq!(stats.wrong_corpus(), None);

    // Committed 0, found 1 — a receipt added without updating the denominator. THE falsifier the row names.
    std::fs::write(dir.join(EXPECTED_FILE), "0\n").expect("write");
    let mut g = Graph::new();
    let stats = extract(&dir, &mut g);
    assert_eq!(stats.wrong_corpus(), Some((0, 1)));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_absent_denominator_is_not_a_mismatch_it_is_a_different_fault() {
    // "no expectation" and "a broken expectation" must not collapse into one state: the first is a missing
    // declaration for the caller to report, the second is WrongCorpus.
    let stats = ParityStats {
        records: 3,
        expected: None,
        ..ParityStats::default()
    };
    assert_eq!(stats.wrong_corpus(), None);
}

#[test]
fn pointing_the_extractor_at_a_subdirectory_trips_the_denominator() {
    // The row's second falsifier: a narrowed walk finds fewer records and must not read as "no violations".
    let dir = tempdir("subdir");
    let deep = dir.join(EVIDENCE_DIR).join("l0-1/lambda");
    std::fs::create_dir_all(&deep).expect("mkdir");
    std::fs::write(
        deep.join("r.json"),
        serde_json::to_string(&record("self", serde_json::json!({}))).expect("json"),
    )
    .expect("write");
    std::fs::write(dir.join(EXPECTED_FILE), "1\n").expect("write");
    let mut g = Graph::new();
    assert_eq!(extract(&dir, &mut g).wrong_corpus(), None);

    // Same denominator, a root whose evidence/parity holds nothing: 1 expected, 0 found.
    let narrow = tempdir("subdir-narrow");
    std::fs::create_dir_all(narrow.join(EVIDENCE_DIR)).expect("mkdir");
    std::fs::write(narrow.join(EXPECTED_FILE), "1\n").expect("write");
    let mut g2 = Graph::new();
    assert_eq!(extract(&narrow, &mut g2).wrong_corpus(), Some((1, 0)));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&narrow).ok();
}

fn tempdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "pmat3577-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    std::fs::create_dir_all(&p).expect("tempdir");
    p
}
