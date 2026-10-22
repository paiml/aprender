//! `certify::analyze_config` derives geometry from a HF `config.json` and flags
//! the algebraic constraints a transformer needs (#5622).

use aprender_contracts_cli::commands::certify::analyze_config;

fn analyze(json: &str) -> serde_json::Value {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("config.json");
    std::fs::write(&p, json).expect("write");
    analyze_config(&p).expect("analyze")
}

fn failed_checks(v: &serde_json::Value) -> Vec<String> {
    v["algebra_checks"]
        .as_array()
        .expect("array")
        .iter()
        .filter(|c| c["passed"] == false)
        .map(|c| c["check"].as_str().expect("name").to_string())
        .collect()
}

#[test]
fn valid_config_derives_geometry_and_passes_every_check() {
    let v = analyze(
        r#"{"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":8,
            "num_key_value_heads":2,"vocab_size":100}"#,
    );
    assert_eq!(v["head_dim"], 8);
    assert_eq!(v["num_kv_heads"], 2);
    // embed + 9 tensors per layer + final norm + lm_head
    assert_eq!(v["expected_tensors"], 1 + 2 * 9 + 2);
    assert_eq!(v["algebra_checks"].as_array().expect("array").len(), 7);
    assert_eq!(v["all_checks_pass"], true);
    assert!(failed_checks(&v).is_empty());
}

#[test]
fn kv_heads_default_to_attention_heads() {
    let v = analyze(
        r#"{"hidden_size":32,"num_hidden_layers":1,"num_attention_heads":4,"vocab_size":10}"#,
    );
    assert_eq!(v["num_kv_heads"], 4);
    assert_eq!(v["all_checks_pass"], true);
}

#[test]
fn empty_config_fails_the_positivity_checks_without_dividing_by_zero() {
    let v = analyze("{}");
    assert_eq!(v["all_checks_pass"], false);
    assert_eq!(v["head_dim"], 0);
    assert_eq!(v["expected_tensors"], 0);
    let failed = failed_checks(&v);
    for name in [
        "hidden_size > 0",
        "num_layers > 0",
        "num_heads > 0",
        "vocab_size > 0",
    ] {
        assert!(
            failed.iter().any(|f| f == name),
            "{name} should fail: {failed:?}"
        );
    }
}

#[test]
fn indivisible_heads_are_named() {
    let v = analyze(
        r#"{"hidden_size":30,"num_hidden_layers":1,"num_attention_heads":4,
            "num_key_value_heads":3,"vocab_size":10}"#,
    );
    assert_eq!(v["all_checks_pass"], false);
    let failed = failed_checks(&v);
    assert!(
        failed.contains(&"hidden_size % num_heads == 0".to_string()),
        "{failed:?}"
    );
    assert!(
        failed.contains(&"num_heads % num_kv_heads == 0".to_string()),
        "{failed:?}"
    );
}

#[test]
fn odd_head_dim_fails_only_the_rope_check() {
    let v = analyze(
        r#"{"hidden_size":12,"num_hidden_layers":1,"num_attention_heads":4,"vocab_size":10}"#,
    );
    assert_eq!(v["head_dim"], 3);
    assert_eq!(failed_checks(&v), vec!["head_dim % 2 == 0".to_string()]);
}

#[test]
fn missing_file_and_bad_json_are_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(analyze_config(&dir.path().join("absent.json")).is_err());
    let p = dir.path().join("bad.json");
    std::fs::write(&p, "{not json").expect("write");
    assert!(analyze_config(&p).is_err());
}
