//! Unit tests for the Modelfile DSL parser (CRUX-K-11).

#![cfg(test)]

use super::parser::{parse_modelfile_str, ModelfileError};
use std::path::Path;

fn parse(text: &str) -> Result<super::ModelfileConfig, ModelfileError> {
    parse_modelfile_str(text, Path::new("<test>"))
}

#[test]
fn falsify_crux_k_11_001_canonical_roundtrip() {
    // From the contract's FALSIFY-CRUX-K-11-001 test body.
    let input = r#"
FROM tinyllama
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM """
You are a helpful assistant.
"""
"#;
    let cfg = parse(input).expect("parse");
    assert_eq!(cfg.from, "tinyllama");
    assert!(
        (cfg.parameters
            .get("temperature")
            .unwrap()
            .as_f64()
            .unwrap()
            - 0.7)
            .abs()
            < 1e-9
    );
    assert!(
        (cfg.parameters.get("top_p").unwrap().as_f64().unwrap() - 0.9).abs() < 1e-9
    );
    let system = cfg.system.expect("system set");
    assert!(
        system.contains("helpful assistant"),
        "got system: {system}"
    );
}

#[test]
fn falsify_crux_k_11_002_case_insensitive_directives() {
    for directive in ["FROM", "from", "From", "fRoM"] {
        let input = format!("{directive} tinyllama\n");
        let cfg = parse(&input).expect("parse");
        assert_eq!(cfg.from, "tinyllama", "directive variant: {directive}");
    }
}

#[test]
fn falsify_crux_k_11_003_unknown_directive_errors_with_location() {
    let input = "FROM tinyllama\nBOGUS 42\n";
    let err = parse(input).unwrap_err();
    // Contract requires `:LINE:` to appear in the error message
    let s = err.to_string();
    assert!(s.contains(":2:"), "missing line number in error: {s}");
    assert!(
        s.to_lowercase().contains("unknown directive"),
        "missing 'unknown directive': {s}"
    );
}

#[test]
fn missing_from_is_an_error() {
    let input = "PARAMETER temperature 0.5\n";
    let err = parse(input).unwrap_err();
    assert!(err.to_string().contains("FROM"), "got: {}", err);
}

#[test]
fn empty_file_is_an_error() {
    let err = parse("").unwrap_err();
    assert!(err.to_string().contains("FROM"), "got: {}", err);
}

#[test]
fn blank_lines_and_comments_ignored() {
    let input = "\n# a comment\n\nFROM model\n# trailing comment\n";
    let cfg = parse(input).expect("parse");
    assert_eq!(cfg.from, "model");
}

#[test]
fn inline_triple_quoted_block() {
    let input = r#"FROM model
SYSTEM """one-line system"""
"#;
    let cfg = parse(input).expect("parse");
    assert_eq!(cfg.system.unwrap(), "one-line system");
}

#[test]
fn unterminated_triple_quote_is_an_error() {
    let input = "FROM model\nSYSTEM \"\"\"\nstart but never end\n";
    let err = parse(input).unwrap_err();
    assert!(
        err.to_string().contains("unterminated"),
        "got: {}",
        err
    );
}

#[test]
fn parameter_int_is_typed() {
    let input = "FROM m\nPARAMETER num_ctx 4096\n";
    let cfg = parse(input).expect("parse");
    let v = cfg.parameters.get("num_ctx").unwrap();
    assert_eq!(v.as_i64(), Some(4096));
}

#[test]
fn parameter_bool_is_typed() {
    let input = "FROM m\nPARAMETER use_cache true\nPARAMETER trace False\n";
    let cfg = parse(input).expect("parse");
    assert_eq!(cfg.parameters["use_cache"].as_bool(), Some(true));
    assert_eq!(cfg.parameters["trace"].as_bool(), Some(false));
}

#[test]
fn parameter_string_value() {
    let input = r#"FROM m
PARAMETER stop "<|im_end|>"
"#;
    let cfg = parse(input).expect("parse");
    assert_eq!(
        cfg.parameters["stop"].as_str(),
        Some("<|im_end|>"),
        "got: {:?}",
        cfg.parameters["stop"]
    );
}

#[test]
fn message_directive_collects_ordered_pairs() {
    let input = "FROM m\nMESSAGE user hello\nMESSAGE assistant world\n";
    let cfg = parse(input).expect("parse");
    assert_eq!(cfg.messages.len(), 2);
    assert_eq!(cfg.messages[0].role, "user");
    assert_eq!(cfg.messages[0].content, "hello");
    assert_eq!(cfg.messages[1].role, "assistant");
    assert_eq!(cfg.messages[1].content, "world");
}

#[test]
fn license_and_adapter_directives() {
    let input = "FROM m\nLICENSE Apache-2.0\nADAPTER /path/to/lora.safetensors\n";
    let cfg = parse(input).expect("parse");
    assert_eq!(cfg.license.unwrap(), "Apache-2.0");
    assert_eq!(cfg.adapter.unwrap(), "/path/to/lora.safetensors");
}

#[test]
fn template_directive_captures_inline() {
    let input = r#"FROM m
TEMPLATE """{{ .System }} {{ .Prompt }}"""
"#;
    let cfg = parse(input).expect("parse");
    assert_eq!(cfg.template.unwrap(), "{{ .System }} {{ .Prompt }}");
}

#[test]
fn full_modelfile_roundtrips_via_json() {
    let input = r#"FROM tinyllama
PARAMETER temperature 0.7
SYSTEM """
You are helpful.
"""
MESSAGE user hi
MESSAGE assistant hello
LICENSE Apache-2.0
"#;
    let cfg = parse(input).expect("parse");
    let json = serde_json::to_string_pretty(&cfg).expect("serialize");
    let round: super::ModelfileConfig =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(round, cfg);
}
