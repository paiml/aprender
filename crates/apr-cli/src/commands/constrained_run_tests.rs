//! #3793: `apr run --json-schema` / `--grammar`, the CLI half.

use super::*;
use realizar::constrain::ConstraintError;

fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "integer", "minimum": 0, "maximum": 1000 } },
        "required": ["answer"],
        "additionalProperties": false
    })
}

fn args(json_schema: Option<&str>, grammar: Option<&str>) -> ConstraintArgs {
    ConstraintArgs {
        json_schema: json_schema.map(String::from),
        grammar: grammar.map(String::from),
    }
}

fn refusal(e: CliError) -> ConstraintRefusal {
    match e {
        CliError::ConstraintRefused(r) => r,
        other => panic!("not a constraint refusal: {other:?}"),
    }
}

#[test]
fn no_flag_is_no_constraint() {
    assert_eq!(constraint_request(&args(None, None)).expect("ok"), None);
}

#[test]
fn a_malformed_schema_is_schema_invalid_before_anything_loads() {
    let r =
        refusal(constraint_request(&args(Some(r#"{"type": "obj"#), None)).expect_err("malformed"));
    assert_eq!(r.kind, "SchemaInvalid");
    assert!(r.message.starts_with("SchemaInvalid: "), "{}", r.message);
    let r = refusal(constraint_request(&args(Some("[1, 2]"), None)).expect_err("not a schema"));
    assert_eq!(r.kind, "SchemaInvalid");
    let r = refusal(
        constraint_request(&args(Some("@/nonexistent/schema.json"), None)).expect_err("no file"),
    );
    assert_eq!(r.kind, "SchemaInvalid");
}

#[test]
fn a_schema_inline_or_at_path_is_the_same_request() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("s.json");
    std::fs::write(&path, schema().to_string()).expect("write");
    let inline = constraint_request(&args(Some(&schema().to_string()), None)).expect("inline");
    let file =
        constraint_request(&args(Some(&format!("@{}", path.display())), None)).expect("@path");
    assert_eq!(inline, file);
    assert_eq!(inline, Some(ConstraintRequest::JsonSchema(schema())));
}

#[test]
fn a_grammar_is_read_and_an_empty_one_is_refused() {
    assert_eq!(
        constraint_request(&args(None, Some(r#"start: "yes" | "no""#))).expect("grammar"),
        Some(ConstraintRequest::Lark(
            r#"start: "yes" | "no""#.to_string()
        ))
    );
    let r = refusal(constraint_request(&args(None, Some("   "))).expect_err("empty"));
    assert_eq!(r.kind, "SchemaInvalid");
}

/// Each refusal is keyed on the error's TYPE: its name, what removes it, and the finish reason
/// it ended on.
#[test]
fn every_refusal_has_its_own_name() {
    let cases = [
        (
            ConstraintError::SchemaInvalid("x".into()),
            "SchemaInvalid",
            false,
            None,
        ),
        (
            ConstraintError::SchemaUnsupported("x".into()),
            "SchemaUnsupported",
            true,
            None,
        ),
        (
            ConstraintError::UnsupportedPath {
                path: "gguf-cuda".into(),
                removed_by: "#3568 PR 3".into(),
            },
            "SchemaUnsupportedPath",
            true,
            None,
        ),
        (
            ConstraintError::WithThinking("x".into()),
            "SchemaWithThinking",
            true,
            None,
        ),
        (
            ConstraintError::Violation("x".into()),
            "SchemaViolation",
            false,
            Some("constraint_complete"),
        ),
        (
            ConstraintError::Truncated { max_tokens: 3 },
            "Truncated",
            false,
            Some("length"),
        ),
        (
            ConstraintError::DeadEnd {
                position: 1,
                reason: "x".into(),
            },
            "ConstraintDeadEnd",
            false,
            Some("dead_end"),
        ),
        (
            ConstraintError::NotCompiled,
            "StructuredOutputNotCompiled",
            false,
            None,
        ),
    ];
    for (e, kind, has_removed_by, finish) in cases {
        let r = refusal_of(&e);
        assert_eq!(r.kind, kind);
        assert_eq!(r.removed_by.is_some(), has_removed_by, "{kind}");
        assert_eq!(r.finish_reason, finish, "{kind}");
        assert!(r.message.starts_with(&format!("{kind}: ")), "{}", r.message);
        let line = r.to_string();
        assert!(!line.contains('\n'), "one line: {line:?}");
        assert!(line.matches("removed_by").count() <= 1, "{line}");
    }
}

#[test]
fn a_constraint_refusal_exits_15() {
    let e = refused(ConstraintError::Truncated { max_tokens: 3 });
    assert_eq!(e.exit_code_value(), 15);
}

#[test]
fn a_cut_document_is_truncated_never_a_success() {
    let request = ConstraintRequest::JsonSchema(schema());
    let r = constraint_verdict(&request, Some(FinishReason::Length), r#"{"answer":"#, 3)
        .expect("a refusal");
    assert_eq!(r.kind, "Truncated");
    assert_eq!(r.finish_reason, Some("length"));
}

#[test]
fn a_complete_conforming_document_stands() {
    let request = ConstraintRequest::JsonSchema(schema());
    assert_eq!(
        constraint_verdict(
            &request,
            Some(FinishReason::ConstraintComplete),
            r#"{"answer":4}"#,
            32
        ),
        None
    );
}

/// The second reader shares nothing with the engine, so it catches what the engine let through.
#[test]
fn the_second_reader_refuses_a_nonconforming_document() {
    let request = ConstraintRequest::JsonSchema(schema());
    for bad in [
        r#"{"answer":4000}"#,
        r#"{"answer":"4"}"#,
        r#"{"answer":4,"x":1}"#,
        "{",
        "",
    ] {
        let r = constraint_verdict(&request, Some(FinishReason::ConstraintComplete), bad, 32)
            .unwrap_or_else(|| panic!("{bad:?} must be refused"));
        assert_eq!(r.kind, "SchemaViolation", "{bad:?}");
    }
}

#[test]
fn a_grammar_has_no_second_reader_and_says_nothing_it_did_not_check() {
    let request = ConstraintRequest::Lark(r#"start: "yes" | "no""#.to_string());
    assert_eq!(
        constraint_verdict(&request, Some(FinishReason::ConstraintComplete), "yes", 8),
        None
    );
    // A cut grammar output is still Truncated
    assert_eq!(
        constraint_verdict(&request, Some(FinishReason::Length), "ye", 1).map(|r| r.kind),
        Some("Truncated")
    );
}

#[test]
fn a_constrained_run_that_reports_no_constrained_finish_is_not_a_success() {
    let request = ConstraintRequest::JsonSchema(schema());
    for finish in [None, Some(FinishReason::Stop)] {
        let r = constraint_verdict(&request, finish, r#"{"answer":4}"#, 32).expect("refused");
        assert_eq!(r.kind, "SchemaViolation");
    }
}

/// Parse `apr run m.gguf --prompt x` plus `extra`, on a thread whose stack the Cli fits in (as
/// parsing.rs does): `Ok(())`, or the clap error kind.
fn parse_run(extra: &'static [&'static str]) -> std::result::Result<(), clap::error::ErrorKind> {
    use clap::Parser;
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let mut argv = vec!["apr", "run", "m.gguf", "--prompt", "x"];
            argv.extend_from_slice(extra);
            crate::Cli::try_parse_from(argv)
                .map(|_| ())
                .map_err(|e| e.kind())
        })
        .expect("spawn")
        .join()
        .expect("join")
}

#[test]
fn json_schema_and_grammar_cannot_both_be_given() {
    // The controls: each alone parses, so the refusal below is the conflict and nothing else
    assert_eq!(parse_run(&["--json-schema", "{}"]), Ok(()));
    assert_eq!(parse_run(&["--grammar", "start: \"a\""]), Ok(()));
    assert_eq!(
        parse_run(&["--json-schema", "{}", "--grammar", "start: \"a\""]),
        Err(clap::error::ErrorKind::ArgumentConflict)
    );
}
