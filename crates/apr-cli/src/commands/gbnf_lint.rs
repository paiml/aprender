//! `apr gbnf-lint` — CRUX-C-10 grammar-constrained output linter.
//!
//! Reads a JSON observation file that captures a single grammar-constrained
//! run and dispatches three classifiers (json output, grammar-error
//! diagnostic, illegal-token masking). Emits a text or `--json` report.
//!
//! Spec: `contracts/crux-C-10-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.
//!
//! Observation schema (top-level keys; all optional — missing fields skip
//! the corresponding classifier):
//!
//!   {
//!     "output":        "{...}",                    // json gate input 1
//!     "finish_reason": "stop",                     // json gate input 2
//!     "grammar_error": {                           // error gate
//!       "exit_code": 1,
//!       "stderr":    "invalid grammar at line 1"
//!     },
//!     "masking": {                                 // masking gate
//!       "logits":     [1.0, null, 2.0],            // null == -Infinity
//!       "legal_mask": [true, false, true]
//!     }
//!   }

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::commands::gbnf_classifier as clf;
use crate::commands::lint_vacuity::{assert_not_vacuous, skipped_label, SectionRun};
use crate::error::{CliError, Result};

/// Top-level observation keys each classifier reads, in report order.
const JSON_KEYS: &[&str] = &["output", "finish_reason"];
const DIAGNOSTIC_KEYS: &[&str] = &["grammar_error"];
const MASKING_KEYS: &[&str] = &["masking"];

pub(crate) fn run(observation_file: &Path, json: bool) -> Result<()> {
    if !observation_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(observation_file)));
    }

    let body = std::fs::read_to_string(observation_file)?;
    let obs: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr gbnf-lint: failed to parse JSON from {}: {e}",
            observation_file.display()
        ))
    })?;

    let json_out = classify_json(&obs);
    let err_diag = classify_error_diagnostic(&obs);
    let masking = classify_masking(&obs);

    let mut fail_reasons: Vec<String> = [
        json_out.as_ref().and_then(json_fail_reason),
        err_diag.as_ref().and_then(error_fail_reason),
        masking.as_ref().and_then(masking_fail_reason),
    ]
    .into_iter()
    .flatten()
    .collect();

    print_report(
        observation_file,
        &obs,
        json_out.as_ref(),
        err_diag.as_ref(),
        masking.as_ref(),
        json,
    );

    // A run in which no gate reached a verdict has proved nothing about the
    // grammar-constrained decode; it must not exit 0.
    if let Err(reason) = assert_not_vacuous(
        "FALSIFY-CRUX-C-10",
        &obs,
        &[
            SectionRun {
                name: "json",
                keys: JSON_KEYS,
                ran: json_out.is_some(),
            },
            SectionRun {
                name: "diagnostic",
                keys: DIAGNOSTIC_KEYS,
                ran: err_diag.is_some(),
            },
            SectionRun {
                name: "masking",
                keys: MASKING_KEYS,
                ran: masking.is_some(),
            },
        ],
    ) {
        fail_reasons.push(reason);
    }

    if fail_reasons.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(fail_reasons.join("; ")))
    }
}

fn classify_json(obs: &Value) -> Option<clf::JsonGrammarOutputOutcome> {
    let output = obs.get("output")?.as_str()?;
    let finish = obs.get("finish_reason")?.as_str()?;
    Some(clf::classify_json_grammar_output(output, finish))
}

fn classify_error_diagnostic(obs: &Value) -> Option<clf::GrammarErrorDiagnosticOutcome> {
    let ge = obs.get("grammar_error")?.as_object()?;
    let exit_code = ge.get("exit_code")?.as_i64()? as i32;
    let stderr = ge.get("stderr")?.as_str()?;
    Some(clf::classify_grammar_error_diagnostic(exit_code, stderr))
}

fn classify_masking(obs: &Value) -> Option<clf::IllegalTokenMaskingOutcome> {
    let m = obs.get("masking")?.as_object()?;
    let logits_raw = m.get("logits")?.as_array()?;
    let mask_raw = m.get("legal_mask")?.as_array()?;

    let logits: Vec<f32> = logits_raw
        .iter()
        .map(|v| match v {
            Value::Null => f32::NEG_INFINITY,
            Value::Number(n) => n.as_f64().map(|x| x as f32).unwrap_or(f32::NAN),
            _ => f32::NAN,
        })
        .collect();
    let mask: Vec<bool> = mask_raw.iter().filter_map(|v| v.as_bool()).collect();

    if mask.len() != mask_raw.len() {
        return None;
    }
    Some(clf::classify_illegal_token_masking(&logits, &mask))
}

fn json_fail_reason(o: &clf::JsonGrammarOutputOutcome) -> Option<String> {
    match o {
        clf::JsonGrammarOutputOutcome::Ok => None,
        clf::JsonGrammarOutputOutcome::EmptyOutput => {
            Some("FALSIFY-CRUX-C-10-001 json: empty output string".to_string())
        }
        clf::JsonGrammarOutputOutcome::NotJson { error } => Some(format!(
            "FALSIFY-CRUX-C-10-001 json: output does not parse as JSON: {error}"
        )),
        clf::JsonGrammarOutputOutcome::WrongFinishReason { got } => Some(format!(
            "FALSIFY-CRUX-C-10-001 json: finish_reason={got:?} not in {{stop, length}}"
        )),
    }
}

fn error_fail_reason(o: &clf::GrammarErrorDiagnosticOutcome) -> Option<String> {
    match o {
        clf::GrammarErrorDiagnosticOutcome::Ok => None,
        clf::GrammarErrorDiagnosticOutcome::ZeroExitCode => Some(
            "FALSIFY-CRUX-C-10-002 diagnostic: malformed grammar silently accepted (exit 0)"
                .to_string(),
        ),
        clf::GrammarErrorDiagnosticOutcome::MissingGrammarDiagnostic { stderr_snippet } => {
            Some(format!(
                "FALSIFY-CRUX-C-10-002 diagnostic: stderr missing 'grammar' keyword; snippet={stderr_snippet:?}"
            ))
        }
    }
}

fn masking_fail_reason(o: &clf::IllegalTokenMaskingOutcome) -> Option<String> {
    match o {
        clf::IllegalTokenMaskingOutcome::Ok => None,
        clf::IllegalTokenMaskingOutcome::LengthMismatch {
            logits_len,
            mask_len,
        } => Some(format!(
            "FALSIFY-CRUX-C-10-001 masking: length mismatch logits={logits_len} mask={mask_len}"
        )),
        clf::IllegalTokenMaskingOutcome::NoLegalTokens => Some(
            "FALSIFY-CRUX-C-10-001 masking: legal_mask has no legal positions".to_string(),
        ),
        clf::IllegalTokenMaskingOutcome::IllegalTokenNotMasked {
            token_index,
            logit,
        } => Some(format!(
            "FALSIFY-CRUX-C-10-001 masking: illegal token at idx {token_index} has logit={logit} (expected -Infinity)"
        )),
    }
}

fn print_report(
    path: &Path,
    obs: &Value,
    json_out: Option<&clf::JsonGrammarOutputOutcome>,
    err_diag: Option<&clf::GrammarErrorDiagnosticOutcome>,
    masking: Option<&clf::IllegalTokenMaskingOutcome>,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "observation_path": path.display().to_string(),
            "json":      json_out.map(|o| format!("{o:?}")),
            "diagnostic": err_diag.map(|o| format!("{o:?}")),
            "masking":   masking.map(|o| format!("{o:?}")),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("gbnf-lint report for {}", path.display());
        print_line(
            "  json:        ",
            json_out.map(|o| format!("{o:?}")),
            obs,
            JSON_KEYS,
        );
        print_line(
            "  diagnostic:  ",
            err_diag.map(|o| format!("{o:?}")),
            obs,
            DIAGNOSTIC_KEYS,
        );
        print_line(
            "  masking:     ",
            masking.map(|o| format!("{o:?}")),
            obs,
            MASKING_KEYS,
        );
    }
}

fn print_line(prefix: &str, v: Option<String>, obs: &Value, keys: &[&str]) {
    match v {
        Some(s) => println!("{prefix}{s}"),
        None => println!("{prefix}{}", skipped_label(obs, keys)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_obs(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn missing_file_is_file_not_found() {
        let err = run(Path::new("/no/such/gbnf.json"), false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    #[test]
    fn invalid_json_is_invalid_format() {
        let f = write_obs("garbage");
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }

    // This test used to assert `is_ok()` on `{}` and so held the defect in
    // place: an observation that engages no classifier proves nothing about
    // the decode and must not exit 0.
    #[test]
    fn falsifier_empty_object_is_rejected() {
        let f = write_obs("{}");
        let err = run(f.path(), false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("has none of json/diagnostic/masking"), "{msg}");
            }
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn falsifier_unrelated_body_is_rejected() {
        let f = write_obs(r#"{"hello": "world", "nested": {"a": 1}}"#);
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }

    #[test]
    fn falsifier_null_observation_is_rejected() {
        let f = write_obs("null");
        let err = run(f.path(), true).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("not a JSON object"), "{msg}"),
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    // 0/1 and true/false express the same legal-token mask. The int form used
    // to suppress the masking gate entirely, hiding a real violation.
    #[test]
    fn falsifier_int_legal_mask_does_not_suppress_the_masking_gate() {
        let bools = write_obs(
            r#"{"masking": {"logits": [1.0, 5.0, 2.0], "legal_mask": [true,false,true]}}"#,
        );
        let ints = write_obs(r#"{"masking": {"logits": [1.0, 5.0, 2.0], "legal_mask": [1,0,1]}}"#);
        assert!(
            run(bools.path(), false).is_err(),
            "control: bool mask must fail"
        );
        let err = run(ints.path(), false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("present but unusable"), "{msg}")
            }
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn falsifier_wrong_typed_finish_reason_is_rejected() {
        let f = write_obs(r#"{"output": "{}", "finish_reason": 7}"#);
        let err = run(f.path(), false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("present but unusable"), "{msg}");
                assert!(msg.contains("json"), "{msg}");
            }
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn json_gate_valid_json_stop_passes() {
        let f = write_obs(r#"{"output": "{\"x\": 1}", "finish_reason": "stop"}"#);
        assert!(run(f.path(), false).is_ok());
    }

    #[test]
    fn json_gate_non_json_output_fails() {
        let f = write_obs(r#"{"output": "not json at all", "finish_reason": "stop"}"#);
        let err = run(f.path(), false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("FALSIFY-CRUX-C-10-001")),
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn masking_gate_legal_finite_illegal_neg_inf_passes() {
        // logits: legal positions finite, illegal position is null (-inf).
        let f = write_obs(
            r#"{"masking": {"logits": [1.0, null, 2.0], "legal_mask": [true, false, true]}}"#,
        );
        assert!(run(f.path(), false).is_ok());
    }

    #[test]
    fn masking_gate_illegal_token_not_masked_fails() {
        // An illegal token (mask false) left at a finite logit violates masking.
        let f = write_obs(r#"{"masking": {"logits": [1.0, 2.0], "legal_mask": [true, false]}}"#);
        let err = run(f.path(), false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("masking: illegal token")),
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn json_mode_runs() {
        let f = write_obs(r#"{"output": "{}", "finish_reason": "stop"}"#);
        assert!(run(f.path(), true).is_ok());
    }
}
