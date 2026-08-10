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
use crate::error::{CliError, Result};

pub(crate) fn run(observation_file: &Path, json: bool) -> Result<()> {
    if !observation_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(observation_file)));
    }

    let body = std::fs::read_to_string(observation_file)?;
    let obs: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidInput(format!(
            "apr gbnf-lint: failed to parse JSON from {}: {e}",
            observation_file.display()
        ))
    })?;

    let json_out = classify_json(&obs);
    let err_diag = classify_error_diagnostic(&obs);
    let masking = classify_masking(&obs);

    let fail_reasons: Vec<String> = [
        json_out.as_ref().and_then(json_fail_reason),
        err_diag.as_ref().and_then(error_fail_reason),
        masking.as_ref().and_then(masking_fail_reason),
    ]
    .into_iter()
    .flatten()
    .collect();

    print_report(
        observation_file,
        json_out.as_ref(),
        err_diag.as_ref(),
        masking.as_ref(),
        json,
    );

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
        print_line("  json:        ", json_out.map(|o| format!("{o:?}")));
        print_line("  diagnostic:  ", err_diag.map(|o| format!("{o:?}")));
        print_line("  masking:     ", masking.map(|o| format!("{o:?}")));
    }
}

fn print_line(prefix: &str, v: Option<String>) {
    match v {
        Some(s) => println!("{prefix}{s}"),
        None => println!("{prefix}(missing fields — classifier skipped)"),
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
        assert!(matches!(err, CliError::InvalidInput(_)));
    }

    #[test]
    fn empty_object_passes_no_gates() {
        let f = write_obs("{}");
        assert!(run(f.path(), false).is_ok());
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
