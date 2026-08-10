//! `apr dry-sampling-lint` — CRUX-C-23 DRY-sampling observation linter.
//!
//! Reads a JSON observation file that captures a single DRY-sampling run and
//! dispatches five classifiers (params, identity, match_len, penalty,
//! monotone). Emits a text or `--json` report.
//!
//! Spec: `contracts/crux-C-23-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.
//!
//! Observation schema (top-level keys; all optional — missing sections skip
//! the corresponding classifier):
//!
//!   {
//!     "params":    { "multiplier": 0.8, "base": 1.75, "allowed_length": 2 },
//!     "identity":  { "logits_before": [0.1, 0.5], "logits_after": [0.1, 0.5],
//!                    "multiplier": 0.0 },
//!     "match_len": { "ctx": [1,2,3,1,2], "candidate": 3,
//!                    "seq_breakers": [], "expected_match_len": 3 },
//!     "penalty":   { "match_len": 5, "allowed_length": 2,
//!                    "multiplier": 0.8, "base": 1.75 },
//!     "monotone":  { "match_len_a": 2, "match_len_b": 5,
//!                    "allowed_length": 2, "multiplier": 0.8, "base": 1.75 }
//!   }

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::commands::dry_sampling_classifier as clf;
use crate::commands::lint_vacuity::{assert_not_vacuous, skipped_label, SectionRun};
use crate::error::{CliError, Result};

/// Each classifier reads exactly one top-level section of the same name.
static SECTION_NAMES: [&str; 5] = ["params", "identity", "match_len", "penalty", "monotone"];

pub(crate) fn run(observation_file: &Path, json: bool) -> Result<()> {
    if !observation_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(observation_file)));
    }

    let body = std::fs::read_to_string(observation_file)?;
    let obs: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr dry-sampling-lint: failed to parse JSON from {}: {e}",
            observation_file.display()
        ))
    })?;

    let params = classify_params(&obs);
    let identity = classify_identity(&obs);
    let match_len = classify_match_len(&obs);
    let penalty = classify_penalty(&obs);
    let monotone = classify_monotone(&obs);

    let mut fail_reasons: Vec<String> = [
        params.as_ref().and_then(params_fail_reason),
        identity.as_ref().and_then(identity_fail_reason),
        match_len.as_ref().and_then(match_len_fail_reason),
        penalty.as_ref().and_then(penalty_fail_reason),
        monotone.as_ref().and_then(monotone_fail_reason),
    ]
    .into_iter()
    .flatten()
    .collect();

    print_report(
        observation_file,
        &obs,
        params.as_ref(),
        identity.as_ref(),
        match_len.as_ref(),
        penalty.as_ref(),
        monotone.as_ref(),
        json,
    );

    // An all-skipped run asserts nothing about DRY sampling. Note this also
    // catches the case where `params.base = 0.5` (a real violation) was hidden
    // because an unrelated sibling field carried the wrong JSON type.
    let ran = [
        params.is_some(),
        identity.is_some(),
        match_len.is_some(),
        penalty.is_some(),
        monotone.is_some(),
    ];
    let sections: Vec<SectionRun> = SECTION_NAMES
        .iter()
        .zip(ran)
        .map(|(name, ran)| SectionRun {
            name,
            keys: std::slice::from_ref(name),
            ran,
        })
        .collect();
    if let Err(reason) = assert_not_vacuous("FALSIFY-CRUX-C-23", &obs, &sections) {
        fail_reasons.push(reason);
    }

    if fail_reasons.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(fail_reasons.join("; ")))
    }
}

fn classify_params(obs: &Value) -> Option<clf::DryParamOutcome> {
    let sec = obs.get("params")?.as_object()?;
    let multiplier = sec.get("multiplier")?.as_f64()?;
    let base = sec.get("base")?.as_f64()?;
    let allowed_length = u32::try_from(sec.get("allowed_length")?.as_u64()?).ok()?;
    Some(clf::classify_dry_params(multiplier, base, allowed_length))
}

fn classify_identity(obs: &Value) -> Option<clf::IdentityOutcome> {
    let sec = obs.get("identity")?.as_object()?;
    let before: Vec<f64> = sec
        .get("logits_before")?
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    let after: Vec<f64> = sec
        .get("logits_after")?
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    let multiplier = sec.get("multiplier")?.as_f64()?;
    Some(clf::classify_dry_identity_zero_multiplier(
        &before, &after, multiplier,
    ))
}

/// Outcome wrapper for match_len gate (comparison with declared expected value).
#[derive(Debug)]
pub(crate) enum MatchLenOutcome {
    Ok { match_len: u32 },
    Mismatch { expected: u32, actual: u32 },
}

fn classify_match_len(obs: &Value) -> Option<MatchLenOutcome> {
    let sec = obs.get("match_len")?.as_object()?;
    let ctx: Vec<u32> = sec
        .get("ctx")?
        .as_array()?
        .iter()
        .filter_map(|v| u32::try_from(v.as_u64()?).ok())
        .collect();
    let candidate = u32::try_from(sec.get("candidate")?.as_u64()?).ok()?;
    let seq_breakers: HashSet<u32> = sec
        .get("seq_breakers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| u32::try_from(v.as_u64()?).ok())
                .collect()
        })
        .unwrap_or_default();
    let expected = u32::try_from(sec.get("expected_match_len")?.as_u64()?).ok()?;
    let actual = clf::classify_dry_match_len(&ctx, candidate, &seq_breakers);
    if actual == expected {
        Some(MatchLenOutcome::Ok { match_len: actual })
    } else {
        Some(MatchLenOutcome::Mismatch { expected, actual })
    }
}

fn classify_penalty(obs: &Value) -> Option<clf::PenaltyOutcome> {
    let sec = obs.get("penalty")?.as_object()?;
    let match_len = u32::try_from(sec.get("match_len")?.as_u64()?).ok()?;
    let allowed_length = u32::try_from(sec.get("allowed_length")?.as_u64()?).ok()?;
    let multiplier = sec.get("multiplier")?.as_f64()?;
    let base = sec.get("base")?.as_f64()?;
    Some(clf::classify_dry_penalty(
        match_len,
        allowed_length,
        multiplier,
        base,
    ))
}

fn classify_monotone(obs: &Value) -> Option<clf::MonotonicityOutcome> {
    let sec = obs.get("monotone")?.as_object()?;
    let a = u32::try_from(sec.get("match_len_a")?.as_u64()?).ok()?;
    let b = u32::try_from(sec.get("match_len_b")?.as_u64()?).ok()?;
    let allowed_length = u32::try_from(sec.get("allowed_length")?.as_u64()?).ok()?;
    let multiplier = sec.get("multiplier")?.as_f64()?;
    let base = sec.get("base")?.as_f64()?;
    Some(clf::classify_dry_penalty_monotone_in_match_len(
        a,
        b,
        allowed_length,
        multiplier,
        base,
    ))
}

fn params_fail_reason(o: &clf::DryParamOutcome) -> Option<String> {
    match o {
        clf::DryParamOutcome::Valid => None,
        clf::DryParamOutcome::NotFinite { field } => Some(format!(
            "FALSIFY-CRUX-C-23-001 params: {field} is not finite"
        )),
        clf::DryParamOutcome::MultiplierNegative { multiplier } => Some(format!(
            "FALSIFY-CRUX-C-23-001 params: multiplier={multiplier} < 0.0"
        )),
        clf::DryParamOutcome::BaseBelowOne { base } => {
            Some(format!("FALSIFY-CRUX-C-23-001 params: base={base} < 1.0"))
        }
        clf::DryParamOutcome::AllowedLengthZero => {
            Some("FALSIFY-CRUX-C-23-001 params: allowed_length == 0".to_string())
        }
    }
}

fn identity_fail_reason(o: &clf::IdentityOutcome) -> Option<String> {
    match o {
        clf::IdentityOutcome::Ok => None,
        clf::IdentityOutcome::InvalidInput { reason } => Some(format!(
            "FALSIFY-CRUX-C-23-001 identity: invalid input: {reason}"
        )),
        clf::IdentityOutcome::LogitsChanged {
            first_diff_index,
            before,
            after,
        } => Some(format!(
            "FALSIFY-CRUX-C-23-001 identity: logit changed at idx {first_diff_index}: before={before} after={after}"
        )),
    }
}

fn match_len_fail_reason(o: &MatchLenOutcome) -> Option<String> {
    match o {
        MatchLenOutcome::Ok { .. } => None,
        MatchLenOutcome::Mismatch { expected, actual } => Some(format!(
            "FALSIFY-CRUX-C-23-002 match_len: expected={expected} actual={actual}"
        )),
    }
}

fn penalty_fail_reason(o: &clf::PenaltyOutcome) -> Option<String> {
    match o {
        clf::PenaltyOutcome::Ok { .. } => None,
        clf::PenaltyOutcome::InvalidInput { reason } => Some(format!(
            "FALSIFY-CRUX-C-23-002 penalty: invalid input: {reason}"
        )),
        clf::PenaltyOutcome::Negative { penalty } => Some(format!(
            "FALSIFY-CRUX-C-23-002 penalty: penalty={penalty} < 0.0"
        )),
    }
}

fn monotone_fail_reason(o: &clf::MonotonicityOutcome) -> Option<String> {
    match o {
        clf::MonotonicityOutcome::Ok => None,
        clf::MonotonicityOutcome::InvalidInput { reason } => Some(format!(
            "FALSIFY-CRUX-C-23-002 monotone: invalid input: {reason}"
        )),
        clf::MonotonicityOutcome::Violation {
            match_len_a,
            match_len_b,
            penalty_a,
            penalty_b,
        } => Some(format!(
            "FALSIFY-CRUX-C-23-002 monotone: violation a={match_len_a}(p={penalty_a}) > b={match_len_b}(p={penalty_b})"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    path: &Path,
    obs: &Value,
    params: Option<&clf::DryParamOutcome>,
    identity: Option<&clf::IdentityOutcome>,
    match_len: Option<&MatchLenOutcome>,
    penalty: Option<&clf::PenaltyOutcome>,
    monotone: Option<&clf::MonotonicityOutcome>,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "observation_path": path.display().to_string(),
            "params":    params.map(|o| format!("{o:?}")),
            "identity":  identity.map(|o| format!("{o:?}")),
            "match_len": match_len.map(|o| format!("{o:?}")),
            "penalty":   penalty.map(|o| format!("{o:?}")),
            "monotone":  monotone.map(|o| format!("{o:?}")),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("dry-sampling-lint report for {}", path.display());
        print_line(
            "  params:    ",
            params.map(|o| format!("{o:?}")),
            obs,
            "params",
        );
        print_line(
            "  identity:  ",
            identity.map(|o| format!("{o:?}")),
            obs,
            "identity",
        );
        print_line(
            "  match_len: ",
            match_len.map(|o| format!("{o:?}")),
            obs,
            "match_len",
        );
        print_line(
            "  penalty:   ",
            penalty.map(|o| format!("{o:?}")),
            obs,
            "penalty",
        );
        print_line(
            "  monotone:  ",
            monotone.map(|o| format!("{o:?}")),
            obs,
            "monotone",
        );
    }
}

fn print_line(prefix: &str, v: Option<String>, obs: &Value, key: &str) {
    match v {
        Some(s) => println!("{prefix}{s}"),
        None => println!("{prefix}{}", skipped_label(obs, &[key])),
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
        let err = run(Path::new("/no/such/dry.json"), false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    #[test]
    fn invalid_json_is_invalid_format() {
        let f = write_obs("not json");
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }

    // This test asserted `is_ok()` on `{}` and its comment ("no fail reasons →
    // Ok") documented the defect as intended behaviour. A run in which no
    // classifier reached a verdict has checked nothing.
    #[test]
    fn falsifier_empty_object_is_rejected() {
        let f = write_obs("{}");
        let err = run(f.path(), false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(
                msg.contains("has none of params/identity/match_len/penalty/monotone"),
                "{msg}"
            ),
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn falsifier_unrelated_body_is_rejected() {
        let f = write_obs(r#"{"unrelated": 123, "nonsense": "hello"}"#);
        assert!(matches!(
            run(f.path(), false).unwrap_err(),
            CliError::ValidationFailed(_)
        ));
    }

    // `base: 0.5` violates the DRY contract in all three bodies below. Only
    // the well-typed one used to be caught; a wrong-typed *sibling* field
    // suppressed the whole gate.
    #[test]
    fn falsifier_wrong_typed_sibling_does_not_suppress_a_real_violation() {
        let ok_types =
            write_obs(r#"{"params": {"multiplier": 0.8, "base": 0.5, "allowed_length": 2}}"#);
        match run(ok_types.path(), false).unwrap_err() {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("base=0.5 < 1.0"), "control: {msg}");
            }
            other => panic!("control: expected ValidationFailed, got {other:?}"),
        }

        for body in [
            r#"{"params": {"multiplier": 0.8, "base": 0.5, "allowed_length": "2"}}"#,
            r#"{"params": {"multiplier": 0.8, "base": 0.5, "allowed_length": -2}}"#,
            r#"{"params": {"multiplier": "oops", "base": 0.5, "allowed_length": 2}}"#,
        ] {
            let f = write_obs(body);
            match run(f.path(), false).unwrap_err() {
                CliError::ValidationFailed(msg) => {
                    assert!(msg.contains("present but unusable"), "{body}: {msg}");
                    assert!(msg.contains("params"), "{body}: {msg}");
                }
                other => panic!("{body}: expected ValidationFailed, got {other:?}"),
            }
        }
    }

    #[test]
    fn params_gate_valid_passes() {
        let f = write_obs(r#"{"params": {"multiplier": 0.8, "base": 1.75, "allowed_length": 2}}"#);
        assert!(run(f.path(), false).is_ok());
    }

    #[test]
    fn params_gate_negative_multiplier_fails() {
        let f = write_obs(r#"{"params": {"multiplier": -1.0, "base": 1.75, "allowed_length": 2}}"#);
        let err = run(f.path(), false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("FALSIFY-CRUX-C-23-001")),
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn params_gate_base_below_one_fails() {
        let f = write_obs(r#"{"params": {"multiplier": 0.8, "base": 0.5, "allowed_length": 2}}"#);
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }

    #[test]
    fn identity_gate_zero_multiplier_unchanged_passes() {
        let f = write_obs(
            r#"{"identity": {"logits_before": [0.1, 0.5], "logits_after": [0.1, 0.5], "multiplier": 0.0}}"#,
        );
        assert!(run(f.path(), false).is_ok());
    }

    #[test]
    fn identity_gate_changed_logits_fail() {
        let f = write_obs(
            r#"{"identity": {"logits_before": [0.1, 0.5], "logits_after": [0.9, 0.5], "multiplier": 0.0}}"#,
        );
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }

    #[test]
    fn match_len_gate_correct_passes() {
        let f = write_obs(
            r#"{"match_len": {"ctx": [1,2,3,1,2], "candidate": 3, "seq_breakers": [], "expected_match_len": 2}}"#,
        );
        // The expected value may or may not match the classifier; assert it runs.
        let _ = run(f.path(), false);
    }

    #[test]
    fn match_len_gate_wrong_expected_fails() {
        let f = write_obs(
            r#"{"match_len": {"ctx": [1,2,3], "candidate": 9, "seq_breakers": [], "expected_match_len": 99}}"#,
        );
        let err = run(f.path(), false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("FALSIFY-CRUX-C-23-002")),
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn penalty_gate_runs() {
        let f = write_obs(
            r#"{"penalty": {"match_len": 5, "allowed_length": 2, "multiplier": 0.8, "base": 1.75}}"#,
        );
        let _ = run(f.path(), false);
    }

    #[test]
    fn monotone_gate_runs_json_mode() {
        let f = write_obs(
            r#"{"monotone": {"match_len_a": 2, "match_len_b": 5, "allowed_length": 2, "multiplier": 0.8, "base": 1.75}}"#,
        );
        let _ = run(f.path(), true);
    }
}
