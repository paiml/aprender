//! `apr typical-p-lint` — CRUX-C-22 typical-p sampling observation linter.
//!
//! Reads a JSON observation file that captures a single typical-p sampling
//! run and dispatches five classifiers (range, identity, mass coverage,
//! sort order, renormalization). Emits a text or `--json` report.
//!
//! Spec: `contracts/crux-C-22-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.
//!
//! Observation schema (top-level keys; all optional — missing sections
//! skip the corresponding classifier):
//!
//!   {
//!     "range":    { "p": 0.95 },
//!     "identity": { "kept_indices": [0,1,2], "total_tokens": 3, "p": 1.0 },
//!     "mass":     { "kept_probs": [0.5, 0.3, 0.15], "p": 0.9 },
//!     "sort":     { "all_probs": [0.5, 0.3, 0.2],
//!                   "kept_probs_in_sort_order": [0.3, 0.2] },
//!     "renorm":   { "filtered_probs": [0.6, 0.4] }
//!   }

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::commands::typical_p_classifier as clf;
use crate::error::{CliError, Result};

pub(crate) fn run(observation_file: &Path, json: bool) -> Result<()> {
    if !observation_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(observation_file)));
    }

    let body = std::fs::read_to_string(observation_file)?;
    let obs: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr typical-p-lint: failed to parse JSON from {}: {e}",
            observation_file.display()
        ))
    })?;

    let range = classify_range(&obs);
    let identity = classify_identity(&obs);
    let mass = classify_mass(&obs);
    let sort = classify_sort(&obs);
    let renorm = classify_renorm(&obs);

    let fail_reasons: Vec<String> = [
        range.as_ref().and_then(range_fail_reason),
        identity.as_ref().and_then(identity_fail_reason),
        mass.as_ref().and_then(mass_fail_reason),
        sort.as_ref().and_then(sort_fail_reason),
        renorm.as_ref().and_then(renorm_fail_reason),
    ]
    .into_iter()
    .flatten()
    .collect();

    print_report(
        observation_file,
        range.as_ref(),
        identity.as_ref(),
        mass.as_ref(),
        sort.as_ref(),
        renorm.as_ref(),
        json,
    );

    if fail_reasons.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(fail_reasons.join("; ")))
    }
}

fn classify_range(obs: &Value) -> Option<clf::TypicalPRangeOutcome> {
    let sec = obs.get("range")?.as_object()?;
    let p = sec.get("p")?.as_f64()?;
    Some(clf::classify_typical_p_range(p))
}

fn classify_identity(obs: &Value) -> Option<clf::IdentityOutcome> {
    let sec = obs.get("identity")?.as_object()?;
    let kept: Vec<usize> = sec
        .get("kept_indices")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();
    let total = sec.get("total_tokens")?.as_u64()? as usize;
    let p = sec.get("p")?.as_f64()?;
    Some(clf::classify_typical_p_identity(&kept, total, p))
}

fn classify_mass(obs: &Value) -> Option<clf::MassCoverageOutcome> {
    let sec = obs.get("mass")?.as_object()?;
    let kept_probs: Vec<f64> = sec
        .get("kept_probs")?
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    let p = sec.get("p")?.as_f64()?;
    Some(clf::classify_typical_p_mass_coverage(&kept_probs, p))
}

fn classify_sort(obs: &Value) -> Option<clf::SortOrderOutcome> {
    let sec = obs.get("sort")?.as_object()?;
    let all_probs: Vec<f64> = sec
        .get("all_probs")?
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    let kept: Vec<f64> = sec
        .get("kept_probs_in_sort_order")?
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    Some(clf::classify_typical_p_sort_order(&all_probs, &kept))
}

fn classify_renorm(obs: &Value) -> Option<clf::RenormOutcome> {
    let sec = obs.get("renorm")?.as_object()?;
    let filtered: Vec<f64> = sec
        .get("filtered_probs")?
        .as_array()?
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    Some(clf::classify_typical_p_renormalization(&filtered))
}

fn range_fail_reason(o: &clf::TypicalPRangeOutcome) -> Option<String> {
    match o {
        clf::TypicalPRangeOutcome::Valid => None,
        clf::TypicalPRangeOutcome::NotFinite => {
            Some("FALSIFY-CRUX-C-22-001 range: p is not finite".to_string())
        }
        clf::TypicalPRangeOutcome::BelowMinimum { p } => Some(format!(
            "FALSIFY-CRUX-C-22-001 range: p={p} <= 0.0 (must be > 0)"
        )),
        clf::TypicalPRangeOutcome::AboveMaximum { p } => {
            Some(format!("FALSIFY-CRUX-C-22-001 range: p={p} > 1.0"))
        }
    }
}

fn identity_fail_reason(o: &clf::IdentityOutcome) -> Option<String> {
    match o {
        clf::IdentityOutcome::Ok { .. } => None,
        clf::IdentityOutcome::InvalidInput { reason } => Some(format!(
            "FALSIFY-CRUX-C-22-001 identity: invalid input: {reason}"
        )),
        clf::IdentityOutcome::DroppedTokens {
            kept_count,
            total_count,
        } => Some(format!(
            "FALSIFY-CRUX-C-22-001 identity: p=1.0 dropped tokens (kept={kept_count}, total={total_count})"
        )),
    }
}

fn mass_fail_reason(o: &clf::MassCoverageOutcome) -> Option<String> {
    match o {
        clf::MassCoverageOutcome::Ok { .. } => None,
        clf::MassCoverageOutcome::InvalidInput { reason } => Some(format!(
            "FALSIFY-CRUX-C-22-002 mass: invalid input: {reason}"
        )),
        clf::MassCoverageOutcome::InsufficientMass {
            kept_mass,
            required,
        } => Some(format!(
            "FALSIFY-CRUX-C-22-002 mass: kept_mass={kept_mass} < required={required}"
        )),
        clf::MassCoverageOutcome::TooLarge { kept_mass, excess } => Some(format!(
            "FALSIFY-CRUX-C-22-002 mass: kept_mass={kept_mass} > 1.0 (excess={excess})"
        )),
    }
}

fn sort_fail_reason(o: &clf::SortOrderOutcome) -> Option<String> {
    match o {
        clf::SortOrderOutcome::Ok => None,
        clf::SortOrderOutcome::InvalidInput { reason } => Some(format!(
            "FALSIFY-CRUX-C-22-002 sort: invalid input: {reason}"
        )),
        clf::SortOrderOutcome::OutOfOrder {
            at_index,
            prev_c,
            curr_c,
        } => Some(format!(
            "FALSIFY-CRUX-C-22-002 sort: out of order at idx {at_index}: prev_c={prev_c} > curr_c={curr_c}"
        )),
    }
}

fn renorm_fail_reason(o: &clf::RenormOutcome) -> Option<String> {
    match o {
        clf::RenormOutcome::Ok { .. } => None,
        clf::RenormOutcome::InvalidInput { reason } => Some(format!(
            "FALSIFY-CRUX-C-22-002 renorm: invalid input: {reason}"
        )),
        clf::RenormOutcome::NotNormalized { sum, deviation } => Some(format!(
            "FALSIFY-CRUX-C-22-002 renorm: sum={sum} deviates from 1.0 by {deviation} (> 1e-6)"
        )),
        clf::RenormOutcome::ContainsNegative {
            first_bad_index,
            value,
        } => Some(format!(
            "FALSIFY-CRUX-C-22-002 renorm: negative prob at idx {first_bad_index}: {value}"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    path: &Path,
    range: Option<&clf::TypicalPRangeOutcome>,
    identity: Option<&clf::IdentityOutcome>,
    mass: Option<&clf::MassCoverageOutcome>,
    sort: Option<&clf::SortOrderOutcome>,
    renorm: Option<&clf::RenormOutcome>,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "observation_path": path.display().to_string(),
            "range":    range.map(|o| format!("{o:?}")),
            "identity": identity.map(|o| format!("{o:?}")),
            "mass":     mass.map(|o| format!("{o:?}")),
            "sort":     sort.map(|o| format!("{o:?}")),
            "renorm":   renorm.map(|o| format!("{o:?}")),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("typical-p-lint report for {}", path.display());
        print_line("  range:    ", range.map(|o| format!("{o:?}")));
        print_line("  identity: ", identity.map(|o| format!("{o:?}")));
        print_line("  mass:     ", mass.map(|o| format!("{o:?}")));
        print_line("  sort:     ", sort.map(|o| format!("{o:?}")));
        print_line("  renorm:   ", renorm.map(|o| format!("{o:?}")));
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
        let err = run(Path::new("/no/such/tp.json"), false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    #[test]
    fn invalid_json_is_invalid_format() {
        let f = write_obs("nope");
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }

    #[test]
    fn empty_object_passes_no_gates() {
        let f = write_obs("{}");
        assert!(run(f.path(), false).is_ok());
    }

    #[test]
    fn range_gate_valid_p_passes() {
        let f = write_obs(r#"{"range": {"p": 0.95}}"#);
        assert!(run(f.path(), false).is_ok());
    }

    #[test]
    fn range_gate_above_one_fails() {
        let f = write_obs(r#"{"range": {"p": 1.5}}"#);
        let err = run(f.path(), false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => assert!(msg.contains("FALSIFY-CRUX-C-22-001")),
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[test]
    fn range_gate_zero_p_fails() {
        let f = write_obs(r#"{"range": {"p": 0.0}}"#);
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }

    #[test]
    fn identity_gate_keep_all_passes() {
        let f =
            write_obs(r#"{"identity": {"kept_indices": [0,1,2], "total_tokens": 3, "p": 1.0}}"#);
        assert!(run(f.path(), false).is_ok());
    }

    #[test]
    fn identity_gate_dropped_tokens_fail() {
        let f = write_obs(r#"{"identity": {"kept_indices": [0], "total_tokens": 3, "p": 1.0}}"#);
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }

    #[test]
    fn mass_gate_runs() {
        let f = write_obs(r#"{"mass": {"kept_probs": [0.5, 0.3, 0.15], "p": 0.9}}"#);
        let _ = run(f.path(), false);
    }

    #[test]
    fn sort_gate_runs() {
        let f = write_obs(
            r#"{"sort": {"all_probs": [0.5, 0.3, 0.2], "kept_probs_in_sort_order": [0.3, 0.2]}}"#,
        );
        let _ = run(f.path(), false);
    }

    #[test]
    fn renorm_gate_runs_json_mode() {
        let f = write_obs(r#"{"renorm": {"filtered_probs": [0.6, 0.4]}}"#);
        let _ = run(f.path(), true);
    }
}
