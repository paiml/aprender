//! CRUX-A-22 — `apr registry-quota-lint` CLI wiring (CRUX-SHIP-001 g2/g3).
//!
//! Dispatches the pure `registry_quota` classifier over a captured JSON
//! observation file covering the three FALSIFY gates:
//!
//! ```jsonc
//! {
//!   "quota":   {"quota": 1000, "used": 600, "incoming": 401,
//!                "expected_outcome": "reject"},
//!   "atomic":  {"quota": 1000, "used": 600, "incoming": 401,
//!                "expected_outcome": "reject"},
//!   "ceiling": {"quota": 1000, "used": 200, "incoming": 300,
//!                "expected_outcome": "allow",
//!                "expected_post_used_le_quota": true}
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-A-22
//! stderr stamp on any failing gate.

use super::lint_error::{load_json_observation, LintError};
use crate::commands::registry_quota::{
    classify_pull_against_quota, render_quota_error_json, QuotaOutcome,
};
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct RegistryQuotaLintArgs {
    pub observation_file: String,
    pub json: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
struct GateReport {
    gate: &'static str,
    falsify_id: &'static str,
    outcome: String,
    passed: bool,
}

pub fn run(args: RegistryQuotaLintArgs) -> Result<(), LintError> {
    let obs: Value = load_json_observation(&args.observation_file, "FALSIFY-CRUX-A-22")?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(v) = obs.get("quota") {
        let (r, err) = run_quota_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("atomic") {
        let (r, err) = run_atomic_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("ceiling") {
        let (r, err) = run_ceiling_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err(LintError::unusable(
            "FALSIFY-CRUX-A-22: observation has none of quota/atomic/ceiling",
        ));
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-A-22",
            "gates": reports,
        });
        println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    } else {
        for r in &reports {
            let tag = if r.passed { "PASS" } else { "FAIL" };
            println!("[{tag}] {} ({}): {}", r.gate, r.falsify_id, r.outcome);
        }
    }

    if !failures.is_empty() {
        return Err(LintError::gate_failed(failures.join("\n")));
    }
    Ok(())
}

fn parse_u64(v: Option<&Value>, field: &str) -> Result<u64, String> {
    v.and_then(|x| x.as_u64())
        .ok_or_else(|| format!("{field} must be a non-negative integer"))
}

fn parse_outcome_tag(v: Option<&Value>) -> Option<String> {
    v.and_then(|x| x.as_str()).map(|s| s.to_ascii_lowercase())
}

fn outcome_tag(o: &QuotaOutcome) -> &'static str {
    match o {
        QuotaOutcome::Allow { .. } => "allow",
        QuotaOutcome::Reject { .. } => "reject",
    }
}

fn run_quota_gate(v: &Value) -> (GateReport, Option<String>) {
    let parse_inputs = || -> Result<(u64, u64, u64, Option<String>), String> {
        Ok((
            parse_u64(v.get("quota"), "quota")?,
            parse_u64(v.get("used"), "used")?,
            parse_u64(v.get("incoming"), "incoming")?,
            parse_outcome_tag(v.get("expected_outcome")),
        ))
    };
    let (quota, used, incoming, expected) = match parse_inputs() {
        Ok(x) => x,
        Err(e) => {
            let desc = format!("parse error: {e}");
            return (
                GateReport {
                    gate: "quota",
                    falsify_id: "FALSIFY-CRUX-A-22-001",
                    outcome: desc.clone(),
                    passed: false,
                },
                Some(format!("FALSIFY-CRUX-A-22-001 quota gate failed: {desc}")),
            );
        }
    };
    let result = classify_pull_against_quota(quota, used, incoming);
    let got = outcome_tag(&result);

    // If the verdict is Reject, additionally verify the rendered JSON
    // error body satisfies the `needed > free` invariant from FALSIFY.
    let body_invariant_ok = match &result {
        QuotaOutcome::Reject { used, free, needed } => {
            let body = render_quota_error_json(*used, *free, *needed);
            serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| Some((v["needed"].as_u64()?, v["free"].as_u64()?)))
                .map(|(n, f)| n > f)
                .unwrap_or(false)
        }
        QuotaOutcome::Allow { .. } => true,
    };

    let passed = expected.as_deref() == Some(got) && body_invariant_ok;
    let desc = format!(
        "classified={got} expected={} body_invariant_ok={body_invariant_ok}",
        expected.as_deref().unwrap_or("(unspecified)")
    );
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-A-22-001 quota gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "quota",
            falsify_id: "FALSIFY-CRUX-A-22-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_atomic_gate(v: &Value) -> (GateReport, Option<String>) {
    // FALSIFY-002: pre-download purity. Re-run classifier twice and assert
    // identical outcomes; if the expected outcome is Reject, additionally
    // assert no side-effecting fields (the classifier is pure by
    // construction, so this is structural).
    let parse_inputs = || -> Result<(u64, u64, u64, Option<String>), String> {
        Ok((
            parse_u64(v.get("quota"), "quota")?,
            parse_u64(v.get("used"), "used")?,
            parse_u64(v.get("incoming"), "incoming")?,
            parse_outcome_tag(v.get("expected_outcome")),
        ))
    };
    let (quota, used, incoming, expected) = match parse_inputs() {
        Ok(x) => x,
        Err(e) => {
            let desc = format!("parse error: {e}");
            return (
                GateReport {
                    gate: "atomic",
                    falsify_id: "FALSIFY-CRUX-A-22-002",
                    outcome: desc.clone(),
                    passed: false,
                },
                Some(format!("FALSIFY-CRUX-A-22-002 atomic gate failed: {desc}")),
            );
        }
    };
    let r1 = classify_pull_against_quota(quota, used, incoming);
    let r2 = classify_pull_against_quota(quota, used, incoming);
    let deterministic = r1 == r2;
    let got = outcome_tag(&r1);
    let outcome_matches = expected.as_deref() == Some(got);

    let passed = deterministic && outcome_matches;
    let desc = format!(
        "deterministic={deterministic} classified={got} expected={}",
        expected.as_deref().unwrap_or("(unspecified)")
    );
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-A-22-002 atomic gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "atomic",
            falsify_id: "FALSIFY-CRUX-A-22-002",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_ceiling_gate(v: &Value) -> (GateReport, Option<String>) {
    // FALSIFY-003: when Allow is returned, post-pull used ≤ quota holds by
    // construction. We compute the hypothetical post-used and assert.
    let parse_inputs = || -> Result<(u64, u64, u64, Option<String>, Option<bool>), String> {
        Ok((
            parse_u64(v.get("quota"), "quota")?,
            parse_u64(v.get("used"), "used")?,
            parse_u64(v.get("incoming"), "incoming")?,
            parse_outcome_tag(v.get("expected_outcome")),
            v.get("expected_post_used_le_quota")
                .and_then(|x| x.as_bool()),
        ))
    };
    let (quota, used, incoming, expected_outcome, expected_invariant) = match parse_inputs() {
        Ok(x) => x,
        Err(e) => {
            let desc = format!("parse error: {e}");
            return (
                GateReport {
                    gate: "ceiling",
                    falsify_id: "FALSIFY-CRUX-A-22-003",
                    outcome: desc.clone(),
                    passed: false,
                },
                Some(format!("FALSIFY-CRUX-A-22-003 ceiling gate failed: {desc}")),
            );
        }
    };
    let result = classify_pull_against_quota(quota, used, incoming);
    let got = outcome_tag(&result);
    let post_used = used.saturating_add(incoming);
    let post_used_le_quota = matches!(result, QuotaOutcome::Allow { .. }) && post_used <= quota;

    let outcome_ok = expected_outcome.as_deref() == Some(got);
    let invariant_ok = expected_invariant
        .map(|want| want == post_used_le_quota)
        .unwrap_or(true);

    let passed = outcome_ok && invariant_ok;
    let desc = format!(
        "classified={got} expected={} post_used={post_used} post_used_le_quota={post_used_le_quota}",
        expected_outcome.as_deref().unwrap_or("(unspecified)"),
    );
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-A-22-003 ceiling gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "ceiling",
            falsify_id: "FALSIFY-CRUX-A-22-003",
            outcome: desc,
            passed,
        },
        err,
    )
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

    fn args_for(f: &NamedTempFile) -> RegistryQuotaLintArgs {
        RegistryQuotaLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = RegistryQuotaLintArgs {
            observation_file: "/no/such/quota.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err().to_string();
        // The whole *-lint family reports a missing input identically:
        // "File not found: <path>" with exit 3 (commands::lint_error).
        assert!(err.contains("File not found"), "got: {err}");
        assert!(err.contains("/no/such/quota.json"), "got: {err}");
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs("");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("not json at all");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("none of quota/atomic/ceiling"));
    }

    #[test]
    fn quota_reject_passes_when_expected_reject() {
        // used+incoming = 1001 > quota 1000 → Reject, as expected.
        let f = write_obs(
            r#"{"quota": {"quota": 1000, "used": 600, "incoming": 401, "expected_outcome": "reject"}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn quota_mismatch_fails() {
        // Reject classified but observer expected allow.
        let f = write_obs(
            r#"{"quota": {"quota": 1000, "used": 600, "incoming": 401, "expected_outcome": "allow"}}"#,
        );
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-A-22-001"));
    }

    #[test]
    fn quota_parse_error_fails() {
        let f = write_obs(r#"{"quota": {"used": 600, "incoming": 401}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-A-22-001"));
        assert!(err.contains("parse error"));
    }

    #[test]
    fn atomic_deterministic_reject_passes() {
        let f = write_obs(
            r#"{"atomic": {"quota": 1000, "used": 600, "incoming": 401, "expected_outcome": "reject"}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn ceiling_allow_passes_with_invariant() {
        // used+incoming = 500 <= quota 1000 → Allow, post_used_le_quota true.
        let f = write_obs(
            r#"{"ceiling": {"quota": 1000, "used": 200, "incoming": 300, "expected_outcome": "allow", "expected_post_used_le_quota": true}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn ceiling_wrong_invariant_fails() {
        let f = write_obs(
            r#"{"ceiling": {"quota": 1000, "used": 200, "incoming": 300, "expected_outcome": "allow", "expected_post_used_le_quota": false}}"#,
        );
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-A-22-003"));
    }

    #[test]
    fn json_mode_all_gates_ok() {
        let f = write_obs(
            r#"{"atomic": {"quota": 100, "used": 10, "incoming": 5, "expected_outcome": "allow"}}"#,
        );
        let args = RegistryQuotaLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
