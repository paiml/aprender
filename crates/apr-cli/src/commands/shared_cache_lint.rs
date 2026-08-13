//! CRUX-A-21 — `apr shared-cache-lint` CLI wiring (CRUX-SHIP-001 g2/g3).
//!
//! Dispatches the pure `shared_cache` classifier over a captured JSON
//! observation file covering the two FALSIFY gates:
//!
//! ```jsonc
//! {
//!   "dedup": {
//!     "apr_models_env": "/var/lib/apr/models",
//!     "home":           "/home/user",
//!     "expected_root":  "/var/lib/apr/models",
//!     "sha256_hex_a":   "a...64 chars",
//!     "sha256_hex_b":   "a...64 chars",
//!     "expected_same_path": true
//!   },
//!   "permission": {
//!     "kind":              "permission_denied",
//!     "expected_outcome":  "permission_denied",
//!     "expected_exit_code": 13,
//!     "expected_hint_substring": "daemon"
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-A-21
//! stderr stamp on any failing gate.

use super::lint_error::{load_json_observation, LintError};
use crate::commands::shared_cache::{
    blob_path_for, classify_pull_permission_outcome, resolve_registry_root, PullPermissionOutcome,
};
use serde_json::Value;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SharedCacheLintArgs {
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

pub fn run(args: SharedCacheLintArgs) -> Result<(), LintError> {
    let obs: Value = load_json_observation(&args.observation_file, "FALSIFY-CRUX-A-21")?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(v) = obs.get("dedup") {
        let (r, err) = run_dedup_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("permission") {
        let (r, err) = run_permission_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err(LintError::unusable(
            "FALSIFY-CRUX-A-21: observation has none of dedup/permission",
        ));
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-A-21",
            "gates": reports,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&payload)
                .expect("the --json report is a locally-built serde_json::Value")
        );
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

fn parse_str(v: Option<&Value>) -> Option<String> {
    v.and_then(|x| x.as_str()).map(|s| s.to_string())
}

fn run_dedup_gate(v: &Value) -> (GateReport, Option<String>) {
    let apr_models_env = parse_str(v.get("apr_models_env"));
    let home = parse_str(v.get("home")).unwrap_or_default();
    let expected_root = parse_str(v.get("expected_root"));
    let hex_a = parse_str(v.get("sha256_hex_a"));
    let hex_b = parse_str(v.get("sha256_hex_b"));
    let expected_same_path = v
        .get("expected_same_path")
        .and_then(|x| x.as_bool())
        .unwrap_or(true);

    // (1) Resolve registry root.
    let root = match resolve_registry_root(apr_models_env.as_deref(), Path::new(&home)) {
        Ok(r) => r,
        Err(e) => {
            let desc = format!("resolve_registry_root error: {e:?}");
            return (
                GateReport {
                    gate: "dedup",
                    falsify_id: "FALSIFY-CRUX-A-21-001",
                    outcome: desc.clone(),
                    passed: false,
                },
                Some(format!("FALSIFY-CRUX-A-21-001 dedup gate failed: {desc}")),
            );
        }
    };
    let root_ok = expected_root
        .as_ref()
        .map(|want| PathBuf::from(want) == root)
        .unwrap_or(true);

    // (2) Compute blob paths for the two digests, if provided.
    let (path_a, path_b, parse_ok) = match (hex_a.as_deref(), hex_b.as_deref()) {
        (Some(a), Some(b)) => {
            let pa = blob_path_for(&root, a);
            let pb = blob_path_for(&root, b);
            match (pa, pb) {
                (Ok(pa), Ok(pb)) => (Some(pa), Some(pb), true),
                _ => (None, None, false),
            }
        }
        (None, None) => (None, None, true),
        _ => (None, None, false),
    };

    let same_path_ok = match (&path_a, &path_b) {
        (Some(a), Some(b)) => (a == b) == expected_same_path,
        _ => true, // hashes not both supplied → skip path-equality check
    };

    let passed = root_ok && parse_ok && same_path_ok;
    let path_a_str = path_a
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(none)".to_string());
    let path_b_str = path_b
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(none)".to_string());
    let desc = format!(
        "resolved={} expected_root={} same_path_ok={same_path_ok} parse_ok={parse_ok} path_a={path_a_str} path_b={path_b_str}",
        root.display(),
        expected_root.as_deref().unwrap_or("(unspecified)")
    );
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-A-21-001 dedup gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "dedup",
            falsify_id: "FALSIFY-CRUX-A-21-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn parse_kind(s: &str) -> Option<ErrorKind> {
    match s.to_ascii_lowercase().as_str() {
        "permission_denied" | "permissiondenied" | "eacces" => Some(ErrorKind::PermissionDenied),
        "not_found" | "notfound" | "enoent" => Some(ErrorKind::NotFound),
        "unexpected_eof" | "unexpectedeof" => Some(ErrorKind::UnexpectedEof),
        "interrupted" => Some(ErrorKind::Interrupted),
        "timed_out" | "timedout" => Some(ErrorKind::TimedOut),
        "already_exists" | "alreadyexists" => Some(ErrorKind::AlreadyExists),
        _ => None,
    }
}

fn outcome_tag(o: &PullPermissionOutcome) -> &'static str {
    match o {
        PullPermissionOutcome::Ok => "ok",
        PullPermissionOutcome::PermissionDenied { .. } => "permission_denied",
        PullPermissionOutcome::NotFound { .. } => "not_found",
        PullPermissionOutcome::Other { .. } => "other",
    }
}

fn outcome_exit_code(o: &PullPermissionOutcome) -> i32 {
    match o {
        PullPermissionOutcome::Ok => 0,
        PullPermissionOutcome::PermissionDenied { exit_code, .. } => *exit_code,
        PullPermissionOutcome::NotFound { exit_code } => *exit_code,
        PullPermissionOutcome::Other { exit_code, .. } => *exit_code,
    }
}

fn run_permission_gate(v: &Value) -> (GateReport, Option<String>) {
    let kind_str = parse_str(v.get("kind"));
    let expected_outcome = parse_str(v.get("expected_outcome")).map(|s| s.to_ascii_lowercase());
    let expected_exit_code = v.get("expected_exit_code").and_then(|x| x.as_i64());
    let expected_hint_substr = parse_str(v.get("expected_hint_substring"));

    let kind = match kind_str.as_deref().and_then(parse_kind) {
        Some(k) => k,
        None => {
            let desc = format!(
                "unknown or missing ErrorKind: {:?}",
                kind_str.as_deref().unwrap_or("(unspecified)")
            );
            return (
                GateReport {
                    gate: "permission",
                    falsify_id: "FALSIFY-CRUX-A-21-002",
                    outcome: desc.clone(),
                    passed: false,
                },
                Some(format!(
                    "FALSIFY-CRUX-A-21-002 permission gate failed: {desc}"
                )),
            );
        }
    };

    let result = classify_pull_permission_outcome(kind);
    let got_tag = outcome_tag(&result);
    let got_exit = outcome_exit_code(&result);

    let outcome_ok = expected_outcome
        .as_deref()
        .map(|want| want == got_tag)
        .unwrap_or(true);
    let exit_ok = expected_exit_code
        .map(|want| want as i32 == got_exit)
        .unwrap_or(true);
    let hint_ok = match (&result, &expected_hint_substr) {
        (PullPermissionOutcome::PermissionDenied { hint, .. }, Some(sub)) => hint
            .to_ascii_lowercase()
            .contains(&sub.to_ascii_lowercase()),
        _ => true,
    };
    // Strengthened invariant from the classifier: EACCES must never → Ok.
    let never_ok_on_eacces = !matches!(
        (kind, &result),
        (ErrorKind::PermissionDenied, PullPermissionOutcome::Ok)
    );

    let passed = outcome_ok && exit_ok && hint_ok && never_ok_on_eacces;
    let desc = format!(
        "kind={} classified={got_tag} exit={got_exit} outcome_ok={outcome_ok} exit_ok={exit_ok} hint_ok={hint_ok}",
        kind_str.as_deref().unwrap_or("(unspecified)")
    );
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-A-21-002 permission gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "permission",
            falsify_id: "FALSIFY-CRUX-A-21-002",
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

    fn args_for(f: &NamedTempFile) -> SharedCacheLintArgs {
        SharedCacheLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = SharedCacheLintArgs {
            observation_file: "/no/such/cache.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err().to_string();
        // The whole *-lint family reports a missing input identically:
        // "File not found: <path>" with exit 3 (commands::lint_error).
        assert!(err.contains("File not found"), "got: {err}");
        assert!(err.contains("/no/such/cache.json"), "got: {err}");
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs(" ");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("@@@");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("none of dedup/permission"));
    }

    #[test]
    fn dedup_gate_resolves_explicit_root() {
        // Explicit APR_MODELS root → resolved root must equal expected_root.
        let f = write_obs(
            r#"{"dedup": {"apr_models_env": "/var/lib/apr/models",
                "home": "/home/user",
                "expected_root": "/var/lib/apr/models"}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn dedup_gate_wrong_expected_root_fails() {
        let f = write_obs(
            r#"{"dedup": {"apr_models_env": "/var/lib/apr/models",
                "home": "/home/user",
                "expected_root": "/somewhere/else"}}"#,
        );
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-A-21-001"));
    }

    #[test]
    fn permission_gate_eacces_classifies_permission_denied() {
        let f = write_obs(
            r#"{"permission": {"kind": "permission_denied", "expected_outcome": "permission_denied"}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn permission_gate_unknown_kind_fails() {
        let f = write_obs(r#"{"permission": {"kind": "banana"}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-A-21-002"));
    }

    #[test]
    fn permission_gate_outcome_mismatch_fails() {
        let f =
            write_obs(r#"{"permission": {"kind": "permission_denied", "expected_outcome": "ok"}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-A-21-002"));
    }

    #[test]
    fn json_mode_ok() {
        let f =
            write_obs(r#"{"permission": {"kind": "not_found", "expected_outcome": "not_found"}}"#);
        let args = SharedCacheLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
