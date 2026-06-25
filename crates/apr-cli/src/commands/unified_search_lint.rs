//! CRUX-A-23 — `apr unified-search-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the pure `search_merge` classifier over a captured JSON
//! observation file covering the two FALSIFY gates:
//!
//! ```jsonc
//! {
//!   "offline": {
//!     "local":            [{"repo": "gpt2", "downloads": 0, "likes": 0, "cached": true}],
//!     "expected_count":   1,
//!     "expected_sources": { "gpt2": "LOCAL" }
//!   },
//!   "dedup": {
//!     "hub":              [{"repo": "gpt2", "downloads": 1000, "likes": 10}],
//!     "local":            [{"repo": "gpt2", "cached": true}],
//!     "expected_count":   1,
//!     "expected_sources": { "gpt2": "BOTH" }
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-A-23
//! stderr stamp on any failing gate.

use crate::commands::search_merge::{merge_search_results, MergedRow, SearchHit, Source};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct UnifiedSearchLintArgs {
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

pub fn run(args: UnifiedSearchLintArgs) -> Result<(), String> {
    let path = Path::new(&args.observation_file);
    if !path.exists() {
        return Err(format!(
            "FALSIFY-CRUX-A-23: observation file not found: {}",
            args.observation_file
        ));
    }
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("FALSIFY-CRUX-A-23: failed to read observation: {e}"))?;
    if raw.trim().is_empty() {
        return Err("FALSIFY-CRUX-A-23: observation file is empty".to_string());
    }
    let obs: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("FALSIFY-CRUX-A-23: observation is not valid JSON: {e}"))?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(v) = obs.get("offline") {
        let (r, err) = run_offline_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("dedup") {
        let (r, err) = run_dedup_gate(v);
        reports.push(r);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err("FALSIFY-CRUX-A-23: observation has none of offline/dedup".into());
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-A-23",
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
        return Err(failures.join("\n"));
    }
    Ok(())
}

fn parse_hits(v: Option<&Value>) -> Vec<SearchHit> {
    v.and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|h| {
                    let repo = h.get("repo")?.as_str()?.to_string();
                    let downloads = h.get("downloads").and_then(|x| x.as_u64()).unwrap_or(0);
                    let likes = h.get("likes").and_then(|x| x.as_u64()).unwrap_or(0);
                    let cached = h.get("cached").and_then(|x| x.as_bool()).unwrap_or(false);
                    Some(SearchHit {
                        repo,
                        downloads,
                        likes,
                        cached,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_expected_sources(v: Option<&Value>) -> BTreeMap<String, String> {
    v.and_then(|x| x.as_object())
        .map(|o| {
            o.iter()
                .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

fn source_tag(s: Source) -> &'static str {
    match s {
        Source::Hub => "HUB",
        Source::Local => "LOCAL",
        Source::Both => "BOTH",
    }
}

fn compare_merge(
    rows: &[MergedRow],
    expected_count: Option<u64>,
    expected_sources: &BTreeMap<String, String>,
) -> Result<String, String> {
    if let Some(want) = expected_count {
        if rows.len() as u64 != want {
            return Err(format!(
                "expected_count={want}, got {} (repos={:?})",
                rows.len(),
                rows.iter().map(|r| &r.repo).collect::<Vec<_>>()
            ));
        }
    }
    for (repo, want_source) in expected_sources {
        match rows.iter().find(|r| &r.repo == repo) {
            None => {
                return Err(format!("expected repo {repo:?} missing from merged rows"));
            }
            Some(r) => {
                let got = source_tag(r.source);
                if got != want_source.as_str() {
                    return Err(format!(
                        "repo {repo:?} expected source={want_source} got={got}"
                    ));
                }
            }
        }
    }
    Ok(format!(
        "rows={} expected_count_ok={} sources_ok={}",
        rows.len(),
        expected_count.is_some(),
        expected_sources.len()
    ))
}

fn run_offline_gate(v: &Value) -> (GateReport, Option<String>) {
    let hub = parse_hits(v.get("hub")); // absent → empty (offline)
    let local = parse_hits(v.get("local"));
    let expected_count = v.get("expected_count").and_then(|x| x.as_u64());
    let expected_sources = parse_expected_sources(v.get("expected_sources"));

    let rows = merge_search_results(&hub, &local);
    let (passed, desc) = match compare_merge(&rows, expected_count, &expected_sources) {
        Ok(msg) => (true, msg),
        Err(msg) => (false, msg),
    };
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-A-23-001 offline gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "offline",
            falsify_id: "FALSIFY-CRUX-A-23-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_dedup_gate(v: &Value) -> (GateReport, Option<String>) {
    let hub = parse_hits(v.get("hub"));
    let local = parse_hits(v.get("local"));
    let expected_count = v.get("expected_count").and_then(|x| x.as_u64());
    let expected_sources = parse_expected_sources(v.get("expected_sources"));

    let rows = merge_search_results(&hub, &local);
    let (passed, desc) = match compare_merge(&rows, expected_count, &expected_sources) {
        Ok(msg) => (true, msg),
        Err(msg) => (false, msg),
    };
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-A-23-002 dedup gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "dedup",
            falsify_id: "FALSIFY-CRUX-A-23-002",
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

    fn args_for(f: &NamedTempFile) -> UnifiedSearchLintArgs {
        UnifiedSearchLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = UnifiedSearchLintArgs {
            observation_file: "/no/such/search.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-A-23"));
        assert!(err.contains("not found"));
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs(" ");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("zzz");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("none of offline/dedup"));
    }

    #[test]
    fn offline_gate_local_only_passes() {
        let f = write_obs(
            r#"{"offline": {"local": [{"repo": "gpt2", "downloads": 0, "likes": 0, "cached": true}],
                "expected_count": 1,
                "expected_sources": {"gpt2": "LOCAL"}}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn offline_gate_wrong_count_fails() {
        let f = write_obs(
            r#"{"offline": {"local": [{"repo": "gpt2", "cached": true}],
                "expected_count": 5}}"#,
        );
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-A-23-001"));
    }

    #[test]
    fn dedup_gate_both_sources_passes() {
        let f = write_obs(
            r#"{"dedup": {"hub": [{"repo": "gpt2", "downloads": 1000, "likes": 10}],
                "local": [{"repo": "gpt2", "cached": true}],
                "expected_count": 1,
                "expected_sources": {"gpt2": "BOTH"}}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn dedup_gate_wrong_source_fails() {
        let f = write_obs(
            r#"{"dedup": {"hub": [{"repo": "gpt2", "downloads": 1000}],
                "local": [{"repo": "gpt2", "cached": true}],
                "expected_sources": {"gpt2": "HUB"}}}"#,
        );
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-A-23-002"));
    }

    #[test]
    fn json_mode_ok() {
        let f = write_obs(
            r#"{"offline": {"local": [{"repo": "x", "cached": true}], "expected_count": 1}}"#,
        );
        let args = UnifiedSearchLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
