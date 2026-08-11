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
//!
//! A gate reaches one of three verdicts, not two. `PASS` means every supplied
//! expectation held; `FAIL` means one did not; `VACUOUS` means the section
//! supplied no expectation to check (or supplied one the parser could not
//! read), so the gate proved nothing. `VACUOUS` exits non-zero: a falsifier
//! that asserted nothing must not be recorded as a discharged obligation.

use crate::commands::lint_input;
use crate::commands::lint_vacuity::{json_type, Verdict};
use crate::commands::search_merge::{merge_search_results, MergedRow, SearchHit, Source};
use crate::error::CliError;
use serde_json::Value;
use std::collections::BTreeMap;
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
    verdict: &'static str,
    passed: bool,
}

pub fn run(args: UnifiedSearchLintArgs) -> crate::error::Result<()> {
    let path = Path::new(&args.observation_file);
    let obs = lint_input::read_json_observation("apr unified-search-lint", path)?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for (key, gate, falsify_id) in [
        ("offline", "offline", "FALSIFY-CRUX-A-23-001"),
        ("dedup", "dedup", "FALSIFY-CRUX-A-23-002"),
    ] {
        if let Some(v) = obs.get(key) {
            let (r, err) = run_gate(gate, falsify_id, v);
            reports.push(r);
            if let Some(e) = err {
                failures.push(e);
            }
        }
    }

    if reports.is_empty() {
        return Err(CliError::ValidationFailed(
            "FALSIFY-CRUX-A-23: observation has none of offline/dedup".into(),
        ));
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-A-23",
            "gates": reports,
        });
        println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    } else {
        for r in &reports {
            println!(
                "[{}] {} ({}): {}",
                r.verdict, r.gate, r.falsify_id, r.outcome
            );
        }
    }

    if !failures.is_empty() {
        return Err(CliError::ValidationFailed(failures.join("\n")));
    }
    Ok(())
}

/// Parse a `hub`/`local` hit array.
///
/// Every wrong JSON type is an error. The 0.63.0 binary used `filter_map` here,
/// so a hit whose `repo` was not a string vanished from the merged rows and
/// silently changed the very count the gate compares against.
fn parse_hits(field: &str, v: Option<&Value>) -> Result<Vec<SearchHit>, String> {
    let Some(v) = v else {
        return Ok(Vec::new());
    };
    if v.is_null() {
        return Ok(Vec::new());
    }
    let arr = v
        .as_array()
        .ok_or_else(|| format!("{field} must be an array of hits, got {}", json_type(v)))?;
    arr.iter()
        .enumerate()
        .map(|(i, h)| {
            let repo = h
                .get("repo")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{field}[{i}] has no string \"repo\" field"))?
                .to_string();
            let downloads = parse_u64_field(h, &format!("{field}[{i}].downloads"))?.unwrap_or(0);
            let likes = parse_u64_field(h, &format!("{field}[{i}].likes"))?.unwrap_or(0);
            let cached = match h.get("cached") {
                None | Some(Value::Null) => false,
                Some(c) => c.as_bool().ok_or_else(|| {
                    format!(
                        "{field}[{i}].cached must be a boolean, got {}",
                        json_type(c)
                    )
                })?,
            };
            Ok(SearchHit {
                repo,
                downloads,
                likes,
                cached,
            })
        })
        .collect()
}

/// Read `<obj>.<last path segment>` as a `u64`, erroring on a wrong type.
fn parse_u64_field(obj: &Value, path: &str) -> Result<Option<u64>, String> {
    let key = path.rsplit('.').next().unwrap_or(path);
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(n) => n.as_u64().map(Some).ok_or_else(|| {
            format!(
                "{path} must be a non-negative integer, got {} ({n})",
                json_type(n)
            )
        }),
    }
}

/// Parse `expected_sources`. A wrong-typed map (or entry) is an error, never
/// an empty map — an empty map silently disarms the whole source comparison.
fn parse_expected_sources(v: Option<&Value>) -> Result<BTreeMap<String, String>, String> {
    let Some(v) = v else {
        return Ok(BTreeMap::new());
    };
    if v.is_null() {
        return Ok(BTreeMap::new());
    }
    let obj = v.as_object().ok_or_else(|| {
        format!(
            "expected_sources must be an object of repo -> source, got {}",
            json_type(v)
        )
    })?;
    obj.iter()
        .map(|(k, val)| {
            let s = val.as_str().ok_or_else(|| {
                format!(
                    "expected_sources[{k:?}] must be a string, got {}",
                    json_type(val)
                )
            })?;
            Ok((k.clone(), s.to_string()))
        })
        .collect()
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
    // A section that supplies no expectation merges rows and compares nothing.
    // 0.63.0 rendered that as `[PASS] … expected_count_ok=false sources_ok=0`.
    if expected_count.is_none() && expected_sources.is_empty() {
        return Err(format!(
            "VACUOUS: section supplies neither expected_count nor expected_sources, so the {} \
             merged row(s) were compared against nothing — a gate that asserts nothing cannot pass",
            rows.len()
        ));
    }
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
        "rows={} expected_count={} expected_sources={} — every supplied expectation held",
        rows.len(),
        expected_count.map_or_else(|| "(not supplied)".to_string(), |c| c.to_string()),
        expected_sources.len()
    ))
}

/// Evaluate one gate section. Schema errors, vacuity and genuine violations
/// are all non-zero; only a section that supplied an expectation and met it
/// reaches `Verdict::Pass`.
fn run_gate(
    gate: &'static str,
    falsify_id: &'static str,
    v: &Value,
) -> (GateReport, Option<String>) {
    let result = evaluate_section(v);
    let verdict = Verdict::of(&result);
    let desc = match result {
        Ok(msg) | Err(msg) => msg,
    };
    let err = if verdict == Verdict::Pass {
        None
    } else {
        Some(format!("{falsify_id} {gate} gate failed: {desc}"))
    };
    (
        GateReport {
            gate,
            falsify_id,
            outcome: desc,
            verdict: verdict.tag(),
            passed: verdict == Verdict::Pass,
        },
        err,
    )
}

fn evaluate_section(v: &Value) -> Result<String, String> {
    if !v.is_object() {
        return Err(format!(
            "section must be a JSON object, got {} — nothing could be read from it",
            json_type(v)
        ));
    }
    let hub = parse_hits("hub", v.get("hub"))?; // absent → empty (offline)
    let local = parse_hits("local", v.get("local"))?;
    let expected_count = parse_u64_field(v, "expected_count")?;
    let expected_sources = parse_expected_sources(v.get("expected_sources"))?;

    let rows = merge_search_results(&hub, &local);
    compare_merge(&rows, expected_count, &expected_sources)
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
        // #2377-8: this used to be exit 1, the same code a *failing falsifier*
        // produced, because `run` returned `Result<(), String>` and dispatch had
        // no class to map. A CI job could not tell the two apart.
        assert_eq!(
            err.exit_code(),
            std::process::ExitCode::from(3),
            "a missing observation file must be exit 3: {err}"
        );
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs(" ");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("is empty"), "{err}");
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("zzz");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("failed to parse JSON"), "{err}");
        // #2377-9: a captured JSON observation is not an APR model.
        assert!(!err.contains("Invalid APR format"), "{err}");
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err().to_string();
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
        let err = run(args_for(&f)).unwrap_err().to_string();
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
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-A-23-002"));
    }

    /// `{"offline": {}}` merged zero rows and compared them to nothing, and
    /// 0.63.0 called that `[PASS] … expected_count_ok=false sources_ok=0`.
    #[test]
    fn falsifier_section_with_no_expectations_is_vacuous_not_pass() {
        for body in [
            r#"{"offline": {}}"#,
            r#"{"dedup": {}}"#,
            r#"{"offline": {"local": [{"repo": "gpt2", "cached": true}]}}"#,
            r#"{"offline": {"local": [{"repo": "gpt2", "cached": true}], "expected_sources": {}}}"#,
        ] {
            let f = write_obs(body);
            let err = run(args_for(&f)).unwrap_err().to_string();
            assert!(err.contains("VACUOUS"), "{body}: {err}");
            assert!(err.contains("asserts nothing"), "{body}: {err}");
        }
    }

    /// The same expectation quoted instead of numeric used to skip the whole
    /// count comparison and exit 0.
    #[test]
    fn falsifier_wrong_typed_expectation_is_a_schema_error() {
        let numeric = write_obs(
            r#"{"offline": {"local": [{"repo": "gpt2", "cached": true}], "expected_count": 99}}"#,
        );
        assert!(
            run(args_for(&numeric))
                .unwrap_err()
                .to_string()
                .contains("expected_count=99, got 1"),
            "control: a numeric expectation must still be compared"
        );

        for (body, want) in [
            (
                r#"{"offline": {"local": [{"repo": "gpt2", "cached": true}], "expected_count": "99"}}"#,
                "expected_count must be a non-negative integer",
            ),
            (
                r#"{"offline": {"local": [{"repo": "gpt2", "cached": true}], "expected_sources": "gpt2=HUB"}}"#,
                "expected_sources must be an object",
            ),
            (
                r#"{"offline": {"local": [{"repo": "gpt2", "cached": true}], "expected_sources": {"gpt2": 7}}}"#,
                "expected_sources[\"gpt2\"] must be a string",
            ),
            (r#"{"offline": "nope"}"#, "section must be a JSON object"),
            (
                r#"{"offline": {"local": "gpt2", "expected_count": 1}}"#,
                "local must be an array of hits",
            ),
            (
                r#"{"offline": {"local": [{"repo": 7}], "expected_count": 1}}"#,
                "local[0] has no string \"repo\" field",
            ),
        ] {
            let f = write_obs(body);
            let err = run(args_for(&f)).unwrap_err().to_string();
            assert!(err.contains(want), "{body}: expected {want:?}, got {err}");
            assert!(!err.contains("VACUOUS"), "{body}: {err}");
        }
    }

    /// The report line and the exit code must agree. 0.63.0's `--json` said
    /// `"passed": true` next to `expected_count_ok=false`.
    #[test]
    fn falsifier_json_report_never_marks_a_vacuous_gate_passed() {
        let f = write_obs(r#"{"offline": {}}"#);
        let args = UnifiedSearchLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_err());

        let (report, err) = run_gate("offline", "FALSIFY-CRUX-A-23-001", &serde_json::json!({}));
        assert_eq!(report.verdict, "VACUOUS");
        assert!(!report.passed);
        assert!(err.is_some());
    }

    #[test]
    fn passing_outcome_string_names_what_was_checked() {
        let (report, err) = run_gate(
            "offline",
            "FALSIFY-CRUX-A-23-001",
            &serde_json::json!({
                "local": [{"repo": "gpt2", "cached": true}],
                "expected_count": 1,
                "expected_sources": {"gpt2": "LOCAL"},
            }),
        );
        assert!(err.is_none());
        assert_eq!(report.verdict, "PASS");
        assert_eq!(
            report.outcome,
            "rows=1 expected_count=1 expected_sources=1 — every supplied expectation held"
        );
        // The self-contradictory 0.63.0 rendering must not come back.
        assert!(!report.outcome.contains("_ok="));
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
