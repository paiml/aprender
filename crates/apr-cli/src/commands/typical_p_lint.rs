//! `apr typical-p-lint` — CRUX-C-22 typical-p sampling observation linter.
//!
//! Reads a JSON observation file that captures a single typical-p sampling
//! run and dispatches five classifiers (range, identity, mass coverage,
//! sort order, renormalization). Emits a text or `--json` report.
//!
//! Spec: `contracts/crux-C-22-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.
//!
//! Observation schema (top-level keys; each section is optional, but a
//! section that IS present must be readable — a wrong-typed or incomplete
//! section is a producer bug and fails the lint rather than being skipped,
//! and an observation carrying none of the five sections is rejected because
//! it asserts nothing):
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

    let mut fail_reasons: Vec<String> = Vec::new();
    if let Some(reason) = unknown_sections_reason(&obs) {
        fail_reasons.push(reason);
    }

    let range = classify_range(&obs);
    let identity = classify_identity(&obs);
    let mass = classify_mass(&obs);
    let sort = classify_sort(&obs);
    let renorm = classify_renorm(&obs);

    fail_reasons.extend(
        [
            section_fail_reason(&range, range_fail_reason),
            section_fail_reason(&identity, identity_fail_reason),
            section_fail_reason(&mass, mass_fail_reason),
            section_fail_reason(&sort, sort_fail_reason),
            section_fail_reason(&renorm, renorm_fail_reason),
        ]
        .into_iter()
        .flatten(),
    );

    let judged_any = [
        matches!(range, Section::Judged(_)),
        matches!(identity, Section::Judged(_)),
        matches!(mass, Section::Judged(_)),
        matches!(sort, Section::Judged(_)),
        matches!(renorm, Section::Judged(_)),
    ]
    .iter()
    .any(|b| *b);

    print_report(
        observation_file,
        &range,
        &identity,
        &mass,
        &sort,
        &renorm,
        json,
    );

    if !judged_any && fail_reasons.is_empty() {
        // Every classifier was skipped and nothing was wrong enough to
        // report: the observation asserted nothing, so a zero exit would be
        // a green gate over no evidence.
        return Err(CliError::ValidationFailed(format!(
            "FALSIFY-CRUX-C-22 {}: observation has none of \
             range/identity/mass/sort/renorm — nothing was checked",
            observation_file.display()
        )));
    }

    if fail_reasons.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(fail_reasons.join("; ")))
    }
}

/// The three states a section can be in. `Absent` is legitimate (an
/// observation may exercise any subset of the gates); `Unreadable` is a
/// producer bug and must never be reported as a pass.
#[derive(Debug)]
enum Section<T> {
    Absent,
    Unreadable(String),
    Judged(T),
}

/// `Ok(None)` = section absent, `Ok(Some)` = judged, `Err` = unreadable.
type StdResult<T> = std::result::Result<T, String>;

const SECTIONS: [&str; 5] = ["range", "identity", "mass", "sort", "renorm"];

/// A misspelled section key (`rnage`) used to be indistinguishable from an
/// intentionally omitted one, so a typo silently disarmed a gate.
fn unknown_sections_reason(obs: &Value) -> Option<String> {
    let map = obs.as_object()?;
    let unknown: Vec<&str> = map
        .keys()
        .map(String::as_str)
        .filter(|k| !SECTIONS.contains(k))
        .collect();
    if unknown.is_empty() {
        return None;
    }
    Some(format!(
        "FALSIFY-CRUX-C-22 unknown section(s) {unknown:?}: expected any of {SECTIONS:?} \
         (a misspelled key would otherwise skip its gate silently)"
    ))
}

fn section_fail_reason<T>(
    section: &Section<T>,
    judged: impl Fn(&T) -> Option<String>,
) -> Option<String> {
    match section {
        Section::Absent => None,
        Section::Unreadable(reason) => Some(reason.clone()),
        Section::Judged(o) => judged(o),
    }
}

/// Read a present section as an object, or explain why it is unreadable.
fn section_of<'a>(
    obs: &'a Value,
    name: &str,
    falsify_id: &str,
) -> StdResult<Option<&'a serde_json::Map<String, Value>>> {
    match obs.get(name) {
        None => Ok(None),
        Some(v) => v.as_object().map(Some).ok_or_else(|| {
            format!("{falsify_id} {name}: evidence unreadable — section is not a JSON object")
        }),
    }
}

fn read_f64(
    sec: &serde_json::Map<String, Value>,
    name: &str,
    key: &str,
    falsify_id: &str,
) -> std::result::Result<f64, String> {
    match sec.get(key) {
        None => Err(format!(
            "{falsify_id} {name}: evidence unreadable — missing `{key}`"
        )),
        Some(v) => v.as_f64().ok_or_else(|| {
            format!("{falsify_id} {name}: evidence unreadable — `{key}` is not a number ({v})")
        }),
    }
}

fn read_f64_array(
    sec: &serde_json::Map<String, Value>,
    name: &str,
    key: &str,
    falsify_id: &str,
) -> std::result::Result<Vec<f64>, String> {
    let Some(v) = sec.get(key) else {
        return Err(format!(
            "{falsify_id} {name}: evidence unreadable — missing `{key}`"
        ));
    };
    let arr = v.as_array().ok_or_else(|| {
        format!("{falsify_id} {name}: evidence unreadable — `{key}` is not an array ({v})")
    })?;
    arr.iter()
        .enumerate()
        .map(|(i, e)| {
            e.as_f64().ok_or_else(|| {
                format!(
                    "{falsify_id} {name}: evidence unreadable — `{key}[{i}]` is not a number ({e})"
                )
            })
        })
        .collect()
}

fn read_usize_array(
    sec: &serde_json::Map<String, Value>,
    name: &str,
    key: &str,
    falsify_id: &str,
) -> std::result::Result<Vec<usize>, String> {
    let Some(v) = sec.get(key) else {
        return Err(format!(
            "{falsify_id} {name}: evidence unreadable — missing `{key}`"
        ));
    };
    let arr = v.as_array().ok_or_else(|| {
        format!("{falsify_id} {name}: evidence unreadable — `{key}` is not an array ({v})")
    })?;
    arr.iter()
        .enumerate()
        .map(|(i, e)| {
            e.as_u64().map(|n| n as usize).ok_or_else(|| {
                format!(
                    "{falsify_id} {name}: evidence unreadable — \
                     `{key}[{i}]` is not a non-negative integer ({e})"
                )
            })
        })
        .collect()
}

/// Turn a `Result<Option<T>, String>` into the three-state [`Section`].
fn into_section<T>(r: StdResult<Option<T>>) -> Section<T> {
    match r {
        Ok(None) => Section::Absent,
        Ok(Some(o)) => Section::Judged(o),
        Err(reason) => Section::Unreadable(reason),
    }
}

fn classify_range(obs: &Value) -> Section<clf::TypicalPRangeOutcome> {
    into_section((|| -> StdResult<Option<_>> {
        const ID: &str = "FALSIFY-CRUX-C-22-001";
        let Some(sec) = section_of(obs, "range", ID)? else {
            return Ok(None);
        };
        let p = read_f64(sec, "range", "p", ID)?;
        Ok(Some(clf::classify_typical_p_range(p)))
    })())
}

fn classify_identity(obs: &Value) -> Section<clf::IdentityOutcome> {
    into_section((|| -> StdResult<Option<_>> {
        const ID: &str = "FALSIFY-CRUX-C-22-001";
        let Some(sec) = section_of(obs, "identity", ID)? else {
            return Ok(None);
        };
        let kept = read_usize_array(sec, "identity", "kept_indices", ID)?;
        let total = read_f64(sec, "identity", "total_tokens", ID)?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let total = total as usize;
        let p = read_f64(sec, "identity", "p", ID)?;
        Ok(Some(clf::classify_typical_p_identity(&kept, total, p)))
    })())
}

fn classify_mass(obs: &Value) -> Section<clf::MassCoverageOutcome> {
    into_section((|| -> StdResult<Option<_>> {
        const ID: &str = "FALSIFY-CRUX-C-22-002";
        let Some(sec) = section_of(obs, "mass", ID)? else {
            return Ok(None);
        };
        let kept_probs = read_f64_array(sec, "mass", "kept_probs", ID)?;
        let p = read_f64(sec, "mass", "p", ID)?;
        Ok(Some(clf::classify_typical_p_mass_coverage(&kept_probs, p)))
    })())
}

fn classify_sort(obs: &Value) -> Section<clf::SortOrderOutcome> {
    into_section((|| -> StdResult<Option<_>> {
        const ID: &str = "FALSIFY-CRUX-C-22-002";
        let Some(sec) = section_of(obs, "sort", ID)? else {
            return Ok(None);
        };
        let all_probs = read_f64_array(sec, "sort", "all_probs", ID)?;
        let kept = read_f64_array(sec, "sort", "kept_probs_in_sort_order", ID)?;
        Ok(Some(clf::classify_typical_p_sort_order(&all_probs, &kept)))
    })())
}

fn classify_renorm(obs: &Value) -> Section<clf::RenormOutcome> {
    into_section((|| -> StdResult<Option<_>> {
        const ID: &str = "FALSIFY-CRUX-C-22-002";
        let Some(sec) = section_of(obs, "renorm", ID)? else {
            return Ok(None);
        };
        let filtered = read_f64_array(sec, "renorm", "filtered_probs", ID)?;
        Ok(Some(clf::classify_typical_p_renormalization(&filtered)))
    })())
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
    range: &Section<clf::TypicalPRangeOutcome>,
    identity: &Section<clf::IdentityOutcome>,
    mass: &Section<clf::MassCoverageOutcome>,
    sort: &Section<clf::SortOrderOutcome>,
    renorm: &Section<clf::RenormOutcome>,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "observation_path": path.display().to_string(),
            "range":    render(range),
            "identity": render(identity),
            "mass":     render(mass),
            "sort":     render(sort),
            "renorm":   render(renorm),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("typical-p-lint report for {}", path.display());
        print_line("  range:    ", render(range));
        print_line("  identity: ", render(identity));
        print_line("  mass:     ", render(mass));
        print_line("  sort:     ", render(sort));
        print_line("  renorm:   ", render(renorm));
    }
}

/// `None` == the section was absent. An unreadable section renders its reason
/// so the report never shows a skipped-looking line for a gate that failed.
fn render<T: std::fmt::Debug>(section: &Section<T>) -> Option<String> {
    match section {
        Section::Absent => None,
        Section::Unreadable(reason) => Some(format!("UNREADABLE — {reason}")),
        Section::Judged(o) => Some(format!("{o:?}")),
    }
}

fn print_line(prefix: &str, v: Option<String>) {
    match v {
        Some(s) => println!("{prefix}{s}"),
        None => println!("{prefix}(section absent — classifier skipped)"),
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

    /// This test used to assert `is_ok()` — it encoded the defect. An
    /// observation with no sections asserts nothing, so exiting 0 hands CI a
    /// green gate for any file the producer failed to write (issue #2391).
    #[test]
    fn empty_object_is_rejected_because_it_checks_nothing() {
        let f = write_obs("{}");
        let err = run(f.path(), false).expect_err("an empty observation checks nothing");
        assert!(
            err.to_string()
                .contains("none of range/identity/mass/sort/renorm"),
            "got: {err}"
        );
    }

    /// FALSIFY-CRUX-C-22-SKIP-001 — a section present with a wrong-typed
    /// value must fail, not be reported as "classifier skipped" with exit 0.
    /// `{"range":{"p":"1.5"}}` used to exit 0 while `{"range":{"p":1.5}}`
    /// exited 5, and `mass` already rejected the same class of error.
    #[test]
    fn every_wrong_typed_shape_is_rejected() {
        for body in [
            r#"{"range": {"p": "1.5"}}"#,
            r#"{"range": {"p": null}}"#,
            r#"{"range": {"p": [1.5]}}"#,
            r#"{"identity": {"kept_indices": [0], "total_tokens": "3", "p": 1.0}}"#,
            r#"{"identity": {"kept_indices": ["0"], "total_tokens": 3, "p": 1.0}}"#,
            r#"{"sort": {"all_probs": 0.5, "kept_probs_in_sort_order": [0.3]}}"#,
            r#"{"renorm": {"filtered_probs": [0.6, "0.4"]}}"#,
        ] {
            let f = write_obs(body);
            assert!(
                run(f.path(), false).is_err(),
                "unreadable evidence reported as a pass: {body}"
            );
        }
    }

    #[test]
    fn wrong_typed_p_is_rejected() {
        let f = write_obs(r#"{"range": {"p": "1.5"}}"#);
        let err = run(f.path(), false).expect_err("a string where a float belongs must fail");
        let msg = err.to_string();
        assert!(msg.contains("evidence unreadable"), "got: {msg}");
        assert!(msg.contains("`p` is not a number"), "got: {msg}");
    }

    #[test]
    fn absent_subfield_is_rejected() {
        let f = write_obs(r#"{"range": {}}"#);
        let err = run(f.path(), false).expect_err("a present section missing `p` must fail");
        assert!(err.to_string().contains("missing `p`"), "got: {err}");
    }

    #[test]
    fn non_object_section_is_rejected() {
        let f = write_obs(r#"{"range": "nonsense"}"#);
        let err = run(f.path(), false).expect_err("a scalar section must fail");
        assert!(err.to_string().contains("not a JSON object"), "got: {err}");
    }

    /// A misspelled section key silently skipped its gate and exited 0.
    #[test]
    fn misspelled_section_key_is_rejected() {
        let f = write_obs(r#"{"rnage": {"p": 1.5}}"#);
        let err = run(f.path(), false).expect_err("a typo'd section must not pass");
        let msg = err.to_string();
        assert!(msg.contains("unknown section"), "got: {msg}");
        assert!(msg.contains("rnage"), "got: {msg}");
    }

    /// Control: the CORRECT spelling of the same body still fails for the
    /// real reason (p > 1.0), so the typo check is not what makes it fail.
    #[test]
    fn correctly_spelled_section_still_fails_on_its_own_merits() {
        let f = write_obs(r#"{"range": {"p": 1.5}}"#);
        let err = run(f.path(), false).expect_err("p=1.5 is out of range");
        let msg = err.to_string();
        assert!(msg.contains("FALSIFY-CRUX-C-22-001"), "got: {msg}");
        assert!(!msg.contains("unknown section"), "got: {msg}");
    }

    /// Control: legitimately omitted sections are still fine — the fix
    /// rejects unreadable evidence, not absent evidence.
    #[test]
    fn a_single_valid_section_still_passes_with_the_others_absent() {
        let f = write_obs(r#"{"range": {"p": 0.95}}"#);
        assert!(run(f.path(), false).is_ok());
    }

    /// The array element type is checked too: a string inside `kept_probs`
    /// used to become NaN and be judged as if it were a probability.
    #[test]
    fn wrong_typed_array_element_is_unreadable() {
        let f = write_obs(r#"{"mass": {"kept_probs": ["a", "b"], "p": 0.9}}"#);
        let err = run(f.path(), false).expect_err("string probs must fail");
        assert!(
            err.to_string().contains("`kept_probs[0]` is not a number"),
            "got: {err}"
        );
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
