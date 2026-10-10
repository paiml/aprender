//! `apr test llm shape-check <witness.json>`, the front door of
//! `apr_test::perf_gate::v3_shape` (#4971 V3-d).
//!
//! The library keeps the reference script's exit codes (0 ADMISSIBLE,
//! 1 NOT ADMISSIBLE, 2 RED). `apr` exits through `CliError`, so here they map
//! onto apr's own convention:
//!
//! | Verdict | apr exit |
//! |---|---|
//! | ADMISSIBLE | 0 |
//! | NOT ADMISSIBLE | 5 (`ValidationFailed`), after every reason is printed |
//! | RED: no such file, or not a file | 3 (`FileNotFound` / `NotAFile`) |
//! | RED: unreadable, empty, or no bands | 4 (`InvalidInput`: a JSON witness, not a model) |
//!
//! A witness that is not admissible and a witness that cannot be read never
//! share a code. That is the draft contract's F1, kept across the mapping.
//!
//! This is not a gate. No merge, queue or release job runs it.

use std::path::Path;

use apr_test::perf_gate::{check_v3_shape_file, ShapeVerdict};

use crate::error::{CliError, Result};

/// Check `witness`, print the verdict with every reason, and exit by it.
///
/// # Errors
/// NOT ADMISSIBLE is `ValidationFailed`. A missing path is `FileNotFound`,
/// a directory `NotAFile`, and an unreadable or empty witness `InvalidInput`.
pub(crate) fn run(witness: &Path) -> Result<()> {
    if !witness.exists() {
        return Err(CliError::FileNotFound(witness.to_path_buf()));
    }
    if !witness.is_file() {
        return Err(CliError::NotAFile(witness.to_path_buf()));
    }
    let verdict = check_v3_shape_file(witness);
    if !matches!(verdict, ShapeVerdict::Red(_)) {
        println!("{}", report(witness, &verdict));
    }
    outcome(witness, verdict)
}

/// The printed verdict: one line naming it, then one line per reason.
fn report(witness: &Path, verdict: &ShapeVerdict) -> String {
    std::iter::once(format!(
        "V3 shape check {}: {}",
        witness.display(),
        verdict.label()
    ))
    .chain(verdict.reasons().iter().map(|r| format!("  - {r}")))
    .collect::<Vec<_>>()
    .join("\n")
}

/// The verdict as apr's result. Only ADMISSIBLE is `Ok`.
fn outcome(witness: &Path, verdict: ShapeVerdict) -> Result<()> {
    match verdict {
        ShapeVerdict::Admissible => Ok(()),
        ShapeVerdict::NotAdmissible(reasons) => Err(CliError::ValidationFailed(format!(
            "V3 shape witness {}: NOT ADMISSIBLE, {} reason(s) printed above (#4971)",
            witness.display(),
            reasons.len()
        ))),
        ShapeVerdict::Red(why) => Err(CliError::InvalidInput(format!(
            "V3 shape witness {}: {why} (#4971)",
            witness.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    /// The library's control witness: every rule S1-S6 holds.
    fn admissible() -> Value {
        let band = |c: u32| json!({"c": c, "m_formed": c, "result": "PASS", "declared_min": 64, "divergence_at": 127});
        json!({
            "_planted": "FIXTURE: not a measurement",
            "binary_sha256": "a".repeat(64),
            "commit": "f".repeat(40),
            "host": "lambda",
            "prompt_sha256": "b".repeat(64),
            "model": {"path": "Qwen3.5-4B-Q4_K_M.gguf", "sha256": "c".repeat(64)},
            "bands": [band(1), band(4), band(8), band(16)],
        })
    }

    /// Run the command on `text` written to a fresh file; the exit code apr
    /// would leave (0 on `Ok`) and the printed report.
    fn exit_of(text: &str) -> (u8, String) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("witness.json");
        std::fs::write(&path, text).expect("write witness");
        let verdict = check_v3_shape_file(&path);
        let printed = report(&path, &verdict);
        let code = run(&path).map_or_else(|e| e.exit_code_value(), |()| 0);
        (code, printed)
    }

    #[test]
    fn the_control_witness_is_admissible_and_exits_0() {
        let (code, printed) = exit_of(&admissible().to_string());
        assert_eq!(code, 0, "{printed}");
        assert!(printed.ends_with(": ADMISSIBLE"), "{printed}");
    }

    /// #4971's falsifier, row 1: one band at `m_formed = c - 1`.
    #[test]
    fn a_band_that_formed_c_minus_1_exits_5_and_names_the_band() {
        let mut w = admissible();
        w["bands"][1]["m_formed"] = json!(3);
        let (code, printed) = exit_of(&w.to_string());
        assert_eq!(code, 5, "{printed}");
        assert!(printed.contains("NOT ADMISSIBLE"), "{printed}");
        assert!(
            printed.contains("  - c=4: m_formed 3 != c (shape not served)"),
            "{printed}"
        );
    }

    /// #4971's falsifier, row 2: one divergence with no margin recorded.
    #[test]
    fn an_early_divergence_without_a_margin_exits_5_and_says_so() {
        let mut w = admissible();
        w["bands"][2]["divergence_at"] = json!(3);
        let (code, printed) = exit_of(&w.to_string());
        assert_eq!(code, 5, "{printed}");
        assert!(
            printed
                .contains("  - c=8: diverges from m=1 at 3 < 64 with no near-tie margin recorded"),
            "{printed}"
        );
    }

    /// The same divergence carried by a near-tie margin is admissible.
    #[test]
    fn an_early_divergence_at_a_near_tie_exits_0() {
        let mut w = admissible();
        w["bands"][2]["divergence_at"] = json!(3);
        w["bands"][2]["top2_margin_at_divergence"] = json!(0.01);
        let (code, printed) = exit_of(&w.to_string());
        assert_eq!(code, 0, "{printed}");
    }

    /// F1 across the mapping: RED never shares NOT ADMISSIBLE's code. And it
    /// names the input, not a model: `InvalidFormat` would print "Invalid APR
    /// format" over a JSON witness.
    #[test]
    fn an_unreadable_or_empty_witness_exits_4_never_5() {
        for text in ["", "   \n", "{not json", "[]", r#"{"bands": []}"#] {
            let (code, _) = exit_of(text);
            assert_eq!(
                code, 4,
                "witness {text:?} must be RED (4), never NOT ADMISSIBLE (5)"
            );
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("witness.json");
        std::fs::write(&path, "{not json").expect("write witness");
        let err = run(&path).expect_err("RED is refused");
        assert!(matches!(err, CliError::InvalidInput(_)), "{err:?}");
        let shown = err.to_string();
        assert!(
            shown.starts_with("Invalid input: V3 shape witness "),
            "{shown}"
        );
        assert!(!shown.contains("APR"), "{shown}");
    }

    #[test]
    fn a_missing_path_exits_3_and_a_directory_exits_3() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = run(&dir.path().join("absent.json")).expect_err("missing is refused");
        assert!(matches!(missing, CliError::FileNotFound(_)), "{missing:?}");
        assert_eq!(missing.exit_code_value(), 3);
        let not_a_file = run(dir.path()).expect_err("a directory is refused");
        assert!(
            matches!(not_a_file, CliError::NotAFile(_)),
            "{not_a_file:?}"
        );
        assert_eq!(not_a_file.exit_code_value(), 3);
    }
}
