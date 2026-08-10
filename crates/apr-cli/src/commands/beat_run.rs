//! `apr beat-run` — evaluate a BeatBenchmark contract against a measured value.
//!
//! The falsifiable runner for the four-pillar "replace AND beat" mission
//! (PMAT-741). Given a `beat-benchmark` contract and (optionally) a measured
//! metric value, it reports the pinned incumbent baseline and the WON/REGRESSED
//! verdict, exiting non-zero on regression so CI hard-fails. The verdict is
//! computed by the single source of truth, `aprender_contracts::schema::Beat::evaluate`
//! — no logic is duplicated here.

use std::path::Path;

use provable_contracts::schema::{parse_contract_str, BeatOutcome};

use crate::error::CliError;

/// Run a beat contract.
///
/// - Without `measured`: report the contract's pinned beat parameters, exit 0.
/// - With `measured`: compute the verdict; return `Err` (non-zero exit) on a
///   regression or an unjudgeable contract, so this can gate CI directly.
pub fn run(contract: &Path, measured: Option<f64>, json: bool) -> Result<(), CliError> {
    let yaml = std::fs::read_to_string(contract).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            // A contract path that does not exist is a missing FILE, not a
            // malformed APR model — `InvalidFormat` printed "Invalid APR
            // format:" and sent the reader looking for a model.
            CliError::FileNotFound(contract.to_path_buf())
        } else {
            CliError::Io(e)
        }
    })?;
    let parsed = parse_contract_str(&yaml).map_err(|e| {
        CliError::ValidationFailed(format!(
            "{} is not a readable contract YAML: {e}",
            contract.display()
        ))
    })?;
    let beat = parsed.beat.ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "{} is not a beat-benchmark contract (no `beat:` block)",
            contract.display()
        ))
    })?;

    let outcome = measured.and_then(|m| beat.evaluate(m));

    if json {
        let f = |v: Option<f64>| v.map_or_else(|| "null".to_string(), |x| x.to_string());
        let verdict = match outcome {
            Some(BeatOutcome::Won) => "\"won\"",
            Some(BeatOutcome::Regressed) => "\"regressed\"",
            None => "null",
        };
        println!(
            "{{\"gate\":\"{}\",\"pillar\":{},\"incumbent\":\"{}\",\"metric\":\"{}\",\
             \"direction\":\"{}\",\"baseline\":{},\"threshold\":{},\"measured\":{},\"verdict\":{}}}",
            beat.ci_gate_name,
            beat.pillar.map_or_else(|| "null".to_string(), |p| p.to_string()),
            beat.incumbent,
            beat.metric,
            beat.direction,
            f(beat.baseline_value),
            f(beat.beat_threshold),
            f(measured),
            verdict,
        );
    } else {
        println!(
            "BEAT {} (pillar {}) — apr vs {}",
            beat.ci_gate_name,
            beat.pillar
                .map_or_else(|| "?".to_string(), |p| p.to_string()),
            beat.incumbent
        );
        println!(
            "  metric={}  direction={}  threshold={}  baseline={}",
            beat.metric,
            beat.direction,
            render_pinned(beat.beat_threshold),
            render_pinned(beat.baseline_value)
        );
        match (measured, outcome) {
            (Some(m), Some(BeatOutcome::Won)) => println!("  measured={m} → WON ✅"),
            (Some(m), Some(BeatOutcome::Regressed)) => println!("  measured={m} → REGRESSED ❌"),
            (Some(m), None) => println!(
                "  measured={m} → UNJUDGEABLE ({})",
                unjudgeable_reason(&beat, m)
            ),
            (None, _) => println!("  (no --measured given; reporting pinned parameters only)"),
        }
    }

    match (measured, outcome) {
        (Some(_), Some(BeatOutcome::Regressed)) => Err(CliError::ValidationFailed(format!(
            "BEAT {} REGRESSED: measured value is on the losing side of the pinned \
             threshold {} ({})",
            beat.ci_gate_name,
            render_pinned(beat.beat_threshold),
            beat.direction
        ))),
        (Some(m), None) => Err(CliError::ValidationFailed(format!(
            "BEAT {} is unjudgeable — {}",
            beat.ci_gate_name,
            unjudgeable_reason(&beat, m)
        ))),
        _ => Ok(()),
    }
}

/// Render a pinned `Option<f64>` for humans: `0.92`, or `(unset)` when the
/// contract does not pin it. Previously `{:?}` leaked `Some(0.92)` into both
/// the report body and the error string.
fn render_pinned(v: Option<f64>) -> String {
    v.map_or_else(|| "(unset)".to_string(), |x| x.to_string())
}

/// Explain WHY [`Beat::evaluate`] could not judge, naming the party at fault.
///
/// The measured value comes from the operator (`--measured`); the threshold and
/// direction come from the contract. Blaming the contract for a non-finite
/// `--measured` sends a release engineer to edit a correct YAML file.
fn unjudgeable_reason(beat: &provable_contracts::schema::Beat, measured: f64) -> String {
    if !measured.is_finite() {
        return format!(
            "--measured {measured} is not a finite number; the contract's threshold \
             {} is fine",
            render_pinned(beat.beat_threshold)
        );
    }
    match beat.beat_threshold {
        None => "the contract pins no beat_threshold".to_string(),
        Some(t) if !t.is_finite() => {
            format!("the contract's beat_threshold {t} is not finite")
        }
        Some(_) => format!(
            "the contract's direction {:?} is neither higher_is_better nor lower_is_better",
            beat.direction
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_contract(body: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tmp");
        write!(
            f,
            "metadata:\n  kind: beat-benchmark\n  version: \"1.0.0\"\n  \
             description: \"t\"\n  references:\n    - \"r\"\nbeat:\n{body}"
        )
        .expect("write");
        f
    }

    const VALID: &str = "  incumbent: scikit-learn\n  metric: accuracy\n  \
        direction: higher_is_better\n  beat_threshold: 0.92\n  ci_gate_name: g\n  \
        approved_compute: CPU\n";

    #[test]
    fn beat_run_won_exits_ok() {
        let f = write_contract(VALID);
        assert!(run(f.path(), Some(0.94), true).is_ok());
    }

    #[test]
    fn beat_run_regressed_is_err() {
        let f = write_contract(VALID);
        assert!(run(f.path(), Some(0.90), true).is_err());
    }

    #[test]
    fn beat_run_report_only_ok() {
        let f = write_contract(VALID);
        assert!(run(f.path(), None, false).is_ok());
    }

    /// FALSIFY-BEAT-RUN-MSG-001 — the report and the error must print the
    /// pinned threshold as a number. `{:?}` on `Option<f64>` shipped
    /// `threshold=Some(0.92)` to users.
    #[test]
    fn regressed_error_prints_bare_threshold_not_debug_option() {
        let f = write_contract(VALID);
        let err = run(f.path(), Some(0.90), false).expect_err("0.90 < 0.92 must regress");
        let msg = err.to_string();
        assert!(
            !msg.contains("Some("),
            "Rust Debug `Some(..)` leaked into a user-facing message: {msg}"
        );
        assert!(
            msg.contains("threshold 0.92"),
            "expected the bare threshold value in: {msg}"
        );
    }

    #[test]
    fn render_pinned_prints_value_or_unset() {
        assert_eq!(render_pinned(Some(0.92)), "0.92");
        assert_eq!(render_pinned(None), "(unset)");
    }

    /// FALSIFY-BEAT-RUN-MSG-002 — a non-finite `--measured` is the operator's
    /// input, not a contract defect. The old message said "contract has no
    /// finite threshold or a bad direction" while the contract pinned 0.92.
    #[test]
    fn non_finite_measured_blames_the_measurement_not_the_contract() {
        let f = write_contract(VALID);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = run(f.path(), Some(bad), false).expect_err("non-finite must not be judged");
            let msg = err.to_string();
            assert!(
                msg.contains("--measured"),
                "unjudgeable message must name --measured, got: {msg}"
            );
            assert!(
                !msg.contains("contract has no finite threshold"),
                "message blames the contract for the operator's value: {msg}"
            );
        }
    }

    /// The contract-side reasons must still be attributed to the contract.
    #[test]
    fn bad_direction_blames_the_contract() {
        let f = write_contract(
            "  incumbent: x\n  metric: m\n  direction: sideways\n  \
             beat_threshold: 0.5\n  ci_gate_name: g\n  approved_compute: CPU\n",
        );
        let err = run(f.path(), Some(0.9), false).expect_err("bad direction is unjudgeable");
        let msg = err.to_string();
        assert!(msg.contains("direction"), "got: {msg}");
        assert!(!msg.contains("--measured"), "got: {msg}");
    }

    #[test]
    fn missing_threshold_blames_the_contract() {
        let f = write_contract(
            "  incumbent: x\n  metric: m\n  direction: higher_is_better\n  \
             ci_gate_name: g\n  approved_compute: CPU\n",
        );
        let err = run(f.path(), Some(0.9), false).expect_err("no threshold is unjudgeable");
        assert!(
            err.to_string().contains("pins no beat_threshold"),
            "got: {err}"
        );
    }

    /// FALSIFY-BEAT-RUN-MSG-003 — a missing contract path is a missing file,
    /// not "Invalid APR format" (`beat-run` never reads an APR model).
    #[test]
    fn missing_contract_file_is_file_not_found() {
        let err = run(Path::new("/no/such/beat.yaml"), Some(0.9), false)
            .expect_err("missing file must error");
        assert!(
            matches!(err, CliError::FileNotFound(_)),
            "expected FileNotFound, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            !msg.contains("Invalid APR format"),
            "a contract path was reported as an APR model problem: {msg}"
        );
    }

    #[test]
    fn unreadable_yaml_is_validation_failed_not_apr_format() {
        let mut f = tempfile::NamedTempFile::new().expect("tmp");
        write!(f, "\t\tthis: [is not: yaml").expect("write");
        let err = run(f.path(), Some(0.9), false).expect_err("bad yaml must error");
        assert!(
            matches!(err, CliError::ValidationFailed(_)),
            "expected ValidationFailed, got {err:?}"
        );
        assert!(
            !err.to_string().contains("Invalid APR format"),
            "got: {err}"
        );
    }

    #[test]
    fn beat_run_missing_beat_block_is_err() {
        let mut f = tempfile::NamedTempFile::new().expect("tmp");
        write!(
            f,
            "metadata:\n  kind: schema\n  version: \"1.0.0\"\n  description: \"t\"\n  \
             references:\n    - \"r\"\n"
        )
        .expect("write");
        assert!(run(f.path(), Some(0.9), true).is_err());
    }
}
