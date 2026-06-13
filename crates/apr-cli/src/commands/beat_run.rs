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
    let yaml = std::fs::read_to_string(contract)
        .map_err(|e| CliError::InvalidFormat(format!("cannot read {}: {e}", contract.display())))?;
    let parsed = parse_contract_str(&yaml)
        .map_err(|e| CliError::InvalidFormat(format!("{}: {e}", contract.display())))?;
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
            "  metric={}  direction={}  threshold={:?}  baseline={:?}",
            beat.metric, beat.direction, beat.beat_threshold, beat.baseline_value
        );
        match (measured, outcome) {
            (Some(m), Some(BeatOutcome::Won)) => println!("  measured={m} → WON ✅"),
            (Some(m), Some(BeatOutcome::Regressed)) => println!("  measured={m} → REGRESSED ❌"),
            (Some(m), None) => println!("  measured={m} → UNJUDGEABLE (bad direction/threshold)"),
            (None, _) => println!("  (no --measured given; reporting pinned parameters only)"),
        }
    }

    match (measured, outcome) {
        (Some(_), Some(BeatOutcome::Regressed)) => Err(CliError::ValidationFailed(format!(
            "BEAT {} REGRESSED: measured value is on the losing side of the pinned \
             threshold {:?} ({})",
            beat.ci_gate_name, beat.beat_threshold, beat.direction
        ))),
        (Some(_), None) => Err(CliError::ValidationFailed(format!(
            "BEAT {} is unjudgeable — contract has no finite threshold or a bad direction",
            beat.ci_gate_name
        ))),
        _ => Ok(()),
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
