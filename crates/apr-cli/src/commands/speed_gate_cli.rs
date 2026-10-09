//! `apr model speed-gate` (EXT-19, aprender#4401): the release-phase speed
//! gate over a ledger file. It prints verdicts only — never a ratio or floor
//! (T28) — so there is no `--json`: the report is the whole output.

use super::speed_gate::gate;
use super::speed_perf_rows::perf_gate;
use crate::error::CliError;
use std::path::Path;

/// Which row shape the ledger holds.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowShape {
    /// `apr-perf-ledger-v1` rows, admitted per `apr-obs-row-identity-v1`
    Perf,
    /// EXT-26 rows: `{tag, cell, outcome: measured | not_run}`
    Ext26,
}

/// The report text and whether the gate passed.
pub(crate) fn report(
    ledger_jsonl: &str,
    tags: &[String],
    cells: &[String],
    rows: RowShape,
    llama_pin: Option<&str>,
) -> Result<(String, bool), String> {
    match rows {
        RowShape::Perf => {
            let pin = llama_pin.ok_or("perf rows need --llama-pin (scripts/llama_pin.toml)")?;
            let r = perf_gate(ledger_jsonl, pin, tags, cells)?;
            Ok((r.render(), r.passed()))
        }
        RowShape::Ext26 => {
            let r = gate(ledger_jsonl, tags, cells)?;
            Ok((r.render(), r.passed()))
        }
    }
}

/// `apr model speed-gate`: print the report; a FAIL or a bad ledger is an error.
pub(crate) fn run(
    ledger: &Path,
    tags: &[String],
    cells: &[String],
    rows: RowShape,
    llama_pin: Option<&str>,
) -> Result<(), CliError> {
    let text = std::fs::read_to_string(ledger).map_err(|e| {
        CliError::ValidationFailed(format!("speed gate: read {}: {e}", ledger.display()))
    })?;
    let (out, passed) = report(&text, tags, cells, rows, llama_pin)
        .map_err(|e| CliError::ValidationFailed(format!("speed gate: {e}")))?;
    print!("{out}");
    if passed {
        Ok(())
    } else {
        Err(CliError::ValidationFailed("speed gate: FAIL".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perf_rows_need_a_pin_and_an_empty_ledger_is_an_error() {
        let t = vec!["v0.71.1".to_string()];
        let c = vec!["gx10/cuda".to_string()];
        let e = report("{}", &t, &c, RowShape::Perf, None).expect_err("pin");
        assert!(e.contains("--llama-pin"), "{e}");
        assert!(report("", &t, &c, RowShape::Perf, Some("d1d3c3396")).is_err());
    }

    #[test]
    fn an_inadmissible_row_fails_through_the_cli_path() {
        let t = vec!["v0.71.1".to_string()];
        let c = vec!["gx10/cuda".to_string()];
        let (out, passed) = report(
            "{\"schema\":\"apr-perf-ledger-v1\"}",
            &t,
            &c,
            RowShape::Perf,
            Some("x"),
        )
        .expect("report");
        assert!(!passed);
        assert!(out.contains("ABSENT   line 1"), "{out}");
        assert!(out.ends_with("speed gate: FAIL\n"), "{out}");
    }
}
