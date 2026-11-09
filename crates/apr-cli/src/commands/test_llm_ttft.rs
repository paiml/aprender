//! `apr test llm ttft-verdict` (#4954): the TTFT ratio row as an exit code.
//!
//! The verdict is `apr_test::perf_gate::ttft_verdict`, read against the
//! `scripts/perf-matrix.yaml` this binary was built from. This file prints its
//! lines and exits with its code: 0 GREEN, 1 RED, 2 NOT MEASURED. Nothing on
//! the way returns a `CliError`; a matrix without the row and a run directory
//! that cannot be read are NOT MEASURED like any other missing evidence, so a
//! caller can tell incomplete evidence from RED
//! (`contracts/apr-ttft-ratio-verdict-v1.yaml`, FALSIFY-APR-TTFT-010).

use std::io::Write;
use std::path::Path;

use apr_test::perf_gate::ttft_verdict::{ttft_verdict_dirs, TtftOutcome, TtftPolicy};

use crate::error::Result;
use crate::DirPath;

/// The exit code and the lines to print, for a policy and the run directories.
fn verdict_code(
    policy: std::result::Result<TtftPolicy, String>,
    dirs: &[&Path],
) -> (i32, Vec<String>) {
    match policy {
        Ok(policy) => {
            let verdict = ttft_verdict_dirs(&policy, dirs);
            (verdict.outcome.exit_code(), verdict.lines)
        }
        Err(why) => {
            let outcome = TtftOutcome::NotMeasured;
            let lines = vec![
                format!("NOT MEASURED {why}"),
                format!(
                    "VERDICT {} ttft ratio: the matrix has no usable row",
                    outcome.label()
                ),
            ];
            (outcome.exit_code(), lines)
        }
    }
}

/// Print the verdict over `runs` and exit with its code.
///
/// # Errors
/// Only when stdout cannot be flushed; every verdict, NOT MEASURED included,
/// leaves through its own exit code.
pub(crate) fn run(runs: &[DirPath]) -> Result<()> {
    let dirs: Vec<&Path> = runs.iter().map(DirPath::as_path).collect();
    let (code, lines) = verdict_code(TtftPolicy::compiled(), &dirs);
    let mut out = std::io::stdout().lock();
    for line in &lines {
        writeln!(out, "{line}")?;
    }
    out.flush()?;
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Subcommand;

    #[test]
    fn ttft_verdict_exit_is_2_for_a_run_dir_that_does_not_exist() {
        let root = tempfile::tempdir().expect("tempdir");
        let missing = root.path().join("lambda");
        let (code, lines) = verdict_code(TtftPolicy::compiled(), &[missing.as_path()]);
        assert_eq!(code, 2, "{lines:#?}");
        assert!(
            lines.iter().any(|l| l.contains("v1-provenance.json")),
            "{lines:#?}"
        );
        assert!(
            lines
                .last()
                .is_some_and(|l| l.starts_with("VERDICT NOT MEASURED")),
            "{lines:#?}"
        );
    }

    #[test]
    fn ttft_verdict_exit_is_2_for_a_matrix_without_the_row() {
        let why = "perf-matrix.yaml has no `arms.L3.v1` block (ttft_floor, hosts)";
        let (code, lines) = verdict_code(Err(why.to_string()), &[]);
        assert_eq!(code, 2, "{lines:#?}");
        assert!(lines.iter().any(|l| l.contains(why)), "{lines:#?}");
    }

    #[test]
    fn ttft_verdict_exit_reads_the_row_this_binary_was_built_with() {
        let policy = TtftPolicy::compiled().expect("the built matrix carries arms.L3.v1");
        assert!(!policy.hosts.is_empty() && policy.floor > 0.0, "{policy:?}");
    }

    #[test]
    fn ttft_verdict_exit_needs_at_least_one_run() {
        let parse = |argv: &[&str]| {
            crate::LlmSubcommand::augment_subcommands(clap::Command::new("llm"))
                .try_get_matches_from(argv)
        };
        assert!(parse(&["llm", "ttft-verdict"]).is_err());
        assert!(parse(&["llm", "ttft-verdict", "--run", "/a", "--run", "/b"]).is_ok());
    }
}
