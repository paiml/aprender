//! Command surface for PTX static analysis, shared by every front end.
//!
//! This module used to live in `src/bin/main.rs` behind the standalone
//! `aprender-ptx-debug` binary. That binary is gone (APR-MONO: one installed
//! binary, `apr`); the capability is reached as `apr ptx-debug <COMMAND>`.
//!
//! The clap [`PtxDebugCommand`] enum lives here rather than in `apr-cli` on
//! purpose: `apr-cli` embeds this exact enum with `#[command(subcommand)]`, so
//! there is no second copy of the argument list that can drift out of sync
//! with this one (defect class #2418 — an argument silently dropped in a
//! re-homed CLI).

use std::fs;
use std::path::{Path, PathBuf};

use clap::Subcommand;

use crate::bugs::BugRegistry;
use crate::falsification::{FalsificationRegistry, FalsificationReport};
use crate::output::{generate_fkr_tests, generate_html_report, AnalysisResult};
use crate::parser::Parser;

/// Default value of `--min-score`: the score at or above which `analyze`
/// does not report failure.
pub const DEFAULT_MIN_SCORE: f64 = 70.0;

/// Score at or above which `analyze` exits 0 rather than 1.
pub const CLEAN_SCORE: f64 = 90.0;

/// Failure modes of the PTX analysis commands.
#[derive(Debug, thiserror::Error)]
pub enum PtxCliError {
    /// A file could not be read or written.
    #[error("failed to read/write {path}: {source}")]
    Io {
        /// The path that could not be read or written.
        path: PathBuf,
        /// The underlying I/O failure.
        #[source]
        source: std::io::Error,
    },

    /// The PTX source did not parse.
    #[error("parse error: {0}")]
    Parse(String),
}

impl PtxCliError {
    /// True when this error is "the input file is not there / not readable",
    /// which callers map onto their own not-found exit code.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        match self {
            Self::Io { source, .. } => source.kind() == std::io::ErrorKind::NotFound,
            Self::Parse(_) => false,
        }
    }
}

/// PTX debugging commands.
///
/// Argument names, short flags and defaults are byte-for-byte the ones the
/// standalone `trueno-ptx-debug` binary accepted, so existing invocations
/// keep working after replacing the program name with `apr ptx-debug`.
#[derive(Debug, Clone, Subcommand)]
pub enum PtxDebugCommand {
    /// Analyze a PTX file for bugs and score it against the 100-point
    /// Popperian falsification framework.
    ///
    /// Exit codes: 0 when the score is >= 90, 1 when it is 70..90, 2 when it
    /// is below `--min-score`, and 3 when any critical bug is detected.
    Analyze {
        /// Path to the PTX source file to analyze.
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Run the full 100-point falsification framework.
        ///
        /// Accepted for command-line compatibility. The full framework is
        /// always evaluated, so this flag selects the behaviour that is
        /// already the default rather than adding to it.
        #[arg(long)]
        falsify: bool,

        /// Report failure (exit 2) when the falsification score is below N.
        #[arg(long, value_name = "N", default_value_t = DEFAULT_MIN_SCORE)]
        min_score: f64,

        /// Also write a standalone HTML report to this path.
        #[arg(long, value_name = "FILE")]
        html: Option<PathBuf>,

        /// Emit the report as JSON instead of human-readable text.
        #[arg(long)]
        json: bool,
    },

    /// Generate FKR (falsifiable kernel regression) tests for jugar-probar
    /// from a PTX file.
    #[command(name = "gen-fkr")]
    GenFkr {
        /// Path to the PTX source file to generate tests from.
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Write the generated tests to this path instead of stdout.
        #[arg(short = 'o', long = "output", value_name = "FILE")]
        output: Option<PathBuf>,
    },
}

/// Read a PTX file, parse it, run every analysis pass, and return the result.
///
/// # Errors
///
/// [`PtxCliError::Io`] when `file_path` cannot be read, [`PtxCliError::Parse`]
/// when the contents are not valid PTX.
pub fn analyze_ptx_file(file_path: &Path) -> Result<AnalysisResult, PtxCliError> {
    let ptx_source = fs::read_to_string(file_path).map_err(|source| PtxCliError::Io {
        path: file_path.to_path_buf(),
        source,
    })?;

    let mut parser = Parser::new(&ptx_source).map_err(|e| PtxCliError::Parse(format!("{e}")))?;
    let module = parser
        .parse()
        .map_err(|e| PtxCliError::Parse(format!("{e}")))?;

    let module_name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let registry = FalsificationRegistry::new();
    let report = registry.evaluate(&module);
    let bugs = BugRegistry::new();
    Ok(AnalysisResult::new(&module_name, report, bugs))
}

/// Map an analysis outcome onto the documented exit-code table.
///
/// `3` (critical bug) outranks `2` (below `--min-score`), which outranks `1`
/// (passed, but under [`CLEAN_SCORE`]).
#[must_use]
pub fn exit_code_for(report: &FalsificationReport, score: f64, min_score: f64) -> u8 {
    if report.has_critical_bugs() {
        3
    } else if score < min_score {
        2
    } else if score < CLEAN_SCORE {
        1
    } else {
        0
    }
}

/// Render the analysis result as JSON.
#[must_use]
pub fn render_json_report(result: &AnalysisResult, report: &FalsificationReport) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"module\": \"{}\",\n", result.module_name));
    out.push_str(&format!(
        "  \"score\": {:.1},\n",
        result.falsification_score
    ));
    out.push_str(&format!("  \"confidence\": {:.2},\n", result.confidence));
    out.push_str(&format!("  \"earned_points\": {},\n", report.earned_points));
    out.push_str(&format!("  \"total_points\": {},\n", report.total_points));
    out.push_str(&format!(
        "  \"critical_bugs_absent\": {}\n",
        report.critical_bugs_absent()
    ));
    out.push_str("}\n");
    out
}

/// Render the analysis result as human-readable text.
#[must_use]
pub fn render_text_report(result: &AnalysisResult, report: &FalsificationReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("PTX Analysis Report: {}\n", result.module_name));
    out.push_str("=========================================\n");
    out.push_str(&format!("Score: {:.1}/100\n", result.falsification_score));
    out.push_str(&format!("Confidence: {:.1}%\n", result.confidence * 100.0));
    out.push_str(&format!(
        "Points: {}/{}\n\n",
        report.earned_points, report.total_points
    ));

    let failed = report.failed_tests();
    if failed.is_empty() {
        out.push_str("All tests passed!\n");
    } else {
        out.push_str(&format!("Failed tests ({}):\n", failed.len()));
        for (id, category, desc, _result) in failed {
            out.push_str(&format!("  {id} [{category}]: {desc}\n"));
        }
    }
    out
}

/// Run one [`PtxDebugCommand`], printing its report to stdout.
///
/// Returns the process exit code the command wants; `0` means success. The
/// caller decides how to surface a non-zero code — this function never calls
/// `std::process::exit`, so it stays testable.
///
/// # Errors
///
/// Propagates [`PtxCliError`] from reading, parsing, or writing files.
pub fn run(command: &PtxDebugCommand) -> Result<u8, PtxCliError> {
    match command {
        PtxDebugCommand::Analyze {
            file,
            falsify: _,
            min_score,
            html,
            json,
        } => run_analyze(file, *min_score, html.as_deref(), *json),
        PtxDebugCommand::GenFkr { file, output } => {
            run_gen_fkr(file, output.as_deref()).map(|()| 0)
        }
    }
}

/// `analyze` implementation. Returns the exit code the report earns.
///
/// # Errors
///
/// Propagates [`PtxCliError`] from reading the PTX or writing the HTML report.
pub fn run_analyze(
    file: &Path,
    min_score: f64,
    html: Option<&Path>,
    json: bool,
) -> Result<u8, PtxCliError> {
    let result = analyze_ptx_file(file)?;
    let report = &result.falsification_report;

    if json {
        print!("{}", render_json_report(&result, report));
    } else {
        print!("{}", render_text_report(&result, report));
    }

    if let Some(html_path) = html {
        let html_body = generate_html_report(&result);
        fs::write(html_path, html_body).map_err(|source| PtxCliError::Io {
            path: html_path.to_path_buf(),
            source,
        })?;
        println!("\nHTML report written to: {}", html_path.display());
    }

    Ok(exit_code_for(report, result.falsification_score, min_score))
}

/// `gen-fkr` implementation: write generated tests to `output`, or stdout.
///
/// # Errors
///
/// Propagates [`PtxCliError`] from reading the PTX or writing the test file.
pub fn run_gen_fkr(file: &Path, output: Option<&Path>) -> Result<(), PtxCliError> {
    let result = analyze_ptx_file(file)?;
    let fkr_tests = generate_fkr_tests(&result);
    match output {
        Some(path) => {
            fs::write(path, &fkr_tests).map_err(|source| PtxCliError::Io {
                path: path.to_path_buf(),
                source,
            })?;
            println!("FKR tests written to: {}", path.display());
        }
        None => println!("{fkr_tests}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;

    /// Minimal harness so the `Subcommand` enum can be parsed on its own.
    #[derive(Debug, clap::Parser)]
    #[command(name = "ptx-debug")]
    struct Harness {
        #[command(subcommand)]
        command: PtxDebugCommand,
    }

    fn parse(argv: &[&str]) -> PtxDebugCommand {
        Harness::try_parse_from(argv).expect("argv parses").command
    }

    #[test]
    fn analyze_accepts_every_flag_the_standalone_binary_accepted() {
        let cmd = parse(&[
            "ptx-debug",
            "analyze",
            "kernel.ptx",
            "--falsify",
            "--min-score",
            "95",
            "--html",
            "report.html",
            "--json",
        ]);
        match cmd {
            PtxDebugCommand::Analyze {
                file,
                falsify,
                min_score,
                html,
                json,
            } => {
                assert_eq!(file, PathBuf::from("kernel.ptx"));
                assert!(falsify);
                assert_eq!(min_score, 95.0);
                assert_eq!(html, Some(PathBuf::from("report.html")));
                assert!(json);
            }
            PtxDebugCommand::GenFkr { .. } => panic!("expected Analyze"),
        }
    }

    #[test]
    fn analyze_min_score_defaults_to_seventy() {
        match parse(&["ptx-debug", "analyze", "k.ptx"]) {
            PtxDebugCommand::Analyze {
                min_score,
                falsify,
                html,
                json,
                ..
            } => {
                assert_eq!(min_score, DEFAULT_MIN_SCORE);
                assert!(!falsify);
                assert_eq!(html, None);
                assert!(!json);
            }
            PtxDebugCommand::GenFkr { .. } => panic!("expected Analyze"),
        }
    }

    #[test]
    fn gen_fkr_keeps_the_short_o_output_flag() {
        match parse(&["ptx-debug", "gen-fkr", "k.ptx", "-o", "tests.rs"]) {
            PtxDebugCommand::GenFkr { file, output } => {
                assert_eq!(file, PathBuf::from("k.ptx"));
                assert_eq!(output, Some(PathBuf::from("tests.rs")));
            }
            PtxDebugCommand::Analyze { .. } => panic!("expected GenFkr"),
        }
    }

    #[test]
    fn gen_fkr_without_output_writes_to_stdout() {
        match parse(&["ptx-debug", "gen-fkr", "k.ptx"]) {
            PtxDebugCommand::GenFkr { output, .. } => assert_eq!(output, None),
            PtxDebugCommand::Analyze { .. } => panic!("expected GenFkr"),
        }
    }

    #[test]
    fn analyze_without_a_file_is_refused() {
        let err = Harness::try_parse_from(["ptx-debug", "analyze"])
            .expect_err("a missing PTX path must be refused, not defaulted");
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn analyze_rejects_a_non_numeric_min_score() {
        let err = Harness::try_parse_from(["ptx-debug", "analyze", "k.ptx", "--min-score", "high"])
            .expect_err("--min-score must reject non-numeric input");
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn missing_file_is_reported_as_not_found() {
        let err = analyze_ptx_file(Path::new("/nonexistent/definitely-not-here.ptx"))
            .expect_err("a missing PTX file must not analyze successfully");
        assert!(
            err.is_not_found(),
            "expected a not-found error, got {err:?}"
        );
    }

    /// Non-PTX input must not come back clean.
    ///
    /// The parser is permissive by design — it does not reject a file for
    /// lacking `.version`/`.target`; the falsification framework does, as
    /// F001/F002/F003. So the assertion that excludes an outcome here is
    /// "the syntax-validity tests FAILED", not "the parse errored".
    #[test]
    fn non_ptx_input_fails_the_syntax_validity_tests() {
        let dir = std::env::temp_dir().join("apr-ptx-debug-cli-test");
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("garbage.ptx");
        fs::write(&path, "\u{1}\u{2}\u{3} not ptx at all @@@").expect("write fixture");

        let result = analyze_ptx_file(&path);
        let _ = fs::remove_file(&path);
        let result = result.expect("the permissive parser accepts this input");

        let failed: Vec<String> = result
            .falsification_report
            .failed_tests()
            .into_iter()
            .map(|(id, _, _, _)| id.to_string())
            .collect();
        for required in ["F001", "F002", "F003"] {
            assert!(
                failed.iter().any(|id| id == required),
                "{required} must FAIL on a file with no .version/.target/address_size; \
                 failed set was {failed:?}"
            );
        }
    }

    /// `--min-score` must be able to turn a report that would otherwise pass
    /// into a failure. Without this the flag is decorative (#2418 class).
    #[test]
    fn min_score_can_turn_a_passing_report_into_a_failure() {
        let dir = std::env::temp_dir().join("apr-ptx-debug-cli-test");
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("min_score.ptx");
        fs::write(&path, "\u{1}\u{2}\u{3} not ptx at all @@@").expect("write fixture");

        let lenient = run_analyze(&path, 0.0, None, true);
        let strict = run_analyze(&path, 100.0, None, true);
        let _ = fs::remove_file(&path);

        let lenient = lenient.expect("analysis runs");
        let strict = strict.expect("analysis runs");
        assert_ne!(
            lenient, strict,
            "--min-score 0 and --min-score 100 must not produce the same exit code"
        );
        assert_eq!(strict, 2, "a score under --min-score must exit 2");
    }

    #[test]
    fn exit_code_table_matches_the_documented_contract() {
        // A report whose only variable is the score: build one by evaluating
        // an empty module, then drive the pure mapping directly.
        let registry = FalsificationRegistry::new();
        let module = Parser::new("// empty\n")
            .expect("lexer accepts a comment-only module")
            .parse()
            .expect("comment-only module parses");
        let report = registry.evaluate(&module);

        // Below --min-score => 2 (unless critical bugs, which outrank it).
        let expected_low = if report.has_critical_bugs() { 3 } else { 2 };
        assert_eq!(exit_code_for(&report, 10.0, 70.0), expected_low);

        // At/above CLEAN_SCORE => 0.
        let expected_clean = if report.has_critical_bugs() { 3 } else { 0 };
        assert_eq!(exit_code_for(&report, 99.0, 70.0), expected_clean);

        // Between --min-score and CLEAN_SCORE => 1.
        let expected_warn = if report.has_critical_bugs() { 3 } else { 1 };
        assert_eq!(exit_code_for(&report, 80.0, 70.0), expected_warn);
    }
}
