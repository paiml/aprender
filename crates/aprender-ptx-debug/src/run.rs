//! Command execution shared by the `aprender-ptx-debug` binary and `apr ptx-debug`
//! (#4062): one implementation, so the two surfaces cannot drift.

use std::fs;

use crate::bugs::BugRegistry;
use crate::cli::{version_string, AnalyzeArgs, Command, GenFkrArgs};
use crate::falsification::{FalsificationRegistry, FalsificationReport};
use crate::output::{generate_fkr_tests, generate_html_report, AnalysisResult};
use crate::parser::Parser;

/// Run one subcommand and return its process exit code.
///
/// `analyze` returns its score verdict (see [`exit_code_for_score`]); the other
/// subcommands return 0. `Err` is a failure to read, parse or write a file.
pub fn run(command: Command) -> Result<i32, String> {
    match command {
        Command::Analyze(args) => cmd_analyze(args),
        Command::GenFkr(args) => cmd_gen_fkr(args),
        Command::Version => {
            print!("{}", version_string());
            Ok(0)
        }
    }
}

/// Print analysis results as JSON.
fn print_json_report(result: &AnalysisResult, report: &FalsificationReport) {
    println!("{{");
    println!("  \"module\": \"{}\",", result.module_name);
    println!("  \"score\": {:.1},", result.falsification_score);
    println!("  \"confidence\": {:.2},", result.confidence);
    println!("  \"earned_points\": {},", report.earned_points);
    println!("  \"total_points\": {},", report.total_points);
    println!(
        "  \"critical_bugs_absent\": {}",
        report.critical_bugs_absent()
    );
    println!("}}");
}

/// Print analysis results as human-readable text.
fn print_text_report(result: &AnalysisResult, report: &FalsificationReport) {
    println!("PTX Analysis Report: {}", result.module_name);
    println!("=========================================");
    println!("Score: {:.1}/100", result.falsification_score);
    println!("Confidence: {:.1}%", result.confidence * 100.0);
    println!("Points: {}/{}", report.earned_points, report.total_points);
    println!();

    let failed = report.failed_tests();
    if failed.is_empty() {
        println!("All tests passed!");
    } else {
        println!("Failed tests ({}):", failed.len());
        for (id, category, desc, _result) in failed {
            println!("  {} [{}]: {}", id, category, desc);
        }
    }
}

/// The verdict exit code for an analysis: 3 = critical bugs present,
/// 2 = score below `--min-score`, 1 = score below 90, 0 = clean.
#[must_use]
pub fn exit_code_for_score(report: &FalsificationReport, score: f64, min_score: f64) -> i32 {
    if report.has_critical_bugs() {
        3
    } else if score < min_score {
        2
    } else if score < 90.0 {
        1
    } else {
        0
    }
}

fn cmd_analyze(opts: AnalyzeArgs) -> Result<i32, String> {
    let result = analyze_ptx_file(&opts.file)?;

    // Output results
    if opts.json {
        print_json_report(&result, &result.falsification_report);
    } else {
        print_text_report(&result, &result.falsification_report);
    }

    // Write HTML report if requested
    if let Some(html_path) = opts.html {
        let html = generate_html_report(&result);
        fs::write(&html_path, html).map_err(|e| format!("Failed to write {}: {}", html_path, e))?;
        println!("\nHTML report written to: {}", html_path);
    }

    Ok(exit_code_for_score(
        &result.falsification_report,
        result.falsification_score,
        opts.min_score,
    ))
}

/// Read a PTX file, parse it, run analysis, and return the result.
fn analyze_ptx_file(file_path: &str) -> Result<AnalysisResult, String> {
    let ptx_source = fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read {}: {}", file_path, e))?;

    let mut parser = Parser::new(&ptx_source).map_err(|e| format!("Parse error: {}", e))?;
    let module = parser.parse().map_err(|e| format!("Parse error: {}", e))?;

    // A PTX module opens with `.version`. Without it (an empty or comment-only
    // file) there is nothing to analyze; scoring it printed "Score: 97.1/100" and
    // exited 0 (#4079).
    if module.version.0 == 0 {
        return Err(format!(
            "{}: not a PTX module (no .version directive)",
            file_path
        ));
    }

    let module_name = std::path::Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let registry = FalsificationRegistry::new();
    let report = registry.evaluate(&module);
    let bugs = BugRegistry::new();
    Ok(AnalysisResult::new(&module_name, report, bugs))
}

/// Write generated content to a file, or print to stdout if no path is given.
fn write_or_print(content: &str, output_path: Option<String>, label: &str) -> Result<(), String> {
    match output_path {
        Some(path) => {
            fs::write(&path, content).map_err(|e| format!("Failed to write {}: {}", path, e))?;
            println!("{} written to: {}", label, path);
        }
        None => println!("{}", content),
    }
    Ok(())
}

fn cmd_gen_fkr(opts: GenFkrArgs) -> Result<i32, String> {
    let result = analyze_ptx_file(&opts.file)?;
    let fkr_tests = generate_fkr_tests(&result);
    write_or_print(&fkr_tests, opts.output, "FKR tests")?;
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falsification::TestResult;
    use std::path::PathBuf;
    use std::process::Command as Proc;

    const VALID_PTX: &str = ".version 8.0\n.target sm_70\n.address_size 64\n\n.entry simple()\n{\n    .reg .u32 %r<4>;\n    mov.u32 %r0, 0;\n    ret;\n}\n";

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ptxdbg-run-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_ptx(name: &str, body: &str) -> String {
        let path = tmp(name).join("k.ptx");
        std::fs::write(&path, body).expect("write ptx");
        path.to_string_lossy().into_owned()
    }

    fn report_with(
        results: Vec<(String, crate::falsification::Category, String, TestResult)>,
    ) -> FalsificationReport {
        FalsificationReport {
            results,
            score: 0.0,
            earned_points: 0,
            total_points: 0,
            confidence: 0.0,
        }
    }

    fn analyze_args(file: &str, min_score: f64, json: bool, html: Option<String>) -> AnalyzeArgs {
        AnalyzeArgs {
            file: file.to_string(),
            falsify: false,
            min_score,
            html,
            json,
        }
    }

    #[test]
    fn exit_code_for_score_table() {
        let clean = report_with(Vec::new());
        assert_eq!(exit_code_for_score(&clean, 95.0, 70.0), 0);
        assert_eq!(exit_code_for_score(&clean, 90.0, 70.0), 0);
        assert_eq!(exit_code_for_score(&clean, 89.9, 70.0), 1);
        assert_eq!(exit_code_for_score(&clean, 70.0, 70.0), 1);
        assert_eq!(exit_code_for_score(&clean, 69.9, 70.0), 2);
        assert_eq!(exit_code_for_score(&clean, 95.0, 96.0), 2);
        assert_eq!(exit_code_for_score(&clean, 95.0, 95.0), 0);
        let critical = report_with(vec![(
            "F082".to_string(),
            crate::falsification::Category::KnownBugs,
            "critical".to_string(),
            TestResult::Fail {
                evidence: "x".to_string(),
                location: None,
            },
        )]);
        assert_eq!(exit_code_for_score(&critical, 100.0, 0.0), 3);
    }

    #[test]
    fn analyze_ptx_file_rejects_versionless_and_accepts_valid() {
        let empty = write_ptx("empty", "");
        let err = analyze_ptx_file(&empty).expect_err("versionless file must be rejected");
        assert!(err.contains("not a PTX module"), "{err}");
        let ok = analyze_ptx_file(&write_ptx("valid", VALID_PTX)).expect("valid ptx analyzes");
        assert_eq!(ok.module_name, "k");
    }

    #[test]
    fn cmd_analyze_returns_verdict_code() {
        let file = write_ptx("verdict", VALID_PTX);
        let result = analyze_ptx_file(&file).expect("analyze");
        let expect_hi = exit_code_for_score(
            &result.falsification_report,
            result.falsification_score,
            101.0,
        );
        assert!(expect_hi == 2 || expect_hi == 3, "{expect_hi}");
        assert_eq!(
            cmd_analyze(analyze_args(&file, 101.0, false, None)),
            Ok(expect_hi)
        );
        assert_eq!(
            run(Command::Analyze(analyze_args(&file, 101.0, true, None))),
            Ok(expect_hi)
        );
        let expect_lo = exit_code_for_score(
            &result.falsification_report,
            result.falsification_score,
            0.0,
        );
        assert_eq!(
            cmd_analyze(analyze_args(&file, 0.0, false, None)),
            Ok(expect_lo)
        );
        assert!(cmd_analyze(analyze_args("/nonexistent/x.ptx", 0.0, false, None)).is_err());
    }

    #[test]
    fn cmd_analyze_writes_html() {
        let file = write_ptx("html", VALID_PTX);
        let html = tmp("html-out").join("r.html");
        let html_s = html.to_string_lossy().into_owned();
        cmd_analyze(analyze_args(&file, 0.0, false, Some(html_s))).expect("analyze with html");
        let body = std::fs::read_to_string(&html).expect("html written");
        assert!(
            body.contains("<html") || body.contains("<!DOCTYPE"),
            "{body:.80}"
        );
    }

    #[test]
    fn version_returns_zero() {
        assert_eq!(run(Command::Version), Ok(0));
    }

    #[test]
    fn gen_fkr_returns_zero_and_writes_output() {
        let file = write_ptx("fkr", VALID_PTX);
        let out = tmp("fkr-out").join("t.rs");
        let out_s = out.to_string_lossy().into_owned();
        let args = GenFkrArgs {
            file: file.clone(),
            output: Some(out_s),
        };
        assert_eq!(cmd_gen_fkr(args.clone()), Ok(0));
        assert_eq!(run(Command::GenFkr(args)), Ok(0));
        let body = std::fs::read_to_string(&out).expect("fkr written");
        assert!(!body.trim().is_empty());
        let missing = GenFkrArgs {
            file: "/nonexistent/x.ptx".to_string(),
            output: None,
        };
        assert!(cmd_gen_fkr(missing).is_err());
    }

    #[test]
    fn write_or_print_writes_file_and_reports_errors() {
        let out = tmp("wop").join("o.txt");
        write_or_print("hello", Some(out.to_string_lossy().into_owned()), "L").expect("write");
        assert_eq!(std::fs::read_to_string(&out).expect("read"), "hello");
        let bad = tmp("wop-bad").join("no-such-dir").join("o.txt");
        assert!(write_or_print("x", Some(bad.to_string_lossy().into_owned()), "L").is_err());
    }

    /// Child half of the stdout tests: only acts when re-executed by
    /// [`child_stdout`] with `--nocapture`, so the real fd 1 is observable.
    #[test]
    fn stdout_child_entry() {
        let Ok(mode) = std::env::var("PTXDBG_CHILD_MODE") else {
            return;
        };
        let file = std::env::var("PTXDBG_CHILD_FILE").expect("file env");
        match mode.as_str() {
            "text" => {
                run(Command::Analyze(analyze_args(&file, 0.0, false, None))).expect("run");
            }
            "json" => {
                run(Command::Analyze(analyze_args(&file, 0.0, true, None))).expect("run");
            }
            "fkr" => {
                cmd_gen_fkr(GenFkrArgs { file, output: None }).expect("fkr");
            }
            "wop" => {
                write_or_print("PAYLOAD-XYZ", None, "L").expect("wop");
            }
            other => panic!("unknown mode {other}"),
        }
    }

    fn child_stdout(mode: &str, file: &str) -> String {
        let out = Proc::new(std::env::current_exe().expect("current_exe"))
            .args([
                "--exact",
                "run::tests::stdout_child_entry",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("PTXDBG_CHILD_MODE", mode)
            .env("PTXDBG_CHILD_FILE", file)
            .output()
            .expect("spawn child test binary");
        assert!(out.status.success(), "child failed: {:?}", out.status);
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    #[test]
    fn text_report_goes_to_stdout() {
        let s = child_stdout("text", &write_ptx("s-text", VALID_PTX));
        assert!(s.contains("PTX Analysis Report: k"), "{s}");
        assert!(s.contains("Score: "), "{s}");
        assert!(s.contains("Points: "), "{s}");
    }

    #[test]
    fn json_report_goes_to_stdout() {
        let s = child_stdout("json", &write_ptx("s-json", VALID_PTX));
        assert!(s.contains("\"module\": \"k\""), "{s}");
        assert!(s.contains("\"critical_bugs_absent\""), "{s}");
        assert!(!s.contains("PTX Analysis Report"), "{s}");
    }

    #[test]
    fn write_or_print_none_prints_content() {
        let s = child_stdout("wop", "unused");
        assert!(s.contains("PAYLOAD-XYZ"), "{s}");
    }

    #[test]
    fn gen_fkr_without_output_prints_tests() {
        let s = child_stdout("fkr", &write_ptx("s-fkr", VALID_PTX));
        assert!(s.contains("fn "), "{s}");
    }
}
