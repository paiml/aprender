//! aprender-ptx-debug CLI
//!
//! Pure Rust PTX debugging and static analysis tool.
//!
//! Usage:
//!   aprender-ptx-debug analyze <file.ptx> [--falsify] [--min-score N]
//!   aprender-ptx-debug gen-fkr <file.ptx> [-o tests.rs]
//!
//! Argument parsing is declarative and lives in `trueno_ptx_debug::cli`.

use std::fs;
use std::process;

// Imported anonymously: `clap::Parser` would otherwise collide with the PTX
// `Parser` used below.
use clap::Parser as _;

use trueno_ptx_debug::bugs::BugRegistry;
use trueno_ptx_debug::cli::{
    exit_code_for_parse_error, version_string, AnalyzeArgs, Cli, Command, GenFkrArgs,
};
use trueno_ptx_debug::falsification::FalsificationRegistry;
use trueno_ptx_debug::output::{generate_fkr_tests, generate_html_report, AnalysisResult};
use trueno_ptx_debug::parser::Parser;

fn main() {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(err) => {
            // clap picks stdout for --help/--version and stderr for real
            // failures; the exit code is chosen the same way.
            let _ = err.print();
            process::exit(exit_code_for_parse_error(&err));
        }
    };

    let result = match cli.command {
        Command::Analyze(args) => cmd_analyze(args),
        Command::GenFkr(args) => cmd_gen_fkr(args),
        Command::Version => {
            print!("{}", version_string());
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

/// Print analysis results as JSON.
fn print_json_report(
    result: &AnalysisResult,
    report: &trueno_ptx_debug::falsification::FalsificationReport,
) {
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
fn print_text_report(
    result: &AnalysisResult,
    report: &trueno_ptx_debug::falsification::FalsificationReport,
) {
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

/// Determine the process exit code from the analysis results.
fn exit_for_score(
    report: &trueno_ptx_debug::falsification::FalsificationReport,
    score: f64,
    min_score: f64,
) {
    if report.has_critical_bugs() {
        process::exit(3);
    } else if score < min_score {
        process::exit(2);
    } else if score < 90.0 {
        process::exit(1);
    }
}

fn cmd_analyze(opts: AnalyzeArgs) -> Result<(), String> {
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

    exit_for_score(
        &result.falsification_report,
        result.falsification_score,
        opts.min_score,
    );

    Ok(())
}

/// Read a PTX file, parse it, run analysis, and return the result.
fn analyze_ptx_file(file_path: &str) -> Result<AnalysisResult, String> {
    let ptx_source = fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read {}: {}", file_path, e))?;

    let mut parser = Parser::new(&ptx_source).map_err(|e| format!("Parse error: {}", e))?;
    let module = parser.parse().map_err(|e| format!("Parse error: {}", e))?;

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

fn cmd_gen_fkr(opts: GenFkrArgs) -> Result<(), String> {
    let result = analyze_ptx_file(&opts.file)?;
    let fkr_tests = generate_fkr_tests(&result);
    write_or_print(&fkr_tests, opts.output, "FKR tests")
}
