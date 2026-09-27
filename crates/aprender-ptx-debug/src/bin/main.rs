//! aprender-ptx-debug CLI
//!
//! Pure Rust PTX debugging and static analysis tool.
//!
//! Usage:
//!   aprender-ptx-debug analyze <file.ptx> [--falsify] [--min-score N]
//!   aprender-ptx-debug gen-fkr <file.ptx> [-o tests.rs]
//!
//! Argument parsing is declarative and lives in `trueno_ptx_debug::cli`.

use std::process;

// Imported anonymously: `clap::Parser` would otherwise collide with the PTX
// `Parser` used by the library.
use clap::Parser as _;

use trueno_ptx_debug::cli::{exit_code_for_parse_error, Cli};

/// #4062: `apr ptx-debug` runs the same code; this binary is on its way out.
const DEPRECATED: &str = "warning: `aprender-ptx-debug` is deprecated and will be removed; run `apr ptx-debug` instead (the same code path). See paiml/aprender#4057.";

fn main() {
    eprintln!("{DEPRECATED}");
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(err) => {
            // clap picks stdout for --help/--version and stderr for real
            // failures; the exit code is chosen the same way.
            let _ = err.print();
            process::exit(exit_code_for_parse_error(&err));
        }
    };

    match trueno_ptx_debug::run::run(cli.command) {
        Ok(code) => process::exit(code),
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
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

fn cmd_gen_fkr(opts: GenFkrArgs) -> Result<(), String> {
    let result = analyze_ptx_file(&opts.file)?;
    let fkr_tests = generate_fkr_tests(&result);
    write_or_print(&fkr_tests, opts.output, "FKR tests")
}
