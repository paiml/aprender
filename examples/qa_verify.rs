//! QA Example: Aprender Quality Gates (PMAT-QA-RUST-001)
//!
//! Comprehensive codebase verification - replaces qa-verify.sh, all-modules-qa-verify.sh,
//! and math-qa-verify.sh.
//!
//! # Tests (20 Points)
//!
//! | ID | Test | Points | Criterion |
//! |----|------|--------|-----------|
//! | P034 | Unit tests pass | 5 | `cargo test --lib` exits 0 |
//! | P035 | Test count > 700 | 2 | Parsed from test output |
//! | P036 | Examples build | 2 | `cargo build --examples` exits 0 |
//! | P037 | Clippy clean | 3 | `cargo clippy` exits 0 |
//! | P038 | Format check | 2 | `cargo fmt --check` exits 0 |
//! | P039 | Docs build | 2 | `cargo doc` exits 0 |
//! | P040 | Math section 1 | 1 | Monte Carlo tests pass |
//! | P041 | Math section 2 | 1 | Statistics tests pass |
//! | P042 | Math section 3 | 1 | ML algorithm tests pass |
//! | P043 | Math section 4 | 1 | Optimization tests pass |
//!
//! # Usage
//!
//! ```bash
//! cargo run --example qa_verify
//! cargo run --example qa_verify -- --section 1
//! cargo run --example qa_verify -- --json
//! ```

use std::env;
use std::process::{Command, Stdio};
use std::time::Instant;

// Colors
const RED: &str = "\x1b[0;31m";
const GREEN: &str = "\x1b[0;32m";
const YELLOW: &str = "\x1b[0;33m";
const BLUE: &str = "\x1b[0;34m";
const CYAN: &str = "\x1b[0;36m";
const NC: &str = "\x1b[0m";

struct TestResult {
    id: &'static str,
    name: &'static str,
    passed: bool,
    details: Option<String>,
    points: u32,
}

impl TestResult {
    fn pass(id: &'static str, name: &'static str, points: u32) -> Self {
        Self { id, name, passed: true, details: None, points }
    }
    fn pass_with_details(id: &'static str, name: &'static str, points: u32, details: String) -> Self {
        Self { id, name, passed: true, details: Some(details), points }
    }
    fn fail(id: &'static str, name: &'static str, points: u32, details: String) -> Self {
        Self { id, name, passed: false, details: Some(details), points }
    }
    fn print(&self, json: bool) {
        if json {
            println!(r#"{{"id":"{}","name":"{}","passed":{},"points":{}}}"#,
                self.id, self.name, self.passed, self.points);
        } else {
            let status = if self.passed { format!("{}[PASS]{}", GREEN, NC) }
                         else { format!("{}[FAIL]{}", RED, NC) };
            println!("{} {}: {}", status, self.id, self.name);
            if let Some(ref d) = self.details { println!("       {}", d); }
        }
    }
}

struct QaConfig {
    json: bool,
    section: Option<u32>,
    verbose: bool,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self { json: false, section: None, verbose: false }
    }
}

/// Run a cargo command and return success/failure with output
fn run_cargo(args: &[&str], _timeout_secs: u64) -> (bool, String) {
    let mut cmd = Command::new("cargo");
    cmd.args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    match cmd.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            (output.status.success(), format!("{}{}", stdout, stderr))
        }
        Err(e) => (false, e.to_string()),
    }
}

/// Extract test count from cargo test output
fn extract_test_count(output: &str) -> u32 {
    // Look for "X passed" pattern
    for line in output.lines() {
        if line.contains("passed") {
            for word in line.split_whitespace() {
                if let Ok(n) = word.parse::<u32>() {
                    return n;
                }
            }
        }
    }
    0
}

// === TEST FUNCTIONS ===

fn test_unit_tests() -> TestResult {
    let (success, _output) = run_cargo(&["test", "--lib", "--quiet"], 300);
    if success {
        TestResult::pass("P034", "Unit Tests Pass", 5)
    } else {
        TestResult::fail("P034", "Unit Tests Pass", 5, "Tests failed".to_string())
    }
}

fn test_count() -> TestResult {
    let (success, output) = run_cargo(&["test", "--lib"], 300);
    let _ = success; // Used for side effect
    let count = extract_test_count(&output);
    if count > 700 {
        TestResult::pass_with_details("P035", "Test Count > 700", 2, format!("{} tests", count))
    } else {
        TestResult::fail("P035", "Test Count > 700", 2, format!("Only {} tests", count))
    }
}

fn test_examples_build() -> TestResult {
    let (success, _) = run_cargo(&["build", "--examples", "--quiet"], 180);
    if success {
        TestResult::pass("P036", "Examples Build", 2)
    } else {
        TestResult::fail("P036", "Examples Build", 2, "Build failed".to_string())
    }
}

fn test_clippy() -> TestResult {
    let (success, output) = run_cargo(&["clippy", "--quiet", "--", "-D", "warnings"], 180);
    if success {
        TestResult::pass("P037", "Clippy Clean", 3)
    } else {
        // Count warnings
        let warn_count = output.matches("warning:").count();
        TestResult::fail("P037", "Clippy Clean", 3, format!("{} warnings", warn_count))
    }
}

fn test_format() -> TestResult {
    let (success, _) = run_cargo(&["fmt", "--check", "--quiet"], 60);
    if success {
        TestResult::pass("P038", "Format Check", 2)
    } else {
        TestResult::fail("P038", "Format Check", 2, "Format issues found".to_string())
    }
}

fn test_docs() -> TestResult {
    let (success, _) = run_cargo(&["doc", "--no-deps", "--quiet"], 180);
    if success {
        TestResult::pass("P039", "Docs Build", 2)
    } else {
        TestResult::fail("P039", "Docs Build", 2, "Doc build failed".to_string())
    }
}

fn test_math_monte_carlo() -> TestResult {
    let (success, _) = run_cargo(&["test", "monte_carlo", "--lib", "--quiet"], 60);
    if success {
        TestResult::pass("P040", "Math: Monte Carlo", 1)
    } else {
        TestResult::fail("P040", "Math: Monte Carlo", 1, "Tests failed".to_string())
    }
}

fn test_math_statistics() -> TestResult {
    let (success, _) = run_cargo(&["test", "stats", "--lib", "--quiet"], 60);
    if success {
        TestResult::pass("P041", "Math: Statistics", 1)
    } else {
        TestResult::fail("P041", "Math: Statistics", 1, "Tests failed".to_string())
    }
}

fn test_math_ml() -> TestResult {
    let (success, _) = run_cargo(&["test", "linear_model", "--lib", "--quiet"], 60);
    if success {
        TestResult::pass("P042", "Math: ML Algorithms", 1)
    } else {
        TestResult::fail("P042", "Math: ML Algorithms", 1, "Tests failed".to_string())
    }
}

fn test_math_optimization() -> TestResult {
    let (success, _) = run_cargo(&["test", "optim", "--lib", "--quiet"], 60);
    if success {
        TestResult::pass("P043", "Math: Optimization", 1)
    } else {
        TestResult::fail("P043", "Math: Optimization", 1, "Tests failed".to_string())
    }
}

fn print_header(json: bool) {
    if !json {
        println!("{}╔══════════════════════════════════════════════════════════════╗{}", BLUE, NC);
        println!("{}║       APRENDER QA VERIFY - Quality Gates Verification        ║{}", BLUE, NC);
        println!("{}║       PMAT-QA-RUST-001 Section D (20 Points)                  ║{}", BLUE, NC);
        println!("{}╚══════════════════════════════════════════════════════════════╝{}", BLUE, NC);
        println!();
    }
}

fn print_summary(results: &[TestResult], json: bool, elapsed_secs: f64) {
    let earned: u32 = results.iter().filter(|r| r.passed).map(|r| r.points).sum();
    let total: u32 = results.iter().map(|r| r.points).sum();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.iter().filter(|r| !r.passed).count();

    if json {
        println!(r#"{{"passed":{},"failed":{},"earned":{},"total":{},"elapsed":{:.1}}}"#,
            passed, failed, earned, total, elapsed_secs);
    } else {
        println!();
        println!("{}═══════════════════════════════════════════════════════════════{}", BLUE, NC);
        println!("Total: {}, Passed: {}{}{}, Failed: {}{}{}",
            results.len(), GREEN, passed, NC, if failed > 0 { RED } else { GREEN }, failed, NC);
        println!("Points: {}/{} ({:.0}%)", earned, total, (earned as f64 / total as f64) * 100.0);
        println!("Elapsed: {:.1}s", elapsed_secs);
        println!();

        // Grade
        let pct = (earned as f64 / total as f64) * 100.0;
        let grade = if pct >= 93.0 { "A+" }
            else if pct >= 90.0 { "A" }
            else if pct >= 87.0 { "A-" }
            else if pct >= 83.0 { "B+" }
            else if pct >= 80.0 { "B" }
            else if pct >= 70.0 { "C" }
            else { "F" };

        println!("Grade: {}{}{}", if pct >= 80.0 { GREEN } else { RED }, grade, NC);

        if failed == 0 {
            println!("{}All quality gates PASSED. Ready for production.{}", GREEN, NC);
        } else {
            println!("{}Quality gates FAILED. Remediation required.{}", RED, NC);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = QaConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => { config.json = true; i += 1; }
            "--section" if i + 1 < args.len() => {
                config.section = args[i + 1].parse().ok();
                i += 2;
            }
            "--verbose" | "-v" => { config.verbose = true; i += 1; }
            "--help" | "-h" => {
                println!("Usage: cargo run --example qa_verify [OPTIONS]");
                println!("  --json         JSON output");
                println!("  --section N    Run specific section (1-4)");
                println!("  --verbose      Verbose output");
                return;
            }
            _ => { i += 1; }
        }
    }

    print_header(config.json);

    let start = Instant::now();
    let mut results = Vec::new();

    if !config.json {
        println!("{}=== Section D: qa_verify.rs Tests (20 Points) ==={}", YELLOW, NC);
        println!();
    }

    // Section 1: Mandatory Gates (12 points)
    if config.section.is_none() || config.section == Some(1) {
        if !config.json {
            println!("{}--- Mandatory Gates ---{}", CYAN, NC);
        }
        results.push(test_unit_tests()); results.last().unwrap().print(config.json);
        results.push(test_count()); results.last().unwrap().print(config.json);
        results.push(test_examples_build()); results.last().unwrap().print(config.json);
        results.push(test_clippy()); results.last().unwrap().print(config.json);
        results.push(test_format()); results.last().unwrap().print(config.json);
        results.push(test_docs()); results.last().unwrap().print(config.json);
    }

    // Section 2: Mathematical Correctness (4 points)
    if config.section.is_none() || config.section == Some(2) {
        if !config.json {
            println!();
            println!("{}--- Mathematical Correctness ---{}", CYAN, NC);
        }
        results.push(test_math_monte_carlo()); results.last().unwrap().print(config.json);
        results.push(test_math_statistics()); results.last().unwrap().print(config.json);
        results.push(test_math_ml()); results.last().unwrap().print(config.json);
        results.push(test_math_optimization()); results.last().unwrap().print(config.json);
    }

    let elapsed = start.elapsed().as_secs_f64();
    print_summary(&results, config.json, elapsed);

    let failed = results.iter().filter(|r| !r.passed).count();
    std::process::exit(if failed == 0 { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_count() {
        let output = "test result: ok. 742 passed; 0 failed; 0 ignored";
        assert_eq!(extract_test_count(output), 742);
    }

    #[test]
    fn test_config_default() {
        let config = QaConfig::default();
        assert!(!config.json);
        assert!(config.section.is_none());
    }
}
