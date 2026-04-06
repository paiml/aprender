//! CLI output formatting.
//!
//! This module contains all output formatting functions for the CLI.
//! Extracted to enable testing of output generation.

use crate::edd::{EmcComplianceReport, ExperimentResult};

/// Print version information.
pub fn print_version() {
    println!("simular {}", env!("CARGO_PKG_VERSION"));
}

/// Print help message.
pub fn print_help() {
    println!(
        r"simular - Unified Simulation Engine for the Sovereign AI Stack

USAGE:
    simular <COMMAND> [OPTIONS]

COMMANDS:
    run <experiment.yaml>       Run an experiment
        --seed <N>              Override the experiment seed
        -v, --verbose           Enable verbose output

    verify <experiment.yaml>    Verify reproducibility across multiple runs
        --runs <N>              Number of verification runs (default: 3)

    emc-check <experiment.yaml> Check EMC compliance and generate report

    emc-validate <file.emc.yaml> Validate an EMC file against the schema

    list-emc                    List available EMCs in the library

    help                        Show this help message
    version                     Show version information

EXAMPLES:
    simular run experiments/harmonic_oscillator.yaml
    simular run experiments/harmonic_oscillator.yaml --seed 12345
    simular verify experiments/harmonic_oscillator.yaml --runs 5
    simular emc-check experiments/littles_law.yaml

EDD COMPLIANCE:
    All experiments are validated against the four pillars of EDD:
    1. Prove It  - Every simulation has an EMC reference
    2. Fail It   - Verification tests are executed
    3. Seed It   - Explicit seed is required
    4. Falsify It - Falsification criteria are checked

For more information, see: https://github.com/paiml/simular
"
    );
}

/// Print experiment result.
///
/// # Arguments
///
/// * `result` - The experiment result to display
/// * `verbose` - Whether to show verbose output
pub fn print_experiment_result(result: &ExperimentResult, verbose: bool) {
    let status = if result.passed { "PASSED" } else { "FAILED" };
    let status_symbol = if result.passed { "✓" } else { "✗" };

    print_header(&result.name, &result.experiment_id, result.seed);
    print_verification(&result.verification, verbose);
    print_falsification(&result.falsification, verbose);
    print_reproducibility(result.reproducibility.as_ref());
    print_execution(&result.execution);
    print_warnings(&result.warnings);
    print_footer(status_symbol, status);
}

fn print_header(name: &str, id: &str, seed: u64) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Experiment: {name}");
    println!("ID: {id}");
    println!("Seed: {seed}");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

fn print_verification(verification: &crate::edd::VerificationSummary, verbose: bool) {
    println!("Verification Tests:");
    println!("  Total:  {}", verification.total);
    println!("  Passed: {}", verification.passed);
    println!("  Failed: {}", verification.failed);

    if verbose && !verification.tests.is_empty() {
        println!();
        for test in &verification.tests {
            let sym = if test.passed { "✓" } else { "✗" };
            println!("  {} {}: {}", sym, test.id, test.name);
            if let Some(ref err) = test.error {
                if !test.passed {
                    println!("      Error: {err}");
                }
            }
        }
    }
}

fn print_falsification(falsification: &crate::edd::FalsificationSummary, verbose: bool) {
    println!("\nFalsification Criteria:");
    println!("  Total:     {}", falsification.total);
    println!("  Passed:    {}", falsification.passed);
    println!("  Triggered: {}", falsification.triggered);

    if falsification.jidoka_triggered {
        println!("  Jidoka:    TRIGGERED (stop-on-error)");
    }

    if verbose && !falsification.criteria.is_empty() {
        println!();
        for crit in &falsification.criteria {
            let sym = if crit.triggered { "✗" } else { "✓" };
            println!("  {} {}: {}", sym, crit.id, crit.name);
            if crit.triggered {
                println!("      Condition: {}", crit.condition);
            }
        }
    }
}

fn print_reproducibility(_reproducibility: Option<&crate::edd::ReproducibilitySummary>) {
    // Reserved for future reproducibility output
}

fn print_execution(execution: &crate::edd::ExecutionMetrics) {
    println!("\nExecution:");
    println!("  Duration:     {} ms", execution.duration_ms);
    println!("  Replications: {}", execution.replications);
}

fn print_warnings(warnings: &[String]) {
    if !warnings.is_empty() {
        println!("\nWarnings:");
        for warning in warnings {
            println!("  ! {warning}");
        }
    }
}

fn print_footer(status_symbol: &str, status: &str) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{status_symbol} Result: {status}");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

/// Print EMC compliance report.
///
/// # Arguments
///
/// * `report` - The EMC compliance report to display
pub fn print_emc_report(report: &EmcComplianceReport) {
    let status = if report.passed {
        "COMPLIANT"
    } else {
        "NON-COMPLIANT"
    };
    let sym = if report.passed { "✓" } else { "✗" };

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EMC Compliance Report: {}", report.experiment_name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // EDD Compliance Checklist
    println!("EDD Compliance Checklist:");
    let check = |passed: bool| if passed { "✓" } else { "✗" };

    println!(
        "  {} EDD-01: EMC Reference",
        check(report.edd_compliance.edd_01_emc_reference)
    );
    println!(
        "  {} EDD-02: Verification Tests",
        check(report.edd_compliance.edd_02_verification_tests)
    );
    println!(
        "  {} EDD-03: Seed Specified",
        check(report.edd_compliance.edd_03_seed_specified)
    );
    println!(
        "  {} EDD-04: Falsification Criteria",
        check(report.edd_compliance.edd_04_falsification_criteria)
    );
    println!(
        "  {} EDD-05: Hypothesis (Optional)",
        check(report.edd_compliance.edd_05_hypothesis)
    );

    // Schema errors
    if !report.schema_errors.is_empty() {
        println!("\nSchema Errors:");
        for err in &report.schema_errors {
            println!("  ✗ {err}");
        }
    }

    // EMC errors
    if !report.emc_errors.is_empty() {
        println!("\nEMC Errors:");
        for err in &report.emc_errors {
            println!("  ✗ {err}");
        }
    }

    // Warnings
    if !report.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &report.warnings {
            println!("  ! {warning}");
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{sym} Result: {status}");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

/// Print EMC validation results.
///
/// # Arguments
///
/// * `yaml` - The parsed YAML value
/// * `errors` - List of validation errors
/// * `warnings` - List of validation warnings
pub fn print_emc_validation_results(
    yaml: &serde_yaml::Value,
    errors: &[String],
    warnings: &[String],
) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if let Some(name) = yaml
        .get("identity")
        .and_then(|i| i.get("name"))
        .and_then(|n| n.as_str())
    {
        println!("EMC: {name}");
    }
    if let Some(id) = yaml.get("emc_id").and_then(|id| id.as_str()) {
        println!("ID: {id}");
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let check = |b: bool| if b { "✓" } else { "✗" };
    println!("Schema Validation:");
    println!("  {} emc_version", check(yaml.get("emc_version").is_some()));
    println!("  {} emc_id", check(yaml.get("emc_id").is_some()));
    println!("  {} identity", check(yaml.get("identity").is_some()));
    println!(
        "  {} governing_equation",
        check(yaml.get("governing_equation").is_some())
    );
    println!(
        "  {} analytical_derivation",
        check(yaml.get("analytical_derivation").is_some())
    );
    println!(
        "  {} domain_of_validity",
        check(yaml.get("domain_of_validity").is_some())
    );
    println!(
        "  {} verification_tests",
        check(yaml.get("verification_tests").is_some())
    );
    println!(
        "  {} falsification_criteria",
        check(yaml.get("falsification_criteria").is_some())
    );

    if !errors.is_empty() {
        println!("\nErrors:");
        for err in errors {
            println!("  ✗ {err}");
        }
    }
    if !warnings.is_empty() {
        println!("\nWarnings:");
        for w in warnings {
            println!("  ! {w}");
        }
    }

    let (status, sym) = if errors.is_empty() {
        ("VALID", "✓")
    } else {
        ("INVALID", "✗")
    };
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{sym} Result: {status}");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}
