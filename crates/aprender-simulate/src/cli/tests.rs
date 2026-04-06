//! CLI module tests.
//!
//! Comprehensive tests for all CLI functionality to achieve 95%+ coverage.

use super::args::{Args, Command};
use super::output::{
    print_emc_report, print_emc_validation_results, print_experiment_result, print_help,
    print_version,
};
use super::schema::validate_emc_schema;
use crate::edd::{
    EddComplianceChecklist, EmcComplianceReport, ExecutionMetrics, ExperimentResult,
    FalsificationCriterionResult, FalsificationSummary, ReproducibilitySummary,
    VerificationSummary, VerificationTestSummary,
};
use std::path::PathBuf;

// ============================================================================
// Args parsing tests
// ============================================================================

#[test]
fn test_parse_no_args_shows_help() {
    let args = Args::parse_from(["simular"]);
    assert_eq!(args.command, Command::Help);
}

#[test]
fn test_parse_help_flag() {
    let args = Args::parse_from(["simular", "-h"]);
    assert_eq!(args.command, Command::Help);
}

#[test]
fn test_parse_help_long_flag() {
    let args = Args::parse_from(["simular", "--help"]);
    assert_eq!(args.command, Command::Help);
}

#[test]
fn test_parse_help_command() {
    let args = Args::parse_from(["simular", "help"]);
    assert_eq!(args.command, Command::Help);
}

#[test]
fn test_parse_version_flag() {
    let args = Args::parse_from(["simular", "-V"]);
    assert_eq!(args.command, Command::Version);
}

#[test]
fn test_parse_version_long_flag() {
    let args = Args::parse_from(["simular", "--version"]);
    assert_eq!(args.command, Command::Version);
}

#[test]
fn test_parse_version_command() {
    let args = Args::parse_from(["simular", "version"]);
    assert_eq!(args.command, Command::Version);
}

#[test]
fn test_parse_unknown_command() {
    let args = Args::parse_from(["simular", "unknown-cmd"]);
    assert!(matches!(args.command, Command::Error(ref msg) if msg.contains("Unknown command")));
}

#[test]
fn test_parse_run_command() {
    let args = Args::parse_from(["simular", "run", "experiment.yaml"]);
    match args.command {
        Command::Run {
            experiment_path,
            seed_override,
            verbose,
        } => {
            assert_eq!(experiment_path, PathBuf::from("experiment.yaml"));
            assert_eq!(seed_override, None);
            assert!(!verbose);
        }
        _ => panic!("Expected Run command"),
    }
}

#[test]
fn test_parse_run_command_with_seed() {
    let args = Args::parse_from(["simular", "run", "experiment.yaml", "--seed", "12345"]);
    match args.command {
        Command::Run {
            experiment_path,
            seed_override,
            verbose,
        } => {
            assert_eq!(experiment_path, PathBuf::from("experiment.yaml"));
            assert_eq!(seed_override, Some(12345));
            assert!(!verbose);
        }
        _ => panic!("Expected Run command"),
    }
}

#[test]
fn test_parse_run_command_with_verbose() {
    for flag in &["-v", "--verbose"] {
        let args = Args::parse_from(["simular", "run", "experiment.yaml", flag]);
        match args.command {
            Command::Run { verbose, .. } => assert!(verbose, "flag {flag}"),
            _ => panic!("Expected Run command for {flag}"),
        }
    }
}

#[test]
fn test_parse_run_command_with_all_options() {
    let args = Args::parse_from([
        "simular",
        "run",
        "experiment.yaml",
        "--seed",
        "999",
        "--verbose",
    ]);
    match args.command {
        Command::Run {
            experiment_path,
            seed_override,
            verbose,
        } => {
            assert_eq!(experiment_path, PathBuf::from("experiment.yaml"));
            assert_eq!(seed_override, Some(999));
            assert!(verbose);
        }
        _ => panic!("Expected Run command"),
    }
}

#[test]
fn test_parse_run_command_missing_path() {
    let args = Args::parse_from(["simular", "run"]);
    assert_eq!(args.command, Command::Help);
}

#[test]
fn test_parse_subcommand_help_flags() {
    let subs = [
        "run",
        "verify",
        "validate",
        "emc-check",
        "emc-validate",
        "render",
    ];
    for sub in subs {
        for flag in ["--help", "-h"] {
            let a = Args::parse_from(["simular", sub, flag]);
            assert_eq!(a.command, Command::Help, "{sub} {flag}");
        }
    }
}

#[test]
fn test_parse_run_command_seed_edge_cases() {
    // Missing value and invalid value both result in None seed
    for argv in [
        vec!["simular", "run", "experiment.yaml", "--seed"],
        vec![
            "simular",
            "run",
            "experiment.yaml",
            "--seed",
            "not-a-number",
        ],
    ] {
        match Args::parse_from(argv).command {
            Command::Run { seed_override, .. } => assert_eq!(seed_override, None),
            _ => panic!("Expected Run command"),
        }
    }
}

#[test]
fn test_parse_run_command_unknown_flag() {
    let args = Args::parse_from(["simular", "run", "experiment.yaml", "--unknown"]);
    // Unknown flags are ignored
    match args.command {
        Command::Run {
            experiment_path, ..
        } => {
            assert_eq!(experiment_path, PathBuf::from("experiment.yaml"));
        }
        _ => panic!("Expected Run command"),
    }
}

#[test]
fn test_parse_verify_command() {
    let args = Args::parse_from(["simular", "verify", "experiment.yaml"]);
    match args.command {
        Command::Verify {
            experiment_path,
            runs,
        } => {
            assert_eq!(experiment_path, PathBuf::from("experiment.yaml"));
            assert_eq!(runs, 3); // default
        }
        _ => panic!("Expected Verify command"),
    }
}

#[test]
fn test_parse_verify_command_with_runs() {
    let args = Args::parse_from(["simular", "verify", "experiment.yaml", "--runs", "10"]);
    match args.command {
        Command::Verify { runs, .. } => {
            assert_eq!(runs, 10);
        }
        _ => panic!("Expected Verify command"),
    }
}

#[test]
fn test_parse_verify_command_missing_path() {
    let args = Args::parse_from(["simular", "verify"]);
    assert_eq!(args.command, Command::Help);
}

#[test]
fn test_parse_verify_command_runs_without_value() {
    let args = Args::parse_from(["simular", "verify", "experiment.yaml", "--runs"]);
    match args.command {
        Command::Verify { runs, .. } => {
            assert_eq!(runs, 3); // default when value missing
        }
        _ => panic!("Expected Verify command"),
    }
}

#[test]
fn test_parse_emc_check_command() {
    let args = Args::parse_from(["simular", "emc-check", "experiment.yaml"]);
    match args.command {
        Command::EmcCheck { experiment_path } => {
            assert_eq!(experiment_path, PathBuf::from("experiment.yaml"));
        }
        _ => panic!("Expected EmcCheck command"),
    }
}

#[test]
fn test_parse_emc_check_missing_path() {
    let args = Args::parse_from(["simular", "emc-check"]);
    assert_eq!(args.command, Command::Help);
}

#[test]
fn test_parse_emc_validate_command() {
    let args = Args::parse_from(["simular", "emc-validate", "littles_law.emc.yaml"]);
    match args.command {
        Command::EmcValidate { emc_path } => {
            assert_eq!(emc_path, PathBuf::from("littles_law.emc.yaml"));
        }
        _ => panic!("Expected EmcValidate command"),
    }
}

#[test]
fn test_parse_emc_validate_missing_path() {
    let args = Args::parse_from(["simular", "emc-validate"]);
    assert_eq!(args.command, Command::Help);
}

#[test]
fn test_parse_list_emc_command() {
    let args = Args::parse_from(["simular", "list-emc"]);
    assert_eq!(args.command, Command::ListEmc);
}

#[test]
fn test_args_clone() {
    let args = Args::parse_from(["simular", "list-emc"]);
    let cloned = args.clone();
    assert_eq!(args.command, cloned.command);
}

#[test]
fn test_command_debug() {
    let cmd = Command::Help;
    let debug_str = format!("{cmd:?}");
    assert!(debug_str.contains("Help"));
}

// ============================================================================
// Output formatting tests
// ============================================================================

#[test]
fn test_print_version() {
    // Just verify it doesn't panic
    print_version();
}

#[test]
fn test_print_help() {
    // Just verify it doesn't panic
    print_help();
}

#[test]
fn test_print_experiment_result_passed() {
    let result = ExperimentResult {
        experiment_id: "test-001".to_string(),
        name: "Test Experiment".to_string(),
        seed: 42,
        passed: true,
        verification: VerificationSummary {
            total: 3,
            passed: 3,
            failed: 0,
            tests: vec![],
        },
        falsification: FalsificationSummary {
            total: 2,
            passed: 2,
            triggered: 0,
            jidoka_triggered: false,
            criteria: vec![],
        },
        reproducibility: None,
        execution: ExecutionMetrics {
            duration_ms: 100,
            steps: 1000,
            replications: 1000,
            peak_memory_bytes: None,
        },
        artifacts: vec![],
        warnings: vec![],
    };

    // Verify it doesn't panic
    print_experiment_result(&result, false);
    print_experiment_result(&result, true);
}

#[test]
fn test_print_experiment_result_failed() {
    let result = ExperimentResult {
        experiment_id: "test-002".to_string(),
        name: "Failed Experiment".to_string(),
        seed: 42,
        passed: false,
        verification: VerificationSummary {
            total: 3,
            passed: 2,
            failed: 1,
            tests: vec![VerificationTestSummary {
                id: "test1".to_string(),
                name: "Failing Test".to_string(),
                passed: false,
                expected: None,
                actual: None,
                tolerance: None,
                error: Some("Assertion failed".to_string()),
            }],
        },
        falsification: FalsificationSummary {
            total: 2,
            passed: 1,
            triggered: 1,
            jidoka_triggered: true,
            criteria: vec![FalsificationCriterionResult {
                id: "crit1".to_string(),
                name: "Energy Conservation".to_string(),
                triggered: true,
                condition: "energy > threshold".to_string(),
                severity: "critical".to_string(),
                value: None,
                threshold: None,
            }],
        },
        reproducibility: None,
        execution: ExecutionMetrics {
            duration_ms: 50,
            steps: 500,
            replications: 500,
            peak_memory_bytes: None,
        },
        artifacts: vec![],
        warnings: vec!["Warning: Something happened".to_string()],
    };

    // Verify it doesn't panic
    print_experiment_result(&result, false);
    print_experiment_result(&result, true);
}

#[test]
fn test_print_experiment_result_with_reproducibility() {
    let result = ExperimentResult {
        experiment_id: "test-003".to_string(),
        name: "Reproducibility Test".to_string(),
        seed: 42,
        passed: true,
        verification: VerificationSummary {
            total: 1,
            passed: 1,
            failed: 0,
            tests: vec![],
        },
        falsification: FalsificationSummary {
            total: 1,
            passed: 1,
            triggered: 0,
            jidoka_triggered: false,
            criteria: vec![],
        },
        reproducibility: Some(ReproducibilitySummary {
            runs: 3,
            identical: true,
            passed: true,
            reference_hash: "abc123".to_string(),
            run_hashes: vec![],
            platform: "linux-x86_64".to_string(),
        }),
        execution: ExecutionMetrics {
            duration_ms: 200,
            steps: 100,
            replications: 100,
            peak_memory_bytes: None,
        },
        artifacts: vec![],
        warnings: vec![],
    };

    print_experiment_result(&result, true);
}

#[test]
fn test_print_emc_report_compliant() {
    let report = EmcComplianceReport {
        experiment_name: "Test Experiment".to_string(),
        passed: true,
        edd_compliance: EddComplianceChecklist {
            edd_01_emc_reference: true,
            edd_02_verification_tests: true,
            edd_03_seed_specified: true,
            edd_04_falsification_criteria: true,
            edd_05_hypothesis: true,
        },
        schema_errors: vec![],
        emc_errors: vec![],
        warnings: vec![],
    };

    print_emc_report(&report);
}

#[test]
fn test_print_emc_report_non_compliant() {
    let report = EmcComplianceReport {
        experiment_name: "Non-Compliant Experiment".to_string(),
        passed: false,
        edd_compliance: EddComplianceChecklist {
            edd_01_emc_reference: false,
            edd_02_verification_tests: false,
            edd_03_seed_specified: true,
            edd_04_falsification_criteria: false,
            edd_05_hypothesis: false,
        },
        schema_errors: vec!["Missing required field: emc_ref".to_string()],
        emc_errors: vec!["EMC not found in registry".to_string()],
        warnings: vec!["Consider adding hypothesis".to_string()],
    };

    print_emc_report(&report);
}

#[test]
fn test_print_emc_validation_results_valid() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        identity:
          name: "Test EMC"
        emc_id: "test/example"
        emc_version: "1.0"
        governing_equation: {}
        analytical_derivation: {}
        domain_of_validity: {}
        verification_tests: []
        falsification_criteria: []
    "#,
    )
    .ok()
    .unwrap_or(serde_yaml::Value::Null);

    print_emc_validation_results(&yaml, &[], &[]);
}

#[test]
fn test_print_emc_validation_results_with_errors() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        identity:
          name: "Incomplete EMC"
    "#,
    )
    .ok()
    .unwrap_or(serde_yaml::Value::Null);

    let errors = vec![
        "Missing required field: emc_version".to_string(),
        "Missing required field: emc_id".to_string(),
    ];
    let warnings = vec!["Missing verification_tests".to_string()];

    print_emc_validation_results(&yaml, &errors, &warnings);
}

#[test]
fn test_print_emc_validation_results_null_yaml() {
    let yaml = serde_yaml::Value::Null;
    print_emc_validation_results(&yaml, &[], &[]);
}

// ============================================================================
// Schema validation tests are in schema.rs
// Re-export key tests here for integration
// ============================================================================

#[test]
fn test_schema_validation_integration() {
    let valid_yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
        emc_version: "1.0"
        emc_id: "test/integration"
        identity:
          name: "Integration Test"
          version: "1.0.0"
        governing_equation:
          latex: "E = mc^2"
        analytical_derivation:
          primary_citation: "Einstein 1905"
        domain_of_validity:
          description: "All"
        verification_tests:
          - id: test1
        falsification_criteria:
          - id: crit1
    "#,
    )
    .ok()
    .unwrap_or(serde_yaml::Value::Null);

    let (errors, warnings) = validate_emc_schema(&valid_yaml);
    assert!(errors.is_empty());
    assert!(warnings.is_empty());
}

// ============================================================================
// Args equality tests for coverage
// ============================================================================

#[test]
fn test_args_equality() {
    let args1 = Args::parse_from(["simular", "list-emc"]);
    let args2 = Args::parse_from(["simular", "list-emc"]);
    assert_eq!(args1, args2);
}

#[test]
fn test_command_equality() {
    assert_eq!(Command::Help, Command::Help);
    assert_eq!(Command::Version, Command::Version);
    assert_eq!(Command::ListEmc, Command::ListEmc);

    let run1 = Command::Run {
        experiment_path: PathBuf::from("test.yaml"),
        seed_override: Some(42),
        verbose: true,
    };
    let run2 = Command::Run {
        experiment_path: PathBuf::from("test.yaml"),
        seed_override: Some(42),
        verbose: true,
    };
    assert_eq!(run1, run2);

    let verify1 = Command::Verify {
        experiment_path: PathBuf::from("test.yaml"),
        runs: 5,
    };
    let verify2 = Command::Verify {
        experiment_path: PathBuf::from("test.yaml"),
        runs: 5,
    };
    assert_eq!(verify1, verify2);

    let emc_check1 = Command::EmcCheck {
        experiment_path: PathBuf::from("test.yaml"),
    };
    let emc_check2 = Command::EmcCheck {
        experiment_path: PathBuf::from("test.yaml"),
    };
    assert_eq!(emc_check1, emc_check2);

    let emc_validate1 = Command::EmcValidate {
        emc_path: PathBuf::from("test.emc.yaml"),
    };
    let emc_validate2 = Command::EmcValidate {
        emc_path: PathBuf::from("test.emc.yaml"),
    };
    assert_eq!(emc_validate1, emc_validate2);
}

#[test]
fn test_command_inequality() {
    assert_ne!(Command::Help, Command::Version);

    let run = Command::Run {
        experiment_path: PathBuf::from("test.yaml"),
        seed_override: None,
        verbose: false,
    };
    assert_ne!(run, Command::Help);
}

// ============================================================================
// Command handler tests
// ============================================================================

use super::commands::{
    emc_check, emc_validate, list_emc, run_cli, run_experiment, validate_experiment,
    verify_reproducibility,
};
use std::process::ExitCode;

#[test]
fn test_run_cli_help() {
    let args = Args::parse_from(["simular", "help"]);
    let exit = run_cli(args);
    assert_eq!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_run_cli_version() {
    let args = Args::parse_from(["simular", "version"]);
    let exit = run_cli(args);
    assert_eq!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_run_cli_list_emc() {
    let args = Args::parse_from(["simular", "list-emc"]);
    let exit = run_cli(args);
    assert_eq!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_run_experiment_file_not_found() {
    let exit = run_experiment(std::path::Path::new("nonexistent.yaml"), None, false);
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_run_experiment_file_not_found_verbose() {
    let exit = run_experiment(std::path::Path::new("nonexistent.yaml"), None, true);
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_run_experiment_with_seed_override() {
    let exit = run_experiment(std::path::Path::new("nonexistent.yaml"), Some(12345), false);
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_verify_reproducibility_file_not_found() {
    let exit = verify_reproducibility(std::path::Path::new("nonexistent.yaml"), 3);
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_verify_reproducibility_custom_runs() {
    let exit = verify_reproducibility(std::path::Path::new("nonexistent.yaml"), 5);
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_emc_check_file_not_found() {
    let exit = emc_check(std::path::Path::new("nonexistent.yaml"));
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_emc_validate_file_not_found() {
    let exit = emc_validate(std::path::Path::new("nonexistent.emc.yaml"));
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_list_emc_returns_success() {
    let exit = list_emc();
    assert_eq!(exit, ExitCode::SUCCESS);
}

// ============================================================================
// validate_experiment tests
// ============================================================================

#[test]
fn test_validate_experiment_file_not_found() {
    let exit = validate_experiment(std::path::Path::new("nonexistent.yaml"));
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_validate_experiment_valid_file() {
    // Note: Existing experiment files use legacy format.
    // This test uses a new-format YAML to verify schema validation works.
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("new_format_experiment.yaml");
    std::fs::write(
        &temp_file,
        r#"
id: "TSP-001"
seed: 42
simulation:
  type: "tsp"
  parameters:
    n_cities: 25
falsification:
  criteria:
    - id: "gap"
      threshold: 0.25
      condition: "gap < threshold"
"#,
    )
    .ok();
    let exit = validate_experiment(&temp_file);
    assert_eq!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
#[cfg(feature = "schema-validation")]
fn test_validate_experiment_invalid_yaml() {
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("invalid_experiment.yaml");
    std::fs::write(
        &temp_file,
        r#"
id: "TEST-001"
# Missing required fields: seed, emc_ref, simulation, falsification
"#,
    )
    .ok();
    let exit = validate_experiment(&temp_file);
    assert_ne!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_validate_experiment_with_valid_structure() {
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("valid_experiment.yaml");
    std::fs::write(
        &temp_file,
        r#"
id: "TEST-001"
seed: 42
emc_ref: "test/emc"
simulation:
  type: "test"
falsification:
  criteria:
    - id: "gap"
      threshold: 0.25
      condition: "gap < threshold"
"#,
    )
    .ok();
    let exit = validate_experiment(&temp_file);
    assert_eq!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_validate_experiment_monte_carlo() {
    // Test Monte Carlo style experiment validation with new format
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("monte_carlo_experiment.yaml");
    std::fs::write(
        &temp_file,
        r#"
id: "MC-001"
seed: 42
simulation:
  type: "monte_carlo"
  parameters:
    samples: 10000
falsification:
  criteria:
    - id: "convergence"
      threshold: 0.01
      condition: "error < threshold"
"#,
    )
    .ok();
    let exit = validate_experiment(&temp_file);
    assert_eq!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_validate_experiment_tsp_grasp() {
    // Test TSP GRASP style experiment validation with new format
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("tsp_grasp_experiment.yaml");
    std::fs::write(
        &temp_file,
        r#"
id: "TSP-GRASP-001"
seed: 42
simulation:
  type: "tsp_grasp"
  parameters:
    n_cities: 25
    rcl_size: 5
falsification:
  criteria:
    - id: "optimality_gap"
      threshold: 0.25
      condition: "gap < threshold"
"#,
    )
    .ok();
    let exit = validate_experiment(&temp_file);
    assert_eq!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_run_cli_validate_command() {
    // Test run_cli with validate command using new format
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("cli_validate_experiment.yaml");
    std::fs::write(
        &temp_file,
        r#"
id: "CLI-001"
seed: 42
simulation:
  type: "test"
falsification:
  criteria:
    - id: "test"
      threshold: 0.1
      condition: "value < threshold"
"#,
    )
    .ok();

    let args = Args {
        command: Command::Validate {
            experiment_path: temp_file.clone(),
        },
    };
    let exit = run_cli(args);
    assert_eq!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

// Test with actual valid experiment file
#[test]
fn test_run_experiment_valid_file() {
    // Use an existing experiment file
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let experiments_dir = std::path::Path::new(&manifest_dir).join("experiments");

    // Try to find a valid experiment file
    if let Ok(entries) = std::fs::read_dir(&experiments_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml") {
                // Just test that it runs (may pass or fail based on EMC availability)
                let _ = run_experiment(&path, None, false);
                return;
            }
        }
    }
    // If no experiments found, create a minimal one in tempdir
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("test_experiment.yaml");
    std::fs::write(
        &temp_file,
        r#"
experiment:
  name: Test
  seed: 42
  steps: 10
equation_model_card:
  emc_ref: "physics/harmonic_oscillator"
verification_tests: []
falsification_criteria: []
"#,
    )
    .ok();
    let _ = run_experiment(&temp_file, None, false);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_emc_validate_valid_file() {
    // Create a valid EMC file for testing
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("test_valid_emc_file.yaml");
    std::fs::write(
        &temp_file,
        r#"
emc_version: "1.0"
emc_id: "test/valid"
identity:
  name: "Valid EMC"
  version: "1.0.0"
governing_equation:
  latex: "E = mc^2"
analytical_derivation:
  primary_citation: "Einstein 1905"
domain_of_validity:
  description: "All"
verification_tests:
  - id: test1
falsification_criteria:
  - id: crit1
"#,
    )
    .ok();
    let exit = emc_validate(&temp_file);
    assert_eq!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_emc_validate_invalid_yaml() {
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("invalid_emc.yaml");
    std::fs::write(&temp_file, "not: valid: yaml: here").ok();
    let exit = emc_validate(&temp_file);
    assert_ne!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_emc_validate_missing_fields() {
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("incomplete_emc.yaml");
    std::fs::write(
        &temp_file,
        r#"
identity:
  name: "Incomplete"
"#,
    )
    .ok();
    let exit = emc_validate(&temp_file);
    // Should fail due to missing required fields
    assert_ne!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_emc_validate_valid_minimal() {
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("valid_minimal_emc.yaml");
    std::fs::write(
        &temp_file,
        r#"
emc_version: "1.0"
emc_id: "test/minimal"
identity:
  name: "Minimal Valid EMC"
  version: "1.0.0"
governing_equation:
  latex: "x = x"
analytical_derivation:
  primary_citation: "Test"
domain_of_validity:
  description: "Test"
verification_tests:
  - id: test1
falsification_criteria:
  - id: crit1
"#,
    )
    .ok();
    let exit = emc_validate(&temp_file);
    assert_eq!(exit, ExitCode::SUCCESS);
    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_run_cli_run_command() {
    let args = Args {
        command: Command::Run {
            experiment_path: PathBuf::from("nonexistent.yaml"),
            seed_override: None,
            verbose: false,
        },
    };
    let exit = run_cli(args);
    // File doesn't exist, should fail
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_run_cli_verify_command() {
    let args = Args {
        command: Command::Verify {
            experiment_path: PathBuf::from("nonexistent.yaml"),
            runs: 3,
        },
    };
    let exit = run_cli(args);
    // File doesn't exist, should fail
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_run_cli_emc_check_command() {
    let args = Args {
        command: Command::EmcCheck {
            experiment_path: PathBuf::from("nonexistent.yaml"),
        },
    };
    let exit = run_cli(args);
    // File doesn't exist, should fail
    assert_ne!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_run_cli_emc_validate_command() {
    let args = Args {
        command: Command::EmcValidate {
            emc_path: PathBuf::from("nonexistent.emc.yaml"),
        },
    };
    let exit = run_cli(args);
    // File doesn't exist, should fail
    assert_ne!(exit, ExitCode::SUCCESS);
}

// ============================================================================
// Success path tests using actual EMC files
// ============================================================================

#[test]
fn test_emc_validate_harmonic_oscillator() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let emc_path =
        std::path::Path::new(&manifest_dir).join("docs/emc/physics/harmonic_oscillator.emc.yaml");

    if emc_path.exists() {
        let exit = emc_validate(&emc_path);
        assert_eq!(exit, ExitCode::SUCCESS);
    }
}

#[test]
fn test_emc_validate_kepler() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let emc_path =
        std::path::Path::new(&manifest_dir).join("docs/emc/physics/kepler_two_body.emc.yaml");

    if emc_path.exists() {
        let exit = emc_validate(&emc_path);
        assert_eq!(exit, ExitCode::SUCCESS);
    }
}

#[test]
fn test_emc_validate_littles_law() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let emc_path =
        std::path::Path::new(&manifest_dir).join("docs/emc/operations/littles_law.emc.yaml");

    if emc_path.exists() {
        let exit = emc_validate(&emc_path);
        assert_eq!(exit, ExitCode::SUCCESS);
    }
}

#[test]
fn test_emc_validate_monte_carlo() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let emc_path = std::path::Path::new(&manifest_dir)
        .join("docs/emc/statistical/monte_carlo_integration.emc.yaml");

    if emc_path.exists() {
        let exit = emc_validate(&emc_path);
        assert_eq!(exit, ExitCode::SUCCESS);
    }
}

#[test]
fn test_run_experiment_harmonic_oscillator() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let exp_path =
        std::path::Path::new(&manifest_dir).join("examples/experiments/harmonic_oscillator.yaml");

    if exp_path.exists() {
        // This test checks that the experiment runs without panicking
        // The result may pass or fail based on configuration
        let _ = run_experiment(&exp_path, None, false);
    }
}

#[test]
fn test_run_experiment_harmonic_oscillator_verbose() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let exp_path =
        std::path::Path::new(&manifest_dir).join("examples/experiments/harmonic_oscillator.yaml");

    if exp_path.exists() {
        let _ = run_experiment(&exp_path, None, true);
    }
}

#[test]
fn test_run_experiment_with_seed() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let exp_path =
        std::path::Path::new(&manifest_dir).join("examples/experiments/monte_carlo_pi.yaml");

    if exp_path.exists() {
        let _ = run_experiment(&exp_path, Some(12345), false);
    }
}

#[test]
fn test_verify_reproducibility_harmonic() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let exp_path =
        std::path::Path::new(&manifest_dir).join("examples/experiments/harmonic_oscillator.yaml");

    if exp_path.exists() {
        // Just run without panicking
        let _ = verify_reproducibility(&exp_path, 2);
    }
}

#[test]
fn test_emc_check_harmonic() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let exp_path =
        std::path::Path::new(&manifest_dir).join("examples/experiments/harmonic_oscillator.yaml");

    if exp_path.exists() {
        let _ = emc_check(&exp_path);
    }
}

#[test]
fn test_list_emc_shows_entries() {
    // list_emc always succeeds - just ensure it runs
    let exit = list_emc();
    assert_eq!(exit, ExitCode::SUCCESS);
}

#[test]
fn test_run_cli_with_run_verbose() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let exp_path =
        std::path::Path::new(&manifest_dir).join("examples/experiments/harmonic_oscillator.yaml");

    if exp_path.exists() {
        let args = Args {
            command: Command::Run {
                experiment_path: exp_path,
                seed_override: None,
                verbose: true,
            },
        };
        let _ = run_cli(args);
    }
}

#[test]
fn test_run_cli_with_verify() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let exp_path =
        std::path::Path::new(&manifest_dir).join("examples/experiments/harmonic_oscillator.yaml");

    if exp_path.exists() {
        let args = Args {
            command: Command::Verify {
                experiment_path: exp_path,
                runs: 2,
            },
        };
        let _ = run_cli(args);
    }
}

#[test]
fn test_run_cli_with_emc_check() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let exp_path =
        std::path::Path::new(&manifest_dir).join("examples/experiments/harmonic_oscillator.yaml");

    if exp_path.exists() {
        let args = Args {
            command: Command::EmcCheck {
                experiment_path: exp_path,
            },
        };
        let _ = run_cli(args);
    }
}

#[test]
fn test_run_cli_with_emc_validate_real() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let emc_path =
        std::path::Path::new(&manifest_dir).join("docs/emc/physics/harmonic_oscillator.emc.yaml");

    if emc_path.exists() {
        let args = Args {
            command: Command::EmcValidate { emc_path },
        };
        let exit = run_cli(args);
        assert_eq!(exit, ExitCode::SUCCESS);
    }
}
