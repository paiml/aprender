//! Sprint 50: Validate Subcommand Tests (EXTREME TDD)
//!
//! Implements `renacer validate` for golden trace comparison.
//!
//! # Toyota Way Principle

#![allow(clippy::unwrap_used)]
//!
//! *Jidoka* (自動化) - Stop on regression detection (Andon principle)
//!
//! # References
//!
//! docs/specifications/apr-runtime-model-tracing-support.md

use renacer::validate::{
    compare_syscalls, compare_timing,
    config::{ValidateConfig, ValidationOutputFormat},
    error::ValidateError,
    golden_trace::{
        PlatformInfo, SyscallTimingStats, TimingStats, TraceFlags, TraceHeader, TraceManifest,
        TraceSyscallEntry,
    },
    output::{format_json_report, format_junit_report, format_text_report},
    ComparisonResult, ValidateExitCode,
};
use std::collections::HashMap;
use std::path::PathBuf;

// ============================================================================
// Step 1: Data Structure Tests (RED PHASE)
// ============================================================================

/// Test that exit codes match specification Section 3.4
#[test]
fn test_validate_exit_code_values() {
    assert_eq!(ValidateExitCode::Passed as i32, 0);
    assert_eq!(ValidateExitCode::Failed as i32, 1);
    assert_eq!(ValidateExitCode::BaselineNotFound as i32, 2);
    assert_eq!(ValidateExitCode::InvalidBaseline as i32, 3);
    assert_eq!(ValidateExitCode::CommandError as i32, 4);
    assert_eq!(ValidateExitCode::ConfigError as i32, 5);
}

/// Test default configuration values per specification
#[test]
fn test_validate_config_defaults() {
    let config = ValidateConfig::default();

    assert!(config.baseline_dir.is_none());
    assert!(config.generate_dir.is_none());
    assert!((config.tolerance_percent - 10.0).abs() < f32::EPSILON);
    assert!(!config.strict_mode);
    assert!(!config.ignore_timing);
    assert!(!config.fail_fast);
    assert!(matches!(config.output_format, ValidationOutputFormat::Text));
}

/// Test binary trace header magic bytes per spec Section 4.3
#[test]
fn test_trace_header_magic_bytes() {
    let header = TraceHeader::new(100, TraceFlags::default());

    assert_eq!(&header.magic, b"RNTR");
    assert_eq!(header.version, 1);
    assert_eq!(header.entry_count, 100);
}

/// Test trace manifest JSON serialization roundtrip
#[test]
fn test_trace_manifest_serialization() {
    let manifest = TraceManifest::new(vec!["echo".to_string(), "hello".to_string()]);

    let json = serde_json::to_string(&manifest).expect("Failed to serialize");
    let parsed: TraceManifest = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(parsed.command, vec!["echo", "hello"]);
    assert_eq!(parsed.version, "1.0.0");
    assert!(!parsed.renacer_version.is_empty());
}

/// Test syscall entry binary encoding
#[test]
fn test_syscall_entry_encoding() {
    let entry = TraceSyscallEntry {
        timestamp_ns: 1_000_000_000,
        syscall_nr: 1, // write
        duration_ns: 50_000,
        return_value: 5,
        args: vec![1, 0x7fff_0000, 5], // fd, buf, count
        string_args: vec![],
    };

    let encoded = entry.encode();
    let decoded = TraceSyscallEntry::decode(&encoded).expect("Failed to decode");

    assert_eq!(decoded.timestamp_ns, 1_000_000_000);
    assert_eq!(decoded.syscall_nr, 1);
    assert_eq!(decoded.duration_ns, 50_000);
    assert_eq!(decoded.return_value, 5);
    assert_eq!(decoded.args.len(), 3);
}

/// Test timing stats JSON schema
#[test]
fn test_timing_stats_json_schema() {
    let stats = TimingStats {
        total_duration_ns: 125_300_000,
        syscall_count: 1234,
        by_syscall: std::collections::HashMap::new(),
    };

    let json = serde_json::to_string(&stats).expect("Failed to serialize");
    assert!(json.contains("total_duration_ns"));
    assert!(json.contains("syscall_count"));
    assert!(json.contains("by_syscall"));
}

/// Test tolerance range validation
#[test]
fn test_tolerance_range_validation() {
    // Valid tolerance
    let config = ValidateConfig::with_tolerance(20.0);
    assert!((config.tolerance_percent - 20.0).abs() < f32::EPSILON);

    // Strict mode overrides tolerance to 0
    let strict_config = ValidateConfig::default().with_strict_mode(true);
    assert!((strict_config.tolerance_percent - 0.0).abs() < f32::EPSILON);
}

/// Test output format enum values
#[test]
fn test_output_format_enum() {
    let text = ValidationOutputFormat::Text;
    let json = ValidationOutputFormat::Json;
    let junit = ValidationOutputFormat::JUnit;

    // They should be distinct
    assert!(!matches!(text, ValidationOutputFormat::Json));
    assert!(!matches!(json, ValidationOutputFormat::JUnit));
    assert!(!matches!(junit, ValidationOutputFormat::Text));
}

/// Test platform info serialization
#[test]
fn test_platform_info_serialization() {
    let platform = PlatformInfo::current();

    assert_eq!(platform.os, "linux");
    assert!(!platform.arch.is_empty());
    assert!(!platform.kernel.is_empty());

    let json = serde_json::to_string(&platform).expect("Failed to serialize");
    let parsed: PlatformInfo = serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(parsed.os, platform.os);
    assert_eq!(parsed.arch, platform.arch);
}

/// Test trace flags bitfield
#[test]
fn test_trace_flags_bitfield() {
    let flags = TraceFlags::default();
    assert!(!flags.compressed());
    assert!(!flags.has_timing());
    assert!(!flags.has_string_args());

    let compressed = TraceFlags::new(true, true, true);
    assert!(compressed.compressed());
    assert!(compressed.has_timing());
    assert!(compressed.has_string_args());
}

// ============================================================================
// Step 3: Comparison Logic Tests (RED PHASE)
// ============================================================================

/// Test identical syscall sequences match
#[test]
fn test_syscall_sequence_match_identical() {
    let baseline =
        vec![TraceSyscallEntry::simple(0, "read", 100), TraceSyscallEntry::simple(1, "write", 50)];
    let current = baseline.clone();

    let result = compare_syscalls(&baseline, &current, false).expect("Comparison failed");

    assert!(result.passed);
    assert!(result.syscall_mismatches.is_empty());
}

/// Test syscall count mismatch detection
#[test]
fn test_syscall_sequence_mismatch_count() {
    let baseline =
        vec![TraceSyscallEntry::simple(0, "read", 100), TraceSyscallEntry::simple(1, "write", 50)];
    let current = vec![TraceSyscallEntry::simple(0, "read", 100)];

    let result = compare_syscalls(&baseline, &current, false).expect("Comparison failed");

    assert!(!result.passed);
    assert!(!result.syscall_mismatches.is_empty());
}

/// Test syscall sequence order mismatch
#[test]
fn test_syscall_sequence_mismatch_order() {
    let baseline =
        vec![TraceSyscallEntry::simple(0, "read", 100), TraceSyscallEntry::simple(1, "write", 50)];
    let current =
        vec![TraceSyscallEntry::simple(1, "write", 50), TraceSyscallEntry::simple(0, "read", 100)];

    let result = compare_syscalls(&baseline, &current, false).expect("Comparison failed");

    assert!(!result.passed);
    assert!(!result.syscall_mismatches.is_empty());
}

/// Test timing within tolerance passes
#[test]
fn test_timing_within_tolerance_passes() {
    let mut baseline = HashMap::new();
    baseline.insert(
        "read".to_string(),
        SyscallTimingStats {
            count: 10,
            total_ns: 1_000_000,
            mean_ns: 100_000,
            std_ns: 10_000,
            min_ns: 90_000,
            max_ns: 110_000,
            p50_ns: 100_000,
            p95_ns: 108_000,
            p99_ns: 109_000,
        },
    );

    let mut current = HashMap::new();
    current.insert(
        "read".to_string(),
        SyscallTimingStats {
            count: 10,
            total_ns: 1_090_000, // 9% increase
            mean_ns: 109_000,
            std_ns: 10_000,
            min_ns: 99_000,
            max_ns: 119_000,
            p50_ns: 109_000,
            p95_ns: 117_000,
            p99_ns: 118_000,
        },
    );

    let result = compare_timing(&baseline, &current, 10.0); // 10% tolerance

    assert!(result.is_empty()); // No regressions
}

/// Test timing exceeds tolerance fails
#[test]
fn test_timing_exceeds_tolerance_fails() {
    let mut baseline = HashMap::new();
    baseline.insert(
        "read".to_string(),
        SyscallTimingStats {
            count: 10,
            total_ns: 1_000_000,
            mean_ns: 100_000,
            std_ns: 10_000,
            min_ns: 90_000,
            max_ns: 110_000,
            p50_ns: 100_000,
            p95_ns: 108_000,
            p99_ns: 109_000,
        },
    );

    let mut current = HashMap::new();
    current.insert(
        "read".to_string(),
        SyscallTimingStats {
            count: 10,
            total_ns: 1_200_000, // 20% increase
            mean_ns: 120_000,
            std_ns: 10_000,
            min_ns: 110_000,
            max_ns: 130_000,
            p50_ns: 120_000,
            p95_ns: 128_000,
            p99_ns: 129_000,
        },
    );

    let result = compare_timing(&baseline, &current, 10.0); // 10% tolerance

    assert!(!result.is_empty()); // Has regressions
}

/// Test strict mode has zero tolerance
#[test]
fn test_strict_mode_zero_tolerance() {
    let mut baseline = HashMap::new();
    baseline.insert(
        "read".to_string(),
        SyscallTimingStats {
            count: 10,
            total_ns: 1_000_000,
            mean_ns: 100_000,
            std_ns: 0,
            min_ns: 100_000,
            max_ns: 100_000,
            p50_ns: 100_000,
            p95_ns: 100_000,
            p99_ns: 100_000,
        },
    );

    let mut current = HashMap::new();
    current.insert(
        "read".to_string(),
        SyscallTimingStats {
            count: 10,
            total_ns: 1_000_001, // +1ns
            mean_ns: 100_001,
            std_ns: 0,
            min_ns: 100_001,
            max_ns: 100_001,
            p50_ns: 100_001,
            p95_ns: 100_001,
            p99_ns: 100_001,
        },
    );

    // With strict mode (0% tolerance), even tiny difference fails
    let result = compare_timing(&baseline, &current, 0.0);

    assert!(!result.is_empty()); // Has regressions
}

/// Test ignore timing mode
#[test]
fn test_ignore_timing_mode() {
    let config = ValidateConfig::default().with_ignore_timing(true);
    assert!(config.ignore_timing);
}

// ============================================================================
// Step 5: I/O Tests (RED PHASE)
// ============================================================================

/// Test generate creates manifest.json
/// Note: This test requires ptrace permissions and may hang in sandboxed environments
#[test]
#[ignore] // Requires ptrace permissions - run with: cargo test -- --ignored
fn test_generate_creates_manifest() {
    use renacer::validate::golden_trace::generate_baseline;
    use tempfile::tempdir;

    let temp = tempdir().expect("Failed to create temp dir");
    let baseline_path = temp.path().join("golden");

    let result = generate_baseline(&["echo", "test"], &baseline_path, &ValidateConfig::default());

    assert!(result.is_ok());
    assert!(baseline_path.join("manifest.json").exists());
}

/// Test generate creates syscalls.trace
/// Note: This test requires ptrace permissions and may hang in sandboxed environments
#[test]
#[ignore] // Requires ptrace permissions - run with: cargo test -- --ignored
fn test_generate_creates_binary_trace() {
    use renacer::validate::golden_trace::generate_baseline;
    use tempfile::tempdir;

    let temp = tempdir().expect("Failed to create temp dir");
    let baseline_path = temp.path().join("golden");

    let result = generate_baseline(&["echo", "test"], &baseline_path, &ValidateConfig::default());

    assert!(result.is_ok());
    assert!(baseline_path.join("syscalls.trace").exists());
}

/// Test load baseline with valid directory
/// Note: This test requires ptrace permissions and may hang in sandboxed environments
#[test]
#[ignore] // Requires ptrace permissions - run with: cargo test -- --ignored
fn test_load_baseline_valid() {
    use renacer::validate::golden_trace::{generate_baseline, load_baseline};
    use tempfile::tempdir;

    let temp = tempdir().expect("Failed to create temp dir");
    let baseline_path = temp.path().join("golden");

    // First generate
    generate_baseline(&["echo", "test"], &baseline_path, &ValidateConfig::default())
        .expect("Failed to generate");

    // Then load
    let loaded = load_baseline(&baseline_path);
    assert!(loaded.is_ok());

    let baseline = loaded.expect("test");
    assert_eq!(baseline.manifest.command, vec!["echo", "test"]);
}

/// Test load baseline returns V001 for missing manifest
#[test]
fn test_load_baseline_missing_manifest_returns_v001() {
    use renacer::validate::golden_trace::load_baseline;
    use tempfile::tempdir;

    let temp = tempdir().expect("Failed to create temp dir");
    let baseline_path = temp.path().join("nonexistent");

    let result = load_baseline(&baseline_path);

    assert!(result.is_err());
    match result.unwrap_err() {
        ValidateError::BaselineNotFound { .. } => {} // Expected V001
        other => panic!("Expected BaselineNotFound, got: {other:?}"),
    }
}

/// Test load baseline returns V002 for invalid JSON
#[test]
fn test_load_baseline_invalid_json_returns_v002() {
    use renacer::validate::golden_trace::load_baseline;
    use std::fs;
    use tempfile::tempdir;

    let temp = tempdir().expect("Failed to create temp dir");
    let baseline_path = temp.path().join("golden");
    fs::create_dir_all(&baseline_path).expect("Failed to create dir");
    fs::write(baseline_path.join("manifest.json"), "invalid json{{{").expect("Failed to write");

    let result = load_baseline(&baseline_path);

    assert!(result.is_err());
    match result.unwrap_err() {
        ValidateError::InvalidManifest { .. } => {} // Expected V002
        other => panic!("Expected InvalidManifest, got: {other:?}"),
    }
}

// ============================================================================
// Step 7: Output Format Tests (RED PHASE)
// ============================================================================

/// Test text output format structure
#[test]
fn test_text_output_format_structure() {
    let result = ComparisonResult::passed();
    let output = format_text_report(&result);

    assert!(output.contains("Validation Report"));
    assert!(output.contains("Status:"));
    assert!(output.contains("PASSED"));
}

/// Test JSON output schema
#[test]
fn test_json_output_schema() {
    let result = ComparisonResult::passed();
    let json_str = format_json_report(&result);
    let json: serde_json::Value = serde_json::from_str(&json_str).expect("Invalid JSON");

    assert!(json.get("status").is_some());
    assert!(json.get("total_compared").is_some());
    assert!(json.get("mismatches").is_some());
}

/// Test JUnit XML is valid
#[test]
fn test_junit_xml_valid() {
    let result = ComparisonResult::passed();
    let xml = format_junit_report(&result);

    // Basic XML structure checks
    assert!(xml.starts_with("<?xml version"));
    assert!(xml.contains("<testsuite"));
    assert!(xml.contains("</testsuite>"));
    assert!(xml.contains("renacer-validate"));
}

// ============================================================================
// Error Type Tests
// ============================================================================

/// Test error codes match specification Section 9.1
#[test]
fn test_error_codes() {
    let e1 = ValidateError::BaselineNotFound { path: PathBuf::from("/test") };
    assert!(format!("{e1}").contains("V001"));

    let e2 = ValidateError::InvalidManifest { reason: "bad json".to_string() };
    assert!(format!("{e2}").contains("V002"));

    let e3 =
        ValidateError::VersionMismatch { expected: "1.0".to_string(), found: "2.0".to_string() };
    assert!(format!("{e3}").contains("V003"));
}

/// Test config builder pattern
#[test]
fn test_config_builder_pattern() {
    let config = ValidateConfig::default()
        .set_tolerance(15.0)
        .with_strict_mode(false)
        .with_ignore_timing(true)
        .with_fail_fast(true)
        .with_output_format(ValidationOutputFormat::Json);

    assert!((config.tolerance_percent - 15.0).abs() < f32::EPSILON);
    assert!(!config.strict_mode);
    assert!(config.ignore_timing);
    assert!(config.fail_fast);
    assert!(matches!(config.output_format, ValidationOutputFormat::Json));
}

/// Test trace header checksum
#[test]
fn test_trace_header_checksum() {
    let header1 = TraceHeader::new(100, TraceFlags::default());
    let header2 = TraceHeader::new(100, TraceFlags::default());
    let header3 = TraceHeader::new(200, TraceFlags::default());

    // Same parameters should produce same checksum
    assert_eq!(header1.checksum, header2.checksum);
    // Different parameters should produce different checksum
    assert_ne!(header1.checksum, header3.checksum);
}

/// Test manifest includes renacer version
#[test]
fn test_manifest_renacer_version() {
    let manifest = TraceManifest::new(vec!["test".to_string()]);

    assert_eq!(manifest.renacer_version, env!("CARGO_PKG_VERSION"));
}

// ============================================================================
// Step 9: CLI Tests
// ============================================================================

/// Test validate subcommand exists and parses
#[test]
fn test_cli_validate_subcommand_exists() {
    use clap::Parser;
    use renacer::cli::{Cli, Commands};

    let cli =
        Cli::parse_from(["renacer", "validate", "--generate", "/tmp/golden", "--", "echo", "test"]);

    assert!(cli.subcommand.is_some());
    match cli.subcommand.expect("test") {
        Commands::Validate(args) => {
            assert_eq!(args.generate.expect("test").to_str().expect("test"), "/tmp/golden");
            assert_eq!(args.command, vec!["echo", "test"]);
        }
        _ => panic!("Expected Validate subcommand"),
    }
}

/// Test validate --baseline flag
#[test]
fn test_cli_validate_baseline_flag() {
    use clap::Parser;
    use renacer::cli::{Cli, Commands};

    let cli = Cli::parse_from([
        "renacer",
        "validate",
        "--baseline",
        "/golden/baseline",
        "--",
        "echo",
        "test",
    ]);

    match cli.subcommand.expect("test") {
        Commands::Validate(args) => {
            assert_eq!(args.baseline.expect("test").to_str().expect("test"), "/golden/baseline");
        }
        _ => panic!("Expected Validate subcommand"),
    }
}

/// Test validate --tolerance flag
#[test]
fn test_cli_validate_tolerance_flag() {
    use clap::Parser;
    use renacer::cli::{Cli, Commands};

    let cli = Cli::parse_from([
        "renacer",
        "validate",
        "--tolerance",
        "15.0",
        "--generate",
        "/tmp/golden",
        "--",
        "echo",
        "test",
    ]);

    match cli.subcommand.expect("test") {
        Commands::Validate(args) => {
            assert!((args.tolerance - 15.0).abs() < f32::EPSILON);
        }
        _ => panic!("Expected Validate subcommand"),
    }
}

/// Test validate --strict flag
#[test]
fn test_cli_validate_strict_flag() {
    use clap::Parser;
    use renacer::cli::{Cli, Commands};

    let cli = Cli::parse_from([
        "renacer",
        "validate",
        "--strict",
        "--generate",
        "/tmp/golden",
        "--",
        "echo",
        "test",
    ]);

    match cli.subcommand.expect("test") {
        Commands::Validate(args) => {
            assert!(args.strict);
        }
        _ => panic!("Expected Validate subcommand"),
    }
}

/// Test validate --output flag options
#[test]
fn test_cli_validate_output_format() {
    use clap::Parser;
    use renacer::cli::{Cli, Commands, ValidationOutputFormat};

    // JSON format
    let cli = Cli::parse_from([
        "renacer",
        "validate",
        "--output",
        "json",
        "--generate",
        "/tmp/golden",
        "--",
        "echo",
        "test",
    ]);

    match cli.subcommand.expect("test") {
        Commands::Validate(args) => {
            assert_eq!(args.output, ValidationOutputFormat::Json);
        }
        _ => panic!("Expected Validate subcommand"),
    }

    // JUnit format
    let cli = Cli::parse_from([
        "renacer",
        "validate",
        "--output",
        "junit",
        "--generate",
        "/tmp/golden",
        "--",
        "echo",
        "test",
    ]);

    match cli.subcommand.expect("test") {
        Commands::Validate(args) => {
            assert_eq!(args.output, ValidationOutputFormat::Junit);
        }
        _ => panic!("Expected Validate subcommand"),
    }
}

/// Test validate default tolerance value
#[test]
fn test_cli_validate_tolerance_default() {
    use clap::Parser;
    use renacer::cli::{Cli, Commands};

    let cli =
        Cli::parse_from(["renacer", "validate", "--generate", "/tmp/golden", "--", "echo", "test"]);

    match cli.subcommand.expect("test") {
        Commands::Validate(args) => {
            assert!((args.tolerance - 10.0).abs() < f32::EPSILON); // Default is 10.0
        }
        _ => panic!("Expected Validate subcommand"),
    }
}
