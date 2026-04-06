//! Golden Trace Validation Example (Sprint 50)
//!
//! This example demonstrates how to use renacer's validate subcommand
//! for regression detection through golden trace comparison.
//!
//! Run with: cargo run --example validate_golden_trace
//!
//! Toyota Way Principle: Jidoka (自動化) - Stop on regression detection

use renacer::validate::{
    compare_syscalls, compare_timing, format_json_report, format_junit_report, format_text_report,
    ComparisonResult, SyscallTimingStats, TimingRegression, TraceFlags, TraceHeader,
    TraceSyscallEntry, ValidateConfig, ValidateExitCode,
};
use std::collections::HashMap;

fn main() {
    println!("🚀 Renacer Golden Trace Validation Demo (Sprint 50)\n");
    println!("Toyota Way Principle: Jidoka (Stop on regression detection)\n");

    // ========================================================================
    // Demo 1: Understanding Exit Codes
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 1: Validate Exit Codes (per APR Spec Section 3.4)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let exit_codes = [
        (ValidateExitCode::Passed, "Validation passed"),
        (ValidateExitCode::Failed, "Regression detected"),
        (ValidateExitCode::BaselineNotFound, "Baseline not found"),
        (ValidateExitCode::InvalidBaseline, "Invalid baseline format"),
        (ValidateExitCode::CommandError, "Command execution error"),
        (ValidateExitCode::ConfigError, "Configuration error"),
    ];

    for (code, description) in exit_codes {
        println!("   Exit {}: {}", code.code(), description);
    }

    // ========================================================================
    // Demo 2: Binary Trace Format
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 2: Binary Trace Format (per APR Spec Section 4.3)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let flags = TraceFlags::new(false, true, true);
    let header = TraceHeader::new(42, flags);

    println!("   Magic bytes: {:?} (\"RNTR\")", header.magic);
    println!("   Version: {}", header.version);
    println!("   Entry count: {}", header.entry_count);
    println!("   Flags:");
    println!("      - compressed: {}", flags.compressed());
    println!("      - has_timing: {}", flags.has_timing());
    println!("      - has_string_args: {}", flags.has_string_args());

    // Demonstrate roundtrip encoding
    let encoded = header.encode();
    let decoded = TraceHeader::decode(&encoded).expect("Failed to decode");
    println!("\n   Roundtrip encoding: {} bytes", encoded.len());
    println!("   Decoded entry_count: {} ✓", decoded.entry_count);

    // ========================================================================
    // Demo 3: Syscall Sequence Comparison
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 3: Syscall Sequence Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create baseline syscall sequence
    let baseline = vec![
        TraceSyscallEntry::simple(257, "openat", 50_000), // 50µs
        TraceSyscallEntry::simple(0, "read", 100_000),    // 100µs
        TraceSyscallEntry::simple(1, "write", 75_000),    // 75µs
        TraceSyscallEntry::simple(3, "close", 10_000),    // 10µs
    ];

    // Identical sequence should pass
    let identical = baseline.clone();
    let result = compare_syscalls(&baseline, &identical, false).expect("test");
    println!("   Identical sequences:");
    println!("      Passed: {} ✓", result.passed);
    println!("      Mismatches: {}", result.syscall_mismatches.len());

    // Different sequence should fail
    let different = vec![
        TraceSyscallEntry::simple(257, "openat", 50_000),
        TraceSyscallEntry::simple(1, "write", 75_000), // Swapped order!
        TraceSyscallEntry::simple(0, "read", 100_000),
        TraceSyscallEntry::simple(3, "close", 10_000),
    ];
    let result = compare_syscalls(&baseline, &different, false).expect("test");
    println!("\n   Different sequences:");
    println!("      Passed: {}", result.passed);
    println!("      Mismatches: {} ✗", result.syscall_mismatches.len());

    // ========================================================================
    // Demo 4: Timing Tolerance
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 4: Timing Tolerance Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut baseline_stats = HashMap::new();
    baseline_stats.insert(
        "read".to_string(),
        SyscallTimingStats {
            count: 100,
            total_ns: 10_000_000,
            mean_ns: 100_000, // 100µs baseline
            std_ns: 10_000,
            min_ns: 80_000,
            max_ns: 120_000,
            p50_ns: 100_000,
            p95_ns: 115_000,
            p99_ns: 118_000,
        },
    );

    // 5% slower - within 10% tolerance
    let mut actual_5pct = HashMap::new();
    actual_5pct.insert(
        "read".to_string(),
        SyscallTimingStats {
            count: 100,
            total_ns: 10_500_000,
            mean_ns: 105_000, // 5% slower
            std_ns: 10_000,
            min_ns: 85_000,
            max_ns: 125_000,
            p50_ns: 105_000,
            p95_ns: 120_000,
            p99_ns: 123_000,
        },
    );

    let regressions = compare_timing(&baseline_stats, &actual_5pct, 10.0);
    println!("   5% slower with 10% tolerance:");
    println!("      Regressions detected: {} ✓", regressions.len());

    // 20% slower - exceeds 10% tolerance
    let mut actual_20pct = HashMap::new();
    actual_20pct.insert(
        "read".to_string(),
        SyscallTimingStats {
            count: 100,
            total_ns: 12_000_000,
            mean_ns: 120_000, // 20% slower
            std_ns: 10_000,
            min_ns: 100_000,
            max_ns: 140_000,
            p50_ns: 120_000,
            p95_ns: 135_000,
            p99_ns: 138_000,
        },
    );

    let regressions = compare_timing(&baseline_stats, &actual_20pct, 10.0);
    println!("\n   20% slower with 10% tolerance:");
    println!("      Regressions detected: {} ✗", regressions.len());
    if let Some(reg) = regressions.first() {
        println!("      Syscall: {}", reg.syscall);
        println!("      Baseline: {:.2}ms", reg.baseline_ms);
        println!("      Actual: {:.2}ms", reg.actual_ms);
        println!("      Delta: {:+.1}%", reg.delta_percent);
    }

    // ========================================================================
    // Demo 5: Configuration Options
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 5: Configuration Options");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let default_config = ValidateConfig::default();
    println!("   Default configuration:");
    println!("      Tolerance: {}%", default_config.tolerance_percent);
    println!("      Strict mode: {}", default_config.strict_mode);
    println!("      Ignore timing: {}", default_config.ignore_timing);

    let strict_config = ValidateConfig::default().with_strict_mode(true);
    println!("\n   Strict mode configuration:");
    println!("      Tolerance: {}% (overridden)", strict_config.tolerance_percent);
    println!("      Strict mode: {}", strict_config.strict_mode);

    let lenient_config = ValidateConfig::default().set_tolerance(25.0).with_ignore_timing(true);
    println!("\n   Lenient configuration:");
    println!("      Tolerance: {}%", lenient_config.tolerance_percent);
    println!("      Ignore timing: {}", lenient_config.ignore_timing);

    // ========================================================================
    // Demo 6: Output Formats
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 6: Output Formats (Text, JSON, JUnit)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create a sample result with regressions
    let result_with_regression = ComparisonResult::failed(
        vec![],
        vec![TimingRegression {
            syscall: "read".to_string(),
            baseline_ms: 0.1,
            actual_ms: 0.12,
            delta_percent: 20.0,
        }],
    );

    println!("   Text format (human readable):");
    let text = format_text_report(&result_with_regression);
    for line in text.lines().take(8) {
        println!("      {}", line);
    }
    println!("      ...");

    println!("\n   JSON format (machine parsing):");
    let json = format_json_report(&result_with_regression);
    for line in json.lines().take(6) {
        println!("      {}", line);
    }
    println!("      ...");

    println!("\n   JUnit format (CI integration):");
    let junit = format_junit_report(&result_with_regression);
    for line in junit.lines().take(5) {
        println!("      {}", line);
    }
    println!("      ...");

    // ========================================================================
    // Demo 7: CLI Usage
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 7: CLI Usage");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("   Generate baseline:");
    println!("      renacer validate --generate ./golden -- echo \"hello\"\n");

    println!("   Compare against baseline:");
    println!("      renacer validate --baseline ./golden -- echo \"hello\"\n");

    println!("   With custom tolerance:");
    println!("      renacer validate --baseline ./golden --tolerance 15.0 -- ./myapp\n");

    println!("   Strict mode (zero tolerance):");
    println!("      renacer validate --baseline ./golden --strict -- ./myapp\n");

    println!("   JSON output for CI:");
    println!("      renacer validate --baseline ./golden --output json -- ./myapp\n");

    println!("   JUnit output for test frameworks:");
    println!("      renacer validate --baseline ./golden --output junit -- ./myapp\n");

    println!("   Ignore timing (behavior only):");
    println!("      renacer validate --baseline ./golden --ignore-timing -- ./myapp\n");

    // ========================================================================
    // Summary
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Summary: Golden Trace Validation (Sprint 50)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("✅ Key Features:");
    println!("   - Golden trace baseline generation");
    println!("   - Syscall sequence comparison");
    println!("   - Timing regression detection with configurable tolerance");
    println!("   - Multiple output formats (Text, JSON, JUnit)");
    println!("   - Strict mode for zero-tolerance validation");
    println!("   - CI/CD integration support");

    println!("\n✅ Exit Codes:");
    println!("   - 0: Validation passed");
    println!("   - 1: Regression detected");
    println!("   - 2: Baseline not found");
    println!("   - 3: Invalid baseline format");
    println!("   - 4: Command execution error");
    println!("   - 5: Configuration error");

    println!("\n✅ Toyota Way Integration:");
    println!("   - Jidoka: Stop on regression detection");
    println!("   - Muda: Eliminate waste from repeated manual testing");
    println!("   - Poka-Yoke: Prevent defects from reaching production");

    println!("\n🎉 Sprint 50 Complete: APR Runtime Model Tracing Support!\n");
}
