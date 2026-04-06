//! PMAT Bridge Demo
//!
//! Demonstrates the pmat integration bridge (PROBAR-PMAT-001) that
//! connects probar's compliance framework with pmat static analysis:
//! - Running pmat quality-gate
//! - Parsing violation counts
//! - Converting to compliance checks
//!
//! Run with: cargo run --example pmat_bridge_demo -p jugar-probar
//!
//! Note: This demo shows the API but actual pmat execution requires
//! pmat to be installed: `cargo install pmat`

use jugar_probar::comply::{ComplianceStatus, PmatBridge, PmatResult};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       PMAT Bridge Demo (PROBAR-PMAT-001)                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_pmat_result();
    demo_bridge_builder();
    demo_output_parsing();
    demo_compliance_checks();
    demo_integration_workflow();
}

/// Demonstrate PmatResult structure
fn demo_pmat_result() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. PmatResult Structure");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Simulate a pmat result (as if we ran pmat quality-gate)
    let result = PmatResult {
        satd_count: 355,
        satd_critical: 17,
        complexity_count: 64,
        dead_code_count: 6,
        duplicate_count: 88,
        security_count: 0,
        coverage_count: 0,
        documentation_count: 0,
        total_violations: 513,
        passed: false,
        raw_output: String::new(),
    };

    println!("  Violation Categories:");
    println!("  ┌────────────────────────────────────────┐");
    println!("  │ Category              │ Count          │");
    println!("  ├────────────────────────────────────────┤");
    println!("  │ SATD (Technical Debt) │ {:>14} │", result.satd_count);
    println!("  │   └─ Critical         │ {:>14} │", result.satd_critical);
    println!(
        "  │ Complexity            │ {:>14} │",
        result.complexity_count
    );
    println!(
        "  │ Dead Code             │ {:>14} │",
        result.dead_code_count
    );
    println!(
        "  │ Duplicates            │ {:>14} │",
        result.duplicate_count
    );
    println!(
        "  │ Security              │ {:>14} │",
        result.security_count
    );
    println!("  ├────────────────────────────────────────┤");
    println!(
        "  │ Total Violations      │ {:>14} │",
        result.total_violations
    );
    println!("  └────────────────────────────────────────┘");
    println!();

    println!("  Result Helpers:");
    println!("    ├─ has_critical(): {}", result.has_critical());
    println!("    ├─ error_count(): {}", result.error_count());
    println!("    ├─ warning_count(): {}", result.warning_count());
    println!("    └─ passed: {}", result.passed);
    println!();
}

/// Demonstrate builder pattern
fn demo_bridge_builder() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. Bridge Builder Pattern");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Default bridge
    let default_bridge = PmatBridge::new();
    println!("  Default Bridge:");
    println!("    └─ Uses 'pmat' from PATH\n");

    // Custom binary path
    let custom_bridge = PmatBridge::new()
        .with_pmat_path("/usr/local/bin/pmat")
        .with_flag("--strict")
        .with_flag("--no-color");

    println!("  Custom Bridge:");
    println!("    ├─ Custom path: /usr/local/bin/pmat");
    println!("    ├─ Flag: --strict");
    println!("    └─ Flag: --no-color\n");

    // Check availability
    println!("  Availability Check:");
    if default_bridge.is_available() {
        println!("    ✓ pmat is installed and available");
    } else {
        println!("    ✗ pmat not found (install with: cargo install pmat)");
    }
    let _ = custom_bridge; // silence unused warning
    println!();
}

/// Demonstrate output parsing
fn demo_output_parsing() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Output Parsing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Simulate pmat output format
    let pmat_output = r#"
🔍 Running quality gate checks...

  🔍 Checking complexity... 64 violations found
  🔍 Checking dead code... 6 violations found
  🔍 Checking technical debt... 355 violations (17 critical)
  🔍 Checking security... 0 violations found
  🔍 Checking duplicates... 88 violations found

Quality Gate: FAILED
Total violations: 513
"#;

    println!("  Sample pmat output:");
    for line in pmat_output.lines().take(10) {
        if !line.is_empty() {
            println!("    {}", line);
        }
    }
    println!("    ...\n");

    // The bridge parses this format internally
    println!("  Parsed Result:");
    println!("    ├─ Complexity: 64");
    println!("    ├─ Dead Code: 6");
    println!("    ├─ SATD: 355 (17 critical)");
    println!("    ├─ Security: 0");
    println!("    ├─ Duplicates: 88");
    println!("    └─ Total: 513");
    println!();
}

/// Demonstrate compliance check conversion
fn demo_compliance_checks() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. Compliance Check Conversion");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let bridge = PmatBridge::new();

    // Create a sample result
    let result = PmatResult {
        satd_count: 10,
        satd_critical: 2,
        complexity_count: 5,
        dead_code_count: 0,
        duplicate_count: 3,
        security_count: 1,
        coverage_count: 0,
        documentation_count: 0,
        total_violations: 19,
        passed: false,
        raw_output: String::new(),
    };

    let checks = bridge.to_compliance_checks(&result);

    println!("  Compliance Checks Generated:");
    println!("  ┌─────────────────────────────────────────────────────────┐");

    for check in &checks {
        let status_icon = match check.status {
            ComplianceStatus::Pass => "✓ PASS",
            ComplianceStatus::Fail => "✗ FAIL",
            ComplianceStatus::Warn => "⚠ WARN",
            ComplianceStatus::Skip => "○ SKIP",
        };
        println!("  │ {} │ {:20} │ {}", check.id, check.name, status_icon);
        if let Some(ref details) = check.details {
            println!("  │                    └─ {}", details);
        }
    }

    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    // Summary
    let pass_count = checks
        .iter()
        .filter(|c| c.status == ComplianceStatus::Pass)
        .count();
    let fail_count = checks
        .iter()
        .filter(|c| c.status == ComplianceStatus::Fail)
        .count();
    let warn_count = checks
        .iter()
        .filter(|c| c.status == ComplianceStatus::Warn)
        .count();

    println!(
        "  Summary: {} passed, {} failed, {} warnings",
        pass_count, fail_count, warn_count
    );
    println!();
}

/// Demonstrate integration workflow
fn demo_integration_workflow() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. Integration Workflow");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Typical Usage:");
    println!();
    println!("    ```rust");
    println!("    use jugar_probar::comply::PmatBridge;");
    println!("    use std::path::Path;");
    println!();
    println!("    let bridge = PmatBridge::new();");
    println!();
    println!("    // Check if pmat is available");
    println!("    if !bridge.is_available() {{");
    println!("        eprintln!(\"pmat not found\");");
    println!("        return;");
    println!("    }}");
    println!();
    println!("    // Run quality gate");
    println!("    let result = bridge.run_quality_gate(Path::new(\"src/\"))?;");
    println!();
    println!("    // Check for critical issues");
    println!("    if result.has_critical() {{");
    println!("        eprintln!(\"Critical issues found!\");");
    println!("    }}");
    println!();
    println!("    // Get compliance result");
    println!("    let compliance = bridge.check_compliance(Path::new(\"src/\"))?;");
    println!("    println!(\"Compliance: {{}}\", compliance.summary());");
    println!("    ```");
    println!();

    println!("  Integration with WASM Compliance:");
    println!();
    println!("    The PmatBridge integrates with probar's WASM compliance");
    println!("    framework, adding pmat checks to the compliance report:");
    println!();
    println!("    ├─ PMAT-SATD-001: SATD Detection");
    println!("    ├─ PMAT-COMPLEXITY-001: Complexity Analysis");
    println!("    ├─ PMAT-DEADCODE-001: Dead Code Detection");
    println!("    ├─ PMAT-SECURITY-001: Security Analysis");
    println!("    └─ PMAT-DUPLICATE-001: Code Duplication");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
