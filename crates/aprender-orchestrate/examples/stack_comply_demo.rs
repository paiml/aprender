//! Stack Compliance Demo
//!
//! Demonstrates the cross-project consistency enforcement using MinHash+LSH.
//!
//! Run with: cargo run --example stack_comply_demo

use batuta::comply::{ComplyConfig, ComplyReportFormat, StackComplyEngine};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    println!("=== Stack Compliance Demo ===\n");

    // 1. Create compliance engine with default configuration
    println!("1. Creating compliance engine");
    let config = ComplyConfig::default();
    let engine = StackComplyEngine::new(config);

    // 2. List available rules
    println!("\n2. Available compliance rules:");
    for (id, description) in engine.available_rules() {
        println!("   • {} - {}", id, description);
    }

    // 3. Discover projects in workspace
    println!("\n3. Discovering projects in current workspace");
    let mut engine = StackComplyEngine::default_for_workspace(Path::new("."));
    let projects = engine.discover_projects(Path::new("."))?;
    println!("   Found {} projects", projects.len());
    for project in projects {
        let paiml_marker = if project.is_paiml_crate { " [PAIML]" } else { "" };
        println!("   • {}{}", project.name, paiml_marker);
    }

    // 4. Run compliance checks
    println!("\n4. Running compliance checks");
    let report = engine.check_all();

    // 5. Display report summary via the summary field
    println!("\n5. Compliance Report Summary:");
    println!("   Pass rate: {:.1}%", report.summary.pass_rate);
    println!("   Total checks: {}", report.summary.total_checks);
    println!("   Passed: {}", report.summary.passed_checks);
    println!("   Failed: {}", report.summary.failed_checks);

    // 6. Show formatted report
    println!("\n6. Text Report:");
    println!("{}", report.format(ComplyReportFormat::Text));

    // 7. Demonstrate rule categories
    println!("\n7. Rule Categories:");
    println!("   • makefile-targets: Ensures Makefile target consistency");
    println!("   • cargo-toml-consistency: Validates Cargo.toml parity");
    println!("   • ci-workflow-parity: Checks CI workflow alignment");
    println!("   • code-duplication: Detects duplicates via MinHash+LSH");

    println!("\n=== Demo Complete ===");
    println!("Use 'batuta stack comply --help' for CLI usage.");

    Ok(())
}
