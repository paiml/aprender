//! Analyze realizar hand-rolled PTX for bugs
//!
//! Run: `cargo run -p trueno-explain --example analyze_realizar`

use std::fs;
use trueno_explain::{BugSeverity, PtxBugAnalyzer};

/// Severity icon for display output.
fn severity_icon(p0: usize, p1: usize, p2: usize) -> &'static str {
    if p0 > 0 {
        "🔴"
    } else if p1 > 0 {
        "🟡"
    } else if p2 > 0 {
        "🟠"
    } else {
        "✅"
    }
}

/// Print detailed bug information for an analysis result.
fn print_bug_details(result: &trueno_explain::PtxBugReport) {
    for bug in &result.bugs {
        println!("   └─ [{}] {}: {}", bug.class.severity(), bug.class.code(), bug.message);
        if bug.line > 0 {
            println!("      Line {}: {}", bug.line, bug.instruction);
        }
        if let Some(fix) = &bug.fix {
            println!("      Fix: {}", fix);
        }
    }
}

/// Analyze a single PTX file and return `(total_bugs, p0, p1, p2)` counts.
fn analyze_file(
    analyzer: &PtxBugAnalyzer,
    ptx_dir: &str,
    file: &str,
) -> Option<(usize, usize, usize, usize)> {
    let path = format!("{}/{}", ptx_dir, file);
    let ptx = match fs::read_to_string(&path) {
        Ok(content) => content,
        Err(e) => {
            println!("❌ Could not read {}: {}", file, e);
            return None;
        }
    };

    let result = analyzer.analyze(&ptx);
    let p0 = result.count_by_severity(BugSeverity::Critical);
    let p1 = result.count_by_severity(BugSeverity::High);
    let p2 = result.count_by_severity(BugSeverity::Medium);

    println!(
        "{} {} - {} bugs ({} P0, {} P1, {} P2)",
        severity_icon(p0, p1, p2),
        file,
        result.bugs.len(),
        p0,
        p1,
        p2
    );
    print_bug_details(&result);

    Some((result.bugs.len(), p0, p1, p2))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    REALIZAR PTX BUG ANALYSIS                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let ptx_dir = "/tmp/realizar_ptx_analysis";
    let files = [
        "bias_activation.ptx",
        "gemm_fp16_wmma.ptx",
        "fused_q4k_q8_dot.ptx",
        "multi_head_attention.ptx",
    ];

    let analyzer = PtxBugAnalyzer::strict();

    let mut total_bugs = 0;
    let mut p0_total = 0;
    let mut p1_total = 0;
    let mut p2_total = 0;

    for file in &files {
        if let Some((bugs, p0, p1, p2)) = analyze_file(&analyzer, ptx_dir, file) {
            total_bugs += bugs;
            p0_total += p0;
            p1_total += p1;
            p2_total += p2;
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("REALIZAR PTX SUMMARY");
    println!("{}", "=".repeat(80));
    println!("  Files analyzed: {}", files.len());
    println!("  Total bugs: {}", total_bugs);
    println!("  🔴 P0 Critical: {}", p0_total);
    println!("  🟡 P1 High: {}", p1_total);
    println!("  🟠 P2 Medium: {}", p2_total);

    if p0_total > 0 {
        println!("\n⚠️  CRITICAL BUGS FOUND in realizar - these need porting to trueno-gpu!");
    }
}
