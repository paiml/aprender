//! Run the contract quality gate (validate + audit + score) on a directory.
//!
//! Demonstrates all lint features: text/json/sarif/github output, suppression,
//! strict mode, severity filter, cache stats, and trend tracking.
//!
//! Usage:
//!   cargo run --example lint -- contracts/
//!   cargo run --example lint -- contracts/ 0.60
//!   cargo run --example lint -- contracts/ 0.0 sarif
//!   cargo run --example lint -- contracts/ 0.0 text --strict
//!   cargo run --example lint -- contracts/ 0.0 text --suppress-rule PV-SCR-001
//!   cargo run --example lint -- contracts/ 0.0 text --cache-stats
//!   cargo run --example lint -- contracts/ 0.0 text --trend

// Test/example/bench binary: `.unwrap()` is idiomatic here; the lib`s
// cfg(test) allow does not reach this separate crate.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process;

use provable_contracts::lint::sarif::{findings_to_sarif, sarif_to_json};
use provable_contracts::lint::trend;
use provable_contracts::lint::{run_lint, GateDetail, LintConfig, LintReport};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let dir = args.get(1).map_or_else(
        || {
            eprintln!(
                "Usage: lint <contracts-dir/> [min-score] [format] [flags]\n\
                 \n\
                 Formats: text (default), json, sarif, github\n\
                 \n\
                 Flags:\n\
                 \x20 --strict             Promote warnings to errors\n\
                 \x20 --suppress-rule ID   Suppress findings by rule (e.g. PV-SCR-001)\n\
                 \x20 --severity error     Filter to error-severity only\n\
                 \x20 --no-cache           Disable content-addressable cache\n\
                 \x20 --cache-stats        Show cache hit/miss statistics\n\
                 \x20 --trend              Record trend snapshot and check for drift"
            );
            process::exit(1);
        },
        PathBuf::from,
    );

    let min_score: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    let format = args.get(3).map_or("text", |s| s.as_str());

    let strict = args.iter().any(|a| a == "--strict");
    let no_cache = args.iter().any(|a| a == "--no-cache");
    let cache_stats = args.iter().any(|a| a == "--cache-stats");
    let do_trend = args.iter().any(|a| a == "--trend");
    let suppress_rule = flag_value(&args, "--suppress-rule");
    let severity = flag_value(&args, "--severity");

    let mut config = LintConfig::new(&dir, None, min_score);
    config.strict = strict;
    config.no_cache = no_cache;
    config.cache_stats = cache_stats;
    if let Some(rule) = suppress_rule {
        config.suppressed_rules = vec![rule];
    }
    if let Some(ref sev) = severity {
        config.severity_filter = provable_contracts::lint::rules::RuleSeverity::from_str_opt(sev);
    }

    let report = run_lint(&config);

    // Print cache statistics if requested
    if cache_stats {
        let cs = &report.cache_stats;
        eprintln!(
            "Cache: {} total, {} hits, {} misses ({:.0}% hit rate)",
            cs.total,
            cs.hits,
            cs.misses,
            cs.hit_rate() * 100.0,
        );
    }

    // Record trend snapshot and detect drift
    if do_trend {
        let trend_root = trend::trend_dir(&dir);
        let contracts_count = count_contracts(&report);
        match trend::record_snapshot(&trend_root, &report, contracts_count) {
            Ok(path) => eprintln!("Trend snapshot: {}", path.display()),
            Err(e) => eprintln!("Warning: trend snapshot failed: {e}"),
        }
        let snapshots = trend::load_snapshots(&trend_root);
        if let Some(drop) = trend::detect_drift(&snapshots, 0.05) {
            eprintln!("Warning: quality drift detected (score dropped {drop:.3})");
        }
        if snapshots.len() >= 2 {
            eprintln!("\n{}", trend::format_trend(&snapshots, 10));
        }
    }

    match format {
        "sarif" => print_sarif(&report),
        "json" => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        "github" => print_github(&report),
        _ => print_text(&report),
    }

    if !report.passed {
        process::exit(1);
    }
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn count_contracts(report: &LintReport) -> usize {
    for gate in &report.gates {
        match &gate.detail {
            GateDetail::Validate { contracts, .. }
            | GateDetail::Audit { contracts, .. }
            | GateDetail::Score { contracts, .. } => return *contracts,
            GateDetail::Skipped { .. }
            | GateDetail::Verify { .. }
            | GateDetail::Enforce { .. }
            | GateDetail::ReverseCoverage { .. }
            | GateDetail::Composition { .. }
            | GateDetail::DuplicateStems { .. } => {}
        }
    }
    0
}

fn print_sarif(report: &LintReport) {
    let sarif = findings_to_sarif(&report.findings, env!("CARGO_PKG_VERSION"));
    println!("{}", sarif_to_json(&sarif, true));
}

fn print_github(report: &LintReport) {
    for f in &report.findings {
        if !f.suppressed {
            println!("{}", f.to_github_annotation());
        }
    }
}

fn print_text(report: &LintReport) {
    println!("pv lint — contract quality gate");
    println!("================================\n");
    for (i, gate) in report.gates.iter().enumerate() {
        print_gate(i + 1, gate);
    }
    print_findings(report);
    print_summary(report);
}

fn print_gate(num: usize, gate: &provable_contracts::lint::GateResult) {
    let status = if gate.skipped {
        "SKIP"
    } else if gate.passed {
        "PASS"
    } else {
        "FAIL"
    };
    let summary = gate_summary(&gate.detail);
    println!(
        "Gate {num}: {:.<30} {status} ({summary}) [{}ms]",
        gate.name, gate.duration_ms
    );
}

fn gate_summary(detail: &GateDetail) -> String {
    match detail {
        GateDetail::Validate {
            contracts,
            errors,
            warnings,
            ..
        } => {
            format!("{contracts} contracts, {errors} errors, {warnings} warnings")
        }
        GateDetail::Audit {
            contracts,
            findings,
            ..
        } => {
            format!("{contracts} contracts, {findings} findings")
        }
        GateDetail::Score {
            contracts,
            min_score,
            mean_score,
            threshold,
            ..
        } => {
            format!(
                "{contracts} contracts, min={min_score:.2}, mean={mean_score:.2}, threshold={threshold:.2}"
            )
        }
        GateDetail::Skipped { reason } => reason.clone(),
        GateDetail::Verify {
            total_refs,
            existing,
            missing,
        } => {
            format!("{total_refs} refs, {existing} existing, {missing} missing")
        }
        GateDetail::Enforce {
            equations_total,
            equations_with_pre,
            equations_with_post,
            ..
        } => {
            format!(
                "{equations_total} equations, {equations_with_pre} pre, {equations_with_post} post"
            )
        }
        GateDetail::ReverseCoverage {
            total_pub_fns,
            bound_fns,
            coverage_pct,
            threshold_pct,
            ..
        } => {
            format!(
                "{bound_fns}/{total_pub_fns} pub fns bound ({coverage_pct:.1}%, threshold {threshold_pct:.1}%)"
            )
        }
        GateDetail::Composition {
            edges_checked,
            edges_satisfied,
            edges_broken,
        } => format!("{edges_checked} edges, {edges_satisfied} satisfied, {edges_broken} broken"),
        GateDetail::DuplicateStems {
            divergent,
            baselined,
            unbaselined,
            stale,
            ..
        } => format!(
            "{divergent} ambiguous stems, {baselined} baselined, {} unbaselined, {} stale",
            unbaselined.len(),
            stale.len()
        ),
    }
}

fn print_findings(report: &LintReport) {
    if report.findings.is_empty() {
        return;
    }
    let unsuppressed = report.findings.iter().filter(|f| !f.suppressed).count();
    let suppressed = report.findings.len() - unsuppressed;
    println!(
        "\nFindings: {} total ({suppressed} suppressed)",
        report.findings.len()
    );
    for f in report.findings.iter().filter(|f| !f.suppressed) {
        println!("  {f}");
    }
}

fn print_summary(report: &LintReport) {
    let passed = report.gates.iter().filter(|g| g.passed).count();
    let total = report.gates.len();
    let result = if report.passed { "PASS" } else { "FAIL" };
    println!(
        "\nResult: {result} ({passed}/{total} gates passed) [{}ms]",
        report.total_duration_ms
    );
}
