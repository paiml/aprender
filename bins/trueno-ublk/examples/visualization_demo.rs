//! Visualization demo example (VIZ-001/002/003/004)
//!
//! Demonstrates trueno-ublk's renacer visualization integration.
//!
//! Run with: `cargo run --example visualization_demo -p trueno-ublk`

use std::collections::HashMap;

/// Simulated metrics for demonstration (actual metrics come from `TruenoCollector`)
fn simulate_metrics() -> HashMap<String, f64> {
    let mut metrics = HashMap::new();

    // Performance metrics
    metrics.insert("throughput_gbps".to_string(), 7.9);
    metrics.insert("iops".to_string(), 666_000.0);
    metrics.insert("compression_ratio".to_string(), 2.8);

    // Tier distribution
    metrics.insert("tier_kernel_zram_pct".to_string(), 0.65);
    metrics.insert("tier_simd_pct".to_string(), 0.25);
    metrics.insert("tier_nvme_pct".to_string(), 0.05);
    metrics.insert("tier_samefill_pct".to_string(), 0.05);

    // Page statistics
    metrics.insert("pages_total".to_string(), 262_144.0);
    metrics.insert("same_fill_pages".to_string(), 13_107.0);

    // Entropy distribution
    metrics.insert("entropy_p50".to_string(), 4.2);
    metrics.insert("entropy_p95".to_string(), 7.3);
    metrics.insert("entropy_p99".to_string(), 7.8);

    metrics
}

fn print_bar(label: &str, value: f64, max: f64, width: usize) {
    #[allow(clippy::cast_sign_loss)]
    let filled = ((value / max) * width as f64) as usize;
    let bar: String = "█".repeat(filled) + &"░".repeat(width - filled);
    println!("  {:<20} [{bar}] {:.1}%", label, value * 100.0);
}

fn main() {
    println!("trueno-ublk Visualization Demo (v3.17.0)");
    println!("=========================================\n");

    println!("This example demonstrates the metrics collected by TruenoCollector");
    println!("for integration with the renacer visualization framework.\n");

    let metrics = simulate_metrics();

    // Performance Section
    println!("Performance Metrics");
    println!("-------------------");
    println!("  Throughput:        {:.1} GB/s", metrics["throughput_gbps"]);
    println!("  IOPS:              {:.0} K", metrics["iops"] / 1000.0);
    println!("  Compression Ratio: {:.1}:1", metrics["compression_ratio"]);
    println!();

    // Tier Distribution (ASCII bar chart)
    println!("Tier Distribution");
    println!("-----------------");
    print_bar("Kernel ZRAM", metrics["tier_kernel_zram_pct"], 1.0, 30);
    print_bar("SIMD ZSTD", metrics["tier_simd_pct"], 1.0, 30);
    print_bar("NVMe Direct", metrics["tier_nvme_pct"], 1.0, 30);
    print_bar("Same-Fill", metrics["tier_samefill_pct"], 1.0, 30);
    println!();

    // Entropy Distribution
    println!("Entropy Distribution");
    println!("--------------------");
    println!("  p50: {:.1} bits/byte (low entropy → kernel zram)", metrics["entropy_p50"]);
    println!("  p95: {:.1} bits/byte (medium → SIMD compression)", metrics["entropy_p95"]);
    println!("  p99: {:.1} bits/byte (high → skip compression)", metrics["entropy_p99"]);
    println!();

    // CLI Integration
    println!("CLI Integration");
    println!("---------------");
    println!("  # Real-time TUI visualization");
    println!("  sudo trueno-ublk create --size 8G --backend tiered --visualize -f");
    println!();
    println!("  # JSON benchmark report");
    println!("  trueno-ublk benchmark --size 4G --format json");
    println!();
    println!("  # HTML benchmark report");
    println!("  trueno-ublk benchmark --size 4G --format html -o report.html");
    println!();
    println!("  # OTLP tracing to Jaeger");
    println!("  sudo trueno-ublk create --size 8G --otlp-endpoint http://localhost:4317");
    println!();

    // Output formats
    println!("Output Formats (VIZ-003)");
    println!("------------------------");
    println!("  text  - Human-readable console output");
    println!("  json  - Machine-parseable JSON (trueno-renacer-v1 schema)");
    println!("  html  - Self-contained HTML report with charts");
    println!();

    println!("Done! See book/src/ublk/visualization.md for full documentation.");
}
