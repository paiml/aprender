//! Descriptive Statistics Example
//!
//! This example demonstrates statistical analysis on a dataset:
//! - Computing quantiles and percentiles
//! - Five-number summary for outlier detection
//! - Histogram generation with different binning methods
//! - Comparing binning strategies
//!
//! Run with: `cargo run --example descriptive_statistics`

use aprender::stats::{BinMethod, DescriptiveStats};
use trueno::Vector;

#[allow(clippy::too_many_lines)]
fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Descriptive Statistics with Aprender");
    println!("═══════════════════════════════════════════════════════\n");

    // Simulate test scores from a class of 30 students
    // Distribution: mostly 60-90 range, with a few outliers
    let test_scores = vec![
        45.0, // outlier (struggling student)
        52.0, // outlier
        62.0, 65.0, 68.0, 70.0, 72.0, 73.0, 75.0, 76.0, // lower cluster
        78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, // middle cluster
        86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, // upper cluster
        95.0, 97.0, 98.0,  // high performers
        100.0, // outlier (perfect score)
    ];

    println!("📊 Analyzing test scores from 30 students");
    println!("   Score range: 0-100\n");

    let data = Vector::from_slice(&test_scores);
    let stats = DescriptiveStats::new(&data);

    // ========================================================================
    // Basic Quantiles
    // ========================================================================
    println!("═══════════════════════════════════════════════════════");
    println!("  1. Quantiles and Percentiles");
    println!("═══════════════════════════════════════════════════════\n");

    let median = stats.quantile(0.5).expect("Failed to compute median");
    let q25 = stats.quantile(0.25).expect("Failed to compute Q1");
    let q75 = stats.quantile(0.75).expect("Failed to compute Q3");

    println!("Key Quantiles:");
    println!("  • 25th percentile (Q1): {q25:.1}");
    println!("  • 50th percentile (Median): {median:.1}");
    println!("  • 75th percentile (Q3): {q75:.1}");

    let percentiles = stats
        .percentiles(&[10.0, 25.0, 50.0, 75.0, 90.0])
        .expect("Failed to compute percentiles");
    println!("\nPercentile Distribution:");
    println!(
        "  • P10: {:.1} - Bottom 10% scored below this",
        percentiles[0]
    );
    println!("  • P25: {:.1} - Bottom quartile", percentiles[1]);
    println!("  • P50: {:.1} - Median score", percentiles[2]);
    println!("  • P75: {:.1} - Top quartile", percentiles[3]);
    println!("  • P90: {:.1} - Top 10% scored above this", percentiles[4]);

    println!("\n💡 Interpretation:");
    println!("   Half the class scored above {median:.1}");
    println!("   The middle 50% of students scored between {q25:.1} and {q75:.1}\n");

    // ========================================================================
    // Five-Number Summary
    // ========================================================================
    println!("═══════════════════════════════════════════════════════");
    println!("  2. Five-Number Summary (Outlier Detection)");
    println!("═══════════════════════════════════════════════════════\n");

    let summary = stats
        .five_number_summary()
        .expect("Failed to compute five-number summary");
    let iqr = stats.iqr().expect("Failed to compute IQR");

    println!("Five-Number Summary:");
    println!("  • Minimum: {:.1}", summary.min);
    println!("  • Q1 (25th percentile): {:.1}", summary.q1);
    println!("  • Median (50th percentile): {:.1}", summary.median);
    println!("  • Q3 (75th percentile): {:.1}", summary.q3);
    println!("  • Maximum: {:.1}", summary.max);
    println!("\n  • IQR (Q3 - Q1): {iqr:.1}");

    // Outlier detection using 1.5 * IQR rule
    let lower_fence = summary.q1 - 1.5 * iqr;
    let upper_fence = summary.q3 + 1.5 * iqr;
    println!("\nOutlier Fences (1.5 × IQR rule):");
    println!("  • Lower fence: {lower_fence:.1}");
    println!("  • Upper fence: {upper_fence:.1}");

    let outliers: Vec<f32> = test_scores
        .iter()
        .filter(|&&score| score < lower_fence || score > upper_fence)
        .copied()
        .collect();

    if outliers.is_empty() {
        println!("  • No outliers detected");
    } else {
        println!("  • {} outliers detected: {:?}", outliers.len(), outliers);
    }

    println!("\n💡 Interpretation:");
    println!("   IQR measures spread of middle 50% of data");
    println!("   Scores outside fences are potential outliers\n");

    // ========================================================================
    // Histogram Comparison
    // ========================================================================
    println!("═══════════════════════════════════════════════════════");
    println!("  3. Histogram Binning Methods");
    println!("═══════════════════════════════════════════════════════\n");

    let methods = vec![
        (BinMethod::FreedmanDiaconis, "Freedman-Diaconis"),
        (BinMethod::Sturges, "Sturges"),
        (BinMethod::Scott, "Scott"),
        (BinMethod::SquareRoot, "Square Root"),
    ];

    println!("Comparing binning strategies:\n");
    for (method, name) in methods {
        let hist = stats
            .histogram_method(method)
            .unwrap_or_else(|_| panic!("Failed to compute {name} histogram"));

        println!("📊 {name} Rule:");
        println!("   {} bins created", hist.bins.len());

        // Print histogram bars
        let max_count = *hist.counts.iter().max().unwrap_or(&0);
        for (i, (&lower, &count)) in hist.bins.iter().zip(hist.counts.iter()).enumerate() {
            let upper = if i < hist.bins.len() - 1 {
                hist.bins[i + 1]
            } else {
                100.0
            };

            let bar_length = if max_count > 0 {
                (count as f64 / max_count as f64 * 30.0) as usize
            } else {
                0
            };
            let bar = "█".repeat(bar_length);

            println!("   [{lower:5.1} - {upper:5.1}): {count:2} {bar}");
        }
        println!();
    }

    println!("💡 Binning Strategy Guide:");
    println!("   • Freedman-Diaconis: Best for outlier-resistant analysis");
    println!("   • Sturges: Fast, works well for normal distributions");
    println!("   • Scott: Minimizes integrated mean squared error");
    println!("   • Square Root: Simple, good for quick exploration\n");

    // ========================================================================
    // Summary Statistics
    // ========================================================================
    println!("═══════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════\n");

    let mean = data.mean().expect("Failed to compute mean");
    let stddev = data.stddev().expect("Failed to compute stddev");
    let min = data.min().expect("Failed to compute min");
    let max = data.max().expect("Failed to compute max");

    println!("Dataset Statistics:");
    println!("  • Sample size: {}", test_scores.len());
    println!("  • Mean: {mean:.2}");
    println!("  • Std Dev: {stddev:.2}");
    println!("  • Range: [{min:.1}, {max:.1}]");
    println!("  • Median: {median:.1}");
    println!("  • IQR: {iqr:.1}");

    println!("\nClass Performance:");
    let pass_count = test_scores.iter().filter(|&&score| score >= 60.0).count();
    let pass_rate = pass_count as f64 / test_scores.len() as f64 * 100.0;
    println!(
        "  • Pass rate (≥60): {:.1}% ({}/{})",
        pass_rate,
        pass_count,
        test_scores.len()
    );

    let a_count = test_scores.iter().filter(|&&score| score >= 90.0).count();
    let a_rate = a_count as f64 / test_scores.len() as f64 * 100.0;
    println!(
        "  • A grade rate (≥90): {:.1}% ({}/{})",
        a_rate,
        a_count,
        test_scores.len()
    );

    println!("\n🚀 Performance Notes:");
    println!("  • QuickSelect: O(n) for single quantile vs O(n log n) full sort");
    println!("  • R-7 method: Linear interpolation between closest ranks");
    println!("  • Single sort: Efficient for multiple percentiles at once");

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Example Complete!");
    println!("═══════════════════════════════════════════════════════");
}
