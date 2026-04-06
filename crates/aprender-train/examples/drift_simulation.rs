//! Drift Detection Simulation Example (APR-073)
//!
//! Demonstrates drift detection with KS test and PSI.
//! Shows how to set up baseline, detect drift, and use callbacks.
//!
//! Run with: cargo run --example drift_simulation

use entrenar::eval::{DriftDetector, DriftTest, Severity};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn main() {
    println!("=== Drift Detection Simulation ===\n");

    // 1. Create detector with KS and PSI tests
    let mut detector = DriftDetector::new(vec![
        DriftTest::KS { threshold: 0.05 },
        DriftTest::PSI { threshold: 0.1 },
    ]);

    // 2. Register Andon callback (triggered when drift detected)
    let drift_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&drift_count);

    detector.on_drift(move |results| {
        count_clone.fetch_add(1, Ordering::SeqCst);
        println!("ANDON ALERT: Drift detected!");
        for r in results.iter().filter(|r| r.drifted) {
            println!(
                "  - {} ({}): statistic={:.4}, severity={:?}",
                r.feature,
                r.test.name(),
                r.statistic,
                r.severity
            );
        }
    });

    // 3. Generate baseline data (training distribution)
    // Feature 1: Normal distribution centered at 50
    // Feature 2: Normal distribution centered at 100
    println!("Generating baseline data (1000 samples, 2 features)...");
    let baseline = generate_data(1000, 50.0, 100.0, 10.0, 42);
    detector.set_baseline(&baseline);
    println!("Baseline set.\n");

    // 4. Test with same distribution (no drift expected)
    println!("--- Test 1: Same Distribution (No Drift Expected) ---");
    let same_dist = generate_data(500, 50.0, 100.0, 10.0, 123);
    let results = detector.check_and_trigger(&same_dist);

    let drifted: Vec<_> = results.iter().filter(|r| r.drifted).collect();
    if drifted.is_empty() {
        println!("Result: No drift detected (as expected)");
    } else {
        println!("Result: Unexpected drift detected in {} features", drifted.len());
    }
    println!();

    // 5. Test with shifted distribution (drift expected)
    println!("--- Test 2: Shifted Distribution (Drift Expected) ---");
    println!("Shifting feature 1 mean from 50 to 80 (+3 std devs)");
    let shifted_dist = generate_data(500, 80.0, 100.0, 10.0, 456);
    let results = detector.check_and_trigger(&shifted_dist);

    let drifted: Vec<_> = results.iter().filter(|r| r.drifted).collect();
    if drifted.is_empty() {
        println!("Result: No drift detected (unexpected!)");
    } else {
        println!("Result: Drift detected in {} tests", drifted.len());
        for r in &drifted {
            let severity_str = match r.severity {
                Severity::None => "none",
                Severity::Warning => "WARNING",
                Severity::Critical => "CRITICAL",
            };
            println!(
                "  {} - {}: statistic={:.4}, severity={}",
                r.feature,
                r.test.name(),
                r.statistic,
                severity_str
            );
        }
    }
    println!();

    // 6. Test with completely different distribution
    println!("--- Test 3: Completely Different Distribution ---");
    println!("Both features shifted significantly");
    let different_dist = generate_data(500, 150.0, 200.0, 10.0, 789);
    let results = detector.check_and_trigger(&different_dist);

    let summary = DriftDetector::summary(&results);
    println!("Summary:");
    println!("  Total features checked: {}", summary.total_features);
    println!("  Features with drift: {}", summary.drifted_features);
    println!("  Critical alerts: {}", summary.critical);
    println!("  Warnings: {}", summary.warnings);
    println!("  Drift percentage: {:.1}%", summary.drift_percentage());
    println!();

    // 7. Report callback invocations
    println!("=== Summary ===");
    println!("Andon callback was triggered {} time(s)", drift_count.load(Ordering::SeqCst));
}

/// Generate synthetic data with 2 features
fn generate_data(n: usize, mean1: f64, mean2: f64, std: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut data = Vec::with_capacity(n);
    let mut state = seed;

    for _ in 0..n {
        // Simple LCG random number generator
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (state >> 33) as f64 / f64::from(u32::MAX);

        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (state >> 33) as f64 / f64::from(u32::MAX);

        // Box-Muller transform for normal distribution
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();

        data.push(vec![mean1 + z1 * std, mean2 + z2 * std]);
    }

    data
}
