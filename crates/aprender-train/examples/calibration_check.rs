//! P-Value Calibration Check Example (APR-073, Section 10.5)
//!
//! Verifies that KS and PSI tests have proper p-value calibration:
//! Under the null hypothesis (no drift), p-values should be uniformly distributed.
//!
//! This demonstrates statistical power and correctness of drift detection.
//!
//! Run with: cargo run --example calibration_check

use entrenar::eval::{DriftDetector, DriftTest};

const NUM_TRIALS: usize = 1000;
const SAMPLE_SIZE: usize = 500;
const SIGNIFICANCE_LEVEL: f64 = 0.05;

fn main() {
    println!("=== P-Value Calibration Check ===\n");
    println!("Testing drift detection statistical calibration under null hypothesis.\n");
    println!("Under H0 (no drift), p-values should be uniformly distributed on [0, 1].");
    println!("This means ~5% of tests should reject at alpha=0.05.\n");

    // Run calibration checks for KS test
    println!("--- Kolmogorov-Smirnov Test Calibration ---");
    let ks_rejection_rate = run_calibration_check(DriftTest::KS { threshold: SIGNIFICANCE_LEVEL });
    println!(
        "KS Test: {:.1}% rejection rate (expected: ~{:.1}%)",
        ks_rejection_rate * 100.0,
        SIGNIFICANCE_LEVEL * 100.0
    );
    let ks_calibrated = is_calibrated(ks_rejection_rate, SIGNIFICANCE_LEVEL);
    println!("Calibration: {}\n", if ks_calibrated { "PASS" } else { "FAIL" });

    // Run calibration checks for PSI test
    println!("--- Population Stability Index Calibration ---");
    let psi_rejection_rate = run_calibration_check(DriftTest::PSI { threshold: 0.1 });
    println!("PSI Test: {:.1}% rejection rate at threshold=0.1", psi_rejection_rate * 100.0);
    // PSI should have low rejection rate under null (same distribution)
    let psi_calibrated = psi_rejection_rate < 0.15; // Allow up to 15% for PSI
    println!(
        "Calibration: {} (rejection rate under 15%)\n",
        if psi_calibrated { "PASS" } else { "FAIL" }
    );

    // Test statistical power (ability to detect actual drift)
    println!("--- Statistical Power Test (Ability to Detect Drift) ---");
    let power = test_statistical_power();
    println!("Power at effect size d=1.0: {:.1}%", power * 100.0);
    let power_adequate = power > 0.80;
    println!(
        "Power: {} (>80%% required)\n",
        if power_adequate { "ADEQUATE" } else { "INSUFFICIENT" }
    );

    // Summary
    println!("=== Summary ===");
    let all_pass = ks_calibrated && psi_calibrated && power_adequate;
    if all_pass {
        println!("All calibration checks PASSED");
    } else {
        println!("Some calibration checks FAILED:");
        if !ks_calibrated {
            println!("  - KS test: p-value distribution not uniform");
        }
        if !psi_calibrated {
            println!("  - PSI test: excessive false positive rate");
        }
        if !power_adequate {
            println!("  - Statistical power: insufficient to detect drift");
        }
    }
}

/// Run calibration check for a given drift test
///
/// Generates pairs of samples from the SAME distribution and measures
/// the rejection rate. Under null hypothesis, this should equal alpha.
fn run_calibration_check(test: DriftTest) -> f64 {
    let mut rejections = 0;

    for trial in 0..NUM_TRIALS {
        // Generate two independent samples from same distribution
        let seed1 = (trial * 2) as u64;
        let seed2 = (trial * 2 + 1) as u64;

        let baseline = generate_normal_data(SAMPLE_SIZE, 50.0, 10.0, seed1);
        let current = generate_normal_data(SAMPLE_SIZE, 50.0, 10.0, seed2);

        // Run drift detection
        let mut detector = DriftDetector::new(vec![test]);
        detector.set_baseline(&baseline);
        let results = detector.check(&current);

        // Count rejections (false positives under null)
        if results.iter().any(|r| r.drifted) {
            rejections += 1;
        }
    }

    f64::from(rejections) / NUM_TRIALS as f64
}

/// Test statistical power - ability to detect actual drift
///
/// Generates samples with known drift and measures detection rate.
fn test_statistical_power() -> f64 {
    let mut detections = 0;

    for trial in 0..NUM_TRIALS {
        let seed1 = (trial * 2) as u64;
        let seed2 = (trial * 2 + 1) as u64;

        // Baseline: mean=50, std=10
        let baseline = generate_normal_data(SAMPLE_SIZE, 50.0, 10.0, seed1);
        // Current: mean=60 (shifted by 1 standard deviation)
        let current = generate_normal_data(SAMPLE_SIZE, 60.0, 10.0, seed2);

        let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);
        detector.set_baseline(&baseline);
        let results = detector.check(&current);

        if results.iter().any(|r| r.drifted) {
            detections += 1;
        }
    }

    f64::from(detections) / NUM_TRIALS as f64
}

/// Check if rejection rate is within acceptable bounds for calibration
fn is_calibrated(rejection_rate: f64, alpha: f64) -> bool {
    // Use a tolerance based on binomial standard error
    // SE = sqrt(alpha * (1 - alpha) / n)
    let se = (alpha * (1.0 - alpha) / NUM_TRIALS as f64).sqrt();
    let tolerance = 3.0 * se; // 3 sigma tolerance

    (rejection_rate - alpha).abs() < tolerance
}

/// Generate synthetic normal data
fn generate_normal_data(n: usize, mean: f64, std: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut data = Vec::with_capacity(n);
    let mut state = seed;

    for _ in 0..n {
        // LCG random number generator
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (state >> 33) as f64 / f64::from(u32::MAX);

        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (state >> 33) as f64 / f64::from(u32::MAX);

        // Box-Muller transform
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        data.push(vec![mean + z * std]);
    }

    data
}
