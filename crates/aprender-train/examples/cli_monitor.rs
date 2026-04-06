//! Monitor Example
//!
//! Demonstrates programmatic drift detection using Population Stability Index (PSI).
//!
//! # Usage
//!
//! ```bash
//! cargo run --example cli_monitor
//! ```
//!
//! Or use the CLI:
//! ```bash
//! entrenar monitor data/current.parquet --threshold 0.2
//! entrenar monitor data/current.parquet --baseline data/baseline.parquet
//! ```

fn main() {
    println!("Drift Monitoring Example");
    println!("========================\n");

    // Example 1: No drift scenario
    println!("Scenario 1: Stable Distribution");
    println!("--------------------------------");
    let baseline_stable = vec![0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05];
    let current_stable = vec![0.11, 0.14, 0.19, 0.26, 0.16, 0.09, 0.05];
    check_drift(&baseline_stable, &current_stable, 0.2);

    println!();

    // Example 2: Minor drift scenario
    println!("Scenario 2: Minor Drift");
    println!("-----------------------");
    let current_minor = vec![0.15, 0.12, 0.18, 0.22, 0.18, 0.10, 0.05];
    check_drift(&baseline_stable, &current_minor, 0.2);

    println!();

    // Example 3: Significant drift scenario
    println!("Scenario 3: Significant Drift");
    println!("-----------------------------");
    let current_major = vec![0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05];
    check_drift(&baseline_stable, &current_major, 0.2);

    println!();

    // Example 4: Custom threshold
    println!("Scenario 4: Strict Threshold (0.1)");
    println!("----------------------------------");
    check_drift(&baseline_stable, &current_minor, 0.1);
}

/// Calculate PSI and check for drift
fn check_drift(baseline: &[f64], current: &[f64], threshold: f64) {
    let psi = calculate_psi(baseline, current);

    // Determine severity
    let (status, severity) = if psi < 0.1 {
        ("NO DRIFT", "low")
    } else if psi < threshold {
        ("MINOR DRIFT", "moderate")
    } else {
        ("SIGNIFICANT DRIFT", "high")
    };

    let pass = psi < threshold;

    println!("  PSI score: {psi:.4}");
    println!("  Threshold: {threshold:.4}");
    println!("  Severity: {severity}");
    println!("  Status: {} {}", status, if pass { "✓" } else { "✗" });

    if !pass {
        println!("  ACTION REQUIRED: Investigate data distribution shift");
    }
}

/// Calculate Population Stability Index (PSI)
///
/// PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
///
/// Interpretation:
/// - PSI < 0.1: No significant shift
/// - PSI 0.1 - 0.2: Moderate shift
/// - PSI > 0.2: Significant shift
fn calculate_psi(baseline: &[f64], current: &[f64]) -> f64 {
    let mut psi = 0.0_f64;

    for (expected, actual) in baseline.iter().zip(current.iter()) {
        if *expected > 0.0 && *actual > 0.0 {
            psi += (*actual - *expected) * (*actual / *expected).ln();
        }
    }

    psi.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_distributions() {
        let dist = vec![0.2, 0.3, 0.3, 0.2];
        let psi = calculate_psi(&dist, &dist);
        assert!(psi < 0.001, "PSI should be ~0 for identical distributions");
    }

    #[test]
    fn test_different_distributions() {
        let baseline = vec![0.25, 0.25, 0.25, 0.25];
        let current = vec![0.40, 0.30, 0.20, 0.10];
        let psi = calculate_psi(&baseline, &current);
        assert!(psi > 0.1, "PSI should be >0.1 for different distributions");
    }

    #[test]
    fn test_psi_is_non_negative() {
        let baseline = vec![0.1, 0.2, 0.3, 0.4];
        let current = vec![0.4, 0.3, 0.2, 0.1];
        let psi = calculate_psi(&baseline, &current);
        assert!(psi >= 0.0, "PSI should always be non-negative");
    }
}
