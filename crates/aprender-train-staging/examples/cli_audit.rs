//! Audit Example
//!
//! Demonstrates programmatic bias detection and fairness auditing.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example cli_audit
//! ```
//!
//! Or use the CLI:
//! ```bash
//! entrenar audit predictions.parquet --type bias --threshold 0.8
//! entrenar audit model.safetensors --type security
//! ```

fn main() {
    println!("Bias & Fairness Audit Example");
    println!("==============================\n");

    // Example: Bias audit
    bias_audit(0.8);

    println!();

    // Example: Stricter threshold
    bias_audit(0.95);

    println!();

    // Example: Security audit
    security_audit();

    println!();

    // Example: Privacy audit
    privacy_audit();
}

fn bias_audit(threshold: f64) {
    println!("Bias Audit (threshold: {threshold:.2})");
    println!("----------------------------");

    // Simulate group statistics
    // In real usage, these would be computed from predictions + protected attributes
    let group_a_positive_rate = 0.72_f64;
    let group_b_positive_rate = 0.78_f64;

    // Demographic Parity Ratio
    // DPR = min(P(Y=1|A=0), P(Y=1|A=1)) / max(P(Y=1|A=0), P(Y=1|A=1))
    let demographic_parity = (group_a_positive_rate / group_b_positive_rate)
        .min(group_b_positive_rate / group_a_positive_rate);

    // Equalized Odds: TPR should be similar across groups
    let group_a_tpr = 0.85_f64;
    let group_b_tpr = 0.82_f64;
    let equalized_odds = 1.0 - (group_a_tpr - group_b_tpr).abs();

    let pass = demographic_parity >= threshold;

    println!("  Group A positive rate: {group_a_positive_rate:.3}");
    println!("  Group B positive rate: {group_b_positive_rate:.3}");
    println!("  Demographic parity ratio: {demographic_parity:.3}");
    println!("  Equalized odds: {equalized_odds:.3}");
    println!("  Threshold: {threshold:.3}");
    println!("  Status: {}", if pass { "PASS ✓" } else { "FAIL ✗" });
}

fn security_audit() {
    println!("Security Audit");
    println!("--------------");

    // Check for common security issues
    println!("  Pickle deserialization: Safe (SafeTensors format)");
    println!("  Code execution vectors: None detected");
    println!("  Untrusted model sources: Not checked (local file)");
    println!("  Status: PASS ✓");
}

fn privacy_audit() {
    println!("Privacy Audit");
    println!("-------------");

    // Check for PII patterns
    println!("  Email pattern scan: 0 found");
    println!("  Phone pattern scan: 0 found");
    println!("  SSN pattern scan: 0 found");
    println!("  Credit card pattern scan: 0 found");
    println!("  Status: PASS ✓");
}
