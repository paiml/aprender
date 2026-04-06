//! Code generation evaluation metrics
//!
//! Provides pass@k — the unbiased estimator for functional correctness
//! of code generation models (Chen et al., 2021 "Evaluating Large Language
//! Models Trained on Code").

/// Compute pass@k: unbiased estimator of functional correctness.
///
/// Formula: `1 - C(n-c, k) / C(n, k)`
///
/// where n = total samples, c = correct samples, k = top-k threshold.
///
/// Returns a value in [0, 1] where 1.0 means all k samples pass.
///
/// # Arguments
/// * `n` - Total number of generated code samples
/// * `c` - Number of correct (passing) samples
/// * `k` - Number of samples to consider (typically 1, 10, or 100)
///
/// # Edge Cases
/// * If `k > n`, returns `if c > 0 { 1.0 } else { 0.0 }`
/// * If `c >= n`, returns 1.0
/// * If `c == 0`, returns 0.0
pub fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if c == 0 {
        return 0.0;
    }
    if c >= n || k > n {
        return 1.0;
    }

    // 1 - C(n-c, k) / C(n, k)
    // Compute in log space to avoid overflow for large n
    // C(n-c, k) / C(n, k) = product_{i=0..k} (n-c-i) / (n-i)
    let mut log_ratio = 0.0f64;
    for i in 0..k {
        let numerator = (n - c - i) as f64;
        let denominator = (n - i) as f64;
        if numerator <= 0.0 {
            return 1.0;
        }
        log_ratio +=
            numerator.max(f64::MIN_POSITIVE).ln() - denominator.max(f64::MIN_POSITIVE).ln();
    }

    1.0 - log_ratio.exp()
}
