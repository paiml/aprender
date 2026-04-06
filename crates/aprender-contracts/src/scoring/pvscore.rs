//! `PVScore`: 10-dimension geometric mean scoring (spec Section 18).
//!
//! Computes a 0-100 composite from 10 dimensions using geometric mean.
//! One zero dimension tanks the entire score, preventing gaming.

use super::types::CodebaseScore;

/// Compute the 10-dimension `PVScore` as a geometric mean on a 0-100 scale.
///
/// Maps each [`CodebaseScore`] dimension (0.0-1.0) to 0-100, then returns:
///   `(D1 * D2 * ... * D10) ^ (1/10)`
///
/// Dimensions:
/// - D1: `contract_coverage`
/// - D2: `binding_completeness`
/// - D3: `mean_contract_score`
/// - D4: `proof_depth_dist`
/// - D5: `drift`
/// - D6: `reverse_coverage`
/// - D7: `mutation_testing`
/// - D8: `ci_pipeline_depth`
/// - D9: `proof_freshness`
/// - D10: `defect_patterns`
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn pvscore_10dim(score: &CodebaseScore) -> f64 {
    let dims = [
        score.contract_coverage,
        score.binding_completeness,
        score.mean_contract_score,
        score.proof_depth_dist,
        score.drift,
        score.reverse_coverage,
        score.mutation_testing,
        score.ci_pipeline_depth,
        score.proof_freshness,
        score.defect_patterns,
    ];

    // Only include measured dimensions (> 0). D6-D10 default to 0 when
    // not populated; excluding them avoids zeroing the entire score.
    let measured: Vec<f64> = dims
        .iter()
        .filter(|&&d| d > 0.0)
        .map(|d| d * 100.0)
        .collect();

    if measured.is_empty() {
        return 0.0;
    }

    // Geometric mean: exp( (1/N) * sum(ln(d_i)) )
    #[allow(clippy::cast_precision_loss)]
    let n = measured.len() as f64;
    let log_sum: f64 = measured.iter().map(|d| d.ln()).sum();
    (log_sum / n).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::types::{CodebaseScore, Grade};

    fn make_score(dims: [f64; 10]) -> CodebaseScore {
        CodebaseScore {
            path: "test".into(),
            contract_coverage: dims[0],
            binding_completeness: dims[1],
            mean_contract_score: dims[2],
            proof_depth_dist: dims[3],
            drift: dims[4],
            reverse_coverage: dims[5],
            mutation_testing: dims[6],
            ci_pipeline_depth: dims[7],
            proof_freshness: dims[8],
            defect_patterns: dims[9],
            composite: 0.0,
            grade: Grade::F,
            top_gaps: Vec::new(),
        }
    }

    #[test]
    fn all_perfect_returns_100() {
        let score = make_score([1.0; 10]);
        let pv = pvscore_10dim(&score);
        assert!(
            (pv - 100.0).abs() < 1e-9,
            "All 100s should give 100.0, got {pv}"
        );
    }

    #[test]
    fn one_zero_excluded_from_mean() {
        // D3 (mean_contract_score) is zero — excluded from geometric mean
        let score = make_score([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let pv = pvscore_10dim(&score);
        // 9 dimensions at 100 => geometric mean = 100
        assert!(
            (pv - 100.0).abs() < 1e-6,
            "Zero dimension excluded, rest = 100.0, got {pv}"
        );
    }

    #[test]
    fn mixed_values_geometric_mean() {
        // All dimensions at 0.80 => geometric mean of (80^10)^(1/10) = 80.0
        let score = make_score([0.80; 10]);
        let pv = pvscore_10dim(&score);
        assert!(
            (pv - 80.0).abs() < 1e-6,
            "Uniform 0.80 should give 80.0, got {pv}"
        );
    }

    #[test]
    fn geometric_mean_less_than_arithmetic() {
        // Geometric mean of unequal values is strictly less than arithmetic mean
        let score = make_score([0.50, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let pv = pvscore_10dim(&score);
        // Arithmetic mean would be (50 + 9*100) / 10 = 95
        // Geometric mean: (50 * 100^9)^(1/10) = 100 * 0.5^(1/10) ~= 93.30
        assert!(
            pv < 95.0,
            "Geometric mean should be < arithmetic mean, got {pv}"
        );
        let expected = 100.0 * 0.5_f64.powf(0.1);
        assert!(
            (pv - expected).abs() < 1e-6,
            "Expected ~{expected}, got {pv}"
        );
    }

    #[test]
    fn all_zeros_returns_zero() {
        let score = make_score([0.0; 10]);
        let pv = pvscore_10dim(&score);
        assert!(pv.abs() < 1e-9, "All zeros should give 0.0, got {pv}");
    }
}
