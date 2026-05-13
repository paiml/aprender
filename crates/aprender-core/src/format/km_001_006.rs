// SHIP-TWO-001 — `kmeans-kernel-v1` algorithm-level PARTIAL discharge
// for FALSIFY-KM-001..006.
//
// Contract: `contracts/kmeans-kernel-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Six K-Means (Lloyd 1982) gates:
//
// - KM-001 (nearest assignment): c_i = argmin_j ||x_i - mu_j||².
// - KM-002 (monotone convergence): J_{t+1} ≤ J_t.
// - KM-003 (objective non-negative): J ≥ 0.
// - KM-004 (valid indices): c_i ∈ [0, K-1].
// - KM-005 (SIMD parity): scalar argmin == SIMD argmin (exact).
// - KM-006 (K=1 boundary): all points → cluster 0, centroid = mean(x).
//
// In-module reference: `kmeans_assign`, `kmeans_update`,
// `kmeans_objective`, `squared_distance`.

/// Tolerance for monotone-decrease check (FP32 wobble).
pub const AC_KM_002_MONOTONE_EPS: f32 = 1e-5;

/// SIMD parity is exact (argmin must agree exactly with scalar).
pub const AC_KM_005_SIMD_EXACT_MATCH: bool = true;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KmVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference K-Means.
// -----------------------------------------------------------------------------

/// Squared L2 distance between two d-dim vectors.
#[must_use]
pub fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// Assign each point in `points` (row-major [N, d]) to its nearest
/// centroid in `centroids` (row-major [K, d]).
#[must_use]
pub fn kmeans_assign(points: &[f32], centroids: &[f32], n: usize, k: usize, d: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let xi = &points[i * d..(i + 1) * d];
        let mut best_k = 0_usize;
        let mut best_d = f32::INFINITY;
        for j in 0..k {
            let mu_j = &centroids[j * d..(j + 1) * d];
            let dist = squared_distance(xi, mu_j);
            if dist < best_d {
                best_d = dist;
                best_k = j;
            }
        }
        out.push(best_k);
    }
    out
}

/// Compute new centroids as cluster means. Empty-cluster centroids
/// remain at their previous value via `prev_centroids`.
#[must_use]
pub fn kmeans_update(
    points: &[f32],
    assignments: &[usize],
    prev_centroids: &[f32],
    n: usize,
    k: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; k * d];
    let mut counts = vec![0_usize; k];
    for i in 0..n {
        let c = assignments[i];
        if c < k {
            for j in 0..d {
                out[c * d + j] += points[i * d + j];
            }
            counts[c] += 1;
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..d {
                out[c * d + j] /= counts[c] as f32;
            }
        } else {
            // Preserve previous centroid for empty cluster.
            for j in 0..d {
                out[c * d + j] = prev_centroids[c * d + j];
            }
        }
    }
    out
}

/// J = Σ_i ||x_i - μ_{c_i}||².
#[must_use]
pub fn kmeans_objective(
    points: &[f32],
    centroids: &[f32],
    assignments: &[usize],
    n: usize,
    d: usize,
) -> f32 {
    let mut j = 0.0_f32;
    for i in 0..n {
        let xi = &points[i * d..(i + 1) * d];
        let c = assignments[i];
        let mu_c = &centroids[c * d..(c + 1) * d];
        j += squared_distance(xi, mu_c);
    }
    j
}

// -----------------------------------------------------------------------------
// Verdict 1: KM-001 — nearest centroid assignment.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_nearest_assignment(
    points: &[f32],
    centroids: &[f32],
    assignments: &[usize],
    n: usize,
    k: usize,
    d: usize,
) -> KmVerdict {
    if k == 0 || n == 0 || d == 0 {
        return KmVerdict::Fail;
    }
    if points.len() != n * d || centroids.len() != k * d || assignments.len() != n {
        return KmVerdict::Fail;
    }
    for i in 0..n {
        let xi = &points[i * d..(i + 1) * d];
        let c_i = assignments[i];
        if c_i >= k {
            return KmVerdict::Fail;
        }
        let dist_assigned = squared_distance(xi, &centroids[c_i * d..(c_i + 1) * d]);
        for j in 0..k {
            let dist_other = squared_distance(xi, &centroids[j * d..(j + 1) * d]);
            if dist_other < dist_assigned - AC_KM_002_MONOTONE_EPS {
                return KmVerdict::Fail;
            }
        }
    }
    KmVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: KM-002 — monotone convergence.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_monotone_convergence(j_values: &[f32]) -> KmVerdict {
    if j_values.is_empty() {
        return KmVerdict::Fail;
    }
    for w in j_values.windows(2) {
        let j_t = w[0];
        let j_tp1 = w[1];
        if !j_t.is_finite() || !j_tp1.is_finite() {
            return KmVerdict::Fail;
        }
        if j_tp1 > j_t + AC_KM_002_MONOTONE_EPS {
            return KmVerdict::Fail;
        }
    }
    KmVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 3: KM-003 — objective non-negative.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_objective_nonneg(j: f32) -> KmVerdict {
    if !j.is_finite() {
        return KmVerdict::Fail;
    }
    if j >= 0.0 {
        KmVerdict::Pass
    } else {
        KmVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: KM-004 — valid cluster indices.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_valid_indices(assignments: &[usize], k: usize) -> KmVerdict {
    if k == 0 {
        return KmVerdict::Fail;
    }
    for &c in assignments {
        if c >= k {
            return KmVerdict::Fail;
        }
    }
    KmVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 5: KM-005 — SIMD parity (exact argmin match).
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_simd_parity(scalar: &[usize], simd: &[usize]) -> KmVerdict {
    if scalar.len() != simd.len() {
        return KmVerdict::Fail;
    }
    if scalar == simd {
        KmVerdict::Pass
    } else {
        KmVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 6: KM-006 — K=1 boundary.
// -----------------------------------------------------------------------------

/// Pass iff `K == 1`, all assignments are 0, and the centroid equals
/// the elementwise mean of all points.
#[must_use]
pub fn verdict_from_k_eq_1_boundary(
    points: &[f32],
    centroid: &[f32],
    assignments: &[usize],
    n: usize,
    d: usize,
) -> KmVerdict {
    if n == 0 || d == 0 {
        return KmVerdict::Fail;
    }
    if points.len() != n * d || centroid.len() != d {
        return KmVerdict::Fail;
    }
    if assignments.len() != n {
        return KmVerdict::Fail;
    }
    if !assignments.iter().all(|&c| c == 0) {
        return KmVerdict::Fail;
    }
    // centroid must be elementwise mean of all n points.
    let mut sum = vec![0.0_f32; d];
    for i in 0..n {
        for j in 0..d {
            sum[j] += points[i * d + j];
        }
    }
    for j in 0..d {
        let mean = sum[j] / n as f32;
        if (centroid[j] - mean).abs() > 1e-5 {
            return KmVerdict::Fail;
        }
    }
    KmVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_monotone_eps_1e_5() {
        assert_eq!(AC_KM_002_MONOTONE_EPS, 1e-5);
    }

    #[test]
    fn provenance_simd_exact_match() {
        assert!(AC_KM_005_SIMD_EXACT_MATCH);
    }

    // -------------------------------------------------------------------------
    // Section 2: KM-001 — nearest assignment.
    // -------------------------------------------------------------------------
    #[test]
    fn km001_pass_two_clusters_correctly_assigned() {
        // 4 points, 2 clusters (K=2), d=2.
        // Points near (0,0): rows 0, 1 ⇒ cluster 0.
        // Points near (10,10): rows 2, 3 ⇒ cluster 1.
        let points = vec![0.0_f32, 0.0, 0.5, 0.0, 10.0, 10.0, 9.5, 10.5];
        let centroids = vec![0.0_f32, 0.0, 10.0, 10.0];
        let assignments = vec![0_usize, 0, 1, 1];
        assert_eq!(
            verdict_from_nearest_assignment(&points, &centroids, &assignments, 4, 2, 2),
            KmVerdict::Pass
        );
    }

    #[test]
    fn km001_pass_using_reference_assign() {
        let points = vec![0.0_f32, 1.0, 2.0, 10.0, 11.0, 12.0];
        let centroids = vec![1.0_f32, 11.0]; // 1D
        let assignments = kmeans_assign(&points, &centroids, 6, 2, 1);
        assert_eq!(
            verdict_from_nearest_assignment(&points, &centroids, &assignments, 6, 2, 1),
            KmVerdict::Pass
        );
    }

    #[test]
    fn km001_fail_assigned_to_far_centroid() {
        // Point near (0,0) but assigned to cluster 1 (far).
        let points = vec![0.0_f32, 0.0];
        let centroids = vec![0.0_f32, 0.0, 10.0, 10.0];
        let assignments = vec![1_usize]; // wrong: 0 is closer
        assert_eq!(
            verdict_from_nearest_assignment(&points, &centroids, &assignments, 1, 2, 2),
            KmVerdict::Fail
        );
    }

    #[test]
    fn km001_fail_index_out_of_range() {
        let points = vec![0.0_f32];
        let centroids = vec![0.0_f32];
        let assignments = vec![5_usize]; // K=1 but assignment=5
        assert_eq!(
            verdict_from_nearest_assignment(&points, &centroids, &assignments, 1, 1, 1),
            KmVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: KM-002 — monotone convergence.
    // -------------------------------------------------------------------------
    #[test]
    fn km002_pass_decreasing_objective() {
        let j = vec![100.0_f32, 50.0, 25.0, 10.0, 5.0];
        assert_eq!(verdict_from_monotone_convergence(&j), KmVerdict::Pass);
    }

    #[test]
    fn km002_pass_constant_after_convergence() {
        let j = vec![100.0_f32, 50.0, 25.0, 25.0, 25.0];
        assert_eq!(verdict_from_monotone_convergence(&j), KmVerdict::Pass);
    }

    #[test]
    fn km002_fail_increasing_objective() {
        let j = vec![10.0_f32, 20.0, 5.0];
        assert_eq!(verdict_from_monotone_convergence(&j), KmVerdict::Fail);
    }

    #[test]
    fn km002_fail_one_jump() {
        let j = vec![100.0_f32, 50.0, 60.0, 25.0]; // 50 → 60 ascent
        assert_eq!(verdict_from_monotone_convergence(&j), KmVerdict::Fail);
    }

    #[test]
    fn km002_fail_nan() {
        let j = vec![100.0_f32, f32::NAN];
        assert_eq!(verdict_from_monotone_convergence(&j), KmVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: KM-003 — objective non-negative.
    // -------------------------------------------------------------------------
    #[test]
    fn km003_pass_zero() {
        assert_eq!(verdict_from_objective_nonneg(0.0), KmVerdict::Pass);
    }

    #[test]
    fn km003_pass_positive() {
        assert_eq!(verdict_from_objective_nonneg(125.5), KmVerdict::Pass);
    }

    #[test]
    fn km003_fail_negative() {
        // Sign-error bug: J should never be < 0 since it's sum of squares.
        assert_eq!(verdict_from_objective_nonneg(-0.001), KmVerdict::Fail);
    }

    #[test]
    fn km003_fail_nan() {
        assert_eq!(verdict_from_objective_nonneg(f32::NAN), KmVerdict::Fail);
    }

    #[test]
    fn km003_fail_inf() {
        assert_eq!(verdict_from_objective_nonneg(f32::INFINITY), KmVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: KM-004 — valid indices.
    // -------------------------------------------------------------------------
    #[test]
    fn km004_pass_all_in_range() {
        let assignments = vec![0_usize, 1, 2, 0, 1, 2];
        assert_eq!(verdict_from_valid_indices(&assignments, 3), KmVerdict::Pass);
    }

    #[test]
    fn km004_pass_empty() {
        let assignments: Vec<usize> = vec![];
        assert_eq!(verdict_from_valid_indices(&assignments, 3), KmVerdict::Pass);
    }

    #[test]
    fn km004_fail_one_out_of_range() {
        let assignments = vec![0_usize, 1, 5]; // 5 >= K=3
        assert_eq!(verdict_from_valid_indices(&assignments, 3), KmVerdict::Fail);
    }

    #[test]
    fn km004_fail_zero_k() {
        let assignments = vec![0_usize];
        assert_eq!(verdict_from_valid_indices(&assignments, 0), KmVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: KM-005 — SIMD parity.
    // -------------------------------------------------------------------------
    #[test]
    fn km005_pass_identical_assignments() {
        let scalar = vec![0_usize, 1, 2, 0];
        let simd = vec![0_usize, 1, 2, 0];
        assert_eq!(verdict_from_simd_parity(&scalar, &simd), KmVerdict::Pass);
    }

    #[test]
    fn km005_pass_empty() {
        let v: Vec<usize> = vec![];
        assert_eq!(verdict_from_simd_parity(&v, &v), KmVerdict::Pass);
    }

    #[test]
    fn km005_fail_one_off() {
        let scalar = vec![0_usize, 1, 2];
        let simd = vec![0_usize, 1, 1]; // last differs
        assert_eq!(verdict_from_simd_parity(&scalar, &simd), KmVerdict::Fail);
    }

    #[test]
    fn km005_fail_length_mismatch() {
        let scalar = vec![0_usize, 1, 2];
        let simd = vec![0_usize, 1];
        assert_eq!(verdict_from_simd_parity(&scalar, &simd), KmVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: KM-006 — K=1 boundary.
    // -------------------------------------------------------------------------
    #[test]
    fn km006_pass_all_zero_assignments_centroid_at_mean() {
        // 4 points in 2D: mean = (3, 3).
        let points = vec![1.0_f32, 1.0, 5.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let centroid = vec![3.0_f32, 3.0];
        let assignments = vec![0_usize; 4];
        assert_eq!(
            verdict_from_k_eq_1_boundary(&points, &centroid, &assignments, 4, 2),
            KmVerdict::Pass
        );
    }

    #[test]
    fn km006_pass_single_point() {
        let points = vec![5.0_f32, 5.0];
        let centroid = vec![5.0_f32, 5.0];
        let assignments = vec![0_usize];
        assert_eq!(
            verdict_from_k_eq_1_boundary(&points, &centroid, &assignments, 1, 2),
            KmVerdict::Pass
        );
    }

    #[test]
    fn km006_fail_nonzero_assignment() {
        let points = vec![1.0_f32, 1.0];
        let centroid = vec![1.0_f32, 1.0];
        let assignments = vec![1_usize]; // K=1 but assigned to 1
        assert_eq!(
            verdict_from_k_eq_1_boundary(&points, &centroid, &assignments, 1, 2),
            KmVerdict::Fail
        );
    }

    #[test]
    fn km006_fail_centroid_not_mean() {
        let points = vec![1.0_f32, 1.0, 5.0, 5.0];
        let centroid = vec![10.0_f32, 10.0]; // not mean = (3, 3)
        let assignments = vec![0_usize; 2];
        assert_eq!(
            verdict_from_k_eq_1_boundary(&points, &centroid, &assignments, 2, 2),
            KmVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: Domain — reference functions.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_squared_distance_zero_at_self() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!((squared_distance(&v, &v) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn domain_kmeans_objective_zero_at_centroid() {
        // All points coincide with single centroid.
        let points = vec![5.0_f32, 5.0, 5.0, 5.0];
        let centroids = vec![5.0_f32, 5.0];
        let assignments = vec![0_usize, 0];
        let j = kmeans_objective(&points, &centroids, &assignments, 2, 2);
        assert!(j.abs() < 1e-6);
    }

    #[test]
    fn domain_kmeans_update_recovers_mean() {
        // 2 points: (1, 1), (3, 3); 1 cluster ⇒ centroid = (2, 2).
        let points = vec![1.0_f32, 1.0, 3.0, 3.0];
        let prev = vec![0.0_f32, 0.0];
        let assignments = vec![0_usize, 0];
        let new = kmeans_update(&points, &assignments, &prev, 2, 1, 2);
        assert!((new[0] - 2.0).abs() < 1e-6);
        assert!((new[1] - 2.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Section 9: Sweep — Lloyd's iteration.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_lloyd_iterations_monotone() {
        // 6 points, K=2, d=1. Initial centroids far from optimum.
        let points = vec![0.0_f32, 1.0, 2.0, 10.0, 11.0, 12.0];
        let mut centroids = vec![5.0_f32, 8.0];
        let mut j_trace = Vec::new();

        for _ in 0..10 {
            let assignments = kmeans_assign(&points, &centroids, 6, 2, 1);
            let j = kmeans_objective(&points, &centroids, &assignments, 6, 1);
            j_trace.push(j);

            let prev = centroids.clone();
            centroids = kmeans_update(&points, &assignments, &prev, 6, 2, 1);
        }
        assert_eq!(verdict_from_monotone_convergence(&j_trace), KmVerdict::Pass);

        // Final centroids should be ~1 and ~11.
        assert!((centroids[0] - 1.0).abs() < 0.5);
        assert!((centroids[1] - 11.0).abs() < 0.5);
    }

    // -------------------------------------------------------------------------
    // Section 10: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_argmin_bug_caught() {
        // KM-001 if_fails: "Distance comparison or argmin
        // implementation incorrect" — point assigned to far centroid.
        let points = vec![0.0_f32, 0.0];
        let centroids = vec![1.0_f32, 1.0, 100.0, 100.0];
        let assignments = vec![1_usize]; // bug: should be 0
        assert_eq!(
            verdict_from_nearest_assignment(&points, &centroids, &assignments, 1, 2, 2),
            KmVerdict::Fail
        );
    }

    #[test]
    fn realistic_centroid_update_bug_caught() {
        // KM-002 if_fails: "Centroid update not correctly computing mean".
        // Simulated: J jumps up after an update.
        let j = vec![25.0_f32, 30.0]; // ascent
        assert_eq!(verdict_from_monotone_convergence(&j), KmVerdict::Fail);
    }

    #[test]
    fn realistic_sign_error_caught() {
        // KM-003 if_fails: "Squared distance computation has sign error".
        assert_eq!(verdict_from_objective_nonneg(-1.5), KmVerdict::Fail);
    }

    #[test]
    fn realistic_oob_index_caught() {
        // KM-004 if_fails: "Out-of-bounds cluster index".
        let assignments = vec![0_usize, 999]; // 999 >> K=2
        assert_eq!(verdict_from_valid_indices(&assignments, 2), KmVerdict::Fail);
    }

    #[test]
    fn realistic_simd_diverges_caught() {
        // KM-005 if_fails: "SIMD distance computation differs".
        let scalar = vec![0_usize, 1, 0, 1];
        let simd = vec![1_usize, 0, 1, 0]; // bit flipped
        assert_eq!(verdict_from_simd_parity(&scalar, &simd), KmVerdict::Fail);
    }

    #[test]
    fn realistic_full_kmeans_pipeline() {
        // 8 points in 2D, K=2 clusters at (0,0) and (10,10).
        let points = vec![
            0.0_f32, 0.0,
            0.5, 0.0,
            -0.5, 0.5,
            0.0, -0.5,
            10.0, 10.0,
            10.5, 10.0,
            9.5, 10.5,
            10.0, 9.5,
        ];
        let mut centroids = vec![1.0_f32, 1.0, 9.0, 9.0]; // initial guess
        let mut j_trace = Vec::new();

        for _ in 0..20 {
            let assignments = kmeans_assign(&points, &centroids, 8, 2, 2);
            let j = kmeans_objective(&points, &centroids, &assignments, 8, 2);
            j_trace.push(j);

            // Verify all 4 algorithm-level invariants per iteration:
            assert_eq!(
                verdict_from_nearest_assignment(&points, &centroids, &assignments, 8, 2, 2),
                KmVerdict::Pass
            );
            assert_eq!(verdict_from_objective_nonneg(j), KmVerdict::Pass);
            assert_eq!(verdict_from_valid_indices(&assignments, 2), KmVerdict::Pass);

            let prev = centroids.clone();
            centroids = kmeans_update(&points, &assignments, &prev, 8, 2, 2);
        }

        assert_eq!(verdict_from_monotone_convergence(&j_trace), KmVerdict::Pass);

        // Centroids should converge near (0, 0) and (10, 10).
        assert!(centroids[0].abs() < 0.5 && centroids[1].abs() < 0.5);
        assert!((centroids[2] - 10.0).abs() < 0.5 && (centroids[3] - 10.0).abs() < 0.5);
    }
}
