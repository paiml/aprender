//! Backend Equivalence Tests (CORE-006)
//!
//! Toyota Way: Jidoka (built-in quality)
//! Ensures GPU == SIMD == Scalar for all aggregation operations
//!
//! References:
//! - Section 7.3 of Sovereign AI spec: Backend equivalence tests
//! - Property-based testing: Claessen & Hughes (2000) `QuickCheck`
//!
//! ## Test Strategy
//!
//! 1. **Property-Based Tests**: Use proptest to generate random inputs
//! 2. **Backend Equivalence**: GPU results == SIMD results == Scalar results
//! 3. **Edge Cases**: NaN, infinity, overflow, empty inputs
//! 4. **CI Integration**: Tests fail on any backend mismatch
//!
//! ## Acceptance Criteria (from roadmap)
//!
//! - [x] Property-based tests with quickcheck/proptest
//! - [ ] Test all aggregations (sum, avg, count, min, max)
//! - [ ] Test edge cases (NaN, infinity, overflow)
//! - [ ] CI fails on any backend mismatch

use proptest::prelude::*;
use trueno::Vector;

/// Aggregation operation trait
///
/// All backends must implement this trait to ensure equivalence
trait Aggregation<T> {
    fn sum(&self, data: &[T]) -> T;
    fn avg(&self, data: &[T]) -> Option<f64>;
    fn count(&self, data: &[T]) -> usize;
    fn min(&self, data: &[T]) -> Option<T>;
    fn max(&self, data: &[T]) -> Option<T>;
}

/// GPU backend (mock implementation for testing)
struct GpuBackend;

/// SIMD backend (mock implementation for testing)
struct SimdBackend;

/// Scalar backend (reference implementation)
struct ScalarBackend;

// ============================================================================
// Scalar Backend (Reference Implementation)
// ============================================================================

impl Aggregation<i32> for ScalarBackend {
    fn sum(&self, data: &[i32]) -> i32 {
        data.iter().fold(0i32, |acc, &x| acc.wrapping_add(x))
    }

    fn avg(&self, data: &[i32]) -> Option<f64> {
        if data.is_empty() {
            None
        } else {
            let sum = data.iter().fold(0i32, |acc, &x| acc.wrapping_add(x));
            Some(f64::from(sum) / data.len() as f64)
        }
    }

    fn count(&self, data: &[i32]) -> usize {
        data.len()
    }

    fn min(&self, data: &[i32]) -> Option<i32> {
        data.iter().min().copied()
    }

    fn max(&self, data: &[i32]) -> Option<i32> {
        data.iter().max().copied()
    }
}

impl Aggregation<f32> for ScalarBackend {
    /// Scalar sum using Kahan (compensated) summation for numerical stability
    ///
    /// This ensures equivalence with SIMD backend which also uses Kahan summation.
    /// Critical for handling large numbers mixed with small numbers.
    fn sum(&self, data: &[f32]) -> f32 {
        // Kahan summation algorithm (compensated summation)
        let mut sum = 0.0_f32;
        let mut compensation = 0.0_f32;

        for &value in data {
            // Early exit for infinity/NaN to avoid compensation artifacts
            if value.is_infinite() || value.is_nan() {
                return data.iter().sum(); // Fallback to naive sum for special values
            }

            let y = value - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        sum
    }

    /// Scalar average using Kahan summation
    fn avg(&self, data: &[f32]) -> Option<f64> {
        if data.is_empty() {
            None
        } else {
            Some(f64::from(self.sum(data)) / data.len() as f64)
        }
    }

    fn count(&self, data: &[f32]) -> usize {
        data.len()
    }

    fn min(&self, data: &[f32]) -> Option<f32> {
        data.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).copied()
    }

    fn max(&self, data: &[f32]) -> Option<f32> {
        data.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).copied()
    }
}

// ============================================================================
// SIMD Backend (CORE-005: Trueno integration with auto-detect SIMD)
// ============================================================================

impl Aggregation<i32> for SimdBackend {
    fn sum(&self, data: &[i32]) -> i32 {
        // Note: i32 operations still use scalar until trueno adds i32 support
        ScalarBackend.sum(data)
    }

    fn avg(&self, data: &[i32]) -> Option<f64> {
        ScalarBackend.avg(data)
    }

    fn count(&self, data: &[i32]) -> usize {
        ScalarBackend.count(data)
    }

    fn min(&self, data: &[i32]) -> Option<i32> {
        ScalarBackend.min(data)
    }

    fn max(&self, data: &[i32]) -> Option<i32> {
        ScalarBackend.max(data)
    }
}

impl Aggregation<f32> for SimdBackend {
    /// SIMD-accelerated sum via trueno (AVX-512/AVX2/SSE2 auto-detect)
    /// Uses Kahan (compensated) summation for numerical stability
    fn sum(&self, data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        // Check for infinity/NaN - use regular sum for special values
        // (trueno's sum_kahan has issues with infinity)
        if data.iter().any(|x| x.is_infinite() || x.is_nan()) {
            return Vector::from_slice(data).sum().unwrap_or(0.0);
        }

        // Toyota Way: Heijunka - trueno uses SIMD but doesn't block
        // Uses Kahan summation for equivalence with scalar backend
        Vector::from_slice(data).sum_kahan().unwrap_or(0.0)
    }

    /// SIMD-accelerated average with Kahan summation
    fn avg(&self, data: &[f32]) -> Option<f64> {
        if data.is_empty() {
            None
        } else {
            let sum = self.sum(data); // Use sum() which handles special values
            Some(f64::from(sum) / data.len() as f64)
        }
    }

    fn count(&self, data: &[f32]) -> usize {
        data.len()
    }

    /// SIMD-accelerated min via trueno
    fn min(&self, data: &[f32]) -> Option<f32> {
        if data.is_empty() {
            None
        } else {
            Vector::from_slice(data).min().ok()
        }
    }

    /// SIMD-accelerated max via trueno
    fn max(&self, data: &[f32]) -> Option<f32> {
        if data.is_empty() {
            None
        } else {
            Vector::from_slice(data).max().ok()
        }
    }
}

// ============================================================================
// GPU Backend (Currently delegates to Scalar - will use wgpu in future)
// ============================================================================

impl Aggregation<i32> for GpuBackend {
    fn sum(&self, data: &[i32]) -> i32 {
        ScalarBackend.sum(data)
    }

    fn avg(&self, data: &[i32]) -> Option<f64> {
        ScalarBackend.avg(data)
    }

    fn count(&self, data: &[i32]) -> usize {
        ScalarBackend.count(data)
    }

    fn min(&self, data: &[i32]) -> Option<i32> {
        ScalarBackend.min(data)
    }

    fn max(&self, data: &[i32]) -> Option<i32> {
        ScalarBackend.max(data)
    }
}

impl Aggregation<f32> for GpuBackend {
    fn sum(&self, data: &[f32]) -> f32 {
        ScalarBackend.sum(data)
    }

    fn avg(&self, data: &[f32]) -> Option<f64> {
        ScalarBackend.avg(data)
    }

    fn count(&self, data: &[f32]) -> usize {
        ScalarBackend.count(data)
    }

    fn min(&self, data: &[f32]) -> Option<f32> {
        ScalarBackend.min(data)
    }

    fn max(&self, data: &[f32]) -> Option<f32> {
        ScalarBackend.max(data)
    }
}

// ============================================================================
// Property-Based Equivalence Tests
// ============================================================================

proptest! {
    /// Test: GPU sum == SIMD sum == Scalar sum (i32)
    ///
    /// Toyota Way: Jidoka - built-in quality check
    /// If backends diverge, tests fail immediately (Andon Cord)
    #[test]
    fn prop_sum_equivalence_i32(data: Vec<i32>) {
        let gpu_result = GpuBackend.sum(&data);
        let simd_result = SimdBackend.sum(&data);
        let scalar_result = ScalarBackend.sum(&data);

        prop_assert_eq!(gpu_result, simd_result, "GPU != SIMD for sum(i32)");
        prop_assert_eq!(simd_result, scalar_result, "SIMD != Scalar for sum(i32)");
    }

    /// Test: GPU sum == SIMD sum == Scalar sum (f32)
    #[test]
    fn prop_sum_equivalence_f32(data: Vec<f32>) {
        let gpu_result = GpuBackend.sum(&data);
        let simd_result = SimdBackend.sum(&data);
        let scalar_result = ScalarBackend.sum(&data);

        // Handle special cases (NaN, Infinity) consistently
        if gpu_result.is_nan() || simd_result.is_nan() || scalar_result.is_nan() {
            prop_assert!(gpu_result.is_nan() && simd_result.is_nan() && scalar_result.is_nan(),
                        "All backends must agree on NaN");
        } else if gpu_result.is_infinite() || simd_result.is_infinite() || scalar_result.is_infinite() {
            prop_assert!(gpu_result == simd_result, "GPU != SIMD for infinite sum");
            prop_assert!(simd_result == scalar_result, "SIMD != Scalar for infinite sum");
        } else {
            // Use approximate equality for finite floats (epsilon = 1e-5)
            let epsilon = 1e-5;
            prop_assert!((gpu_result - simd_result).abs() < epsilon, "GPU != SIMD for sum(f32)");
            prop_assert!((simd_result - scalar_result).abs() < epsilon, "SIMD != Scalar for sum(f32)");
        }
    }

    /// Test: GPU avg == SIMD avg == Scalar avg (i32)
    #[test]
    fn prop_avg_equivalence_i32(data: Vec<i32>) {
        let gpu_result = GpuBackend.avg(&data);
        let simd_result = SimdBackend.avg(&data);
        let scalar_result = ScalarBackend.avg(&data);

        prop_assert_eq!(gpu_result, simd_result, "GPU != SIMD for avg(i32)");
        prop_assert_eq!(simd_result, scalar_result, "SIMD != Scalar for avg(i32)");
    }

    /// Test: GPU avg == SIMD avg == Scalar avg (f32)
    #[test]
    fn prop_avg_equivalence_f32(data: Vec<f32>) {
        let gpu_result = GpuBackend.avg(&data);
        let simd_result = SimdBackend.avg(&data);
        let scalar_result = ScalarBackend.avg(&data);

        match (gpu_result, simd_result, scalar_result) {
            (Some(gpu), Some(simd), Some(scalar)) => {
                // Handle special cases (NaN, Infinity) consistently
                if gpu.is_nan() || simd.is_nan() || scalar.is_nan() {
                    prop_assert!(gpu.is_nan() && simd.is_nan() && scalar.is_nan(),
                                "All backends must agree on NaN for avg");
                } else if gpu.is_infinite() || simd.is_infinite() || scalar.is_infinite() {
                    prop_assert!(gpu == simd, "GPU != SIMD for infinite avg");
                    prop_assert!(simd == scalar, "SIMD != Scalar for infinite avg");
                } else {
                    let epsilon = 1e-4; // Larger epsilon for avg due to division
                    prop_assert!((gpu - simd).abs() < epsilon, "GPU != SIMD for avg(f32)");
                    prop_assert!((simd - scalar).abs() < epsilon, "SIMD != Scalar for avg(f32)");
                }
            }
            (None, None, None) => {
                // All backends correctly return None for empty input
            }
            _ => {
                prop_assert!(false, "Backends disagree on None vs Some for avg");
            }
        }
    }

    /// Test: GPU count == SIMD count == Scalar count
    #[test]
    fn prop_count_equivalence_i32(data: Vec<i32>) {
        let gpu_result = GpuBackend.count(&data);
        let simd_result = SimdBackend.count(&data);
        let scalar_result = ScalarBackend.count(&data);

        prop_assert_eq!(gpu_result, simd_result, "GPU != SIMD for count");
        prop_assert_eq!(simd_result, scalar_result, "SIMD != Scalar for count");
    }

    /// Test: GPU min == SIMD min == Scalar min (i32)
    #[test]
    fn prop_min_equivalence_i32(data: Vec<i32>) {
        let gpu_result = GpuBackend.min(&data);
        let simd_result = SimdBackend.min(&data);
        let scalar_result = ScalarBackend.min(&data);

        prop_assert_eq!(gpu_result, simd_result, "GPU != SIMD for min(i32)");
        prop_assert_eq!(simd_result, scalar_result, "SIMD != Scalar for min(i32)");
    }

    /// Test: GPU max == SIMD max == Scalar max (i32)
    #[test]
    fn prop_max_equivalence_i32(data: Vec<i32>) {
        let gpu_result = GpuBackend.max(&data);
        let simd_result = SimdBackend.max(&data);
        let scalar_result = ScalarBackend.max(&data);

        prop_assert_eq!(gpu_result, simd_result, "GPU != SIMD for max(i32)");
        prop_assert_eq!(simd_result, scalar_result, "SIMD != Scalar for max(i32)");
    }
}

// ============================================================================
// Edge Case Tests (NaN, Infinity, Overflow)
// ============================================================================

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_input_all_backends() {
        let empty: Vec<i32> = vec![];

        // Sum of empty should be 0
        assert_eq!(GpuBackend.sum(&empty), 0);
        assert_eq!(SimdBackend.sum(&empty), 0);
        assert_eq!(ScalarBackend.sum(&empty), 0);

        // Avg of empty should be None
        assert_eq!(GpuBackend.avg(&empty), None);
        assert_eq!(SimdBackend.avg(&empty), None);
        assert_eq!(ScalarBackend.avg(&empty), None);

        // Count of empty should be 0
        assert_eq!(GpuBackend.count(&empty), 0);
        assert_eq!(SimdBackend.count(&empty), 0);
        assert_eq!(ScalarBackend.count(&empty), 0);

        // Min/Max of empty should be None
        assert_eq!(GpuBackend.min(&empty), None);
        assert_eq!(SimdBackend.min(&empty), None);
        assert_eq!(ScalarBackend.min(&empty), None);
    }

    #[test]
    fn test_nan_handling() {
        let data_with_nan = vec![1.0_f32, f32::NAN, 3.0];

        // Sum with NaN
        let gpu_sum = GpuBackend.sum(&data_with_nan);
        let simd_sum = SimdBackend.sum(&data_with_nan);
        let scalar_sum = ScalarBackend.sum(&data_with_nan);

        // All backends should produce NaN
        assert!(gpu_sum.is_nan(), "GPU sum should be NaN");
        assert!(simd_sum.is_nan(), "SIMD sum should be NaN");
        assert!(scalar_sum.is_nan(), "Scalar sum should be NaN");

        // Min/Max with NaN (implementation-dependent behavior)
        let _gpu_min = GpuBackend.min(&data_with_nan);
        let _simd_min = SimdBackend.min(&data_with_nan);
        let _scalar_min = ScalarBackend.min(&data_with_nan);
        // Note: Behavior with NaN is implementation-dependent
    }

    #[test]
    fn test_infinity_handling() {
        let data_with_inf = vec![1.0_f32, f32::INFINITY, 3.0];

        let gpu_sum = GpuBackend.sum(&data_with_inf);
        let simd_sum = SimdBackend.sum(&data_with_inf);
        let scalar_sum = ScalarBackend.sum(&data_with_inf);

        assert_eq!(gpu_sum, f32::INFINITY);
        assert_eq!(simd_sum, f32::INFINITY);
        assert_eq!(scalar_sum, f32::INFINITY);
    }

    #[test]
    fn test_overflow_i32() {
        let data_overflow = vec![i32::MAX, 1];

        let gpu_sum = GpuBackend.sum(&data_overflow);
        let simd_sum = SimdBackend.sum(&data_overflow);
        let scalar_sum = ScalarBackend.sum(&data_overflow);

        // All backends should overflow consistently (wrapping)
        assert_eq!(gpu_sum, simd_sum, "GPU and SIMD disagree on overflow behavior");
        assert_eq!(simd_sum, scalar_sum, "SIMD and Scalar disagree on overflow behavior");
    }

    #[test]
    fn test_large_dataset_equivalence() {
        // Test with 1 million elements
        let large_data: Vec<i32> = (0..1_000_000).collect();

        let gpu_sum = GpuBackend.sum(&large_data);
        let simd_sum = SimdBackend.sum(&large_data);
        let scalar_sum = ScalarBackend.sum(&large_data);

        assert_eq!(gpu_sum, simd_sum);
        assert_eq!(simd_sum, scalar_sum);

        let gpu_count = GpuBackend.count(&large_data);
        let simd_count = SimdBackend.count(&large_data);
        let scalar_count = ScalarBackend.count(&large_data);

        assert_eq!(gpu_count, 1_000_000);
        assert_eq!(simd_count, 1_000_000);
        assert_eq!(scalar_count, 1_000_000);
    }
}
