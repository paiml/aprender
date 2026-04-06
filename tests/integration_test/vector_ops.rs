//! Vector operations integration tests: element-wise, reductions, activations, scalar ops

use proptest::prelude::*;
use trueno::Vector;

// ============================================================================
// PROPERTY TEST CONFIGURATION
// ============================================================================

const PROPTEST_CASES: u32 = 50; // Reduced from 100 for 30s target

// ============================================================================
// VECTOR OPERATIONS - ELEMENT-WISE
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: All element-wise binary operations
    #[test]
    fn integration_vector_elementwise_binary(
        a_data in prop::collection::vec(-100.0f32..100.0, 10..100),
        b_data in prop::collection::vec(-100.0f32..100.0, 10..100)
    ) {
        let len = a_data.len().min(b_data.len());
        let a = Vector::from_slice(&a_data[..len]);
        let b = Vector::from_slice(&b_data[..len]);

        // Test all binary operations
        prop_assert!(a.add(&b).is_ok());
        prop_assert!(a.sub(&b).is_ok());
        prop_assert!(a.mul(&b).is_ok());

        // div requires non-zero b
        let b_nonzero: Vec<f32> = b_data[..len].iter()
            .map(|&x| if x.abs() < 0.01 { 1.0 } else { x })
            .collect();
        let b_nz = Vector::from_slice(&b_nonzero);
        prop_assert!(a.div(&b_nz).is_ok());

        // Comparison operations
        prop_assert!(a.minimum(&b).is_ok());
        prop_assert!(a.maximum(&b).is_ok());

        // Dot product
        let dot = a.dot(&b)?;
        prop_assert!(dot.is_finite());

        // Lerp and FMA
        prop_assert!(a.lerp(&b, 0.5).is_ok());
        let c = Vector::from_slice(&a_data[..len]);
        prop_assert!(a.fma(&b, &c).is_ok());
    }

    /// Integration test: All element-wise unary operations
    #[test]
    fn integration_vector_elementwise_unary(
        data in prop::collection::vec(-50.0f32..50.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Basic unary operations
        prop_assert!(v.abs().is_ok());
        prop_assert!(v.neg().is_ok());
        prop_assert!(v.floor().is_ok());
        prop_assert!(v.ceil().is_ok());
        prop_assert!(v.round().is_ok());
        prop_assert!(v.trunc().is_ok());
        prop_assert!(v.fract().is_ok());
        prop_assert!(v.signum().is_ok());

        // Sqrt on positive values
        let v_pos = v.abs().unwrap();
        prop_assert!(v_pos.sqrt().is_ok());

        // Recip on non-zero values
        let v_nonzero: Vec<f32> = data.iter()
            .map(|&x| if x.abs() < 0.1 { 1.0 } else { x })
            .collect();
        let v_nz = Vector::from_slice(&v_nonzero);
        prop_assert!(v_nz.recip().is_ok());

        // Exponential and logarithm (restricted ranges)
        let v_restricted: Vec<f32> = data.iter()
            .map(|&x| x.abs().min(10.0))
            .collect();
        let v_r = Vector::from_slice(&v_restricted);
        prop_assert!(v_r.exp().is_ok());

        let v_pos_restricted: Vec<f32> = data.iter()
            .map(|&x| x.abs().clamp(0.1, 100.0))
            .collect();
        let v_pr = Vector::from_slice(&v_pos_restricted);
        prop_assert!(v_pr.ln().is_ok());
    }

    /// Integration test: Trigonometric functions
    #[test]
    fn integration_vector_trig(
        data in prop::collection::vec(-10.0f32..10.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Basic trig functions
        prop_assert!(v.sin().is_ok());
        prop_assert!(v.cos().is_ok());
        prop_assert!(v.tan().is_ok());

        // Inverse trig (restricted domain)
        let v_unit: Vec<f32> = data.iter()
            .map(|&x| x.clamp(-0.9, 0.9))
            .collect();
        let vu = Vector::from_slice(&v_unit);
        prop_assert!(vu.asin().is_ok());
        prop_assert!(vu.acos().is_ok());
        prop_assert!(v.atan().is_ok());

        // Hyperbolic functions (restricted range)
        let v_restricted: Vec<f32> = data.iter()
            .map(|&x| x.clamp(-5.0, 5.0))
            .collect();
        let vr = Vector::from_slice(&v_restricted);
        prop_assert!(vr.sinh().is_ok());
        prop_assert!(vr.cosh().is_ok());
        prop_assert!(vr.tanh().is_ok());

        // Inverse hyperbolic (appropriate domains)
        prop_assert!(v.asinh().is_ok());

        let v_pos: Vec<f32> = data.iter()
            .map(|&x| x.abs().max(1.01))
            .collect();
        let vp = Vector::from_slice(&v_pos);
        prop_assert!(vp.acosh().is_ok());

        let v_tanh_domain: Vec<f32> = data.iter()
            .map(|&x| x.clamp(-0.9, 0.9))
            .collect();
        let vtd = Vector::from_slice(&v_tanh_domain);
        prop_assert!(vtd.atanh().is_ok());
    }
}

// ============================================================================
// VECTOR OPERATIONS - REDUCTIONS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: All reduction operations
    #[test]
    fn integration_vector_reductions(
        data in prop::collection::vec(-100.0f32..100.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Basic reductions
        let sum = v.sum()?;
        let sum_kahan = v.sum_kahan()?;
        let min = v.min()?;
        let max = v.max()?;
        let sum_sq = v.sum_of_squares()?;

        prop_assert!(sum.is_finite());
        prop_assert!(sum_kahan.is_finite());
        prop_assert!(min <= max);
        prop_assert!(sum_sq >= 0.0);

        // Statistical operations
        let mean = v.mean()?;
        let variance = v.variance()?;
        let stddev = v.stddev()?;

        prop_assert!(mean.is_finite());
        prop_assert!(variance >= -1e-5); // Allow small numerical error
        prop_assert!(stddev >= -1e-5);

        // Index operations
        let argmin_idx = v.argmin()?;
        let argmax_idx = v.argmax()?;
        prop_assert!(argmin_idx < v.len());
        prop_assert!(argmax_idx < v.len());

        // Norms
        let l1 = v.norm_l1()?;
        let l2 = v.norm_l2()?;
        let linf = v.norm_linf()?;

        prop_assert!(l1 >= 0.0);
        prop_assert!(l2 >= 0.0);
        prop_assert!(linf >= 0.0);
        prop_assert!(linf <= l1); // L-inf <= L1 for normalized comparison
    }

    /// Integration test: Statistical operations with two vectors
    #[test]
    fn integration_vector_statistics_two_vectors(
        a_data in prop::collection::vec(-50.0f32..50.0, 10..100),
        b_data in prop::collection::vec(-50.0f32..50.0, 10..100)
    ) {
        let len = a_data.len().min(b_data.len());
        let a = Vector::from_slice(&a_data[..len]);
        let b = Vector::from_slice(&b_data[..len]);

        // Covariance and correlation
        let cov = a.covariance(&b)?;
        let corr = a.correlation(&b)?;

        prop_assert!(cov.is_finite());
        prop_assert!(corr.is_finite());
        prop_assert!((-1.0 - 1e-4..=1.0 + 1e-4).contains(&corr));
    }
}

// ============================================================================
// VECTOR OPERATIONS - ACTIVATION FUNCTIONS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: All activation functions
    #[test]
    fn integration_vector_activations(
        data in prop::collection::vec(-10.0f32..10.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // ReLU variants
        prop_assert!(v.relu().is_ok());
        prop_assert!(v.leaky_relu(0.01).is_ok());
        prop_assert!(v.elu(1.0).is_ok());

        // Sigmoid and variants
        prop_assert!(v.sigmoid().is_ok());

        // Softmax and log_softmax (restricted range for stability)
        let v_restricted: Vec<f32> = data.iter()
            .map(|&x| x.clamp(-10.0, 10.0))
            .collect();
        let vr = Vector::from_slice(&v_restricted);
        let softmax = vr.softmax()?;
        let _log_softmax = vr.log_softmax()?;

        // Verify softmax sums to 1
        let sum: f32 = softmax.as_slice().iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4);

        // Modern activations
        prop_assert!(v.gelu().is_ok());
        prop_assert!(v.swish().is_ok()); // Also known as SiLU
    }

    /// Integration test: Preprocessing operations
    #[test]
    fn integration_vector_preprocessing(
        data in prop::collection::vec(-100.0f32..100.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Clipping
        prop_assert!(v.clip(-50.0, 50.0).is_ok());

        // Min-max normalization
        if data.iter().any(|&x| x != data[0]) {
            let normalized = v.minmax_normalize()?;
            let min = *normalized.as_slice().iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let max = *normalized.as_slice().iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            prop_assert!(min >= -1e-5);
            prop_assert!(max <= 1.0 + 1e-5);
        }

        // Z-score normalization (requires n >= 3 and variance > 0)
        if data.len() >= 3 {
            let variance = v.variance()?;
            if variance > 0.1 {
                prop_assert!(v.zscore().is_ok());
            }
        }
    }
}

// ============================================================================
// VECTOR OPERATIONS - SCALAR AND UTILITY
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

    /// Integration test: Scalar operations
    #[test]
    fn integration_vector_scalar_ops(
        data in prop::collection::vec(-50.0f32..50.0, 10..100),
        scalar in -10.0f32..10.0
    ) {
        let v = Vector::from_slice(&data);

        // Scale
        prop_assert!(v.scale(scalar).is_ok());

        // Pow
        let v_pos = v.abs().unwrap();
        prop_assert!(v_pos.pow(2.0).is_ok());

        // Clamp
        prop_assert!(v.clamp(-10.0, 10.0).is_ok());
    }

    /// Integration test: Vector normalization
    #[test]
    fn integration_vector_normalization(
        data in prop::collection::vec(-50.0f32..50.0, 10..100)
    ) {
        let v = Vector::from_slice(&data);

        // Skip if vector is near-zero
        let norm = v.norm_l2()?;
        if norm > 1e-6 {
            let normalized = v.normalize()?;
            let new_norm = normalized.norm_l2()?;
            prop_assert!((new_norm - 1.0).abs() < 1e-4);
        }
    }
}
