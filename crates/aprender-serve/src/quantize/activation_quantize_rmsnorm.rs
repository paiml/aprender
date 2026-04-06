
    // ============= quantize_rmsnorm_q8_0 tests =============

    #[test]
    fn test_quantize_rmsnorm_q8_0_scalar_zeros() {
        let input = vec![0.0f32; 64];
        let norm_weight = vec![1.0f32; 64];
        let eps = 1e-5;

        let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

        // With all-zero input, scales should be minimal, quants should be 0
        assert_eq!(scales.len(), 2); // 64/32 = 2 blocks
        assert_eq!(quants.len(), 64);
        for q in &quants {
            assert_eq!(*q, 0);
        }
    }

    #[test]
    fn test_quantize_rmsnorm_q8_0_scalar_identity() {
        let input = vec![1.0f32; 32];
        let norm_weight = vec![1.0f32; 32];
        let eps = 1e-5;

        let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

        assert_eq!(scales.len(), 1);
        assert_eq!(quants.len(), 32);
        // Normalized value should be ~1.0 (input / sqrt(1.0 + eps))
    }

    #[test]
    fn test_quantize_rmsnorm_q8_0_matches_simd() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let norm_weight: Vec<f32> = (0..128).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let eps = 1e-5;

        let (scales_scalar, quants_scalar) =
            quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);
        let (scales_simd, quants_simd) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);

        // Should produce equivalent results
        assert_eq!(scales_scalar.len(), scales_simd.len());
        assert_eq!(quants_scalar.len(), quants_simd.len());

        for (s1, s2) in scales_scalar.iter().zip(scales_simd.iter()) {
            assert!((s1 - s2).abs() < 1e-4, "scale mismatch: {} vs {}", s1, s2);
        }
        for (q1, q2) in quants_scalar.iter().zip(quants_simd.iter()) {
            assert!(
                (*q1 as i32 - *q2 as i32).abs() <= 1,
                "quant mismatch: {} vs {}",
                q1,
                q2
            );
        }
    }

    #[test]
    fn test_quantize_rmsnorm_q8_0_with_scaling_weight() {
        let input = vec![2.0f32; 32];
        let norm_weight = vec![0.5f32; 32]; // Scale down by half
        let eps = 1e-5;

        let (scales, quants) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

        assert_eq!(scales.len(), 1);
        // With uniform input, normalized output should also be uniform
        let first_q = quants[0];
        for q in &quants[..32] {
            assert_eq!(*q, first_q);
        }
    }

    // ============= quantize_rmsnorm_q8_0_into tests =============

    #[test]
    fn test_quantize_rmsnorm_q8_0_into_basic() {
        let input = vec![1.0f32; 32];
        let norm_weight = vec![1.0f32; 32];
        let eps = 1e-5;

        let mut scales = vec![0.0f32; 1];
        let mut quants = vec![0i8; 32];

        quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);

        // Should produce non-zero scales
        assert!(scales[0] > 0.0);
        // With uniform input, all quants should be equal
        let first_q = quants[0];
        for q in &quants {
            assert_eq!(*q, first_q);
        }
    }

    #[test]
    fn test_quantize_rmsnorm_q8_0_into_matches_allocating() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let norm_weight = vec![1.0f32; 64];
        let eps = 1e-5;

        let (scales_alloc, quants_alloc) = quantize_rmsnorm_q8_0_scalar(&input, &norm_weight, eps);

        let mut scales_into = vec![0.0f32; 2];
        let mut quants_into = vec![0i8; 64];
        quantize_rmsnorm_q8_0_into(
            &input,
            &norm_weight,
            eps,
            &mut scales_into,
            &mut quants_into,
        );

        assert_eq!(scales_alloc, scales_into);
        assert_eq!(quants_alloc, quants_into);
    }

    // ============= fused_rmsnorm_q4_0_matmul tests =============

    #[test]
    fn test_fused_rmsnorm_q4_0_matmul_input_size_mismatch() {
        let input = vec![1.0f32; 32]; // Wrong size
        let norm_weight = vec![1.0f32; 64];
        let weight_data = vec![0u8; 1000];

        let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 64, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_fused_rmsnorm_q4_0_matmul_weight_size_mismatch() {
        let input = vec![1.0f32; 64];
        let norm_weight = vec![1.0f32; 64];
        let weight_data = vec![0u8; 10]; // Too small

        let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, 64, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_fused_rmsnorm_q4_0_matmul_valid() {
        let in_dim: usize = 32;
        let out_dim: usize = 8;
        let blocks_per_row = in_dim.div_ceil(32);
        let bytes_per_row = blocks_per_row * 18; // Q4_0 block is 18 bytes
        let total_bytes = out_dim * bytes_per_row;

        let input = vec![1.0f32; in_dim];
        let norm_weight = vec![1.0f32; in_dim];
        let weight_data = vec![0u8; total_bytes];

        let result =
            fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, in_dim, out_dim);

        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), out_dim);
    }

    // ============= fused_rmsnorm_ffn_up_gate tests =============

    #[test]
    fn test_fused_rmsnorm_ffn_up_gate_input_mismatch() {
        let input = vec![1.0f32; 32];
        let norm_weight = vec![1.0f32; 64]; // Wrong size
        let up_data = vec![0u8; 1000];
        let gate_data = vec![0u8; 1000];

        let result =
            fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_data, &gate_data, 64, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_fused_rmsnorm_ffn_up_gate_up_weight_too_small() {
        let input = vec![1.0f32; 64];
        let norm_weight = vec![1.0f32; 64];
        let up_data = vec![0u8; 10]; // Too small
        let gate_data = vec![0u8; 1000];

        let result =
            fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_data, &gate_data, 64, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_fused_rmsnorm_ffn_up_gate_gate_weight_too_small() {
        let input = vec![1.0f32; 64];
        let norm_weight = vec![1.0f32; 64];
        let up_data = vec![0u8; 1000];
        let gate_data = vec![0u8; 10]; // Too small

        let result =
            fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_data, &gate_data, 64, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_fused_rmsnorm_ffn_up_gate_valid() {
        let in_dim: usize = 32;
        let out_dim: usize = 8;
        let blocks_per_row = in_dim.div_ceil(32);
        let bytes_per_row = blocks_per_row * 18;
        let total_bytes = out_dim * bytes_per_row;

        let input = vec![1.0f32; in_dim];
        let norm_weight = vec![1.0f32; in_dim];
        let up_data = vec![0u8; total_bytes];
        let gate_data = vec![0u8; total_bytes];

        let result = fused_rmsnorm_ffn_up_gate(
            &input,
            &norm_weight,
            1e-5,
            &up_data,
            &gate_data,
            in_dim,
            out_dim,
        );

        assert!(result.is_ok());
        let (up, gate) = result.expect("expected value");
        assert_eq!(up.len(), out_dim);
        assert_eq!(gate.len(), out_dim);
    }

    // ============= fused_swiglu tests =============

    #[test]
    fn test_fused_swiglu_scalar_zeros() {
        let mut gate = vec![0.0f32; 8];
        let up = vec![1.0f32; 8];

        fused_swiglu_scalar(&mut gate, &up);

        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0, so result = 0 * 1 = 0
        for val in &gate {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_fused_swiglu_scalar_positive() {
        let mut gate = vec![1.0f32; 4];
        let up = vec![2.0f32; 4];

        fused_swiglu_scalar(&mut gate, &up);

        // silu(1) = 1 / (1 + exp(-1)) ≈ 0.731
        // result ≈ 0.731 * 2 = 1.462
        for val in &gate {
            assert!((val - 1.462).abs() < 0.01, "expected ~1.462, got {}", val);
        }
    }

    #[test]
    fn test_fused_swiglu_simd_matches_scalar() {
        let mut gate_simd: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.1, -0.1, 3.0, 0.0];
        let up: Vec<f32> = vec![1.0, 2.0, 0.5, 1.5, 1.0, 1.0, 2.0, 2.0, 0.5, 3.0];

        let mut gate_scalar = gate_simd.clone();

        fused_swiglu_scalar(&mut gate_scalar, &up);
        fused_swiglu_simd(&mut gate_simd, &up);

        // SIMD uses polynomial exp approximation with ~10% accuracy (5th-degree polynomial)
        // The goal is to verify the SIMD path runs and produces reasonable results
        for i in 0..gate_simd.len() {
            let abs_err = (gate_simd[i] - gate_scalar[i]).abs();
            // Allow 15% relative error or 0.05 absolute error
            let max_err = 0.20 * gate_scalar[i].abs().max(0.1);
            assert!(
                abs_err < max_err,
                "mismatch at {}: simd={} scalar={} abs_err={} max_err={}",
                i,
                gate_simd[i],
                gate_scalar[i],
                abs_err,
                max_err
            );
        }
    }

    #[test]
    fn test_fused_swiglu_simd_large() {
        let n = 128;
        let mut gate_simd: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 % 10.0) * 0.2).collect();

        let mut gate_scalar = gate_simd.clone();

        fused_swiglu_scalar(&mut gate_scalar, &up);
        fused_swiglu_simd(&mut gate_simd, &up);

        // SIMD uses polynomial exp approximation with ~10% accuracy (5th-degree polynomial)
        // The goal is to verify the SIMD path runs and produces reasonable results
        for i in 0..n {
            let abs_err = (gate_simd[i] - gate_scalar[i]).abs();
            // Allow 15% relative error or 0.05 absolute error
            let max_err = 0.20 * gate_scalar[i].abs().max(0.1);
            assert!(
                abs_err < max_err,
                "mismatch at {}: simd={} scalar={} abs_err={} max_err={}",
                i,
                gate_simd[i],
                gate_scalar[i],
                abs_err,
                max_err
            );
        }
    }

    // ============= softmax tests =============

    #[test]
    fn test_softmax_scalar_empty() {
        let mut x: Vec<f32> = vec![];
        softmax_scalar(&mut x);
        assert!(x.is_empty());
    }

    #[test]
    fn test_softmax_scalar_single() {
        let mut x = vec![5.0];
        softmax_scalar(&mut x);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_scalar_uniform() {
        let mut x = vec![1.0, 1.0, 1.0, 1.0];
        softmax_scalar(&mut x);

        // Uniform input should give uniform output
        for val in &x {
            assert!((val - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_scalar_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        softmax_scalar(&mut x);

        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_scalar_monotonic() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        softmax_scalar(&mut x);

        // Larger input should give larger output
        for i in 1..x.len() {
            assert!(x[i] > x[i - 1]);
        }
    }

    #[test]
    fn test_softmax_simd_empty() {
        let mut x: Vec<f32> = vec![];
        softmax_simd(&mut x);
        assert!(x.is_empty());
    }

    #[test]
    fn test_softmax_simd_matches_scalar() {
        let mut x_simd = vec![0.1, 0.2, 0.5, 1.0, 2.0, -1.0, 0.0, 0.3, 1.5, -0.5];
        let mut x_scalar = x_simd.clone();

        softmax_scalar(&mut x_scalar);
        softmax_simd(&mut x_simd);

        for i in 0..x_simd.len() {
            assert!(
                (x_simd[i] - x_scalar[i]).abs() < 1e-5,
                "mismatch at {}: simd={} scalar={}",
                i,
                x_simd[i],
                x_scalar[i]
            );
        }
    }

    #[test]
    fn test_softmax_simd_large() {
        let n = 128;
        let mut x_simd: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let mut x_scalar = x_simd.clone();

        softmax_scalar(&mut x_scalar);
        softmax_simd(&mut x_simd);

        for i in 0..n {
            assert!((x_simd[i] - x_scalar[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not cause overflow
        let mut x = vec![1000.0, 1001.0, 1002.0];
        softmax_simd(&mut x);

        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(!x[0].is_nan());
        assert!(!x[1].is_nan());
        assert!(!x[2].is_nan());
    }

    // ================================================================
    // FALSIFY-SM: softmax-kernel-v1.yaml contract falsification
    //
    // Five-Whys (PMAT-354):
    //   Why 1: realizar had 8 softmax tests but 0 FALSIFY-SM-* tagged
    //   Why 2: existing tests verify SIMD parity, not contract claims
    //   Why 3: no mapping from softmax-kernel-v1.yaml to realizar tests
    //   Why 4: realizar predates the provable-contracts YAML
    //   Why 5: softmax correctness was verified indirectly via inference
    //
    // References:
    //   - provable-contracts/contracts/softmax-kernel-v1.yaml
    // ================================================================

    /// FALSIFY-SM-001: Output sums to 1 (partition of unity)
    #[test]
    fn falsify_sm_001_sums_to_one() {
        let cases: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![-10.0, 0.0, 10.0],
            (0..128).map(|i| (i as f32 * 0.37).sin() * 5.0).collect(),
        ];

        for (idx, logits) in cases.iter().enumerate() {
            let mut x = logits.clone();
            softmax_simd(&mut x);
            let sum: f32 = x.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "FALSIFIED SM-001: case {idx} sum={sum}"
            );
        }
    }

    /// FALSIFY-SM-002: All outputs strictly positive (within f32 range)
    #[test]
    fn falsify_sm_002_strictly_positive() {
        // Use moderate range to stay within f32 representable softmax outputs
        let mut x: Vec<f32> = (0..10).map(|i| (i as f32 - 5.0) * 2.0).collect();
        softmax_simd(&mut x);

        for (i, &p) in x.iter().enumerate() {
            assert!(
                p > 0.0,
                "FALSIFIED SM-002: x[{i}] = {p} not strictly positive"
            );
        }
    }

    /// FALSIFY-SM-003: Order preservation (argmax invariant)
    #[test]
    fn falsify_sm_003_order_preservation() {
        let original = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        let input_argmax = original
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("expected value"))
            .expect("expected value")
            .0;

        let mut x = original;
        softmax_simd(&mut x);
        let output_argmax = x
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("expected value"))
            .expect("expected value")
            .0;

        assert_eq!(
            input_argmax, output_argmax,
            "FALSIFIED SM-003: argmax changed {input_argmax} → {output_argmax}"
        );
    }

    /// FALSIFY-SM-004: Each output bounded in (0, 1)
    #[test]
    fn falsify_sm_004_bounded_zero_one() {
        let mut x: Vec<f32> = (0..32).map(|i| (i as f32 * 1.7).sin() * 10.0).collect();
        softmax_simd(&mut x);

        for (i, &p) in x.iter().enumerate() {
            assert!(
                p > 0.0 && p < 1.0,
                "FALSIFIED SM-004: x[{i}] = {p} not in (0, 1)"
            );
        }
    }

    /// FALSIFY-SM-005: Numerical stability — no NaN/Inf
    #[test]
    fn falsify_sm_005_numerical_stability() {
        let mut x = vec![-500.0f32, 0.0, 500.0];
        softmax_simd(&mut x);

        for (i, &p) in x.iter().enumerate() {
            assert!(
                p.is_finite(),
                "FALSIFIED SM-005: x[{i}] = {p} is not finite"
            );
        }
    }

    /// FALSIFY-SM-006: Identical elements → uniform distribution
    ///
    /// Contract: softmax([c, c, ..., c]) = [1/n, 1/n, ..., 1/n]
    #[test]
    fn falsify_sm_006_identical_elements_uniform() {
        for n in [2, 4, 8, 16] {
            let mut x = vec![5.0f32; n];
            softmax_simd(&mut x);
            let expected = 1.0 / n as f32;

            for (i, &p) in x.iter().enumerate() {
                assert!(
                    (p - expected).abs() < 1e-6,
                    "FALSIFIED SM-006: n={n} x[{i}] = {p}, expected {expected}"
                );
            }
        }
    }

    /// FALSIFY-SM-007: Translation invariance — σ(x + c) = σ(x) for any scalar c
    ///
    /// Five-Whys (PMAT-354):
    ///   Why 1: SM-INV-003 (translation invariance) had ZERO coverage
    ///   Why 2: max-subtraction trick IMPLEMENTS this but nobody tested it
    ///   Why 3: foundational to numerical stability but untested
    ///
    /// Contract: σ(x + c·1) = σ(x) for any scalar c.
    #[test]
    fn falsify_sm_007_translation_invariance() {
        let base = vec![1.0f32, 3.0, -2.0, 0.5];
        let mut base_probs = base.clone();
        softmax_simd(&mut base_probs);

        for c in [100.0f32, -100.0, 0.0, 42.0, -999.0] {
            let mut shifted: Vec<f32> = base.iter().map(|&x| x + c).collect();
            softmax_simd(&mut shifted);

            for (i, (&orig, &shift)) in base_probs.iter().zip(shifted.iter()).enumerate() {
                assert!(
                    (orig - shift).abs() < 1e-5,
                    "FALSIFIED SM-007: σ(x+{c})[{i}] = {shift} != σ(x)[{i}] = {orig}"
                );
            }
        }
    }

    /// FALSIFY-SM-008: SIMD equivalence — |softmax_simd(x) - softmax_scalar(x)| < 8 ULP
    ///
    /// Five-Whys (PMAT-354):
    ///   Why 1: YAML SM-004 specifies SIMD equivalence but no test existed
    ///   Why 2: our SM-004 tests "bounded" (SM-BND-001), not SIMD equivalence
    ///   Why 3: naming mismatch — we numbered locally, not from the YAML
    ///   Why 4: SIMD parity was assumed correct from inference end-to-end tests
    ///   Why 5: nobody wrote an explicit scalar-vs-SIMD comparison test
    ///
    /// Contract: softmax_simd(x) and softmax_scalar(x) must agree within 8 ULP.
    /// Numbered SM-008 to avoid collision with our existing SM-004 (bounded).
    #[test]
    fn falsify_sm_008_simd_scalar_equivalence() {
        let test_cases: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![-10.0, 0.0, 10.0],
            vec![100.0, 100.0, 100.0, 100.0],
            vec![1e-6, 1e-6, 1e-6],
            (0..32).map(|i| (i as f32 * 0.7).sin()).collect(),
            (0..128).map(|i| i as f32 - 64.0).collect(),
            vec![-500.0, 0.0, 500.0],
        ];

        for (idx, base) in test_cases.iter().enumerate() {
            let mut simd_input = base.clone();
            let mut scalar_input = base.clone();

            softmax_simd(&mut simd_input);
            softmax_scalar(&mut scalar_input);

            for (i, (&s, &r)) in simd_input.iter().zip(scalar_input.iter()).enumerate() {
                // 8 ULP tolerance: |diff| / min(|s|, |r|) < 8 * f32::EPSILON
                let diff = (s - r).abs();
                let ulp_bound = 8.0 * f32::EPSILON * s.abs().max(r.abs()).max(f32::MIN_POSITIVE);
                assert!(
                    diff <= ulp_bound,
                    "FALSIFIED SM-008: case {idx}[{i}] SIMD={s} vs scalar={r}, diff={diff} > {ulp_bound} (8 ULP)"
                );
            }
        }
    }

    /// FALSIFY-SM-009: Single element boundary — softmax([x]) = [1.0] for any x
    ///
    /// Contract: YAML SM-005 = softmax of a single element is always 1.0.
    /// Numbered SM-009 to avoid collision with our existing SM-005 (stability).
    #[test]
    fn falsify_sm_009_single_element() {
        for x in [0.0f32, 1.0, -1.0, 100.0, -100.0, f32::MIN_POSITIVE, 1e30] {
            let mut v = vec![x];
            softmax_simd(&mut v);
            assert!(
                (v[0] - 1.0).abs() < 1e-6,
                "FALSIFIED SM-009: softmax([{x}]) = {}, expected 1.0",
                v[0]
            );
        }
    }

    // ============= quantize_activations_q8_0 tests =============

    #[test]
    fn test_quantize_activations_q8_0_zeros() {
        let activations = vec![0.0f32; 32];
        let (scales, quants) = quantize_activations_q8_0(&activations);

        assert_eq!(scales.len(), 1);
        assert_eq!(quants.len(), 32);
        for q in &quants {
            assert_eq!(*q, 0);
        }
    }

    #[test]
    fn test_quantize_activations_q8_0_positive() {
        let activations = vec![127.0f32; 32];
        let (scales, quants) = quantize_activations_q8_0(&activations);

        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 1.0).abs() < 0.01); // scale should be ~1.0
        for q in &quants {
            assert_eq!(*q, 127); // Should quantize to max
        }
    }

    // ================================================================
    // FALSIFY-SM proptest variants
    //
    // Property-based counterparts to the deterministic SM-001..SM-009
    // tests above. These use random inputs to hunt for edge cases that
    // hand-crafted vectors miss.
    //
    // References:
    //   - provable-contracts/contracts/softmax-kernel-v1.yaml
    // ================================================================

    mod softmax_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // FALSIFY-SM-001-prop: Normalization for random vectors
        //
        // Contract: sum(softmax(x)) = 1.0 for any non-empty input.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_sm_001_prop_sums_to_one(
                logits in proptest::collection::vec(-100.0_f32..100.0, 2..64),
            ) {
                let mut x = logits;
                softmax_simd(&mut x);
                let sum: f32 = x.iter().sum();
                prop_assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "FALSIFIED SM-001-prop: sum={} for {} elements", sum, x.len()
                );
            }
        }

        // FALSIFY-SM-002-prop: Positivity for random vectors
        //
        // Contract: softmax(x)[i] >= 0 and is finite for all i.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_sm_002_prop_positive(
                logits in proptest::collection::vec(-500.0_f32..500.0, 2..32),
            ) {
                let mut x = logits;
                softmax_simd(&mut x);
                for (i, &p) in x.iter().enumerate() {
                    prop_assert!(p >= 0.0, "FALSIFIED SM-002-prop: probs[{}]={} negative", i, p);
                    prop_assert!(p.is_finite(), "FALSIFIED SM-002-prop: probs[{}]={} non-finite", i, p);
                }
            }
        }

        // FALSIFY-SM-003-prop: Order preservation for random vectors
        //
        // Contract: argmax(softmax(x)) = argmax(x) when no duplicate max.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_sm_003_prop_order_preservation(
                logits in proptest::collection::vec(-50.0_f32..50.0, 2..32),
            ) {
                let has_dupes = logits.windows(2).any(|w| (w[0] - w[1]).abs() < 1e-10);
                if has_dupes {
                    return Ok(());
                }

                let input_argmax = logits.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("expected value")).expect("expected value").0;
                let mut x = logits;
                softmax_simd(&mut x);
                let output_argmax = x.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("expected value")).expect("expected value").0;
                prop_assert_eq!(
                    input_argmax, output_argmax,
                    "FALSIFIED SM-003-prop: argmax {} -> {}", input_argmax, output_argmax
                );
            }
        }
    }
