//! Tests for mixed-precision training utilities.

#![allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::autograd::precision::{
        bf16_to_f32, bf16_truncate, estimate_memory_savings, f32_to_bf16, f32_to_fp16, fp16_to_f32,
        gemm_bf16_reference, GradScaler, MixedPrecisionConfig, Precision,
    };

    #[test]
    fn test_precision_size_bytes() {
        assert_eq!(Precision::Fp32.size_bytes(), 4);
        assert_eq!(Precision::Fp16.size_bytes(), 2);
        assert_eq!(Precision::Bf16.size_bytes(), 2);
    }

    #[test]
    fn test_precision_name() {
        assert_eq!(Precision::Fp32.name(), "fp32");
        assert_eq!(Precision::Fp16.name(), "fp16");
        assert_eq!(Precision::Bf16.name(), "bf16");
    }

    #[test]
    fn test_precision_is_reduced() {
        assert!(!Precision::Fp32.is_reduced());
        assert!(Precision::Fp16.is_reduced());
        assert!(Precision::Bf16.is_reduced());
    }

    #[test]
    fn test_precision_memory_multiplier() {
        assert_eq!(Precision::Fp32.memory_multiplier(), 1.0);
        assert_eq!(Precision::Fp16.memory_multiplier(), 0.5);
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(format!("{}", Precision::Bf16), "bf16");
    }

    #[test]
    fn test_precision_default() {
        assert_eq!(Precision::default(), Precision::Fp32);
    }

    #[test]
    fn test_mixed_precision_config_fp32() {
        let config = MixedPrecisionConfig::fp32();
        assert!(!config.is_mixed());
        assert_eq!(config.compute_precision, Precision::Fp32);
    }

    #[test]
    fn test_mixed_precision_config_fp16() {
        let config = MixedPrecisionConfig::fp16();
        assert!(config.is_mixed());
        assert_eq!(config.compute_precision, Precision::Fp16);
        assert!(config.dynamic_scaling);
        assert_eq!(config.initial_scale, 65536.0);
    }

    #[test]
    fn test_mixed_precision_config_bf16() {
        let config = MixedPrecisionConfig::bf16();
        assert!(config.is_mixed());
        assert_eq!(config.compute_precision, Precision::Bf16);
        assert!(!config.dynamic_scaling); // bf16 typically doesn't need scaling
    }

    #[test]
    fn test_mixed_precision_config_builders() {
        let config =
            MixedPrecisionConfig::fp16().with_initial_scale(1024.0).with_dynamic_scaling(false);
        assert_eq!(config.initial_scale, 1024.0);
        assert!(!config.dynamic_scaling);
    }

    #[test]
    fn test_grad_scaler_new() {
        let scaler = GradScaler::new(65536.0);
        assert_eq!(scaler.scale(), 65536.0);
        assert!(scaler.is_dynamic());
    }

    #[test]
    fn test_grad_scaler_from_config() {
        let config = MixedPrecisionConfig::fp16();
        let scaler = GradScaler::from_config(&config);
        assert_eq!(scaler.scale(), config.initial_scale);
    }

    #[test]
    fn test_grad_scaler_scale_loss() {
        let scaler = GradScaler::new(1000.0);
        assert_eq!(scaler.scale_loss(0.001), 1.0);
    }

    #[test]
    fn test_grad_scaler_unscale_grad() {
        let scaler = GradScaler::new(1000.0);
        assert_eq!(scaler.unscale_grad(1000.0), 1.0);
    }

    #[test]
    fn test_grad_scaler_unscale_and_check_valid() {
        let scaler = GradScaler::new(100.0);
        let mut grads = vec![100.0, 200.0, 300.0];
        let valid = scaler.unscale_and_check(&mut grads);

        assert!(valid);
        assert_eq!(grads[0], 1.0);
        assert_eq!(grads[1], 2.0);
        assert_eq!(grads[2], 3.0);
    }

    #[test]
    fn test_grad_scaler_unscale_and_check_overflow() {
        let scaler = GradScaler::new(100.0);
        let mut grads = vec![100.0, f32::INFINITY, 300.0];
        let valid = scaler.unscale_and_check(&mut grads);

        assert!(!valid);
    }

    #[test]
    fn test_grad_scaler_update_success() {
        let mut scaler = GradScaler::new(1000.0);
        scaler.growth_interval = 2; // Fast growth for testing

        scaler.update(true);
        scaler.update(true);

        // After growth_interval successful steps, scale should grow
        assert!(scaler.scale() > 1000.0);
        assert_eq!(scaler.successful_steps(), 2);
    }

    #[test]
    fn test_grad_scaler_update_overflow() {
        let mut scaler = GradScaler::new(1000.0);

        scaler.update(false); // Overflow

        assert!(scaler.scale() < 1000.0);
        assert_eq!(scaler.overflow_count(), 1);
    }

    #[test]
    fn test_grad_scaler_scale_floor() {
        let mut scaler = GradScaler::new(1.0);

        scaler.update(false); // Overflow

        // Scale should not go below 1.0
        assert!(scaler.scale() >= 1.0);
    }

    #[test]
    fn test_grad_scaler_dynamic_disabled() {
        let mut scaler = GradScaler::new(1000.0);
        scaler.set_dynamic(false);

        scaler.update(false);

        // Scale should not change when dynamic is disabled
        assert_eq!(scaler.scale(), 1000.0);
    }

    #[test]
    fn test_f32_to_bf16_roundtrip() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 100.0, -0.001];
        for &val in &values {
            let bf16 = f32_to_bf16(val);
            let back = bf16_to_f32(bf16);
            // BF16 has limited precision, so we check approximate equality
            if val.abs() > 1e-6 {
                let rel_err = (back - val).abs() / val.abs();
                assert!(rel_err < 0.01, "BF16 roundtrip error too large for {val}");
            }
        }
    }

    #[test]
    fn test_f32_to_fp16_roundtrip() {
        let values = vec![0.0, 1.0, -1.0, 0.5, 100.0];
        for &val in &values {
            let fp16 = f32_to_fp16(val);
            let back = fp16_to_f32(fp16);
            // FP16 has limited precision
            if val.abs() > 1e-4 {
                let rel_err = (back - val).abs() / val.abs();
                assert!(rel_err < 0.01, "FP16 roundtrip error too large for {val}");
            }
        }
    }

    #[test]
    fn test_bf16_special_values() {
        // Zero
        let zero = f32_to_bf16(0.0);
        assert_eq!(bf16_to_f32(zero), 0.0);

        // Negative zero
        let neg_zero = f32_to_bf16(-0.0);
        assert_eq!(bf16_to_f32(neg_zero), -0.0);
    }

    #[test]
    fn test_fp16_infinity() {
        let inf = f32_to_fp16(f32::INFINITY);
        let back = fp16_to_f32(inf);
        assert!(back.is_infinite() && back > 0.0);

        let neg_inf = f32_to_fp16(f32::NEG_INFINITY);
        let back_neg = fp16_to_f32(neg_inf);
        assert!(back_neg.is_infinite() && back_neg < 0.0);
    }

    #[test]
    fn test_estimate_memory_savings() {
        let (fp32, mixed, savings) =
            estimate_memory_savings(1_000_000, 8, 512, 4096, Precision::Bf16);

        assert!(mixed < fp32);
        assert!(savings > 0.0);
        assert!(savings < 1.0);
    }

    #[test]
    fn test_memory_savings_no_reduction_for_fp32() {
        let (fp32, mixed, savings) =
            estimate_memory_savings(1_000_000, 8, 512, 4096, Precision::Fp32);

        assert_eq!(fp32, mixed);
        assert_eq!(savings, 0.0);
    }

    #[test]
    fn test_grad_scaler_default() {
        let scaler = GradScaler::default();
        assert_eq!(scaler.scale(), 65536.0);
    }

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert!(!config.is_mixed());
    }

    // ── R-002: BF16 mixed precision integration tests ──────────────────

    #[test]
    fn test_grad_scaler_bf16_is_noop() {
        // BF16 has same exponent range as f32, so no loss scaling needed.
        // GradScaler for BF16 config must be a no-op (scale=1.0, dynamic=false).
        let config = MixedPrecisionConfig::bf16();
        let mut scaler = GradScaler::from_config(&config);

        assert_eq!(scaler.scale(), 1.0, "BF16 scaler should have scale=1.0");
        assert!(!scaler.is_dynamic(), "BF16 scaler should not be dynamic");

        // scale_loss should be identity
        assert_eq!(scaler.scale_loss(0.5), 0.5);

        // unscale_grad should be identity
        assert_eq!(scaler.unscale_grad(42.0), 42.0);

        // update should not change scale
        scaler.update(true);
        assert_eq!(scaler.scale(), 1.0);
        scaler.update(false);
        assert_eq!(scaler.scale(), 1.0);
    }

    #[test]
    fn test_grad_scaler_fp16_is_active() {
        // FP16 needs dynamic loss scaling to prevent gradient underflow.
        let config = MixedPrecisionConfig::fp16();
        let scaler = GradScaler::from_config(&config);

        assert_eq!(scaler.scale(), 65536.0, "FP16 scaler should start at 65536");
        assert!(scaler.is_dynamic(), "FP16 scaler should be dynamic");

        // scale_loss should amplify
        assert_eq!(scaler.scale_loss(0.001), 0.001 * 65536.0);
    }

    #[test]
    fn test_bf16_nan_preserved() {
        let nan_bits = f32_to_bf16(f32::NAN);
        let back = bf16_to_f32(nan_bits);
        assert!(back.is_nan(), "NaN must be preserved through BF16 roundtrip");
    }

    #[test]
    fn test_bf16_infinity_preserved() {
        let inf = f32_to_bf16(f32::INFINITY);
        assert_eq!(bf16_to_f32(inf), f32::INFINITY);

        let neg_inf = f32_to_bf16(f32::NEG_INFINITY);
        assert_eq!(bf16_to_f32(neg_inf), f32::NEG_INFINITY);
    }

    #[test]
    fn test_bf16_precision_characteristics() {
        // BF16 has 7-bit mantissa → ~2 decimal digits of precision
        // Relative error should be <= 2^-7 ≈ 0.0078
        let test_values = [1.0_f32, 0.1, 3.14159, 100.0, 0.001, 65504.0];
        for &val in &test_values {
            let bf16 = f32_to_bf16(val);
            let back = bf16_to_f32(bf16);
            let rel_err = (back - val).abs() / val.abs();
            assert!(rel_err < 0.008, "BF16 relative error {rel_err} too large for {val}");
        }
    }

    #[test]
    fn test_bf16_vram_savings() {
        // BF16 uses 2 bytes vs f32's 4 bytes → 50% VRAM savings on activations
        let (fp32_bytes, mixed_bytes, savings) =
            estimate_memory_savings(1_000_000, 8, 512, 4096, Precision::Bf16);
        assert!(savings > 0.3, "BF16 should save >30% memory (got {savings})");
        assert!(mixed_bytes < fp32_bytes);
    }

    // ── R-002 Batch 14: BF16 GEMM reference + truncation tests ─────────

    #[test]
    fn test_bf16_truncate_basic() {
        // 1.0 = 0x3F800000 → bf16 = 0x3F80 → f32 = 0x3F800000 (exact)
        assert_eq!(bf16_truncate(1.0), 1.0);

        // 0.1 = 0x3DCCCCCD → bf16 truncation zeros lower 16 bits → 0x3DCC0000
        let t = bf16_truncate(0.1);
        assert_ne!(t, 0.1, "0.1 should lose precision under bf16 truncation");
        assert!((t - 0.1).abs() < 0.002, "bf16(0.1) should be close: got {t}");
    }

    #[test]
    fn test_bf16_truncate_special_values() {
        assert!(bf16_truncate(f32::NAN).is_nan());
        assert!(bf16_truncate(f32::INFINITY).is_infinite());
        assert!(bf16_truncate(f32::NEG_INFINITY).is_infinite());
        assert_eq!(bf16_truncate(0.0), 0.0);
        assert_eq!(bf16_truncate(-0.0).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn test_bf16_truncate_precision_loss() {
        // BF16 has 7 mantissa bits → ~2 decimal digits of precision
        // 1.0 + 2^-8 = 1.00390625 should be distinguishable in f32 but not bf16
        let val = 1.0f32 + (1.0 / 256.0);
        let truncated = bf16_truncate(val);
        assert_eq!(truncated, 1.0, "bf16 should lose the 8th mantissa bit");
    }

    #[test]
    fn test_bf16_truncate_matches_roundtrip() {
        // bf16_truncate(x) must equal bf16_to_f32(f32_to_bf16(x)) for all x
        let test_values = [0.0, 1.0, -1.0, 0.1, 3.14159, 65504.0, -0.001, 1e38];
        for &val in &test_values {
            let truncated = bf16_truncate(val);
            let roundtrip = bf16_to_f32(f32_to_bf16(val));
            assert_eq!(
                truncated.to_bits(),
                roundtrip.to_bits(),
                "bf16_truncate({val}) = {truncated} != roundtrip {roundtrip}"
            );
        }
    }

    #[test]
    fn test_bf16_truncate_lower_bits_zeroed() {
        // C-BF16GEMM-001: lower 16 bits must always be zero
        let test_values = [0.1, 0.2, 0.3, 1.5, 42.42, -99.99, f32::MAX, f32::MIN_POSITIVE];
        for &val in &test_values {
            let truncated = bf16_truncate(val);
            assert_eq!(truncated.to_bits() & 0x0000FFFF, 0, "lower 16 bits not zeroed for {val}");
        }
    }

    #[test]
    fn test_gemm_bf16_reference_identity() {
        // A @ I = A (with bf16 precision on elements)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let c = gemm_bf16_reference(&a, &b, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
        assert!((c[2] - 3.0).abs() < 1e-6);
        assert!((c[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_bf16_reference_vs_fp32() {
        // Compare bf16 GEMM vs fp32 GEMM — should differ but be close
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2x3
        let b = vec![0.7, 0.8, 0.9, 1.0, 1.1, 1.2]; // 3x2

        // FP32 reference
        let mut c_fp32 = vec![0.0f32; 4];
        for row in 0..2 {
            for col in 0..2 {
                let mut acc = 0.0f32;
                for i in 0..3 {
                    acc += a[row * 3 + i] * b[i * 2 + col];
                }
                c_fp32[row * 2 + col] = acc;
            }
        }

        let c_bf16 = gemm_bf16_reference(&a, &b, 2, 3, 2);

        // Results should be close but not identical due to bf16 truncation
        let mut any_different = false;
        for i in 0..4 {
            let diff = (c_fp32[i] - c_bf16[i]).abs();
            assert!(diff < 0.1, "BF16 vs FP32 diff too large at [{i}]: {diff}");
            if diff > 1e-7 {
                any_different = true;
            }
        }
        assert!(any_different, "BF16 GEMM should differ from FP32 due to truncation");
    }

    #[test]
    fn test_gemm_bf16_reference_fp32_accumulation() {
        // Verify accumulation is in f32 (not bf16)
        // Sum 1024 products of bf16(0.01)^2 — in bf16 accum this would lose precision
        let k = 1024;
        let a = vec![0.01f32; k];
        let b = vec![0.01f32; k];
        let c = gemm_bf16_reference(&a, &b, 1, k, 1);

        let bf16_val = bf16_truncate(0.01);
        let expected = bf16_val * bf16_val * k as f32;
        assert!(
            (c[0] - expected).abs() < 1e-4,
            "BF16 GEMM accumulation should be in f32: got {}, expected {expected}",
            c[0]
        );
    }

    #[test]
    fn test_gemm_bf16_reference_350m_dims() {
        // Verify BF16 GEMM works with 350M model dimensions (hidden=1024)
        let m = 4; // seq_len
        let k = 1024; // hidden_size
        let n = 1024; // hidden_size (Q projection)

        let a: Vec<f32> = (0..m * k).map(|i| 0.001 * (i as f32 % 100.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| 0.001 * (i as f32 % 100.0)).collect();

        let c = gemm_bf16_reference(&a, &b, m, k, n);

        // Verify output is finite (no NaN/Inf from accumulation)
        assert!(c.iter().all(|v: &f32| v.is_finite()), "All outputs should be finite");
        assert_eq!(c.len(), m * n);

        // Verify non-trivial result (not all zeros)
        assert!(c.iter().any(|&v| v != 0.0), "Output should not be all zeros");
    }
}
