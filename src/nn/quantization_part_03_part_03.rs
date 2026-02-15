
    // ========================================================================
    // Coverage: QuantizedLinear from_float with zero weights
    // ========================================================================

    #[test]
    fn test_quantized_linear_from_float_zero_weights() {
        let weights = vec![0.0, 0.0, 0.0, 0.0];
        let config = FakeQuantConfig::int8();
        let ql = QuantizedLinear::from_float(&weights, None, 2, 2, &config);
        // When max_abs is 0, scale should default to 1.0
        assert!((ql.weight_scale - 1.0).abs() < 1e-6);
    }

    // ========================================================================
    // Coverage: FakeQuantize enable_calibration
    // ========================================================================

    #[test]
    fn test_fake_quantize_enable_calibration() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::int8());
        fq.fake_quantize(&[1.0, 2.0]);
        fq.disable_calibration();
        let frozen_scale = fq.scale();

        // Re-enable calibration
        fq.enable_calibration();
        assert!(fq.calibrating);
        fq.fake_quantize(&[10.0, 20.0]);
        // Scale should change now that calibration is re-enabled
        assert!(fq.scale() > frozen_scale);
    }

    // ========================================================================
    // Coverage: FakeQuantize config accessor
    // ========================================================================

    #[test]
    fn test_fake_quantize_config_accessor() {
        let fq = FakeQuantize::new(FakeQuantConfig::int4());
        let config = fq.config();
        assert_eq!(config.bits, 4);
        assert!(config.symmetric);
    }

    // ========================================================================
    // Coverage: FakeQuantize zero_point accessor
    // ========================================================================

    #[test]
    fn test_fake_quantize_zero_point_accessor() {
        let fq = FakeQuantize::new(FakeQuantConfig::int8());
        // Default zero point for symmetric quantization
        assert!((fq.zero_point() - 0.0).abs() < 1e-6);
    }

    // ========================================================================
    // Coverage: MixedPrecision Default impl
    // ========================================================================

    #[test]
    fn test_mixed_precision_default() {
        let mp = MixedPrecision::default();
        assert!((mp.loss_scale() - 65536.0).abs() < 1.0);
    }

    // ========================================================================
    // Coverage: DynamicQuantizer Clone
    // ========================================================================

    #[test]
    fn test_dynamic_quantizer_clone() {
        let dq = DynamicQuantizer::new(FakeQuantConfig::int8());
        let cloned = dq.clone();
        let data = vec![0.5, -0.5, 1.0, -1.0];
        let (q_orig, s_orig, zp_orig) = dq.quantize(&data);
        let (q_cloned, s_cloned, zp_cloned) = cloned.quantize(&data);
        assert_eq!(q_orig, q_cloned);
        assert!((s_orig - s_cloned).abs() < 1e-6);
        assert!((zp_orig - zp_cloned).abs() < 1e-6);
    }

    // ========================================================================
    // Coverage: MixedPrecision Clone
    // ========================================================================

    #[test]
    fn test_mixed_precision_clone() {
        let mut mp = MixedPrecision::new();
        mp.update(true); // Reduce scale
        let cloned = mp.clone();
        assert!((mp.loss_scale() - cloned.loss_scale()).abs() < 1e-6);
    }

    // ========================================================================
    // Coverage: FakeQuantize Module parameters (empty)
    // ========================================================================

    #[test]
    fn test_fake_quantize_module_parameters() {
        let fq = FakeQuantize::new(FakeQuantConfig::int8());
        assert!(fq.parameters().is_empty());
    }

    #[test]
    fn test_fake_quantize_module_parameters_mut() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::int8());
        assert!(fq.parameters_mut().is_empty());
    }

    // ========================================================================
    // Coverage: FakeQuantize forward with non-default scale
    // ========================================================================

    #[test]
    fn test_fake_quantize_module_forward_after_calibration() {
        let mut fq = FakeQuantize::new(FakeQuantConfig::int8());
        // Calibrate first
        fq.fake_quantize(&[-2.0, 2.0]);
        fq.disable_calibration();

        // Now use Module::forward
        let input = Tensor::new(&[0.5, -0.5, 1.0, -1.0], &[4]);
        let output = fq.forward(&input);
        assert_eq!(output.shape(), &[4]);

        // Values should be close to original (within quantization error)
        for (orig, quant) in input.data().iter().zip(output.data().iter()) {
            assert!((orig - quant).abs() < 0.05);
        }
    }

    // ========================================================================
    // Coverage: MixedPrecision growth at exactly growth_interval boundary
    // ========================================================================

    #[test]
    fn test_mixed_precision_growth_boundary() {
        let mut mp = MixedPrecision::new();
        let initial = mp.loss_scale();

        // Do exactly growth_interval - 1 good steps (no growth yet)
        for _ in 0..1999 {
            mp.update(false);
        }
        assert!((mp.loss_scale() - initial).abs() < 1.0);

        // One more good step triggers growth
        mp.update(false);
        assert!((mp.loss_scale() - initial * 2.0).abs() < 1.0);
    }

    // ========================================================================
    // Coverage: Observer method enum Debug/Clone/PartialEq
    // ========================================================================

    #[test]
    fn test_observer_method_debug_clone_eq() {
        let method = ObserverMethod::MinMax;
        let cloned = method;
        assert_eq!(method, cloned);
        let debug_str = format!("{:?}", method);
        assert!(debug_str.contains("MinMax"));

        assert_ne!(ObserverMethod::MinMax, ObserverMethod::Percentile);
        assert_ne!(ObserverMethod::MovingAverage, ObserverMethod::MeanStd);
    }

    // ========================================================================
    // Coverage: FakeQuantConfig quant_range for asymmetric int4
    // ========================================================================

    #[test]
    fn test_fake_quant_config_asymmetric_int4() {
        let config = FakeQuantConfig {
            bits: 4,
            symmetric: false,
            learnable: false,
            observer: ObserverMethod::MinMax,
        };
        let (qmin, qmax) = config.quant_range();
        assert_eq!(qmin, 0.0);
        assert_eq!(qmax, 15.0);
    }

    // ========================================================================
    // Coverage: DynamicQuantizer with asymmetric config
    // ========================================================================

    #[test]
    fn test_dynamic_quantizer_asymmetric() {
        let config = FakeQuantConfig {
            bits: 8,
            symmetric: false,
            learnable: false,
            observer: ObserverMethod::MinMax,
        };
        let dq = DynamicQuantizer::new(config);
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let (quantized, scale, zp) = dq.quantize(&data);
        let dequantized = dq.dequantize(&quantized, scale, zp);

        // Verify basic properties: correct count, finite values, scale > 0
        assert_eq!(quantized.len(), data.len());
        assert_eq!(dequantized.len(), data.len());
        assert!(scale > 0.0);
        assert!(zp.is_finite());
        for val in &dequantized {
            assert!(val.is_finite());
        }
    }

    // ========================================================================
    // Coverage: MixedPrecision check_overflow with all-finite
    // ========================================================================

    #[test]
    fn test_mixed_precision_no_overflow_empty() {
        let mp = MixedPrecision::new();
        assert!(!mp.check_overflow(&[]));
    }

    #[test]
    fn test_mixed_precision_overflow_nan_only() {
        let mp = MixedPrecision::new();
        assert!(mp.check_overflow(&[f32::NAN]));
    }
