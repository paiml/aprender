use crate::nn::quantization::*;

#[test]
fn test_fake_quant_config_default() {
    let config = FakeQuantConfig::default();
    assert_eq!(config.bits, 8);
    assert!(config.symmetric);
    assert!(!config.learnable);
}

#[test]
fn test_fake_quant_config_int4() {
    let config = FakeQuantConfig::int4();
    assert_eq!(config.bits, 4);
    let (qmin, qmax) = config.quant_range();
    assert_eq!(qmin, -7.0);
    assert_eq!(qmax, 7.0);
}

#[test]
fn test_fake_quant_config_int8() {
    let config = FakeQuantConfig::int8();
    assert_eq!(config.bits, 8);
    let (qmin, qmax) = config.quant_range();
    assert_eq!(qmin, -127.0);
    assert_eq!(qmax, 127.0);
}

#[test]
fn test_quant_observer_minmax() {
    let mut observer = QuantObserver::new(ObserverMethod::MinMax);
    observer.observe(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let (min, max) = observer.range();
    assert!((min - 1.0).abs() < 1e-6);
    assert!((max - 5.0).abs() < 1e-6);
}

#[test]
fn test_quant_observer_accumulates() {
    let mut observer = QuantObserver::new(ObserverMethod::MinMax);
    observer.observe(&[2.0, 3.0]);
    observer.observe(&[1.0, 4.0]);
    let (min, max) = observer.range();
    assert!((min - 1.0).abs() < 1e-6);
    assert!((max - 4.0).abs() < 1e-6);
}

#[test]
fn test_quant_observer_percentile() {
    let mut observer = QuantObserver::new(ObserverMethod::Percentile);
    let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    observer.observe(&data);
    let (min, max) = observer.range();
    // Percentile should exclude extreme values
    assert!(min < 10.0);
    assert!(max > 990.0);
}

#[test]
fn test_quant_observer_moving_average() {
    let mut observer = QuantObserver::new(ObserverMethod::MovingAverage);
    observer.observe(&[0.0, 10.0]);
    let (min1, max1) = observer.range();
    observer.observe(&[5.0, 5.0]);
    let (min2, max2) = observer.range();
    // Moving average should smooth values
    assert!(min2 > min1 - 0.5);
    assert!(max2 < max1 + 0.5);
}

#[test]
fn test_quant_observer_mean_std() {
    let mut observer = QuantObserver::new(ObserverMethod::MeanStd);
    // Normal-ish distribution around 0
    let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 20.0).collect();
    observer.observe(&data);
    let (min, max) = observer.range();
    // 3-sigma should cover most values
    assert!(min < -1.5);
    assert!(max > 1.5);
}

#[test]
fn test_observer_compute_qparams_symmetric() {
    let mut observer = QuantObserver::new(ObserverMethod::MinMax);
    observer.observe(&[-2.0, 2.0]);
    let config = FakeQuantConfig::int8();
    let (scale, zp) = observer.compute_qparams(&config);
    assert!((zp - 0.0).abs() < 1e-6); // Symmetric = zero point is 0
    assert!((scale - 2.0 / 127.0).abs() < 1e-6);
}

#[test]
fn test_fake_quantize_roundtrip() {
    let mut fq = FakeQuantize::new(FakeQuantConfig::int8());
    let data = vec![0.5, -0.5, 1.0, -1.0];
    let quantized = fq.fake_quantize(&data);

    // Should be close to original (within quantization error)
    for (orig, quant) in data.iter().zip(quantized.iter()) {
        assert!((orig - quant).abs() < 0.02, "orig={orig} quant={quant}");
    }
}

#[test]
fn test_fake_quantize_zeros() {
    let mut fq = FakeQuantize::new(FakeQuantConfig::int8());
    let data = vec![0.0, 0.0, 0.0];
    let quantized = fq.fake_quantize(&data);

    for q in &quantized {
        assert!(q.abs() < 1e-6);
    }
}

#[test]
fn test_fake_quantize_calibration() {
    let mut fq = FakeQuantize::new(FakeQuantConfig::int8());

    // First observation
    fq.fake_quantize(&[1.0, 2.0]);
    let scale1 = fq.scale();

    // Second observation expands range
    fq.fake_quantize(&[1.0, 4.0]);
    let scale2 = fq.scale();

    assert!(scale2 > scale1);
}

#[test]
fn test_fake_quantize_disable_calibration() {
    let mut fq = FakeQuantize::new(FakeQuantConfig::int8());
    fq.fake_quantize(&[1.0, 2.0]);
    fq.disable_calibration();
    let scale_frozen = fq.scale();

    // After disabling, scale should not change
    fq.fake_quantize(&[10.0, 20.0]);
    assert!((fq.scale() - scale_frozen).abs() < 1e-6);
}

#[test]
fn test_quantized_linear_from_float() {
    let weights = vec![1.0, 0.5, -0.5, -1.0];
    let config = FakeQuantConfig::int8();
    let ql = QuantizedLinear::from_float(&weights, None, 2, 2, &config);

    assert!(ql.weight_scale > 0.0);
    assert_eq!(ql.weights_q.len(), 4);
}

#[test]
fn test_quantized_linear_forward() {
    let weights = vec![1.0, 0.0, 0.0, 1.0]; // Identity-ish
    let config = FakeQuantConfig::int8();
    let ql = QuantizedLinear::from_float(&weights, None, 2, 2, &config);

    // Quantize input
    let input = vec![10i8, 20i8];
    let output = ql.forward_quantized(&input);

    assert_eq!(output.len(), 2);
}

#[test]
fn test_dynamic_quantizer() {
    let dq = DynamicQuantizer::new(FakeQuantConfig::int8());
    let data = vec![0.5, -0.5, 1.0, -1.0];

    let (quantized, scale, zp) = dq.quantize(&data);
    let dequantized = dq.dequantize(&quantized, scale, zp);

    for (orig, deq) in data.iter().zip(dequantized.iter()) {
        assert!((orig - deq).abs() < 0.02);
    }
}

#[test]
fn test_mixed_precision_scale_loss() {
    let mp = MixedPrecision::new();
    let loss = 0.5;
    let scaled = mp.scale_loss(loss);
    assert!((scaled - 0.5 * 65536.0).abs() < 1.0);
}

#[test]
fn test_mixed_precision_unscale() {
    let mp = MixedPrecision::new();
    let mut grads = vec![65536.0, 131072.0];
    mp.unscale_gradients(&mut grads);
    assert!((grads[0] - 1.0).abs() < 1e-6);
    assert!((grads[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_mixed_precision_check_overflow() {
    let mp = MixedPrecision::new();
    assert!(!mp.check_overflow(&[1.0, 2.0]));
    assert!(mp.check_overflow(&[1.0, f32::INFINITY]));
    assert!(mp.check_overflow(&[f32::NAN, 1.0]));
}

#[test]
fn test_mixed_precision_update_no_overflow() {
    let mut mp = MixedPrecision::new();
    let initial = mp.loss_scale();

    // Simulate many good steps
    for _ in 0..2000 {
        mp.update(false);
    }

    assert!(mp.loss_scale() > initial);
}

#[test]
fn test_mixed_precision_update_with_overflow() {
    let mut mp = MixedPrecision::new();
    let initial = mp.loss_scale();

    mp.update(true);

    assert!(mp.loss_scale() < initial);
}

#[test]
fn test_mixed_precision_reset() {
    let mut mp = MixedPrecision::new();
    mp.update(true); // Reduce scale
    mp.reset();
    assert!((mp.loss_scale() - 65536.0).abs() < 1.0);
}

#[test]
fn test_fake_quantize_module() {
    let fq = FakeQuantize::new(FakeQuantConfig::int8());
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let output = fq.forward(&input);
    assert_eq!(output.shape(), &[4]);
}

#[test]
fn test_observer_reset() {
    let mut observer = QuantObserver::new(ObserverMethod::MinMax);
    observer.observe(&[1.0, 10.0]);
    observer.reset();
    let (min, max) = observer.range();
    assert!(min.is_infinite() && min > 0.0); // +inf
    assert!(max.is_infinite() && max < 0.0); // -inf
}

#[test]
fn test_fake_quant_config_asymmetric() {
    let config = FakeQuantConfig {
        bits: 8,
        symmetric: false,
        learnable: false,
        observer: ObserverMethod::MinMax,
    };
    let (qmin, qmax) = config.quant_range();
    assert_eq!(qmin, 0.0);
    assert_eq!(qmax, 255.0);
}

#[test]
fn test_observer_asymmetric_qparams() {
    let mut observer = QuantObserver::new(ObserverMethod::MinMax);
    observer.observe(&[0.0, 4.0]);

    let config = FakeQuantConfig {
        bits: 8,
        symmetric: false,
        learnable: false,
        observer: ObserverMethod::MinMax,
    };
    let (scale, zp) = observer.compute_qparams(&config);

    // scale = 4/255, zp = 0 - 0/scale = 0
    assert!((scale - 4.0 / 255.0).abs() < 1e-6);
    assert!((zp - 0.0).abs() < 1e-6);
}

// ========================================================================
// Coverage: FakeQuantConfig Clone and builder methods
// ========================================================================

#[test]
fn test_fake_quant_config_clone() {
    let config = FakeQuantConfig::int8();
    let cloned = config.clone();
    assert_eq!(cloned.bits, 8);
    assert!(cloned.symmetric);
    assert!(!cloned.learnable);
}

#[test]
fn test_fake_quant_config_with_learnable() {
    let config = FakeQuantConfig::int8().with_learnable();
    assert!(config.learnable);
    assert_eq!(config.bits, 8);
}

#[test]
fn test_fake_quant_config_with_observer() {
    let config = FakeQuantConfig::int8().with_observer(ObserverMethod::Percentile);
    assert_eq!(config.observer, ObserverMethod::Percentile);
}

#[test]
fn test_fake_quant_config_with_observer_mean_std() {
    let config = FakeQuantConfig::int4().with_observer(ObserverMethod::MeanStd);
    assert_eq!(config.observer, ObserverMethod::MeanStd);
    assert_eq!(config.bits, 4);
}

// ========================================================================
// Coverage: QuantObserver Clone
// ========================================================================

#[test]
fn test_quant_observer_clone() {
    let mut observer = QuantObserver::new(ObserverMethod::MinMax);
    observer.observe(&[1.0, 5.0]);
    let cloned = observer.clone();
    let (min, max) = cloned.range();
    assert!((min - 1.0).abs() < 1e-6);
    assert!((max - 5.0).abs() < 1e-6);
}

// ========================================================================
// Coverage: observer with empty data
// ========================================================================

#[test]
fn test_quant_observer_empty_data() {
    let mut observer = QuantObserver::new(ObserverMethod::MinMax);
    observer.observe(&[]);
    // Range should remain at initial infinity values
    let (min, max) = observer.range();
    assert!(min.is_infinite());
    assert!(max.is_infinite());
}

// ========================================================================
// Coverage: symmetric qparams with zero max_abs
// ========================================================================

#[test]
fn test_observer_compute_qparams_symmetric_zero_range() {
    let mut observer = QuantObserver::new(ObserverMethod::MinMax);
    observer.observe(&[0.0, 0.0, 0.0]);
    let config = FakeQuantConfig::int8();
    let (scale, zp) = observer.compute_qparams(&config);
    // When max_abs is 0, scale defaults to 1.0
    assert!((scale - 1.0).abs() < 1e-6);
    assert!((zp - 0.0).abs() < 1e-6);
}

// ========================================================================
// Coverage: asymmetric qparams with near-zero range
// ========================================================================

#[test]
fn test_observer_compute_qparams_asymmetric_zero_range() {
    let mut observer = QuantObserver::new(ObserverMethod::MinMax);
    observer.observe(&[5.0, 5.0]); // min == max

    let config = FakeQuantConfig {
        bits: 8,
        symmetric: false,
        learnable: false,
        observer: ObserverMethod::MinMax,
    };
    let (scale, _zp) = observer.compute_qparams(&config);
    // When max - min < 1e-10, scale defaults to 1.0
    assert!((scale - 1.0).abs() < 1e-6);
}

// ========================================================================
// Coverage: QuantizedLinear with bias
// ========================================================================

#[test]
fn test_quantized_linear_from_float_with_bias() {
    let weights = vec![1.0, 0.5, -0.5, -1.0];
    let bias = vec![0.1, -0.2];
    let config = FakeQuantConfig::int8();
    let ql = QuantizedLinear::from_float(&weights, Some(&bias), 2, 2, &config);

    assert!(ql.bias_q.is_some());
    assert_eq!(ql.bias_q.as_ref().unwrap().len(), 2);
}

#[test]
fn test_quantized_linear_forward_with_bias() {
    let weights = vec![1.0, 0.0, 0.0, 1.0]; // Identity-ish
    let bias = vec![10.0, 20.0];
    let config = FakeQuantConfig::int8();
    let ql = QuantizedLinear::from_float(&weights, Some(&bias), 2, 2, &config);

    let input = vec![10i8, 20i8];
    let output = ql.forward_quantized(&input);

    assert_eq!(output.len(), 2);
    // Output should include bias contribution
    let bias_q = ql.bias_q.as_ref().unwrap();
    // Verify bias was added (output[0] includes bias_q[0])
    assert_ne!(output[0], 0);
    assert!(output[0] != i32::from(input[0]) * i32::from(ql.weights_q[0]) || bias_q[0] != 0);
}

// ========================================================================
// Coverage: QuantizedLinear batch forward
// ========================================================================

#[test]
fn test_quantized_linear_forward_batch() {
    let weights = vec![1.0, 0.0, 0.0, 1.0];
    let config = FakeQuantConfig::int8();
    let ql = QuantizedLinear::from_float(&weights, None, 2, 2, &config);

    // Batch of 2 samples
    let input = vec![10i8, 20i8, 30i8, 40i8];
    let output = ql.forward_quantized(&input);
    assert_eq!(output.len(), 4); // 2 samples * 2 outputs
}

// ========================================================================
// Coverage: QuantizedLinear set_input_scale
// ========================================================================

#[test]
fn test_quantized_linear_set_input_scale() {
    let weights = vec![1.0, 0.5, -0.5, -1.0];
    let config = FakeQuantConfig::int8();
    let mut ql = QuantizedLinear::from_float(&weights, None, 2, 2, &config);

    let original_output_scale = ql.output_scale();
    ql.set_input_scale(2.0);

    assert!((ql.input_scale - 2.0).abs() < 1e-6);
    // output_scale = weight_scale * input_scale
    assert!((ql.output_scale() - ql.weight_scale * 2.0).abs() < 1e-6);
    assert!((ql.output_scale() - original_output_scale * 2.0).abs() < 1e-6);
}
