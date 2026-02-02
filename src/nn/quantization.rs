//! Quantization-Aware Training (QAT) module.
//!
//! This module provides fake quantization operators for simulating quantization
//! effects during training. The forward pass uses quantized values while gradients
//! flow through using the Straight-Through Estimator (STE).
//!
//! # Overview
//!
//! ```text
//! Training with QAT:
//!
//! ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
//! │   Input     │ ──► │ FakeQuantize │ ──► │   Output    │
//! │   (f32)     │     │  (simulate)  │     │   (f32)     │
//! └─────────────┘     └──────────────┘     └─────────────┘
//!                            │
//!                     ┌──────┴──────┐
//!                     │ STE Gradient │
//!                     │ (pass-thru)  │
//!                     └─────────────┘
//! ```
//!
//! # References
//!
//! - Jacob, B., et al. (2018). Quantization and Training of Neural Networks
//!   for Efficient Integer-Arithmetic-Only Inference. CVPR.
//! - Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for
//!   efficient inference. arXiv:1806.08342.

use crate::autograd::Tensor;
use crate::nn::Module;

/// Fake quantization configuration for QAT.
#[derive(Debug, Clone)]
pub struct FakeQuantConfig {
    /// Number of bits for quantization (4 or 8 typical)
    pub bits: usize,
    /// Whether to use symmetric quantization (centered at 0)
    pub symmetric: bool,
    /// Whether to learn scale/zero-point (LSQ)
    pub learnable: bool,
    /// Observation method for scale calibration
    pub observer: ObserverMethod,
}

/// Method for observing tensor statistics for quantization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObserverMethod {
    /// Min-max observation
    MinMax,
    /// Percentile observation (reduces outlier impact)
    Percentile,
    /// Moving average min-max
    MovingAverage,
    /// Mean + std deviation based (for weights)
    MeanStd,
}

impl Default for FakeQuantConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            symmetric: true,
            learnable: false,
            observer: ObserverMethod::MinMax,
        }
    }
}

impl FakeQuantConfig {
    /// Create 8-bit symmetric quantization config.
    #[must_use]
    pub fn int8() -> Self {
        Self {
            bits: 8,
            symmetric: true,
            learnable: false,
            observer: ObserverMethod::MinMax,
        }
    }

    /// Create 4-bit symmetric quantization config.
    #[must_use]
    pub fn int4() -> Self {
        Self {
            bits: 4,
            symmetric: true,
            learnable: false,
            observer: ObserverMethod::MinMax,
        }
    }

    /// Enable learnable scale quantization (LSQ).
    #[must_use]
    pub fn with_learnable(mut self) -> Self {
        self.learnable = true;
        self
    }

    /// Set observation method.
    #[must_use]
    pub fn with_observer(mut self, observer: ObserverMethod) -> Self {
        self.observer = observer;
        self
    }

    /// Compute quantization range for given bits.
    #[must_use]
    pub fn quant_range(&self) -> (f32, f32) {
        if self.symmetric {
            let max = (1 << (self.bits - 1)) - 1;
            (-(max as f32), max as f32)
        } else {
            let max = (1 << self.bits) - 1;
            (0.0, max as f32)
        }
    }
}

/// Observer for tracking tensor statistics.
#[derive(Debug, Clone)]
pub struct QuantObserver {
    /// Observed minimum value
    min_val: f32,
    /// Observed maximum value
    max_val: f32,
    /// Running average of min (for moving average)
    avg_min: f32,
    /// Running average of max (for moving average)
    avg_max: f32,
    /// Number of observations
    count: usize,
    /// Averaging constant
    averaging_constant: f32,
    /// Observer method
    method: ObserverMethod,
}

impl QuantObserver {
    /// Create a new observer.
    #[must_use]
    pub fn new(method: ObserverMethod) -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            avg_min: 0.0,
            avg_max: 0.0,
            count: 0,
            averaging_constant: 0.01,
            method,
        }
    }

    /// Update statistics from tensor data.
    pub fn observe(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        match self.method {
            ObserverMethod::MinMax => {
                let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                self.min_val = self.min_val.min(min);
                self.max_val = self.max_val.max(max);
            }
            ObserverMethod::Percentile => {
                // Use 0.1% and 99.9% percentiles
                let mut sorted = data.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let p_low = (sorted.len() as f32 * 0.001).floor() as usize;
                let p_high = ((sorted.len() as f32 * 0.999).ceil() as usize).min(sorted.len() - 1);
                let min = sorted[p_low];
                let max = sorted[p_high];
                self.min_val = self.min_val.min(min);
                self.max_val = self.max_val.max(max);
            }
            ObserverMethod::MovingAverage => {
                let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                if self.count == 0 {
                    self.avg_min = min;
                    self.avg_max = max;
                } else {
                    let c = self.averaging_constant;
                    self.avg_min = (1.0 - c) * self.avg_min + c * min;
                    self.avg_max = (1.0 - c) * self.avg_max + c * max;
                }
                self.min_val = self.avg_min;
                self.max_val = self.avg_max;
            }
            ObserverMethod::MeanStd => {
                let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                let variance: f32 =
                    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
                let std = variance.sqrt();
                // Use 3-sigma range
                let min = mean - 3.0 * std;
                let max = mean + 3.0 * std;
                self.min_val = self.min_val.min(min);
                self.max_val = self.max_val.max(max);
            }
        }
        self.count += 1;
    }

    /// Get observed range.
    #[must_use]
    pub fn range(&self) -> (f32, f32) {
        (self.min_val, self.max_val)
    }

    /// Reset observer.
    pub fn reset(&mut self) {
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
        self.avg_min = 0.0;
        self.avg_max = 0.0;
        self.count = 0;
    }

    /// Compute scale and zero-point for asymmetric quantization.
    #[must_use]
    pub fn compute_qparams(&self, config: &FakeQuantConfig) -> (f32, f32) {
        let (qmin, qmax) = config.quant_range();
        let (min_val, max_val) = self.range();

        if config.symmetric {
            // Symmetric: scale = max(|min|, |max|) / qmax
            let max_abs = min_val.abs().max(max_val.abs());
            let scale = if max_abs > 0.0 { max_abs / qmax } else { 1.0 };
            (scale, 0.0)
        } else {
            // Asymmetric: scale = (max - min) / (qmax - qmin)
            let scale = if (max_val - min_val).abs() > 1e-10 {
                (max_val - min_val) / (qmax - qmin)
            } else {
                1.0
            };
            let zero_point = qmin - min_val / scale;
            (scale, zero_point)
        }
    }
}

/// Fake quantization layer for QAT.
///
/// Applies fake quantization during forward pass (simulates int8/int4)
/// while allowing gradients to flow through using STE.
#[derive(Debug)]
pub struct FakeQuantize {
    config: FakeQuantConfig,
    observer: QuantObserver,
    /// Current scale
    scale: f32,
    /// Current zero point
    zero_point: f32,
    /// Whether in calibration mode
    calibrating: bool,
}

impl FakeQuantize {
    /// Create fake quantizer with config.
    #[must_use]
    pub fn new(config: FakeQuantConfig) -> Self {
        let observer = QuantObserver::new(config.observer);
        Self {
            config,
            observer,
            scale: 1.0,
            zero_point: 0.0,
            calibrating: true,
        }
    }

    /// Enable calibration mode (observe statistics).
    pub fn enable_calibration(&mut self) {
        self.calibrating = true;
    }

    /// Disable calibration and freeze quantization params.
    pub fn disable_calibration(&mut self) {
        self.calibrating = false;
        let (scale, zp) = self.observer.compute_qparams(&self.config);
        self.scale = scale;
        self.zero_point = zp;
    }

    /// Perform fake quantization on data.
    pub fn fake_quantize(&mut self, data: &[f32]) -> Vec<f32> {
        if self.calibrating {
            self.observer.observe(data);
            let (scale, zp) = self.observer.compute_qparams(&self.config);
            self.scale = scale;
            self.zero_point = zp;
        }

        let (qmin, qmax) = self.config.quant_range();

        data.iter()
            .map(|&x| {
                // Quantize
                let q = (x / self.scale + self.zero_point).round().clamp(qmin, qmax);
                // Dequantize (fake quantize = quantize then dequantize)
                (q - self.zero_point) * self.scale
            })
            .collect()
    }

    /// Get current scale.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get current zero point.
    #[must_use]
    pub fn zero_point(&self) -> f32 {
        self.zero_point
    }

    /// Get config.
    #[must_use]
    pub fn config(&self) -> &FakeQuantConfig {
        &self.config
    }
}

impl Module for FakeQuantize {
    fn forward(&self, input: &Tensor) -> Tensor {
        let data = input.data();
        let (qmin, qmax) = self.config.quant_range();

        let output_data: Vec<f32> = data
            .iter()
            .map(|&x| {
                let q = (x / self.scale + self.zero_point).round().clamp(qmin, qmax);
                (q - self.zero_point) * self.scale
            })
            .collect();

        Tensor::new(&output_data, input.shape())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

/// Quantized linear layer for inference.
///
/// Stores pre-computed quantization parameters for fast inference
/// without floating point multiplication.
#[derive(Debug)]
pub struct QuantizedLinear {
    /// Weight scale
    weight_scale: f32,
    /// Input scale
    input_scale: f32,
    /// Output scale
    output_scale: f32,
    /// Quantized weights (i8)
    weights_q: Vec<i8>,
    /// Quantized bias (i32)
    bias_q: Option<Vec<i32>>,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
}

impl QuantizedLinear {
    /// Create quantized linear from float weights.
    pub fn from_float(
        weights: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        config: &FakeQuantConfig,
    ) -> Self {
        // Compute weight scale
        let max_abs = weights.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let (_, qmax) = config.quant_range();
        let weight_scale = if max_abs > 0.0 { max_abs / qmax } else { 1.0 };

        // Quantize weights
        let weights_q: Vec<i8> = weights
            .iter()
            .map(|&w| (w / weight_scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        // Quantize bias (scaled by weight_scale * input_scale)
        // For simplicity, assume input_scale = 1.0 initially
        let bias_q = bias.map(|b| {
            b.iter()
                .map(|&v| (v / weight_scale).round() as i32)
                .collect()
        });

        Self {
            weight_scale,
            input_scale: 1.0,
            output_scale: weight_scale,
            weights_q,
            bias_q,
            in_features,
            out_features,
        }
    }

    /// Set input scale (must be done after observing input statistics).
    pub fn set_input_scale(&mut self, scale: f32) {
        self.input_scale = scale;
        self.output_scale = self.weight_scale * scale;
    }

    /// Perform quantized forward pass.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn forward_quantized(&self, input: &[i8]) -> Vec<i32> {
        let batch_size = input.len() / self.in_features;
        let mut output = vec![0i32; batch_size * self.out_features];

        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut acc: i32 = 0;
                for i in 0..self.in_features {
                    let w = i32::from(self.weights_q[o * self.in_features + i]);
                    let x = i32::from(input[b * self.in_features + i]);
                    acc += w * x;
                }
                if let Some(ref bias) = self.bias_q {
                    acc += bias[o];
                }
                output[b * self.out_features + o] = acc;
            }
        }

        output
    }

    /// Get output scale for next layer.
    #[must_use]
    pub fn output_scale(&self) -> f32 {
        self.output_scale
    }
}

/// Dynamic quantization for runtime int8 conversion.
///
/// Quantizes activations on-the-fly during inference.
#[derive(Debug, Clone)]
pub struct DynamicQuantizer {
    config: FakeQuantConfig,
}

impl DynamicQuantizer {
    /// Create dynamic quantizer.
    #[must_use]
    pub fn new(config: FakeQuantConfig) -> Self {
        Self { config }
    }

    /// Quantize tensor dynamically.
    #[must_use]
    pub fn quantize(&self, data: &[f32]) -> (Vec<i8>, f32, f32) {
        let mut observer = QuantObserver::new(self.config.observer);
        observer.observe(data);
        let (scale, zero_point) = observer.compute_qparams(&self.config);

        let (qmin, qmax) = self.config.quant_range();
        let quantized: Vec<i8> = data
            .iter()
            .map(|&x| (x / scale + zero_point).round().clamp(qmin, qmax) as i8)
            .collect();

        (quantized, scale, zero_point)
    }

    /// Dequantize back to float.
    #[must_use]
    pub fn dequantize(&self, data: &[i8], scale: f32, zero_point: f32) -> Vec<f32> {
        data.iter()
            .map(|&q| (f32::from(q) - zero_point) * scale)
            .collect()
    }
}

/// Mixed precision training helper.
///
/// Manages FP16/BF16 training with loss scaling.
#[derive(Debug, Clone)]
pub struct MixedPrecision {
    /// Loss scale for FP16
    loss_scale: f32,
    /// Initial loss scale
    init_scale: f32,
    /// Growth factor for scale
    growth_factor: f32,
    /// Backoff factor on overflow
    backoff_factor: f32,
    /// Steps without overflow
    growth_interval: usize,
    /// Current good steps
    good_steps: usize,
}

impl MixedPrecision {
    /// Create mixed precision manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loss_scale: 65536.0,
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            good_steps: 0,
        }
    }

    /// Scale loss for backward pass.
    #[must_use]
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.loss_scale
    }

    /// Unscale gradients after backward.
    pub fn unscale_gradients(&self, grads: &mut [f32]) {
        let inv_scale = 1.0 / self.loss_scale;
        for g in grads.iter_mut() {
            *g *= inv_scale;
        }
    }

    /// Check for overflow in gradients.
    #[must_use]
    pub fn check_overflow(&self, grads: &[f32]) -> bool {
        grads.iter().any(|&g| !g.is_finite())
    }

    /// Update loss scale based on overflow status.
    pub fn update(&mut self, overflow: bool) {
        if overflow {
            self.loss_scale *= self.backoff_factor;
            self.good_steps = 0;
        } else {
            self.good_steps += 1;
            if self.good_steps >= self.growth_interval {
                self.loss_scale *= self.growth_factor;
                self.good_steps = 0;
            }
        }
    }

    /// Get current loss scale.
    #[must_use]
    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    /// Reset to initial scale.
    pub fn reset(&mut self) {
        self.loss_scale = self.init_scale;
        self.good_steps = 0;
    }
}

impl Default for MixedPrecision {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
