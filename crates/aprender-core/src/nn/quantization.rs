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
    pub(crate) weight_scale: f32,
    /// Input scale
    pub(crate) input_scale: f32,
    /// Output scale
    pub(crate) output_scale: f32,
    /// Quantized weights (i8)
    pub(crate) weights_q: Vec<i8>,
    /// Quantized bias (i32)
    pub(crate) bias_q: Option<Vec<i32>>,
    /// Input dimension
    pub(crate) in_features: usize,
    /// Output dimension
    pub(crate) out_features: usize,
}

#[path = "quantization_linear.rs"]
mod quantization_linear;
pub use quantization_linear::*;

#[path = "quantization_tests.rs"]
mod quantization_tests;
