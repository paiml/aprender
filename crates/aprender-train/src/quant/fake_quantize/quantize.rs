//! Fake quantization operation with Straight-Through Estimator (STE).

use crate::Tensor;

use super::config::FakeQuantConfig;

/// Fake quantization operation with Straight-Through Estimator (STE)
///
/// This struct holds the state for fake quantization including learned
/// or calibrated scale and zero_point parameters.
#[derive(Clone, Debug)]
pub struct FakeQuantize {
    /// Quantization configuration
    pub config: FakeQuantConfig,
    /// Scale factor for quantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Whether scale has been initialized
    pub initialized: bool,
}

impl FakeQuantize {
    /// Create new fake quantization operation
    pub fn new(config: FakeQuantConfig) -> Self {
        Self { config, scale: 1.0, zero_point: 0, initialized: false }
    }

    /// Create with 4-bit symmetric quantization
    pub fn q4() -> Self {
        Self::new(FakeQuantConfig::q4_symmetric())
    }

    /// Create with 8-bit symmetric quantization
    pub fn q8() -> Self {
        Self::new(FakeQuantConfig::q8_symmetric())
    }

    /// Initialize scale from data (min-max calibration)
    ///
    /// For symmetric: scale = max(|min|, |max|) / qmax
    /// For asymmetric: scale = (max - min) / (qmax - qmin)
    pub fn calibrate(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if self.config.symmetric {
            // Symmetric: scale from max absolute value
            let max_abs = min_val.abs().max(max_val.abs());
            self.scale = max_abs / self.config.qmax as f32;
            self.zero_point = 0;
        } else {
            // Asymmetric: scale from range
            self.scale = (max_val - min_val) / (self.config.qmax - self.config.qmin) as f32;
            self.zero_point = (self.config.qmin as f32 - min_val / self.scale).round() as i32;
            self.zero_point = self.zero_point.clamp(self.config.qmin, self.config.qmax);
        }

        // Prevent division by zero
        if self.scale < 1e-10 {
            self.scale = 1e-10;
        }

        self.initialized = true;
    }

    /// Forward pass: fake quantize (quantize → dequantize)
    ///
    /// Simulates quantization effects while keeping values in floating point.
    /// Output = dequantize(quantize(input))
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let data: Vec<f32> = input.data().iter().map(|&x| self.fake_quantize_value(x)).collect();

        Tensor::new(ndarray::arr1(&data), input.requires_grad())
    }

    /// Forward pass with auto-calibration
    ///
    /// If not initialized, calibrates from input data first.
    pub fn forward_with_calibration(&mut self, input: &Tensor) -> Tensor {
        if !self.initialized {
            self.calibrate(input.data().as_slice().unwrap_or(&[]));
        }
        self.forward(input)
    }

    /// Backward pass: Straight-Through Estimator (STE)
    ///
    /// The gradient passes through unchanged:
    /// ∂L/∂x = ∂L/∂y (where y = fake_quantize(x))
    ///
    /// This allows gradients to flow during training despite the
    /// non-differentiable quantization operation.
    pub fn backward(&self, grad_output: &Tensor) -> Tensor {
        // STE: gradient passes through unchanged
        grad_output.clone()
    }

    /// Backward pass with gradient clipping (clamped STE)
    ///
    /// Clips gradients to zero outside the quantization range.
    /// This can improve training stability.
    pub fn backward_clamped(&self, grad_output: &Tensor, input: &Tensor) -> Tensor {
        let qmin_float = self.config.qmin as f32 * self.scale;
        let qmax_float = self.config.qmax as f32 * self.scale;

        let data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(input.data().iter())
            .map(|(&grad, &x)| {
                // Zero gradient outside quantization range
                if x < qmin_float || x > qmax_float {
                    0.0
                } else {
                    grad
                }
            })
            .collect();

        Tensor::new(ndarray::arr1(&data), grad_output.requires_grad())
    }

    /// Fake quantize a single value
    fn fake_quantize_value(&self, x: f32) -> f32 {
        // Quantize
        let q = if self.config.symmetric {
            (x / self.scale).round().clamp(self.config.qmin as f32, self.config.qmax as f32) as i32
        } else {
            ((x / self.scale) + self.zero_point as f32)
                .round()
                .clamp(self.config.qmin as f32, self.config.qmax as f32) as i32
        };

        // Dequantize
        if self.config.symmetric {
            q as f32 * self.scale
        } else {
            (q - self.zero_point) as f32 * self.scale
        }
    }

    /// Get the quantization scale
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get the zero point
    pub fn zero_point(&self) -> i32 {
        self.zero_point
    }

    /// Check if calibrated
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get number of quantization levels
    pub fn num_levels(&self) -> usize {
        (self.config.qmax - self.config.qmin + 1) as usize
    }
}
