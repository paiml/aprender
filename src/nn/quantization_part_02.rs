use super::*;

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
