//! Convenience functions for fake quantization operations.

use crate::Tensor;

use super::config::FakeQuantConfig;
use super::quantize::FakeQuantize;

/// Convenience function for fake quantization forward pass
pub fn fake_quantize(input: &Tensor, bits: usize, symmetric: bool) -> Tensor {
    let config = if symmetric {
        FakeQuantConfig::symmetric(bits)
    } else {
        FakeQuantConfig::asymmetric(bits)
    };
    let mut fq = FakeQuantize::new(config);
    fq.forward_with_calibration(input)
}

/// Convenience function for STE backward pass
pub fn ste_backward(grad_output: &Tensor) -> Tensor {
    // STE: gradient passes through unchanged
    grad_output.clone()
}
