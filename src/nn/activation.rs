//! Activation function modules.
//!
//! These modules wrap activation functions for use in Sequential containers.
//! For functional versions, see `nn::functional`.
//!
//! # References
//!
//! - Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted
//!   Boltzmann machines. ICML.
//! - He, K., et al. (2015). Delving deep into rectifiers. ICCV.

use super::module::Module;
use crate::autograd::Tensor;

/// Rectified Linear Unit activation: ReLU(x) = max(0, x)
///
/// # Shape
///
/// - Input: `(*)` any shape
/// - Output: `(*)` same shape as input
///
/// # Example
///
/// ```ignore
/// use aprender::nn::{Module, ReLU};
/// use aprender::autograd::Tensor;
///
/// let relu = ReLU::new();
/// let x = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0]);
/// let y = relu.forward(&x);  // [0.0, 0.0, 1.0, 2.0]
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU;

impl ReLU {
    /// Create a new ReLU activation.
    pub fn new() -> Self {
        Self
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }
}

/// Leaky ReLU activation: LeakyReLU(x) = max(negative_slope * x, x)
///
/// # Arguments
///
/// * `negative_slope` - Controls angle of negative slope (default: 0.01)
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    negative_slope: f32,
}

impl LeakyReLU {
    /// Create a new LeakyReLU with default negative slope (0.01).
    pub fn new() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }

    /// Create a new LeakyReLU with specified negative slope.
    pub fn with_slope(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.leaky_relu(self.negative_slope)
    }
}

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
///
/// Maps inputs to (0, 1) range.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
    }
}

/// Tanh activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
///
/// Maps inputs to (-1, 1) range.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.tanh_()
    }
}

/// Gaussian Error Linear Unit (GELU) activation.
///
/// GELU(x) = x * Φ(x) where Φ is the CDF of standard normal.
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// # Reference
///
/// - Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
#[derive(Debug, Clone, Copy, Default)]
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }
}

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.gelu()
    }
}

/// Softmax activation: softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
///
/// Converts logits to probabilities that sum to 1.
///
/// # Arguments
///
/// * `dim` - Dimension along which to compute softmax
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    #[allow(dead_code)]
    dim: i32,
}

impl Softmax {
    /// Create a new Softmax along the specified dimension.
    pub fn new(dim: i32) -> Self {
        Self { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.softmax()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let relu = ReLU::new();
        let x = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = relu.forward(&x);

        assert_eq!(y.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let lrelu = LeakyReLU::with_slope(0.1);
        let x = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = lrelu.forward(&x);

        assert_eq!(y.data(), &[-0.2, -0.1, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid::new();
        let x = Tensor::from_slice(&[0.0]);
        let y = sigmoid.forward(&x);

        assert!((y.data()[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_bounds() {
        let sigmoid = Sigmoid::new();
        let x = Tensor::from_slice(&[-10.0, 0.0, 10.0]);
        let y = sigmoid.forward(&x);

        // Should be in (0, 1)
        for &val in y.data() {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_tanh() {
        let tanh = Tanh::new();
        let x = Tensor::from_slice(&[0.0]);
        let y = tanh.forward(&x);

        assert!((y.data()[0]).abs() < 1e-5);
    }

    #[test]
    fn test_tanh_bounds() {
        let tanh = Tanh::new();
        let x = Tensor::from_slice(&[-2.0, 0.0, 2.0]);
        let y = tanh.forward(&x);

        // Should be in (-1, 1)
        for &val in y.data() {
            assert!((-1.0..=1.0).contains(&val));
        }

        // More specific bounds for non-extreme values
        assert!(y.data()[0] > -1.0 && y.data()[0] < -0.9); // tanh(-2) ≈ -0.964
        assert!(y.data()[2] > 0.9 && y.data()[2] < 1.0); // tanh(2) ≈ 0.964
    }

    #[test]
    fn test_gelu() {
        let gelu = GELU::new();
        let x = Tensor::from_slice(&[0.0]);
        let y = gelu.forward(&x);

        // GELU(0) = 0
        assert!((y.data()[0]).abs() < 1e-5);
    }

    #[test]
    fn test_gelu_positive() {
        let gelu = GELU::new();
        let x = Tensor::from_slice(&[1.0]);
        let y = gelu.forward(&x);

        // GELU(1) ≈ 0.841
        assert!((y.data()[0] - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let softmax = Softmax::new(-1);
        let x = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let y = softmax.forward(&x);

        // Each row should sum to 1
        let (batch, features) = (2, 3);
        for b in 0..batch {
            let sum: f32 = (0..features).map(|j| y.data()[b * features + j]).sum();
            assert!((sum - 1.0).abs() < 1e-5, "Row {b} sums to {sum}");
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let softmax = Softmax::new(-1);
        // Large values that could cause overflow without proper handling
        let x = Tensor::new(&[1000.0, 1001.0, 1002.0], &[1, 3]);
        let y = softmax.forward(&x);

        // Should not have NaN or Inf
        for &val in y.data() {
            assert!(val.is_finite());
            assert!((0.0..=1.0).contains(&val));
        }

        // Should still sum to 1
        let sum: f32 = y.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
