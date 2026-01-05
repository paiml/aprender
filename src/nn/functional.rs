//! Functional interface for neural network operations.
//!
//! This module provides stateless functions that mirror the module-based
//! activations and operations. Use these when you don't need a module
//! wrapper (e.g., in custom forward passes).
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::F;
//! use aprender::autograd::Tensor;
//!
//! let x = Tensor::randn(&[32, 10]);
//! let y = F::relu(&x);
//! let probs = F::softmax(&y, -1);
//! ```

use crate::autograd::Tensor;

/// `ReLU` activation: max(0, x)
#[must_use] 
pub fn relu(x: &Tensor) -> Tensor {
    x.relu()
}

/// Leaky `ReLU` activation: `max(negative_slope` * x, x)
#[must_use] 
pub fn leaky_relu(x: &Tensor, negative_slope: f32) -> Tensor {
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| if v > 0.0 { v } else { negative_slope * v })
        .collect();
    Tensor::new(&data, x.shape())
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[must_use] 
pub fn sigmoid(x: &Tensor) -> Tensor {
    x.sigmoid()
}

/// Tanh activation
#[must_use] 
pub fn tanh(x: &Tensor) -> Tensor {
    x.tanh_()
}

/// GELU activation (Gaussian Error Linear Unit)
#[must_use] 
pub fn gelu(x: &Tensor) -> Tensor {
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();

    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| {
            let inner = sqrt_2_over_pi * (v + 0.044715 * v.powi(3));
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();

    Tensor::new(&data, x.shape())
}

/// Softmax along a dimension
///
/// Currently only supports 2D tensors with dim=-1 (last dimension).
#[must_use] 
pub fn softmax(x: &Tensor, _dim: i32) -> Tensor {
    assert_eq!(x.ndim(), 2, "softmax currently only supports 2D tensors");

    let (batch, features) = (x.shape()[0], x.shape()[1]);
    let mut output = vec![0.0; batch * features];

    for b in 0..batch {
        let row_start = b * features;

        // Numerical stability: subtract max
        let max_val = x.data()[row_start..row_start + features]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let mut sum = 0.0;
        for j in 0..features {
            let exp_val = (x.data()[row_start + j] - max_val).exp();
            output[row_start + j] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for j in 0..features {
            output[row_start + j] /= sum;
        }
    }

    Tensor::new(&output, x.shape())
}

/// Log softmax along a dimension
///
/// More numerically stable than log(softmax(x)).
#[must_use] 
pub fn log_softmax(x: &Tensor, _dim: i32) -> Tensor {
    assert_eq!(
        x.ndim(),
        2,
        "log_softmax currently only supports 2D tensors"
    );

    let (batch, features) = (x.shape()[0], x.shape()[1]);
    let mut output = vec![0.0; batch * features];

    for b in 0..batch {
        let row_start = b * features;

        // Numerical stability: subtract max
        let max_val = x.data()[row_start..row_start + features]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute log(sum(exp(x - max)))
        let log_sum_exp: f32 = x.data()[row_start..row_start + features]
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum::<f32>()
            .ln();

        // log_softmax = x - max - log_sum_exp
        for j in 0..features {
            output[row_start + j] = x.data()[row_start + j] - max_val - log_sum_exp;
        }
    }

    Tensor::new(&output, x.shape())
}

/// Dropout (must be called with training flag)
#[must_use] 
pub fn dropout(x: &Tensor, p: f32, training: bool) -> Tensor {
    if !training || p == 0.0 {
        return x.clone();
    }

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let scale = 1.0 / (1.0 - p);

    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| if rng.gen::<f32>() < p { 0.0 } else { v * scale })
        .collect();

    Tensor::new(&data, x.shape())
}

/// Linear transformation: y = x @ weight^T + bias
#[must_use] 
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let weight_t = weight.transpose();
    let output = x.matmul(&weight_t);

    match bias {
        Some(b) => broadcast_add_1d(&output, b),
        None => output,
    }
}

/// Helper: broadcast-add 1D bias to 2D output
fn broadcast_add_1d(matrix: &Tensor, vector: &Tensor) -> Tensor {
    let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
    let mut result = vec![0.0; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            result[i * cols + j] = matrix.data()[i * cols + j] + vector.data()[j];
        }
    }

    Tensor::new(&result, &[rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_functional() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0]);
        let y = relu(&x);
        assert_eq!(y.data(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_leaky_relu_functional() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0]);
        let y = leaky_relu(&x, 0.1);
        assert_eq!(y.data(), &[-0.1, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_functional() {
        let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
        let y = softmax(&x, -1);

        // Should sum to 1
        let sum: f32 = y.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_functional() {
        let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
        let y = log_softmax(&x, -1);

        // exp(log_softmax) should sum to 1
        let sum: f32 = y.data().iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_functional() {
        // Identity weight, no bias
        let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
        let weight = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);

        let y = linear(&x, &weight, None);
        assert_eq!(y.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_linear_functional_with_bias() {
        let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
        let weight = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let bias = Tensor::new(&[10.0, 20.0], &[2]);

        let y = linear(&x, &weight, Some(&bias));
        assert_eq!(y.data(), &[11.0, 22.0]);
    }

    #[test]
    fn test_dropout_eval() {
        let x = Tensor::ones(&[100]);
        let y = dropout(&x, 0.5, false); // eval mode
        assert_eq!(y.data(), x.data());
    }

    // =========================================================================
    // Additional coverage tests for functional interface
    // =========================================================================

    #[test]
    fn test_sigmoid_functional() {
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
        let y = sigmoid(&x);
        // sigmoid(0) = 0.5
        assert!((y.data()[0] - 0.5).abs() < 1e-5);
        // sigmoid(x) is always in (0, 1)
        for &val in y.data() {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_tanh_functional() {
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
        let y = tanh(&x);
        // tanh(0) = 0
        assert!((y.data()[0]).abs() < 1e-5);
        // tanh is in (-1, 1)
        for &val in y.data() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_gelu_functional() {
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
        let y = gelu(&x);
        // GELU(0) = 0
        assert!((y.data()[0]).abs() < 1e-5);
        // GELU(1) ≈ 0.841
        assert!((y.data()[1] - 0.841).abs() < 0.01);
        // GELU(-1) ≈ -0.158
        assert!((y.data()[2] - (-0.158)).abs() < 0.02);
    }

    #[test]
    fn test_dropout_training_zeros_some() {
        // With training=true and p=0.5, some values should be zeroed
        let x = Tensor::ones(&[1000]);
        let y = dropout(&x, 0.5, true);

        let zeros = y.data().iter().filter(|&&v| v == 0.0).count();
        let scaled = y
            .data()
            .iter()
            .filter(|&&v| (v - 2.0).abs() < 0.001)
            .count();

        // With p=0.5, roughly half should be zero, half should be scaled by 2
        assert!(
            zeros > 300 && zeros < 700,
            "Expected ~500 zeros, got {zeros}"
        );
        assert!(
            scaled > 300 && scaled < 700,
            "Expected ~500 scaled, got {scaled}"
        );
    }

    #[test]
    fn test_dropout_zero_probability() {
        let x = Tensor::ones(&[100]);
        let y = dropout(&x, 0.0, true); // p=0 means no dropout
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_softmax_multi_batch() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = softmax(&x, -1);

        // Each row should sum to 1
        let sum1: f32 = y.data()[0..3].iter().sum();
        let sum2: f32 = y.data()[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_multi_batch() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = log_softmax(&x, -1);

        // exp(log_softmax) should sum to 1 for each row
        let sum1: f32 = y.data()[0..3].iter().map(|v| v.exp()).sum();
        let sum2: f32 = y.data()[3..6].iter().map(|v| v.exp()).sum();
        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_batch() {
        // Test linear with batch dimension
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let weight = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let bias = Tensor::new(&[10.0, 20.0], &[2]);

        let y = linear(&x, &weight, Some(&bias));
        // Row 1: [1,2,3] @ [[1,0],[0,1],[0,0]]^T + [10,20] = [1,2] + [10,20] = [11,22]
        assert!((y.data()[0] - 11.0).abs() < 1e-5);
        assert!((y.data()[1] - 22.0).abs() < 1e-5);
    }
}
