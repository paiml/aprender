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

/// ReLU activation: max(0, x)
pub fn relu(x: &Tensor) -> Tensor {
    x.relu()
}

/// Leaky ReLU activation: max(negative_slope * x, x)
pub fn leaky_relu(x: &Tensor, negative_slope: f32) -> Tensor {
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| if v > 0.0 { v } else { negative_slope * v })
        .collect();
    Tensor::new(&data, x.shape())
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(x: &Tensor) -> Tensor {
    x.sigmoid()
}

/// Tanh activation
pub fn tanh(x: &Tensor) -> Tensor {
    x.tanh_()
}

/// GELU activation (Gaussian Error Linear Unit)
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
}
