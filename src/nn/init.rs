//! Weight initialization functions.
//!
//! Proper initialization is critical for training deep networks.
//! This module provides initialization schemes from the literature:
//!
//! - Xavier/Glorot (Glorot & Bengio, 2010) - for tanh/sigmoid activations
//! - Kaiming/He (He et al., 2015) - for `ReLU` activations
//!
//! # References
//!
//! - Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training
//!   deep feedforward neural networks. AISTATS.
//! - He, K., et al. (2015). Delving deep into rectifiers: Surpassing human-level
//!   performance on `ImageNet` classification. ICCV.

use crate::autograd::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Xavier uniform initialization (Glorot & Bengio, 2010).
///
/// Samples from U(-a, a) where a = sqrt(6 / (`fan_in` + `fan_out`)).
/// Suitable for tanh and sigmoid activations.
///
/// # Arguments
///
/// * `shape` - Shape of the tensor to initialize
/// * `fan_in` - Number of input features
/// * `fan_out` - Number of output features
/// * `seed` - Optional random seed for reproducibility
///
/// # Example
///
/// ```ignore
/// use aprender::nn::init::xavier_uniform;
///
/// // Initialize weight for layer with 784 inputs and 256 outputs
/// let weight = xavier_uniform(&[256, 784], 784, 256, None);
/// ```
#[must_use] 
pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize, seed: Option<u64>) -> Tensor {
    let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
    uniform(shape, -a, a, seed)
}

/// Xavier normal initialization (Glorot & Bengio, 2010).
///
/// Samples from N(0, std) where std = sqrt(2 / (`fan_in` + `fan_out`)).
/// Suitable for tanh and sigmoid activations.
#[must_use] 
pub fn xavier_normal(shape: &[usize], fan_in: usize, fan_out: usize, seed: Option<u64>) -> Tensor {
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    normal(shape, 0.0, std, seed)
}

/// Kaiming uniform initialization (He et al., 2015).
///
/// Samples from U(-bound, bound) where bound = sqrt(6 / `fan_in`).
/// Optimal for `ReLU` activations.
///
/// # Arguments
///
/// * `shape` - Shape of the tensor
/// * `fan_in` - Number of input features
/// * `seed` - Optional random seed
#[must_use] 
pub fn kaiming_uniform(shape: &[usize], fan_in: usize, seed: Option<u64>) -> Tensor {
    let bound = (6.0 / fan_in as f32).sqrt();
    uniform(shape, -bound, bound, seed)
}

/// Kaiming normal initialization (He et al., 2015).
///
/// Samples from N(0, std) where std = sqrt(2 / `fan_in`).
/// Optimal for `ReLU` activations.
#[must_use] 
pub fn kaiming_normal(shape: &[usize], fan_in: usize, seed: Option<u64>) -> Tensor {
    let std = (2.0 / fan_in as f32).sqrt();
    normal(shape, 0.0, std, seed)
}

/// Uniform distribution initialization.
///
/// Samples from U(low, high).
pub(crate) fn uniform(shape: &[usize], low: f32, high: f32, seed: Option<u64>) -> Tensor {
    let numel: usize = shape.iter().product();
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(low..high)).collect();

    Tensor::new(&data, shape)
}

/// Normal distribution initialization.
///
/// Samples from N(mean, std).
pub(crate) fn normal(shape: &[usize], mean: f32, std: f32, seed: Option<u64>) -> Tensor {
    let numel: usize = shape.iter().product();
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Box-Muller transform for normal distribution
    let data: Vec<f32> = (0..numel)
        .map(|_| {
            let u1: f32 = rng.gen_range(0.0001_f32..1.0_f32);
            let u2: f32 = rng.gen_range(0.0_f32..1.0_f32);
            let z = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
            mean + std * z
        })
        .collect();

    Tensor::new(&data, shape)
}

/// Constant initialization.
pub(crate) fn constant(shape: &[usize], value: f32) -> Tensor {
    let numel: usize = shape.iter().product();
    Tensor::new(&vec![value; numel], shape)
}

/// Zeros initialization.
pub(crate) fn zeros(shape: &[usize]) -> Tensor {
    constant(shape, 0.0)
}

/// Ones initialization.
#[allow(dead_code)]
pub(crate) fn ones(shape: &[usize]) -> Tensor {
    constant(shape, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_uniform_bounds() {
        let t = xavier_uniform(&[100, 100], 100, 100, Some(42));
        let a = (6.0 / 200.0_f32).sqrt();

        for &val in t.data() {
            assert!(
                (-a..=a).contains(&val),
                "Value {val} out of bounds [-{a}, {a}]"
            );
        }
    }

    #[test]
    fn test_xavier_uniform_reproducible() {
        let t1 = xavier_uniform(&[10, 10], 10, 10, Some(42));
        let t2 = xavier_uniform(&[10, 10], 10, 10, Some(42));

        assert_eq!(t1.data(), t2.data());
    }

    #[test]
    fn test_kaiming_uniform_bounds() {
        let t = kaiming_uniform(&[100, 50], 50, Some(42));
        let bound = (6.0 / 50.0_f32).sqrt();

        for &val in t.data() {
            assert!(val >= -bound && val <= bound);
        }
    }

    #[test]
    fn test_normal_mean_std() {
        let t = normal(&[10000], 5.0, 2.0, Some(42));

        let mean: f32 = t.data().iter().sum::<f32>() / t.numel() as f32;
        let var: f32 = t.data().iter().map(|x| (x - mean).powi(2)).sum::<f32>() / t.numel() as f32;
        let std = var.sqrt();

        // Allow 10% tolerance for statistical tests
        assert!((mean - 5.0).abs() < 0.5, "Mean {mean} too far from 5.0");
        assert!((std - 2.0).abs() < 0.3, "Std {std} too far from 2.0");
    }

    #[test]
    fn test_zeros_ones() {
        let z = zeros(&[3, 3]);
        assert!(z.data().iter().all(|&x| x == 0.0));

        let o = ones(&[3, 3]);
        assert!(o.data().iter().all(|&x| x == 1.0));
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_xavier_normal_distribution() {
        let t = xavier_normal(&[1000], 100, 100, Some(42));
        let std = (2.0 / 200.0_f32).sqrt();

        // Check mean is close to 0
        let mean: f32 = t.data().iter().sum::<f32>() / t.numel() as f32;
        assert!((mean).abs() < 0.1, "Mean {mean} too far from 0");

        // Check std is reasonable
        let variance: f32 =
            t.data().iter().map(|x| (x - mean).powi(2)).sum::<f32>() / t.numel() as f32;
        let actual_std = variance.sqrt();
        assert!(
            (actual_std - std).abs() < 0.05,
            "Std {actual_std} too far from {std}"
        );
    }

    #[test]
    fn test_xavier_normal_reproducible() {
        let t1 = xavier_normal(&[10, 10], 10, 10, Some(42));
        let t2 = xavier_normal(&[10, 10], 10, 10, Some(42));
        assert_eq!(t1.data(), t2.data());
    }

    #[test]
    fn test_kaiming_normal_distribution() {
        let t = kaiming_normal(&[1000], 100, Some(42));
        let expected_std = (2.0 / 100.0_f32).sqrt();

        let mean: f32 = t.data().iter().sum::<f32>() / t.numel() as f32;
        assert!((mean).abs() < 0.1, "Mean {mean} too far from 0");

        let variance: f32 =
            t.data().iter().map(|x| (x - mean).powi(2)).sum::<f32>() / t.numel() as f32;
        let actual_std = variance.sqrt();
        assert!(
            (actual_std - expected_std).abs() < 0.05,
            "Std {actual_std} too far from {expected_std}"
        );
    }

    #[test]
    fn test_constant_initialization() {
        let t = constant(&[5, 5], 3.14);
        assert!(t.data().iter().all(|&x| (x - 3.14).abs() < 1e-6));
        assert_eq!(t.numel(), 25);
    }

    #[test]
    fn test_uniform_no_seed() {
        // Without seed, should still work (entropy-based)
        let t1 = uniform(&[100], 0.0, 1.0, None);
        let t2 = uniform(&[100], 0.0, 1.0, None);

        // Very unlikely to be identical
        let same = t1
            .data()
            .iter()
            .zip(t2.data())
            .all(|(a, b)| (a - b).abs() < 1e-10);
        assert!(!same, "Two entropy-seeded tensors should differ");
    }

    #[test]
    fn test_normal_no_seed() {
        // Without seed, should still work (entropy-based)
        let t1 = normal(&[100], 0.0, 1.0, None);
        let t2 = normal(&[100], 0.0, 1.0, None);

        let same = t1
            .data()
            .iter()
            .zip(t2.data())
            .all(|(a, b)| (a - b).abs() < 1e-10);
        assert!(!same, "Two entropy-seeded tensors should differ");
    }
}
