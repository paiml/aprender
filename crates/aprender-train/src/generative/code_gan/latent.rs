//! Latent code representation for GAN latent space operations.

use rand::Rng;

/// Latent code representation (vector in latent space)
#[derive(Debug, Clone, PartialEq)]
pub struct LatentCode {
    /// The latent vector
    pub vector: Vec<f32>,
}

impl LatentCode {
    /// Create a new latent code from a vector
    #[must_use]
    pub fn new(vector: Vec<f32>) -> Self {
        Self { vector }
    }

    /// Sample from standard normal distribution using Box-Muller transform
    pub fn sample<R: Rng>(rng: &mut R, dim: usize) -> Self {
        contract_pre_sample!();
        let vector: Vec<f32> = (0..dim)
            .map(|_| {
                // Box-Muller transform for standard normal
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random::<f64>();
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
            })
            .collect();
        Self { vector }
    }

    /// Dimension of the latent code
    #[must_use]
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Linear interpolation between two latent codes
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        assert_eq!(self.dim(), other.dim(), "Latent dimensions must match");
        let vector =
            self.vector.iter().zip(&other.vector).map(|(a, b)| a * (1.0 - t) + b * t).collect();
        Self { vector }
    }

    /// Spherical linear interpolation between two latent codes
    #[must_use]
    pub fn slerp(&self, other: &Self, t: f32) -> Self {
        assert_eq!(self.dim(), other.dim(), "Latent dimensions must match");

        let norm_self: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_other: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Fall back to lerp if either vector has near-zero norm
        if norm_self < 1e-10 || norm_other < 1e-10 {
            return self.lerp(other, t);
        }

        let dot: f32 = self.vector.iter().zip(&other.vector).map(|(a, b)| a * b).sum();

        let cos_omega = (dot / (norm_self * norm_other)).clamp(-1.0, 1.0);
        let omega = cos_omega.acos();

        if omega.abs() < 1e-6 {
            return self.lerp(other, t);
        }

        let sin_omega = omega.sin();
        let factor_self = ((1.0 - t) * omega).sin() / sin_omega;
        let factor_other = (t * omega).sin() / sin_omega;

        let vector = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| a * factor_self + b * factor_other)
            .collect();

        Self { vector }
    }

    /// Compute L2 norm
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.vector.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length
    #[must_use]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-10 {
            return self.clone();
        }
        let vector = self.vector.iter().map(|x| x / n).collect();
        Self { vector }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_latent_code_creation() {
        let code = LatentCode::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(code.dim(), 3);
        assert_eq!(code.vector, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_latent_code_sample() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let code = LatentCode::sample(&mut rng, 128);
        assert_eq!(code.dim(), 128);
    }

    #[test]
    fn test_latent_code_lerp() {
        let z1 = LatentCode::new(vec![0.0, 0.0]);
        let z2 = LatentCode::new(vec![1.0, 1.0]);

        let mid = z1.lerp(&z2, 0.5);
        assert!((mid.vector[0] - 0.5).abs() < 1e-6);
        assert!((mid.vector[1] - 0.5).abs() < 1e-6);

        let start = z1.lerp(&z2, 0.0);
        assert!((start.vector[0] - 0.0).abs() < 1e-6);

        let end = z1.lerp(&z2, 1.0);
        assert!((end.vector[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_latent_code_slerp() {
        let z1 = LatentCode::new(vec![1.0, 0.0]);
        let z2 = LatentCode::new(vec![0.0, 1.0]);

        let mid = z1.slerp(&z2, 0.5);
        // At midpoint, should have roughly equal components
        assert!((mid.vector[0] - mid.vector[1]).abs() < 0.1);
    }

    #[test]
    fn test_latent_code_norm() {
        let code = LatentCode::new(vec![3.0, 4.0]);
        assert!((code.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_latent_code_normalize() {
        let code = LatentCode::new(vec![3.0, 4.0]);
        let normalized = code.normalize();
        assert!((normalized.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_slerp_maintains_norm() {
        let z1 = LatentCode::new(vec![1.0, 0.0, 0.0]).normalize();
        let z2 = LatentCode::new(vec![0.0, 1.0, 0.0]).normalize();

        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let z = z1.slerp(&z2, t);
            // SLERP should maintain approximate unit norm
            assert!((z.norm() - 1.0).abs() < 0.1);
        }
    }

    proptest! {
        #[test]
        fn test_latent_lerp_bounds(t in 0.0f32..=1.0) {
            let z1 = LatentCode::new(vec![0.0, 0.0, 0.0]);
            let z2 = LatentCode::new(vec![1.0, 1.0, 1.0]);

            let result = z1.lerp(&z2, t);

            for v in &result.vector {
                prop_assert!(*v >= 0.0 && *v <= 1.0);
            }
        }

        #[test]
        fn test_latent_norm_non_negative(values in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let code = LatentCode::new(values);
            prop_assert!(code.norm() >= 0.0);
        }

        #[test]
        fn test_normalize_unit_length(values in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let code = LatentCode::new(values);
            if code.norm() > 1e-6 {
                let normalized = code.normalize();
                prop_assert!((normalized.norm() - 1.0).abs() < 1e-5);
            }
        }
    }
}
