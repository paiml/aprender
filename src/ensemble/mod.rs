//! Mixture of Experts (MoE) ensemble learning (GH-101)

mod gating;
mod moe;

pub use gating::{GatingNetwork, SoftmaxGating};
pub use moe::{MixtureOfExperts, MoeConfig};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_network_forward_returns_weights() {
        let gating = SoftmaxGating::new(4, 3);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = gating.forward(&input);
        assert_eq!(weights.len(), 3);
    }

    #[test]
    fn test_gating_network_weights_sum_to_one() {
        let gating = SoftmaxGating::new(4, 3);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = gating.forward(&input);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_gating_low_temperature_peaked() {
        let gating = SoftmaxGating::new(4, 3).with_temperature(0.01);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = gating.forward(&input);
        let max_weight = weights.iter().copied().fold(0.0f32, f32::max);
        assert!(max_weight > 0.9);
    }

    #[test]
    fn test_moe_builder_basic() {
        let gating = SoftmaxGating::new(4, 2);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(1))
            .expert(MockExpert::new(2))
            .build()
            .expect("build");
        assert_eq!(moe.n_experts(), 2);
    }

    #[test]
    fn test_moe_predict_uses_gating() {
        let gating = SoftmaxGating::new(4, 2);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .build()
            .expect("build");
        let output = moe.predict(&[1.0, 2.0, 3.0, 4.0]);
        assert!((10.0..=20.0).contains(&output));
    }

    #[test]
    fn test_moe_top_k_sparse() {
        let gating = SoftmaxGating::new(4, 3);
        let config = MoeConfig::default().with_top_k(1);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .expert(MockExpert::new(30))
            .config(config)
            .build()
            .expect("build");
        let output = moe.predict(&[1.0, 2.0, 3.0, 4.0]);
        assert!(
            (output - 10.0).abs() < 1e-6
                || (output - 20.0).abs() < 1e-6
                || (output - 30.0).abs() < 1e-6
        );
    }

    #[test]
    fn test_moe_save_creates_file() {
        let gating = SoftmaxGating::new(4, 2);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .build()
            .expect("build");
        let tmp = tempfile::NamedTempFile::new().expect("temp");
        moe.save(tmp.path()).expect("save");
        assert!(tmp.path().exists());
    }

    #[test]
    fn test_moe_roundtrip_preserves_predictions() {
        let gating = SoftmaxGating::new(4, 2);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .build()
            .expect("build");
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let original = moe.predict(&input);
        let tmp = tempfile::NamedTempFile::new().expect("temp");
        moe.save(tmp.path()).expect("save");
        let loaded = MixtureOfExperts::<MockExpert, SoftmaxGating>::load(tmp.path()).expect("load");
        let restored = loaded.predict(&input);
        assert!((original - restored).abs() < 1e-6);
    }

    #[test]
    fn test_moe_save_apr_format() {
        let gating = SoftmaxGating::new(4, 2);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .build()
            .expect("build");
        let tmp = tempfile::NamedTempFile::new().expect("temp");
        moe.save_apr(tmp.path()).expect("save apr");
        let bytes = std::fs::read(tmp.path()).expect("read");
        assert_eq!(&bytes[0..4], b"APRN");
    }

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub(crate) struct MockExpert {
        pub(crate) value: f32,
    }

    impl MockExpert {
        fn new(value: i32) -> Self {
            Self {
                value: value as f32,
            }
        }
    }

    impl crate::traits::Estimator for MockExpert {
        fn fit(&mut self, _x: &crate::Matrix<f32>, _y: &crate::Vector<f32>) -> crate::Result<()> {
            Ok(())
        }
        fn predict(&self, _x: &crate::Matrix<f32>) -> crate::Vector<f32> {
            crate::Vector::from_slice(&[self.value])
        }
        fn score(&self, _x: &crate::Matrix<f32>, _y: &crate::Vector<f32>) -> f32 {
            1.0
        }
    }

    // ========== New tests for enhanced MoE features ==========

    #[test]
    fn test_moe_predict_batch() {
        let gating = SoftmaxGating::new(4, 2);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .build()
            .expect("build");

        let inputs = crate::Matrix::from_vec(
            3,
            4,
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5],
        )
        .expect("matrix");

        let predictions = moe.predict_batch(&inputs);
        assert_eq!(predictions.len(), 3);

        // Each prediction should be in range of expert outputs
        for &pred in predictions.as_slice() {
            assert!((10.0..=20.0).contains(&pred));
        }
    }

    #[test]
    fn test_moe_load_balance_loss_computation() {
        let gating = SoftmaxGating::new(4, 3);
        let config = MoeConfig::default()
            .with_top_k(1)
            .with_load_balance_weight(0.1);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .expert(MockExpert::new(30))
            .config(config)
            .build()
            .expect("build");

        let inputs = crate::Matrix::from_vec(
            10,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1,
                1.5, 2.5, 3.5, 4.5, 4.5, 3.5, 2.5, 1.5,
            ],
        )
        .expect("matrix");

        let loss = moe.compute_load_balance_loss(&inputs);
        // Loss should be non-negative
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_moe_load_balance_loss_empty_inputs() {
        let gating = SoftmaxGating::new(4, 2);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .build()
            .expect("build");

        let empty_inputs = crate::Matrix::from_vec(0, 4, vec![]).expect("matrix");
        let loss = moe.compute_load_balance_loss(&empty_inputs);
        assert!((loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_moe_expert_usage_statistics() {
        let gating = SoftmaxGating::new(4, 3);
        let config = MoeConfig::default().with_top_k(2);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .expert(MockExpert::new(30))
            .config(config)
            .build()
            .expect("build");

        let inputs = crate::Matrix::from_vec(
            5,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0,
            ],
        )
        .expect("matrix");

        let usage = moe.expert_usage(&inputs);
        assert_eq!(usage.len(), 3);

        // Usage should sum to top_k (approximately)
        let total: f32 = usage.iter().sum();
        assert!((total - 2.0).abs() < 0.1); // top_k = 2
    }

    #[test]
    fn test_moe_get_routing_weights() {
        let gating = SoftmaxGating::new(4, 3);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .expert(MockExpert::new(30))
            .build()
            .expect("build");

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = moe.get_routing_weights(&input);

        assert_eq!(weights.len(), 3);

        // Weights should sum to 1 (softmax output)
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_moe_fit_succeeds() {
        let gating = SoftmaxGating::new(4, 2);
        let mut moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .build()
            .expect("build");

        let x = crate::Matrix::from_vec(
            5,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0,
            ],
        )
        .expect("matrix");
        let y = crate::Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Fit should succeed (even though it's a no-op in simple impl)
        moe.fit(&x, &y).expect("fit should succeed");
    }

    #[test]
    fn test_moe_config_builder_methods() {
        let config = MoeConfig::default()
            .with_top_k(3)
            .with_capacity_factor(1.5)
            .with_expert_dropout(0.1)
            .with_load_balance_weight(0.05);

        assert_eq!(config.top_k, 3);
        assert!((config.capacity_factor - 1.5).abs() < 1e-6);
        assert!((config.expert_dropout - 0.1).abs() < 1e-6);
        assert!((config.load_balance_weight - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_moe_with_single_expert() {
        let gating = SoftmaxGating::new(4, 1);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(42))
            .build()
            .expect("build");

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = moe.predict(&input);

        // With single expert, output should match expert value
        assert!((output - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_moe_predict_batch_matches_single() {
        let gating = SoftmaxGating::new(4, 2);
        let moe = MixtureOfExperts::<MockExpert, _>::builder()
            .gating(gating)
            .expert(MockExpert::new(10))
            .expert(MockExpert::new(20))
            .build()
            .expect("build");

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let single_pred = moe.predict(&input);

        let batch_input = crate::Matrix::from_vec(1, 4, input).expect("matrix");
        let batch_pred = moe.predict_batch(&batch_input);

        assert!((single_pred - batch_pred.as_slice()[0]).abs() < 1e-6);
    }
}
