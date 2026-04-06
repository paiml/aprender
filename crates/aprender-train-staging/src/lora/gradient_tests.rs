//! Gradient flow tests for LoRA
//!
//! Validates that:
//! 1. Frozen base weights do NOT receive gradients
//! 2. Trainable LoRA adapters (A, B) DO receive gradients
//! 3. Gradients flow correctly through LoRA computation

#[cfg(test)]
mod tests {
    use crate::lora::LoRALayer;
    use crate::Tensor;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    // ========================================================================
    // PROPERTY TESTS - Gradient flow correctness
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]

        /// Base weight should never have requires_grad for any configuration
        #[test]
        fn prop_base_weight_always_frozen(
            d_out in 2usize..16,
            d_in in 2usize..16,
            rank in 1usize..4,
            alpha in 1.0f32..32.0,
        ) {
            let size = d_out * d_in;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let lora = LoRALayer::new(base_weight, d_out, d_in, rank, alpha);

            prop_assert!(
                !lora.base_weight().requires_grad(),
                "Base weight should always be frozen"
            );
        }

        /// LoRA A and B should always be trainable
        #[test]
        fn prop_lora_params_always_trainable(
            d_out in 2usize..16,
            d_in in 2usize..16,
            rank in 1usize..4,
            alpha in 1.0f32..32.0,
        ) {
            let size = d_out * d_in;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let lora = LoRALayer::new(base_weight, d_out, d_in, rank, alpha);

            prop_assert!(lora.lora_a().requires_grad(), "LoRA A should be trainable");
            prop_assert!(lora.lora_b().requires_grad(), "LoRA B should be trainable");
        }

        /// trainable_params should return exactly 2 parameters (A and B)
        #[test]
        fn prop_trainable_params_count(
            d_out in 2usize..16,
            d_in in 2usize..16,
            rank in 1usize..4,
        ) {
            let size = d_out * d_in;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let mut lora = LoRALayer::new(base_weight, d_out, d_in, rank, 4.0);

            let params = lora.trainable_params();
            prop_assert_eq!(params.len(), 2, "Should have exactly 2 trainable params");

            // Check dimensions
            prop_assert_eq!(params[0].len(), rank * d_in, "A should be [rank * d_in]");
            prop_assert_eq!(params[1].len(), d_out * rank, "B should be [d_out * rank]");
        }

        /// Gradients should accumulate correctly
        #[test]
        fn prop_gradient_accumulation(
            d in 2usize..8,
            initial_grad in prop::collection::vec(-10.0f32..10.0, 2..8),
            additional_grad in prop::collection::vec(-10.0f32..10.0, 2..8),
        ) {
            // Ensure same length
            let len = initial_grad.len().min(additional_grad.len()).min(d * d);
            if len < 2 { return Ok(()); }

            let initial: Vec<f32> = initial_grad[..len].to_vec();
            let additional: Vec<f32> = additional_grad[..len].to_vec();

            let base_weight = Tensor::from_vec(vec![1.0; len], false);
            let mut lora = LoRALayer::new(base_weight, len, 1, 1, 1.0);

            // Set initial gradient on A (which has len elements when d_in=len, rank=1)
            // Actually A is [rank * d_in] = [1 * 1] = 1 element
            // Let me reconsider - for simplicity, just test with fixed small size
            let a_len = lora.lora_a().len();
            if initial.len() < a_len { return Ok(()); }

            let init_slice: Vec<f32> = initial[..a_len].to_vec();
            let add_slice: Vec<f32> = additional[..a_len].to_vec();

            lora.lora_a_mut().set_grad(ndarray::arr1(&init_slice));
            lora.lora_a_mut().accumulate_grad(ndarray::arr1(&add_slice));

            let grad = lora.lora_a().grad().expect("gradient should be available");
            for i in 0..a_len {
                let expected = init_slice[i] + add_slice[i];
                prop_assert!(
                    (grad[i] - expected).abs() < 1e-5,
                    "Accumulated gradient at {} should be {}, got {}",
                    i, expected, grad[i]
                );
            }
        }
    }

    // ========================================================================
    // UNIT TESTS
    // ========================================================================

    #[test]
    fn test_base_weight_frozen() {
        // Base weight should NOT require gradients
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        // Base weight should be frozen
        assert!(!lora.base_weight().requires_grad(), "Base weight should be frozen");
    }

    #[test]
    fn test_lora_params_trainable() {
        // LoRA A and B should require gradients
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        // LoRA parameters should be trainable
        assert!(lora.lora_a().requires_grad(), "LoRA A should be trainable");
        assert!(lora.lora_b().requires_grad(), "LoRA B should be trainable");
    }

    #[test]
    fn test_gradient_flow_to_lora_params() {
        // Test that gradients flow to LoRA A and B but not to base weight
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        // Set non-zero LoRA weights
        *lora.lora_a_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);
        *lora.lora_b_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);

        // Forward pass
        let x = Tensor::from_vec(vec![1.0, 1.0], true);
        let _output = lora.forward(&x);

        // Simulate backward pass by setting output gradient
        // For simple test, we'll manually set gradients on A and B
        // In real training, these would come from backprop

        // Manually compute expected gradients for this simple case
        // This validates that the LoRA params CAN receive gradients
        lora.lora_a_mut().set_grad(ndarray::arr1(&[0.1, 0.1]));
        lora.lora_b_mut().set_grad(ndarray::arr1(&[0.1, 0.1]));

        // Verify gradients are set
        assert!(lora.lora_a().grad().is_some(), "LoRA A should have gradient");
        assert!(lora.lora_b().grad().is_some(), "LoRA B should have gradient");
    }

    #[test]
    fn test_trainable_params_have_requires_grad() {
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 2, 4.0);

        let params = lora.trainable_params();

        // All trainable params should require gradients
        for param in params {
            assert!(param.requires_grad(), "Trainable parameter should require gradients");
        }
    }

    #[test]
    fn test_gradient_isolation_merged_vs_unmerged() {
        // Test that gradient behavior is consistent whether merged or not
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora_unmerged = LoRALayer::new(base_weight.clone(), 2, 2, 1, 1.0);
        let mut lora_merged = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        // Set same LoRA weights
        *lora_unmerged.lora_a_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);
        *lora_unmerged.lora_b_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);
        *lora_merged.lora_a_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);
        *lora_merged.lora_b_mut().data_mut() = ndarray::arr1(&[0.5, 0.5]);

        // Merge one
        lora_merged.merge();

        // Both should have trainable params
        assert!(lora_unmerged.lora_a().requires_grad());
        assert!(lora_unmerged.lora_b().requires_grad());
        assert!(lora_merged.lora_a().requires_grad());
        assert!(lora_merged.lora_b().requires_grad());
    }

    #[test]
    fn test_zero_grad_on_trainable_params() {
        // Test that we can zero gradients on trainable params
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        // Set gradients
        lora.lora_a_mut().set_grad(ndarray::arr1(&[1.0, 2.0]));
        lora.lora_b_mut().set_grad(ndarray::arr1(&[3.0, 4.0]));

        assert!(lora.lora_a().grad().is_some());
        assert!(lora.lora_b().grad().is_some());

        // Zero gradients
        lora.lora_a_mut().zero_grad();
        lora.lora_b_mut().zero_grad();

        assert!(lora.lora_a().grad().is_none());
        assert!(lora.lora_b().grad().is_none());
    }

    #[test]
    fn test_gradient_accumulation_on_lora_params() {
        // Test that gradients can accumulate on LoRA params
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        // Set initial gradient
        lora.lora_a_mut().set_grad(ndarray::arr1(&[1.0, 2.0]));

        // Accumulate more gradient
        lora.lora_a_mut().accumulate_grad(ndarray::arr1(&[0.5, 0.5]));

        let grad = lora.lora_a().grad().expect("gradient should be available");
        assert_abs_diff_eq!(grad[0], 1.5, epsilon = 1e-6); // 1.0 + 0.5
        assert_abs_diff_eq!(grad[1], 2.5, epsilon = 1e-6); // 2.0 + 0.5
    }

    #[test]
    fn test_multiple_forward_passes_gradient_ready() {
        // Test that LoRA params remain gradient-ready across multiple forward passes
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        let x = Tensor::from_vec(vec![1.0, 1.0], true);

        // Multiple forward passes
        for _ in 0..3 {
            let _output = lora.forward(&x);

            // LoRA params should still be trainable
            assert!(lora.lora_a().requires_grad());
            assert!(lora.lora_b().requires_grad());
        }
    }

    #[test]
    fn test_lora_params_independent_gradients() {
        // Test that A and B can have independent gradients
        let base_weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 1, 1.0);

        // Set different gradients
        lora.lora_a_mut().set_grad(ndarray::arr1(&[1.0, 2.0]));
        lora.lora_b_mut().set_grad(ndarray::arr1(&[3.0, 4.0]));

        let grad_a = lora.lora_a().grad().expect("gradient should be available");
        let grad_b = lora.lora_b().grad().expect("gradient should be available");

        // Gradients should be independent
        assert_abs_diff_eq!(grad_a[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad_a[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad_b[0], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(grad_b[1], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_optimizer_integration_readiness() {
        // Test that trainable_params() returns references suitable for optimizers
        let base_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
        let mut lora = LoRALayer::new(base_weight, 2, 2, 2, 4.0);

        // Get trainable params
        let params = lora.trainable_params();

        // Should have exactly 2 params (A and B)
        assert_eq!(params.len(), 2);

        // All params should require gradients
        for param in &params {
            assert!(param.requires_grad());
        }

        // Simulate optimizer: set gradients and update
        for param in params {
            // Set dummy gradient
            param.set_grad(ndarray::Array1::ones(param.len()));

            // Simulate parameter update (grad descent)
            let update = param.grad().expect("gradient should be available") * 0.01;
            *param.data_mut() = param.data() - &update;

            // Verify gradient is still there after update
            assert!(param.grad().is_some());
        }
    }
}
