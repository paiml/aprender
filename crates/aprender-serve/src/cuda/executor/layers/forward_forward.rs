
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    #[test]
    fn test_forward_to_logits_sequential() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        for position in 0..3 {
            let input = vec![0.5f32; config.hidden_dim];
            let mut logits = vec![0.0f32; config.vocab_size];

            let result = exec.forward_all_layers_gpu_to_logits(
                &input,
                &mut logits,
                position,
                config.num_layers,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                config.vocab_size as u32,
                1e-5,
            );
            let _ = result;
        }
    }

    #[test]
    fn test_forward_with_varying_inputs() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test with different input patterns
        let inputs = [
            vec![0.0f32; config.hidden_dim],
            vec![1.0f32; config.hidden_dim],
            (0..config.hidden_dim)
                .map(|i| (i as f32 / 1000.0).sin())
                .collect::<Vec<_>>(),
        ];

        for input in inputs {
            let mut output = vec![0.0f32; config.hidden_dim];
            let result = exec.forward_all_layers_gpu(
                &input,
                &mut output,
                0,
                config.num_layers,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                1e-5,
            );
            let _ = result;
        }
    }

    #[test]
    fn test_forward_gqa_heads_config() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        // Configure GQA: 32 heads, 8 KV heads (4:1 ratio)
        config.num_heads = 32;
        config.num_kv_heads = 8;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = vec![0.1f32; config.hidden_dim];
        let mut output = vec![0.0f32; config.hidden_dim];

        let result = exec.forward_all_layers_gpu(
            &input,
            &mut output,
            0,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_to_logits_output_check() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = vec![0.1f32; config.hidden_dim];
        let mut logits = vec![-999.0f32; config.vocab_size];

        let result = exec.forward_all_layers_gpu_to_logits(
            &input,
            &mut logits,
            0,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );

        // If successful, logits should have changed from -999
        if result.is_ok() {
            // Check that logits were modified
            let modified = logits.iter().any(|&x| x != -999.0);
            assert!(modified, "Logits should be modified after forward pass");
        }
    }

    #[test]
    fn test_forward_workspace_path() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Ensure workspace is set up
        let has_workspace = exec.has_workspace();
        assert!(has_workspace, "Harness should set up workspace");

        let input = vec![0.1f32; config.hidden_dim];
        let mut output = vec![0.0f32; config.hidden_dim];

        let result = exec.forward_all_layers_gpu(
            &input,
            &mut output,
            0,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }
include!("forward_validation_tests.rs");
}
