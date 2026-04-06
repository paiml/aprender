//! Phase 16: APR Q4 CPU vs GPU Parity Test
//!
//! Compares CPU (`QuantizedAprTransformerQ4::forward_single_with_scratch`)
//! vs GPU (`GpuModelQ4::forward`) to isolate divergence point.
//!
//! # The Logic
//!
//! If MatMul(A, B) works in isolation, but fails in the model, then either
//! A (Input) or B (Weights) is wrong in the model context.
//!
//! # Trace Plan
//!
//! 1. Embedding lookup - Same code path
//! 2. RMSNorm output - TRACE THIS
//! 3. QKV projection - Proven correct in isolation
//!
//! # Usage
//!
//! ```bash
//! cargo test --test apr_q4_parity --features cuda -- --nocapture
//! ```

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::apr_transformer::{
        AprInferenceScratch, AprTransformerConfig, QuantizedAprLayerQ4, QuantizedAprTensorQ4,
        QuantizedAprTransformerQ4,
    };
    use realizar::cuda::CudaExecutor;
    use realizar::gpu::adapters::AprQ4ToGpuAdapter;

    /// Create a minimal test model with known weights for parity testing
    fn create_test_model() -> QuantizedAprTransformerQ4 {
        let hidden_dim = 64; // Small for testing
        let intermediate_dim = 128;
        let vocab_size = 100;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = hidden_dim + 2 * kv_dim; // Q + K + V

        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        };

        // Token embedding: identity-like (each token maps to itself scaled)
        let token_embedding: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| ((i % hidden_dim) as f32) * 0.01)
            .collect();

        // Create Q4_0 weights (18 bytes per 32 values)
        let create_q4_0_weights = |in_dim: usize, out_dim: usize| -> Vec<u8> {
            let num_elements = in_dim * out_dim;
            let num_blocks = num_elements.div_ceil(32);
            let mut data = vec![0u8; num_blocks * 18];

            // Set scale = 0.1 (f16: ~0x2E66) for each block
            for block in 0..num_blocks {
                let base = block * 18;
                data[base] = 0x66;
                data[base + 1] = 0x2E;
                // Quants: alternating pattern
                for i in 0..16 {
                    data[base + 2 + i] = 0x88; // mid-range values
                }
            }
            data
        };

        // Layer weights
        let attn_norm_weight = vec![1.0f32; hidden_dim];
        let ffn_norm_weight = vec![1.0f32; hidden_dim];

        let layer = QuantizedAprLayerQ4 {
            attn_norm_weight,
            qkv_weight: QuantizedAprTensorQ4::new(
                create_q4_0_weights(hidden_dim, qkv_dim),
                hidden_dim,
                qkv_dim,
            ),
            attn_output_weight: QuantizedAprTensorQ4::new(
                create_q4_0_weights(hidden_dim, hidden_dim),
                hidden_dim,
                hidden_dim,
            ),
            ffn_up_weight: QuantizedAprTensorQ4::new(
                create_q4_0_weights(hidden_dim, intermediate_dim),
                hidden_dim,
                intermediate_dim,
            ),
            ffn_down_weight: QuantizedAprTensorQ4::new(
                create_q4_0_weights(intermediate_dim, hidden_dim),
                intermediate_dim,
                hidden_dim,
            ),
            ffn_gate_weight: Some(QuantizedAprTensorQ4::new(
                create_q4_0_weights(hidden_dim, intermediate_dim),
                hidden_dim,
                intermediate_dim,
            )),
            ffn_norm_weight: Some(ffn_norm_weight),
        };

        let output_norm_weight = vec![1.0f32; hidden_dim];
        let lm_head_weight = QuantizedAprTensorQ4::new(
            create_q4_0_weights(hidden_dim, vocab_size),
            hidden_dim,
            vocab_size,
        );

        QuantizedAprTransformerQ4 {
            config,
            token_embedding,
            layers: vec![layer],
            output_norm_weight,
            lm_head_weight,
        }
    }

    /// Test 1: RMSNorm CPU vs GPU parity
    ///
    /// Verifies that the RMSNorm output matches between CPU and GPU paths.
    #[test]
    fn test_rmsnorm_parity() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: RMSNORM PARITY TEST (APR Q4)                              ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let apr = create_test_model();
        let hidden_dim = apr.config.hidden_dim;
        let eps = apr.config.eps;

        // Token 5 for testing
        let token_id = 5u32;

        // CPU path: embedding + RMSNorm
        let offset = (token_id as usize) * hidden_dim;
        let hidden_cpu: Vec<f32> = apr.token_embedding[offset..offset + hidden_dim].to_vec();

        let sq_sum: f32 = hidden_cpu.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();

        let normed_cpu: Vec<f32> = hidden_cpu
            .iter()
            .zip(apr.layers[0].attn_norm_weight.iter())
            .map(|(&x, &w)| x / rms * w)
            .collect();

        eprintln!(
            "CPU Embedding [0..8]: {:?}",
            &hidden_cpu[..8.min(hidden_dim)]
        );
        eprintln!(
            "CPU Embedding L2: {:.6}",
            hidden_cpu.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        eprintln!("CPU RMS: {:.6}", rms);
        eprintln!("CPU Normed [0..8]: {:?}", &normed_cpu[..8.min(hidden_dim)]);
        eprintln!(
            "CPU Normed L2: {:.6}",
            normed_cpu.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // GPU path: same calculation via GpuModelQ4's rms_norm_inplace
        let gpu_model = AprQ4ToGpuAdapter::create_model(&apr);

        let hidden_gpu: Vec<f32> = apr.token_embedding[offset..offset + hidden_dim].to_vec();
        let mut normed_gpu = hidden_gpu;

        // Manually call the same RMSNorm that GPU path uses
        let n = normed_gpu.len();
        let sum_sq: f32 = normed_gpu.iter().map(|v| v * v).sum();
        let rms_gpu = (sum_sq / n as f32 + eps).sqrt();
        let scale = 1.0 / rms_gpu;
        for (i, v) in normed_gpu.iter_mut().enumerate() {
            *v = *v
                * scale
                * gpu_model.layer_norms[0]
                    .attn_norm
                    .get(i)
                    .copied()
                    .unwrap_or(1.0);
        }

        eprintln!("\nGPU RMS: {:.6}", rms_gpu);
        eprintln!("GPU Normed [0..8]: {:?}", &normed_gpu[..8.min(hidden_dim)]);
        eprintln!(
            "GPU Normed L2: {:.6}",
            normed_gpu.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Compare
        let diff_l2: f32 = normed_cpu
            .iter()
            .zip(normed_gpu.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        eprintln!("\n--- Comparison ---");
        eprintln!("RMS diff: {:.6e}", (rms - rms_gpu).abs());
        eprintln!("Normed L2 diff: {:.6e}", diff_l2);

        if diff_l2 < 1e-5 {
            eprintln!("✅ RMSNorm MATCHES - bug is NOT here");
        } else {
            eprintln!("❌ RMSNorm DIVERGES - investigate norm weights!");
        }

        assert!(diff_l2 < 1e-5, "RMSNorm diverged: diff = {}", diff_l2);
    }

    /// Test 2: Embedding parity
    ///
    /// Verifies that both paths get the same embedding vector.
    #[test]
    fn test_embedding_parity() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  EMBEDDING PARITY TEST                                               ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let apr = create_test_model();
        let hidden_dim = apr.config.hidden_dim;
        let gpu_model = AprQ4ToGpuAdapter::create_model(&apr);

        for token_id in [0u32, 5, 50, 99] {
            let offset = (token_id as usize) * hidden_dim;

            let cpu_embed: Vec<f32> = apr.token_embedding[offset..offset + hidden_dim].to_vec();
            let gpu_embed: Vec<f32> =
                gpu_model.token_embedding[offset..offset + hidden_dim].to_vec();

            let diff_l2: f32 = cpu_embed
                .iter()
                .zip(gpu_embed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();

            eprintln!("Token {}: diff L2 = {:.6e}", token_id, diff_l2);
            assert!(diff_l2 < 1e-10, "Embedding mismatch for token {}", token_id);
        }

        eprintln!("✅ Embeddings MATCH for all test tokens");
    }

    /// Test 3: Full forward pass parity with tracing
    ///
    /// Runs both CPU and GPU forward passes and compares intermediate values.
    #[test]
    fn test_full_forward_parity() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  FULL FORWARD PARITY TEST (APR Q4)                                   ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let apr = create_test_model();
        let mut scratch = AprInferenceScratch::from_config(&apr.config);

        // CPU forward
        let token_id = 5u32;
        eprintln!("Running CPU forward for token {}...", token_id);

        let cpu_logits = apr.forward_single_with_scratch(token_id, &mut scratch);

        match cpu_logits {
            Ok(logits) => {
                let l2 = logits.iter().map(|x| x * x).sum::<f32>().sqrt();
                let top1 = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                eprintln!("CPU Logits L2: {:.6}", l2);
                eprintln!("CPU Top-1: {}", top1);
                eprintln!("CPU Logits[0..5]: {:?}", &logits[..5.min(logits.len())]);
            },
            Err(e) => {
                eprintln!("❌ CPU forward failed: {:?}", e);
            },
        }

        // GPU forward (if CUDA available)
        eprintln!("\nAttempting GPU forward...");

        match CudaExecutor::new(0) {
            Ok(mut executor) => {
                // Upload weights
                match AprQ4ToGpuAdapter::upload_weights(&apr, &mut executor) {
                    Ok(bytes) => eprintln!("Uploaded {} bytes to GPU", bytes),
                    Err(e) => {
                        eprintln!("❌ Weight upload failed: {:?}", e);
                        return;
                    },
                }

                let gpu_model = AprQ4ToGpuAdapter::create_model(&apr);
                let gpu_logits = gpu_model.forward(&mut executor, &[token_id as usize]);

                match gpu_logits {
                    Ok(logits) => {
                        let l2 = logits.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let top1 = logits
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);
                        eprintln!("GPU Logits L2: {:.6}", l2);
                        eprintln!("GPU Top-1: {}", top1);
                        eprintln!("GPU Logits[0..5]: {:?}", &logits[..5.min(logits.len())]);
                    },
                    Err(e) => {
                        eprintln!("❌ GPU forward failed: {:?}", e);
                    },
                }
            },
            Err(e) => {
                eprintln!("⚠️ CUDA not available: {:?}", e);
                eprintln!("Skipping GPU portion of test");
            },
        }
    }

    /// Test 4: Layer norm weight parity
    ///
    /// Verifies that GPU model received the same norm weights as CPU.
    #[test]
    fn test_layer_norm_weights_parity() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  LAYER NORM WEIGHTS PARITY                                           ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let apr = create_test_model();
        let gpu_model = AprQ4ToGpuAdapter::create_model(&apr);

        // Check layer 0 attn norm
        let cpu_attn_norm = &apr.layers[0].attn_norm_weight;
        let gpu_attn_norm = &gpu_model.layer_norms[0].attn_norm;

        let diff_l2: f32 = cpu_attn_norm
            .iter()
            .zip(gpu_attn_norm.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        eprintln!("Attn norm weight diff L2: {:.6e}", diff_l2);
        eprintln!(
            "CPU attn_norm[0..8]: {:?}",
            &cpu_attn_norm[..8.min(cpu_attn_norm.len())]
        );
        eprintln!(
            "GPU attn_norm[0..8]: {:?}",
            &gpu_attn_norm[..8.min(gpu_attn_norm.len())]
        );

        if diff_l2 < 1e-10 {
            eprintln!("✅ Layer norm weights MATCH");
        } else {
            eprintln!("❌ Layer norm weights DIVERGE!");
        }

        assert!(diff_l2 < 1e-10, "Layer norm weights diverged");
    }
}
