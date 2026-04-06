//! Phase 21: Trace the F32 GPU Divergence
//!
//! "Stop grepping. Start tracing." - The Popperian Command
//!
//! This test uses BrickTracer to instrument both CPU and GPU forward passes
//! and identify EXACTLY where they diverge.
//!
//! # Usage
//!
//! ```bash
//! # Run with trace feature enabled and full output
//! cargo test --test phase21_trace_divergence --features "cuda trace" -- --nocapture
//! ```

#![allow(unused_variables)]

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
    use realizar::brick::BrickTracer;
    use realizar::gpu::adapters::AprF32ToGpuAdapter;

    /// Create a small synthetic model for deterministic testing.
    ///
    /// Model dimensions:
    /// - vocab_size: 256
    /// - hidden_dim: 64
    /// - num_heads: 4
    /// - num_kv_heads: 2 (GQA)
    /// - num_layers: 1
    /// - intermediate_dim: 128
    fn create_small_test_model() -> AprTransformer {
        let hidden_dim = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_dim / num_heads; // 16
        let kv_dim = num_kv_heads * head_dim; // 32
        let intermediate_dim = 128;
        let vocab_size = 256;

        // Create config
        let config = AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers: 1,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 32,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        };

        // Token embedding: [vocab_size, hidden_dim]
        // Initialize with deterministic pattern
        let token_embedding: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();

        // Output norm weight: all 1.0
        let output_norm_weight = vec![1.0f32; hidden_dim];

        // LM head weight: [hidden_dim, vocab_size]
        // APR stores as [out_dim, in_dim] = [vocab_size, hidden_dim]
        let lm_head_weight: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| ((i as f32) * 0.001).cos())
            .collect();

        // Create one layer with deterministic weights
        let qkv_out_dim = hidden_dim + 2 * kv_dim; // 64 + 64 = 128 for Q + K + V

        // QKV weight: [qkv_out_dim, hidden_dim]
        let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();

        // Attention output weight: [hidden_dim, hidden_dim]
        let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.02).cos() * 0.1)
            .collect();

        // Attention norm weight
        let attn_norm_weight = vec![1.0f32; hidden_dim];

        // FFN weights
        let ffn_up_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.03).sin() * 0.1)
            .collect();
        let ffn_down_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
            .map(|i| ((i as f32) * 0.04).cos() * 0.1)
            .collect();
        let ffn_gate_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.05).sin() * 0.1)
            .collect();
        let ffn_norm_weight = vec![1.0f32; hidden_dim];

        let layer = AprTransformerLayer {
            qkv_weight,
            qkv_bias: None,
            attn_output_weight,
            attn_output_bias: None,
            attn_norm_weight,
            attn_norm_bias: None,
            ffn_up_weight,
            ffn_up_bias: None,
            ffn_down_weight,
            ffn_down_bias: None,
            ffn_gate_weight: Some(ffn_gate_weight),
            ffn_gate_bias: None,
            ffn_norm_weight: Some(ffn_norm_weight),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        };

        AprTransformer {
            config,
            token_embedding,
            layers: vec![layer],
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
        }
    }

    /// Phase 21: The Golden Trace - Find where GPU diverges from CPU
    #[test]
    fn test_phase21_trace_divergence() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 21: THE TRACER'S RETURN - Find the Divergence                 ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // 1. Create small test model
        eprintln!("Step 1: Creating synthetic test model...");
        let apr_model = create_small_test_model();
        eprintln!("  - vocab_size: {}", apr_model.config.vocab_size);
        eprintln!("  - hidden_dim: {}", apr_model.config.hidden_dim);
        eprintln!("  - num_heads: {}", apr_model.config.num_heads);
        eprintln!("  - num_kv_heads: {}", apr_model.config.num_kv_heads);
        eprintln!("  - num_layers: {}", apr_model.config.num_layers);
        eprintln!(
            "  - intermediate_dim: {}",
            apr_model.config.intermediate_dim
        );

        // 2. Convert to GPU model
        eprintln!("\nStep 2: Converting to GpuModel via AprF32ToGpuAdapter...");
        let mut gpu_model = match AprF32ToGpuAdapter::to_gpu_model(&apr_model) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  ERROR: Failed to create GPU model: {:?}", e);
                return;
            },
        };
        eprintln!("  - GPU model created successfully");

        // 3. Test with single token
        let token_id: usize = 42;

        eprintln!("\nStep 3: Forward pass with token_id={}", token_id);

        // =====================================================================
        // CPU Path - APR Forward
        // =====================================================================
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("CPU FORWARD PASS (APR Reference)");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        let mut cpu_tracer = BrickTracer::new();

        // CPU: Embedding
        let hidden_dim = apr_model.config.hidden_dim;
        let embed_start = token_id * hidden_dim;
        let cpu_embedding = &apr_model.token_embedding[embed_start..embed_start + hidden_dim];
        cpu_tracer.log("embedding", cpu_embedding);
        let cpu_embed_l2 = cpu_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  embedding L2: {:.6}", cpu_embed_l2);
        eprintln!(
            "  embedding[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            cpu_embedding[0], cpu_embedding[1], cpu_embedding[2], cpu_embedding[3]
        );

        // CPU: Full forward pass via APR
        let cpu_logits = apr_model.forward(&[token_id as u32]).unwrap();
        cpu_tracer.log("final_logits", &cpu_logits);
        let cpu_logits_l2 = cpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  final_logits L2: {:.6}", cpu_logits_l2);

        let cpu_argmax = cpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        eprintln!("  CPU argmax: {}", cpu_argmax);

        // =====================================================================
        // GPU Path - GpuModel Forward
        // =====================================================================
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("GPU FORWARD PASS");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        let mut gpu_tracer = BrickTracer::new();

        // GPU: Full forward pass
        let gpu_logits = match gpu_model.forward_gpu(&[token_id]) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("  ERROR: GPU forward failed: {:?}", e);
                return;
            },
        };
        gpu_tracer.log("final_logits", &gpu_logits);
        let gpu_logits_l2 = gpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  final_logits L2: {:.6}", gpu_logits_l2);

        let gpu_argmax = gpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        eprintln!("  GPU argmax: {}", gpu_argmax);

        // =====================================================================
        // COMPARISON
        // =====================================================================
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("TRACE COMPARISON");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        let comparison = BrickTracer::compare(&cpu_tracer, &gpu_tracer, 0.01);
        eprintln!("{}", comparison);

        let logits_diff = (cpu_logits_l2 - gpu_logits_l2).abs() / cpu_logits_l2.max(1e-10);
        eprintln!("\nLogits L2 diff: {:.4}%", logits_diff * 100.0);
        eprintln!("CPU argmax: {} | GPU argmax: {}", cpu_argmax, gpu_argmax);

        if cpu_argmax == gpu_argmax {
            eprintln!("RESULT: ARGMAX MATCH - Same prediction");
        } else {
            eprintln!("RESULT: ARGMAX MISMATCH - Different prediction!");
        }

        // Element-wise comparison
        eprintln!("\nFirst 10 logits comparison:");
        eprintln!("  idx |    CPU    |    GPU    |   diff");
        eprintln!("  ----|-----------|-----------|----------");
        for i in 0..10.min(cpu_logits.len()) {
            let cpu_val = cpu_logits[i];
            let gpu_val = gpu_logits[i];
            let diff = (cpu_val - gpu_val).abs();
            eprintln!(
                "  {:3} | {:9.4} | {:9.4} | {:9.6}",
                i, cpu_val, gpu_val, diff
            );
        }
    }

    /// Test matmul operation match - THE CRITICAL CHECKPOINT
    /// Phase 15 says "GPU matmul is the culprit" - this test verifies.
    #[test]
    fn test_phase21_matmul_parity() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 21: MATMUL PARITY TEST - The Suspected Culprit                ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // Small dimensions for easy debugging
        let m = 1; // batch
        let k = 64; // input dim
        let n = 128; // output dim

        // Create deterministic inputs
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.02).cos()).collect();

        eprintln!(
            "Matrix dimensions: A[{},{}] @ B[{},{}] = C[{},{}]",
            m, k, k, n, m, n
        );
        eprintln!("A L2: {:.6}", a.iter().map(|x| x * x).sum::<f32>().sqrt());
        eprintln!("B L2: {:.6}", b.iter().map(|x| x * x).sum::<f32>().sqrt());

        // CPU matmul: C[m,n] = A[m,k] @ B[k,n]
        let mut cpu_output = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                cpu_output[i * n + j] = sum;
            }
        }

        let cpu_l2 = cpu_output.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("\nCPU matmul L2: {:.6}", cpu_l2);
        eprintln!(
            "CPU output[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            cpu_output[0], cpu_output[1], cpu_output[2], cpu_output[3]
        );

        // GPU matmul via CudaScheduler
        use realizar::gpu::CudaScheduler;

        let mut scheduler = match CudaScheduler::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("CUDA not available: {:?}", e);
                return;
            },
        };

        let gpu_output = scheduler.matmul(&a, &b, m, k, n).unwrap();
        let gpu_l2 = gpu_output.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("GPU matmul L2: {:.6}", gpu_l2);
        eprintln!(
            "GPU output[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            gpu_output[0], gpu_output[1], gpu_output[2], gpu_output[3]
        );

        let diff = (cpu_l2 - gpu_l2).abs() / cpu_l2.max(1e-10);
        eprintln!("\nL2 diff: {:.6}%", diff * 100.0);

        // Element-wise comparison
        let max_elem_diff = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Max element diff: {:.6}", max_elem_diff);

        if diff > 0.01 {
            eprintln!("\n!!! MATMUL DIVERGENCE DETECTED !!!");
            eprintln!("This confirms Phase 15: GPU matmul is the culprit.\n");

            eprintln!("Element-wise comparison (first 16):");
            eprintln!("  idx |    CPU    |    GPU    |   diff");
            eprintln!("  ----|-----------|-----------|----------");
            for i in 0..n.min(16) {
                let cpu_val = cpu_output[i];
                let gpu_val = gpu_output[i];
                eprintln!(
                    "  {:3} | {:9.6} | {:9.6} | {:9.6}",
                    i,
                    cpu_val,
                    gpu_val,
                    (cpu_val - gpu_val).abs()
                );
            }
        } else {
            eprintln!("\nMatmul matches within tolerance.");
        }

        // Don't assert - just report
        if diff > 0.05 {
            eprintln!(
                "\nWARNING: Matmul mismatch {:.4}% exceeds 5% threshold",
                diff * 100.0
            );
        }
    }

    /// Test RMSNorm parity (second checkpoint in the pipeline)
    #[test]
    fn test_phase21_rmsnorm_parity() {
        eprintln!("\n=== Phase 21: RMSNorm Parity Test ===\n");

        let hidden_dim = 64;
        let eps = 1e-5f32;

        // Test input
        let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1).collect();
        let weight = vec![1.0f32; hidden_dim];
        let bias = vec![0.0f32; hidden_dim];

        // CPU RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight + bias
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
        let cpu_output: Vec<f32> = input
            .iter()
            .zip(weight.iter())
            .zip(bias.iter())
            .map(|((&x, &w), &b)| (x / rms) * w + b)
            .collect();

        let cpu_l2 = cpu_output.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("CPU RMSNorm L2: {:.6}", cpu_l2);
        eprintln!("  (rms={:.6}, sum_sq={:.6})", rms, sum_sq);

        // The GPU uses the same formula in layer_norm_static, so this should match
        // We can't call the private function, but we verify the formula is correct
        eprintln!("\nRMSNorm formula verification:");
        eprintln!(
            "  input L2: {:.6}",
            input.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        eprintln!("  output L2: {:.6}", cpu_l2);
        eprintln!(
            "  output[0..4]: [{:.6}, {:.6}, {:.6}, {:.6}]",
            cpu_output[0], cpu_output[1], cpu_output[2], cpu_output[3]
        );
    }

    /// DEEP DIVE: Test each step of the forward pass to find exact divergence point
    #[test]
    fn test_phase21_step_by_step_divergence() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 21: STEP-BY-STEP DIVERGENCE HUNT                              ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // Create synthetic model
        let apr_model = create_small_test_model();
        let hidden_dim = apr_model.config.hidden_dim;

        // Test embedding parity (should match)
        let token_id = 42usize;
        let cpu_embed =
            &apr_model.token_embedding[token_id * hidden_dim..(token_id + 1) * hidden_dim];
        let cpu_embed_l2 = cpu_embed.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("Step 1: EMBEDDING");
        eprintln!("  CPU embedding L2: {:.6}", cpu_embed_l2);

        // Create GPU model
        let gpu_model = AprF32ToGpuAdapter::to_gpu_model(&apr_model).unwrap();

        // Key insight: The GPU forward_gpu uses forward_block_idx
        // which does QKV projection differently from APR forward

        eprintln!("\nStep 2: QKV PROJECTION");
        eprintln!("  Comparing weight dimensions...");

        // APR QKV weight: [qkv_out_dim, hidden_dim]
        let qkv_out_dim = hidden_dim
            + 2 * (apr_model.config.num_kv_heads * hidden_dim / apr_model.config.num_heads);
        eprintln!(
            "  APR QKV weight: [{}, {}] = {} elements",
            qkv_out_dim,
            hidden_dim,
            apr_model.layers[0].qkv_weight.len()
        );

        // The GPU model transposes during conversion:
        // GPU QKV weight should be: [hidden_dim, qkv_out_dim]
        let gpu_qkv_dim = gpu_model.config().qkv_dim();
        eprintln!("  GPU expects qkv_dim: {}", gpu_qkv_dim);

        // Manual CPU QKV projection: C = input @ W^T where W is [qkv_out_dim, hidden_dim]
        // This is equivalent to: C = input @ (W^T) = input @ [hidden_dim, qkv_out_dim]
        eprintln!("\nStep 3: MANUAL QKV PROJECTION");

        // APR does: qkv = matmul(normed, qkv_weight, hidden_dim, qkv_dim)
        // where qkv_weight is [qkv_dim, hidden_dim] in row-major
        // matmul(A, B, m, k, n) computes A[m,k] @ B[k,n] = C[m,n]
        // So: normed[1, hidden_dim] @ qkv_weight[hidden_dim, qkv_dim] = qkv[1, qkv_dim]
        // But qkv_weight is stored as [qkv_dim, hidden_dim]!

        // This means APR's matmul interprets the weight differently
        // Let me trace through the actual APR code path

        // In APR forward (apr_transformer/mod.rs line 1031):
        // let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
        // APR matmul is: C[m,n] = A[m,k] @ B[k,n]
        // So: normed[seq, hidden] @ weight[hidden, qkv_dim]

        // But APR stores qkv_weight as [qkv_dim, hidden_dim]
        // This means APR's matmul must be treating the weight as transposed!

        eprintln!("\nDIAGNOSIS:");
        eprintln!(
            "  APR stores QKV weight as [qkv_out, hidden] but uses matmul(in, W, hidden, qkv_out)"
        );
        eprintln!("  This implies APR matmul does: input @ W where W is read as [hidden, qkv_out]");
        eprintln!("  But W is stored as [qkv_out, hidden], so APR must be transposing internally");
        eprintln!();
        eprintln!("  GPU adapter transposes: [qkv_out, hidden] -> [hidden, qkv_out]");
        eprintln!("  GPU matmul does: input @ W where W is [hidden, qkv_out]");
        eprintln!("  This should match... unless the transpose is wrong or dimensions mismatch");

        // Let me check the actual APR matmul implementation
        eprintln!("\nChecking APR matmul interpretation...");

        // APR matmul signature: fn matmul(&self, a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32>
        // It computes: for each row i, for each col j: sum over l: a[i*k + l] * b[l*n + j]
        // This is: A[?, k] @ B[k, n] = C[?, n] where ? is inferred from a.len()/k

        // So when APR calls matmul(&normed, &qkv_weight, hidden_dim, qkv_dim):
        // It does: normed[seq, hidden_dim] @ qkv_weight[hidden_dim, qkv_dim]
        // But qkv_weight is stored as [qkv_dim, hidden_dim] in row-major!
        // This means APR is treating the weight incorrectly... OR
        // APR's matmul accesses b[l*n + j] = weight[l * qkv_dim + j]
        // If weight is [qkv_dim, hidden_dim], then weight[l * qkv_dim + j] makes no sense

        // WAIT - I need to re-read the APR matmul

        eprintln!("\n  The bug might be:");
        eprintln!("  1. APR's matmul has a different convention than I thought");
        eprintln!("  2. The adapter's transpose logic is inverted");
        eprintln!("  3. GPU's forward_block_idx has wrong dimension ordering");

        // Let's compute a small manual test
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // [1, 4]
        let b = vec![
            1.0f32, 0.0, 0.0, 1.0, // [4, 2] stored row-major
            2.0, 0.0, 0.0, 2.0,
        ];
        let b_transposed = vec![
            1.0f32, 2.0, // [2, 4] stored row-major
            0.0, 0.0, 0.0, 0.0, 1.0, 2.0,
        ];

        // Standard matmul A[1,4] @ B[4,2] = C[1,2]
        // c[0] = a[0]*b[0] + a[1]*b[2] + a[2]*b[4] + a[3]*b[6]
        //      = 1*1 + 2*0 + 3*2 + 4*0 = 1 + 0 + 6 + 0 = 7
        // c[1] = a[0]*b[1] + a[1]*b[3] + a[2]*b[5] + a[3]*b[7]
        //      = 1*0 + 2*1 + 3*0 + 4*2 = 0 + 2 + 0 + 8 = 10
        eprintln!("\nManual matmul test:");
        eprintln!("  A[1,4] @ B[4,2] should give [7, 10]");

        let m = 1;
        let k = 4;
        let n = 2;
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        eprintln!("  Result: {:?}", c);
    }
}
