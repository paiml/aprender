//! Compare CPU and GPU forward passes using proper tracing
//!
//! Uses forward_traced for CPU to get layer-by-layer stats,
//! then manually traces GPU at the same checkpoints.

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::apr_transformer::{
        ActivationStats,
        AprTransformer,
        AprTransformerConfig,
        AprTransformerLayer,
        TracedForward, // PMAT-216: Import the trait
    };
    use realizar::gpu::adapters::AprF32ToGpuAdapter;

    fn compute_stats(data: &[f32]) -> ActivationStats {
        ActivationStats::from_slice(data)
    }

    /// Create a minimal test model
    fn create_test_model() -> AprTransformer {
        let hidden_dim = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_dim = 128;
        let vocab_size = 256;

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

        let token_embedding: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();

        let output_norm_weight = vec![1.0f32; hidden_dim];

        let lm_head_weight: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| ((i as f32) * 0.001).cos())
            .collect();

        let qkv_out_dim = hidden_dim + 2 * kv_dim;
        let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();

        let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.02).cos() * 0.1)
            .collect();

        let attn_norm_weight = vec![1.0f32; hidden_dim];

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

    #[test]
    fn test_trace_comparison() {
        eprintln!("\n╔═════════════════════════════════════════════════════════════╗");
        eprintln!("║  GPU/CPU TRACE COMPARISON - Using forward_traced            ║");
        eprintln!("╚═════════════════════════════════════════════════════════════╝\n");

        let mut apr_model = create_test_model();
        let token_id = 42u32;

        // CPU traced forward (using TracedForward trait - PMAT-216)
        eprintln!("=== CPU Forward (traced via TracedForward trait) ===");
        let cpu_trace =
            TracedForward::forward_traced(&mut apr_model, &[token_id]).expect("CPU forward failed");

        eprintln!(
            "Embed: mean={:.4}, std={:.4}, L2≈{:.4}",
            cpu_trace.embed_stats.mean,
            cpu_trace.embed_stats.std_dev,
            (cpu_trace.embed_stats.mean.powi(2) + cpu_trace.embed_stats.std_dev.powi(2)).sqrt()
                * (apr_model.config.hidden_dim as f32).sqrt()
        );

        for layer in &cpu_trace.layer_activations {
            eprintln!("Layer {}: qkv mean={:.4} std={:.4}, attn_out mean={:.4} std={:.4}, ffn_out mean={:.4} std={:.4}",
                layer.layer_idx,
                layer.qkv_stats.mean, layer.qkv_stats.std_dev,
                layer.attn_out_stats.mean, layer.attn_out_stats.std_dev,
                layer.ffn_out_stats.mean, layer.ffn_out_stats.std_dev,
            );
        }

        eprintln!(
            "Final norm: mean={:.4}, std={:.4}",
            cpu_trace.final_norm_stats.mean, cpu_trace.final_norm_stats.std_dev
        );
        eprintln!(
            "Logits: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            cpu_trace.logits_stats.mean,
            cpu_trace.logits_stats.std_dev,
            cpu_trace.logits_stats.min,
            cpu_trace.logits_stats.max
        );

        let cpu_argmax = cpu_trace
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        eprintln!("CPU argmax: {}", cpu_argmax);

        // GPU forward with tracing (PMAT-216: using TracedForward trait)
        eprintln!("\n=== GPU Forward (traced via TracedForward trait) ===");
        let mut gpu_model =
            AprF32ToGpuAdapter::to_gpu_model(&apr_model).expect("GPU model creation failed");

        let gpu_trace = TracedForward::forward_traced(&mut gpu_model, &[token_id])
            .expect("GPU forward_traced failed");

        let gpu_logits = &gpu_trace.logits;

        eprintln!(
            "Embed: mean={:.4}, std={:.4}",
            gpu_trace.embed_stats.mean, gpu_trace.embed_stats.std_dev
        );

        for layer in &gpu_trace.layer_activations {
            eprintln!("Layer {}: qkv mean={:.4} std={:.4}, attn_out mean={:.4} std={:.4}, ffn_out mean={:.4} std={:.4}",
                layer.layer_idx,
                layer.qkv_stats.mean, layer.qkv_stats.std_dev,
                layer.attn_out_stats.mean, layer.attn_out_stats.std_dev,
                layer.ffn_out_stats.mean, layer.ffn_out_stats.std_dev,
            );
        }

        let gpu_stats = compute_stats(gpu_logits);
        eprintln!(
            "GPU Logits: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            gpu_stats.mean, gpu_stats.std_dev, gpu_stats.min, gpu_stats.max
        );

        let gpu_argmax = gpu_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        eprintln!("GPU argmax: {}", gpu_argmax);

        // Compare
        eprintln!("\n=== Comparison ===");
        let cpu_l2: f32 = cpu_trace.logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let gpu_l2: f32 = gpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let l2_diff_pct = ((cpu_l2 - gpu_l2).abs() / cpu_l2.max(1e-10)) * 100.0;

        eprintln!(
            "CPU L2: {:.4}, GPU L2: {:.4}, diff: {:.2}%",
            cpu_l2, gpu_l2, l2_diff_pct
        );
        eprintln!("Argmax match: {}", cpu_argmax == gpu_argmax);

        // Layer-by-layer comparison (PMAT-216: enabled by forward_traced_gpu)
        eprintln!("\n=== Layer-by-Layer Comparison ===");
        eprintln!("Layer | CPU embed | GPU embed | Diff %");
        let embed_diff = ((cpu_trace.embed_stats.mean - gpu_trace.embed_stats.mean).abs()
            / cpu_trace.embed_stats.mean.abs().max(1e-10))
            * 100.0;
        eprintln!(
            "Embed | {:.4} | {:.4} | {:.2}%",
            cpu_trace.embed_stats.mean, gpu_trace.embed_stats.mean, embed_diff
        );

        for (cpu_layer, gpu_layer) in cpu_trace
            .layer_activations
            .iter()
            .zip(gpu_trace.layer_activations.iter())
        {
            let qkv_diff = ((cpu_layer.qkv_stats.mean - gpu_layer.qkv_stats.mean).abs()
                / cpu_layer.qkv_stats.mean.abs().max(1e-10))
                * 100.0;
            let out_diff = ((cpu_layer.output_stats.mean - gpu_layer.output_stats.mean).abs()
                / cpu_layer.output_stats.mean.abs().max(1e-10))
                * 100.0;
            eprintln!(
                "L{:2}   | qkv diff: {:.2}% | out diff: {:.2}%",
                cpu_layer.layer_idx, qkv_diff, out_diff
            );
        }

        // Element-wise comparison of first 10
        eprintln!("\nFirst 10 logits:");
        eprintln!("  idx |    CPU    |    GPU    |   ratio");
        eprintln!("  ----|-----------|-----------|----------");
        for i in 0..10 {
            let cpu_val = cpu_trace.logits[i];
            let gpu_val = gpu_logits[i];
            let ratio = if gpu_val.abs() > 1e-10 {
                cpu_val / gpu_val
            } else {
                f32::NAN
            };
            eprintln!(
                "  {:3} | {:9.4} | {:9.4} | {:9.2}",
                i, cpu_val, gpu_val, ratio
            );
        }

        // The ratio should be consistent if it's just a scale factor
        let ratios: Vec<f32> = (0..50)
            .filter_map(|i| {
                let cpu_val = cpu_trace.logits[i];
                let gpu_val = gpu_logits[i];
                if gpu_val.abs() > 0.1 {
                    Some(cpu_val / gpu_val)
                } else {
                    None
                }
            })
            .collect();

        if !ratios.is_empty() {
            let avg_ratio: f32 = ratios.iter().sum::<f32>() / ratios.len() as f32;
            let ratio_std: f32 = (ratios.iter().map(|r| (r - avg_ratio).powi(2)).sum::<f32>()
                / ratios.len() as f32)
                .sqrt();
            eprintln!(
                "\nRatio analysis: avg={:.2}, std={:.4}",
                avg_ratio, ratio_std
            );
            if ratio_std < 0.1 {
                eprintln!(
                    "  → Consistent scale factor detected: CPU ≈ {:.1}x GPU",
                    avg_ratio
                );
            }
        }

        // PMAT-216 MANDATORY ASSERTIONS
        assert!(
            l2_diff_pct < 1.0,
            "PMAT-216 FAIL: GPU L2 diverged {:.2}% from CPU (max 1%)",
            l2_diff_pct
        );
        assert_eq!(
            cpu_argmax, gpu_argmax,
            "PMAT-216 FAIL: GPU argmax {} != CPU argmax {}",
            gpu_argmax, cpu_argmax
        );
    }
}
