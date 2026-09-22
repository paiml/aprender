//! Compare CPU and GPU forward passes using proper tracing
//!
//! Uses forward_traced for CPU to get layer-by-layer stats,
//! then manually traces GPU at the same checkpoints.

// #3809: the fixture now lives in `tests/common/` so the GPU-free CPU control
// (`apr_trace_fixture_cpu_half.rs`) drives the SAME model rather than a copy.
mod common;

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::apr_transformer::{
        ActivationStats,
        TracedForward, // PMAT-216: Import the trait
    };
    use realizar::gpu::adapters::AprF32ToGpuAdapter;

    use crate::common::create_test_model;

    fn compute_stats(data: &[f32]) -> ActivationStats {
        ActivationStats::from_slice(data)
    }

    /// Create a minimal test model

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
        for (i, (&cpu_val, &gpu_val)) in cpu_trace
            .logits
            .iter()
            .zip(gpu_logits.iter())
            .take(10)
            .enumerate()
        {
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
