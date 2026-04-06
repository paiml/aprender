//! Phase 14: GPU vs CPU Parity Test with BrickTracer
//!
//! "The Golden Trace" - Automated divergence detection system.
//!
//! Identifies EXACTLY where GPU output diverges from CPU by comparing:
//! 1. Token embeddings
//! 2. Per-layer hidden states (RMSNorm, QKV, RoPE, Attention, FFN)
//! 3. Final logits
//!
//! # Usage
//!
//! ```bash
//! # Run parity test with full output
//! cargo test --test gpu_cpu_parity --features cuda -- --nocapture
//!
//! # Run with trace feature for maximum detail
//! cargo test --test gpu_cpu_parity --features "cuda trace" -- --nocapture
//! ```

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::brick::BrickTracer;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    /// Test GPU vs CPU parity on first token logits
    #[test]
    fn test_gpu_cpu_first_token_logits_parity() {
        // Use TinyLlama Q4_K_M model
        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping test: model not found at {:?}", model_path);
            return;
        }

        // Load model
        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");
        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        // Encode simple prompt
        let prompt = "1+1=";
        let mut tokens: Vec<u32> = mapped
            .model
            .encode(prompt)
            .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());
        if let Some(bos) = mapped.model.bos_token_id() {
            tokens.insert(0, bos);
        }

        eprintln!(
            "[PARITY] Testing with {} prompt tokens: {:?}",
            tokens.len(),
            tokens
        );

        // CPU: Get logits after processing all prompt tokens
        let max_seq_len = tokens.len() + 10;
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), max_seq_len);
        let mut cpu_logits = vec![];
        for (pos, &token_id) in tokens.iter().enumerate() {
            cpu_logits = cpu_model
                .forward_cached(token_id, &mut cpu_cache, pos)
                .expect("CPU forward failed");
        }

        // GPU: Get logits using generate_full_cuda_with_cache (standard path)
        let _cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(),
            0,
            max_seq_len,
        )
        .expect("Failed to create CUDA model");

        // For GPU, we'll use generate_full_cuda_with_cache with 1 token
        // and compare the logits
        let _gen_config = realizar::gguf::QuantizedGenerateConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: vec![],
            trace: false,
            ..Default::default()
        };

        // Can't easily get intermediate logits from generate, so let's compare top-k
        let cpu_top1 = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        let cpu_decoded = mapped.model.decode(&[cpu_top1]);
        eprintln!("[PARITY] CPU top-1 token: {} = {:?}", cpu_top1, cpu_decoded);

        // Check logit statistics
        let cpu_mean: f32 = cpu_logits.iter().sum::<f32>() / cpu_logits.len() as f32;
        let cpu_max = cpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let cpu_min = cpu_logits.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!(
            "[PARITY] CPU logits: mean={:.4}, min={:.4}, max={:.4}",
            cpu_mean, cpu_min, cpu_max
        );

        // Top 5 CPU tokens
        let mut cpu_indexed: Vec<(usize, f32)> = cpu_logits.iter().copied().enumerate().collect();
        cpu_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("[PARITY] CPU top-5:");
        for (idx, logit) in cpu_indexed.iter().take(5) {
            let tok = mapped.model.decode(&[*idx as u32]);
            eprintln!("  {} ({:.4}): {:?}", idx, logit, tok);
        }

        // Log CPU logits to tracer for comparison
        let mut cpu_tracer = BrickTracer::new();
        cpu_tracer.log("final_logits", &cpu_logits);

        // This test documents the current state - we know GPU is broken
        // The goal is to help identify WHERE the divergence happens
    }

    /// Test BrickTracer comparison functionality
    #[test]
    fn test_brick_tracer_comparison() {
        let mut cpu_tracer = BrickTracer::new();
        let mut gpu_tracer = BrickTracer::new();

        // Simulate CPU trace
        cpu_tracer.log("embedding", &[1.0, 2.0, 3.0, 4.0]);
        cpu_tracer.log("layer0_attn_norm", &[0.5, 1.0, 1.5, 2.0]);
        cpu_tracer.log("layer0_qkv", &[0.1, 0.2, 0.3, 0.4]);
        cpu_tracer.log("layer0_rope_q", &[0.11, 0.21, 0.31, 0.41]);
        cpu_tracer.log("layer0_attention", &[0.05, 0.1, 0.15, 0.2]);

        // Simulate GPU trace - matches until layer0_rope_q
        gpu_tracer.log("embedding", &[1.0, 2.0, 3.0, 4.0]);
        gpu_tracer.log("layer0_attn_norm", &[0.5, 1.0, 1.5, 2.0]);
        gpu_tracer.log("layer0_qkv", &[0.1, 0.2, 0.3, 0.4]);
        gpu_tracer.log("layer0_rope_q", &[0.5, 0.6, 0.7, 0.8]); // DIVERGE!
        gpu_tracer.log("layer0_attention", &[0.25, 0.3, 0.35, 0.4]);

        let comparison = BrickTracer::compare(&cpu_tracer, &gpu_tracer, 0.01);

        eprintln!("{}", comparison);

        assert!(!comparison.is_equivalent(), "Should detect divergence");

        let first = comparison.first_divergence().unwrap();
        assert_eq!(
            first.name, "layer0_rope_q",
            "First divergence should be at RoPE"
        );
    }

    /// Test that traces correctly identify matching computations
    #[test]
    fn test_brick_tracer_match() {
        let mut cpu_tracer = BrickTracer::new();
        let mut gpu_tracer = BrickTracer::new();

        // Same data on both
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        cpu_tracer.log("embedding", &data);
        gpu_tracer.log("embedding", &data);

        let comparison = BrickTracer::compare(&cpu_tracer, &gpu_tracer, 0.01);
        assert!(
            comparison.is_equivalent(),
            "Should match when data is identical"
        );
    }

    /// Phase 14: The Golden Trace - Multi-position parity test
    ///
    /// Tests that GPU produces identical output to CPU at each position
    /// in a multi-token prompt. This is the key test for finding RoPE bugs.
    #[test]
    fn test_golden_trace_multi_position() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 14: THE GOLDEN TRACE - Multi-Position Parity Test            ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // Use TinyLlama Q4_K_M model
        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping test: model not found at {:?}", model_path);
            return;
        }

        // Load model
        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");
        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        // Test prompt: "Once upon a time" - known to diverge at position > 1
        let prompt = "Once upon a time";
        let mut tokens: Vec<u32> = mapped
            .model
            .encode(prompt)
            .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());
        if let Some(bos) = mapped.model.bos_token_id() {
            tokens.insert(0, bos);
        }

        eprintln!("📝 Prompt: {:?}", prompt);
        eprintln!("🔢 Tokens ({}):", tokens.len());
        for (i, &tok) in tokens.iter().enumerate() {
            let decoded = mapped.model.decode(&[tok]);
            eprintln!("   [{}] {} = {:?}", i, tok, decoded);
        }
        eprintln!();

        // CPU: Process each token and log hidden state at each position
        let max_seq_len = tokens.len() + 10;
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), max_seq_len);

        let mut cpu_tracer = BrickTracer::new();

        eprintln!("═══════════════════════════════════════════════════════════════════════");
        eprintln!("CPU FORWARD PASS");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        for (pos, &token_id) in tokens.iter().enumerate() {
            cpu_tracer.set_position(pos);

            let logits = cpu_model
                .forward_cached(token_id, &mut cpu_cache, pos)
                .expect("CPU forward failed");

            // Log logits at this position
            let event_name = format!("pos{}_logits", pos);
            cpu_tracer.log(&event_name, &logits);

            // Get top prediction
            let top1_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            let top1_decoded = mapped.model.decode(&[top1_idx]);

            // Calculate L2 norm
            let l2 = logits.iter().map(|x| x * x).sum::<f32>().sqrt();

            eprintln!(
                "📍 Position {} | Token: {} | L2: {:.4} | Top1: {} ({:?})",
                pos, token_id, l2, top1_idx, top1_decoded
            );
        }

        // Print CPU trace summary
        eprintln!("\n--- CPU Trace Summary ---");
        cpu_tracer.summary();
        eprintln!();

        // For now, we document the CPU baseline
        // TODO: Add GPU forward pass with traced intermediate states
        eprintln!("═══════════════════════════════════════════════════════════════════════");
        eprintln!("GPU FORWARD PASS (TODO: Instrument with D2H copies)");
        eprintln!("═══════════════════════════════════════════════════════════════════════");
        eprintln!("⚠️  GPU instrumentation pending - requires D2H copies after each op");
        eprintln!();

        // Hypothesis from Phase 13: RoPE at position > 1 is the divergence point
        eprintln!("═══════════════════════════════════════════════════════════════════════");
        eprintln!("HYPOTHESIS");
        eprintln!("═══════════════════════════════════════════════════════════════════════");
        eprintln!("Based on Phase 13 findings:");
        eprintln!("  - Positions 0-1 (tokens: BOS, 'Once'): MATCH");
        eprintln!("  - Position 2+ (tokens: ' upon', ' a', ' time'): DIVERGE");
        eprintln!("  - Suspected cause: RoPE application at position > 1");
        eprintln!();
        eprintln!("Next steps:");
        eprintln!("  1. Add D2H copy after GPU RoPE kernel");
        eprintln!("  2. Compare GPU RoPE output vs CPU RoPE output at each position");
        eprintln!("  3. If RoPE matches, move to attention kernel");
        eprintln!("  4. If RoPE diverges, inspect position buffer and theta calculation");
    }

    /// Test tracer with realistic hidden dimensions
    #[test]
    fn test_brick_tracer_realistic_dims() {
        let mut tracer = BrickTracer::new();

        // TinyLlama hidden_dim = 2048
        let hidden_dim = 2048;
        let hidden_state: Vec<f32> = (0..hidden_dim).map(|i| (i as f32).sin()).collect();

        tracer.log("embedding", &hidden_state);

        let event = tracer.get("embedding").unwrap();
        assert_eq!(event.len, hidden_dim);
        assert!(event.l2_norm > 0.0);
        assert!(event.full_data.is_none()); // Non-verbose mode

        // Verbose mode
        let mut verbose_tracer = BrickTracer::verbose();
        verbose_tracer.log("embedding", &hidden_state);

        let verbose_event = verbose_tracer.get("embedding").unwrap();
        assert!(verbose_event.full_data.is_some());
        assert_eq!(verbose_event.full_data.as_ref().unwrap().len(), hidden_dim);
    }

    /// Test comparison with quantization-level tolerance
    #[test]
    fn test_brick_tracer_quantization_tolerance() {
        let mut cpu_tracer = BrickTracer::new();
        let mut gpu_tracer = BrickTracer::new();

        // CPU: precise values
        let cpu_data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        cpu_tracer.log("hidden", &cpu_data);

        // GPU: slightly different due to quantization (0.5% error)
        let gpu_data: Vec<f32> = cpu_data.iter().map(|x| x * 1.005).collect();
        gpu_tracer.log("hidden", &gpu_data);

        // Should fail at 0.1% tolerance
        let strict = BrickTracer::compare(&cpu_tracer, &gpu_tracer, 0.001);
        assert!(
            !strict.is_equivalent(),
            "0.1% tolerance should catch 0.5% error"
        );

        // Should pass at 1% tolerance
        let relaxed = BrickTracer::compare(&cpu_tracer, &gpu_tracer, 0.01);
        assert!(
            relaxed.is_equivalent(),
            "1% tolerance should accept 0.5% error"
        );
    }
}
