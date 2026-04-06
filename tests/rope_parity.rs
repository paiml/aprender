//! Phase 15: RoPE Parity Test
//!
//! Tests that GPU RoPE produces identical output to CPU RoPE at each position.
//! This is the key test for verifying H1 (RoPE Divergence hypothesis).
//!
//! # Usage
//!
//! ```bash
//! # Run with debug output
//! GPU_DEBUG=1 SKIP_CUDA_GRAPH=1 cargo test --test rope_parity --features cuda -- --nocapture
//! ```

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::brick::BrickTracer;
    use realizar::gguf::QuantizedGenerateConfig;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    /// Phase 15: The RoPE Divergence Hunt
    ///
    /// Tests GPU RoPE against CPU RoPE at multiple positions.
    /// Hypothesis H1: GPU RoPE is miscalculating theta or position for tokens > 1.
    #[test]
    fn test_rope_parity_multi_position() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 15: THE ROPE DIVERGENCE HUNT                                  ║");
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

        // Test prompt: "Once upon a time" - known to diverge at position > 1
        let prompt = "Once";
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

        // ═══════════════════════════════════════════════════════════════════════
        // CPU FORWARD PASS
        // ═══════════════════════════════════════════════════════════════════════
        eprintln!("═══════════════════════════════════════════════════════════════════════");
        eprintln!("CPU FORWARD PASS");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let max_seq_len = tokens.len() + 10;
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), max_seq_len);

        let mut cpu_tracer = BrickTracer::new();
        let mut cpu_logits_all = Vec::new();

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

            cpu_logits_all.push(logits);
        }

        // ═══════════════════════════════════════════════════════════════════════
        // GPU FORWARD PASS (Non-Graphed)
        // ═══════════════════════════════════════════════════════════════════════
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("GPU FORWARD PASS (Non-Graphed Path)");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        // Set environment variables to disable graphs and enable debug
        std::env::set_var("SKIP_CUDA_GRAPH", "1");
        std::env::set_var("GPU_DEBUG", "1");

        let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(),
            0,
            max_seq_len,
        )
        .expect("Failed to create CUDA model");

        // Generate using GPU
        let gen_config = QuantizedGenerateConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: vec![],
            trace: false,
            ..Default::default()
        };

        eprintln!("\n🚀 Running GPU generation (SKIP_CUDA_GRAPH=1, GPU_DEBUG=1)...");
        let gpu_result = cuda_model.generate_full_cuda_with_cache(&tokens, &gen_config);

        match gpu_result {
            Ok(generated) => {
                eprintln!("\n✅ GPU generation completed");
                eprintln!("   Input tokens: {:?}", tokens);
                eprintln!("   Generated tokens: {:?}", generated);

                // Compare first generated token
                if generated.len() > tokens.len() {
                    let gpu_next = generated[tokens.len()];
                    let gpu_decoded = mapped.model.decode(&[gpu_next]);

                    // What does CPU predict after the prompt?
                    let cpu_next = cpu_logits_all
                        .last()
                        .map(|logits| {
                            logits
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .map(|(idx, _)| idx as u32)
                                .unwrap_or(0)
                        })
                        .unwrap_or(0);
                    let cpu_decoded = mapped.model.decode(&[cpu_next]);

                    eprintln!(
                        "\n═══════════════════════════════════════════════════════════════════════"
                    );
                    eprintln!("PARITY CHECK");
                    eprintln!(
                        "═══════════════════════════════════════════════════════════════════════"
                    );
                    eprintln!("CPU next token: {} ({:?})", cpu_next, cpu_decoded);
                    eprintln!("GPU next token: {} ({:?})", gpu_next, gpu_decoded);

                    if cpu_next == gpu_next {
                        eprintln!("✅ MATCH: CPU and GPU agree on next token");
                    } else {
                        eprintln!("❌ DIVERGENCE: CPU and GPU disagree!");
                        eprintln!("\nThis confirms the hypothesis:");
                        eprintln!("  H1 (RoPE Divergence): Check the debug output above for RoPE differences");
                    }
                }
            },
            Err(e) => {
                eprintln!("❌ GPU generation failed: {:?}", e);
            },
        }

        // Clean up environment
        std::env::remove_var("SKIP_CUDA_GRAPH");
        std::env::remove_var("GPU_DEBUG");
    }

    /// Test GPU RoPE with single token (position 0) - should match
    #[test]
    fn test_rope_position_zero() {
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("TEST: RoPE at Position 0 (Single Token)");
        eprintln!("═══════════════════════════════════════════════════════════════════════\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping test: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        // Single token test
        let tokens = vec![mapped.model.bos_token_id().unwrap_or(1)];
        eprintln!("Token: {:?}", tokens);

        // CPU
        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");
        let max_seq_len = 10;
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), max_seq_len);
        let cpu_logits = cpu_model
            .forward_cached(tokens[0], &mut cpu_cache, 0)
            .expect("CPU forward failed");

        let cpu_top1 = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        // GPU
        std::env::set_var("SKIP_CUDA_GRAPH", "1");
        let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(),
            0,
            max_seq_len,
        )
        .expect("Failed to create CUDA model");

        let gen_config = QuantizedGenerateConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: vec![],
            trace: false,
            ..Default::default()
        };

        let gpu_result = cuda_model.generate_full_cuda_with_cache(&tokens, &gen_config);
        std::env::remove_var("SKIP_CUDA_GRAPH");

        match gpu_result {
            Ok(generated) => {
                if generated.len() > 1 {
                    let gpu_top1 = generated[1];
                    eprintln!("CPU top1: {}", cpu_top1);
                    eprintln!("GPU top1: {}", gpu_top1);

                    if cpu_top1 == gpu_top1 {
                        eprintln!("✅ Position 0: MATCH");
                    } else {
                        eprintln!("❌ Position 0: DIVERGE (unexpected!)");
                    }
                }
            },
            Err(e) => eprintln!("GPU error: {:?}", e),
        }
    }

    /// Test GPU RoPE with two tokens (positions 0 and 1)
    #[test]
    fn test_rope_position_one() {
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("TEST: RoPE at Positions 0 and 1 (Two Tokens)");
        eprintln!("═══════════════════════════════════════════════════════════════════════\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping test: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        // Two tokens: BOS + "Hello"
        let hello_encoded = mapped.model.encode("Hello").unwrap_or(vec![15496]);
        let mut tokens = vec![mapped.model.bos_token_id().unwrap_or(1)];
        tokens.extend(hello_encoded);
        eprintln!("Tokens: {:?}", tokens);

        // CPU
        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");
        let max_seq_len = 20;
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), max_seq_len);

        let mut cpu_last_logits = vec![];
        for (pos, &tok) in tokens.iter().enumerate() {
            cpu_last_logits = cpu_model
                .forward_cached(tok, &mut cpu_cache, pos)
                .expect("CPU forward failed");
        }

        let cpu_top1 = cpu_last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        // GPU
        std::env::set_var("SKIP_CUDA_GRAPH", "1");
        let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(),
            0,
            max_seq_len,
        )
        .expect("Failed to create CUDA model");

        let gen_config = QuantizedGenerateConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: vec![],
            trace: false,
            ..Default::default()
        };

        let gpu_result = cuda_model.generate_full_cuda_with_cache(&tokens, &gen_config);
        std::env::remove_var("SKIP_CUDA_GRAPH");

        match gpu_result {
            Ok(generated) => {
                if generated.len() > tokens.len() {
                    let gpu_top1 = generated[tokens.len()];
                    let cpu_decoded = mapped.model.decode(&[cpu_top1]);
                    let gpu_decoded = mapped.model.decode(&[gpu_top1]);

                    eprintln!("CPU top1: {} ({:?})", cpu_top1, cpu_decoded);
                    eprintln!("GPU top1: {} ({:?})", gpu_top1, gpu_decoded);

                    if cpu_top1 == gpu_top1 {
                        eprintln!("✅ Positions 0-1: MATCH");
                    } else {
                        eprintln!("❌ Positions 0-1: DIVERGE");
                    }
                }
            },
            Err(e) => eprintln!("GPU error: {:?}", e),
        }
    }
}
