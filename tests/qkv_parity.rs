//! Phase 15b: Step-by-Step Forward Parity Test
//!
//! Compares CPU and GPU forward pass results to isolate the divergence point.

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::gguf::{
        MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel, OwnedQuantizedModelCuda,
        QuantizedGenerateConfig,
    };
    use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};

    /// Test full forward pass comparison
    #[test]
    fn test_forward_parity() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 15b: CPU vs GPU FORWARD PARITY                                ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        // Test with 2 tokens: BOS + "Once"
        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        let once_tokens = mapped.model.encode("Once").unwrap_or(vec![9038]);
        let tokens = vec![bos_token, once_tokens[0]];

        eprintln!("Test tokens: {:?}", tokens);
        for (i, &tok) in tokens.iter().enumerate() {
            let decoded = mapped.model.decode(&[tok]);
            eprintln!("  [{}] {} = {:?}", i, tok, decoded);
        }

        // CPU forward
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("CPU FORWARD PASS");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), 20);

        let mut cpu_logits = vec![];
        for (pos, &tok) in tokens.iter().enumerate() {
            cpu_logits = cpu_model
                .forward_cached(tok, &mut cpu_cache, pos)
                .expect("CPU forward failed");
            let l2 = cpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
            let top1 = cpu_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            let decoded = mapped.model.decode(&[top1]);
            eprintln!(
                "Position {}: L2={:.4}, top1={} ({:?})",
                pos, l2, top1, decoded
            );
        }

        // GPU forward
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("GPU FORWARD PASS (generate_full_cuda_with_cache)");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        std::env::set_var("SKIP_CUDA_GRAPH", "1");

        let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(),
            0,
            20,
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

        let gpu_result = cuda_model
            .generate_full_cuda_with_cache(&tokens, &gen_config)
            .expect("GPU forward failed");

        std::env::remove_var("SKIP_CUDA_GRAPH");

        let gpu_next_token = if gpu_result.len() > tokens.len() {
            gpu_result[tokens.len()]
        } else {
            0
        };
        let gpu_decoded = mapped.model.decode(&[gpu_next_token]);

        eprintln!("Generated: {:?}", gpu_result);
        eprintln!("Next token: {} ({:?})", gpu_next_token, gpu_decoded);

        // Compare
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("COMPARISON");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        let cpu_top1 = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        let cpu_decoded = mapped.model.decode(&[cpu_top1]);

        eprintln!("CPU next token: {} ({:?})", cpu_top1, cpu_decoded);
        eprintln!("GPU next token: {} ({:?})", gpu_next_token, gpu_decoded);

        if cpu_top1 == gpu_next_token {
            eprintln!("\n✅ MATCH: CPU and GPU agree!");
        } else {
            eprintln!("\n❌ DIVERGENCE DETECTED!");
            eprintln!("\nAnalysis:");
            eprintln!("  The generate_full_cuda_with_cache path uses:");
            eprintln!("  - CPU embedding lookup");
            eprintln!("  - GPU QKV projection (matmul)");
            eprintln!("  - CPU RoPE");
            eprintln!("  - CPU attention");
            eprintln!("  - GPU output/FFN projections (matmul)");
            eprintln!("  - GPU LM head projection (matmul)");
            eprintln!("\n  Since CPU RoPE and attention are shared, the bug is in GPU matmul.");
        }
    }

    /// Single token test - should match since no KV cache involved
    #[test]
    fn test_single_token_parity() {
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("SINGLE TOKEN PARITY TEST");
        eprintln!("═══════════════════════════════════════════════════════════════════════\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        // Single token: just BOS
        let tokens = vec![mapped.model.bos_token_id().unwrap_or(1)];
        eprintln!("Token: {:?}", tokens);

        // CPU
        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), 10);
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
            10,
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

        let gpu_result = cuda_model
            .generate_full_cuda_with_cache(&tokens, &gen_config)
            .expect("GPU forward failed");
        std::env::remove_var("SKIP_CUDA_GRAPH");

        let gpu_top1 = if gpu_result.len() > 1 {
            gpu_result[1]
        } else {
            0
        };

        let cpu_decoded = mapped.model.decode(&[cpu_top1]);
        let gpu_decoded = mapped.model.decode(&[gpu_top1]);

        eprintln!("CPU: {} ({:?})", cpu_top1, cpu_decoded);
        eprintln!("GPU: {} ({:?})", gpu_top1, gpu_decoded);

        if cpu_top1 == gpu_top1 {
            eprintln!("✅ Single token: MATCH");
        } else {
            eprintln!("❌ Single token: DIVERGE (unexpected for position 0!)");
        }
    }

    /// Phase 16: Trace RMSNorm and QKV intermediate values
    ///
    /// This test manually computes the first layer's RMSNorm and QKV to compare
    /// CPU vs what the GPU path should be receiving.
    #[test]
    fn test_phase16_rmsnorm_trace() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: RMSNORM INPUT TRACE                                       ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let config = cpu_model.config();
        let hidden_dim = config.hidden_dim;
        let eps = config.eps;

        // Use BOS token
        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        eprintln!("Token: {} (BOS)", bos_token);
        eprintln!("hidden_dim: {}, eps: {}\n", hidden_dim, eps);

        // Step 1: Embedding lookup
        let embedding = cpu_model.embed(&[bos_token]);
        let embed_l2 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("STEP 1: Embedding");
        eprintln!("  L2: {:.6}", embed_l2);
        eprintln!("  Sum: {:.6}", embedding.iter().sum::<f32>());
        eprintln!("  First 8: {:?}\n", &embedding[..8.min(embedding.len())]);

        // Step 2: RMSNorm (first layer, attention norm)
        let layer = &cpu_model.layers()[0];
        let attn_norm_weight = &layer.attn_norm_weight;

        // Manual RMSNorm computation
        let sq_sum: f32 = embedding.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        let normed: Vec<f32> = embedding
            .iter()
            .zip(attn_norm_weight.iter())
            .map(|(&x, &w)| x / rms * w)
            .collect();

        let normed_l2 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("STEP 2: RMSNorm (Layer 0, Attention)");
        eprintln!("  RMS: {:.6}", rms);
        eprintln!("  Normed L2: {:.6}", normed_l2);
        eprintln!("  Normed Sum: {:.6}", normed.iter().sum::<f32>());
        eprintln!("  First 8: {:?}\n", &normed[..8.min(normed.len())]);

        // Step 3: QKV projection (CPU)
        eprintln!("STEP 3: QKV Projection (CPU)");
        let qkv_weight = &layer.qkv_weight;
        let qkv = match qkv_weight {
            OwnedQKVWeights::Fused(tensor) => {
                eprintln!("  Using FUSED QKV (qtype={})", tensor.qtype);
                fused_q4k_parallel_matvec(&tensor.data, &normed, tensor.in_dim, tensor.out_dim)
                    .expect("QKV matmul failed")
            },
            OwnedQKVWeights::Separate { q, k, v } => {
                eprintln!("  Using SEPARATE Q/K/V");
                eprintln!("    Q: qtype={}, dims={}x{}", q.qtype, q.in_dim, q.out_dim);
                eprintln!("    K: qtype={}, dims={}x{}", k.qtype, k.in_dim, k.out_dim);
                eprintln!("    V: qtype={}, dims={}x{}", v.qtype, v.in_dim, v.out_dim);

                let q_out = fused_q4k_parallel_matvec(&q.data, &normed, q.in_dim, q.out_dim)
                    .expect("Q matmul failed");
                let k_out = fused_q4k_parallel_matvec(&k.data, &normed, k.in_dim, k.out_dim)
                    .expect("K matmul failed");
                // V might be Q6_K
                let v_out = if v.qtype == 14 {
                    eprintln!("    V uses Q6_K - calling fused_q6k_parallel_matvec");
                    fused_q6k_parallel_matvec(&v.data, &normed, v.in_dim, v.out_dim)
                        .expect("V matmul failed")
                } else {
                    fused_q4k_parallel_matvec(&v.data, &normed, v.in_dim, v.out_dim)
                        .expect("V matmul failed")
                };

                eprintln!(
                    "  Q L2: {:.6}",
                    q_out.iter().map(|x| x * x).sum::<f32>().sqrt()
                );
                eprintln!(
                    "  K L2: {:.6}",
                    k_out.iter().map(|x| x * x).sum::<f32>().sqrt()
                );
                eprintln!(
                    "  V L2: {:.6}",
                    v_out.iter().map(|x| x * x).sum::<f32>().sqrt()
                );

                let mut combined = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                combined.extend_from_slice(&q_out);
                combined.extend_from_slice(&k_out);
                combined.extend_from_slice(&v_out);
                combined
            },
        };

        let qkv_l2 = qkv.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  QKV total L2: {:.6}", qkv_l2);
        eprintln!("  QKV len: {}", qkv.len());

        // Now let's compare what the GPU forward pass actually produces
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("GPU PATH COMPARISON");
        eprintln!("═══════════════════════════════════════════════════════════════════════\n");

        std::env::set_var("SKIP_CUDA_GRAPH", "1");
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");

        let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(),
            0,
            10,
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

        let tokens = vec![bos_token];
        let _gpu_result = cuda_model.generate_full_cuda_with_cache(&tokens, &gen_config);

        std::env::remove_var("SKIP_CUDA_GRAPH");
        std::env::remove_var("REALIZAR_DEBUG_FORWARD");

        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("SUMMARY");
        eprintln!("═══════════════════════════════════════════════════════════════════════");
        eprintln!("\nIf PAR-052 debug output shows same embedding and normed values,");
        eprintln!("then the bug is in the GPU matmul dispatch, not the input.");
    }

    /// Phase 16b: Direct CPU vs GPU Q4K GEMV comparison with same input
    ///
    /// Uses the exact same normed input vector and weight pointer to compare
    /// CPU fused_q4k_parallel_matvec vs GPU q4k_gemv_cached.
    #[test]
    fn test_phase16b_direct_qkv_gemv() {
        use realizar::cuda::CudaExecutor;

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16b: DIRECT CPU vs GPU Q4K GEMV COMPARISON                    ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let config = cpu_model.config();
        let hidden_dim = config.hidden_dim;
        let eps = config.eps;

        // Get embedding for BOS token
        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        let embedding = cpu_model.embed(&[bos_token]);

        // Compute RMSNorm
        let layer = &cpu_model.layers()[0];
        let attn_norm_weight = &layer.attn_norm_weight;
        let sq_sum: f32 = embedding.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        let normed: Vec<f32> = embedding
            .iter()
            .zip(attn_norm_weight.iter())
            .map(|(&x, &w)| x / rms * w)
            .collect();

        eprintln!(
            "Normed input L2: {:.6}\n",
            normed.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Get Q weight (Q4_K, qtype=12)
        let (q_data, q_in_dim, q_out_dim) = match &layer.qkv_weight {
            OwnedQKVWeights::Separate { q, .. } => (&q.data, q.in_dim, q.out_dim),
            OwnedQKVWeights::Fused(_) => {
                eprintln!("⚠️  Skipping: model uses fused QKV");
                return;
            },
        };

        eprintln!("Q weight: {}x{} (Q4_K)", q_in_dim, q_out_dim);

        // CPU Q projection
        let cpu_q = fused_q4k_parallel_matvec(q_data, &normed, q_in_dim, q_out_dim)
            .expect("CPU Q matmul failed");
        let cpu_q_l2 = cpu_q.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("CPU Q output L2: {:.6}", cpu_q_l2);
        eprintln!("CPU Q first 8: {:?}", &cpu_q[..8.min(cpu_q.len())]);

        // GPU Q projection
        eprintln!("\nAttempting GPU Q projection...");
        match CudaExecutor::new(0) {
            Ok(mut executor) => {
                // Upload Q weight to GPU cache
                let cache_key = "test_q_weight".to_string();
                executor
                    .load_quantized_weights_with_type(&cache_key, q_data, 12)
                    .expect("Failed to upload Q weight");

                // Allocate output buffer
                let mut gpu_q = vec![0.0f32; q_out_dim];

                // Run GPU GEMV
                match executor.q4k_gemv_cached(
                    &cache_key,
                    &normed,
                    &mut gpu_q,
                    q_out_dim as u32,
                    q_in_dim as u32,
                ) {
                    Ok(()) => {
                        let gpu_q_l2 = gpu_q.iter().map(|x| x * x).sum::<f32>().sqrt();
                        eprintln!("GPU Q output L2: {:.6}", gpu_q_l2);
                        eprintln!("GPU Q first 8: {:?}", &gpu_q[..8.min(gpu_q.len())]);

                        // Compare
                        let diff_l2: f32 = cpu_q
                            .iter()
                            .zip(gpu_q.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                            .sqrt();

                        let relative_diff = diff_l2 / cpu_q_l2 * 100.0;
                        eprintln!("\n--- COMPARISON ---");
                        eprintln!("Diff L2: {:.6}", diff_l2);
                        eprintln!("Relative diff: {:.4}%", relative_diff);

                        if relative_diff < 1.0 {
                            eprintln!("✅ Q projection MATCHES within 1%");
                            eprintln!("   The matmul kernel is correct.");
                            eprintln!("   The bug must be ELSEWHERE in the forward pass.");
                        } else {
                            eprintln!("❌ Q projection DIVERGES!");
                            eprintln!("   Investigating further...");

                            // Check for systematic offset
                            let avg_diff: f32 = cpu_q
                                .iter()
                                .zip(gpu_q.iter())
                                .map(|(a, b)| a - b)
                                .sum::<f32>()
                                / cpu_q.len() as f32;
                            eprintln!("   Average element difference: {:.6}", avg_diff);

                            // Check ratio
                            let ratio: f32 = gpu_q_l2 / cpu_q_l2;
                            eprintln!("   GPU/CPU L2 ratio: {:.4}", ratio);
                        }
                    },
                    Err(e) => {
                        eprintln!("❌ GPU GEMV failed: {:?}", e);
                    },
                }
            },
            Err(e) => {
                eprintln!("⚠️ CUDA not available: {:?}", e);
            },
        }
    }

    /// Phase 16c: Trace entire first layer CPU vs GPU
    ///
    /// Traces Q, K, V, attention output, FFN to find where divergence starts.
    #[test]
    fn test_phase16c_layer_trace() {
        use realizar::cuda::CudaExecutor;

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16c: FIRST LAYER TRACE - FINDING DIVERGENCE POINT             ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let config = cpu_model.config();
        let hidden_dim = config.hidden_dim;
        let eps = config.eps;

        // Get embedding for BOS token
        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        let embedding = cpu_model.embed(&[bos_token]);

        // Compute RMSNorm
        let layer = &cpu_model.layers()[0];
        let attn_norm_weight = &layer.attn_norm_weight;
        let sq_sum: f32 = embedding.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        let normed: Vec<f32> = embedding
            .iter()
            .zip(attn_norm_weight.iter())
            .map(|(&x, &w)| x / rms * w)
            .collect();

        eprintln!(
            "Input: embedding after RMSNorm, L2={:.6}\n",
            normed.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Get weight references
        let (q_data, q_in, q_out, k_data, k_in, k_out, v_data, v_in, v_out, v_qtype) =
            match &layer.qkv_weight {
                OwnedQKVWeights::Separate { q, k, v } => (
                    &q.data, q.in_dim, q.out_dim, &k.data, k.in_dim, k.out_dim, &v.data, v.in_dim,
                    v.out_dim, v.qtype,
                ),
                _ => {
                    eprintln!("⚠️  Model uses fused QKV, skipping");
                    return;
                },
            };

        match CudaExecutor::new(0) {
            Ok(mut executor) => {
                // Upload Q, K, V weights
                let q_key = "test_layer0_q".to_string();
                let k_key = "test_layer0_k".to_string();
                let v_key = "test_layer0_v".to_string();

                executor
                    .load_quantized_weights_with_type(&q_key, q_data, 12)
                    .expect("Q upload");
                executor
                    .load_quantized_weights_with_type(&k_key, k_data, 12)
                    .expect("K upload");
                executor
                    .load_quantized_weights_with_type(&v_key, v_data, v_qtype)
                    .expect("V upload");

                // === Q Projection ===
                let cpu_q = fused_q4k_parallel_matvec(q_data, &normed, q_in, q_out).expect("CPU Q");
                let mut gpu_q = vec![0.0f32; q_out];
                executor
                    .q4k_gemv_cached(&q_key, &normed, &mut gpu_q, q_out as u32, q_in as u32)
                    .expect("GPU Q");

                let q_diff = cpu_q
                    .iter()
                    .zip(gpu_q.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                let q_rel = q_diff / cpu_q.iter().map(|x| x * x).sum::<f32>().sqrt() * 100.0;
                eprintln!(
                    "Q: CPU L2={:.4}, GPU L2={:.4}, diff={:.6} ({:.4}%) {}",
                    cpu_q.iter().map(|x| x * x).sum::<f32>().sqrt(),
                    gpu_q.iter().map(|x| x * x).sum::<f32>().sqrt(),
                    q_diff,
                    q_rel,
                    if q_rel < 1.0 { "✅" } else { "❌" }
                );

                // === K Projection ===
                let cpu_k = fused_q4k_parallel_matvec(k_data, &normed, k_in, k_out).expect("CPU K");
                let mut gpu_k = vec![0.0f32; k_out];
                executor
                    .q4k_gemv_cached(&k_key, &normed, &mut gpu_k, k_out as u32, k_in as u32)
                    .expect("GPU K");

                let k_diff = cpu_k
                    .iter()
                    .zip(gpu_k.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                let k_rel = k_diff / cpu_k.iter().map(|x| x * x).sum::<f32>().sqrt() * 100.0;
                eprintln!(
                    "K: CPU L2={:.4}, GPU L2={:.4}, diff={:.6} ({:.4}%) {}",
                    cpu_k.iter().map(|x| x * x).sum::<f32>().sqrt(),
                    gpu_k.iter().map(|x| x * x).sum::<f32>().sqrt(),
                    k_diff,
                    k_rel,
                    if k_rel < 1.0 { "✅" } else { "❌" }
                );

                // === V Projection (Q6_K, qtype=14) ===
                let cpu_v = fused_q6k_parallel_matvec(v_data, &normed, v_in, v_out).expect("CPU V");
                let mut gpu_v = vec![0.0f32; v_out];

                // V uses Q6_K - need q6k_gemv_cached
                let v_result = executor.q6k_gemv_cached(
                    &v_key,
                    &normed,
                    &mut gpu_v,
                    v_out as u32,
                    v_in as u32,
                );
                match v_result {
                    Ok(()) => {
                        let v_diff = cpu_v
                            .iter()
                            .zip(gpu_v.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                            .sqrt();
                        let v_rel =
                            v_diff / cpu_v.iter().map(|x| x * x).sum::<f32>().sqrt() * 100.0;
                        eprintln!(
                            "V (Q6_K): CPU L2={:.4}, GPU L2={:.4}, diff={:.6} ({:.4}%) {}",
                            cpu_v.iter().map(|x| x * x).sum::<f32>().sqrt(),
                            gpu_v.iter().map(|x| x * x).sum::<f32>().sqrt(),
                            v_diff,
                            v_rel,
                            if v_rel < 1.0 { "✅" } else { "❌" }
                        );
                    },
                    Err(e) => {
                        eprintln!("V (Q6_K): GPU GEMV failed: {:?}", e);
                        eprintln!(
                            "  CPU V L2={:.4}",
                            cpu_v.iter().map(|x| x * x).sum::<f32>().sqrt()
                        );
                    },
                }

                eprintln!("\n--- Summary ---");
                if q_rel < 1.0 && k_rel < 1.0 {
                    eprintln!("✅ Q and K match. Bug might be in:");
                    eprintln!("   - V projection (Q6_K path)");
                    eprintln!("   - Attention computation");
                    eprintln!("   - FFN projections");
                    eprintln!("   - Layer accumulation");
                } else {
                    eprintln!("❌ Q or K diverges - this is unexpected!");
                }
            },
            Err(e) => {
                eprintln!("⚠️ CUDA not available: {:?}", e);
            },
        }
    }
}
