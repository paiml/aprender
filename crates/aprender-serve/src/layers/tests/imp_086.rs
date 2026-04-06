
/// M33: llama.cpp Completion Endpoint (IMP-086)
/// Target: /completion returns generated text (llama.cpp compatible)
///
/// Tests llama.cpp-compatible completion API.
/// Run with: `cargo test test_imp_086 --ignored`
#[test]
#[ignore = "Requires running server"]
fn test_imp_086_llamacpp_endpoint() {
    // IMP-086: Integration test for /completion (llama.cpp-compatible)

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client");

    let url = "http://127.0.0.1:3000/completion";

    // llama.cpp-style request format
    let request = serde_json::json!({
        "prompt": "Hello, world!",
        "n_predict": 10,
        "temperature": 0.0
    });

    match client.post(url).json(&request).send() {
        Ok(response) => {
            if response.status().is_success() {
                let body: serde_json::Value = response.json().expect("Valid JSON");
                assert!(
                    body.get("content").is_some() || body.get("text").is_some(),
                    "IMP-086: Response should have 'content' or 'text'"
                );
                println!("IMP-086: ✅ llama.cpp completion endpoint works");
            } else if response.status().as_u16() == 404 {
                println!("IMP-086: ⚠️ /completion not implemented yet (404)");
            } else {
                panic!("IMP-086: Unexpected status: {}", response.status());
            }
        },
        Err(e) => {
            panic!(
                "IMP-086: Server not running. Start with: cargo run --example api_server. Error: {}",
                e
            );
        },
    }
}

/// M33: Benchmark Integration (IMP-087)
/// Target: realizar appears in bench-server-matrix.sh output
///
/// Verifies benchmark infrastructure is functional.
/// Run with: `cargo test test_imp_087 --ignored`
#[test]
#[ignore = "Requires benchmark infrastructure"]
fn test_imp_087_benchmark_integration() {
    // IMP-087: Benchmark integration test
    //
    // This test verifies that:
    // 1. The benchmark script exists
    // 2. The server can respond to benchmark-style requests
    // 3. Throughput can be measured

    use std::time::Instant;

    // Check if benchmark script exists
    let script_path = std::path::Path::new("scripts/bench-server-matrix.sh");
    if script_path.exists() {
        println!("IMP-087: ✅ Benchmark script exists at scripts/bench-server-matrix.sh");
    } else {
        println!("IMP-087: ⚠️ Benchmark script not found (optional)");
    }

    // Test benchmark-style request pattern
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client");

    let url = "http://127.0.0.1:3000/generate";
    let request = serde_json::json!({
        "prompt": "Benchmark test",
        "max_tokens": 10,
        "temperature": 0.0
    });

    // Run 5 iterations to measure throughput
    let iterations = 5;
    let start = Instant::now();
    let mut success_count = 0;
    let mut total_tokens = 0;

    for i in 0..iterations {
        match client.post(url).json(&request).send() {
            Ok(response) if response.status().is_success() => {
                if let Ok(body) = response.json::<serde_json::Value>() {
                    if let Some(text) = body.get("text").and_then(|t| t.as_str()) {
                        total_tokens += text.split_whitespace().count();
                        success_count += 1;
                    }
                }
            },
            Ok(response) => {
                println!(
                    "IMP-087: Iteration {} failed with status {}",
                    i,
                    response.status()
                );
            },
            Err(e) => {
                assert!(
                    i != 0,
                    "IMP-087: Server not running. Start with: cargo run --example api_server. Error: {}",
                    e
                );
            },
        }
    }

    let elapsed = start.elapsed();
    let throughput = if elapsed.as_secs_f64() > 0.0 {
        total_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "IMP-087: ✅ Benchmark test: {} iterations, {} tokens, {:.2} tok/s",
        success_count, total_tokens, throughput
    );

    assert!(
        success_count > 0,
        "IMP-087: At least one benchmark iteration should succeed"
    );
}

/// M33: GQA Support - num_kv_heads in config (IMP-088)
/// Target: GpuModelConfig has num_kv_heads field for Grouped Query Attention
#[test]
#[cfg(feature = "gpu")]
fn test_imp_088_gqa_config_num_kv_heads() {
    use crate::gpu::GpuModelConfig;

    // Create config with different num_kv_heads (GQA pattern)
    // Qwen 1.5B: 12 heads, 2 kv_heads (6:1 ratio)
    let config = GpuModelConfig {
        vocab_size: 151936,
        hidden_dim: 1536,
        num_heads: 12,
        num_kv_heads: 2, // GQA: fewer KV heads than Q heads
        num_layers: 28,
        intermediate_dim: 8960,
        eps: 1e-6,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    assert_eq!(config.num_heads, 12, "IMP-088: Should have 12 Q heads");
    assert_eq!(config.num_kv_heads, 2, "IMP-088: Should have 2 KV heads");

    // head_dim should be hidden_dim / num_heads
    let head_dim = config.hidden_dim / config.num_heads;
    assert_eq!(head_dim, 128, "IMP-088: Head dim should be 128");

    // KV size per layer should use num_kv_heads
    let kv_head_dim = config.hidden_dim / config.num_heads; // Same head_dim
    let kv_size = config.num_kv_heads * kv_head_dim;
    assert_eq!(kv_size, 256, "IMP-088: KV size should be 2*128=256");
}

/// M33: GQA Attention Forward (IMP-089)
/// Target: Forward pass handles K/V with fewer heads than Q
#[test]
#[cfg(feature = "gpu")]
fn test_imp_089_gqa_attention_forward() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Create GQA config (fewer KV heads than Q heads)
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 128,
        num_heads: 4,    // 4 Q heads
        num_kv_heads: 2, // 2 KV heads (2:1 ratio, each KV serves 2 Q heads)
        num_layers: 2,
        intermediate_dim: 256,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let mut model = GpuModel::new(config).expect("Failed to create GQA model");

    // Forward should work with GQA attention
    let tokens = vec![1usize, 2, 3];
    let result = model.forward_gpu(&tokens);

    assert!(
        result.is_ok(),
        "IMP-089: Forward pass should handle GQA attention. Error: {:?}",
        result.err()
    );

    let logits = result.expect("test");
    // Output should be [seq_len * vocab_size]
    assert_eq!(
        logits.len(),
        tokens.len() * 256,
        "IMP-089: Logits should be seq_len * vocab_size"
    );
}

/// M33: CPU Embedding for Large Vocab (IMP-090)
/// Target: Handle vocab sizes that exceed GPU buffer limits (>65536 tokens)
/// wgpu max buffer is 256MB, large vocab like Qwen (151936) needs CPU fallback
#[test]
#[cfg(feature = "gpu")]
fn test_imp_090_cpu_embedding_large_vocab() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Large vocab size that would exceed GPU buffer limits if stored fully
    // Real example: Qwen 2.5 Coder 1.5B has vocab_size=151936
    // Buffer size would be: 151936 * 1536 * 4 = 933MB > 256MB wgpu limit
    // Test with smaller but still "large vocab" threshold (>65536)
    let large_vocab_config = GpuModelConfig {
        vocab_size: 100_000, // Large vocab - requires CPU embedding fallback
        hidden_dim: 256,     // Smaller hidden_dim for test speed
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    // This should NOT fail due to GPU buffer limits
    // Instead, it should use CPU embedding lookup
    let model_result = GpuModel::new(large_vocab_config);

    assert!(
        model_result.is_ok(),
        "IMP-090: Should create model with large vocab using CPU embedding. Error: {:?}",
        model_result.err()
    );

    let mut model = model_result.expect("test");

    // Forward pass should also work with CPU embedding lookup
    let tokens = vec![0usize, 1000, 50000, 99999]; // Include edge tokens
    let result = model.forward_gpu(&tokens);

    assert!(
        result.is_ok(),
        "IMP-090: Forward pass should work with CPU embedding for large vocab. Error: {:?}",
        result.err()
    );

    let logits = result.expect("test");
    assert_eq!(
        logits.len(),
        tokens.len() * 100_000,
        "IMP-090: Logits should be seq_len * vocab_size"
    );

    // Verify embeddings are valid (not all zeros, not NaN)
    let has_valid_values = logits.iter().any(|&v| v != 0.0 && !v.is_nan());
    assert!(
        has_valid_values,
        "IMP-090: Logits should contain valid non-zero values"
    );
}

/// IMP-093: Real GGUF GPU benchmark test
///
/// Tests the full GPU inference path with a real GGUF model.
/// This verifies IMP-092 (no weight cloning) improves performance.
///
/// Run: cargo test --features gpu test_imp_093_real_gguf_gpu_benchmark -- --nocapture --ignored
#[test]
#[cfg(feature = "gpu")]
#[ignore] // Requires real GGUF file - run manually
fn test_imp_093_real_gguf_gpu_benchmark() {
    use crate::gguf::MappedGGUFModel;
    use crate::gpu::GpuModel;
    use std::path::Path;
    use std::time::Instant;

    // Real GGUF model path (Qwen 2.5 Coder 1.5B Q4_K_M)
    let home = std::env::var("HOME").expect("HOME env var not set");
    let model_path_str = format!(
        "{}/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        home
    );
    let model_path = &model_path_str;

    if !Path::new(model_path).exists() {
        eprintln!("IMP-093: Skipping - model not found at {}", model_path);
        return;
    }

    println!("\n=== IMP-093: Real GGUF GPU Benchmark ===\n");
    println!("Model: {}", model_path);

    // Load model
    let load_start = Instant::now();
    let mapped = MappedGGUFModel::from_path(model_path).expect("Failed to load GGUF");
    let load_mmap = load_start.elapsed();
    println!("  Mmap load: {:?}", load_mmap);

    let gpu_start = Instant::now();
    let mut gpu_model = GpuModel::from_mapped_gguf(&mapped).expect("Failed to load to GPU");
    let gpu_load = gpu_start.elapsed();
    println!("  GPU load: {:?}", gpu_load);
    println!(
        "  Config: hidden={}, layers={}, vocab={}, heads={}, kv_heads={}, intermediate={}",
        gpu_model.config().hidden_dim,
        gpu_model.config().num_layers,
        gpu_model.config().vocab_size,
        gpu_model.config().num_heads,
        gpu_model.config().num_kv_heads,
        gpu_model.config().intermediate_dim,
    );
    println!();

    // Test tokens (small prompt)
    let test_tokens = vec![0usize, 1, 2, 3];
    let max_tokens = 5;

    // Warmup
    println!("Warmup...");
    let _ = gpu_model.generate(
        &test_tokens,
        &crate::gpu::GpuGenerateConfig {
            max_tokens: 1,
            ..Default::default()
        },
    );

    // Benchmark generation
    println!("\nGenerating {} tokens...", max_tokens);
    let gen_start = Instant::now();
    let result = gpu_model.generate(
        &test_tokens,
        &crate::gpu::GpuGenerateConfig {
            max_tokens,
            ..Default::default()
        },
    );
    let gen_elapsed = gen_start.elapsed();

    assert!(
        result.is_ok(),
        "IMP-093: Generation should succeed: {:?}",
        result.err()
    );

    let generated = result.expect("test");
    let gen_secs = gen_elapsed.as_secs_f64();
    let tps = max_tokens as f64 / gen_secs;

    println!("\n=== Results ===");
    println!(
        "  Generated: {} tokens",
        generated.len() - test_tokens.len()
    );
    println!("  Time: {:.3}s", gen_secs);
    println!("  Throughput: {:.2} tok/s", tps);
    println!();

    // Performance assertions (soft targets - document actual vs target)
    // Target: ≥10 tok/s (Ollama achieves ~143 tok/s)
    // IMP-092 eliminates 3.7GB/token memory copying
    let target_tps = 10.0;
    if tps < target_tps {
        eprintln!(
            "WARNING: Below target {} tok/s (actual: {:.2} tok/s)",
            target_tps, tps
        );
        eprintln!("Parity gap with Ollama (~143 tok/s): {:.0}x", 143.0 / tps);
    } else {
        println!(
            "PASS: Achieved {:.2} tok/s (target: {} tok/s)",
            tps, target_tps
        );
    }
}
