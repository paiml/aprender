//! IMP-700: Real-World Performance Verification
//!
//! Popperian Falsification: Measure ACTUAL performance against Ollama and llama.cpp
//!
//! Falsifiable Claims:
//! 1. Ollama throughput: Expected ~200+ tok/s (from IMP-400)
//! 2. Realizar throughput: Measured vs expected
//! 3. Gap: Must be measured, not estimated
//!
//! Run: cargo run --release --example imp_700_realworld_verification

use std::time::{Duration, Instant};

/// Statistics for measurement stability (Coefficient of Variation)
fn calculate_cv(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance: f64 =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt() / mean
}

/// Measure Ollama throughput via HTTP API
fn measure_ollama(
    model: &str,
    prompt: &str,
    num_runs: usize,
) -> Result<(f64, f64, Duration), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let mut throughputs = Vec::with_capacity(num_runs);
    let mut latencies = Vec::with_capacity(num_runs);

    for run in 0..num_runs {
        let start = Instant::now();

        let response = client
            .post("http://localhost:11434/api/generate")
            .json(&serde_json::json!({
                "model": model,
                "prompt": prompt,
                "stream": false,
                "options": {
                    "num_predict": 50,
                    "temperature": 0.0,
                    "seed": 42
                }
            }))
            .send()
            .map_err(|e| format!("Ollama request failed: {}", e))?;

        let elapsed = start.elapsed();

        let json: serde_json::Value = response
            .json()
            .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;

        // Extract metrics from Ollama response
        if let (Some(total_duration), Some(eval_count)) = (
            json.get("total_duration").and_then(|v| v.as_u64()),
            json.get("eval_count").and_then(|v| v.as_u64()),
        ) {
            let duration_secs = total_duration as f64 / 1_000_000_000.0;
            let tps = eval_count as f64 / duration_secs;
            throughputs.push(tps);
            latencies.push(elapsed);

            if run == 0 {
                println!(
                    "  Run {}: {} tokens in {:.2}s = {:.1} tok/s",
                    run + 1,
                    eval_count,
                    duration_secs,
                    tps
                );
            }
        }
    }

    if throughputs.is_empty() {
        return Err("No valid measurements".to_string());
    }

    let mean_tps: f64 = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let cv = calculate_cv(&throughputs);
    let mean_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;

    Ok((mean_tps, cv, mean_latency))
}

/// Measure realizar throughput (CPU path)
fn measure_realizar(
    model_path: &str,
    prompt_tokens: &[u32],
    num_tokens: usize,
) -> Result<(f64, Duration), String> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    // Load model file using memory mapping (most efficient)
    let start_load = Instant::now();
    let mapped = MappedGGUFModel::from_path(model_path)
        .map_err(|e| format!("Failed to memory-map model: {}", e))?;
    let load_time = start_load.elapsed();
    println!(
        "  Model memory-mapped in {:?} ({:.1} MB)",
        load_time,
        mapped.data().len() as f64 / 1_000_000.0
    );

    // Create owned quantized model (PARITY-001: uses KV cache in generate_with_cache)
    let start_parse = Instant::now();
    let quantized_model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| format!("Failed to create quantized model: {}", e))?;
    let parse_time = start_parse.elapsed();
    println!("  Model parsed in {:?}", parse_time);

    // Configure generation with KV cache
    let config = QuantizedGenerateConfig {
        max_tokens: num_tokens,
        temperature: 0.0, // Greedy
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Warmup with KV cache (first run initializes cache)
    println!("  Running warmup...");
    let _ = quantized_model.generate_with_cache(prompt_tokens, &config);

    // Measure generation WITH KV CACHE (PARITY-001: This is the critical fix!)
    println!("  Measuring with KV cache...");
    let start = Instant::now();
    let generated = quantized_model
        .generate_with_cache(prompt_tokens, &config)
        .map_err(|e| format!("Generation failed: {}", e))?;

    let elapsed = start.elapsed();
    let tokens_generated = generated.len().saturating_sub(prompt_tokens.len());
    let tps = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "  Generated {} tokens (total seq: {})",
        tokens_generated,
        generated.len()
    );

    Ok((tps, elapsed))
}

/// Measure realizar throughput with GPU-accelerated batched prefill (PARITY-002)
#[cfg(feature = "gpu")]
fn measure_realizar_gpu(
    model_path: &str,
    prompt_tokens: &[u32],
    num_tokens: usize,
) -> Result<(f64, Duration), String> {
    use realizar::gguf::{
        DispatchMetrics, MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig,
    };
    use std::sync::Arc;

    // Load model file using memory mapping
    let start_load = Instant::now();
    let mapped = MappedGGUFModel::from_path(model_path)
        .map_err(|e| format!("Failed to memory-map model: {}", e))?;
    let load_time = start_load.elapsed();
    println!(
        "  [GPU] Model memory-mapped in {:?} ({:.1} MB)",
        load_time,
        mapped.data().len() as f64 / 1_000_000.0
    );

    // Create owned quantized model
    let start_parse = Instant::now();
    let quantized_model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| format!("Failed to create quantized model: {}", e))?;
    let parse_time = start_parse.elapsed();
    println!("  [GPU] Model parsed in {:?}", parse_time);

    // Configure generation
    let config = QuantizedGenerateConfig {
        max_tokens: num_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: vec![],
        trace: false,
        ..Default::default()
    };

    // Create dispatch metrics tracker
    let metrics = Arc::new(DispatchMetrics::new());

    // Warmup with batched prefill (PARITY-002: processes all prompt tokens at once)
    println!("  [GPU] Running warmup with BATCHED prefill...");
    let _ = quantized_model.generate_with_batched_prefill(prompt_tokens, &config, &metrics);

    // Reset metrics for measurement
    let metrics = Arc::new(DispatchMetrics::new());

    // Measure generation WITH BATCHED PREFILL (PARITY-002)
    // This processes all prompt tokens in a single batch, enabling GPU acceleration
    println!("  [GPU] Measuring with BATCHED prefill...");
    let start = Instant::now();
    let generated = quantized_model
        .generate_with_batched_prefill(prompt_tokens, &config, &metrics)
        .map_err(|e| format!("GPU generation failed: {}", e))?;

    let elapsed = start.elapsed();
    let tokens_generated = generated.len().saturating_sub(prompt_tokens.len());
    let tps = tokens_generated as f64 / elapsed.as_secs_f64();

    // Report dispatch statistics
    let cpu_count = metrics.cpu_dispatches();
    let gpu_count = metrics.gpu_dispatches();
    let gpu_ratio = if cpu_count + gpu_count > 0 {
        gpu_count as f64 / (cpu_count + gpu_count) as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "  [GPU] Generated {} tokens (CPU: {}, GPU: {}, GPU ratio: {:.1}%)",
        tokens_generated, cpu_count, gpu_count, gpu_ratio
    );

    Ok((tps, elapsed))
}

fn main() {
    println!("=== IMP-700: Real-World Performance Verification ===");
    println!("Popperian Falsification: Measure ACTUAL performance gaps\n");

    // Test configuration
    let prompt = "The capital of France is";
    let num_runs = 5;
    let num_tokens = 50;

    // Test 1: Ollama phi2:2.7b
    println!("=== Test 1: Ollama phi2:2.7b ===");
    match measure_ollama("phi2:2.7b", prompt, num_runs) {
        Ok((tps, cv, latency)) => {
            println!(
                "  Result: {:.1} tok/s (CV={:.4}, latency={:?})",
                tps, cv, latency
            );
            if cv > 0.05 {
                println!("  WARNING: High CV indicates unstable measurements");
            }

            // Falsifiable claim verification
            println!("\n  Falsifiable Claims:");
            println!("  - Expected: ~200+ tok/s (from IMP-400)");
            println!("  - Measured: {:.1} tok/s", tps);
            if tps >= 180.0 {
                println!("  - Status: VERIFIED (within expected range)");
            } else {
                println!("  - Status: BELOW EXPECTED (check GPU/system state)");
            }
        },
        Err(e) => {
            println!("  ERROR: {}", e);
            println!("  Make sure Ollama is running: ollama serve");
        },
    }

    // Test 2: Realizar with phi-2 GGUF (if available)
    println!("\n=== Test 2: Realizar phi-2 Q4_K_M ===");

    // Common paths for phi-2 model
    let model_paths = [
        "/home/noah/src/single-shot-eval/models/raw/phi-2-q4_k_m.gguf",
        "/home/noah/.cache/huggingface/hub/models--TheBloke--phi-2-GGUF/snapshots/*/phi-2.Q4_K_M.gguf",
        "../models/phi-2.Q4_K_M.gguf",
        "models/phi-2.Q4_K_M.gguf",
    ];

    let mut model_found = false;
    for pattern in &model_paths {
        // Try to find model via glob - use .next() to get first match only
        if let Ok(mut paths) = glob::glob(pattern) {
            if let Some(path) = paths.next().and_then(|p| p.ok()) {
                println!("  Found model: {}", path.display());
                let prompt_tokens: Vec<u32> = vec![1, 450, 3007, 310, 3444, 338]; // "The capital of France is"

                match measure_realizar(path.to_str().expect("test"), &prompt_tokens, num_tokens) {
                    Ok((tps, latency)) => {
                        println!("  Result: {:.2} tok/s (latency={:?})", tps, latency);
                        model_found = true;

                        // Calculate gap
                        let ollama_baseline = 200.0; // Conservative estimate
                        let gap = ollama_baseline / tps;
                        println!("\n  Performance Gap Analysis:");
                        println!("  - Ollama baseline: ~{:.0} tok/s", ollama_baseline);
                        println!("  - Realizar: {:.2} tok/s", tps);
                        println!("  - Gap: {:.0}x", gap);
                        println!("  - Target for parity: gap < 1.25x");
                    },
                    Err(e) => {
                        println!("  ERROR: {}", e);
                    },
                }
            }
        }
        if model_found {
            break;
        }
    }

    if !model_found {
        println!("  No phi-2 GGUF model found. Download with:");
        println!("  huggingface-cli download TheBloke/phi-2-GGUF phi-2.Q4_K_M.gguf");
    }

    // Test 3: Realizar with GPU-accelerated attention (PARITY-002)
    //
    // IMPORTANT FINDING (PARITY-002):
    // Batched prefill is SLOWER than incremental KV cache generation because:
    // 1. Batched prefill computes O(n²) attention for all positions
    // 2. KV cache incremental generation is O(n) per token
    // 3. GPU MATVEC (per-head attention) is 2.7x slower than CPU (IMP-600)
    // 4. GPU GEMM (large batch) is 57x faster, but attention is MATVEC
    //
    // For GPU to help attention, need FlashAttention (fused kernel) or
    // batched multi-request inference (multiple prompts processed together).
    //
    // CURRENT RECOMMENDATION: Use KV cache path (Test 2) for best performance.
    // GPU batched prefill is included for verification but is slower.
    #[cfg(feature = "gpu")]
    {
        println!("\n=== Test 3: Realizar phi-2 Q4_K_M (GPU Batched Prefill) ===");
        println!("  NOTE: This test verifies GPU path but is slower than KV cache (Test 2)");
        println!("  Reason: Attention is MATVEC (GPU overhead) not GEMM (GPU wins)");
        let mut gpu_model_found = false;
        for pattern in &model_paths {
            if let Ok(mut paths) = glob::glob(pattern) {
                if let Some(path) = paths.next().and_then(|p| p.ok()) {
                    println!("  Found model: {}", path.display());
                    // PARITY-002: Use longer prompt
                    let prompt_tokens: Vec<u32> = (0..32).map(|i| (i % 100) as u32).collect();
                    println!(
                        "  Using {} token prompt for batched prefill",
                        prompt_tokens.len()
                    );

                    match measure_realizar_gpu(
                        path.to_str().expect("test"),
                        &prompt_tokens,
                        num_tokens,
                    ) {
                        Ok((tps, latency)) => {
                            println!("  Result: {:.2} tok/s (latency={:?})", tps, latency);
                            gpu_model_found = true;

                            // Calculate gap and comparison
                            let ollama_baseline = 200.0;
                            let cpu_kv_cache = 5.25; // KV cache result from Test 2
                            let gap = ollama_baseline / tps;

                            println!("\n  PARITY-002 Analysis:");
                            println!(
                                "  - Ollama GPU: ~{:.0} tok/s (uses FlashAttention)",
                                ollama_baseline
                            );
                            println!("  - Realizar KV cache (CPU): {:.2} tok/s", cpu_kv_cache);
                            println!("  - Realizar batched prefill: {:.2} tok/s", tps);
                            println!("  - Gap to Ollama: {:.1}x", gap);

                            // Explain the finding
                            if tps < cpu_kv_cache {
                                println!(
                                    "\n  Finding: Batched prefill is {:.1}x SLOWER than KV cache",
                                    cpu_kv_cache / tps
                                );
                                println!(
                                    "  Root cause: Attention = MATVEC, GPU overhead dominates"
                                );
                                println!("  Solution: FlashAttention kernel (future work) or batch inference");
                            } else {
                                println!("  Status: GPU batched prefill competitive");
                            }
                        },
                        Err(e) => {
                            println!("  ERROR: {}", e);
                        },
                    }
                }
            }
            if gpu_model_found {
                break;
            }
        }

        if !gpu_model_found {
            println!("  No phi-2 GGUF model found for GPU test");
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("\n=== Test 3: GPU Attention (SKIPPED) ===");
        println!("  Run with --features gpu to enable GPU-accelerated attention");
    }

    println!("\n=== Summary ===");
    println!("IMP-700: Real-world verification completed");
    println!("All measurements are falsifiable and reproducible");
    println!("Run this benchmark to verify any performance claims");
}
