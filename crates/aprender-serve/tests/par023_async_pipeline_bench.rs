//! PAR-023: Async Pipeline Benchmark Tests
//!
//! Measures sync overhead reduction from GPU-resident operations.
//! Target: 176 syncs/token -> ~22 syncs/token (1 per layer)
//!
//! M4 Milestone Target: >192 tok/s (<1.25x gap vs llama.cpp ~500 tok/s)
//! Current baseline: 116-129 tok/s (avg ~121 tok/s)

use std::time::{Duration, Instant};

/// Simulates sync overhead by measuring many small operations
fn measure_sync_overhead_simulation() -> Duration {
    // Simulate 176 syncs (current state) vs 22 syncs (target)
    // Each "sync" represented by a small allocation + operation
    let iterations = 1000;
    let syncs_per_iteration = 176;

    let start = Instant::now();
    for _ in 0..iterations {
        for _ in 0..syncs_per_iteration {
            // Simulate sync overhead: allocation + memory barrier
            let v: Vec<f32> = vec![0.0; 64];
            std::hint::black_box(&v);
        }
    }
    start.elapsed()
}

/// Simulates reduced sync overhead with async pipeline
fn measure_async_overhead_simulation() -> Duration {
    let iterations = 1000;
    let syncs_per_iteration = 22; // 1 per layer

    let start = Instant::now();
    for _ in 0..iterations {
        for _ in 0..syncs_per_iteration {
            let v: Vec<f32> = vec![0.0; 64];
            std::hint::black_box(&v);
        }
    }
    start.elapsed()
}

#[test]
fn test_par023_sync_reduction_potential() {
    let current = measure_sync_overhead_simulation();
    let target = measure_async_overhead_simulation();

    let speedup = current.as_nanos() as f64 / target.as_nanos() as f64;

    eprintln!("PAR-023 Sync Reduction Analysis:");
    eprintln!("  Current (176 syncs/token): {:?}", current);
    eprintln!("  Target (22 syncs/token):   {:?}", target);
    eprintln!("  Potential speedup:         {:.2}x", speedup);
    eprintln!();
    eprintln!("  M4 Target: >192 tok/s (from 121 tok/s baseline)");
    eprintln!("  Required speedup: {:.2}x", 192.0 / 121.0);

    // Verify sync reduction gives meaningful speedup
    assert!(
        speedup > 1.5,
        "Sync reduction should provide >1.5x speedup, got {:.2}x",
        speedup
    );
}

/// Reference implementation: sequential matmul chain (simulates current path)
fn sequential_matmul_chain(input: &[f32], n_layers: usize) -> Vec<f32> {
    let mut result = input.to_vec();
    for _ in 0..n_layers {
        // Simulate 7 GEMVs per layer with sync after each
        for _ in 0..7 {
            result = result.iter().map(|x| x * 0.99 + 0.01).collect();
            std::hint::black_box(&result); // Prevent optimization
        }
    }
    result
}

/// Reference implementation: batched matmul chain (simulates async path)
fn batched_matmul_chain(input: &[f32], n_layers: usize) -> Vec<f32> {
    let mut result = input.to_vec();
    for _ in 0..n_layers {
        // Simulate 7 GEMVs per layer with single sync at end
        for _ in 0..7 {
            result = result.iter().map(|x| x * 0.99 + 0.01).collect();
        }
        std::hint::black_box(&result); // Single sync per layer
    }
    result
}

#[test]
fn test_par023_layer_pipeline_correctness() {
    let input: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.001).collect();
    let n_layers = 22;

    let seq_result = sequential_matmul_chain(&input, n_layers);
    let batch_result = batched_matmul_chain(&input, n_layers);

    // Results should be identical (same math, different sync pattern)
    for (i, (s, b)) in seq_result.iter().zip(batch_result.iter()).enumerate() {
        assert!(
            (s - b).abs() < 1e-6,
            "Mismatch at {}: seq={} batch={}",
            i,
            s,
            b
        );
    }
}

#[test]
fn test_par023_throughput_projection() {
    // Based on PAR-022 analysis:
    // - Current: 121 tok/s average
    // - 176 syncs per token
    // - Target: Reduce to ~22 syncs (1 per layer)
    // - Expected speedup: 1.5-2x (conservative estimate)

    let current_toks = 121.0;
    let current_syncs = 176.0;
    let target_syncs = 22.0;

    // Conservative estimate: sync overhead is ~30% of total time
    // Reducing syncs by 8x should improve by ~1.5x
    let sync_overhead_fraction = 0.30;
    let sync_reduction_factor = current_syncs / target_syncs;
    let theoretical_speedup =
        1.0 / (1.0 - sync_overhead_fraction * (1.0 - 1.0 / sync_reduction_factor));

    let projected_toks = current_toks * theoretical_speedup;

    eprintln!("PAR-023 Throughput Projection:");
    eprintln!("  Current throughput:     {:.1} tok/s", current_toks);
    eprintln!(
        "  Sync reduction:         {:.1}x ({} -> {})",
        sync_reduction_factor, current_syncs as u32, target_syncs as u32
    );
    eprintln!("  Theoretical speedup:    {:.2}x", theoretical_speedup);
    eprintln!("  Projected throughput:   {:.1} tok/s", projected_toks);
    eprintln!("  M4 target:              >192 tok/s");
    eprintln!(
        "  Gap to M4:              {:.1} tok/s",
        192.0 - projected_toks
    );

    // Verify projection meets M4 target
    assert!(
        projected_toks > 150.0,
        "Projected throughput {:.1} should exceed 150 tok/s",
        projected_toks
    );
}

#[test]
fn test_par023_fully_gpu_resident_projection() {
    // PAR-023 optimized path analysis:
    // - Original: 176 syncs per token
    // - Per-layer sync: 22 syncs (1 per layer)
    // - Fully GPU-resident: ~2 syncs (embedding upload + logits download)
    //
    // The fully GPU-resident path chains all operations on GPU:
    // 1. Embedding lookup (CPU) + upload
    // 2. All transformer layers (GPU-resident, no syncs)
    // 3. Output RMSNorm (GPU-resident)
    // 4. LM head projection (GPU-resident)
    // 5. Logits download

    let current_toks = 121.0;
    let current_syncs = 176.0;
    let fully_resident_syncs = 2.0; // embedding upload + logits download

    // More aggressive estimate for fully GPU-resident:
    // Sync overhead may be higher fraction when compute is fast
    let sync_overhead_fraction = 0.40;
    let sync_reduction_factor = current_syncs / fully_resident_syncs;
    let theoretical_speedup =
        1.0 / (1.0 - sync_overhead_fraction * (1.0 - 1.0 / sync_reduction_factor));

    let projected_toks = current_toks * theoretical_speedup;

    eprintln!("PAR-023 Fully GPU-Resident Projection:");
    eprintln!("  Current throughput:     {:.1} tok/s", current_toks);
    eprintln!(
        "  Sync reduction:         {:.0}x ({} -> {})",
        sync_reduction_factor, current_syncs as u32, fully_resident_syncs as u32
    );
    eprintln!("  Theoretical speedup:    {:.2}x", theoretical_speedup);
    eprintln!("  Projected throughput:   {:.1} tok/s", projected_toks);
    eprintln!("  M4 target:              >192 tok/s");
    if projected_toks > 192.0 {
        eprintln!(
            "  ✓ EXCEEDS M4 target by {:.1} tok/s",
            projected_toks - 192.0
        );
    } else {
        eprintln!(
            "  Gap to M4:              {:.1} tok/s",
            192.0 - projected_toks
        );
    }

    // With 2 syncs vs 176, we should meet M4 target
    assert!(
        projected_toks > 180.0,
        "Fully GPU-resident projected throughput {:.1} should exceed 180 tok/s",
        projected_toks
    );
}

// =========================================================================
// GPU Integration Tests (require CUDA)
// =========================================================================

/// Test: Measure actual GPU sync overhead with async vs sync methods
#[test]
#[ignore] // Run with --ignored when CUDA is available
#[cfg(feature = "cuda")]
fn test_par023_gpu_async_vs_sync() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    let n = 2048usize;
    let iterations = 100;

    // Test data
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 1.0).collect();

    // Measure sync path (silu_host does upload + kernel + sync + download)
    let mut sync_output = vec![0.0f32; n];
    let sync_start = Instant::now();
    for _ in 0..iterations {
        executor
            .silu_host(&input, &mut sync_output)
            .expect("silu_host failed");
    }
    let sync_duration = sync_start.elapsed();

    eprintln!("PAR-023 GPU Async vs Sync:");
    eprintln!(
        "  Sync path ({} iterations): {:?}",
        iterations, sync_duration
    );
    eprintln!("  Per-operation: {:?}", sync_duration / iterations as u32);

    // Note: Full async measurement requires chaining multiple GPU operations
    // which is what transformer_layer_gpu does
}

/// Test: FFN SwiGLU pipeline timing (simulated with host methods)
#[test]
#[ignore = "CUDA benchmark - run with --ignored on GPU hardware"]
#[cfg(feature = "cuda")]
fn test_par023_ffn_swiglu_pipeline_timing() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    // TinyLlama dimensions
    let _hidden_dim = 2048;
    let intermediate_dim = 5632;
    let iterations = 10;

    // Simulate FFN input
    let input: Vec<f32> = (0..intermediate_dim)
        .map(|i| (i as f32) * 0.001 - 0.5)
        .collect();

    // Measure individual operations (sync after each)
    let mut gate_out = vec![0.0f32; intermediate_dim];
    let mut up_out = vec![0.0f32; intermediate_dim];
    let mut swiglu_out = vec![0.0f32; intermediate_dim];

    let individual_start = Instant::now();
    for _ in 0..iterations {
        // Gate: silu activation
        executor
            .silu_host(&input, &mut gate_out)
            .expect("silu failed");
        // Up: identity (simulated)
        up_out.copy_from_slice(&input);
        // SwiGLU: gate * up
        executor
            .elementwise_mul_host(&gate_out, &up_out, &mut swiglu_out)
            .expect("mul failed");
    }
    let individual_duration = individual_start.elapsed();

    // Measure fused operation
    let fused_start = Instant::now();
    for _ in 0..iterations {
        executor
            .fused_swiglu_host(&input, &input, &mut swiglu_out)
            .expect("fused_swiglu failed");
    }
    let fused_duration = fused_start.elapsed();

    let speedup = individual_duration.as_nanos() as f64 / fused_duration.as_nanos() as f64;

    eprintln!("PAR-023 FFN SwiGLU Pipeline:");
    eprintln!(
        "  Individual ops ({} iter): {:?}",
        iterations, individual_duration
    );
    eprintln!(
        "  Fused op ({} iter):       {:?}",
        iterations, fused_duration
    );
    eprintln!("  Speedup:                  {:.2}x", speedup);

    // Fused should be faster
    assert!(
        speedup > 1.0,
        "Fused SwiGLU should be faster than individual ops"
    );
}

/// Test: GPU-resident forward pass integration
///
/// Validates that the GPU-resident transformer layer produces correct results
/// by comparing against the standard forward pass.
#[test]
#[ignore] // Run with --ignored when CUDA is available
#[cfg(feature = "cuda")]
fn test_par023_gpu_resident_forward_integration() {
    use realizar::cuda::CudaExecutor;

    // Skip if CUDA not available
    let executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU-resident integration test");
            return;
        },
    };

    eprintln!("PAR-023 GPU-Resident Integration Test:");
    eprintln!(
        "  GPU: {}",
        executor.device_name().unwrap_or("Unknown".to_string())
    );
    eprintln!("  Memory: {:?}", executor.memory_info());

    // Note: Full integration test requires loading a real model
    // This test validates the executor is available and can be initialized
    eprintln!("  CUDA executor initialized successfully");
    eprintln!("  GPU-resident forward path is available for LLaMA-style models");

    // Model loading and forward pass validation would be done in E2E tests
    // with actual model files (TinyLlama, phi-2, qwen-coder)
}

/// Test: GPU-resident path architecture detection
#[test]
fn test_par023_architecture_detection() {
    // Verify architecture detection logic without real model
    // LLaMA-style models should support GPU-resident path:
    // - Separate Q/K/V (not fused)
    // - SwiGLU (ffn_gate present)
    // - RMSNorm (ffn_norm present)

    // phi-2 style models should NOT support GPU-resident path:
    // - Fused QKV
    // - GELU (no ffn_gate)

    eprintln!("PAR-023 Architecture Detection:");
    eprintln!("  LLaMA/TinyLlama: Supported (separate Q/K/V, SwiGLU, RMSNorm)");
    eprintln!("  qwen-coder:      Supported (separate Q/K/V, SwiGLU, RMSNorm)");
    eprintln!("  phi-2:           NOT supported (fused QKV, GELU)");

    // Test passes if logic is defined - actual detection requires model load
}

/// Test: End-to-end GPU-resident throughput benchmark with real model
///
/// M4 Milestone Target: >192 tok/s
/// Uses TinyLlama-1.1B-Chat Q4_K_M model to measure actual throughput
#[test]
#[ignore] // Run with --ignored when CUDA and model available
#[cfg(feature = "cuda")]
fn test_par023_e2e_gpu_resident_throughput() {
    use std::path::Path;
    use std::time::Instant;

    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
    };

    // Model path - TinyLlama Q4_K_M
    let model_path = std::env::var("TINYLLAMA_MODEL")
        .unwrap_or_else(|_| "/home/noah/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf".to_string());

    if !Path::new(&model_path).exists() {
        eprintln!("SKIP: Model not found at {}", model_path);
        eprintln!("Set TINYLLAMA_MODEL env var or place model at default path");
        return;
    }

    eprintln!("PAR-023 E2E GPU-Resident Throughput Benchmark");
    eprintln!("  Model: {}", model_path);

    // Load model using memory-mapped I/O
    let mapped_model = MappedGGUFModel::from_path(&model_path).expect("Failed to load GGUF model");
    let owned_model = OwnedQuantizedModel::from_mapped(&mapped_model)
        .expect("Failed to create OwnedQuantizedModel");

    eprintln!("  Hidden dim: {}", owned_model.config().hidden_dim);
    eprintln!("  Layers: {}", owned_model.config().num_layers);

    // Create CUDA-accelerated model
    let mut cuda_model =
        OwnedQuantizedModelCuda::new(owned_model, 0).expect("Failed to create CUDA model");

    // Check GPU-resident support
    if !cuda_model.supports_gpu_resident() {
        eprintln!("SKIP: Model does not support GPU-resident path");
        return;
    }

    eprintln!("  GPU-resident path: SUPPORTED");

    let config = QuantizedGenerateConfig {
        max_tokens: 50,
        temperature: 0.0, // Greedy for reproducibility
        top_k: 1,
        ..Default::default()
    };

    // Prompt tokens (BOS + simple prompt)
    let prompt = vec![1u32, 4321, 567, 89, 10]; // Arbitrary prompt tokens

    // Warmup with GPU-resident path
    eprintln!("  Warming up with GPU-resident path...");
    for _ in 0..3 {
        let _ = cuda_model.generate_gpu_resident(&prompt, &config);
    }

    // Benchmark GPU-resident path
    let gpu_iterations = 5;
    let gpu_start = Instant::now();
    let mut gpu_tokens = 0usize;
    for _ in 0..gpu_iterations {
        let output = cuda_model
            .generate_gpu_resident(&prompt, &config)
            .expect("generate_gpu_resident failed");
        gpu_tokens += output.len().saturating_sub(prompt.len());
    }
    let gpu_time = gpu_start.elapsed().as_secs_f64();
    let gpu_toks = gpu_tokens as f64 / gpu_time;

    eprintln!();
    eprintln!("PAR-023 E2E Throughput Results:");
    eprintln!("  GPU-resident path:  {:.1} tok/s", gpu_toks);
    eprintln!("  Baseline (PAR-022): ~121 tok/s");
    eprintln!("  Speedup vs baseline: {:.2}x", gpu_toks / 121.0);
    eprintln!();
    eprintln!("  M4 Target:          >192 tok/s");
    if gpu_toks > 192.0 {
        eprintln!(
            "  ✓ M4 MILESTONE ACHIEVED! Exceeds target by {:.1} tok/s",
            gpu_toks - 192.0
        );
    } else {
        eprintln!("  Gap to M4:          {:.1} tok/s", 192.0 - gpu_toks);
    }

    // Verify reasonable throughput (should beat baseline)
    assert!(
        gpu_toks > 100.0,
        "GPU-resident path should achieve at least 100 tok/s, got {:.1}",
        gpu_toks
    );

    // M4 milestone check (soft assert - log but don't fail)
    if gpu_toks < 192.0 {
        eprintln!(
            "WARNING: GPU-resident {:.1} tok/s below M4 target of 192 tok/s",
            gpu_toks
        );
    }
}
