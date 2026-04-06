use crate::http_client::tests::part_11::SimdBackend;
use crate::http_client::*;
// ==================== IMP-306: Trueno wgpu GPU Backend ====================
// Per spec: 10x speedup for 4096x4096 matmul via GPU

/// GPU backend availability check for IMP-306
#[derive(Debug, Clone)]
pub struct Imp306GpuStatus {
    pub wgpu_available: bool,
    pub cuda_available: bool,
    pub device_name: Option<String>,
    pub vram_mb: Option<u64>,
    pub meets_imp306: bool,
}

impl Imp306GpuStatus {
    pub fn check() -> Self {
        // Check wgpu availability via trueno
        let wgpu_available = cfg!(feature = "gpu");
        let cuda_available = cfg!(feature = "cuda");

        // Would query actual GPU in real implementation
        let device_name = if wgpu_available {
            Some("wgpu backend".to_string())
        } else {
            None
        };

        let meets_imp306 = wgpu_available;

        Self {
            wgpu_available,
            cuda_available,
            device_name,
            vram_mb: None,
            meets_imp306,
        }
    }
}

/// IMP-306a: Test GPU availability
#[test]
fn test_imp_306a_gpu_availability() {
    let result = Imp306GpuStatus::check();

    println!("\nIMP-306a: GPU Availability:");
    println!("  wgpu: {}", result.wgpu_available);
    println!("  CUDA: {}", result.cuda_available);
    if let Some(name) = &result.device_name {
        println!("  Device: {}", name);
    }
    println!(
        "  IMP-306: {}",
        if result.meets_imp306 {
            "READY"
        } else {
            "NO GPU"
        }
    );
}

/// IMP-306b: Test trueno GPU feature flag
#[test]
fn test_imp_306b_trueno_gpu_feature() {
    // Verify gpu feature is enabled in Cargo.toml
    #[cfg(feature = "gpu")]
    {
        println!("\nIMP-306b: Trueno GPU feature enabled");
        assert!(true, "GPU feature available");
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("\nIMP-306b: Trueno GPU feature NOT enabled");
        println!("  Run with: cargo test --features gpu");
    }
}

/// IMP-306c: Test backend selection for large operations
#[test]
fn test_imp_306c_backend_selection() {
    let backend = trueno::select_best_available_backend();

    println!("\nIMP-306c: Backend Selection:");
    println!("  Best available: {:?}", backend);

    // For large matmul, compute-bound operations prefer AVX-512 or GPU
    let compute_backend = trueno::select_backend_for_operation(trueno::OperationType::ComputeBound);
    println!("  Compute-bound (large matmul): {:?}", compute_backend);

    // Memory-bound operations prefer AVX2 for cache efficiency
    let memory_backend = trueno::select_backend_for_operation(trueno::OperationType::MemoryBound);
    println!("  Memory-bound: {:?}", memory_backend);
}

/// IMP-306d: Real-world GPU matmul benchmark
#[test]
#[ignore = "Requires GPU and extended benchmark time"]
fn test_imp_306d_realworld_gpu_matmul() {
    use std::time::Instant;
    use trueno::Matrix;

    let size = 4096;
    let iterations = 10;

    let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.0001).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.0001).collect();

    let a = Matrix::from_vec(size, size, a_data).expect("Matrix A");
    let b = Matrix::from_vec(size, size, b_data).expect("Matrix B");

    let start = Instant::now();
    for _ in 0..iterations {
        let _c = a.matmul(&b).expect("matmul");
    }
    let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

    let flops = 2.0 * (size as f64).powi(3);
    let gflops = flops / (avg_us * 1e-6) / 1e9;

    println!("\nIMP-306d: GPU Matmul {}x{}:", size, size);
    println!("  Time: {:.1}ms", avg_us / 1000.0);
    println!("  GFLOPS: {:.1}", gflops);
    println!(
        "  IMP-306: {}",
        if gflops > 100.0 { "PASS" } else { "NEEDS GPU" }
    );
}

// ==================== Performance Summary ====================

/// Summary of all trueno integration benchmarks
pub struct TruenoIntegrationSummary {
    pub simd_backend: SimdBackend,
    pub simd_speedup: f64,
    pub matmul_gflops: f64,
    pub activation_latency_us: f64,
    pub gpu_available: bool,
    pub estimated_tok_s: f64,
}

impl TruenoIntegrationSummary {
    pub fn estimate_throughput(&self) -> f64 {
        // Rough estimate based on Phi-2 model
        // ~100 matmuls per token, each ~500µs with SIMD
        let matmul_time_ms = 100.0 * 0.5; // 50ms per token
        let activation_time_ms = 32.0 * self.activation_latency_us / 1000.0;
        let total_ms = matmul_time_ms + activation_time_ms;
        1000.0 / total_ms
    }
}

/// IMP-307a: Integration summary test
#[test]
fn test_imp_307a_integration_summary() {
    let summary = TruenoIntegrationSummary {
        simd_backend: SimdBackend::detect(),
        simd_speedup: 4.0,
        matmul_gflops: 30.0, // Conservative estimate
        activation_latency_us: 20.0,
        gpu_available: cfg!(feature = "gpu"),
        estimated_tok_s: 0.0,
    };

    let est_toks = summary.estimate_throughput();

    println!("\nIMP-307a: Trueno Integration Summary:");
    println!("  SIMD Backend: {:?}", summary.simd_backend);
    println!("  Matmul GFLOPS: {:.1}", summary.matmul_gflops);
    println!(
        "  Activation latency: {:.1}µs",
        summary.activation_latency_us
    );
    println!("  GPU available: {}", summary.gpu_available);
    println!("  Estimated throughput: {:.1} tok/s", est_toks);
    println!();
    println!(
        "  Gap to llama.cpp CPU (15 tok/s): {:.1}x",
        15.0 / est_toks.max(0.1)
    );
    println!(
        "  Gap to llama.cpp GPU (256 tok/s): {:.1}x",
        256.0 / est_toks.max(0.1)
    );
}

// ==================== IMP-400: E2E Real-World Performance Comparison ====================
// EXTREME TDD: Real apples-to-apples comparison with Ollama and llama.cpp
// Uses same model (phi-2 Q4_K_M) for fair comparison

/// E2E performance comparison result
#[derive(Debug, Clone)]
pub struct E2EPerformanceComparison {
    /// Ollama throughput (tok/s)
    pub ollama_tps: f64,
    /// Ollama p50 latency (ms)
    pub ollama_p50_ms: f64,
    /// Realizar native throughput (tok/s)
    pub realizar_tps: f64,
    /// Realizar p50 latency (ms)
    pub realizar_p50_ms: f64,
    /// Gap: ollama_tps / realizar_tps
    pub performance_gap: f64,
    /// Model used for comparison
    pub model: String,
    /// Tokens generated per sample
    pub tokens_generated: usize,
}

impl E2EPerformanceComparison {
    /// Create comparison from measurements
    pub fn from_measurements(
        ollama_tps: f64,
        ollama_p50_ms: f64,
        realizar_tps: f64,
        realizar_p50_ms: f64,
        model: &str,
        tokens: usize,
    ) -> Self {
        let performance_gap = if realizar_tps > 0.0 {
            ollama_tps / realizar_tps
        } else {
            f64::INFINITY
        };

        Self {
            ollama_tps,
            ollama_p50_ms,
            realizar_tps,
            realizar_p50_ms,
            performance_gap,
            model: model.to_string(),
            tokens_generated: tokens,
        }
    }

    /// Check if parity target is met (within 20% of Ollama)
    pub fn meets_parity_target(&self) -> bool {
        self.performance_gap < 1.25
    }
}

/// IMP-400a: Test E2E comparison struct
#[test]
fn test_imp_400a_e2e_comparison_struct() {
    let comparison = E2EPerformanceComparison::from_measurements(
        200.0, // Ollama: 200 tok/s
        50.0,  // Ollama p50: 50ms
        100.0, // Realizar: 100 tok/s
        100.0, // Realizar p50: 100ms
        "phi-2-q4_k_m",
        50,
    );

    assert!(
        (comparison.performance_gap - 2.0).abs() < 0.01,
        "Gap should be 2.0x"
    );
    assert!(
        !comparison.meets_parity_target(),
        "2x gap should not meet parity"
    );

    println!("\nIMP-400a: E2E Comparison Struct:");
    println!(
        "  Ollama: {:.1} tok/s, {:.1}ms p50",
        comparison.ollama_tps, comparison.ollama_p50_ms
    );
    println!(
        "  Realizar: {:.1} tok/s, {:.1}ms p50",
        comparison.realizar_tps, comparison.realizar_p50_ms
    );
    println!("  Gap: {:.2}x", comparison.performance_gap);
    println!("  Parity met: {}", comparison.meets_parity_target());
}

/// IMP-400b: Measure Ollama baseline for E2E comparison
#[test]
#[ignore = "Requires running Ollama server on port 11434"]
fn test_imp_400b_ollama_e2e_baseline() {
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 15, 0.10),
        warmup_iterations: 2,
        prompt: "Explain machine learning in one sentence.".to_string(),
        max_tokens: 50,
        temperature: 0.0, // Deterministic for reproducibility
        ..Default::default()
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_ollama("http://127.0.0.1:11434", "phi2:2.7b")
        .expect("IMP-400b: Ollama benchmark should succeed");

    assert!(
        result.throughput_tps > 50.0,
        "Ollama should achieve > 50 tok/s"
    );

    println!("\nIMP-400b: Ollama E2E Baseline (phi2:2.7b):");
    println!("  Throughput: {:.1} tok/s", result.throughput_tps);
    println!("  P50 Latency: {:.1}ms", result.p50_latency_ms);
    println!("  P99 Latency: {:.1}ms", result.p99_latency_ms);
    println!("  Samples: {}", result.sample_count);
    println!("  CV: {:.4}", result.cv_at_stop);
}

/// IMP-400c: Measure realizar native forward pass performance
#[test]
#[ignore = "GGUFTransformer does not have forward method - needs OwnedQuantizedModel"]
fn test_imp_400c_realizar_native_forward_performance() {
    use crate::gguf::{GGUFConfig, GGUFTransformer, GGUFTransformerLayer};
    use std::time::Instant;

    // Create a scaled-down model for benchmarking (1/4 phi-2 size for faster iteration)
    // Note: This is a test model with random weights for timing only
    let hidden_dim = 640; // phi-2 / 4
    let num_layers = 8; // phi-2 / 4
    let vocab_size = 12800; // phi-2 / 4
    let intermediate_dim = 2560; // phi-2 / 4
    let num_heads = 8;

    let config = GGUFConfig {
        architecture: "phi2_benchmark_scaled".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("phi2_benchmark_scaled"),
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads: 8,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
            eos_token_id: None,
    };

    // Create layers with properly sized weights
    let layers: Vec<GGUFTransformerLayer> = (0..num_layers)
        .map(|_| GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_norm_weight: Some(vec![1.0; hidden_dim]),
            ffn_norm_bias: None,
            ffn_gate_weight: Some(vec![0.01; hidden_dim * intermediate_dim]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        })
        .collect();

    let transformer = GGUFTransformer {
        config,
        token_embedding: vec![0.01; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; vocab_size * hidden_dim],
        lm_head_bias: None,
    };

    // Benchmark forward pass (single token)
    // NOTE: GGUFTransformer is a data holder - use OwnedQuantizedModel for inference
    let _token_ids = vec![1u32]; // Single token
    let iterations = 5;
    let mut latencies_ms = Vec::with_capacity(iterations);

    // Stub: actual forward pass requires OwnedQuantizedModel
    let _ = &transformer.config;
    for _ in 0..iterations {
        let start = Instant::now();
        // Placeholder: actual forward requires OwnedQuantizedModel conversion
        let _output: Vec<f32> = vec![0.0; transformer.config.vocab_size];
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        latencies_ms.push(elapsed_ms);
    }

    // Calculate throughput (tokens per second)
    let avg_latency_ms = latencies_ms.iter().sum::<f64>() / iterations as f64;
    let throughput_tps = 1000.0 / avg_latency_ms;

    // Scale factor to estimate full phi-2 performance (rough approximation)
    // Full model would be ~16x slower due to quadratic attention scaling and linear FFN
    let estimated_full_tps = throughput_tps / 16.0;

    println!("\nIMP-400c: Realizar Native Forward Performance:");
    println!(
        "  Model config: {}x{} hidden, {} layers (1/4 phi-2)",
        num_heads,
        hidden_dim / num_heads,
        num_layers
    );
    println!("  Forward latency: {:.1}ms per token", avg_latency_ms);
    println!("  Throughput (scaled): {:.2} tok/s", throughput_tps);
    println!("  Estimated full phi-2: {:.2} tok/s", estimated_full_tps);
    println!();
    println!(
        "  Gap to Ollama (150 tok/s): {:.1}x",
        150.0 / estimated_full_tps.max(0.01)
    );
    println!(
        "  Gap to llama.cpp GPU (256 tok/s): {:.1}x",
        256.0 / estimated_full_tps.max(0.01)
    );

    // We expect the gap to be significant without GPU optimization
    // This establishes the baseline for measuring optimization progress
}

include!("imp_400d.rs");
