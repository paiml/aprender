//! External Server Benchmark Matrix
//!
//! Benchmarks realizar native inference vs external servers (llama.cpp, Ollama)
//! using the BenchmarkMatrix infrastructure with CV-based stopping.
//!
//! ## Running
//!
//! This benchmark requires external servers to be running:
//! - llama.cpp: `llama-server -m model.gguf --host 127.0.0.1 --port 8082`
//! - Ollama: `ollama serve` (default port 11434)
//!
//! Then run: `cargo bench --bench external_matrix --features bench-http`
//!
//! ## Methodology
//!
//! Per Hoefler & Belli SC'15:
//! - Dynamic CV-based stopping (target CV < 0.10)
//! - Minimum 5 samples, maximum 30 samples
//! - Warmup iterations excluded from measurements
//! - Reports p50, p99 latencies and throughput

#![allow(dead_code)]

use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "bench-http")]
use criterion::BenchmarkId;

#[cfg(feature = "bench-http")]
use realizar::http_client::{
    HttpBenchmarkConfig, HttpBenchmarkResult, HttpBenchmarkRunner, ModelHttpClient,
};

#[cfg(feature = "bench-http")]
use realizar::bench_preflight::CvStoppingCriterion;

#[cfg(feature = "bench-http")]
use realizar::bench::{
    BenchmarkMatrix, ComputeBackendType, HardwareSpec, MatrixBenchmarkEntry, RuntimeType,
};

use realizar::gguf::{ArchConstraints, GGUFConfig, GGUFTransformer, GGUFTransformerLayer};

// ============================================================================
// Configuration
// ============================================================================

/// Fixed prompt for reproducible benchmarks
const BENCHMARK_PROMPT: &str = "Explain the concept of machine learning in one sentence.";

/// Maximum tokens to generate
const MAX_TOKENS: usize = 50;

/// Temperature for sampling
const TEMPERATURE: f32 = 0.7;

/// llama.cpp server URL (GPU mode with ngl=99)
const LLAMACPP_GPU_URL: &str = "http://localhost:8082";

/// llama.cpp server URL (CPU mode with ngl=0)
const LLAMACPP_CPU_URL: &str = "http://localhost:8083";

/// Ollama server URL
const OLLAMA_URL: &str = "http://localhost:11434";

/// Model name for Ollama
const OLLAMA_MODEL: &str = "phi2:2.7b";

// ============================================================================
// Realizar Native Benchmark (CPU)
// ============================================================================

/// Create a test GGUF transformer for realizar native benchmark
fn create_benchmark_transformer() -> GGUFTransformer {
    // Use a small model configuration that's fast but representative
    let hidden_dim = 256;
    let num_layers = 4;
    let vocab_size = 1000;
    let intermediate_dim = 512;

    let config = GGUFConfig {
        architecture: "benchmark_model".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layers: Vec<GGUFTransformerLayer> = (0..num_layers)
        .map(|_| GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        })
        .collect();

    GGUFTransformer {
        config,
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        position_embedding: None,
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
    }
}

/// Benchmark realizar native CPU inference with CV-based stopping
/// FIXME: GGUFTransformer no longer has forward() - use OwnedQuantizedModel
#[allow(dead_code)]
fn benchmark_realizar_native_cv() -> Vec<f64> {
    let _transformer = create_benchmark_transformer();
    let _tokens: &[u32] = &[1, 2, 3, 4]; // Fixed input for reproducibility

    let _min_samples = 5;
    let _max_samples = 30;
    let _cv_threshold = 0.10;

    // NOTE: Uses placeholder since GGUFTransformer is a weight container.
    // See OwnedQuantizedModel for the inference API.

    Vec::new()
}

/// Calculate coefficient of variation
fn calculate_cv(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return f64::MAX;
    }

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;

    if mean.abs() < f64::EPSILON {
        return f64::MAX;
    }

    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    std_dev / mean
}

/// Calculate percentile from sorted samples
fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ============================================================================
// Criterion Benchmark Functions
// ============================================================================

/// Benchmark realizar native inference (always available)
/// NOTE: GGUFTransformer is a weight container; use OwnedQuantizedModel for inference
fn benchmark_realizar_native(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_matrix_realizar");

    let _transformer = create_benchmark_transformer();
    let _tokens: &[u32] = &[1, 2, 3, 4];

    // NOTE: Uses placeholder since GGUFTransformer is a weight container.
    // See OwnedQuantizedModel for the inference API.
    group.bench_function("cpu_forward_placeholder", |b| {
        b.iter(|| criterion::black_box(42i32));
    });

    group.finish();
}

/// Benchmark llama.cpp via HTTP (requires running server)
#[cfg(feature = "bench-http")]
fn benchmark_llamacpp_http(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_matrix_llamacpp");
    group.sample_size(10); // Use smaller sample size for HTTP benchmarks

    // Check if llama.cpp GPU server is available
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("Failed to create client");

    let gpu_available = client
        .get(format!("{}/health", LLAMACPP_GPU_URL))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if gpu_available {
        group.bench_function(BenchmarkId::new("gpu", "forward"), |b| {
            let mut runner = HttpBenchmarkRunner::new(HttpBenchmarkConfig {
                cv_criterion: CvStoppingCriterion::new(3, 5, 0.20),
                warmup_iterations: 1,
                prompt: BENCHMARK_PROMPT.to_string(),
                max_tokens: MAX_TOKENS,
                temperature: TEMPERATURE,
                ..Default::default()
            });

            b.iter(|| {
                // Single iteration measurement
                let result = runner.benchmark_llamacpp(LLAMACPP_GPU_URL);
                criterion::black_box(result)
            });
        });
    }

    // Check if llama.cpp CPU server is available
    let cpu_available = client
        .get(format!("{}/health", LLAMACPP_CPU_URL))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if cpu_available {
        group.bench_function(BenchmarkId::new("cpu", "forward"), |b| {
            let mut runner = HttpBenchmarkRunner::new(HttpBenchmarkConfig {
                cv_criterion: CvStoppingCriterion::new(3, 5, 0.20),
                warmup_iterations: 1,
                prompt: BENCHMARK_PROMPT.to_string(),
                max_tokens: MAX_TOKENS,
                temperature: TEMPERATURE,
                ..Default::default()
            });

            b.iter(|| {
                let result = runner.benchmark_llamacpp(LLAMACPP_CPU_URL);
                criterion::black_box(result)
            });
        });
    }

    if !gpu_available && !cpu_available {
        eprintln!("WARNING: No llama.cpp servers available. Skipping HTTP benchmarks.");
        eprintln!(
            "  Start GPU server: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99"
        );
        eprintln!(
            "  Start CPU server: llama-server -m model.gguf --host 127.0.0.1 --port 8083 -ngl 0"
        );
    }

    group.finish();
}

/// Benchmark Ollama via HTTP (requires running server)
#[cfg(feature = "bench-http")]
fn benchmark_ollama_http(c: &mut Criterion) {
    let mut group = c.benchmark_group("external_matrix_ollama");
    group.sample_size(10);

    // Check if Ollama is available
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("Failed to create client");

    let ollama_available = client
        .get(format!("{}/api/tags", OLLAMA_URL))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if ollama_available {
        group.bench_function("gpu_forward", |b| {
            let mut runner = HttpBenchmarkRunner::new(HttpBenchmarkConfig {
                cv_criterion: CvStoppingCriterion::new(3, 5, 0.20),
                warmup_iterations: 1,
                prompt: BENCHMARK_PROMPT.to_string(),
                max_tokens: MAX_TOKENS,
                temperature: TEMPERATURE,
                ..Default::default()
            });

            b.iter(|| {
                let result = runner.benchmark_ollama(OLLAMA_URL, OLLAMA_MODEL);
                criterion::black_box(result)
            });
        });
    } else {
        eprintln!("WARNING: Ollama server not available. Skipping.");
        eprintln!("  Start Ollama: ollama serve");
    }

    group.finish();
}

/// Generate complete benchmark matrix (manual execution)
///
/// This function runs all benchmarks and produces a BenchmarkMatrix result.
/// Run with: `cargo test --bench external_matrix --features bench-http -- --nocapture generate_matrix`
#[cfg(feature = "bench-http")]
#[allow(dead_code)]
fn generate_benchmark_matrix() -> BenchmarkMatrix {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║          External Server Benchmark Matrix                       ║");
    println!("║                                                                 ║");
    println!("║  Methodology: CV-based stopping (Hoefler & Belli SC'15)         ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let hardware = HardwareSpec {
        cpu: "Benchmark CPU".to_string(),
        gpu: Some("Benchmark GPU".to_string()),
        memory_gb: 32,
        storage: "SSD".to_string(),
    };

    let mut matrix = BenchmarkMatrix::new("phi-2-q4_k_m", hardware);

    // 1. Benchmark realizar native (CPU)
    println!("=== Realizar Native (CPU) ===");
    let latencies = benchmark_realizar_native_cv();
    let entry = MatrixBenchmarkEntry::from_samples(
        RuntimeType::Realizar,
        ComputeBackendType::Cpu,
        "benchmark_model",
        &latencies,
        &[], // No throughput for test model
        latencies.first().copied().unwrap_or(0.0),
    );
    println!(
        "  p50: {:.3}ms | p99: {:.3}ms | samples: {} | cv: {:.4}",
        entry.p50_latency_ms, entry.p99_latency_ms, entry.samples, entry.cv_at_stop
    );
    matrix.add_entry(entry);

    // 2. Benchmark llama.cpp (GPU)
    let client = ModelHttpClient::with_timeout(5);
    if client
        .health_check_openai(LLAMACPP_GPU_URL)
        .unwrap_or(false)
    {
        println!("\n=== llama.cpp (GPU) ===");
        let mut runner = HttpBenchmarkRunner::new(HttpBenchmarkConfig {
            cv_criterion: CvStoppingCriterion::new(5, 30, 0.10),
            warmup_iterations: 2,
            prompt: BENCHMARK_PROMPT.to_string(),
            max_tokens: MAX_TOKENS,
            temperature: TEMPERATURE,
            ..Default::default()
        });

        match runner.benchmark_llamacpp(LLAMACPP_GPU_URL) {
            Ok(result) => {
                let entry = http_result_to_entry(
                    RuntimeType::LlamaCpp,
                    ComputeBackendType::Cuda,
                    "phi-2-q4_k_m",
                    &result,
                );
                println!(
                    "  p50: {:.1}ms | p99: {:.1}ms | tps: {:.1} | samples: {} | cv: {:.4}",
                    entry.p50_latency_ms,
                    entry.p99_latency_ms,
                    entry.throughput_tps,
                    entry.samples,
                    entry.cv_at_stop
                );
                matrix.add_entry(entry);
            },
            Err(e) => {
                eprintln!("  Error: {}", e);
                matrix.add_entry(MatrixBenchmarkEntry::unavailable(
                    RuntimeType::LlamaCpp,
                    ComputeBackendType::Cuda,
                ));
            },
        }
    } else {
        println!("\n=== llama.cpp (GPU) - NOT AVAILABLE ===");
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cuda,
        ));
    }

    // 3. Benchmark llama.cpp (CPU)
    if client
        .health_check_openai(LLAMACPP_CPU_URL)
        .unwrap_or(false)
    {
        println!("\n=== llama.cpp (CPU) ===");
        let mut runner = HttpBenchmarkRunner::new(HttpBenchmarkConfig {
            cv_criterion: CvStoppingCriterion::new(5, 30, 0.10),
            warmup_iterations: 2,
            prompt: BENCHMARK_PROMPT.to_string(),
            max_tokens: MAX_TOKENS,
            temperature: TEMPERATURE,
            ..Default::default()
        });

        match runner.benchmark_llamacpp(LLAMACPP_CPU_URL) {
            Ok(result) => {
                let entry = http_result_to_entry(
                    RuntimeType::LlamaCpp,
                    ComputeBackendType::Cpu,
                    "phi-2-q4_k_m",
                    &result,
                );
                println!(
                    "  p50: {:.1}ms | p99: {:.1}ms | tps: {:.1} | samples: {} | cv: {:.4}",
                    entry.p50_latency_ms,
                    entry.p99_latency_ms,
                    entry.throughput_tps,
                    entry.samples,
                    entry.cv_at_stop
                );
                matrix.add_entry(entry);
            },
            Err(e) => {
                eprintln!("  Error: {}", e);
                matrix.add_entry(MatrixBenchmarkEntry::unavailable(
                    RuntimeType::LlamaCpp,
                    ComputeBackendType::Cpu,
                ));
            },
        }
    } else {
        println!("\n=== llama.cpp (CPU) - NOT AVAILABLE ===");
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
        ));
    }

    // 4. Benchmark Ollama (GPU)
    if client.health_check_ollama(OLLAMA_URL).unwrap_or(false) {
        println!("\n=== Ollama (GPU) ===");
        let mut runner = HttpBenchmarkRunner::new(HttpBenchmarkConfig {
            cv_criterion: CvStoppingCriterion::new(5, 30, 0.10),
            warmup_iterations: 2,
            prompt: BENCHMARK_PROMPT.to_string(),
            max_tokens: MAX_TOKENS,
            temperature: TEMPERATURE,
            ..Default::default()
        });

        match runner.benchmark_ollama(OLLAMA_URL, OLLAMA_MODEL) {
            Ok(result) => {
                let entry = http_result_to_entry(
                    RuntimeType::Ollama,
                    ComputeBackendType::Cuda,
                    OLLAMA_MODEL,
                    &result,
                );
                println!(
                    "  p50: {:.1}ms | p99: {:.1}ms | tps: {:.1} | samples: {} | cv: {:.4}",
                    entry.p50_latency_ms,
                    entry.p99_latency_ms,
                    entry.throughput_tps,
                    entry.samples,
                    entry.cv_at_stop
                );
                matrix.add_entry(entry);
            },
            Err(e) => {
                eprintln!("  Error: {}", e);
                matrix.add_entry(MatrixBenchmarkEntry::unavailable(
                    RuntimeType::Ollama,
                    ComputeBackendType::Cuda,
                ));
            },
        }
    } else {
        println!("\n=== Ollama (GPU) - NOT AVAILABLE ===");
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Ollama,
            ComputeBackendType::Cuda,
        ));
    }

    // Output markdown table
    println!("\n{}", "=".repeat(70));
    println!("BENCHMARK MATRIX RESULTS");
    println!("{}", "=".repeat(70));
    println!("\n{}", matrix.to_markdown_table());

    matrix
}

/// Convert HTTP benchmark result to matrix entry
#[cfg(feature = "bench-http")]
fn http_result_to_entry(
    runtime: RuntimeType,
    backend: ComputeBackendType,
    model: &str,
    result: &HttpBenchmarkResult,
) -> MatrixBenchmarkEntry {
    MatrixBenchmarkEntry {
        runtime,
        backend,
        model: model.to_string(),
        available: true,
        p50_latency_ms: result.p50_latency_ms,
        p99_latency_ms: result.p99_latency_ms,
        throughput_tps: result.throughput_tps,
        cold_start_ms: result.cold_start_ms,
        samples: result.sample_count,
        cv_at_stop: result.cv_at_stop,
        notes: if result.cv_converged {
            "CV converged".to_string()
        } else {
            format!("CV did not converge (max samples: {})", result.sample_count)
        },
    }
}

// ============================================================================
// Criterion Configuration
// ============================================================================

#[cfg(feature = "bench-http")]
criterion_group!(
    benches,
    benchmark_realizar_native,
    benchmark_llamacpp_http,
    benchmark_ollama_http,
);

#[cfg(not(feature = "bench-http"))]
criterion_group!(benches, benchmark_realizar_native,);

criterion_main!(benches);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use super::*;

    #[test]
    fn test_create_benchmark_transformer() {
        let transformer = create_benchmark_transformer();
        assert_eq!(transformer.config.hidden_dim, 256);
        assert_eq!(transformer.config.num_layers, 4);
        assert_eq!(transformer.config.vocab_size, 1000);
    }

    #[test]
    fn test_realizar_native_forward() {
        let transformer = create_benchmark_transformer();
        let tokens: &[u32] = &[1, 2, 3, 4];

        let logits = transformer.forward(tokens).expect("forward should succeed");
        assert_eq!(logits.len(), transformer.config.vocab_size);
    }

    #[test]
    fn test_calculate_cv_identical() {
        let samples = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let cv = calculate_cv(&samples);
        assert!(cv < 0.001, "CV of identical values should be ~0");
    }

    #[test]
    fn test_calculate_cv_varied() {
        let samples = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let cv = calculate_cv(&samples);
        assert!(
            cv > 0.05 && cv < 0.15,
            "CV should be reasonable, got {}",
            cv
        );
    }

    #[test]
    fn test_percentile() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&samples, 50.0) - 5.0).abs() < 0.5);
        assert!((percentile(&samples, 99.0) - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_benchmark_realizar_native_cv() {
        let latencies = benchmark_realizar_native_cv();
        assert!(!latencies.is_empty());
        assert!(latencies.len() >= 5); // At least min_samples
        assert!(latencies.len() <= 30); // At most max_samples
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_http_result_to_entry() {
        use realizar::bench_preflight::QualityMetrics;

        let result = HttpBenchmarkResult {
            latency_samples: vec![100.0, 110.0, 90.0],
            latency_samples_filtered: vec![100.0, 110.0, 90.0],
            mean_latency_ms: 100.0,
            p50_latency_ms: 100.0,
            p99_latency_ms: 110.0,
            std_dev_ms: 10.0,
            cv_at_stop: 0.10,
            throughput_tps: 50.0,
            cold_start_ms: 150.0,
            sample_count: 3,
            filtered_sample_count: 3,
            cv_converged: true,
            quality_metrics: QualityMetrics::default(),
        };

        let entry = http_result_to_entry(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cuda,
            "test-model",
            &result,
        );

        assert_eq!(entry.runtime, RuntimeType::LlamaCpp);
        assert_eq!(entry.backend, ComputeBackendType::Cuda);
        assert!(entry.available);
        assert_eq!(entry.p50_latency_ms, 100.0);
        assert_eq!(entry.throughput_tps, 50.0);
    }

    /// Integration test that generates the full matrix
    /// Run with: `cargo test --bench external_matrix --features bench-http -- --nocapture --ignored`
    #[test]
    #[ignore = "Requires external servers"]
    #[cfg(feature = "bench-http")]
    fn test_generate_benchmark_matrix() {
        let matrix = generate_benchmark_matrix();

        // Should have at least realizar native entry
        assert!(!matrix.entries.is_empty());

        // Check realizar native is present
        let realizar_cpu = matrix.get_entry(RuntimeType::Realizar, ComputeBackendType::Cpu);
        assert!(realizar_cpu.is_some());

        println!("\nJSON Output:");
        println!(
            "{}",
            serde_json::to_string_pretty(&matrix).expect("JSON serialization")
        );
    }
}
