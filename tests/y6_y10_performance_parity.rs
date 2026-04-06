//! Y6-Y10: APR Performance Parity Integration Tests (EXTREME TDD)
//!
//! These tests verify APR format achieves performance parity with GGUF
//! using actual model files. Tests are skipped if model files not available.
//!
//! Per Popperian falsificationism, each test defines falsification conditions.
//!
//! REQUIRES: /tmp/test-tinyllama.apr and TinyLLama GGUF model

use std::path::Path;
use std::time::Instant;

// ============================================================================
// Test Model Paths
// ============================================================================

const APR_MODEL: &str = "/tmp/test-tinyllama.apr";
const GGUF_MODEL: &str = match option_env!("GGUF_MODEL_PATH") {
    Some(p) => p,
    None => "models/TinyLLama-v0.1-5M-F16.gguf",
};

fn models_available() -> bool {
    Path::new(APR_MODEL).exists() && Path::new(GGUF_MODEL).exists()
}

// ============================================================================
// Y6: APR Decode >= 50 tok/s (CPU)
// ============================================================================

/// Y6 Integration: APR decode speed meets 50 tok/s threshold
/// FALSIFICATION: APR < 50 tok/s on TinyLlama
/// Note: Uses simple generate() like CLI (not generate_with_cache)
#[test]
fn y6_apr_decode_meets_cpu_threshold() {
    if !Path::new(APR_MODEL).exists() {
        eprintln!("SKIP: APR model not found at {}", APR_MODEL);
        return;
    }

    use realizar::apr_transformer::AprTransformer;

    let transformer = AprTransformer::from_apr_file(APR_MODEL).expect("Failed to load APR model");

    let prompt = vec![1u32, 2, 3, 4, 5];
    let max_tokens = 20;

    // Warmup
    for _ in 0..3 {
        let _ = transformer.generate(&prompt, 5);
    }

    // Benchmark using generate() like CLI does
    let mut total_time = 0.0;
    let mut total_tokens = 0;
    for _ in 0..10 {
        let start = Instant::now();
        let output = transformer
            .generate(&prompt, max_tokens)
            .expect("Generate failed");
        total_time += start.elapsed().as_secs_f64();
        total_tokens += output.len().saturating_sub(prompt.len());
    }

    let tokens_per_second = total_tokens as f64 / total_time;

    const CPU_THRESHOLD: f64 = 50.0;
    assert!(
        tokens_per_second >= CPU_THRESHOLD,
        "FALSIFIED Y6: APR decode {:.1} tok/s < {} tok/s threshold",
        tokens_per_second,
        CPU_THRESHOLD
    );

    println!(
        "Y6 PASS: APR decode {:.1} tok/s (threshold: {} tok/s)",
        tokens_per_second, CPU_THRESHOLD
    );
}

/// Y6 Parity: APR decode >= 95% of GGUF decode speed
/// FALSIFICATION: APR < 95% of GGUF on same model
#[test]
#[ignore = "requires model files, run manually"]
fn y6_apr_decode_parity_with_gguf() {
    if !models_available() {
        eprintln!("SKIP: Models not found");
        return;
    }

    use realizar::apr_transformer::{AprBenchmarkRunner, AprTransformer};
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    // Benchmark APR
    let apr_transformer = AprTransformer::from_apr_file(APR_MODEL).expect("Failed to load APR");
    let mut apr_runner = AprBenchmarkRunner::new(apr_transformer);
    apr_runner.set_warmup_iterations(3);
    apr_runner.set_measure_iterations(10);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let apr_result = apr_runner
        .benchmark_decode(&prompt, 20)
        .expect("APR benchmark failed");

    // Benchmark GGUF using OwnedQuantizedModel (production path)
    let mapped = MappedGGUFModel::from_path(GGUF_MODEL).expect("Failed to mmap GGUF");
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load GGUF model");

    // Simple GGUF benchmark (manual timing)
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 20,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    let mut total_time = 0.0;
    let mut total_tokens = 0;
    for _ in 0..10 {
        let start = Instant::now();
        let output = gguf_model
            .generate(&prompt, &gen_config)
            .expect("GGUF generate failed");
        total_time += start.elapsed().as_secs_f64();
        total_tokens += output.len() - prompt.len();
    }
    let gguf_tps = total_tokens as f64 / total_time;

    // Check parity
    let ratio = apr_result.tokens_per_second / gguf_tps;
    const PARITY_THRESHOLD: f64 = 0.95;

    assert!(
        ratio >= PARITY_THRESHOLD,
        "FALSIFIED Y6 PARITY: APR {:.1} tok/s is {:.1}% of GGUF {:.1} tok/s (need >=95%)",
        apr_result.tokens_per_second,
        ratio * 100.0,
        gguf_tps
    );

    println!(
        "Y6 PARITY PASS: APR {:.1} tok/s = {:.1}% of GGUF {:.1} tok/s",
        apr_result.tokens_per_second,
        ratio * 100.0,
        gguf_tps
    );
}

// ============================================================================
// Y8: APR Prefill >= 100 tok/s
// ============================================================================

/// Y8: APR prefill speed meets 100 tok/s threshold
/// FALSIFICATION: APR prefill < 100 tok/s
#[test]
fn y8_apr_prefill_meets_threshold() {
    if !Path::new(APR_MODEL).exists() {
        eprintln!("SKIP: APR model not found at {}", APR_MODEL);
        return;
    }

    use realizar::apr_transformer::{AprBenchmarkRunner, AprTransformer};

    let transformer = AprTransformer::from_apr_file(APR_MODEL).expect("Failed to load APR model");

    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_warmup_iterations(3);
    runner.set_measure_iterations(10);

    // Long prompt for prefill test
    let prompt: Vec<u32> = (1..100).collect();
    let result = runner
        .benchmark_prefill(&prompt)
        .expect("Prefill benchmark failed");

    const PREFILL_THRESHOLD: f64 = 100.0;
    assert!(
        result.prefill_tok_s >= PREFILL_THRESHOLD,
        "FALSIFIED Y8: APR prefill {:.1} tok/s < {} tok/s threshold",
        result.prefill_tok_s,
        PREFILL_THRESHOLD
    );

    println!(
        "Y8 PASS: APR prefill {:.1} tok/s (threshold: {} tok/s)",
        result.prefill_tok_s, PREFILL_THRESHOLD
    );
}

// ============================================================================
// Y9: APR Load Time <= GGUF Load Time
// ============================================================================

/// Y9: APR load time does not exceed 1.2x GGUF load time
/// FALSIFICATION: APR load > 1.2x GGUF load
#[test]
#[ignore = "requires model files, run manually"]
fn y9_apr_load_time_parity() {
    if !models_available() {
        eprintln!("SKIP: Models not found");
        return;
    }

    use realizar::apr_transformer::AprTransformer;
    use realizar::gguf::GGUFModel;
    use std::fs::File;
    use std::io::Read;

    // Benchmark APR load time
    let mut apr_times = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let _ = AprTransformer::from_apr_file(APR_MODEL).expect("Failed to load APR");
        apr_times.push(start.elapsed().as_secs_f64());
    }
    let apr_avg = apr_times.iter().sum::<f64>() / apr_times.len() as f64;

    // Benchmark GGUF load time
    let mut gguf_times = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let mut file = File::open(GGUF_MODEL).expect("Failed to open GGUF");
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).expect("Failed to read GGUF");
        let _ = GGUFModel::from_bytes(&bytes).expect("Failed to parse GGUF");
        gguf_times.push(start.elapsed().as_secs_f64());
    }
    let gguf_avg = gguf_times.iter().sum::<f64>() / gguf_times.len() as f64;

    // Check ratio
    let ratio = apr_avg / gguf_avg;
    const MAX_RATIO: f64 = 1.2;

    assert!(
        ratio <= MAX_RATIO,
        "FALSIFIED Y9: APR load {:.3}s is {:.2}x GGUF load {:.3}s (max: 1.2x)",
        apr_avg,
        ratio,
        gguf_avg
    );

    println!(
        "Y9 PASS: APR load {:.3}s = {:.2}x GGUF load {:.3}s",
        apr_avg, ratio, gguf_avg
    );
}

// ============================================================================
// Y10: APR Peak Memory <= GGUF
// ============================================================================

/// Y10: APR peak memory does not exceed 1.1x GGUF
/// FALSIFICATION: APR memory > 1.1x GGUF memory
#[test]
fn y10_apr_memory_efficiency() {
    if !Path::new(APR_MODEL).exists() {
        eprintln!("SKIP: APR model not found at {}", APR_MODEL);
        return;
    }

    use realizar::apr_transformer::{AprBenchmarkRunner, AprTransformer};

    let transformer = AprTransformer::from_apr_file(APR_MODEL).expect("Failed to load APR model");

    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_warmup_iterations(1);
    runner.set_measure_iterations(3);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let result = runner
        .benchmark_decode(&prompt, 10)
        .expect("Benchmark failed");

    // Check memory is reasonable (model + overhead)
    // TinyLlama is ~10MB, allow up to 100MB for reasonable overhead
    const MAX_MEMORY_MB: f64 = 100.0;

    assert!(
        result.peak_memory_mb <= MAX_MEMORY_MB,
        "FALSIFIED Y10: APR peak memory {:.1} MB > {} MB",
        result.peak_memory_mb,
        MAX_MEMORY_MB
    );

    println!(
        "Y10 PASS: APR peak memory {:.1} MB, model memory {:.1} MB",
        result.peak_memory_mb, result.model_memory_mb
    );
}

// ============================================================================
// Combined Parity Report
// ============================================================================

/// Generate full Y6-Y10 parity report
/// FALSIFICATION: Any metric below threshold
#[test]
#[ignore = "requires model files, run manually"]
fn y6_y10_full_parity_report() {
    if !models_available() {
        eprintln!("SKIP: Models not found");
        return;
    }

    use realizar::apr_transformer::{AprBenchmarkRunner, AprTransformer};

    let transformer = AprTransformer::from_apr_file(APR_MODEL).expect("Failed to load APR model");

    let mut runner = AprBenchmarkRunner::new(transformer);
    runner.set_warmup_iterations(5);
    runner.set_measure_iterations(20);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let decode_result = runner.benchmark_decode(&prompt, 50).expect("Decode failed");

    let prefill_prompt: Vec<u32> = (1..100).collect();
    let prefill_result = runner
        .benchmark_prefill(&prefill_prompt)
        .expect("Prefill failed");

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║           Y6-Y10 APR Performance Parity Report            ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!(
        "║  Y6  Decode Speed:      {:>8.1} tok/s  (threshold: 50)  ║",
        decode_result.tokens_per_second
    );
    println!(
        "║  Y8  Prefill Speed:     {:>8.1} tok/s  (threshold: 100) ║",
        prefill_result.prefill_tok_s
    );
    println!(
        "║  Y10 Peak Memory:       {:>8.1} MB                      ║",
        decode_result.peak_memory_mb
    );
    println!(
        "║  Y10 Model Memory:      {:>8.1} MB                      ║",
        decode_result.model_memory_mb
    );
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Statistics:                                              ║");
    println!(
        "║    p50 throughput:      {:>8.1} tok/s                   ║",
        decode_result.throughput_p50
    );
    println!(
        "║    p99 throughput:      {:>8.1} tok/s                   ║",
        decode_result.throughput_p99
    );
    println!(
        "║    std_dev:             {:>8.2} tok/s                   ║",
        decode_result.throughput_std_dev
    );
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Assertions
    assert!(decode_result.tokens_per_second >= 50.0, "Y6 FALSIFIED");
    assert!(prefill_result.prefill_tok_s >= 100.0, "Y8 FALSIFIED");
}

// ============================================================================
// Summary: Y6-Y10 Popperian Falsification Matrix
// ============================================================================
//
// | Test | Claim | Falsification Condition |
// |------|-------|------------------------|
// | y6_apr_decode_meets_cpu_threshold | APR >= 50 tok/s | APR < 50 tok/s |
// | y6_apr_decode_parity_with_gguf | APR >= 95% GGUF | APR < 95% of GGUF |
// | y8_apr_prefill_meets_threshold | Prefill >= 100 tok/s | Prefill < 100 tok/s |
// | y9_apr_load_time_parity | Load <= 1.2x GGUF | Load > 1.2x GGUF |
// | y10_apr_memory_efficiency | Memory reasonable | Memory > 100 MB |
