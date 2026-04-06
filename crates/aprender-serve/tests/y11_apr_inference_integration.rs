//! Y11: APR Inference Integration Tests (EXTREME TDD - RED Phase)
//!
//! Per Section Y.3 of the spec, APR inference must be wired into realizar.
//! These tests define Popperian falsification conditions for Y11-Y14.
//!
//! FALSIFICATION: `realizar run model.apr` fails or falls back to GGUF parser

use std::path::Path;

// ============================================================================
// Y11.1: APR File Detection
// ============================================================================

/// Y11.1a: detect_format correctly identifies APR v2 files
/// FALSIFICATION: APR v2 magic not recognized
/// NOTE: APR v2 detection requires implementation in format module
#[test]
#[ignore = "APR v2 format detection not yet implemented"]
fn y11_1a_detect_apr_v2_format() {
    use realizar::format::{detect_format, ModelFormat};

    // APR v2 magic: "APR2" (0x41 0x50 0x52 0x32)
    let apr_v2_data = b"APR2\x02\x00\x00\x00\x00\x00\x00\x00";

    let format = detect_format(apr_v2_data);
    assert!(format.is_ok(), "Should detect APR v2 format");
    assert_eq!(
        format.expect("test"),
        ModelFormat::Apr,
        "Should identify as APR format"
    );
}

/// Y11.1b: APR v1 magic also recognized for backwards compatibility
/// FALSIFICATION: APR v1 magic rejected
/// NOTE: APR v1 detection requires implementation in format module
#[test]
#[ignore = "APR v1 format detection not yet implemented"]
fn y11_1b_detect_apr_v1_format() {
    use realizar::format::{detect_format, ModelFormat};

    // APR v1 magic: "APRN" (0x41 0x50 0x52 0x4E)
    let apr_v1_data = b"APRN\x01\x00\x00\x00\x00\x00\x00\x00";

    let format = detect_format(apr_v1_data);
    assert!(format.is_ok(), "Should detect APR v1 format");
    assert_eq!(
        format.expect("test"),
        ModelFormat::Apr,
        "Should identify as APR format"
    );
}

// ============================================================================
// Y11.2: APR Transformer Loading
// ============================================================================

/// Y11.2a: AprTransformer can be constructed from APR file
/// FALSIFICATION: AprTransformer::from_apr_file() fails
#[test]
fn y11_2a_apr_transformer_from_file() {
    use realizar::apr_transformer::AprTransformer;

    // Skip if no test model available
    let test_model = "/tmp/test-tinyllama.apr";
    if !Path::new(test_model).exists() {
        eprintln!("SKIP: Test model not found at {}", test_model);
        return;
    }

    let transformer = AprTransformer::from_apr_file(test_model);
    assert!(
        transformer.is_ok(),
        "Should load transformer from APR file: {:?}",
        transformer.err()
    );
}

/// Y11.2b: APR transformer has correct architecture metadata
/// FALSIFICATION: Architecture info missing or wrong
#[test]
fn y11_2b_apr_architecture_detection() {
    use realizar::apr_transformer::AprTransformer;

    let test_model = "/tmp/test-tinyllama.apr";
    if !Path::new(test_model).exists() {
        eprintln!("SKIP: Test model not found at {}", test_model);
        return;
    }

    let transformer = AprTransformer::from_apr_file(test_model).expect("test");
    let config = transformer.config();

    // TinyLlama architecture
    assert!(config.hidden_dim > 0, "Should have hidden_dim");
    assert!(config.num_layers > 0, "Should have layers");
    assert!(config.vocab_size > 0, "Should have vocab_size");
}

// ============================================================================
// Y11.3: APR Inference Execution
// ============================================================================

/// Y11.3a: APR transformer can run forward pass
/// FALSIFICATION: forward() fails or returns wrong shape
#[test]
fn y11_3a_apr_forward_pass() {
    use realizar::apr_transformer::AprTransformer;

    let test_model = "/tmp/test-tinyllama.apr";
    if !Path::new(test_model).exists() {
        eprintln!("SKIP: Test model not found at {}", test_model);
        return;
    }

    let transformer = AprTransformer::from_apr_file(test_model).expect("test");

    // Run forward pass
    let input_tokens = vec![1u32, 2, 3];
    let logits = transformer.forward(&input_tokens);

    assert!(logits.is_ok(), "Forward pass should succeed");
    let logits = logits.expect("test");
    assert_eq!(
        logits.len(),
        transformer.config().vocab_size,
        "Output should have vocab_size logits"
    );
}

/// Y11.3b: APR transformer can generate tokens
/// FALSIFICATION: generate() fails or produces invalid tokens
#[test]
fn y11_3b_apr_generate() {
    use realizar::apr_transformer::AprTransformer;

    let test_model = "/tmp/test-tinyllama.apr";
    if !Path::new(test_model).exists() {
        eprintln!("SKIP: Test model not found at {}", test_model);
        return;
    }

    let transformer = AprTransformer::from_apr_file(test_model).expect("test");

    let prompt = vec![1u32, 2, 3];
    let max_tokens = 5;

    let generated = transformer.generate(&prompt, max_tokens);
    assert!(generated.is_ok(), "Generate should succeed");

    let tokens = generated.expect("test");
    assert!(
        tokens.len() >= prompt.len(),
        "Should generate at least prompt tokens"
    );
}

// ============================================================================
// Y11.4: CLI Integration
// ============================================================================

/// Y11.4a: `realizar run model.apr` uses native APR loader
/// FALSIFICATION: Falls back to GGUF parser (error mentions GGUF magic)
#[test]
#[ignore = "Run manually with: cargo test y11_4a --ignored"]
fn y11_4a_cli_uses_native_apr_loader() {
    use std::process::Command;

    let test_model = "/tmp/test-tinyllama.apr";
    if !Path::new(test_model).exists() {
        eprintln!("SKIP: Test model not found at {}", test_model);
        return;
    }

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "realizar",
            "--release",
            "--",
            "run",
            test_model,
            "Hello",
            "--max-tokens",
            "5",
        ])
        .output()
        .expect("Failed to run realizar");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should NOT mention GGUF parser errors
    assert!(
        !stderr.contains("Invalid GGUF magic"),
        "Should NOT fall back to GGUF parser. Stderr: {}",
        stderr
    );

    // Should successfully load APR
    assert!(
        stdout.contains("APR") || stdout.contains("loaded") || output.status.success(),
        "Should load APR file. Status: {}, Stdout: {}, Stderr: {}",
        output.status,
        stdout,
        stderr
    );
}

// ============================================================================
// Y12: APR Performance Parity
// ============================================================================

/// Y12.1: APR inference speed >= 95% of GGUF
/// FALSIFICATION: APR throughput < 95% of GGUF on same model
#[test]
#[ignore = "Requires both APR and GGUF versions of same model"]
fn y12_1_apr_performance_parity() {
    use realizar::apr_transformer::{AprBenchmarkRunner, AprTransformer};

    let apr_model = "/tmp/test-tinyllama.apr";
    let gguf_model = option_env!("GGUF_MODEL_PATH").unwrap_or("models/TinyLLama-v0.1-5M-F16.gguf");

    if !Path::new(apr_model).exists() || !Path::new(gguf_model).exists() {
        eprintln!("SKIP: Test models not found");
        return;
    }

    // Benchmark APR
    let apr_transformer = AprTransformer::from_apr_file(apr_model).expect("test");
    let mut apr_runner = AprBenchmarkRunner::new(apr_transformer);
    apr_runner.set_warmup_iterations(3);
    apr_runner.set_measure_iterations(10);

    let prompt = vec![1u32, 2, 3, 4, 5];
    let apr_result = apr_runner.benchmark_decode(&prompt, 20).expect("test");

    // For now, just check APR meets minimum threshold
    // Full GGUF comparison requires GGUFBenchmarkRunner
    assert!(
        apr_result.tokens_per_second >= 50.0 * 0.95, // 95% of 50 tok/s threshold
        "APR should achieve at least 47.5 tok/s, got {:.1}",
        apr_result.tokens_per_second
    );
}

// ============================================================================
// Summary: Y11 Popperian Falsification Matrix
// ============================================================================
//
// | Test | Claim | Falsification Condition |
// |------|-------|------------------------|
// | Y11.1a | APR v2 format detected | Magic bytes not recognized |
// | Y11.1b | APR v1 format detected | Magic bytes not recognized |
// | Y11.2a | AprTransformer::from_apr_file() | Method fails to load |
// | Y11.2b | Architecture metadata present | Config fields missing |
// | Y11.3a | Forward pass works | forward() fails |
// | Y11.3b | Generate works | generate() fails |
// | Y11.4a | CLI uses native APR | Falls back to GGUF parser |
// | Y12.1 | APR >= 95% of GGUF speed | Throughput below threshold |
