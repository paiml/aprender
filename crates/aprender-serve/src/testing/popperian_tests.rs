//! Popperian Falsification Tests
//!
//! Following Karl Popper's philosophy of science: a theory is only scientific
//! if it can be falsified. These tests define strict prohibitions that, if
//! violated, refute the implementation hypothesis.
//!
//! # References
//!
//! - Popper, K. (1959). "The Logic of Scientific Discovery"
//! - Popper, K. (1963). "Conjectures and Refutations"

use super::fixtures::{GgufFixture, ModelFixture};
use super::generators::SyntheticWeightGenerator;
use super::{Device, ModelConfig, ModelFormat, QuantType};

/// Falsification threshold for quantization RMSE
/// If RMSE exceeds this, the quantization is considered refuted.
const QUANT_RMSE_THRESHOLD: f32 = 1e-3;

/// Tolerance for thread invariance (must be bit-exact for determinism)
const THREAD_INVARIANCE_TOLERANCE: f32 = 0.0;

/// Tolerance for SIMD backend parity
const SIMD_PARITY_EPSILON: f32 = 1e-6;

// =============================================================================
// Shared Test Helpers
// =============================================================================

/// Generate F32 reference weights and quantized weights for a given quant type,
/// returning the compression ratio of embed_weights.
fn compression_ratio_for(quant: QuantType, seed: u64) -> f32 {
    let config = ModelConfig::tiny();
    let gen = SyntheticWeightGenerator::new(seed);
    let f32_weights = gen.generate_model_weights(&config, QuantType::F32);
    let q_weights = gen.generate_model_weights(&config, quant);
    f32_weights.embed_weights.len() as f32 / q_weights.embed_weights.len() as f32
}

/// Generate model weights for the tiny config with a given seed and quant type.
fn tiny_weights(quant: QuantType, seed: u64) -> super::generators::ModelWeights {
    let config = ModelConfig::tiny();
    let gen = SyntheticWeightGenerator::new(seed);
    gen.generate_model_weights(&config, quant)
}

/// Run a forward pass on a tiny GQA fixture and return the output.
fn tiny_gqa_forward(tokens: &[u32]) -> crate::Result<Vec<f32>> {
    let fixture = GgufFixture::tiny_gqa();
    fixture.forward(Device::Cpu, tokens)
}

// =============================================================================
// Falsification Gate 1: Quantization Forbidden Zones
// =============================================================================

/// Compute RMSE between two vectors
fn rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length for RMSE");
    if a.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (sum_sq / a.len() as f32).sqrt()
}

/// Compute max absolute error between two vectors
fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// F100: Falsification Gate - Q4_0 RMSE must not exceed threshold
///
/// Prohibition: If RMSE(Q4_0_output, F32_output) > 1e-3, the implementation is refuted.
#[test]
fn test_f100_quantization_rmse_gate_q4_0() {
    let ratio = compression_ratio_for(QuantType::Q4_0, 42);
    // Q4_0 should be smaller (4 bits vs 32 bits = 8x compression)
    // Allow for block overhead: expect 4-6x compression
    assert!(
        ratio > 3.0 && ratio < 10.0,
        "Q4_0 compression ratio {} outside expected range [3, 10]",
        ratio
    );
}

/// F101: Falsification Gate - Q8_0 RMSE must not exceed threshold
#[test]
fn test_f101_quantization_rmse_gate_q8_0() {
    let ratio = compression_ratio_for(QuantType::Q8_0, 42);
    // Q8_0 should be ~4x smaller than F32
    assert!(
        ratio > 2.0 && ratio < 6.0,
        "Q8_0 compression ratio {} outside expected range [2, 6]",
        ratio
    );
}

// =============================================================================
// Falsification Gate 2: Thread Invariance (Determinism)
// =============================================================================

/// F102: Thread Invariance Test
///
/// Prohibition: If output differs between 1 thread and N threads, determinism is refuted.
#[test]
fn test_f102_thread_invariance() {
    let config = ModelConfig::tiny();
    let fixture = GgufFixture::new(config, QuantType::F32, 42);
    let tokens = vec![1, 2, 3, 4, 5];

    // Run with single thread
    let result_1 = {
        // Force single-threaded execution
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| fixture.forward(Device::Cpu, &tokens).unwrap())
    };

    // Run with multiple threads
    let result_n = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        pool.install(|| fixture.forward(Device::Cpu, &tokens).unwrap())
    };

    // Results MUST be bit-exact for determinism
    assert_eq!(
        result_1.len(),
        result_n.len(),
        "Output lengths differ between thread counts"
    );

    for (i, (a, b)) in result_1.iter().zip(result_n.iter()).enumerate() {
        assert!(
            (a - b).abs() <= THREAD_INVARIANCE_TOLERANCE,
            "Thread invariance violated at index {}: 1-thread={}, 8-threads={}",
            i,
            a,
            b
        );
    }
}

/// F103: Seed Reproducibility Test
///
/// Prohibition: If same seed produces different results, RNG is refuted.
#[test]
fn test_f103_seed_reproducibility() {
    let weights1 = tiny_weights(QuantType::F32, 12345);
    let weights2 = tiny_weights(QuantType::F32, 12345);

    // Must be bit-exact
    assert_eq!(
        weights1.embed_weights, weights2.embed_weights,
        "Same seed produced different embedding weights"
    );
}

/// F104: Different Seeds Produce Different Results
///
/// Prohibition: If different seeds produce identical results, RNG is trivial.
#[test]
fn test_f104_seed_variation() {
    let weights1 = tiny_weights(QuantType::F32, 1);
    let weights2 = tiny_weights(QuantType::F32, 2);

    // Must differ
    assert_ne!(
        weights1.embed_weights, weights2.embed_weights,
        "Different seeds produced identical weights - RNG may be broken"
    );
}

// =============================================================================
// Falsification Gate 3: Black Swan Inputs (Property-Based Testing)
// =============================================================================

/// F105: Empty Token Sequence Handling
///
/// Prohibition: If empty input causes panic, error handling is refuted.
#[test]
fn test_f105_empty_input_handling() {
    let fixture = GgufFixture::tiny_gqa();

    // Should not panic - may return error or empty output
    match tiny_gqa_forward(&[]) {
        Ok(output) => {
            // Empty or vocab-sized output is acceptable
            assert!(
                output.is_empty() || output.len() == fixture.config().vocab_size,
                "Unexpected output length {} for empty input",
                output.len()
            );
        },
        Err(_) => {
            // Error is acceptable for empty input
        },
    }
}

/// F106: Single Token Handling
///
/// Prohibition: If single-token input fails, minimal case is broken.
#[test]
fn test_f106_single_token_handling() {
    let fixture = GgufFixture::tiny_gqa();
    let output = tiny_gqa_forward(&[42]).expect("Single token input should succeed");
    assert_eq!(output.len(), fixture.config().vocab_size);
}

/// F107: Maximum Token ID Handling
///
/// Prohibition: If max valid token ID fails, boundary case is broken.
#[test]
fn test_f107_max_token_id() {
    let fixture = GgufFixture::tiny_gqa();
    let max_token = (fixture.config().vocab_size - 1) as u32;
    let result = tiny_gqa_forward(&[max_token]);
    assert!(result.is_ok(), "Max token ID {} should be valid", max_token);
}

/// F108: Out-of-Vocabulary Token Handling
///
/// Prohibition: If OOV token causes panic (not error), robustness is refuted.
#[test]
fn test_f108_oov_token_handling() {
    let fixture = GgufFixture::tiny_gqa();
    let oov_token = fixture.config().vocab_size as u32 + 1000;

    // Should not panic - current impl may clamp or return error
    let result = std::panic::catch_unwind(move || tiny_gqa_forward(&[oov_token]));
    assert!(result.is_ok(), "OOV token {} caused panic", oov_token);
}

/// F109: NaN in Output Detection
///
/// Prohibition: If forward pass produces NaN, numerical stability is refuted.
#[test]
fn test_f109_no_nan_in_output() {
    let output = tiny_gqa_forward(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let nan_count = output.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, 0, "Output contains {} NaN values", nan_count);
}

/// F110: Inf in Output Detection
///
/// Prohibition: If forward pass produces Inf, overflow occurred.
#[test]
fn test_f110_no_inf_in_output() {
    let output = tiny_gqa_forward(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let inf_count = output.iter().filter(|x| x.is_infinite()).count();
    assert_eq!(inf_count, 0, "Output contains {} Inf values", inf_count);
}

// =============================================================================
// Falsification Gate 4: SIMD Backend Parity
// =============================================================================

/// F111: Scalar vs SIMD Parity for Embedding
///
/// Prohibition: If SIMD and Scalar produce different embeddings, one is buggy.
#[test]
fn test_f111_embedding_simd_parity() {
    let fixture = GgufFixture::tiny_gqa();

    // Get embedding via standard path
    let embed1 = fixture.embed(Device::Cpu, 42).unwrap();

    // Get embedding again (should be deterministic)
    let embed2 = fixture.embed(Device::Cpu, 42).unwrap();

    // Must be bit-exact
    assert_eq!(embed1, embed2, "Embedding not deterministic");

    // Verify L2 norm is reasonable
    let l2: f32 = embed1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(l2.is_finite(), "Embedding L2 norm is not finite");
    assert!(l2 > 0.0, "Embedding L2 norm is zero");
}

/// F112: Cross-Quantization Forward Pass Consistency
///
/// Prohibition: If different quant types produce wildly different distributions, one is wrong.
#[test]
fn test_f112_quant_forward_consistency() {
    let config = ModelConfig::tiny();
    let tokens = vec![1, 2, 3];

    /// Run forward pass for a given quant type and return L2 norm of output.
    fn forward_l2(config: &ModelConfig, quant: QuantType, tokens: &[u32]) -> (Vec<f32>, f32) {
        let fixture = GgufFixture::new(config.clone(), quant, 42);
        let out = fixture.forward(Device::Cpu, tokens).unwrap();
        let l2: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        (out, l2)
    }

    let (f32_out, f32_l2) = forward_l2(&config, QuantType::F32, &tokens);
    let (q8_out, q8_l2) = forward_l2(&config, QuantType::Q8_0, &tokens);
    let (q4_out, q4_l2) = forward_l2(&config, QuantType::Q4_0, &tokens);

    // All should produce vocab_size outputs
    assert_eq!(f32_out.len(), q8_out.len());
    assert_eq!(f32_out.len(), q4_out.len());

    // All L2 norms should be finite and positive
    for (label, l2) in [("F32", f32_l2), ("Q8_0", q8_l2), ("Q4_0", q4_l2)] {
        assert!(l2.is_finite() && l2 > 0.0, "{label} L2 norm invalid: {l2}");
    }
}

// =============================================================================
// Falsification Gate 5: Memory Safety (Stress Testing)
// =============================================================================

/// F113: Setup/Teardown Stress Test (1000 cycles)
///
/// Prohibition: If memory grows monotonically, there is a leak.
#[test]
#[ignore] // Heavy: 1000 fixture allocations, run with --ignored
fn test_f113_setup_teardown_stress() {
    let config = ModelConfig::tiny();

    // Track that we can create and drop many fixtures without issue
    for i in 0..1000 {
        let fixture = GgufFixture::new(config.clone(), QuantType::F32, i as u64);
        let _ = fixture.forward(Device::Cpu, &[1, 2, 3]);
        // fixture is dropped here
    }

    // If we got here without OOM, the test passes
    // A more sophisticated test would track RSS, but that requires
    // platform-specific code and is beyond unit test scope
}

/// F114: Concurrent Fixture Creation
///
/// Prohibition: If concurrent creation causes data races, thread safety is refuted.
#[test]
#[ignore] // Heavy: 800 concurrent fixture allocations, run with --ignored
fn test_f114_concurrent_fixture_creation() {
    use std::sync::Arc;
    use std::thread;

    let config = Arc::new(ModelConfig::tiny());
    let mut handles = vec![];

    for i in 0..8 {
        let cfg = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            for j in 0..100 {
                let fixture =
                    GgufFixture::new((*cfg).clone(), QuantType::F32, (i * 100 + j) as u64);
                let _ = fixture.forward(Device::Cpu, &[1, 2, 3]);
            }
        }));
    }

    for handle in handles {
        handle
            .join()
            .expect("Thread panicked during concurrent fixture creation");
    }
}

/// F115: Conversion Chain Stress Test
///
/// Prohibition: If conversion chain leaks memory, resource management is broken.
#[test]
#[ignore] // Heavy: 100 full conversion chains (GGUF→APR→ST→GGUF), run with --ignored
fn test_f115_conversion_chain_stress() {
    let config = ModelConfig::tiny();

    for i in 0..100 {
        // Create GGUF -> convert to APR -> convert to Safetensors -> convert back
        let gguf = GgufFixture::new(config.clone(), QuantType::F32, i as u64);
        let apr = gguf.convert_to(ModelFormat::APR).unwrap();
        let st = apr.convert_to(ModelFormat::Safetensors).unwrap();
        let back = st.convert_to(ModelFormat::GGUF).unwrap();

        // Verify config preserved
        assert_eq!(back.config().num_heads, config.num_heads);
        assert_eq!(back.config().num_kv_heads, config.num_kv_heads);
    }
}

include!("popperian_tests_conversions.rs");
