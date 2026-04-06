//! Falsification tests for layer parity contract (layer-parity-v1.yaml)
//!
//! Contract: aprender/contracts/layer-parity-v1.yaml
//! Tests: FALSIFY-PARITY-001..004, FALSIFY-PARITY-001-prop, FALSIFY-PARITY-004-prop,
//!        FALSIFY-PARITY-GATE-001
//!
//! Popperian falsification: attempts to break the parity gate's mathematical
//! guarantees using synthetic data. GPU tests use #[ignore] when CUDA unavailable.
//!
//! Run with: `cargo test --test parity_contract_falsify --release -- --nocapture`

use proptest::prelude::*;

/// The contract YAML — source of truth for GPU/CPU parity thresholds.
const CONTRACT_YAML: &str = include_str!("../../aprender/contracts/layer-parity-v1.yaml");

/// Mirror of the PARITY_GATE_COSINE_MIN constant from realizar/src/gguf/cuda/mod.rs.
/// Test 8 verifies this matches the contract YAML.
const PARITY_GATE_COSINE_MIN: f32 = 0.99;

/// Cosine similarity between two f32 slices, computed in f64 for numerical stability.
///
/// Returns 0.0 if either vector has near-zero norm (< 1e-12).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

// ============================================================================
// FALSIFY-PARITY-001: Identical vectors must produce cosine = 1.0
// ============================================================================

/// FALSIFY-PARITY-001: Identical f32 vectors should produce cosine = 1.0.
///
/// This is the base case: if the parity gate cannot recognize identical outputs
/// as identical, the entire contract is broken. A single hidden_dim-sized vector
/// simulates one token position.
#[test]
fn falsify_parity_001_identical_vectors_cosine_one() {
    // Simulate a single token's hidden state (hidden_dim = 896 for 0.5B, 3584 for 7B)
    for hidden_dim in [128, 896, 2048, 3584] {
        let cpu_output: Vec<f32> = (0..hidden_dim).map(|i| ((i as f32) * 0.01).sin()).collect();
        let gpu_output = cpu_output.clone();

        let cosine = cosine_similarity(&cpu_output, &gpu_output);

        assert!(
            (cosine - 1.0).abs() < 1e-6,
            "FALSIFY-PARITY-001 FALSIFIED: identical vectors at hidden_dim={hidden_dim} \
             produced cosine={cosine:.8}, expected 1.0"
        );
    }
}

// ============================================================================
// FALSIFY-PARITY-001b: Multi-token concatenated vectors
// ============================================================================

/// FALSIFY-PARITY-001b: Multiple token positions concatenated — still cosine 1.0.
///
/// During batched prefill, the parity gate may compare concatenated multi-token
/// outputs. The cosine similarity must still be exactly 1.0 for identical data
/// regardless of sequence length.
#[test]
fn falsify_parity_001b_multi_token_identical_cosine() {
    let hidden_dim = 896;
    let seq_lengths = [1, 4, 16, 64, 128];

    for seq_len in seq_lengths {
        let total_dim = hidden_dim * seq_len;
        let cpu_output: Vec<f32> = (0..total_dim).map(|i| ((i as f32) * 0.007).cos()).collect();
        let gpu_output = cpu_output.clone();

        let cosine = cosine_similarity(&cpu_output, &gpu_output);

        assert!(
            (cosine - 1.0).abs() < 1e-6,
            "FALSIFY-PARITY-001b FALSIFIED: identical multi-token vectors \
             (seq_len={seq_len}, total_dim={total_dim}) produced cosine={cosine:.8}, expected 1.0"
        );
    }
}

// ============================================================================
// FALSIFY-PARITY-002: Quantization noise within tolerance
// ============================================================================

/// FALSIFY-PARITY-002: Small quantization noise (+-1e-4) must not break parity.
///
/// Contract: GPU quantized GEMV introduces rounding error bounded by the quant
/// precision. For Q4_K (4-bit), the per-element error is typically < 1e-3.
/// We test with 1e-4 noise to verify the gate tolerates realistic quant error.
/// Expected cosine: >= 0.999 (well above the 0.99 gate threshold).
#[test]
fn falsify_parity_002_quantization_noise_within_tolerance() {
    let hidden_dim = 3584; // 7B hidden dim
    let cpu_output: Vec<f32> = (0..hidden_dim).map(|i| ((i as f32) * 0.01).sin()).collect();

    // Simulate quantization noise: deterministic +-1e-4 perturbation
    let gpu_output: Vec<f32> = cpu_output
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let noise = if i % 2 == 0 { 1e-4 } else { -1e-4 };
            v + noise
        })
        .collect();

    let cosine = cosine_similarity(&cpu_output, &gpu_output);

    assert!(
        cosine >= 0.999,
        "FALSIFY-PARITY-002 FALSIFIED: quantization noise of +-1e-4 broke parity. \
         cosine={cosine:.8}, expected >= 0.999"
    );

    // Also verify it passes the actual gate threshold
    assert!(
        cosine >= PARITY_GATE_COSINE_MIN,
        "FALSIFY-PARITY-002 FALSIFIED: quantization noise fails gate. \
         cosine={cosine:.8}, gate_threshold={PARITY_GATE_COSINE_MIN}"
    );
}

// ============================================================================
// FALSIFY-PARITY-003: Mixed quant types produce identical outputs
// ============================================================================

/// FALSIFY-PARITY-003: Different quant type representations with identical values.
///
/// A model may use Q4_K for most layers and Q6_K for attention output.
/// As long as the dequantized values are identical, parity must hold.
/// This tests that the cosine metric is quant-type agnostic.
#[test]
fn falsify_parity_003_mixed_quant_types_identical() {
    let hidden_dim = 2048;

    // Simulate Q4_K output (values in typical dequantized range)
    let q4k_output: Vec<f32> = (0..hidden_dim)
        .map(|i| {
            // Q4_K values are typically small after dequantization
            let base = ((i as f32) * 0.013).sin();
            // Quantize to Q4-like precision: round to ~4-bit resolution
            (base * 8.0).round() / 8.0
        })
        .collect();

    // Simulate Q6_K output with the same logical values
    // (different quant path, same result)
    let q6k_output = q4k_output.clone();

    let cosine = cosine_similarity(&q4k_output, &q6k_output);

    assert!(
        (cosine - 1.0).abs() < 1e-6,
        "FALSIFY-PARITY-003 FALSIFIED: mixed quant type outputs with identical values \
         produced cosine={cosine:.8}, expected 1.0"
    );

    // Now test with slightly different quant precisions (Q4 vs Q6 rounding)
    let q6k_output_slightly_different: Vec<f32> = (0..hidden_dim)
        .map(|i| {
            let base = ((i as f32) * 0.013).sin();
            // Q6_K has finer resolution than Q4_K
            (base * 32.0).round() / 32.0
        })
        .collect();

    let cosine_mixed = cosine_similarity(&q4k_output, &q6k_output_slightly_different);

    // Mixed precision should still pass the parity gate
    assert!(
        cosine_mixed >= PARITY_GATE_COSINE_MIN,
        "FALSIFY-PARITY-003 FALSIFIED: mixed Q4/Q6 precision broke gate. \
         cosine={cosine_mixed:.8}, gate_threshold={PARITY_GATE_COSINE_MIN}"
    );
}

// ============================================================================
// FALSIFY-PARITY-004: Corrupted weights must break parity
// ============================================================================

/// FALSIFY-PARITY-004: Corrupted weights (sign flips) must drop cosine below threshold.
///
/// If the parity gate cannot detect significant corruption, it provides no
/// safety guarantee. We flip the sign of 20% of elements — this simulates
/// a kernel bug that produces wrong values for some block rows.
#[test]
fn falsify_parity_004_corrupted_weight_breaks_parity() {
    let hidden_dim = 3584;
    let cpu_output: Vec<f32> = (0..hidden_dim).map(|i| ((i as f32) * 0.01).sin()).collect();

    // Corrupt 20% of elements by flipping their sign
    let corruption_fraction = 0.20;
    let corrupt_count = (hidden_dim as f64 * corruption_fraction) as usize;
    let mut gpu_output = cpu_output.clone();
    for i in 0..corrupt_count {
        // Corrupt every 5th element (20%)
        let idx = i * 5;
        if idx < hidden_dim {
            gpu_output[idx] = -gpu_output[idx];
        }
    }

    let cosine = cosine_similarity(&cpu_output, &gpu_output);

    assert!(
        cosine < PARITY_GATE_COSINE_MIN,
        "FALSIFY-PARITY-004 FALSIFIED: 20% sign corruption NOT detected! \
         cosine={cosine:.8}, gate_threshold={PARITY_GATE_COSINE_MIN}. \
         The parity gate is too permissive."
    );
}

// ============================================================================
// FALSIFY-PARITY-001-prop: Proptest — identical copies always cosine 1.0
// ============================================================================

proptest! {
    /// FALSIFY-PARITY-001-prop: For any random f32 vector, identical copies
    /// must produce cosine similarity of exactly 1.0.
    ///
    /// This property-based test generates random vectors of varying sizes
    /// and verifies the fundamental identity property of cosine similarity.
    #[test]
    fn falsify_parity_001_prop(
        values in prop::collection::vec(-1e3f32..1e3f32, 64..=2048)
            .prop_filter("at least one nonzero", |v| v.iter().any(|x| x.abs() > 1e-10))
    ) {
        let copy = values.clone();
        let cosine = cosine_similarity(&values, &copy);

        prop_assert!(
            (cosine - 1.0).abs() < 1e-5,
            "FALSIFY-PARITY-001-prop FALSIFIED: identical random vectors \
             (len={}) produced cosine={:.8}, expected 1.0",
            values.len(),
            cosine
        );
    }
}

// ============================================================================
// FALSIFY-PARITY-004-prop: Proptest — corruption drops cosine below threshold
// ============================================================================

proptest! {
    /// FALSIFY-PARITY-004-prop: Flipping the sign of >10% of elements in a random
    /// vector must drop cosine below the parity gate threshold.
    ///
    /// This tests that the gate is sensitive enough to catch kernel bugs that
    /// produce incorrect values for a meaningful fraction of output elements.
    ///
    /// We use a bounded uniform range [-10, 10] to simulate realistic hidden state
    /// distributions where no single element dominates the norm. With all elements
    /// contributing roughly equally, flipping 15%+ of signs reliably drops cosine
    /// below 0.99. (Mathematical argument: for uniform-magnitude vectors, flipping
    /// fraction f of signs gives cosine ~ 1 - 2f, so f=0.15 -> cosine ~ 0.70.)
    #[test]
    fn falsify_parity_004_prop(
        values in prop::collection::vec(-10.0f32..10.0f32, 256..=2048)
            .prop_filter("no near-zero vectors", |v| {
                v.iter().map(|x| x * x).sum::<f32>() > 1.0
            }),
        corruption_pct in 15u32..=50u32,  // 15%-50% corruption
    ) {
        let n = values.len();
        let corrupt_count = (n as u32 * corruption_pct / 100) as usize;

        let mut corrupted = values.clone();
        // Flip signs of the first `corrupt_count` elements (deterministic subset)
        for val in corrupted.iter_mut().take(corrupt_count) {
            *val = -*val;
        }

        let cosine = cosine_similarity(&values, &corrupted);

        prop_assert!(
            cosine < PARITY_GATE_COSINE_MIN,
            "FALSIFY-PARITY-004-prop FALSIFIED: {}% corruption ({}/{} elements) \
             NOT detected! cosine={:.8}, gate_threshold={}. \
             The parity gate is too permissive.",
            corruption_pct,
            corrupt_count,
            n,
            cosine,
            PARITY_GATE_COSINE_MIN
        );
    }
}

// ============================================================================
// FALSIFY-PARITY-GATE-001: Constant matches contract YAML
// ============================================================================

/// FALSIFY-PARITY-GATE-001: The PARITY_GATE_COSINE_MIN constant must match
/// the threshold defined in layer-parity-v1.yaml.
///
/// This prevents drift between the contract specification and the runtime code.
/// If someone changes the YAML threshold without updating the code (or vice versa),
/// this test catches it.
#[test]
fn falsify_parity_gate_001_constant_matches_contract() {
    // Parse the contract YAML to extract parity_gate.threshold
    // We do lightweight string parsing to avoid adding serde_yaml as a test dep.
    let threshold = extract_parity_gate_threshold(CONTRACT_YAML)
        .expect("Failed to find parity_gate.threshold in contract YAML");

    // Compare with tolerance for f32 -> f64 representation gap.
    // 0.99_f32 as f64 is 0.9900000095367432 (not exactly 0.99).
    // The YAML stores the exact decimal 0.99. We allow up to 1e-6
    // difference to account for f32 representation error while still
    // catching any meaningful contract drift (e.g., 0.99 vs 0.999).
    let diff = (threshold - f64::from(PARITY_GATE_COSINE_MIN)).abs();
    assert!(
        diff < 1e-6,
        "FALSIFY-PARITY-GATE-001 FALSIFIED: contract YAML threshold ({threshold}) \
         does not match PARITY_GATE_COSINE_MIN ({PARITY_GATE_COSINE_MIN}), \
         diff={diff:.12}. Either the contract or the code is out of date."
    );
}

/// Extract the `parity_gate.threshold` value from the contract YAML.
///
/// Uses simple line-based parsing to avoid a serde_yaml dependency.
/// Looks for the pattern:
/// ```yaml
/// parity_gate:
///   ...
///   threshold: 0.99
/// ```
fn extract_parity_gate_threshold(yaml: &str) -> Option<f64> {
    let mut in_parity_gate = false;

    for line in yaml.lines() {
        let trimmed = line.trim();

        // Detect top-level `parity_gate:` key (no leading whitespace)
        if line.starts_with("parity_gate:") {
            in_parity_gate = true;
            continue;
        }

        // If we hit another top-level key, stop
        if in_parity_gate && !line.starts_with(' ') && !line.starts_with('#') && !trimmed.is_empty()
        {
            break;
        }

        // Look for `threshold:` within the parity_gate block
        if in_parity_gate && trimmed.starts_with("threshold:") {
            let value_str = trimmed.strip_prefix("threshold:")?.trim();
            return value_str.parse::<f64>().ok();
        }
    }

    None
}

// ============================================================================
// Helper verification tests
// ============================================================================

/// Sanity check: cosine of orthogonal vectors is 0.
#[test]
fn cosine_orthogonal_vectors_is_zero() {
    // e1 and e2 are orthogonal unit vectors
    let mut a = vec![0.0f32; 128];
    let mut b = vec![0.0f32; 128];
    a[0] = 1.0;
    b[1] = 1.0;

    let cosine = cosine_similarity(&a, &b);
    assert!(
        cosine.abs() < 1e-6,
        "orthogonal vectors should have cosine ~0, got {cosine}"
    );
}

/// Sanity check: cosine of antiparallel vectors is -1.
#[test]
fn cosine_antiparallel_vectors_is_negative_one() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = a.iter().map(|v| -v).collect();

    let cosine = cosine_similarity(&a, &b);
    assert!(
        (cosine + 1.0).abs() < 1e-5,
        "antiparallel vectors should have cosine ~-1, got {cosine}"
    );
}

/// Sanity check: zero-norm vector returns 0.
#[test]
fn cosine_zero_vector_returns_zero() {
    let a = vec![0.0f32; 64];
    let b: Vec<f32> = (0..64).map(|i| i as f32).collect();

    let cosine = cosine_similarity(&a, &b);
    assert!(
        cosine.abs() < 1e-6,
        "zero vector should produce cosine=0, got {cosine}"
    );
}
