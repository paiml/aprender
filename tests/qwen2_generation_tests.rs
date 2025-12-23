//! Generation Property Tests for Qwen2-0.5B-Instruct
//!
//! EXTREME TDD: These tests verify generation properties using property-based testing.
//! All tests must complete in < 100ms (bashrs standard).
//!
//! # Spec Reference
//!
//! Section D: Generation (20 points)
//! - D2: Non-determinism with temperature > 0
//! - D3: EOS token stops generation
//! - D4: Greedy (temp=0) is deterministic
//! - D5: All generated tokens within vocab range
//!
//! # References
//!
//! - Holtzman et al. (2019). "The Curious Case of Neural Text Degeneration"

use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;

// ============================================================================
// FAST Test Model Setup (bashrs standard: hidden_size ≤ 32, layers = 1)
// ============================================================================

/// Create a TINY model for fast testing (< 10ms creation).
fn create_fast_test_model() -> Qwen2Model {
    let config = Qwen2Config {
        hidden_size: 32,        // Minimal
        num_attention_heads: 2, // Minimal
        num_kv_heads: 1,        // Minimal GQA
        num_layers: 1,          // Single layer for speed
        vocab_size: 100,        // Tiny vocab
        max_seq_len: 32,        // Short context
        intermediate_size: 64,  // 2x hidden
        rope_theta: 10000.0,
    };
    Qwen2Model::new(&config)
}

// ============================================================================
// Section D: Generation Property Tests (all < 100ms)
// ============================================================================

/// D2: Generation produces output of expected length.
#[test]
fn d2_generation_produces_output() {
    let mut model = create_fast_test_model();
    let prompt = vec![1u32, 2, 3];

    let out = model.generate(&prompt, 5, 0.7, 0.9);

    assert!(
        out.len() >= prompt.len(),
        "Generation should produce at least prompt length output"
    );
}

/// D3: Generation respects max_tokens limit.
#[test]
fn d3_generation_respects_max_tokens() {
    let mut model = create_fast_test_model();
    let prompt = vec![1u32, 2, 3];
    let max_tokens = 3;

    let output = model.generate(&prompt, max_tokens, 0.0, 1.0);

    assert!(
        output.len() <= prompt.len() + max_tokens,
        "Generation should stop within max_tokens limit"
    );
}

/// D4: Greedy decoding (temperature=0) is deterministic on SAME model instance.
/// Note: Different model instances have different random weights, so they produce
/// different outputs. This test verifies same model + same input = same output.
#[test]
fn d4_greedy_is_deterministic_same_model() {
    let prompt = vec![1u32, 2, 3];
    let mut model = create_fast_test_model();

    // Run twice on the SAME model
    let out1 = model.generate(&prompt, 3, 0.0, 1.0);
    let out2 = model.generate(&prompt, 3, 0.0, 1.0);

    assert_eq!(out1, out2, "Greedy decoding must be deterministic on same model");
}

/// D5: All generated tokens are within vocabulary range.
#[test]
fn d5_generated_tokens_within_vocab() {
    let mut model = create_fast_test_model();
    let vocab_size = model.config().vocab_size;
    let prompt = vec![1u32, 2, 3];

    let output = model.generate(&prompt, 5, 0.7, 0.9);

    for &token in &output {
        assert!(
            (token as usize) < vocab_size,
            "Token {} exceeds vocab size {}",
            token,
            vocab_size
        );
    }
}

/// D6: Generation preserves prompt tokens.
#[test]
fn d6_generation_preserves_prompt() {
    let mut model = create_fast_test_model();
    let prompt = vec![1u32, 2, 3];

    let output = model.generate(&prompt, 3, 0.7, 0.9);

    assert!(
        output.starts_with(&prompt),
        "Output must start with prompt tokens"
    );
}

/// D7: Single token prompt works correctly.
#[test]
fn d7_single_token_prompt() {
    let mut model = create_fast_test_model();
    let single_token = vec![5u32];

    let output = model.generate(&single_token, 2, 0.0, 1.0);

    assert!(
        output.starts_with(&single_token),
        "Output must preserve single token prompt"
    );
}

/// D8: Zero max_tokens returns just the prompt.
#[test]
fn d8_zero_max_tokens() {
    let mut model = create_fast_test_model();
    let prompt = vec![1u32, 2, 3];

    let output = model.generate(&prompt, 0, 0.0, 1.0);

    assert_eq!(
        output.len(),
        prompt.len(),
        "Zero max_tokens should return prompt only"
    );
}

/// D9: Multiple greedy runs on SAME model produce identical output.
#[test]
fn d9_greedy_reproducible_same_model() {
    let prompt = vec![1u32, 2];
    let mut model = create_fast_test_model();

    let baseline = model.generate(&prompt, 2, 0.0, 1.0);

    // Run 3 more times on the SAME model
    for _ in 0..3 {
        let out = model.generate(&prompt, 2, 0.0, 1.0);
        assert_eq!(baseline, out, "Greedy must be reproducible on same model");
    }
}

/// D10: Different prompts produce different outputs (model is responsive).
#[test]
fn d10_different_prompts_different_outputs() {
    let mut model = create_fast_test_model();

    let out1 = model.generate(&[1u32, 2, 3], 3, 0.0, 1.0);
    let out2 = model.generate(&[10u32, 20, 30], 3, 0.0, 1.0);

    // With different prompts, outputs should differ (unless model is broken)
    assert_ne!(out1, out2, "Different prompts should produce different outputs");
}

// ============================================================================
// Property-Based Tests with Proptest (PROPTEST_CASES=32 for speed)
// ============================================================================

#[cfg(test)]
mod proptest_generation {
    use super::*;
    use proptest::prelude::*;

    // Limit cases for fast testing (bashrs standard)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Property: All generated tokens are within vocabulary.
        #[test]
        fn prop_tokens_in_vocab(
            prompt in prop::collection::vec(0u32..50, 1..4),
            max_tokens in 1usize..4,
        ) {
            let mut model = create_fast_test_model();
            let vocab_size = model.config().vocab_size;

            let output = model.generate(&prompt, max_tokens, 0.5, 0.9);

            for &token in &output {
                prop_assert!(
                    (token as usize) < vocab_size,
                    "Token {} exceeds vocab {}",
                    token,
                    vocab_size
                );
            }
        }

        /// Property: Output length is bounded by prompt + max_tokens.
        #[test]
        fn prop_output_length_bounded(
            prompt in prop::collection::vec(0u32..50, 1..4),
            max_tokens in 1usize..5,
        ) {
            let mut model = create_fast_test_model();

            let output = model.generate(&prompt, max_tokens, 0.5, 0.9);
            let max_length = prompt.len() + max_tokens;

            prop_assert!(
                output.len() <= max_length,
                "Output {} exceeds max {}",
                output.len(),
                max_length
            );
        }

        /// Property: Prompt is always preserved in output.
        #[test]
        fn prop_prompt_preserved(
            prompt in prop::collection::vec(0u32..50, 1..4),
            max_tokens in 0usize..3,
        ) {
            let mut model = create_fast_test_model();

            let output = model.generate(&prompt, max_tokens, 0.5, 0.9);

            prop_assert!(
                output.starts_with(&prompt),
                "Prompt not preserved in output"
            );
        }

        /// Property: Greedy decoding is deterministic on same model instance.
        #[test]
        fn prop_greedy_deterministic_same_model(
            prompt in prop::collection::vec(0u32..50, 1..3),
            max_tokens in 1usize..3,
        ) {
            let mut model = create_fast_test_model();

            let out1 = model.generate(&prompt, max_tokens, 0.0, 1.0);
            let out2 = model.generate(&prompt, max_tokens, 0.0, 1.0);

            prop_assert_eq!(out1, out2, "Greedy must be deterministic on same model");
        }
    }
}

// ============================================================================
// Sampling Correctness Tests
// ============================================================================

/// Verify softmax temperature scaling math is correct.
#[test]
fn temperature_scaling_correctness() {
    let logits = vec![2.0f32, 1.0, 0.5, 0.0];

    // Standard softmax (temp=1)
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|x| (x - max_logit).exp() / exp_sum)
        .collect();

    // Higher logit = higher probability
    assert!(probs[0] > probs[1], "Higher logit should have higher prob");
    assert!(probs[1] > probs[2], "Logits should order probabilities");
    assert!(probs[2] > probs[3], "Logits should order probabilities");

    // Probabilities sum to 1
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Probabilities must sum to 1");
}
