//! Comprehensive tests for advanced sampling algorithms
//!
//! This module tests edge cases and code paths in algorithms.rs
//! that may not be covered by the main test suite.

use crate::generate::algorithms::*;
use crate::tensor::Tensor;

// =============================================================================
// Min-P Sampling Tests
// =============================================================================

#[test]
fn test_sample_min_p_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_min_p_invalid_min_p_negative() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let result = sample_min_p(&logits, -0.01, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_sample_min_p_invalid_min_p_greater_than_one() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let result = sample_min_p(&logits, 1.01, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_sample_min_p_boundary_zero() {
    let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    // min_p = 0.0 should include all tokens
    let result = sample_min_p(&logits, 0.0, 0.5).expect("test");
    assert!(result < 4);
}

#[test]
fn test_sample_min_p_boundary_one() {
    let logits = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 100.0]).expect("test");
    // min_p = 1.0 should only include max prob token
    let result = sample_min_p(&logits, 1.0, 0.5).expect("test");
    assert_eq!(result, 3);
}

#[test]
fn test_sample_min_p_all_equal_probs() {
    let logits = Tensor::from_vec(vec![5], vec![0.0; 5]).expect("test");
    // All equal, all should pass threshold
    let result = sample_min_p(&logits, 0.5, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_min_p_rng_selection() {
    let logits = Tensor::from_vec(vec![3], vec![10.0, 10.0, 0.0]).expect("test");
    // Two equal high tokens, rng=0.0 should pick first
    let result = sample_min_p(&logits, 0.5, 0.0).expect("test");
    assert!(result == 0 || result == 1);
}

// =============================================================================
// MirostatState Tests
// =============================================================================

#[test]
fn test_mirostat_state_default_values() {
    let state = MirostatState::default();
    assert!((state.tau - 5.0).abs() < 1e-6);
    assert!((state.eta - 0.1).abs() < 1e-6);
    assert!((state.mu - 10.0).abs() < 1e-6);
}

#[test]
fn test_mirostat_state_new_tau_sets_mu() {
    let state = MirostatState::new(3.0);
    assert!((state.tau - 3.0).abs() < 1e-6);
    assert!((state.mu - 6.0).abs() < 1e-6); // mu = 2 * tau
}

#[test]
fn test_mirostat_state_with_eta_builder() {
    let state = MirostatState::new(5.0).with_eta(0.5);
    assert!((state.eta - 0.5).abs() < 1e-6);
}

#[test]
fn test_mirostat_state_update_increases_mu() {
    let mut state = MirostatState::new(5.0).with_eta(0.1);
    let initial_mu = state.mu;
    // Observed surprise < tau, so mu should increase
    state.update(2.0);
    assert!(state.mu > initial_mu);
}

#[test]
fn test_mirostat_state_update_decreases_mu() {
    let mut state = MirostatState::new(5.0).with_eta(0.1);
    let initial_mu = state.mu;
    // Observed surprise > tau, so mu should decrease
    state.update(10.0);
    assert!(state.mu < initial_mu);
}

#[test]
fn test_mirostat_state_clone() {
    let state = MirostatState::new(3.0).with_eta(0.2);
    let cloned = state.clone();
    assert!((cloned.tau - state.tau).abs() < 1e-6);
    assert!((cloned.eta - state.eta).abs() < 1e-6);
    assert!((cloned.mu - state.mu).abs() < 1e-6);
}

// =============================================================================
// Mirostat Sampling Tests
// =============================================================================

#[test]
fn test_sample_mirostat_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_mirostat_single_token() {
    let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let mut state = MirostatState::default();
    let result = sample_mirostat(&logits, &mut state, 0.5).expect("test");
    assert_eq!(result, 0);
}

#[test]
fn test_sample_mirostat_low_mu_fallback() {
    // Very low mu should still return at least top candidate
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let mut state = MirostatState::new(0.01); // Very low tau
    state.mu = 0.001; // Extremely low mu
    let result = sample_mirostat(&logits, &mut state, 0.5).expect("test");
    assert!(result < 3);
}

#[test]
fn test_sample_mirostat_updates_state() {
    let logits = Tensor::from_vec(vec![3], vec![1.0, 2.0, 10.0]).expect("test");
    let mut state = MirostatState::default();
    let initial_mu = state.mu;
    let _ = sample_mirostat(&logits, &mut state, 0.5).expect("test");
    assert!((state.mu - initial_mu).abs() > 1e-6);
}

/// PMAT-857 falsifier: Mirostat 2.0 surprise MUST be measured in bits (log2),
/// not nats (ln), per Basu et al. 2021 and llama.cpp
/// `llama_sampler_mirostat_v2_apply` (which uses `-log2f(p)`).
///
/// Setup: logits `[0.0, -1.3863]` -> softmax ~= `[0.80, 0.20]`.
/// `MirostatState::new(1.0)` -> mu = 2*tau = 2.0 (bits).
///
/// Token-1 (prob 0.20) surprise:
///   - bits: `-log2(0.20) ~= 2.3219` > mu=2.0  => TRUNCATED (break)
///   - nats: `-ln(0.20)   ~= 1.6094` < mu=2.0  => KEPT
///
/// With the correct (bits) computation the candidate set is `{token-0}` only,
/// so even with rng_value = 0.99 the selected token is 0. With the buggy
/// (nats) computation token-1 is also a candidate and rng=0.99 selects it.
/// This makes the assertion RED on `ln` and GREEN on `log2`.
#[test]
fn mirostat_truncation_uses_bits_not_nats() {
    // softmax([0.0, -1.3863]) = [e^0, e^-1.3863] / sum = [1.0, 0.25] / 1.25 = [0.8, 0.2]
    let logits = Tensor::from_vec(vec![2], vec![0.0, -1.386_294_4]).expect("test");
    let mut state = MirostatState::new(1.0); // mu = 2.0 bits

    // rng=0.99 deliberately biases toward the LAST candidate. If token-1 were a
    // candidate (nats bug), it would be selected. With bits it is truncated, so
    // the only candidate is token-0 and the result must be 0.
    let result = sample_mirostat(&logits, &mut state, 0.99).expect("test");

    assert_eq!(
        result, 0,
        "Mirostat must measure surprise in bits (-log2 p): token-1 surprise \
         -log2(0.20)=2.32 > mu=2.0 so it must be truncated. A result of 1 means \
         surprise was computed in nats (-ln p)=1.61 < mu, the PMAT-857 bug."
    );

    // Observed surprise of the selected token-0 must also be in bits:
    // -log2(0.8) = 0.3219, so mu update is mu -= eta*(0.3219 - tau=1.0).
    // observed < tau => mu increases above the initial 2.0.
    assert!(
        state.mu > 2.0,
        "mu must increase: observed bits surprise -log2(0.8)=0.32 < tau=1.0; got mu={}",
        state.mu
    );
}

// =============================================================================
// TFS (Tail-Free Sampling) Tests
// =============================================================================

#[test]
fn test_sample_tfs_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_tfs_two_tokens_greedy() {
    // Less than 3 tokens, uses greedy
    let logits = Tensor::from_vec(vec![2], vec![1.0, 5.0]).expect("test");
    let result = sample_tfs(&logits, 0.95, 0.5).expect("test");
    assert_eq!(result, 1);
}

#[test]
fn test_sample_tfs_z_zero_strict() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    // z=0 should be very restrictive
    let result = sample_tfs(&logits, 0.0, 0.0).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_tfs_z_one_permissive() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    // z=1 should include many tokens
    let result = sample_tfs(&logits, 1.0, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_tfs_uniform_distribution() {
    let logits = Tensor::from_vec(vec![5], vec![0.0; 5]).expect("test");
    // Uniform distribution - second derivatives all zero
    let result = sample_tfs(&logits, 0.5, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_tfs_single_dominant() {
    let logits = Tensor::from_vec(vec![5], vec![100.0, 0.0, 0.0, 0.0, 0.0]).expect("test");
    let result = sample_tfs(&logits, 0.95, 0.0).expect("test");
    assert_eq!(result, 0);
}

// =============================================================================
// Typical Sampling Tests
// =============================================================================

#[test]
fn test_sample_typical_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_typical_single_token() {
    let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let result = sample_typical(&logits, 0.95, 0.5).expect("test");
    assert_eq!(result, 0);
}

#[test]
fn test_sample_typical_p_very_small() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    // Very small p should select most typical token(s)
    let result = sample_typical(&logits, 0.01, 0.0).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_typical_p_one() {
    let logits = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
    let result = sample_typical(&logits, 1.0, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_typical_all_zero_entropy() {
    // One token has all the probability
    let logits = Tensor::from_vec(vec![3], vec![100.0, -100.0, -100.0]).expect("test");
    let result = sample_typical(&logits, 0.95, 0.5).expect("test");
    assert_eq!(result, 0);
}

// =============================================================================
// DryConfig Tests
// =============================================================================

#[test]
fn test_dry_config_default() {
    let config = DryConfig::default();
    assert!((config.multiplier - 0.8).abs() < 1e-6);
    assert!((config.base - 1.75).abs() < 1e-6);
    assert_eq!(config.allowed_length, 2);
    assert_eq!(config.penalty_last_n, 256);
    assert!(config.is_enabled());
}

#[test]
fn test_dry_config_new() {
    let config = DryConfig::new(0.5);
    assert!((config.multiplier - 0.5).abs() < 1e-6);
}

#[test]
fn test_dry_config_disabled() {
    let config = DryConfig::new(0.0);
    assert!(!config.is_enabled());
}

#[test]
fn test_dry_config_builders() {
    let config = DryConfig::new(1.0)
        .with_base(2.0)
        .with_allowed_length(3)
        .with_penalty_last_n(128);
    assert!((config.base - 2.0).abs() < 1e-6);
    assert_eq!(config.allowed_length, 3);
    assert_eq!(config.penalty_last_n, 128);
}

// =============================================================================
// DRY Penalty Tests
// =============================================================================

#[test]
fn test_apply_dry_penalty_disabled() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = DryConfig::new(0.0);
    let result = apply_dry_penalty(&logits, &[0, 1, 0, 1], &config);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_dry_penalty_short_context() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = DryConfig::new(1.0).with_allowed_length(5);
    // Context shorter than allowed_length
    let result = apply_dry_penalty(&logits, &[0, 1, 2], &config);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_dry_penalty_window_truncation() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = DryConfig::new(1.0).with_penalty_last_n(3);
    // Long context, but only last 3 tokens used
    let long_context: Vec<usize> = (0..100).collect();
    let result = apply_dry_penalty(&logits, &long_context, &config);
    // Should still work
    assert_eq!(result.data().len(), 5);
}

#[test]
fn test_apply_dry_penalty_repetition_detected() {
    let logits = Tensor::from_vec(vec![5], vec![10.0; 5]).expect("test");
    let config = DryConfig::new(1.0).with_allowed_length(2);
    // Pattern: [0,1] repeats, next token 0 would extend
    let context = vec![0, 1, 0, 1];
    let result = apply_dry_penalty(&logits, &context, &config);
    // Token 0 should be penalized
    assert!(result.data()[0] < 10.0);
}

#[test]
fn test_apply_dry_penalty_no_repetition() {
    let logits = Tensor::from_vec(vec![5], vec![10.0; 5]).expect("test");
    let config = DryConfig::new(1.0).with_allowed_length(2);
    // No repetition pattern
    let context = vec![0, 1, 2, 3];
    let result = apply_dry_penalty(&logits, &context, &config);
    // No penalty should be applied
    for val in result.data() {
        assert!((*val - 10.0).abs() < 1e-6);
    }
}

// =============================================================================
// XtcConfig Tests
// =============================================================================

#[test]
fn test_xtc_config_default() {
    let config = XtcConfig::default();
    assert!((config.probability - 0.0).abs() < 1e-6);
    assert!((config.threshold - 0.5).abs() < 1e-6);
    assert_eq!(config.min_keep, 1);
    assert!(!config.is_enabled());
}

#[test]
fn test_xtc_config_new() {
    let config = XtcConfig::new(0.5);
    assert!((config.probability - 0.5).abs() < 1e-6);
    assert!(config.is_enabled());
}

#[test]
fn test_xtc_config_builders() {
    let config = XtcConfig::new(0.8).with_threshold(0.3).with_min_keep(2);
    assert!((config.threshold - 0.3).abs() < 1e-6);
    assert_eq!(config.min_keep, 2);
}

// =============================================================================
// XTC (Exclude Top Choices) Tests
// =============================================================================

#[test]
fn test_apply_xtc_disabled() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = XtcConfig::default(); // probability = 0
    let result = apply_xtc(&logits, &config, 0.5);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_xtc_rng_above_probability() {
    let logits = Tensor::from_vec(vec![5], vec![1.0; 5]).expect("test");
    let config = XtcConfig::new(0.5); // 50% chance
                                      // rng = 0.6 > 0.5, so no exclusion
    let result = apply_xtc(&logits, &config, 0.6);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_xtc_too_few_tokens() {
    let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let config = XtcConfig::new(1.0).with_min_keep(2);
    // Only 1 token, can't exclude
    let result = apply_xtc(&logits, &config, 0.0);
    assert_eq!(result.data(), logits.data());
}

#[test]
fn test_apply_xtc_excludes_top_token() {
    // PMAT-846: canonical XTC removes the strictly-most-probable above-threshold
    // tokens but KEEPS the boundary (least-probable above-threshold) token. With
    // logits [0,5,6] and threshold 0.05, tokens 1 and 2 are above threshold; XTC
    // removes the top (idx 2) and keeps the boundary (idx 1).
    let logits = Tensor::from_vec(vec![3], vec![0.0, 5.0, 6.0]).expect("test");
    let config = XtcConfig::new(1.0).with_threshold(0.05).with_min_keep(1);
    let result = apply_xtc(&logits, &config, 0.0);
    assert_eq!(result.data()[2], f32::NEG_INFINITY, "top token excluded");
    assert!(result.data()[1].is_finite(), "boundary token kept");
}

#[test]
fn test_apply_xtc_respects_min_keep() {
    let logits = Tensor::from_vec(vec![3], vec![100.0, 100.0, 100.0]).expect("test");
    let config = XtcConfig::new(1.0).with_threshold(0.1).with_min_keep(2);
    let result = apply_xtc(&logits, &config, 0.0);
    // Should keep at least 2 tokens (not NEG_INFINITY)
    let finite_count = result.data().iter().filter(|&&x| x.is_finite()).count();
    assert!(finite_count >= 2);
}

// PMAT-846 falsifier: XTC must KEEP the boundary token (the LAST token whose
// prob >= threshold), matching llama.cpp `llama_sample_xtc_apply`. The buggy
// implementation subtracted the excluded count from the FULL vocab length, so
// with a real vocab + small min_keep it removed the ENTIRE above-threshold set,
// including the boundary token it must preserve.
//
// Repro: logits=[4.0,3.0,2.0,-10.0,-10.0], threshold=0.1, min_keep=1, rng=0.0.
// softmax ~= [0.665, 0.245, 0.090, ~0, ~0]; tokens 0 and 1 are >= 0.1.
// llama.cpp: pos_last = 1 (sorted index of LAST token >= threshold) -> remove
// only sorted index 0 (orig idx 0, the strictly-most-probable), KEEP boundary
// (orig idx 1 = value 3.0) and the whole below-threshold tail.
#[test]
fn test_apply_xtc_keeps_boundary_token_pmat846() {
    let logits = Tensor::from_vec(vec![5], vec![4.0, 3.0, 2.0, -10.0, -10.0]).expect("test");
    let config = XtcConfig::new(1.0).with_threshold(0.1).with_min_keep(1);
    // rng=0.0 < probability=1.0 => XTC fires deterministically.
    let result = apply_xtc(&logits, &config, 0.0);
    let out = result.data();

    // The strictly-most-probable above-threshold token (orig idx 0) is removed.
    assert_eq!(
        out[0],
        f32::NEG_INFINITY,
        "top token (idx 0) must be excluded"
    );
    // The BOUNDARY token (orig idx 1, the LAST above-threshold token) must stay
    // finite and unchanged at 3.0. The buggy impl set this to NEG_INFINITY.
    assert!(
        out[1].is_finite(),
        "boundary token (idx 1) must NOT be excluded, got {}",
        out[1]
    );
    assert!(
        (out[1] - 3.0).abs() < 1e-6,
        "boundary token must be unchanged (3.0), got {}",
        out[1]
    );
    // Below-threshold tail must remain finite (XTC only touches top choices).
    assert!(out[2].is_finite(), "below-threshold token idx 2 must stay");
    assert!(out[3].is_finite(), "below-threshold token idx 3 must stay");
    assert!(out[4].is_finite(), "below-threshold token idx 4 must stay");
}

// PMAT-846: canonical `threshold > 0.5` no-op guard (llama.cpp short-circuits
// when xtc_threshold > 0.5 because at most one token can clear that bar, so
// removing it would violate the keep-the-boundary invariant). Output == input.
#[test]
fn test_apply_xtc_threshold_above_half_is_noop_pmat846() {
    let logits = Tensor::from_vec(vec![5], vec![4.0, 3.0, 2.0, -10.0, -10.0]).expect("test");
    let config = XtcConfig::new(1.0).with_threshold(0.6).with_min_keep(1);
    let result = apply_xtc(&logits, &config, 0.0);
    assert_eq!(
        result.data(),
        logits.data(),
        "threshold > 0.5 must be a no-op (output == input)"
    );
}

// =============================================================================
// EtaConfig Tests
// =============================================================================

#[test]
fn test_eta_config_default() {
    let config = EtaConfig::default();
    assert!((config.eta - 0.3).abs() < 1e-6);
    assert!((config.min_p - 0.0001).abs() < 1e-6);
    assert!(config.is_enabled());
}

#[test]
fn test_eta_config_new() {
    let config = EtaConfig::new(0.5);
    assert!((config.eta - 0.5).abs() < 1e-6);
}

#[test]
fn test_eta_config_disabled() {
    let config = EtaConfig::new(0.0);
    assert!(!config.is_enabled());
}

#[test]
fn test_eta_config_with_min_p() {
    let config = EtaConfig::new(0.5).with_min_p(0.01);
    assert!((config.min_p - 0.01).abs() < 1e-6);
}

include!("algorithms_tests_sample_eta.rs");
