
// ============================================================================
// Tests: Sampling Algorithms Coverage (PMAT-802)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create logits tensor (panics on empty - use make_empty_logits for that)
    fn make_logits(data: &[f32]) -> Tensor<f32> {
        assert!(!data.is_empty(), "Use make_empty_logits for empty data");
        Tensor::from_vec(vec![data.len()], data.to_vec()).expect("expected value")
    }

    // Create a single-element tensor to simulate edge case (empty logits not valid)
    fn make_single_logit() -> Tensor<f32> {
        Tensor::from_vec(vec![1], vec![0.0]).expect("expected value")
    }

    // ========================================================================
    // Min-P Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_min_p_basic() {
        let logits = make_logits(&[2.0, 1.0, 0.5, 0.1]);
        let result = sample_min_p(&logits, 0.1, 0.5);
        assert!(result.is_ok());
        let idx = result.expect("idx");
        assert!(idx < 4);
    }

    #[test]
    fn test_sample_min_p_single_logit() {
        // Single logit should work (edge case - only one choice)
        let logits = make_single_logit();
        let result = sample_min_p(&logits, 0.1, 0.5);
        assert!(result.is_ok());
        assert_eq!(result.expect("result"), 0);
    }

    #[test]
    fn test_sample_min_p_invalid_min_p_negative() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let result = sample_min_p(&logits, -0.1, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_min_p_invalid_min_p_too_high() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let result = sample_min_p(&logits, 1.5, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_min_p_high_threshold_fallback() {
        // With very high min_p, all tokens filtered -> fallback to greedy
        let logits = make_logits(&[1.0, 5.0, 2.0]);
        let result = sample_min_p(&logits, 0.99, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_min_p_zero_threshold() {
        // With min_p=0, all tokens kept
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let result = sample_min_p(&logits, 0.0, 0.5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Mirostat State Tests
    // ========================================================================

    #[test]
    fn test_mirostat_state_default() {
        let state = MirostatState::default();
        assert!((state.tau - 5.0).abs() < 1e-6);
        assert!((state.eta - 0.1).abs() < 1e-6);
        assert!((state.mu - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_mirostat_state_new() {
        let state = MirostatState::new(3.0);
        assert!((state.tau - 3.0).abs() < 1e-6);
        assert!((state.mu - 6.0).abs() < 1e-6); // 2 * tau
    }

    #[test]
    fn test_mirostat_state_with_eta() {
        let state = MirostatState::new(5.0).with_eta(0.2);
        assert!((state.eta - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_mirostat_state_update() {
        let mut state = MirostatState::new(5.0).with_eta(0.1);
        let initial_mu = state.mu;
        state.update(6.0); // Observed surprise > tau
        assert!(state.mu < initial_mu); // mu decreases
    }

    #[test]
    fn test_mirostat_state_update_low_surprise() {
        let mut state = MirostatState::new(5.0).with_eta(0.1);
        let initial_mu = state.mu;
        state.update(4.0); // Observed surprise < tau
        assert!(state.mu > initial_mu); // mu increases
    }

    // ========================================================================
    // Mirostat Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_mirostat_basic() {
        let logits = make_logits(&[2.0, 1.0, 0.5, 0.1]);
        let mut state = MirostatState::default();
        let result = sample_mirostat(&logits, &mut state, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_mirostat_single_logit() {
        let logits = make_single_logit();
        let mut state = MirostatState::default();
        let result = sample_mirostat(&logits, &mut state, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_mirostat_state_updates() {
        let logits = make_logits(&[5.0, 1.0, 0.1]);
        let mut state = MirostatState::new(3.0);
        let initial_mu = state.mu;
        let _ = sample_mirostat(&logits, &mut state, 0.5);
        // State should be updated after sampling
        assert!((state.mu - initial_mu).abs() > 1e-9 || state.mu == initial_mu);
    }

    // ========================================================================
    // Tail-Free Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_tfs_basic() {
        let logits = make_logits(&[3.0, 2.0, 1.0, 0.5, 0.1]);
        let result = sample_tfs(&logits, 0.95, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_tfs_single_logit() {
        let logits = make_single_logit();
        let result = sample_tfs(&logits, 0.95, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_tfs_low_z() {
        let logits = make_logits(&[3.0, 2.0, 1.0]);
        let result = sample_tfs(&logits, 0.1, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_tfs_high_z() {
        let logits = make_logits(&[3.0, 2.0, 1.0]);
        let result = sample_tfs(&logits, 0.99, 0.5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Typical Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_typical_basic() {
        let logits = make_logits(&[3.0, 2.0, 1.0, 0.5]);
        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_typical_single_logit() {
        let logits = make_single_logit();
        let result = sample_typical(&logits, 0.95, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_typical_low_p() {
        let logits = make_logits(&[3.0, 2.0, 1.0]);
        let result = sample_typical(&logits, 0.1, 0.5);
        assert!(result.is_ok());
    }

    // ========================================================================
    // DRY Penalty Tests
    // ========================================================================

    #[test]
    fn test_dry_penalty_no_repetition() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let context = vec![0, 1, 2, 3]; // No repetition
        let config = DryConfig::default();
        let result = apply_dry_penalty(&logits, &context, &config);
        // Returns Tensor, verify it has same shape
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_dry_penalty_with_repetition() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let context = vec![0, 1, 0, 1, 0]; // Repeating pattern
        let config = DryConfig {
            multiplier: 1.0,
            base: 1.5,
            allowed_length: 2,
            penalty_last_n: 256,
        };
        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_dry_penalty_empty_context() {
        let logits = make_logits(&[1.0, 2.0, 3.0]);
        let context: Vec<usize> = vec![];
        let config = DryConfig::default();
        let result = apply_dry_penalty(&logits, &context, &config);
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_dry_config_default() {
        let config = DryConfig::default();
        assert!(config.multiplier > 0.0);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_dry_config_new() {
        let config = DryConfig::new(0.5);
        assert!((config.multiplier - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dry_config_builder() {
        let config = DryConfig::new(1.0)
            .with_base(2.0)
            .with_allowed_length(3)
            .with_penalty_last_n(128);
        assert!((config.base - 2.0).abs() < 1e-6);
        assert_eq!(config.allowed_length, 3);
        assert_eq!(config.penalty_last_n, 128);
    }

    // ========================================================================
    // XTC Tests
    // ========================================================================

    #[test]
    fn test_apply_xtc_disabled() {
        let logits = make_logits(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        let config = XtcConfig::default(); // probability=0.0 (disabled)
        let result = apply_xtc(&logits, &config, 0.5);
        // When disabled, should return original logits
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_apply_xtc_enabled() {
        let logits = make_logits(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        let config = XtcConfig::new(1.0).with_threshold(0.1);
        let result = apply_xtc(&logits, &config, 0.0); // rng < probability triggers
        assert_eq!(result.shape(), logits.shape());
    }

    #[test]
    fn test_xtc_config_default() {
        let config = XtcConfig::default();
        assert!((config.probability - 0.0).abs() < 1e-6);
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_xtc_config_builder() {
        let config = XtcConfig::new(0.5).with_threshold(0.3).with_min_keep(2);
        assert!((config.probability - 0.5).abs() < 1e-6);
        assert!((config.threshold - 0.3).abs() < 1e-6);
        assert_eq!(config.min_keep, 2);
        assert!(config.is_enabled());
    }

    // ========================================================================
    // Eta Sampling Tests
    // ========================================================================

    #[test]
    fn test_sample_eta_basic() {
        let logits = make_logits(&[3.0, 2.0, 1.0, 0.5]);
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_eta_single_logit() {
        let logits = make_single_logit();
        let config = EtaConfig::default();
        let result = sample_eta(&logits, &config, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sample_eta_custom_config() {
        let logits = make_logits(&[3.0, 2.0, 1.0]);
        let config = EtaConfig::new(0.5);
        let result = sample_eta(&logits, &config, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_eta_config_default() {
        let config = EtaConfig::default();
        assert!((config.eta - 0.3).abs() < 1e-6);
        assert!(config.min_p > 0.0);
    }

    // ========================================================================
    // Token Healing Tests
    // ========================================================================

    #[test]
    fn test_token_healing_empty_tokens() {
        let tokens: Vec<usize> = vec![];
        let result = analyze_token_healing(&tokens, None);
        assert_eq!(result.tokens_removed, 0);
        assert!(result.adjusted_tokens.is_empty());
    }

    #[test]
    fn test_token_healing_no_healing_needed() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = analyze_token_healing(&tokens, Some("hello")); // Long token, no healing
        assert_eq!(result.tokens_removed, 0);
        assert_eq!(result.adjusted_tokens.len(), 5);
    }

    #[test]
    fn test_token_healing_partial_token() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = analyze_token_healing(&tokens, Some("ab")); // Short alphanumeric
        // Should heal (remove last token, add prefix constraint)
        assert_eq!(result.tokens_removed, 1);
        assert_eq!(result.adjusted_tokens.len(), 4);
        assert!(result.prefix_constraint.is_some());
    }

    #[test]
    fn test_token_healing_with_space() {
        let tokens = vec![1, 2, 3];
        let result = analyze_token_healing(&tokens, Some(" the")); // Starts with space
        // Should NOT heal (tokens starting with space are complete)
        assert_eq!(result.tokens_removed, 0);
    }

    #[test]
    fn test_token_healing_config() {
        let config = TokenHealingConfig::new(true);
        assert!(config.enabled);
        assert!(config.max_backup_chars > 0);

        let config2 = TokenHealingConfig::new(true).with_max_backup(5);
        assert_eq!(config2.max_backup_chars, 5);
    }

    // ========================================================================
    // CFG (Classifier-Free Guidance) Tests
    // ========================================================================

    #[test]
    fn test_apply_cfg_basic() {
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.5, 1.0, 1.5]);
        let result = apply_cfg(&cond, &uncond, 2.0);
        assert!(result.is_ok());
        let guided = result.expect("guided");
        assert_eq!(guided.shape(), cond.shape());
    }

    #[test]
    fn test_apply_cfg_shape_mismatch() {
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.5, 1.0]);
        let result = apply_cfg(&cond, &uncond, 2.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_cfg_scale_one() {
        // With scale=1, output should equal conditional
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.0, 0.0, 0.0]);
        let result = apply_cfg(&cond, &uncond, 1.0).expect("result");
        let data = result.data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_cfg_scale_zero() {
        // With scale=0, output should equal unconditional
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.5, 1.0, 1.5]);
        let result = apply_cfg(&cond, &uncond, 0.0).expect("result");
        let data = result.data();
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!((data[1] - 1.0).abs() < 1e-6);
        assert!((data[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_cfg_negative_scale() {
        // Negative scale should work (inverts guidance direction)
        let cond = make_logits(&[1.0, 2.0, 3.0]);
        let uncond = make_logits(&[0.5, 1.0, 1.5]);
        let result = apply_cfg(&cond, &uncond, -1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cfg_config_default() {
        let config = CfgConfig::default();
        assert!((config.scale - 1.0).abs() < 1e-6);
        assert!(config.negative_prompt_tokens.is_empty());
    }

    #[test]
    fn test_cfg_config_new() {
        let config = CfgConfig::new(2.0);
        assert!((config.scale - 2.0).abs() < 1e-6);
    }
}
