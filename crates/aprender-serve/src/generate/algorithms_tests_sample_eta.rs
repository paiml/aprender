
// =============================================================================
// Eta Sampling Tests
// =============================================================================

#[test]
fn test_sample_eta_empty_logits() {
    // Zero-dimension tensors are rejected at creation time
    let result = Tensor::<f32>::from_vec(vec![0], vec![]);
    assert!(result.is_err());
}

#[test]
fn test_sample_eta_single_token() {
    let logits = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let config = EtaConfig::default();
    let result = sample_eta(&logits, &config, 0.5).expect("test");
    assert_eq!(result, 0);
}

#[test]
fn test_sample_eta_high_entropy() {
    // Uniform distribution has high entropy
    let logits = Tensor::from_vec(vec![5], vec![0.0; 5]).expect("test");
    let config = EtaConfig::new(0.5);
    let result = sample_eta(&logits, &config, 0.5).expect("test");
    assert!(result < 5);
}

#[test]
fn test_sample_eta_low_entropy_fallback() {
    // Single dominant token
    let logits = Tensor::from_vec(vec![3], vec![100.0, -100.0, -100.0]).expect("test");
    let config = EtaConfig::new(0.5);
    let result = sample_eta(&logits, &config, 0.5).expect("test");
    assert_eq!(result, 0);
}

// =============================================================================
// TokenHealingConfig Tests
// =============================================================================

#[test]
fn test_token_healing_config_default() {
    let config = TokenHealingConfig::default();
    assert!(!config.enabled);
    assert_eq!(config.max_backup_chars, 0);
}

#[test]
fn test_token_healing_config_new() {
    let config = TokenHealingConfig::new(true);
    assert!(config.enabled);
    assert_eq!(config.max_backup_chars, 10);
}

#[test]
fn test_token_healing_config_with_max_backup() {
    let config = TokenHealingConfig::new(true).with_max_backup(20);
    assert_eq!(config.max_backup_chars, 20);
}

// =============================================================================
// Token Healing Analysis Tests
// =============================================================================

#[test]
fn test_analyze_token_healing_no_healing_needed() {
    let tokens = vec![1, 2, 3];
    let result = analyze_token_healing(&tokens, Some(" complete"));
    assert_eq!(result.tokens_removed, 0);
    assert!(result.prefix_constraint.is_none());
    assert_eq!(result.adjusted_tokens, tokens);
}

#[test]
fn test_analyze_token_healing_partial_token() {
    let tokens = vec![1, 2, 3];
    // Short alphanumeric without space = partial
    let result = analyze_token_healing(&tokens, Some("wo"));
    assert_eq!(result.tokens_removed, 1);
    assert_eq!(result.prefix_constraint, Some("wo".to_string()));
    assert_eq!(result.adjusted_tokens, vec![1, 2]);
}

#[test]
fn test_analyze_token_healing_empty_tokens() {
    let tokens: Vec<usize> = vec![];
    let result = analyze_token_healing(&tokens, Some("a"));
    assert_eq!(result.tokens_removed, 0);
    assert_eq!(result.adjusted_tokens.len(), 0);
}

#[test]
fn test_analyze_token_healing_none_text() {
    let tokens = vec![1, 2, 3];
    let result = analyze_token_healing(&tokens, None);
    assert_eq!(result.tokens_removed, 0);
    assert!(result.prefix_constraint.is_none());
}

#[test]
fn test_analyze_token_healing_long_text_no_heal() {
    let tokens = vec![1, 2, 3];
    // More than 3 chars = not partial
    let result = analyze_token_healing(&tokens, Some("word"));
    assert_eq!(result.tokens_removed, 0);
}

#[test]
fn test_analyze_token_healing_non_alphanumeric() {
    let tokens = vec![1, 2, 3];
    // Contains non-alphanumeric
    let result = analyze_token_healing(&tokens, Some("a!"));
    assert_eq!(result.tokens_removed, 0);
}

// =============================================================================
// CfgConfig Tests
// =============================================================================

#[test]
fn test_cfg_config_default() {
    let config = CfgConfig::default();
    assert!((config.scale - 1.0).abs() < 1e-6);
    assert!(config.negative_prompt_tokens.is_empty());
    assert!(!config.is_enabled());
}

#[test]
fn test_cfg_config_new() {
    let config = CfgConfig::new(2.0);
    assert!((config.scale - 2.0).abs() < 1e-6);
    assert!(config.is_enabled());
}

#[test]
fn test_cfg_config_boundary_not_enabled() {
    let config = CfgConfig::new(1.0);
    assert!(!config.is_enabled());
}

#[test]
fn test_cfg_config_with_negative_prompt() {
    let config = CfgConfig::new(1.5).with_negative_prompt(vec![1, 2, 3]);
    assert_eq!(config.negative_prompt_tokens, vec![1, 2, 3]);
}

// =============================================================================
// CFG (Classifier-Free Guidance) Tests
// =============================================================================

#[test]
fn test_apply_cfg_shape_mismatch() {
    let cond = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let uncond = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    let result = apply_cfg(&cond, &uncond, 1.5);
    assert!(result.is_err());
}

#[test]
fn test_apply_cfg_scale_one_returns_conditional() {
    let cond = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let uncond = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).expect("test");
    let result = apply_cfg(&cond, &uncond, 1.0).expect("test");
    for (i, val) in result.data().iter().enumerate() {
        assert!((*val - cond.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_apply_cfg_scale_zero_returns_unconditional() {
    let cond = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    let uncond = Tensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]).expect("test");
    let result = apply_cfg(&cond, &uncond, 0.0).expect("test");
    for (i, val) in result.data().iter().enumerate() {
        assert!((*val - uncond.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_apply_cfg_amplifies_difference() {
    let cond = Tensor::from_vec(vec![3], vec![2.0, 4.0, 6.0]).expect("test");
    let uncond = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
    // scale=2: uncond + 2*(cond - uncond) = uncond + 2*diff
    // = [1,2,3] + 2*[1,2,3] = [3,6,9]
    let result = apply_cfg(&cond, &uncond, 2.0).expect("test");
    assert!((result.data()[0] - 3.0).abs() < 1e-6);
    assert!((result.data()[1] - 6.0).abs() < 1e-6);
    assert!((result.data()[2] - 9.0).abs() < 1e-6);
}

#[test]
fn test_apply_cfg_negative_scale() {
    // Negative scale reverses direction
    let cond = Tensor::from_vec(vec![2], vec![5.0, 10.0]).expect("test");
    let uncond = Tensor::from_vec(vec![2], vec![0.0, 0.0]).expect("test");
    let result = apply_cfg(&cond, &uncond, -1.0).expect("test");
    // uncond + (-1)*(cond - uncond) = -cond + 2*uncond = [-5, -10]
    assert!((result.data()[0] - (-5.0)).abs() < 1e-6);
    assert!((result.data()[1] - (-10.0)).abs() < 1e-6);
}

// =============================================================================
// Serialization Tests (DryConfig, XtcConfig, EtaConfig, CfgConfig)
// =============================================================================

#[test]
fn test_dry_config_debug() {
    let config = DryConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("multiplier"));
}

#[test]
fn test_xtc_config_debug() {
    let config = XtcConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("probability"));
}

#[test]
fn test_eta_config_debug() {
    let config = EtaConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("eta"));
}

#[test]
fn test_cfg_config_debug() {
    let config = CfgConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("scale"));
}

#[test]
fn test_mirostat_state_debug() {
    let state = MirostatState::default();
    let debug_str = format!("{:?}", state);
    assert!(debug_str.contains("tau"));
}

#[test]
fn test_token_healing_result_debug() {
    let result = TokenHealingResult {
        adjusted_tokens: vec![1, 2],
        prefix_constraint: Some("test".to_string()),
        tokens_removed: 1,
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("adjusted_tokens"));
}
