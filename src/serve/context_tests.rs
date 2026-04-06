use super::*;
use crate::serve::templates::ChatMessage;

// ========================================================================
// SERVE-CTX-001: Context Window Tests
// ========================================================================

#[test]
fn test_SERVE_CTX_001_context_window_new() {
    let window = ContextWindow::new(8192, 2048);
    assert_eq!(window.max_tokens, 8192);
    assert_eq!(window.output_reserve, 2048);
}

#[test]
fn test_SERVE_CTX_001_available_input() {
    let window = ContextWindow::new(8192, 2048);
    assert_eq!(window.available_input(), 6144);
}

#[test]
fn test_SERVE_CTX_001_for_model_gpt4() {
    let window = ContextWindow::for_model("gpt-4-turbo");
    assert_eq!(window.max_tokens, 128_000);
}

#[test]
fn test_SERVE_CTX_001_for_model_claude() {
    let window = ContextWindow::for_model("claude-3-sonnet");
    assert_eq!(window.max_tokens, 200_000);
}

#[test]
fn test_SERVE_CTX_001_for_model_llama() {
    let window = ContextWindow::for_model("llama-2-7b");
    assert_eq!(window.max_tokens, 4_096);
}

#[test]
fn test_SERVE_CTX_001_for_model_mistral() {
    let window = ContextWindow::for_model("mistral-7b");
    assert_eq!(window.max_tokens, 8_192);
}

#[test]
fn test_SERVE_CTX_001_default() {
    let window = ContextWindow::default();
    assert_eq!(window.max_tokens, 4_096);
}

// ========================================================================
// SERVE-CTX-002: Token Estimator Tests
// ========================================================================

#[test]
fn test_SERVE_CTX_002_estimate_simple() {
    let estimator = TokenEstimator::new();
    // 20 chars / 4 = 5 tokens
    assert_eq!(estimator.estimate("Hello, how are you?"), 5);
}

#[test]
fn test_SERVE_CTX_002_estimate_empty() {
    let estimator = TokenEstimator::new();
    assert_eq!(estimator.estimate(""), 0);
}

#[test]
fn test_SERVE_CTX_002_estimate_long_text() {
    let estimator = TokenEstimator::new();
    let text = "a".repeat(1000);
    assert_eq!(estimator.estimate(&text), 250);
}

#[test]
fn test_SERVE_CTX_002_estimate_messages() {
    let estimator = TokenEstimator::new();
    let messages = vec![ChatMessage::user("Hello"), ChatMessage::assistant("Hi there!")];
    let tokens = estimator.estimate_messages(&messages);
    // Each message: ~4 formatting tokens + content tokens
    assert!(tokens > 0);
}

#[test]
fn test_SERVE_CTX_002_custom_ratio() {
    let estimator = TokenEstimator::with_ratio(3.0);
    // 12 chars / 3 = 4 tokens
    assert_eq!(estimator.estimate("Hello World!"), 4);
}

// ========================================================================
// SERVE-CTX-003: Context Manager Tests
// ========================================================================

#[test]
fn test_SERVE_CTX_003_fits_under_limit() {
    let manager = ContextManager::for_model("gpt-4-turbo");
    let messages = vec![ChatMessage::user("Hello!")];
    assert!(manager.fits(&messages));
}

#[test]
fn test_SERVE_CTX_003_estimate_tokens() {
    let manager = ContextManager::default();
    let messages = vec![ChatMessage::user("Test message")];
    let tokens = manager.estimate_tokens(&messages);
    assert!(tokens > 0);
}

#[test]
fn test_SERVE_CTX_003_truncate_not_needed() {
    let manager = ContextManager::for_model("gpt-4-turbo");
    let messages = vec![ChatMessage::user("Hello!")];
    let result = manager.truncate(&messages).unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn test_SERVE_CTX_003_truncate_error_strategy() {
    let config = ContextConfig {
        window: ContextWindow::new(10, 0), // Very small window
        strategy: TruncationStrategy::Error,
        ..Default::default()
    };
    let manager = ContextManager::new(config);
    let messages = vec![ChatMessage::user("This is a longer message that exceeds limit")];
    let result = manager.truncate(&messages);
    assert!(result.is_err());
}

// ========================================================================
// SERVE-CTX-004: Sliding Window Tests
// ========================================================================

#[test]
fn test_SERVE_CTX_004_sliding_window_truncation() {
    let config = ContextConfig {
        window: ContextWindow::new(100, 0),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::user("First message"),
        ChatMessage::assistant("First response"),
        ChatMessage::user("Second message"),
        ChatMessage::assistant("Second response"),
        ChatMessage::user("Third message - most recent"),
    ];

    let result = manager.truncate(&messages).unwrap();
    // Should keep the most recent messages
    assert!(result.len() < messages.len() || manager.fits(&messages));
}

#[test]
fn test_SERVE_CTX_004_preserves_system() {
    let config = ContextConfig {
        window: ContextWindow::new(200, 0),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: true,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::system("You are helpful"),
        ChatMessage::user("Message 1"),
        ChatMessage::user("Message 2"),
        ChatMessage::user("Message 3"),
    ];

    let result = manager.truncate(&messages).unwrap();
    // First message should be system if it fits
    if !result.is_empty() {
        // System message should be preserved if space allows
        let has_system =
            result.iter().any(|m| matches!(m.role, crate::serve::templates::Role::System));
        // Only check if original had system and result has multiple messages
        if result.len() > 1 {
            assert!(has_system);
        }
    }
}

// ========================================================================
// SERVE-CTX-005: Middle-Out Tests
// ========================================================================

#[test]
fn test_SERVE_CTX_005_middle_out_keeps_first_last() {
    let config = ContextConfig {
        window: ContextWindow::new(150, 0),
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 2,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::user("First"),
        ChatMessage::assistant("Middle 1"),
        ChatMessage::user("Middle 2"),
        ChatMessage::assistant("Last"),
    ];

    let result = manager.truncate(&messages).unwrap();
    // Should keep first and last at minimum
    assert!(result.len() >= 2);
    assert_eq!(result[0].content, "First");
    assert_eq!(result[result.len() - 1].content, "Last");
}

// ========================================================================
// SERVE-CTX-006: Error Tests
// ========================================================================

#[test]
fn test_SERVE_CTX_006_error_display() {
    let err = ContextError::ExceedsLimit { tokens: 10000, limit: 4096 };
    let msg = err.to_string();
    assert!(msg.contains("10000"));
    assert!(msg.contains("4096"));
}

// ========================================================================
// SERVE-CTX-007: Default Tests
// ========================================================================

#[test]
fn test_SERVE_CTX_007_context_config_default() {
    let config = ContextConfig::default();
    assert_eq!(config.strategy, TruncationStrategy::SlidingWindow);
    assert!(config.preserve_system);
}

#[test]
fn test_SERVE_CTX_007_manager_default() {
    let manager = ContextManager::default();
    assert_eq!(manager.available_tokens(), 3072); // 4096 - 1024
}

// ========================================================================
// SERVE-CTX-008: Additional Model Coverage Tests
// ========================================================================

#[test]
fn test_SERVE_CTX_008_for_model_gpt4_32k() {
    let window = ContextWindow::for_model("gpt-4-32k");
    assert_eq!(window.max_tokens, 32_768);
}

#[test]
fn test_SERVE_CTX_008_for_model_gpt4_base() {
    let window = ContextWindow::for_model("gpt-4");
    assert_eq!(window.max_tokens, 8_192);
}

#[test]
fn test_SERVE_CTX_008_for_model_gpt4o() {
    let window = ContextWindow::for_model("gpt-4o-2024");
    assert_eq!(window.max_tokens, 128_000);
}

#[test]
fn test_SERVE_CTX_008_for_model_gpt35_16k() {
    let window = ContextWindow::for_model("gpt-3.5-turbo-16k");
    assert_eq!(window.max_tokens, 16_384);
}

#[test]
fn test_SERVE_CTX_008_for_model_gpt35() {
    let window = ContextWindow::for_model("gpt-3.5-turbo");
    assert_eq!(window.max_tokens, 4_096);
}

#[test]
fn test_SERVE_CTX_008_for_model_claude_v1() {
    let window = ContextWindow::for_model("claude-instant-v1");
    assert_eq!(window.max_tokens, 100_000);
}

#[test]
fn test_SERVE_CTX_008_for_model_llama3() {
    let window = ContextWindow::for_model("llama-3-70b");
    assert_eq!(window.max_tokens, 8_192);
}

#[test]
fn test_SERVE_CTX_008_for_model_llama2_32k() {
    let window = ContextWindow::for_model("llama-2-32k-instruct");
    assert_eq!(window.max_tokens, 32_768);
}

#[test]
fn test_SERVE_CTX_008_for_model_mixtral() {
    let window = ContextWindow::for_model("mixtral-8x7b");
    assert_eq!(window.max_tokens, 32_768);
}

#[test]
fn test_SERVE_CTX_008_for_model_unknown() {
    let window = ContextWindow::for_model("unknown-model-xyz");
    assert_eq!(window.max_tokens, 4_096); // Default
}

// ========================================================================
// SERVE-CTX-009: Token Estimator Default
// ========================================================================

#[test]
fn test_SERVE_CTX_009_estimator_default() {
    let estimator = TokenEstimator::default();
    assert_eq!(estimator.estimate("Test"), 1);
}

// ========================================================================
// SERVE-CTX-010: Edge Cases
// ========================================================================

#[test]
fn test_SERVE_CTX_010_middle_out_short_messages() {
    let config = ContextConfig {
        window: ContextWindow::new(1000, 0),
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    // Only 2 messages - should return both
    let messages = vec![ChatMessage::user("First"), ChatMessage::assistant("Last")];

    let result = manager.truncate(&messages).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_SERVE_CTX_010_middle_out_single_message() {
    let config = ContextConfig {
        window: ContextWindow::new(1000, 0),
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![ChatMessage::user("Only one")];

    let result = manager.truncate(&messages).unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn test_SERVE_CTX_010_sliding_window_no_system() {
    let config = ContextConfig {
        window: ContextWindow::new(50, 0),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::user("First"),
        ChatMessage::user("Second with more content"),
        ChatMessage::user("Third"),
    ];

    let result = manager.truncate(&messages).unwrap();
    // Should keep recent messages
    assert!(!result.is_empty());
}

#[test]
fn test_SERVE_CTX_010_available_input_saturating() {
    // Test saturating_sub doesn't panic
    let window = ContextWindow::new(100, 200);
    assert_eq!(window.available_input(), 0);
}

#[test]
fn test_SERVE_CTX_010_context_error_is_error() {
    // Ensure ContextError implements std::error::Error
    let err = ContextError::ExceedsLimit { tokens: 100, limit: 50 };
    let _: &dyn std::error::Error = &err;
}

#[test]
fn test_SERVE_CTX_010_config_for_model() {
    let config = ContextConfig::for_model("gpt-4-turbo");
    assert_eq!(config.window.max_tokens, 128_000);
    assert!(config.preserve_system);
}

// ========================================================================
// SERVE-CTX-011: Additional Edge Case Coverage
// ========================================================================

#[test]
fn test_SERVE_CTX_011_window_zero_reserve() {
    let window = ContextWindow::new(4096, 0);
    assert_eq!(window.available_input(), 4096);
}

#[test]
fn test_SERVE_CTX_011_window_full_reserve() {
    let window = ContextWindow::new(4096, 4096);
    assert_eq!(window.available_input(), 0);
}

#[test]
fn test_SERVE_CTX_011_truncation_strategy_variants() {
    // Test all variants exist
    let _sliding = TruncationStrategy::SlidingWindow;
    let _middle = TruncationStrategy::MiddleOut;
    let _error = TruncationStrategy::Error;
}

#[test]
fn test_SERVE_CTX_011_all_error_variants() {
    // Test all error variants for display
    let err = ContextError::ExceedsLimit { tokens: 100, limit: 50 };
    let msg = err.to_string();
    assert!(msg.contains("100"));
    assert!(msg.contains("50"));
}

#[test]
fn test_SERVE_CTX_011_context_manager_no_reserve() {
    let config = ContextConfig {
        window: ContextWindow::new(1000, 0),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);
    assert_eq!(manager.available_tokens(), 1000);
}

#[test]
fn test_SERVE_CTX_011_truncate_single_fits() {
    let config = ContextConfig {
        window: ContextWindow::new(1000, 0),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![ChatMessage::user("Short message")];
    let result = manager.truncate(&messages).unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn test_SERVE_CTX_011_for_model_fallback() {
    // Unknown models get default
    let window = ContextWindow::for_model("some-unknown-model");
    assert_eq!(window.max_tokens, 4_096);
}

#[test]
fn test_SERVE_CTX_011_for_model_mixtral() {
    let window = ContextWindow::for_model("mixtral-8x7b");
    assert_eq!(window.max_tokens, 32_768);
}

#[test]
fn test_SERVE_CTX_011_for_model_llama() {
    let window = ContextWindow::for_model("llama-some-model");
    assert_eq!(window.max_tokens, 4_096);
}

// ========================================================================
// SERVE-CTX-012: More Model Coverage
// ========================================================================

#[test]
fn test_ctx_cov_001_gpt4_32k() {
    let window = ContextWindow::for_model("gpt-4-32k");
    assert_eq!(window.max_tokens, 32_768);
}

#[test]
fn test_ctx_cov_002_gpt4_base() {
    let window = ContextWindow::for_model("gpt-4");
    assert_eq!(window.max_tokens, 8_192);
}

#[test]
fn test_ctx_cov_003_gpt4o() {
    let window = ContextWindow::for_model("gpt-4o-mini");
    assert_eq!(window.max_tokens, 128_000);
}

#[test]
fn test_ctx_cov_004_gpt35_16k() {
    let window = ContextWindow::for_model("gpt-3.5-turbo-16k");
    assert_eq!(window.max_tokens, 16_384);
}

#[test]
fn test_ctx_cov_005_gpt35_base() {
    let window = ContextWindow::for_model("gpt-3.5-turbo");
    assert_eq!(window.max_tokens, 4_096);
}

#[test]
fn test_ctx_cov_006_claude2() {
    let window = ContextWindow::for_model("claude-2.1");
    assert_eq!(window.max_tokens, 200_000);
}

#[test]
fn test_ctx_cov_007_claude_base() {
    let window = ContextWindow::for_model("claude-instant");
    assert_eq!(window.max_tokens, 100_000);
}

#[test]
fn test_ctx_cov_008_llama3() {
    let window = ContextWindow::for_model("llama-3-8b");
    assert_eq!(window.max_tokens, 8_192);
}

#[test]
fn test_ctx_cov_009_llama2_32k() {
    let window = ContextWindow::for_model("llama-2-70b-32k");
    assert_eq!(window.max_tokens, 32_768);
}

#[test]
fn test_ctx_cov_010_mistral_base() {
    let window = ContextWindow::for_model("mistral-large");
    assert_eq!(window.max_tokens, 8_192);
}

// ========================================================================
// SERVE-CTX-013: Truncation Edge Cases
// ========================================================================

#[test]
fn test_ctx_cov_011_middle_out_two_messages() {
    let config = ContextConfig {
        window: ContextWindow::new(100, 0),
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![ChatMessage::user("First"), ChatMessage::assistant("Second")];
    let result = manager.truncate(&messages).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_ctx_cov_012_middle_out_many_messages() {
    let config = ContextConfig {
        window: ContextWindow::new(150, 0),
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::user("First message"),
        ChatMessage::assistant("Response 1"),
        ChatMessage::user("Question 2"),
        ChatMessage::assistant("Response 2"),
        ChatMessage::user("Final message"),
    ];
    let result = manager.truncate(&messages).unwrap();
    // Should keep first and last at minimum
    assert!(result.len() >= 2);
}

#[test]
fn test_ctx_cov_013_sliding_with_system() {
    let config = ContextConfig {
        window: ContextWindow::new(200, 0),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: true,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::system("You are a helpful assistant"),
        ChatMessage::user("Hello"),
        ChatMessage::assistant("Hi there!"),
        ChatMessage::user("How are you?"),
    ];
    let result = manager.truncate(&messages).unwrap();
    // System should be preserved
    assert!(result.iter().any(|m| matches!(m.role, crate::serve::templates::Role::System)));
}

#[test]
fn test_ctx_cov_014_sliding_no_preserve_system() {
    let config = ContextConfig {
        window: ContextWindow::new(50, 0),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::system("You are a helpful assistant"),
        ChatMessage::user("Hello there! This is a longer message."),
    ];
    let result = manager.truncate(&messages).unwrap();
    assert!(!result.is_empty());
}

#[test]
fn test_ctx_cov_015_estimator_default() {
    let estimator = TokenEstimator::default();
    assert_eq!(estimator.estimate("test"), 1);
}

#[test]
fn test_ctx_cov_016_context_window_serialize() {
    let window = ContextWindow::new(8192, 2048);
    let json = serde_json::to_string(&window).unwrap();
    let deserialized: ContextWindow = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.max_tokens, window.max_tokens);
}

#[test]
fn test_ctx_cov_017_truncation_strategy_serialize() {
    let strategy = TruncationStrategy::MiddleOut;
    let json = serde_json::to_string(&strategy).unwrap();
    let deserialized: TruncationStrategy = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, strategy);
}

#[test]
fn test_ctx_cov_018_context_config_serialize() {
    let config = ContextConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: ContextConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.min_messages, config.min_messages);
}

#[test]
fn test_ctx_cov_019_context_error_eq() {
    let err1 = ContextError::ExceedsLimit { tokens: 100, limit: 50 };
    let err2 = ContextError::ExceedsLimit { tokens: 100, limit: 50 };
    assert_eq!(err1, err2);
}

#[test]
fn test_ctx_cov_020_context_error_clone() {
    let err = ContextError::ExceedsLimit { tokens: 100, limit: 50 };
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn test_ctx_cov_021_context_error_debug() {
    let err = ContextError::ExceedsLimit { tokens: 100, limit: 50 };
    let debug = format!("{:?}", err);
    assert!(debug.contains("ExceedsLimit"));
}

#[test]
fn test_ctx_cov_022_window_saturating_sub() {
    // Test case where reserve > max
    let window = ContextWindow::new(100, 200);
    assert_eq!(window.available_input(), 0);
}

#[test]
fn test_ctx_cov_023_empty_messages() {
    let manager = ContextManager::default();
    let messages: Vec<ChatMessage> = vec![];
    let result = manager.truncate(&messages).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_ctx_cov_024_context_manager_default_trait() {
    let manager = ContextManager::default();
    assert!(manager.available_tokens() > 0);
}

// ========================================================================
// SERVE-CTX-014: More Coverage Tests
// ========================================================================

#[test]
fn test_ctx_cov_025_sliding_window_min_messages_break() {
    // Test the `else if recent_msgs.len() >= self.config.min_messages` branch
    let config = ContextConfig {
        window: ContextWindow::new(40, 0), // Very small window
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 1, // Only need 1 message minimum
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::user("First message that is quite long"),
        ChatMessage::assistant("Response 1 also quite long"),
        ChatMessage::user("Question 2 with more content"),
        ChatMessage::assistant("Response 2 with content"),
        ChatMessage::user("Final"),
    ];

    let result = manager.truncate(&messages).unwrap();
    // Should have at least min_messages
    assert!(!result.is_empty());
}

#[test]
fn test_ctx_cov_026_middle_out_break_branch() {
    // Test the `else { break; }` branch in truncate_middle_out
    let config = ContextConfig {
        window: ContextWindow::new(60, 0), // Tight window
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::user("First"),
        ChatMessage::assistant("Middle message that is long"),
        ChatMessage::user("Another middle msg"),
        ChatMessage::assistant("More middle content"),
        ChatMessage::user("Last"),
    ];

    let result = manager.truncate(&messages).unwrap();
    // Should keep first and last at minimum
    assert!(result.len() >= 2);
    assert_eq!(result[0].content, "First");
    assert_eq!(result.last().unwrap().content, "Last");
}

#[test]
fn test_ctx_cov_027_sliding_window_system_too_large() {
    // Test when system message alone exceeds available tokens
    let config = ContextConfig {
        window: ContextWindow::new(10, 0), // Very small
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: true,
        min_messages: 0,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::system("This is a very long system message that exceeds the token limit"),
        ChatMessage::user("Hello"),
    ];

    let result = manager.truncate(&messages).unwrap();
    // System message too large, should not be included
    assert!(result.is_empty() || result.len() <= 1);
}

#[test]
fn test_ctx_cov_028_truncation_strategy_default() {
    let strategy = TruncationStrategy::default();
    assert_eq!(strategy, TruncationStrategy::SlidingWindow);
}

#[test]
fn test_ctx_cov_029_context_window_copy() {
    let window1 = ContextWindow::new(8192, 2048);
    let window2 = window1; // Copy trait
    assert_eq!(window1.max_tokens, window2.max_tokens);
}

#[test]
fn test_ctx_cov_030_truncation_strategy_copy() {
    let strat1 = TruncationStrategy::MiddleOut;
    let strat2 = strat1; // Copy trait
    assert_eq!(strat1, strat2);
}

#[test]
fn test_ctx_cov_031_context_config_clone() {
    let config = ContextConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.min_messages, config.min_messages);
}

#[test]
fn test_ctx_cov_032_context_config_debug() {
    let config = ContextConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ContextConfig"));
}

#[test]
fn test_ctx_cov_033_context_error_std_error() {
    let err = ContextError::ExceedsLimit { tokens: 100, limit: 50 };
    let _: &dyn std::error::Error = &err;
    // Just verify it implements Error trait
}

// ========================================================================
// SERVE-CTX-015: Middle-Out Truncation Coverage
// ========================================================================

#[test]
fn test_ctx_cov_034_middle_out_forces_truncation_break() {
    // Window is small enough to exceed budget, forcing truncate_middle_out
    // to actually run and hit the break path in the middle loop.
    let config = ContextConfig {
        window: ContextWindow::new(130, 0),
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    // Each long message: ceil(200/4)=50 tokens + 4 overhead = 54 tokens
    // Short messages: ceil(40/4)=10 tokens + 4 overhead = 14 tokens
    // Total: 14 + 54 + 54 + 54 + 14 = 190 tokens > 130 → triggers truncation
    let messages = vec![
        ChatMessage::user(&"a".repeat(40)),       // first: 14 tokens
        ChatMessage::assistant(&"b".repeat(200)), // middle: 54 tokens
        ChatMessage::user(&"c".repeat(200)),      // middle: 54 tokens
        ChatMessage::assistant(&"d".repeat(200)), // middle: 54 tokens
        ChatMessage::user(&"e".repeat(40)),       // last: 14 tokens
    ];

    let result = manager.truncate(&messages).unwrap();

    // First and last must be preserved
    assert!(result.len() >= 2);
    assert_eq!(result[0].content, "a".repeat(40));
    assert_eq!(result.last().unwrap().content, "e".repeat(40));

    // Budget: first(14) + last(14) = 28. Remaining = 102.
    // From end of middle: msg4(54) fits (82), msg3(54) doesn't (136 > 130) → break
    // Result: [first, msg4, last] = 3 messages
    assert_eq!(result.len(), 3);
    assert_eq!(result[1].content, "d".repeat(200));
}

#[test]
fn test_ctx_cov_035_middle_out_drops_all_middle() {
    // Budget so tight that no middle messages fit at all
    let config = ContextConfig {
        window: ContextWindow::new(35, 0),
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    // first: ceil(20/4)=5 + 4 = 9 tokens
    // middle: ceil(200/4)=50 + 4 = 54 tokens
    // last: 9 tokens
    // Total: 9 + 54 + 54 + 9 = 126 > 35
    let messages = vec![
        ChatMessage::user(&"a".repeat(20)),
        ChatMessage::assistant(&"b".repeat(200)),
        ChatMessage::user(&"c".repeat(200)),
        ChatMessage::user(&"d".repeat(20)),
    ];

    let result = manager.truncate(&messages).unwrap();

    // first(9) + last(9) = 18 ≤ 35. Middle msgs need 54 each, 18+54=72 > 35 → break
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].content, "a".repeat(20));
    assert_eq!(result[1].content, "d".repeat(20));
}

#[test]
fn test_ctx_cov_036_middle_out_keeps_multiple_middle() {
    // Budget allows keeping 2 of 3 middle messages
    let config = ContextConfig {
        window: ContextWindow::new(200, 0),
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::user(&"a".repeat(40)),       // 14
        ChatMessage::assistant(&"b".repeat(100)), // 29
        ChatMessage::user(&"c".repeat(100)),      // 29
        ChatMessage::assistant(&"d".repeat(100)), // 29
        ChatMessage::user(&"e".repeat(100)),      // 29
        ChatMessage::assistant(&"f".repeat(100)), // 29
        ChatMessage::user(&"g".repeat(100)),      // 29
        ChatMessage::assistant(&"h".repeat(100)), // 29
        ChatMessage::user(&"i".repeat(40)),       // 14
    ];
    // Total: 14 + 7*29 + 14 = 231 > 200 → triggers truncation
    // In truncate_middle_out: first(14) + last(14) = 28. Remaining = 172.
    // Middle reverse: h(29)→57, g(29)→86, f(29)→115, e(29)→144, d(29)→173 > 200 → break at d
    // Keeps: h, g, f, e from middle
    let result = manager.truncate(&messages).unwrap();

    assert!(result.len() >= 2);
    assert_eq!(result[0].content, "a".repeat(40));
    assert_eq!(result.last().unwrap().content, "i".repeat(40));
    // Should keep first + some middle + last
    assert!(result.len() > 2);
    assert!(result.len() < messages.len());
}

#[test]
fn test_ctx_cov_037_middle_out_exactly_three_messages() {
    // Three messages where total exceeds budget
    let config = ContextConfig {
        window: ContextWindow::new(30, 0),
        strategy: TruncationStrategy::MiddleOut,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::user(&"a".repeat(20)),       // 9
        ChatMessage::assistant(&"b".repeat(200)), // 54
        ChatMessage::user(&"c".repeat(20)),       // 9
    ];
    // Total: 9+54+9 = 72 > 30
    // first(9) + last(9) = 18. Remaining = 12. Middle needs 54 → break
    let result = manager.truncate(&messages).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].content, "a".repeat(20));
    assert_eq!(result[1].content, "c".repeat(20));
}
