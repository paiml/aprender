//! Contract: tokenizer-v1 enforcement tests (PMAT-189)
//!
//! Falsification tests for batuta's token estimation and context
//! window management. Tests determinism, empty input, monotonicity,
//! truncation correctness, and error strategy enforcement.

use super::*;
use crate::serve::templates::ChatMessage;

#[test]
fn falsify_tok_001_deterministic_encode() {
    // FALSIFY-TOK-001: Same text always produces same token estimate.
    let estimator = TokenEstimator::new();
    let text = "Hello, world! This is a test of the tokenizer contract.";
    let first = estimator.estimate(text);
    for _ in 0..100 {
        assert_eq!(
            estimator.estimate(text),
            first,
            "FALSIFY-TOK-001: estimate must be deterministic"
        );
    }
}

#[test]
fn falsify_tok_002_empty_input_zero_tokens() {
    // FALSIFY-TOK-002: Empty string produces zero tokens.
    let estimator = TokenEstimator::new();
    assert_eq!(estimator.estimate(""), 0, "FALSIFY-TOK-002: empty input must produce 0 tokens");
}

#[test]
fn falsify_tok_003_positive_for_nonempty() {
    // FALSIFY-TOK-003: Non-empty text always produces > 0 tokens.
    let estimator = TokenEstimator::new();
    for text in &["a", "hello", "fn main() {}", "\n", " ", "\t"] {
        assert!(
            estimator.estimate(text) > 0,
            "FALSIFY-TOK-003: non-empty text '{}' must produce > 0 tokens",
            text
        );
    }
}

#[test]
fn falsify_tok_004_monotonic_with_length() {
    // FALSIFY-TOK-004: Longer text produces >= token estimate of shorter text.
    let estimator = TokenEstimator::new();
    let short = "Hello";
    let long = "Hello, world! This is a much longer piece of text for testing.";
    assert!(
        estimator.estimate(long) >= estimator.estimate(short),
        "FALSIFY-TOK-004: longer text must produce >= tokens"
    );
}

#[test]
fn falsify_tok_005_message_overhead() {
    // FALSIFY-TOK-005: Message estimation adds overhead per message.
    let estimator = TokenEstimator::new();
    let single = vec![ChatMessage::user("Hello")];
    let double = vec![ChatMessage::user("Hello"), ChatMessage::assistant("Hi")];
    assert!(
        estimator.estimate_messages(&double) > estimator.estimate_messages(&single),
        "FALSIFY-TOK-005: more messages must produce more tokens (formatting overhead)"
    );
}

#[test]
fn falsify_tok_006_context_window_truncation_fits() {
    // FALSIFY-TOK-006: After truncation, messages always fit within window.
    let config = ContextConfig {
        window: ContextWindow::new(100, 0),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![
        ChatMessage::user("This is message one with some content"),
        ChatMessage::assistant("Response one with content too"),
        ChatMessage::user("Message two here"),
        ChatMessage::assistant("Response two also"),
        ChatMessage::user("Final message"),
    ];

    let result = manager.truncate(&messages).unwrap();
    assert!(
        manager.fits(&result),
        "FALSIFY-TOK-006: truncated messages must always fit within context window"
    );
}

#[test]
fn falsify_tok_007_error_strategy_refuses_truncation() {
    // FALSIFY-TOK-007: Error strategy returns Err, never silently truncates.
    let config = ContextConfig {
        window: ContextWindow::new(10, 0),
        strategy: TruncationStrategy::Error,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![ChatMessage::user("This text is definitely longer than ten tokens")];
    let result = manager.truncate(&messages);
    assert!(
        result.is_err(),
        "FALSIFY-TOK-007: Error strategy must return Err when truncation needed"
    );
}

#[test]
fn falsify_tok_008_fitting_messages_unchanged() {
    // FALSIFY-TOK-008: Messages that already fit pass through unchanged.
    let config = ContextConfig {
        window: ContextWindow::new(10000, 0),
        strategy: TruncationStrategy::SlidingWindow,
        preserve_system: false,
        min_messages: 1,
    };
    let manager = ContextManager::new(config);

    let messages = vec![ChatMessage::user("Short"), ChatMessage::assistant("Also short")];
    let result = manager.truncate(&messages).unwrap();
    assert_eq!(
        result.len(),
        messages.len(),
        "FALSIFY-TOK-008: fitting messages must pass through unchanged"
    );
    assert_eq!(result[0].content, messages[0].content);
    assert_eq!(result[1].content, messages[1].content);
}

#[test]
fn falsify_tok_009_token_count_monotonicity() {
    // FALSIFY-TOK-009: Adding messages never decreases total token count.
    let estimator = TokenEstimator::new();
    let mut messages = vec![ChatMessage::user("Hello")];
    let mut prev_count = estimator.estimate_messages(&messages);

    for content in &["World", "How are you?", "I'm fine thanks", "Good to hear"] {
        messages.push(ChatMessage::assistant(*content));
        let new_count = estimator.estimate_messages(&messages);
        assert!(
            new_count >= prev_count,
            "FALSIFY-TOK-009: token count must be monotonically non-decreasing"
        );
        prev_count = new_count;
    }
}
