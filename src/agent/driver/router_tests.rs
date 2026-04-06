//! Tests for RoutingDriver (local-first, remote fallback).

use super::*;
use crate::agent::driver::mock::MockDriver;
use crate::agent::driver::{CompletionRequest, LlmDriver};
use crate::agent::result::{AgentError, DriverError, StopReason, TokenUsage};
use crate::serve::backends::PrivacyTier;

/// A driver that always fails with InferenceFailed.
struct FailingDriver;

#[async_trait]
impl LlmDriver for FailingDriver {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError> {
        Err(AgentError::Driver(DriverError::InferenceFailed("primary failed".into())))
    }

    fn context_window(&self) -> usize {
        4096
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Sovereign
    }
}

/// A driver that fails with a non-retryable error.
struct CircuitBreakDriver;

#[async_trait]
impl LlmDriver for CircuitBreakDriver {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError> {
        Err(AgentError::CircuitBreak("budget exhausted".into()))
    }

    fn context_window(&self) -> usize {
        4096
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Sovereign
    }
}

/// A mock driver that tracks privacy tier.
struct StandardDriver;

#[async_trait]
impl LlmDriver for StandardDriver {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError> {
        Ok(CompletionResponse {
            text: "from standard".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens: 5, output_tokens: 3 },
        })
    }

    fn context_window(&self) -> usize {
        8192
    }

    fn privacy_tier(&self) -> PrivacyTier {
        PrivacyTier::Standard
    }
}

fn test_request() -> CompletionRequest {
    CompletionRequest {
        model: "test".into(),
        messages: vec![],
        tools: vec![],
        max_tokens: 100,
        temperature: 0.5,
        system: None,
    }
}

#[tokio::test]
async fn test_primary_succeeds_no_fallback() {
    let primary = MockDriver::single_response("primary ok");
    let fallback = MockDriver::single_response("fallback ok");
    let driver = RoutingDriver::new(Box::new(primary), Box::new(fallback));

    let resp = driver.complete(test_request()).await.expect("complete");
    assert_eq!(resp.text, "primary ok");
    assert_eq!(driver.metrics().primary_attempts(), 1);
    assert_eq!(driver.metrics().spillover_count(), 0);
}

#[tokio::test]
async fn test_primary_fails_fallback_succeeds() {
    let fallback = MockDriver::single_response("fallback ok");
    let driver = RoutingDriver::new(Box::new(FailingDriver), Box::new(fallback));

    let resp = driver.complete(test_request()).await.expect("complete");
    assert_eq!(resp.text, "fallback ok");
    assert_eq!(driver.metrics().spillover_count(), 1);
}

#[tokio::test]
async fn test_primary_fails_fallback_fails() {
    let driver = RoutingDriver::new(Box::new(FailingDriver), Box::new(FailingDriver));

    let result = driver.complete(test_request()).await;
    assert!(result.is_err());
    assert_eq!(driver.metrics().spillover_count(), 1);
}

#[tokio::test]
async fn test_non_retryable_error_skips_fallback() {
    let fallback = MockDriver::single_response("fallback ok");
    let driver = RoutingDriver::new(Box::new(CircuitBreakDriver), Box::new(fallback));

    let result = driver.complete(test_request()).await;
    assert!(result.is_err());
    // CircuitBreak is not retryable — fallback not attempted
    assert_eq!(driver.metrics().spillover_count(), 0);
}

#[tokio::test]
async fn test_primary_only_strategy() {
    let driver = RoutingDriver::new(
        Box::new(FailingDriver),
        Box::new(MockDriver::single_response("fallback")),
    )
    .with_strategy(RoutingStrategy::PrimaryOnly);

    let result = driver.complete(test_request()).await;
    assert!(result.is_err());
    // Should NOT fallback
    assert_eq!(driver.metrics().spillover_count(), 0);
}

#[tokio::test]
async fn test_fallback_only_strategy() {
    let driver = RoutingDriver::new(
        Box::new(FailingDriver),
        Box::new(MockDriver::single_response("fallback ok")),
    )
    .with_strategy(RoutingStrategy::FallbackOnly);

    let resp = driver.complete(test_request()).await.expect("complete");
    assert_eq!(resp.text, "fallback ok");
    // Primary never attempted
    assert_eq!(driver.metrics().primary_attempts(), 0);
}

#[tokio::test]
async fn test_fallback_only_no_fallback_configured() {
    let driver = RoutingDriver::primary_only(Box::new(MockDriver::single_response("primary")))
        .with_strategy(RoutingStrategy::FallbackOnly);

    let result = driver.complete(test_request()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_privacy_tier_inherits_most_permissive() {
    // Sovereign primary + Standard fallback → Standard
    let driver = RoutingDriver::new(
        Box::new(MockDriver::single_response("primary")),
        Box::new(StandardDriver),
    );
    assert_eq!(driver.privacy_tier(), PrivacyTier::Standard);
}

#[tokio::test]
async fn test_privacy_tier_both_sovereign() {
    let driver = RoutingDriver::new(
        Box::new(MockDriver::single_response("a")),
        Box::new(MockDriver::single_response("b")),
    );
    // MockDriver is Sovereign, so both sovereign
    assert_eq!(driver.privacy_tier(), PrivacyTier::Sovereign);
}

#[tokio::test]
async fn test_privacy_tier_primary_only() {
    let driver = RoutingDriver::primary_only(Box::new(MockDriver::single_response("primary")));
    assert_eq!(driver.privacy_tier(), PrivacyTier::Sovereign);
}

#[tokio::test]
async fn test_context_window_uses_primary() {
    let primary = MockDriver::single_response("a").with_context_window(8192);
    let fallback = MockDriver::single_response("b").with_context_window(16384);
    let driver = RoutingDriver::new(Box::new(primary), Box::new(fallback));

    assert_eq!(driver.context_window(), 8192);
}

#[tokio::test]
async fn test_context_window_fallback_only() {
    let primary = MockDriver::single_response("a").with_context_window(8192);
    let fallback = MockDriver::single_response("b").with_context_window(16384);
    let driver = RoutingDriver::new(Box::new(primary), Box::new(fallback))
        .with_strategy(RoutingStrategy::FallbackOnly);

    assert_eq!(driver.context_window(), 16384);
}

#[tokio::test]
async fn test_metrics_fallback_success_rate() {
    let driver =
        RoutingDriver::new(Box::new(FailingDriver), Box::new(MockDriver::single_response("ok")));

    let _ = driver.complete(test_request()).await;
    assert!((driver.metrics().fallback_success_rate() - 1.0).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_metrics_initial_state() {
    let driver = RoutingDriver::new(
        Box::new(MockDriver::single_response("a")),
        Box::new(MockDriver::single_response("b")),
    );

    assert_eq!(driver.metrics().primary_attempts(), 0);
    assert_eq!(driver.metrics().spillover_count(), 0);
    assert!((driver.metrics().fallback_success_rate() - 0.0).abs() < f64::EPSILON);
}
