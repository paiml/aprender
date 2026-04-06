//! Routing driver for local-first, remote fallback inference.
//!
//! Wraps a primary (typically sovereign/local) and fallback
//! (typically remote/cloud) `LlmDriver`. Attempts the primary
//! driver first; on failure, spills over to the fallback.
//!
//! Phase 2: Implements `RoutingDriver` from the agent spec.
//!
//! Privacy tier: inherits the more permissive of the two
//! underlying drivers (if fallback is Standard, routing is
//! Standard — data *may* leave the machine on spillover).

use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::agent::driver::{CompletionRequest, CompletionResponse, LlmDriver};
use crate::agent::result::AgentError;
use crate::serve::backends::PrivacyTier;

/// Strategy for selecting which driver to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Try primary first, fallback on any error.
    PrimaryWithFallback,
    /// Use primary only (no fallback). Equivalent to
    /// using the primary driver directly, but keeps the
    /// `RoutingDriver` interface for config uniformity.
    PrimaryOnly,
    /// Use fallback only (no primary). Useful for testing
    /// or when local inference is unavailable.
    FallbackOnly,
}

/// Metrics for routing decisions.
#[derive(Debug)]
pub struct RoutingMetrics {
    /// Number of successful primary completions.
    primary_successes: AtomicU64,
    /// Number of primary failures (all kinds).
    primary_failures: AtomicU64,
    /// Number of times fallback was actually attempted.
    spillovers: AtomicU64,
    /// Number of successful fallback completions.
    fallback_successes: AtomicU64,
    /// Number of fallback failures.
    fallback_failures: AtomicU64,
}

impl RoutingMetrics {
    fn new() -> Self {
        Self {
            primary_successes: AtomicU64::new(0),
            primary_failures: AtomicU64::new(0),
            spillovers: AtomicU64::new(0),
            fallback_successes: AtomicU64::new(0),
            fallback_failures: AtomicU64::new(0),
        }
    }

    /// Total primary attempts.
    pub fn primary_attempts(&self) -> u64 {
        self.primary_successes.load(Ordering::Relaxed)
            + self.primary_failures.load(Ordering::Relaxed)
    }

    /// Total spillover count (primary failures → fallback).
    pub fn spillover_count(&self) -> u64 {
        self.spillovers.load(Ordering::Relaxed)
    }

    /// Fallback success rate (0.0–1.0).
    pub fn fallback_success_rate(&self) -> f64 {
        let successes = self.fallback_successes.load(Ordering::Relaxed);
        let failures = self.fallback_failures.load(Ordering::Relaxed);
        let total = successes + failures;
        if total == 0 {
            0.0
        } else {
            // Precision loss acceptable for metrics display
            #[allow(clippy::cast_precision_loss)]
            {
                successes as f64 / total as f64
            }
        }
    }
}

/// Routing driver: local-first with remote fallback.
///
/// Wraps two `LlmDriver` implementations. The primary driver
/// (typically `RealizarDriver` for sovereign inference) is tried
/// first. On failure, the fallback (typically `RemoteDriver`)
/// handles the request.
///
/// Privacy tier is the more permissive of the two drivers — if
/// the fallback is `Standard`, data may leave the machine on
/// spillover, so the routing driver reports `Standard`.
pub struct RoutingDriver {
    primary: Box<dyn LlmDriver>,
    fallback: Option<Box<dyn LlmDriver>>,
    strategy: RoutingStrategy,
    metrics: Arc<RoutingMetrics>,
}

impl RoutingDriver {
    /// Create a new routing driver with primary and fallback.
    pub fn new(primary: Box<dyn LlmDriver>, fallback: Box<dyn LlmDriver>) -> Self {
        Self {
            primary,
            fallback: Some(fallback),
            strategy: RoutingStrategy::PrimaryWithFallback,
            metrics: Arc::new(RoutingMetrics::new()),
        }
    }

    /// Create a routing driver with primary only (no fallback).
    pub fn primary_only(primary: Box<dyn LlmDriver>) -> Self {
        Self {
            primary,
            fallback: None,
            strategy: RoutingStrategy::PrimaryOnly,
            metrics: Arc::new(RoutingMetrics::new()),
        }
    }

    /// Set the routing strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Get routing metrics.
    pub fn metrics(&self) -> &RoutingMetrics {
        &self.metrics
    }

    /// Check if the error is retryable (should trigger fallback).
    fn should_fallback(error: &AgentError) -> bool {
        use crate::agent::result::DriverError;
        match error {
            AgentError::Driver(driver_err) => {
                matches!(
                    driver_err,
                    DriverError::InferenceFailed(_)
                        | DriverError::ModelNotFound(_)
                        | DriverError::Network(_)
                )
            }
            _ => false,
        }
    }

    /// Record primary result in metrics.
    fn record_primary(&self, result: &Result<CompletionResponse, AgentError>) {
        match result {
            Ok(_) => {
                self.metrics.primary_successes.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.metrics.primary_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Record fallback result in metrics.
    fn record_fallback(&self, result: &Result<CompletionResponse, AgentError>) {
        match result {
            Ok(_) => {
                self.metrics.fallback_successes.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.metrics.fallback_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Try primary, spillover to fallback on retryable error.
    async fn complete_with_fallback(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError> {
        let primary_result = self.primary.complete(request.clone()).await;

        match primary_result {
            Ok(response) => {
                self.metrics.primary_successes.fetch_add(1, Ordering::Relaxed);
                Ok(response)
            }
            Err(ref e) if Self::should_fallback(e) && self.fallback.is_some() => {
                self.metrics.primary_failures.fetch_add(1, Ordering::Relaxed);
                self.metrics.spillovers.fetch_add(1, Ordering::Relaxed);
                self.run_fallback(request).await
            }
            Err(e) => {
                self.metrics.primary_failures.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Execute on fallback driver and record metrics.
    async fn run_fallback(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, AgentError> {
        if let Some(ref fallback) = self.fallback {
            let result = fallback.complete(request).await;
            self.record_fallback(&result);
            return result;
        }
        Err(AgentError::Driver(crate::agent::result::DriverError::InferenceFailed(
            "No fallback driver configured".into(),
        )))
    }
}

#[async_trait]
impl LlmDriver for RoutingDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, AgentError> {
        match self.strategy {
            RoutingStrategy::FallbackOnly => self.run_fallback(request).await,
            RoutingStrategy::PrimaryOnly => {
                let result = self.primary.complete(request).await;
                self.record_primary(&result);
                result
            }
            RoutingStrategy::PrimaryWithFallback => self.complete_with_fallback(request).await,
        }
    }

    fn context_window(&self) -> usize {
        match self.strategy {
            RoutingStrategy::FallbackOnly => {
                self.fallback.as_ref().map_or(self.primary.context_window(), |f| f.context_window())
            }
            _ => self.primary.context_window(),
        }
    }

    fn privacy_tier(&self) -> PrivacyTier {
        let primary_tier = self.primary.privacy_tier();
        let fallback_tier = self.fallback.as_ref().map_or(primary_tier, |f| f.privacy_tier());

        // Most permissive tier wins (Standard > Private > Sovereign)
        match (&primary_tier, &fallback_tier) {
            (PrivacyTier::Standard, _) | (_, PrivacyTier::Standard) => PrivacyTier::Standard,
            (PrivacyTier::Private, _) | (_, PrivacyTier::Private) => PrivacyTier::Private,
            _ => PrivacyTier::Sovereign,
        }
    }
}

#[cfg(test)]
#[path = "router_tests.rs"]
mod tests;
