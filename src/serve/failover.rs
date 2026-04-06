//! Stateful Failover Protocol
//!
//! Implements Toyota Way "Jidoka" (Autonomation) for streaming requests.
//!
//! When a streaming response fails mid-generation, the failover protocol:
//! 1. Caches the prompt and generated prefix
//! 2. Re-initiates the request to a secondary backend
//! 3. Continues from the failure point transparently

use crate::serve::backends::ServingBackend;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ============================================================================
// SERVE-FLO-001: Streaming State
// ============================================================================

/// State of a streaming response
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamState {
    /// Not started
    Pending,
    /// Actively receiving tokens
    Streaming,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed(String),
    /// Recovered via failover
    Recovered,
}

/// Cached streaming context for failover recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingContext {
    /// Original prompt
    pub prompt: String,
    /// Generated tokens so far
    pub generated_prefix: String,
    /// Total tokens generated
    pub token_count: usize,
    /// Primary backend that was used
    pub primary_backend: String,
    /// Request ID for correlation
    pub request_id: String,
}

impl StreamingContext {
    /// Create a new streaming context
    #[must_use]
    pub fn new(prompt: impl Into<String>, request_id: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            generated_prefix: String::new(),
            token_count: 0,
            primary_backend: String::new(),
            request_id: request_id.into(),
        }
    }

    /// Append generated tokens
    pub fn append(&mut self, tokens: &str) {
        self.generated_prefix.push_str(tokens);
        // Rough token count estimate
        self.token_count += tokens.split_whitespace().count().max(1);
    }

    /// Get continuation prompt (original + generated so far)
    #[must_use]
    pub fn continuation_prompt(&self) -> String {
        if self.generated_prefix.is_empty() {
            self.prompt.clone()
        } else {
            format!("{}{}", self.prompt, self.generated_prefix)
        }
    }

    /// Check if recovery is worthwhile (has meaningful progress)
    #[must_use]
    pub fn worth_recovering(&self) -> bool {
        self.token_count >= 5 // At least 5 tokens generated
    }
}

// ============================================================================
// SERVE-FLO-002: Failover Configuration
// ============================================================================

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Maximum retries per request
    pub max_retries: u32,
    /// Timeout for failover attempt
    pub failover_timeout: Duration,
    /// Minimum tokens before considering recovery
    pub min_tokens_for_recovery: usize,
    /// Whether to include prefix in failover request
    pub include_prefix: bool,
    /// Backends to try in order after primary fails
    pub fallback_order: Vec<ServingBackend>,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            failover_timeout: Duration::from_secs(30),
            min_tokens_for_recovery: 5,
            include_prefix: true,
            fallback_order: vec![
                ServingBackend::Realizar,
                ServingBackend::Ollama,
                ServingBackend::Together,
                ServingBackend::Groq,
            ],
        }
    }
}

// ============================================================================
// SERVE-FLO-003: Failover Manager
// ============================================================================

/// Failover attempt record
#[derive(Debug, Clone)]
pub struct FailoverAttempt {
    pub backend: ServingBackend,
    pub started_at: Instant,
    pub result: Option<FailoverResult>,
}

/// Result of a failover attempt
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailoverResult {
    Success,
    Timeout,
    BackendError(String),
    NoBackendsAvailable,
}

/// Stateful failover manager
pub struct FailoverManager {
    config: FailoverConfig,
    /// Active streaming contexts by request ID
    contexts: std::collections::HashMap<String, StreamingContext>,
    /// Recent failover history for observability
    history: VecDeque<FailoverAttempt>,
    /// Maximum history entries
    max_history: usize,
}

impl FailoverManager {
    /// Create a new failover manager
    #[must_use]
    pub fn new(config: FailoverConfig) -> Self {
        Self {
            config,
            contexts: std::collections::HashMap::new(),
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Create with default config
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(FailoverConfig::default())
    }

    /// Start tracking a streaming request
    pub fn start_tracking(&mut self, request_id: &str, prompt: &str) {
        let context = StreamingContext::new(prompt, request_id);
        self.contexts.insert(request_id.to_string(), context);
    }

    /// Update context with new tokens
    pub fn append_tokens(&mut self, request_id: &str, tokens: &str) {
        if let Some(ctx) = self.contexts.get_mut(request_id) {
            ctx.append(tokens);
        }
    }

    /// Mark request as completed
    pub fn complete(&mut self, request_id: &str) {
        self.contexts.remove(request_id);
    }

    /// Get context for failover
    #[must_use]
    pub fn get_context(&self, request_id: &str) -> Option<&StreamingContext> {
        self.contexts.get(request_id)
    }

    /// Check if failover should be attempted
    #[must_use]
    pub fn should_failover(&self, request_id: &str) -> bool {
        self.contexts.get(request_id).map(|ctx| ctx.worth_recovering()).unwrap_or(false)
    }

    /// Get next backend to try for failover
    #[must_use]
    pub fn next_backend(&self, failed_backend: ServingBackend) -> Option<ServingBackend> {
        self.config.fallback_order.iter().find(|&&b| b != failed_backend).copied()
    }

    /// Prepare failover request
    #[must_use]
    pub fn prepare_failover(&self, request_id: &str) -> Option<FailoverRequest> {
        let ctx = self.contexts.get(request_id)?;

        let prompt =
            if self.config.include_prefix { ctx.continuation_prompt() } else { ctx.prompt.clone() };

        Some(FailoverRequest {
            request_id: request_id.to_string(),
            prompt,
            generated_prefix: ctx.generated_prefix.clone(),
            token_count: ctx.token_count,
        })
    }

    /// Record a failover attempt
    pub fn record_attempt(&mut self, attempt: FailoverAttempt) {
        self.history.push_back(attempt);
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Get failover statistics
    #[must_use]
    pub fn stats(&self) -> FailoverStats {
        let total = self.history.len();
        let successes =
            self.history.iter().filter(|a| a.result == Some(FailoverResult::Success)).count();
        let timeouts =
            self.history.iter().filter(|a| a.result == Some(FailoverResult::Timeout)).count();

        FailoverStats {
            total_attempts: total,
            successful: successes,
            timeouts,
            active_contexts: self.contexts.len(),
        }
    }

    /// Get config
    #[must_use]
    pub fn config(&self) -> &FailoverConfig {
        &self.config
    }
}

impl Default for FailoverManager {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Failover request prepared for retry
#[derive(Debug, Clone)]
pub struct FailoverRequest {
    pub request_id: String,
    pub prompt: String,
    pub generated_prefix: String,
    pub token_count: usize,
}

/// Failover statistics
#[derive(Debug, Clone, Default)]
pub struct FailoverStats {
    pub total_attempts: usize,
    pub successful: usize,
    pub timeouts: usize,
    pub active_contexts: usize,
}

impl FailoverStats {
    /// Success rate as percentage
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            0.0
        } else {
            (self.successful as f64 / self.total_attempts as f64) * 100.0
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // SERVE-FLO-001: Streaming Context Tests
    // ========================================================================

    #[test]
    fn test_SERVE_FLO_001_context_new() {
        let ctx = StreamingContext::new("Hello, how are you?", "req-123");
        assert_eq!(ctx.prompt, "Hello, how are you?");
        assert_eq!(ctx.request_id, "req-123");
        assert!(ctx.generated_prefix.is_empty());
        assert_eq!(ctx.token_count, 0);
    }

    #[test]
    fn test_SERVE_FLO_001_context_append() {
        let mut ctx = StreamingContext::new("Test", "req-1");
        ctx.append("Hello ");
        ctx.append("world!");
        assert_eq!(ctx.generated_prefix, "Hello world!");
        assert!(ctx.token_count > 0);
    }

    #[test]
    fn test_SERVE_FLO_001_continuation_prompt() {
        let mut ctx = StreamingContext::new("Prompt: ", "req-1");
        ctx.append("Response so far");
        assert_eq!(ctx.continuation_prompt(), "Prompt: Response so far");
    }

    #[test]
    fn test_SERVE_FLO_001_continuation_prompt_empty() {
        let ctx = StreamingContext::new("Just prompt", "req-1");
        assert_eq!(ctx.continuation_prompt(), "Just prompt");
    }

    #[test]
    fn test_SERVE_FLO_001_worth_recovering() {
        let mut ctx = StreamingContext::new("Test", "req-1");
        assert!(!ctx.worth_recovering());

        ctx.append("one two three four five six");
        assert!(ctx.worth_recovering());
    }

    // ========================================================================
    // SERVE-FLO-002: Failover Config Tests
    // ========================================================================

    #[test]
    fn test_SERVE_FLO_002_default_config() {
        let config = FailoverConfig::default();
        assert_eq!(config.max_retries, 2);
        assert!(config.include_prefix);
        assert!(!config.fallback_order.is_empty());
    }

    #[test]
    fn test_SERVE_FLO_002_fallback_order() {
        let config = FailoverConfig::default();
        assert!(config.fallback_order.contains(&ServingBackend::Realizar));
        assert!(config.fallback_order.contains(&ServingBackend::Together));
    }

    // ========================================================================
    // SERVE-FLO-003: Failover Manager Tests
    // ========================================================================

    #[test]
    fn test_SERVE_FLO_003_start_tracking() {
        let mut manager = FailoverManager::with_defaults();
        manager.start_tracking("req-1", "Test prompt");
        assert!(manager.get_context("req-1").is_some());
    }

    #[test]
    fn test_SERVE_FLO_003_append_tokens() {
        let mut manager = FailoverManager::with_defaults();
        manager.start_tracking("req-1", "Prompt");
        manager.append_tokens("req-1", "Generated");

        let ctx = manager.get_context("req-1").expect("unexpected failure");
        assert_eq!(ctx.generated_prefix, "Generated");
    }

    #[test]
    fn test_SERVE_FLO_003_complete_removes() {
        let mut manager = FailoverManager::with_defaults();
        manager.start_tracking("req-1", "Prompt");
        manager.complete("req-1");
        assert!(manager.get_context("req-1").is_none());
    }

    #[test]
    fn test_SERVE_FLO_003_should_failover() {
        let mut manager = FailoverManager::with_defaults();
        manager.start_tracking("req-1", "Prompt");

        // Not enough tokens yet
        assert!(!manager.should_failover("req-1"));

        // Add enough tokens
        manager.append_tokens("req-1", "one two three four five six");
        assert!(manager.should_failover("req-1"));
    }

    // ========================================================================
    // SERVE-FLO-004: Next Backend Tests
    // ========================================================================

    #[test]
    fn test_SERVE_FLO_004_next_backend_skips_failed() {
        let manager = FailoverManager::with_defaults();
        let next = manager.next_backend(ServingBackend::Realizar);
        assert!(next.is_some());
        assert_ne!(next.expect("unexpected failure"), ServingBackend::Realizar);
    }

    #[test]
    fn test_SERVE_FLO_004_next_backend_order() {
        let config = FailoverConfig {
            fallback_order: vec![ServingBackend::Ollama, ServingBackend::Together],
            ..Default::default()
        };
        let manager = FailoverManager::new(config);

        let next = manager.next_backend(ServingBackend::Realizar);
        assert_eq!(next, Some(ServingBackend::Ollama));
    }

    // ========================================================================
    // SERVE-FLO-005: Prepare Failover Tests
    // ========================================================================

    #[test]
    fn test_SERVE_FLO_005_prepare_failover() {
        let mut manager = FailoverManager::with_defaults();
        manager.start_tracking("req-1", "Original prompt");
        manager.append_tokens("req-1", " partial response");

        let request = manager.prepare_failover("req-1").expect("unexpected failure");
        assert_eq!(request.request_id, "req-1");
        assert!(request.prompt.contains("Original prompt"));
        assert!(request.prompt.contains("partial response"));
    }

    #[test]
    fn test_SERVE_FLO_005_prepare_failover_not_found() {
        let manager = FailoverManager::with_defaults();
        assert!(manager.prepare_failover("nonexistent").is_none());
    }

    // ========================================================================
    // SERVE-FLO-006: Statistics Tests
    // ========================================================================

    #[test]
    fn test_SERVE_FLO_006_empty_stats() {
        let manager = FailoverManager::with_defaults();
        let stats = manager.stats();
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_SERVE_FLO_006_record_attempt() {
        let mut manager = FailoverManager::with_defaults();
        manager.record_attempt(FailoverAttempt {
            backend: ServingBackend::Together,
            started_at: Instant::now(),
            result: Some(FailoverResult::Success),
        });

        let stats = manager.stats();
        assert_eq!(stats.total_attempts, 1);
        assert_eq!(stats.successful, 1);
        assert_eq!(stats.success_rate(), 100.0);
    }

    #[test]
    fn test_SERVE_FLO_006_mixed_results() {
        let mut manager = FailoverManager::with_defaults();

        manager.record_attempt(FailoverAttempt {
            backend: ServingBackend::Together,
            started_at: Instant::now(),
            result: Some(FailoverResult::Success),
        });
        manager.record_attempt(FailoverAttempt {
            backend: ServingBackend::Groq,
            started_at: Instant::now(),
            result: Some(FailoverResult::Timeout),
        });

        let stats = manager.stats();
        assert_eq!(stats.total_attempts, 2);
        assert_eq!(stats.successful, 1);
        assert_eq!(stats.timeouts, 1);
        assert_eq!(stats.success_rate(), 50.0);
    }

    // ========================================================================
    // SERVE-FLO-007: Stream State Tests
    // ========================================================================

    #[test]
    fn test_SERVE_FLO_007_stream_states() {
        assert_eq!(StreamState::Pending, StreamState::Pending);
        assert_ne!(StreamState::Streaming, StreamState::Completed);

        let failed = StreamState::Failed("Connection reset".to_string());
        if let StreamState::Failed(msg) = failed {
            assert!(msg.contains("reset"));
        }
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_stream_state_clone() {
        let state = StreamState::Failed("error".to_string());
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_stream_state_debug() {
        let state = StreamState::Recovered;
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("Recovered"));
    }

    #[test]
    fn test_stream_state_all_variants() {
        let states = [
            StreamState::Pending,
            StreamState::Streaming,
            StreamState::Completed,
            StreamState::Failed("err".to_string()),
            StreamState::Recovered,
        ];
        assert_eq!(states.len(), 5);
    }

    #[test]
    fn test_streaming_context_clone() {
        let mut ctx = StreamingContext::new("prompt", "req-1");
        ctx.append("tokens");
        let cloned = ctx.clone();
        assert_eq!(ctx.prompt, cloned.prompt);
        assert_eq!(ctx.generated_prefix, cloned.generated_prefix);
    }

    #[test]
    fn test_streaming_context_debug() {
        let ctx = StreamingContext::new("test prompt", "req-debug");
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("test prompt"));
        assert!(debug_str.contains("req-debug"));
    }

    #[test]
    fn test_streaming_context_serialize() {
        let ctx = StreamingContext::new("serializable", "req-ser");
        let json = serde_json::to_string(&ctx).expect("json serialize failed");
        assert!(json.contains("serializable"));
        assert!(json.contains("req-ser"));
    }

    #[test]
    fn test_streaming_context_deserialize() {
        let json = r#"{"prompt":"deserialized","generated_prefix":"prefix","token_count":5,"primary_backend":"test","request_id":"req-de"}"#;
        let ctx: StreamingContext = serde_json::from_str(json).expect("json deserialize failed");
        assert_eq!(ctx.prompt, "deserialized");
        assert_eq!(ctx.generated_prefix, "prefix");
        assert_eq!(ctx.token_count, 5);
    }

    #[test]
    fn test_failover_config_clone() {
        let config = FailoverConfig::default();
        let cloned = config.clone();
        assert_eq!(config.max_retries, cloned.max_retries);
    }

    #[test]
    fn test_failover_config_debug() {
        let config = FailoverConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("max_retries"));
    }

    #[test]
    fn test_failover_config_serialize() {
        let config = FailoverConfig::default();
        let json = serde_json::to_string(&config).expect("json serialize failed");
        assert!(json.contains("max_retries"));
        assert!(json.contains("include_prefix"));
    }

    #[test]
    fn test_failover_attempt_clone() {
        let attempt = FailoverAttempt {
            backend: ServingBackend::Ollama,
            started_at: Instant::now(),
            result: Some(FailoverResult::Success),
        };
        let cloned = attempt.clone();
        assert_eq!(attempt.backend, cloned.backend);
    }

    #[test]
    fn test_failover_attempt_debug() {
        let attempt = FailoverAttempt {
            backend: ServingBackend::Together,
            started_at: Instant::now(),
            result: None,
        };
        let debug_str = format!("{:?}", attempt);
        assert!(debug_str.contains("Together"));
    }

    #[test]
    fn test_failover_result_clone() {
        let result = FailoverResult::BackendError("timeout".to_string());
        let cloned = result.clone();
        assert_eq!(result, cloned);
    }

    #[test]
    fn test_failover_result_debug() {
        let result = FailoverResult::NoBackendsAvailable;
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("NoBackendsAvailable"));
    }

    #[test]
    fn test_failover_result_all_variants() {
        let results = [
            FailoverResult::Success,
            FailoverResult::Timeout,
            FailoverResult::BackendError("err".to_string()),
            FailoverResult::NoBackendsAvailable,
        ];
        assert_eq!(results.len(), 4);
        assert_eq!(results[0], FailoverResult::Success);
    }

    #[test]
    fn test_failover_request_clone() {
        let request = FailoverRequest {
            request_id: "req-1".to_string(),
            prompt: "prompt".to_string(),
            generated_prefix: "prefix".to_string(),
            token_count: 10,
        };
        let cloned = request.clone();
        assert_eq!(request.request_id, cloned.request_id);
    }

    #[test]
    fn test_failover_request_debug() {
        let request = FailoverRequest {
            request_id: "debug-req".to_string(),
            prompt: "test".to_string(),
            generated_prefix: "".to_string(),
            token_count: 0,
        };
        let debug_str = format!("{:?}", request);
        assert!(debug_str.contains("debug-req"));
    }

    #[test]
    fn test_failover_stats_clone() {
        let stats =
            FailoverStats { total_attempts: 10, successful: 8, timeouts: 1, active_contexts: 2 };
        let cloned = stats.clone();
        assert_eq!(stats.total_attempts, cloned.total_attempts);
    }

    #[test]
    fn test_failover_stats_debug() {
        let stats = FailoverStats::default();
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("FailoverStats"));
    }

    #[test]
    fn test_failover_stats_default() {
        let stats = FailoverStats::default();
        assert_eq!(stats.total_attempts, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.timeouts, 0);
        assert_eq!(stats.active_contexts, 0);
    }

    #[test]
    fn test_failover_manager_config() {
        let config = FailoverConfig { max_retries: 5, ..Default::default() };
        let manager = FailoverManager::new(config);
        assert_eq!(manager.config().max_retries, 5);
    }

    #[test]
    fn test_failover_manager_default() {
        let manager = FailoverManager::default();
        let stats = manager.stats();
        assert_eq!(stats.total_attempts, 0);
    }

    #[test]
    fn test_failover_manager_history_trimming() {
        let mut manager = FailoverManager::with_defaults();
        // max_history is 100, add 110 entries
        for _ in 0..110 {
            manager.record_attempt(FailoverAttempt {
                backend: ServingBackend::Together,
                started_at: Instant::now(),
                result: Some(FailoverResult::Success),
            });
        }
        let stats = manager.stats();
        // Should be trimmed to max_history (100)
        assert_eq!(stats.total_attempts, 100);
    }

    #[test]
    fn test_append_tokens_nonexistent() {
        let mut manager = FailoverManager::with_defaults();
        // Should not panic when appending to nonexistent request
        manager.append_tokens("nonexistent", "tokens");
        assert!(manager.get_context("nonexistent").is_none());
    }

    #[test]
    fn test_should_failover_nonexistent() {
        let manager = FailoverManager::with_defaults();
        assert!(!manager.should_failover("nonexistent"));
    }

    #[test]
    fn test_prepare_failover_without_prefix() {
        let config = FailoverConfig { include_prefix: false, ..Default::default() };
        let mut manager = FailoverManager::new(config);
        manager.start_tracking("req-1", "Original prompt");
        manager.append_tokens("req-1", " generated");

        let request = manager.prepare_failover("req-1").expect("unexpected failure");
        // Without prefix, should use original prompt only
        assert_eq!(request.prompt, "Original prompt");
    }

    #[test]
    fn test_stats_active_contexts() {
        let mut manager = FailoverManager::with_defaults();
        manager.start_tracking("req-1", "p1");
        manager.start_tracking("req-2", "p2");
        manager.start_tracking("req-3", "p3");

        let stats = manager.stats();
        assert_eq!(stats.active_contexts, 3);

        manager.complete("req-1");
        let stats = manager.stats();
        assert_eq!(stats.active_contexts, 2);
    }

    #[test]
    fn test_streaming_context_primary_backend() {
        let mut ctx = StreamingContext::new("prompt", "req-1");
        ctx.primary_backend = "realizar".to_string();
        assert_eq!(ctx.primary_backend, "realizar");
    }
}
