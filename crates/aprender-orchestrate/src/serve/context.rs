//! Context Window Management
//!
//! Automatic token counting and context truncation.
//! Prevents silent failures when prompts exceed model context limits.

use crate::serve::templates::ChatMessage;
use serde::{Deserialize, Serialize};

// ============================================================================
// SERVE-CTX-001: Context Configuration
// ============================================================================

/// Known model context window sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextWindow {
    /// Maximum context size in tokens
    pub max_tokens: usize,
    /// Reserved tokens for output
    pub output_reserve: usize,
}

impl ContextWindow {
    /// Create a new context window configuration
    #[must_use]
    pub const fn new(max_tokens: usize, output_reserve: usize) -> Self {
        Self { max_tokens, output_reserve }
    }

    /// Available tokens for input after reserving output space
    #[must_use]
    pub const fn available_input(&self) -> usize {
        self.max_tokens.saturating_sub(self.output_reserve)
    }

    /// Model name patterns mapped to (max_tokens, output_reserve).
    /// Order matters: more specific patterns must come first.
    /// Each entry uses ALL semantics: every pattern in the slice must match.
    const MODEL_WINDOWS: &[(&[&str], usize, usize)] = &[
        (&["gpt-4-turbo"], 128_000, 4096),
        (&["gpt-4o"], 128_000, 4096),
        (&["gpt-4-32k"], 32_768, 4096),
        (&["gpt-4"], 8_192, 2048),
        (&["gpt-3.5-turbo-16k"], 16_384, 4096),
        (&["gpt-3.5"], 4_096, 1024),
        (&["claude-3"], 200_000, 4096),
        (&["claude-2"], 200_000, 4096),
        (&["claude"], 100_000, 4096),
        (&["llama-3"], 8_192, 2048),
        (&["llama-2", "32k"], 32_768, 4096),
        (&["llama"], 4_096, 1024),
        (&["mixtral"], 32_768, 4096),
        (&["mistral"], 8_192, 2048),
    ];

    /// Get context window for known model
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        let lower = model.to_lowercase();
        Self::MODEL_WINDOWS
            .iter()
            .find(|(pats, _, _)| pats.iter().all(|p| lower.contains(p)))
            .map_or_else(Self::default, |&(_, max, reserve)| Self::new(max, reserve))
    }
}

impl Default for ContextWindow {
    fn default() -> Self {
        Self::new(4_096, 1024)
    }
}

// ============================================================================
// SERVE-CTX-002: Token Estimation
// ============================================================================

/// Simple token estimator (approximation without full tokenizer)
///
/// Uses heuristic: ~4 characters per token for English text.
/// For accurate counts, use a proper tokenizer.
pub struct TokenEstimator {
    /// Characters per token (default: 4.0)
    chars_per_token: f64,
}

impl TokenEstimator {
    /// Create with default settings
    #[must_use]
    pub fn new() -> Self {
        Self { chars_per_token: 4.0 }
    }

    /// Create with custom chars-per-token ratio
    #[must_use]
    pub fn with_ratio(chars_per_token: f64) -> Self {
        Self { chars_per_token }
    }

    /// Estimate token count for a string
    #[must_use]
    pub fn estimate(&self, text: &str) -> usize {
        if self.chars_per_token <= 0.0 {
            return text.len();
        }
        (text.len() as f64 / self.chars_per_token).ceil() as usize
    }

    /// Estimate tokens for chat messages
    #[must_use]
    pub fn estimate_messages(&self, messages: &[ChatMessage]) -> usize {
        let mut total = 0;
        for msg in messages {
            // Role tokens (approximately 3-4 tokens per message for formatting)
            total += 4;
            total += self.estimate(&msg.content);
        }
        total
    }
}

impl Default for TokenEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SERVE-CTX-003: Context Manager
// ============================================================================

/// Truncation strategy when context is exceeded
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Remove oldest messages first (sliding window)
    #[default]
    SlidingWindow,
    /// Remove from the middle, keep first and last
    MiddleOut,
    /// Fail with error instead of truncating
    Error,
}

/// Context management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Context window settings
    pub window: ContextWindow,
    /// Truncation strategy
    pub strategy: TruncationStrategy,
    /// Always preserve system message
    pub preserve_system: bool,
    /// Minimum messages to keep
    pub min_messages: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            window: ContextWindow::default(),
            strategy: TruncationStrategy::SlidingWindow,
            preserve_system: true,
            min_messages: 2,
        }
    }
}

impl ContextConfig {
    /// Create config for a specific model
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        Self { window: ContextWindow::for_model(model), ..Default::default() }
    }
}

/// Context manager for handling token limits
pub struct ContextManager {
    config: ContextConfig,
    estimator: TokenEstimator,
}

impl ContextManager {
    /// Create a new context manager
    #[must_use]
    pub fn new(config: ContextConfig) -> Self {
        Self { config, estimator: TokenEstimator::new() }
    }

    /// Create for a specific model
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        Self::new(ContextConfig::for_model(model))
    }

    /// Check if messages fit within context window
    #[must_use]
    pub fn fits(&self, messages: &[ChatMessage]) -> bool {
        let tokens = self.estimator.estimate_messages(messages);
        tokens <= self.config.window.available_input()
    }

    /// Get estimated token count for messages
    #[must_use]
    pub fn estimate_tokens(&self, messages: &[ChatMessage]) -> usize {
        self.estimator.estimate_messages(messages)
    }

    /// Get available token budget
    #[must_use]
    pub fn available_tokens(&self) -> usize {
        self.config.window.available_input()
    }

    /// Truncate messages to fit within context window
    ///
    /// Returns truncated messages or error if strategy is `Error` and truncation needed.
    pub fn truncate(&self, messages: &[ChatMessage]) -> Result<Vec<ChatMessage>, ContextError> {
        let available = self.config.window.available_input();
        let current = self.estimator.estimate_messages(messages);

        if current <= available {
            return Ok(messages.to_vec());
        }

        match self.config.strategy {
            TruncationStrategy::Error => {
                Err(ContextError::ExceedsLimit { tokens: current, limit: available })
            }
            TruncationStrategy::SlidingWindow => {
                Ok(self.truncate_sliding_window(messages, available))
            }
            TruncationStrategy::MiddleOut => Ok(self.truncate_middle_out(messages, available)),
        }
    }

    fn truncate_sliding_window(
        &self,
        messages: &[ChatMessage],
        available: usize,
    ) -> Vec<ChatMessage> {
        let mut result = Vec::new();
        let mut tokens_used = 0;

        // Extract system message if preserving
        let (system_msg, other_msgs): (Vec<_>, Vec<_>) = if self.config.preserve_system {
            messages.iter().partition(|m| matches!(m.role, crate::serve::templates::Role::System))
        } else {
            (vec![], messages.iter().collect())
        };

        // Add system message first
        for msg in &system_msg {
            let msg_tokens = self.estimator.estimate(&msg.content) + 4;
            if tokens_used + msg_tokens <= available {
                result.push((*msg).clone());
                tokens_used += msg_tokens;
            }
        }

        // Add messages from the end (most recent first)
        let mut recent_msgs: Vec<ChatMessage> = Vec::new();
        for msg in other_msgs.into_iter().rev() {
            let msg_tokens = self.estimator.estimate(&msg.content) + 4;
            if tokens_used + msg_tokens <= available {
                recent_msgs.push(msg.clone());
                tokens_used += msg_tokens;
            } else if recent_msgs.len() >= self.config.min_messages {
                break;
            }
        }

        // Reverse to restore chronological order
        recent_msgs.reverse();
        result.extend(recent_msgs);

        result
    }

    fn truncate_middle_out(&self, messages: &[ChatMessage], available: usize) -> Vec<ChatMessage> {
        if messages.len() <= 2 {
            return messages.to_vec();
        }

        let mut result = Vec::new();
        let mut tokens_used = 0;

        // Always keep first message (often system)
        let first = &messages[0];
        let first_tokens = self.estimator.estimate(&first.content) + 4;
        result.push(first.clone());
        tokens_used += first_tokens;

        // Always keep last message
        let last = &messages[messages.len() - 1];
        let last_tokens = self.estimator.estimate(&last.content) + 4;
        tokens_used += last_tokens;

        // Add messages from the end, working backwards
        let middle = &messages[1..messages.len() - 1];
        let mut kept_from_end: Vec<ChatMessage> = Vec::new();

        for msg in middle.iter().rev() {
            let msg_tokens = self.estimator.estimate(&msg.content) + 4;
            if tokens_used + msg_tokens <= available {
                kept_from_end.push(msg.clone());
                tokens_used += msg_tokens;
            } else {
                break;
            }
        }

        // Reverse and add
        kept_from_end.reverse();
        result.extend(kept_from_end);
        result.push(last.clone());

        result
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new(ContextConfig::default())
    }
}

/// Context management errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextError {
    /// Context window exceeded and strategy is Error
    ExceedsLimit { tokens: usize, limit: usize },
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExceedsLimit { tokens, limit } => {
                write!(f, "Context exceeds limit: {} tokens, max {} tokens", tokens, limit)
            }
        }
    }
}

impl std::error::Error for ContextError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
#[path = "context_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "context_contract_tests.rs"]
mod contract_tests;
