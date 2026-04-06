//! Loop guard — prevents runaway agent loops (Jidoka pattern).
//!
//! Tracks iteration count, tool call hashes (FxHash ping-pong detection),
//! cost budget, and consecutive `MaxTokens` responses. Verdicts: `Allow`,
//! `Warn`, `Block`, or `CircuitBreak`. See arXiv:2512.10350 (Tacheny).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::result::TokenUsage;

/// Prevents runaway agent loops (Jidoka pattern).
pub struct LoopGuard {
    max_iterations: u32,
    current_iteration: u32,
    max_tool_calls: u32,
    total_tool_calls: u32,
    tool_call_counts: HashMap<u64, u32>, // FxHash(tool,input) → count
    consecutive_max_tokens: u32,
    usage: TokenUsage,
    max_cost_usd: f64, // 0.0 = unlimited (sovereign)
    accumulated_cost_usd: f64,
    max_tokens_budget: Option<u64>, // None = unlimited
}

/// Verdict from the loop guard on whether to proceed.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopVerdict {
    /// Proceed with execution.
    Allow,
    /// Proceed but warn (approaching limits).
    Warn(String),
    /// Block this specific tool call (repeated pattern).
    Block(String),
    /// Hard stop the entire loop.
    CircuitBreak(String),
}

/// Configuration for ping-pong detection thresholds.
const PINGPONG_THRESHOLD: u32 = 3;
/// Maximum consecutive `MaxTokens` before circuit break.
const MAX_CONSECUTIVE_TRUNCATION: u32 = 5;
/// Warn when iteration count reaches this fraction of max.
const WARN_ITERATION_FRACTION: f64 = 0.8;

impl LoopGuard {
    /// Create a new guard from resource quotas.
    pub fn new(max_iterations: u32, max_tool_calls: u32, max_cost_usd: f64) -> Self {
        Self {
            max_iterations,
            current_iteration: 0,
            max_tool_calls,
            total_tool_calls: 0,
            tool_call_counts: HashMap::new(),
            consecutive_max_tokens: 0,
            usage: TokenUsage::default(),
            max_cost_usd,
            accumulated_cost_usd: 0.0,
            max_tokens_budget: None,
        }
    }

    /// Set the token budget (input+output cumulative limit).
    pub fn with_token_budget(mut self, budget: Option<u64>) -> Self {
        self.max_tokens_budget = budget;
        self
    }

    /// Check if another iteration is allowed.
    pub fn check_iteration(&mut self) -> LoopVerdict {
        self.current_iteration += 1;

        if self.current_iteration > self.max_iterations {
            return LoopVerdict::CircuitBreak(format!(
                "max iterations reached ({})",
                self.max_iterations
            ));
        }

        // Precision loss acceptable: max_iterations is small enough for f64
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_lossless)]
        let threshold = (self.max_iterations as f64 * WARN_ITERATION_FRACTION) as u32;
        if self.current_iteration >= threshold {
            return LoopVerdict::Warn(format!(
                "iteration {}/{} ({}% of budget)",
                self.current_iteration,
                self.max_iterations,
                self.current_iteration * 100 / self.max_iterations
            ));
        }

        LoopVerdict::Allow
    }

    /// Check if a tool call is allowed (ping-pong detection).
    pub fn check_tool_call(&mut self, tool_name: &str, input: &serde_json::Value) -> LoopVerdict {
        self.total_tool_calls += 1;

        if self.total_tool_calls > self.max_tool_calls {
            return LoopVerdict::CircuitBreak(format!(
                "max tool calls reached ({})",
                self.max_tool_calls
            ));
        }

        let hash = fx_hash_tool_call(tool_name, input);
        let count = self.tool_call_counts.entry(hash).or_insert(0);
        *count += 1;

        if *count >= PINGPONG_THRESHOLD {
            return LoopVerdict::Block(format!(
                "ping-pong detected: tool '{tool_name}' called \
                 {count} times with same input"
            ));
        }

        LoopVerdict::Allow
    }

    /// Record a `MaxTokens` stop reason. Returns `CircuitBreak` if
    /// consecutive truncations exceed threshold.
    pub fn record_max_tokens(&mut self) -> LoopVerdict {
        self.consecutive_max_tokens += 1;
        if self.consecutive_max_tokens >= MAX_CONSECUTIVE_TRUNCATION {
            LoopVerdict::CircuitBreak(format!(
                "{MAX_CONSECUTIVE_TRUNCATION} consecutive MaxTokens responses"
            ))
        } else {
            LoopVerdict::Allow
        }
    }

    /// Reset consecutive `MaxTokens` counter (on `EndTurn` or `ToolUse`).
    pub fn reset_max_tokens(&mut self) {
        self.consecutive_max_tokens = 0;
    }

    /// Record token usage from a completion. Returns verdict
    /// based on token budget (if configured).
    pub fn record_usage(&mut self, usage: &TokenUsage) -> LoopVerdict {
        self.usage.accumulate(usage);
        self.check_token_budget()
    }

    /// Check cumulative token usage against budget.
    fn check_token_budget(&self) -> LoopVerdict {
        let Some(budget) = self.max_tokens_budget else {
            return LoopVerdict::Allow;
        };
        let total = self.usage.input_tokens + self.usage.output_tokens;
        if total > budget {
            return LoopVerdict::CircuitBreak(format!(
                "token budget exhausted: {total} > {budget}"
            ));
        }
        let threshold = (budget as f64 * WARN_ITERATION_FRACTION) as u64;
        if total >= threshold {
            return LoopVerdict::Warn(format!(
                "token usage {total}/{budget} ({}% of budget)",
                total * 100 / budget
            ));
        }
        LoopVerdict::Allow
    }

    /// Record estimated cost and check budget.
    #[cfg_attr(
        feature = "agents-contracts",
        provable_contracts_macros::contract("agent-loop-v1", equation = "guard_budget")
    )]
    pub fn record_cost(&mut self, cost_usd: f64) -> LoopVerdict {
        self.accumulated_cost_usd += cost_usd;
        if self.max_cost_usd > 0.0 && self.accumulated_cost_usd > self.max_cost_usd {
            LoopVerdict::CircuitBreak(format!(
                "cost budget exceeded: ${:.4} > ${:.4}",
                self.accumulated_cost_usd, self.max_cost_usd
            ))
        } else {
            LoopVerdict::Allow
        }
    }

    /// Get accumulated usage.
    pub fn usage(&self) -> &TokenUsage {
        &self.usage
    }

    /// Get current iteration count.
    pub fn current_iteration(&self) -> u32 {
        self.current_iteration
    }

    /// Get total tool calls made.
    pub fn total_tool_calls(&self) -> u32 {
        self.total_tool_calls
    }
}

/// `FxHash` a tool call for ping-pong detection.
///
/// Uses a simple multiplicative hash (non-cryptographic) — we only
/// need collision resistance across ~50 values, not security.
fn fx_hash_tool_call(tool_name: &str, input: &serde_json::Value) -> u64 {
    let mut hasher = FxHasher::default();
    tool_name.hash(&mut hasher);
    let input_str = input.to_string();
    input_str.hash(&mut hasher);
    hasher.finish()
}

/// Minimal `FxHash` implementation (no external dependency).
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.hash = (self.hash.rotate_left(5) ^ u64::from(byte)).wrapping_mul(FX_SEED);
        }
    }
}

#[cfg(test)]
#[path = "guard_tests.rs"]
mod tests;
