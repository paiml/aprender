//! Inference request and sequence types for continuous batching.

use std::time::Instant;

use crate::paged_kv::SeqId;

/// Token type (simplified - u32 vocabulary index).
pub type Token = u32;

/// Request priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(pub u8);

impl Default for Priority {
    fn default() -> Self {
        Priority(128) // Middle priority
    }
}

/// Inference request.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request ID
    pub id: SeqId,
    /// Input tokens (prompt)
    pub input_tokens: Vec<Token>,
    /// Maximum output tokens to generate
    pub max_new_tokens: usize,
    /// Request priority
    pub priority: Priority,
    /// Arrival timestamp
    pub arrival_time: Instant,
    /// Estimated total tokens (input + output)
    pub estimated_tokens: usize,
}

impl InferenceRequest {
    /// Create a new inference request.
    pub fn new(id: SeqId, input_tokens: Vec<Token>, max_new_tokens: usize) -> Self {
        let estimated_tokens = input_tokens.len() + max_new_tokens;
        Self {
            id,
            input_tokens,
            max_new_tokens,
            priority: Priority::default(),
            arrival_time: Instant::now(),
            estimated_tokens,
        }
    }

    /// Create request with priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Input sequence length.
    pub fn input_len(&self) -> usize {
        self.input_tokens.len()
    }
}

/// Sequence group (request + generation state).
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    /// Original request
    pub request: InferenceRequest,
    /// Generated tokens so far
    pub output_tokens: Vec<Token>,
    /// Is generation complete?
    pub is_finished: bool,
    /// Last access timestamp (for LRU)
    pub last_access: Instant,
    /// Number of decode steps
    pub num_steps: usize,
}

impl SequenceGroup {
    /// Create new sequence group from request.
    pub fn new(request: InferenceRequest) -> Self {
        Self {
            request,
            output_tokens: Vec::new(),
            is_finished: false,
            last_access: Instant::now(),
            num_steps: 0,
        }
    }

    /// Total tokens (input + output so far).
    pub fn total_tokens(&self) -> usize {
        self.request.input_tokens.len() + self.output_tokens.len()
    }

    /// Remaining tokens to generate.
    pub fn remaining_tokens(&self) -> usize {
        self.request
            .max_new_tokens
            .saturating_sub(self.output_tokens.len())
    }

    /// Add generated token.
    pub fn add_token(&mut self, token: Token) {
        self.output_tokens.push(token);
        self.last_access = Instant::now();
        self.num_steps += 1;

        // Check if finished
        if self.output_tokens.len() >= self.request.max_new_tokens {
            self.is_finished = true;
        }
    }

    /// Mark as finished.
    pub fn finish(&mut self) {
        self.is_finished = true;
    }
}
