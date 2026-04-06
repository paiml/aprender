//! Continuous Batching Scheduler
//!
//! Per spec §8: Implements continuous batching for LLM serving based on vLLM.
//! Reference: [8] Yu et al. (2022) "Orca: A Distributed Serving System for Transformer-Based Generative Models"
//!
//! ## Key Features
//!
//! - **Iteration-Level Scheduling**: New requests join batch at any iteration
//! - **Preemption**: Low-priority requests can be preempted for high-priority
//! - **Memory-Aware**: Respects KV cache limits when scheduling
//! - **Fair Queuing**: Prevents starvation of long requests
//!
//! ## Scheduling Algorithm
//!
//! ```text
//! while running:
//!   1. Check for completed sequences (EOS or max_tokens)
//!   2. Preempt sequences if memory pressure
//!   3. Schedule waiting sequences if space available
//!   4. Run one iteration of generation for batch
//! ```

// Module-level clippy allows
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_wraps)] // Result wrapping for API consistency
#![allow(clippy::derivable_impls)] // Manual impl for documentation clarity
#![allow(clippy::option_if_let_else)] // map_or is more readable

use crate::paged_kv::{PagedCacheError, PagedKvCache, SeqId};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::time::Instant;
use thiserror::Error;

// PMAT-802: Extracted modules
mod chunked_prefill;
mod types;
pub use chunked_prefill::{
    ChunkedPrefillConfig, ChunkedPrefillScheduler, ChunkedPrefillState, ChunkedPrefillStats,
};
pub use types::{Priority, SchedulerStats, SequenceState};

/// Error type for scheduler operations
#[derive(Debug, Error)]
pub enum SchedulerError {
    /// Queue is full
    #[error("Request queue full: capacity {capacity}")]
    QueueFull {
        /// Queue capacity
        capacity: usize,
    },

    /// Request not found
    #[error("Request not found: {0}")]
    RequestNotFound(u64),

    /// KV cache error
    #[error("KV cache error: {0}")]
    CacheError(#[from] PagedCacheError),

    /// Invalid state
    #[error("Invalid scheduler state: {0}")]
    InvalidState(String),
}

/// Generation request
#[derive(Debug, Clone)]
pub struct SchedulerRequest {
    /// Unique request ID
    pub request_id: u64,
    /// Input token IDs
    pub input_ids: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Priority level
    pub priority: Priority,
    /// Arrival time
    pub arrival_time: Instant,
    /// Sequence ID (assigned when scheduled)
    pub seq_id: Option<SeqId>,
    /// Current state
    pub state: SequenceState,
    /// Generated tokens so far
    pub generated_tokens: Vec<u32>,
    /// Number of decode iterations
    pub iterations: usize,
}

impl SchedulerRequest {
    /// Create a new request
    pub fn new(request_id: u64, input_ids: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            request_id,
            input_ids,
            max_tokens,
            priority: Priority::default(),
            arrival_time: Instant::now(),
            seq_id: None,
            state: SequenceState::Waiting,
            generated_tokens: Vec::new(),
            iterations: 0,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Total tokens (input + generated)
    pub fn total_tokens(&self) -> usize {
        self.input_ids.len() + self.generated_tokens.len()
    }

    /// Remaining tokens to generate
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.generated_tokens.len())
    }

    /// Check if generation is complete
    pub fn is_complete(&self, eos_token: u32) -> bool {
        self.generated_tokens.len() >= self.max_tokens
            || self.generated_tokens.last() == Some(&eos_token)
    }

    /// Time waiting in queue
    pub fn wait_time(&self) -> std::time::Duration {
        self.arrival_time.elapsed()
    }
}

/// Priority-aware entry for the waiting queue
#[derive(Debug)]
struct PriorityEntry {
    priority: Priority,
    arrival_time: Instant,
    request_id: u64,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then earlier arrival
        match self.priority.cmp(&other.priority) {
            std::cmp::Ordering::Equal => other.arrival_time.cmp(&self.arrival_time),
            other => other,
        }
    }
}

/// Scheduler output for one iteration
#[derive(Debug, Clone, Default)]
pub struct SchedulerOutput {
    /// Sequences to run this iteration
    pub scheduled_seq_ids: Vec<SeqId>,
    /// Request IDs for scheduled sequences
    pub scheduled_request_ids: Vec<u64>,
    /// Sequences that were preempted
    pub preempted_seq_ids: Vec<SeqId>,
    /// Sequences that completed
    pub completed_request_ids: Vec<u64>,
    /// Number of tokens in prefill phase
    pub num_prefill_tokens: usize,
    /// Number of tokens in decode phase
    pub num_decode_tokens: usize,
}

impl SchedulerOutput {
    /// Total tokens scheduled
    pub fn total_tokens(&self) -> usize {
        self.num_prefill_tokens + self.num_decode_tokens
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.scheduled_seq_ids.is_empty()
    }
}

/// Continuous batching scheduler
pub struct Scheduler {
    /// All requests by ID
    requests: HashMap<u64, SchedulerRequest>,
    /// Waiting queue (priority-ordered)
    waiting_queue: BinaryHeap<PriorityEntry>,
    /// Running sequences
    running: Vec<u64>,
    /// Preempted sequences (can be resumed)
    preempted: VecDeque<u64>,
    /// Maximum batch size
    max_batch_size: usize,
    /// Maximum queue size
    max_queue_size: usize,
    /// Maximum tokens per batch
    max_tokens_per_batch: usize,
    /// Next request ID
    next_request_id: u64,
    /// Statistics
    stats: SchedulerStats,
    /// Total wait time for completed requests (for averaging)
    total_wait_time_ms: f64,
}

include!("mod_max_scheduler.rs");
include!("mod_num_slots_idle.rs");
include!("mod_max_default_batch.rs");
include!("mod_dynamic_scheduler.rs");
