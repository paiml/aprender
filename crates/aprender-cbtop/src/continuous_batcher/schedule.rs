//! Scheduling types for continuous batching.

use std::fmt;
use std::time::Instant;

use crate::paged_kv::SeqId;

use super::request::Token;

/// Scheduling policy for request prioritization.
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingPolicy {
    /// First-come, first-served
    FCFS,
    /// Shortest job first (by estimated tokens)
    SJF,
    /// Priority-based (API tiers)
    Priority { preempt_enabled: bool },
    /// Fair share (equal GPU time per user)
    FairShare,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        SchedulingPolicy::FCFS
    }
}

impl fmt::Display for SchedulingPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedulingPolicy::FCFS => write!(f, "FCFS"),
            SchedulingPolicy::SJF => write!(f, "SJF"),
            SchedulingPolicy::Priority { preempt_enabled } => {
                write!(f, "Priority(preempt={})", preempt_enabled)
            }
            SchedulingPolicy::FairShare => write!(f, "FairShare"),
        }
    }
}

/// Batch schedule result.
#[derive(Debug, Clone)]
pub struct BatchSchedule {
    /// Sequence IDs in this batch
    pub sequence_ids: Vec<SeqId>,
    /// Number of sequences in batch
    pub batch_size: usize,
    /// Total tokens to process
    pub total_tokens: usize,
    /// Prefill sequences (first token)
    pub prefill_count: usize,
    /// Decode sequences (continuation)
    pub decode_count: usize,
}

impl BatchSchedule {
    /// Create empty schedule.
    pub fn empty() -> Self {
        Self {
            sequence_ids: Vec::new(),
            batch_size: 0,
            total_tokens: 0,
            prefill_count: 0,
            decode_count: 0,
        }
    }

    /// Check if schedule is empty.
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }
}

/// Token output from a decode step.
#[derive(Debug, Clone)]
pub struct TokenOutput {
    /// Sequence ID
    pub seq_id: SeqId,
    /// Generated token
    pub token: Token,
    /// Is EOS token?
    pub is_eos: bool,
}

/// Batcher statistics.
#[derive(Debug, Clone, Default)]
pub struct BatcherStats {
    /// Total tokens processed
    pub total_tokens: u64,
    /// Total requests completed
    pub total_requests: u64,
    /// Total preemptions
    pub total_preemptions: u64,
    /// Total swaps (CPU<->GPU)
    pub total_swaps: u64,
    /// Processing start time
    pub start_time: Option<Instant>,
}

impl BatcherStats {
    /// Calculate throughput (tokens/sec).
    pub fn throughput(&self) -> f64 {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                return self.total_tokens as f64 / elapsed;
            }
        }
        0.0
    }
}
