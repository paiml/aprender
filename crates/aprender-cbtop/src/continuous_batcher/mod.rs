//! ContinuousBatcher Implementation (PMAT-015)
//!
//! Implements continuous batching for LLM inference per cbtop spec §19.
//!
//! # Overview
//!
//! Continuous batching processes inference requests dynamically, allowing
//! new requests to join and completed requests to leave mid-batch.
//!
//! # Citations
//!
//! - [Yu et al. 2022] "ORCA: Continuous Batching for LLM Inference" OSDI
//! - [Leviathan et al. 2023] "Fast Inference from Transformers via Speculative Decoding" ICML
//! - [Chen et al. 2023] "Accelerating LLM Decoding with Speculative Sampling" arXiv

mod request;
mod schedule;
mod speculative;

pub use request::{InferenceRequest, Priority, SequenceGroup, Token};
pub use schedule::{BatchSchedule, BatcherStats, SchedulingPolicy, TokenOutput};
pub use speculative::{ExponentialMovingAverage, SpeculativeDecoder, SpeculativeOutput};

use std::collections::VecDeque;
use std::fmt;
use std::time::Instant;

use crate::paged_kv::SeqId;

/// Continuous batching scheduler for LLM inference.
///
/// Processes requests as they arrive without waiting for batch completion.
/// Based on ORCA continuous batching algorithm.
#[derive(Debug)]
pub struct ContinuousBatcher {
    /// Maximum batch size (GPU memory limited)
    max_batch_size: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Active sequences in current batch
    running: Vec<SequenceGroup>,
    /// Waiting queue (sorted by policy)
    waiting: VecDeque<SequenceGroup>,
    /// Swapped sequences (offloaded to CPU)
    swapped: Vec<SequenceGroup>,
    /// Scheduling policy
    policy: SchedulingPolicy,
    /// Statistics
    stats: BatcherStats,
    /// Memory threshold for preemption (0.0-1.0)
    memory_threshold: f64,
}

impl ContinuousBatcher {
    /// Create a new continuous batcher.
    pub fn new(max_batch_size: usize, max_seq_len: usize) -> Self {
        Self {
            max_batch_size,
            max_seq_len,
            running: Vec::new(),
            waiting: VecDeque::new(),
            swapped: Vec::new(),
            policy: SchedulingPolicy::default(),
            stats: BatcherStats { start_time: Some(Instant::now()), ..Default::default() },
            memory_threshold: 0.9,
        }
    }

    /// Set scheduling policy.
    pub fn with_policy(mut self, policy: SchedulingPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set memory threshold for preemption.
    pub fn with_memory_threshold(mut self, threshold: f64) -> Self {
        self.memory_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Get scheduling policy.
    pub fn policy(&self) -> &SchedulingPolicy {
        &self.policy
    }

    /// Get max batch size.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Get max sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get number of running sequences.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Get number of waiting sequences.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Get number of swapped sequences.
    pub fn swapped_count(&self) -> usize {
        self.swapped.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> &BatcherStats {
        &self.stats
    }

    /// Current throughput (tokens/sec).
    pub fn throughput(&self) -> f64 {
        self.stats.throughput()
    }

    /// Add new inference request.
    pub fn add_request(&mut self, request: InferenceRequest) {
        let seq_group = SequenceGroup::new(request);
        self.insert_waiting(seq_group);
    }

    /// Insert into waiting queue according to policy.
    fn insert_waiting(&mut self, seq_group: SequenceGroup) {
        match &self.policy {
            SchedulingPolicy::FCFS => {
                // Add to back of queue
                self.waiting.push_back(seq_group);
            }
            SchedulingPolicy::SJF => {
                // Insert sorted by estimated tokens (shortest first)
                let insert_idx = self
                    .waiting
                    .iter()
                    .position(|s| s.request.estimated_tokens > seq_group.request.estimated_tokens)
                    .unwrap_or(self.waiting.len());
                self.waiting.insert(insert_idx, seq_group);
            }
            SchedulingPolicy::Priority { .. } => {
                // Insert sorted by priority (highest first)
                let insert_idx = self
                    .waiting
                    .iter()
                    .position(|s| s.request.priority < seq_group.request.priority)
                    .unwrap_or(self.waiting.len());
                self.waiting.insert(insert_idx, seq_group);
            }
            SchedulingPolicy::FairShare => {
                // Simple round-robin for now
                self.waiting.push_back(seq_group);
            }
        }
    }

    /// Schedule next iteration batch.
    pub fn schedule(&mut self) -> BatchSchedule {
        // Remove finished sequences from running
        self.running.retain(|s| !s.is_finished);

        // Calculate available slots
        let _available = self.max_batch_size.saturating_sub(self.running.len());

        // Promote from waiting to running
        let mut prefill_count = 0;
        while !self.waiting.is_empty() && self.running.len() < self.max_batch_size {
            if let Some(seq_group) = self.waiting.pop_front() {
                // Check if sequence fits
                if seq_group.total_tokens() <= self.max_seq_len {
                    prefill_count += 1;
                    self.running.push(seq_group);
                } else {
                    // Too long, put back
                    self.waiting.push_front(seq_group);
                    break;
                }
            }
        }

        // Swap in from swapped if space available
        while !self.swapped.is_empty() && self.running.len() < self.max_batch_size {
            if let Some(seq_group) = self.swapped.pop() {
                self.running.push(seq_group);
                self.stats.total_swaps += 1;
            }
        }

        // Build schedule
        let sequence_ids: Vec<SeqId> = self.running.iter().map(|s| s.request.id).collect();
        let total_tokens: usize = self.running.iter().map(|s| s.total_tokens()).sum();
        let decode_count = self.running.len() - prefill_count;

        BatchSchedule {
            batch_size: sequence_ids.len(),
            sequence_ids,
            total_tokens,
            prefill_count,
            decode_count,
        }
    }

    /// Process outputs from a decode step.
    pub fn process_outputs(&mut self, outputs: Vec<TokenOutput>) {
        for output in outputs {
            // Find the sequence and add the token
            if let Some(seq_group) = self.running.iter_mut().find(|s| s.request.id == output.seq_id)
            {
                seq_group.add_token(output.token);
                self.stats.total_tokens += 1;

                // Check for EOS
                if output.is_eos {
                    seq_group.finish();
                    self.stats.total_requests += 1;
                }
            }
        }
    }

    /// Preempt sequences under memory pressure.
    pub fn preempt(&mut self, num_to_preempt: usize) -> Vec<SeqId> {
        let mut preempted = Vec::new();

        // Preempt from running (lowest priority or longest)
        for _ in 0..num_to_preempt {
            if self.running.is_empty() {
                break;
            }

            // Find victim (longest sequence for simplicity)
            let victim_idx = self
                .running
                .iter()
                .enumerate()
                .max_by_key(|(_, s)| s.total_tokens())
                .map(|(i, _)| i);

            if let Some(idx) = victim_idx {
                let victim = self.running.remove(idx);
                preempted.push(victim.request.id);
                self.swapped.push(victim);
                self.stats.total_preemptions += 1;
            }
        }

        preempted
    }

    /// Check if preemption is needed (simulated memory pressure).
    pub fn needs_preemption(&self, current_utilization: f64) -> bool {
        current_utilization >= self.memory_threshold
            && !self.running.is_empty()
            && matches!(self.policy, SchedulingPolicy::Priority { preempt_enabled: true })
    }

    /// Get sequence by ID.
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&SequenceGroup> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.swapped.iter())
            .find(|s| s.request.id == seq_id)
    }

    /// Get all sequence IDs.
    pub fn all_sequence_ids(&self) -> Vec<SeqId> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.swapped.iter())
            .map(|s| s.request.id)
            .collect()
    }
}

impl fmt::Display for ContinuousBatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ContinuousBatcher")?;
        writeln!(
            f,
            "  Policy: {} | Max Batch: {} | Max Seq: {}",
            self.policy, self.max_batch_size, self.max_seq_len
        )?;
        writeln!(
            f,
            "  Running: {} | Waiting: {} | Swapped: {}",
            self.running.len(),
            self.waiting.len(),
            self.swapped.len()
        )?;
        writeln!(f, "  Throughput: {:.1} tok/s", self.throughput())?;
        writeln!(
            f,
            "  Stats: tokens={}, requests={}, preemptions={}, swaps={}",
            self.stats.total_tokens,
            self.stats.total_requests,
            self.stats.total_preemptions,
            self.stats.total_swaps
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
