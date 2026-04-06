//! Chunked Prefill Scheduler Component
//!
//! Implements chunked prefill per Sarathi-Serve/vLLM for reduced TTFT variance.
//! Extracted from scheduler/mod.rs (PMAT-802).

#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// CHUNKED PREFILL (per Sarathi-Serve / vLLM)
// ============================================================================
//
// Chunked prefill breaks long prompt processing into smaller chunks, allowing
// decode steps for other requests to interleave. This reduces TTFT variance
// and prevents head-of-line blocking from long prompts.
//
// Reference: Agrawal et al. (2024) "Sarathi-Serve: Large Language Model Inference
// with Chunked Prefill and Decode Disaggregation"

/// Configuration for chunked prefill
///
/// Controls how long prompts are split into chunks for interleaved processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkedPrefillConfig {
    /// Enable chunked prefill (default: true)
    pub enabled: bool,
    /// Maximum tokens per prefill chunk (default: 512)
    pub chunk_size: usize,
    /// Minimum prompt length to trigger chunking (default: 256)
    pub min_prompt_length: usize,
    /// Allow decode interleaving between chunks (default: true)
    pub allow_decode_interleave: bool,
    /// Priority boost for partially-prefilled sequences (default: true)
    pub boost_partial_prefill: bool,
    /// Maximum chunks before forcing completion (default: 16)
    pub max_chunks: usize,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            chunk_size: 512,
            min_prompt_length: 256,
            allow_decode_interleave: true,
            boost_partial_prefill: true,
            max_chunks: 16,
        }
    }
}

impl ChunkedPrefillConfig {
    /// Create config with specified chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Disable chunked prefill
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create config for low-latency scenarios (smaller chunks)
    pub fn low_latency() -> Self {
        Self {
            enabled: true,
            chunk_size: 128,
            min_prompt_length: 64,
            allow_decode_interleave: true,
            boost_partial_prefill: true,
            max_chunks: 32,
        }
    }

    /// Create config for high-throughput scenarios (larger chunks)
    pub fn high_throughput() -> Self {
        Self {
            enabled: true,
            chunk_size: 1024,
            min_prompt_length: 512,
            allow_decode_interleave: false,
            boost_partial_prefill: false,
            max_chunks: 8,
        }
    }
}

/// State of chunked prefill for a sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkedPrefillState {
    /// Sequence ID
    pub seq_id: u64,
    /// Total prompt tokens
    pub total_tokens: usize,
    /// Tokens processed so far
    pub processed_tokens: usize,
    /// Current chunk index
    pub current_chunk: usize,
    /// Total number of chunks
    pub total_chunks: usize,
    /// Start time of prefill (for latency tracking)
    pub start_time_ms: u64,
    /// Time spent on each chunk (milliseconds)
    pub chunk_latencies: Vec<u64>,
}

impl ChunkedPrefillState {
    /// Create new chunked prefill state
    pub fn new(seq_id: u64, total_tokens: usize, chunk_size: usize) -> Self {
        let total_chunks = total_tokens.div_ceil(chunk_size);
        Self {
            seq_id,
            total_tokens,
            processed_tokens: 0,
            current_chunk: 0,
            total_chunks,
            start_time_ms: 0,
            chunk_latencies: Vec::with_capacity(total_chunks),
        }
    }

    /// Get next chunk of tokens to process
    pub fn next_chunk(&self, chunk_size: usize) -> std::ops::Range<usize> {
        let start = self.processed_tokens;
        let end = (start + chunk_size).min(self.total_tokens);
        start..end
    }

    /// Advance to next chunk
    pub fn advance(&mut self, tokens_processed: usize, latency_ms: u64) {
        self.processed_tokens += tokens_processed;
        self.current_chunk += 1;
        self.chunk_latencies.push(latency_ms);
    }

    /// Check if prefill is complete
    pub fn is_complete(&self) -> bool {
        self.processed_tokens >= self.total_tokens
    }

    /// Get progress as percentage
    pub fn progress(&self) -> f64 {
        if self.total_tokens == 0 {
            100.0
        } else {
            (self.processed_tokens as f64 / self.total_tokens as f64) * 100.0
        }
    }

    /// Remaining tokens to prefill
    pub fn remaining_tokens(&self) -> usize {
        self.total_tokens.saturating_sub(self.processed_tokens)
    }

    /// Average chunk latency
    pub fn avg_chunk_latency_ms(&self) -> f64 {
        if self.chunk_latencies.is_empty() {
            0.0
        } else {
            self.chunk_latencies.iter().sum::<u64>() as f64 / self.chunk_latencies.len() as f64
        }
    }
}

/// Statistics for chunked prefill scheduler
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkedPrefillStats {
    /// Total sequences that used chunked prefill
    pub chunked_sequences: u64,
    /// Total sequences that bypassed chunking (short prompts)
    pub bypassed_sequences: u64,
    /// Total chunks processed
    pub chunks_processed: u64,
    /// Total decode interleaves (decode steps between prefill chunks)
    pub decode_interleaves: u64,
    /// Sum of all chunk latencies (for averaging)
    pub total_chunk_latency_ms: u64,
    /// Maximum single chunk latency
    pub max_chunk_latency_ms: u64,
    /// Total tokens saved from prefix cache hits during chunked prefill
    pub prefix_cache_hits: u64,
}

impl ChunkedPrefillStats {
    /// Average chunk latency
    pub fn avg_chunk_latency_ms(&self) -> f64 {
        if self.chunks_processed == 0 {
            0.0
        } else {
            self.total_chunk_latency_ms as f64 / self.chunks_processed as f64
        }
    }

    /// Chunking rate (what % of sequences used chunking)
    pub fn chunking_rate(&self) -> f64 {
        let total = self.chunked_sequences + self.bypassed_sequences;
        if total == 0 {
            0.0
        } else {
            self.chunked_sequences as f64 / total as f64
        }
    }
}

/// Chunked prefill scheduler
///
/// Manages the chunked processing of long prompts, interleaving with decode
/// operations for improved TTFT across all requests.
pub struct ChunkedPrefillScheduler {
    /// Configuration
    config: ChunkedPrefillConfig,
    /// Active chunked prefill states (seq_id -> state)
    active_prefills: HashMap<u64, ChunkedPrefillState>,
    /// Queue of sequences waiting for prefill
    prefill_queue: VecDeque<u64>,
    /// Statistics
    stats: ChunkedPrefillStats,
    /// Next sequence ID
    next_seq_id: u64,
}

impl ChunkedPrefillScheduler {
    /// Create new chunked prefill scheduler
    pub fn new(config: ChunkedPrefillConfig) -> Self {
        Self {
            config,
            active_prefills: HashMap::new(),
            prefill_queue: VecDeque::new(),
            stats: ChunkedPrefillStats::default(),
            next_seq_id: 0,
        }
    }

    /// Submit a new sequence for prefill
    ///
    /// Returns the sequence ID and whether it will use chunked prefill.
    pub fn submit(&mut self, prompt_tokens: usize) -> (u64, bool) {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let use_chunking = self.config.enabled && prompt_tokens >= self.config.min_prompt_length;

        if use_chunking {
            let state = ChunkedPrefillState::new(seq_id, prompt_tokens, self.config.chunk_size);
            self.active_prefills.insert(seq_id, state);
            self.prefill_queue.push_back(seq_id);
            self.stats.chunked_sequences += 1;
        } else {
            self.stats.bypassed_sequences += 1;
        }

        (seq_id, use_chunking)
    }

    /// Get the next chunk to process
    ///
    /// Returns (seq_id, token_range) for the next prefill chunk, or None if
    /// all prefills are complete or queue is empty.
    pub fn next_chunk(&mut self) -> Option<(u64, std::ops::Range<usize>)> {
        // Find next sequence with remaining prefill work
        while let Some(&seq_id) = self.prefill_queue.front() {
            if let Some(state) = self.active_prefills.get(&seq_id) {
                if !state.is_complete() {
                    let range = state.next_chunk(self.config.chunk_size);
                    return Some((seq_id, range));
                }
            }
            // Remove completed or missing sequences from queue
            self.prefill_queue.pop_front();
        }
        None
    }

    /// Record completion of a prefill chunk
    pub fn complete_chunk(&mut self, seq_id: u64, tokens_processed: usize, latency_ms: u64) {
        if let Some(state) = self.active_prefills.get_mut(&seq_id) {
            state.advance(tokens_processed, latency_ms);
            self.stats.chunks_processed += 1;
            self.stats.total_chunk_latency_ms += latency_ms;
            self.stats.max_chunk_latency_ms = self.stats.max_chunk_latency_ms.max(latency_ms);

            // If complete, clean up
            if state.is_complete() {
                // Move to back of queue or remove
                if let Some(pos) = self.prefill_queue.iter().position(|&id| id == seq_id) {
                    self.prefill_queue.remove(pos);
                }
            } else if self.config.boost_partial_prefill {
                // Keep at front for priority
            } else {
                // Move to back of queue for round-robin
                if let Some(pos) = self.prefill_queue.iter().position(|&id| id == seq_id) {
                    self.prefill_queue.remove(pos);
                    self.prefill_queue.push_back(seq_id);
                }
            }
        }
    }

    /// Record a decode interleave (decode step between prefill chunks)
    pub fn record_decode_interleave(&mut self) {
        self.stats.decode_interleaves += 1;
    }

    /// Check if we should allow decode interleaving
    pub fn should_interleave_decode(&self) -> bool {
        self.config.allow_decode_interleave && !self.prefill_queue.is_empty()
    }

    /// Get state for a sequence
    pub fn get_state(&self, seq_id: u64) -> Option<&ChunkedPrefillState> {
        self.active_prefills.get(&seq_id)
    }

    /// Check if sequence has pending prefill work
    pub fn has_pending_prefill(&self, seq_id: u64) -> bool {
        self.active_prefills
            .get(&seq_id)
            .is_some_and(|s| !s.is_complete())
    }

    /// Remove completed sequence
    pub fn remove(&mut self, seq_id: u64) -> Option<ChunkedPrefillState> {
        if let Some(pos) = self.prefill_queue.iter().position(|&id| id == seq_id) {
            self.prefill_queue.remove(pos);
        }
        self.active_prefills.remove(&seq_id)
    }

    /// Number of sequences with pending prefill
    pub fn pending_count(&self) -> usize {
        self.active_prefills
            .values()
            .filter(|s| !s.is_complete())
            .count()
    }

    /// Total prefill queue length
    pub fn queue_len(&self) -> usize {
        self.prefill_queue.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &ChunkedPrefillStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &ChunkedPrefillConfig {
        &self.config
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.active_prefills.clear();
        self.prefill_queue.clear();
    }

    /// Record prefix cache hit during prefill
    pub fn record_prefix_cache_hit(&mut self, tokens_saved: usize) {
        self.stats.prefix_cache_hits += tokens_saved as u64;
    }
}

impl Default for ChunkedPrefillScheduler {
    fn default() -> Self {
        Self::new(ChunkedPrefillConfig::default())
    }
}

include!("chunked_prefill_config_default.rs");
