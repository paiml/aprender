
/// Statistics for multi-request scheduler (PARITY-034)
#[cfg(feature = "gpu")]
pub struct MultiRequestStats {
    /// Total requests submitted
    pub requests_submitted: u64,
    /// Total requests completed
    pub requests_completed: u64,
    /// Total tokens generated
    pub tokens_generated: u64,
    /// Batch iterations performed
    pub batch_iterations: u64,
    /// Current pending requests
    pub pending_requests: usize,
    /// Current active requests
    pub active_requests: usize,
    /// Average batch size
    pub avg_batch_size: f64,
}

// =============================================================================
// PARITY-035: Chunked Prefill for Long Contexts (IMP-320)
// =============================================================================
//
// Enables streaming prompt processing by breaking long prefills into chunks.
// Key optimization for TTFT (Time to First Token) with long contexts.
//
// Architecture:
// - Prompt is split into chunks (default 512 tokens)
// - Each chunk processes incrementally, updating KV cache
// - First token can be generated after first chunk completes
// - Total prefill time is spread across chunks
// =============================================================================

/// Configuration for chunked prefill
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Chunk size in tokens (default: 512)
    pub chunk_size: usize,
    /// Maximum context length (default: 8192)
    pub max_context: usize,
    /// Whether to yield after each chunk for streaming
    pub stream_chunks: bool,
}

#[cfg(feature = "gpu")]
impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            max_context: 8192,
            stream_chunks: true,
        }
    }
}

#[cfg(feature = "gpu")]
impl ChunkedPrefillConfig {
    /// Create config with custom chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            ..Default::default()
        }
    }
}

/// Progress report for a single chunk
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ChunkProgress {
    /// Chunk index (0-based)
    pub chunk_idx: usize,
    /// Total chunks
    pub total_chunks: usize,
    /// Tokens processed so far
    pub tokens_processed: usize,
    /// Total tokens to process
    pub total_tokens: usize,
    /// Time for this chunk (ms)
    pub chunk_time_ms: f64,
    /// Cumulative time so far (ms)
    pub cumulative_time_ms: f64,
}

/// Chunked prefill processor for long context handling
#[cfg(feature = "gpu")]
pub struct ChunkedPrefill {
    /// Configuration
    config: ChunkedPrefillConfig,
    /// Chunks created from prompt
    chunks: Vec<Vec<u32>>,
    /// Current chunk being processed
    current_chunk: usize,
    /// Tokens processed so far
    tokens_processed: usize,
    /// Start time for timing
    start_time: Option<std::time::Instant>,
    /// Timing for each chunk
    chunk_times_ms: Vec<f64>,
}

#[cfg(feature = "gpu")]
impl ChunkedPrefill {
    /// Create new chunked prefill from prompt tokens
    pub fn new(prompt_tokens: &[u32], config: ChunkedPrefillConfig) -> Self {
        let chunks: Vec<Vec<u32>> = prompt_tokens
            .chunks(config.chunk_size)
            .map(<[u32]>::to_vec)
            .collect();

        Self {
            config,
            chunks,
            current_chunk: 0,
            tokens_processed: 0,
            start_time: None,
            chunk_times_ms: Vec::new(),
        }
    }

    /// Get total number of chunks
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Get total tokens
    pub fn total_tokens(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum()
    }

    /// Check if there are more chunks to process
    pub fn has_more_chunks(&self) -> bool {
        self.current_chunk < self.chunks.len()
    }

    /// Get the next chunk to process
    ///
    /// Returns None if all chunks are processed
    pub fn next_chunk(&mut self) -> Option<&[u32]> {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        if self.current_chunk < self.chunks.len() {
            let chunk = &self.chunks[self.current_chunk];
            Some(chunk.as_slice())
        } else {
            None
        }
    }

    /// Mark current chunk as complete
    pub fn complete_chunk(&mut self, chunk_time_ms: f64) {
        if self.current_chunk < self.chunks.len() {
            self.tokens_processed += self.chunks[self.current_chunk].len();
            self.chunk_times_ms.push(chunk_time_ms);
            self.current_chunk += 1;
        }
    }

    /// Get progress after completing a chunk
    pub fn progress(&self) -> ChunkProgress {
        let cumulative_time_ms: f64 = self.chunk_times_ms.iter().sum();

        ChunkProgress {
            chunk_idx: self.current_chunk.saturating_sub(1),
            total_chunks: self.chunks.len(),
            tokens_processed: self.tokens_processed,
            total_tokens: self.total_tokens(),
            chunk_time_ms: self.chunk_times_ms.last().copied().unwrap_or(0.0),
            cumulative_time_ms,
        }
    }

    /// Get estimated time to first token (after first chunk)
    pub fn estimated_ttft_ms(&self) -> f64 {
        if let Some(first_chunk_time) = self.chunk_times_ms.first() {
            *first_chunk_time
        } else {
            // Estimate based on chunk size and typical throughput
            let tokens = self.chunks.first().map_or(0, Vec::len);
            // Conservative estimate: 0.5ms per token for prefill
            tokens as f64 * 0.5
        }
    }

    /// Get statistics after completion
    pub fn stats(&self) -> ChunkedPrefillStats {
        let total_time_ms: f64 = self.chunk_times_ms.iter().sum();
        let total_tokens = self.total_tokens();
        let avg_chunk_time_ms = if !self.chunk_times_ms.is_empty() {
            total_time_ms / self.chunk_times_ms.len() as f64
        } else {
            0.0
        };

        ChunkedPrefillStats {
            total_chunks: self.chunks.len(),
            chunk_size: self.config.chunk_size,
            total_tokens,
            total_time_ms,
            avg_chunk_time_ms,
            ttft_ms: self.estimated_ttft_ms(),
            tokens_per_second: if total_time_ms > 0.0 {
                total_tokens as f64 / (total_time_ms / 1000.0)
            } else {
                0.0
            },
        }
    }
}

/// Statistics for chunked prefill
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ChunkedPrefillStats {
    /// Total chunks processed
    pub total_chunks: usize,
    /// Chunk size used
    pub chunk_size: usize,
    /// Total tokens processed
    pub total_tokens: usize,
    /// Total time (ms)
    pub total_time_ms: f64,
    /// Average time per chunk (ms)
    pub avg_chunk_time_ms: f64,
    /// Time to first token (ms)
    pub ttft_ms: f64,
    /// Prefill throughput (tokens/sec)
    pub tokens_per_second: f64,
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[path = "batch_scheduler_tests.rs"]
mod batch_scheduler_tests;
