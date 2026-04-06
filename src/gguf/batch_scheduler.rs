//! Request Batching Infrastructure for GPU-Accelerated Inference
//!
//! Extracted from gguf_monolith.rs (PMAT-802) for vertical production partitioning.
//!
//! ## Contents
//!
//! - `BatchGenerationStats`: Statistics for batch generation capabilities
//! - `PendingRequest`, `RequestBatch`, `BatchRequestCollector`: Request batching
//! - `BatchingConfig`: Batching configuration
//! - `SlotState`, `ContinuousBatchScheduler`: Continuous batching
//! - `SpeculativeConfig`, `SpeculativeDecoder`: Speculative decoding
//! - `GpuBufferPool`, `GpuBufferPoolStats`: GPU buffer management
//! - `AsyncCommandQueue`, `CommandSlot`, `AsyncQueueStats`: Async command queue
//! - `PrefixCache`, `PrefixCacheEntry`, `PrefixCacheStats`: Prefix caching
//! - `MultiRequestState`, `MultiSchedulerRequest`, `SchedulingPolicy`, `MultiRequestScheduler`: Multi-request scheduling
//! - `ChunkedPrefillConfig`, `ChunkProgress`, `ChunkedPrefill`, `ChunkedPrefillStats`: Chunked prefill
//!
//! ## Feature Gate
//!
//! This entire module is gated behind `#[cfg(feature = "gpu")]`.

// Note: This module is feature-gated in mod.rs with #[cfg(feature = "gpu")]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]

#[cfg(feature = "gpu")]
use super::runtime::OwnedQuantizedKVCache;

/// Statistics for batch generation configuration
#[derive(Debug, Clone)]
pub struct BatchGenerationStats {
    /// Whether GPU cache is ready
    pub gpu_cache_ready: bool,
    /// Memory used by GPU cache in GB
    pub cache_memory_gb: f64,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Recommended batch size for GPU efficiency
    pub recommended_batch_size: usize,
    /// Maximum batch size before memory pressure
    pub max_batch_size: usize,
}

// ============================================================================
// PARITY-023: Request Batching Infrastructure
// ============================================================================

/// A pending request waiting to be batched (PARITY-023)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct PendingRequest {
    /// Request ID for tracking
    pub id: u64,
    /// Prompt tokens
    pub prompt: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Time when request was submitted
    pub submitted_at: std::time::Instant,
}

#[cfg(feature = "gpu")]
impl PendingRequest {
    /// Create a new pending request
    pub fn new(
        id: u64,
        prompt: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Self {
        Self {
            id,
            prompt,
            max_tokens,
            temperature,
            top_k,
            submitted_at: std::time::Instant::now(),
        }
    }

    /// Time spent waiting in queue
    pub fn wait_time(&self) -> std::time::Duration {
        self.submitted_at.elapsed()
    }
}

/// A batch of requests ready for processing (PARITY-023)
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct RequestBatch {
    /// Requests in this batch
    pub requests: Vec<PendingRequest>,
    /// When batch was formed
    pub formed_at: std::time::Instant,
}

#[cfg(feature = "gpu")]
impl RequestBatch {
    /// Create batch from requests
    pub fn new(requests: Vec<PendingRequest>) -> Self {
        Self {
            requests,
            formed_at: std::time::Instant::now(),
        }
    }

    /// Number of requests in batch
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Extract prompts for batch processing
    pub fn prompts(&self) -> Vec<Vec<u32>> {
        self.requests.iter().map(|r| r.prompt.clone()).collect()
    }

    /// Average wait time for requests in this batch
    pub fn avg_wait_time(&self) -> std::time::Duration {
        if self.requests.is_empty() {
            return std::time::Duration::ZERO;
        }
        let total: std::time::Duration = self.requests.iter().map(PendingRequest::wait_time).sum();
        total / self.requests.len() as u32
    }
}

/// Request batch collector with configurable thresholds (PARITY-023)
///
/// Collects incoming requests and forms batches when:
/// - Batch size reaches `batch_threshold`, OR
/// - Wait time exceeds `timeout_ms`
///
/// This enables efficient GPU utilization by batching multiple requests.
#[cfg(feature = "gpu")]
pub struct BatchRequestCollector {
    /// Pending requests
    pending: std::sync::Mutex<Vec<PendingRequest>>,
    /// Next request ID
    next_id: std::sync::atomic::AtomicU64,
    /// Batch size threshold (32 = GPU GEMM threshold from IMP-600)
    pub batch_threshold: usize,
    /// Maximum wait time before forcing batch formation (ms)
    pub timeout_ms: u64,
    /// Maximum batch size (memory limit)
    pub max_batch_size: usize,
}

#[cfg(feature = "gpu")]
impl BatchRequestCollector {
    /// Create new collector with default thresholds
    ///
    /// Default: batch_threshold=32, timeout_ms=50, max_batch_size=64
    pub fn new() -> Self {
        Self {
            pending: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
            batch_threshold: 32,
            timeout_ms: 50,
            max_batch_size: 64,
        }
    }

    /// Create collector with custom thresholds
    pub fn with_thresholds(batch_threshold: usize, timeout_ms: u64, max_batch_size: usize) -> Self {
        Self {
            pending: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
            batch_threshold,
            timeout_ms,
            max_batch_size,
        }
    }

    /// Submit a request to the collector
    ///
    /// Returns the request ID for tracking
    pub fn submit(
        &self,
        prompt: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> u64 {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let request = PendingRequest::new(id, prompt, max_tokens, temperature, top_k);

        let mut pending = self.pending.lock().expect("Mutex poisoned");
        pending.push(request);

        id
    }

    /// Check if batch is ready to be formed
    pub fn is_batch_ready(&self) -> bool {
        let pending = self.pending.lock().expect("Mutex poisoned");
        if pending.is_empty() {
            return false;
        }

        // Batch ready if threshold reached
        if pending.len() >= self.batch_threshold {
            return true;
        }

        // Batch ready if oldest request has waited too long
        if let Some(oldest) = pending.first() {
            let wait_ms = oldest.wait_time().as_millis() as u64;
            if wait_ms >= self.timeout_ms {
                return true;
            }
        }

        false
    }

    /// Collect a batch of requests
    ///
    /// Returns None if no requests are pending or batch not ready
    pub fn collect_batch(&self) -> Option<RequestBatch> {
        let mut pending = self.pending.lock().expect("Mutex poisoned");
        if pending.is_empty() {
            return None;
        }

        // Check if batch is ready (threshold or timeout)
        let ready = pending.len() >= self.batch_threshold
            || pending
                .first()
                .is_some_and(|r| r.wait_time().as_millis() as u64 >= self.timeout_ms);

        if !ready {
            return None;
        }

        // Take up to max_batch_size requests
        let batch_size = pending.len().min(self.max_batch_size);
        let requests: Vec<PendingRequest> = pending.drain(..batch_size).collect();

        Some(RequestBatch::new(requests))
    }

    /// Force collect all pending requests as a batch
    pub fn flush(&self) -> Option<RequestBatch> {
        let mut pending = self.pending.lock().expect("Mutex poisoned");
        if pending.is_empty() {
            return None;
        }

        let requests: Vec<PendingRequest> = pending.drain(..).collect();
        Some(RequestBatch::new(requests))
    }

    /// Number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending.lock().expect("Mutex poisoned").len()
    }

    /// Total requests submitted
    pub fn total_submitted(&self) -> u64 {
        self.next_id.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(feature = "gpu")]
impl Default for BatchRequestCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Batching configuration for request collector (PARITY-023)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    /// Minimum batch size to trigger GPU processing (32 from IMP-600)
    pub batch_threshold: usize,
    /// Maximum wait time before processing smaller batch (ms)
    pub timeout_ms: u64,
    /// Maximum batch size (memory limit)
    pub max_batch_size: usize,
    /// Whether to prefer latency (process immediately) or throughput (wait for batch)
    pub prefer_throughput: bool,
}

#[cfg(feature = "gpu")]
impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            batch_threshold: 32,
            timeout_ms: 50,
            max_batch_size: 64,
            prefer_throughput: true,
        }
    }
}

#[cfg(feature = "gpu")]
impl BatchingConfig {
    /// Config optimized for latency (smaller batches, shorter timeout)
    pub fn latency_optimized() -> Self {
        Self {
            batch_threshold: 8,
            timeout_ms: 10,
            max_batch_size: 32,
            prefer_throughput: false,
        }
    }

    /// Config optimized for throughput (larger batches, longer timeout)
    pub fn throughput_optimized() -> Self {
        Self {
            batch_threshold: 32,
            timeout_ms: 100,
            max_batch_size: 64,
            prefer_throughput: true,
        }
    }
}

/// Slot state for continuous batching (PARITY-028)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub enum SlotState {
    /// Slot is available for new request
    Empty,
    /// Slot has active request being generated
    Active {
        /// Unique request identifier
        request_id: u64,
        /// Input prompt tokens
        prompt_tokens: Vec<u32>,
        /// Tokens generated so far
        generated_tokens: Vec<u32>,
        /// Maximum tokens to generate
        max_tokens: usize,
        /// Sampling temperature
        temperature: f32,
        /// Top-k sampling parameter
        top_k: usize,
    },
    /// Slot has completed request waiting for retrieval
    Completed {
        /// Unique request identifier
        request_id: u64,
        /// All generated tokens
        generated_tokens: Vec<u32>,
    },
}

#[cfg(feature = "gpu")]
impl SlotState {
    /// Check if slot is available
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Check if slot has active generation
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active { .. })
    }

    /// Check if slot has completed request
    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    /// Get request ID if slot has one
    pub fn request_id(&self) -> Option<u64> {
        match self {
            Self::Empty => None,
            Self::Active { request_id, .. } | Self::Completed { request_id, .. } => {
                Some(*request_id)
            },
        }
    }
}

/// Continuous batch scheduler (PARITY-028)
///
/// Enables dynamic addition/removal of requests from a running batch:
/// - Requests are assigned to slots
/// - Each slot can be in Empty, Active, or Completed state
/// - New requests fill empty slots immediately
/// - Completed requests free their slots for reuse
///
/// This maximizes GPU utilization by keeping the batch full.
#[cfg(feature = "gpu")]
pub struct ContinuousBatchScheduler {
    /// Fixed-size array of slots
    slots: std::sync::Mutex<Vec<SlotState>>,
    /// KV caches for each slot (pre-allocated)
    caches: std::sync::Mutex<Vec<OwnedQuantizedKVCache>>,
    /// Total slots (max concurrent requests)
    pub num_slots: usize,
    /// Completed request IDs for polling
    completed: std::sync::Mutex<Vec<(u64, Vec<u32>)>>,
    /// Next request ID
    next_id: std::sync::atomic::AtomicU64,
}

include!("batch_scheduler_lock_slots_completed.rs");
include!("batch_scheduler_counter_inc_load.rs");
include!("batch_scheduler_prefix_cache.rs");
include!("batch_scheduler_multi_request.rs");
