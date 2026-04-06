//! GPU batch inference handlers
//!
//! Extracted from api/mod.rs (PMAT-802) to reduce module size.
//! Contains batch completions, warmup, and status handlers for GPU inference.
#![allow(unreachable_pub)] // Items re-exported as pub from api/mod.rs

use std::convert::Infallible;

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};

use super::{
    default_max_tokens, default_top_k, AppState, BatchGenerateRequest, BatchGenerateResponse,
    BatchTokenizeRequest, BatchTokenizeResponse, ErrorResponse, GenerateRequest, GenerateResponse,
    ModelsResponse, StreamDoneEvent, StreamTokenEvent, TokenizeRequest, TokenizeResponse,
};
use crate::generate::{GenerationConfig, SamplingStrategy};
use crate::registry::ModelInfo;
use crate::tokenizer::BPETokenizer;

// ============================================================================
// Shared helpers
// ============================================================================

/// Shorthand for the error tuple used across all gpu_handlers endpoints.
type ApiErr = (StatusCode, Json<ErrorResponse>);

/// Build an API error response.
fn api_err(status: StatusCode, msg: impl std::fmt::Display) -> ApiErr {
    (
        status,
        Json(ErrorResponse {
            error: msg.to_string(),
        }),
    )
}

/// Get tokenizer from state or return 500.
fn require_tok(state: &AppState) -> Result<std::sync::Arc<BPETokenizer>, ApiErr> {
    state
        .tokenizer
        .clone()
        .ok_or_else(|| api_err(StatusCode::INTERNAL_SERVER_ERROR, "No tokenizer available"))
}

/// Tokenize a prompt, returning error if empty.
fn tokenize_prompt(tokenizer: &BPETokenizer, prompt: &str) -> Result<Vec<u32>, ApiErr> {
    let ids = tokenizer.encode(prompt);
    if ids.is_empty() {
        return Err(api_err(StatusCode::BAD_REQUEST, "Prompt cannot be empty"));
    }
    Ok(ids)
}

/// GH-330: Get the EOS token ID using Design by Contract priority.
///
/// 1. Model config (class invariant from GGUF/APR metadata)
/// 2. Tokenizer vocabulary lookup (runtime fallback)
/// 3. 0 (disabled — no EOS checking) instead of hardcoded Qwen2 value
fn eos_id(tokenizer: &BPETokenizer, model_eos: Option<u32>) -> u32 {
    model_eos
        .or_else(|| tokenizer.get_token_id("<|im_end|>"))
        .or_else(|| tokenizer.get_token_id("<|endoftext|>"))
        .unwrap_or(0)
}

// ============================================================================
// PARITY-022: GPU Batch Inference API
// ============================================================================

/// GPU batch completions request (PARITY-022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBatchRequest {
    /// List of prompts to process in batch
    pub prompts: Vec<String>,
    /// Maximum tokens to generate per prompt
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy)
    #[serde(default)]
    pub temperature: f32,
    /// Top-k sampling (1 = greedy)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Stop tokens (optional)
    #[serde(default)]
    pub stop: Vec<String>,
}

/// GPU batch completions response (PARITY-022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBatchResponse {
    /// Results for each prompt
    pub results: Vec<GpuBatchResult>,
    /// Batch statistics
    pub stats: GpuBatchStats,
}

/// Single result in GPU batch response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBatchResult {
    /// Prompt index
    pub index: usize,
    /// Generated token IDs
    pub token_ids: Vec<u32>,
    /// Decoded text
    pub text: String,
    /// Number of tokens generated
    pub num_generated: usize,
}

/// GPU batch statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBatchStats {
    /// Batch size
    pub batch_size: usize,
    /// Whether GPU was used
    pub gpu_used: bool,
    /// Total tokens generated
    pub total_tokens: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Throughput in tokens per second
    pub throughput_tps: f64,
}

/// GPU warmup response (PARITY-022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuWarmupResponse {
    /// Whether warmup succeeded
    pub success: bool,
    /// Memory used in bytes
    pub memory_bytes: usize,
    /// Number of layers cached
    pub num_layers: usize,
    /// Message
    pub message: String,
}

/// GPU status response (PARITY-022)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatusResponse {
    /// Whether GPU cache is warmed up
    pub cache_ready: bool,
    /// Memory used by cache in bytes
    pub cache_memory_bytes: usize,
    /// GPU batch threshold
    pub batch_threshold: usize,
    /// Recommended minimum batch size
    pub recommended_min_batch: usize,
}

// ==================== PARITY-052: Batch Request Queuing ====================
//
// Infrastructure for continuous batch inference via HTTP API.
// Requests are queued and processed in batches for higher throughput.
//
// Architecture:
//   - BatchConfig: Configuration for batch window and size thresholds
//   - ContinuousBatchRequest: Internal request with oneshot response channel
//   - ContinuousBatchResponse: Result returned via oneshot channel
//   - AppState extensions: batch_scheduler, batch_request_tx, batch_config
// ============================================================================

/// Configuration for continuous batch inference (PARITY-052)
#[derive(Debug, Clone)]
#[cfg(feature = "gpu")]
pub struct BatchConfig {
    /// Maximum time to wait for batch to fill (milliseconds)
    pub window_ms: u64,
    /// Minimum batch size to process (below this, use single-request path)
    pub min_batch: usize,
    /// Optimal batch size for M4 parity (process immediately when reached)
    /// PARITY-095: This also controls GPU batch threshold
    pub optimal_batch: usize,
    /// Maximum batch size (GPU memory constraint)
    pub max_batch: usize,
    /// Channel buffer size for request queue
    pub queue_size: usize,
    /// GPU batch threshold (use GPU path when batch >= this)
    /// PARITY-095: GPU GEMM wins at batch >= 32 (from IMP-600 analysis)
    pub gpu_threshold: usize,
}

#[cfg(feature = "gpu")]
impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            window_ms: 50,     // 50ms batch window (allow time for requests to accumulate)
            min_batch: 4,      // Minimum for any batching benefit
            optimal_batch: 32, // PARITY-095: Aligned with GPU threshold for M4 parity
            max_batch: 64,     // Allow larger batches for better GPU utilization
            queue_size: 1024,  // Request queue buffer
            gpu_threshold: 32, // GPU GEMM crossover point (from PARITY-046b)
        }
    }
}

#[cfg(feature = "gpu")]
impl BatchConfig {
    /// Create config optimized for low latency (smaller batches)
    /// Note: GPU batch disabled (threshold > max_batch) for consistent latency
    pub fn low_latency() -> Self {
        Self {
            window_ms: 5,
            min_batch: 2,
            optimal_batch: 8,
            max_batch: 16,
            queue_size: 512,
            gpu_threshold: 32, // Effectively disabled since max_batch=16
        }
    }

    /// Create config optimized for high throughput (larger batches)
    /// PARITY-095: GPU batch enabled for batch >= 32
    pub fn high_throughput() -> Self {
        Self {
            window_ms: 100, // 100ms window for maximum batching
            min_batch: 8,
            optimal_batch: 32, // Trigger processing at GPU threshold
            max_batch: 128,    // Large batches for maximum throughput
            queue_size: 2048,
            gpu_threshold: 32, // GPU GEMM crossover
        }
    }

    /// Check if batch size is sufficient for processing
    pub fn should_process(&self, batch_size: usize) -> bool {
        batch_size >= self.optimal_batch
    }

    /// Check if batch size meets minimum threshold
    pub fn meets_minimum(&self, batch_size: usize) -> bool {
        batch_size >= self.min_batch
    }
}

/// Internal batch request with response channel (PARITY-052)
#[cfg(feature = "gpu")]
pub struct ContinuousBatchRequest {
    /// Tokenized input prompt
    pub prompt_tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature
    pub temperature: f32,
    /// Top-k sampling parameter
    pub top_k: usize,
    /// Channel to send response back to handler
    pub response_tx: tokio::sync::oneshot::Sender<ContinuousBatchResponse>,
    /// Request timestamp for latency tracking
    pub submitted_at: std::time::Instant,
}

/// Response from batch processor (PARITY-052)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct ContinuousBatchResponse {
    /// Generated token IDs (includes prompt)
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens (to skip when decoding)
    pub prompt_len: usize,
    /// Whether request was processed in batch or single-request path
    pub batched: bool,
    /// Batch size when processed (1 for single-request)
    pub batch_size: usize,
    /// Processing latency in milliseconds
    pub latency_ms: f64,
}

#[cfg(feature = "gpu")]
impl ContinuousBatchResponse {
    /// Create response for single-request path
    pub fn single(token_ids: Vec<u32>, prompt_len: usize, latency_ms: f64) -> Self {
        Self {
            token_ids,
            prompt_len,
            batched: false,
            batch_size: 1,
            latency_ms,
        }
    }

    /// Create response for batched path
    pub fn batched(
        token_ids: Vec<u32>,
        prompt_len: usize,
        batch_size: usize,
        latency_ms: f64,
    ) -> Self {
        Self {
            token_ids,
            prompt_len,
            batched: true,
            batch_size,
            latency_ms,
        }
    }

    /// Get generated tokens (excluding prompt)
    pub fn generated_tokens(&self) -> &[u32] {
        if self.token_ids.len() > self.prompt_len {
            &self.token_ids[self.prompt_len..]
        } else {
            &[]
        }
    }
}

/// Batch queue statistics (PARITY-052)
#[derive(Debug, Clone, Default)]
#[cfg(feature = "gpu")]
pub struct BatchQueueStats {
    /// Total requests queued
    pub total_queued: u64,
    /// Total batches processed
    pub total_batches: u64,
    /// Total requests processed via single-request path
    pub total_single: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average queue wait time in milliseconds
    pub avg_wait_ms: f64,
}

// ==================== PARITY-053: Batch Processor Background Task ====================
//
// Background task that processes batched inference requests.
// Collects requests until batch is ready (size threshold or timeout), then processes.
//
// Flow:
//   1. Receive requests via mpsc channel
//   2. Accumulate until batch_size >= optimal_batch OR window_ms timeout
//   3. Process batch using model.generate_with_cache() for each request
//   4. Send results via oneshot channels
//
// Note: True batch inference (single forward pass for multiple requests) requires
// additional model infrastructure. This implementation processes requests in
// parallel within a batch window, which still improves throughput under load.
// ==================================================================================

/// Result from batch processing
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct BatchProcessResult {
    /// Number of requests processed
    pub requests_processed: usize,
    /// Whether processed as batch or single
    pub was_batched: bool,
    /// Total processing time in milliseconds
    pub total_time_ms: f64,
    /// Average latency per request in milliseconds
    pub avg_latency_ms: f64,
}

/// Spawn the batch processor background task (PARITY-053)
///
/// Returns the sender channel for submitting requests.
/// The receiver is consumed by the spawned task.
///
/// # Arguments
/// * `model` - The cached model for inference
/// * `config` - Batch configuration
///
/// # Returns
/// * Sender channel for batch requests
#[cfg(feature = "gpu")]
pub fn spawn_batch_processor(
    model: std::sync::Arc<crate::gguf::OwnedQuantizedModelCachedSync>,
    config: BatchConfig,
) -> tokio::sync::mpsc::Sender<ContinuousBatchRequest> {
    let (tx, rx) = tokio::sync::mpsc::channel(config.queue_size);

    // Spawn the background processor task
    tokio::spawn(batch_processor_task(rx, model, config));

    tx
}

/// Background task that processes batched requests (PARITY-053)
///
/// This task runs continuously, collecting requests and processing them in batches.
/// It uses a timeout-based batching strategy:
/// - Process immediately if batch reaches optimal_batch size
/// - Process on timeout (window_ms) if batch has requests
/// - Fall back to single-request processing for very small batches
#[cfg(feature = "gpu")]
async fn batch_processor_task(
    mut rx: tokio::sync::mpsc::Receiver<ContinuousBatchRequest>,
    model: std::sync::Arc<crate::gguf::OwnedQuantizedModelCachedSync>,
    config: BatchConfig,
) {
    use std::time::{Duration, Instant};
    use tokio::time::timeout;

    let mut batch: Vec<ContinuousBatchRequest> = Vec::with_capacity(config.max_batch);
    let mut window_start = Instant::now();

    loop {
        // Calculate remaining time in window
        let elapsed = window_start.elapsed();
        let remaining = Duration::from_millis(config.window_ms).saturating_sub(elapsed);

        // Try to receive with timeout
        match timeout(remaining, rx.recv()).await {
            Ok(Some(request)) => {
                batch.push(request);

                // Process immediately if we hit optimal batch size
                if batch.len() >= config.optimal_batch {
                    process_batch(&model, &config, &mut batch).await;
                    window_start = Instant::now();
                }
            },
            Ok(None) => {
                // Channel closed, process remaining and exit
                if !batch.is_empty() {
                    process_batch(&model, &config, &mut batch).await;
                }
                break;
            },
            Err(_) => {
                // Timeout - process current batch if we have requests
                if !batch.is_empty() {
                    process_batch(&model, &config, &mut batch).await;
                }
                window_start = Instant::now();
            },
        }
    }
}

include!("batch_processing.rs");
include!("batch.rs");
include!("stream_generate.rs");
