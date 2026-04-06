//! HTTP API for model inference
//!
//! Provides REST endpoints for tokenization and text generation using axum.
//!
//! ## Endpoints
//!
//! - `GET /health` - Health check
//! - `GET /metrics` - Prometheus-formatted metrics
//! - `GET /metrics/dispatch` - CPU/GPU dispatch statistics (?format=prometheus|json)
//! - `POST /tokenize` - Tokenize text
//! - `POST /generate` - Generate text from prompt
//! - `POST /batch/tokenize` - Batch tokenize multiple texts
//! - `POST /batch/generate` - Batch generate for multiple prompts
//! - `POST /stream/generate` - Stream generated tokens via SSE
//! - `POST /v1/gpu/warmup` - Warmup GPU cache for batch inference (PARITY-022)
//! - `GET /v1/gpu/status` - Check GPU cache status (PARITY-022)
//! - `POST /v1/batch/completions` - GPU-accelerated batch inference (PARITY-022)
//! - `GET /v1/metrics` - JSON metrics for TUI monitoring (PARITY-107)
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::api::{create_router, AppState};
//!
//! let state = AppState::new(model, tokenizer);
//! let app = create_router(state);
//! axum::serve(listener, app).await?;
//! ```

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{
    apr::{AprModel, HEADER_SIZE, MAGIC},
    audit::{AuditLogger, AuditRecord, InMemoryAuditSink},
    cache::{CacheKey, ModelCache},
    error::RealizarError,
    explain::ShapExplanation,
    layers::{Model, ModelConfig},
    metrics::MetricsCollector,
    registry::ModelRegistry,
    tokenizer::BPETokenizer,
};

// PMAT-802: Extracted handlers
#[cfg(feature = "cuda")]
pub mod apr_q4k_scheduler;
#[cfg(feature = "cuda")]
pub mod cuda_batch_scheduler;
#[cfg(feature = "cuda")]
pub mod iteration_scheduler;
mod openai_handlers;
pub(crate) use openai_handlers::{
    openai_chat_completions_handler, openai_chat_completions_stream_handler, openai_models_handler,
};
mod gpu_handlers;
pub(crate) use gpu_handlers::{
    batch_generate_handler, batch_tokenize_handler, generate_handler,
    gpu_batch_completions_handler, gpu_status_handler, gpu_warmup_handler, models_handler,
    stream_generate_handler, tokenize_handler,
};
// Public exports for tests (GPU-only types)
#[cfg(feature = "gpu")]
pub use gpu_handlers::{
    BatchProcessResult, BatchQueueStats, ContinuousBatchRequest, ContinuousBatchResponse,
    GpuBatchRequest, GpuBatchResponse, GpuBatchResult, GpuBatchStats, GpuStatusResponse,
    GpuWarmupResponse,
};
// Public exports for apr-cli CUDA integration (PMAT-GPU-001)
#[cfg(feature = "gpu")]
pub use gpu_handlers::{spawn_batch_processor, BatchConfig};
mod realize_handlers;
pub(crate) use realize_handlers::{
    clean_chat_output, format_chat_messages, openai_completions_handler, openai_embeddings_handler,
    realize_embed_handler, realize_model_handler, realize_reload_handler,
};
#[cfg(feature = "cuda")]
pub(crate) use realize_handlers::{logprobs_handler, perplexity_handler};
// Public exports for tests
pub use realize_handlers::{
    CompletionChoice, CompletionRequest, CompletionResponse, ContextWindowConfig,
    ContextWindowManager, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    ModelLineage, ModelMetadataResponse, ReloadRequest, ReloadResponse,
};
mod apr_handlers;
pub(crate) use apr_handlers::{apr_audit_handler, apr_explain_handler, apr_predict_handler};
mod types;
pub use crate::registry::ModelInfo;
pub use types::{default_max_tokens, default_top_k};
#[cfg(test)]
pub(crate) use types::{default_strategy, default_temperature, default_top_p};
pub use types::{
    BatchGenerateRequest, BatchGenerateResponse, BatchTokenizeRequest, BatchTokenizeResponse,
    ErrorResponse, GenerateRequest, GenerateResponse, HealthResponse, ModelsResponse,
    StreamDoneEvent, StreamTokenEvent, TokenizeRequest, TokenizeResponse,
};

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Model for inference (single model mode)
    model: Option<Arc<Model>>,
    /// Tokenizer for encoding/decoding (single model mode)
    tokenizer: Option<Arc<BPETokenizer>>,
    /// Model cache for multi-model support
    #[allow(dead_code)]
    cache: Option<Arc<ModelCache>>,
    /// Default cache key for single model mode
    #[allow(dead_code)]
    cache_key: Option<CacheKey>,
    /// Metrics collector for monitoring
    metrics: Arc<MetricsCollector>,
    /// Model registry for multi-model serving
    registry: Option<Arc<ModelRegistry>>,
    /// Default model ID for multi-model mode
    default_model_id: Option<String>,
    /// APR model for /v1/predict endpoint (real inference, not mock)
    apr_model: Option<Arc<AprModel>>,
    /// Audit logger for /v1/audit endpoint (real records, not mock)
    audit_logger: Arc<AuditLogger>,
    /// In-memory audit sink for record retrieval
    audit_sink: Arc<InMemoryAuditSink>,
    /// GPU model for GGUF inference (M33: IMP-084)
    #[cfg(feature = "gpu")]
    gpu_model: Option<Arc<std::sync::RwLock<crate::gpu::GpuModel>>>,
    /// Quantized model for fused Q4_K inference (IMP-100)
    /// This is 1.37x faster than dequantized GpuModel due to reduced memory bandwidth
    quantized_model: Option<Arc<crate::gguf::OwnedQuantizedModel>>,
    /// Thread-safe cached model for HTTP serving (IMP-116)
    /// Uses Mutex-based scheduler caching for 10.6x speedup
    #[cfg(feature = "gpu")]
    cached_model: Option<Arc<crate::gguf::OwnedQuantizedModelCachedSync>>,
    /// Dispatch metrics for adaptive CPU/GPU tracking (IMP-126)
    #[cfg(feature = "gpu")]
    dispatch_metrics: Option<Arc<crate::gguf::DispatchMetrics>>,
    /// Batch request channel for continuous batching (PARITY-052)
    /// Requests sent here are queued and processed in batches
    #[cfg(feature = "gpu")]
    batch_request_tx: Option<tokio::sync::mpsc::Sender<ContinuousBatchRequest>>,
    /// Batch configuration for window timing and size thresholds (PARITY-052)
    #[cfg(feature = "gpu")]
    batch_config: Option<BatchConfig>,
    /// CUDA-optimized model for high-performance GPU inference (PAR-111)
    /// Uses pre-uploaded weights and batched workspaces for 755+ tok/s (2.6x Ollama)
    #[cfg(feature = "cuda")]
    cuda_model: Option<Arc<std::sync::RwLock<crate::gguf::OwnedQuantizedModelCuda>>>,
    /// PMAT-044: CUDA batch scheduler for continuous batching on /v1/chat/completions
    #[cfg(feature = "cuda")]
    cuda_batch_tx: Option<tokio::sync::mpsc::Sender<cuda_batch_scheduler::CudaBatchRequest>>,
    /// ALB-095: APR Q4K GPU inference channel (dedicated thread owns CudaExecutor)
    #[cfg(feature = "cuda")]
    apr_q4k_tx: Option<tokio::sync::mpsc::Sender<apr_q4k_scheduler::AprQ4kRequest>>,
    /// APR Transformer for SafeTensors/APR inference (PMAT-SERVE-FIX-001)
    /// Supports F32 weights from SafeTensors or APR format
    apr_transformer: Option<Arc<crate::apr_transformer::AprTransformer>>,
    /// #169: SafeTensors CUDA model for GPU-accelerated F16/F32 inference
    #[cfg(feature = "cuda")]
    safetensors_cuda_model:
        Option<Arc<std::sync::Mutex<crate::safetensors_cuda::SafeTensorsCudaModel>>>,
    /// GH-319: Cached model architecture string (avoids RwLock in hot path)
    cached_architecture: Option<String>,
    /// GH-330: Cached EOS token ID (avoids RwLock in hot path)
    cached_eos_token_id: Option<u32>,
    /// GH-152: Enable verbose request/response logging
    verbose: bool,
    /// GH-103: Enable inference tracing (propagates into QuantizedGenerateConfig.trace)
    trace: bool,
}

/// Helper to create default audit infrastructure
fn create_audit_state() -> (Arc<AuditLogger>, Arc<InMemoryAuditSink>) {
    let sink = Arc::new(InMemoryAuditSink::new());
    let logger = AuditLogger::new(Box::new(InMemorySinkWrapper(sink.clone())))
        .with_model_hash("demo-model-hash");
    (Arc::new(logger), sink)
}

/// Wrapper to make Arc<InMemoryAuditSink> implement AuditSink
struct InMemorySinkWrapper(Arc<InMemoryAuditSink>);

impl crate::audit::AuditSink for InMemorySinkWrapper {
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), crate::audit::AuditError> {
        self.0.write_batch(records)
    }

    fn flush(&self) -> Result<(), crate::audit::AuditError> {
        self.0.flush()
    }
}

include!("mod_app_state_gpu.rs");
include!("mod_create_demo.rs");
include!("router.rs");
include!("dispatch_metrics.rs");
