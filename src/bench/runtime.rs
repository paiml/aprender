//! Runtime backend abstraction for benchmark comparison
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - BENCH-002: Runtime Backend Abstraction
//! - LlamaCppBackend, VllmBackend, OllamaBackend implementations

#![allow(clippy::cast_precision_loss)]
// Items are conditionally re-exported by parent mod via feature gates (bench-http)
#![allow(unreachable_pub)]
// Backend abstraction types used by tests but not production inference
#![allow(dead_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::RealizarError;

#[cfg(feature = "bench-http")]
use crate::http_client::{CompletionRequest, ModelHttpClient, OllamaOptions, OllamaRequest};

/// Supported runtime types for inference benchmarking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeType {
    /// Native Realizar runtime (.apr format)
    Realizar,
    /// llama.cpp (GGUF format)
    LlamaCpp,
    /// vLLM (safetensors, HuggingFace)
    Vllm,
    /// Ollama (wraps llama.cpp)
    Ollama,
}

impl RuntimeType {
    /// Get string representation
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Realizar => "realizar",
            Self::LlamaCpp => "llama-cpp",
            Self::Vllm => "vllm",
            Self::Ollama => "ollama",
        }
    }

    /// Parse from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "realizar" => Some(Self::Realizar),
            "llama-cpp" | "llama.cpp" | "llamacpp" => Some(Self::LlamaCpp),
            "vllm" => Some(Self::Vllm),
            "ollama" => Some(Self::Ollama),
            _ => None,
        }
    }
}

/// Request for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature
    pub temperature: f64,
    /// Optional stop sequences
    pub stop: Vec<String>,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 100,
            temperature: 0.7,
            stop: Vec::new(),
        }
    }
}

impl InferenceRequest {
    /// Create new request with prompt
    #[must_use]
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            ..Default::default()
        }
    }

    /// Set max tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set stop sequences
    #[must_use]
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }
}

/// Response from inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Time to first token (ms)
    pub ttft_ms: f64,
    /// Total generation time (ms)
    pub total_time_ms: f64,
    /// Inter-token latencies (ms)
    pub itl_ms: Vec<f64>,
}

impl InferenceResponse {
    /// Calculate tokens per second
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_time_ms <= 0.0 {
            return 0.0;
        }
        (self.tokens_generated as f64) / (self.total_time_ms / 1000.0)
    }
}

/// Runtime backend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Runtime type
    pub runtime_type: RuntimeType,
    /// Version string
    pub version: String,
    /// Whether streaming is supported
    pub supports_streaming: bool,
    /// Model currently loaded (if any)
    pub loaded_model: Option<String>,
}

/// Trait for inference runtime backends
pub trait RuntimeBackend: Send + Sync {
    /// Get backend information
    fn info(&self) -> BackendInfo;

    /// Run inference
    ///
    /// # Errors
    ///
    /// Returns `RealizarError` if inference fails due to:
    /// - Model not loaded
    /// - Backend communication failure
    /// - Invalid request parameters
    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError>;

    /// Load a model (if applicable)
    ///
    /// # Errors
    ///
    /// Returns `RealizarError` if model loading fails due to:
    /// - Model file not found
    /// - Invalid model format
    /// - Insufficient memory
    fn load_model(&mut self, _model_path: &str) -> Result<(), RealizarError> {
        Ok(()) // Default: no-op
    }
}

/// Mock backend for testing
pub struct MockBackend {
    ttft_ms: f64,
    tokens_per_second: f64,
}

impl MockBackend {
    /// Create a new mock backend with specified latencies
    #[must_use]
    pub fn new(ttft_ms: f64, tokens_per_second: f64) -> Self {
        Self {
            ttft_ms,
            tokens_per_second,
        }
    }
}

impl RuntimeBackend for MockBackend {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            runtime_type: RuntimeType::Realizar,
            version: env!("CARGO_PKG_VERSION").to_string(),
            supports_streaming: true,
            loaded_model: None,
        }
    }

    fn inference(&self, request: &InferenceRequest) -> Result<InferenceResponse, RealizarError> {
        let tokens = request.max_tokens.min(100);
        let gen_time_ms = (tokens as f64) / self.tokens_per_second * 1000.0;

        Ok(InferenceResponse {
            text: "Mock response".to_string(),
            tokens_generated: tokens,
            ttft_ms: self.ttft_ms,
            total_time_ms: self.ttft_ms + gen_time_ms,
            itl_ms: vec![gen_time_ms / tokens as f64; tokens],
        })
    }
}

/// Registry of available backends
pub struct BackendRegistry {
    backends: HashMap<RuntimeType, Box<dyn RuntimeBackend>>,
}

impl BackendRegistry {
    /// Create empty registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Register a backend
    pub fn register(&mut self, runtime: RuntimeType, backend: Box<dyn RuntimeBackend>) {
        self.backends.insert(runtime, backend);
    }

    /// Get a backend by type
    #[must_use]
    pub fn get(&self, runtime: RuntimeType) -> Option<&dyn RuntimeBackend> {
        self.backends.get(&runtime).map(AsRef::as_ref)
    }

    /// List registered runtimes
    #[must_use]
    pub fn list(&self) -> Vec<RuntimeType> {
        self.backends.keys().copied().collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for llama.cpp backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    /// Path to llama-cli binary
    pub binary_path: String,
    /// Path to model file
    pub model_path: Option<String>,
    /// Number of GPU layers to offload
    pub n_gpu_layers: u32,
    /// Context size
    pub ctx_size: usize,
    /// Number of threads
    pub threads: usize,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            binary_path: "llama-cli".to_string(),
            model_path: None,
            n_gpu_layers: 0,
            ctx_size: 2048,
            threads: 4,
        }
    }
}

impl LlamaCppConfig {
    /// Create new config with binary path
    #[must_use]
    pub fn new(binary_path: &str) -> Self {
        Self {
            binary_path: binary_path.to_string(),
            ..Default::default()
        }
    }

    /// Set model path
    #[must_use]
    pub fn with_model(mut self, model_path: &str) -> Self {
        self.model_path = Some(model_path.to_string());
        self
    }

    /// Set GPU layers
    #[must_use]
    pub fn with_gpu_layers(mut self, layers: u32) -> Self {
        self.n_gpu_layers = layers;
        self
    }

    /// Set context size
    #[must_use]
    pub fn with_ctx_size(mut self, ctx_size: usize) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    /// Set number of threads
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }
}

/// Configuration for vLLM backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmConfig {
    /// Base URL for vLLM server
    pub base_url: String,
    /// API version
    pub api_version: String,
    /// Model name/path
    pub model: Option<String>,
    /// API key (if required)
    pub api_key: Option<String>,
}

impl Default for VllmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8000".to_string(),
            api_version: "v1".to_string(),
            model: None,
            api_key: None,
        }
    }
}

impl VllmConfig {
    /// Create new config with base URL
    #[must_use]
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            ..Default::default()
        }
    }

    /// Set model
    #[must_use]
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Set API key
    #[must_use]
    pub fn with_api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }
}

// ============================================================================
// LlamaCppBackend Implementation (BENCH-002)
// ============================================================================

/// llama.cpp backend for GGUF model inference via subprocess
pub struct LlamaCppBackend {
    config: LlamaCppConfig,
}

include!("backend.rs");
include!("runtime_type.rs");
