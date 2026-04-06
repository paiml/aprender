//! HTTP Client for Real Model Server Benchmarking
//!
//! This module implements REAL HTTP calls to external model servers.
//! **NO MOCK DATA** - actual network requests with timing measurements.
//!
//! ## Supported Backends
//! - vLLM: OpenAI-compatible `/v1/completions` endpoint
//! - Ollama: `/api/generate` endpoint
//! - llama.cpp: OpenAI-compatible `/v1/completions` endpoint
//!
//! ## Quality Features
//! - Preflight validation per Toyota Way principles (Jidoka, Poka-yoke)
//! - CV-based stopping criterion per Hoefler & Belli SC'15
//! - MAD-based outlier detection per Chen et al.
//!
//! ## References
//! - [1] OpenAI API Spec: https://platform.openai.com/docs/api-reference
//! - [2] Ollama API Spec: https://github.com/ollama/ollama/blob/main/docs/api.md
//! - [3] Hoefler & Belli SC'15: CV-based stopping for reproducible benchmarks
//! - [4] Chen et al.: Robust outlier detection using MAD

use std::time::Instant;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::bench_preflight::{
    canonical_inputs, CvStoppingCriterion, OutlierDetector, PreflightRunner, QualityMetrics,
    ServerAvailabilityCheck, StopDecision,
};
use crate::error::{RealizarError, Result};

/// OpenAI-compatible completion request (vLLM, llama.cpp)
#[derive(Debug, Clone, Serialize)]
pub struct CompletionRequest {
    /// Model identifier
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
}

/// OpenAI-compatible completion response
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionResponse {
    /// Response ID
    pub id: String,
    /// Completion choices
    pub choices: Vec<CompletionChoice>,
    /// Usage statistics
    pub usage: Option<UsageStats>,
}

/// A single completion choice
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionChoice {
    /// Generated text
    pub text: String,
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// Token usage statistics
#[derive(Debug, Clone, Deserialize)]
pub struct UsageStats {
    /// Prompt tokens
    pub prompt_tokens: usize,
    /// Completion tokens
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

/// Ollama generate request
#[derive(Debug, Clone, Serialize)]
pub struct OllamaRequest {
    /// Model name
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Whether to stream
    #[serde(default)]
    pub stream: bool,
    /// Generation options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
}

/// Ollama generation options
#[derive(Debug, Clone, Serialize)]
pub struct OllamaOptions {
    /// Maximum tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<usize>,
    /// Temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// Ollama generate response
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaResponse {
    /// Model used
    pub model: String,
    /// Generated response
    pub response: String,
    /// Whether generation is done
    pub done: bool,
    /// Total duration in nanoseconds
    #[serde(default)]
    pub total_duration: u64,
    /// Load duration in nanoseconds
    #[serde(default)]
    pub load_duration: u64,
    /// Prompt evaluation count
    #[serde(default)]
    pub prompt_eval_count: usize,
    /// Prompt evaluation duration in nanoseconds
    #[serde(default)]
    pub prompt_eval_duration: u64,
    /// Evaluation count (tokens generated)
    #[serde(default)]
    pub eval_count: usize,
    /// Evaluation duration in nanoseconds
    #[serde(default)]
    pub eval_duration: u64,
}

/// llama.cpp native completion response (different from OpenAI format)
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaCppResponse {
    /// Generated content
    pub content: String,
    /// Model path
    #[serde(default)]
    pub model: String,
    /// Number of tokens predicted
    #[serde(default)]
    pub tokens_predicted: usize,
    /// Number of tokens evaluated (prompt)
    #[serde(default)]
    pub tokens_evaluated: usize,
    /// Whether generation stopped
    #[serde(default)]
    pub stop: bool,
    /// Timing information
    #[serde(default)]
    pub timings: Option<LlamaCppTimings>,
}

/// llama.cpp timing information
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaCppTimings {
    /// Prompt tokens
    #[serde(default)]
    pub prompt_n: usize,
    /// Prompt processing time in ms
    #[serde(default)]
    pub prompt_ms: f64,
    /// Predicted tokens
    #[serde(default)]
    pub predicted_n: usize,
    /// Prediction time in ms
    #[serde(default)]
    pub predicted_ms: f64,
    /// Tokens per second for generation
    #[serde(default)]
    pub predicted_per_second: f64,
}

/// Timing measurements from an HTTP inference request
#[derive(Debug, Clone)]
pub struct InferenceTiming {
    /// Time to first byte (TTFT) in milliseconds
    pub ttft_ms: f64,
    /// Total request time in milliseconds
    pub total_time_ms: f64,
    /// Tokens generated
    pub tokens_generated: usize,
    /// Generated text
    pub text: String,
}

/// HTTP client for model server communication
pub struct ModelHttpClient {
    client: Client,
    timeout_secs: u64,
}

impl Default for ModelHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelHttpClient {
    /// Create a new HTTP client with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to create HTTP client"),
            timeout_secs: 60,
        }
    }

    /// Create a new HTTP client with custom timeout
    #[must_use]
    pub fn with_timeout(timeout_secs: u64) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .build()
                .expect("Failed to create HTTP client"),
            timeout_secs,
        }
    }

    /// Get the configured timeout
    #[must_use]
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// Call OpenAI-compatible `/v1/completions` endpoint (vLLM, llama.cpp)
    ///
    /// # Errors
    /// Returns error if network request fails or response parsing fails
    pub fn openai_completion(
        &self,
        base_url: &str,
        request: &CompletionRequest,
        api_key: Option<&str>,
    ) -> Result<InferenceTiming> {
        let url = format!("{}/v1/completions", base_url.trim_end_matches('/'));
        let start = Instant::now();

        let mut req_builder = self.client.post(&url).json(request);

        if let Some(key) = api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder
            .send()
            .map_err(|e| RealizarError::ConnectionError(format!("HTTP request failed: {}", e)))?;

        let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(RealizarError::ConnectionError(format!(
                "HTTP {} from {}: {}",
                status, url, body
            )));
        }

        let completion: CompletionResponse =
            response.json().map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to parse completion response: {}", e),
            })?;

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let text = completion
            .choices
            .first()
            .map(|c| c.text.clone())
            .unwrap_or_default();

        let tokens_generated = completion.usage.map_or(0, |u| u.completion_tokens);

        Ok(InferenceTiming {
            ttft_ms,
            total_time_ms,
            tokens_generated,
            text,
        })
    }

    /// Call Ollama `/api/generate` endpoint
    ///
    /// # Errors
    /// Returns error if network request fails or response parsing fails
    pub fn ollama_generate(
        &self,
        base_url: &str,
        request: &OllamaRequest,
    ) -> Result<InferenceTiming> {
        let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
        let start = Instant::now();

        let response =
            self.client.post(&url).json(request).send().map_err(|e| {
                RealizarError::ConnectionError(format!("HTTP request failed: {}", e))
            })?;

        let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(RealizarError::ConnectionError(format!(
                "HTTP {} from {}: {}",
                status, url, body
            )));
        }

        let ollama_resp: OllamaResponse =
            response.json().map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to parse Ollama response: {}", e),
            })?;

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Use eval_count if available, otherwise estimate from response text
        // Ollama doesn't always return eval_count (depends on model/config)
        let tokens_generated = if ollama_resp.eval_count > 0 {
            ollama_resp.eval_count
        } else {
            // Estimate: ~4 chars per token (common for LLMs)
            // This is a reasonable fallback for scientific reproducibility
            (ollama_resp.response.len() / 4).max(1)
        };

        Ok(InferenceTiming {
            ttft_ms,
            total_time_ms,
            tokens_generated,
            text: ollama_resp.response,
        })
    }

    /// Call llama.cpp native `/completion` endpoint
    ///
    /// Note: llama.cpp also has `/v1/completions` but it returns a non-OpenAI format.
    /// This method uses the native endpoint which is more reliable.
    ///
    /// # Errors
    /// Returns error if network request fails or response parsing fails
    pub fn llamacpp_completion(
        &self,
        base_url: &str,
        request: &CompletionRequest,
    ) -> Result<InferenceTiming> {
        // llama.cpp uses /completion for native format, /v1/completions for OpenAI-ish
        let url = format!("{}/completion", base_url.trim_end_matches('/'));
        let start = Instant::now();

        // llama.cpp expects slightly different field names
        let body = serde_json::json!({
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature.unwrap_or(0.8),
            "stream": false
        });

        let response =
            self.client.post(&url).json(&body).send().map_err(|e| {
                RealizarError::ConnectionError(format!("HTTP request failed: {}", e))
            })?;

        let ttft_ms = start.elapsed().as_secs_f64() * 1000.0;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(RealizarError::ConnectionError(format!(
                "HTTP {} from {}: {}",
                status, url, body
            )));
        }

        let llama_resp: LlamaCppResponse =
            response.json().map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to parse llama.cpp response: {}", e),
            })?;

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(InferenceTiming {
            ttft_ms,
            total_time_ms,
            tokens_generated: llama_resp.tokens_predicted,
            text: llama_resp.content,
        })
    }

    /// Health check for OpenAI-compatible server
    ///
    /// # Errors
    /// Returns error if server is not reachable
    pub fn health_check_openai(&self, base_url: &str) -> Result<bool> {
        let url = format!("{}/v1/models", base_url.trim_end_matches('/'));

        let response =
            self.client.get(&url).send().map_err(|e| {
                RealizarError::ConnectionError(format!("Health check failed: {}", e))
            })?;

        Ok(response.status().is_success())
    }

    /// Health check for Ollama server
    ///
    /// # Errors
    /// Returns error if server is not reachable
    pub fn health_check_ollama(&self, base_url: &str) -> Result<bool> {
        let url = format!("{}/api/tags", base_url.trim_end_matches('/'));

        let response =
            self.client.get(&url).send().map_err(|e| {
                RealizarError::ConnectionError(format!("Health check failed: {}", e))
            })?;

        Ok(response.status().is_success())
    }
}

include!("mod_http_benchmark.rs");
include!("benchmark_runner.rs");
include!("mod_04.rs");
