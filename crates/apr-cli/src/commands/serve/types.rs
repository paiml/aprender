//! Server type definitions and data models for APR serve command

// Allow dead code and unused during development - these are planned features
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::needless_return)]
#![allow(clippy::format_push_string)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::if_not_else)]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::inefficient_to_string)]

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port to listen on
    pub port: u16,
    /// Host to bind to
    pub host: String,
    /// Enable CORS (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub cors: bool,
    /// Request timeout in seconds (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub timeout_secs: u64,
    /// Maximum concurrent requests (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub max_concurrent: usize,
    /// Enable Prometheus metrics endpoint
    pub metrics: bool,
    /// Disable GPU acceleration (accepted but not yet implemented - GH-80)
    #[allow(dead_code)]
    pub no_gpu: bool,
    /// Force GPU acceleration (requires CUDA feature)
    pub gpu: bool,
    /// Enable batched GPU inference for 2X+ throughput
    pub batch: bool,
    /// Enable inference tracing (PMAT-SHOWCASE-METHODOLOGY-001)
    pub trace: bool,
    /// Trace detail level (none, basic, layer)
    pub trace_level: String,
    /// Enable inline Roofline profiling (adds X-Profile headers)
    pub profile: bool,
    /// GH-152: Enable verbose request/response logging
    pub verbose: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            cors: true,
            timeout_secs: 30,
            max_concurrent: 10,
            metrics: true,
            no_gpu: false,
            gpu: false,
            batch: false,
            trace: false,
            trace_level: "basic".to_string(),
            profile: false,
            verbose: false,
        }
    }
}

impl ServerConfig {
    /// Create config with custom port (builder pattern, used in tests)
    #[cfg(test)]
    pub(crate) fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Create config with custom host (builder pattern, used in tests)
    #[cfg(test)]
    pub(crate) fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Get bind address
    pub(super) fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Server metrics (thread-safe)
///
/// Implements APR-SPEC §4.15.9 metrics accuracy requirements (MA01-MA10).
/// All counters are thread-safe and exposed via /metrics endpoint.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    /// Total requests received (MA01)
    pub requests_total: AtomicU64,
    /// Successful requests (2xx)
    pub requests_success: AtomicU64,
    /// Client errors (4xx)
    pub requests_client_error: AtomicU64,
    /// Server errors (5xx)
    pub requests_server_error: AtomicU64,
    /// Total tokens generated (MA03)
    pub tokens_generated: AtomicU64,
    /// Total inference time in milliseconds
    pub inference_time_ms: AtomicU64,
    /// Model memory in bytes (MM01)
    pub model_memory_bytes: AtomicU64,
    /// Server start time (for uptime calculation)
    start_time: std::sync::OnceLock<Instant>,
}

impl ServerMetrics {
    /// Create new metrics with server start time
    pub fn new() -> Arc<Self> {
        let metrics = Arc::new(Self::default());
        let _ = metrics.start_time.set(Instant::now());
        metrics
    }

    /// Record a request with outcome
    pub fn record_request(&self, success: bool, tokens: u64, duration_ms: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        if success {
            self.requests_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_server_error.fetch_add(1, Ordering::Relaxed);
        }
        self.tokens_generated.fetch_add(tokens, Ordering::Relaxed);
        self.inference_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Record client error (4xx)
    pub fn record_client_error(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_client_error.fetch_add(1, Ordering::Relaxed);
    }

    /// Get uptime in seconds (HR04)
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time
            .get()
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0)
    }

    /// Get Prometheus-format metrics (MA06: valid Prometheus format)
    ///
    /// Format follows https://prometheus.io/docs/instrumenting/exposition_formats/
    pub fn prometheus_output(&self) -> String {
        let total = self.requests_total.load(Ordering::Relaxed);
        let success = self.requests_success.load(Ordering::Relaxed);
        let client_errors = self.requests_client_error.load(Ordering::Relaxed);
        let server_errors = self.requests_server_error.load(Ordering::Relaxed);
        let tokens = self.tokens_generated.load(Ordering::Relaxed);
        let inference_ms = self.inference_time_ms.load(Ordering::Relaxed);
        let model_bytes = self.model_memory_bytes.load(Ordering::Relaxed);
        let uptime = self.uptime_seconds();

        format!(
            r#"# HELP apr_requests_total Total number of HTTP requests
# TYPE apr_requests_total counter
apr_requests_total {total}

# HELP apr_requests_success Successful requests (2xx)
# TYPE apr_requests_success counter
apr_requests_success {success}

# HELP apr_requests_client_error Client error requests (4xx)
# TYPE apr_requests_client_error counter
apr_requests_client_error {client_errors}

# HELP apr_requests_server_error Server error requests (5xx)
# TYPE apr_requests_server_error counter
apr_requests_server_error {server_errors}

# HELP apr_tokens_generated_total Total tokens generated
# TYPE apr_tokens_generated_total counter
apr_tokens_generated_total {tokens}

# HELP apr_inference_duration_seconds_total Total inference time in seconds
# TYPE apr_inference_duration_seconds_total counter
apr_inference_duration_seconds_total {:.3}

# HELP apr_memory_bytes Memory usage by type
# TYPE apr_memory_bytes gauge
apr_memory_bytes{{type="model"}} {model_bytes}

# HELP apr_uptime_seconds Server uptime in seconds
# TYPE apr_uptime_seconds gauge
apr_uptime_seconds {uptime}
"#,
            inference_ms as f64 / 1000.0
        )
    }
}

// =============================================================================
// OpenAI-Compatible Tool Calling Types (GH-160, PMAT-186)
// =============================================================================

/// Tool definition for function calling (OpenAI-compatible)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Tool {
    /// Tool type (always "function" for now)
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: FunctionDef,
}

/// Function definition within a tool
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FunctionDef {
    /// Function name (e.g., "get_weather")
    pub name: String,
    /// Function description for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool call generated by the model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// Tool type (always "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function call details
    pub function: FunctionCall,
}

/// Function call within a tool call
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FunctionCall {
    /// Function name being called
    pub name: String,
    /// Arguments as JSON string
    pub arguments: String,
}

/// Tool choice option
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String options: "none", "auto", "required"
    Mode(String),
    /// Specific function to call
    Function {
        #[serde(rename = "type")]
        tool_type: String,
        function: ToolChoiceFunction,
    },
}

/// Specific function for tool_choice
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// Chat message with tool support (OpenAI-compatible)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    /// Role: system, user, assistant, tool
    pub role: String,
    /// Text content (optional for assistant messages with tool_calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls made by assistant (assistant messages only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID this message responds to (tool messages only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Function name (deprecated, for tool messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Chat completion request with tool support
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatCompletionRequest {
    /// Model name (ignored, uses loaded model)
    #[serde(default)]
    pub model: String,
    /// Conversation messages
    pub messages: Vec<ChatMessage>,
    /// Available tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Tool choice mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Enable streaming
    #[serde(default)]
    pub stream: bool,
    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

/// Chat completion response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique response ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model name
    pub model: String,
    /// Response choices
    pub choices: Vec<ChatChoice>,
    /// Token usage
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

/// Chat completion choice
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatChoice {
    /// Choice index
    pub index: u32,
    /// Response message
    pub message: ChatMessage,
    /// Finish reason: "stop", "length", "tool_calls"
    pub finish_reason: Option<String>,
}

/// Token usage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(clippy::struct_field_names)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Format tools into prompt for model (ChatML format with tool definitions)
pub(super) fn format_tools_prompt(tools: &[Tool]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut prompt = String::from("\n\nYou have access to the following tools:\n\n");

    for tool in tools {
        prompt.push_str(&format!("### {}\n", tool.function.name));
        if let Some(desc) = &tool.function.description {
            prompt.push_str(&format!("{}\n", desc));
        }
        if let Some(params) = &tool.function.parameters {
            prompt.push_str(&format!("Parameters: {}\n", params));
        }
        prompt.push('\n');
    }

    prompt.push_str("To use a tool, respond with a JSON object in this format:\n");
    prompt.push_str(r#"{"tool_call": {"name": "function_name", "arguments": {...}}}"#);
    prompt.push_str("\n\nIf you don't need to use a tool, respond normally.\n");

    prompt
}

/// Parse model output to detect tool calls
pub(super) fn parse_tool_calls(output: &str) -> Option<Vec<ToolCall>> {
    let output_trimmed = output.trim();

    // Try to parse entire output as JSON with tool_call field
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(output_trimmed) {
        if let Some(call) = extract_tool_call(&parsed) {
            return Some(vec![call]);
        }
    }

    // Check for embedded JSON in text
    if let Some(json_str) = find_embedded_tool_json(output_trimmed) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
            if let Some(call) = extract_tool_call(&parsed) {
                return Some(vec![call]);
            }
        }
    }

    None
}

/// Extract a ToolCall from a parsed JSON value containing a "tool_call" field.
fn extract_tool_call(parsed: &serde_json::Value) -> Option<ToolCall> {
    let tool_call = parsed.get("tool_call")?;
    let name = tool_call.get("name")?.as_str()?;
    let arguments = tool_call.get("arguments")?;

    Some(ToolCall {
        id: format!("call_{}", uuid_simple()),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    })
}

/// Find embedded `{"tool_call"...}` JSON in text, returning the balanced JSON substring.
fn find_embedded_tool_json(text: &str) -> Option<String> {
    let start = text.find(r#"{"tool_call""#)?;
    let json_part = &text[start..];
    let mut depth = 0;
    for (i, c) in json_part.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(json_part[..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Generate simple UUID for tool call IDs
pub(super) fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:016x}", timestamp & 0xFFFFFFFFFFFFFFFF)
}

// =============================================================================
// End Tool Calling Types
// =============================================================================

/// Server state
///
/// Stores model and configuration for the server.
/// Implements APR-SPEC §4.15.8.2 Server Architecture.
pub struct ServerState {
    /// Model path
    pub model_path: PathBuf,
    /// Model ID (filename without extension)
    pub model_id: String,
    /// Server configuration
    pub config: ServerConfig,
    /// Metrics
    pub metrics: Arc<ServerMetrics>,
    /// Whether model uses mmap (SL03: >50MB threshold)
    pub uses_mmap: bool,
    /// Model file size in bytes
    pub model_size_bytes: u64,
    /// Server readiness state
    pub ready: std::sync::atomic::AtomicBool,
}

impl ServerState {
    /// Create new server state
    pub fn new(model_path: PathBuf, config: ServerConfig) -> Result<Self> {
        // Check model exists (SL06: Invalid model path returns exit 1)
        if !model_path.exists() {
            return Err(CliError::FileNotFound(model_path));
        }

        // Get model metadata
        let metadata = std::fs::metadata(&model_path)?;
        let model_size_bytes = metadata.len();

        // Determine if mmap should be used (SL03: >50MB threshold)
        let uses_mmap = model_size_bytes > 50 * 1024 * 1024;

        // Extract model ID from filename
        let model_id = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let metrics = ServerMetrics::new();
        metrics
            .model_memory_bytes
            .store(model_size_bytes, Ordering::Relaxed);

        Ok(Self {
            model_path,
            model_id,
            config,
            metrics,
            uses_mmap,
            model_size_bytes,
            ready: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Mark server as ready (SL10: Ready only after model loaded)
    pub fn set_ready(&self) {
        self.ready.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Check if server is ready
    pub fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Health status enum (HR06, HR07)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Server is healthy and ready
    Healthy,
    /// Server is experiencing degraded performance (HR06: p99 > 1s)
    Degraded,
    /// Server is unhealthy (HR07: OOM or critical error)
    Unhealthy,
}

/// Health check response (HR01-HR10)
///
/// Implements APR-SPEC §4.15.8.3 HealthResponse schema.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthResponse {
    /// Health status (HR01, HR06, HR07)
    pub status: HealthStatus,
    /// Model identifier (HR03)
    pub model_id: String,
    /// Server version (SL09)
    pub version: String,
    /// Uptime in seconds (HR04: monotonically increases)
    pub uptime_seconds: u64,
    /// Total requests processed (HR05)
    pub requests_total: u64,
    /// GPU availability (HR08)
    pub gpu_available: bool,
}

/// Error response (EH09: Error response is valid JSON)
///
/// Implements APR-SPEC §4.15.8.7 error format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorResponse {
    /// Error type
    pub error: String,
    /// Human-readable message
    pub message: String,
    /// Request correlation ID (EH10)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

impl ErrorResponse {
    /// Create a new error response
    pub fn new(error: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            message: message.into(),
            request_id: None,
        }
    }

    /// Create with request ID
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }
}

/// Generate request (§4.15.8.3 GenerateRequest)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerateRequest {
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate (IC02)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Temperature for sampling (IC01, IC04)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Enable streaming (SP01-SP10)
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences (IC03)
    #[serde(default)]
    pub stop: Vec<String>,
}

fn default_max_tokens() -> u32 {
    256
}

fn default_temperature() -> f32 {
    1.0
}

/// Generate response (§4.15.8.3 GenerateResponse)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenerateResponse {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: u32,
    /// Reason for stopping (stop, length, error)
    pub finish_reason: String,
    /// Latency in milliseconds (LG03)
    pub latency_ms: u64,
}

/// SSE Stream Event (SP02-SP03)
#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamEvent {
    /// Event type: token, done, error
    pub event: String,
    /// Token text or error message
    pub data: String,
    /// Token ID (SP07, SP08)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_id: Option<u32>,
}

impl StreamEvent {
    /// Create token event
    pub fn token(text: &str, id: u32) -> Self {
        Self {
            event: "token".to_string(),
            data: text.to_string(),
            token_id: Some(id),
        }
    }

    /// Create done event (SP03)
    pub fn done(finish_reason: &str, tokens: u32) -> Self {
        Self {
            event: "done".to_string(),
            data: format!(
                r#"{{"finish_reason":"{}","tokens_generated":{}}}"#,
                finish_reason, tokens
            ),
            token_id: None,
        }
    }

    /// Create error event (SP04)
    pub fn error(message: &str) -> Self {
        Self {
            event: "error".to_string(),
            data: message.to_string(),
            token_id: None,
        }
    }

    /// Format as SSE (SP02)
    pub fn to_sse(&self) -> String {
        let mut result = format!("event: {}\n", self.event);
        result.push_str(&format!("data: {}\n\n", self.data));
        result
    }
}

/// Transcribe request (§4.15.8.3 TranscribeRequest)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscribeRequest {
    /// Language code (optional)
    #[serde(default)]
    pub language: Option<String>,
    /// Task: transcribe or translate
    #[serde(default = "default_task")]
    pub task: String,
}

fn default_task() -> String {
    "transcribe".to_string()
}

/// Transcribe response (§4.15.8.3 TranscribeResponse)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscribeResponse {
    /// Transcribed text
    pub text: String,
    /// Detected/specified language
    pub language: String,
    /// Audio duration in seconds
    pub duration_seconds: f64,
    /// Latency in milliseconds (LG03)
    pub latency_ms: u64,
}

/// Request size limit (SE02: default 10MB)
pub const MAX_REQUEST_SIZE: usize = 10 * 1024 * 1024;

/// Server information response (SL09)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Server version (semver)
    pub version: String,
    /// Loaded model ID
    pub model_id: String,
}

/// Generate health response from server state
pub fn health_check(state: &ServerState) -> HealthResponse {
    let metrics = &state.metrics;

    // Determine health status (HR06, HR07)
    let status = if !state.is_ready() {
        HealthStatus::Unhealthy
    } else {
        // Check for degraded performance (p99 > 1s approximation)
        let total_requests = metrics.requests_total.load(Ordering::Relaxed);
        let total_time_ms = metrics.inference_time_ms.load(Ordering::Relaxed);
        let avg_latency_ms = if total_requests > 0 {
            total_time_ms / total_requests
        } else {
            0
        };

        if avg_latency_ms > 1000 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    };

    // Detect GPU availability
    #[cfg(feature = "inference")]
    let gpu_available = {
        use realizar::cuda::CudaExecutor;
        CudaExecutor::is_available() && CudaExecutor::num_devices() > 0
    };
    #[cfg(not(feature = "inference"))]
    let gpu_available = false;

    HealthResponse {
        status,
        model_id: state.model_id.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: metrics.uptime_seconds(),
        requests_total: metrics.requests_total.load(Ordering::Relaxed),
        gpu_available,
    }
}
