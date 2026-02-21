
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
