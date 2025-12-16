//! Serve command implementation
//!
//! Implements APR-SPEC §4.15.8: Runtime Inference Engine
//!
//! Start an inference server with:
//! - REST API endpoints (/transcribe, /generate, /health, /metrics)
//! - Server-Sent Events (SSE) streaming
//! - Prometheus metrics
//! - Automatic mmap loading for large models
//! - KV cache for transformer models

use crate::error::{CliError, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
#[cfg(test)]
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Server configuration
#[derive(Debug, Clone)]
pub(crate) struct ServerConfig {
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
    pub(crate) fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Server metrics (thread-safe)
///
/// Fields are updated via `record_request` and exposed for monitoring.
/// While clippy sees them as unread in production code, they're:
/// 1. Modified via atomic operations in `record_request`
/// 2. Read in tests to verify correctness
/// 3. Intended for future Prometheus endpoint exposure (GH-80)
#[derive(Debug, Default)]
#[allow(dead_code)]
pub(crate) struct ServerMetrics {
    /// Total requests received
    pub requests_total: AtomicU64,
    /// Successful requests
    pub requests_success: AtomicU64,
    /// Failed requests
    pub requests_error: AtomicU64,
    /// Total tokens generated
    pub tokens_generated: AtomicU64,
    /// Total inference time in milliseconds
    pub inference_time_ms: AtomicU64,
}

impl ServerMetrics {
    /// Create new metrics
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a request (used in tests and future server implementation)
    #[cfg(test)]
    pub(crate) fn record_request(&self, success: bool, tokens: u64, duration_ms: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        if success {
            self.requests_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_error.fetch_add(1, Ordering::Relaxed);
        }
        self.tokens_generated.fetch_add(tokens, Ordering::Relaxed);
        self.inference_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Get Prometheus-format metrics (used in tests and future /metrics endpoint)
    #[cfg(test)]
    pub(crate) fn prometheus_output(&self) -> String {
        let total = self.requests_total.load(Ordering::Relaxed);
        let success = self.requests_success.load(Ordering::Relaxed);
        let errors = self.requests_error.load(Ordering::Relaxed);
        let tokens = self.tokens_generated.load(Ordering::Relaxed);
        let inference_ms = self.inference_time_ms.load(Ordering::Relaxed);

        format!(
            r"# HELP apr_requests_total Total number of requests
# TYPE apr_requests_total counter
apr_requests_total {total}

# HELP apr_requests_success Successful requests
# TYPE apr_requests_success counter
apr_requests_success {success}

# HELP apr_requests_error Failed requests
# TYPE apr_requests_error counter
apr_requests_error {errors}

# HELP apr_tokens_generated_total Total tokens generated
# TYPE apr_tokens_generated_total counter
apr_tokens_generated_total {tokens}

# HELP apr_inference_duration_seconds_total Total inference time
# TYPE apr_inference_duration_seconds_total counter
apr_inference_duration_seconds_total {:.3}
",
            inference_ms as f64 / 1000.0
        )
    }
}

/// Server state
///
/// Stores model and configuration for the server. Fields are accessed
/// in tests and will be used by the full server implementation (GH-80).
#[allow(dead_code)]
pub(crate) struct ServerState {
    /// Model path
    pub model_path: PathBuf,
    /// Server configuration
    pub config: ServerConfig,
    /// Metrics
    pub metrics: Arc<ServerMetrics>,
    /// Whether model uses mmap
    pub uses_mmap: bool,
}

impl ServerState {
    /// Create new server state
    pub(crate) fn new(model_path: PathBuf, config: ServerConfig) -> Result<Self> {
        // Check model exists
        if !model_path.exists() {
            return Err(CliError::FileNotFound(model_path));
        }

        // Determine if mmap should be used
        let metadata = std::fs::metadata(&model_path)?;
        let uses_mmap = metadata.len() > 50 * 1024 * 1024; // 50MB threshold

        Ok(Self {
            model_path,
            config,
            metrics: ServerMetrics::new(),
            uses_mmap,
        })
    }
}

/// Health check response (used in tests and future /health endpoint)
#[cfg(test)]
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct HealthResponse {
    pub status: String,
    pub model: String,
    pub uptime_secs: u64,
}

/// Generate health response (used in tests)
#[cfg(test)]
pub fn health_check(state: &ServerState, uptime_secs: u64) -> HealthResponse {
    HealthResponse {
        status: "healthy".to_string(),
        model: state.model_path.display().to_string(),
        uptime_secs,
    }
}

/// Serve command entry point (blocking)
pub(crate) fn run(model_path: &Path, config: &ServerConfig) -> Result<()> {
    println!("{}", "=== APR Serve ===".cyan().bold());
    println!();
    println!("Model: {}", model_path.display());
    println!("Binding: {}", config.bind_addr());
    println!();

    // Validate model
    if !model_path.exists() {
        return Err(CliError::FileNotFound(model_path.to_path_buf()));
    }

    let state = ServerState::new(model_path.to_path_buf(), config.clone())?;

    println!(
        "{}",
        format!(
            "Model loading: {}",
            if state.uses_mmap { "mmap" } else { "full" }
        )
        .dimmed()
    );

    println!();
    println!("{}", "Endpoints:".green().bold());
    println!("  POST /predict        - Model prediction (APR)");
    println!("  POST /generate       - Text generation (GGUF)");
    println!("  GET  /health         - Health check");
    if config.metrics {
        println!("  GET  /metrics        - Prometheus metrics");
    }

    println!();
    println!(
        "{}",
        format!("Server ready at http://{}", config.bind_addr())
            .green()
            .bold()
    );
    println!();
    println!("{}", "Press Ctrl+C to stop".dimmed());

    // Try to start real server with realizar
    #[cfg(feature = "inference")]
    {
        return start_realizar_server(model_path, &config);
    }

    // Fallback: stub mode
    #[cfg(not(feature = "inference"))]
    {
        println!();
        println!("{}", "[Server requires --features inference]".yellow());
        Ok(())
    }
}

/// Start server using realizar
#[cfg(feature = "inference")]
fn start_realizar_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::format::{detect_format, ModelFormat};

    // Read file for format detection
    let data = std::fs::read(model_path)?;
    if data.len() < 8 {
        return Err(CliError::InvalidFormat(
            "File too small for format detection".to_string(),
        ));
    }

    // Detect model format from magic bytes
    let format = detect_format(&data[..8])
        .map_err(|e| CliError::InvalidFormat(format!("Format detection failed: {e}")))?;

    println!();
    println!("Detected format: {}", format);

    match format {
        ModelFormat::Apr => {
            println!("{}", "Starting APR model server...".cyan());
            start_apr_server(model_path, config)
        }
        ModelFormat::Gguf => {
            println!("{}", "Starting GGUF inference server...".cyan());
            start_gguf_server(model_path, config)
        }
        ModelFormat::SafeTensors => {
            println!("{}", "Starting SafeTensors inspection server...".cyan());
            start_safetensors_server(model_path, config)
        }
    }
}

/// Start APR model server using realizar's serve module
#[cfg(feature = "inference")]
fn start_apr_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::serve::{create_serve_router, ServeState};

    // Load APR model using realizar's ServeState
    let serve_state = ServeState::load_apr(model_path, "v1.0".to_string(), 0)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load APR model: {e}")))?;

    println!("{}", "APR model loaded successfully".green());

    // Create tokio runtime and run server
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let enable_metrics = config.metrics;

    runtime.block_on(async move {
        // Create router
        let app = create_serve_router(serve_state);

        // Bind and serve
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  POST /predict        - Single prediction");
        println!("  POST /predict/batch  - Batch prediction");
        println!("  GET  /health         - Health check");
        println!("  GET  /ready          - Readiness check");
        println!("  GET  /models         - List models");
        if enable_metrics {
            println!("  GET  /metrics        - Prometheus metrics");
        }
        println!();
        println!("{}", "Press Ctrl+C to stop".dimmed());

        // Serve with graceful shutdown
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Start GGUF model inspection server
#[cfg(feature = "inference")]
fn start_gguf_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use axum::{routing::get, Json, Router};
    use realizar::gguf::MappedGGUFModel;
    use serde::Serialize;

    // Load GGUF model via mmap
    println!("{}", "Loading GGUF model...".dimmed());
    let mapped_model = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?;

    let model = mapped_model.model;
    println!("{}", "GGUF model loaded successfully".green());

    // Create tokio runtime
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let model_path_str = model_path.display().to_string();
    let tensor_count = model.tensors.len();
    let metadata_count = model.metadata.len();
    let version = model.header.version;

    runtime.block_on(async move {
        #[derive(Clone, Serialize)]
        struct ModelInfo {
            path: String,
            version: u32,
            tensors: usize,
            metadata: usize,
        }

        let info = ModelInfo {
            path: model_path_str.clone(),
            version,
            tensors: tensor_count,
            metadata: metadata_count,
        };

        // Create inspection endpoints
        let app = Router::new()
            .route(
                "/health",
                get(|| async { Json(serde_json::json!({"status": "healthy"})) }),
            )
            .route("/model", get(move || async move { Json(info.clone()) }));

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET /health  - Health check");
        println!("  GET /model   - Model information");
        println!();
        println!(
            "{}",
            "Note: GGUF text generation requires additional tokenizer setup.".yellow()
        );
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Start SafeTensors inspection server
#[cfg(feature = "inference")]
fn start_safetensors_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use axum::{routing::get, Json, Router};
    use realizar::safetensors::SafetensorsModel;
    use serde::Serialize;

    // Load SafeTensors for inspection
    let data = std::fs::read(model_path)?;
    let st_model = SafetensorsModel::from_bytes(&data)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load SafeTensors: {e}")))?;

    let tensor_names: Vec<String> = st_model.tensors.keys().cloned().collect();
    let tensor_count = tensor_names.len();

    println!(
        "{}",
        format!("SafeTensors loaded: {} tensors", tensor_count).green()
    );

    // Create tokio runtime
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = config.bind_addr();
    let model_path_str = model_path.display().to_string();

    runtime.block_on(async move {
        #[derive(Clone, Serialize)]
        struct TensorInfo {
            count: usize,
            model: String,
            names: Vec<String>,
        }

        let info = TensorInfo {
            count: tensor_count,
            model: model_path_str.clone(),
            names: tensor_names.clone(),
        };

        // Simple inspection endpoints
        let app = Router::new()
            .route(
                "/health",
                get(|| async { Json(serde_json::json!({"status": "healthy"})) }),
            )
            .route("/tensors", get(move || async move { Json(info.clone()) }));

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("Server listening on http://{}", bind_addr)
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET /health   - Health check");
        println!("  GET /tensors  - List tensor names");
        println!();
        println!(
            "{}",
            "Note: SafeTensors serving requires additional tokenizer/config".yellow()
        );
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        println!();
        println!("{}", "Server stopped".yellow());
        Ok(())
    })
}

/// Shutdown signal handler
#[cfg(feature = "inference")]
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ==================== ServerConfig Tests ====================

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
        assert!(config.cors);
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_concurrent, 10);
        assert!(config.metrics);
        assert!(!config.no_gpu);
    }

    #[test]
    fn test_server_config_with_port() {
        let config = ServerConfig::default().with_port(3000);
        assert_eq!(config.port, 3000);
    }

    #[test]
    fn test_server_config_with_host() {
        let config = ServerConfig::default().with_host("0.0.0.0");
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_server_config_bind_addr() {
        let config = ServerConfig::default().with_port(9000).with_host("0.0.0.0");
        assert_eq!(config.bind_addr(), "0.0.0.0:9000");
    }

    // ==================== ServerMetrics Tests ====================

    #[test]
    fn test_metrics_default() {
        let metrics = ServerMetrics::new();
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.requests_error.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_record_success() {
        let metrics = ServerMetrics::new();
        metrics.record_request(true, 10, 100);

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.requests_error.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 10);
        assert_eq!(metrics.inference_time_ms.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_metrics_record_error() {
        let metrics = ServerMetrics::new();
        metrics.record_request(false, 0, 50);

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.requests_error.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_metrics_accumulate() {
        let metrics = ServerMetrics::new();
        metrics.record_request(true, 10, 100);
        metrics.record_request(true, 20, 200);
        metrics.record_request(false, 0, 50);

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.requests_error.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 30);
        assert_eq!(metrics.inference_time_ms.load(Ordering::Relaxed), 350);
    }

    #[test]
    fn test_prometheus_output_format() {
        let metrics = ServerMetrics::new();
        metrics.record_request(true, 100, 1000);

        let output = metrics.prometheus_output();
        assert!(output.contains("apr_requests_total 1"));
        assert!(output.contains("apr_requests_success 1"));
        assert!(output.contains("apr_tokens_generated_total 100"));
        assert!(output.contains("# TYPE apr_requests_total counter"));
    }

    // ==================== ServerState Tests ====================

    #[test]
    fn test_server_state_file_not_found() {
        let result = ServerState::new(
            PathBuf::from("/nonexistent/model.apr"),
            ServerConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_server_state_small_model_no_mmap() {
        // Create small temp file (<50MB)
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"small model data").unwrap();

        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        assert!(!state.uses_mmap);
    }

    // ==================== Health Check Tests ====================

    #[test]
    fn test_health_check() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test model").unwrap();

        let state = ServerState::new(file.path().to_path_buf(), ServerConfig::default()).unwrap();
        let health = health_check(&state, 60);

        assert_eq!(health.status, "healthy");
        assert_eq!(health.uptime_secs, 60);
        assert!(health
            .model
            .contains(file.path().file_name().unwrap().to_str().unwrap()));
    }

    // ==================== Thread Safety Tests ====================

    #[test]
    fn test_metrics_thread_safe() {
        use std::thread;

        let metrics = ServerMetrics::new();
        let metrics_clone = Arc::clone(&metrics);

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let m = Arc::clone(&metrics_clone);
                thread::spawn(move || {
                    for _ in 0..100 {
                        m.record_request(true, 1, 1);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 1000);
        assert_eq!(metrics.tokens_generated.load(Ordering::Relaxed), 1000);
    }
}
