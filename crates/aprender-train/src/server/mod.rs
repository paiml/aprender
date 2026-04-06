//! REST/HTTP API Server (#67)
//!
//! Remote access to experiment tracking with built-in quality stops.
//!
//! # Toyota Principle: Jidoka (自働化)
//!
//! Built-in quality - Remote access enables team-wide visibility while
//! maintaining quality through input validation and error handling.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::server::{TrackingServer, ServerConfig};
//! use std::net::SocketAddr;
//!
//! let config = ServerConfig::default();
//! let server = TrackingServer::new(config);
//! server.run("127.0.0.1:5000".parse().expect("valid address")).await?;
//! ```

#[cfg(feature = "server")]
mod api;
#[cfg(feature = "server")]
mod handlers;
#[cfg(feature = "server")]
mod state;

#[cfg(feature = "server")]
pub use api::*;
#[cfg(feature = "server")]
pub use handlers::*;
#[cfg(feature = "server")]
pub use state::*;

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use thiserror::Error;

/// Server errors
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("Bind error: {0}")]
    Bind(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for server operations
pub type Result<T> = std::result::Result<T, ServerError>;

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server address
    pub address: SocketAddr,
    /// Enable CORS
    pub cors_enabled: bool,
    /// Allowed origins for CORS
    pub cors_origins: Vec<String>,
    /// API key for authentication (None = no auth)
    pub api_key: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum request body size in bytes
    pub max_body_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:5000".parse().expect("default server address must be valid"),
            cors_enabled: true,
            cors_origins: vec!["*".to_string()],
            api_key: None,
            timeout_secs: 30,
            max_body_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

impl ServerConfig {
    /// Create config with custom address
    pub fn with_address(mut self, addr: SocketAddr) -> Self {
        self.address = addr;
        self
    }

    /// Create config with API key authentication
    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    /// Disable CORS
    pub fn without_cors(mut self) -> Self {
        self.cors_enabled = false;
        self
    }

    /// Set allowed CORS origins
    pub fn with_cors_origins(mut self, origins: Vec<String>) -> Self {
        self.cors_origins = origins;
        self
    }
}

/// API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Whether the request was successful
    pub success: bool,
    /// Response data (if successful)
    pub data: Option<T>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Request ID for tracing
    pub request_id: String,
}

impl<T> ApiResponse<T> {
    /// Create success response
    pub fn success(data: T, request_id: &str) -> Self {
        Self { success: true, data: Some(data), error: None, request_id: request_id.to_string() }
    }

    /// Create error response
    pub fn error(message: &str, request_id: &str) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.to_string()),
            request_id: request_id.to_string(),
        }
    }
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Server status
    pub status: String,
    /// Server version
    pub version: String,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Number of active experiments
    pub experiments_count: usize,
    /// Number of active runs
    pub runs_count: usize,
}

// =============================================================================
// Request/Response DTOs
// =============================================================================

/// Create experiment request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateExperimentRequest {
    /// Experiment name
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Optional tags
    pub tags: Option<std::collections::HashMap<String, String>>,
}

/// Create run request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateRunRequest {
    /// Experiment ID
    pub experiment_id: String,
    /// Optional run name
    pub name: Option<String>,
    /// Optional tags
    pub tags: Option<std::collections::HashMap<String, String>>,
}

/// Log parameters request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogParamsRequest {
    /// Parameters to log
    pub params: std::collections::HashMap<String, serde_json::Value>,
}

/// Log metrics request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogMetricsRequest {
    /// Metrics to log (name -> value)
    pub metrics: std::collections::HashMap<String, f64>,
    /// Optional step number
    pub step: Option<u64>,
}

/// Update run request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRunRequest {
    /// New status
    pub status: Option<String>,
    /// End time (ISO 8601)
    pub end_time: Option<String>,
}

/// Experiment response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResponse {
    /// Experiment ID
    pub id: String,
    /// Experiment name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Creation time
    pub created_at: String,
    /// Tags
    pub tags: std::collections::HashMap<String, String>,
}

/// Run response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponse {
    /// Run ID
    pub id: String,
    /// Experiment ID
    pub experiment_id: String,
    /// Run name
    pub name: Option<String>,
    /// Status
    pub status: String,
    /// Start time
    pub start_time: String,
    /// End time
    pub end_time: Option<String>,
    /// Parameters
    pub params: std::collections::HashMap<String, serde_json::Value>,
    /// Latest metrics
    pub metrics: std::collections::HashMap<String, f64>,
    /// Tags
    pub tags: std::collections::HashMap<String, String>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.address.port(), 5000);
        assert!(config.cors_enabled);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_server_config_with_address() {
        let addr: SocketAddr = "0.0.0.0:8080".parse().expect("parsing should succeed");
        let config = ServerConfig::default().with_address(addr);
        assert_eq!(config.address.port(), 8080);
    }

    #[test]
    fn test_server_config_with_api_key() {
        let config = ServerConfig::default().with_api_key("secret123");
        assert_eq!(config.api_key, Some("secret123".to_string()));
    }

    #[test]
    fn test_server_config_without_cors() {
        let config = ServerConfig::default().without_cors();
        assert!(!config.cors_enabled);
    }

    #[test]
    fn test_api_response_success() {
        let response = ApiResponse::success("hello", "req-123");
        assert!(response.success);
        assert_eq!(response.data, Some("hello"));
        assert!(response.error.is_none());
    }

    #[test]
    fn test_api_response_error() {
        let response: ApiResponse<String> = ApiResponse::error("not found", "req-456");
        assert!(!response.success);
        assert!(response.data.is_none());
        assert_eq!(response.error, Some("not found".to_string()));
    }

    #[test]
    fn test_health_response_serialize() {
        let health = HealthResponse {
            status: "healthy".to_string(),
            version: "0.2.3".to_string(),
            uptime_secs: 3600,
            experiments_count: 10,
            runs_count: 50,
        };
        let json = serde_json::to_string(&health).expect("JSON serialization should succeed");
        assert!(json.contains("healthy"));
    }

    #[test]
    fn test_create_experiment_request() {
        let json = r#"{"name": "test-exp", "description": "A test"}"#;
        let req: CreateExperimentRequest =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert_eq!(req.name, "test-exp");
        assert_eq!(req.description, Some("A test".to_string()));
    }

    #[test]
    fn test_create_run_request() {
        let json = r#"{"experiment_id": "exp-123", "name": "run-1"}"#;
        let req: CreateRunRequest =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert_eq!(req.experiment_id, "exp-123");
        assert_eq!(req.name, Some("run-1".to_string()));
    }

    #[test]
    fn test_log_params_request() {
        let json = r#"{"params": {"lr": 0.001, "batch_size": 32}}"#;
        let req: LogParamsRequest =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert!(req.params.contains_key("lr"));
        assert!(req.params.contains_key("batch_size"));
    }

    #[test]
    fn test_log_metrics_request() {
        let json = r#"{"metrics": {"loss": 0.5, "accuracy": 0.9}, "step": 100}"#;
        let req: LogMetricsRequest =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert_eq!(req.metrics.get("loss"), Some(&0.5));
        assert_eq!(req.step, Some(100));
    }

    #[test]
    fn test_update_run_request() {
        let json = r#"{"status": "completed", "end_time": "2024-01-15T10:30:00Z"}"#;
        let req: UpdateRunRequest =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert_eq!(req.status, Some("completed".to_string()));
    }

    #[test]
    fn test_experiment_response_serialize() {
        let exp = ExperimentResponse {
            id: "exp-123".to_string(),
            name: "My Experiment".to_string(),
            description: Some("Test".to_string()),
            created_at: "2024-01-15T10:00:00Z".to_string(),
            tags: std::collections::HashMap::new(),
        };
        let json = serde_json::to_string(&exp).expect("JSON serialization should succeed");
        assert!(json.contains("exp-123"));
    }

    #[test]
    fn test_run_response_serialize() {
        let run = RunResponse {
            id: "run-456".to_string(),
            experiment_id: "exp-123".to_string(),
            name: Some("training-run".to_string()),
            status: "running".to_string(),
            start_time: "2024-01-15T10:00:00Z".to_string(),
            end_time: None,
            params: std::collections::HashMap::new(),
            metrics: std::collections::HashMap::new(),
            tags: std::collections::HashMap::new(),
        };
        let json = serde_json::to_string(&run).expect("JSON serialization should succeed");
        assert!(json.contains("run-456"));
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_server_config_port_preserved(port in 1024u16..65535) {
            let addr: SocketAddr = format!("127.0.0.1:{port}").parse().expect("parsing should succeed");
            let config = ServerConfig::default().with_address(addr);
            prop_assert_eq!(config.address.port(), port);
        }

        #[test]
        fn prop_api_response_success_has_data(data in "[a-zA-Z0-9]{1,100}") {
            let response = ApiResponse::success(data.clone(), "req-1");
            prop_assert!(response.success);
            prop_assert_eq!(response.data, Some(data));
        }

        #[test]
        fn prop_api_response_error_has_message(msg in "[a-zA-Z0-9 ]{1,100}") {
            let response: ApiResponse<String> = ApiResponse::error(&msg, "req-1");
            prop_assert!(!response.success);
            prop_assert_eq!(response.error, Some(msg));
        }

        #[test]
        fn prop_create_experiment_roundtrip(name in "[a-zA-Z0-9-]{1,50}") {
            let req = CreateExperimentRequest {
                name: name.clone(),
                description: None,
                tags: None,
            };
            let json = serde_json::to_string(&req).expect("JSON serialization should succeed");
            let parsed: CreateExperimentRequest = serde_json::from_str(&json).expect("JSON deserialization should succeed");
            prop_assert_eq!(parsed.name, name);
        }

        #[test]
        fn prop_log_metrics_roundtrip(
            metric_name in "[a-z_]{1,20}",
            value in -1000.0f64..1000.0
        ) {
            let mut metrics = std::collections::HashMap::new();
            metrics.insert(metric_name.clone(), value);
            let req = LogMetricsRequest { metrics, step: None };
            let json = serde_json::to_string(&req).expect("JSON serialization should succeed");
            let parsed: LogMetricsRequest = serde_json::from_str(&json).expect("JSON deserialization should succeed");
            prop_assert!((parsed.metrics.get(&metric_name).expect("parsing should succeed") - value).abs() < 1e-10);
        }
    }
}
