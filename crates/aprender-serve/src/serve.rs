//! Aprender ML Model Serving API
//!
//! HTTP REST API for serving classical ML models in `.apr` format from aprender.
//! Separate from the LLM inference API in `api.rs`.
//!
//! ## The `.apr` Format
//!
//! Aprender's proprietary binary format with built-in quality (Jidoka):
//! - CRC32 checksum (integrity verification)
//! - Ed25519 signatures (provenance)
//! - AES-256-GCM encryption (confidentiality)
//! - Zstd compression (efficiency)
//! - Quantization support (`Q4_0`, `Q8_0`)
//! - Streaming/mmap (JIT loading)
//!
//! ## Endpoints
//!
//! - `GET /health` - Health check / liveness probe
//! - `GET /ready` - Readiness check (model loaded)
//! - `POST /predict` - Single prediction
//! - `POST /predict/batch` - Batch predictions
//! - `GET /models` - List loaded models
//! - `GET /metrics` - Prometheus metrics
//!
//! ## Architecture
//!
//! Per `docs/specifications/serve-deploy-apr.md`:
//! - Pure Rust, WASM-compatible
//! - Sub-10ms p50 latency target (actual: ~0.5µs)
//! - Supports single-binary deployment via `include_bytes!()`
//! - 9.6x faster than `PyTorch` (statistically validated)
//!
//! ## Example: Load from `.apr` File
//!
//! ```rust,ignore
//! use realizar::serve::{create_serve_router, ServeState};
//! use aprender::format::{load, ModelType};
//! use aprender::classification::LogisticRegression;
//!
//! // Load trained model from .apr format (with CRC32 verification)
//! let model: LogisticRegression = load("model.apr", ModelType::LogisticRegression).expect("test");
//! let state = ServeState::with_logistic_regression(model, "mnist-v1".to_string(), 784);
//! let app = create_serve_router(state);
//! axum::serve(listener, app).await?;
//! ```
//!
//! ## Example: Embedded Model (Single Binary)
//!
//! ```rust,ignore
//! use realizar::serve::{create_serve_router, ServeState};
//! use aprender::format::{load_from_bytes, ModelType};
//! use aprender::classification::LogisticRegression;
//!
//! // Embed model at compile time
//! const MODEL_BYTES: &[u8] = include_bytes!("../models/sentiment.apr");
//!
//! // Load from embedded bytes (zero-copy where possible)
//! let model: LogisticRegression = load_from_bytes(MODEL_BYTES, ModelType::LogisticRegression).expect("test");
//! let state = ServeState::with_logistic_regression(model, "sentiment-v1".to_string(), 768);
//! ```

use std::{sync::Arc, time::Instant};

#[cfg(feature = "aprender-serve")]
use aprender::{
    classification::{GaussianNB, KNearestNeighbors, LinearSVM, LogisticRegression},
    linear_model::LinearRegression,
    primitives::Matrix,
    tree::{DecisionTreeClassifier, GradientBoostingClassifier, RandomForestClassifier},
    Estimator,
};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

/// Loaded model variant for type-erased serving
///
/// Per spec §5: Supports all 18 APR model types.
/// Currently implemented: 8 most common prediction-capable types.
#[cfg(feature = "aprender-serve")]
#[derive(Clone)]
pub enum LoadedModel {
    // ===== Classification Models =====
    /// Logistic regression model (binary classification)
    LogisticRegression(Arc<LogisticRegression>),
    /// K-nearest neighbors classifier
    KNearestNeighbors(Arc<KNearestNeighbors>),
    /// Gaussian Naive Bayes classifier
    GaussianNB(Arc<GaussianNB>),
    /// Linear SVM classifier
    LinearSVM(Arc<LinearSVM>),
    /// Decision tree classifier (CART)
    DecisionTreeClassifier(Arc<DecisionTreeClassifier>),
    /// Random forest classifier (bagging ensemble)
    RandomForestClassifier(Arc<RandomForestClassifier>),
    /// Gradient boosting classifier
    GradientBoostingClassifier(Arc<GradientBoostingClassifier>),

    // ===== Regression Models =====
    /// Linear regression model (OLS/Ridge/Lasso)
    LinearRegression(Arc<LinearRegression>),
}

/// Application state for aprender model serving
#[derive(Clone)]
pub struct ServeState {
    /// Currently loaded model (type-erased)
    #[cfg(feature = "aprender-serve")]
    model: Option<LoadedModel>,
    /// Model name/identifier
    model_name: String,
    /// Model version
    model_version: String,
    /// Input feature dimension (for validation)
    input_dim: usize,
    /// Request counter for metrics
    request_count: Arc<std::sync::atomic::AtomicU64>,
}

impl ServeState {
    /// Create new serving state without a model (for testing/scaffolding)
    #[must_use]
    pub fn new(model_name: String, model_version: String) -> Self {
        Self {
            #[cfg(feature = "aprender-serve")]
            model: None,
            model_name,
            model_version,
            input_dim: 0,
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Create serving state with a loaded `LogisticRegression` model
    #[cfg(feature = "aprender-serve")]
    #[must_use]
    pub fn with_logistic_regression(
        model: LogisticRegression,
        model_version: String,
        input_dim: usize,
    ) -> Self {
        Self {
            model: Some(LoadedModel::LogisticRegression(Arc::new(model))),
            model_name: "LogisticRegression".to_string(),
            model_version,
            input_dim,
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Load a `LogisticRegression` model from an `.apr` file
    ///
    /// The `.apr` format provides:
    /// - CRC32 integrity verification on load
    /// - Optional Ed25519 signature verification
    /// - Optional AES-256-GCM decryption
    /// - Automatic Zstd decompression
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be read
    /// - CRC32 checksum fails (file corrupted)
    /// - Signature verification fails (if signed)
    /// - Decryption fails (if encrypted, wrong key)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use realizar::serve::ServeState;
    ///
    /// let state = ServeState::load_apr("model.apr", "v1.0".to_string(), 784)?;
    /// ```
    #[cfg(feature = "aprender-serve")]
    pub fn load_apr(
        path: impl AsRef<std::path::Path>,
        model_version: String,
        input_dim: usize,
    ) -> Result<Self, anyhow::Error> {
        use aprender::format::{load, ModelType};

        let model: LogisticRegression = load(path, ModelType::LogisticRegression)?;

        Ok(Self {
            model: Some(LoadedModel::LogisticRegression(Arc::new(model))),
            model_name: "LogisticRegression".to_string(),
            model_version,
            input_dim,
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Load a `LogisticRegression` model from embedded `.apr` bytes
    ///
    /// Use this with `include_bytes!()` for single-binary deployment:
    ///
    /// ```rust,ignore
    /// use realizar::serve::ServeState;
    ///
    /// const MODEL: &[u8] = include_bytes!("../models/sentiment.apr");
    ///
    /// let state = ServeState::load_apr_from_bytes(MODEL, "v1.0".to_string(), 768)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - CRC32 checksum fails (data corrupted)
    /// - Signature verification fails (if signed)
    /// - Decryption fails (if encrypted)
    #[cfg(feature = "aprender-serve")]
    pub fn load_apr_from_bytes(
        bytes: &[u8],
        model_version: String,
        input_dim: usize,
    ) -> Result<Self, anyhow::Error> {
        use aprender::format::{load_from_bytes, ModelType};

        let model: LogisticRegression = load_from_bytes(bytes, ModelType::LogisticRegression)?;

        Ok(Self {
            model: Some(LoadedModel::LogisticRegression(Arc::new(model))),
            model_name: "LogisticRegression".to_string(),
            model_version,
            input_dim,
            request_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Check if a model is loaded
    #[must_use]
    pub fn has_model(&self) -> bool {
        #[cfg(feature = "aprender-serve")]
        {
            self.model.is_some()
        }
        #[cfg(not(feature = "aprender-serve"))]
        {
            false
        }
    }
}

/// Health check response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Service version
    pub version: String,
}

/// Readiness check response
#[derive(Debug, Serialize, Deserialize)]
pub struct ReadyResponse {
    /// Ready status
    pub ready: bool,
    /// Model loaded
    pub model_loaded: bool,
    /// Model name
    pub model_name: String,
}

/// Prediction request
///
/// Per spec §5.2: Request schema for single prediction
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictRequest {
    /// Model ID (optional, uses default if not specified)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Input features as flat array
    pub features: Vec<f32>,
    /// Optional prediction options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<PredictOptions>,
}

/// Prediction options
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictOptions {
    /// Return class probabilities (for classification models)
    #[serde(default)]
    pub return_probabilities: bool,
    /// Return top-k predictions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
}

/// Prediction response
///
/// Per spec §5.2: Response schema for single prediction
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictResponse {
    /// Predicted value or class
    pub prediction: f32,
    /// Class probabilities (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<Vec<f32>>,
    /// Inference latency in milliseconds
    pub latency_ms: f64,
    /// Model version used
    pub model_version: String,
}

/// Batch prediction request
///
/// Per spec §5.2: Batch request schema
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchPredictRequest {
    /// Model ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// Multiple instances to predict
    pub instances: Vec<PredictInstance>,
}

/// Single instance in batch request
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictInstance {
    /// Input features
    pub features: Vec<f32>,
}

/// Batch prediction response
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchPredictResponse {
    /// Predictions for all instances
    pub predictions: Vec<PredictResponse>,
    /// Total batch processing time in milliseconds
    pub total_latency_ms: f64,
}

/// Models list response
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsResponse {
    /// Available models
    pub models: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model type (e.g., "`LinearRegression`", "`RandomForest`")
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Loaded status
    pub loaded: bool,
}

/// Error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error message
    pub error: String,
    /// Error code (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Create a standardized HTTP error response
#[inline]
fn http_error(
    status: StatusCode,
    message: impl Into<String>,
    code: &str,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: message.into(),
            code: Some(code.to_string()),
        }),
    )
}

/// Validate that model is loaded, return error if not
#[inline]
fn require_model(state: &ServeState) -> Result<&LoadedModel, (StatusCode, Json<ErrorResponse>)> {
    state.model.as_ref().ok_or_else(|| {
        http_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "No model loaded",
            "E_NO_MODEL",
        )
    })
}

/// Validate input dimensions match expected
#[inline]
fn validate_dimensions(
    expected: usize,
    actual: usize,
) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if expected > 0 && actual != expected {
        return Err(http_error(
            StatusCode::BAD_REQUEST,
            format!("Invalid input dimension: expected {expected}, got {actual}"),
            "E_INVALID_INPUT",
        ));
    }
    Ok(())
}

/// Type alias for prediction result
type HttpResult<T> = Result<T, (StatusCode, Json<ErrorResponse>)>;

/// Extract first prediction as f32 (macro to handle i32/usize/f32 return types)
#[cfg(feature = "aprender-serve")]
macro_rules! first_pred {
    ($preds:expr) => {{
        #[allow(clippy::cast_precision_loss)]
        {
            *$preds.first().unwrap_or(&Default::default()) as f32
        }
    }};
}

include!("serve_run_model.rs");
include!("serve_state_predict.rs");
