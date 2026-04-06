//! AWS Lambda Handler for Aprender ML Model Serving
//!
//! Single-binary deployment via `include_bytes!()` for sub-50ms cold starts.
//! Optimized for ARM64 Graviton processors.
//!
//! ## Architecture
//!
//! Per `docs/specifications/serve-deploy-apr.md` Section 7:
//! - Model embedded at compile time via `include_bytes!()`
//! - `OnceLock` for lazy model initialization (amortizes cold start)
//! - ARM64-native binary for Graviton Lambda
//! - Target: <50ms cold start, <10ms warm inference
//!
//! ## Usage
//!
//! ```rust,ignore
//! use realizar::lambda::{LambdaHandler, LambdaRequest, LambdaResponse};
//!
//! // Model bytes embedded at compile time
//! static MODEL_BYTES: &[u8] = include_bytes!("../models/model.apr");
//!
//! let handler = LambdaHandler::from_bytes(MODEL_BYTES)?;
//! ```

use std::{sync::OnceLock, time::Instant};

use serde::{Deserialize, Serialize};

use crate::apr::{AprModel, MAGIC as APR_MAGIC};

/// Embedded model bytes placeholder
/// In production: `static MODEL_BYTES: &[u8] = include_bytes!("../models/model.apr");`
pub static MODEL_BYTES: OnceLock<&'static [u8]> = OnceLock::new();

/// Lambda request format
///
/// Per spec §5.2: Compatible with `serve::PredictRequest`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaRequest {
    /// Input features for prediction
    pub features: Vec<f32>,
    /// Optional model ID (for multi-model deployments)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

/// Lambda response format
///
/// Per spec §5.2: Compatible with `serve::PredictResponse`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaResponse {
    /// Prediction result
    pub prediction: f32,
    /// Class probabilities (classification models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<Vec<f32>>,
    /// Inference latency in milliseconds
    pub latency_ms: f64,
    /// Cold start indicator (true on first invocation)
    pub cold_start: bool,
}

/// Batch Lambda request format
///
/// Per spec §5.3: Batch inference for throughput optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchLambdaRequest {
    /// Multiple instances to predict
    pub instances: Vec<LambdaRequest>,
    /// Optional: Maximum parallel workers (default: CPU count)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_parallelism: Option<usize>,
}

/// Batch Lambda response format
///
/// Per spec §5.3: Batch response with individual results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchLambdaResponse {
    /// Individual predictions
    pub predictions: Vec<LambdaResponse>,
    /// Total batch processing time in milliseconds
    pub total_latency_ms: f64,
    /// Number of successful predictions
    pub success_count: usize,
    /// Number of failed predictions
    pub error_count: usize,
}

/// Prometheus metrics for Lambda handler
///
/// Per spec §11.3: Production metrics endpoint
#[derive(Debug, Clone, Default)]
pub struct LambdaMetrics {
    /// Total requests processed
    pub requests_total: u64,
    /// Successful requests
    pub requests_success: u64,
    /// Failed requests
    pub requests_failed: u64,
    /// Total inference latency (for computing mean)
    pub latency_total_ms: f64,
    /// Cold starts observed
    pub cold_starts: u64,
    /// Batch requests processed
    pub batch_requests: u64,
}

impl LambdaMetrics {
    /// Create new empty metrics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful request
    pub fn record_success(&mut self, latency_ms: f64, is_cold_start: bool) {
        self.requests_total += 1;
        self.requests_success += 1;
        self.latency_total_ms += latency_ms;
        if is_cold_start {
            self.cold_starts += 1;
        }
    }

    /// Record a failed request
    pub fn record_failure(&mut self) {
        self.requests_total += 1;
        self.requests_failed += 1;
    }

    /// Record a batch request
    pub fn record_batch(&mut self, success_count: usize, error_count: usize, latency_ms: f64) {
        self.batch_requests += 1;
        self.requests_total += (success_count + error_count) as u64;
        self.requests_success += success_count as u64;
        self.requests_failed += error_count as u64;
        self.latency_total_ms += latency_ms;
    }

    /// Get average latency in milliseconds
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Acceptable precision loss for metrics
    pub fn avg_latency_ms(&self) -> f64 {
        if self.requests_success == 0 {
            0.0
        } else {
            self.latency_total_ms / self.requests_success as f64
        }
    }

    /// Export as Prometheus text format
    #[must_use]
    pub fn to_prometheus(&self) -> String {
        format!(
            "# HELP lambda_requests_total Total number of requests\n\
             # TYPE lambda_requests_total counter\n\
             lambda_requests_total {}\n\
             # HELP lambda_requests_success Successful requests\n\
             # TYPE lambda_requests_success counter\n\
             lambda_requests_success {}\n\
             # HELP lambda_requests_failed Failed requests\n\
             # TYPE lambda_requests_failed counter\n\
             lambda_requests_failed {}\n\
             # HELP lambda_latency_avg_ms Average inference latency\n\
             # TYPE lambda_latency_avg_ms gauge\n\
             lambda_latency_avg_ms {:.3}\n\
             # HELP lambda_cold_starts Cold start count\n\
             # TYPE lambda_cold_starts counter\n\
             lambda_cold_starts {}\n\
             # HELP lambda_batch_requests Batch requests processed\n\
             # TYPE lambda_batch_requests counter\n\
             lambda_batch_requests {}\n",
            self.requests_total,
            self.requests_success,
            self.requests_failed,
            self.avg_latency_ms(),
            self.cold_starts,
            self.batch_requests
        )
    }
}

/// Cold start timing breakdown
///
/// Per spec §7.1: Instrumented cold start measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartMetrics {
    /// Runtime initialization time (ms)
    pub runtime_init_ms: f64,
    /// Model load time (ms)
    pub model_load_ms: f64,
    /// First inference time (ms)
    pub first_inference_ms: f64,
    /// Total cold start time (ms)
    pub total_ms: f64,
}

/// Lambda handler state
///
/// Per spec §6.1: Uses `OnceLock` for lazy initialization
pub struct LambdaHandler {
    /// Model bytes (embedded via `include_bytes!()`)
    model_bytes: &'static [u8],
    /// Parsed APR model (lazy-initialized on first inference)
    model: OnceLock<AprModel>,
    /// Initialization time tracking
    init_time: OnceLock<Instant>,
    /// Cold start metrics
    cold_start_metrics: OnceLock<ColdStartMetrics>,
}

// Manual Debug impl since AprModel doesn't derive Debug for OnceLock
impl std::fmt::Debug for LambdaHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LambdaHandler")
            .field("model_bytes_len", &self.model_bytes.len())
            .field("model_initialized", &self.model.get().is_some())
            .field("init_time", &self.init_time)
            .field("cold_start_metrics", &self.cold_start_metrics)
            .finish()
    }
}

impl LambdaHandler {
    /// Create handler from embedded model bytes
    ///
    /// Per spec §6.1: `include_bytes!()` pattern
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Embedded .apr model bytes
    ///
    /// # Errors
    ///
    /// Returns `LambdaError::EmptyModel` if model bytes are empty.
    /// Returns `LambdaError::InvalidMagic` if magic bytes don't match .apr format.
    pub fn from_bytes(model_bytes: &'static [u8]) -> Result<Self, LambdaError> {
        if model_bytes.is_empty() {
            return Err(LambdaError::EmptyModel);
        }

        // Validate .apr magic bytes
        if model_bytes.len() >= 4 {
            let magic = model_bytes.get(0..4).expect("len >= 4 checked above");
            if magic != APR_MAGIC {
                return Err(LambdaError::InvalidMagic {
                    expected: "APR".to_string(),
                    found: format!(
                        "{:?}",
                        model_bytes.get(0..4.min(model_bytes.len())).unwrap_or(&[])
                    ),
                });
            }
        }

        Ok(Self {
            model_bytes,
            model: OnceLock::new(),
            init_time: OnceLock::new(),
            cold_start_metrics: OnceLock::new(),
        })
    }

    /// Check if this is a cold start (first invocation)
    #[must_use]
    pub fn is_cold_start(&self) -> bool {
        self.init_time.get().is_none()
    }

    /// Get model bytes size
    #[must_use]
    pub fn model_size_bytes(&self) -> usize {
        contract_pre_model_bytes!();
        self.model_bytes.len()
    }

    /// Get cold start metrics (only available after first invocation)
    #[must_use]
    pub fn cold_start_metrics(&self) -> Option<&ColdStartMetrics> {
        self.cold_start_metrics.get()
    }

    /// Handle Lambda invocation
    ///
    /// Per spec §6.1: Lambda handler pattern
    /// Per spec §8.1: Target <10ms warm latency
    ///
    /// # Errors
    ///
    /// Returns `LambdaError::EmptyFeatures` if request features are empty.
    /// Returns `LambdaError::InferenceFailed` if model inference fails.
    pub fn handle(&self, request: &LambdaRequest) -> Result<LambdaResponse, LambdaError> {
        let start = Instant::now();
        let is_cold = self.is_cold_start();

        // Record initialization time on first invocation
        let _ = self.init_time.get_or_init(|| start);

        // Validate request
        if request.features.is_empty() {
            return Err(LambdaError::EmptyFeatures);
        }

        // Lazy-initialize model from bytes (per spec §6.1: OnceLock pattern)
        let model_load_start = Instant::now();
        let model = self.model.get_or_init(|| {
            AprModel::from_bytes(self.model_bytes.to_vec())
                .expect("Model bytes already validated in from_bytes()")
        });
        let model_load_ms = model_load_start.elapsed().as_secs_f64() * 1000.0;

        // Run real inference using AprModel::predict()
        // Note: predict() is a stub returning sum of features until real inference is implemented
        let inference_start = Instant::now();
        let output = model
            .predict(&request.features)
            .map_err(|e| LambdaError::InferenceFailed(format!("Model inference failed: {e}")))?;
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

        // Extract scalar prediction (first output element or sum for multi-output)
        let prediction = if output.len() == 1 {
            output[0]
        } else {
            output.iter().sum()
        };

        let latency = start.elapsed();

        // Record cold start metrics on first invocation
        if is_cold {
            let _ = self.cold_start_metrics.get_or_init(|| ColdStartMetrics {
                runtime_init_ms: 0.0,
                model_load_ms,
                first_inference_ms: inference_ms,
                total_ms: latency.as_secs_f64() * 1000.0,
            });
        }

        Ok(LambdaResponse {
            prediction,
            probabilities: if output.len() > 1 { Some(output) } else { None },
            latency_ms: latency.as_secs_f64() * 1000.0,
            cold_start: is_cold,
        })
    }

    /// Handle batch Lambda invocation
    ///
    /// Per spec §5.3: Batch inference for throughput optimization
    /// Per spec §11.3: Production hardening
    ///
    /// # Errors
    ///
    /// Individual predictions may fail; errors are captured in response.
    /// Returns `LambdaError::EmptyBatch` if batch has no instances.
    pub fn handle_batch(
        &self,
        request: &BatchLambdaRequest,
    ) -> Result<BatchLambdaResponse, LambdaError> {
        let start = Instant::now();

        if request.instances.is_empty() {
            return Err(LambdaError::EmptyBatch);
        }

        let mut predictions = Vec::with_capacity(request.instances.len());
        let mut success_count = 0;
        let mut error_count = 0;

        // Process each instance sequentially (Lambda single-threaded for cold start optimization)
        for instance in &request.instances {
            if let Ok(response) = self.handle(instance) {
                predictions.push(response);
                success_count += 1;
            } else {
                // Push error placeholder
                predictions.push(LambdaResponse {
                    prediction: f32::NAN,
                    probabilities: None,
                    latency_ms: 0.0,
                    cold_start: false,
                });
                error_count += 1;
            }
        }

        let total_latency = start.elapsed();

        Ok(BatchLambdaResponse {
            predictions,
            total_latency_ms: total_latency.as_secs_f64() * 1000.0,
            success_count,
            error_count,
        })
    }
}

/// Lambda handler errors
#[derive(Debug, Clone, PartialEq)]
pub enum LambdaError {
    /// Model bytes are empty
    EmptyModel,
    /// Invalid .apr magic bytes
    InvalidMagic {
        /// Expected magic bytes
        expected: String,
        /// Found magic bytes
        found: String,
    },
    /// Request features are empty
    EmptyFeatures,
    /// Batch request has no instances
    EmptyBatch,
    /// Model deserialization failed
    ModelDeserialize(String),
    /// Inference failed
    InferenceFailed(String),
}

impl std::fmt::Display for LambdaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LambdaError::EmptyModel => write!(f, "Model bytes are empty"),
            LambdaError::InvalidMagic { expected, found } => {
                write!(f, "Invalid magic bytes: expected {expected}, found {found}")
            },
            LambdaError::EmptyFeatures => write!(f, "Request features are empty"),
            LambdaError::EmptyBatch => write!(f, "Batch request has no instances"),
            LambdaError::ModelDeserialize(msg) => write!(f, "Model deserialization failed: {msg}"),
            LambdaError::InferenceFailed(msg) => write!(f, "Inference failed: {msg}"),
        }
    }
}

impl std::error::Error for LambdaError {}

include!("lambda_arm64_optimizations.rs");
include!("lambda_batch_handler.rs");
