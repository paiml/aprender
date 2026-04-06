//! Audit Trail and Provenance Logging
//!
//! Per spec §12: Comprehensive audit record for every inference request.
//! Implements GDPR Article 13/14 and SOC 2 compliance requirements.
//!
//! ## Toyota Way: Jidoka (Built-in Quality)
//!
//! Every inference operation includes automatic quality checks:
//! - CRC32 checksum verification
//! - NaN/Inf detection in outputs
//! - Latency anomaly detection
//! - Confidence score validation
//!
//! ## Key Features
//!
//! - Full provenance tracking (model hash, distillation lineage)
//! - Latency breakdown (preprocessing, inference, postprocessing)
//! - Quality gates (NaN check, confidence check)
//! - Batched async flush for high throughput

// Module-level clippy allows for audit infrastructure
#![allow(clippy::must_use_candidate)] // Builder pattern methods don't need must_use
#![allow(clippy::return_self_not_must_use)] // Builder pattern methods return Self
#![allow(clippy::missing_errors_doc)] // Error conditions are self-explanatory

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use uuid::Uuid;

/// Comprehensive audit record for every inference request
/// Per GDPR Article 13/14 and SOC 2 compliance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    // === Identification ===
    /// Unique request identifier
    pub request_id: String,
    /// Timestamp (ISO 8601 with timezone)
    pub timestamp: DateTime<Utc>,
    /// Client identifier (hashed for privacy)
    pub client_id_hash: Option<String>,

    // === Model Information ===
    /// Model file SHA256 hash (provenance)
    pub model_hash: String,
    /// Model version string
    pub model_version: String,
    /// Model type (LogisticRegression, RandomForest, etc.)
    pub model_type: String,
    /// Distillation lineage (if applicable)
    pub distillation_teacher_hash: Option<String>,

    // === Request Details ===
    /// Input feature dimensions
    pub input_dims: Vec<usize>,
    /// Input feature hash (for reproducibility without storing raw data)
    pub input_hash: String,
    /// Request options (explain, confidence threshold, etc.)
    pub options: AuditOptions,

    // === Response Details ===
    /// Output prediction (class or value)
    pub prediction: serde_json::Value,
    /// Confidence score (if classification)
    pub confidence: Option<f32>,
    /// Explanation summary (if explainability enabled)
    pub explanation_summary: Option<String>,

    // === Performance ===
    /// Total latency in milliseconds
    pub latency_ms: f64,
    /// Breakdown: preprocessing, inference, postprocessing
    pub latency_breakdown: LatencyBreakdown,
    /// Peak memory usage in bytes
    pub memory_peak_bytes: u64,

    // === Quality Checks (Jidoka) ===
    /// Did output pass NaN/Inf check?
    pub quality_nan_check: bool,
    /// Did output pass confidence threshold?
    pub quality_confidence_check: bool,
    /// Any warnings generated?
    pub warnings: Vec<String>,
}

impl AuditRecord {
    /// Create a new audit record with minimal required fields
    pub fn new(request_id: Uuid, model_hash: &str, model_type: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            timestamp: Utc::now(),
            client_id_hash: None,
            model_hash: model_hash.to_string(),
            model_version: String::new(),
            model_type: model_type.to_string(),
            distillation_teacher_hash: None,
            input_dims: Vec::new(),
            input_hash: String::new(),
            options: AuditOptions::default(),
            prediction: serde_json::Value::Null,
            confidence: None,
            explanation_summary: None,
            latency_ms: 0.0,
            latency_breakdown: LatencyBreakdown::default(),
            memory_peak_bytes: 0,
            quality_nan_check: true,
            quality_confidence_check: true,
            warnings: Vec::new(),
        }
    }

    /// Builder pattern: set client ID hash
    pub fn with_client_hash(mut self, hash: impl Into<String>) -> Self {
        self.client_id_hash = Some(hash.into());
        self
    }

    /// Builder pattern: set model version
    pub fn with_model_version(mut self, version: impl Into<String>) -> Self {
        self.model_version = version.into();
        self
    }

    /// Builder pattern: set input dimensions
    pub fn with_input_dims(mut self, dims: Vec<usize>) -> Self {
        self.input_dims = dims;
        self
    }

    /// Builder pattern: set input hash
    pub fn with_input_hash(mut self, hash: impl Into<String>) -> Self {
        self.input_hash = hash.into();
        self
    }

    /// Builder pattern: set prediction result
    pub fn with_prediction(mut self, prediction: serde_json::Value) -> Self {
        self.prediction = prediction;
        self
    }

    /// Builder pattern: set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Builder pattern: set latency
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency_ms = latency.as_secs_f64() * 1000.0;
        self
    }

    /// Builder pattern: set latency breakdown
    pub fn with_latency_breakdown(mut self, breakdown: LatencyBreakdown) -> Self {
        self.latency_breakdown = breakdown;
        self
    }

    /// Builder pattern: add a warning
    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }

    /// Builder pattern: set quality check results
    pub fn with_quality_checks(mut self, nan_check: bool, confidence_check: bool) -> Self {
        self.quality_nan_check = nan_check;
        self.quality_confidence_check = confidence_check;
        self
    }

    /// Builder pattern: set distillation teacher hash
    pub fn with_distillation_teacher(mut self, teacher_hash: impl Into<String>) -> Self {
        self.distillation_teacher_hash = Some(teacher_hash.into());
        self
    }
}

/// Request options included in audit record
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditOptions {
    /// Was explainability requested?
    pub explain: bool,
    /// Confidence threshold (if any)
    pub confidence_threshold: Option<f32>,
    /// Max tokens (for LLM requests)
    pub max_tokens: Option<usize>,
    /// Temperature (for LLM requests)
    pub temperature: Option<f32>,
}

impl AuditOptions {
    /// Create options for APR model request
    pub fn apr(explain: bool, confidence_threshold: Option<f32>) -> Self {
        Self {
            explain,
            confidence_threshold,
            max_tokens: None,
            temperature: None,
        }
    }

    /// Create options for LLM request
    pub fn llm(max_tokens: usize, temperature: f32) -> Self {
        Self {
            explain: false,
            confidence_threshold: None,
            max_tokens: Some(max_tokens),
            temperature: Some(temperature),
        }
    }
}

/// Latency breakdown for detailed performance analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    /// Preprocessing time in milliseconds
    pub preprocessing_ms: f64,
    /// Inference time in milliseconds
    pub inference_ms: f64,
    /// Postprocessing time in milliseconds
    pub postprocessing_ms: f64,
    /// Explanation generation time (if enabled)
    pub explanation_ms: Option<f64>,
}

impl LatencyBreakdown {
    /// Create a new latency breakdown
    pub fn new(preprocessing_ms: f64, inference_ms: f64, postprocessing_ms: f64) -> Self {
        Self {
            preprocessing_ms,
            inference_ms,
            postprocessing_ms,
            explanation_ms: None,
        }
    }

    /// Add explanation time
    pub fn with_explanation(mut self, explanation_ms: f64) -> Self {
        self.explanation_ms = Some(explanation_ms);
        self
    }

    /// Total time in milliseconds
    pub fn total_ms(&self) -> f64 {
        self.preprocessing_ms
            + self.inference_ms
            + self.postprocessing_ms
            + self.explanation_ms.unwrap_or(0.0)
    }
}

/// Model provenance tracking (Jidoka: full traceability)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChain {
    /// Original training data hash
    pub training_data_hash: Option<String>,
    /// Training code commit SHA
    pub training_code_sha: Option<String>,
    /// Training environment specification
    pub training_env: Option<TrainingEnv>,
    /// Distillation chain (for distilled models)
    pub distillation_chain: Vec<DistillationStep>,
    /// Quantization provenance
    pub quantization: Option<QuantizationProvenance>,
    /// Signature chain (all signers)
    pub signatures: Vec<SignatureRecord>,
}

impl Default for ProvenanceChain {
    fn default() -> Self {
        Self::new()
    }
}

impl ProvenanceChain {
    /// Create an empty provenance chain
    pub fn new() -> Self {
        Self {
            training_data_hash: None,
            training_code_sha: None,
            training_env: None,
            distillation_chain: Vec::new(),
            quantization: None,
            signatures: Vec::new(),
        }
    }

    /// Add training data hash
    pub fn with_training_data(mut self, hash: impl Into<String>) -> Self {
        self.training_data_hash = Some(hash.into());
        self
    }

    /// Add training code commit
    pub fn with_code_sha(mut self, sha: impl Into<String>) -> Self {
        self.training_code_sha = Some(sha.into());
        self
    }

    /// Add distillation step
    pub fn add_distillation(&mut self, step: DistillationStep) {
        self.distillation_chain.push(step);
    }

    /// Add signature
    pub fn add_signature(&mut self, signature: SignatureRecord) {
        self.signatures.push(signature);
    }
}

/// Training environment specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEnv {
    /// Framework version (e.g., "aprender 0.1.0")
    pub framework: String,
    /// Hardware description
    pub hardware: String,
    /// Random seed used
    pub seed: Option<u64>,
}

/// Distillation step in provenance chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationStep {
    /// Teacher model hash
    pub teacher_hash: String,
    /// Distillation method (Standard, Progressive, Ensemble)
    pub method: String,
    /// Temperature used
    pub temperature: f32,
    /// Alpha (soft vs hard loss weight)
    pub alpha: f32,
    /// Final distillation loss
    pub final_loss: f32,
    /// Timestamp of distillation
    pub timestamp: DateTime<Utc>,
}

/// Quantization provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationProvenance {
    /// Original model hash (before quantization)
    pub original_hash: String,
    /// Quantization method (Q4_0, Q8_0, AWQ, etc.)
    pub method: String,
    /// Bits per weight
    pub bits: u8,
    /// Calibration data hash (if applicable)
    pub calibration_hash: Option<String>,
}

/// Signature record for signed models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureRecord {
    /// Signer identity (e.g., "paiml-release-key")
    pub signer: String,
    /// Signature algorithm (Ed25519)
    pub algorithm: String,
    /// Signature timestamp
    pub timestamp: DateTime<Utc>,
    /// Signature bytes (base64 encoded)
    pub signature: String,
}

/// Audit sink trait for different output destinations
pub trait AuditSink: Send + Sync {
    /// Write a batch of audit records
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), AuditError>;

    /// Flush any pending writes
    fn flush(&self) -> Result<(), AuditError>;
}

/// In-memory audit sink for testing
#[derive(Debug, Default)]
pub struct InMemoryAuditSink {
    records: Mutex<Vec<AuditRecord>>,
}

impl InMemoryAuditSink {
    /// Create a new in-memory sink
    pub fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
        }
    }

    /// Get all stored records
    pub fn records(&self) -> Vec<AuditRecord> {
        self.records.lock().expect("test").clone()
    }

    /// Get record count
    pub fn count(&self) -> usize {
        self.records.lock().expect("test").len()
    }
}

impl AuditSink for InMemoryAuditSink {
    fn write_batch(&self, records: &[AuditRecord]) -> Result<(), AuditError> {
        let mut storage = self.records.lock().expect("test");
        storage.extend(records.iter().cloned());
        Ok(())
    }

    fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }
}

/// JSON file audit sink
pub struct JsonFileAuditSink {
    path: std::path::PathBuf,
}

impl JsonFileAuditSink {
    /// Create a new JSON file sink
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

include!("audit_write_batch.rs");
include!("audit_record.rs");
