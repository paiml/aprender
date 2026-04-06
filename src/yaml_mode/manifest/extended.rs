//! Extended Configurations
//!
//! Contains extended configuration types for the YAML Mode QA Epic.

use serde::{Deserialize, Serialize};

/// CITL (Compiler-in-the-Loop) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitlConfig {
    /// Mode: suggest, trace, index, tarantula
    pub mode: String,

    /// Error code for suggestions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,

    /// Top K suggestions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,

    /// Workspace mode for cross-crate analysis
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<bool>,

    /// Include dependencies in analysis
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_deps: Option<bool>,
}

/// RAG (Retrieval-Augmented Generation) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Pattern store path
    pub store: String,

    /// Similarity threshold
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity_threshold: Option<f64>,

    /// Max results to return
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
}

/// Graph output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Output file path
    pub output: String,

    /// Output format (dot, json, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Include edges in graph
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_edges: Option<bool>,
}

/// Distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Teacher model configuration
    pub teacher: DistillModelRef,

    /// Student model configuration
    pub student: DistillModelRef,

    /// Distillation temperature
    pub temperature: f64,

    /// Alpha weight for distillation loss vs hard labels
    pub alpha: f64,

    /// Loss function (kl_div, mse, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss: Option<String>,
}

/// Model reference for distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillModelRef {
    /// Model source path
    pub source: String,

    /// Device placement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// Inspection configuration for data analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectConfig {
    /// Inspection mode: outliers, distribution, correlation
    pub mode: String,

    /// Z-score threshold for outlier detection
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_threshold: Option<f64>,

    /// Action on detection: log, drop, flag
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,

    /// Columns to inspect
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub columns: Option<Vec<String>>,
}

/// Privacy configuration for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Enable differential privacy
    pub differential: bool,

    /// Privacy budget epsilon
    pub epsilon: f64,

    /// Privacy budget delta
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<f64>,

    /// Maximum gradient norm for clipping
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_grad_norm: Option<f64>,

    /// Noise multiplier
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_multiplier: Option<f64>,

    /// Privacy accountant type (rdp, gdp, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accountant: Option<String>,
}

/// Audit configuration for bias and fairness testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Audit type: bias, fairness, security
    #[serde(rename = "type")]
    pub audit_type: String,

    /// Protected attribute for bias testing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protected_attr: Option<String>,

    /// Favorable outcome value
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub favorable_outcome: Option<i32>,

    /// Metrics to compute
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,

    /// Threshold for passing audit
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,

    /// Subgroups to analyze
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subgroups: Option<Vec<String>>,
}

/// Session configuration for stateful training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Unique session identifier
    pub id: String,

    /// Auto-save session state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_save: Option<bool>,

    /// Resume on crash
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_on_crash: Option<bool>,

    /// State directory
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_dir: Option<String>,
}

/// Stress testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressConfig {
    /// Number of parallel jobs
    pub parallel_jobs: usize,

    /// Test duration (e.g., "24h")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,

    /// Memory limit as fraction (0.0-1.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<f64>,

    /// Backpressure configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backpressure: Option<BackpressureConfig>,
}

/// Backpressure configuration for stress testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureConfig {
    /// Enable backpressure handling
    pub enabled: bool,

    /// Queue size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_size: Option<usize>,

    /// Drop policy: oldest, newest, random
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drop_policy: Option<String>,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Benchmark mode: inference, training, throughput
    pub mode: String,

    /// Warmup iterations
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup: Option<usize>,

    /// Number of iterations
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iterations: Option<usize>,

    /// Batch sizes to test
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_sizes: Option<Vec<usize>>,

    /// Percentiles to report
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub percentiles: Option<Vec<String>>,
}

/// Debug configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Enable memory profiling
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_profile: Option<bool>,

    /// Log interval in steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_interval: Option<usize>,

    /// GC interval in steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gc_interval: Option<usize>,
}

/// Signing configuration for model artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigningConfig {
    /// Enable signing
    pub enabled: bool,

    /// Signing algorithm (ed25519, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,

    /// Signing key (env var reference)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

/// Verification configuration for production releases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Require all 25 QA checks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all_25_checks: Option<bool>,

    /// QA lead sign-off requirement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qa_lead_sign_off: Option<String>,

    /// Engineering lead sign-off requirement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eng_lead_sign_off: Option<String>,

    /// Safety officer sign-off requirement
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_officer_sign_off: Option<String>,
}
