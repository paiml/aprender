//! Sovereign AI Stack Integration (spec §9)
//!
//! Defines integration types for the broader Pragmatic AI Labs Sovereign AI Stack.
//! The `.apr` format is standalone but interoperates with sibling tools:
//!
//! - **alimentar** ("to feed"): Data loading and transformation (`.ald` format)
//! - **pacha**: Model registry with versioning and lineage
//! - **realizar** ("to accomplish"): Pure Rust inference engine
//! - **presentar**: WASM visualization and playgrounds
//! - **batuta**: Orchestration and oracle mode
//!
//! # Architecture
//! ```text
//! alimentar → aprender → pacha → realizar
//!     ↓           ↓          ↓         ↓
//!              presentar (WASM viz)
//!                    ↓
//!              batuta (orchestration)
//! ```
//!
//! # Design Principles
//! - **Pure Rust**: Zero cloud dependencies
//! - **Format Independence**: Each tool has its own binary format
//! - **Toyota Way**: Jidoka, Muda elimination, Kaizen

use std::collections::HashMap;
use std::path::PathBuf;

/// Stack component identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StackComponent {
    /// Data loading and transformation
    Alimentar,
    /// ML algorithms and training
    Aprender,
    /// Model registry and versioning
    Pacha,
    /// Inference engine
    Realizar,
    /// WASM visualization
    Presentar,
    /// Orchestration and oracle
    Batuta,
}

impl StackComponent {
    /// Spanish name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Alimentar => "alimentar",
            Self::Aprender => "aprender",
            Self::Pacha => "pacha",
            Self::Realizar => "realizar",
            Self::Presentar => "presentar",
            Self::Batuta => "batuta",
        }
    }

    /// English translation
    #[must_use]
    pub const fn english(&self) -> &'static str {
        match self {
            Self::Alimentar => "to feed",
            Self::Aprender => "to learn",
            Self::Pacha => "earth/universe",
            Self::Realizar => "to accomplish",
            Self::Presentar => "to present",
            Self::Batuta => "baton",
        }
    }

    /// Component description
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Alimentar => "Data loading and transformation library",
            Self::Aprender => "Machine learning algorithms and training",
            Self::Pacha => "Model registry with versioning and lineage",
            Self::Realizar => "Pure Rust ML inference engine",
            Self::Presentar => "WASM visualization and playgrounds",
            Self::Batuta => "Orchestration and oracle mode",
        }
    }

    /// Primary file format
    #[must_use]
    pub const fn format(&self) -> Option<&'static str> {
        match self {
            Self::Alimentar => Some(".ald"),
            Self::Aprender => Some(".apr"),
            Self::Batuta => Some(".bat"), // orchestration config
            // Registry and consumers don't have their own format
            Self::Pacha | Self::Realizar | Self::Presentar => None,
        }
    }

    /// Format magic bytes (first 4 bytes)
    #[must_use]
    pub const fn magic(&self) -> Option<[u8; 4]> {
        match self {
            Self::Alimentar => Some([0x41, 0x4C, 0x44, 0x46]), // "ALDF"
            Self::Aprender => Some([0x41, 0x50, 0x52, 0x4E]),  // "APRN"
            Self::Batuta => Some([0x42, 0x41, 0x54, 0x41]),    // "BATA"
            _ => None,
        }
    }

    /// All components in stack order
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Alimentar,
            Self::Aprender,
            Self::Pacha,
            Self::Realizar,
            Self::Presentar,
            Self::Batuta,
        ]
    }
}

impl std::fmt::Display for StackComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.name(), self.english())
    }
}

/// Model derivation types tracked by Pacha (spec §9.3.2)
///
/// Tracks model lineage via a DAG (Directed Acyclic Graph):
/// - Original training runs
/// - Fine-tuning relationships
/// - Knowledge distillation
/// - Model merging
/// - Quantization
/// - Pruning
#[derive(Debug, Clone, PartialEq)]
pub enum DerivationType {
    /// Original training run
    Original,

    /// Fine-tuning from parent model
    FineTune {
        /// Hash of parent model
        parent_hash: [u8; 32],
        /// Number of training epochs
        epochs: u32,
    },

    /// Knowledge distillation from teacher
    Distillation {
        /// Hash of teacher model
        teacher_hash: [u8; 32],
        /// Temperature parameter
        temperature: f32,
    },

    /// Model merging (e.g., TIES, DARE)
    Merge {
        /// Hashes of parent models
        parent_hashes: Vec<[u8; 32]>,
        /// Merging method name
        method: String,
    },

    /// Quantization (precision reduction)
    Quantize {
        /// Hash of parent model
        parent_hash: [u8; 32],
        /// Quantization type
        quant_type: QuantizationType,
    },

    /// Pruning (weight removal)
    Prune {
        /// Hash of parent model
        parent_hash: [u8; 32],
        /// Target sparsity (0.0-1.0)
        sparsity: f32,
    },
}

impl DerivationType {
    /// Human-readable derivation type name
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::FineTune { .. } => "fine-tune",
            Self::Distillation { .. } => "distillation",
            Self::Merge { .. } => "merge",
            Self::Quantize { .. } => "quantize",
            Self::Prune { .. } => "prune",
        }
    }

    /// Check if this is a derived model (has parent)
    #[must_use]
    pub const fn is_derived(&self) -> bool {
        !matches!(self, Self::Original)
    }

    /// Get parent hash(es) if derived
    #[must_use]
    pub fn parent_hashes(&self) -> Vec<[u8; 32]> {
        match self {
            Self::Original => vec![],
            Self::FineTune { parent_hash, .. }
            | Self::Distillation {
                teacher_hash: parent_hash,
                ..
            }
            | Self::Quantize { parent_hash, .. }
            | Self::Prune { parent_hash, .. } => vec![*parent_hash],
            Self::Merge { parent_hashes, .. } => parent_hashes.clone(),
        }
    }
}

/// Quantization types for model compression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization
    Int4,
    /// 16-bit floating point (half precision)
    Float16,
    /// Brain floating point (bfloat16)
    BFloat16,
    /// Dynamic quantization (runtime)
    Dynamic,
    /// Quantization-aware training
    QAT,
}

impl QuantizationType {
    /// Bits per weight
    #[must_use]
    pub const fn bits(&self) -> u8 {
        match self {
            Self::Int8 | Self::Dynamic | Self::QAT => 8,
            Self::Int4 => 4,
            Self::Float16 | Self::BFloat16 => 16,
        }
    }

    /// Name string
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Float16 => "fp16",
            Self::BFloat16 => "bf16",
            Self::Dynamic => "dynamic",
            Self::QAT => "qat",
        }
    }
}

/// Pacha registry stage for model lifecycle (spec §9.3.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelStage {
    /// Model under development
    #[default]
    Development,
    /// Ready for testing
    Staging,
    /// Deployed in production
    Production,
    /// No longer in use
    Archived,
}

impl ModelStage {
    /// Stage name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Development => "development",
            Self::Staging => "staging",
            Self::Production => "production",
            Self::Archived => "archived",
        }
    }

    /// Can transition to target stage?
    #[must_use]
    pub const fn can_transition_to(&self, target: Self) -> bool {
        // Valid transitions:
        // - Same stage (identity)
        // - Development -> Staging or Archived
        // - Staging -> Production or Development
        // - Production -> Archived
        // Everything else (including any transition from Archived) is invalid
        matches!(
            (self, target),
            (
                Self::Development,
                Self::Development | Self::Staging | Self::Archived
            ) | (
                Self::Staging,
                Self::Staging | Self::Production | Self::Development
            ) | (Self::Production, Self::Production | Self::Archived)
        )
    }
}

impl std::fmt::Display for ModelStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Pacha model version entry
#[derive(Debug, Clone)]
pub struct ModelVersion {
    /// Version string (semver)
    pub version: String,
    /// Model file hash (SHA-256)
    pub hash: [u8; 32],
    /// Stage in lifecycle
    pub stage: ModelStage,
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Quality score (0-100)
    pub quality_score: Option<f32>,
    /// Derivation information
    pub derivation: DerivationType,
    /// Tags for discovery
    pub tags: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl ModelVersion {
    /// Create a new model version
    #[must_use]
    pub fn new(version: impl Into<String>, hash: [u8; 32]) -> Self {
        Self {
            version: version.into(),
            hash,
            stage: ModelStage::Development,
            created_at: "2025-01-01T00:00:00Z".into(),
            size_bytes: 0,
            quality_score: None,
            derivation: DerivationType::Original,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set stage
    #[must_use]
    pub const fn with_stage(mut self, stage: ModelStage) -> Self {
        self.stage = stage;
        self
    }

    /// Set size
    #[must_use]
    pub const fn with_size(mut self, size_bytes: u64) -> Self {
        self.size_bytes = size_bytes;
        self
    }

    /// Set quality score
    #[must_use]
    pub fn with_quality_score(mut self, score: f32) -> Self {
        self.quality_score = Some(score);
        self
    }

    /// Set derivation
    #[must_use]
    pub fn with_derivation(mut self, derivation: DerivationType) -> Self {
        self.derivation = derivation;
        self
    }

    /// Add tag
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Hash as hex string
    #[must_use]
    pub fn hash_hex(&self) -> String {
        use std::fmt::Write;
        self.hash
            .iter()
            .fold(String::with_capacity(64), |mut acc, b| {
                let _ = write!(acc, "{b:02x}");
                acc
            })
    }

    /// Is production ready?
    #[must_use]
    pub fn is_production_ready(&self) -> bool {
        self.stage == ModelStage::Production && self.quality_score.is_some_and(|s| s >= 85.0)
    }
}

/// Realizar inference endpoint configuration (spec §9.4.1)
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Model file path
    pub model_path: PathBuf,
    /// Server port
    pub port: u16,
    /// Maximum batch size
    pub max_batch_size: u32,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Enable CORS
    pub enable_cors: bool,
    /// Metrics endpoint path
    pub metrics_path: Option<String>,
    /// Health endpoint path
    pub health_path: Option<String>,
}

impl InferenceConfig {
    /// Create default config for a model
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            port: 8080,
            max_batch_size: 32,
            timeout_ms: 100,
            enable_cors: true,
            metrics_path: Some("/metrics".into()),
            health_path: Some("/health".into()),
        }
    }

    /// Set port
    #[must_use]
    pub const fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set max batch size
    #[must_use]
    pub const fn with_batch_size(mut self, size: u32) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set timeout
    #[must_use]
    pub const fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Disable CORS
    #[must_use]
    pub const fn without_cors(mut self) -> Self {
        self.enable_cors = false;
        self
    }

    /// Inference endpoint URL
    #[must_use]
    pub fn predict_url(&self) -> String {
        format!("http://localhost:{}/predict", self.port)
    }

    /// Batch inference endpoint URL
    #[must_use]
    pub fn batch_predict_url(&self) -> String {
        format!("http://localhost:{}/batch_predict", self.port)
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self::new("model.apr")
    }
}

/// Stack health status
#[derive(Debug, Clone)]
pub struct StackHealth {
    /// Component availability
    pub components: HashMap<StackComponent, ComponentHealth>,
    /// Overall status
    pub overall: HealthStatus,
    /// Last check timestamp
    pub checked_at: String,
}

impl StackHealth {
    /// Create new health status
    #[must_use]
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            overall: HealthStatus::Unknown,
            checked_at: "2025-01-01T00:00:00Z".into(),
        }
    }

    /// Set component health
    pub fn set_component(&mut self, component: StackComponent, health: ComponentHealth) {
        self.components.insert(component, health);
        self.update_overall();
    }

    /// Update overall status based on components
    fn update_overall(&mut self) {
        if self.components.is_empty() {
            self.overall = HealthStatus::Unknown;
            return;
        }

        let all_healthy = self
            .components
            .values()
            .all(|h| h.status == HealthStatus::Healthy);
        let any_unhealthy = self
            .components
            .values()
            .any(|h| h.status == HealthStatus::Unhealthy);

        self.overall = if all_healthy {
            HealthStatus::Healthy
        } else if any_unhealthy {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        };
    }

    /// Check if stack is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.overall == HealthStatus::Healthy
    }
}

impl Default for StackHealth {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual component health
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    /// Health status
    pub status: HealthStatus,
    /// Version if available
    pub version: Option<String>,
    /// Response time in ms
    pub response_time_ms: Option<u64>,
    /// Error message if unhealthy
    pub error: Option<String>,
}

impl ComponentHealth {
    /// Create healthy status
    #[must_use]
    pub fn healthy(version: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Healthy,
            version: Some(version.into()),
            response_time_ms: None,
            error: None,
        }
    }

    /// Create unhealthy status
    #[must_use]
    pub fn unhealthy(error: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Unhealthy,
            version: None,
            response_time_ms: None,
            error: Some(error.into()),
        }
    }

    /// Create degraded status
    #[must_use]
    pub fn degraded(version: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Degraded,
            version: Some(version.into()),
            response_time_ms: None,
            error: Some(reason.into()),
        }
    }

    /// Set response time
    #[must_use]
    pub const fn with_response_time(mut self, ms: u64) -> Self {
        self.response_time_ms = Some(ms);
        self
    }
}

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HealthStatus {
    /// Component is functioning normally
    Healthy,
    /// Component has issues but is operational
    Degraded,
    /// Component is not operational
    Unhealthy,
    /// Status unknown
    #[default]
    Unknown,
}

impl HealthStatus {
    /// Status name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
            Self::Unknown => "unknown",
        }
    }

    /// Is operational (healthy or degraded)?
    #[must_use]
    pub const fn is_operational(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Format compatibility information
#[derive(Debug, Clone, Copy)]
pub struct FormatCompatibility {
    /// Aprender format version
    pub apr_version: (u8, u8),
    /// Alimentar format version
    pub ald_version: (u8, u8),
    /// Are formats compatible?
    pub compatible: bool,
}

impl FormatCompatibility {
    /// Current version compatibility
    #[must_use]
    pub const fn current() -> Self {
        Self {
            apr_version: (1, 0),
            ald_version: (1, 2),
            compatible: true,
        }
    }

    /// Check if a specific APR version is compatible
    #[must_use]
    pub const fn is_apr_compatible(&self, major: u8, minor: u8) -> bool {
        major == self.apr_version.0 && minor <= self.apr_version.1
    }

    /// Check if a specific ALD version is compatible
    #[must_use]
    pub const fn is_ald_compatible(&self, major: u8, minor: u8) -> bool {
        major == self.ald_version.0 && minor <= self.ald_version.1
    }
}

impl Default for FormatCompatibility {
    fn default() -> Self {
        Self::current()
    }
}

#[cfg(test)]
mod tests;
