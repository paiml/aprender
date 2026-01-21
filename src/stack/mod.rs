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
mod tests {
    use super::*;

    #[test]
    fn test_stack_component() {
        assert_eq!(StackComponent::Aprender.name(), "aprender");
        assert_eq!(StackComponent::Aprender.english(), "to learn");
        assert_eq!(StackComponent::Aprender.format(), Some(".apr"));
        assert_eq!(
            StackComponent::Aprender.magic(),
            Some([0x41, 0x50, 0x52, 0x4E])
        );
    }

    #[test]
    fn test_stack_component_all() {
        let all = StackComponent::all();
        assert_eq!(all.len(), 6);
        assert_eq!(all[0], StackComponent::Alimentar);
        assert_eq!(all[5], StackComponent::Batuta);
    }

    #[test]
    fn test_stack_component_display() {
        let comp = StackComponent::Realizar;
        assert_eq!(format!("{comp}"), "realizar (to accomplish)");
    }

    #[test]
    fn test_derivation_type_original() {
        let deriv = DerivationType::Original;
        assert_eq!(deriv.type_name(), "original");
        assert!(!deriv.is_derived());
        assert!(deriv.parent_hashes().is_empty());
    }

    #[test]
    fn test_derivation_type_fine_tune() {
        let parent = [1u8; 32];
        let deriv = DerivationType::FineTune {
            parent_hash: parent,
            epochs: 10,
        };

        assert_eq!(deriv.type_name(), "fine-tune");
        assert!(deriv.is_derived());
        assert_eq!(deriv.parent_hashes(), vec![parent]);
    }

    #[test]
    fn test_derivation_type_merge() {
        let parents = vec![[1u8; 32], [2u8; 32]];
        let deriv = DerivationType::Merge {
            parent_hashes: parents.clone(),
            method: "TIES".into(),
        };

        assert_eq!(deriv.type_name(), "merge");
        assert_eq!(deriv.parent_hashes(), parents);
    }

    #[test]
    fn test_quantization_type() {
        assert_eq!(QuantizationType::Int8.bits(), 8);
        assert_eq!(QuantizationType::Int4.bits(), 4);
        assert_eq!(QuantizationType::Float16.bits(), 16);
        assert_eq!(QuantizationType::Int8.name(), "int8");
    }

    #[test]
    fn test_model_stage_transitions() {
        assert!(ModelStage::Development.can_transition_to(ModelStage::Staging));
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Production));
        assert!(ModelStage::Production.can_transition_to(ModelStage::Archived));
        assert!(!ModelStage::Archived.can_transition_to(ModelStage::Development));
        assert!(ModelStage::Development.can_transition_to(ModelStage::Development));
    }

    #[test]
    fn test_model_version() {
        let version = ModelVersion::new("1.0.0", [0u8; 32])
            .with_stage(ModelStage::Production)
            .with_size(1_000_000)
            .with_quality_score(95.0)
            .with_tag("classification");

        assert_eq!(version.version, "1.0.0");
        assert_eq!(version.stage, ModelStage::Production);
        assert_eq!(version.size_bytes, 1_000_000);
        assert_eq!(version.quality_score, Some(95.0));
        assert!(version.tags.contains(&"classification".to_string()));
        assert!(version.is_production_ready());
    }

    #[test]
    fn test_model_version_not_production_ready() {
        let version = ModelVersion::new("1.0.0", [0u8; 32])
            .with_stage(ModelStage::Development)
            .with_quality_score(95.0);
        assert!(!version.is_production_ready()); // wrong stage

        let version = ModelVersion::new("1.0.0", [0u8; 32])
            .with_stage(ModelStage::Production)
            .with_quality_score(70.0);
        assert!(!version.is_production_ready()); // low quality
    }

    #[test]
    fn test_model_version_hash_hex() {
        let hash = [
            0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let version = ModelVersion::new("1.0.0", hash);
        assert!(version.hash_hex().starts_with("abcdef01234567890000"));
    }

    #[test]
    fn test_inference_config() {
        let config = InferenceConfig::new("model.apr")
            .with_port(9000)
            .with_batch_size(64)
            .with_timeout_ms(200)
            .without_cors();

        assert_eq!(config.port, 9000);
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.timeout_ms, 200);
        assert!(!config.enable_cors);
        assert_eq!(config.predict_url(), "http://localhost:9000/predict");
    }

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.port, 8080);
        assert!(config.enable_cors);
    }

    #[test]
    fn test_stack_health() {
        let mut health = StackHealth::new();
        assert_eq!(health.overall, HealthStatus::Unknown);

        health.set_component(StackComponent::Aprender, ComponentHealth::healthy("0.15.0"));
        health.set_component(StackComponent::Pacha, ComponentHealth::healthy("1.0.0"));

        assert!(health.is_healthy());
        assert_eq!(health.overall, HealthStatus::Healthy);
    }

    #[test]
    fn test_stack_health_degraded() {
        let mut health = StackHealth::new();

        health.set_component(StackComponent::Aprender, ComponentHealth::healthy("0.15.0"));
        health.set_component(
            StackComponent::Pacha,
            ComponentHealth::degraded("1.0.0", "high latency"),
        );

        assert!(!health.is_healthy());
        assert_eq!(health.overall, HealthStatus::Degraded);
    }

    #[test]
    fn test_stack_health_unhealthy() {
        let mut health = StackHealth::new();

        health.set_component(StackComponent::Aprender, ComponentHealth::healthy("0.15.0"));
        health.set_component(
            StackComponent::Pacha,
            ComponentHealth::unhealthy("connection refused"),
        );

        assert!(!health.is_healthy());
        assert_eq!(health.overall, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_component_health() {
        let healthy = ComponentHealth::healthy("1.0.0").with_response_time(50);
        assert_eq!(healthy.status, HealthStatus::Healthy);
        assert_eq!(healthy.response_time_ms, Some(50));

        let unhealthy = ComponentHealth::unhealthy("timeout");
        assert_eq!(unhealthy.status, HealthStatus::Unhealthy);
        assert_eq!(unhealthy.error, Some("timeout".into()));
    }

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_operational());
        assert!(HealthStatus::Degraded.is_operational());
        assert!(!HealthStatus::Unhealthy.is_operational());
        assert!(!HealthStatus::Unknown.is_operational());

        assert_eq!(HealthStatus::Healthy.name(), "healthy");
        assert_eq!(format!("{}", HealthStatus::Degraded), "degraded");
    }

    #[test]
    fn test_format_compatibility() {
        let compat = FormatCompatibility::current();

        assert!(compat.is_apr_compatible(1, 0));
        assert!(!compat.is_apr_compatible(2, 0));

        assert!(compat.is_ald_compatible(1, 0));
        assert!(compat.is_ald_compatible(1, 2));
        assert!(!compat.is_ald_compatible(1, 3));
    }

    #[test]
    fn test_model_stage_display() {
        assert_eq!(format!("{}", ModelStage::Production), "production");
    }

    #[test]
    fn test_derivation_distillation() {
        let teacher = [0xAA; 32];
        let deriv = DerivationType::Distillation {
            teacher_hash: teacher,
            temperature: 2.0,
        };

        assert_eq!(deriv.type_name(), "distillation");
        assert_eq!(deriv.parent_hashes(), vec![teacher]);
    }

    #[test]
    fn test_derivation_quantize() {
        let parent = [0xBB; 32];
        let deriv = DerivationType::Quantize {
            parent_hash: parent,
            quant_type: QuantizationType::Int8,
        };

        assert_eq!(deriv.type_name(), "quantize");
        assert_eq!(deriv.parent_hashes(), vec![parent]);
    }

    #[test]
    fn test_derivation_prune() {
        let parent = [0xCC; 32];
        let deriv = DerivationType::Prune {
            parent_hash: parent,
            sparsity: 0.5,
        };

        assert_eq!(deriv.type_name(), "prune");
        assert_eq!(deriv.parent_hashes(), vec![parent]);
    }

    // =========================================================================
    // Extended coverage tests
    // =========================================================================

    #[test]
    fn test_stack_component_all_names() {
        assert_eq!(StackComponent::Alimentar.name(), "alimentar");
        assert_eq!(StackComponent::Pacha.name(), "pacha");
        assert_eq!(StackComponent::Realizar.name(), "realizar");
        assert_eq!(StackComponent::Presentar.name(), "presentar");
        assert_eq!(StackComponent::Batuta.name(), "batuta");
    }

    #[test]
    fn test_stack_component_all_english() {
        assert_eq!(StackComponent::Alimentar.english(), "to feed");
        assert_eq!(StackComponent::Pacha.english(), "earth/universe");
        assert_eq!(StackComponent::Presentar.english(), "to present");
        assert_eq!(StackComponent::Batuta.english(), "baton");
    }

    #[test]
    fn test_stack_component_descriptions() {
        assert!(StackComponent::Alimentar
            .description()
            .contains("Data loading"));
        assert!(StackComponent::Aprender
            .description()
            .contains("Machine learning"));
        assert!(StackComponent::Pacha.description().contains("registry"));
        assert!(StackComponent::Realizar.description().contains("inference"));
        assert!(StackComponent::Presentar
            .description()
            .contains("visualization"));
        assert!(StackComponent::Batuta
            .description()
            .contains("Orchestration"));
    }

    #[test]
    fn test_stack_component_formats_none() {
        assert_eq!(StackComponent::Pacha.format(), None);
        assert_eq!(StackComponent::Realizar.format(), None);
        assert_eq!(StackComponent::Presentar.format(), None);
    }

    #[test]
    fn test_stack_component_formats_some() {
        assert_eq!(StackComponent::Alimentar.format(), Some(".ald"));
        assert_eq!(StackComponent::Batuta.format(), Some(".bat"));
    }

    #[test]
    fn test_stack_component_magic_none() {
        assert_eq!(StackComponent::Pacha.magic(), None);
        assert_eq!(StackComponent::Realizar.magic(), None);
        assert_eq!(StackComponent::Presentar.magic(), None);
    }

    #[test]
    fn test_stack_component_magic_some() {
        assert_eq!(
            StackComponent::Alimentar.magic(),
            Some([0x41, 0x4C, 0x44, 0x46])
        ); // "ALDF"
        assert_eq!(
            StackComponent::Batuta.magic(),
            Some([0x42, 0x41, 0x54, 0x41])
        ); // "BATA"
    }

    #[test]
    fn test_quantization_type_all_bits() {
        assert_eq!(QuantizationType::BFloat16.bits(), 16);
        assert_eq!(QuantizationType::Dynamic.bits(), 8);
        assert_eq!(QuantizationType::QAT.bits(), 8);
    }

    #[test]
    fn test_quantization_type_all_names() {
        assert_eq!(QuantizationType::Int4.name(), "int4");
        assert_eq!(QuantizationType::Float16.name(), "fp16");
        assert_eq!(QuantizationType::BFloat16.name(), "bf16");
        assert_eq!(QuantizationType::Dynamic.name(), "dynamic");
        assert_eq!(QuantizationType::QAT.name(), "qat");
    }

    #[test]
    fn test_model_stage_default() {
        let stage: ModelStage = Default::default();
        assert_eq!(stage, ModelStage::Development);
    }

    #[test]
    fn test_model_stage_all_transitions() {
        // Development transitions
        assert!(ModelStage::Development.can_transition_to(ModelStage::Archived));

        // Staging transitions
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Staging));
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Development));

        // Production transitions
        assert!(ModelStage::Production.can_transition_to(ModelStage::Production));
        assert!(!ModelStage::Production.can_transition_to(ModelStage::Development));
        assert!(!ModelStage::Production.can_transition_to(ModelStage::Staging));

        // Archived - cannot transition anywhere
        assert!(!ModelStage::Archived.can_transition_to(ModelStage::Archived));
        assert!(!ModelStage::Archived.can_transition_to(ModelStage::Production));
        assert!(!ModelStage::Archived.can_transition_to(ModelStage::Staging));
    }

    #[test]
    fn test_model_stage_all_names() {
        assert_eq!(ModelStage::Development.name(), "development");
        assert_eq!(ModelStage::Staging.name(), "staging");
        assert_eq!(ModelStage::Archived.name(), "archived");
    }

    #[test]
    fn test_model_version_with_derivation() {
        let parent = [0xDD; 32];
        let deriv = DerivationType::FineTune {
            parent_hash: parent,
            epochs: 5,
        };
        let version = ModelVersion::new("2.0.0", [0u8; 32]).with_derivation(deriv);
        assert!(version.derivation.is_derived());
    }

    #[test]
    fn test_model_version_no_quality_score_not_production_ready() {
        let version = ModelVersion::new("1.0.0", [0u8; 32]).with_stage(ModelStage::Production);
        // No quality score set
        assert!(!version.is_production_ready());
    }

    #[test]
    fn test_inference_config_batch_predict_url() {
        let config = InferenceConfig::new("model.apr").with_port(3000);
        assert_eq!(
            config.batch_predict_url(),
            "http://localhost:3000/batch_predict"
        );
    }

    #[test]
    fn test_stack_health_default() {
        let health: StackHealth = Default::default();
        assert_eq!(health.overall, HealthStatus::Unknown);
        assert!(health.components.is_empty());
    }

    #[test]
    fn test_format_compatibility_default() {
        let compat: FormatCompatibility = Default::default();
        assert_eq!(compat.apr_version, (1, 0));
        assert!(compat.compatible);
    }

    #[test]
    fn test_health_status_default() {
        let status: HealthStatus = Default::default();
        assert_eq!(status, HealthStatus::Unknown);
    }

    #[test]
    fn test_health_status_all_names() {
        assert_eq!(HealthStatus::Unhealthy.name(), "unhealthy");
        assert_eq!(HealthStatus::Unknown.name(), "unknown");
    }

    #[test]
    fn test_stack_component_debug() {
        let comp = StackComponent::Aprender;
        let debug_str = format!("{:?}", comp);
        assert!(debug_str.contains("Aprender"));
    }

    #[test]
    fn test_derivation_type_debug() {
        let deriv = DerivationType::Original;
        let debug_str = format!("{:?}", deriv);
        assert!(debug_str.contains("Original"));
    }

    #[test]
    fn test_quantization_type_debug() {
        let qt = QuantizationType::Int8;
        let debug_str = format!("{:?}", qt);
        assert!(debug_str.contains("Int8"));
    }

    #[test]
    fn test_model_stage_debug() {
        let stage = ModelStage::Staging;
        let debug_str = format!("{:?}", stage);
        assert!(debug_str.contains("Staging"));
    }

    #[test]
    fn test_model_version_debug() {
        let version = ModelVersion::new("1.0.0", [0u8; 32]);
        let debug_str = format!("{:?}", version);
        assert!(debug_str.contains("ModelVersion"));
    }

    #[test]
    fn test_inference_config_debug() {
        let config = InferenceConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("InferenceConfig"));
    }

    #[test]
    fn test_stack_health_debug() {
        let health = StackHealth::new();
        let debug_str = format!("{:?}", health);
        assert!(debug_str.contains("StackHealth"));
    }

    #[test]
    fn test_component_health_debug() {
        let health = ComponentHealth::healthy("1.0.0");
        let debug_str = format!("{:?}", health);
        assert!(debug_str.contains("ComponentHealth"));
    }

    #[test]
    fn test_health_status_debug() {
        let status = HealthStatus::Degraded;
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("Degraded"));
    }

    #[test]
    fn test_format_compatibility_debug() {
        let compat = FormatCompatibility::current();
        let debug_str = format!("{:?}", compat);
        assert!(debug_str.contains("FormatCompatibility"));
    }

    #[test]
    fn test_stack_component_clone() {
        let comp = StackComponent::Pacha;
        let cloned = comp;
        assert_eq!(comp, cloned);
    }

    #[test]
    fn test_stack_component_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StackComponent::Aprender);
        set.insert(StackComponent::Pacha);
        assert!(set.contains(&StackComponent::Aprender));
        assert!(!set.contains(&StackComponent::Batuta));
    }

    #[test]
    fn test_derivation_type_clone() {
        let deriv = DerivationType::Original;
        let cloned = deriv.clone();
        assert_eq!(deriv, cloned);
    }

    #[test]
    fn test_quantization_type_clone() {
        let qt = QuantizationType::Float16;
        let cloned = qt;
        assert_eq!(qt, cloned);
    }

    #[test]
    fn test_quantization_type_eq() {
        assert_eq!(QuantizationType::Int8, QuantizationType::Int8);
        assert_ne!(QuantizationType::Int8, QuantizationType::Int4);
    }

    #[test]
    fn test_model_stage_clone() {
        let stage = ModelStage::Production;
        let cloned = stage;
        assert_eq!(stage, cloned);
    }

    #[test]
    fn test_model_version_clone() {
        let version = ModelVersion::new("1.0.0", [0u8; 32]).with_tag("test");
        let cloned = version.clone();
        assert_eq!(cloned.version, "1.0.0");
        assert!(cloned.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_inference_config_clone() {
        let config = InferenceConfig::new("model.apr").with_port(9000);
        let cloned = config.clone();
        assert_eq!(cloned.port, 9000);
    }

    #[test]
    fn test_stack_health_clone() {
        let mut health = StackHealth::new();
        health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0.0"));
        let cloned = health.clone();
        assert!(cloned.components.contains_key(&StackComponent::Aprender));
    }

    #[test]
    fn test_component_health_clone() {
        let health = ComponentHealth::degraded("1.0.0", "slow").with_response_time(100);
        let cloned = health.clone();
        assert_eq!(cloned.status, HealthStatus::Degraded);
        assert_eq!(cloned.response_time_ms, Some(100));
    }

    #[test]
    fn test_health_status_clone() {
        let status = HealthStatus::Healthy;
        let cloned = status;
        assert_eq!(status, cloned);
    }

    #[test]
    fn test_format_compatibility_clone() {
        let compat = FormatCompatibility::current();
        let cloned = compat;
        assert_eq!(compat.apr_version, cloned.apr_version);
    }

    // =========================================================================
    // Additional coverage tests for all branches
    // =========================================================================

    #[test]
    fn test_stack_component_display_all() {
        for comp in StackComponent::all() {
            let display = format!("{}", comp);
            assert!(display.contains(comp.name()));
            assert!(display.contains(comp.english()));
        }
    }

    #[test]
    fn test_model_stage_display_all() {
        let stages = [
            ModelStage::Development,
            ModelStage::Staging,
            ModelStage::Production,
            ModelStage::Archived,
        ];
        for stage in stages {
            let display = format!("{}", stage);
            assert_eq!(display, stage.name());
        }
    }

    #[test]
    fn test_health_status_display_all() {
        let statuses = [
            HealthStatus::Healthy,
            HealthStatus::Degraded,
            HealthStatus::Unhealthy,
            HealthStatus::Unknown,
        ];
        for status in statuses {
            let display = format!("{}", status);
            assert_eq!(display, status.name());
        }
    }

    #[test]
    fn test_stack_health_only_degraded_components() {
        let mut health = StackHealth::new();
        health.set_component(
            StackComponent::Aprender,
            ComponentHealth::degraded("1.0.0", "high latency"),
        );
        health.set_component(
            StackComponent::Pacha,
            ComponentHealth::degraded("1.0.0", "disk full"),
        );
        // All degraded, none unhealthy -> overall should be Degraded
        assert_eq!(health.overall, HealthStatus::Degraded);
        assert!(!health.is_healthy());
    }

    #[test]
    fn test_stack_health_mixed_healthy_degraded() {
        let mut health = StackHealth::new();
        health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0.0"));
        health.set_component(
            StackComponent::Pacha,
            ComponentHealth::degraded("1.0.0", "slow"),
        );
        // Some healthy, some degraded -> overall should be Degraded
        assert_eq!(health.overall, HealthStatus::Degraded);
    }

    #[test]
    fn test_stack_health_clear_and_recheck() {
        let mut health = StackHealth::new();
        health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0.0"));
        assert!(health.is_healthy());

        // Replace with unhealthy
        health.set_component(
            StackComponent::Aprender,
            ComponentHealth::unhealthy("crashed"),
        );
        assert_eq!(health.overall, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_model_version_multiple_tags() {
        let version = ModelVersion::new("1.0.0", [0u8; 32])
            .with_tag("classification")
            .with_tag("nlp")
            .with_tag("transformer");

        assert_eq!(version.tags.len(), 3);
        assert!(version.tags.contains(&"nlp".to_string()));
    }

    #[test]
    fn test_model_version_empty_metadata() {
        let version = ModelVersion::new("1.0.0", [0u8; 32]);
        assert!(version.metadata.is_empty());
    }

    #[test]
    fn test_model_version_quality_boundary() {
        // Exactly at 85.0 boundary
        let version = ModelVersion::new("1.0.0", [0u8; 32])
            .with_stage(ModelStage::Production)
            .with_quality_score(85.0);
        assert!(version.is_production_ready());

        // Just below boundary
        let version = ModelVersion::new("1.0.0", [0u8; 32])
            .with_stage(ModelStage::Production)
            .with_quality_score(84.99);
        assert!(!version.is_production_ready());
    }

    #[test]
    fn test_inference_config_with_no_metrics_or_health() {
        let mut config = InferenceConfig::new("model.apr");
        config.metrics_path = None;
        config.health_path = None;
        assert!(config.metrics_path.is_none());
        assert!(config.health_path.is_none());
    }

    #[test]
    fn test_format_compatibility_apr_major_mismatch() {
        let compat = FormatCompatibility::current();
        assert!(!compat.is_apr_compatible(0, 1)); // Major 0 != 1
        assert!(!compat.is_apr_compatible(2, 0)); // Major 2 != 1
    }

    #[test]
    fn test_format_compatibility_ald_major_mismatch() {
        let compat = FormatCompatibility::current();
        assert!(!compat.is_ald_compatible(0, 1)); // Major 0 != 1
        assert!(!compat.is_ald_compatible(2, 0)); // Major 2 != 1
    }

    #[test]
    fn test_derivation_merge_empty_parents() {
        let deriv = DerivationType::Merge {
            parent_hashes: vec![],
            method: "empty".into(),
        };
        assert!(deriv.parent_hashes().is_empty());
        assert!(deriv.is_derived());
    }

    #[test]
    fn test_derivation_merge_many_parents() {
        let parents = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let deriv = DerivationType::Merge {
            parent_hashes: parents.clone(),
            method: "ensemble".into(),
        };
        assert_eq!(deriv.parent_hashes().len(), 4);
    }

    #[test]
    fn test_component_health_degraded_has_version_and_error() {
        let health = ComponentHealth::degraded("2.0.0", "high memory");
        assert_eq!(health.version, Some("2.0.0".into()));
        assert_eq!(health.error, Some("high memory".into()));
        assert_eq!(health.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_model_version_hash_all_ff() {
        let hash = [0xFF; 32];
        let version = ModelVersion::new("1.0.0", hash);
        let hex = version.hash_hex();
        assert!(hex.chars().all(|c| c == 'f'));
        assert_eq!(hex.len(), 64);
    }

    #[test]
    fn test_model_version_hash_mixed() {
        let mut hash = [0u8; 32];
        hash[0] = 0x12;
        hash[1] = 0x34;
        hash[31] = 0xAB;
        let version = ModelVersion::new("1.0.0", hash);
        let hex = version.hash_hex();
        assert!(hex.starts_with("1234"));
        assert!(hex.ends_with("ab"));
    }

    #[test]
    fn test_stack_component_eq() {
        assert_eq!(StackComponent::Aprender, StackComponent::Aprender);
        assert_ne!(StackComponent::Aprender, StackComponent::Pacha);
    }

    #[test]
    fn test_health_status_eq() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Degraded);
    }

    #[test]
    fn test_model_stage_eq() {
        assert_eq!(ModelStage::Production, ModelStage::Production);
        assert_ne!(ModelStage::Production, ModelStage::Staging);
    }

    #[test]
    fn test_derivation_type_eq() {
        assert_eq!(DerivationType::Original, DerivationType::Original);
        let d1 = DerivationType::FineTune {
            parent_hash: [0; 32],
            epochs: 10,
        };
        let d2 = DerivationType::FineTune {
            parent_hash: [0; 32],
            epochs: 10,
        };
        let d3 = DerivationType::FineTune {
            parent_hash: [1; 32],
            epochs: 10,
        };
        assert_eq!(d1, d2);
        assert_ne!(d1, d3);
    }

    #[test]
    fn test_inference_config_model_path() {
        let config = InferenceConfig::new("/path/to/model.apr");
        assert_eq!(config.model_path, PathBuf::from("/path/to/model.apr"));
    }

    #[test]
    fn test_stack_health_timestamp() {
        let health = StackHealth::new();
        assert!(!health.checked_at.is_empty());
    }

    #[test]
    fn test_model_version_created_at_default() {
        let version = ModelVersion::new("1.0.0", [0u8; 32]);
        assert!(!version.created_at.is_empty());
    }

    #[test]
    fn test_model_version_size_default() {
        let version = ModelVersion::new("1.0.0", [0u8; 32]);
        assert_eq!(version.size_bytes, 0);
    }

    #[test]
    fn test_derivation_finetune_details() {
        let parent = [0xAA; 32];
        let deriv = DerivationType::FineTune {
            parent_hash: parent,
            epochs: 100,
        };
        if let DerivationType::FineTune {
            epochs,
            parent_hash,
        } = deriv
        {
            assert_eq!(epochs, 100);
            assert_eq!(parent_hash, parent);
        } else {
            panic!("Expected FineTune");
        }
    }

    #[test]
    fn test_derivation_distillation_details() {
        let teacher = [0xBB; 32];
        let deriv = DerivationType::Distillation {
            teacher_hash: teacher,
            temperature: 1.5,
        };
        if let DerivationType::Distillation {
            temperature,
            teacher_hash,
        } = deriv
        {
            assert!((temperature - 1.5).abs() < 1e-6);
            assert_eq!(teacher_hash, teacher);
        } else {
            panic!("Expected Distillation");
        }
    }

    #[test]
    fn test_derivation_quantize_details() {
        let parent = [0xCC; 32];
        let deriv = DerivationType::Quantize {
            parent_hash: parent,
            quant_type: QuantizationType::Int4,
        };
        if let DerivationType::Quantize {
            quant_type,
            parent_hash,
        } = deriv
        {
            assert_eq!(quant_type, QuantizationType::Int4);
            assert_eq!(parent_hash, parent);
        } else {
            panic!("Expected Quantize");
        }
    }

    #[test]
    fn test_derivation_prune_details() {
        let parent = [0xDD; 32];
        let deriv = DerivationType::Prune {
            parent_hash: parent,
            sparsity: 0.9,
        };
        if let DerivationType::Prune {
            sparsity,
            parent_hash,
        } = deriv
        {
            assert!((sparsity - 0.9).abs() < 1e-6);
            assert_eq!(parent_hash, parent);
        } else {
            panic!("Expected Prune");
        }
    }

    #[test]
    fn test_stack_component_copy() {
        let comp = StackComponent::Realizar;
        let copy = comp;
        // Both should be usable (Copy trait)
        assert_eq!(comp.name(), "realizar");
        assert_eq!(copy.name(), "realizar");
    }

    #[test]
    fn test_quantization_type_copy() {
        let qt = QuantizationType::BFloat16;
        let copy = qt;
        assert_eq!(qt.bits(), 16);
        assert_eq!(copy.bits(), 16);
    }

    #[test]
    fn test_model_stage_copy() {
        let stage = ModelStage::Archived;
        let copy = stage;
        assert_eq!(stage.name(), "archived");
        assert_eq!(copy.name(), "archived");
    }

    #[test]
    fn test_health_status_copy() {
        let status = HealthStatus::Unknown;
        let copy = status;
        assert!(!status.is_operational());
        assert!(!copy.is_operational());
    }

    #[test]
    fn test_format_compatibility_copy() {
        let compat = FormatCompatibility::current();
        let copy = compat;
        assert_eq!(compat.apr_version, copy.apr_version);
        assert_eq!(compat.ald_version, copy.ald_version);
    }

    #[test]
    fn test_stack_health_multiple_updates() {
        let mut health = StackHealth::new();

        // First: all healthy
        health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0"));
        health.set_component(StackComponent::Pacha, ComponentHealth::healthy("1.0"));
        assert_eq!(health.overall, HealthStatus::Healthy);

        // Second: one becomes unhealthy
        health.set_component(StackComponent::Aprender, ComponentHealth::unhealthy("down"));
        assert_eq!(health.overall, HealthStatus::Unhealthy);

        // Third: fix it, now degraded
        health.set_component(
            StackComponent::Aprender,
            ComponentHealth::degraded("1.0", "slow"),
        );
        assert_eq!(health.overall, HealthStatus::Degraded);

        // Fourth: all healthy again
        health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0"));
        assert_eq!(health.overall, HealthStatus::Healthy);
    }

    #[test]
    fn test_component_health_unhealthy_no_version() {
        let health = ComponentHealth::unhealthy("service unavailable");
        assert!(health.version.is_none());
        assert!(health.response_time_ms.is_none());
    }

    #[test]
    fn test_inference_config_port_range() {
        let config = InferenceConfig::new("m.apr").with_port(0);
        assert_eq!(config.port, 0);

        let config = InferenceConfig::new("m.apr").with_port(65535);
        assert_eq!(config.port, 65535);
    }

    #[test]
    fn test_inference_config_large_batch_size() {
        let config = InferenceConfig::new("m.apr").with_batch_size(1_000_000);
        assert_eq!(config.max_batch_size, 1_000_000);
    }

    #[test]
    fn test_inference_config_zero_timeout() {
        let config = InferenceConfig::new("m.apr").with_timeout_ms(0);
        assert_eq!(config.timeout_ms, 0);
    }

    #[test]
    fn test_model_version_large_size() {
        let version = ModelVersion::new("1.0.0", [0u8; 32]).with_size(100_000_000_000); // 100GB
        assert_eq!(version.size_bytes, 100_000_000_000);
    }

    #[test]
    fn test_model_version_quality_score_max() {
        let version = ModelVersion::new("1.0.0", [0u8; 32])
            .with_stage(ModelStage::Production)
            .with_quality_score(100.0);
        assert!(version.is_production_ready());
    }
}
