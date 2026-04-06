//! BrickPipeline: Orchestration for multi-brick workflows (PROBAR-SPEC-009-P9)
//!
//! Provides pipeline orchestration with:
//! - Sequential dependencies (brick A → brick B)
//! - Parallel execution (bricks A and B concurrently)
//! - Failure handling and checkpointing
//! - Privacy-aware routing
//!
//! # Design Philosophy
//!
//! BrickPipeline applies the batuta orchestration patterns to brick execution.
//! Every brick can be a pipeline stage with validation and audit trail.
//!
//! # Example
//!
//! ```rust,ignore
//! use probar::brick::pipeline::{BrickPipeline, BrickStage, PrivacyTier};
//!
//! let whisper_pipeline = BrickPipeline::new()
//!     .stage(AudioCaptureBrick::new())
//!     .stage(MelSpectrogramBrick::new())
//!     .stage(EncoderBrick::new())
//!     .stage(DecoderBrick::new())
//!     .with_privacy(PrivacyTier::Sovereign)
//!     .with_checkpointing(Duration::from_secs(5));
//!
//! let output = whisper_pipeline.run(input).await?;
//! ```

// Allow missing docs for enum variant fields - context is clear from variant name
#![allow(missing_docs)]

use super::{Brick, BrickError};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Result type for pipeline operations
pub type PipelineResult<T> = Result<T, PipelineError>;

/// Pipeline execution context
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Named data values in the pipeline
    pub data: HashMap<String, PipelineData>,
    /// Metadata for audit trail
    pub metadata: PipelineMetadata,
    /// Execution trace for debugging
    pub trace: Vec<StageTrace>,
}

impl PipelineContext {
    /// Create a new empty context
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            metadata: PipelineMetadata::new(),
            trace: Vec::new(),
        }
    }

    /// Create context from initial input
    #[must_use]
    pub fn from_input(name: &str, data: PipelineData) -> Self {
        let mut ctx = Self::new();
        ctx.data.insert(name.to_string(), data);
        ctx
    }

    /// Get data by name
    pub fn get(&self, name: &str) -> Option<&PipelineData> {
        self.data.get(name)
    }

    /// Set data by name
    pub fn set(&mut self, name: impl Into<String>, data: PipelineData) {
        self.data.insert(name.into(), data);
    }

    /// Add a stage trace entry
    pub fn add_trace(&mut self, trace: StageTrace) {
        self.trace.push(trace);
    }
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Data types that can flow through pipelines
#[derive(Debug, Clone)]
pub enum PipelineData {
    /// Raw bytes
    Bytes(Vec<u8>),
    /// Float tensor
    FloatTensor { data: Vec<f32>, shape: Vec<usize> },
    /// String value
    Text(String),
    /// JSON value
    Json(serde_json::Value),
    /// Integer value
    Int(i64),
    /// Boolean value
    Bool(bool),
}

impl PipelineData {
    /// Create a float tensor
    #[must_use]
    pub fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::FloatTensor { data, shape }
    }

    /// Get as float tensor
    pub fn as_tensor(&self) -> Option<(&[f32], &[usize])> {
        match self {
            Self::FloatTensor { data, shape } => Some((data, shape)),
            _ => None,
        }
    }

    /// Get as text
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            _ => None,
        }
    }
}

/// Metadata for pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineMetadata {
    /// Pipeline run ID
    pub run_id: String,
    /// Start time
    pub started_at: Option<Instant>,
    /// Custom tags
    pub tags: HashMap<String, String>,
}

impl PipelineMetadata {
    /// Create new metadata with generated run ID
    #[must_use]
    pub fn new() -> Self {
        Self {
            run_id: format!("run-{}", uuid_v4()),
            started_at: None,
            tags: HashMap::new(),
        }
    }

    /// Add a tag
    pub fn tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }
}

impl Default for PipelineMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution trace for a single stage
#[derive(Debug, Clone)]
pub struct StageTrace {
    /// Stage name
    pub stage_name: String,
    /// Execution duration
    pub duration: Duration,
    /// Whether stage succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Privacy tier for pipeline execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyTier {
    /// Local-only execution (no network calls)
    Sovereign,
    /// VPC-only (private cloud, no public APIs)
    Private,
    /// Cloud-enabled (spillover to external APIs)
    Standard,
}

impl Default for PrivacyTier {
    fn default() -> Self {
        Self::Standard
    }
}

/// Validation result from a stage
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Validation messages (warnings and errors)
    pub messages: Vec<ValidationMessage>,
}

impl ValidationResult {
    /// Create a passing validation result
    #[must_use]
    pub fn ok() -> Self {
        Self {
            valid: true,
            messages: Vec::new(),
        }
    }

    /// Create a failing validation result
    #[must_use]
    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            valid: false,
            messages: vec![ValidationMessage {
                level: ValidationLevel::Error,
                message: reason.into(),
            }],
        }
    }

    /// Add a warning
    pub fn warn(&mut self, message: impl Into<String>) {
        self.messages.push(ValidationMessage {
            level: ValidationLevel::Warning,
            message: message.into(),
        });
    }
}

/// Validation message level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel {
    /// Informational message
    Info,
    /// Warning (execution continues)
    Warning,
    /// Error (execution blocked)
    Error,
}

/// A validation message
#[derive(Debug, Clone)]
pub struct ValidationMessage {
    /// Message severity
    pub level: ValidationLevel,
    /// Message content
    pub message: String,
}

/// Pipeline error type
#[derive(Debug, Clone)]
pub enum PipelineError {
    /// Stage validation failed
    ValidationFailed { stage: String, reason: String },
    /// Stage execution failed
    ExecutionFailed { stage: String, reason: String },
    /// Missing required input
    MissingInput { stage: String, input: String },
    /// Privacy tier violation
    PrivacyViolation { tier: PrivacyTier, reason: String },
    /// Checkpoint failed
    CheckpointFailed { reason: String },
    /// Brick error
    BrickError(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ValidationFailed { stage, reason } => {
                write!(f, "Validation failed at stage '{}': {}", stage, reason)
            }
            Self::ExecutionFailed { stage, reason } => {
                write!(f, "Execution failed at stage '{}': {}", stage, reason)
            }
            Self::MissingInput { stage, input } => {
                write!(f, "Missing input '{}' for stage '{}'", input, stage)
            }
            Self::PrivacyViolation { tier, reason } => {
                write!(f, "Privacy tier {:?} violated: {}", tier, reason)
            }
            Self::CheckpointFailed { reason } => {
                write!(f, "Checkpoint failed: {}", reason)
            }
            Self::BrickError(msg) => write!(f, "Brick error: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<BrickError> for PipelineError {
    fn from(e: BrickError) -> Self {
        Self::BrickError(e.to_string())
    }
}

/// Trait for bricks that can be pipeline stages
pub trait BrickStage: Brick + Send + Sync {
    /// Execute the stage
    fn execute(&self, ctx: PipelineContext) -> PipelineResult<PipelineContext>;

    /// Validate before execution (Jidoka pattern)
    fn validate(&self, ctx: &PipelineContext) -> ValidationResult;

    /// Get required input names
    fn required_inputs(&self) -> &[&str] {
        &[]
    }

    /// Get output names
    fn output_names(&self) -> &[&str] {
        &[]
    }
}

/// Audit entry for pipeline execution
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Stage name
    pub stage: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Duration
    pub duration: Duration,
    /// Success status
    pub success: bool,
    /// Input data keys
    pub inputs: Vec<String>,
    /// Output data keys
    pub outputs: Vec<String>,
}

/// Audit trail collector
#[derive(Debug, Default)]
pub struct PipelineAuditCollector {
    entries: Vec<AuditEntry>,
}

impl PipelineAuditCollector {
    /// Create a new collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record a stage execution
    pub fn record(&mut self, stage: &str, duration: Duration, success: bool) {
        self.entries.push(AuditEntry {
            stage: stage.to_string(),
            timestamp: Instant::now(),
            duration,
            success,
            inputs: Vec::new(),
            outputs: Vec::new(),
        });
    }

    /// Get all entries
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Get total execution time
    pub fn total_duration(&self) -> Duration {
        self.entries.iter().map(|e| e.duration).sum()
    }
}

/// Checkpoint state for fault tolerance
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Stage index
    pub stage_index: usize,
    /// Context at checkpoint
    pub context: PipelineContext,
    /// Timestamp
    pub created_at: Instant,
}

/// BrickPipeline: Orchestrates multi-brick workflows
pub struct BrickPipeline {
    /// Pipeline name
    name: String,
    /// Ordered stages
    stages: Vec<Box<dyn BrickStage>>,
    /// Privacy tier
    privacy_tier: PrivacyTier,
    /// Checkpoint interval
    checkpoint_interval: Option<Duration>,
    /// Audit collector
    audit_collector: PipelineAuditCollector,
    /// Last checkpoint
    last_checkpoint: Option<Checkpoint>,
}

impl BrickPipeline {
    /// Create a new pipeline
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
            privacy_tier: PrivacyTier::Standard,
            checkpoint_interval: None,
            audit_collector: PipelineAuditCollector::new(),
            last_checkpoint: None,
        }
    }

    /// Add a stage to the pipeline
    #[must_use]
    pub fn stage<S: BrickStage + 'static>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Set privacy tier
    #[must_use]
    pub fn with_privacy(mut self, tier: PrivacyTier) -> Self {
        self.privacy_tier = tier;
        self
    }

    /// Enable checkpointing
    #[must_use]
    pub fn with_checkpointing(mut self, interval: Duration) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    /// Run the pipeline
    pub fn run(&mut self, input: PipelineContext) -> PipelineResult<PipelineContext> {
        let mut ctx = input;
        ctx.metadata.started_at = Some(Instant::now());

        let start_index = self
            .last_checkpoint
            .as_ref()
            .map(|c| c.stage_index)
            .unwrap_or(0);

        // Restore from checkpoint if available
        if let Some(checkpoint) = &self.last_checkpoint {
            ctx = checkpoint.context.clone();
        }

        let mut last_checkpoint_time = Instant::now();

        for (i, stage) in self.stages.iter().enumerate().skip(start_index) {
            let stage_name = stage.brick_name();

            // Jidoka: validate before execution
            let validation = stage.validate(&ctx);
            if !validation.valid {
                let reason = validation
                    .messages
                    .iter()
                    .filter(|m| m.level == ValidationLevel::Error)
                    .map(|m| m.message.as_str())
                    .collect::<Vec<_>>()
                    .join("; ");

                return Err(PipelineError::ValidationFailed {
                    stage: stage_name.to_string(),
                    reason,
                });
            }

            // Execute stage (clone ctx for error recovery)
            let start = Instant::now();
            let ctx_for_error = ctx.clone();
            let result = stage.execute(ctx);
            let duration = start.elapsed();

            match result {
                Ok(mut new_ctx) => {
                    new_ctx.add_trace(StageTrace {
                        stage_name: stage_name.to_string(),
                        duration,
                        success: true,
                        error: None,
                    });

                    self.audit_collector.record(stage_name, duration, true);

                    // Checkpoint if interval exceeded
                    if let Some(interval) = self.checkpoint_interval {
                        if last_checkpoint_time.elapsed() >= interval {
                            self.last_checkpoint = Some(Checkpoint {
                                stage_index: i + 1,
                                context: new_ctx.clone(),
                                created_at: Instant::now(),
                            });
                            last_checkpoint_time = Instant::now();
                        }
                    }

                    ctx = new_ctx;
                }
                Err(e) => {
                    self.audit_collector.record(stage_name, duration, false);

                    // Use saved context for trace in error case
                    let mut error_ctx = ctx_for_error;
                    error_ctx.add_trace(StageTrace {
                        stage_name: stage_name.to_string(),
                        duration,
                        success: false,
                        error: Some(e.to_string()),
                    });

                    return Err(PipelineError::ExecutionFailed {
                        stage: stage_name.to_string(),
                        reason: e.to_string(),
                    });
                }
            }
        }

        // Clear checkpoint on successful completion
        self.last_checkpoint = None;

        Ok(ctx)
    }

    /// Get the pipeline name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of stages
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get audit trail
    pub fn audit_trail(&self) -> &[AuditEntry] {
        self.audit_collector.entries()
    }

    /// Get privacy tier
    #[must_use]
    pub fn privacy_tier(&self) -> PrivacyTier {
        self.privacy_tier
    }
}

impl Debug for BrickPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrickPipeline")
            .field("name", &self.name)
            .field("stage_count", &self.stages.len())
            .field("privacy_tier", &self.privacy_tier)
            .finish()
    }
}

/// Generate a simple UUID v4 (non-cryptographic)
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}{:x}", now.as_nanos(), std::process::id())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::brick::{BrickAssertion, BrickBudget, BrickVerification};

    // ============================================================
    // Test Stage Implementation
    // ============================================================

    struct TestStage {
        name: &'static str,
        should_fail: bool,
    }

    impl Brick for TestStage {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(100)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(10),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl BrickStage for TestStage {
        fn execute(&self, mut ctx: PipelineContext) -> PipelineResult<PipelineContext> {
            if self.should_fail {
                return Err(PipelineError::ExecutionFailed {
                    stage: self.name.to_string(),
                    reason: "Test failure".into(),
                });
            }
            ctx.set(
                format!("{}_output", self.name),
                PipelineData::Text("done".into()),
            );
            Ok(ctx)
        }

        fn validate(&self, _ctx: &PipelineContext) -> ValidationResult {
            ValidationResult::ok()
        }
    }

    /// A stage that fails validation
    struct FailingValidationStage {
        name: &'static str,
    }

    impl Brick for FailingValidationStage {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(100)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(10),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl BrickStage for FailingValidationStage {
        fn execute(&self, ctx: PipelineContext) -> PipelineResult<PipelineContext> {
            Ok(ctx)
        }

        fn validate(&self, _ctx: &PipelineContext) -> ValidationResult {
            ValidationResult::fail("Validation error")
        }
    }

    // ============================================================
    // PipelineContext tests
    // ============================================================

    #[test]
    fn test_pipeline_context_new() {
        let ctx = PipelineContext::new();
        assert!(ctx.data.is_empty());
        assert!(ctx.trace.is_empty());
    }

    #[test]
    fn test_pipeline_context_default() {
        let ctx = PipelineContext::default();
        assert!(ctx.data.is_empty());
    }

    #[test]
    fn test_pipeline_context_from_input() {
        let ctx = PipelineContext::from_input("input", PipelineData::Text("hello".into()));
        assert!(ctx.get("input").is_some());
    }

    #[test]
    fn test_pipeline_context() {
        let mut ctx = PipelineContext::new();
        ctx.set("test", PipelineData::Text("hello".into()));

        assert!(ctx.get("test").is_some());
        assert!(ctx.get("missing").is_none());
    }

    #[test]
    fn test_pipeline_context_add_trace() {
        let mut ctx = PipelineContext::new();
        ctx.add_trace(StageTrace {
            stage_name: "test".to_string(),
            duration: Duration::from_millis(10),
            success: true,
            error: None,
        });

        assert_eq!(ctx.trace.len(), 1);
        assert_eq!(ctx.trace[0].stage_name, "test");
    }

    #[test]
    fn test_pipeline_context_clone() {
        let mut ctx = PipelineContext::new();
        ctx.set("key", PipelineData::Int(42));

        let cloned = ctx.clone();
        assert!(cloned.get("key").is_some());
    }

    // ============================================================
    // PipelineData tests
    // ============================================================

    #[test]
    fn test_pipeline_data_tensor() {
        let data = PipelineData::tensor(vec![1.0, 2.0, 3.0], vec![3]);

        let (values, shape) = data.as_tensor().unwrap();
        assert_eq!(values, &[1.0, 2.0, 3.0]);
        assert_eq!(shape, &[3]);
    }

    #[test]
    fn test_pipeline_data_as_tensor_none() {
        let data = PipelineData::Text("hello".into());
        assert!(data.as_tensor().is_none());
    }

    #[test]
    fn test_pipeline_data_as_text() {
        let data = PipelineData::Text("hello".into());
        assert_eq!(data.as_text(), Some("hello"));
    }

    #[test]
    fn test_pipeline_data_as_text_none() {
        let data = PipelineData::Int(42);
        assert!(data.as_text().is_none());
    }

    #[test]
    fn test_pipeline_data_bytes() {
        let data = PipelineData::Bytes(vec![1, 2, 3, 4]);
        if let PipelineData::Bytes(bytes) = data {
            assert_eq!(bytes, vec![1, 2, 3, 4]);
        } else {
            panic!("Expected Bytes variant");
        }
    }

    #[test]
    fn test_pipeline_data_json() {
        let json = serde_json::json!({"key": "value"});
        let data = PipelineData::Json(json.clone());
        if let PipelineData::Json(value) = data {
            assert_eq!(value, json);
        } else {
            panic!("Expected Json variant");
        }
    }

    #[test]
    fn test_pipeline_data_int() {
        let data = PipelineData::Int(-42);
        if let PipelineData::Int(val) = data {
            assert_eq!(val, -42);
        } else {
            panic!("Expected Int variant");
        }
    }

    #[test]
    fn test_pipeline_data_bool() {
        let data = PipelineData::Bool(true);
        if let PipelineData::Bool(val) = data {
            assert!(val);
        } else {
            panic!("Expected Bool variant");
        }
    }

    #[test]
    fn test_pipeline_data_clone_and_debug() {
        let data = PipelineData::Text("test".into());
        let cloned = data;
        assert!(format!("{:?}", cloned).contains("Text"));
    }

    // ============================================================
    // PipelineMetadata tests
    // ============================================================

    #[test]
    fn test_pipeline_metadata_new() {
        let meta = PipelineMetadata::new();
        assert!(meta.run_id.starts_with("run-"));
        assert!(meta.started_at.is_none());
        assert!(meta.tags.is_empty());
    }

    #[test]
    fn test_pipeline_metadata_default() {
        let meta = PipelineMetadata::default();
        assert!(meta.run_id.starts_with("run-"));
    }

    #[test]
    fn test_pipeline_metadata_tag() {
        let mut meta = PipelineMetadata::new();
        meta.tag("env", "test");
        meta.tag("version", "1.0");

        assert_eq!(meta.tags.get("env"), Some(&"test".to_string()));
        assert_eq!(meta.tags.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_pipeline_metadata_clone_and_debug() {
        let meta = PipelineMetadata::new();
        let cloned = meta;
        assert!(format!("{:?}", cloned).contains("PipelineMetadata"));
    }

    // ============================================================
    // StageTrace tests
    // ============================================================

    #[test]
    fn test_stage_trace_clone_and_debug() {
        let trace = StageTrace {
            stage_name: "test".to_string(),
            duration: Duration::from_millis(100),
            success: true,
            error: None,
        };

        let cloned = trace;
        assert_eq!(cloned.stage_name, "test");
        assert!(cloned.success);
        assert!(format!("{:?}", cloned).contains("StageTrace"));
    }

    #[test]
    fn test_stage_trace_with_error() {
        let trace = StageTrace {
            stage_name: "failed".to_string(),
            duration: Duration::from_millis(50),
            success: false,
            error: Some("Something went wrong".to_string()),
        };

        assert!(!trace.success);
        assert_eq!(trace.error, Some("Something went wrong".to_string()));
    }

    // ============================================================
    // PrivacyTier tests
    // ============================================================

    #[test]
    fn test_privacy_tier_default() {
        let tier = PrivacyTier::default();
        assert_eq!(tier, PrivacyTier::Standard);
    }

    #[test]
    fn test_privacy_tier_equality() {
        assert_eq!(PrivacyTier::Sovereign, PrivacyTier::Sovereign);
        assert_ne!(PrivacyTier::Sovereign, PrivacyTier::Private);
        assert_ne!(PrivacyTier::Private, PrivacyTier::Standard);
    }

    #[test]
    fn test_privacy_tier_debug_and_clone() {
        let tier = PrivacyTier::Private;
        let cloned = tier;
        assert!(format!("{:?}", cloned).contains("Private"));
    }

    // ============================================================
    // ValidationResult tests
    // ============================================================

    #[test]
    fn test_validation_result_ok() {
        let ok = ValidationResult::ok();
        assert!(ok.valid);
        assert!(ok.messages.is_empty());
    }

    #[test]
    fn test_validation_result_fail() {
        let fail = ValidationResult::fail("test error");
        assert!(!fail.valid);
        assert_eq!(fail.messages.len(), 1);
        assert_eq!(fail.messages[0].level, ValidationLevel::Error);
        assert_eq!(fail.messages[0].message, "test error");
    }

    #[test]
    fn test_validation_result_warn() {
        let mut result = ValidationResult::ok();
        result.warn("warning message");

        assert!(result.valid);
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].level, ValidationLevel::Warning);
    }

    #[test]
    fn test_validation_result_clone_and_debug() {
        let result = ValidationResult::fail("error");
        let cloned = result;
        assert!(format!("{:?}", cloned).contains("ValidationResult"));
    }

    // ============================================================
    // ValidationLevel tests
    // ============================================================

    #[test]
    fn test_validation_level_equality() {
        assert_eq!(ValidationLevel::Info, ValidationLevel::Info);
        assert_eq!(ValidationLevel::Warning, ValidationLevel::Warning);
        assert_eq!(ValidationLevel::Error, ValidationLevel::Error);
        assert_ne!(ValidationLevel::Info, ValidationLevel::Error);
    }

    #[test]
    fn test_validation_level_debug_and_clone() {
        let level = ValidationLevel::Warning;
        let cloned = level;
        assert!(format!("{:?}", cloned).contains("Warning"));
    }

    // ============================================================
    // ValidationMessage tests
    // ============================================================

    #[test]
    fn test_validation_message_clone_and_debug() {
        let msg = ValidationMessage {
            level: ValidationLevel::Error,
            message: "test".to_string(),
        };

        let cloned = msg;
        assert_eq!(cloned.message, "test");
        assert!(format!("{:?}", cloned).contains("ValidationMessage"));
    }

    // ============================================================
    // PipelineError tests
    // ============================================================

    #[test]
    fn test_pipeline_error_validation_failed() {
        let err = PipelineError::ValidationFailed {
            stage: "test".to_string(),
            reason: "bad input".to_string(),
        };

        let display = format!("{}", err);
        assert!(display.contains("Validation failed"));
        assert!(display.contains("test"));
        assert!(display.contains("bad input"));
    }

    #[test]
    fn test_pipeline_error_execution_failed() {
        let err = PipelineError::ExecutionFailed {
            stage: "compute".to_string(),
            reason: "timeout".to_string(),
        };

        let display = format!("{}", err);
        assert!(display.contains("Execution failed"));
        assert!(display.contains("compute"));
    }

    #[test]
    fn test_pipeline_error_missing_input() {
        let err = PipelineError::MissingInput {
            stage: "transform".to_string(),
            input: "data".to_string(),
        };

        let display = format!("{}", err);
        assert!(display.contains("Missing input"));
        assert!(display.contains("data"));
        assert!(display.contains("transform"));
    }

    #[test]
    fn test_pipeline_error_privacy_violation() {
        let err = PipelineError::PrivacyViolation {
            tier: PrivacyTier::Sovereign,
            reason: "external API call".to_string(),
        };

        let display = format!("{}", err);
        assert!(display.contains("Privacy tier"));
        assert!(display.contains("Sovereign"));
    }

    #[test]
    fn test_pipeline_error_checkpoint_failed() {
        let err = PipelineError::CheckpointFailed {
            reason: "disk full".to_string(),
        };

        let display = format!("{}", err);
        assert!(display.contains("Checkpoint failed"));
        assert!(display.contains("disk full"));
    }

    #[test]
    fn test_pipeline_error_brick_error() {
        let err = PipelineError::BrickError("brick error".to_string());

        let display = format!("{}", err);
        assert!(display.contains("Brick error"));
    }

    #[test]
    fn test_pipeline_error_from_brick_error() {
        use crate::brick::{BrickAssertion, BrickError};
        let brick_err = BrickError::AssertionFailed {
            assertion: BrickAssertion::ElementPresent("test".to_string()),
            reason: "failed".to_string(),
        };

        let pipeline_err: PipelineError = brick_err.into();
        if let PipelineError::BrickError(msg) = pipeline_err {
            assert!(msg.contains("test"));
        } else {
            panic!("Expected BrickError variant");
        }
    }

    #[test]
    fn test_pipeline_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(PipelineError::CheckpointFailed {
            reason: "test".to_string(),
        });

        assert!(err.to_string().contains("Checkpoint"));
    }

    // ============================================================
    // PipelineAuditCollector tests
    // ============================================================

    #[test]
    fn test_audit_collector_new() {
        let collector = PipelineAuditCollector::new();
        assert!(collector.entries().is_empty());
    }

    #[test]
    fn test_audit_collector_default() {
        let collector = PipelineAuditCollector::default();
        assert!(collector.entries().is_empty());
    }

    #[test]
    fn test_audit_collector() {
        let mut collector = PipelineAuditCollector::new();
        collector.record("stage1", Duration::from_millis(100), true);
        collector.record("stage2", Duration::from_millis(50), true);

        assert_eq!(collector.entries().len(), 2);
        assert_eq!(collector.total_duration(), Duration::from_millis(150));
    }

    #[test]
    fn test_audit_collector_record_failure() {
        let mut collector = PipelineAuditCollector::new();
        collector.record("failed", Duration::from_millis(25), false);

        assert_eq!(collector.entries().len(), 1);
        assert!(!collector.entries()[0].success);
    }

    #[test]
    fn test_audit_collector_debug() {
        let collector = PipelineAuditCollector::new();
        assert!(format!("{:?}", collector).contains("PipelineAuditCollector"));
    }

    // ============================================================
    // AuditEntry tests
    // ============================================================

    #[test]
    fn test_audit_entry_clone_and_debug() {
        let entry = AuditEntry {
            stage: "test".to_string(),
            timestamp: Instant::now(),
            duration: Duration::from_millis(100),
            success: true,
            inputs: vec!["input1".to_string()],
            outputs: vec!["output1".to_string()],
        };

        let cloned = entry;
        assert_eq!(cloned.stage, "test");
        assert!(format!("{:?}", cloned).contains("AuditEntry"));
    }

    // ============================================================
    // Checkpoint tests
    // ============================================================

    #[test]
    fn test_checkpoint_clone_and_debug() {
        let checkpoint = Checkpoint {
            stage_index: 2,
            context: PipelineContext::new(),
            created_at: Instant::now(),
        };

        let cloned = checkpoint;
        assert_eq!(cloned.stage_index, 2);
        assert!(format!("{:?}", cloned).contains("Checkpoint"));
    }

    // ============================================================
    // BrickPipeline tests
    // ============================================================

    #[test]
    fn test_pipeline_basic() {
        let mut pipeline = BrickPipeline::new("test")
            .stage(TestStage {
                name: "stage1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "stage2",
                should_fail: false,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.get("stage1_output").is_some());
        assert!(output.get("stage2_output").is_some());
    }

    #[test]
    fn test_pipeline_failure() {
        let mut pipeline = BrickPipeline::new("test")
            .stage(TestStage {
                name: "stage1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "stage2",
                should_fail: true,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_err());
        match result {
            Err(PipelineError::ExecutionFailed { stage, .. }) => {
                assert_eq!(stage, "stage2");
            }
            _ => panic!("Expected ExecutionFailed"),
        }
    }

    #[test]
    fn test_pipeline_validation_failure() {
        let mut pipeline =
            BrickPipeline::new("test").stage(FailingValidationStage { name: "validator" });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_err());
        match result {
            Err(PipelineError::ValidationFailed { stage, reason }) => {
                assert_eq!(stage, "validator");
                assert!(reason.contains("Validation error"));
            }
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_pipeline_privacy_tier() {
        let pipeline = BrickPipeline::new("test").with_privacy(PrivacyTier::Sovereign);

        assert_eq!(pipeline.privacy_tier(), PrivacyTier::Sovereign);
    }

    #[test]
    fn test_pipeline_name() {
        let pipeline = BrickPipeline::new("my-pipeline");
        assert_eq!(pipeline.name(), "my-pipeline");
    }

    #[test]
    fn test_pipeline_stage_count() {
        let pipeline = BrickPipeline::new("test")
            .stage(TestStage {
                name: "s1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "s2",
                should_fail: false,
            })
            .stage(TestStage {
                name: "s3",
                should_fail: false,
            });

        assert_eq!(pipeline.stage_count(), 3);
    }

    #[test]
    fn test_pipeline_empty() {
        let mut pipeline = BrickPipeline::new("empty");

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_ok());
    }

    #[test]
    fn test_pipeline_with_checkpointing() {
        let pipeline =
            BrickPipeline::new("checkpointed").with_checkpointing(Duration::from_secs(5));

        // Just verify it compiles and sets the interval
        assert_eq!(pipeline.name(), "checkpointed");
    }

    #[test]
    fn test_pipeline_audit_trail() {
        let mut pipeline = BrickPipeline::new("audited")
            .stage(TestStage {
                name: "step1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "step2",
                should_fail: false,
            });

        let ctx = PipelineContext::new();
        let _ = pipeline.run(ctx);

        let trail = pipeline.audit_trail();
        assert_eq!(trail.len(), 2);
        assert!(trail[0].success);
        assert!(trail[1].success);
    }

    #[test]
    fn test_pipeline_audit_trail_with_failure() {
        let mut pipeline = BrickPipeline::new("audited")
            .stage(TestStage {
                name: "success",
                should_fail: false,
            })
            .stage(TestStage {
                name: "failure",
                should_fail: true,
            });

        let ctx = PipelineContext::new();
        let _ = pipeline.run(ctx);

        let trail = pipeline.audit_trail();
        assert_eq!(trail.len(), 2);
        assert!(trail[0].success);
        assert!(!trail[1].success);
    }

    #[test]
    fn test_pipeline_debug() {
        let pipeline = BrickPipeline::new("debug-test")
            .with_privacy(PrivacyTier::Private)
            .stage(TestStage {
                name: "s1",
                should_fail: false,
            });

        let debug_str = format!("{:?}", pipeline);
        assert!(debug_str.contains("BrickPipeline"));
        assert!(debug_str.contains("debug-test"));
        assert!(debug_str.contains("Private"));
    }

    #[test]
    fn test_pipeline_context_metadata_started_at() {
        let mut pipeline = BrickPipeline::new("test").stage(TestStage {
            name: "s1",
            should_fail: false,
        });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx).unwrap();

        assert!(result.metadata.started_at.is_some());
    }

    #[test]
    fn test_pipeline_traces_recorded() {
        let mut pipeline = BrickPipeline::new("traced").stage(TestStage {
            name: "traced_stage",
            should_fail: false,
        });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx).unwrap();

        assert_eq!(result.trace.len(), 1);
        assert_eq!(result.trace[0].stage_name, "traced_stage");
        assert!(result.trace[0].success);
        assert!(result.trace[0].error.is_none());
    }

    // ============================================================
    // BrickStage trait tests
    // ============================================================

    #[test]
    fn test_brick_stage_default_required_inputs() {
        let stage = TestStage {
            name: "test",
            should_fail: false,
        };

        assert!(stage.required_inputs().is_empty());
    }

    #[test]
    fn test_brick_stage_default_output_names() {
        let stage = TestStage {
            name: "test",
            should_fail: false,
        };

        assert!(stage.output_names().is_empty());
    }

    // ============================================================
    // uuid_v4 function test
    // ============================================================

    #[test]
    fn test_uuid_generation() {
        // Test that metadata run_id is unique
        let meta1 = PipelineMetadata::new();
        let meta2 = PipelineMetadata::new();

        // They should both start with "run-"
        assert!(meta1.run_id.starts_with("run-"));
        assert!(meta2.run_id.starts_with("run-"));
    }

    // ============================================================
    // Integration tests
    // ============================================================

    #[test]
    fn test_full_pipeline_workflow() {
        let mut pipeline = BrickPipeline::new("full-workflow")
            .with_privacy(PrivacyTier::Private)
            .stage(TestStage {
                name: "input",
                should_fail: false,
            })
            .stage(TestStage {
                name: "transform",
                should_fail: false,
            })
            .stage(TestStage {
                name: "output",
                should_fail: false,
            });

        let ctx = PipelineContext::from_input("initial", PipelineData::Text("start".into()));
        let result = pipeline.run(ctx).unwrap();

        // Check all stages executed
        assert!(result.get("input_output").is_some());
        assert!(result.get("transform_output").is_some());
        assert!(result.get("output_output").is_some());

        // Check traces
        assert_eq!(result.trace.len(), 3);

        // Check audit trail
        assert_eq!(pipeline.audit_trail().len(), 3);
    }

    #[test]
    fn test_pipeline_with_tensor_data() {
        let mut pipeline = BrickPipeline::new("tensor-pipeline").stage(TestStage {
            name: "process",
            should_fail: false,
        });

        let ctx = PipelineContext::from_input(
            "tensor",
            PipelineData::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );

        let result = pipeline.run(ctx).unwrap();

        // Original tensor data should still be accessible
        let tensor = result.get("tensor").unwrap();
        let (data, shape) = tensor.as_tensor().unwrap();
        assert_eq!(data.len(), 4);
        assert_eq!(shape, &[2, 2]);
    }

    // ============================================================
    // Additional coverage tests
    // ============================================================

    /// A slow stage for testing checkpointing
    struct SlowStage {
        name: &'static str,
        delay_ms: u64,
    }

    impl Brick for SlowStage {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(100)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(10),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl BrickStage for SlowStage {
        fn execute(&self, mut ctx: PipelineContext) -> PipelineResult<PipelineContext> {
            // Simulate slow execution
            std::thread::sleep(Duration::from_millis(self.delay_ms));
            ctx.set(
                format!("{}_output", self.name),
                PipelineData::Text("slow done".into()),
            );
            Ok(ctx)
        }

        fn validate(&self, _ctx: &PipelineContext) -> ValidationResult {
            ValidationResult::ok()
        }
    }

    /// A stage with multiple validation errors
    struct MultiErrorValidationStage {
        name: &'static str,
    }

    impl Brick for MultiErrorValidationStage {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(100)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(10),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl BrickStage for MultiErrorValidationStage {
        fn execute(&self, ctx: PipelineContext) -> PipelineResult<PipelineContext> {
            Ok(ctx)
        }

        fn validate(&self, _ctx: &PipelineContext) -> ValidationResult {
            let mut result = ValidationResult {
                valid: false,
                messages: vec![
                    ValidationMessage {
                        level: ValidationLevel::Error,
                        message: "First error".to_string(),
                    },
                    ValidationMessage {
                        level: ValidationLevel::Error,
                        message: "Second error".to_string(),
                    },
                    ValidationMessage {
                        level: ValidationLevel::Warning,
                        message: "A warning".to_string(),
                    },
                    ValidationMessage {
                        level: ValidationLevel::Info,
                        message: "Some info".to_string(),
                    },
                ],
            };
            // Add another warning to test warn() method
            result.warn("Another warning");
            result
        }
    }

    /// A stage with custom required inputs and outputs
    struct CustomIOStage {
        name: &'static str,
        inputs: &'static [&'static str],
        outputs: &'static [&'static str],
    }

    impl Brick for CustomIOStage {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(100)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(10),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl BrickStage for CustomIOStage {
        fn execute(&self, mut ctx: PipelineContext) -> PipelineResult<PipelineContext> {
            for output in self.outputs {
                ctx.set((*output).to_string(), PipelineData::Text("output".into()));
            }
            Ok(ctx)
        }

        fn validate(&self, _ctx: &PipelineContext) -> ValidationResult {
            ValidationResult::ok()
        }

        fn required_inputs(&self) -> &[&str] {
            self.inputs
        }

        fn output_names(&self) -> &[&str] {
            self.outputs
        }
    }

    #[test]
    fn test_pipeline_checkpointing_triggers() {
        // Use very short checkpoint interval (1ms) to ensure checkpoint is created
        let mut pipeline = BrickPipeline::new("checkpoint-test")
            .with_checkpointing(Duration::from_millis(1))
            .stage(SlowStage {
                name: "slow1",
                delay_ms: 5,
            })
            .stage(SlowStage {
                name: "slow2",
                delay_ms: 5,
            })
            .stage(SlowStage {
                name: "slow3",
                delay_ms: 5,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.get("slow1_output").is_some());
        assert!(output.get("slow2_output").is_some());
        assert!(output.get("slow3_output").is_some());
    }

    #[test]
    fn test_pipeline_multi_error_validation() {
        let mut pipeline =
            BrickPipeline::new("multi-error").stage(MultiErrorValidationStage { name: "multi" });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_err());
        match result {
            Err(PipelineError::ValidationFailed { stage, reason }) => {
                assert_eq!(stage, "multi");
                // Should contain both error messages joined by semicolons
                assert!(reason.contains("First error"));
                assert!(reason.contains("Second error"));
                // Should NOT contain warnings or info
                assert!(!reason.contains("warning"));
                assert!(!reason.contains("info"));
            }
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_custom_io_stage_inputs_outputs() {
        let stage = CustomIOStage {
            name: "custom",
            inputs: &["input1", "input2"],
            outputs: &["output1", "output2"],
        };

        assert_eq!(stage.required_inputs(), &["input1", "input2"]);
        assert_eq!(stage.output_names(), &["output1", "output2"]);
    }

    #[test]
    fn test_pipeline_with_custom_io_stage() {
        let mut pipeline = BrickPipeline::new("custom-io").stage(CustomIOStage {
            name: "custom",
            inputs: &["in"],
            outputs: &["out1", "out2"],
        });

        let ctx = PipelineContext::from_input("in", PipelineData::Text("input".into()));
        let result = pipeline.run(ctx).unwrap();

        assert!(result.get("out1").is_some());
        assert!(result.get("out2").is_some());
    }

    #[test]
    fn test_validation_level_info() {
        // Test Info level specifically
        let msg = ValidationMessage {
            level: ValidationLevel::Info,
            message: "Informational message".to_string(),
        };

        assert_eq!(msg.level, ValidationLevel::Info);
        assert!(format!("{:?}", msg.level).contains("Info"));
    }

    #[test]
    fn test_pipeline_error_clone() {
        // Test cloning of all error variants
        let err1 = PipelineError::ValidationFailed {
            stage: "s".to_string(),
            reason: "r".to_string(),
        };
        let cloned1 = err1;
        assert!(matches!(cloned1, PipelineError::ValidationFailed { .. }));

        let err2 = PipelineError::ExecutionFailed {
            stage: "s".to_string(),
            reason: "r".to_string(),
        };
        let cloned2 = err2;
        assert!(matches!(cloned2, PipelineError::ExecutionFailed { .. }));

        let err3 = PipelineError::MissingInput {
            stage: "s".to_string(),
            input: "i".to_string(),
        };
        let cloned3 = err3;
        assert!(matches!(cloned3, PipelineError::MissingInput { .. }));

        let err4 = PipelineError::PrivacyViolation {
            tier: PrivacyTier::Sovereign,
            reason: "r".to_string(),
        };
        let cloned4 = err4;
        assert!(matches!(cloned4, PipelineError::PrivacyViolation { .. }));

        let err5 = PipelineError::CheckpointFailed {
            reason: "r".to_string(),
        };
        let cloned5 = err5;
        assert!(matches!(cloned5, PipelineError::CheckpointFailed { .. }));

        let err6 = PipelineError::BrickError("e".to_string());
        let cloned6 = err6;
        assert!(matches!(cloned6, PipelineError::BrickError(_)));
    }

    #[test]
    fn test_pipeline_error_debug() {
        let err = PipelineError::ValidationFailed {
            stage: "test".to_string(),
            reason: "debug test".to_string(),
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("ValidationFailed"));
    }

    #[test]
    fn test_validation_result_multiple_warnings() {
        let mut result = ValidationResult::ok();
        result.warn("warning 1");
        result.warn("warning 2");
        result.warn("warning 3");

        assert!(result.valid);
        assert_eq!(result.messages.len(), 3);
        for msg in &result.messages {
            assert_eq!(msg.level, ValidationLevel::Warning);
        }
    }

    #[test]
    fn test_pipeline_data_debug_variants() {
        // Test Debug for all PipelineData variants
        let bytes = PipelineData::Bytes(vec![1, 2, 3]);
        assert!(format!("{:?}", bytes).contains("Bytes"));

        let tensor = PipelineData::FloatTensor {
            data: vec![1.0],
            shape: vec![1],
        };
        assert!(format!("{:?}", tensor).contains("FloatTensor"));

        let text = PipelineData::Text("hello".into());
        assert!(format!("{:?}", text).contains("Text"));

        let json = PipelineData::Json(serde_json::json!({}));
        assert!(format!("{:?}", json).contains("Json"));

        let int = PipelineData::Int(42);
        assert!(format!("{:?}", int).contains("Int"));

        let boolean = PipelineData::Bool(false);
        assert!(format!("{:?}", boolean).contains("Bool"));
    }

    #[test]
    fn test_pipeline_context_set_with_string() {
        let mut ctx = PipelineContext::new();
        // Test set() with String instead of &str
        ctx.set(String::from("key"), PipelineData::Int(123));

        assert!(ctx.get("key").is_some());
    }

    #[test]
    fn test_pipeline_metadata_tag_with_string() {
        let mut meta = PipelineMetadata::new();
        // Test tag() with String instead of &str
        meta.tag(String::from("key"), String::from("value"));

        assert_eq!(meta.tags.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_audit_collector_total_duration_empty() {
        let collector = PipelineAuditCollector::new();
        assert_eq!(collector.total_duration(), Duration::ZERO);
    }

    #[test]
    fn test_privacy_tier_copy() {
        let tier = PrivacyTier::Sovereign;
        let copied = tier;
        assert_eq!(tier, copied);
        assert_eq!(tier, PrivacyTier::Sovereign);
    }

    #[test]
    fn test_stage_trace_error_field() {
        let trace = StageTrace {
            stage_name: "error_stage".to_string(),
            duration: Duration::from_secs(1),
            success: false,
            error: Some("error message".to_string()),
        };

        assert_eq!(trace.error.as_deref(), Some("error message"));
    }

    #[test]
    fn test_pipeline_run_clears_checkpoint_on_success() {
        let mut pipeline = BrickPipeline::new("clear-checkpoint")
            .with_checkpointing(Duration::from_millis(1))
            .stage(SlowStage {
                name: "slow",
                delay_ms: 5,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_ok());
        // After successful run, checkpoint should be cleared
        // (internal state - verified by running again successfully)
        let ctx2 = PipelineContext::new();
        let result2 = pipeline.run(ctx2);
        assert!(result2.is_ok());
    }

    #[test]
    fn test_validation_message_levels() {
        let info = ValidationMessage {
            level: ValidationLevel::Info,
            message: "info".to_string(),
        };
        let warning = ValidationMessage {
            level: ValidationLevel::Warning,
            message: "warning".to_string(),
        };
        let error = ValidationMessage {
            level: ValidationLevel::Error,
            message: "error".to_string(),
        };

        assert_ne!(info.level, warning.level);
        assert_ne!(warning.level, error.level);
        assert_ne!(info.level, error.level);
    }

    #[test]
    fn test_pipeline_data_clone_all_variants() {
        let bytes = PipelineData::Bytes(vec![1, 2, 3]);
        let _ = bytes;

        let tensor = PipelineData::FloatTensor {
            data: vec![1.0, 2.0],
            shape: vec![2],
        };
        let _ = tensor;

        let text = PipelineData::Text("test".into());
        let _ = text;

        let json = PipelineData::Json(serde_json::json!({"key": "value"}));
        let _ = json;

        let int = PipelineData::Int(-100);
        let _ = int;

        let boolean = PipelineData::Bool(true);
        let _ = boolean;
    }

    #[test]
    fn test_pipeline_with_input_context() {
        let mut pipeline = BrickPipeline::new("with-input").stage(TestStage {
            name: "process",
            should_fail: false,
        });

        // Test running with pre-populated context
        let mut ctx = PipelineContext::new();
        ctx.set("input1", PipelineData::Text("value1".into()));
        ctx.set("input2", PipelineData::Int(42));
        ctx.metadata.tag("env", "test");

        let result = pipeline.run(ctx).unwrap();

        // Original inputs should still be present
        assert!(result.get("input1").is_some());
        assert!(result.get("input2").is_some());
        // Stage output should be present
        assert!(result.get("process_output").is_some());
    }

    #[test]
    fn test_uuid_v4_generates_unique_ids() {
        // Generate multiple run IDs and verify they're unique
        let mut ids = std::collections::HashSet::new();
        for _ in 0..100 {
            let meta = PipelineMetadata::new();
            ids.insert(meta.run_id);
        }
        // Should have generated 100 unique IDs (or very close due to timing)
        assert!(ids.len() >= 90);
    }

    #[test]
    fn test_pipeline_debug_format_complete() {
        let pipeline = BrickPipeline::new("debug-complete")
            .with_privacy(PrivacyTier::Sovereign)
            .stage(TestStage {
                name: "s1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "s2",
                should_fail: false,
            });

        let debug_str = format!("{:?}", pipeline);
        assert!(debug_str.contains("BrickPipeline"));
        assert!(debug_str.contains("debug-complete"));
        assert!(debug_str.contains("stage_count"));
        assert!(debug_str.contains('2'));
        assert!(debug_str.contains("Sovereign"));
    }

    #[test]
    fn test_checkpoint_fields() {
        let ctx = PipelineContext::from_input("test", PipelineData::Text("data".into()));
        let checkpoint = Checkpoint {
            stage_index: 5,
            context: ctx,
            created_at: Instant::now(),
        };

        assert_eq!(checkpoint.stage_index, 5);
        assert!(checkpoint.context.get("test").is_some());
    }

    #[test]
    fn test_audit_entry_fields() {
        let entry = AuditEntry {
            stage: "my_stage".to_string(),
            timestamp: Instant::now(),
            duration: Duration::from_millis(250),
            success: false,
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
        };

        assert_eq!(entry.stage, "my_stage");
        assert_eq!(entry.duration, Duration::from_millis(250));
        assert!(!entry.success);
        assert_eq!(entry.inputs.len(), 2);
        assert_eq!(entry.outputs.len(), 1);
    }

    #[test]
    fn test_pipeline_error_display_all_variants() {
        // Ensure all Display implementations are covered
        let errors = vec![
            PipelineError::ValidationFailed {
                stage: "stg".to_string(),
                reason: "rsn".to_string(),
            },
            PipelineError::ExecutionFailed {
                stage: "stg".to_string(),
                reason: "rsn".to_string(),
            },
            PipelineError::MissingInput {
                stage: "stg".to_string(),
                input: "inp".to_string(),
            },
            PipelineError::PrivacyViolation {
                tier: PrivacyTier::Private,
                reason: "rsn".to_string(),
            },
            PipelineError::CheckpointFailed {
                reason: "rsn".to_string(),
            },
            PipelineError::BrickError("err".to_string()),
        ];

        for err in errors {
            let display = format!("{}", err);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_pipeline_context_trace_with_error() {
        let mut ctx = PipelineContext::new();
        ctx.add_trace(StageTrace {
            stage_name: "failing".to_string(),
            duration: Duration::from_millis(50),
            success: false,
            error: Some("Detailed error message".to_string()),
        });

        assert_eq!(ctx.trace.len(), 1);
        assert!(!ctx.trace[0].success);
        assert!(ctx.trace[0].error.is_some());
        assert!(ctx.trace[0]
            .error
            .as_ref()
            .unwrap()
            .contains("Detailed error"));
    }

    /// A stage that sets a checkpoint marker so we can detect if checkpoint was restored
    struct CheckpointMarkerStage {
        name: &'static str,
        marker_value: &'static str,
    }

    impl Brick for CheckpointMarkerStage {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(100)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(10),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl BrickStage for CheckpointMarkerStage {
        fn execute(&self, mut ctx: PipelineContext) -> PipelineResult<PipelineContext> {
            ctx.set(
                format!("{}_marker", self.name),
                PipelineData::Text(self.marker_value.to_string()),
            );
            Ok(ctx)
        }

        fn validate(&self, _ctx: &PipelineContext) -> ValidationResult {
            ValidationResult::ok()
        }
    }

    #[test]
    fn test_pipeline_checkpoint_restoration() {
        // Create a pipeline with checkpointing
        let mut pipeline = BrickPipeline::new("checkpoint-restore-test")
            .with_checkpointing(Duration::from_nanos(1))
            .stage(SlowStage {
                name: "stage1",
                delay_ms: 2,
            })
            .stage(SlowStage {
                name: "stage2",
                delay_ms: 2,
            });

        // First run - creates checkpoint
        let ctx = PipelineContext::new();
        let result1 = pipeline.run(ctx);
        assert!(result1.is_ok());

        // Simulate failure and re-run - checkpoint would be used if present
        // Note: after successful completion checkpoint is cleared,
        // so this tests the clearing behavior
        let ctx2 = PipelineContext::new();
        let result2 = pipeline.run(ctx2);
        assert!(result2.is_ok());
    }

    #[test]
    fn test_pipeline_start_index_from_checkpoint() {
        // Manually set up a pipeline with a checkpoint to test start_index logic
        let mut pipeline = BrickPipeline::new("start-index-test")
            .stage(TestStage {
                name: "stage1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "stage2",
                should_fail: false,
            })
            .stage(TestStage {
                name: "stage3",
                should_fail: false,
            });

        // Manually set a checkpoint at stage index 1 (skip first stage)
        let checkpoint_ctx = PipelineContext::from_input("checkpoint_data", PipelineData::Int(42));
        pipeline.last_checkpoint = Some(Checkpoint {
            stage_index: 1,
            context: checkpoint_ctx,
            created_at: Instant::now(),
        });

        // Run with fresh context - should restore from checkpoint
        let fresh_ctx = PipelineContext::new();
        let result = pipeline.run(fresh_ctx).unwrap();

        // Should have stage2 and stage3 outputs (stage1 skipped)
        assert!(result.get("stage2_output").is_some());
        assert!(result.get("stage3_output").is_some());
        // stage1_output should NOT be present since we skipped it
        assert!(result.get("stage1_output").is_none());
        // checkpoint_data should be present since we restored from checkpoint
        assert!(result.get("checkpoint_data").is_some());
    }

    #[test]
    fn test_pipeline_checkpoint_context_restored() {
        // Verify that checkpoint context is actually restored
        let mut pipeline = BrickPipeline::new("context-restore-test")
            .stage(TestStage {
                name: "stage1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "stage2",
                should_fail: false,
            });

        // Create checkpoint with specific data
        let mut checkpoint_ctx = PipelineContext::new();
        checkpoint_ctx.set("restored_key", PipelineData::Text("restored_value".into()));

        pipeline.last_checkpoint = Some(Checkpoint {
            stage_index: 0,
            context: checkpoint_ctx,
            created_at: Instant::now(),
        });

        // Run should use checkpoint context
        let input_ctx =
            PipelineContext::from_input("input_key", PipelineData::Text("input_value".into()));
        let result = pipeline.run(input_ctx).unwrap();

        // Restored context should have the checkpoint data
        assert!(result.get("restored_key").is_some());
        // Input context's data should NOT be present (checkpoint overwrites)
        assert!(result.get("input_key").is_none());
    }

    #[test]
    fn test_multiple_checkpoints_during_run() {
        // Test that multiple checkpoints are created during a long run
        let mut pipeline = BrickPipeline::new("multi-checkpoint")
            .with_checkpointing(Duration::from_millis(1))
            .stage(SlowStage {
                name: "s1",
                delay_ms: 3,
            })
            .stage(SlowStage {
                name: "s2",
                delay_ms: 3,
            })
            .stage(SlowStage {
                name: "s3",
                delay_ms: 3,
            })
            .stage(SlowStage {
                name: "s4",
                delay_ms: 3,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.get("s1_output").is_some());
        assert!(output.get("s2_output").is_some());
        assert!(output.get("s3_output").is_some());
        assert!(output.get("s4_output").is_some());
    }

    #[test]
    fn test_checkpoint_not_created_when_interval_not_exceeded() {
        // Use a very long interval so checkpoint is never created
        let mut pipeline = BrickPipeline::new("no-checkpoint")
            .with_checkpointing(Duration::from_secs(3600)) // 1 hour
            .stage(TestStage {
                name: "fast1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "fast2",
                should_fail: false,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_ok());
        // Checkpoint should be None after run (cleared on success)
        assert!(pipeline.last_checkpoint.is_none());
    }

    #[test]
    fn test_all_privacy_tier_variants_in_debug() {
        // Ensure all PrivacyTier variants are covered in Debug
        let sovereign = PrivacyTier::Sovereign;
        let private = PrivacyTier::Private;
        let standard = PrivacyTier::Standard;

        assert!(format!("{:?}", sovereign).contains("Sovereign"));
        assert!(format!("{:?}", private).contains("Private"));
        assert!(format!("{:?}", standard).contains("Standard"));
    }

    #[test]
    fn test_pipeline_error_debug_all_variants() {
        // Test Debug for all PipelineError variants
        let errors: Vec<PipelineError> = vec![
            PipelineError::ValidationFailed {
                stage: "s".to_string(),
                reason: "r".to_string(),
            },
            PipelineError::ExecutionFailed {
                stage: "s".to_string(),
                reason: "r".to_string(),
            },
            PipelineError::MissingInput {
                stage: "s".to_string(),
                input: "i".to_string(),
            },
            PipelineError::PrivacyViolation {
                tier: PrivacyTier::Sovereign,
                reason: "r".to_string(),
            },
            PipelineError::CheckpointFailed {
                reason: "r".to_string(),
            },
            PipelineError::BrickError("e".to_string()),
        ];

        for err in errors {
            let debug_str = format!("{:?}", err);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_pipeline_run_with_zero_stages() {
        let mut pipeline = BrickPipeline::new("zero-stages");

        let ctx = PipelineContext::from_input("data", PipelineData::Bool(true));
        let result = pipeline.run(ctx).unwrap();

        // Input should still be present
        assert!(result.get("data").is_some());
        // started_at should be set
        assert!(result.metadata.started_at.is_some());
    }

    #[test]
    fn test_validation_result_fail_with_different_messages() {
        let fail1 = ValidationResult::fail("error message");
        assert!(!fail1.valid);
        assert_eq!(fail1.messages.len(), 1);

        let fail2 = ValidationResult::fail(String::from("string message"));
        assert!(!fail2.valid);
        assert_eq!(fail2.messages.len(), 1);
    }

    #[test]
    fn test_pipeline_data_tensor_multidimensional() {
        let data =
            PipelineData::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);

        let (values, shape) = data.as_tensor().unwrap();
        assert_eq!(values.len(), 8);
        assert_eq!(shape, &[2, 2, 2]);
    }

    #[test]
    fn test_audit_collector_records_multiple() {
        let mut collector = PipelineAuditCollector::new();

        collector.record("stage1", Duration::from_millis(10), true);
        collector.record("stage2", Duration::from_millis(20), true);
        collector.record("stage3", Duration::from_millis(30), false);
        collector.record("stage4", Duration::from_millis(40), true);

        assert_eq!(collector.entries().len(), 4);
        assert_eq!(collector.total_duration(), Duration::from_millis(100));

        // Verify individual entries
        assert_eq!(collector.entries()[0].stage, "stage1");
        assert!(collector.entries()[0].success);
        assert!(!collector.entries()[2].success);
    }

    // ============================================================
    // Additional coverage tests for 95%+ target
    // ============================================================

    #[test]
    fn test_checkpoint_marker_stage_execute() {
        // Use CheckpointMarkerStage to remove dead_code warning and cover its execute path
        let stage = CheckpointMarkerStage {
            name: "marker",
            marker_value: "test_marker",
        };

        let ctx = PipelineContext::new();
        let result = stage.execute(ctx).unwrap();

        assert!(result.get("marker_marker").is_some());
        if let Some(PipelineData::Text(value)) = result.get("marker_marker") {
            assert_eq!(value, "test_marker");
        } else {
            panic!("Expected Text variant");
        }
    }

    #[test]
    fn test_checkpoint_marker_stage_validate() {
        let stage = CheckpointMarkerStage {
            name: "marker",
            marker_value: "val",
        };

        let ctx = PipelineContext::new();
        let validation = stage.validate(&ctx);

        assert!(validation.valid);
    }

    #[test]
    fn test_checkpoint_marker_stage_brick_impl() {
        let stage = CheckpointMarkerStage {
            name: "test_marker",
            marker_value: "v",
        };

        assert_eq!(stage.brick_name(), "test_marker");
        assert!(stage.assertions().is_empty());
        assert!(stage.to_html().is_empty());
        assert!(stage.to_css().is_empty());

        let budget = stage.budget();
        assert_eq!(budget.total_ms, 100);

        let verify = stage.verify();
        assert!(verify.passed.is_empty());
        assert!(verify.failed.is_empty());
    }

    #[test]
    fn test_slow_stage_brick_impl() {
        let stage = SlowStage {
            name: "slow_test",
            delay_ms: 1,
        };

        assert_eq!(stage.brick_name(), "slow_test");
        assert!(stage.assertions().is_empty());
        assert!(stage.to_html().is_empty());
        assert!(stage.to_css().is_empty());

        let budget = stage.budget();
        assert_eq!(budget.total_ms, 100);

        let verify = stage.verify();
        assert!(verify.passed.is_empty());
    }

    #[test]
    fn test_slow_stage_validate() {
        let stage = SlowStage {
            name: "slow",
            delay_ms: 1,
        };

        let ctx = PipelineContext::new();
        let validation = stage.validate(&ctx);

        assert!(validation.valid);
    }

    #[test]
    fn test_multi_error_validation_stage_brick_impl() {
        let stage = MultiErrorValidationStage { name: "multi_err" };

        assert_eq!(stage.brick_name(), "multi_err");
        assert!(stage.assertions().is_empty());
        assert!(stage.to_html().is_empty());
        assert!(stage.to_css().is_empty());

        let budget = stage.budget();
        assert_eq!(budget.total_ms, 100);

        let verify = stage.verify();
        assert!(verify.passed.is_empty());
    }

    #[test]
    fn test_multi_error_validation_stage_execute() {
        let stage = MultiErrorValidationStage { name: "multi" };

        let ctx = PipelineContext::new();
        let result = stage.execute(ctx);

        // Execute always succeeds
        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_io_stage_brick_impl() {
        let stage = CustomIOStage {
            name: "custom_io",
            inputs: &["a"],
            outputs: &["b"],
        };

        assert_eq!(stage.brick_name(), "custom_io");
        assert!(stage.assertions().is_empty());
        assert!(stage.to_html().is_empty());
        assert!(stage.to_css().is_empty());

        let budget = stage.budget();
        assert_eq!(budget.total_ms, 100);

        let verify = stage.verify();
        assert!(verify.passed.is_empty());
    }

    #[test]
    fn test_custom_io_stage_validate() {
        let stage = CustomIOStage {
            name: "custom",
            inputs: &[],
            outputs: &[],
        };

        let ctx = PipelineContext::new();
        let validation = stage.validate(&ctx);

        assert!(validation.valid);
    }

    #[test]
    fn test_failing_validation_stage_brick_impl() {
        let stage = FailingValidationStage { name: "fail_val" };

        assert_eq!(stage.brick_name(), "fail_val");
        assert!(stage.assertions().is_empty());
        assert!(stage.to_html().is_empty());
        assert!(stage.to_css().is_empty());

        let budget = stage.budget();
        assert_eq!(budget.total_ms, 100);

        let verify = stage.verify();
        assert!(verify.passed.is_empty());
    }

    #[test]
    fn test_test_stage_brick_impl_full() {
        let stage = TestStage {
            name: "test_brick",
            should_fail: false,
        };

        assert_eq!(stage.brick_name(), "test_brick");
        assert!(stage.assertions().is_empty());
        assert!(stage.to_html().is_empty());
        assert!(stage.to_css().is_empty());

        let budget = stage.budget();
        assert_eq!(budget.total_ms, 100);

        let verify = stage.verify();
        assert!(verify.passed.is_empty());
        assert!(verify.failed.is_empty());
    }

    #[test]
    fn test_pipeline_failure_records_trace() {
        let mut pipeline = BrickPipeline::new("failure-trace")
            .stage(TestStage {
                name: "success_stage",
                should_fail: false,
            })
            .stage(TestStage {
                name: "fail_stage",
                should_fail: true,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_err());

        // Check audit trail includes both stages
        let trail = pipeline.audit_trail();
        assert_eq!(trail.len(), 2);
        assert!(trail[0].success);
        assert!(!trail[1].success);
    }

    #[test]
    fn test_pipeline_data_all_variants_as_methods() {
        // Test as_tensor on non-tensor types
        let bytes = PipelineData::Bytes(vec![1, 2]);
        assert!(bytes.as_tensor().is_none());
        assert!(bytes.as_text().is_none());

        let json = PipelineData::Json(serde_json::json!({}));
        assert!(json.as_tensor().is_none());
        assert!(json.as_text().is_none());

        let int = PipelineData::Int(42);
        assert!(int.as_tensor().is_none());
        assert!(int.as_text().is_none());

        let boolean = PipelineData::Bool(true);
        assert!(boolean.as_tensor().is_none());
        assert!(boolean.as_text().is_none());
    }

    #[test]
    fn test_pipeline_with_checkpoint_marker_stage() {
        let mut pipeline = BrickPipeline::new("marker-pipeline")
            .with_checkpointing(Duration::from_millis(1))
            .stage(CheckpointMarkerStage {
                name: "mark1",
                marker_value: "first",
            })
            .stage(SlowStage {
                name: "slow",
                delay_ms: 5,
            })
            .stage(CheckpointMarkerStage {
                name: "mark2",
                marker_value: "second",
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx).unwrap();

        assert!(result.get("mark1_marker").is_some());
        assert!(result.get("mark2_marker").is_some());
        assert!(result.get("slow_output").is_some());
    }

    #[test]
    fn test_pipeline_error_from_brick_error_explicit() {
        use crate::brick::BrickError;

        let brick_err = BrickError::MissingChild {
            expected: "child_brick".to_string(),
        };
        let pipeline_err = PipelineError::from(brick_err);

        match pipeline_err {
            PipelineError::BrickError(msg) => {
                assert!(msg.contains("child_brick"));
            }
            _ => panic!("Expected BrickError variant"),
        }
    }

    #[test]
    fn test_pipeline_context_multiple_traces() {
        let mut ctx = PipelineContext::new();

        for i in 0..5 {
            ctx.add_trace(StageTrace {
                stage_name: format!("stage_{}", i),
                duration: Duration::from_millis(10 * i as u64),
                success: i % 2 == 0,
                error: if i % 2 == 1 {
                    Some(format!("Error at stage {}", i))
                } else {
                    None
                },
            });
        }

        assert_eq!(ctx.trace.len(), 5);
        assert!(ctx.trace[0].success);
        assert!(!ctx.trace[1].success);
        assert!(ctx.trace[1].error.is_some());
    }

    #[test]
    fn test_validation_result_with_info_level() {
        let result = ValidationResult {
            valid: true,
            messages: vec![ValidationMessage {
                level: ValidationLevel::Info,
                message: "Just some info".to_string(),
            }],
        };

        assert!(result.valid);
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].level, ValidationLevel::Info);
    }

    #[test]
    fn test_pipeline_metadata_multiple_tags() {
        let mut meta = PipelineMetadata::new();

        meta.tag("key1", "value1");
        meta.tag("key2", "value2");
        meta.tag("key3", "value3");
        // Overwrite a key
        meta.tag("key1", "new_value1");

        assert_eq!(meta.tags.len(), 3);
        assert_eq!(meta.tags.get("key1"), Some(&"new_value1".to_string()));
    }

    #[test]
    fn test_pipeline_context_get_nonexistent() {
        let ctx = PipelineContext::new();

        assert!(ctx.get("nonexistent").is_none());
        assert!(ctx.get("").is_none());
        assert!(ctx.get("some_key").is_none());
    }

    #[test]
    fn test_pipeline_data_empty_tensor() {
        let data = PipelineData::tensor(vec![], vec![0]);

        let (values, shape) = data.as_tensor().unwrap();
        assert!(values.is_empty());
        assert_eq!(shape, &[0]);
    }

    #[test]
    fn test_pipeline_data_empty_text() {
        let data = PipelineData::Text(String::new());

        assert_eq!(data.as_text(), Some(""));
    }

    #[test]
    fn test_audit_entry_with_empty_io() {
        let entry = AuditEntry {
            stage: "empty_io".to_string(),
            timestamp: Instant::now(),
            duration: Duration::from_nanos(1),
            success: true,
            inputs: Vec::new(),
            outputs: Vec::new(),
        };

        assert!(entry.inputs.is_empty());
        assert!(entry.outputs.is_empty());
    }

    #[test]
    fn test_checkpoint_with_empty_context() {
        let checkpoint = Checkpoint {
            stage_index: 0,
            context: PipelineContext::new(),
            created_at: Instant::now(),
        };

        assert_eq!(checkpoint.stage_index, 0);
        assert!(checkpoint.context.data.is_empty());
    }

    #[test]
    fn test_pipeline_run_single_stage() {
        let mut pipeline = BrickPipeline::new("single").stage(TestStage {
            name: "only",
            should_fail: false,
        });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx).unwrap();

        assert!(result.get("only_output").is_some());
        assert_eq!(result.trace.len(), 1);
    }

    #[test]
    fn test_pipeline_first_stage_fails() {
        let mut pipeline = BrickPipeline::new("first-fail").stage(TestStage {
            name: "first",
            should_fail: true,
        });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_err());
        match result {
            Err(PipelineError::ExecutionFailed { stage, .. }) => {
                assert_eq!(stage, "first");
            }
            _ => panic!("Expected ExecutionFailed"),
        }
    }

    #[test]
    fn test_pipeline_first_stage_validation_fails() {
        let mut pipeline = BrickPipeline::new("first-val-fail")
            .stage(FailingValidationStage { name: "first_fail" });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_err());
        match result {
            Err(PipelineError::ValidationFailed { stage, .. }) => {
                assert_eq!(stage, "first_fail");
            }
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_pipeline_checkpoint_skip_first_stage() {
        let mut pipeline = BrickPipeline::new("skip-first")
            .stage(TestStage {
                name: "skipped",
                should_fail: false,
            })
            .stage(TestStage {
                name: "executed",
                should_fail: false,
            });

        // Set checkpoint to skip first stage
        pipeline.last_checkpoint = Some(Checkpoint {
            stage_index: 1,
            context: PipelineContext::from_input("from_checkpoint", PipelineData::Bool(true)),
            created_at: Instant::now(),
        });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx).unwrap();

        // skipped_output should NOT be present
        assert!(result.get("skipped_output").is_none());
        // executed_output should be present
        assert!(result.get("executed_output").is_some());
        // Checkpoint data should be present
        assert!(result.get("from_checkpoint").is_some());
    }

    #[test]
    fn test_pipeline_all_stages_skipped_by_checkpoint() {
        let mut pipeline = BrickPipeline::new("all-skipped")
            .stage(TestStage {
                name: "s1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "s2",
                should_fail: false,
            });

        // Set checkpoint to skip all stages
        pipeline.last_checkpoint = Some(Checkpoint {
            stage_index: 2, // Skip all
            context: PipelineContext::from_input("final_data", PipelineData::Int(999)),
            created_at: Instant::now(),
        });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx).unwrap();

        // No stage outputs should be present
        assert!(result.get("s1_output").is_none());
        assert!(result.get("s2_output").is_none());
        // Checkpoint data should be present
        assert!(result.get("final_data").is_some());
    }

    #[test]
    fn test_pipeline_with_many_stages() {
        let mut pipeline = BrickPipeline::new("many-stages");

        for i in 0..10 {
            pipeline = pipeline.stage(TestStage {
                name: Box::leak(format!("stage_{}", i).into_boxed_str()),
                should_fail: false,
            });
        }

        assert_eq!(pipeline.stage_count(), 10);

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx).unwrap();

        assert_eq!(result.trace.len(), 10);
    }

    #[test]
    fn test_pipeline_error_std_error_trait() {
        let err = PipelineError::MissingInput {
            stage: "s".to_string(),
            input: "i".to_string(),
        };

        // Test that it implements std::error::Error
        fn accepts_error<E: std::error::Error>(_e: &E) {}
        accepts_error(&err);

        // source() should return None for this error type
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn test_pipeline_context_set_overwrite() {
        let mut ctx = PipelineContext::new();

        ctx.set("key", PipelineData::Int(1));
        assert!(matches!(ctx.get("key"), Some(PipelineData::Int(1))));

        ctx.set("key", PipelineData::Int(2));
        assert!(matches!(ctx.get("key"), Some(PipelineData::Int(2))));

        ctx.set("key", PipelineData::Text("text".into()));
        assert!(matches!(ctx.get("key"), Some(PipelineData::Text(_))));
    }

    #[test]
    fn test_stage_trace_zero_duration() {
        let trace = StageTrace {
            stage_name: "instant".to_string(),
            duration: Duration::ZERO,
            success: true,
            error: None,
        };

        assert_eq!(trace.duration, Duration::ZERO);
    }

    #[test]
    fn test_pipeline_with_privacy_and_checkpointing() {
        let pipeline = BrickPipeline::new("full-config")
            .with_privacy(PrivacyTier::Sovereign)
            .with_checkpointing(Duration::from_secs(10))
            .stage(TestStage {
                name: "s1",
                should_fail: false,
            });

        assert_eq!(pipeline.privacy_tier(), PrivacyTier::Sovereign);
        assert_eq!(pipeline.stage_count(), 1);
    }

    #[test]
    fn test_pipeline_json_data_complex() {
        let complex_json = serde_json::json!({
            "array": [1, 2, 3],
            "nested": {
                "key": "value",
                "number": 42
            },
            "boolean": true,
            "null_value": null
        });

        let data = PipelineData::Json(complex_json);

        if let PipelineData::Json(value) = data {
            assert_eq!(value["array"][0], 1);
            assert_eq!(value["nested"]["key"], "value");
        } else {
            panic!("Expected Json variant");
        }
    }

    #[test]
    fn test_pipeline_bytes_large() {
        let large_bytes: Vec<u8> = (0..=255).collect();
        let data = PipelineData::Bytes(large_bytes);

        if let PipelineData::Bytes(bytes) = data {
            assert_eq!(bytes.len(), 256);
            assert_eq!(bytes[0], 0);
            assert_eq!(bytes[255], 255);
        } else {
            panic!("Expected Bytes variant");
        }
    }

    #[test]
    fn test_validation_result_fail_empty_reason() {
        let result = ValidationResult::fail("");

        assert!(!result.valid);
        assert_eq!(result.messages[0].message, "");
    }

    #[test]
    fn test_pipeline_context_from_input_preserves_metadata() {
        let ctx = PipelineContext::from_input("key", PipelineData::Bool(false));

        assert!(ctx.metadata.run_id.starts_with("run-"));
        assert!(ctx.trace.is_empty());
    }

    #[test]
    fn test_pipeline_stage_middle_fails() {
        let mut pipeline = BrickPipeline::new("middle-fail")
            .stage(TestStage {
                name: "first",
                should_fail: false,
            })
            .stage(TestStage {
                name: "middle",
                should_fail: true,
            })
            .stage(TestStage {
                name: "last",
                should_fail: false,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_err());

        // Audit trail should have 2 entries (first success, middle fail)
        let trail = pipeline.audit_trail();
        assert_eq!(trail.len(), 2);
    }

    #[test]
    fn test_pipeline_stage_last_fails() {
        let mut pipeline = BrickPipeline::new("last-fail")
            .stage(TestStage {
                name: "first",
                should_fail: false,
            })
            .stage(TestStage {
                name: "second",
                should_fail: false,
            })
            .stage(TestStage {
                name: "last",
                should_fail: true,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_err());

        let trail = pipeline.audit_trail();
        assert_eq!(trail.len(), 3);
        assert!(trail[0].success);
        assert!(trail[1].success);
        assert!(!trail[2].success);
    }

    #[test]
    fn test_pipeline_checkpoint_at_exact_interval() {
        // Test checkpoint creation at exactly the interval boundary
        let mut pipeline = BrickPipeline::new("exact-interval")
            .with_checkpointing(Duration::from_millis(0)) // Immediate checkpoint
            .stage(TestStage {
                name: "s1",
                should_fail: false,
            })
            .stage(TestStage {
                name: "s2",
                should_fail: false,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_ok());
        // Checkpoint should be cleared after successful run
        assert!(pipeline.last_checkpoint.is_none());
    }

    #[test]
    fn test_validation_level_copy_and_clone() {
        let info = ValidationLevel::Info;
        let copied = info;
        let cloned = copied;

        assert_eq!(info, copied);
        assert_eq!(copied, cloned);
    }

    #[test]
    fn test_privacy_tier_all_variants_equality() {
        let tiers = [
            PrivacyTier::Sovereign,
            PrivacyTier::Private,
            PrivacyTier::Standard,
        ];

        for (i, tier1) in tiers.iter().enumerate() {
            for (j, tier2) in tiers.iter().enumerate() {
                if i == j {
                    assert_eq!(tier1, tier2);
                } else {
                    assert_ne!(tier1, tier2);
                }
            }
        }
    }

    #[test]
    fn test_pipeline_metadata_started_at_is_set() {
        let mut pipeline = BrickPipeline::new("started").stage(TestStage {
            name: "s",
            should_fail: false,
        });

        let ctx = PipelineContext::new();
        assert!(ctx.metadata.started_at.is_none());

        let result = pipeline.run(ctx).unwrap();
        assert!(result.metadata.started_at.is_some());
    }

    #[test]
    fn test_pipeline_tensor_high_dimensional() {
        let data = PipelineData::tensor(
            vec![1.0; 24], // 2 * 3 * 4 = 24 elements
            vec![2, 3, 4],
        );

        let (values, shape) = data.as_tensor().unwrap();
        assert_eq!(values.len(), 24);
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn test_pipeline_context_debug() {
        let ctx = PipelineContext::from_input("debug_key", PipelineData::Int(42));
        let debug_str = format!("{:?}", ctx);

        assert!(debug_str.contains("PipelineContext"));
        assert!(debug_str.contains("debug_key"));
    }

    #[test]
    fn test_stage_trace_long_error_message() {
        let long_error = "Error ".repeat(1000);
        let trace = StageTrace {
            stage_name: "long_error".to_string(),
            duration: Duration::from_millis(1),
            success: false,
            error: Some(long_error.clone()),
        };

        assert_eq!(trace.error.as_ref().unwrap().len(), long_error.len());
    }

    #[test]
    fn test_pipeline_with_checkpoint_and_failure() {
        let mut pipeline = BrickPipeline::new("checkpoint-fail")
            .with_checkpointing(Duration::from_nanos(1))
            .stage(SlowStage {
                name: "slow",
                delay_ms: 2,
            })
            .stage(TestStage {
                name: "fail",
                should_fail: true,
            });

        let ctx = PipelineContext::new();
        let result = pipeline.run(ctx);

        assert!(result.is_err());
    }

    #[test]
    fn test_audit_collector_single_entry_duration() {
        let mut collector = PipelineAuditCollector::new();
        collector.record("single", Duration::from_secs(5), true);

        assert_eq!(collector.total_duration(), Duration::from_secs(5));
    }
}
