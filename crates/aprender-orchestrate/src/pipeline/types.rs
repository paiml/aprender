//! Pipeline types and trait definitions.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Context passed between pipeline stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineContext {
    /// Input path (source project)
    pub input_path: PathBuf,

    /// Output path (transpiled Rust project)
    pub output_path: PathBuf,

    /// Primary language detected
    pub primary_language: Option<crate::types::Language>,

    /// Transpiled file mappings (source -> output)
    pub file_mappings: Vec<(PathBuf, PathBuf)>,

    /// Optimization passes applied
    pub optimizations: Vec<String>,

    /// Validation results
    pub validation_results: Vec<ValidationResult>,

    /// Metadata accumulated during pipeline
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl PipelineContext {
    pub fn new(input_path: PathBuf, output_path: PathBuf) -> Self {
        Self {
            input_path,
            output_path,
            primary_language: None,
            file_mappings: Vec::new(),
            optimizations: Vec::new(),
            validation_results: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Get final output artifacts
    pub fn output(&self) -> PipelineOutput {
        PipelineOutput {
            output_path: self.output_path.clone(),
            file_mappings: self.file_mappings.clone(),
            optimizations: self.optimizations.clone(),
            validation_passed: self.validation_results.iter().all(|v| v.passed),
        }
    }
}

/// Validation result from a pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub stage: String,
    pub passed: bool,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

/// Final output from the pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOutput {
    pub output_path: PathBuf,
    pub file_mappings: Vec<(PathBuf, PathBuf)>,
    pub optimizations: Vec<String>,
    pub validation_passed: bool,
}

/// Validation strategy for pipeline stages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationStrategy {
    /// Stop on first error (Jidoka principle)
    StopOnError,
    /// Continue on errors but collect them
    ContinueOnError,
    /// Skip validation
    None,
}

/// Trait for pipeline stages
#[async_trait::async_trait]
pub trait PipelineStage: Send + Sync {
    /// Name of this stage
    fn name(&self) -> &str;

    /// Execute this stage
    async fn execute(&self, ctx: PipelineContext) -> Result<PipelineContext>;

    /// Validate the output of this stage
    fn validate(&self, _ctx: &PipelineContext) -> Result<ValidationResult> {
        // Default: always pass
        Ok(ValidationResult {
            stage: self.name().to_string(),
            passed: true,
            message: "No validation configured".to_string(),
            details: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_context_new() {
        let ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
        assert_eq!(ctx.input_path, PathBuf::from("/input"));
        assert_eq!(ctx.output_path, PathBuf::from("/output"));
        assert!(ctx.primary_language.is_none());
        assert!(ctx.file_mappings.is_empty());
        assert!(ctx.optimizations.is_empty());
        assert!(ctx.validation_results.is_empty());
        assert!(ctx.metadata.is_empty());
    }

    #[test]
    fn test_pipeline_context_output() {
        let ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
        let output = ctx.output();
        assert_eq!(output.output_path, PathBuf::from("/output"));
        assert!(output.validation_passed);
    }

    #[test]
    fn test_pipeline_context_output_with_failed_validation() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
        ctx.validation_results.push(ValidationResult {
            stage: "test".to_string(),
            passed: false,
            message: "Failed".to_string(),
            details: None,
        });
        let output = ctx.output();
        assert!(!output.validation_passed);
    }

    #[test]
    fn test_pipeline_context_output_with_mixed_validations() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
        ctx.validation_results.push(ValidationResult {
            stage: "stage1".to_string(),
            passed: true,
            message: "OK".to_string(),
            details: None,
        });
        ctx.validation_results.push(ValidationResult {
            stage: "stage2".to_string(),
            passed: false,
            message: "Failed".to_string(),
            details: None,
        });
        let output = ctx.output();
        assert!(!output.validation_passed); // One failure means overall failure
    }

    #[test]
    fn test_validation_result_serialization() {
        let result = ValidationResult {
            stage: "test".to_string(),
            passed: true,
            message: "Success".to_string(),
            details: Some(serde_json::json!({"key": "value"})),
        };
        let json = serde_json::to_string(&result).expect("json serialize failed");
        let deserialized: ValidationResult =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized.stage, "test");
        assert!(deserialized.passed);
    }

    #[test]
    fn test_pipeline_output_serialization() {
        let output = PipelineOutput {
            output_path: PathBuf::from("/out"),
            file_mappings: vec![(PathBuf::from("a.py"), PathBuf::from("a.rs"))],
            optimizations: vec!["opt1".to_string()],
            validation_passed: true,
        };
        let json = serde_json::to_string(&output).expect("json serialize failed");
        let deserialized: PipelineOutput =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized.file_mappings.len(), 1);
    }

    #[test]
    fn test_validation_strategy_equality() {
        assert_eq!(ValidationStrategy::StopOnError, ValidationStrategy::StopOnError);
        assert_ne!(ValidationStrategy::StopOnError, ValidationStrategy::ContinueOnError);
        assert_ne!(ValidationStrategy::ContinueOnError, ValidationStrategy::None);
    }

    #[test]
    fn test_pipeline_context_serialization() {
        let ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
        let json = serde_json::to_string(&ctx).expect("json serialize failed");
        let deserialized: PipelineContext =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized.input_path, PathBuf::from("/input"));
    }

    #[test]
    fn test_pipeline_context_with_file_mappings() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
        ctx.file_mappings.push((PathBuf::from("src/main.py"), PathBuf::from("src/main.rs")));
        let output = ctx.output();
        assert_eq!(output.file_mappings.len(), 1);
    }

    #[test]
    fn test_pipeline_context_with_optimizations() {
        let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
        ctx.optimizations.push("dead_code_elimination".to_string());
        ctx.optimizations.push("inlining".to_string());
        let output = ctx.output();
        assert_eq!(output.optimizations.len(), 2);
    }
}
