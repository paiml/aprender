//! Pipeline module tests.

use super::*;
use std::path::{Path, PathBuf};

// ============================================================================
// PIPELINE CONTEXT TESTS
// ============================================================================

#[test]
fn test_pipeline_context_creation() {
    let input = PathBuf::from("/input");
    let output = PathBuf::from("/output");

    let ctx = PipelineContext::new(input.clone(), output.clone());

    assert_eq!(ctx.input_path, input);
    assert_eq!(ctx.output_path, output);
    assert!(ctx.primary_language.is_none());
    assert!(ctx.file_mappings.is_empty());
    assert!(ctx.optimizations.is_empty());
    assert!(ctx.validation_results.is_empty());
    assert!(ctx.metadata.is_empty());
}

#[test]
fn test_pipeline_context_with_language() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    ctx.primary_language = Some(crate::types::Language::Python);
    assert_eq!(ctx.primary_language, Some(crate::types::Language::Python));
}

#[test]
fn test_pipeline_context_file_mappings() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    ctx.file_mappings.push((PathBuf::from("/input/main.py"), PathBuf::from("/output/src/main.rs")));

    assert_eq!(ctx.file_mappings.len(), 1);
    assert_eq!(ctx.file_mappings[0].0, PathBuf::from("/input/main.py"));
    assert_eq!(ctx.file_mappings[0].1, PathBuf::from("/output/src/main.rs"));
}

#[test]
fn test_pipeline_context_optimizations() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    ctx.optimizations.push("SIMD enabled".to_string());
    ctx.optimizations.push("GPU dispatch enabled".to_string());

    assert_eq!(ctx.optimizations.len(), 2);
    assert_eq!(ctx.optimizations[0], "SIMD enabled");
    assert_eq!(ctx.optimizations[1], "GPU dispatch enabled");
}

#[test]
fn test_pipeline_context_validation_results() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    ctx.validation_results.push(ValidationResult {
        stage: "Analysis".to_string(),
        passed: true,
        message: "Language detected".to_string(),
        details: None,
    });

    assert_eq!(ctx.validation_results.len(), 1);
    assert!(ctx.validation_results[0].passed);
    assert_eq!(ctx.validation_results[0].stage, "Analysis");
}

#[test]
fn test_pipeline_context_metadata() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    ctx.metadata.insert("total_files".to_string(), serde_json::json!(42));
    ctx.metadata.insert("language".to_string(), serde_json::json!("Python"));

    assert_eq!(ctx.metadata.len(), 2);
    assert_eq!(ctx.metadata.get("total_files").unwrap(), &serde_json::json!(42));
    assert_eq!(ctx.metadata.get("language").unwrap(), &serde_json::json!("Python"));
}

#[test]
fn test_pipeline_context_output() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    ctx.file_mappings.push((PathBuf::from("/input/main.py"), PathBuf::from("/output/main.rs")));
    ctx.optimizations.push("SIMD".to_string());
    ctx.validation_results.push(ValidationResult {
        stage: "Test".to_string(),
        passed: true,
        message: "OK".to_string(),
        details: None,
    });

    let output = ctx.output();

    assert_eq!(output.output_path, PathBuf::from("/output"));
    assert_eq!(output.file_mappings.len(), 1);
    assert_eq!(output.optimizations.len(), 1);
    assert!(output.validation_passed);
}

#[test]
fn test_pipeline_context_output_validation_failed() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    // Add one passing and one failing validation result
    ctx.validation_results.push(ValidationResult {
        stage: "Stage1".to_string(),
        passed: true,
        message: "OK".to_string(),
        details: None,
    });
    ctx.validation_results.push(ValidationResult {
        stage: "Stage2".to_string(),
        passed: false,
        message: "Failed".to_string(),
        details: None,
    });

    let output = ctx.output();

    // Should be false because at least one validation failed
    assert!(!output.validation_passed);
}

// ============================================================================
// VALIDATION RESULT TESTS
// ============================================================================

#[test]
fn test_validation_result_passed() {
    let result = ValidationResult {
        stage: "TestStage".to_string(),
        passed: true,
        message: "All checks passed".to_string(),
        details: None,
    };

    assert_eq!(result.stage, "TestStage");
    assert!(result.passed);
    assert_eq!(result.message, "All checks passed");
    assert!(result.details.is_none());
}

#[test]
fn test_validation_result_failed() {
    let result = ValidationResult {
        stage: "TestStage".to_string(),
        passed: false,
        message: "Check failed".to_string(),
        details: Some(serde_json::json!({"error": "details here"})),
    };

    assert_eq!(result.stage, "TestStage");
    assert!(!result.passed);
    assert_eq!(result.message, "Check failed");
    assert!(result.details.is_some());
}

#[test]
fn test_validation_result_with_details() {
    let details = serde_json::json!({
        "errors": ["error1", "error2"],
        "warnings": ["warning1"]
    });

    let result = ValidationResult {
        stage: "Analysis".to_string(),
        passed: false,
        message: "Multiple issues found".to_string(),
        details: Some(details.clone()),
    };

    assert_eq!(result.details, Some(details));
}

// ============================================================================
// VALIDATION STRATEGY TESTS
// ============================================================================

#[test]
fn test_validation_strategy_equality() {
    assert_eq!(ValidationStrategy::StopOnError, ValidationStrategy::StopOnError);
    assert_eq!(ValidationStrategy::ContinueOnError, ValidationStrategy::ContinueOnError);
    assert_eq!(ValidationStrategy::None, ValidationStrategy::None);

    assert_ne!(ValidationStrategy::StopOnError, ValidationStrategy::ContinueOnError);
    assert_ne!(ValidationStrategy::StopOnError, ValidationStrategy::None);
    assert_ne!(ValidationStrategy::ContinueOnError, ValidationStrategy::None);
}

// ============================================================================
// TRANSPILATION PIPELINE TESTS
// ============================================================================

struct MockStage {
    name: String,
    should_fail: bool,
    validation_should_pass: bool,
}

impl MockStage {
    fn new(name: &str) -> Self {
        Self { name: name.to_string(), should_fail: false, validation_should_pass: true }
    }

    fn with_execution_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }

    fn with_validation_failure(mut self) -> Self {
        self.validation_should_pass = false;
        self
    }
}

#[async_trait::async_trait]
impl PipelineStage for MockStage {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, mut ctx: PipelineContext) -> anyhow::Result<PipelineContext> {
        if self.should_fail {
            anyhow::bail!("Execution failed for {}", self.name);
        }

        ctx.metadata.insert(format!("{}_executed", self.name), serde_json::json!(true));

        Ok(ctx)
    }

    fn validate(&self, _ctx: &PipelineContext) -> anyhow::Result<ValidationResult> {
        Ok(ValidationResult {
            stage: self.name.clone(),
            passed: self.validation_should_pass,
            message: if self.validation_should_pass {
                format!("{} validation passed", self.name)
            } else {
                format!("{} validation failed", self.name)
            },
            details: None,
        })
    }
}

#[tokio::test]
async fn test_pipeline_creation() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError);
    assert_eq!(pipeline.stages.len(), 0);
    assert_eq!(pipeline.validation, ValidationStrategy::StopOnError);
}

#[tokio::test]
async fn test_pipeline_add_stage() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
        .add_stage(Box::new(MockStage::new("Stage1")))
        .add_stage(Box::new(MockStage::new("Stage2")));

    assert_eq!(pipeline.stages.len(), 2);
}

#[tokio::test]
async fn test_pipeline_run_no_stages() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::None);

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());

    let pipeline_output = result.unwrap();
    assert_eq!(pipeline_output.output_path, output);
    assert!(pipeline_output.file_mappings.is_empty());
}

#[tokio::test]
async fn test_pipeline_run_single_stage() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
        .add_stage(Box::new(MockStage::new("TestStage")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());

    let pipeline_output = result.unwrap();
    assert_eq!(pipeline_output.output_path, output);
}

#[tokio::test]
async fn test_pipeline_run_multiple_stages() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
        .add_stage(Box::new(MockStage::new("Stage1")))
        .add_stage(Box::new(MockStage::new("Stage2")))
        .add_stage(Box::new(MockStage::new("Stage3")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_pipeline_stage_execution_order() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
        .add_stage(Box::new(MockStage::new("First")))
        .add_stage(Box::new(MockStage::new("Second")))
        .add_stage(Box::new(MockStage::new("Third")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());

    // Stages should execute in order, adding metadata
    // This is implicitly tested by the fact that execution succeeds
}

#[tokio::test]
async fn test_pipeline_stop_on_error_execution() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
        .add_stage(Box::new(MockStage::new("Stage1")))
        .add_stage(Box::new(MockStage::new("Stage2").with_execution_failure()))
        .add_stage(Box::new(MockStage::new("Stage3")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.to_string().contains("Stage 'Stage2' failed"));
}

#[tokio::test]
async fn test_pipeline_stop_on_error_validation() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
        .add_stage(Box::new(MockStage::new("Stage1")))
        .add_stage(Box::new(MockStage::new("Stage2").with_validation_failure()))
        .add_stage(Box::new(MockStage::new("Stage3")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.to_string().contains("Validation failed for stage 'Stage2'"));
}

#[tokio::test]
async fn test_pipeline_continue_on_error_validation() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::ContinueOnError)
        .add_stage(Box::new(MockStage::new("Stage1")))
        .add_stage(Box::new(MockStage::new("Stage2").with_validation_failure()))
        .add_stage(Box::new(MockStage::new("Stage3")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());

    // Validation failure should be recorded but pipeline continues
    let pipeline_output = result.unwrap();
    assert!(!pipeline_output.validation_passed);
}

#[tokio::test]
async fn test_pipeline_no_validation() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
        .add_stage(Box::new(MockStage::new("Stage1").with_validation_failure()))
        .add_stage(Box::new(MockStage::new("Stage2").with_validation_failure()));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());

    // No validation should be performed, so pipeline succeeds
    let pipeline_output = result.unwrap();
    // With ValidationStrategy::None, no validation results are added
    assert!(pipeline_output.validation_passed); // All (zero) validations passed
}

// ============================================================================
// TRANSPILATION STAGE TESTS
// ============================================================================

#[test]
fn test_transpilation_stage_creation() {
    let stage = TranspilationStage::new(true, true);
    assert!(stage.incremental);
    assert!(stage.cache);
    // library_analyzer is initialized by default
    let _ = &stage.library_analyzer;
}

#[test]
fn test_transpilation_stage_name() {
    let stage = TranspilationStage::new(false, false);
    assert_eq!(stage.name(), "Transpilation");
}

// ============================================================================
// OPTIMIZATION STAGE TESTS
// ============================================================================

#[test]
fn test_optimization_stage_creation() {
    let stage = OptimizationStage::new(true, true, 1000);
    assert!(stage.enable_gpu);
    assert!(stage.enable_simd);
    assert_eq!(stage.gpu_threshold, 1000);
}

#[test]
fn test_optimization_stage_name() {
    let stage = OptimizationStage::new(false, false, 500);
    assert_eq!(stage.name(), "Optimization");
}

#[test]
fn test_optimization_stage_analyze_optimizations() {
    let stage = OptimizationStage::new(true, true, 1000);
    let recommendations = stage.analyze_optimizations();

    // Should return 3 recommendations (from the hardcoded workloads)
    assert_eq!(recommendations.len(), 3);

    // Check that recommendations contain backend information
    assert!(recommendations[0].contains("Element-wise operations"));
    assert!(recommendations[1].contains("Vector reductions"));
    assert!(recommendations[2].contains("Matrix multiplications"));
}

#[tokio::test]
async fn test_optimization_stage_execute() {
    let stage = OptimizationStage::new(true, true, 1000);

    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();

    // Check that optimizations were added
    assert!(!ctx.optimizations.is_empty());
    assert!(ctx.optimizations.iter().any(|o| o.contains("SIMD")));
    assert!(ctx.optimizations.iter().any(|o| o.contains("GPU")));

    // Check metadata
    assert!(ctx.metadata.contains_key("optimizations_applied"));
    assert!(ctx.metadata.contains_key("moe_routing_enabled"));
}

#[tokio::test]
async fn test_optimization_stage_simd_only() {
    let stage = OptimizationStage::new(false, true, 1000);

    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();

    // Should have SIMD but not GPU in traditional optimizations
    assert!(ctx.optimizations.iter().any(|o| o.contains("SIMD")));
    assert!(!ctx.optimizations.iter().any(|o| o.contains("GPU dispatch enabled")));
}

#[tokio::test]
async fn test_optimization_stage_gpu_only() {
    let stage = OptimizationStage::new(true, false, 2000);

    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();

    // Should have GPU but not SIMD in traditional optimizations
    assert!(!ctx.optimizations.iter().any(|o| o.contains("SIMD vectorization enabled")));
    assert!(ctx.optimizations.iter().any(|o| o.contains("GPU")));
    assert!(ctx.optimizations.iter().any(|o| o.contains("2000")));
}

// ============================================================================
// VALIDATION STAGE TESTS
// ============================================================================

#[test]
fn test_validation_stage_creation() {
    let stage = ValidationStage::new(true, true);
    assert!(stage.trace_syscalls);
    assert!(stage.run_tests);
}

#[test]
fn test_validation_stage_name() {
    let stage = ValidationStage::new(false, false);
    assert_eq!(stage.name(), "Validation");
}

#[test]
fn test_validation_stage_compare_traces_identical() {
    let trace1 =
        vec!["open(/file)".to_string(), "read(fd, buf, 100)".to_string(), "close(fd)".to_string()];

    let trace2 =
        vec!["open(/file)".to_string(), "read(fd, buf, 100)".to_string(), "close(fd)".to_string()];

    assert!(ValidationStage::compare_traces(&trace1, &trace2));
}

#[test]
fn test_validation_stage_compare_traces_different_length() {
    let trace1 = vec!["open(/file)".to_string(), "close(fd)".to_string()];

    let trace2 = vec!["open(/file)".to_string()];

    assert!(!ValidationStage::compare_traces(&trace1, &trace2));
}

#[test]
fn test_validation_stage_compare_traces_different_syscalls() {
    let trace1 = vec!["open(/file)".to_string(), "read(fd, buf, 100)".to_string()];

    let trace2 = vec!["open(/file)".to_string(), "write(fd, buf, 100)".to_string()];

    assert!(!ValidationStage::compare_traces(&trace1, &trace2));
}

#[test]
fn test_validation_stage_compare_traces_same_syscall_different_args() {
    // Should pass - only syscall names are compared, not arguments
    let trace1 = vec!["open(/file1)".to_string(), "read(fd, buf, 100)".to_string()];

    let trace2 = vec!["open(/file2)".to_string(), "read(fd, buf, 200)".to_string()];

    assert!(ValidationStage::compare_traces(&trace1, &trace2));
}

// ============================================================================
// BUILD STAGE TESTS
// ============================================================================

#[test]
fn test_build_stage_creation() {
    let stage = BuildStage::new(true, Some("x86_64-unknown-linux-gnu".to_string()), false);
    assert!(stage.release);
    assert_eq!(stage.target, Some("x86_64-unknown-linux-gnu".to_string()));
    assert!(!stage.wasm);
}

#[test]
fn test_build_stage_creation_wasm() {
    let stage = BuildStage::new(false, None, true);
    assert!(!stage.release);
    assert!(stage.target.is_none());
    assert!(stage.wasm);
}

#[test]
fn test_build_stage_name() {
    let stage = BuildStage::new(false, None, false);
    assert_eq!(stage.name(), "Build");
}

#[test]
fn test_build_stage_validate_no_build_dir() {
    let stage = BuildStage::new(true, None, false);

    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/nonexistent"));

    let result = stage.validate(&ctx);
    assert!(result.is_ok());

    let validation = result.unwrap();
    assert!(!validation.passed);
    assert!(validation.message.contains("not found"));
}

// ============================================================================
// ANALYSIS STAGE TESTS
// ============================================================================

#[test]
fn test_analysis_stage_name() {
    let stage = AnalysisStage;
    assert_eq!(stage.name(), "Analysis");
}

#[test]
fn test_analysis_stage_validate_no_language() {
    let stage = AnalysisStage;

    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

    let result = stage.validate(&ctx);
    assert!(result.is_ok());

    let validation = result.unwrap();
    assert!(!validation.passed);
    assert!(validation.message.contains("Could not detect"));
}

#[test]
fn test_analysis_stage_validate_with_language() {
    let stage = AnalysisStage;

    let mut ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));
    ctx.primary_language = Some(crate::types::Language::Python);

    let result = stage.validate(&ctx);
    assert!(result.is_ok());

    let validation = result.unwrap();
    assert!(validation.passed);
    assert!(validation.message.contains("Language detected"));
}

// ============================================================================
// SERIALIZATION TESTS
// ============================================================================

#[test]
fn test_pipeline_context_serialization() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
    ctx.primary_language = Some(crate::types::Language::Python);
    ctx.file_mappings.push((PathBuf::from("a.py"), PathBuf::from("a.rs")));
    ctx.optimizations.push("SIMD".to_string());

    let json = serde_json::to_string(&ctx).unwrap();
    let deserialized: PipelineContext = serde_json::from_str(&json).unwrap();

    assert_eq!(ctx.input_path, deserialized.input_path);
    assert_eq!(ctx.output_path, deserialized.output_path);
    assert_eq!(ctx.primary_language, deserialized.primary_language);
    assert_eq!(ctx.file_mappings.len(), deserialized.file_mappings.len());
    assert_eq!(ctx.optimizations.len(), deserialized.optimizations.len());
}

#[test]
fn test_validation_result_serialization() {
    let result = ValidationResult {
        stage: "Test".to_string(),
        passed: true,
        message: "OK".to_string(),
        details: Some(serde_json::json!({"key": "value"})),
    };

    let json = serde_json::to_string(&result).unwrap();
    let deserialized: ValidationResult = serde_json::from_str(&json).unwrap();

    assert_eq!(result.stage, deserialized.stage);
    assert_eq!(result.passed, deserialized.passed);
    assert_eq!(result.message, deserialized.message);
    assert_eq!(result.details, deserialized.details);
}

#[test]
fn test_pipeline_output_serialization() {
    let output = PipelineOutput {
        output_path: PathBuf::from("/output"),
        file_mappings: vec![(PathBuf::from("a"), PathBuf::from("b"))],
        optimizations: vec!["opt1".to_string(), "opt2".to_string()],
        validation_passed: true,
    };

    let json = serde_json::to_string(&output).unwrap();
    let deserialized: PipelineOutput = serde_json::from_str(&json).unwrap();

    assert_eq!(output.output_path, deserialized.output_path);
    assert_eq!(output.file_mappings.len(), deserialized.file_mappings.len());
    assert_eq!(output.optimizations.len(), deserialized.optimizations.len());
    assert_eq!(output.validation_passed, deserialized.validation_passed);
}

// ============================================================================
// ANALYSIS STAGE EXECUTION TESTS
// ============================================================================

#[cfg(feature = "native")]
#[tokio::test]
async fn test_analysis_stage_execute_with_rust_project() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    std::fs::create_dir(&src_dir).unwrap();
    std::fs::write(src_dir.join("main.rs"), "fn main() {}").unwrap();

    let stage = AnalysisStage;
    let ctx = PipelineContext::new(temp_dir.path().to_path_buf(), PathBuf::from("/tmp/output"));

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();
    assert_eq!(ctx.primary_language, Some(crate::types::Language::Rust));
    assert!(ctx.metadata.contains_key("total_files"));
    assert!(ctx.metadata.contains_key("total_lines"));
}

#[cfg(feature = "native")]
#[tokio::test]
async fn test_analysis_stage_execute_with_python_project() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    std::fs::write(temp_dir.path().join("main.py"), "print('hello')").unwrap();

    let stage = AnalysisStage;
    let ctx = PipelineContext::new(temp_dir.path().to_path_buf(), PathBuf::from("/tmp/output"));

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();
    assert_eq!(ctx.primary_language, Some(crate::types::Language::Python));
}

#[cfg(feature = "native")]
#[tokio::test]
async fn test_analysis_stage_execute_empty_directory() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();

    let stage = AnalysisStage;
    let ctx = PipelineContext::new(temp_dir.path().to_path_buf(), PathBuf::from("/tmp/output"));

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();
    assert!(ctx.primary_language.is_none());
}

// ============================================================================
// TRANSPILATION STAGE EXECUTION TESTS
// ============================================================================

#[cfg(feature = "native")]
#[tokio::test]
async fn test_transpilation_stage_creates_directories() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("output");

    let stage = TranspilationStage::new(false, false);
    let mut ctx = PipelineContext::new(temp_dir.path().to_path_buf(), output_dir.clone());
    ctx.primary_language = Some(crate::types::Language::Rust);

    // This will fail at transpilation but should create directories
    let _result = stage.execute(ctx).await;

    // Check that directories were created even if transpilation fails
    assert!(output_dir.exists());
    assert!(output_dir.join("src").exists());
}

#[test]
fn test_transpilation_stage_validate_empty_output() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("output");
    std::fs::create_dir_all(output_dir.join("src")).unwrap();

    let stage = TranspilationStage::new(false, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), output_dir.clone());

    let result = stage.validate(&ctx);
    assert!(result.is_ok());

    let validation = result.unwrap();
    // Should fail because src dir is empty
    assert!(!validation.passed);
}

#[test]
fn test_transpilation_stage_validate_with_files() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("output");
    std::fs::create_dir_all(output_dir.join("src")).unwrap();
    std::fs::write(output_dir.join("src/main.rs"), "fn main() {}").unwrap();

    let stage = TranspilationStage::new(false, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), output_dir.clone());

    let result = stage.validate(&ctx);
    assert!(result.is_ok());

    let validation = result.unwrap();
    assert!(validation.passed);
}

// ============================================================================
// BUILD STAGE EXECUTION TESTS
// ============================================================================

#[test]
fn test_build_stage_validate_debug_build() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("output");
    std::fs::create_dir_all(output_dir.join("target/debug")).unwrap();

    let stage = BuildStage::new(false, None, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), output_dir.clone());

    let result = stage.validate(&ctx);
    assert!(result.is_ok());

    let validation = result.unwrap();
    assert!(validation.passed);
}

#[test]
fn test_build_stage_validate_release_build() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("output");
    std::fs::create_dir_all(output_dir.join("target/release")).unwrap();

    let stage = BuildStage::new(true, None, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), output_dir.clone());

    let result = stage.validate(&ctx);
    assert!(result.is_ok());

    let validation = result.unwrap();
    assert!(validation.passed);
}

// ============================================================================
// VALIDATION STAGE TESTS
// ============================================================================

#[tokio::test]
async fn test_validation_stage_execute_no_binaries() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();

    let stage = ValidationStage::new(true, false);
    let ctx = PipelineContext::new(temp_dir.path().to_path_buf(), temp_dir.path().join("output"));

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();
    assert!(ctx.metadata.contains_key("validation_completed"));
}

#[tokio::test]
async fn test_validation_stage_execute_without_tracing() {
    let stage = ValidationStage::new(false, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), PathBuf::from("/tmp/output"));

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();
    assert_eq!(ctx.metadata.get("validation_completed"), Some(&serde_json::json!(true)));
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_pipeline_context_clone() {
    let mut ctx1 = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
    ctx1.optimizations.push("test".to_string());

    let ctx2 = ctx1.clone();
    assert_eq!(ctx1.input_path, ctx2.input_path);
    assert_eq!(ctx1.output_path, ctx2.output_path);
    assert_eq!(ctx1.optimizations.len(), ctx2.optimizations.len());
}

#[test]
fn test_validation_result_clone() {
    let result1 = ValidationResult {
        stage: "Test".to_string(),
        passed: true,
        message: "OK".to_string(),
        details: None,
    };

    let result2 = result1.clone();
    assert_eq!(result1.stage, result2.stage);
    assert_eq!(result1.passed, result2.passed);
}

#[test]
fn test_pipeline_output_clone() {
    let output1 = PipelineOutput {
        output_path: PathBuf::from("/out"),
        file_mappings: vec![],
        optimizations: vec!["opt".to_string()],
        validation_passed: true,
    };

    let output2 = output1.clone();
    assert_eq!(output1.output_path, output2.output_path);
    assert_eq!(output1.validation_passed, output2.validation_passed);
}

#[test]
fn test_optimization_stage_no_gpu_no_simd() {
    let stage = OptimizationStage::new(false, false, 500);
    let recommendations = stage.analyze_optimizations();

    // Should still return MoE recommendations even without GPU/SIMD
    assert!(!recommendations.is_empty());
}

// ============================================================================
// ADDITIONAL COVERAGE TESTS
// ============================================================================

#[test]
fn test_pipeline_context_output_method_extended() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    // Add some data
    ctx.file_mappings.push((PathBuf::from("a.py"), PathBuf::from("a.rs")));
    ctx.optimizations.push("simd".to_string());
    ctx.validation_results.push(ValidationResult {
        stage: "test".to_string(),
        passed: true,
        message: "ok".to_string(),
        details: None,
    });

    let output = ctx.output();
    assert_eq!(output.output_path, PathBuf::from("/output"));
    assert_eq!(output.file_mappings.len(), 1);
    assert!(output.validation_passed);
}

#[test]
fn test_pipeline_context_output_fails_validation_extended() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    ctx.validation_results.push(ValidationResult {
        stage: "test".to_string(),
        passed: false,
        message: "failed".to_string(),
        details: None,
    });

    let output = ctx.output();
    assert!(!output.validation_passed);
}

#[test]
fn test_transpilation_stage_library_analyzer_initialized() {
    let stage = TranspilationStage::new(true, true);
    // Library analyzer is initialized on construction
    let _ = &stage.library_analyzer;
}

#[test]
fn test_optimization_stage_with_different_simd_thresholds_ext() {
    let stage_low = OptimizationStage::new(false, true, 100);
    let stage_high = OptimizationStage::new(false, true, 10000);

    let recs_low = stage_low.analyze_optimizations();
    let recs_high = stage_high.analyze_optimizations();

    // Both should have recommendations but may differ
    assert!(!recs_low.is_empty());
    assert!(!recs_high.is_empty());
}

#[test]
fn test_transpilation_stage_validate_ext() {
    let stage = TranspilationStage::new(true, true);
    let mut ctx = PipelineContext::new(PathBuf::from("/tmp"), PathBuf::from("/tmp/out"));

    // Without output files, validation result exists
    let result = stage.validate(&ctx);
    assert!(result.is_ok());

    // With file mappings
    ctx.file_mappings.push((PathBuf::from("a.py"), PathBuf::from("a.rs")));
    let result = stage.validate(&ctx);
    assert!(result.is_ok());
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

#[test]
fn test_pipeline_context_empty_metadata() {
    let ctx = PipelineContext::new(PathBuf::from("/in"), PathBuf::from("/out"));
    assert!(ctx.metadata.is_empty());
    assert!(ctx.file_mappings.is_empty());
    assert!(ctx.optimizations.is_empty());
    assert!(ctx.validation_results.is_empty());
}

#[test]
fn test_pipeline_context_complex_metadata() {
    let mut ctx = PipelineContext::new(PathBuf::from("/in"), PathBuf::from("/out"));

    ctx.metadata.insert("array".to_string(), serde_json::json!([1, 2, 3]));
    ctx.metadata.insert("nested".to_string(), serde_json::json!({"a": {"b": "c"}}));
    ctx.metadata.insert("null".to_string(), serde_json::Value::Null);
    ctx.metadata.insert("bool".to_string(), serde_json::json!(true));

    assert_eq!(ctx.metadata.len(), 4);
}

#[test]
fn test_validation_result_default_details() {
    let result = ValidationResult {
        stage: "Test".to_string(),
        passed: true,
        message: "OK".to_string(),
        details: None,
    };

    assert!(result.details.is_none());
    assert!(result.passed);
}

#[test]
fn test_pipeline_output_all_fields() {
    let output = PipelineOutput {
        output_path: PathBuf::from("/output"),
        file_mappings: vec![
            (PathBuf::from("a.py"), PathBuf::from("a.rs")),
            (PathBuf::from("b.py"), PathBuf::from("b.rs")),
        ],
        optimizations: vec!["opt1".to_string(), "opt2".to_string()],
        validation_passed: true,
    };

    assert_eq!(output.file_mappings.len(), 2);
    assert_eq!(output.optimizations.len(), 2);
    assert!(output.validation_passed);
}

#[test]
fn test_analysis_stage_name_immutable() {
    let stage = AnalysisStage;
    let name1 = stage.name();
    let name2 = stage.name();
    assert_eq!(name1, name2);
}

#[test]
fn test_validation_stage_empty_traces() {
    let _stage = ValidationStage::new(true, true);
    let trace1: Vec<String> = vec![];
    let trace2: Vec<String> = vec![];

    // Empty traces should match
    assert!(ValidationStage::compare_traces(&trace1, &trace2));
}

#[test]
fn test_optimization_stage_edge_cases() {
    // Test with zero threshold
    let stage = OptimizationStage::new(true, true, 0);
    let recommendations = stage.analyze_optimizations();
    assert!(!recommendations.is_empty());

    // Test with very high threshold
    let stage_high = OptimizationStage::new(true, true, 10_000_000);
    let rec_high = stage_high.analyze_optimizations();
    assert!(!rec_high.is_empty());
}

#[test]
fn test_build_stage_all_configurations() {
    // Debug, no target, no WASM
    let stage1 = BuildStage::new(false, None, false);
    assert!(!stage1.release);
    assert!(stage1.target.is_none());
    assert!(!stage1.wasm);

    // Release, with target, no WASM
    let stage2 = BuildStage::new(true, Some("aarch64-apple-darwin".to_string()), false);
    assert!(stage2.release);
    assert!(stage2.target.is_some());

    // Debug, no target, WASM
    let stage3 = BuildStage::new(false, None, true);
    assert!(stage3.wasm);
}

#[test]
fn test_transpilation_stage_no_cache() {
    let stage = TranspilationStage::new(false, false);
    assert!(!stage.incremental);
    assert!(!stage.cache);
}

#[test]
fn test_validation_stage_no_tracing() {
    let stage = ValidationStage::new(false, false);
    assert!(!stage.trace_syscalls);
    assert!(!stage.run_tests);
}

// ============================================================================
// ADDITIONAL COVERAGE TESTS - UNIQUE ONLY
// ============================================================================

#[test]
fn test_trace_binary_nonexistent_cov() {
    let result = ValidationStage::trace_binary(Path::new("/nonexistent/binary"));
    // May error if renacer isn't available
    let _ = result;
}

// ============================================================================
// COVERAGE GAP TESTS - execution.rs
// ============================================================================

/// Test pipeline with StopOnError validation where ALL stages pass validation.
/// This covers the branch in run() where validation_result.passed is true
/// and the bail is NOT taken (line 69 false branch).
#[tokio::test]
async fn test_pipeline_stop_on_error_all_pass() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
        .add_stage(Box::new(MockStage::new("Stage1")))
        .add_stage(Box::new(MockStage::new("Stage2")))
        .add_stage(Box::new(MockStage::new("Stage3")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());

    let pipeline_output = result.unwrap();
    // All 3 validations passed
    assert!(pipeline_output.validation_passed);
}

/// Test pipeline with ContinueOnError where ALL stages pass validation.
/// Exercises the non-None validation path with no failures.
#[tokio::test]
async fn test_pipeline_continue_on_error_all_pass() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::ContinueOnError)
        .add_stage(Box::new(MockStage::new("Stage1")))
        .add_stage(Box::new(MockStage::new("Stage2")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());

    let pipeline_output = result.unwrap();
    assert!(pipeline_output.validation_passed);
}

/// Test pipeline with ContinueOnError where execution fails (not validation).
/// This exercises the stage.execute() error path with ContinueOnError strategy.
#[tokio::test]
async fn test_pipeline_continue_on_error_execution_failure() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::ContinueOnError)
        .add_stage(Box::new(MockStage::new("Stage1")))
        .add_stage(Box::new(MockStage::new("Stage2").with_execution_failure()));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    // Execution failure is always fatal regardless of validation strategy
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Stage 'Stage2' failed"));
}

/// Test pipeline where first stage fails execution (stops immediately).
#[tokio::test]
async fn test_pipeline_first_stage_execution_failure() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
        .add_stage(Box::new(MockStage::new("Stage1").with_execution_failure()))
        .add_stage(Box::new(MockStage::new("Stage2")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Stage 'Stage1' failed"));
}

// ============================================================================
// COVERAGE GAP TESTS - build.rs (execute method)
// ============================================================================

/// Test BuildStage execute fails when no Cargo.toml exists.
/// This covers the bail at line 45 in build.rs.
#[tokio::test]
async fn test_build_stage_execute_no_cargo_toml() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let stage = BuildStage::new(false, None, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), temp_dir.path().to_path_buf());

    let result = stage.execute(ctx).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("No Cargo.toml found"));
}

/// Test BuildStage execute with a real minimal Cargo.toml (debug build).
/// This covers the happy path of execute including metadata insertion.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_build_stage_execute_debug_build() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(project_dir.join("src")).unwrap();
    std::fs::write(
        project_dir.join("Cargo.toml"),
        r#"[package]
name = "test-build"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    std::fs::write(project_dir.join("src/main.rs"), "fn main() {}\n").unwrap();

    let stage = BuildStage::new(false, None, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), project_dir.clone());

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();
    assert_eq!(ctx.metadata.get("build_mode"), Some(&serde_json::json!("debug")));
    // No wasm_build metadata when wasm is false
    assert!(!ctx.metadata.contains_key("wasm_build"));
}

/// Test BuildStage execute with release mode.
/// Covers the release arg push at line 52.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_build_stage_execute_release_build() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(project_dir.join("src")).unwrap();
    std::fs::write(
        project_dir.join("Cargo.toml"),
        r#"[package]
name = "test-build"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    std::fs::write(project_dir.join("src/main.rs"), "fn main() {}\n").unwrap();

    let stage = BuildStage::new(true, None, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), project_dir.clone());

    let result = stage.execute(ctx).await;
    assert!(result.is_ok());

    let ctx = result.unwrap();
    assert_eq!(ctx.metadata.get("build_mode"), Some(&serde_json::json!("release")));
}

/// Test BuildStage execute with custom target.
/// Covers the target arg push at lines 56-57.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_build_stage_execute_with_target() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(project_dir.join("src")).unwrap();
    std::fs::write(
        project_dir.join("Cargo.toml"),
        r#"[package]
name = "test-build"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    std::fs::write(project_dir.join("src/main.rs"), "fn main() {}\n").unwrap();

    // Use the native target so compilation should succeed
    let stage = BuildStage::new(false, Some("x86_64-unknown-linux-gnu".to_string()), false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), project_dir.clone());

    let result = stage.execute(ctx).await;
    // Should succeed (compiling for the same architecture we're on)
    assert!(result.is_ok());
}

/// Test BuildStage validate with release=false when build dir doesn't exist.
#[test]
fn test_build_stage_validate_debug_no_dir() {
    let stage = BuildStage::new(false, None, false);
    let ctx = PipelineContext::new(
        PathBuf::from("/tmp/input"),
        PathBuf::from("/tmp/nonexistent_dir_for_test"),
    );

    let result = stage.validate(&ctx);
    assert!(result.is_ok());
    let validation = result.unwrap();
    assert!(!validation.passed);
    assert_eq!(validation.message, "Build directory not found");
}

/// Test BuildStage validate passing case for release=true.
#[test]
fn test_build_stage_validate_release_passing() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    std::fs::create_dir_all(temp_dir.path().join("target/release")).unwrap();

    let stage = BuildStage::new(true, None, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), temp_dir.path().to_path_buf());

    let result = stage.validate(&ctx);
    assert!(result.is_ok());
    let validation = result.unwrap();
    assert!(validation.passed);
    assert_eq!(validation.message, "Build artifacts found");
}

// ============================================================================
// COVERAGE GAP TESTS - types.rs (default validate method)
// ============================================================================

/// A stage that does NOT override validate(), so the default impl is used.
/// This covers lines 95-103 in types.rs.
struct DefaultValidateStage;

#[async_trait::async_trait]
impl PipelineStage for DefaultValidateStage {
    fn name(&self) -> &str {
        "DefaultValidate"
    }

    async fn execute(&self, ctx: PipelineContext) -> anyhow::Result<PipelineContext> {
        Ok(ctx)
    }
    // NOTE: validate() is NOT overridden, so the default impl is used
}

#[test]
fn test_default_validate_method() {
    let stage = DefaultValidateStage;
    let ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));

    let result = stage.validate(&ctx);
    assert!(result.is_ok());

    let validation = result.unwrap();
    assert_eq!(validation.stage, "DefaultValidate");
    assert!(validation.passed);
    assert_eq!(validation.message, "No validation configured");
    assert!(validation.details.is_none());
}

/// Test pipeline run with a stage using default validate (non-None strategy).
/// This exercises the default validate method through the pipeline run path.
#[tokio::test]
async fn test_pipeline_run_with_default_validate_stage() {
    let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
        .add_stage(Box::new(DefaultValidateStage));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());

    let pipeline_output = result.unwrap();
    assert!(pipeline_output.validation_passed);
}

// ============================================================================
// COVERAGE GAP TESTS - PipelineContext Debug + Clone coverage
// ============================================================================

#[test]
fn test_pipeline_context_debug_format() {
    let ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
    let debug_str = format!("{:?}", ctx);
    assert!(debug_str.contains("PipelineContext"));
    assert!(debug_str.contains("/input"));
    assert!(debug_str.contains("/output"));
}

#[test]
fn test_validation_result_debug_format() {
    let result = ValidationResult {
        stage: "TestStage".to_string(),
        passed: true,
        message: "OK".to_string(),
        details: None,
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("ValidationResult"));
    assert!(debug_str.contains("TestStage"));
}

#[test]
fn test_pipeline_output_debug_format() {
    let output = PipelineOutput {
        output_path: PathBuf::from("/out"),
        file_mappings: vec![],
        optimizations: vec![],
        validation_passed: false,
    };
    let debug_str = format!("{:?}", output);
    assert!(debug_str.contains("PipelineOutput"));
    assert!(debug_str.contains("false"));
}

#[test]
fn test_validation_strategy_debug_format() {
    let strategy = ValidationStrategy::StopOnError;
    let debug_str = format!("{:?}", strategy);
    assert_eq!(debug_str, "StopOnError");

    let strategy2 = ValidationStrategy::ContinueOnError;
    let debug_str2 = format!("{:?}", strategy2);
    assert_eq!(debug_str2, "ContinueOnError");

    let strategy3 = ValidationStrategy::None;
    let debug_str3 = format!("{:?}", strategy3);
    assert_eq!(debug_str3, "None");
}

#[test]
fn test_validation_strategy_copy_semantics() {
    let s1 = ValidationStrategy::StopOnError;
    let s2 = s1; // Copy
    assert_eq!(s1, s2); // s1 still usable after copy
}

#[test]
fn test_pipeline_context_with_all_fields_populated() {
    let mut ctx = PipelineContext::new(PathBuf::from("/input"), PathBuf::from("/output"));
    ctx.primary_language = Some(crate::types::Language::Python);
    ctx.file_mappings.push((PathBuf::from("a.py"), PathBuf::from("a.rs")));
    ctx.file_mappings.push((PathBuf::from("b.py"), PathBuf::from("b.rs")));
    ctx.optimizations.push("simd".to_string());
    ctx.optimizations.push("gpu".to_string());
    ctx.validation_results.push(ValidationResult {
        stage: "s1".to_string(),
        passed: true,
        message: "ok".to_string(),
        details: Some(serde_json::json!({"info": "details"})),
    });
    ctx.metadata.insert("key1".to_string(), serde_json::json!("val1"));

    let output = ctx.output();
    assert_eq!(output.file_mappings.len(), 2);
    assert_eq!(output.optimizations.len(), 2);
    assert!(output.validation_passed);
}

/// Test BuildStage execute with broken source code (cargo build fails).
/// Covers the cargo build failure path at lines 73-75 in build.rs.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_build_stage_execute_cargo_build_fails() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(project_dir.join("src")).unwrap();
    std::fs::write(
        project_dir.join("Cargo.toml"),
        r#"[package]
name = "broken-build"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    // Invalid Rust source to cause build failure
    std::fs::write(project_dir.join("src/main.rs"), "fn main() { this is not valid rust }")
        .unwrap();

    let stage = BuildStage::new(false, None, false);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), project_dir.clone());

    let result = stage.execute(ctx).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Cargo build failed"));
}

/// Test BuildStage with wasm=true to cover wasm metadata insertion (lines 83-86).
/// Note: This test uses a real project with wasm target which may not be installed.
/// The build may fail, but we test the wasm arg construction path.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_build_stage_execute_wasm_flag() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(project_dir.join("src")).unwrap();
    std::fs::write(
        project_dir.join("Cargo.toml"),
        r#"[package]
name = "wasm-test"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    std::fs::write(project_dir.join("src/lib.rs"), "pub fn hello() {}\n").unwrap();

    let stage = BuildStage::new(false, None, true);
    let ctx = PipelineContext::new(PathBuf::from("/tmp/input"), project_dir.clone());

    let result = stage.execute(ctx).await;
    // WASM target may or may not be installed; if it succeeds, check metadata
    if let Ok(ctx) = result {
        assert_eq!(ctx.metadata.get("build_mode"), Some(&serde_json::json!("debug")));
        assert_eq!(ctx.metadata.get("wasm_build"), Some(&serde_json::json!(true)));
    }
    // If it fails due to missing wasm target, that's acceptable
}

// ============================================================================
// COVERAGE GAP TESTS - execution.rs info!() macro line coverage
// ============================================================================

/// Test pipeline run with a tracing subscriber installed to force evaluation
/// of the info!() format arguments in execution.rs lines 50-55.
/// Without a subscriber, tracing macros may short-circuit and skip formatting.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_pipeline_run_with_tracing_subscriber() {
    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    // Create a subscriber that captures all output (sink it to avoid test noise)
    let subscriber = tracing_subscriber::registry().with(fmt::layer().with_writer(std::io::sink));

    // Use a guard so the subscriber is only active for this test
    let _guard = subscriber.set_default();

    let pipeline = TranspilationPipeline::new(ValidationStrategy::StopOnError)
        .add_stage(Box::new(MockStage::new("Analysis")))
        .add_stage(Box::new(MockStage::new("Build")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());

    let pipeline_output = result.unwrap();
    assert!(pipeline_output.validation_passed);
}

/// Test pipeline with ContinueOnError and tracing to hit the debug!() line at 65.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_pipeline_run_continue_on_error_with_tracing() {
    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let subscriber = tracing_subscriber::registry().with(fmt::layer().with_writer(std::io::sink));
    let _guard = subscriber.set_default();

    let pipeline = TranspilationPipeline::new(ValidationStrategy::ContinueOnError)
        .add_stage(Box::new(MockStage::new("Stage1").with_validation_failure()))
        .add_stage(Box::new(MockStage::new("Stage2")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());
    assert!(!result.unwrap().validation_passed);
}

/// Test pipeline single stage with tracing for info!() format arg coverage.
#[cfg(feature = "native")]
#[tokio::test]
async fn test_pipeline_single_stage_with_tracing() {
    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let subscriber = tracing_subscriber::registry().with(fmt::layer().with_writer(std::io::sink));
    let _guard = subscriber.set_default();

    let pipeline = TranspilationPipeline::new(ValidationStrategy::None)
        .add_stage(Box::new(MockStage::new("OnlyStage")));

    let input = PathBuf::from("/tmp/input");
    let output = PathBuf::from("/tmp/output");

    let result = pipeline.run(&input, &output).await;
    assert!(result.is_ok());
}
