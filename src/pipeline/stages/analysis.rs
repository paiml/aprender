//! Analysis stage - detects languages and dependencies.

use anyhow::Result;

#[cfg(feature = "native")]
use tracing::info;

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

use crate::pipeline::types::{PipelineContext, PipelineStage, ValidationResult};

/// Analysis stage - detects languages and dependencies
pub struct AnalysisStage;

#[async_trait::async_trait]
impl PipelineStage for AnalysisStage {
    fn name(&self) -> &'static str {
        "Analysis"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Analyzing project at {:?}", ctx.input_path);

        let analysis = crate::analyzer::analyze_project(
            &ctx.input_path,
            false, // TDG - skip for pipeline
            true,  // languages
            true,  // dependencies
        )?;

        ctx.primary_language = analysis.primary_language;
        ctx.metadata.insert("total_files".to_string(), serde_json::json!(analysis.total_files));
        ctx.metadata.insert("total_lines".to_string(), serde_json::json!(analysis.total_lines));

        Ok(ctx)
    }

    fn validate(&self, ctx: &PipelineContext) -> Result<ValidationResult> {
        let passed = ctx.primary_language.is_some();
        Ok(ValidationResult {
            stage: self.name().to_string(),
            passed,
            message: if passed {
                format!("Language detected: {:?}", ctx.primary_language)
            } else {
                "Could not detect primary language".to_string()
            },
            details: None,
        })
    }
}
