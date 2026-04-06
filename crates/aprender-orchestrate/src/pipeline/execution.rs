//! Pipeline execution engine.

use anyhow::{Context as AnyhowContext, Result};
use std::path::Path;

#[cfg(feature = "native")]
use tracing::{debug, info};

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! debug {
    ($($arg:tt)*) => {{}};
}

use super::types::{PipelineContext, PipelineOutput, PipelineStage, ValidationStrategy};

/// Main transpilation pipeline
pub struct TranspilationPipeline {
    pub(crate) stages: Vec<Box<dyn PipelineStage>>,
    pub(crate) validation: ValidationStrategy,
}

impl TranspilationPipeline {
    pub fn new(validation: ValidationStrategy) -> Self {
        Self { stages: Vec::new(), validation }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Run the complete pipeline
    #[allow(clippy::cognitive_complexity)]
    pub async fn run(&self, input: &Path, output: &Path) -> Result<PipelineOutput> {
        info!("Starting pipeline with {} stages", self.stages.len());

        let mut ctx = PipelineContext::new(input.to_path_buf(), output.to_path_buf());

        for (idx, stage) in self.stages.iter().enumerate() {
            info!("Running stage {}/{}: {}", idx + 1, self.stages.len(), stage.name());

            // Execute stage
            ctx = stage
                .execute(ctx)
                .await
                .with_context(|| format!("Stage '{}' failed", stage.name()))?;

            // Validate if strategy requires it
            if self.validation != ValidationStrategy::None {
                debug!("Validating stage: {}", stage.name());
                let validation_result = stage.validate(&ctx)?;
                ctx.validation_results.push(validation_result.clone());

                if !validation_result.passed && self.validation == ValidationStrategy::StopOnError {
                    anyhow::bail!(
                        "Validation failed for stage '{}': {}",
                        stage.name(),
                        validation_result.message
                    );
                }
            }
        }

        info!("Pipeline completed successfully");
        Ok(ctx.output())
    }
}
