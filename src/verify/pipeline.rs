//! Pipeline definition and verification orchestration.
//!
//! A pipeline consists of multiple stages, each with ground truth
//! and tolerance. Verification proceeds stage-by-stage with
//! Jidoka-style stop-the-line on first failure.

#[cfg(test)]
use super::StageStatus;
use super::{GroundTruth, Stage, StageBuilder, StageResult, Tolerance, VerifyReport};
use std::collections::HashSet;

/// Error type for pipeline construction and verification.
#[derive(Debug, Clone)]
pub enum PipelineError {
    /// Pipeline name cannot be empty
    EmptyName,
    /// Duplicate stage name detected
    DuplicateStageName(String),
    /// No stages defined
    NoStages,
    /// Stage not found
    StageNotFound(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::EmptyName => write!(f, "Pipeline name cannot be empty"),
            PipelineError::DuplicateStageName(name) => {
                write!(f, "Duplicate stage name: '{name}'")
            }
            PipelineError::NoStages => write!(f, "Pipeline has no stages"),
            PipelineError::StageNotFound(name) => write!(f, "Stage not found: '{name}'"),
        }
    }
}

impl std::error::Error for PipelineError {}

/// A verification pipeline consisting of multiple stages.
#[derive(Debug, Clone)]
pub struct Pipeline {
    /// Pipeline name (e.g., "whisper-tiny")
    name: String,
    /// Ordered list of stages
    stages: Vec<Stage>,
    /// Whether to stop on first failure (Jidoka)
    stop_on_failure: bool,
}

impl Pipeline {
    /// Create a new pipeline builder.
    pub fn builder(name: impl Into<String>) -> PipelineBuilder {
        PipelineBuilder::new(name)
    }

    /// Get the pipeline name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get all stages.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Get a stage by name.
    pub fn get_stage(&self, name: &str) -> Option<&Stage> {
        self.stages.iter().find(|s| s.name() == name)
    }

    /// Verify all stages with provided outputs.
    ///
    /// `outputs` is a function that produces the GroundTruth for each stage name.
    /// Verification proceeds in order; if `stop_on_failure` is true, stages
    /// after a failure are marked as skipped.
    pub fn verify<F>(&self, mut outputs: F) -> VerifyReport
    where
        F: FnMut(&str) -> Option<GroundTruth>,
    {
        let mut report = VerifyReport::new(&self.name);
        let mut failed = false;

        for stage in &self.stages {
            if failed && self.stop_on_failure {
                report.add_result(StageResult::skipped(stage.name()));
                continue;
            }

            match outputs(stage.name()) {
                Some(output) => {
                    let result = stage.verify(&output);
                    if result.status().is_failed() {
                        failed = true;
                    }
                    report.add_result(result);
                }
                None => {
                    report.add_result(StageResult::skipped(stage.name()));
                }
            }
        }

        report
    }

    /// Verify a single stage by name.
    pub fn verify_stage(
        &self,
        stage_name: &str,
        output: &GroundTruth,
    ) -> Result<StageResult, PipelineError> {
        let stage = self
            .get_stage(stage_name)
            .ok_or_else(|| PipelineError::StageNotFound(stage_name.to_string()))?;
        Ok(stage.verify(output))
    }
}

/// Builder for constructing pipelines.
#[derive(Debug)]
pub struct PipelineBuilder {
    name: String,
    stages: Vec<Stage>,
    stop_on_failure: bool,
}

impl PipelineBuilder {
    /// Create a new pipeline builder with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
            stop_on_failure: true, // Jidoka by default
        }
    }

    /// Add a stage using the fluent builder API.
    ///
    /// Returns a StageBuilder that will add the stage when build_stage() is called.
    pub fn stage(self, name: impl Into<String>) -> PipelineStageBuilder {
        PipelineStageBuilder {
            pipeline: self,
            stage: StageBuilder::new(name),
        }
    }

    /// Add a pre-built stage.
    pub fn add_stage(mut self, stage: Stage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Disable stop-on-failure behavior.
    pub fn continue_on_failure(mut self) -> Self {
        self.stop_on_failure = false;
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> Result<Pipeline, PipelineError> {
        // Validate name
        if self.name.is_empty() {
            return Err(PipelineError::EmptyName);
        }

        // Check for duplicate stage names
        let mut seen = HashSet::new();
        for stage in &self.stages {
            if !seen.insert(stage.name()) {
                return Err(PipelineError::DuplicateStageName(stage.name().to_string()));
            }
        }

        Ok(Pipeline {
            name: self.name,
            stages: self.stages,
            stop_on_failure: self.stop_on_failure,
        })
    }
}

/// Intermediate builder for adding stages with fluent API.
#[derive(Debug)]
pub struct PipelineStageBuilder {
    pipeline: PipelineBuilder,
    stage: StageBuilder,
}

impl PipelineStageBuilder {
    /// Set ground truth from statistics.
    pub fn ground_truth_stats(mut self, mean: f32, std: f32) -> Self {
        self.stage = self.stage.ground_truth_stats(mean, std);
        self
    }

    /// Set ground truth.
    pub fn ground_truth(mut self, gt: GroundTruth) -> Self {
        self.stage = self.stage.ground_truth(gt);
        self
    }

    /// Set tolerance.
    pub fn tolerance(mut self, tolerance: Tolerance) -> Self {
        self.stage = self.stage.tolerance(tolerance);
        self
    }

    /// Set description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.stage = self.stage.description(desc);
        self
    }

    /// Finish building this stage and return to the pipeline builder.
    pub fn build_stage(self) -> PipelineBuilder {
        let stage = self.stage.build();
        self.pipeline.add_stage(stage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creates_pipeline() {
        let pipeline = Pipeline::builder("test").build().unwrap();
        assert_eq!(pipeline.name(), "test");
    }

    #[test]
    fn test_empty_name_fails() {
        let result = Pipeline::builder("").build();
        assert!(matches!(result, Err(PipelineError::EmptyName)));
    }

    #[test]
    fn test_duplicate_stage_fails() {
        let result = Pipeline::builder("test")
            .stage("same")
            .build_stage()
            .stage("same")
            .build_stage()
            .build();
        assert!(matches!(result, Err(PipelineError::DuplicateStageName(_))));
    }

    #[test]
    fn test_stage_order_preserved() {
        let pipeline = Pipeline::builder("ordered")
            .stage("first")
            .build_stage()
            .stage("second")
            .build_stage()
            .stage("third")
            .build_stage()
            .build()
            .unwrap();

        let names: Vec<_> = pipeline.stages().iter().map(|s| s.name()).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
    }

    #[test]
    fn test_verify_all_pass() {
        let pipeline = Pipeline::builder("test")
            .stage("a")
            .ground_truth_stats(0.0, 1.0)
            .tolerance(Tolerance::percent(10.0))
            .build_stage()
            .stage("b")
            .ground_truth_stats(0.0, 1.0)
            .tolerance(Tolerance::percent(10.0))
            .build_stage()
            .build()
            .unwrap();

        let report = pipeline.verify(|name| {
            Some(GroundTruth::from_stats(
                0.05,
                if name == "a" { 1.02 } else { 0.98 },
            ))
        });

        assert!(report.all_passed());
    }

    #[test]
    fn test_verify_stops_on_failure() {
        let pipeline = Pipeline::builder("test")
            .stage("a")
            .ground_truth_stats(-0.5, 1.0)
            .tolerance(Tolerance::percent(5.0))
            .build_stage()
            .stage("b")
            .ground_truth_stats(0.0, 1.0)
            .build_stage()
            .build()
            .unwrap();

        let report = pipeline.verify(|name| {
            if name == "a" {
                // Sign flipped - will fail
                Some(GroundTruth::from_stats(0.5, 1.0))
            } else {
                Some(GroundTruth::from_stats(0.0, 1.0))
            }
        });

        assert!(!report.all_passed());
        // Stage b should be skipped due to Jidoka
        let results = report.results();
        assert_eq!(results[0].status(), StageStatus::Failed);
        assert_eq!(results[1].status(), StageStatus::Skipped);
    }

    #[test]
    fn test_continue_on_failure() {
        let pipeline = Pipeline::builder("test")
            .stage("a")
            .ground_truth_stats(-0.5, 1.0)
            .tolerance(Tolerance::percent(5.0))
            .build_stage()
            .stage("b")
            .ground_truth_stats(0.0, 1.0)
            .tolerance(Tolerance::percent(50.0))
            .build_stage()
            .continue_on_failure()
            .build()
            .unwrap();

        let report = pipeline.verify(|name| {
            if name == "a" {
                Some(GroundTruth::from_stats(0.5, 1.0))
            } else {
                Some(GroundTruth::from_stats(0.0, 1.0))
            }
        });

        let results = report.results();
        assert_eq!(results[0].status(), StageStatus::Failed);
        assert_eq!(results[1].status(), StageStatus::Passed); // Not skipped
    }
}
