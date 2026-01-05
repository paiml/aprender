//! Pipeline stage definition and results.
//!
//! Each stage represents a step in the ML inference pipeline
//! with associated ground truth and tolerance.

use super::{Delta, GroundTruth, Tolerance};

/// Status of a stage verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    /// Stage output matches ground truth within tolerance
    Passed,
    /// Stage output diverges from ground truth beyond tolerance
    Failed,
    /// Stage was skipped (e.g., due to prior failure)
    Skipped,
    /// Stage hasn't been evaluated yet
    Pending,
}

impl StageStatus {
    /// Get the Unicode icon for this status.
    #[must_use] 
    pub fn icon(&self) -> &'static str {
        match self {
            StageStatus::Passed => "✓",
            StageStatus::Failed => "✗",
            StageStatus::Skipped => "○",
            StageStatus::Pending => "?",
        }
    }

    /// Get the ANSI color code for this status.
    #[must_use] 
    pub fn color(&self) -> &'static str {
        match self {
            StageStatus::Passed => "\x1b[32m",  // Green
            StageStatus::Failed => "\x1b[31m",  // Red
            StageStatus::Skipped => "\x1b[90m", // Gray
            StageStatus::Pending => "\x1b[33m", // Yellow
        }
    }

    /// Check if this is a passing status.
    #[must_use] 
    pub fn is_passed(&self) -> bool {
        matches!(self, StageStatus::Passed)
    }

    /// Check if this is a failing status.
    #[must_use] 
    pub fn is_failed(&self) -> bool {
        matches!(self, StageStatus::Failed)
    }
}

/// Definition of a pipeline stage.
#[derive(Debug, Clone)]
pub struct Stage {
    /// Stage name (e.g., "mel", "encoder", "decoder")
    name: String,
    /// Expected ground truth for this stage
    ground_truth: Option<GroundTruth>,
    /// Tolerance threshold
    tolerance: Tolerance,
    /// Human-readable description
    description: Option<String>,
}

impl Stage {
    /// Create a new stage with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ground_truth: None,
            tolerance: Tolerance::default(),
            description: None,
        }
    }

    /// Get the stage name.
    #[must_use] 
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the ground truth for this stage.
    #[must_use] 
    pub fn ground_truth(&self) -> Option<&GroundTruth> {
        self.ground_truth.as_ref()
    }

    /// Get the tolerance for this stage.
    #[must_use] 
    pub fn tolerance(&self) -> &Tolerance {
        &self.tolerance
    }

    /// Get the description.
    #[must_use] 
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Verify output against ground truth.
    ///
    /// Returns a `StageResult` indicating pass/fail and computed delta.
    #[must_use] 
    pub fn verify(&self, output: &GroundTruth) -> StageResult {
        match &self.ground_truth {
            Some(gt) => {
                let delta = Delta::compute(output, gt);
                let status = if self.tolerance.is_satisfied(&delta) {
                    StageStatus::Passed
                } else {
                    StageStatus::Failed
                };
                StageResult {
                    name: self.name.clone(),
                    status,
                    delta: Some(delta),
                    our_stats: Some(output.clone()),
                    gt_stats: Some(gt.clone()),
                }
            }
            None => StageResult {
                name: self.name.clone(),
                status: StageStatus::Skipped,
                delta: None,
                our_stats: Some(output.clone()),
                gt_stats: None,
            },
        }
    }
}

/// Builder for constructing stages.
#[derive(Debug)]
pub struct StageBuilder {
    name: String,
    ground_truth: Option<GroundTruth>,
    tolerance: Tolerance,
    description: Option<String>,
}

impl StageBuilder {
    /// Create a new stage builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ground_truth: None,
            tolerance: Tolerance::default(),
            description: None,
        }
    }

    /// Set the ground truth from statistics.
    #[must_use] 
    pub fn ground_truth_stats(mut self, mean: f32, std: f32) -> Self {
        self.ground_truth = Some(GroundTruth::from_stats(mean, std));
        self
    }

    /// Set the ground truth from a `GroundTruth` instance.
    #[must_use] 
    pub fn ground_truth(mut self, gt: GroundTruth) -> Self {
        self.ground_truth = Some(gt);
        self
    }

    /// Set the tolerance.
    #[must_use] 
    pub fn tolerance(mut self, tolerance: Tolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set a description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Build the stage.
    #[must_use] 
    pub fn build(self) -> Stage {
        Stage {
            name: self.name,
            ground_truth: self.ground_truth,
            tolerance: self.tolerance,
            description: self.description,
        }
    }

    /// Build the stage and return control to the pipeline builder.
    ///
    /// This is for fluent API usage with `PipelineBuilder`.
    #[must_use] 
    pub fn build_stage(self) -> Stage {
        self.build()
    }
}

/// Result of verifying a single stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage name
    name: String,
    /// Verification status
    status: StageStatus,
    /// Computed delta (if ground truth was available)
    delta: Option<Delta>,
    /// Our computed statistics
    our_stats: Option<GroundTruth>,
    /// Ground truth statistics
    gt_stats: Option<GroundTruth>,
}

impl StageResult {
    /// Create a passed result.
    pub fn passed(name: impl Into<String>, delta: Delta) -> Self {
        Self {
            name: name.into(),
            status: StageStatus::Passed,
            delta: Some(delta),
            our_stats: None,
            gt_stats: None,
        }
    }

    /// Create a failed result.
    pub fn failed(name: impl Into<String>, delta: Delta) -> Self {
        Self {
            name: name.into(),
            status: StageStatus::Failed,
            delta: Some(delta),
            our_stats: None,
            gt_stats: None,
        }
    }

    /// Create a skipped result.
    pub fn skipped(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: StageStatus::Skipped,
            delta: None,
            our_stats: None,
            gt_stats: None,
        }
    }

    /// Get the stage name.
    #[must_use] 
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the verification status.
    #[must_use] 
    pub fn status(&self) -> StageStatus {
        self.status
    }

    /// Get the delta if available.
    #[must_use] 
    pub fn delta(&self) -> Option<&Delta> {
        self.delta.as_ref()
    }

    /// Get our computed statistics.
    #[must_use] 
    pub fn our_stats(&self) -> Option<&GroundTruth> {
        self.our_stats.as_ref()
    }

    /// Get ground truth statistics.
    #[must_use] 
    pub fn gt_stats(&self) -> Option<&GroundTruth> {
        self.gt_stats.as_ref()
    }

    /// Generate 5-Whys style diagnosis for failures.
    ///
    /// Returns a list of diagnostic messages based on the delta.
    #[must_use] 
    pub fn diagnose(&self) -> Vec<String> {
        let mut diagnosis = Vec::new();

        // Why 1: What failed?
        if let Some(delta) = &self.delta {
            diagnosis.push(format!(
                "Stage '{}' {} with delta {:.1}%",
                self.name,
                if self.status.is_failed() {
                    "failed"
                } else {
                    "passed"
                },
                delta.percent()
            ));

            // Why 2: Sign flip?
            if delta.is_sign_flipped() {
                diagnosis.push("Sign is FLIPPED (positive vs negative)".to_string());
                diagnosis.push("Likely cause: Normalization formula error".to_string());
                diagnosis.push("Check: Log base, subtraction order, sign convention".to_string());
            }

            // Why 3: Magnitude?
            if delta.mean_delta() > 1.0 {
                diagnosis.push(format!(
                    "Mean difference is large: {:.3}",
                    delta.mean_delta()
                ));
                diagnosis.push("Likely cause: Scale factor or offset error".to_string());
            }

            // Why 4: Variance?
            if delta.std_delta() > 0.5 {
                diagnosis.push(format!("Std difference is large: {:.3}", delta.std_delta()));
                diagnosis.push("Likely cause: Distribution shape mismatch".to_string());
            }

            // Why 5: Cosine similarity?
            if let Some(cos) = delta.cosine() {
                if cos < 0.9 {
                    diagnosis.push(format!("Low cosine similarity: {cos:.3}"));
                    diagnosis.push("Likely cause: Structural/layout error in data".to_string());
                }
            }
        }

        diagnosis
    }

    /// Format this result as a table row.
    #[must_use] 
    pub fn format_row(&self) -> String {
        let status_icon = self.status.icon();
        let our_mean = self
            .our_stats
            .as_ref()
            .map_or_else(|| "N/A".to_string(), |s| format!("{:+.4}", s.mean()));
        let our_std = self
            .our_stats
            .as_ref()
            .map_or_else(|| "N/A".to_string(), |s| format!("{:.4}", s.std()));
        let gt_mean = self
            .gt_stats
            .as_ref()
            .map_or_else(|| "N/A".to_string(), |s| format!("{:+.4}", s.mean()));
        let gt_std = self
            .gt_stats
            .as_ref()
            .map_or_else(|| "N/A".to_string(), |s| format!("{:.4}", s.std()));
        let delta_pct = self
            .delta
            .as_ref()
            .map_or_else(|| "N/A".to_string(), |d| format!("{:.1}%", d.percent()));
        let first_char = self.name.chars().next().unwrap_or('?');
        let name = &self.name;

        format!(
            "│ {first_char:^4} │ {name:15} │   {status_icon}    │ μ={our_mean} σ={our_std} │ μ={gt_mean} σ={gt_std} │ {delta_pct:>7} │"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_new() {
        let stage = Stage::new("test");
        assert_eq!(stage.name(), "test");
    }

    #[test]
    fn test_stage_builder() {
        let stage = StageBuilder::new("mel")
            .ground_truth_stats(-0.215, 0.448)
            .tolerance(Tolerance::percent(5.0))
            .description("Mel spectrogram")
            .build();

        assert_eq!(stage.name(), "mel");
        assert!(stage.ground_truth().is_some());
        assert!(stage.description().is_some());
    }

    #[test]
    fn test_stage_verify_pass() {
        let stage = StageBuilder::new("test")
            .ground_truth_stats(0.0, 1.0)
            .tolerance(Tolerance::percent(10.0))
            .build();

        let output = GroundTruth::from_stats(0.05, 1.02);
        let result = stage.verify(&output);
        assert!(result.status().is_passed());
    }

    #[test]
    fn test_stage_verify_fail() {
        let stage = StageBuilder::new("test")
            .ground_truth_stats(-0.215, 0.448)
            .tolerance(Tolerance::percent(5.0))
            .build();

        let output = GroundTruth::from_stats(0.184, 0.5); // Sign flipped!
        let result = stage.verify(&output);
        assert!(result.status().is_failed());
    }

    #[test]
    fn test_diagnosis_sign_flip() {
        let delta = Delta::compute(
            &GroundTruth::from_stats(0.184, 0.5),
            &GroundTruth::from_stats(-0.215, 0.5),
        );
        let result = StageResult::failed("mel", delta);
        let diagnosis = result.diagnose();
        assert!(diagnosis.iter().any(|d| d.contains("FLIPPED")));
    }

    #[test]
    fn test_stage_status_icons() {
        assert_eq!(StageStatus::Passed.icon(), "✓");
        assert_eq!(StageStatus::Failed.icon(), "✗");
        assert_eq!(StageStatus::Skipped.icon(), "○");
        assert_eq!(StageStatus::Pending.icon(), "?");
    }

    #[test]
    fn test_stage_status_colors() {
        assert!(StageStatus::Passed.color().contains("32")); // Green
        assert!(StageStatus::Failed.color().contains("31")); // Red
        assert!(StageStatus::Skipped.color().contains("90")); // Gray
        assert!(StageStatus::Pending.color().contains("33")); // Yellow
    }

    #[test]
    fn test_stage_status_is_passed_is_failed() {
        assert!(StageStatus::Passed.is_passed());
        assert!(!StageStatus::Passed.is_failed());
        assert!(StageStatus::Failed.is_failed());
        assert!(!StageStatus::Failed.is_passed());
        assert!(!StageStatus::Skipped.is_passed());
        assert!(!StageStatus::Skipped.is_failed());
        assert!(!StageStatus::Pending.is_passed());
        assert!(!StageStatus::Pending.is_failed());
    }

    #[test]
    fn test_stage_verify_no_ground_truth() {
        let stage = Stage::new("test"); // No ground truth
        let output = GroundTruth::from_stats(0.0, 1.0);
        let result = stage.verify(&output);
        assert_eq!(result.status(), StageStatus::Skipped);
        assert!(result.delta().is_none());
        assert!(result.our_stats().is_some());
        assert!(result.gt_stats().is_none());
    }

    #[test]
    fn test_stage_builder_with_ground_truth() {
        let gt = GroundTruth::from_slice(&[1.0, 2.0, 3.0]);
        let stage = StageBuilder::new("test").ground_truth(gt).build();
        assert!(stage.ground_truth().is_some());
        assert!(stage.ground_truth().unwrap().has_data());
    }

    #[test]
    fn test_stage_builder_build_stage_alias() {
        let stage = StageBuilder::new("alias_test")
            .ground_truth_stats(1.0, 0.5)
            .build_stage();
        assert_eq!(stage.name(), "alias_test");
    }

    #[test]
    fn test_stage_result_accessors() {
        let delta = Delta::from_percent(5.0);
        let result = StageResult::passed("test_stage", delta);
        assert_eq!(result.name(), "test_stage");
        assert!(result.status().is_passed());
        assert!(result.delta().is_some());
        assert!((result.delta().unwrap().percent() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_stage_result_skipped() {
        let result = StageResult::skipped("skip_me");
        assert_eq!(result.name(), "skip_me");
        assert_eq!(result.status(), StageStatus::Skipped);
        assert!(result.delta().is_none());
    }

    #[test]
    fn test_stage_result_format_row() {
        let delta = Delta::from_percent(5.0);
        let result = StageResult::passed("mel", delta);
        let row = result.format_row();
        assert!(row.contains("mel"));
        assert!(row.contains("✓"));
    }

    #[test]
    fn test_diagnosis_large_mean_delta() {
        let delta = Delta::from_stats(2.0, 0.1);
        let result = StageResult::failed("test", delta);
        let diagnosis = result.diagnose();
        assert!(diagnosis.iter().any(|d| d.contains("Mean difference")));
    }

    #[test]
    fn test_diagnosis_large_std_delta() {
        let delta = Delta::from_stats(0.1, 1.0);
        let result = StageResult::failed("test", delta);
        let diagnosis = result.diagnose();
        assert!(diagnosis.iter().any(|d| d.contains("Std difference")));
    }

    #[test]
    fn test_diagnosis_low_cosine() {
        // Need data for cosine computation
        let our = GroundTruth::from_slice(&[1.0, 0.0, 0.0]);
        let gt = GroundTruth::from_slice(&[0.0, 0.0, 1.0]); // Orthogonal
        let delta = Delta::compute(&our, &gt);
        let result = StageResult {
            name: "test".to_string(),
            status: StageStatus::Failed,
            delta: Some(delta),
            our_stats: None,
            gt_stats: None,
        };
        let diagnosis = result.diagnose();
        assert!(diagnosis.iter().any(|d| d.contains("cosine")));
    }

    #[test]
    fn test_stage_tolerance_accessor() {
        let stage = StageBuilder::new("test")
            .tolerance(Tolerance::percent(10.0))
            .build();
        // Default tolerance check
        let tol = stage.tolerance();
        assert!(tol.is_satisfied(&Delta::from_percent(5.0)));
    }
}
