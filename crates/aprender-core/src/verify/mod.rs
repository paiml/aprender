//! Pipeline Verification & Visualization System (APR-VERIFY-001)
//!
//! Deterministic, visual pipeline verification for ML model debugging.
//! Combines Pixar/Weta-style stage gates with Probar-style TUI testing.
//!
//! # Overview
//!
//! This module provides systematic ground truth comparison at each pipeline stage,
//! enabling rapid identification of divergence points in ML inference pipelines.
//!
//! # Example
//!
//! ```ignore
//! use aprender::verify::{Pipeline, Stage, GroundTruth, Tolerance};
//!
//! let pipeline = Pipeline::builder("whisper-tiny")
//!     .stage("mel")
//!         .ground_truth(GroundTruth::from_stats(0.0, -0.215, 0.448))
//!         .tolerance(Tolerance::percent(5.0))
//!         .build_stage()
//!     .build()
//!     .expect("valid pipeline config");
//!
//! let result = pipeline.verify(&input_data);
//! assert!(result.all_passed());
//! ```

mod delta;
mod ground_truth;
mod pipeline;
mod report;
mod stage;
mod tolerance;

pub use delta::{Delta, Metric};
pub use ground_truth::GroundTruth;
pub use pipeline::{Pipeline, PipelineBuilder};
pub use report::VerifyReport;
pub use stage::{Stage, StageBuilder, StageResult, StageStatus};
pub use tolerance::Tolerance;

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // Section A: Pipeline Definition Tests (20 points from QA spec)
    // =============================================================================

    #[test]
    fn a01_pipeline_builder_creates_valid_pipeline() {
        let pipeline = Pipeline::builder("test-pipeline").build().unwrap();
        assert_eq!(pipeline.name(), "test-pipeline");
    }

    #[test]
    fn a02_pipeline_accepts_multiple_stages() {
        let pipeline = Pipeline::builder("multi-stage")
            .stage("stage_a")
            .build_stage()
            .stage("stage_b")
            .build_stage()
            .build()
            .unwrap();
        assert_eq!(pipeline.stages().len(), 2);
    }

    #[test]
    fn a03_pipeline_name_cannot_be_empty() {
        let result = Pipeline::builder("").build();
        assert!(result.is_err());
    }

    #[test]
    fn a04_pipeline_preserves_stage_order() {
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
    fn a05_stage_names_must_be_unique() {
        let result = Pipeline::builder("dupe-check")
            .stage("same")
            .build_stage()
            .stage("same")
            .build_stage()
            .build();
        assert!(result.is_err());
    }

    // =============================================================================
    // Section B: Ground Truth Tests (20 points from QA spec)
    // =============================================================================

    #[test]
    fn b01_ground_truth_from_stats() {
        let gt = GroundTruth::from_stats(-0.215, 0.448);
        assert!((gt.mean() - (-0.215)).abs() < 1e-6);
        assert!((gt.std() - 0.448).abs() < 1e-6);
    }

    #[test]
    fn b02_ground_truth_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gt = GroundTruth::from_slice(&data);
        assert!((gt.mean() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn b03_ground_truth_std_calculation() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let gt = GroundTruth::from_slice(&data);
        // Population std of this data is 2.0
        assert!((gt.std() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn b04_ground_truth_handles_empty_slice() {
        let data: Vec<f32> = vec![];
        let gt = GroundTruth::from_slice(&data);
        assert!(gt.mean().is_nan() || gt.mean() == 0.0);
    }

    #[test]
    fn b05_ground_truth_min_max() {
        let data = vec![-5.0, 0.0, 10.0, 3.0];
        let gt = GroundTruth::from_slice(&data);
        assert!((gt.min() - (-5.0)).abs() < 1e-6);
        assert!((gt.max() - 10.0).abs() < 1e-6);
    }

    // =============================================================================
    // Section C: Delta Computation Tests (20 points from QA spec)
    // =============================================================================

    #[test]
    fn c01_delta_mean_difference() {
        let our = GroundTruth::from_stats(0.184, 0.5);
        let gt = GroundTruth::from_stats(-0.215, 0.5);
        let delta = Delta::compute(&our, &gt);
        assert!((delta.mean_delta() - 0.399).abs() < 1e-3);
    }

    #[test]
    fn c02_delta_std_difference() {
        let our = GroundTruth::from_stats(0.0, 0.6);
        let gt = GroundTruth::from_stats(0.0, 0.4);
        let delta = Delta::compute(&our, &gt);
        assert!((delta.std_delta() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn c03_delta_percent_calculation() {
        let our = GroundTruth::from_stats(0.184, 0.5);
        let gt = GroundTruth::from_stats(-0.215, 0.448);
        let delta = Delta::compute(&our, &gt);
        // Significant divergence expected
        assert!(delta.percent() > 50.0);
    }

    #[test]
    fn c04_delta_sign_flip_detection() {
        let our = GroundTruth::from_stats(0.184, 0.5);
        let gt = GroundTruth::from_stats(-0.215, 0.5);
        let delta = Delta::compute(&our, &gt);
        assert!(delta.is_sign_flipped());
    }

    #[test]
    fn c05_delta_no_sign_flip_when_same_sign() {
        let our = GroundTruth::from_stats(0.1, 0.5);
        let gt = GroundTruth::from_stats(0.2, 0.5);
        let delta = Delta::compute(&our, &gt);
        assert!(!delta.is_sign_flipped());
    }

    #[test]
    fn c06_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let cos = Delta::cosine_similarity(&a, &b);
        assert!((cos - 1.0).abs() < 1e-6);
    }

    #[test]
    fn c07_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let cos = Delta::cosine_similarity(&a, &b);
        assert!((cos - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn c08_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let cos = Delta::cosine_similarity(&a, &b);
        assert!(cos.abs() < 1e-6);
    }

    #[test]
    fn c09_kl_divergence_identical_distributions() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let kl = Delta::kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-6);
    }

    #[test]
    fn c10_kl_divergence_different_distributions() {
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let kl = Delta::kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    // =============================================================================
    // Section D: Tolerance Tests (20 points from QA spec)
    // =============================================================================

    #[test]
    fn d01_tolerance_percent_pass() {
        let tolerance = Tolerance::percent(5.0);
        let delta = Delta::from_percent(3.0);
        assert!(tolerance.is_satisfied(&delta));
    }

    #[test]
    fn d02_tolerance_percent_fail() {
        let tolerance = Tolerance::percent(5.0);
        let delta = Delta::from_percent(10.0);
        assert!(!tolerance.is_satisfied(&delta));
    }

    #[test]
    fn d03_tolerance_stats_both_pass() {
        let tolerance = Tolerance::stats(0.1, 0.1);
        let delta = Delta::from_stats(0.05, 0.05);
        assert!(tolerance.is_satisfied(&delta));
    }

    #[test]
    fn d04_tolerance_stats_mean_fail() {
        let tolerance = Tolerance::stats(0.1, 0.1);
        let delta = Delta::from_stats(0.2, 0.05);
        assert!(!tolerance.is_satisfied(&delta));
    }

    #[test]
    fn d05_default_tolerance_is_five_percent() {
        let tolerance = Tolerance::default();
        let delta_pass = Delta::from_percent(4.9);
        let delta_fail = Delta::from_percent(5.1);
        assert!(tolerance.is_satisfied(&delta_pass));
        assert!(!tolerance.is_satisfied(&delta_fail));
    }

    // =============================================================================
    // Section E: Stage Result Tests (20 points from QA spec)
    // =============================================================================

    #[test]
    fn e01_stage_result_passed() {
        let result = StageResult::passed("mel", Delta::from_percent(2.0));
        assert_eq!(result.status(), StageStatus::Passed);
    }

    #[test]
    fn e02_stage_result_failed() {
        let result = StageResult::failed("mel", Delta::from_percent(90.0));
        assert_eq!(result.status(), StageStatus::Failed);
    }

    #[test]
    fn e03_stage_status_icon_passed() {
        assert_eq!(StageStatus::Passed.icon(), "✓");
    }

    #[test]
    fn e04_stage_status_icon_failed() {
        assert_eq!(StageStatus::Failed.icon(), "✗");
    }

    #[test]
    fn e05_stage_status_icon_skipped() {
        assert_eq!(StageStatus::Skipped.icon(), "○");
    }

    #[test]
    fn e06_stage_diagnosis_sign_flip() {
        let our = GroundTruth::from_stats(0.184, 0.5);
        let gt = GroundTruth::from_stats(-0.215, 0.5);
        let delta = Delta::compute(&our, &gt);
        let result = StageResult::failed("mel", delta);
        let diagnosis = result.diagnose();
        assert!(diagnosis.iter().any(|d| d.contains("Sign is FLIPPED")));
    }

    // =============================================================================
    // Section F: Verify Report Tests
    // =============================================================================

    #[test]
    fn f01_verify_report_all_passed() {
        let mut report = VerifyReport::new("test-pipeline");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::passed("b", Delta::from_percent(2.0)));
        assert!(report.all_passed());
    }

    #[test]
    fn f02_verify_report_has_failures() {
        let mut report = VerifyReport::new("test-pipeline");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::failed("b", Delta::from_percent(90.0)));
        assert!(!report.all_passed());
    }

    #[test]
    fn f03_verify_report_first_failure() {
        let mut report = VerifyReport::new("test-pipeline");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::failed("b", Delta::from_percent(90.0)));
        report.add_result(StageResult::skipped("c"));
        assert_eq!(report.first_failure().unwrap().name(), "b");
    }

    #[test]
    fn f04_verify_report_skips_after_failure() {
        let mut report = VerifyReport::new("test-pipeline");
        report.add_result(StageResult::failed("a", Delta::from_percent(90.0)));
        report.add_result(StageResult::skipped("b"));
        let skipped: Vec<_> = report
            .results()
            .iter()
            .filter(|r| r.status() == StageStatus::Skipped)
            .collect();
        assert_eq!(skipped.len(), 1);
    }
}
