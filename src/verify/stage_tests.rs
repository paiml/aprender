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
