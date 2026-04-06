//! Tests for scoring types (SpectrumData, MutationData, ScoredLocation).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::bug_hunter::types::{ChannelWeights, SbflFormula};

use super::*;

#[test]
fn test_sbfl_ochiai() {
    let mut data = SpectrumData { total_failed: 2, total_passed: 8, ..Default::default() };
    data.failed_coverage.insert((PathBuf::from("test.rs"), 10), 2);
    data.passed_coverage.insert((PathBuf::from("test.rs"), 10), 1);

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Ochiai);
    assert!(score > 0.0);
    assert!(score <= 1.0);
}

#[test]
fn test_channel_weights() {
    let weights = ChannelWeights::default();
    let score = weights.combine(0.8, 0.6, 0.4, 0.2, 0.5);
    // 0.30*0.8 + 0.25*0.6 + 0.20*0.4 + 0.15*0.2 + 0.10*0.5
    // = 0.24 + 0.15 + 0.08 + 0.03 + 0.05 = 0.55
    assert!((score - 0.55).abs() < 0.01);
}

#[test]
fn test_scored_location_new() {
    let loc = ScoredLocation::new(PathBuf::from("test.rs"), 42);
    assert_eq!(loc.line, 42);
    assert_eq!(loc.spectrum_score, 0.0);
    assert_eq!(loc.mutation_score, 0.0);
    assert_eq!(loc.static_score, 0.0);
    assert_eq!(loc.semantic_score, 0.0);
    assert_eq!(loc.quality_score, 0.0);
    assert_eq!(loc.final_score, 0.0);
}

#[test]
fn test_scored_location_compute_final_score() {
    let mut loc = ScoredLocation::new(PathBuf::from("test.rs"), 10);
    loc.spectrum_score = 0.8;
    loc.mutation_score = 0.6;
    loc.static_score = 0.4;
    loc.semantic_score = 0.2;

    let weights = ChannelWeights::default();
    loc.compute_final_score(&weights);

    assert!(loc.final_score > 0.0);
}

#[test]
fn test_sbfl_tarantula() {
    let mut data = SpectrumData { total_failed: 2, total_passed: 8, ..Default::default() };
    data.failed_coverage.insert((PathBuf::from("test.rs"), 10), 2);
    data.passed_coverage.insert((PathBuf::from("test.rs"), 10), 2);

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Tarantula);
    assert!(score >= 0.0);
    assert!(score <= 1.0);
}

#[test]
fn test_sbfl_dstar2() {
    let mut data = SpectrumData { total_failed: 2, total_passed: 8, ..Default::default() };
    data.failed_coverage.insert((PathBuf::from("test.rs"), 10), 2);
    data.passed_coverage.insert((PathBuf::from("test.rs"), 10), 1);

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar2);
    assert!(score > 0.0);
}

#[test]
fn test_sbfl_dstar3() {
    let mut data = SpectrumData { total_failed: 2, total_passed: 8, ..Default::default() };
    data.failed_coverage.insert((PathBuf::from("test.rs"), 10), 2);
    data.passed_coverage.insert((PathBuf::from("test.rs"), 10), 1);

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar3);
    assert!(score > 0.0);
}

#[test]
fn test_sbfl_zero_tests() {
    let data = SpectrumData::default();
    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Tarantula);
    assert_eq!(score, 0.0);
}

#[test]
fn test_sbfl_no_coverage() {
    let data = SpectrumData { total_failed: 2, total_passed: 8, ..Default::default() };
    // No coverage entries

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Ochiai);
    assert_eq!(score, 0.0);
}

#[test]
fn test_test_coverage() {
    let cov = TestCoverage {
        test_name: "test_example".to_string(),
        passed: true,
        executed_lines: HashMap::new(),
    };
    assert!(cov.passed);
    assert_eq!(cov.test_name, "test_example");
}

// =========================================================================
// Coverage gap: MutationData edge cases
// =========================================================================

#[test]
fn test_mutation_data_no_mutants() {
    let data = MutationData::default();
    let score = data.compute_score(Path::new("test.rs"), 10);
    assert_eq!(score, 0.0, "No mutants should yield 0.0");
}

#[test]
fn test_mutation_data_zero_total() {
    let mut data = MutationData::default();
    data.mutants.insert((PathBuf::from("test.rs"), 10), (0, 0));
    let score = data.compute_score(Path::new("test.rs"), 10);
    assert_eq!(score, 0.0, "Zero total mutants should yield 0.0");
}

#[test]
fn test_mutation_data_all_killed() {
    let mut data = MutationData::default();
    data.mutants.insert((PathBuf::from("test.rs"), 10), (5, 5));
    let score = data.compute_score(Path::new("test.rs"), 10);
    assert!((score - 1.0).abs() < f64::EPSILON, "All killed should yield 1.0");
}

#[test]
fn test_mutation_data_partial_killed() {
    let mut data = MutationData::default();
    data.mutants.insert((PathBuf::from("test.rs"), 10), (4, 2));
    let score = data.compute_score(Path::new("test.rs"), 10);
    assert!((score - 0.5).abs() < f64::EPSILON, "2/4 killed should yield 0.5");
}

// =========================================================================
// Coverage gap: SBFL edge cases -- DStar2/DStar3 denom=0 with ef>0
// =========================================================================

#[test]
fn test_sbfl_dstar2_denom_zero_ef_positive() {
    let mut data = SpectrumData { total_failed: 1, total_passed: 0, ..Default::default() };
    // ef=1, ep=0, nf=0 -> denom = ep + nf = 0
    data.failed_coverage.insert((PathBuf::from("test.rs"), 10), 1);

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar2);
    assert_eq!(score, f64::MAX, "DStar2 with denom=0 and ef>0 should be MAX");
}

#[test]
fn test_sbfl_dstar3_denom_zero_ef_positive() {
    let mut data = SpectrumData { total_failed: 1, total_passed: 0, ..Default::default() };
    data.failed_coverage.insert((PathBuf::from("test.rs"), 10), 1);

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar3);
    assert_eq!(score, f64::MAX, "DStar3 with denom=0 and ef>0 should be MAX");
}

#[test]
fn test_sbfl_dstar2_denom_zero_ef_zero() {
    let data = SpectrumData { total_failed: 0, total_passed: 0, ..Default::default() };

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar2);
    assert_eq!(score, 0.0, "DStar2 with denom=0 and ef=0 should be 0.0");
}

#[test]
fn test_sbfl_dstar3_denom_zero_ef_zero() {
    let data = SpectrumData { total_failed: 0, total_passed: 0, ..Default::default() };

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::DStar3);
    assert_eq!(score, 0.0, "DStar3 with denom=0 and ef=0 should be 0.0");
}

#[test]
fn test_sbfl_ochiai_denom_zero() {
    let data = SpectrumData { total_failed: 0, total_passed: 0, ..Default::default() };

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Ochiai);
    assert_eq!(score, 0.0, "Ochiai with denom=0 should be 0.0");
}

#[test]
fn test_sbfl_tarantula_only_failed() {
    let mut data = SpectrumData { total_failed: 3, total_passed: 0, ..Default::default() };
    data.failed_coverage.insert((PathBuf::from("test.rs"), 10), 3);

    let score = data.compute_score(Path::new("test.rs"), 10, SbflFormula::Tarantula);
    // fail_ratio = 3/3 = 1.0, pass_ratio = 0.0 -> score = 1.0 / (1.0 + 0.0) = 1.0
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "Tarantula: all failed, score should be 1.0, got {}",
        score
    );
}
