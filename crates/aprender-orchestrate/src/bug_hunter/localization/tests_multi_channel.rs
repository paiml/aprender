//! Tests for MultiChannelLocalizer (localize and add_coverage).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::bug_hunter::types::{ChannelWeights, LocalizationStrategy};

use super::*;

// =========================================================================
// Coverage gap: MultiChannelLocalizer::localize()
// =========================================================================

#[test]
fn test_localize_sbfl_strategy() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    localizer.spectrum_data.total_failed = 1;
    localizer.spectrum_data.total_passed = 5;
    localizer.spectrum_data.failed_coverage.insert((PathBuf::from("src/lib.rs"), 42), 1);
    localizer.spectrum_data.passed_coverage.insert((PathBuf::from("src/lib.rs"), 42), 1);

    let results = localizer.localize(Path::new("/tmp"));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].line, 42);
    assert!(results[0].final_score > 0.0);
    // SBFL strategy: final_score == spectrum_score
    assert_eq!(results[0].final_score, results[0].spectrum_score);
}

#[test]
fn test_localize_mbfl_strategy() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Mbfl, ChannelWeights::default());
    localizer.mutation_data.mutants.insert((PathBuf::from("src/lib.rs"), 10), (5, 2));

    let results = localizer.localize(Path::new("/tmp"));
    assert_eq!(results.len(), 1);
    // MBFL strategy: final_score == mutation_score
    assert_eq!(results[0].final_score, results[0].mutation_score);
}

#[test]
fn test_localize_causal_strategy() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Causal, ChannelWeights::default());
    localizer.spectrum_data.total_failed = 1;
    localizer.spectrum_data.total_passed = 5;
    localizer.spectrum_data.failed_coverage.insert((PathBuf::from("src/lib.rs"), 20), 1);

    let results = localizer.localize(Path::new("/tmp"));
    assert_eq!(results.len(), 1);
    // Causal uses spectrum as approximation
    assert_eq!(results[0].final_score, results[0].spectrum_score);
}

#[test]
fn test_localize_multichannel_strategy() {
    let mut localizer = MultiChannelLocalizer::new(
        LocalizationStrategy::MultiChannel,
        ChannelWeights {
            spectrum: 0.4,
            mutation: 0.3,
            static_analysis: 0.2,
            semantic: 0.1,
            quality: 0.0,
        },
    );
    localizer.spectrum_data.total_failed = 1;
    localizer.spectrum_data.total_passed = 5;
    let key = (PathBuf::from("src/lib.rs"), 30);
    localizer.spectrum_data.failed_coverage.insert(key.clone(), 1);
    localizer.static_findings.insert(key.clone(), 0.7);

    let results = localizer.localize(Path::new("/tmp"));
    assert_eq!(results.len(), 1);
    // MultiChannel strategy uses compute_final_score with weights
    assert!(results[0].final_score > 0.0);
}

#[test]
fn test_localize_hybrid_strategy() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Hybrid, ChannelWeights::default());
    let key = (PathBuf::from("src/lib.rs"), 5);
    localizer.static_findings.insert(key, 0.5);

    let results = localizer.localize(Path::new("/tmp"));
    assert_eq!(results.len(), 1);
    // Hybrid also uses compute_final_score
    assert!(results[0].final_score >= 0.0);
}

#[test]
fn test_localize_multiple_locations_sorted() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    localizer.spectrum_data.total_failed = 2;
    localizer.spectrum_data.total_passed = 8;
    // High suspiciousness location
    localizer.spectrum_data.failed_coverage.insert((PathBuf::from("src/a.rs"), 10), 2);
    localizer.spectrum_data.passed_coverage.insert((PathBuf::from("src/a.rs"), 10), 0);
    // Low suspiciousness location
    localizer.spectrum_data.failed_coverage.insert((PathBuf::from("src/b.rs"), 20), 1);
    localizer.spectrum_data.passed_coverage.insert((PathBuf::from("src/b.rs"), 20), 7);

    let results = localizer.localize(Path::new("/tmp"));
    assert_eq!(results.len(), 2);
    // Should be sorted descending by final_score
    assert!(results[0].final_score >= results[1].final_score);
}

#[test]
fn test_localize_empty() {
    let localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    let results = localizer.localize(Path::new("/tmp"));
    assert!(results.is_empty());
}

#[test]
fn test_localize_merges_channels() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::MultiChannel, ChannelWeights::default());
    // Same location from spectrum and static
    let key = (PathBuf::from("src/lib.rs"), 42);
    localizer.spectrum_data.total_failed = 1;
    localizer.spectrum_data.total_passed = 5;
    localizer.spectrum_data.failed_coverage.insert(key.clone(), 1);
    localizer.mutation_data.mutants.insert(key.clone(), (3, 1));
    localizer.static_findings.insert(key, 0.8);

    let results = localizer.localize(Path::new("/tmp"));
    // Should merge into a single location
    assert_eq!(results.len(), 1);
    assert!(results[0].spectrum_score > 0.0);
    assert!(results[0].mutation_score > 0.0);
    assert_eq!(results[0].static_score, 0.8);
}

// =========================================================================
// Coverage gap: add_coverage()
// =========================================================================

#[test]
fn test_add_coverage_passing_tests() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    let mut lines = HashMap::new();
    lines.insert((PathBuf::from("src/lib.rs"), 10), 1);
    lines.insert((PathBuf::from("src/lib.rs"), 20), 3);

    let coverage = vec![TestCoverage {
        test_name: "test_pass_1".to_string(),
        passed: true,
        executed_lines: lines,
    }];

    localizer.add_coverage(&coverage);

    assert_eq!(localizer.spectrum_data.total_passed, 1);
    assert_eq!(localizer.spectrum_data.total_failed, 0);
    assert_eq!(
        *localizer.spectrum_data.passed_coverage.get(&(PathBuf::from("src/lib.rs"), 10)).unwrap(),
        1
    );
    assert_eq!(
        *localizer.spectrum_data.passed_coverage.get(&(PathBuf::from("src/lib.rs"), 20)).unwrap(),
        3
    );
}

#[test]
fn test_add_coverage_failing_tests() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    let mut lines = HashMap::new();
    lines.insert((PathBuf::from("src/lib.rs"), 42), 2);

    let coverage = vec![TestCoverage {
        test_name: "test_fail_1".to_string(),
        passed: false,
        executed_lines: lines,
    }];

    localizer.add_coverage(&coverage);

    assert_eq!(localizer.spectrum_data.total_passed, 0);
    assert_eq!(localizer.spectrum_data.total_failed, 1);
    assert_eq!(
        *localizer.spectrum_data.failed_coverage.get(&(PathBuf::from("src/lib.rs"), 42)).unwrap(),
        2
    );
}

#[test]
fn test_add_coverage_mixed_pass_fail() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

    let mut pass_lines = HashMap::new();
    pass_lines.insert((PathBuf::from("src/lib.rs"), 10), 1);

    let mut fail_lines = HashMap::new();
    fail_lines.insert((PathBuf::from("src/lib.rs"), 10), 1);
    fail_lines.insert((PathBuf::from("src/lib.rs"), 42), 1);

    let coverage = vec![
        TestCoverage {
            test_name: "test_pass".to_string(),
            passed: true,
            executed_lines: pass_lines,
        },
        TestCoverage {
            test_name: "test_fail".to_string(),
            passed: false,
            executed_lines: fail_lines,
        },
    ];

    localizer.add_coverage(&coverage);

    assert_eq!(localizer.spectrum_data.total_passed, 1);
    assert_eq!(localizer.spectrum_data.total_failed, 1);
    // Line 10 hit by both passing and failing
    assert_eq!(
        *localizer.spectrum_data.passed_coverage.get(&(PathBuf::from("src/lib.rs"), 10)).unwrap(),
        1
    );
    assert_eq!(
        *localizer.spectrum_data.failed_coverage.get(&(PathBuf::from("src/lib.rs"), 10)).unwrap(),
        1
    );
    // Line 42 only hit by failing
    assert!(!localizer
        .spectrum_data
        .passed_coverage
        .contains_key(&(PathBuf::from("src/lib.rs"), 42)));
    assert_eq!(
        *localizer.spectrum_data.failed_coverage.get(&(PathBuf::from("src/lib.rs"), 42)).unwrap(),
        1
    );
}

#[test]
fn test_add_coverage_empty() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    localizer.add_coverage(&[]);
    assert_eq!(localizer.spectrum_data.total_passed, 0);
    assert_eq!(localizer.spectrum_data.total_failed, 0);
}

#[test]
fn test_add_coverage_accumulates_counts() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

    let mut lines1 = HashMap::new();
    lines1.insert((PathBuf::from("a.rs"), 5), 2);

    let mut lines2 = HashMap::new();
    lines2.insert((PathBuf::from("a.rs"), 5), 3);

    let coverage = vec![
        TestCoverage { test_name: "t1".to_string(), passed: true, executed_lines: lines1 },
        TestCoverage { test_name: "t2".to_string(), passed: true, executed_lines: lines2 },
    ];

    localizer.add_coverage(&coverage);

    assert_eq!(localizer.spectrum_data.total_passed, 2);
    // Counts accumulate: 2 + 3 = 5
    assert_eq!(
        *localizer.spectrum_data.passed_coverage.get(&(PathBuf::from("a.rs"), 5)).unwrap(),
        5
    );
}
