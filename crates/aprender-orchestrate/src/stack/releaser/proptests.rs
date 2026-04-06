//! Property-based tests for Release Orchestrator
//!
//! Uses proptest for randomized testing of release orchestration.

use super::orchestrator::ReleaseOrchestrator;
use crate::stack::releaser_types::{format_plan_text, BumpType, ReleaseConfig};
use crate::stack::types::*;
use proptest::prelude::*;
use std::collections::HashMap;

proptest! {
    /// PROPERTY: BumpType::Patch always increments patch by exactly 1
    #[test]
    fn prop_bump_patch_increments_by_one(
        major in 0u64..100,
        minor in 0u64..100,
        patch in 0u64..1000
    ) {
        let version = semver::Version::new(major, minor, patch);
        let bumped = BumpType::Patch.apply(&version);

        prop_assert_eq!(bumped.major, major);
        prop_assert_eq!(bumped.minor, minor);
        prop_assert_eq!(bumped.patch, patch + 1);
    }

    /// PROPERTY: BumpType::Minor increments minor and resets patch to 0
    #[test]
    fn prop_bump_minor_resets_patch(
        major in 0u64..100,
        minor in 0u64..100,
        patch in 0u64..1000
    ) {
        let version = semver::Version::new(major, minor, patch);
        let bumped = BumpType::Minor.apply(&version);

        prop_assert_eq!(bumped.major, major);
        prop_assert_eq!(bumped.minor, minor + 1);
        prop_assert_eq!(bumped.patch, 0);
    }

    /// PROPERTY: BumpType::Major increments major and resets minor/patch to 0
    #[test]
    fn prop_bump_major_resets_all(
        major in 0u64..100,
        minor in 0u64..100,
        patch in 0u64..1000
    ) {
        let version = semver::Version::new(major, minor, patch);
        let bumped = BumpType::Major.apply(&version);

        prop_assert_eq!(bumped.major, major + 1);
        prop_assert_eq!(bumped.minor, 0);
        prop_assert_eq!(bumped.patch, 0);
    }

    /// PROPERTY: Bumped version is always greater than original
    #[test]
    fn prop_bumped_version_always_greater(
        major in 0u64..100,
        minor in 0u64..100,
        patch in 0u64..1000,
        bump_idx in 0usize..3
    ) {
        let version = semver::Version::new(major, minor, patch);
        let bump = match bump_idx {
            0 => BumpType::Patch,
            1 => BumpType::Minor,
            _ => BumpType::Major,
        };

        let bumped = bump.apply(&version);

        prop_assert!(bumped > version, "Bumped {} should be > {}", bumped, version);
    }

    /// PROPERTY: format_plan_text never panics
    #[test]
    fn prop_format_plan_text_never_panics(
        num_releases in 0usize..10,
        dry_run: bool
    ) {
        let releases: Vec<PlannedRelease> = (0..num_releases)
            .map(|i| PlannedRelease {
                crate_name: format!("crate-{}", i),
                current_version: semver::Version::new(0, i as u64, 0),
                new_version: semver::Version::new(0, i as u64, 1),
                dependents: vec![],
                ready: true,
            })
            .collect();

        let plan = ReleasePlan {
            releases,
            dry_run,
            preflight_results: HashMap::new(),
        };

        // Should not panic
        let _text = format_plan_text(&plan);
    }

    /// PROPERTY: ReleaseConfig clone is identical to original
    #[test]
    fn prop_release_config_clone_identical(
        no_verify: bool,
        dry_run: bool,
        publish: bool,
        min_coverage in 0.0f64..100.0,
        fail_on_comply in proptest::bool::ANY
    ) {
        let config = ReleaseConfig {
            bump_type: None,
            no_verify,
            dry_run,
            publish,
            min_coverage,
            lint_command: "cargo clippy".to_string(),
            coverage_command: "cargo tarpaulin".to_string(),
            comply_command: "pmat comply".to_string(),
            fail_on_comply_violations: fail_on_comply,
            // PMAT Quality Gate Integration (use defaults)
            ..Default::default()
        };

        let cloned = config.clone();

        prop_assert_eq!(config.no_verify, cloned.no_verify);
        prop_assert_eq!(config.dry_run, cloned.dry_run);
        prop_assert_eq!(config.publish, cloned.publish);
        prop_assert_eq!(config.fail_on_comply_violations, cloned.fail_on_comply_violations);
        prop_assert!((config.min_coverage - cloned.min_coverage).abs() < f64::EPSILON);
    }
}

// ============================================================================
// PMAT-STACK-GATES: PMAT Quality Gate Integration Tests
// ============================================================================

#[test]
fn test_pmat_gates_config_defaults() {
    // ARRANGE
    let config = ReleaseConfig::default();

    // ASSERT - verify all PMAT gate defaults
    assert_eq!(config.quality_gate_command, "pmat quality-gate");
    assert!(config.fail_on_quality_gate);
    assert_eq!(config.tdg_command, "pmat tdg --format json");
    assert_eq!(config.min_tdg_score, 80.0);
    assert!(config.fail_on_tdg);
    assert_eq!(config.dead_code_command, "pmat analyze dead-code --format json");
    assert!(!config.fail_on_dead_code); // Warning only by default
    assert_eq!(config.complexity_command, "pmat analyze complexity --format json");
    assert_eq!(config.max_complexity, 20);
    assert!(config.fail_on_complexity);
    assert_eq!(config.satd_command, "pmat analyze satd --format json");
    assert_eq!(config.max_satd_items, 10);
    assert!(!config.fail_on_satd); // Warning only by default
    assert_eq!(config.popper_command, "pmat popper-score --format json");
    assert_eq!(config.min_popper_score, 60.0);
    assert!(config.fail_on_popper);
}

#[test]
fn test_pmat_gates_parse_score_from_json() {
    // ARRANGE
    let json = r#"{"score": 85.5, "other": "value"}"#;

    // ACT
    let score = ReleaseOrchestrator::parse_score_from_json(json, "score");

    // ASSERT
    assert_eq!(score, Some(85.5));
}

#[test]
fn test_pmat_gates_parse_score_missing_key() {
    // ARRANGE
    let json = r#"{"other": 100}"#;

    // ACT
    let score = ReleaseOrchestrator::parse_score_from_json(json, "score");

    // ASSERT
    assert_eq!(score, None);
}

#[test]
fn test_pmat_gates_parse_count_from_json() {
    // ARRANGE
    let json = r#"{"count": 42, "total": 100}"#;

    // ACT
    let count = ReleaseOrchestrator::parse_count_from_json(json, "count");

    // ASSERT
    assert_eq!(count, Some(42));
}

#[test]
fn test_pmat_gates_parse_tdg_score() {
    // ARRANGE
    let json = r#"{"tdg_score": 92.5, "files_analyzed": 50}"#;

    // ACT
    let score = ReleaseOrchestrator::parse_score_from_json(json, "tdg_score");

    // ASSERT
    assert_eq!(score, Some(92.5));
}

#[test]
fn test_pmat_gates_parse_popper_score() {
    // ARRANGE
    let json = r#"{"popper_score": 75.0, "category": "A"}"#;

    // ACT
    let score = ReleaseOrchestrator::parse_score_from_json(json, "popper_score");

    // ASSERT
    assert_eq!(score, Some(75.0));
}

#[test]
fn test_pmat_gates_config_custom_thresholds() {
    // ARRANGE/ACT
    let config = ReleaseConfig {
        min_tdg_score: 90.0,
        min_popper_score: 70.0,
        max_complexity: 15,
        max_satd_items: 5,
        fail_on_dead_code: true,
        fail_on_satd: true,
        ..Default::default()
    };

    // ASSERT
    assert_eq!(config.min_tdg_score, 90.0);
    assert_eq!(config.min_popper_score, 70.0);
    assert_eq!(config.max_complexity, 15);
    assert_eq!(config.max_satd_items, 5);
    assert!(config.fail_on_dead_code);
    assert!(config.fail_on_satd);
}

#[test]
fn test_pmat_gates_disabled_checks() {
    // ARRANGE/ACT - disable all PMAT checks
    let config = ReleaseConfig {
        quality_gate_command: String::new(),
        tdg_command: String::new(),
        dead_code_command: String::new(),
        complexity_command: String::new(),
        satd_command: String::new(),
        popper_command: String::new(),
        ..Default::default()
    };

    // ASSERT
    assert!(config.quality_gate_command.is_empty());
    assert!(config.tdg_command.is_empty());
    assert!(config.dead_code_command.is_empty());
    assert!(config.complexity_command.is_empty());
    assert!(config.satd_command.is_empty());
    assert!(config.popper_command.is_empty());
}
