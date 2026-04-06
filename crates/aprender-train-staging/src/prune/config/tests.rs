//! Tests for pruning configuration module.

use super::*;

// =============================================================================
// PruneMethod Tests
// =============================================================================

#[test]
fn test_prune_method_requires_calibration() {
    // TEST_ID: CFG-001
    assert!(
        !PruneMethod::Magnitude.requires_calibration(),
        "CFG-001 FALSIFIED: Magnitude should not require calibration"
    );
    assert!(
        PruneMethod::Wanda.requires_calibration(),
        "CFG-001 FALSIFIED: Wanda should require calibration"
    );
    assert!(
        PruneMethod::SparseGpt.requires_calibration(),
        "CFG-001 FALSIFIED: SparseGPT should require calibration"
    );
    assert!(
        PruneMethod::MinitronDepth.requires_calibration(),
        "CFG-001 FALSIFIED: MinitronDepth should require calibration"
    );
    assert!(
        PruneMethod::MinitronWidth.requires_calibration(),
        "CFG-001 FALSIFIED: MinitronWidth should require calibration"
    );
}

#[test]
fn test_prune_method_display_names() {
    // TEST_ID: CFG-002
    assert_eq!(PruneMethod::Magnitude.display_name(), "Magnitude");
    assert_eq!(PruneMethod::Wanda.display_name(), "Wanda");
    assert_eq!(PruneMethod::SparseGpt.display_name(), "SparseGPT");
    assert_eq!(PruneMethod::MinitronDepth.display_name(), "Minitron (Depth)");
    assert_eq!(PruneMethod::MinitronWidth.display_name(), "Minitron (Width)");
}

#[test]
fn test_prune_method_default() {
    // TEST_ID: CFG-003
    assert_eq!(
        PruneMethod::default(),
        PruneMethod::Magnitude,
        "CFG-003 FALSIFIED: Default method should be Magnitude"
    );
}

// =============================================================================
// SparsityPatternConfig Tests
// =============================================================================

#[test]
fn test_sparsity_pattern_nm_2_4() {
    // TEST_ID: CFG-010
    let pattern = SparsityPatternConfig::nm_2_4();
    match pattern {
        SparsityPatternConfig::NM { n, m } => {
            assert_eq!(n, 2);
            assert_eq!(m, 4);
        }
        _ => panic!("CFG-010 FALSIFIED: Expected NM pattern"),
    }
}

#[test]
fn test_sparsity_pattern_nm_4_8() {
    // TEST_ID: CFG-011
    let pattern = SparsityPatternConfig::nm_4_8();
    match pattern {
        SparsityPatternConfig::NM { n, m } => {
            assert_eq!(n, 4);
            assert_eq!(m, 8);
        }
        _ => panic!("CFG-011 FALSIFIED: Expected NM pattern"),
    }
}

#[test]
fn test_sparsity_pattern_theoretical_sparsity() {
    // TEST_ID: CFG-012
    // 2:4 = 50% sparsity (2 zeros out of 4)
    let nm_2_4 = SparsityPatternConfig::nm_2_4();
    assert!(
        (nm_2_4.theoretical_sparsity() - 0.5).abs() < 1e-6,
        "CFG-012 FALSIFIED: 2:4 should have 50% sparsity"
    );

    // 4:8 = 50% sparsity
    let nm_4_8 = SparsityPatternConfig::nm_4_8();
    assert!(
        (nm_4_8.theoretical_sparsity() - 0.5).abs() < 1e-6,
        "CFG-012 FALSIFIED: 4:8 should have 50% sparsity"
    );

    // Unstructured has variable sparsity (returns 0 as placeholder)
    let unstructured = SparsityPatternConfig::Unstructured;
    assert_eq!(unstructured.theoretical_sparsity(), 0.0);
}

#[test]
fn test_sparsity_pattern_block_theoretical_sparsity() {
    // TEST_ID: CFG-014
    // Block patterns have variable sparsity (returns 0 as placeholder)
    let block = SparsityPatternConfig::Block { height: 4, width: 4 };
    assert_eq!(
        block.theoretical_sparsity(),
        0.0,
        "CFG-014 FALSIFIED: Block should return 0.0 for variable sparsity"
    );
}

#[test]
fn test_sparsity_pattern_row_theoretical_sparsity() {
    // TEST_ID: CFG-015
    // Row patterns have variable sparsity (returns 0 as placeholder)
    let row = SparsityPatternConfig::Row;
    assert_eq!(
        row.theoretical_sparsity(),
        0.0,
        "CFG-015 FALSIFIED: Row should return 0.0 for variable sparsity"
    );
}

#[test]
fn test_sparsity_pattern_column_theoretical_sparsity() {
    // TEST_ID: CFG-016
    // Column patterns have variable sparsity (returns 0 as placeholder)
    let column = SparsityPatternConfig::Column;
    assert_eq!(
        column.theoretical_sparsity(),
        0.0,
        "CFG-016 FALSIFIED: Column should return 0.0 for variable sparsity"
    );
}

#[test]
fn test_sparsity_pattern_default() {
    // TEST_ID: CFG-013
    assert_eq!(
        SparsityPatternConfig::default(),
        SparsityPatternConfig::Unstructured,
        "CFG-013 FALSIFIED: Default pattern should be Unstructured"
    );
}

// =============================================================================
// PruningConfig Tests
// =============================================================================

#[test]
fn test_config_default_values() {
    // TEST_ID: CFG-020
    let config = PruningConfig::default();
    assert_eq!(config.method(), PruneMethod::Magnitude);
    assert!((config.target_sparsity() - 0.5).abs() < 1e-6);
    assert_eq!(*config.pattern(), SparsityPatternConfig::Unstructured);
    assert!(config.fine_tune_after_pruning());
    assert_eq!(config.fine_tune_steps(), 1000);
    assert!((config.fine_tune_lr() - 1e-5).abs() < 1e-10);
    assert!(config.skip_embed_layers());
}

#[test]
fn test_config_builder_pattern() {
    // TEST_ID: CFG-021
    let config = PruningConfig::new()
        .with_method(PruneMethod::Wanda)
        .with_target_sparsity(0.7)
        .with_pattern(SparsityPatternConfig::nm_2_4())
        .with_fine_tune(false)
        .with_fine_tune_steps(500)
        .with_fine_tune_lr(1e-4)
        .with_skip_embed_layers(false);

    assert_eq!(config.method(), PruneMethod::Wanda);
    assert!((config.target_sparsity() - 0.7).abs() < 1e-6);
    match config.pattern() {
        SparsityPatternConfig::NM { n, m } => {
            assert_eq!(*n, 2);
            assert_eq!(*m, 4);
        }
        _ => panic!("CFG-021 FALSIFIED: Expected NM pattern"),
    }
    assert!(!config.fine_tune_after_pruning());
    assert_eq!(config.fine_tune_steps(), 500);
    assert!((config.fine_tune_lr() - 1e-4).abs() < 1e-10);
    assert!(!config.skip_embed_layers());
}

#[test]
fn test_config_target_sparsity_clamped() {
    // TEST_ID: CFG-022
    let config = PruningConfig::new().with_target_sparsity(1.5);
    assert_eq!(
        config.target_sparsity(),
        1.0,
        "CFG-022 FALSIFIED: Sparsity should be clamped to 1.0"
    );

    let config2 = PruningConfig::new().with_target_sparsity(-0.5);
    assert_eq!(
        config2.target_sparsity(),
        0.0,
        "CFG-022 FALSIFIED: Sparsity should be clamped to 0.0"
    );
}

#[test]
fn test_config_requires_calibration() {
    // TEST_ID: CFG-023
    let magnitude_config = PruningConfig::new().with_method(PruneMethod::Magnitude);
    assert!(
        !magnitude_config.requires_calibration(),
        "CFG-023 FALSIFIED: Magnitude config should not require calibration"
    );

    let wanda_config = PruningConfig::new().with_method(PruneMethod::Wanda);
    assert!(
        wanda_config.requires_calibration(),
        "CFG-023 FALSIFIED: Wanda config should require calibration"
    );
}

// =============================================================================
// Validation Tests
// =============================================================================

#[test]
fn test_config_validate_valid() {
    // TEST_ID: CFG-030
    let config = PruningConfig::default();
    assert!(config.validate().is_ok(), "CFG-030 FALSIFIED: Default config should be valid");
}

#[test]
fn test_config_validate_invalid_nm() {
    // TEST_ID: CFG-031
    let config = PruningConfig::new().with_pattern(SparsityPatternConfig::NM {
        n: 5, // Invalid: n >= m
        m: 4,
    });
    assert!(config.validate().is_err(), "CFG-031 FALSIFIED: N >= M should be invalid");
}

#[test]
fn test_config_validate_zero_m() {
    // TEST_ID: CFG-032
    let config = PruningConfig::new().with_pattern(SparsityPatternConfig::NM { n: 0, m: 0 });
    assert!(config.validate().is_err(), "CFG-032 FALSIFIED: M=0 should be invalid");
}

#[test]
fn test_config_validate_zero_block() {
    // TEST_ID: CFG-033
    let config =
        PruningConfig::new().with_pattern(SparsityPatternConfig::Block { height: 0, width: 4 });
    assert!(
        config.validate().is_err(),
        "CFG-033 FALSIFIED: Zero block dimension should be invalid"
    );
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_config_serialize_json() {
    // TEST_ID: CFG-040
    let config = PruningConfig::new().with_method(PruneMethod::Wanda).with_target_sparsity(0.5);

    let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
    assert!(json.contains("wanda"), "CFG-040 FALSIFIED: JSON should contain method name");

    let deserialized: PruningConfig =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(
        deserialized.method(),
        PruneMethod::Wanda,
        "CFG-040 FALSIFIED: Deserialized method should match"
    );
}

#[test]
fn test_config_serialize_yaml() {
    // TEST_ID: CFG-041
    let config = PruningConfig::new()
        .with_method(PruneMethod::SparseGpt)
        .with_pattern(SparsityPatternConfig::nm_2_4());

    let yaml = serde_yaml::to_string(&config).expect("config should be valid");
    assert!(yaml.contains("sparse_gpt"), "CFG-041 FALSIFIED: YAML should contain method name");
}

#[test]
fn test_config_deserialize_from_yaml() {
    // TEST_ID: CFG-042
    let yaml = r"
method: wanda
target_sparsity: 0.5
pattern:
  type: nm
  n: 2
  m: 4
schedule:
  type: one_shot
  step: 1000
fine_tune_after_pruning: true
fine_tune_steps: 500
fine_tune_lr: 0.00001
skip_embed_layers: true
";
    let config: PruningConfig = serde_yaml::from_str(yaml).expect("config should be valid");
    assert_eq!(config.method(), PruneMethod::Wanda);
    assert!((config.target_sparsity() - 0.5).abs() < 1e-6);
    match config.pattern() {
        SparsityPatternConfig::NM { n, m } => {
            assert_eq!(*n, 2);
            assert_eq!(*m, 4);
        }
        _ => panic!("CFG-042 FALSIFIED: Expected NM pattern"),
    }
}

// =============================================================================
// Clone Tests
// =============================================================================

#[test]
fn test_config_clone() {
    // TEST_ID: CFG-050
    let config = PruningConfig::new().with_method(PruneMethod::Wanda).with_target_sparsity(0.7);

    let cloned = config.clone();
    assert_eq!(config.method(), cloned.method(), "CFG-050 FALSIFIED: Cloned method should match");
    assert!(
        (config.target_sparsity() - cloned.target_sparsity()).abs() < 1e-6,
        "CFG-050 FALSIFIED: Cloned target_sparsity should match"
    );
}

// =============================================================================
// Debug Tests
// =============================================================================

#[test]
fn test_config_debug() {
    // TEST_ID: CFG-060
    let config = PruningConfig::new().with_method(PruneMethod::Wanda);
    let debug = format!("{config:?}");
    assert!(debug.contains("Wanda"), "CFG-060 FALSIFIED: Debug should contain method name");
}

#[test]
fn test_pattern_debug() {
    // TEST_ID: CFG-061
    let pattern = SparsityPatternConfig::nm_2_4();
    let debug = format!("{pattern:?}");
    assert!(debug.contains("NM"), "CFG-061 FALSIFIED: Debug should contain pattern type");
}
