//! Snapshot tests for pruning module (PMAT QA Phase 6)
//!
//! These tests use insta for snapshot testing to verify:
//! - Configuration serialization stability
//! - Schedule behavior across step ranges
//! - Pipeline state transitions
//! - Metrics computation consistency

#[cfg(test)]
mod tests {
    use crate::prune::{
        CalibrationConfig, PruneMethod, PruningConfig, PruningMetrics, PruningSchedule,
        PruningStage, SparsityPatternConfig,
    };

    // =========================================================================
    // Schedule Snapshot Tests
    // =========================================================================

    #[test]
    fn snapshot_oneshot_schedule_serialization() {
        // TEST_ID: SNAP-001
        // Snapshot test for OneShot schedule JSON serialization
        let schedule = PruningSchedule::OneShot { step: 1000 };
        insta::assert_json_snapshot!("oneshot_schedule", schedule);
    }

    #[test]
    fn snapshot_gradual_schedule_serialization() {
        // TEST_ID: SNAP-002
        // Snapshot test for Gradual schedule JSON serialization
        let schedule = PruningSchedule::Gradual {
            start_step: 100,
            end_step: 1000,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 100,
        };
        insta::assert_json_snapshot!("gradual_schedule", schedule);
    }

    #[test]
    fn snapshot_cubic_schedule_serialization() {
        // TEST_ID: SNAP-003
        // Snapshot test for Cubic schedule JSON serialization
        let schedule =
            PruningSchedule::Cubic { start_step: 0, end_step: 10000, final_sparsity: 0.7 };
        insta::assert_json_snapshot!("cubic_schedule", schedule);
    }

    #[test]
    fn snapshot_gradual_sparsity_progression() {
        // TEST_ID: SNAP-004
        // Snapshot sparsity values at key points in a gradual schedule
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 0.5,
            frequency: 10,
        };

        let progression: Vec<(usize, f32)> =
            (0..=100).step_by(10).map(|step| (step, schedule.sparsity_at_step(step))).collect();

        insta::assert_json_snapshot!("gradual_sparsity_progression", progression);
    }

    #[test]
    fn snapshot_cubic_sparsity_progression() {
        // TEST_ID: SNAP-005
        // Snapshot sparsity values at key points in a cubic schedule
        let schedule = PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: 0.5 };

        let progression: Vec<(usize, f32)> =
            (0..=100).step_by(10).map(|step| (step, schedule.sparsity_at_step(step))).collect();

        insta::assert_json_snapshot!("cubic_sparsity_progression", progression);
    }

    // =========================================================================
    // Config Snapshot Tests
    // =========================================================================

    #[test]
    fn snapshot_default_config() {
        // TEST_ID: SNAP-010
        // Snapshot default PruningConfig to detect breaking changes
        let config = PruningConfig::default();
        insta::assert_json_snapshot!("default_pruning_config", config);
    }

    #[test]
    fn snapshot_wanda_config() {
        // TEST_ID: SNAP-011
        // Snapshot Wanda configuration
        let config = PruningConfig::new()
            .with_method(PruneMethod::Wanda)
            .with_target_sparsity(0.5)
            .with_pattern(SparsityPatternConfig::nm_2_4())
            .with_schedule(PruningSchedule::Gradual {
                start_step: 1000,
                end_step: 5000,
                initial_sparsity: 0.0,
                final_sparsity: 0.5,
                frequency: 100,
            });
        insta::assert_json_snapshot!("wanda_config", config);
    }

    #[test]
    fn snapshot_sparsegpt_config() {
        // TEST_ID: SNAP-012
        // Snapshot SparseGPT configuration
        let config = PruningConfig::new()
            .with_method(PruneMethod::SparseGpt)
            .with_target_sparsity(0.7)
            .with_pattern(SparsityPatternConfig::Unstructured)
            .with_fine_tune(true)
            .with_fine_tune_steps(2000);
        insta::assert_json_snapshot!("sparsegpt_config", config);
    }

    #[test]
    fn snapshot_all_sparsity_patterns() {
        // TEST_ID: SNAP-013
        // Snapshot all sparsity pattern configurations
        let patterns = vec![
            ("unstructured", SparsityPatternConfig::Unstructured),
            ("nm_2_4", SparsityPatternConfig::nm_2_4()),
            ("nm_4_8", SparsityPatternConfig::nm_4_8()),
            ("block_4x4", SparsityPatternConfig::Block { height: 4, width: 4 }),
            ("row", SparsityPatternConfig::Row),
            ("column", SparsityPatternConfig::Column),
        ];
        insta::assert_json_snapshot!("all_sparsity_patterns", patterns);
    }

    #[test]
    fn snapshot_all_prune_methods() {
        // TEST_ID: SNAP-014
        // Snapshot all pruning methods with their calibration requirements
        let methods: Vec<(&str, PruneMethod, bool)> = vec![
            ("magnitude", PruneMethod::Magnitude, false),
            ("wanda", PruneMethod::Wanda, true),
            ("sparsegpt", PruneMethod::SparseGpt, true),
            ("minitron_depth", PruneMethod::MinitronDepth, true),
            ("minitron_width", PruneMethod::MinitronWidth, true),
        ];

        let method_info: Vec<_> = methods
            .iter()
            .map(|(name, method, requires_cal)| {
                serde_json::json!({
                    "name": name,
                    "display_name": method.display_name(),
                    "requires_calibration": requires_cal,
                })
            })
            .collect();

        insta::assert_json_snapshot!("all_prune_methods", method_info);
    }

    // =========================================================================
    // Pipeline Snapshot Tests
    // =========================================================================

    #[test]
    fn snapshot_pipeline_stages() {
        // TEST_ID: SNAP-020
        // Snapshot all pipeline stages
        let stages = vec![
            PruningStage::Idle,
            PruningStage::Calibrating,
            PruningStage::ComputingImportance,
            PruningStage::Pruning,
            PruningStage::FineTuning,
            PruningStage::Evaluating,
            PruningStage::Exporting,
            PruningStage::Complete,
            PruningStage::Failed,
        ];

        let stage_info: Vec<_> = stages
            .iter()
            .map(|stage| {
                serde_json::json!({
                    "name": stage.display_name(),
                    "is_active": stage.is_active(),
                    "is_terminal": stage.is_terminal(),
                })
            })
            .collect();

        insta::assert_json_snapshot!("pipeline_stages", stage_info);
    }

    #[test]
    fn snapshot_initial_metrics() {
        // TEST_ID: SNAP-021
        // Snapshot initial metrics state
        let metrics = PruningMetrics::new(0.5);
        insta::assert_json_snapshot!("initial_metrics", metrics);
    }

    #[test]
    fn snapshot_metrics_after_sparsity_update() {
        // TEST_ID: SNAP-022
        // Snapshot metrics after sparsity updates
        let mut metrics = PruningMetrics::new(0.5);
        // update_sparsity(pruned, total)
        metrics.update_sparsity(250, 1000);
        // Add per-layer sparsity info
        metrics.add_layer_sparsity("layer1", 0.25);
        metrics.add_layer_sparsity("layer2", 0.25);
        metrics.add_layer_sparsity("layer3", 0.30);
        insta::assert_json_snapshot!("metrics_with_sparsity", metrics);
    }

    // =========================================================================
    // Calibration Config Snapshot Tests
    // =========================================================================

    #[test]
    fn snapshot_calibration_config_default() {
        // TEST_ID: SNAP-030
        // Snapshot default calibration configuration
        let config = CalibrationConfig::default();
        insta::assert_json_snapshot!("calibration_config_default", config);
    }

    #[test]
    fn snapshot_calibration_config_custom() {
        // TEST_ID: SNAP-031
        // Snapshot custom calibration configuration
        let config = CalibrationConfig::new()
            .with_num_samples(1024)
            .with_batch_size(16)
            .with_sequence_length(4096);
        insta::assert_json_snapshot!("calibration_config_custom", config);
    }

    // =========================================================================
    // Schedule Validation Snapshot Tests
    // =========================================================================

    #[test]
    fn snapshot_schedule_validation_errors() {
        // TEST_ID: SNAP-040
        // Snapshot validation error messages for invalid schedules
        let invalid_schedules = [
            (
                "gradual_end_before_start",
                PruningSchedule::Gradual {
                    start_step: 100,
                    end_step: 50,
                    initial_sparsity: 0.0,
                    final_sparsity: 0.5,
                    frequency: 10,
                },
            ),
            (
                "gradual_negative_initial",
                PruningSchedule::Gradual {
                    start_step: 0,
                    end_step: 100,
                    initial_sparsity: -0.1,
                    final_sparsity: 0.5,
                    frequency: 10,
                },
            ),
            (
                "gradual_final_over_one",
                PruningSchedule::Gradual {
                    start_step: 0,
                    end_step: 100,
                    initial_sparsity: 0.0,
                    final_sparsity: 1.5,
                    frequency: 10,
                },
            ),
            (
                "cubic_end_before_start",
                PruningSchedule::Cubic { start_step: 100, end_step: 50, final_sparsity: 0.5 },
            ),
            (
                "cubic_negative_sparsity",
                PruningSchedule::Cubic { start_step: 0, end_step: 100, final_sparsity: -0.1 },
            ),
        ];

        let errors: Vec<_> = invalid_schedules
            .iter()
            .map(|(name, schedule)| {
                serde_json::json!({
                    "name": name,
                    "error": schedule.validate().err(),
                })
            })
            .collect();

        insta::assert_json_snapshot!("schedule_validation_errors", errors);
    }

    // =========================================================================
    // YAML Serialization Snapshot Tests
    // =========================================================================

    #[test]
    fn snapshot_config_yaml_format() {
        // TEST_ID: SNAP-050
        // Snapshot YAML format for config (used in user-facing config files)
        let config = PruningConfig::new()
            .with_method(PruneMethod::Wanda)
            .with_target_sparsity(0.5)
            .with_pattern(SparsityPatternConfig::nm_2_4())
            .with_schedule(PruningSchedule::Gradual {
                start_step: 1000,
                end_step: 5000,
                initial_sparsity: 0.0,
                final_sparsity: 0.5,
                frequency: 100,
            });

        insta::assert_yaml_snapshot!("config_yaml_format", config);
    }
}
