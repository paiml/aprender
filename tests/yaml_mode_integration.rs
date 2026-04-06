//! YAML Mode Integration Tests
//!
//! Tests for the YAML Mode QA Epic - validates that all YAML example files
//! work with the binary's TrainSpec schema.

use entrenar::config::{load_config, validate_config};
use std::path::Path;

/// Helper to validate that a YAML file can be loaded by the binary
fn validate_yaml_file(filename: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/yaml").join(filename);

    let spec = load_config(&path).unwrap_or_else(|e| panic!("Failed to load {filename}: {e}"));

    validate_config(&spec).unwrap_or_else(|e| panic!("Failed to validate {filename}: {e}"));

    // Basic QA checks
    assert!(!spec.model.path.as_os_str().is_empty(), "{filename}: model path must be specified");
    assert!(
        !spec.data.train.as_os_str().is_empty(),
        "{filename}: data train path must be specified"
    );
    assert!(spec.data.batch_size > 0, "{filename}: batch_size must be > 0");
    assert!(spec.training.epochs > 0, "{filename}: epochs must be > 0");
}

// ============================================================================
// SECTION A: BASIC TRAINING & DATA
// ============================================================================

mod section_a_data {
    use super::*;

    #[test]
    fn yaml_001_mnist_baseline_cpu() {
        validate_yaml_file("mnist_cpu.yaml");
    }

    #[test]
    fn yaml_003_csv_data() {
        validate_yaml_file("csv_data.yaml");
    }

    #[test]
    fn yaml_004_parquet_data() {
        validate_yaml_file("parquet_data.yaml");
    }

    #[test]
    fn yaml_005_deterministic() {
        validate_yaml_file("deterministic.yaml");
    }

    #[test]
    fn yaml_009_multiworker() {
        validate_yaml_file("multiworker.yaml");
    }
}

// ============================================================================
// SECTION B: CITL
// ============================================================================

mod section_b_citl {
    use super::*;

    #[test]
    fn yaml_014_citl_suggest() {
        validate_yaml_file("citl_suggest.yaml");
    }

    #[test]
    fn yaml_017_citl_workspace() {
        validate_yaml_file("citl_workspace.yaml");
    }
}

// ============================================================================
// SECTION C: MODEL ARCHITECTURE
// ============================================================================

mod section_c_architecture {
    use super::*;

    #[test]
    fn yaml_021_llama2_mock() {
        validate_yaml_file("llama2_mock.yaml");
    }

    #[test]
    fn yaml_023_custom_arch() {
        validate_yaml_file("custom_arch.yaml");
    }

    #[test]
    fn yaml_025_dropout() {
        validate_yaml_file("dropout.yaml");
    }

    #[test]
    fn yaml_028_grad_clip() {
        validate_yaml_file("grad_clip.yaml");
    }
}

// ============================================================================
// SECTION D: OPTIMIZATION
// ============================================================================

mod section_d_optimization {
    use super::*;

    #[test]
    fn yaml_031_lora() {
        validate_yaml_file("lora.yaml");
    }

    #[test]
    fn yaml_032_qlora() {
        validate_yaml_file("qlora.yaml");
    }

    #[test]
    fn yaml_034_distillation() {
        validate_yaml_file("distillation.yaml");
    }

    #[test]
    fn yaml_037_grad_accum() {
        validate_yaml_file("grad_accum.yaml");
    }

    #[test]
    fn yaml_040_lr_schedule() {
        validate_yaml_file("lr_schedule.yaml");
    }
}

// ============================================================================
// SECTION E: MONITORING
// ============================================================================

mod section_e_monitoring {
    use super::*;

    #[test]
    fn yaml_044_andon() {
        validate_yaml_file("andon.yaml");
    }

    #[test]
    fn yaml_048_outlier() {
        validate_yaml_file("outlier.yaml");
    }
}

// ============================================================================
// SECTION F: RELIABILITY
// ============================================================================

mod section_f_reliability {
    use super::*;

    #[test]
    fn yaml_051_checkpoint() {
        validate_yaml_file("checkpoint.yaml");
    }

    #[test]
    fn yaml_056_config_validate() {
        validate_yaml_file("config_validate.yaml");
    }

    #[test]
    fn yaml_058_long_run() {
        validate_yaml_file("long_run.yaml");
    }

    #[test]
    fn yaml_060_locked() {
        validate_yaml_file("locked.yaml");
    }
}

// ============================================================================
// SECTION G: INFERENCE
// ============================================================================

mod section_g_inference {
    use super::*;

    #[test]
    fn yaml_062_latency() {
        validate_yaml_file("latency.yaml");
    }

    #[test]
    fn yaml_067_json_output() {
        validate_yaml_file("json_output.yaml");
    }
}

// ============================================================================
// SECTION H: RESEARCH
// ============================================================================

mod section_h_research {
    use super::*;

    #[test]
    fn yaml_073_dp() {
        validate_yaml_file("dp.yaml");
    }

    #[test]
    fn yaml_076_bias() {
        validate_yaml_file("bias.yaml");
    }
}

// ============================================================================
// SECTION I: ECOSYSTEM
// ============================================================================

mod section_i_ecosystem {
    use super::*;

    #[test]
    fn yaml_083_session() {
        validate_yaml_file("session.yaml");
    }
}

// ============================================================================
// SECTION J: EDGE CASES
// ============================================================================

mod section_j_edge_cases {
    use super::*;

    #[test]
    fn yaml_096_soak() {
        validate_yaml_file("soak.yaml");
    }

    #[test]
    fn yaml_098_drift() {
        validate_yaml_file("drift.yaml");
    }

    #[test]
    fn yaml_100_golden_run() {
        validate_yaml_file("release.yaml");
    }
}

// ============================================================================
// VALIDATION TESTS
// ============================================================================

mod validation {
    use super::*;
    use std::fs;

    /// Test that all YAML files in examples/yaml/ can be loaded
    #[test]
    fn all_yaml_files_load() {
        let yaml_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/yaml");

        let entries = fs::read_dir(&yaml_dir).expect("Failed to read examples/yaml directory");

        let mut count = 0;
        for entry in entries {
            let entry = entry.expect("Failed to read directory entry");
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "yaml") {
                let filename = path
                    .file_name()
                    .expect("operation should succeed")
                    .to_str()
                    .expect("operation should succeed");
                validate_yaml_file(filename);
                count += 1;
            }
        }

        assert!(count >= 25, "Expected at least 25 YAML files, found {count}");
    }
}
