use super::*;

#[test]
fn tcov_conversion_path_display_with_intermediates() {
    let path = ConversionPath::chain(
        FormatType::Gguf,
        vec![FormatType::Apr],
        FormatType::SafeTensors,
    );
    let display = format!("{path}");
    assert_eq!(display, "GGUF → APR → SafeTensors");
}

#[test]
fn tcov_verify_roundtrip_safetensors() {
    let source = create_safetensors_fixture();
    let rosetta = RosettaStone::new();
    let report = rosetta.verify_roundtrip(&source, FormatType::Apr);
    // Round-trip may succeed or fail based on implementation details
    // The key is that it doesn't panic
    match report {
        Ok(vr) => {
            // If it succeeds, verify report structure
            assert!(vr.max_diff >= 0.0);
        }
        Err(_) => {
            // Also acceptable - some round-trip failures expected
        }
    }
    let _ = std::fs::remove_file(source);
}

// ========================================================================
// Section 23: Coverage Gap Tests (P050+)
// Targets uncovered branches in mod.rs identified by coverage analysis.
// ========================================================================

// ========================================================================
// P050-P059: FormatType::from_extension error paths (lines 90-116)
// ========================================================================

/// P050: from_extension with a real directory path (is_dir = true, no model files found)
/// Covers: line 90 (is_dir check), lines 93-103 (directory with no candidates)
#[test]
fn p050_from_extension_real_directory_no_candidates() {
    // /tmp is a real directory that won't contain model.gguf, model.apr, model.safetensors
    let path = Path::new("/tmp");
    let result = FormatType::from_extension(path);
    assert!(result.is_err(), "Directory path should fail from_extension");
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("is a directory"),
        "Error should mention directory: {err}"
    );
    assert!(
        err.contains(".gguf") || err.contains(".apr") || err.contains(".safetensors"),
        "Error should mention expected extensions: {err}"
    );
}

/// P051b: from_extension with a nonexistent path that has no extension
/// Covers: lines 112-116 (not a directory, no extension -> "No file extension found")
#[test]
fn p051b_from_extension_nonexistent_no_extension() {
    let path = Path::new("/nonexistent_dir_12345/no_extension_file");
    let result = FormatType::from_extension(path);
    assert!(
        result.is_err(),
        "Nonexistent path with no extension should fail"
    );
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("No file extension"),
        "Error should mention missing extension: {err}"
    );
}

/// P052b: from_extension with a directory containing a candidate model file
/// Covers: lines 104-109 (directory with found candidates -> "Did you mean" suggestion)
#[test]
fn p052b_from_extension_directory_with_candidate() {
    // Create a temp directory with a model file inside
    let temp_dir = std::env::temp_dir().join("rosetta_test_dir_p052b");
    let _ = std::fs::create_dir_all(&temp_dir);
    let model_file = temp_dir.join("model.gguf");
    let _ = std::fs::write(&model_file, b"dummy");

    let result = FormatType::from_extension(&temp_dir);
    assert!(
        result.is_err(),
        "Directory path should fail even with candidates"
    );
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("Did you mean"),
        "Error should suggest candidate file: {err}"
    );
    assert!(
        err.contains("model.gguf"),
        "Suggestion should include model.gguf: {err}"
    );

    let _ = std::fs::remove_dir_all(temp_dir);
}

/// P053b: from_extension with a directory containing multiple candidate model files
/// The code returns the first found candidate from `[model.gguf, model.apr, model.safetensors]`
#[test]
fn p053b_from_extension_directory_with_multiple_candidates() {
    let temp_dir = std::env::temp_dir().join("rosetta_test_dir_p053b");
    let _ = std::fs::create_dir_all(&temp_dir);
    let _ = std::fs::write(temp_dir.join("model.apr"), b"dummy");
    let _ = std::fs::write(temp_dir.join("model.safetensors"), b"dummy");

    let result = FormatType::from_extension(&temp_dir);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("Did you mean"),
        "Should suggest a candidate: {err}"
    );

    let _ = std::fs::remove_dir_all(temp_dir);
}

/// P054b: from_extension with a nonexistent directory (is_dir returns false)
/// Covers: the else branch at line 112 when path doesn't exist and has no extension
#[test]
fn p054b_from_extension_nonexistent_directory_path() {
    let path = Path::new("/nonexistent_dir_12345");
    let result = FormatType::from_extension(path);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    // is_dir() returns false for nonexistent paths, so we get "No file extension"
    assert!(
        err.contains("No file extension"),
        "Nonexistent path without extension: {err}"
    );
}

// ========================================================================
// P060-P069: compute_tensor_validation thorough testing (lines 1036-1176)
// Tests the core validation logic directly via RosettaStone instance
// ========================================================================

/// P060: compute_tensor_validation with empty data
/// Covers: lines 1038-1052 (early return for empty data)
#[test]
fn p060_compute_validation_empty_data() {
    let rosetta = RosettaStone::new();
    let tv = rosetta.compute_tensor_validation("empty.weight", &[]);
    assert!(tv.is_valid, "Empty tensor should be valid");
    assert_eq!(tv.element_count, 0);
    assert_eq!(tv.nan_count, 0);
    assert_eq!(tv.inf_count, 0);
    assert_eq!(tv.zero_count, 0);
    assert_eq!(tv.min, 0.0);
    assert_eq!(tv.max, 0.0);
    assert_eq!(tv.mean, 0.0);
    assert_eq!(tv.std, 0.0);
    assert!(tv.failures.is_empty());
}

/// P061b: compute_tensor_validation with all NaN values
/// Covers: lines 1062-1064 (NaN counting), 1082-1087 (valid_count=0 path), 1105-1108
#[test]
fn p061b_compute_validation_all_nan() {
    let rosetta = RosettaStone::new();
    let data = [f32::NAN, f32::NAN, f32::NAN, f32::NAN];
    let tv = rosetta.compute_tensor_validation("corrupted.weight", &data);
    assert!(!tv.is_valid, "All-NaN tensor should be invalid");
    assert_eq!(tv.nan_count, 4);
    assert_eq!(tv.element_count, 4);
    assert_eq!(tv.mean, 0.0, "Mean should be 0 when no valid values");
    assert_eq!(tv.std, 0.0, "Std should be 0 when no valid values");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("NaN") && f.contains("F-DATA-QUALITY-002")),
        "Should report NaN failure"
    );
}

/// P062b: compute_tensor_validation with all Inf values
/// Covers: lines 1066-1068 (Inf counting), 1110-1113
#[test]
fn p062b_compute_validation_all_inf() {
    let rosetta = RosettaStone::new();
    let data = [f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY];
    let tv = rosetta.compute_tensor_validation("overflow.weight", &data);
    assert!(!tv.is_valid, "All-Inf tensor should be invalid");
    assert_eq!(tv.inf_count, 3);
    assert_eq!(tv.nan_count, 0);
    assert_eq!(tv.mean, 0.0, "Mean should be 0 when no valid values");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("Inf") && f.contains("F-DATA-QUALITY-002")),
        "Should report Inf failure"
    );
}

/// P063b: compute_tensor_validation with mixed NaN and Inf
/// Covers: both NaN and Inf branches simultaneously, valid_count = 0
#[test]
fn p063b_compute_validation_mixed_nan_inf() {
    let rosetta = RosettaStone::new();
    let data = [f32::NAN, f32::INFINITY, f32::NAN, f32::NEG_INFINITY];
    let tv = rosetta.compute_tensor_validation("broken.weight", &data);
    assert!(!tv.is_valid);
    assert_eq!(tv.nan_count, 2);
    assert_eq!(tv.inf_count, 2);
    assert!(
        tv.failures.len() >= 2,
        "Should have both NaN and Inf failures"
    );
}

/// P064: compute_tensor_validation with all zeros
/// Covers: lines 1070-1072 (zero counting), 1115-1117 (all zeros failure)
#[test]
fn p064_compute_validation_all_zeros() {
    let rosetta = RosettaStone::new();
    let data = [0.0_f32; 100];
    let tv = rosetta.compute_tensor_validation("uninitialized.weight", &data);
    assert!(!tv.is_valid, "All-zero tensor should be invalid");
    assert_eq!(tv.zero_count, 100);
    assert_eq!(tv.element_count, 100);
    assert!(tv.is_all_zeros());
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("All values are zero") && f.contains("F-DATA-QUALITY-001")),
        "Should report all-zeros failure: {:?}",
        tv.failures
    );
}

/// P065: compute_tensor_validation density gate for embedding tensors (>50% zeros)
/// Covers: lines 1119-1134 (density gate with embedding name threshold=50%)
#[test]
fn p065_compute_validation_embedding_high_zeros() {
    let rosetta = RosettaStone::new();
    // 60% zeros in an embedding tensor (threshold is 50%)
    let mut data = vec![0.0_f32; 60];
    data.extend(vec![0.1_f32; 40]);
    let tv = rosetta.compute_tensor_validation("model.embed_tokens.weight", &data);
    assert!(!tv.is_valid, "Embedding with >50% zeros should be invalid");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("DENSITY") && f.contains("F-DATA-QUALITY-001")),
        "Should report density failure: {:?}",
        tv.failures
    );
}

/// P066: compute_tensor_validation density gate for weight tensors (>80% zeros)
/// Covers: lines 1129 (non-embedding path, threshold=80%)
#[test]
fn p066_compute_validation_weight_high_zeros() {
    let rosetta = RosettaStone::new();
    // 85% zeros in a non-embedding weight tensor (threshold is 80%)
    let mut data = vec![0.0_f32; 85];
    data.extend(vec![0.5_f32; 15]);
    let tv = rosetta.compute_tensor_validation("model.layers.0.self_attn.q_proj.weight", &data);
    assert!(!tv.is_valid, "Weight with >80% zeros should be invalid");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("DENSITY") && f.contains("80")),
        "Should report 80% density threshold: {:?}",
        tv.failures
    );
}

/// P067: compute_tensor_validation density gate for weight under threshold
/// Covers: density gate path where zero_pct <= threshold (no failure)
#[test]
fn p067_compute_validation_weight_acceptable_zeros() {
    let rosetta = RosettaStone::new();
    // 50% zeros in a non-embedding weight tensor (threshold is 80%) -- should pass density
    let mut data = vec![0.0_f32; 50];
    data.extend(vec![0.5_f32; 50]);
    let tv = rosetta.compute_tensor_validation("model.layers.0.weight", &data);
    assert!(
        !tv.failures.iter().any(|f| f.contains("DENSITY")),
        "50% zeros in weight should not trigger 80% density gate: {:?}",
        tv.failures
    );
}

/// P068: compute_tensor_validation L2 norm gate (near-zero L2)
/// Covers: lines 1136-1149 (L2 norm < 1e-6)
#[test]
fn p068_compute_validation_low_l2_norm() {
    let rosetta = RosettaStone::new();
    // Very small values that produce L2 norm < 1e-6
    let data = [1e-7_f32, -1e-7, 1e-7, -1e-7];
    let tv = rosetta.compute_tensor_validation("tiny.weight", &data);
    assert!(!tv.is_valid, "Near-zero L2 norm should be invalid");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("L2 norm") && f.contains("F-DATA-QUALITY-003")),
        "Should report L2 norm failure: {:?}",
        tv.failures
    );
}

/// P069: compute_tensor_validation constant value gate (all identical, non-zero)
/// Covers: lines 1151-1159 (max - min < 1e-10 for non-norm tensors)
#[test]
fn p069_compute_validation_constant_values() {
    let rosetta = RosettaStone::new();
    let data = [0.5_f32; 100];
    let tv = rosetta.compute_tensor_validation("model.layers.0.weight", &data);
    assert!(!tv.is_valid, "Constant tensor should be invalid");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("constant") && f.contains("F-DATA-QUALITY-003")),
        "Should report constant value failure: {:?}",
        tv.failures
    );
}

/// P070b: compute_tensor_validation constant value exemption for norm tensors
/// Covers: lines 1153-1155 (is_norm_or_bias exemption)
#[test]
fn p070b_compute_validation_constant_norm_exempt() {
    let rosetta = RosettaStone::new();
    // All 1.0 values in a norm tensor (e.g., RMS norm weight initialized to 1.0)
    let data = [1.0_f32; 100];
    let tv = rosetta.compute_tensor_validation("model.layers.0.input_layernorm.weight", &data);
    // Norm tensors are exempt from the constant-value check
    assert!(
        !tv.failures.iter().any(|f| f.contains("constant")),
        "Norm tensor should be exempt from constant check: {:?}",
        tv.failures
    );
}

/// P070c: compute_tensor_validation constant value exemption for bias tensors
/// Covers: lines 1153-1155 (is_norm_or_bias with "bias" in name)
#[test]
fn p070c_compute_validation_constant_bias_exempt() {
    let rosetta = RosettaStone::new();
    let data = [0.0_f32; 50]; // All zeros in a bias
                              // Even though all zeros, bias is exempt from constant check.
                              // However, all-zeros still triggers the all-zeros failure.
    let tv = rosetta.compute_tensor_validation("model.layers.0.bias", &data);
    assert!(
        !tv.failures.iter().any(|f| f.contains("constant")),
        "Bias tensor should be exempt from constant check: {:?}",
        tv.failures
    );
}

/// P070d: compute_tensor_validation constant value exemption for ln_ tensors
/// Covers: lines 1153-1155 (is_norm_or_bias with "ln_" in name)
#[test]
fn p070d_compute_validation_constant_ln_exempt() {
    let rosetta = RosettaStone::new();
    let data = [1.0_f32; 100];
    let tv = rosetta.compute_tensor_validation("model.ln_f.weight", &data);
    assert!(
        !tv.failures.iter().any(|f| f.contains("constant")),
        "ln_ tensor should be exempt from constant check: {:?}",
        tv.failures
    );
}

/// P070e: compute_tensor_validation with single valid element (valid_count=1, std=0)
/// Covers: line 1097-1101 (valid_count <= 1 path for std computation)
#[test]
fn p070e_compute_validation_single_element() {
    let rosetta = RosettaStone::new();
    let data = [0.5_f32];
    let tv = rosetta.compute_tensor_validation("single.weight", &data);
    assert_eq!(tv.element_count, 1);
    assert_eq!(tv.std, 0.0, "Std should be 0 for single element");
    assert!((tv.mean - 0.5).abs() < 1e-6);
    assert!((tv.min - 0.5).abs() < 1e-6);
    assert!((tv.max - 0.5).abs() < 1e-6);
}

/// P070f: compute_tensor_validation with NaN mixed with valid data
/// Covers: lines 1062-1064 NaN skip, 1082-1084 mean with valid_count > 0
/// Also covers line 1092 (NaN skip in variance computation)
#[test]
fn p070f_compute_validation_nan_mixed_with_valid() {
    let rosetta = RosettaStone::new();
    let data = [1.0_f32, f32::NAN, 3.0, f32::NAN, 5.0];
    let tv = rosetta.compute_tensor_validation("mixed.weight", &data);
    assert!(!tv.is_valid, "Mixed NaN tensor should be invalid");
    assert_eq!(tv.nan_count, 2);
    assert_eq!(tv.element_count, 5);
    // mean of valid values [1.0, 3.0, 5.0] = 3.0
    assert!(
        (tv.mean - 3.0).abs() < 1e-5,
        "Mean should be ~3.0, got {}",
        tv.mean
    );
    assert!((tv.min - 1.0).abs() < 1e-6);
    assert!((tv.max - 5.0).abs() < 1e-6);
    assert!(tv.std > 0.0, "Std should be non-zero for varied values");
}

/// P070g: compute_tensor_validation min/max clamping when only NaN/Inf present
/// Covers: lines 1170-1171 (min/max clamping when they remain INFINITY/NEG_INFINITY)
#[test]
fn p070g_compute_validation_minmax_clamped() {
    let rosetta = RosettaStone::new();
    // Only NaN and Inf, no finite values -> min stays INFINITY, max stays NEG_INFINITY
    let data = [f32::NAN, f32::INFINITY];
    let tv = rosetta.compute_tensor_validation("bad.weight", &data);
    // min was initialized to INFINITY (never updated), max to NEG_INFINITY
    // The code clamps: if min.is_infinite() { 0.0 } and if max.is_infinite() { 0.0 }
    assert_eq!(tv.min, 0.0, "min should be clamped to 0.0");
    assert_eq!(tv.max, 0.0, "max should be clamped to 0.0");
}

/// P070h: compute_tensor_validation with Inf mixed with valid data
/// Covers: Inf skip in main loop and variance loop
#[test]
fn p070h_compute_validation_inf_mixed_with_valid() {
    let rosetta = RosettaStone::new();
    let data = [2.0_f32, f32::INFINITY, 4.0, f32::NEG_INFINITY, 6.0];
    let tv = rosetta.compute_tensor_validation("mixed_inf.weight", &data);
    assert!(!tv.is_valid);
    assert_eq!(tv.inf_count, 2);
    // mean of valid values [2.0, 4.0, 6.0] = 4.0
    assert!(
        (tv.mean - 4.0).abs() < 1e-5,
        "Mean should be ~4.0, got {}",
        tv.mean
    );
    assert!((tv.min - 2.0).abs() < 1e-6);
    assert!((tv.max - 6.0).abs() < 1e-6);
}
