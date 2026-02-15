use super::super::*;
use super::{create_apr_fixture, create_safetensors_fixture, unique_temp_path};

// ========================================================================
// Pygmy-Based Tests (T-COV-95)
// Testing rosetta paths with in-memory generated models
// ========================================================================

#[test]
fn pygmy_inspect_safetensors() {
    use crate::format::test_factory::build_pygmy_safetensors;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_safetensors();

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(temp.path());

    assert!(result.is_ok(), "Should inspect pygmy SafeTensors");
    let inspection = result.expect("inspection");
    assert_eq!(inspection.format, FormatType::SafeTensors);
    assert!(!inspection.tensors.is_empty());
}

#[test]
fn pygmy_inspect_apr() {
    use crate::format::test_factory::build_pygmy_apr;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_apr();

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(temp.path());

    assert!(result.is_ok(), "Should inspect pygmy APR");
    let inspection = result.expect("inspection");
    assert_eq!(inspection.format, FormatType::Apr);
    assert!(!inspection.tensors.is_empty());
}

#[test]
fn pygmy_validate_apr() {
    use crate::format::test_factory::build_pygmy_apr;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_apr();

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.validate(temp.path());

    assert!(result.is_ok(), "Should validate pygmy APR");
    let validation = result.expect("validation");
    assert!(
        validation.is_valid,
        "Pygmy APR should be valid (no NaN/Inf)"
    );
    assert_eq!(validation.total_nan_count, 0);
    assert_eq!(validation.total_inf_count, 0);
}

#[test]
fn pygmy_validate_safetensors() {
    use crate::format::test_factory::build_pygmy_safetensors;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_safetensors();

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.validate(temp.path());

    assert!(result.is_ok(), "Should validate pygmy SafeTensors");
    let validation = result.expect("validation");
    assert!(validation.is_valid, "Pygmy SafeTensors should be valid");
}

#[test]
fn pygmy_inspect_apr_with_llama_style_config() {
    use crate::format::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = PygmyConfig::llama_style();
    let data = build_pygmy_apr_with_config(config);

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(temp.path());

    assert!(result.is_ok(), "Should inspect LLaMA-style pygmy APR");
    let inspection = result.expect("inspection");

    // Should have LLaMA-style tensor names
    assert!(
        inspection
            .tensors
            .iter()
            .any(|t| t.name.contains("self_attn")),
        "Should have attention tensors"
    );
}

#[test]
fn pygmy_inspect_quantized_apr() {
    use crate::format::test_factory::{build_pygmy_apr_f16, build_pygmy_apr_q8};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Test Q8 APR
    let q8_data = build_pygmy_apr_q8();
    let mut temp_q8 = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp_q8.write_all(&q8_data).expect("Write data");
    temp_q8.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(temp_q8.path());
    assert!(result.is_ok(), "Should inspect Q8 pygmy APR");

    // Test F16 APR
    let f16_data = build_pygmy_apr_f16();
    let mut temp_f16 = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp_f16.write_all(&f16_data).expect("Write data");
    temp_f16.flush().expect("Flush");

    let result = rosetta.inspect(temp_f16.path());
    assert!(result.is_ok(), "Should inspect F16 pygmy APR");
}

#[test]
fn pygmy_format_from_magic_apr() {
    use crate::format::test_factory::build_pygmy_apr;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_apr();

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let format = FormatType::from_magic(temp.path());
    assert!(format.is_ok(), "Should detect format from magic");
    assert_eq!(format.expect("format"), FormatType::Apr);
}

#[test]
fn pygmy_format_from_magic_safetensors() {
    use crate::format::test_factory::build_pygmy_safetensors;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_safetensors();

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let format = FormatType::from_magic(temp.path());
    assert!(format.is_ok(), "Should detect SafeTensors from magic");
    assert_eq!(format.expect("format"), FormatType::SafeTensors);
}

// ========================================================================
// T-GH192: Pre-Load Inspection Tests (rosetta-testing.md §Test Gaps)
// ========================================================================

/// T-GH192-01: Pre-load inspection returns correct metadata for APR format
#[test]
fn t_gh192_01_inspect_apr_returns_metadata() {
    use crate::format::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = PygmyConfig::realistic();
    let data = build_pygmy_apr_with_config(config);

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let inspection = rosetta
        .inspect(temp.path())
        .expect("T-GH192-01: APR inspection must succeed");

    // Verify inspection contains meaningful metadata
    assert!(
        !inspection.tensors.is_empty(),
        "T-GH192-01: Inspection must report tensors"
    );
    assert!(
        inspection.format == FormatType::Apr,
        "T-GH192-01: Inspection must identify APR format"
    );
}

/// T-GH192-01: Pre-load inspection returns correct metadata for SafeTensors format
#[test]
fn t_gh192_01_inspect_safetensors_returns_metadata() {
    use crate::format::test_factory::{build_pygmy_safetensors_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = PygmyConfig::realistic();
    let data = build_pygmy_safetensors_with_config(config);

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let inspection = rosetta
        .inspect(temp.path())
        .expect("T-GH192-01: SafeTensors inspection must succeed");

    assert!(
        !inspection.tensors.is_empty(),
        "T-GH192-01: Inspection must report tensors"
    );
    assert!(
        inspection.format == FormatType::SafeTensors,
        "T-GH192-01: Inspection must identify SafeTensors format"
    );
}

/// T-GH192-02: Sequential model loading with different sizes
/// This test verifies that the system correctly handles loading models
/// of different sizes sequentially without config leakage.
#[test]
fn t_gh192_02_sequential_model_size_switching() {
    use crate::format::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create "small" model (few layers)
    let small_config = PygmyConfig::default(); // 1 layer
    let small_data = build_pygmy_apr_with_config(small_config);

    let mut small_temp = NamedTempFile::with_suffix(".apr").expect("Create small temp");
    small_temp.write_all(&small_data).expect("Write small data");
    small_temp.flush().expect("Flush small");

    // Create "large" model (more layers)
    let large_config = PygmyConfig::realistic(); // More layers
    let large_data = build_pygmy_apr_with_config(large_config);

    let mut large_temp = NamedTempFile::with_suffix(".apr").expect("Create large temp");
    large_temp.write_all(&large_data).expect("Write large data");
    large_temp.flush().expect("Flush large");

    let rosetta = RosettaStone::new();

    // Inspect small model first
    let small_inspection = rosetta
        .inspect(small_temp.path())
        .expect("T-GH192-02: Small model inspection must succeed");

    // Inspect large model second
    let large_inspection = rosetta
        .inspect(large_temp.path())
        .expect("T-GH192-02: Large model inspection must succeed");

    // Verify they report different tensor counts or file sizes (no config leakage)
    // Note: realistic() has more tensors than default()
    assert!(
        small_inspection.tensors.len() != large_inspection.tensors.len()
            || small_inspection.file_size != large_inspection.file_size,
        "T-GH192-02: Different model sizes must report different metadata. \
         Small: {} tensors, {} bytes. Large: {} tensors, {} bytes.",
        small_inspection.tensors.len(),
        small_inspection.file_size,
        large_inspection.tensors.len(),
        large_inspection.file_size
    );

    // Re-inspect small model to verify no state leakage from large model
    let small_reinspection = rosetta
        .inspect(small_temp.path())
        .expect("T-GH192-02: Re-inspection must succeed");

    assert_eq!(
        small_inspection.tensors.len(),
        small_reinspection.tensors.len(),
        "T-GH192-02: Re-inspection must match original (no state leakage)"
    );
}

// ========================================================================
// T-GH194: Tensor Count Preservation Tests (rosetta-testing.md §Test Gaps)
// ========================================================================

/// T-GH194-01: APR round-trip preserves ALL tensor count
/// This test verifies that converting to APR and back doesn't drop tensors.
/// Note: Uses SafeTensors as source since GGUF builder is not available.
#[test]
fn t_gh194_01_safetensors_apr_preserves_tensor_count() {
    use crate::format::tensors::{list_tensors_from_bytes, TensorListOptions};
    use crate::format::test_factory::{build_pygmy_safetensors_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Use realistic config to ensure we have a meaningful number of tensors
    let config = PygmyConfig::realistic();
    let st_data = build_pygmy_safetensors_with_config(config);

    // Count tensors in original SafeTensors
    let st_result = list_tensors_from_bytes(&st_data, TensorListOptions::default())
        .expect("T-GH194-01: SafeTensors tensor listing must succeed");
    let st_count = st_result.tensor_count;

    assert!(
        st_count > 5,
        "T-GH194-01: Test requires SafeTensors with >5 tensors, got {}",
        st_count
    );

    // Write SafeTensors to temp file for import
    let mut st_temp = NamedTempFile::with_suffix(".safetensors").expect("Create ST temp");
    st_temp.write_all(&st_data).expect("Write ST data");
    st_temp.flush().expect("Flush ST");

    // Import SafeTensors to APR
    let apr_temp = NamedTempFile::with_suffix(".apr").expect("Create APR temp");
    let apr_path = apr_temp.path().to_path_buf();

    use crate::format::converter::apr_import;
    use crate::format::converter_types::{Architecture, ImportOptions};

    let import_result = apr_import(
        st_temp.path().to_str().expect("Path to string"),
        &apr_path,
        ImportOptions {
            architecture: Architecture::Auto,
            allow_no_config: true,
            ..Default::default()
        },
    );

    assert!(
        import_result.is_ok(),
        "T-GH194-01: SafeTensors→APR import must succeed: {:?}",
        import_result.err()
    );

    // Count tensors in resulting APR
    let apr_data = std::fs::read(&apr_path).expect("Read APR file");
    let apr_result = list_tensors_from_bytes(&apr_data, TensorListOptions::default())
        .expect("T-GH194-01: APR tensor listing must succeed");
    let apr_count = apr_result.tensor_count;

    // INVARIANT: APR must have at least as many tensors as SafeTensors
    // (it may have more if weight tying is resolved, but never fewer)
    assert!(
        apr_count >= st_count,
        "T-GH194-01: APR must preserve all tensors. SafeTensors: {}, APR: {} (dropped {})",
        st_count,
        apr_count,
        st_count.saturating_sub(apr_count)
    );
}

// ========================================================================
// Section 17: TensorValidation Helper Methods (T-COV-95)
// ========================================================================

#[test]
fn tcov_tensor_validation_has_nan_true() {
    let tv = TensorValidation {
        name: "test.weight".to_string(),
        is_valid: false,
        nan_count: 5,
        inf_count: 0,
        zero_count: 0,
        element_count: 100,
        min: -1.0,
        max: 1.0,
        mean: 0.0,
        std: 0.5,
        failures: vec!["NaN detected".to_string()],
    };
    assert!(tv.has_nan());
    assert!(!tv.has_inf());
    assert!(!tv.is_all_zeros());
}

#[test]
fn tcov_tensor_validation_has_inf_true() {
    let tv = TensorValidation {
        name: "test.weight".to_string(),
        is_valid: false,
        nan_count: 0,
        inf_count: 3,
        zero_count: 0,
        element_count: 100,
        min: -1.0,
        max: 1.0,
        mean: 0.0,
        std: 0.5,
        failures: vec!["Inf detected".to_string()],
    };
    assert!(!tv.has_nan());
    assert!(tv.has_inf());
    assert!(!tv.is_all_zeros());
}

#[test]
fn tcov_tensor_validation_is_all_zeros_true() {
    let tv = TensorValidation {
        name: "test.weight".to_string(),
        is_valid: false,
        nan_count: 0,
        inf_count: 0,
        zero_count: 100,
        element_count: 100,
        min: 0.0,
        max: 0.0,
        mean: 0.0,
        std: 0.0,
        failures: vec!["All zeros".to_string()],
    };
    assert!(!tv.has_nan());
    assert!(!tv.has_inf());
    assert!(tv.is_all_zeros());
}

include!("tests_pygmy_part_02.rs");
include!("tests_pygmy_part_03.rs");
include!("tests_pygmy_part_04.rs");
