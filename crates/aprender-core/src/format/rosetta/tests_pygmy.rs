use super::super::*;
pub(crate) use super::super::{create_apr_fixture, create_safetensors_fixture, unique_temp_path};

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

    // Use isolated temp directory to prevent stale config.json from polluting
    // architecture detection (the import pipeline looks for sibling config.json)
    let work_dir = tempfile::tempdir().expect("Create temp dir");
    let st_path = work_dir.path().join("model.safetensors");
    std::fs::write(&st_path, &st_data).expect("Write ST data");

    let apr_path = work_dir.path().join("output.apr");

    use crate::format::converter::apr_import;
    use crate::format::converter_types::{Architecture, ImportOptions};

    let import_result = apr_import(
        st_path.to_str().expect("Path to string"),
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

#[path = "tcov.rs"]
mod tcov;
#[path = "computation.rs"]
mod computation;
#[path = "tests_pygmy_validation.rs"]
mod tests_pygmy_validation;

// ========================================================================
// #3903: is_all_zeros must not be VACUOUSLY true on an empty tensor.
//
// `empty_tensor_validation` deliberately returns `is_valid: true` for a
// zero-element tensor, and `p060_compute_validation_empty_data` asserts that
// ("Empty tensor should be valid"). But `is_all_zeros()` was
// `zero_count == element_count`, which is `0 == 0` for that same tensor — so
// the two verdicts contradicted each other, and `strict_blocking` reads the
// vacuous one:
//
//     strict_blocking(report) = nan > 0 OR inf > 0 OR all_zero_tensors non-empty
//
// The live consequence: `apr import` emits `lm_head.weight` as a 0-byte
// tied-embedding placeholder (#2309 — the runtime ties it to
// `model.embed_tokens.weight`), so EVERY tied-embedding model imported from
// SafeTensors was reported as carrying one all-zero tensor and rejected by
// `apr validate --strict`, though it loads and generates correctly. That
// violates the stated obligation of `contracts/apr-validate-fail-closed-v1.yaml`:
// "A healthy .apr (report.is_valid, no strict-blocking findings) passes — no
// false positive".
//
// "every element is zero" is not a claim you can make about no elements.
// ========================================================================

/// Builds a `TensorValidation` with the counts under test; every other field is
/// the neutral value `empty_tensor_validation` uses, so these tests isolate the
/// `zero_count`/`element_count` relation and nothing else.
fn tv_with_counts(name: &str, zero_count: usize, element_count: usize) -> TensorValidation {
    TensorValidation {
        name: name.to_string(),
        is_valid: true,
        nan_count: 0,
        inf_count: 0,
        zero_count,
        element_count,
        min: 0.0,
        max: 0.0,
        mean: 0.0,
        std: 0.0,
        failures: Vec::new(),
    }
}

#[test]
fn empty_tensor_is_not_all_zeros_3903() {
    let tv = tv_with_counts("lm_head.weight", 0, 0);
    assert!(
        !tv.is_all_zeros(),
        "a 0-element tensor has no elements to be zero; reporting it as all-zeros \
         makes --strict reject every tied-embedding model (#2309 placeholder; see #3903)"
    );
}

#[test]
fn the_empty_tensor_the_validator_actually_builds_is_not_all_zeros_3903() {
    // Not a hand-built struct: the value the production path returns for the
    // 0-byte placeholder, so the test cannot drift from what ships.
    let rosetta = RosettaStone::new();
    let tv = rosetta.compute_tensor_validation("lm_head.weight", &[]);
    assert_eq!(tv.element_count, 0, "precondition: this is the empty path");
    assert!(
        tv.is_valid,
        "precondition: empty is valid (p060_compute_validation_empty_data)"
    );
    assert!(
        !tv.is_all_zeros(),
        "is_valid and is_all_zeros must not contradict each other on the same tensor"
    );
}

#[test]
fn a_genuinely_all_zero_tensor_is_still_all_zeros_3903() {
    // The control: the guard must not weaken real detection. This is the shape of
    // the 11 dead `self_attn.v_proj.weight` tensors in the stale
    // qwen2.5-coder-1.5b-instruct-st.apr — 256 * 1536 elements, every one zero.
    let tv = tv_with_counts("model.layers.0.self_attn.v_proj.weight", 393_216, 393_216);
    assert!(
        tv.is_all_zeros(),
        "a populated tensor whose every element is zero is still a real finding"
    );
}

#[test]
fn a_partially_zero_tensor_is_not_all_zeros_3903() {
    let tv = tv_with_counts("model.layers.0.self_attn.v_proj.weight", 393_215, 393_216);
    assert!(!tv.is_all_zeros(), "one nonzero element is enough to disqualify");
}
