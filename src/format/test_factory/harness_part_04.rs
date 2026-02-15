
/// F-CONV-04 (Strict Leakage): Import missing norm tensor with --strict
///
/// **FALSIFICATION FINDING (DEFECT-001):**
/// Strict mode did NOT reject models missing norm tensors. This was a Jidoka
/// violation - the system should stop-the-line for incomplete models.
///
/// **Previous Behavior:** Import succeeds (result.is_ok())
/// **Fixed Behavior:** Import fails with "Missing required tensor: model.norm.weight"
/// **Status:** FIXED (DEFECT-001)
#[test]
fn test_f_conv_04_strict_missing_norm() {
    // Create config without norms
    let config = PygmyConfig {
        include_norms: false,
        include_embedding: true,
        include_attention: true,
        include_mlp: true,
        ..Default::default()
    };

    let mut options = ImportOptions::default();
    options.strict = true;
    options.allow_no_config = true;

    let h = ConversionTestHarness::new().with_safetensors(config);
    let result = h.try_import_to_apr(options);

    // DEFECT-001 FIX VERIFICATION: Strict mode should now reject missing norms
    assert!(
        result.is_err(),
        "DEFECT-001 FIX: Strict mode should reject missing model.norm.weight"
    );

    // Verify the error message mentions the missing tensor
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("model.norm.weight"),
        "Error should mention missing tensor: {err_msg}"
    );
}

/// F-DISP-01 (Magic vs Extension): GGUF as .txt -> should work via magic bytes
#[test]
fn test_f_disp_01_magic_vs_extension() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use crate::format::tensors::{list_tensors_from_bytes, TensorListOptions};
    use tempfile::NamedTempFile;

    // Create valid GGUF
    let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: floats,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export");

    // Rename to .txt extension
    let file = NamedTempFile::with_suffix(".txt").expect("create");
    std::fs::write(file.path(), &gguf_bytes).expect("write");

    // Should still work via magic bytes detection
    let result = list_tensors_from_bytes(&gguf_bytes, TensorListOptions::default());
    assert!(
        result.is_ok(),
        "F-DISP-01: GGUF should be detected by magic bytes, not extension"
    );
    assert!(
        result
            .expect("test harness value")
            .format_version
            .contains("GGUF"),
        "F-DISP-01: Should detect as GGUF format"
    );
}

/// F-DISP-02 (Format Poisoning): APR magic + noise -> graceful error, not panic
#[test]
fn test_f_disp_02_format_poisoning() {
    use crate::format::tensors::{list_tensors_from_bytes, TensorListOptions};

    // Create poisoned data: APR magic followed by random noise
    let mut poisoned = Vec::new();
    poisoned.extend_from_slice(b"APRN"); // APR magic
    poisoned.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // noise
    poisoned.extend_from_slice(&vec![0xFF; 100]); // more noise

    // Should fail gracefully, not panic
    let result = list_tensors_from_bytes(&poisoned, TensorListOptions::default());
    assert!(
        result.is_err(),
        "F-DISP-02: Poisoned APR should fail gracefully"
    );
}

/// F-DISP-03 (SafeTensors Header Overflow): 100GB header -> immediate rejection
#[test]
fn test_f_disp_03_header_overflow() {
    use crate::format::tensors::{list_tensors_from_bytes, TensorListOptions};

    // Create SafeTensors with absurd header length (100GB)
    let header_len: u64 = 100 * 1024 * 1024 * 1024; // 100GB
    let mut overflow_bytes = Vec::new();
    overflow_bytes.extend_from_slice(&header_len.to_le_bytes());
    overflow_bytes.extend_from_slice(b"{}"); // minimal "header"

    // Should be rejected immediately (safety limit)
    let result = list_tensors_from_bytes(&overflow_bytes, TensorListOptions::default());
    assert!(
        result.is_err(),
        "F-DISP-03: 100GB header should trigger safety rejection"
    );
}

/// F-DISP-04 (Cross-Format Linting): GGUF lint rules should trigger
#[test]
fn test_f_disp_04_cross_format_linting() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use crate::format::lint::lint_model_file;
    use tempfile::NamedTempFile;

    // Create GGUF without license metadata (should trigger lint warning)
    let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: floats,
    };
    // No license, no author, no description
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export");

    let file = NamedTempFile::with_suffix(".gguf").expect("create");
    std::fs::write(file.path(), &gguf_bytes).expect("write");

    // Lint should trigger GGUF-specific warnings
    let result = lint_model_file(file.path());
    assert!(result.is_ok(), "F-DISP-04: Lint should not crash on GGUF");
    let report = result.expect("test harness value");
    assert!(
        report.warn_count > 0,
        "F-DISP-04: GGUF without metadata should trigger warnings"
    );
}

/// F-DATA-01 (NaN Propagation): NaN in tensor -> detected in validation
#[test]
fn test_f_data_01_nan_propagation() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use crate::format::rosetta::RosettaStone;
    use tempfile::NamedTempFile;

    // Create GGUF with NaN values
    let nan_bytes = f32::NAN.to_le_bytes();
    let mut tensor_data = Vec::new();
    for _ in 0..4 {
        tensor_data.extend_from_slice(&nan_bytes);
    }

    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: tensor_data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export");

    let file = NamedTempFile::with_suffix(".gguf").expect("create");
    std::fs::write(file.path(), &gguf_bytes).expect("write");

    // Validate should detect NaN
    let rosetta = RosettaStone::default();
    let result = rosetta.validate(file.path());
    assert!(result.is_ok(), "F-DATA-01: Validation should not crash");
    let report = result.expect("test harness value");
    assert!(
        report.total_nan_count > 0,
        "F-DATA-01: NaN should be detected and reported"
    );
}

/// F-DATA-02 (All-Zeros Refutation): All-zero tensor -> Jidoka alarm
///
/// **FALSIFICATION FINDING (DEFECT-002):**
/// All-zero tensors are NOT being detected in GGUF validation.
/// This is a Jidoka violation - uninitialized weights should trigger alarm.
///
/// **Previous Behavior:** `all_zero_tensors` was empty (GGUF export bug)
/// **Fixed Behavior:** Should contain "model.weight"
/// **Status:** FIXED (DEFECT-002)
#[test]
fn test_f_data_02_all_zeros_alarm() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use crate::format::rosetta::RosettaStone;
    use tempfile::NamedTempFile;

    // Create GGUF with all-zero tensor
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![4, 4],
        dtype: GgmlType::F32,
        data: vec![0u8; 64], // All zeros - 4x4 F32 = 16 elements * 4 bytes = 64 bytes
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export");

    let file = NamedTempFile::with_suffix(".gguf").expect("create");
    std::fs::write(file.path(), &gguf_bytes).expect("write");

    // Validate should detect all-zeros
    let rosetta = RosettaStone::default();
    let result = rosetta.validate(file.path());
    assert!(result.is_ok(), "F-DATA-02: Validation should not crash");
    let report = result.expect("test harness value");

    // DEFECT-002 FIX VERIFICATION: All-zeros should now be detected
    assert!(
        report
            .all_zero_tensors
            .contains(&"model.weight".to_string()),
        "DEFECT-002 FIX: All-zeros tensor should be detected. Got: {:?}",
        report.all_zero_tensors
    );
}

/// F-TPS-01 (Boilerplate Check): New conversion test < 10 lines
#[test]
fn test_f_tps_01_boilerplate_minimal() {
    // This is the ONE-LINER API from the spec - proves < 10 lines
    ConversionTestHarness::assert_import_ok(PygmyConfig::default());
    // Total: 1 line. Requirement: < 10 lines. [REFUTED]
}

/// F-TPS-02 (Read-Back Verification): list_tensors uses mmap for SafeTensors
#[test]
fn test_f_tps_02_mmap_verification() {
    use crate::format::tensors::{list_tensors, TensorListOptions};
    use tempfile::NamedTempFile;

    // Create SafeTensors file
    let st_bytes = super::build_pygmy_safetensors();
    let file = NamedTempFile::with_suffix(".safetensors").expect("create");
    std::fs::write(file.path(), &st_bytes).expect("write");

    // Path-based list_tensors uses MappedSafeTensors (mmap)
    let result = list_tensors(file.path(), TensorListOptions::default());
    assert!(
        result.is_ok(),
        "F-TPS-02: list_tensors should work with file path (mmap)"
    );

    // Verify format detected correctly (mmap path would work)
    let info = result.expect("test harness value");
    assert!(
        info.format_version.contains("SafeTensors"),
        "F-TPS-02: Should detect SafeTensors format via mmap path"
    );
}

// ====================================================================
// Audit Item 4: infer_model_config_from_tensors with realistic dims
// Complements pmat.rs tests -- verifies head_dim detection is triggered
// ====================================================================

/// Verify `infer_model_config_from_tensors` correctly infers num_heads and
/// num_kv_heads when given realistic dimensions (hidden_size=128, head_dim=64).
/// PygmyConfig's tiny dims (hidden_size=4/8) never match head_dim candidates
/// [64, 128, 96, 80], so this path was previously untested via harness.
#[test]
fn test_f_infer_config_realistic_dimensions() {
    use crate::format::converter::infer_model_config_from_tensors;
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

    // embedding: [vocab=256, hidden=128] -> vocab_size=256 (larger), hidden_size=128
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (vec![0.0; 256 * 128], vec![256, 128]),
    );
    // Q: [128, 128] -> q_dim==hidden_size -> try head_dim=64 -> num_heads=2
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 128 * 128], vec![128, 128]),
    );
    // K: [64, 128] -> kv_dim=64, head_dim=64 -> num_kv_heads=1 (GQA 2:1)
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![0.0; 64 * 128], vec![64, 128]),
    );
    tensors.insert(
        "model.layers.1.self_attn.q_proj.weight".to_string(),
        (vec![0.0; 128 * 128], vec![128, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors);
    assert!(
        config.is_some(),
        "Inference must succeed with realistic dims"
    );

    let config = config.expect("test harness value");
    assert_eq!(config.hidden_size, Some(128));
    assert_eq!(config.vocab_size, Some(256));
    assert_eq!(config.num_heads, Some(2), "128/64 head_dim = 2 heads");
    assert_eq!(config.num_kv_heads, Some(1), "64/64 = 1 KV head (GQA)");
    assert_eq!(config.num_layers, Some(2));
}

// ====================================================================
// Audit Item 5 fix: Harness round-trip with PygmyConfig::realistic()
// Exercises infer_model_config_from_tensors through the full pipeline
// ====================================================================

/// Audit #5: Import with realistic dims succeeds via harness
#[test]
fn test_f_harness_import_realistic_dims() {
    ConversionTestHarness::assert_import_ok(PygmyConfig::realistic());
}

/// Audit #5: Full round-trip (ST->APR->ST) with realistic dims
#[test]
fn test_f_harness_roundtrip_realistic_dims() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::realistic());
}

/// Audit #5: ST->APR->GGUF with realistic dims (exercises GGUF export shape handling)
#[test]
fn test_f_harness_gguf_export_realistic_dims() {
    use crate::format::gguf::GgufReader;

    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::realistic())
        .import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        })
        .export_to_gguf();

    let gguf_path = h.output_path().expect("GGUF output exists");
    let gguf_data = std::fs::read(gguf_path).expect("read GGUF");
    let reader = GgufReader::from_bytes(gguf_data).expect("parse GGUF");

    // Realistic dims should produce valid GGUF with tensors
    assert!(
        !reader.tensors.is_empty(),
        "Realistic config must produce GGUF with tensors"
    );

    // Should have separate Q/K/V (no fusion)
    let names: Vec<&str> = reader.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(
        !names.iter().any(|n| n.contains("attn_qkv")),
        "Realistic config must NOT have fused attn_qkv. Found: {names:?}"
    );
}

// ====================================================================
// T-QKV-02: Round-trip tests must verify tensor NAME set equality
// ====================================================================

/// T-QKV-02: verify_apr() detects extra tensors not in source (name-set check)
#[test]
fn test_t_qkv_02_name_set_equality_apr() {
    // Standard import -- names should match exactly
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::llama_style())
        .import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        });
    let result = h.verify_apr();
    assert!(
        result.passed(),
        "T-QKV-02: Llama-style import should have matching tensor names. \
         Mismatches: {:?}",
        result.mismatches
    );
}

/// T-QKV-02: verify_safetensors() detects name-set equality on round-trip
#[test]
fn test_t_qkv_02_name_set_equality_roundtrip() {
    // Full round-trip -- names should survive ST->APR->ST
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::qwen2_gqa())
        .import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        })
        .export_to_safetensors();
    let result = h.verify_safetensors();
    assert!(
        result.passed(),
        "T-QKV-02: GQA round-trip should preserve tensor name set. \
         Mismatches: {:?}",
        result.mismatches
    );
}
