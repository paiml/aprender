#[test]
fn test_validate_2d_tensor_shape_3d() {
    let spec = make_spec("test.weight", "[vocab, hidden]", true);
    let config = make_config_full();
    let result = validate_2d_tensor_shape("test", &[10, 20, 30], &spec, &config);
    assert!(!result.passed);
    assert!(result.details.contains("must be 2D, got 3D"));
}

// ========================================================================
// 30. validate_1d_tensor_shape 0D tensor
// ========================================================================

#[test]
fn test_validate_1d_tensor_shape_0d() {
    let spec = make_spec("test.bias", "[hidden]", false);
    let config = make_config_full();
    let result = validate_1d_tensor_shape("test.bias", &[], &spec, &config);
    assert!(!result.passed);
    assert!(result.details.contains("must be 1D, got 0D"));
}

// ========================================================================
// 31. load_contract_from - valid YAML
// ========================================================================

#[test]
fn test_load_contract_from_valid_yaml() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("contract.yaml");
    let yaml_content = r#"
metadata:
  version: "1.0"
  created: "2026-01-01"
  updated: "2026-01-01"
  author: "test"
  description: "test contract"
formats:
  apr:
    layout: "row-major"
    shape_convention: "[out_dim, in_dim]"
kernel:
  signature: "matmul(W, x, out_dim, in_dim)"
  weight_shape: "[out_dim, in_dim]"
  computation: "y = W @ x"
  byte_calculation: "out * in * block_size / QK_K"
  block_sizes:
    Q4_K: 144
  QK_K: 256
tensors: {}
validation_rules:
  - id: "F-LAYOUT-CONTRACT-001"
    name: "2D Transpose Check"
    description: "All 2D weights are transposed"
    severity: "P0"
    critical: true
"#;
    std::fs::write(&yaml_path, yaml_content).unwrap();

    let contract = load_contract_from(&yaml_path).unwrap();
    assert_eq!(contract.metadata.version, "1.0");
    assert_eq!(contract.metadata.author, "test");
    assert_eq!(contract.kernel.qk_k, 256);
    assert_eq!(contract.validation_rules.len(), 1);
    assert_eq!(contract.validation_rules[0].id, "F-LAYOUT-CONTRACT-001");
}

#[test]
fn test_load_contract_from_invalid_yaml() {
    let dir = tempfile::tempdir().unwrap();
    let yaml_path = dir.path().join("bad.yaml");
    std::fs::write(&yaml_path, "this: is: not: valid: yaml: [[[").unwrap();

    let result = load_contract_from(&yaml_path);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Failed to parse"));
}

// ========================================================================
// 32. read_safetensors_metadata - invalid UTF-8
// ========================================================================

#[test]
fn test_read_safetensors_metadata_invalid_utf8() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("badutf8.safetensors");

    // Write invalid UTF-8 bytes as header
    let bad_bytes: &[u8] = &[0xFF, 0xFE, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85];
    let header_len = bad_bytes.len() as u64;
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(bad_bytes).unwrap();

    let result = read_safetensors_metadata(&file_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid UTF-8"));
}

// ========================================================================
// 33. load_contract (uses DEFAULT_CONTRACT_PATH)
// ========================================================================

#[test]
fn test_load_contract_default_path_missing() {
    // The default path is relative and almost certainly not present in test env
    let result = load_contract();
    // Either succeeds (if contract exists) or fails with file-not-found
    if let Err(e) = result {
        let err_msg = format!("{e}");
        assert!(err_msg.contains("Failed to read"));
    }
}
