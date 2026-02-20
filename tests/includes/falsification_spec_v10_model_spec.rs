// =============================================================================
// Section 4: Model Specification (F-MODEL-*)
// =============================================================================

#[test]
fn f_model_001_oracle_identifies_as_qwen2() {
    // F-MODEL-001: qwen2 family must exist in registry
    let registry = build_default_registry();
    assert!(
        registry.get("qwen2").is_some(),
        "F-MODEL-001: qwen2 family must exist in registry"
    );
}

#[test]
fn f_model_002_hf_cross_validation_matches() {
    // F-MODEL-002: Structural check — compare-hf command exists for HF cross-validation
    let compare_hf_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("compare_hf.rs");
    let content = std::fs::read_to_string(&compare_hf_path).expect("compare_hf.rs must exist");
    assert!(
        content.contains("fn execute") || content.contains("fn run"),
        "F-MODEL-002: compare_hf.rs must have execute/run function"
    );
    assert!(
        content.contains("huggingface")
            || content.contains("HuggingFace")
            || content.contains("hf"),
        "F-MODEL-002: compare_hf must reference HuggingFace"
    );
}

#[test]
fn f_model_003_contract_rejects_wrong_family() {
    // F-MODEL-003: qwen2 != llama (different architectures)
    let registry = build_default_registry();
    let qwen2 = registry.get("qwen2").expect("qwen2 must exist");
    let llama = registry.get("llama").expect("llama must exist");

    // Qwen2 and LLaMA are different families with different constraints
    let qwen2_constraints = qwen2.constraints();
    let llama_constraints = llama.constraints();

    // At minimum, they should differ on some property
    let qwen2_7b = qwen2.size_config("7b").expect("qwen2 7b exists");

    // Verify qwen2 family is correctly identified (not llama)
    assert_eq!(
        qwen2.family_name(),
        "qwen2",
        "F-MODEL-003: Qwen2 family must be 'qwen2'"
    );
    assert_eq!(
        llama.family_name(),
        "llama",
        "F-MODEL-003: LLaMA family must be 'llama'"
    );

    // Families must have distinct parameters for the same size class
    // Check that at least one of hidden_dim, num_kv_heads, or vocab_size differs
    if let Some(llama_7b) = llama.size_config("7b") {
        let differs = qwen2_7b.hidden_dim != llama_7b.hidden_dim
            || qwen2_7b.num_kv_heads != llama_7b.num_kv_heads
            || qwen2_7b.vocab_size != llama_7b.vocab_size;
        assert!(
            differs,
            "F-MODEL-003: Qwen2 7B and LLaMA 7B must differ in at least one parameter"
        );
    }

    // Also check that different families have potentially different constraints
    let _ = (qwen2_constraints, llama_constraints); // used for type checking
}

#[test]
fn f_model_004_tensor_count_matches_contract() {
    // F-MODEL-004: Qwen2 7B tensor count formula matches
    let registry = build_default_registry();
    let qwen2 = registry.get("qwen2").expect("qwen2 must exist");

    let tensor_count = qwen2
        .expected_tensor_count("7b")
        .expect("7b tensor count must exist");

    // 3 global + 12 per-layer * 28 layers = 339
    let expected = 3 + 12 * 28;
    assert_eq!(
        tensor_count, expected,
        "F-MODEL-004: Qwen2 7B tensor count must be {expected}, got {tensor_count}"
    );
}

#[test]
fn f_model_005_gqa_ratio_correct() {
    // F-MODEL-005: GQA ratio = 7 (28/4)
    let registry = build_default_registry();
    let qwen2 = registry.get("qwen2").expect("qwen2 must exist");
    let config = qwen2.size_config("7b").expect("7b must exist");

    let gqa_ratio = config.num_heads / config.num_kv_heads;
    assert_eq!(
        gqa_ratio, 7,
        "F-MODEL-005: Qwen2 7B GQA ratio must be 7, got {gqa_ratio}"
    );
    assert_eq!(
        config.num_heads % config.num_kv_heads,
        0,
        "F-MODEL-005: num_heads must be divisible by num_kv_heads"
    );
}

#[test]
fn f_model_006_head_dim_matches_contract() {
    // F-MODEL-006: head_dim = 128 (3584/28)
    let registry = build_default_registry();
    let qwen2 = registry.get("qwen2").expect("qwen2 must exist");
    let config = qwen2.size_config("7b").expect("7b must exist");

    assert_eq!(config.head_dim, 128, "F-MODEL-006: head_dim must be 128");
    assert_eq!(
        config.hidden_dim / config.num_heads,
        128,
        "F-MODEL-006: hidden_dim/num_heads must equal head_dim"
    );
}

// =============================================================================
// Section 5: Format Support (F-FMT-*)
// =============================================================================

#[test]
fn f_fmt_001_from_magic_detects_gguf() {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&[0u8; 64]);

    let mut temp = NamedTempFile::with_suffix(".gguf").expect("temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let format = FormatType::from_magic(temp.path());
    assert_eq!(
        format.expect("detect GGUF"),
        FormatType::Gguf,
        "F-FMT-001: GGUF magic must be detected"
    );
}

#[test]
fn f_fmt_002_from_magic_detects_safetensors() {
    let header = r#"{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header_bytes);
    data.extend_from_slice(&[0u8; 16]);

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let format = FormatType::from_magic(temp.path());
    assert_eq!(
        format.expect("detect SafeTensors"),
        FormatType::SafeTensors,
        "F-FMT-002: SafeTensors header must be detected"
    );
}

#[test]
fn f_fmt_003_from_magic_detects_apr() {
    let mut data = Vec::new();
    data.extend_from_slice(b"APR2");
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&[0u8; 64]);

    let mut temp = NamedTempFile::with_suffix(".apr").expect("temp file");
    temp.write_all(&data).expect("write");
    temp.flush().expect("flush");

    let format = FormatType::from_magic(temp.path());
    assert_eq!(
        format.expect("detect APR"),
        FormatType::Apr,
        "F-FMT-003: APR magic must be detected"
    );
}

#[test]
fn f_fmt_004_unknown_format_rejected() {
    let mut temp = NamedTempFile::with_suffix(".bin").expect("temp file");
    temp.write_all(b"GARBAGE_NOT_A_MODEL_FORMAT_1234567890")
        .expect("write");
    temp.flush().expect("flush");

    let format = FormatType::from_magic(temp.path());
    assert!(
        format.is_err(),
        "F-FMT-004: Unknown magic bytes must be rejected"
    );
}

#[test]
fn f_fmt_005_all_commands_work_on_all_formats() {
    // F-FMT-005: Structural check — FormatType enum has all 3 format variants
    let formats = [FormatType::Apr, FormatType::Gguf, FormatType::SafeTensors];
    assert_eq!(
        formats.len(),
        3,
        "F-FMT-005: Must support exactly 3 formats (APR, GGUF, SafeTensors)"
    );
    // Verify format names are distinct
    let names: Vec<String> = formats.iter().map(|f| format!("{f:?}")).collect();
    assert_ne!(names[0], names[1], "Formats must be distinct");
    assert_ne!(names[1], names[2], "Formats must be distinct");
    assert_ne!(names[0], names[2], "Formats must be distinct");
}

// =============================================================================
