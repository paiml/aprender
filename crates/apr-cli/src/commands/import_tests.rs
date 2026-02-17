use super::*;
use aprender::format::converter::QuantizationType;

// =========================================================================
// derive_output_path() tests
// =========================================================================

#[test]
fn test_derive_output_path_hf_repo() {
    let result = derive_output_path("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct").unwrap();
    assert_eq!(result, PathBuf::from("Qwen2.5-Coder-1.5B-Instruct.apr"));
}

#[test]
fn test_derive_output_path_hf_with_file() {
    let result =
        derive_output_path("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model-q4k.gguf").unwrap();
    assert_eq!(result, PathBuf::from("model-q4k.apr"));
}

#[test]
fn test_derive_output_path_local_gguf() {
    let result = derive_output_path("/path/to/model.gguf").unwrap();
    assert_eq!(result, PathBuf::from("model.apr"));
}

#[test]
fn test_derive_output_path_local_safetensors() {
    let result = derive_output_path("model.safetensors").unwrap();
    assert_eq!(result, PathBuf::from("model.apr"));
}

#[test]
fn test_derive_output_path_url() {
    let result = derive_output_path("https://example.com/models/qwen-1.5b.gguf").unwrap();
    assert_eq!(result, PathBuf::from("qwen-1.5b.apr"));
}

#[test]
fn test_derive_output_path_url_no_extension() {
    let result = derive_output_path("https://example.com/models/mymodel").unwrap();
    assert_eq!(result, PathBuf::from("mymodel.apr"));
}

#[test]
fn test_derive_output_path_hf_nested_file() {
    let result = derive_output_path("hf://openai/whisper-tiny/pytorch_model.bin").unwrap();
    assert_eq!(result, PathBuf::from("pytorch_model.apr"));
}

#[test]
fn test_derive_output_path_relative_path() {
    let result = derive_output_path("./models/test.safetensors").unwrap();
    assert_eq!(result, PathBuf::from("test.apr"));
}

// =========================================================================
// parse_quantize() tests
// =========================================================================

#[test]
fn test_parse_quantize_none() {
    let result = parse_quantize(None).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_parse_quantize_int8() {
    let result = parse_quantize(Some("int8")).unwrap();
    assert_eq!(result, Some(QuantizationType::Int8));
}

#[test]
fn test_parse_quantize_int4() {
    let result = parse_quantize(Some("int4")).unwrap();
    assert_eq!(result, Some(QuantizationType::Int4));
}

#[test]
fn test_parse_quantize_fp16() {
    let result = parse_quantize(Some("fp16")).unwrap();
    assert_eq!(result, Some(QuantizationType::Fp16));
}

#[test]
fn test_parse_quantize_q4k() {
    let result = parse_quantize(Some("q4k")).unwrap();
    assert_eq!(result, Some(QuantizationType::Q4K));
}

#[test]
fn test_parse_quantize_q4_k_underscore() {
    let result = parse_quantize(Some("q4_k")).unwrap();
    assert_eq!(result, Some(QuantizationType::Q4K));
}

#[test]
fn test_parse_quantize_unknown() {
    let result = parse_quantize(Some("q8_0"));
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Unknown quantization"));
            assert!(msg.contains("Supported: int8, int4, fp16, q4k"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_parse_quantize_invalid() {
    let result = parse_quantize(Some("notaquant"));
    assert!(result.is_err());
}

// =========================================================================
// run() error cases tests
// =========================================================================

#[test]
fn test_run_unknown_architecture() {
    let result = run(
        "hf://test/model",
        Some(Path::new("output.apr")),
        Some("unknown_arch"), // Invalid architecture
        None,
        false,
        false,
        None,  // tokenizer
        false, // enforce_provenance
        false, // allow_no_config
    );

    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Unknown architecture"));
            assert!(msg.contains("Supported: whisper, llama, bert, qwen2, auto"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_run_with_whisper_arch() {
    // This will fail at import stage but tests architecture parsing
    let result = run(
        "hf://test/model",
        Some(Path::new("output.apr")),
        Some("whisper"),
        None,
        false,
        false,
        None,  // tokenizer
        false, // enforce_provenance
        false, // allow_no_config
    );

    // Will fail at network stage, not architecture parsing
    assert!(result.is_err());
}

#[test]
fn test_run_with_llama_arch() {
    // This will fail at import stage but tests architecture parsing
    let result = run(
        "hf://test/model",
        Some(Path::new("output.apr")),
        Some("llama"),
        None,
        false,
        false,
        None,  // tokenizer
        false, // enforce_provenance
        false, // allow_no_config
    );

    // Will fail at network stage, not architecture parsing
    assert!(result.is_err());
}

#[test]
fn test_run_with_bert_arch() {
    // This will fail at import stage but tests architecture parsing
    let result = run(
        "hf://test/model",
        Some(Path::new("output.apr")),
        Some("bert"),
        None,
        false,
        false,
        None,  // tokenizer
        false, // enforce_provenance
        false, // allow_no_config
    );

    // Will fail at network stage, not architecture parsing
    assert!(result.is_err());
}

#[test]
fn test_run_with_qwen2_arch() {
    // This will fail at import stage but tests architecture parsing
    let result = run(
        "hf://test/model",
        Some(Path::new("output.apr")),
        Some("qwen2"),
        None,
        false,
        false,
        None,  // tokenizer
        false, // enforce_provenance
        false, // allow_no_config
    );

    // Will fail at network stage, not architecture parsing
    assert!(result.is_err());
}

#[test]
fn test_run_with_auto_arch() {
    // This will fail at import stage but tests architecture parsing
    let result = run(
        "hf://test/model",
        Some(Path::new("output.apr")),
        Some("auto"),
        None,
        false,
        false,
        None,  // tokenizer
        false, // enforce_provenance
        false, // allow_no_config
    );

    // Will fail at network stage, not architecture parsing
    assert!(result.is_err());
}

#[test]
fn test_run_with_quantize_option() {
    // This will fail at import stage but tests quantize parsing
    let result = run(
        "hf://test/model",
        Some(Path::new("output.apr")),
        None,
        Some("int8"),
        false,
        false,
        None,  // tokenizer
        false, // enforce_provenance
        false, // allow_no_config
    );

    // Will fail at network stage, not quantize parsing
    assert!(result.is_err());
}

#[test]
fn test_run_with_force_flag() {
    // This will fail at import stage but tests force flag
    let result = run(
        "hf://test/model",
        Some(Path::new("output.apr")),
        None,
        None,
        true, // force
        false,
        None,  // tokenizer
        false, // enforce_provenance
        false, // allow_no_config
    );

    // Will fail at network stage
    assert!(result.is_err());
}

#[test]
fn test_run_invalid_source() {
    // Empty source should fail
    let result = run(
        "",
        Some(Path::new("output.apr")),
        None,
        None,
        false,
        false,
        None,
        false, // enforce_provenance
        false, // allow_no_config
    );

    assert!(result.is_err());
}

// =========================================================================
// F-GT-001: --enforce-provenance tests
// =========================================================================

#[test]
fn t_f_gt_001_enforce_provenance_rejects_gguf_source() {
    let result = run(
        "model.gguf",
        Some(Path::new("output.apr")),
        None,
        None,
        false,
        false,
        None,
        true,  // enforce_provenance = ON
        false, // allow_no_config
    );
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("F-GT-001"),
        "Error must cite F-GT-001 gate: {err_msg}"
    );
    assert!(
        err_msg.contains("provenance"),
        "Error must mention provenance: {err_msg}"
    );
}

#[test]
fn t_f_gt_001_enforce_provenance_rejects_gguf_hub_pattern() {
    // Hub-style paths with -GGUF suffix should also be rejected
    let result = run(
        "hf://TheBloke/Qwen2.5-Coder-7B-GGUF",
        Some(Path::new("output.apr")),
        None,
        None,
        false,
        false,
        None,
        true,  // enforce_provenance = ON
        false, // allow_no_config
    );
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("F-GT-001"),
        "Error must cite F-GT-001 gate: {err_msg}"
    );
}

#[test]
fn t_f_gt_001_no_provenance_allows_gguf() {
    // Without --enforce-provenance, GGUF should NOT be rejected
    // (it will fail for other reasons like file not found, but NOT F-GT-001)
    let result = run(
        "model.gguf",
        Some(Path::new("output.apr")),
        None,
        None,
        false,
        false,
        None,
        false, // enforce_provenance = OFF
        false, // allow_no_config
    );
    // Should fail (file doesn't exist) but NOT with F-GT-001
    if let Err(e) = &result {
        let err_msg = format!("{e}");
        assert!(
            !err_msg.contains("F-GT-001"),
            "Without --enforce-provenance, F-GT-001 must not trigger: {err_msg}"
        );
    }
}

#[test]
fn t_f_gt_001_enforce_provenance_allows_safetensors() {
    // SafeTensors source should pass provenance check (fail later for file not found)
    let result = run(
        "model.safetensors",
        Some(Path::new("output.apr")),
        None,
        None,
        false,
        false,
        None,
        true,  // enforce_provenance = ON
        false, // allow_no_config
    );
    // Should fail (file doesn't exist) but NOT with F-GT-001
    if let Err(e) = &result {
        let err_msg = format!("{e}");
        assert!(
            !err_msg.contains("F-GT-001"),
            "SafeTensors must pass provenance check: {err_msg}"
        );
    }
}

// =========================================================================
// Source parsing tests (via derive_output_path)
// =========================================================================

#[test]
fn test_source_parse_huggingface_basic() {
    let source = Source::parse("hf://openai/whisper-tiny").unwrap();
    match source {
        Source::HuggingFace { org, repo, file } => {
            assert_eq!(org, "openai");
            assert_eq!(repo, "whisper-tiny");
            assert!(file.is_none());
        }
        _ => panic!("Expected HuggingFace source"),
    }
}

#[test]
fn test_source_parse_huggingface_with_file() {
    let source = Source::parse("hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/model.gguf").unwrap();
    match source {
        Source::HuggingFace { org, repo, file } => {
            assert_eq!(org, "Qwen");
            assert_eq!(repo, "Qwen2.5-0.5B-Instruct-GGUF");
            assert_eq!(file, Some("model.gguf".to_string()));
        }
        _ => panic!("Expected HuggingFace source"),
    }
}

#[test]
fn test_source_parse_local() {
    let source = Source::parse("/path/to/model.safetensors").unwrap();
    match source {
        Source::Local(path) => {
            assert_eq!(path, PathBuf::from("/path/to/model.safetensors"));
        }
        _ => panic!("Expected Local source"),
    }
}

#[test]
fn test_source_parse_url() {
    let source = Source::parse("https://example.com/model.gguf").unwrap();
    match source {
        Source::Url(url) => {
            assert_eq!(url, "https://example.com/model.gguf");
        }
        _ => panic!("Expected URL source"),
    }
}

// =========================================================================
// GH-267: PyTorch format detection tests
// =========================================================================

#[test]
fn t_gh267_is_pytorch_magic_zip() {
    let magic = *b"PK\x03\x04";
    assert!(is_pytorch_magic(&magic));
}

#[test]
fn t_gh267_is_pytorch_magic_pickle_v2() {
    let magic = [0x80, 0x02, 0x00, 0x00];
    assert!(is_pytorch_magic(&magic));
}

#[test]
fn t_gh267_is_pytorch_magic_pickle_v5() {
    let magic = [0x80, 0x05, 0x00, 0x00];
    assert!(is_pytorch_magic(&magic));
}

#[test]
fn t_gh267_not_pytorch_gguf() {
    let magic = *b"GGUF";
    assert!(!is_pytorch_magic(&magic));
}

#[test]
fn t_gh267_not_pytorch_apr() {
    let magic = *b"APR\0";
    assert!(!is_pytorch_magic(&magic));
}

#[test]
fn t_gh267_reject_remote_bin_file() {
    // Remote .bin files should be rejected based on extension
    let result = reject_pytorch_format("hf://openai/whisper-tiny/pytorch_model.bin");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("GH-267"), "Error must cite GH-267: {err}");
    assert!(err.contains("SafeTensors"), "Error must suggest conversion: {err}");
}

#[test]
fn t_gh267_reject_pt_extension() {
    let result = reject_pytorch_format("model.pt");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("GH-267"));
}

#[test]
fn t_gh267_allow_safetensors() {
    let result = reject_pytorch_format("model.safetensors");
    assert!(result.is_ok());
}

#[test]
fn t_gh267_allow_gguf() {
    let result = reject_pytorch_format("model.gguf");
    assert!(result.is_ok());
}

#[test]
fn t_gh267_allow_apr() {
    let result = reject_pytorch_format("model.apr");
    assert!(result.is_ok());
}

#[test]
fn t_gh267_local_bin_with_pytorch_magic() {
    // Create a temp file with PyTorch ZIP magic
    let dir = tempfile::tempdir().expect("tempdir");
    let bin_path = dir.path().join("model.bin");
    std::fs::write(&bin_path, b"PK\x03\x04rest_of_zip_data").expect("write");
    let result = reject_pytorch_format(bin_path.to_str().expect("path"));
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("GH-267"));
}

#[test]
fn t_gh267_local_bin_not_pytorch() {
    // A .bin file that doesn't have PyTorch magic should be allowed through
    let dir = tempfile::tempdir().expect("tempdir");
    let bin_path = dir.path().join("model.bin");
    std::fs::write(&bin_path, b"\x00\x00\x00\x08{\"__metadata__\":{}}").expect("write");
    let result = reject_pytorch_format(bin_path.to_str().expect("path"));
    assert!(result.is_ok());
}

#[test]
fn t_gh267_run_rejects_pytorch_bin() {
    let result = run(
        "pytorch_model.bin",
        Some(Path::new("output.apr")),
        None,
        None,
        false,
        false,
        None,
        false,
        false,
    );
    // Should fail with GH-267 PyTorch error (not file-not-found or other errors)
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("GH-267"),
        "Expected GH-267 PyTorch detection error, got: {err}"
    );
}
