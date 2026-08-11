use super::*;
use aprender::format::converter::QuantizationType;

// =========================================================================
// derive_output_path() tests
// =========================================================================

#[test]
fn test_derive_output_path_hf_repo() {
    let result = derive_output_path("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct").expect("5B-Instruct'");
    assert_eq!(result, PathBuf::from("Qwen2.5-Coder-1.5B-Instruct.apr"));
}

#[test]
fn test_derive_output_path_hf_with_file() {
    let result = derive_output_path("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model-q4k.gguf")
        .expect("gguf'");
    assert_eq!(result, PathBuf::from("model-q4k.apr"));
}

#[test]
fn test_derive_output_path_local_gguf() {
    let result = derive_output_path("/path/to/model.gguf").expect("gguf'");
    assert_eq!(result, PathBuf::from("model.apr"));
}

#[test]
fn test_derive_output_path_local_safetensors() {
    let result = derive_output_path("model.safetensors").expect("safetensors'");
    assert_eq!(result, PathBuf::from("model.apr"));
}

#[test]
fn test_derive_output_path_url() {
    let result = derive_output_path("https://example.com/models/qwen-1.5b.gguf").expect("gguf'");
    assert_eq!(result, PathBuf::from("qwen-1.5b.apr"));
}

#[test]
fn test_derive_output_path_url_no_extension() {
    let result =
        derive_output_path("https://example.com/models/mymodel").expect("com/models/mymodel'");
    assert_eq!(result, PathBuf::from("mymodel.apr"));
}

#[test]
fn test_derive_output_path_hf_nested_file() {
    let result = derive_output_path("hf://openai/whisper-tiny/pytorch_model.bin").expect("bin'");
    assert_eq!(result, PathBuf::from("pytorch_model.apr"));
}

#[test]
fn test_derive_output_path_relative_path() {
    let result = derive_output_path("./models/test.safetensors").expect("safetensors'");
    assert_eq!(result, PathBuf::from("test.apr"));
}

// =========================================================================
// parse_quantize() tests
// =========================================================================

#[test]
fn test_parse_quantize_none() {
    let result = parse_quantize(None).expect("value");
    assert!(result.is_none());
}

#[test]
fn test_parse_quantize_int8() {
    let result = parse_quantize(Some("int8")).expect("value");
    assert_eq!(result, Some(QuantizationType::Int8));
}

#[test]
fn test_parse_quantize_int4() {
    let result = parse_quantize(Some("int4")).expect("value");
    assert_eq!(result, Some(QuantizationType::Int4));
}

#[test]
fn test_parse_quantize_fp16() {
    let result = parse_quantize(Some("fp16")).expect("value");
    assert_eq!(result, Some(QuantizationType::Fp16));
}

#[test]
fn test_parse_quantize_q4k() {
    let result = parse_quantize(Some("q4k")).expect("value");
    assert_eq!(result, Some(QuantizationType::Q4K));
}

#[test]
fn test_parse_quantize_q4_k_underscore() {
    let result = parse_quantize(Some("q4_k")).expect("value");
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
        false, // json
    );

    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Unknown architecture"));
            assert!(msg.contains("Supported: whisper, llama, bert, qwen2, qwen3, qwen3_5, gpt2, starcoder, gpt-neox, opt, phi, gemma, falcon, mamba, t5, auto"));
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
        false, // json
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
        false, // json
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
        false, // json
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
        false, // json
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
        false, // json
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
        false, // json
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
        false, // json
    );

    // Will fail at network stage
    assert!(result.is_err());
}

#[test]
fn test_run_invalid_source() {
    // Non-existent source should return Err (empty string panics on contract)
    let result = run(
        "/nonexistent/model.gguf",
        Some(Path::new("output.apr")),
        None,
        None,
        false,
        false,
        None,
        false, // enforce_provenance
        false, // allow_no_config
        false, // json
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
        false, // json
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
        false, // json
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
        false, // json
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
        false, // json
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
    let source = Source::parse("hf://openai/whisper-tiny").expect("value");
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
    let source = Source::parse("hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/model.gguf").expect("gguf'");
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
    let source = Source::parse("/path/to/model.safetensors").expect("safetensors'");
    match source {
        Source::Local(path) => {
            assert_eq!(path, PathBuf::from("/path/to/model.safetensors"));
        }
        _ => panic!("Expected Local source"),
    }
}

#[test]
fn test_source_parse_url() {
    let source = Source::parse("https://example.com/model.gguf").expect("gguf'");
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

// =========================================================================
// `--json` must emit JSON, not human-formatted text.
//
// apr 0.63.0 (from crates.io) never forwarded the global `--json` flag into
// `import::run`, so `apr import model.safetensors -o out.apr --json` printed
// the "APR Import Pipeline" banner and the aligned key/value tables, and a
// consumer piping it to a parser got
// `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`.
// =========================================================================

fn import_description_fixture() -> ImportDescription {
    ImportDescription {
        source: "Local file: model.safetensors".to_string(),
        output: "out.apr".to_string(),
        architecture: "Qwen2".to_string(),
        validation: "Basic".to_string(),
        quantize: None,
    }
}

#[test]
fn test_import_json_stdout_parses_as_json() {
    let stdout = import_json_stdout(&import_description_fixture(), 97, "A", true);

    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!(
            "`apr import --json` must write parseable JSON to stdout, but a consumer \
             got {e}. Actual stdout was:\n{stdout}"
        )
    });
    assert_eq!(parsed["output"], "out.apr");
    assert_eq!(parsed["architecture"], "Qwen2");
    assert_eq!(parsed["score"], 97);
    assert_eq!(parsed["grade"], "A");
    assert_eq!(parsed["status"], "ok");
    assert_eq!(parsed["quantize"], serde_json::Value::Null);
}

#[test]
fn test_import_json_stdout_carries_no_human_decoration() {
    let stdout = import_json_stdout(&import_description_fixture(), 60, "D", false);

    // These are the strings apr 0.63.0 put on stdout ahead of any document.
    for leak in [
        "APR Import Pipeline",
        "Validation Report",
        "Import successful",
        "Import completed with warnings",
    ] {
        assert!(
            !stdout.contains(leak),
            "human decoration {leak:?} leaked into `--json` stdout:\n{stdout}"
        );
    }
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("warning path must still be one JSON document");
    assert_eq!(parsed["status"], "warnings");
}

#[test]
fn test_import_json_stdout_reports_quantization_when_requested() {
    let mut describe = import_description_fixture();
    describe.quantize = Some("Int8".to_string());
    let stdout = import_json_stdout(&describe, 91, "A", true);

    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("quantized import must still be one JSON document");
    assert_eq!(parsed["quantize"], "Int8");
}

#[cfg(feature = "inference")]
#[test]
fn test_q4k_import_json_stdout_parses_as_json() {
    let stats = realizar::convert::Q4KConversionStats {
        tensor_count: 291,
        q4k_tensor_count: 197,
        total_bytes: 1_073_741_824,
        architecture: "qwen2".to_string(),
        num_layers: 28,
        hidden_size: 1536,
    };
    let stdout = q4k_import_json_stdout("model.gguf", "model.apr", &stats);

    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!(
            "`apr import --preserve-q4k --json` must write parseable JSON to stdout, \
             but a consumer got {e}. Actual stdout was:\n{stdout}"
        )
    });
    assert_eq!(parsed["mode"], "q4k");
    assert_eq!(parsed["q4k_tensor_count"], 197);
    assert_eq!(parsed["architecture"], "qwen2");
    assert!(
        !stdout.contains("Q4K Import Report"),
        "human subheader leaked into `--json` stdout:\n{stdout}"
    );
}

include!("import_tests_include_01.rs");
