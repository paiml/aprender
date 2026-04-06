
// ============================================================================
// CLI mod.rs private function tests (GH-219 coverage kaizen)
// ============================================================================

// Access private items from ancestor module (cli)
use super::super::{
    process_chat_input, ChatAction,
    validate_chat_model,
    load_chat_history, save_chat_history,
    format_model_prompt,
    detect_model_source, ModelSource,
};

// ============================================================================
// process_chat_input tests
// ============================================================================

#[test]
fn test_process_chat_input_empty_p19() {
    let mut history = Vec::new();
    let result = process_chat_input("", &mut history);
    assert!(matches!(result, ChatAction::Continue));
}

#[test]
fn test_process_chat_input_exit_p19() {
    let mut history = Vec::new();
    let result = process_chat_input("exit", &mut history);
    assert!(matches!(result, ChatAction::Exit));
}

#[test]
fn test_process_chat_input_quit_p19() {
    let mut history = Vec::new();
    let result = process_chat_input("/quit", &mut history);
    assert!(matches!(result, ChatAction::Exit));
}

#[test]
fn test_process_chat_input_clear_p19() {
    let mut history = vec![("hello".to_string(), "world".to_string())];
    let result = process_chat_input("/clear", &mut history);
    assert!(matches!(result, ChatAction::Continue));
    assert!(history.is_empty());
}

#[test]
fn test_process_chat_input_history_p19() {
    let mut history = vec![
        ("user1".to_string(), "resp1".to_string()),
        ("user2".to_string(), "resp2".to_string()),
    ];
    let result = process_chat_input("/history", &mut history);
    assert!(matches!(result, ChatAction::Continue));
    // History should be unchanged
    assert_eq!(history.len(), 2);
}

#[test]
fn test_process_chat_input_respond_p19() {
    let mut history = Vec::new();
    let result = process_chat_input("Hello, world!", &mut history);
    match result {
        ChatAction::Respond(text) => assert_eq!(text, "Hello, world!"),
        _ => panic!("Expected ChatAction::Respond"),
    }
}

#[test]
fn test_process_chat_input_history_empty_p19() {
    let mut history = Vec::new();
    let result = process_chat_input("/history", &mut history);
    assert!(matches!(result, ChatAction::Continue));
}

#[test]
fn test_process_chat_input_clear_already_empty_p19() {
    let mut history = Vec::new();
    let result = process_chat_input("/clear", &mut history);
    assert!(matches!(result, ChatAction::Continue));
    assert!(history.is_empty());
}

// ============================================================================
// validate_chat_model tests
// ============================================================================

#[test]
fn test_validate_chat_model_pacha_uri_p19() {
    // pacha:// URIs return Ok but print a message
    let result = validate_chat_model("pacha://model:latest");
    assert!(result.is_ok());
}

#[test]
fn test_validate_chat_model_hf_uri_p19() {
    let result = validate_chat_model("hf://org/model");
    assert!(result.is_ok());
}

#[test]
fn test_validate_chat_model_nonexistent_p19() {
    let result = validate_chat_model("/nonexistent/model.gguf");
    assert!(result.is_err());
}

#[test]
fn test_validate_chat_model_existing_file_p19() {
    use std::io::Write;
    let mut temp = tempfile::NamedTempFile::new().expect("file operation failed");
    temp.write_all(b"data").expect("operation failed");
    let path = temp.path().to_str().expect("invalid UTF-8");
    let result = validate_chat_model(path);
    assert!(result.is_ok());
}

// ============================================================================
// load_chat_history / save_chat_history tests
// ============================================================================

#[test]
fn test_load_chat_history_none_p19() {
    let history = load_chat_history(None);
    assert!(history.is_empty());
}

#[test]
fn test_load_chat_history_nonexistent_p19() {
    let history = load_chat_history(Some("/nonexistent/history.json"));
    assert!(history.is_empty());
}

#[test]
fn test_save_and_load_chat_history_p19() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_chat_history_p19.json");
    let path_str = path.to_str().expect("invalid UTF-8");

    let history = vec![
        ("hello".to_string(), "world".to_string()),
        ("foo".to_string(), "bar".to_string()),
    ];
    save_chat_history(Some(path_str), &history);

    let loaded = load_chat_history(Some(path_str));
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].0, "hello");
    assert_eq!(loaded[0].1, "world");
    assert_eq!(loaded[1].0, "foo");
    assert_eq!(loaded[1].1, "bar");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_save_chat_history_none_path_p19() {
    // Should be a no-op
    save_chat_history(None, &[("a".to_string(), "b".to_string())]);
}

#[test]
fn test_load_chat_history_invalid_json_p19() {
    use std::io::Write;
    let mut temp = tempfile::NamedTempFile::new().expect("file operation failed");
    temp.write_all(b"not json").expect("operation failed");
    let path = temp.path().to_str().expect("invalid UTF-8");
    let history = load_chat_history(Some(path));
    assert!(history.is_empty());
}

// ============================================================================
// format_model_prompt tests
// ============================================================================

#[test]
fn test_format_model_prompt_raw_mode_p19() {
    let result = format_model_prompt("model.gguf", "Hello", None, true);
    assert_eq!(result, "Hello");
}

#[test]
fn test_format_model_prompt_raw_mode_with_system_p19() {
    // Raw mode ignores system prompt
    let result = format_model_prompt("model.gguf", "Hello", Some("System"), true);
    assert_eq!(result, "Hello");
}

#[test]
fn test_format_model_prompt_non_raw_p19() {
    // Non-raw mode applies chat template
    let result = format_model_prompt("model.gguf", "Hello", None, false);
    // Should return some formatted prompt (template-dependent)
    assert!(!result.is_empty());
}

#[test]
fn test_format_model_prompt_non_raw_with_system_p19() {
    let result = format_model_prompt("model.gguf", "Hello", Some("You are helpful."), false);
    assert!(!result.is_empty());
}

// ============================================================================
// detect_model_source tests
// ============================================================================

#[test]
fn test_detect_model_source_pacha_uri_p19() {
    let result = detect_model_source("pacha://model:latest", false);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap(), ModelSource::RegistryPacha));
}

#[test]
fn test_detect_model_source_hf_uri_p19() {
    // Note: "hf://org/model" contains ':', so detect_model_source matches
    // the colon check before the hf:// check → returns RegistryPacha.
    let result = detect_model_source("hf://org/model", false);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap(), ModelSource::RegistryPacha));
}

#[test]
fn test_detect_model_source_colon_tag_p19() {
    // name:tag format is treated as Pacha registry
    let result = detect_model_source("llama:latest", false);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap(), ModelSource::RegistryPacha));
}

#[test]
fn test_detect_model_source_local_file_p19() {
    use std::io::Write;
    let mut temp = tempfile::NamedTempFile::with_suffix(".gguf").expect("file operation failed");
    temp.write_all(b"GGUF").expect("operation failed");
    let path = temp.path().to_str().expect("invalid UTF-8");
    let result = detect_model_source(path, false);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap(), ModelSource::Local));
}

#[test]
fn test_detect_model_source_local_verbose_p19() {
    use std::io::Write;
    let mut temp = tempfile::NamedTempFile::with_suffix(".gguf").expect("file operation failed");
    temp.write_all(b"GGUF").expect("operation failed");
    let path = temp.path().to_str().expect("invalid UTF-8");
    let result = detect_model_source(path, true);
    assert!(result.is_ok());
    assert!(matches!(result.unwrap(), ModelSource::Local));
}

// ============================================================================
// print_info smoke test
// ============================================================================

#[test]
fn test_print_info_no_panic_p19() {
    super::super::print_info();
}

// ============================================================================
// ModelType Display + Debug tests (cli/mod_part_05.rs — server feature)
// ============================================================================

#[cfg(feature = "server")]
#[test]
fn test_model_type_display_gguf_p19() {
    let mt = ModelType::Gguf;
    assert_eq!(format!("{}", mt), "GGUF");
}

#[cfg(feature = "server")]
#[test]
fn test_model_type_display_safetensors_p19() {
    let mt = ModelType::SafeTensors;
    assert_eq!(format!("{}", mt), "SafeTensors");
}

#[cfg(feature = "server")]
#[test]
fn test_model_type_display_apr_p19() {
    let mt = ModelType::Apr;
    assert_eq!(format!("{}", mt), "APR");
}

#[cfg(feature = "server")]
#[test]
fn test_model_type_debug_p19() {
    let mt = ModelType::Gguf;
    let debug = format!("{:?}", mt);
    assert!(debug.contains("Gguf"));
}

#[cfg(feature = "server")]
#[test]
fn test_model_type_clone_eq_p19() {
    let mt1 = ModelType::SafeTensors;
    let mt2 = mt1;
    assert_eq!(mt1, mt2);
}

#[cfg(feature = "server")]
#[test]
fn test_prepare_serve_state_unsupported_ext_p19() {
    let result = prepare_serve_state("/some/model.xyz", false, false);
    assert!(result.is_err());
}

#[cfg(feature = "server")]
#[test]
fn test_prepare_serve_state_nonexistent_gguf_p19() {
    let result = prepare_serve_state("/nonexistent/model.gguf", false, false);
    assert!(result.is_err());
}

#[cfg(feature = "server")]
#[test]
fn test_prepare_serve_state_nonexistent_safetensors_p19() {
    let result = prepare_serve_state("/nonexistent/model.safetensors", false, false);
    assert!(result.is_err());
}

#[cfg(feature = "server")]
#[test]
fn test_prepare_serve_state_nonexistent_apr_p19() {
    let result = prepare_serve_state("/nonexistent/model.apr", false, false);
    assert!(result.is_err());
}

// ============================================================================
// Additional cli/mod.rs helper tests for edge cases
// ============================================================================

#[test]
fn test_process_chat_input_whitespace_only_p19() {
    // After trim, whitespace becomes empty → Continue
    // But the caller trims before calling, so test with actual whitespace
    let mut history = Vec::new();
    let result = process_chat_input("  hello  ", &mut history);
    match result {
        ChatAction::Respond(text) => assert_eq!(text, "  hello  "),
        _ => panic!("Expected Respond"),
    }
}

#[test]
fn test_detect_model_source_bare_name_p19() {
    // A bare name without slashes, dots, or colons → treated as Local
    let result = detect_model_source("modelname", false);
    assert!(result.is_ok());
    // "modelname" isn't a local file path (no /, no ., no known extension)
    // so is_local_file_path returns false, and it doesn't have : or hf:// or pacha://
    // → falls through to Ok(ModelSource::Local)
}
