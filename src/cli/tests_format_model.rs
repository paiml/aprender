//! T-COV-95 Deep Coverage: CLI mod.rs pure functions (Part 11)
//!
//! Tests format_model_prompt, process_chat_input, detect_model_source,
//! parse_bench_line, parse_cargo_bench_output, validate_suite_or_error,
//! load_chat_history, and print_bench_config.

#![allow(clippy::needless_pass_by_value)]

use super::*;

// ============================================================================
// format_model_prompt tests
// ============================================================================

#[test]
fn test_format_model_prompt_raw_mode_returns_verbatim() {
    let result = format_model_prompt("model.gguf", "Hello world", None, true);
    assert_eq!(result, "Hello world");
}

#[test]
fn test_format_model_prompt_raw_mode_with_system_returns_verbatim() {
    let result = format_model_prompt("model.gguf", "Hello", Some("Be helpful"), true);
    assert_eq!(result, "Hello");
}

#[test]
fn test_format_model_prompt_raw_mode_empty_prompt() {
    let result = format_model_prompt("model.gguf", "", None, true);
    assert_eq!(result, "");
}

#[test]
fn test_format_model_prompt_non_raw_returns_string() {
    // Non-raw mode returns formatted prompt (could be original or templated)
    let result = format_model_prompt("unknown_model.gguf", "Hi", None, false);
    assert!(!result.is_empty());
}

#[test]
fn test_format_model_prompt_non_raw_with_system_prompt() {
    let result = format_model_prompt("unknown_model.gguf", "Hi", Some("You are helpful"), false);
    assert!(!result.is_empty());
}

// ============================================================================
// process_chat_input tests
// ============================================================================

#[test]
fn test_process_chat_input_empty_returns_continue() {
    let mut history = Vec::new();
    match process_chat_input("", &mut history) {
        ChatAction::Continue => {},
        _ => panic!("Expected Continue for empty input"),
    }
}

#[test]
fn test_process_chat_input_exit_returns_exit() {
    let mut history = Vec::new();
    match process_chat_input("exit", &mut history) {
        ChatAction::Exit => {},
        _ => panic!("Expected Exit for 'exit'"),
    }
}

#[test]
fn test_process_chat_input_quit_returns_exit() {
    let mut history = Vec::new();
    match process_chat_input("/quit", &mut history) {
        ChatAction::Exit => {},
        _ => panic!("Expected Exit for '/quit'"),
    }
}

#[test]
fn test_process_chat_input_clear_clears_history() {
    let mut history = vec![("hello".to_string(), "world".to_string())];
    match process_chat_input("/clear", &mut history) {
        ChatAction::Continue => {
            assert!(history.is_empty());
        },
        _ => panic!("Expected Continue for '/clear'"),
    }
}

#[test]
fn test_process_chat_input_history_returns_continue() {
    let mut history = vec![("q1".to_string(), "a1".to_string())];
    match process_chat_input("/history", &mut history) {
        ChatAction::Continue => {
            // History should remain unchanged
            assert_eq!(history.len(), 1);
        },
        _ => panic!("Expected Continue for '/history'"),
    }
}

#[test]
fn test_process_chat_input_normal_text_returns_respond() {
    let mut history = Vec::new();
    match process_chat_input("What is ML?", &mut history) {
        ChatAction::Respond(text) => {
            assert_eq!(text, "What is ML?");
        },
        _ => panic!("Expected Respond for normal text"),
    }
}

#[test]
fn test_process_chat_input_history_empty() {
    let mut history: Vec<(String, String)> = Vec::new();
    match process_chat_input("/history", &mut history) {
        ChatAction::Continue => {},
        _ => panic!("Expected Continue for '/history' with empty history"),
    }
}

// ============================================================================
// is_local_file_path tests (more edge cases)
// ============================================================================

#[test]
fn test_is_local_file_path_dot_slash() {
    assert!(is_local_file_path("./model.gguf"));
}

#[test]
fn test_is_local_file_path_absolute() {
    assert!(is_local_file_path("/home/user/model.gguf"));
}

#[test]
fn test_is_local_file_path_gguf_suffix() {
    assert!(is_local_file_path("model.gguf"));
}

#[test]
fn test_is_local_file_path_safetensors_suffix() {
    assert!(is_local_file_path("model.safetensors"));
}

#[test]
fn test_is_local_file_path_apr_suffix() {
    assert!(is_local_file_path("model.apr"));
}

#[test]
fn test_is_local_file_path_registry_uri() {
    assert!(!is_local_file_path("pacha://llama:latest"));
}

#[test]
fn test_is_local_file_path_hf_uri() {
    assert!(!is_local_file_path("hf://openai/whisper-tiny"));
}

#[test]
fn test_is_local_file_path_bare_name() {
    assert!(!is_local_file_path("llama3.2"));
}

// ============================================================================
// parse_bench_line tests
// ============================================================================

#[test]
fn test_parse_bench_line_valid() {
    let line = "test tensor_add ... bench:      1234 ns/iter (+/- 56)";
    let result = parse_bench_line(line, Some("tensor_ops"));
    assert!(result.is_some());
    let val = result.expect("should parse");
    assert_eq!(val["name"], "tensor_add");
    assert_eq!(val["time_ns"], 1234);
    assert_eq!(val["suite"], "tensor_ops");
}

#[test]
fn test_parse_bench_line_with_commas() {
    let line = "test matmul_128 ... bench:    145,300 ns/iter (+/- 200)";
    let result = parse_bench_line(line, None);
    assert!(result.is_some());
    let val = result.expect("should parse");
    assert_eq!(val["time_ns"], 145300);
}

#[test]
fn test_parse_bench_line_no_bench_keyword() {
    let line = "running 5 tests";
    let result = parse_bench_line(line, None);
    assert!(result.is_none());
}

#[test]
fn test_parse_bench_line_no_ns_iter() {
    let line = "test tensor_add ... ok (1.5s)";
    let result = parse_bench_line(line, None);
    assert!(result.is_none());
}

#[test]
fn test_parse_bench_line_too_few_parts() {
    let line = "bench: 100";
    let result = parse_bench_line(line, None);
    assert!(result.is_none());
}

#[test]
fn test_parse_bench_line_none_suite() {
    let line = "test softmax ... bench:      500 ns/iter (+/- 10)";
    let result = parse_bench_line(line, None);
    assert!(result.is_some());
    let val = result.expect("should parse");
    assert!(val["suite"].is_null());
}

// ============================================================================
// parse_cargo_bench_output tests
// ============================================================================

#[test]
fn test_parse_cargo_bench_output_empty() {
    let results = parse_cargo_bench_output("", None);
    assert!(results.is_empty());
}

#[test]
fn test_parse_cargo_bench_output_no_bench_lines() {
    let output = "running 3 tests\ntest foo ... ok\ntest bar ... ok\n";
    let results = parse_cargo_bench_output(output, None);
    assert!(results.is_empty());
}

#[test]
fn test_parse_cargo_bench_output_single_bench() {
    let output = "test tensor_add ... bench:      1234 ns/iter (+/- 56)\n";
    let results = parse_cargo_bench_output(output, Some("ops"));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["name"], "tensor_add");
    assert_eq!(results[0]["suite"], "ops");
}

#[test]
fn test_parse_cargo_bench_output_multiple_benches() {
    let output = "test add ... bench:      100 ns/iter (+/- 5)\n\
                  test mul ... bench:      200 ns/iter (+/- 10)\n\
                  test div ... bench:      300 ns/iter (+/- 15)\n";
    let results = parse_cargo_bench_output(output, None);
    assert_eq!(results.len(), 3);
}

// ============================================================================
// validate_suite_name tests
// ============================================================================

#[test]
fn test_validate_suite_name_tensor_ops() {
    assert!(validate_suite_name("tensor_ops"));
}

#[test]
fn test_validate_suite_name_inference() {
    assert!(validate_suite_name("inference"));
}

#[test]
fn test_validate_suite_name_cache() {
    assert!(validate_suite_name("cache"));
}

#[test]
fn test_validate_suite_name_tokenizer() {
    assert!(validate_suite_name("tokenizer"));
}

#[test]
fn test_validate_suite_name_quantize() {
    assert!(validate_suite_name("quantize"));
}

#[test]
fn test_validate_suite_name_lambda() {
    assert!(validate_suite_name("lambda"));
}

#[test]
fn test_validate_suite_name_comparative() {
    assert!(validate_suite_name("comparative"));
}

#[test]
fn test_validate_suite_name_invalid() {
    assert!(!validate_suite_name("nonexistent_suite"));
}

#[test]
fn test_validate_suite_name_empty() {
    assert!(!validate_suite_name(""));
}

// ============================================================================
// validate_suite_or_error tests
// ============================================================================

#[test]
fn test_validate_suite_or_error_valid() {
    assert!(validate_suite_or_error("tensor_ops"));
}

#[test]
fn test_validate_suite_or_error_invalid() {
    assert!(!validate_suite_or_error("bogus"));
}

// ============================================================================
// format_size tests (edge cases)
// ============================================================================

#[test]
fn test_format_size_zero_bytes() {
    assert_eq!(format_size(0), "0 B");
}

#[test]
fn test_format_size_one_byte() {
    assert_eq!(format_size(1), "1 B");
}

#[test]
fn test_format_size_1023_bytes() {
    assert_eq!(format_size(1023), "1023 B");
}

#[test]
fn test_format_size_exact_1kb() {
    assert_eq!(format_size(1024), "1.0 KB");
}

#[test]
fn test_format_size_exact_1mb() {
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
}

#[test]
fn test_format_size_exact_1gb() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
}

#[test]
fn test_format_size_fractional_gb() {
    let size = (2.5 * 1024.0 * 1024.0 * 1024.0) as u64;
    let result = format_size(size);
    assert!(result.contains("GB"));
    assert!(result.starts_with("2.5"));
}

#[test]
fn test_format_size_fractional_mb() {
    let size = (3.7 * 1024.0 * 1024.0) as u64;
    let result = format_size(size);
    assert!(result.contains("MB"));
}

// ============================================================================
// BENCHMARK_SUITES constant tests
// ============================================================================

#[test]
fn test_benchmark_suites_has_seven_entries() {
    assert_eq!(BENCHMARK_SUITES.len(), 7);
}

#[test]
fn test_benchmark_suites_names_unique() {
    let names: Vec<&str> = BENCHMARK_SUITES.iter().map(|(n, _)| *n).collect();
    let mut unique = names.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(names.len(), unique.len());
}

#[test]
fn test_benchmark_suites_descriptions_nonempty() {
    for (_, desc) in BENCHMARK_SUITES {
        assert!(!desc.is_empty(), "Suite description should not be empty");
    }
}

// ============================================================================
// load_chat_history tests
// ============================================================================

#[test]
fn test_load_chat_history_none() {
    let history = load_chat_history(None);
    assert!(history.is_empty());
}

#[test]
fn test_load_chat_history_nonexistent_file() {
    let history = load_chat_history(Some("/nonexistent/path/chat.json"));
    assert!(history.is_empty());
}

// ============================================================================
// home_dir tests
// ============================================================================

#[test]
fn test_home_dir_returns_some_when_home_set() {
    // HOME should be set in any test environment
    let result = home_dir();
    if std::env::var_os("HOME").is_some() {
        assert!(result.is_some());
    }
}

// ============================================================================
// detect_model_source tests (non-file paths)
// ============================================================================

#[test]
fn test_detect_model_source_pacha_uri() {
    let result = detect_model_source("pacha://model:latest", false);
    assert!(result.is_ok());
    match result.expect("should succeed") {
        ModelSource::RegistryPacha => {},
        _ => panic!("Expected RegistryPacha"),
    }
}

#[test]
fn test_detect_model_source_hf_uri() {
    let result = detect_model_source("hf://org/model", false);
    assert!(result.is_ok());
    match result.expect("should succeed") {
        ModelSource::RegistryHf => {},
        _ => panic!("Expected RegistryHf"),
    }
}

#[test]
fn test_detect_model_source_name_tag() {
    // "name:tag" format should be treated as registry
    let result = detect_model_source("llama:latest", false);
    assert!(result.is_ok());
    match result.expect("should succeed") {
        ModelSource::RegistryPacha => {},
        _ => panic!("Expected RegistryPacha for name:tag"),
    }
}
