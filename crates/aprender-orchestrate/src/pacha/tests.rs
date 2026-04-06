//! Pacha Module Tests
//!
//! Unit tests for the Pacha model registry CLI functionality.

use super::commands::PachaCommand;
use super::helpers::{
    create_progress_bar, format_size, get_cached_models, is_cached, resolve_model_ref, truncate_str,
};
use super::run::{
    generate_simulated_response, parse_simple_modelfile, truncate_context, ChatMessage,
};

// ============================================================================
// Command Parsing Tests
// ============================================================================

#[test]
fn test_resolve_model_ref_alias() {
    let resolved = resolve_model_ref("llama3", None).unwrap();
    assert!(resolved.starts_with("hf://"));
    assert!(resolved.contains("Llama-3-8B"));
}

#[test]
fn test_resolve_model_ref_with_quant() {
    let resolved = resolve_model_ref("llama3", Some("q8_0")).unwrap();
    assert!(resolved.contains("Q8_0"));
}

#[test]
fn test_resolve_model_ref_full_uri() {
    let resolved = resolve_model_ref("hf://custom/model", None).unwrap();
    assert_eq!(resolved, "hf://custom/model");
}

#[test]
fn test_resolve_model_ref_unknown() {
    let resolved = resolve_model_ref("custom-model", None).unwrap();
    assert_eq!(resolved, "pacha://custom-model");
}

// ============================================================================
// Format Tests
// ============================================================================

#[test]
fn test_format_size_bytes() {
    assert_eq!(format_size(500), "500 B");
}

#[test]
fn test_format_size_kb() {
    assert_eq!(format_size(2048), "2 KB");
}

#[test]
fn test_format_size_mb() {
    assert_eq!(format_size(5 * 1024 * 1024), "5.0 MB");
}

#[test]
fn test_format_size_gb() {
    assert_eq!(format_size(4 * 1024 * 1024 * 1024), "4.00 GB");
}

// ============================================================================
// Progress Bar Tests
// ============================================================================

#[test]
fn test_progress_bar_empty() {
    let bar = create_progress_bar(0.0, 10);
    assert!(bar.contains("          ")); // 10 spaces
}

#[test]
fn test_progress_bar_half() {
    let bar = create_progress_bar(50.0, 10);
    assert!(bar.contains("====="));
    assert!(bar.contains("     "));
}

#[test]
fn test_progress_bar_full() {
    let bar = create_progress_bar(100.0, 10);
    assert!(bar.contains("=========="));
}

// ============================================================================
// Helper Tests
// ============================================================================

#[test]
fn test_get_cached_models() {
    let models = get_cached_models();
    assert!(!models.is_empty());
    assert!(models.iter().any(|(n, _, _)| n.contains("llama")));
}

#[test]
fn test_is_cached() {
    // Currently always returns false
    assert!(!is_cached("llama3"));
}

// ============================================================================
// Command Enum Tests
// ============================================================================

#[test]
fn test_pacha_command_clone() {
    let cmd = PachaCommand::Pull { model: "llama3".to_string(), force: false, quant: None };
    let cloned = cmd.clone();
    if let PachaCommand::Pull { model, .. } = cloned {
        assert_eq!(model, "llama3");
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_pacha_command_debug() {
    let cmd = PachaCommand::List { verbose: true, format: "json".to_string() };
    let debug = format!("{:?}", cmd);
    assert!(debug.contains("List"));
    assert!(debug.contains("verbose"));
}

// ============================================================================
// Run Command Tests
// ============================================================================

#[test]
fn test_truncate_str_short() {
    let s = "hello";
    assert_eq!(truncate_str(s, 10), "hello");
}

#[test]
fn test_truncate_str_exact() {
    let s = "hello";
    assert_eq!(truncate_str(s, 5), "hello");
}

#[test]
fn test_truncate_str_long() {
    let s = "hello world this is a long string";
    let result = truncate_str(s, 15);
    assert!(result.ends_with("..."));
    assert!(result.len() <= 15);
}

#[test]
fn test_parse_simple_modelfile_system() {
    let content = "FROM llama3\nSYSTEM You are helpful.";
    let mf = parse_simple_modelfile(content).unwrap();
    assert_eq!(mf.system, Some("You are helpful.".to_string()));
}

#[test]
fn test_parse_simple_modelfile_temperature() {
    let content = "FROM llama3\nPARAMETER temperature 0.8";
    let mf = parse_simple_modelfile(content).unwrap();
    assert_eq!(mf.temperature, Some(0.8));
}

#[test]
fn test_parse_simple_modelfile_max_tokens() {
    let content = "FROM llama3\nPARAMETER max_tokens 512";
    let mf = parse_simple_modelfile(content).unwrap();
    assert_eq!(mf.max_tokens, Some(512));
}

#[test]
fn test_parse_simple_modelfile_num_predict() {
    let content = "FROM llama3\nPARAMETER num_predict 256";
    let mf = parse_simple_modelfile(content).unwrap();
    assert_eq!(mf.max_tokens, Some(256));
}

#[test]
fn test_parse_simple_modelfile_empty() {
    let content = "";
    let mf = parse_simple_modelfile(content).unwrap();
    assert!(mf.system.is_none());
    assert!(mf.temperature.is_none());
}

#[test]
fn test_parse_simple_modelfile_comments() {
    let content = "# comment\nFROM llama3\n# another comment";
    let mf = parse_simple_modelfile(content).unwrap();
    assert!(mf.system.is_none());
}

#[test]
fn test_generate_simulated_response_hello() {
    let response = generate_simulated_response("hello", &[]);
    assert!(response.contains("Hello"));
}

#[test]
fn test_generate_simulated_response_code() {
    let response = generate_simulated_response("show me some code", &[]);
    assert!(response.contains("```"));
}

#[test]
fn test_generate_simulated_response_default() {
    let response = generate_simulated_response("random query", &[]);
    assert!(response.contains("simulated"));
}

#[test]
fn test_truncate_context_small() {
    let mut messages = vec![ChatMessage::user("hi"), ChatMessage::assistant("hello")];
    truncate_context(&mut messages, 1000, false);
    assert_eq!(messages.len(), 2);
}

#[test]
fn test_truncate_context_with_system() {
    let mut messages = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("x".repeat(500)),
        ChatMessage::assistant("y".repeat(500)),
        ChatMessage::user("z".repeat(500)),
    ];
    truncate_context(&mut messages, 100, true);
    // Should keep system message
    assert_eq!(messages[0].role, "system");
}

#[test]
fn test_chat_message_clone() {
    let msg = ChatMessage::user("test");
    let cloned = msg.clone();
    assert_eq!(cloned.role, "user");
    assert_eq!(cloned.content, "test");
}

#[test]
fn test_run_command_enum() {
    let cmd = PachaCommand::Run {
        model: "llama3".to_string(),
        system: Some("You are helpful".to_string()),
        modelfile: None,
        temperature: 0.7,
        max_tokens: Some(1024),
        context: 4096,
        verbose: false,
    };
    if let PachaCommand::Run { model, temperature, .. } = cmd {
        assert_eq!(model, "llama3");
        assert!((temperature - 0.7).abs() < 0.001);
    } else {
        panic!("Expected Run command");
    }
}
