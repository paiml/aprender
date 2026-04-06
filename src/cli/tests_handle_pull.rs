//! T-COV-95 Deep Coverage Bridge: cli/handlers.rs + cli/mod.rs
//!
//! Targets: handle_pull, handle_push, handle_list paths, handle_serve stub,
//! validate_model_path, ServeConfig/RunConfig construction, parse_trace_config.

use crate::cli::handlers::*;
use clap::Parser;

// ============================================================================
// handle_pull paths
// ============================================================================

#[tokio::test]
async fn test_handle_pull_default_registry() {
    let result = handle_pull("llama3:8b", false, None).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("registry") || err.contains("Registry"), "got: {err}");
}

#[tokio::test]
async fn test_handle_pull_force_download() {
    let result = handle_pull("llama3:8b", true, None).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("registry") || err.contains("Registry"), "got: {err}");
}

#[tokio::test]
async fn test_handle_pull_with_quantize() {
    let result = handle_pull("model:latest", false, Some("q4_k")).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("registry") || err.contains("Registry"), "got: {err}");
}

#[tokio::test]
async fn test_handle_pull_hf_prefix() {
    let result = handle_pull("hf://Qwen/Qwen2-0.5B", false, None).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("registry") || err.contains("Registry"), "got: {err}");
}

#[tokio::test]
async fn test_handle_pull_pacha_prefix() {
    let result = handle_pull("pacha://llama3:8b-q4k", false, None).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("registry") || err.contains("Registry"), "got: {err}");
}

// ============================================================================
// handle_push paths
// ============================================================================

#[tokio::test]
async fn test_handle_push_default_target() {
    let result = handle_push("my-model:latest", None).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("registry") || err.contains("Registry"), "got: {err}");
}

#[tokio::test]
async fn test_handle_push_custom_target() {
    let result = handle_push("my-model:v1", Some("pacha://registry")).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("registry") || err.contains("Registry"), "got: {err}");
}

// ============================================================================
// handle_list paths
// ============================================================================

#[test]
fn test_handle_list_remote_url() {
    let result = handle_list(Some("https://registry.example.com"), "table");
    assert!(result.is_ok());
}

#[test]
fn test_handle_list_table_format() {
    let result = handle_list(None, "table");
    assert!(result.is_ok());
}

#[test]
fn test_handle_list_json_format_no_models() {
    let result = handle_list(None, "json");
    assert!(result.is_ok());
}

// ============================================================================
// handle_serve stub (without server feature)
// ============================================================================

#[tokio::test]
async fn test_handle_serve_stub_without_feature() {
    let config = ServeConfig {
        host: "0.0.0.0".to_string(),
        port: 9090,
        model: Some("test.gguf".to_string()),
        demo: false,
        batch: false,
        gpu: false,
        openai_api: true,
        trace: false,
        context_length: 4096,
        no_fp8_cache: false,
    };
    // Without server feature, should return error
    #[cfg(not(feature = "server"))]
    {
        let result = handle_serve(config).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("server") || err.contains("Server") || err.contains("feature"),
            "got: {err}"
        );
    }
    // With server feature, just verify config creation doesn't panic
    #[cfg(feature = "server")]
    {
        let _ = config;
    }
}

// ============================================================================
// validate_model_path
// ============================================================================

#[test]
fn test_validate_model_path_nonexistent() {
    let result = validate_model_path("/nonexistent/path/model.gguf");
    assert!(result.is_err());
}

#[test]
fn test_validate_model_path_directory() {
    let result = validate_model_path("/tmp");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not a file") || err.contains("file"),
        "got: {err}"
    );
}

#[test]
fn test_validate_model_path_valid_file() {
    let path = "/tmp/test_validate_handler_cov95.txt";
    std::fs::write(path, b"test data").unwrap();
    let result = validate_model_path(path);
    let _ = std::fs::remove_file(path);
    assert!(result.is_ok());
}

// ============================================================================
// ServeConfig construction
// ============================================================================

#[test]
fn test_serve_config_demo_mode() {
    let config = ServeConfig {
        host: "127.0.0.1".to_string(),
        port: 8080,
        model: None,
        demo: true,
        batch: false,
        gpu: false,
        openai_api: true,
        trace: false,
        context_length: 4096,
        no_fp8_cache: false,
    };
    assert!(config.demo);
    assert!(config.model.is_none());
    assert!(!config.batch);
    assert!(!config.gpu);
}

#[test]
fn test_serve_config_model_mode() {
    let config = ServeConfig {
        host: "0.0.0.0".to_string(),
        port: 3000,
        model: Some("model.gguf".to_string()),
        demo: false,
        batch: true,
        gpu: true,
        openai_api: true,
        trace: false,
        context_length: 4096,
        no_fp8_cache: false,
    };
    assert!(!config.demo);
    assert_eq!(config.model.as_deref(), Some("model.gguf"));
    assert!(config.batch);
    assert!(config.gpu);
    assert_eq!(config.port, 3000);
}

// ============================================================================
// RunConfig construction
// ============================================================================

#[test]
fn test_run_config_minimal() {
    let config = RunConfig {
        model: "model.gguf".to_string(),
        prompt: None,
        max_tokens: 32,
        temperature: 0.0,
        format: "text".to_string(),
        system: None,
        raw: false,
        gpu: false,
        verbose: false,
        trace: None,
    };
    assert_eq!(config.model, "model.gguf");
    assert!(config.prompt.is_none());
    assert!(!config.raw);
}

#[test]
fn test_run_config_with_all_options() {
    let config = RunConfig {
        model: "hf://Qwen/Qwen2-0.5B".to_string(),
        prompt: Some("Hello, world!".to_string()),
        max_tokens: 512,
        temperature: 0.8,
        format: "json".to_string(),
        system: Some("You are helpful.".to_string()),
        raw: true,
        gpu: true,
        verbose: true,
        trace: Some(Some("attention,ffn".to_string())),
    };
    assert_eq!(config.max_tokens, 512);
    assert!(config.raw);
    assert!(config.gpu);
    assert!(config.verbose);
    assert!(config.system.is_some());
}

// ============================================================================
// parse_trace_config
// ============================================================================

#[test]
fn test_parse_trace_config_none() {
    let result = parse_trace_config(None);
    assert!(result.is_none());
}

#[test]
fn test_parse_trace_config_enabled_no_steps() {
    let result = parse_trace_config(Some(None));
    assert!(result.is_some());
    let config = result.unwrap();
    assert!(config.verbose);
}

#[test]
fn test_parse_trace_config_with_steps() {
    let result = parse_trace_config(Some(Some("attention,ffn,norm".to_string())));
    assert!(result.is_some());
    let config = result.unwrap();
    assert!(config.verbose);
    assert!(!config.steps.is_empty());
}

#[test]
fn test_parse_trace_config_single_step() {
    let result = parse_trace_config(Some(Some("attention".to_string())));
    assert!(result.is_some());
    let config = result.unwrap();
    assert!(config.verbose);
}

#[test]
fn test_parse_trace_config_empty_string() {
    let result = parse_trace_config(Some(Some(String::new())));
    assert!(result.is_some());
}

// ============================================================================
// Cli parsing coverage
// ============================================================================

#[test]
fn test_cli_parse_pull_with_force() {
    let cli = Cli::try_parse_from(["realizar", "pull", "llama3:8b", "--force"]).unwrap();
    match cli.command {
        Commands::Pull { model, force, .. } => {
            assert_eq!(model, "llama3:8b");
            assert!(force);
        },
        _ => panic!("Expected Pull command"),
    }
}

#[test]
fn test_cli_parse_push() {
    let cli = Cli::try_parse_from(["realizar", "push", "my-model:v1"]).unwrap();
    match cli.command {
        Commands::Push { model, .. } => {
            assert_eq!(model, "my-model:v1");
        },
        _ => panic!("Expected Push command"),
    }
}

#[test]
fn test_cli_parse_run_with_all_flags() {
    let cli = Cli::try_parse_from([
        "realizar",
        "run",
        "model.gguf",
        "test prompt",
        "-n",
        "64",
        "-t",
        "0.5",
        "--format",
        "json",
        "--raw",
        "--gpu",
        "--verbose",
    ])
    .unwrap();
    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            format,
            raw,
            gpu,
            verbose,
            ..
        } => {
            assert_eq!(model, "model.gguf");
            assert_eq!(prompt, Some("test prompt".to_string()));
            assert_eq!(max_tokens, 64);
            assert!((temperature - 0.5).abs() < f32::EPSILON);
            assert_eq!(format, "json");
            assert!(raw);
            assert!(gpu);
            assert!(verbose);
        },
        _ => panic!("Expected Run command"),
    }
}

#[test]
fn test_cli_parse_chat_with_system() {
    let cli =
        Cli::try_parse_from(["realizar", "chat", "model.gguf", "-s", "You are helpful"]).unwrap();
    match cli.command {
        Commands::Chat { model, system, .. } => {
            assert_eq!(model, "model.gguf");
            assert_eq!(system, Some("You are helpful".to_string()));
        },
        _ => panic!("Expected Chat command"),
    }
}

#[test]
fn test_cli_parse_serve_batch_mode() {
    let cli = Cli::try_parse_from([
        "realizar",
        "serve",
        "--model",
        "model.gguf",
        "--batch",
        "--port",
        "3000",
    ])
    .unwrap();
    match cli.command {
        Commands::Serve {
            model, batch, port, ..
        } => {
            assert_eq!(model, Some("model.gguf".to_string()));
            assert!(batch);
            assert_eq!(port, 3000);
        },
        _ => panic!("Expected Serve command"),
    }
}

#[test]
fn test_cli_parse_list_remote() {
    let cli = Cli::try_parse_from(["realizar", "list", "--remote", "https://example.com"]).unwrap();
    match cli.command {
        Commands::List { remote, .. } => {
            assert_eq!(remote, Some("https://example.com".to_string()));
        },
        _ => panic!("Expected List command"),
    }
}

#[test]
fn test_cli_parse_bench_compare() {
    let cli =
        Cli::try_parse_from(["realizar", "bench-compare", "base.json", "current.json"]).unwrap();
    match cli.command {
        Commands::BenchCompare { file1, file2, .. } => {
            assert_eq!(file1, "base.json");
            assert_eq!(file2, "current.json");
        },
        _ => panic!("Expected BenchCompare command"),
    }
}

// ============================================================================
// Debug trait implementations
// ============================================================================

#[test]
fn test_serve_config_debug() {
    let config = ServeConfig {
        host: "localhost".to_string(),
        port: 8080,
        model: None,
        demo: true,
        batch: false,
        gpu: false,
        openai_api: true,
        trace: false,
        context_length: 4096,
        no_fp8_cache: false,
    };
    let debug = format!("{config:?}");
    assert!(debug.contains("localhost"));
    assert!(debug.contains("8080"));
}

#[test]
fn test_run_config_debug() {
    let config = RunConfig {
        model: "test.gguf".to_string(),
        prompt: Some("hello".to_string()),
        max_tokens: 32,
        temperature: 0.0,
        format: "text".to_string(),
        system: None,
        raw: false,
        gpu: false,
        verbose: false,
        trace: None,
    };
    let debug = format!("{config:?}");
    assert!(debug.contains("test.gguf"));
    assert!(debug.contains("hello"));
}
