//! CLI command implementations
//!
//! This module contains all the business logic for CLI commands,
//! extracted from main.rs for testability.

// CLI glue code - relaxed lint requirements
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]

use crate::error::{RealizarError, Result};

// PMAT-802: Extracted inference runners
pub mod inference;
#[cfg(feature = "cuda")]
pub use inference::run_gguf_inference_gpu;
pub use inference::{run_apr_inference, run_gguf_inference, run_safetensors_inference};

// T-COV-001: CLI handlers extracted from main.rs for testability
pub mod handlers;
pub use handlers::{Cli, Commands, RunConfig, ServeConfig};

/// Main CLI entrypoint - dispatches commands to handlers (T-COV-001)
pub async fn entrypoint(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            format,
            system,
            raw,
            gpu,
            verbose,
            trace,
        } => {
            run_model_command(
                &model,
                prompt.as_deref(),
                max_tokens,
                temperature,
                &format,
                system.as_deref(),
                raw,
                gpu,
                verbose,
                trace,
            )
            .await
        },
        Commands::Chat {
            model,
            system,
            history,
        } => run_chat_command(&model, system.as_deref(), history.as_deref()).await,
        Commands::List { remote, format } => handlers::handle_list(remote.as_deref(), &format),
        Commands::Pull {
            model,
            force,
            quantize,
        } => handlers::handle_pull(&model, force, quantize.as_deref()).await,
        Commands::Push { model, to } => handlers::handle_push(&model, to.as_deref()).await,
        Commands::Serve {
            host,
            port,
            model,
            demo,
            openai_api,
            batch,
            gpu,
            trace,
            context_length,
            no_fp8_cache,
        } => {
            // GH-148: openai_api flag now wired through to router
            // GH-103: trace flag propagates into QuantizedGenerateConfig.trace
            handlers::handle_serve(ServeConfig {
                host,
                port,
                model,
                demo,
                batch,
                gpu,
                openai_api,
                trace,
                context_length,
                no_fp8_cache,
            })
            .await
        },
        Commands::Bench {
            suite,
            list,
            runtime,
            model,
            url,
            output,
        } => run_benchmarks(suite, list, runtime, model, url, output),
        Commands::BenchConvoy {
            runtime,
            model,
            output,
        } => run_convoy_test(runtime, model, output),
        Commands::BenchSaturation {
            runtime,
            model,
            output,
        } => run_saturation_test(runtime, model, output),
        Commands::BenchCompare {
            file1,
            file2,
            threshold,
        } => run_bench_compare(&file1, &file2, threshold),
        Commands::BenchRegression {
            baseline,
            current,
            strict,
        } => {
            if run_bench_regression(&baseline, &current, strict).is_err() {
                std::process::exit(1);
            }
            Ok(())
        },
        Commands::Viz { color, samples } => {
            run_visualization(color, samples);
            Ok(())
        },
        Commands::Info => {
            print_info();
            Ok(())
        },
    }
}

/// Format a prompt using chat template detection (unless raw mode)
fn format_model_prompt(
    model_ref: &str,
    prompt_text: &str,
    system_prompt: Option<&str>,
    raw_mode: bool,
) -> String {
    use crate::chat_template::{auto_detect_template, ChatMessage};

    if raw_mode {
        return prompt_text.to_string();
    }
    let template = auto_detect_template(model_ref);
    let mut messages = Vec::new();
    if let Some(sys) = system_prompt {
        messages.push(ChatMessage::system(sys));
    }
    messages.push(ChatMessage::user(prompt_text));
    template
        .format_conversation(&messages)
        .unwrap_or_else(|_| prompt_text.to_string())
}

/// Model source detection result
enum ModelSource {
    /// Local file, continue processing
    Local,
    /// Registry URI (pacha:// or name:tag), feature not enabled
    RegistryPacha,
    /// HuggingFace Hub (hf://), feature not enabled
    RegistryHf,
}

/// Detect and validate model source
fn detect_model_source(model_ref: &str, verbose: bool) -> Result<ModelSource> {
    if is_local_file_path(model_ref) {
        handlers::validate_model_path(model_ref)?;
        if verbose {
            println!("  Source: local file");
        }
        return Ok(ModelSource::Local);
    }
    if model_ref.starts_with("pacha://") || model_ref.contains(':') {
        return Ok(ModelSource::RegistryPacha);
    }
    if model_ref.starts_with("hf://") {
        return Ok(ModelSource::RegistryHf);
    }
    Ok(ModelSource::Local)
}

/// Setup trace environment if enabled
/// Parse trace config and set up environment
#[allow(clippy::option_option)]
fn setup_trace_config(
    trace: Option<Option<String>>,
) -> Option<crate::inference_trace::TraceConfig> {
    let trace_config = handlers::parse_trace_config(trace);
    if trace_config.is_some() {
        std::env::set_var("GPU_DEBUG", "1");
        eprintln!("[TRACE] Inference tracing enabled - GPU_DEBUG=1");
    }
    trace_config
}

/// Dispatch inference based on model format
#[allow(clippy::too_many_arguments)]
fn dispatch_inference(
    model_ref: &str,
    file_data: &[u8],
    formatted_prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    force_gpu: bool,
    verbose: bool,
    trace_config: Option<crate::inference_trace::TraceConfig>,
) -> Result<()> {
    use crate::format::{detect_format, ModelFormat};
    match detect_format(file_data).unwrap_or(ModelFormat::Gguf) {
        ModelFormat::Apr => run_apr_inference(
            model_ref,
            file_data,
            formatted_prompt,
            max_tokens,
            temperature,
            format,
            force_gpu,
            verbose,
            trace_config.clone(),
        ),
        ModelFormat::SafeTensors => run_safetensors_inference(
            model_ref,
            formatted_prompt,
            max_tokens,
            temperature,
            format,
            trace_config.clone(),
        ),
        ModelFormat::Gguf => run_gguf_inference(
            model_ref,
            file_data,
            formatted_prompt,
            max_tokens,
            temperature,
            format,
            force_gpu,
            verbose,
            trace_config,
        ),
    }
}

/// Run model command handler
#[allow(
    clippy::too_many_arguments,
    clippy::unused_async,
    clippy::option_option
)]
async fn run_model_command(
    model_ref: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    system_prompt: Option<&str>,
    raw_mode: bool,
    force_gpu: bool,
    verbose: bool,
    trace: Option<Option<String>>,
) -> Result<()> {
    // APR-TRACE-001: Parse trace config and pass through to inference
    let trace_config = setup_trace_config(trace);

    match detect_model_source(model_ref, verbose)? {
        ModelSource::Local => {},
        ModelSource::RegistryPacha => {
            println!("  Source: Pacha registry");
            println!("Enable registry support: --features registry");
            return Ok(());
        },
        ModelSource::RegistryHf => {
            println!("  Source: HuggingFace Hub");
            println!("Enable registry support: --features registry");
            return Ok(());
        },
    }

    // #170: Disabled broken Q4K APR GPU shortcut (run_apr_inference_gpu_q4k produces garbage).
    // Falls through to dispatch_inference → run_apr_inference → from_apr → OwnedQuantizedModelCuda.

    let file_data = std::fs::read(model_ref).map_err(|e| RealizarError::UnsupportedOperation {
        operation: "read_model".to_string(),
        reason: format!("Failed to read {model_ref}: {e}"),
    })?;

    if verbose {
        display_model_info(model_ref, &file_data)?;
    }

    if let Some(prompt_text) = prompt {
        let formatted_prompt = format_model_prompt(model_ref, prompt_text, system_prompt, raw_mode);
        dispatch_inference(
            model_ref,
            &file_data,
            &formatted_prompt,
            max_tokens,
            temperature,
            format,
            force_gpu,
            verbose,
            trace_config,
        )?;
    } else {
        println!("Interactive mode - use a prompt argument");
    }
    Ok(())
}

/// Result of processing a single chat input line
enum ChatAction {
    /// Continue reading input (empty line)
    Continue,
    /// Exit the chat loop
    Exit,
    /// Process the input and generate a response
    Respond(String),
}

/// Process a single chat input line into an action
fn process_chat_input(input: &str, history: &mut Vec<(String, String)>) -> ChatAction {
    if input.is_empty() {
        return ChatAction::Continue;
    }
    if input == "exit" || input == "/quit" {
        return ChatAction::Exit;
    }
    if input == "/clear" {
        history.clear();
        println!("Cleared.");
        return ChatAction::Continue;
    }
    if input == "/history" {
        for (i, (u, a)) in history.iter().enumerate() {
            println!("[{}] {}: {}", i + 1, u, a);
        }
        return ChatAction::Continue;
    }
    ChatAction::Respond(input.to_string())
}

/// Validate model reference for chat mode
fn validate_chat_model(model_ref: &str) -> Result<()> {
    if model_ref.starts_with("pacha://") || model_ref.starts_with("hf://") {
        println!("Registry URIs require --features registry");
        return Ok(());
    }
    if !std::path::Path::new(model_ref).exists() {
        return Err(RealizarError::ModelNotFound(model_ref.to_string()));
    }
    Ok(())
}

/// Load chat history from file
fn load_chat_history(history_file: Option<&str>) -> Vec<(String, String)> {
    history_file
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_default()
}

/// Save chat history to file
fn save_chat_history(history_file: Option<&str>, history: &[(String, String)]) {
    if let Some(path) = history_file {
        if let Ok(json) = serde_json::to_string_pretty(&history) {
            let _ = std::fs::write(path, json);
        }
    }
}

/// Run the chat input loop
fn run_chat_loop(history: &mut Vec<(String, String)>) {
    use std::io::{BufRead, Write};

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    loop {
        print!(">>> ");
        stdout.flush().ok();
        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) | Err(_) => break,
            Ok(_) => match process_chat_input(input.trim(), history) {
                ChatAction::Exit => break,
                ChatAction::Respond(user_input) => {
                    let response = format!("[Echo] {}", user_input);
                    println!("{response}");
                    history.push((user_input, response));
                },
                ChatAction::Continue => {},
            },
        }
    }
}

/// Chat command handler
#[allow(clippy::unused_async)]
async fn run_chat_command(
    model_ref: &str,
    system_prompt: Option<&str>,
    history_file: Option<&str>,
) -> Result<()> {
    validate_chat_model(model_ref)?;
    if model_ref.starts_with("pacha://") || model_ref.starts_with("hf://") {
        return Ok(());
    }

    let file_data = std::fs::read(model_ref).map_err(|e| RealizarError::UnsupportedOperation {
        operation: "read_model".to_string(),
        reason: format!("Failed to read: {e}"),
    })?;

    display_model_info(model_ref, &file_data)?;

    let mut history = load_chat_history(history_file);

    if let Some(sys) = system_prompt {
        println!("System: {sys}");
    }
    println!("Chat mode active. Type 'exit' to quit.");

    run_chat_loop(&mut history);
    save_chat_history(history_file, &history);

    println!("Goodbye!");
    Ok(())
}

/// Available benchmark suites
pub const BENCHMARK_SUITES: &[(&str, &str)] = &[
    (
        "tensor_ops",
        "Core tensor operations (add, mul, matmul, softmax)",
    ),
    ("inference", "End-to-end inference pipeline benchmarks"),
    ("cache", "KV cache operations and memory management"),
    ("tokenizer", "BPE and SentencePiece tokenization"),
    ("quantize", "Quantization/dequantization (Q4_0, Q8_0)"),
    ("lambda", "AWS Lambda cold start and warm invocation"),
    (
        "comparative",
        "Framework comparison (MNIST, CIFAR-10, Iris)",
    ),
];

include!("display_utils.rs");
include!("mod_print_benchmark.rs");
include!("mod_gguf_info.rs");
include!("mod_server_commands.rs");
include!("mod_06.rs");
