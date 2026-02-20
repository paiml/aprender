//! Interactive Chat Command
//!
//! Provides a REPL interface for interactive chat with language models.
//!
//! Toyota Way: Genchi Genbutsu + Visual Control
//! - Go and see the actual model outputs
//! - Optionally visualize token probabilities for transparency
//!
//! # Y13/Y14 Compliance
//!
//! Per spec Section Y.3:
//! - Y13: Architecture-agnostic (auto-detect from model metadata)
//! - Y14: Format-agnostic (supports APR, GGUF, SafeTensors)
//!
//! # Example
//!
//! ```bash
//! apr chat model.apr              # APR format
//! apr chat model.gguf             # GGUF format
//! apr chat model.safetensors      # SafeTensors format
//! apr chat model.safetensors --inspect        # Show top-k probabilities
//! apr chat model.safetensors --temperature 0.7 --top-p 0.9
//! ```

use crate::error::CliError;
use crate::output;
#[cfg(feature = "inference")]
use aprender::text::llama_tokenizer::LlamaTokenizer;
// PMAT-109: Qwen tokenizer needed for APR/SafeTensors format
use aprender::text::bpe::Qwen2BpeTokenizer;
// PMAT-181: Read EOS token from APR metadata (fixes GH-170)
use aprender::serialization::apr::AprReader;
// Chat template support (Toyota Way: Standardized Work)
use aprender::text::chat_template::{
    auto_detect_template, detect_format_from_name, ChatMessage, ChatTemplateEngine, TemplateFormat,
};
use colored::Colorize;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

// No fallback imports — inference feature is required for chat

/// Chat configuration options
pub(crate) struct ChatConfig {
    /// Sampling temperature (0 = greedy)
    pub temperature: f32,
    /// Nucleus sampling threshold
    pub top_p: f32,
    /// Maximum tokens to generate per response
    pub max_tokens: usize,
    /// System prompt (optional)
    pub system: Option<String>,
    /// Show inspection info (top-k probs, tokens/sec)
    pub inspect: bool,
    /// Force CPU inference (skip CUDA even if available)
    /// Default: false - GPU is preferred when available (F-GPU-134b)
    pub force_cpu: bool,
    /// Enable inference tracing (APR-TRACE-001)
    pub trace: bool,
    /// Trace output file path
    pub trace_output: Option<std::path::PathBuf>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            system: None,
            inspect: false,
            force_cpu: false, // F-GPU-134b: Default to GPU when available
            trace: false,
            trace_output: None,
        }
    }
}

/// Run the chat command with optional inference tracing (APR-TRACE-001)
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    path: &Path,
    temperature: f32,
    top_p: f32,
    max_tokens: usize,
    system: Option<&str>,
    inspect: bool,
    force_cpu: bool,
    trace: bool,
    trace_steps: Option<&[String]>,
    trace_verbose: bool,
    trace_output: Option<std::path::PathBuf>,
    trace_level: &str,
    profile: bool,
) -> Result<(), CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    // Setup trace config if tracing enabled (APR-TRACE-001)
    if trace {
        eprintln!(
            "{}",
            "Inference tracing enabled for chat (APR-TRACE-001)".cyan()
        );
        eprintln!("  Trace level: {}", trace_level);
        if let Some(steps) = trace_steps {
            eprintln!("  Trace steps: {}", steps.join(", "));
        }
        if trace_verbose {
            eprintln!("  Verbose mode enabled");
        }
        if let Some(ref path) = trace_output {
            eprintln!("  Output: {}", path.display());
        }
        if profile {
            eprintln!("  Roofline profiling enabled");
        }
    }

    // trace_steps, trace_verbose, trace_level, profile reserved for future use
    let _ = (trace_steps, trace_verbose, trace_level, profile);

    let config = ChatConfig {
        temperature,
        top_p,
        max_tokens,
        system: system.map(String::from),
        inspect,
        force_cpu,
        trace,
        trace_output,
    };

    print_welcome_banner(path, &config);

    // Run the REPL
    run_repl(path, &config)
}

/// Model format variants (Y14: format-agnostic)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFormat {
    /// Native APR v2 format (preferred per Native Library Mandate)
    Apr,
    /// GGUF format (llama.cpp compatible)
    Gguf,
    /// HuggingFace SafeTensors format
    SafeTensors,
    /// Demo mode with tiny random weights
    Demo,
}

/// Detect model format from file extension (Y14: format-agnostic)
fn detect_format(path: &Path) -> ModelFormat {
    match path.extension().and_then(|e| e.to_str()) {
        Some("apr") => ModelFormat::Apr,
        Some("gguf") => ModelFormat::Gguf,
        Some("safetensors") => ModelFormat::SafeTensors,
        _ => ModelFormat::Demo,
    }
}

/// Try loading a tokenizer from a specific path, printing success/failure.
fn try_load_tokenizer(path: &Path, label: &str) -> Option<Qwen2BpeTokenizer> {
    match Qwen2BpeTokenizer::from_file(path) {
        Ok(tok) => {
            println!(
                "{} {} ({})",
                format!("Loaded tokenizer{label}:").green(),
                path.display(),
                format!("{} tokens", tok.vocab_size()).dimmed()
            );
            Some(tok)
        }
        Err(e) => {
            println!(
                "{} {}",
                format!("Warning: Failed to load tokenizer{label}:").yellow(),
                e
            );
            None
        }
    }
}

/// Search HuggingFace cache for Qwen tokenizer.json files.
fn search_hf_cache_tokenizer(hf_cache: &Path) -> Option<Qwen2BpeTokenizer> {
    let entries = std::fs::read_dir(hf_cache).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("models--Qwen") {
            continue;
        }
        let snapshots_dir = entry.path().join("snapshots");
        let snapshots = std::fs::read_dir(&snapshots_dir).ok()?;
        for snapshot in snapshots.flatten() {
            let tokenizer_path = snapshot.path().join("tokenizer.json");
            if tokenizer_path.exists() {
                if let Some(tok) = try_load_tokenizer(&tokenizer_path, " from HuggingFace cache") {
                    return Some(tok);
                }
            }
        }
    }
    None
}

/// Try to load tokenizer from a path if it exists.
fn try_tokenizer_at(path: &Path, label: &str) -> Option<Qwen2BpeTokenizer> {
    if path.exists() {
        try_load_tokenizer(path, label)
    } else {
        None
    }
}

/// PMAT-109: Find Qwen tokenizer from model dir, HF cache, or APR cache.
fn find_qwen_tokenizer(model_path: &Path) -> Result<Option<Qwen2BpeTokenizer>, CliError> {
    // 1. Model's parent directory
    if let Some(tok) = model_path
        .parent()
        .and_then(|p| try_tokenizer_at(&p.join("tokenizer.json"), ""))
    {
        return Ok(Some(tok));
    }

    // 2. HuggingFace + APR caches
    if let Some(home) = dirs::home_dir() {
        if let Some(tok) = search_hf_cache_tokenizer(&home.join(".cache/huggingface/hub")) {
            return Ok(Some(tok));
        }
        if let Some(tok) = try_tokenizer_at(
            &home.join(".apr/tokenizers/qwen2/tokenizer.json"),
            " from APR cache",
        ) {
            return Ok(Some(tok));
        }
    }

    Err(CliError::InvalidFormat(
        "No Qwen tokenizer found. Searched:\n\
         1. Model directory (tokenizer.json)\n\
         2. HuggingFace cache (~/.cache/huggingface/hub/models--Qwen--*/snapshots/*/tokenizer.json)\n\
         3. APR cache (~/.apr/tokenizers/qwen2/tokenizer.json)\n\n\
         To fix: Download a Qwen model with tokenizer:\n\
           apr pull hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF"
            .to_string(),
    ))
}

/// Normalize repeated punctuation (max 3 repeats of `!`, `?`, `.`).
fn normalize_repeated_punctuation(s: &str) -> String {
    let mut prev_char = '\0';
    let mut repeat_count = 0;
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        if c == prev_char && matches!(c, '!' | '?' | '.') {
            repeat_count += 1;
            if repeat_count < 3 {
                result.push(c);
            }
        } else {
            repeat_count = 0;
            result.push(c);
        }
        prev_char = c;
    }
    result
}

/// Check if text looks like the start of a new conversational turn.
fn looks_like_new_turn(text: &str) -> bool {
    text.starts_with("Suggest")
        || text.starts_with("What")
        || text.starts_with("How")
        || text.starts_with("Why")
        || text.starts_with("Can")
        || text.starts_with("Human:")
        || text.contains("<|im_start|>")
}

/// Clean ChatML markers and artifacts from model response (F-PIPE-166b).
fn clean_chat_response(raw: &str) -> String {
    let mut cleaned = raw.to_string();

    // Remove ChatML markers
    for marker in &[
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
    ] {
        cleaned = cleaned.replace(marker, "");
    }

    // F-PIPE-166b: Clean GPT-2/BPE tokenizer artifacts
    cleaned = cleaned.replace('\u{0120}', " ");
    cleaned = cleaned.replace('\u{010A}', "\n");
    cleaned = cleaned.replace("Ġ", " ");
    cleaned = cleaned.replace("Ċ", "\n");

    cleaned = normalize_repeated_punctuation(&cleaned);

    while cleaned.contains("  ") {
        cleaned = cleaned.replace("  ", " ");
    }

    let trimmed = cleaned.trim();

    // Stop at first line if the model started a new turn
    if let Some(first_newline) = trimmed.find('\n') {
        let first_line = trimmed[..first_newline].trim();
        let rest = trimmed[first_newline..].trim();
        if looks_like_new_turn(rest) {
            return first_line.to_string();
        }
    }

    trimmed.to_string()
}

/// Detect format from magic bytes (more reliable than extension)
#[cfg(feature = "inference")]
fn detect_format_from_bytes(data: &[u8]) -> ModelFormat {
    if data.len() < 8 {
        return ModelFormat::Demo;
    }
    // APR v1: "APRN", APR v2: "APR2" or "APR\0"
    if &data[0..4] == b"APRN" || &data[0..4] == b"APR2" || &data[0..4] == b"APR\0" {
        return ModelFormat::Apr;
    }
    // GGUF: "GGUF"
    if &data[0..4] == b"GGUF" {
        return ModelFormat::Gguf;
    }
    // SafeTensors: first 8 bytes are header size (reasonable value)
    let header_size = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8]));
    if header_size > 0 && header_size < 100_000_000 {
        return ModelFormat::SafeTensors;
    }
    ModelFormat::Demo
}

fn print_welcome_banner(path: &Path, config: &ChatConfig) {
    let format = detect_format(path);

    // Detect chat template format from model name (Toyota Way: Visual Control)
    let model_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let template_format = detect_format_from_name(model_name);
    let template_name = match template_format {
        TemplateFormat::ChatML => "ChatML",
        TemplateFormat::Llama2 => "LLaMA2",
        TemplateFormat::Mistral => "Mistral",
        TemplateFormat::Phi => "Phi",
        TemplateFormat::Alpaca => "Alpaca",
        TemplateFormat::Custom => "Custom",
        TemplateFormat::Raw => "Raw",
    };

    match format {
        ModelFormat::Apr => {
            // Y13: Architecture detected from metadata, not hardcoded
            output::section("Model Chat (APR Format)");
            println!();
            println!(
                "{}",
                "Using APR v2 format with mmap (Native Library Mandate)".cyan()
            );
        }
        ModelFormat::Gguf => {
            // Y13/Y14: Architecture-agnostic GGUF support
            output::section("Model Chat (GGUF Format)");
            println!();
            println!(
                "{}",
                "Using GGUF format with realizar inference engine".cyan()
            );
        }
        ModelFormat::SafeTensors => {
            output::section("Model Chat (SafeTensors Format)");
            println!();
            println!(
                "{}",
                "Using SafeTensors with mmap (Native Library Mandate)".cyan()
            );
        }
        ModelFormat::Demo => {
            output::section("Chat Demo (Tiny Model)");
            println!();
            println!(
                "{}",
                "Note: Using tiny demo model. Pass .apr, .gguf, or .safetensors file for full model."
                    .yellow()
            );
        }
    }
    println!();
    output::kv("Model", path.display());
    output::kv("Chat Template", template_name);
    output::kv("Temperature", config.temperature);
    output::kv("Top-P", config.top_p);
    output::kv("Max Tokens", config.max_tokens);

    if let Some(system) = &config.system {
        output::kv("System", system);
    }

    if config.inspect {
        println!();
        println!(
            "{}",
            "Inspection mode enabled - showing token probabilities".cyan()
        );
    }

    println!();
    println!("{}", "Commands:".white().bold());
    println!("  /quit     Exit the chat");
    println!("  /clear    Clear conversation history");
    println!("  /system   Set system prompt");
    println!("  /help     Show help");
    println!();
    println!("{}", "═".repeat(60));
    println!();
}

include!("chat_session.rs");
include!("chat_part_03.rs");
include!("chat_part_04.rs");
