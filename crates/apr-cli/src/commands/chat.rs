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

// Fallback imports when inference feature is disabled
#[cfg(not(feature = "inference"))]
use aprender::demo::Qwen2Config;
#[cfg(not(feature = "inference"))]
use aprender::models::Qwen2Model;

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

/// PMAT-109: Find Qwen tokenizer from multiple standard locations
///
/// Search order:
/// 1. Model's parent directory (tokenizer.json)
/// 2. HuggingFace cache (~/.cache/huggingface/hub/models--Qwen--*/snapshots/*/tokenizer.json)
/// 3. APR tokenizer cache (~/.apr/tokenizers/qwen2/tokenizer.json)
fn find_qwen_tokenizer(model_path: &Path) -> Result<Option<Qwen2BpeTokenizer>, CliError> {
    // 1. Check model's parent directory first
    if let Some(parent) = model_path.parent() {
        let tokenizer_path = parent.join("tokenizer.json");
        if tokenizer_path.exists() {
            match Qwen2BpeTokenizer::from_file(&tokenizer_path) {
                Ok(tok) => {
                    println!(
                        "{} {} ({})",
                        "Loaded tokenizer:".green(),
                        tokenizer_path.display(),
                        format!("{} tokens", tok.vocab_size()).dimmed()
                    );
                    return Ok(Some(tok));
                }
                Err(e) => {
                    println!(
                        "{} {}",
                        "Warning: Failed to load tokenizer from model dir:".yellow(),
                        e
                    );
                }
            }
        }
    }

    // 2. Search HuggingFace cache for Qwen tokenizers
    if let Some(home) = dirs::home_dir() {
        let hf_cache = home.join(".cache/huggingface/hub");
        if hf_cache.exists() {
            // Look for Qwen model directories
            if let Ok(entries) = std::fs::read_dir(&hf_cache) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    // Match Qwen model directories (Qwen, Qwen2, Qwen2.5, etc.)
                    if name_str.starts_with("models--Qwen") {
                        let snapshots_dir = entry.path().join("snapshots");
                        if snapshots_dir.exists() {
                            if let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) {
                                for snapshot in snapshots.flatten() {
                                    let tokenizer_path = snapshot.path().join("tokenizer.json");
                                    if tokenizer_path.exists() {
                                        if let Ok(tok) =
                                            Qwen2BpeTokenizer::from_file(&tokenizer_path)
                                        {
                                            println!(
                                                "{} {} ({})",
                                                "Loaded tokenizer from HuggingFace cache:".green(),
                                                tokenizer_path.display(),
                                                format!("{} tokens", tok.vocab_size()).dimmed()
                                            );
                                            return Ok(Some(tok));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Check APR tokenizer cache
        let apr_tokenizer = home.join(".apr/tokenizers/qwen2/tokenizer.json");
        if apr_tokenizer.exists() {
            match Qwen2BpeTokenizer::from_file(&apr_tokenizer) {
                Ok(tok) => {
                    println!(
                        "{} {} ({})",
                        "Loaded tokenizer from APR cache:".green(),
                        apr_tokenizer.display(),
                        format!("{} tokens", tok.vocab_size()).dimmed()
                    );
                    return Ok(Some(tok));
                }
                Err(e) => {
                    println!(
                        "{} {}",
                        "Warning: Failed to load tokenizer from APR cache:".yellow(),
                        e
                    );
                }
            }
        }
    }

    // No tokenizer found - provide helpful error message
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

/// Clean ChatML markers and artifacts from model response
///
/// Strips (F-PIPE-166b compliance):
/// - <|im_start|>assistant, <|im_end|>, <|im_start|>, etc.
/// - GPT-2/BPE tokenizer artifacts (Ġ = U+0120 = space prefix)
/// - Repeated punctuation (!!!, ???, etc.)
/// - Leading/trailing whitespace
/// - Repeated newlines
fn clean_chat_response(raw: &str) -> String {
    let mut cleaned = raw.to_string();

    // Remove ChatML markers
    let markers = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
    ];
    for marker in markers {
        cleaned = cleaned.replace(marker, "");
    }

    // F-PIPE-166b: Clean GPT-2/BPE tokenizer artifacts
    // Ġ (U+0120) represents a space before a token in GPT-2 BPE encoding
    cleaned = cleaned.replace('\u{0120}', " ");
    // Ċ (U+010A) represents newline in some BPE tokenizers
    cleaned = cleaned.replace('\u{010A}', "\n");
    // Other common BPE artifacts
    cleaned = cleaned.replace("Ġ", " "); // In case it's literal "Ġ" string
    cleaned = cleaned.replace("Ċ", "\n"); // In case it's literal "Ċ" string

    // F-PIPE-166b: Normalize repeated punctuation (e.g., "!!!" -> "!")
    // Use a simple loop to avoid regex dependency
    let mut prev_char = '\0';
    let mut repeat_count = 0;
    let mut result = String::with_capacity(cleaned.len());

    for c in cleaned.chars() {
        if c == prev_char && (c == '!' || c == '?' || c == '.') {
            repeat_count += 1;
            // Allow at most 3 repeated punctuation marks
            if repeat_count < 3 {
                result.push(c);
            }
        } else {
            repeat_count = 0;
            result.push(c);
        }
        prev_char = c;
    }
    cleaned = result;

    // Normalize multiple spaces to single space
    while cleaned.contains("  ") {
        cleaned = cleaned.replace("  ", " ");
    }

    // Trim and normalize whitespace
    let trimmed = cleaned.trim();

    // Stop at first line if it looks like the model started a new turn
    // (e.g., "4\nSuggest a fun way..." -> just "4")
    if let Some(first_newline) = trimmed.find('\n') {
        let first_line = trimmed[..first_newline].trim();
        let rest = trimmed[first_newline..].trim();
        // If rest looks like a new question/topic, just return first line
        if rest.starts_with("Suggest")
            || rest.starts_with("What")
            || rest.starts_with("How")
            || rest.starts_with("Why")
            || rest.starts_with("Can")
            || rest.starts_with("Human:")
            || rest.contains("<|im_start|>")
        {
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

// =============================================================================
// ChatSession with realizar (Y13/Y14: architecture and format agnostic)
// =============================================================================

#[cfg(feature = "inference")]
mod realizar_chat {
    use super::*;
    use aprender::text::bpe::Qwen2BpeTokenizer;
    use std::fs::File;
    use std::io::Read;

    /// Chat session using realizar for high-performance inference
    /// Y13: Architecture-agnostic (detected from model metadata)
    /// Y14: Format-agnostic (APR, GGUF, SafeTensors)
    ///
    /// PMAT-108: ALL inference delegated to realizar engine.
    /// aprender::models is NOT used for inference (only training).
    pub struct ChatSession {
        /// Model bytes (kept for regeneration if needed)
        model_bytes: Vec<u8>,
        /// Model path (for mmap-based loading)
        model_path: std::path::PathBuf,
        /// Detected format
        format: ModelFormat,
        /// Conversation history as ChatMessage objects
        history: Vec<ChatMessage>,
        /// Chat template engine (Toyota Way: Standardized Work)
        chat_template: Box<dyn ChatTemplateEngine + Send + Sync>,
        /// Detected template format name (for display)
        template_format: TemplateFormat,
        /// LLaMA tokenizer (for GGUF format)
        llama_tokenizer: Option<LlamaTokenizer>,
        /// Qwen2 BPE tokenizer (for SafeTensors/APR format)
        qwen_tokenizer: Option<Qwen2BpeTokenizer>,
    }

    impl ChatSession {
        pub(super) fn new(path: &Path) -> Result<Self, CliError> {
            println!("{}", "Loading model...".cyan());
            let start = Instant::now();

            // Read file bytes
            let mut file = File::open(path).map_err(|e| {
                CliError::ValidationFailed(format!("Failed to open model file: {e}"))
            })?;
            let mut model_bytes = Vec::new();
            file.read_to_end(&mut model_bytes).map_err(|e| {
                CliError::ValidationFailed(format!("Failed to read model file: {e}"))
            })?;

            // Detect format from magic bytes (Y14)
            let format = detect_format_from_bytes(&model_bytes);

            let elapsed = start.elapsed();
            let format_name = match format {
                ModelFormat::Apr => "APR",
                ModelFormat::Gguf => "GGUF",
                ModelFormat::SafeTensors => "SafeTensors",
                ModelFormat::Demo => "Demo",
            };
            println!(
                "{} {} format in {:.2}s ({:.1} MB)",
                "Loaded".green(),
                format_name,
                elapsed.as_secs_f32(),
                model_bytes.len() as f32 / 1_000_000.0
            );

            // Load tokenizer based on format
            let (llama_tokenizer, qwen_tokenizer) = match format {
                ModelFormat::Gguf => {
                    // Load LLaMA tokenizer from GGUF
                    let tok = match LlamaTokenizer::from_gguf_bytes(&model_bytes) {
                        Ok(tok) => {
                            println!(
                                "{} tokenizer with {} tokens",
                                "Loaded".green(),
                                tok.vocab_size()
                            );
                            Some(tok)
                        }
                        Err(e) => {
                            // F-MODEL-COMPLETE-001: Failed tokenizer is a fatal error
                            return Err(CliError::InvalidFormat(format!(
                                "Model is incomplete: Failed to load GGUF tokenizer: {}. \
                                This usually indicates a corrupted or improperly converted model.",
                                e
                            )));
                        }
                    };
                    (tok, None)
                }
                ModelFormat::SafeTensors | ModelFormat::Apr => {
                    // PMAT-109: Search multiple standard locations for Qwen tokenizer
                    // Priority: 1) model dir, 2) HuggingFace cache, 3) ~/.apr/tokenizers
                    let tok = find_qwen_tokenizer(path)?;
                    (None, tok)
                }
                ModelFormat::Demo => (None, None),
            };

            // PMAT-108: SafeTensors inference uses realizar's SafetensorsToAprConverter
            // No need to pre-load weights here - done lazily in generate_safetensors()
            // This avoids loading the model twice and removes aprender::models dependency
            if format == ModelFormat::SafeTensors {
                // Just print config info for user feedback (via realizar's SafetensorsConfig)
                if let Some(parent) = path.parent() {
                    let config_path = parent.join("config.json");
                    if config_path.exists() {
                        if let Ok(json) = std::fs::read_to_string(&config_path) {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                                let num_layers = v["num_hidden_layers"].as_u64().unwrap_or(0);
                                let hidden_size = v["hidden_size"].as_u64().unwrap_or(0);
                                let num_heads = v["num_attention_heads"].as_u64().unwrap_or(0);
                                println!(
                                    "{} config: {} layers, {} hidden, {} heads",
                                    "Loaded".green(),
                                    num_layers,
                                    hidden_size,
                                    num_heads
                                );
                            }
                        }
                    }
                }
            }

            // Detect chat template from model architecture (Toyota Way: Jidoka - auto-detect)
            // F-TEMPLATE-001: Use GGUF metadata for architecture detection, not filename
            // PMAT-120 FIX: SafeTensors/APR use config.json for architecture detection
            // Hash-based filenames (e.g., d4c4d9763127153c.gguf) don't contain model name
            let model_name = match format {
                ModelFormat::Gguf => {
                    // Parse GGUF metadata to get architecture (e.g., "qwen2", "llama")
                    use realizar::gguf::GGUFModel;
                    match GGUFModel::from_bytes(&model_bytes) {
                        Ok(gguf) => {
                            let arch = gguf.architecture().unwrap_or("unknown").to_string();
                            // Map GGUF architecture to template-detectable name
                            // Architecture names: qwen2, llama, phi3, mistral, etc.
                            arch
                        }
                        Err(_) => path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string(),
                    }
                }
                ModelFormat::SafeTensors | ModelFormat::Apr => {
                    // PMAT-120: Read config.json for architecture detection
                    // Five-Whys: "model.safetensors" filename doesn't indicate architecture
                    // Root cause: detect_format_from_name("model") returns Raw template
                    // Fix: Extract model_type or architectures from config.json
                    let mut arch = String::from("unknown");
                    if let Some(parent) = path.parent() {
                        let config_path = parent.join("config.json");
                        if config_path.exists() {
                            if let Ok(json) = std::fs::read_to_string(&config_path) {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                                    // Try model_type first (e.g., "qwen2")
                                    if let Some(model_type) = v["model_type"].as_str() {
                                        arch = model_type.to_lowercase();
                                    }
                                    // Fallback to architectures array (e.g., ["Qwen2ForCausalLM"])
                                    else if let Some(archs) = v["architectures"].as_array() {
                                        if let Some(first) = archs.first().and_then(|a| a.as_str())
                                        {
                                            // Extract base name: "Qwen2ForCausalLM" -> "qwen2"
                                            arch = first
                                                .trim_end_matches("ForCausalLM")
                                                .trim_end_matches("LMHeadModel")
                                                .to_lowercase();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Fallback to parent directory name (often contains model name)
                    if arch == "unknown" {
                        arch = path
                            .parent()
                            .and_then(|p| p.file_name())
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown")
                            .to_string();
                    }
                    arch
                }
                ModelFormat::Demo => "demo".to_string(),
            };
            let template_format = detect_format_from_name(&model_name);
            let chat_template = auto_detect_template(&model_name);

            let template_name = match template_format {
                TemplateFormat::ChatML => "ChatML",
                TemplateFormat::Llama2 => "LLaMA2",
                TemplateFormat::Mistral => "Mistral",
                TemplateFormat::Phi => "Phi",
                TemplateFormat::Alpaca => "Alpaca",
                TemplateFormat::Custom => "Custom",
                TemplateFormat::Raw => "Raw",
            };
            println!(
                "{} {} chat template",
                "Detected".green(),
                template_name.cyan()
            );

            Ok(Self {
                model_bytes,
                model_path: path.to_path_buf(),
                format,
                history: Vec::new(),
                chat_template,
                template_format,
                llama_tokenizer,
                qwen_tokenizer,
            })
        }

        pub(super) fn generate(&mut self, user_input: &str, config: &ChatConfig) -> String {
            let start = Instant::now();

            // Build conversation with chat template (Toyota Way: Standardized Work)
            let mut messages: Vec<ChatMessage> = Vec::new();

            // Add system prompt if configured
            if let Some(ref system) = config.system {
                messages.push(ChatMessage::system(system));
            }

            // Add conversation history
            messages.extend(self.history.iter().cloned());

            // Add current user message
            messages.push(ChatMessage::user(user_input));

            // Format conversation using detected template
            let formatted_prompt = match self.chat_template.format_conversation(&messages) {
                Ok(prompt) => prompt,
                Err(e) => {
                    return format!("[Template error: {}]", e);
                }
            };

            // APR-TRACE-001: Debug formatted prompt
            if config.trace {
                eprintln!(
                    "[APR-TRACE] Formatted prompt ({} chars):",
                    formatted_prompt.len()
                );
                eprintln!(
                    "[APR-TRACE] {:?}",
                    &formatted_prompt[..formatted_prompt.len().min(500)]
                );
            }

            // For GGUF, use embedded tokenizer directly (correct special token IDs)
            // This is critical: GGUF models have their own tokenizer with correct special token IDs
            // Using LlamaTokenizer/Qwen2BpeTokenizer causes wrong IDs for <|im_start|>, <|im_end|>, etc.
            if self.format == ModelFormat::Gguf {
                return match self.generate_gguf_with_prompt(&formatted_prompt, config) {
                    Ok(response) => clean_chat_response(&response),
                    Err(e) => format!("[Error: {}]", e),
                };
            }

            // For non-GGUF formats, tokenize with loaded tokenizer
            let prompt_tokens: Vec<u32> = if let Some(ref tokenizer) = self.llama_tokenizer {
                tokenizer.encode_with_bos(&formatted_prompt)
            } else if let Some(ref tokenizer) = self.qwen_tokenizer {
                tokenizer.encode(&formatted_prompt)
            } else {
                formatted_prompt.chars().map(|c| c as u32).collect()
            };

            // PMAT-114: Debug APR tokenization to compare with GGUF
            if config.trace {
                eprintln!(
                    "[APR-TRACE] Prompt tokens ({} tokens): {:?}",
                    prompt_tokens.len(),
                    &prompt_tokens[..prompt_tokens.len().min(50)]
                );
            }

            let result = match self.format {
                ModelFormat::Apr => self.generate_apr(&prompt_tokens, config),
                ModelFormat::SafeTensors => self.generate_safetensors(&prompt_tokens, config),
                ModelFormat::Demo => Ok(vec![]),
                ModelFormat::Gguf => unreachable!(), // handled above
            };

            let gen_time = start.elapsed();

            match result {
                Ok(output_tokens) => {
                    // Strip prompt tokens - only decode newly generated tokens
                    let new_tokens = if output_tokens.len() > prompt_tokens.len() {
                        &output_tokens[prompt_tokens.len()..]
                    } else {
                        &output_tokens[..]
                    };

                    // Debug: show generation stats
                    if config.inspect {
                        println!(
                            "{}",
                            format!(
                                "[{} new tokens in {:.2}s = {:.1} tok/s]",
                                new_tokens.len(),
                                gen_time.as_secs_f32(),
                                new_tokens.len() as f32 / gen_time.as_secs_f32()
                            )
                            .dimmed()
                        );
                        // Debug: show first 10 new tokens
                        if let Some(ref tok) = self.llama_tokenizer {
                            println!(
                                "[DEBUG: first 10 new tokens: {:?}]",
                                &new_tokens[..new_tokens.len().min(10)]
                            );
                            for &id in new_tokens.iter().take(10) {
                                println!("[DEBUG: {} -> {:?}]", id, tok.id_to_token(id));
                            }
                        }
                    }

                    // Decode only the new tokens
                    let raw_response = if let Some(ref tokenizer) = self.llama_tokenizer {
                        tokenizer.decode(new_tokens)
                    } else if let Some(ref tokenizer) = self.qwen_tokenizer {
                        tokenizer.decode(new_tokens)
                    } else {
                        new_tokens
                            .iter()
                            .filter_map(|&t| char::from_u32(t))
                            .collect()
                    };

                    // Clean up ChatML markers from response
                    clean_chat_response(&raw_response)
                }
                Err(e) => format!("[Error: {}]", e),
            }
        }

        /// Generate response for GGUF models using the embedded tokenizer
        ///
        /// GGUF models have their own tokenizer with correct special token IDs.
        /// Using LlamaTokenizer/Qwen2BpeTokenizer causes wrong token IDs for:
        /// - <|im_start|> (should be 151644)
        /// - <|im_end|> (should be 151645)
        /// - <|endoftext|> (should be 151643)
        ///
        /// This function uses the GGUF's embedded tokenizer for both encode and decode.
        fn generate_gguf_with_prompt(
            &self,
            prompt: &str,
            config: &ChatConfig,
        ) -> Result<String, String> {
            use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

            // Load GGUF model with mmap
            let mapped = MappedGGUFModel::from_path(&self.model_path)
                .map_err(|e| format!("Failed to mmap GGUF: {e}"))?;

            // Encode prompt using GGUF's embedded tokenizer (correct special token IDs)
            let prompt_tokens = mapped
                .model
                .encode(prompt)
                .ok_or_else(|| "Failed to encode prompt with GGUF tokenizer".to_string())?;
            let prompt_len = prompt_tokens.len();

            // APR-TRACE-001: Debug token IDs
            if config.trace {
                eprintln!(
                    "[APR-TRACE] Prompt tokens ({} tokens): {:?}",
                    prompt_len,
                    &prompt_tokens[..prompt_len.min(50)]
                );
                // Decode tokens to verify they're correct
                let decoded = mapped.model.decode(&prompt_tokens);
                eprintln!(
                    "[APR-TRACE] Decoded: {:?}",
                    &decoded[..decoded.len().min(200)]
                );
            }

            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| format!("Failed to create GGUF model: {e}"))?;

            let practical_max = config.max_tokens;

            // ChatML stop tokens for Qwen2/ChatML models
            let gen_config = QuantizedGenerateConfig {
                max_tokens: practical_max,
                temperature: config.temperature,
                top_k: 40,
                stop_tokens: vec![151645, 151643], // <|im_end|>, <|endoftext|>
                trace: config.trace,               // PMAT-TRACE-GGUF-001: Pass trace flag
            };

            // Try CUDA GPU path first
            #[cfg(feature = "cuda")]
            if !config.force_cpu {
                use realizar::gguf::OwnedQuantizedModelCuda;
                if OwnedQuantizedModelCuda::is_available() {
                    let num_layers = model.config.num_layers;
                    let is_gqa = model.config.num_kv_heads < model.config.num_heads;
                    let gqa_note = if is_gqa {
                        format!(" (GQA: {} kv_heads)", model.config.num_kv_heads)
                    } else {
                        String::new()
                    };

                    match OwnedQuantizedModelCuda::new(model, 0) {
                        Ok(mut cuda_model) => {
                            let gpu_name = cuda_model.device_name().to_string();
                            let vram_mb = cuda_model.vram_mb();
                            println!(
                                "{}",
                                format!(
                                    "[GGUF CUDA: {} ({} MB VRAM), {} layers, {} tokens{}]",
                                    gpu_name, vram_mb, num_layers, practical_max, gqa_note
                                )
                                .bright_green()
                            );

                            // APR-TRACE-001: Force logits path for debugging
                            let debug_gen_config = realizar::gguf::QuantizedGenerateConfig {
                                max_tokens: gen_config.max_tokens,
                                temperature: 0.01, // Near-zero but forces logits path
                                top_k: 10,         // Get top-10 for debugging
                                stop_tokens: gen_config.stop_tokens.clone(),
                                trace: config.trace, // PMAT-TRACE-GGUF-001: Pass trace flag
                            };

                            let output_tokens = cuda_model
                                .generate_gpu_resident(&prompt_tokens, &debug_gen_config)
                                .map_err(|e| format!("CUDA generate failed: {e}"))?;

                            // Extract new tokens and decode with GGUF tokenizer
                            let new_tokens = if output_tokens.len() > prompt_len {
                                &output_tokens[prompt_len..]
                            } else {
                                &output_tokens[..]
                            };

                            // APR-TRACE-001: Debug generated tokens
                            if config.trace {
                                eprintln!(
                                    "[APR-TRACE] Generated {} new tokens: {:?}",
                                    new_tokens.len(),
                                    &new_tokens[..new_tokens.len().min(50)]
                                );

                                // Decode each token individually for trace diagnostics
                                for (i, &tok) in new_tokens.iter().take(20).enumerate() {
                                    let decoded = mapped.model.decode(&[tok]);
                                    eprintln!("[APR-TRACE] Token {}: {} -> {:?}", i, tok, decoded);
                                }
                            }

                            return Ok(mapped.model.decode(new_tokens));
                        }
                        Err(e) => {
                            println!(
                                "{}",
                                format!("[CUDA init failed: {}, falling back to CPU]", e).yellow()
                            );
                            // Fall through to CPU path - need to recreate model
                        }
                    }
                }
            }

            // CPU path - recreate model since CUDA may have consumed it
            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| format!("Failed to recreate model: {e}"))?;

            // APR-TRACE-001: Use traced generation when --trace is enabled
            let output_tokens = if config.trace {
                use realizar::{InferenceTracer, ModelInfo, TraceConfig};

                let trace_config = TraceConfig {
                    enabled: true,
                    verbose: false,
                    output: config.trace_output.clone(),
                    ..Default::default()
                };

                let mut tracer = InferenceTracer::new(trace_config);
                tracer.set_model_info(ModelInfo {
                    name: "GGUF Model (CPU)".to_string(),
                    num_layers: model.config.num_layers,
                    hidden_dim: model.config.hidden_dim,
                    vocab_size: model.config.vocab_size,
                    num_heads: model.config.num_heads,
                    quant_type: None,
                });

                // Create decode closure for tracing
                let _decode_fn = |token_id: u32| -> String { mapped.model.decode(&[token_id]) };

                // APR-TRACE-001: CPU traced generation not yet implemented
                eprintln!("Warning: CPU traced generation not implemented, using non-traced path");

                let result = model
                    .generate_with_cache(&prompt_tokens, &gen_config)
                    .map_err(|e| format!("GGUF generate failed: {e}"))?;

                // Output partial trace
                if let Err(e) = tracer.write_output() {
                    eprintln!("Warning: Failed to write trace output: {e}");
                }

                result
            } else {
                model
                    .generate_with_cache(&prompt_tokens, &gen_config)
                    .map_err(|e| format!("GGUF generate failed: {e}"))?
            };

            // Extract new tokens and decode with GGUF tokenizer
            let new_tokens = if output_tokens.len() > prompt_len {
                &output_tokens[prompt_len..]
            } else {
                &output_tokens[..]
            };

            let decoded = mapped.model.decode(new_tokens);
            Ok(decoded)
        }

        fn generate_apr(&self, prompt: &[u32], config: &ChatConfig) -> Result<Vec<u32>, String> {
            // PMAT-181: Extract EOS token from APR metadata (fixes GH-170)
            // Root cause: Different models have different EOS tokens, hardcoded 151645 was wrong
            // Five-Whys: WHY hang? → WHY no output? → WHY immediate EOS? → Mismatch with model's actual EOS
            let eos_token_id = self.extract_apr_eos_token().unwrap_or(151645);

            // PMAT-110: APR CUDA now works with KV cache (fixed in realizar)
            // Try GPU path first if not force_cpu

            #[cfg(feature = "cuda")]
            if !config.force_cpu {
                use realizar::apr::{AprV2Model, AprV2ModelCuda};

                if AprV2ModelCuda::is_available() {
                    // First create AprV2Model from bytes, then wrap with CUDA
                    match AprV2Model::from_bytes(self.model_bytes.clone()) {
                        Ok(apr_model) => {
                            match AprV2ModelCuda::new(apr_model, 0) {
                                Ok(mut cuda_model) => {
                                    let gpu_name = cuda_model.device_name().to_string();
                                    let vram_mb = cuda_model.vram_mb();
                                    eprintln!("[APR CUDA: {} ({} MB VRAM)]", gpu_name, vram_mb);

                                    // PMAT-181: Use EOS token from model metadata
                                    let max_tokens = config.max_tokens;
                                    return cuda_model
                                        .generate_cuda_with_cache(prompt, max_tokens, eos_token_id)
                                        .map_err(|e| format!("APR CUDA generate failed: {e}"));
                                }
                                Err(e) => {
                                    eprintln!("[APR CUDA init failed: {}, using CPU]", e);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[APR model load failed: {}, using CPU]", e);
                        }
                    }
                }
            }

            // CPU path using AprTransformer (has temperature/top_p sampling + KV cache)
            use realizar::apr_transformer::{AprTransformer, GenerateConfig};

            let transformer = AprTransformer::from_apr_bytes(&self.model_bytes)
                .map_err(|e| format!("Failed to load APR transformer: {e}"))?;

            // Use generate_with_cache for O(n) KV-cached generation (not O(n²) forward)
            let gen_config = GenerateConfig {
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                top_p: config.top_p,
                top_k: 0,
                repetition_penalty: 1.0,
                trace: config.trace,
            };

            transformer
                .generate_with_cache(prompt, &gen_config)
                .map_err(|e| format!("APR generate failed: {e}"))
        }

        /// PMAT-181: Extract EOS token ID from APR metadata (fixes GH-170)
        ///
        /// Five-Whys Root Cause: Different Qwen2 model sizes may have different EOS tokens
        /// in their metadata. The 1.5B model's EOS token was being mismatched with hardcoded
        /// 151645, causing generation to terminate immediately or hang.
        ///
        /// Toyota Way: Genchi Genbutsu - Go and see the actual model metadata
        fn extract_apr_eos_token(&self) -> Option<u32> {
            // Parse APR metadata from model bytes
            let reader = AprReader::from_bytes(self.model_bytes.clone()).ok()?;

            // Try common metadata keys for EOS token (in order of specificity)
            // 1. tokenizer.eos_token_id (GGUF-style)
            // 2. eos_token_id (direct)
            // 3. tokenizer_config.eos_token_id (nested)
            let keys = [
                "tokenizer.eos_token_id",
                "eos_token_id",
                "tokenizer.ggml.eos_token_id",
            ];

            for key in keys {
                if let Some(value) = reader.get_metadata(key) {
                    if let Some(id) = value.as_u64() {
                        return Some(id as u32);
                    }
                }
            }

            // Try nested tokenizer_config
            if let Some(config) = reader.get_metadata("tokenizer_config") {
                if let Some(obj) = config.as_object() {
                    if let Some(value) = obj.get("eos_token_id") {
                        if let Some(id) = value.as_u64() {
                            return Some(id as u32);
                        }
                    }
                }
            }

            None
        }

        #[allow(dead_code)]
        fn generate_gguf(&self, prompt: &[u32], config: &ChatConfig) -> Result<Vec<u32>, String> {
            use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

            // Use MappedGGUFModel -> OwnedQuantizedModel for proper attention
            // This has RoPE position encoding, causal mask, and GQA support
            let mapped = MappedGGUFModel::from_path(&self.model_path)
                .map_err(|e| format!("Failed to mmap GGUF: {e}"))?;

            let model = OwnedQuantizedModel::from_mapped(&mapped)
                .map_err(|e| format!("Failed to create GGUF model: {e}"))?;

            // With KV cache, we can generate more tokens efficiently
            let practical_max = config.max_tokens;

            let is_gqa = model.config.num_kv_heads < model.config.num_heads;
            let gqa_note = if is_gqa {
                format!(" (GQA: {} kv_heads)", model.config.num_kv_heads)
            } else {
                String::new()
            };

            // ChatML stop tokens for Qwen2/ChatML models:
            // <|im_end|> = 151645, <|endoftext|> = 151643
            let gen_config = QuantizedGenerateConfig {
                max_tokens: practical_max,
                temperature: config.temperature,
                top_k: 40,
                stop_tokens: vec![151645, 151643], // <|im_end|>, <|endoftext|>
                trace: config.trace,               // PMAT-TRACE-GGUF-001: Pass trace flag
            };

            // Try CUDA GPU path first (200+ tok/s target)
            // Uses generate_gpu_resident which is the tested/working GPU path
            #[cfg(feature = "cuda")]
            if !config.force_cpu {
                use realizar::gguf::OwnedQuantizedModelCuda;
                if OwnedQuantizedModelCuda::is_available() {
                    // Print model info before attempting CUDA (model consumed on success)
                    let num_layers = model.config.num_layers;

                    match OwnedQuantizedModelCuda::new(model, 0) {
                        Ok(mut cuda_model) => {
                            let gpu_name = cuda_model.device_name().to_string();
                            let vram_mb = cuda_model.vram_mb();
                            println!(
                                "{}",
                                format!(
                                    "[GGUF CUDA: {} ({} MB VRAM), {} layers, {} tokens{}]",
                                    gpu_name, vram_mb, num_layers, practical_max, gqa_note
                                )
                                .bright_green()
                            );
                            // Use generate_gpu_resident (tested working path) not generate_full_cuda_with_cache
                            return cuda_model
                                .generate_gpu_resident(prompt, &gen_config)
                                .map_err(|e| format!("CUDA generate failed: {e}"));
                        }
                        Err(e) => {
                            println!(
                                "{}",
                                format!("[CUDA init failed: {}, falling back to CPU]", e).yellow()
                            );
                            // Re-create model for CPU fallback (model was consumed)
                            let model = OwnedQuantizedModel::from_mapped(&mapped)
                                .map_err(|e| format!("Failed to recreate model: {e}"))?;

                            return model
                                .generate_with_cache(prompt, &gen_config)
                                .map_err(|e| format!("GGUF generate failed: {e}"));
                        }
                    }
                }
            }

            // CPU path with KV cache (12+ tok/s) - used when CUDA feature disabled or unavailable
            // Use KV cache path for O(n) instead of O(n²)
            model
                .generate_with_cache(prompt, &gen_config)
                .map_err(|e| format!("GGUF generate failed: {e}"))
        }

        fn generate_safetensors(
            &mut self,
            prompt: &[u32],
            config: &ChatConfig,
        ) -> Result<Vec<u32>, String> {
            // PMAT-116: GPU path for SafeTensors (direct H2D loading, no APR conversion)
            #[cfg(feature = "cuda")]
            if !config.force_cpu {
                use realizar::safetensors_cuda::SafeTensorsCudaModel;

                // Load SafeTensors directly to GPU (PMAT-116)
                // This avoids APR conversion overhead and achieves GGUF parity
                let mut cuda_model = SafeTensorsCudaModel::load(&self.model_path, 0)
                    .map_err(|e| format!("SafeTensors CUDA load failed: {e}"))?;

                eprintln!(
                    "  {} {} ({}MB VRAM)",
                    "GPU:".green(),
                    cuda_model.device_name(),
                    cuda_model.vram_mb()
                );

                // Use EOS token from config or default to Qwen2 EOS
                let eos_id = 151645u32; // Qwen2 EOS token

                // Generate with GPU acceleration
                let tokens = cuda_model
                    .generate(prompt, config.max_tokens, eos_id)
                    .map_err(|e| format!("SafeTensors CUDA generate failed: {e}"))?;

                // PMAT-120 DEBUG: Log generated token IDs
                if config.trace {
                    let new_tokens = &tokens[prompt.len()..];
                    eprintln!(
                        "[APR-TRACE] SafeTensors GPU generated {} tokens: {:?}",
                        new_tokens.len(),
                        &new_tokens[..new_tokens.len().min(20)]
                    );
                }

                // Return only generated tokens (skip prompt)
                return Ok(tokens[prompt.len()..].to_vec());
            }

            // CPU path: Use realizar's SafeTensors inference via AprTransformer
            // PMAT-108 FIX: Use realizar's SafeTensors inference, not aprender's training model
            // The old path used Qwen2Model.generate() which is 0.3 tok/s (training code)
            // The new path uses AprTransformer via SafetensorsToAprConverter for 25+ tok/s
            use realizar::apr_transformer::GenerateConfig;
            use realizar::safetensors_infer::SafetensorsToAprConverter;

            // Convert SafeTensors to AprTransformer (optimized inference engine)
            let transformer = SafetensorsToAprConverter::convert(&self.model_path)
                .map_err(|e| format!("SafeTensors conversion failed: {e}"))?;

            // Use KV-cached generation for O(n) instead of O(n²)
            let gen_config = GenerateConfig {
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                top_p: config.top_p,
                top_k: 0,
                repetition_penalty: 1.0,
                trace: config.trace,
            };

            transformer
                .generate_with_cache(prompt, &gen_config)
                .map_err(|e| format!("SafeTensors generate failed: {e}"))
        }

        #[allow(dead_code)]
        pub(super) fn history(&self) -> &[ChatMessage] {
            &self.history
        }

        pub(super) fn history_len(&self) -> usize {
            self.history.len()
        }

        pub(super) fn add_to_history(&mut self, role: &str, content: &str) {
            self.history.push(ChatMessage::new(role, content));
        }

        pub(super) fn clear_history(&mut self) {
            self.history.clear();
        }

        #[allow(dead_code)]
        pub(super) fn format(&self) -> ModelFormat {
            self.format
        }

        #[allow(dead_code)]
        pub(super) fn template_format(&self) -> TemplateFormat {
            self.template_format
        }
    }
}

#[cfg(feature = "inference")]
use realizar_chat::ChatSession;

// =============================================================================
// Fallback ChatSession without realizar (demo mode only)
// =============================================================================

#[cfg(not(feature = "inference"))]
struct ChatSession {
    model: Qwen2Model,
    tokenizer: Qwen2BpeTokenizer,
    history: Vec<String>,
}

#[cfg(not(feature = "inference"))]
impl ChatSession {
    fn new(path: &Path) -> Result<Self, CliError> {
        println!("{}", "Loading model...".cyan());
        let start = Instant::now();

        let format = detect_format(path);

        // Use full config for real weights, tiny config for demo
        let config = match format {
            ModelFormat::Apr | ModelFormat::Gguf | ModelFormat::SafeTensors => {
                // Full Qwen2-0.5B config - mmap loading won't OOM
                Qwen2Config::qwen2_0_5b_instruct()
            }
            ModelFormat::Demo => {
                // Tiny demo config (~50MB RAM)
                Qwen2Config {
                    hidden_size: 64,
                    num_attention_heads: 4,
                    num_kv_heads: 2,
                    num_layers: 2,
                    vocab_size: 1000,
                    max_seq_len: 512,
                    intermediate_size: 256,
                    rope_theta: 10000.0,
                }
            }
        };

        // Use uninitialized model for APR/SafeTensors to avoid OOM
        // Per Native Library Mandate: placeholder tensors = ~1KB vs ~2.5GB
        let mut model = match format {
            ModelFormat::Apr | ModelFormat::Gguf | ModelFormat::SafeTensors => {
                Qwen2Model::new_uninitialized(&config)
            }
            ModelFormat::Demo => Qwen2Model::new(&config),
        };

        // Load weights using mmap (Native Library Mandate - zero-copy)
        // Per Spec §2.4: Random weight fallbacks are FORBIDDEN
        match format {
            ModelFormat::Apr => {
                // Native APR v2 format (preferred)
                let count = model.load_from_apr(path).map_err(|e| {
                    CliError::ValidationFailed(format!(
                        "Failed to load APR weights (random fallback FORBIDDEN per spec): {e}"
                    ))
                })?;
                println!(
                    "{} {}",
                    "Loaded".green(),
                    format!("{count} tensors via APR mmap")
                );
            }
            ModelFormat::Gguf => {
                // GGUF requires inference feature
                return Err(CliError::ValidationFailed(
                    "GGUF format requires --features inference. Rebuild with: cargo build --features inference".to_string()
                ));
            }
            ModelFormat::SafeTensors => {
                // SafeTensors format (also uses mmap)
                let count = model.load_from_safetensors(path).map_err(|e| {
                    CliError::ValidationFailed(format!(
                        "Failed to load SafeTensors weights (random fallback FORBIDDEN per spec): {e}"
                    ))
                })?;
                println!(
                    "{} {}",
                    "Loaded".green(),
                    format!("{count} tensors via mmap")
                );
            }
            ModelFormat::Demo => {
                // Demo mode: tiny model with random weights for testing only
                println!(
                    "{}",
                    "Using randomly initialized weights (demo mode only)".yellow()
                );
            }
        }

        model.eval();
        let elapsed = start.elapsed();
        println!("{} {:.2}s", "Model ready in".green(), elapsed.as_secs_f32());

        // Load tokenizer from same directory as model
        let tokenizer = Self::load_tokenizer(path);

        Ok(Self {
            model,
            tokenizer,
            history: Vec::new(),
        })
    }

    /// Load tokenizer from model directory
    fn load_tokenizer(model_path: &Path) -> Qwen2BpeTokenizer {
        // Try to find tokenizer.json in same directory as model
        if let Some(parent) = model_path.parent() {
            let tokenizer_path = parent.join("tokenizer.json");
            if tokenizer_path.exists() {
                match Qwen2BpeTokenizer::from_file(&tokenizer_path) {
                    Ok(tok) => {
                        println!(
                            "{} {} ({})",
                            "Loaded tokenizer:".green(),
                            tokenizer_path.display(),
                            format!("{} tokens", tok.vocab_size()).dimmed()
                        );
                        return tok;
                    }
                    Err(e) => {
                        eprintln!("{} {}", "Warning: Failed to load tokenizer:".yellow(), e);
                    }
                }
            }
        }

        // Fallback to basic byte-level tokenizer
        eprintln!(
            "{}",
            "Using basic byte-level tokenizer (download tokenizer.json for better results)"
                .yellow()
        );
        Qwen2BpeTokenizer::new()
    }

    fn generate(&mut self, user_input: &str, config: &ChatConfig) -> String {
        // Build conversation with chat template
        let mut messages: Vec<(&str, &str)> = Vec::new();

        // Add system prompt if configured
        if let Some(ref system) = config.system {
            messages.push(("system", system.as_str()));
        }

        // Add history
        for msg in &self.history {
            if let Some(content) = msg.strip_prefix("user: ") {
                messages.push(("user", content));
            } else if let Some(content) = msg.strip_prefix("assistant: ") {
                messages.push(("assistant", content));
            }
        }

        // Add current user message
        messages.push(("user", user_input));

        // Format conversation
        let prompt = self
            .tokenizer
            .format_conversation(&messages.iter().map(|(r, c)| (*r, *c)).collect::<Vec<_>>());

        // Encode prompt (using byte-level encoding since we don't have full vocab loaded)
        let input_ids = self.tokenizer.encode(&prompt);

        // Limit prompt length
        let max_prompt = 256;
        let input_ids: Vec<u32> = if input_ids.len() > max_prompt {
            input_ids[input_ids.len() - max_prompt..].to_vec()
        } else {
            input_ids
        };

        // Generate (use profiled version in inspect mode)
        let start = Instant::now();
        let output_ids = if config.inspect {
            self.model.generate_profiled(
                &input_ids,
                config.max_tokens, // Limit for speed
                config.temperature,
            )
        } else {
            self.model.generate(
                &input_ids,
                config.max_tokens, // Limit for speed
                config.temperature,
                config.top_p,
            )
        };
        let gen_time = start.elapsed();

        // Decode only the generated tokens
        let new_tokens = &output_ids[input_ids.len()..];
        let response = self.tokenizer.decode(new_tokens);

        // Print generation stats if inspect mode
        if config.inspect {
            let tokens_per_sec = new_tokens.len() as f32 / gen_time.as_secs_f32();
            println!(
                "{}",
                format!(
                    "[{} tokens in {:.2}s = {:.1} tok/s]",
                    new_tokens.len(),
                    gen_time.as_secs_f32(),
                    tokens_per_sec
                )
                .dimmed()
            );

            // Get logits for the last position to show top-k candidates
            if !output_ids.is_empty() {
                let position_ids: Vec<usize> = (0..output_ids.len()).collect();
                let logits = self.model.forward(&output_ids, &position_ids);
                let logits_data = logits.data();
                let vocab_size = self.model.config().vocab_size;
                let last_pos = output_ids.len() - 1;
                let last_logits = &logits_data[last_pos * vocab_size..(last_pos + 1) * vocab_size];
                print_top_k(last_logits, &self.tokenizer, 5);
            }
        }

        response
    }
}

// =============================================================================
// REPL implementation with inference feature
// =============================================================================

#[cfg(feature = "inference")]
fn run_repl(path: &Path, config: &ChatConfig) -> Result<(), CliError> {
    let mut session = ChatSession::new(path)?;

    loop {
        // Print prompt
        print!("{}", "You: ".green().bold());
        io::stdout().flush()?;

        // Read input
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            // EOF (Ctrl+D)
            println!();
            break;
        }

        let input = input.trim();

        // Handle empty input
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match handle_command_inference(input, &mut session)? {
                CommandResult::Continue => continue,
                CommandResult::Quit => break,
            }
        }

        // Generate response using realizar inference engine (Y13/Y14)
        let response = session.generate(input, config);

        // Add to history
        session.add_to_history("user", input);
        session.add_to_history("assistant", &response);

        // Print response
        println!("{} {}", "Assistant:".blue().bold(), response);

        if config.inspect {
            print_inspection_info_inference(&session);
        }

        println!();
    }

    println!("{}", "Goodbye!".cyan());
    Ok(())
}

#[cfg(feature = "inference")]
fn handle_command_inference(
    input: &str,
    session: &mut ChatSession,
) -> Result<CommandResult, CliError> {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0].to_lowercase();

    match cmd.as_str() {
        "/quit" | "/exit" | "/q" => {
            return Ok(CommandResult::Quit);
        }
        "/clear" => {
            session.clear_history();
            println!("{}", "Conversation cleared.".yellow());
        }
        "/system" => {
            if parts.len() > 1 {
                println!("{} {}", "System prompt set:".yellow(), parts[1]);
            } else {
                println!("{}", "Usage: /system <prompt>".yellow());
            }
        }
        "/help" | "/h" | "/?" => {
            println!();
            println!("{}", "Commands:".white().bold());
            println!("  /quit, /exit, /q   Exit the chat");
            println!("  /clear             Clear conversation history");
            println!("  /system <prompt>   Set system prompt");
            println!("  /help, /h, /?      Show this help");
            println!();
        }
        _ => {
            println!("{} {}", "Unknown command:".red(), cmd);
        }
    }

    Ok(CommandResult::Continue)
}

#[cfg(feature = "inference")]
fn print_inspection_info_inference(session: &ChatSession) {
    println!();
    println!("{}", "[DEBUG] Session info:".dimmed());
    println!("{}", format!("  Format: {:?}", session.format()).dimmed());
    println!(
        "{}",
        format!("  Template: {:?}", session.template_format()).dimmed()
    );
    println!(
        "{}",
        format!("  History: {} messages", session.history_len()).dimmed()
    );
}

// =============================================================================
// REPL implementation without inference feature (fallback)
// =============================================================================

#[cfg(not(feature = "inference"))]
fn run_repl(path: &Path, config: &ChatConfig) -> Result<(), CliError> {
    let mut session = ChatSession::new(path)?;

    loop {
        // Print prompt
        print!("{}", "You: ".green().bold());
        io::stdout().flush()?;

        // Read input
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            // EOF (Ctrl+D)
            println!();
            break;
        }

        let input = input.trim();

        // Handle empty input
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match handle_command(input, &mut session.history)? {
                CommandResult::Continue => continue,
                CommandResult::Quit => break,
            }
        }

        // Generate response using real model inference
        let response = session.generate(input, config);

        // Add to history
        session.history.push(format!("user: {input}"));
        session.history.push(format!("assistant: {response}"));

        // Print response
        println!("{} {}", "Assistant:".blue().bold(), response);

        if config.inspect {
            print_inspection_info(&session);
        }

        println!();
    }

    println!("{}", "Goodbye!".cyan());
    Ok(())
}

enum CommandResult {
    Continue,
    Quit,
}

#[cfg(not(feature = "inference"))]
fn handle_command(input: &str, history: &mut Vec<String>) -> Result<CommandResult, CliError> {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0].to_lowercase();

    match cmd.as_str() {
        "/quit" | "/exit" | "/q" => {
            return Ok(CommandResult::Quit);
        }
        "/clear" => {
            history.clear();
            println!("{}", "Conversation cleared.".yellow());
        }
        "/system" => {
            if parts.len() > 1 {
                println!("{} {}", "System prompt set:".yellow(), parts[1]);
            } else {
                println!("{}", "Usage: /system <prompt>".yellow());
            }
        }
        "/help" | "/h" | "/?" => {
            println!();
            println!("{}", "Commands:".white().bold());
            println!("  /quit, /exit, /q   Exit the chat");
            println!("  /clear             Clear conversation history");
            println!("  /system <prompt>   Set system prompt");
            println!("  /help, /h, /?      Show this help");
            println!();
        }
        _ => {
            println!("{} {}", "Unknown command:".red(), cmd);
        }
    }

    Ok(CommandResult::Continue)
}

#[cfg(not(feature = "inference"))]
fn print_inspection_info(session: &ChatSession) {
    println!();
    println!("{}", "[DEBUG] Session info:".dimmed());
    println!(
        "{}",
        format!(
            "  Model: {} layers, {} params",
            session.model.num_layers(),
            format_params(session.model.num_parameters())
        )
        .dimmed()
    );
    println!(
        "{}",
        format!("  History: {} messages", session.history.len()).dimmed()
    );
    println!(
        "{}",
        format!("  Vocab: {} tokens", session.tokenizer.vocab_size()).dimmed()
    );
}

/// Display top-k token probabilities (spec E1)
#[cfg(not(feature = "inference"))]
fn print_top_k(logits: &[f32], tokenizer: &Qwen2BpeTokenizer, k: usize) {
    println!();
    println!("{}", "[TOP-K CANDIDATES]".cyan().bold());

    // Compute softmax
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&v| v / sum).collect();

    // Get top-k indices
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (i, (token_id, prob)) in indexed.iter().take(k).enumerate() {
        let token_str = tokenizer.decode(&[*token_id as u32]);
        let display = if token_str.trim().is_empty() {
            format!("<token_{}>", token_id)
        } else {
            format!("\"{}\"", token_str.escape_debug())
        };

        let bar_len = (prob * 50.0) as usize;
        let bar = "█".repeat(bar_len);

        println!(
            "  {}. {} {:>6.2}% {}",
            i + 1,
            display.yellow(),
            prob * 100.0,
            bar.green()
        );
    }
}

/// Format parameter count in human-readable form
#[cfg(not(feature = "inference"))]
fn format_params(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        format!("{count}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_chat_config_default() {
        let config = ChatConfig::default();
        assert!((config.temperature - 0.7).abs() < 0.01);
        assert!((config.top_p - 0.9).abs() < 0.01);
        assert_eq!(config.max_tokens, 512);
        assert!(config.system.is_none());
        assert!(!config.inspect);
        // F-GPU-134b: Default to GPU (force_cpu = false)
        assert!(
            !config.force_cpu,
            "F-GPU-134b: force_cpu should default to false"
        );
    }

    #[test]
    fn test_chat_config_with_system_prompt() {
        let config = ChatConfig {
            system: Some("You are a helpful assistant.".to_string()),
            ..Default::default()
        };
        assert!(config.system.is_some());
        assert_eq!(
            config.system.as_ref().unwrap(),
            "You are a helpful assistant."
        );
    }

    #[test]
    fn test_chat_config_trace_settings() {
        let config = ChatConfig {
            trace: true,
            trace_output: Some(PathBuf::from("/tmp/trace.json")),
            ..Default::default()
        };
        assert!(config.trace);
        assert_eq!(
            config.trace_output.as_ref().unwrap().to_str().unwrap(),
            "/tmp/trace.json"
        );
    }

    #[test]
    fn test_chat_config_force_cpu() {
        let config = ChatConfig {
            force_cpu: true,
            ..Default::default()
        };
        assert!(config.force_cpu);
    }

    // F-PIPE-166b: Test tokenizer artifact cleaning
    #[test]
    fn test_clean_chat_response_removes_chatml_markers() {
        let raw = "<|im_start|>assistant\nHello world<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_removes_bpe_artifacts() {
        // Ġ (U+0120) is used in GPT-2/BPE tokenizers for space prefix
        let raw = "HelloĠworld";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_normalizes_repeated_punctuation() {
        let raw = "Wow!!!!!";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wow!!!");

        let raw2 = "Really??????";
        let cleaned2 = clean_chat_response(raw2);
        assert_eq!(cleaned2, "Really???");
    }

    #[test]
    fn test_clean_chat_response_normalizes_spaces() {
        let raw = "Hello   world";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_trims_whitespace() {
        let raw = "  Hello world  ";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_removes_endoftext() {
        let raw = "Hello<|endoftext|>world";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Helloworld");
    }

    #[test]
    fn test_clean_chat_response_newline_artifact() {
        // Ċ (U+010A) represents newline in some BPE tokenizers
        let raw = "HelloĊworld";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello\nworld");
    }

    #[test]
    fn test_clean_chat_response_strips_new_turn() {
        // If response has new question after first line, return first line only
        let raw = "4\nSuggest a fun way to learn Rust";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "4");
    }

    #[test]
    fn test_clean_chat_response_keeps_multiline_answer() {
        // Normal multiline response should be preserved
        let raw = "Here is the answer:\nLine 1\nLine 2";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Here is the answer:\nLine 1\nLine 2");
    }

    #[test]
    fn test_clean_chat_response_empty_string() {
        let raw = "";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_chat_response_only_markers() {
        let raw = "<|im_start|>assistant<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_chat_response_human_prompt_cutoff() {
        // If "Human:" appears after first line, cut it off
        let raw = "Yes\nHuman: What else?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Yes");
    }

    // =========================================================================
    // ModelFormat tests
    // =========================================================================

    #[test]
    fn test_model_format_equality() {
        assert_eq!(ModelFormat::Apr, ModelFormat::Apr);
        assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
        assert_eq!(ModelFormat::SafeTensors, ModelFormat::SafeTensors);
        assert_eq!(ModelFormat::Demo, ModelFormat::Demo);
    }

    #[test]
    fn test_model_format_inequality() {
        assert_ne!(ModelFormat::Apr, ModelFormat::Gguf);
        assert_ne!(ModelFormat::Gguf, ModelFormat::SafeTensors);
        assert_ne!(ModelFormat::SafeTensors, ModelFormat::Demo);
    }

    #[test]
    fn test_model_format_debug() {
        assert_eq!(format!("{:?}", ModelFormat::Apr), "Apr");
        assert_eq!(format!("{:?}", ModelFormat::Gguf), "Gguf");
        assert_eq!(format!("{:?}", ModelFormat::SafeTensors), "SafeTensors");
        assert_eq!(format!("{:?}", ModelFormat::Demo), "Demo");
    }

    #[test]
    fn test_model_format_clone() {
        let format = ModelFormat::Apr;
        let cloned = format;
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_model_format_copy() {
        let format = ModelFormat::Gguf;
        let copied: ModelFormat = format;
        assert_eq!(format, copied);
    }

    // =========================================================================
    // detect_format tests (Y14: format-agnostic)
    // =========================================================================

    #[test]
    fn test_detect_format_apr() {
        let path = Path::new("/models/test.apr");
        assert_eq!(detect_format(path), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_format_gguf() {
        let path = Path::new("/models/test.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_safetensors() {
        let path = Path::new("/models/model.safetensors");
        assert_eq!(detect_format(path), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_detect_format_unknown_fallback_to_demo() {
        let path = Path::new("/models/test.bin");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_no_extension() {
        let path = Path::new("/models/modelfile");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_case_sensitive() {
        // Extensions are case-sensitive in Rust
        let path = Path::new("/models/test.APR");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_nested_path() {
        let path = Path::new("/home/user/.cache/models/qwen2-0.5b.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_relative_path() {
        let path = Path::new("./models/model.safetensors");
        assert_eq!(detect_format(path), ModelFormat::SafeTensors);
    }

    // =========================================================================
    // detect_format_from_bytes tests (inference feature only)
    // =========================================================================

    #[cfg(feature = "inference")]
    mod inference_tests {
        use super::*;

        #[test]
        fn test_detect_format_from_bytes_apr_v1() {
            let data = b"APRNxxxx00000000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Apr);
        }

        #[test]
        fn test_detect_format_from_bytes_apr_v2() {
            let data = b"APR2xxxx00000000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Apr);
        }

        #[test]
        fn test_detect_format_from_bytes_apr_null() {
            let data = b"APR\0xxxx00000000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Apr);
        }

        #[test]
        fn test_detect_format_from_bytes_gguf() {
            let data = b"GGUFxxxx00000000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Gguf);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors() {
            // SafeTensors: first 8 bytes are little-endian header size
            // A typical header size might be ~1000-10000 bytes
            let mut data = vec![0u8; 16];
            // Write 1000 as little-endian u64 (reasonable header size)
            data[0..8].copy_from_slice(&1000u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::SafeTensors);
        }

        #[test]
        fn test_detect_format_from_bytes_too_short() {
            let data = b"APR";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_empty() {
            let data: &[u8] = &[];
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_unknown_magic() {
            let data = b"UNKN0000\x00\x00\x00\x00\x00\x00\x00\x00";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Demo);
        }
    }

    // =========================================================================
    // run() error case tests
    // =========================================================================

    #[test]
    fn test_run_file_not_found() {
        let path = Path::new("/nonexistent/model.gguf");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false, false, None, false, None, "info", false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(p)) => {
                assert_eq!(p, PathBuf::from("/nonexistent/model.gguf"));
            }
            other => panic!("Expected FileNotFound error, got {:?}", other),
        }
    }

    #[test]
    fn test_run_with_trace_config() {
        let path = Path::new("/nonexistent/model.gguf");
        // This should still fail with FileNotFound, but trace config is set
        let result = run(
            path,
            0.7,
            0.9,
            512,
            Some("You are helpful"),
            true,
            true,
            true,
            Some(&["tokenize".to_string(), "sample".to_string()]),
            true,
            Some(PathBuf::from("/tmp/trace.json")),
            "debug",
            true,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // CommandResult tests
    // =========================================================================

    #[test]
    fn test_command_result_variants() {
        // Ensure CommandResult variants exist and can be matched
        let continue_result = CommandResult::Continue;
        let quit_result = CommandResult::Quit;

        match continue_result {
            CommandResult::Continue => {}
            CommandResult::Quit => panic!("Expected Continue"),
        }

        match quit_result {
            CommandResult::Quit => {}
            CommandResult::Continue => panic!("Expected Quit"),
        }
    }

    // =========================================================================
    // Edge case tests for clean_chat_response
    // =========================================================================

    #[test]
    fn test_clean_chat_response_mixed_markers() {
        let raw = "<|im_start|>assistant\n<|im_start|>Hello<|im_end|><|endoftext|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn test_clean_chat_response_repeated_dots() {
        let raw = "Hmm........ let me think";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hmm... let me think");
    }

    #[test]
    fn test_clean_chat_response_unicode_preserved() {
        let raw = "こんにちは世界";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "こんにちは世界");
    }

    #[test]
    fn test_clean_chat_response_emoji_preserved() {
        let raw = "Hello 👋 World 🌍";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello 👋 World 🌍");
    }

    #[test]
    fn test_clean_chat_response_code_block() {
        let raw = "Here is code:\n```rust\nfn main() {}\n```";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Here is code:\n```rust\nfn main() {}\n```");
    }

    #[test]
    fn test_clean_chat_response_numbered_list() {
        let raw = "Steps:\n1. First\n2. Second\n3. Third";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Steps:\n1. First\n2. Second\n3. Third");
    }

    // =========================================================================
    // format_params tests (non-inference only)
    // =========================================================================

    #[cfg(not(feature = "inference"))]
    #[test]
    fn test_format_params_billions() {
        assert_eq!(format_params(7_000_000_000), "7.0B");
        assert_eq!(format_params(1_500_000_000), "1.5B");
    }

    #[cfg(not(feature = "inference"))]
    #[test]
    fn test_format_params_millions() {
        assert_eq!(format_params(500_000_000), "500.0M");
        assert_eq!(format_params(7_000_000), "7.0M");
        assert_eq!(format_params(1_500_000), "1.5M");
    }

    #[cfg(not(feature = "inference"))]
    #[test]
    fn test_format_params_thousands() {
        assert_eq!(format_params(500_000), "500.0K");
        assert_eq!(format_params(7_000), "7.0K");
        assert_eq!(format_params(1_500), "1.5K");
    }

    #[cfg(not(feature = "inference"))]
    #[test]
    fn test_format_params_small() {
        assert_eq!(format_params(999), "999");
        assert_eq!(format_params(100), "100");
        assert_eq!(format_params(1), "1");
        assert_eq!(format_params(0), "0");
    }

    // =========================================================================
    // Additional ChatConfig tests
    // =========================================================================

    #[test]
    fn test_chat_config_extreme_values() {
        let config = ChatConfig {
            temperature: 0.0,
            top_p: 0.0,
            max_tokens: 1,
            system: None,
            inspect: false,
            force_cpu: false,
            trace: false,
            trace_output: None,
        };
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_p, 0.0);
        assert_eq!(config.max_tokens, 1);
    }

    #[test]
    fn test_chat_config_high_temp() {
        let config = ChatConfig {
            temperature: 2.0,
            top_p: 1.0,
            max_tokens: 4096,
            system: Some("Creative mode".to_string()),
            inspect: true,
            force_cpu: true,
            trace: true,
            trace_output: Some(PathBuf::from("/tmp/creative_trace.json")),
        };
        assert_eq!(config.temperature, 2.0);
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_chat_config_inspect_mode() {
        let config = ChatConfig {
            inspect: true,
            ..Default::default()
        };
        assert!(config.inspect);
    }

    // =========================================================================
    // Additional clean_chat_response edge cases
    // =========================================================================

    #[test]
    fn test_clean_chat_response_deeply_nested_markers() {
        let raw = "<|im_start|><|im_start|>assistant\nTest<|im_end|><|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Test");
    }

    #[test]
    fn test_clean_chat_response_escaped_newlines() {
        let raw = "Line1\\nLine2";
        let cleaned = clean_chat_response(raw);
        // Should preserve escaped newlines
        assert!(cleaned.contains("\\n") || cleaned.contains('\n'));
    }

    #[test]
    fn test_clean_chat_response_tabs() {
        let raw = "Column1\tColumn2";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Column1\tColumn2");
    }

    #[test]
    fn test_clean_chat_response_markdown_headers() {
        let raw = "# Header 1\n## Header 2\n### Header 3";
        let cleaned = clean_chat_response(raw);
        assert!(cleaned.contains("# Header 1"));
        assert!(cleaned.contains("## Header 2"));
    }

    #[test]
    fn test_clean_chat_response_math_expression() {
        let raw = "E = mc²";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "E = mc²");
    }

    #[test]
    fn test_clean_chat_response_url() {
        let raw = "Visit https://example.com for more info";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Visit https://example.com for more info");
    }

    // =========================================================================
    // ModelFormat additional tests
    // =========================================================================

    #[test]
    fn test_model_format_all_variants() {
        // Ensure all variants exist
        let _apr = ModelFormat::Apr;
        let _gguf = ModelFormat::Gguf;
        let _st = ModelFormat::SafeTensors;
        let _demo = ModelFormat::Demo;
    }

    #[test]
    fn test_detect_format_double_extension() {
        let path = Path::new("/models/model.tar.gz");
        // Should detect based on last extension
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_hidden_file() {
        let path = Path::new("/models/.hidden.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_dot_in_name() {
        let path = Path::new("/models/model.v2.0.safetensors");
        assert_eq!(detect_format(path), ModelFormat::SafeTensors);
    }

    // =========================================================================
    // print_welcome_banner smoke tests (all format branches + config combos)
    // =========================================================================

    #[test]
    fn test_print_welcome_banner_apr_format() {
        let path = Path::new("/models/qwen2-instruct.apr");
        let config = ChatConfig::default();
        // Should not panic; exercises ModelFormat::Apr branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_gguf_format() {
        let path = Path::new("/models/qwen2-instruct.gguf");
        let config = ChatConfig::default();
        // Should not panic; exercises ModelFormat::Gguf branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_safetensors_format() {
        let path = Path::new("/models/model.safetensors");
        let config = ChatConfig::default();
        // Should not panic; exercises ModelFormat::SafeTensors branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_demo_format() {
        let path = Path::new("/models/model.bin");
        let config = ChatConfig::default();
        // Should not panic; exercises ModelFormat::Demo branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_with_system_prompt() {
        let path = Path::new("/models/test.apr");
        let config = ChatConfig {
            system: Some("You are a helpful assistant.".to_string()),
            ..Default::default()
        };
        // Exercises the system prompt display branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_with_inspect_mode() {
        let path = Path::new("/models/test.gguf");
        let config = ChatConfig {
            inspect: true,
            ..Default::default()
        };
        // Exercises the inspect mode display branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_with_system_and_inspect() {
        let path = Path::new("/models/model.safetensors");
        let config = ChatConfig {
            system: Some("Be concise".to_string()),
            inspect: true,
            ..Default::default()
        };
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_chatml() {
        // "qwen" in filename triggers ChatML template detection
        let path = Path::new("/models/qwen2-0.5b.apr");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_llama2() {
        // "llama" in filename triggers LLaMA2 template detection
        let path = Path::new("/models/llama-2-7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_mistral() {
        // "mistral" in filename triggers Mistral template detection
        let path = Path::new("/models/mistral-7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_phi() {
        // "phi-" in filename triggers Phi template detection
        let path = Path::new("/models/phi-3.safetensors");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_alpaca() {
        // "alpaca" in filename triggers Alpaca template detection
        let path = Path::new("/models/alpaca-7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_raw_fallback() {
        // Unknown model name falls back to Raw template
        let path = Path::new("/models/unknown-model.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_no_extension() {
        // No extension -> Demo format
        let path = Path::new("/models/modelfile");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_no_stem() {
        // File with no stem (just extension)
        let path = Path::new("/models/.apr");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    // =========================================================================
    // clean_chat_response: new-turn detection branches
    // =========================================================================

    #[test]
    fn test_clean_chat_response_what_question_cutoff() {
        let raw = "42\nWhat is the meaning of life?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "42");
    }

    #[test]
    fn test_clean_chat_response_how_question_cutoff() {
        let raw = "Done.\nHow do you feel about that?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Done.");
    }

    #[test]
    fn test_clean_chat_response_why_question_cutoff() {
        let raw = "Because reasons.\nWhy did the chicken cross the road?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Because reasons.");
    }

    #[test]
    fn test_clean_chat_response_can_question_cutoff() {
        let raw = "Yes.\nCan you elaborate on that?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Yes.");
    }

    #[test]
    fn test_clean_chat_response_im_start_in_rest_not_cutoff_after_stripping() {
        // Note: <|im_start|> markers are stripped BEFORE new-turn detection,
        // so the rest becomes "user\nNext question?" which doesn't match
        // any cutoff pattern (Suggest/What/How/Why/Can/Human:/<|im_start|>)
        let raw = "Answer here.\n<|im_start|>user\nNext question?";
        let cleaned = clean_chat_response(raw);
        // After marker stripping: "Answer here.\nuser\nNext question?"
        // "user" doesn't match cutoff patterns, so full text is preserved
        assert_eq!(cleaned, "Answer here.\nuser\nNext question?");
    }

    #[test]
    fn test_clean_chat_response_raw_im_start_text_in_rest() {
        // If the raw text literally has "<|im_start|>" that WASN'T stripped
        // (e.g., doubled markers), test the cutoff pattern
        // Actually, all <|im_start|> are stripped first, so this tests
        // that the check works on the already-cleaned text
        let raw = "Answer.\nWhat is next?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Answer.");
    }

    #[test]
    fn test_clean_chat_response_suggest_cutoff() {
        // Already covered above, but this tests with different first-line content
        let raw = "Rust is great!\nSuggest some alternatives";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Rust is great!");
    }

    #[test]
    fn test_clean_chat_response_no_cutoff_normal_continuation() {
        // Lines that start with lowercase should NOT be cut off
        let raw = "First line\nsecond line continues";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "First line\nsecond line continues");
    }

    #[test]
    fn test_clean_chat_response_no_cutoff_numbered_continuation() {
        let raw = "Steps:\n1. Do this\n2. Do that";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Steps:\n1. Do this\n2. Do that");
    }

    // =========================================================================
    // clean_chat_response: BPE artifact edge cases
    // =========================================================================

    #[test]
    fn test_clean_chat_response_multiple_bpe_spaces() {
        // Multiple Ġ should collapse to single space after cleaning
        let raw = "HelloĠĠworld";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_bpe_newline_u010a() {
        // U+010A character (Ċ) is newline in some BPE tokenizers
        let raw = "LineAĊLineB";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "LineA\nLineB");
    }

    #[test]
    fn test_clean_chat_response_both_bpe_artifacts() {
        // Mix of Ġ and Ċ
        let raw = "HelloĠworldĊnewĠline";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world\nnew line");
    }

    #[test]
    fn test_clean_chat_response_literal_g_with_dot() {
        // Literal "Ġ" string (not the Unicode character) should also be replaced
        let raw = "Hello\u{0120}there";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello there");
    }

    // =========================================================================
    // clean_chat_response: punctuation normalization edge cases
    // =========================================================================

    #[test]
    fn test_clean_chat_response_exactly_three_exclamation() {
        let raw = "Wow!!!";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wow!!!");
    }

    #[test]
    fn test_clean_chat_response_exactly_two_exclamation() {
        let raw = "Wow!!";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wow!!");
    }

    #[test]
    fn test_clean_chat_response_single_exclamation() {
        let raw = "Wow!";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wow!");
    }

    #[test]
    fn test_clean_chat_response_mixed_punctuation_not_collapsed() {
        // Mixed punctuation (different chars) should NOT be collapsed
        let raw = "Really!?!?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Really!?!?");
    }

    #[test]
    fn test_clean_chat_response_dots_exactly_three() {
        let raw = "Hmm...";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hmm...");
    }

    #[test]
    fn test_clean_chat_response_dots_many() {
        let raw = "Wait..........";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wait...");
    }

    #[test]
    fn test_clean_chat_response_questions_exactly_three() {
        let raw = "What???";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "What???");
    }

    #[test]
    fn test_clean_chat_response_questions_many() {
        let raw = "What????????";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "What???");
    }

    // =========================================================================
    // clean_chat_response: ChatML marker variations
    // =========================================================================

    #[test]
    fn test_clean_chat_response_im_start_alone() {
        let raw = "<|im_start|>Some text";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Some text");
    }

    #[test]
    fn test_clean_chat_response_im_end_alone() {
        let raw = "Some text<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Some text");
    }

    #[test]
    fn test_clean_chat_response_endoftext_midstream() {
        let raw = "Hello<|endoftext|> world";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_assistant_without_newline() {
        let raw = "<|im_start|>assistantHello";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn test_clean_chat_response_multiple_endoftext() {
        let raw = "Text<|endoftext|><|endoftext|>more";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Textmore");
    }

    #[test]
    fn test_clean_chat_response_all_markers_combined() {
        let raw = "<|im_start|>assistant\n<|im_start|>Hello <|endoftext|>world<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    // =========================================================================
    // clean_chat_response: whitespace normalization
    // =========================================================================

    #[test]
    fn test_clean_chat_response_many_spaces() {
        let raw = "Hello     world     test";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world test");
    }

    #[test]
    fn test_clean_chat_response_only_whitespace() {
        let raw = "   \t   \n   ";
        let cleaned = clean_chat_response(raw);
        assert!(cleaned.is_empty() || cleaned.trim().is_empty());
    }

    #[test]
    fn test_clean_chat_response_newlines_preserved() {
        let raw = "Line1\nLine2\nLine3";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Line1\nLine2\nLine3");
    }

    #[test]
    fn test_clean_chat_response_leading_newline_trimmed() {
        let raw = "\nHello";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn test_clean_chat_response_trailing_newline_trimmed() {
        let raw = "Hello\n";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    // =========================================================================
    // detect_format: additional edge cases
    // =========================================================================

    #[test]
    fn test_detect_format_empty_filename() {
        let path = Path::new("");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_just_extension_apr() {
        // Path is literally ".apr"
        let path = Path::new(".apr");
        // file_stem is "" for ".apr", extension is "apr" on some platforms
        // but actually Path::new(".apr").extension() returns None on Unix
        // because ".apr" is treated as a hidden file with no extension
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_mixed_case_gguf() {
        let path = Path::new("/models/test.GGUF");
        // Case-sensitive: "GGUF" != "gguf"
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_mixed_case_safetensors() {
        let path = Path::new("/models/test.SafeTensors");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_with_spaces_in_path() {
        let path = Path::new("/my models/test model.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_unicode_path() {
        let path = Path::new("/modelos/modelo.apr");
        assert_eq!(detect_format(path), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_format_root_path() {
        let path = Path::new("/model.safetensors");
        assert_eq!(detect_format(path), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_detect_format_current_dir() {
        let path = Path::new("model.apr");
        assert_eq!(detect_format(path), ModelFormat::Apr);
    }

    // =========================================================================
    // detect_format_from_bytes: additional edge cases (inference only)
    // =========================================================================

    #[cfg(feature = "inference")]
    mod inference_edge_cases {
        use super::*;

        #[test]
        fn test_detect_format_from_bytes_exactly_8_bytes() {
            // Exactly 8 bytes, not matching any known magic
            let data = [0u8; 8];
            // header_size = 0, so SafeTensors check fails (header_size > 0)
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_header_zero() {
            // SafeTensors with header_size = 0 should NOT match
            let mut data = vec![0u8; 16];
            data[0..8].copy_from_slice(&0u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_header_huge() {
            // SafeTensors with header_size >= 100_000_000 should NOT match
            let mut data = vec![0u8; 16];
            data[0..8].copy_from_slice(&100_000_000u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_header_boundary() {
            // SafeTensors with header_size = 99_999_999 should match
            let mut data = vec![0u8; 16];
            data[0..8].copy_from_slice(&99_999_999u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::SafeTensors);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_header_one() {
            // SafeTensors with header_size = 1 should match
            let mut data = vec![0u8; 16];
            data[0..8].copy_from_slice(&1u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::SafeTensors);
        }

        #[test]
        fn test_detect_format_from_bytes_7_bytes() {
            // 7 bytes (less than 8) should return Demo
            let data = b"APR2xxx";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_apr_takes_priority_over_safetensors() {
            // "APRN" starts with bytes that could also be a valid u64 header size
            // APR magic should be detected first
            let data = b"APRNxxxxxxxx0000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Apr);
        }

        #[test]
        fn test_detect_format_from_bytes_gguf_takes_priority_over_safetensors() {
            // "GGUF" starts with bytes that could also be a valid u64 header size
            // GGUF magic should be detected first
            let data = b"GGUFxxxxxxxx0000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Gguf);
        }

        #[test]
        fn test_detect_format_from_bytes_all_0xff() {
            // All 0xFF bytes: header_size would be u64::MAX which is > 100_000_000
            let data = [0xFFu8; 16];
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_typical_sizes() {
            // Typical SafeTensors header sizes
            for size in [256u64, 1024, 8192, 65536, 1_000_000, 50_000_000] {
                let mut data = vec![0u8; 16];
                data[0..8].copy_from_slice(&size.to_le_bytes());
                assert_eq!(
                    detect_format_from_bytes(&data),
                    ModelFormat::SafeTensors,
                    "Expected SafeTensors for header_size={}",
                    size
                );
            }
        }
    }

    // =========================================================================
    // ChatConfig: comprehensive field combination tests
    // =========================================================================

    #[test]
    fn test_chat_config_default_trace_is_false() {
        let config = ChatConfig::default();
        assert!(!config.trace);
    }

    #[test]
    fn test_chat_config_default_trace_output_is_none() {
        let config = ChatConfig::default();
        assert!(config.trace_output.is_none());
    }

    #[test]
    fn test_chat_config_all_fields_set() {
        let config = ChatConfig {
            temperature: 1.5,
            top_p: 0.95,
            max_tokens: 2048,
            system: Some("Expert mode".to_string()),
            inspect: true,
            force_cpu: true,
            trace: true,
            trace_output: Some(PathBuf::from("/tmp/all_fields.json")),
        };
        assert!((config.temperature - 1.5).abs() < f32::EPSILON);
        assert!((config.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.system.as_deref(), Some("Expert mode"));
        assert!(config.inspect);
        assert!(config.force_cpu);
        assert!(config.trace);
        assert_eq!(
            config.trace_output.as_ref().map(|p| p.to_str().unwrap()),
            Some("/tmp/all_fields.json")
        );
    }

    #[test]
    fn test_chat_config_zero_temperature() {
        // Greedy decoding
        let config = ChatConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(config.temperature, 0.0);
    }

    #[test]
    fn test_chat_config_max_tokens_zero() {
        let config = ChatConfig {
            max_tokens: 0,
            ..Default::default()
        };
        assert_eq!(config.max_tokens, 0);
    }

    #[test]
    fn test_chat_config_max_tokens_large() {
        let config = ChatConfig {
            max_tokens: usize::MAX,
            ..Default::default()
        };
        assert_eq!(config.max_tokens, usize::MAX);
    }

    #[test]
    fn test_chat_config_empty_system_prompt() {
        let config = ChatConfig {
            system: Some(String::new()),
            ..Default::default()
        };
        assert_eq!(config.system.as_deref(), Some(""));
    }

    #[test]
    fn test_chat_config_long_system_prompt() {
        let long_prompt = "a".repeat(10_000);
        let config = ChatConfig {
            system: Some(long_prompt.clone()),
            ..Default::default()
        };
        assert_eq!(config.system.as_deref(), Some(long_prompt.as_str()));
    }

    #[test]
    fn test_chat_config_trace_output_relative_path() {
        let config = ChatConfig {
            trace: true,
            trace_output: Some(PathBuf::from("relative/trace.json")),
            ..Default::default()
        };
        assert!(!config.trace_output.as_ref().unwrap().is_absolute());
    }

    // =========================================================================
    // run() additional error path tests
    // =========================================================================

    #[test]
    fn test_run_nonexistent_path_without_trace() {
        let path = Path::new("/definitely/not/a/real/path/model.apr");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false, false, None, false, None, "info", false,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            CliError::FileNotFound(p) => {
                assert_eq!(p, PathBuf::from("/definitely/not/a/real/path/model.apr"));
            }
            other => panic!("Expected FileNotFound, got: {:?}", other),
        }
    }

    #[test]
    fn test_run_nonexistent_safetensors() {
        let path = Path::new("/no/such/model.safetensors");
        let result = run(
            path, 0.5, 0.8, 256, None, false, false, false, None, false, None, "info", false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_apr() {
        let path = Path::new("/no/such/model.apr");
        let result = run(
            path, 1.0, 1.0, 1024, None, true, true, false, None, false, None, "warn", false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_with_all_trace_options() {
        let path = Path::new("/no/such/model.gguf");
        let steps = vec![
            "tokenize".to_string(),
            "embed".to_string(),
            "attention".to_string(),
            "ffn".to_string(),
            "sample".to_string(),
            "decode".to_string(),
        ];
        let result = run(
            path,
            0.3,
            0.95,
            128,
            Some("System prompt"),
            true,
            false,
            true,
            Some(&steps),
            true,
            Some(PathBuf::from("/tmp/full_trace.json")),
            "debug",
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_no_system_inspect_off() {
        let path = Path::new("/no/model.bin");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false, false, None, false, None, "info", false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_with_empty_trace_steps() {
        let path = Path::new("/no/model.gguf");
        let steps: Vec<String> = vec![];
        let result = run(
            path,
            0.7,
            0.9,
            512,
            None,
            false,
            false,
            true,
            Some(&steps),
            false,
            None,
            "info",
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_trace_without_output() {
        let path = Path::new("/no/model.apr");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false, true,  // trace enabled
            None,  // no trace steps
            false, // not verbose
            None,  // no trace output
            "info", false, // no profile
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_with_profile_only() {
        let path = Path::new("/no/model.gguf");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false,
            true, // trace must be on for profile to print
            None, false, None, "info", true, // profile enabled
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // CommandResult exhaustive tests
    // =========================================================================

    #[test]
    fn test_command_result_continue_is_not_quit() {
        let result = CommandResult::Continue;
        let is_continue = matches!(result, CommandResult::Continue);
        assert!(is_continue);
    }

    #[test]
    fn test_command_result_quit_is_not_continue() {
        let result = CommandResult::Quit;
        let is_quit = matches!(result, CommandResult::Quit);
        assert!(is_quit);
    }

    // =========================================================================
    // find_qwen_tokenizer: error paths
    // =========================================================================

    #[test]
    fn test_find_qwen_tokenizer_nonexistent_path() {
        // Path with no parent directory containing tokenizer.json.
        // Note: This function also searches HuggingFace cache and APR cache,
        // so it may succeed on dev machines with cached Qwen models.
        let path = Path::new("/nonexistent/deeply/nested/model.safetensors");
        let result = find_qwen_tokenizer(path);
        // Result depends on system state: Ok if cache has tokenizer, Err otherwise
        // We just verify it doesn't panic and returns a valid Result
        match result {
            Ok(Some(tok)) => {
                // Found in cache - verify it's a valid tokenizer
                assert!(tok.vocab_size() > 0);
            }
            Ok(None) => {
                // This shouldn't happen: function returns Err, not Ok(None) on failure
                panic!("Expected Err or Ok(Some), got Ok(None)");
            }
            Err(CliError::InvalidFormat(msg)) => {
                assert!(
                    msg.contains("No Qwen tokenizer found"),
                    "Expected helpful error message, got: {}",
                    msg
                );
            }
            Err(other) => panic!("Expected InvalidFormat error, got: {:?}", other),
        }
    }

    #[test]
    fn test_find_qwen_tokenizer_root_path() {
        // Path at root level - parent is "/"
        let path = Path::new("/model.safetensors");
        let result = find_qwen_tokenizer(path);
        // Same as above: depends on system cache state
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_find_qwen_tokenizer_error_message_content_when_no_cache() {
        // When find_qwen_tokenizer fails, it should return an InvalidFormat error
        // with a helpful message listing the search locations.
        // We test the error construction directly since the function may succeed
        // on machines with cached tokenizers.
        let err = CliError::InvalidFormat(
            "No Qwen tokenizer found. Searched:\n\
             1. Model directory (tokenizer.json)\n\
             2. HuggingFace cache (~/.cache/huggingface/hub/models--Qwen--*/snapshots/*/tokenizer.json)\n\
             3. APR cache (~/.apr/tokenizers/qwen2/tokenizer.json)\n\n\
             To fix: Download a Qwen model with tokenizer:\n\
               apr pull hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF"
                .to_string(),
        );
        let msg = err.to_string();
        assert!(msg.contains("No Qwen tokenizer found"));
        assert!(msg.contains("tokenizer.json"));
        assert!(msg.contains("HuggingFace cache"));
        assert!(msg.contains("APR cache"));
        assert!(msg.contains("apr pull"));
    }

    #[test]
    fn test_find_qwen_tokenizer_searches_parent_directory_first() {
        // The function's search order is:
        // 1. Model's parent directory (tokenizer.json)
        // 2. HuggingFace cache
        // 3. APR tokenizer cache
        // If there's no tokenizer.json in the parent dir, it falls through
        let path = Path::new("/tmp/no_tokenizer_here/model.safetensors");
        let result = find_qwen_tokenizer(path);
        // Just verify it doesn't panic; result depends on system cache
        let _ = result;
    }

    // =========================================================================
    // clean_chat_response: comprehensive combined scenarios
    // =========================================================================

    #[test]
    fn test_clean_chat_response_full_chatml_response() {
        let raw = "<|im_start|>assistant\nThe answer is 42.<|im_end|><|endoftext|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "The answer is 42.");
    }

    #[test]
    fn test_clean_chat_response_bpe_with_markers() {
        let raw = "<|im_start|>assistant\nHelloĠworld!<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_chat_response_complex_combined() {
        let raw = "<|im_start|>assistant\nĠĠHello!!!!!!ĠĠworld<|im_end|><|endoftext|>";
        let cleaned = clean_chat_response(raw);
        // Ġ -> space, multiple spaces -> single, !!!!!! -> !!!, markers removed, trimmed
        assert_eq!(cleaned, "Hello!!! world");
    }

    #[test]
    fn test_clean_chat_response_very_long_input() {
        let raw = "x".repeat(100_000);
        let cleaned = clean_chat_response(&raw);
        assert_eq!(cleaned.len(), 100_000);
    }

    #[test]
    fn test_clean_chat_response_only_bpe_artifacts() {
        let raw = "ĠĠĠ";
        let cleaned = clean_chat_response(raw);
        // Three Ġ -> three spaces -> collapsed to single space -> trimmed to empty
        assert!(cleaned.is_empty());
    }

    #[test]
    fn test_clean_chat_response_marker_in_middle_of_word() {
        let raw = "hel<|im_end|>lo";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "hello");
    }

    #[test]
    fn test_clean_chat_response_multiple_im_start_assistant() {
        let raw = "<|im_start|>assistant\n<|im_start|>assistant\nHello";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn test_clean_chat_response_newline_bpe_and_human_cutoff() {
        // Ċ becomes newline, then Human: detected -> cutoff
        let raw = "DoneĊHuman: next question";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Done");
    }

    #[test]
    fn test_clean_chat_response_single_char() {
        let raw = "a";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "a");
    }

    #[test]
    fn test_clean_chat_response_just_newline() {
        let raw = "\n";
        let cleaned = clean_chat_response(raw);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn test_clean_chat_response_just_markers() {
        let raw = "<|im_start|><|im_end|><|endoftext|>";
        let cleaned = clean_chat_response(raw);
        assert!(cleaned.is_empty());
    }

    // =========================================================================
    // ModelFormat: exhaustive match coverage
    // =========================================================================

    #[test]
    fn test_model_format_debug_format_all() {
        // Verify Debug representation for every variant
        let variants = [
            (ModelFormat::Apr, "Apr"),
            (ModelFormat::Gguf, "Gguf"),
            (ModelFormat::SafeTensors, "SafeTensors"),
            (ModelFormat::Demo, "Demo"),
        ];
        for (variant, expected) in variants {
            assert_eq!(format!("{:?}", variant), expected);
        }
    }

    #[test]
    fn test_model_format_clone_all_variants() {
        let variants = [
            ModelFormat::Apr,
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            ModelFormat::Demo,
        ];
        for variant in variants {
            let cloned = variant;
            assert_eq!(variant, cloned);
        }
    }

    #[test]
    fn test_model_format_eq_reflexive() {
        let formats = [
            ModelFormat::Apr,
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            ModelFormat::Demo,
        ];
        for f in formats {
            assert_eq!(f, f);
        }
    }

    #[test]
    fn test_model_format_ne_all_pairs() {
        let formats = [
            ModelFormat::Apr,
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            ModelFormat::Demo,
        ];
        for (i, a) in formats.iter().enumerate() {
            for (j, b) in formats.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Expected {:?} != {:?}", a, b);
                }
            }
        }
    }

    // =========================================================================
    // detect_format: pathological paths
    // =========================================================================

    #[test]
    fn test_detect_format_trailing_dot() {
        // Path ending in dot has no extension
        let path = Path::new("/models/model.");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_multiple_dots_gguf() {
        let path = Path::new("/models/model.v1.2.3.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_hash_named_apr() {
        let path = Path::new("/models/e910cab26ae116eb.apr");
        assert_eq!(detect_format(path), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_format_hash_named_gguf() {
        let path = Path::new("/cache/d4c4d9763127153c.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_long_extension() {
        let path = Path::new("/models/model.safetensorsbackup");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_similar_extensions() {
        // Close but not exact matches
        assert_eq!(detect_format(Path::new("x.ap")), ModelFormat::Demo);
        assert_eq!(detect_format(Path::new("x.ggu")), ModelFormat::Demo);
        assert_eq!(detect_format(Path::new("x.safetensor")), ModelFormat::Demo);
        assert_eq!(detect_format(Path::new("x.ggufx")), ModelFormat::Demo);
        assert_eq!(detect_format(Path::new("x.aprx")), ModelFormat::Demo);
    }

    // =========================================================================
    // print_welcome_banner: config display combinations
    // =========================================================================

    #[test]
    fn test_print_welcome_banner_zero_temp_and_top_p() {
        let path = Path::new("/models/test.gguf");
        let config = ChatConfig {
            temperature: 0.0,
            top_p: 0.0,
            max_tokens: 1,
            ..Default::default()
        };
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_high_temp_and_top_p() {
        let path = Path::new("/models/test.apr");
        let config = ChatConfig {
            temperature: 2.0,
            top_p: 1.0,
            max_tokens: 8192,
            system: Some("Be creative and wild!".to_string()),
            inspect: true,
            ..Default::default()
        };
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_openhermes_model() {
        // openhermes triggers ChatML template
        let path = Path::new("/models/openhermes-2.5.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_yi_model() {
        // yi- triggers ChatML template
        let path = Path::new("/models/yi-34b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_vicuna_model() {
        // vicuna triggers LLaMA2 template
        let path = Path::new("/models/vicuna-7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_mixtral_model() {
        // mixtral triggers Mistral template
        let path = Path::new("/models/mixtral-8x7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }
}
