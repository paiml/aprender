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
#[cfg(not(feature = "inference"))]
use aprender::text::bpe::Qwen2BpeTokenizer;

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
        if let Some(steps) = trace_steps {
            eprintln!("  Trace steps: {}", steps.join(", "));
        }
        if trace_verbose {
            eprintln!("  Verbose mode enabled");
        }
        if let Some(ref path) = trace_output {
            eprintln!("  Output: {}", path.display());
        }
    }

    // trace_steps and trace_verbose reserved for future use
    let _ = (trace_steps, trace_verbose);

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
    use aprender::demo::Qwen2Config;
    use aprender::models::Qwen2Model;
    use aprender::text::bpe::Qwen2BpeTokenizer;
    use std::fs::File;
    use std::io::Read;

    /// Chat session using realizar for high-performance inference
    /// Y13: Architecture-agnostic (detected from model metadata)
    /// Y14: Format-agnostic (APR, GGUF, SafeTensors)
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
        /// Qwen2 model (for SafeTensors inference)
        qwen_model: Option<Qwen2Model>,
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
                            println!(
                                "{} Failed to load GGUF tokenizer: {} (using byte fallback)",
                                "Warning:".yellow(),
                                e
                            );
                            None
                        }
                    };
                    (tok, None)
                }
                ModelFormat::SafeTensors | ModelFormat::Apr => {
                    // Load Qwen tokenizer from tokenizer.json in model directory
                    let tok = if let Some(parent) = path.parent() {
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
                                    Some(tok)
                                }
                                Err(e) => {
                                    println!(
                                        "{} {}",
                                        "Warning: Failed to load tokenizer:".yellow(),
                                        e
                                    );
                                    None
                                }
                            }
                        } else {
                            println!(
                                "{}",
                                "Warning: No tokenizer.json found, using byte fallback".yellow()
                            );
                            None
                        }
                    } else {
                        None
                    };
                    (None, tok)
                }
                ModelFormat::Demo => (None, None),
            };

            // Load Qwen2Model for SafeTensors format
            let qwen_model = if format == ModelFormat::SafeTensors {
                // Try to load config.json for model architecture
                let config_path = path.parent().map(|p| p.join("config.json"));
                let qwen_config = if let Some(ref cp) = config_path {
                    if cp.exists() {
                        // Parse config.json to get model parameters
                        match std::fs::read_to_string(cp) {
                            Ok(json) => {
                                // Parse the config - extract key fields
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                                    let hidden_size =
                                        v["hidden_size"].as_u64().unwrap_or(896) as usize;
                                    let num_heads =
                                        v["num_attention_heads"].as_u64().unwrap_or(14) as usize;
                                    let num_kv_heads =
                                        v["num_key_value_heads"].as_u64().unwrap_or(2) as usize;
                                    let num_layers =
                                        v["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
                                    let vocab_size =
                                        v["vocab_size"].as_u64().unwrap_or(151936) as usize;
                                    let intermediate_size =
                                        v["intermediate_size"].as_u64().unwrap_or(4864) as usize;
                                    let rope_theta =
                                        v["rope_theta"].as_f64().unwrap_or(1_000_000.0);

                                    println!(
                                        "{} config: {} layers, {} hidden, {} heads",
                                        "Loaded".green(),
                                        num_layers,
                                        hidden_size,
                                        num_heads
                                    );

                                    Qwen2Config {
                                        hidden_size,
                                        num_attention_heads: num_heads,
                                        num_kv_heads,
                                        num_layers,
                                        vocab_size,
                                        max_seq_len: 32768,
                                        intermediate_size,
                                        rope_theta,
                                    }
                                } else {
                                    Qwen2Config::qwen2_0_5b_instruct()
                                }
                            }
                            Err(_) => Qwen2Config::qwen2_0_5b_instruct(),
                        }
                    } else {
                        Qwen2Config::qwen2_0_5b_instruct()
                    }
                } else {
                    Qwen2Config::qwen2_0_5b_instruct()
                };

                // Create model and load weights
                let mut model = Qwen2Model::new_uninitialized(&qwen_config);
                match model.load_from_safetensors(path) {
                    Ok(count) if count > 0 => {
                        println!("{} {} tensors from SafeTensors", "Loaded".green(), count);
                        model.eval();
                        Some(model)
                    }
                    Ok(0) => {
                        // GH-140: No tensors loaded - model has different naming convention
                        println!(
                            "{} No matching tensors found in SafeTensors file. \
                             Model may use different weight naming convention.",
                            "Error:".red()
                        );
                        println!(
                            "{}",
                            "Tip: Use GGUF format for better compatibility: apr run model.gguf"
                                .yellow()
                        );
                        None
                    }
                    Ok(_) => unreachable!(), // count is always >= 0
                    Err(e) => {
                        println!("{} Failed to load weights: {}", "Warning:".yellow(), e);
                        None
                    }
                }
            } else {
                None
            };

            // Detect chat template from model name (Toyota Way: Jidoka - auto-detect)
            let model_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let template_format = detect_format_from_name(model_name);
            let chat_template = auto_detect_template(model_name);

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
                qwen_model,
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

            let practical_max = config.max_tokens.min(128);

            // ChatML stop tokens for Qwen2/ChatML models
            let gen_config = QuantizedGenerateConfig {
                max_tokens: practical_max,
                temperature: config.temperature,
                top_k: 40,
                stop_tokens: vec![151645, 151643], // <|im_end|>, <|endoftext|>
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
            use realizar::apr_transformer::{AprTransformer, GenerateConfig};

            let transformer = AprTransformer::from_apr_bytes(&self.model_bytes)
                .map_err(|e| format!("Failed to load APR transformer: {e}"))?;

            // Use generate_with_cache for O(n) KV-cached generation (not O(n²) forward)
            let gen_config = GenerateConfig {
                max_tokens: config.max_tokens.min(128),
                temperature: config.temperature,
                top_p: config.top_p,
                top_k: 0,
                repetition_penalty: 1.0,
            };

            transformer
                .generate_with_cache(prompt, &gen_config)
                .map_err(|e| format!("APR generate failed: {e}"))
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
            let practical_max = config.max_tokens.min(128);

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
            // Use loaded Qwen2Model for SafeTensors inference
            let model = self.qwen_model.as_mut().ok_or_else(|| {
                "SafeTensors model not loaded. Check config.json and model weights.".to_string()
            })?;

            // Limit tokens for reasonable CPU inference speed
            let max_tokens = config.max_tokens.min(32);

            // Generate using the model
            let output = model.generate(prompt, max_tokens, config.temperature, config.top_p);

            // Return only the newly generated tokens (after prompt)
            Ok(output[prompt.len()..].to_vec())
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
                config.max_tokens.min(128), // Limit for speed
                config.temperature,
            )
        } else {
            self.model.generate(
                &input_ids,
                config.max_tokens.min(128), // Limit for speed
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
}
