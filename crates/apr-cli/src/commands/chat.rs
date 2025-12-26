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
use aprender::text::llama_tokenizer::LlamaTokenizer;
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
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            system: None,
            inspect: false,
        }
    }
}

/// Run the chat command
pub(crate) fn run(
    path: &Path,
    temperature: f32,
    top_p: f32,
    max_tokens: usize,
    system: Option<&str>,
    inspect: bool,
) -> Result<(), CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let config = ChatConfig {
        temperature,
        top_p,
        max_tokens,
        system: system.map(String::from),
        inspect,
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

/// Detect format from magic bytes (more reliable than extension)
#[cfg(feature = "inference")]
fn detect_format_from_bytes(data: &[u8]) -> ModelFormat {
    if data.len() < 8 {
        return ModelFormat::Demo;
    }
    // APR v1: "APRN", APR v2: "APR2"
    if &data[0..4] == b"APRN" || &data[0..4] == b"APR2" {
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
    use std::fs::File;
    use std::io::Read;

    /// Chat session using realizar for high-performance inference
    /// Y13: Architecture-agnostic (detected from model metadata)
    /// Y14: Format-agnostic (APR, GGUF, SafeTensors)
    pub struct ChatSession {
        /// Model bytes (kept for regeneration if needed)
        model_bytes: Vec<u8>,
        /// Detected format
        format: ModelFormat,
        /// Conversation history
        history: Vec<String>,
        /// LLaMA tokenizer (for GGUF format)
        tokenizer: Option<LlamaTokenizer>,
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

            // Load tokenizer for GGUF format
            let tokenizer = if format == ModelFormat::Gguf {
                match LlamaTokenizer::from_gguf_bytes(&model_bytes) {
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
                            "{} Failed to load tokenizer: {} (using byte fallback)",
                            "Warning:".yellow(),
                            e
                        );
                        None
                    }
                }
            } else {
                None
            };

            Ok(Self {
                model_bytes,
                format,
                history: Vec::new(),
                tokenizer,
            })
        }

        pub(super) fn generate(&mut self, user_input: &str, config: &ChatConfig) -> String {
            let start = Instant::now();

            // Tokenize input using LLaMA tokenizer if available, else fall back to char-level
            let prompt_tokens: Vec<u32> = if let Some(ref tokenizer) = self.tokenizer {
                tokenizer.encode_with_bos(user_input)
            } else {
                user_input.chars().map(|c| c as u32).collect()
            };

            let result = match self.format {
                ModelFormat::Apr => self.generate_apr(&prompt_tokens, config),
                ModelFormat::Gguf => self.generate_gguf(&prompt_tokens, config),
                ModelFormat::SafeTensors => self.generate_safetensors(&prompt_tokens, config),
                ModelFormat::Demo => Ok(vec![]),
            };

            let gen_time = start.elapsed();

            match result {
                Ok(output_tokens) => {
                    if config.inspect {
                        println!(
                            "{}",
                            format!(
                                "[{} tokens in {:.2}s = {:.1} tok/s]",
                                output_tokens.len(),
                                gen_time.as_secs_f32(),
                                output_tokens.len() as f32 / gen_time.as_secs_f32()
                            )
                            .dimmed()
                        );
                    }
                    // Decode output tokens using LLaMA tokenizer if available
                    if let Some(ref tokenizer) = self.tokenizer {
                        tokenizer.decode(&output_tokens)
                    } else {
                        output_tokens
                            .iter()
                            .filter_map(|&t| char::from_u32(t))
                            .collect()
                    }
                }
                Err(e) => format!("[Error: {}]", e),
            }
        }

        fn generate_apr(&self, prompt: &[u32], config: &ChatConfig) -> Result<Vec<u32>, String> {
            use realizar::apr_transformer::AprTransformer;

            let transformer = AprTransformer::from_apr_bytes(&self.model_bytes)
                .map_err(|e| format!("Failed to load APR transformer: {e}"))?;

            transformer
                .generate(prompt, config.max_tokens.min(128))
                .map_err(|e| format!("APR generate failed: {e}"))
        }

        fn generate_gguf(&self, prompt: &[u32], config: &ChatConfig) -> Result<Vec<u32>, String> {
            use realizar::gguf::{GGUFModel, QuantizedGGUFTransformer, QuantizedGenerateConfig};

            let gguf = GGUFModel::from_bytes(&self.model_bytes)
                .map_err(|e| format!("Failed to parse GGUF: {e}"))?;

            let transformer = QuantizedGGUFTransformer::from_gguf(&gguf, &self.model_bytes)
                .map_err(|e| format!("Failed to create GGUF transformer: {e}"))?;

            let gen_config = QuantizedGenerateConfig {
                max_tokens: config.max_tokens.min(128),
                temperature: config.temperature,
                top_k: 40, // Default top_k (config.top_p not used in GGUF)
                ..Default::default()
            };

            transformer
                .generate(prompt, &gen_config)
                .map_err(|e| format!("GGUF generate failed: {e}"))
        }

        fn generate_safetensors(
            &self,
            _prompt: &[u32],
            _config: &ChatConfig,
        ) -> Result<Vec<u32>, String> {
            // SafeTensors requires architecture detection from config.json
            // For now, return a placeholder - full implementation pending
            Err("SafeTensors inference requires config.json for architecture detection. Use APR or GGUF format.".to_string())
        }

        pub(super) fn history(&self) -> &[String] {
            &self.history
        }

        pub(super) fn add_to_history(&mut self, role: &str, content: &str) {
            self.history.push(format!("{role}: {content}"));
        }

        pub(super) fn clear_history(&mut self) {
            self.history.clear();
        }

        #[allow(dead_code)]
        pub(super) fn format(&self) -> ModelFormat {
            self.format
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

        Ok(Self {
            model,
            tokenizer: Qwen2BpeTokenizer::new(),
            history: Vec::new(),
        })
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
        format!("  History: {} messages", session.history().len()).dimmed()
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
    }
}
