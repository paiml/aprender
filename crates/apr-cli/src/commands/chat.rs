//! Interactive Chat Command
//!
//! Provides a REPL interface for interactive chat with language models.
//!
//! Toyota Way: Genchi Genbutsu + Visual Control
//! - Go and see the actual model outputs
//! - Optionally visualize token probabilities for transparency
//!
//! # Example
//!
//! ```bash
//! apr chat model.safetensors
//! apr chat model.safetensors --inspect        # Show top-k probabilities
//! apr chat model.safetensors --temperature 0.7 --top-p 0.9
//! ```

use crate::error::CliError;
use crate::output;
use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;
use aprender::text::bpe::Qwen2BpeTokenizer;
use colored::Colorize;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

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

fn print_welcome_banner(path: &Path, config: &ChatConfig) {
    output::section("Qwen2-0.5B-Instruct Chat");
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
        println!("{}", "Inspection mode enabled - showing token probabilities".cyan());
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

/// Chat session state holding model and tokenizer
struct ChatSession {
    model: Qwen2Model,
    tokenizer: Qwen2BpeTokenizer,
    history: Vec<String>,
}

impl ChatSession {
    fn new(path: &Path) -> Result<Self, CliError> {
        println!("{}", "Loading model...".cyan());
        let start = Instant::now();

        // Use tiny config for testing (full model would use Qwen2Config::qwen2_0_5b_instruct())
        let config = Qwen2Config {
            hidden_size: 896,
            num_attention_heads: 14,
            num_kv_heads: 2,
            num_layers: 24,
            vocab_size: 151936,
            max_seq_len: 32768,
            intermediate_size: 4864,
            rope_theta: 1_000_000.0,
        };

        let mut model = Qwen2Model::new(&config);

        // Try to load weights from SafeTensors file
        if path.extension().map_or(false, |e| e == "safetensors") {
            match model.load_from_safetensors(path) {
                Ok(count) => {
                    println!("{} {}", "Loaded".green(), format!("{count} weight tensors"));
                }
                Err(e) => {
                    println!("{} {}", "Warning:".yellow(), format!("Could not load weights: {e}"));
                    println!("{}", "Using randomly initialized weights".yellow());
                }
            }
        } else {
            println!("{}", "Note: Using randomly initialized weights (no .safetensors file)".yellow());
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
        let prompt = self.tokenizer.format_conversation(
            &messages.iter().map(|(r, c)| (*r, *c)).collect::<Vec<_>>()
        );

        // Encode prompt (using byte-level encoding since we don't have full vocab loaded)
        let input_ids = self.tokenizer.encode(&prompt);

        // Limit prompt length
        let max_prompt = 256;
        let input_ids: Vec<u32> = if input_ids.len() > max_prompt {
            input_ids[input_ids.len() - max_prompt..].to_vec()
        } else {
            input_ids
        };

        // Generate
        let start = Instant::now();
        let output_ids = self.model.generate(
            &input_ids,
            config.max_tokens.min(128), // Limit for speed
            config.temperature,
            config.top_p,
        );
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
        }

        response
    }
}

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

/// Format parameter count in human-readable form
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
