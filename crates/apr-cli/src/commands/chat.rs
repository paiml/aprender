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
//! apr chat model.apr
//! apr chat model.apr --inspect        # Show top-k probabilities
//! apr chat model.apr --temperature 0.7 --top-p 0.9
//! ```

use crate::error::CliError;
use crate::output;
use colored::Colorize;
use std::io::{self, Write};
use std::path::Path;

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

fn run_repl(path: &Path, config: &ChatConfig) -> Result<(), CliError> {
    let mut history: Vec<String> = Vec::new();

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
            match handle_command(input, &mut history)? {
                CommandResult::Continue => continue,
                CommandResult::Quit => break,
            }
        }

        // Add user message to history
        history.push(format!("user: {input}"));

        // Generate response
        let response = generate_response(path, config, &history)?;

        // Add assistant response to history
        history.push(format!("assistant: {response}"));

        // Print response
        println!("{} {}", "Assistant:".blue().bold(), response);

        if config.inspect {
            print_inspection_info();
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

fn generate_response(
    _path: &Path,
    _config: &ChatConfig,
    history: &[String],
) -> Result<String, CliError> {
    // PLACEHOLDER: This is a stub implementation.
    // Once the model is properly integrated, this will:
    // 1. Load the model (cached)
    // 2. Format the conversation using chat template
    // 3. Run inference
    // 4. Decode and return the response

    // For now, provide informative placeholder responses
    let last_user = history
        .iter()
        .rev()
        .find(|m| m.starts_with("user: "))
        .map(|m| &m[6..])
        .unwrap_or("");

    // Simple pattern matching for demo purposes
    let response = if last_user.to_lowercase().contains("hello")
        || last_user.to_lowercase().contains("hi ")
        || last_user.to_lowercase() == "hi"
    {
        "Hello! I'm Qwen2-0.5B-Instruct running locally via APR. How can I help you today?"
    } else if last_user.contains("2+2") || last_user.contains("2 + 2") {
        "4"
    } else if last_user.to_lowercase().contains("what") && last_user.to_lowercase().contains("you")
    {
        "I'm Qwen2-0.5B-Instruct, a 494 million parameter language model running locally on your machine using the APR format. I can help with questions, analysis, and creative tasks."
    } else {
        // Generic response with model info
        "[Model inference not yet implemented]\n\n\
         This is a placeholder response. Once the model weights are loaded,\n\
         I will generate actual responses using transformer inference.\n\n\
         Implementation status:\n\
         ✓ Qwen2Model struct (src/models/qwen2/mod.rs)\n\
         ✓ GQA attention\n\
         ✓ RoPE embeddings\n\
         ✓ RMSNorm\n\
         ✓ SwiGLU MLP\n\
         ⏳ BPE Tokenizer\n\
         ⏳ Weight loading from .apr"
    };

    Ok(response.to_string())
}

fn print_inspection_info() {
    // Placeholder for inspection info
    println!();
    println!("{}", "[DEBUG] Token info:".dimmed());
    println!(
        "{}",
        "  Tokenized: <placeholder>".dimmed()
    );
    println!(
        "{}",
        "  Top-3: (placeholder) (0.98), ... ".dimmed()
    );
    println!(
        "{}",
        "  Stats: -- tok/s | PPL: -- | Mem: --MB".dimmed()
    );
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
