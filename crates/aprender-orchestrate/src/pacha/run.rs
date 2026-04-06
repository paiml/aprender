//! Pacha Run Command - Interactive Chat
//!
//! Implements the interactive chat functionality for the Pacha CLI,
//! providing an ollama-like experience for model inference.

use crate::ansi_colors::Colorize;
use std::io::{self, Write};

use super::helpers::{print_command_header, truncate_str};

// ============================================================================
// PACHA-CLI-013: Run Command (Interactive Chat)
// ============================================================================

/// Chat message structure
#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}
impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".to_string(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".to_string(), content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".to_string(), content: content.into() }
    }
}

/// Result of handling a chat command
pub enum ChatCommandResult {
    Continue,
    Exit,
    Error(String),
}

/// Simple modelfile parser for run command
pub struct SimpleModelfile {
    pub system: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
}

// ============================================================================
// Configuration Loading
// ============================================================================

/// Load effective chat configuration from modelfile or defaults.
pub fn load_chat_config(
    system: Option<&str>,
    modelfile: Option<&str>,
    temperature: f32,
    max_tokens: Option<usize>,
) -> anyhow::Result<(Option<String>, f32, Option<usize>)> {
    if let Some(mf_path) = modelfile {
        let content = std::fs::read_to_string(mf_path)?;
        let manifest = parse_simple_modelfile(&content)?;
        Ok((
            manifest.system.or_else(|| system.map(String::from)),
            manifest.temperature.unwrap_or(temperature),
            manifest.max_tokens.or(max_tokens),
        ))
    } else {
        Ok((system.map(String::from), temperature, max_tokens))
    }
}

/// Parse a PARAMETER directive from a modelfile line.
fn parse_modelfile_parameter(value: &str, result: &mut SimpleModelfile) {
    let param_parts: Vec<&str> = value.splitn(2, char::is_whitespace).collect();
    if param_parts.len() == 2 {
        match param_parts[0].to_lowercase().as_str() {
            "temperature" => result.temperature = param_parts[1].parse().ok(),
            "max_tokens" | "num_predict" => result.max_tokens = param_parts[1].parse().ok(),
            _ => {}
        }
    }
}

/// Parse a simple modelfile format
pub fn parse_simple_modelfile(content: &str) -> anyhow::Result<SimpleModelfile> {
    let mut result = SimpleModelfile { system: None, temperature: None, max_tokens: None };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            continue;
        }

        match parts[0].to_uppercase().as_str() {
            "SYSTEM" => result.system = Some(parts[1].to_string()),
            "PARAMETER" => parse_modelfile_parameter(parts[1], &mut result),
            _ => {}
        }
    }

    Ok(result)
}

// ============================================================================
// Chat Display
// ============================================================================

/// Print chat session header with model info.
fn print_chat_header(
    model: &str,
    system: &Option<String>,
    temp: f32,
    context: usize,
    max_tokens: Option<usize>,
) {
    print_command_header("Interactive Chat");
    println!("Model:       {}", model.cyan());
    if let Some(ref sys) = system {
        println!("System:      {}", truncate_str(sys, 50).dimmed());
    }
    println!("Temperature: {}", format!("{:.1}", temp).yellow());
    println!("Context:     {} tokens", context);
    if let Some(max) = max_tokens {
        println!("Max Tokens:  {}", max);
    }
    println!();
    println!("{}", "Type your message and press Enter. Commands:".dimmed());
    println!("{}", "  /bye, /exit, /quit - Exit chat".dimmed());
    println!("{}", "  /clear             - Clear context".dimmed());
    println!("{}", "  /system <prompt>   - Change system prompt".dimmed());
    println!("{}", "  /temp <value>      - Change temperature".dimmed());
    println!("{}", "  /save <file>       - Save conversation".dimmed());
    println!();
    println!("{}", "-".repeat(60).dimmed());
}

// ============================================================================
// Chat Commands
// ============================================================================

/// Handle /clear command: reset messages, re-add system prompt if set.
fn chat_cmd_clear(
    messages: &mut Vec<ChatMessage>,
    current_system: &Option<String>,
) -> ChatCommandResult {
    messages.clear();
    if let Some(ref sys) = current_system {
        messages.push(ChatMessage::system(sys.clone()));
    }
    println!("{} Context cleared", "ok".bright_green());
    ChatCommandResult::Continue
}

/// Handle /system command: update system prompt.
fn chat_cmd_system(
    arg: Option<&str>,
    messages: &mut Vec<ChatMessage>,
    current_system: &mut Option<String>,
) -> ChatCommandResult {
    let Some(prompt) = arg else {
        return ChatCommandResult::Error("Usage: /system <prompt>".to_string());
    };
    *current_system = Some(prompt.to_string());
    if let Some(msg) = messages.iter_mut().find(|m| m.role == "system") {
        msg.content = prompt.to_string();
    } else {
        messages.insert(0, ChatMessage::system(prompt));
    }
    println!("{} System prompt updated", "ok".bright_green());
    ChatCommandResult::Continue
}

/// Handle /temp command: set temperature.
fn chat_cmd_temp(arg: Option<&str>, current_temp: &mut f32) -> ChatCommandResult {
    let Some(val) = arg else {
        return ChatCommandResult::Error("Usage: /temp <value>".to_string());
    };
    match val.parse::<f32>() {
        Ok(t) if (0.0..=2.0).contains(&t) => {
            *current_temp = t;
            println!("{} Temperature set to {:.1}", "ok".bright_green(), t);
            ChatCommandResult::Continue
        }
        _ => ChatCommandResult::Error("Temperature must be between 0.0 and 2.0".to_string()),
    }
}

/// Handle /save command: save conversation to file.
fn chat_cmd_save(arg: Option<&str>, messages: &[ChatMessage]) -> ChatCommandResult {
    let Some(path) = arg else {
        return ChatCommandResult::Error("Usage: /save <file>".to_string());
    };
    match save_conversation(messages, path) {
        Ok(()) => {
            println!("{} Conversation saved to {}", "ok".bright_green(), path);
            ChatCommandResult::Continue
        }
        Err(e) => ChatCommandResult::Error(format!("Failed to save: {}", e)),
    }
}

/// Handle chat slash commands
fn handle_chat_command(
    input: &str,
    messages: &mut Vec<ChatMessage>,
    current_system: &mut Option<String>,
    current_temp: &mut f32,
) -> ChatCommandResult {
    let parts: Vec<&str> = input.splitn(2, char::is_whitespace).collect();
    let cmd = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim());

    match cmd.as_str() {
        "/bye" | "/exit" | "/quit" => ChatCommandResult::Exit,
        "/clear" => chat_cmd_clear(messages, current_system),
        "/system" => chat_cmd_system(arg, messages, current_system),
        "/temp" => chat_cmd_temp(arg, current_temp),
        "/save" => chat_cmd_save(arg, messages),
        "/help" => {
            println!("{}", "Commands:".bright_white().bold());
            println!("  /bye, /exit, /quit - Exit chat");
            println!("  /clear             - Clear context");
            println!("  /system <prompt>   - Change system prompt");
            println!("  /temp <value>      - Change temperature");
            println!("  /save <file>       - Save conversation");
            println!("  /help              - Show this help");
            ChatCommandResult::Continue
        }
        _ => ChatCommandResult::Error(format!("Unknown command: {}", cmd)),
    }
}

// ============================================================================
// Response Generation
// ============================================================================

/// Generate simulated response (placeholder for real inference)
pub fn generate_simulated_response(input: &str, _messages: &[ChatMessage]) -> String {
    // Simple pattern matching for demo purposes
    let input_lower = input.to_lowercase();

    if input_lower.contains("hello") || input_lower.contains("hi") {
        return "Hello! How can I help you today?".to_string();
    }

    if input_lower.contains("how are you") {
        return "I'm doing well, thank you for asking! I'm ready to assist you with any questions or tasks you might have.".to_string();
    }

    if input_lower.contains("what is") || input_lower.contains("explain") {
        return format!(
            "That's an interesting question about \"{}\"! Let me explain: This is a simulated response. In a real implementation, I would provide a detailed explanation based on my training data and the context of our conversation.",
            input.chars().take(30).collect::<String>()
        );
    }

    if input_lower.contains("code") || input_lower.contains("program") {
        return "Here's a simple example:\n\n```rust\nfn main() {\n    println!(\"Hello, world!\");\n}\n```\n\nThis is a basic Rust program that prints a greeting. Would you like me to explain any part of it?".to_string();
    }

    // Default response
    format!(
        "I understand you're asking about \"{}\". This is a simulated response for demonstration purposes. In production, this would use the actual inference engine to generate contextually appropriate responses.",
        truncate_str(input, 40)
    )
}

// ============================================================================
// Context Management
// ============================================================================

/// Truncate context to fit within window size
pub fn truncate_context(messages: &mut Vec<ChatMessage>, max_tokens: usize, has_system: bool) {
    // Keep system message and recent messages
    let start_idx = if has_system { 1 } else { 0 };

    while messages.len() > start_idx + 2 {
        let token_estimate: usize = messages.iter().map(|m| m.content.len() / 4).sum();
        if token_estimate <= max_tokens {
            break;
        }
        // Remove oldest non-system message
        messages.remove(start_idx);
    }
}

/// Save conversation to file
fn save_conversation(messages: &[ChatMessage], path: &str) -> anyhow::Result<()> {
    let mut output = String::new();

    for msg in messages {
        output.push_str(&format!("[{}]\n{}\n\n", msg.role.to_uppercase(), msg.content));
    }

    std::fs::write(path, output)?;
    Ok(())
}

// ============================================================================
// Chat Loop
// ============================================================================

/// Display a simulated streamed response and manage context window.
fn display_streamed_response(
    input: &str,
    messages: &mut Vec<ChatMessage>,
    context: usize,
    verbose: bool,
    has_system: bool,
) -> anyhow::Result<()> {
    print!("\n{} ", "<<<".bright_cyan().bold());
    io::stdout().flush()?;

    let response = generate_simulated_response(input, messages);
    for chunk in response.chars() {
        print!("{}", chunk);
        io::stdout().flush()?;
        std::thread::sleep(std::time::Duration::from_millis(15));
    }
    println!();

    messages.push(ChatMessage::assistant(response));

    let token_estimate: usize = messages.iter().map(|m| m.content.len() / 4).sum();
    if token_estimate > context {
        if verbose {
            println!("{}", format!("[Context truncated: ~{} tokens]", token_estimate).dimmed());
        }
        truncate_context(messages, context, has_system);
    }

    Ok(())
}

/// Run one iteration of the chat loop. Returns false to exit.
fn chat_loop_iteration(
    messages: &mut Vec<ChatMessage>,
    current_system: &mut Option<String>,
    current_temp: &mut f32,
    context: usize,
    verbose: bool,
) -> anyhow::Result<bool> {
    use std::io::BufRead;

    print!("\n{} ", ">>>".bright_green().bold());
    io::stdout().flush()?;

    let mut input = String::new();
    match io::stdin().lock().read_line(&mut input) {
        Ok(0) => {
            println!();
            return Ok(false);
        }
        Ok(_) => {}
        Err(e) => {
            println!("{} Input error: {}", "error:".red(), e);
            return Ok(true);
        }
    }

    let input = input.trim();
    if input.is_empty() {
        return Ok(true);
    }

    if input.starts_with('/') {
        return Ok(match handle_chat_command(input, messages, current_system, current_temp) {
            ChatCommandResult::Exit => false,
            ChatCommandResult::Error(msg) => {
                println!("{} {}", "warning:".yellow(), msg);
                true
            }
            ChatCommandResult::Continue => true,
        });
    }

    messages.push(ChatMessage::user(input));

    display_streamed_response(input, messages, context, verbose, current_system.is_some())?;

    Ok(true)
}

/// Run interactive chat with a model
pub fn cmd_run(
    model: &str,
    system: Option<&str>,
    modelfile: Option<&str>,
    temperature: f32,
    max_tokens: Option<usize>,
    context: usize,
    verbose: bool,
) -> anyhow::Result<()> {
    let (effective_system, effective_temp, effective_max_tokens) =
        load_chat_config(system, modelfile, temperature, max_tokens)?;

    print_chat_header(model, &effective_system, effective_temp, context, effective_max_tokens);

    if verbose {
        println!("{}", "Loading model...".dimmed());
        std::thread::sleep(std::time::Duration::from_millis(500));
        println!("{} Model loaded", "ok".bright_green());
        println!();
    }

    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut current_system = effective_system.clone();
    let mut current_temp = effective_temp;

    if let Some(ref sys) = current_system {
        messages.push(ChatMessage::system(sys.clone()));
    }

    while chat_loop_iteration(
        &mut messages,
        &mut current_system,
        &mut current_temp,
        context,
        verbose,
    )? {}

    println!();
    println!("{} Chat ended. Goodbye!", "bye".bright_cyan());
    Ok(())
}
