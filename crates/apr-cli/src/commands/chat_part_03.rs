
// =============================================================================
// Fallback ChatSession without realizar (demo mode only)
// =============================================================================

/// Fallback ChatSession stub when inference feature is disabled.
/// Chat requires realizar for inference — aprender is training only.
#[cfg(not(feature = "inference"))]
struct ChatSession;

#[cfg(not(feature = "inference"))]
impl ChatSession {
    fn new(_path: &Path) -> Result<Self, CliError> {
        Err(CliError::ValidationFailed(
            "Chat requires the 'inference' feature (realizar). Rebuild with: \
             cargo install --path crates/apr-cli --features inference"
                .to_string(),
        ))
    }

    fn generate(&mut self, _user_input: &str, _config: &ChatConfig) -> String {
        unreachable!("ChatSession::new always returns Err without inference feature")
    }
}

// =============================================================================
// REPL implementation with inference feature
// =============================================================================

/// Read one line of user input. Returns `None` on EOF, `Some("")` for empty, `Some(text)` otherwise.
fn read_repl_line() -> Result<Option<String>, CliError> {
    print!("{}", "You: ".green().bold());
    io::stdout().flush()?;
    let mut input = String::new();
    if io::stdin().read_line(&mut input)? == 0 {
        println!();
        return Ok(None);
    }
    Ok(Some(input.trim().to_string()))
}

/// Generate a response, update history, and print (inference mode).
#[cfg(feature = "inference")]
fn generate_and_print(session: &mut ChatSession, input: &str, config: &ChatConfig) {
    let response = session.generate(input, config);
    session.add_to_history("user", input);
    session.add_to_history("assistant", &response);
    println!("{} {}", "Assistant:".blue().bold(), response);
    if config.inspect {
        print_inspection_info_inference(session);
    }
    println!();
}

/// Process a single REPL input line. Returns `true` to quit, `false` to continue.
#[cfg(feature = "inference")]
fn process_repl_input(
    input: &str,
    session: &mut ChatSession,
    config: &ChatConfig,
) -> Result<bool, CliError> {
    if input.starts_with('/') {
        match handle_command_inference(input, session)? {
            CommandResult::Continue => return Ok(false),
            CommandResult::Quit => return Ok(true),
        }
    }
    generate_and_print(session, input, config);
    Ok(false)
}

#[cfg(feature = "inference")]
fn run_repl(path: &Path, config: &ChatConfig) -> Result<(), CliError> {
    let mut session = ChatSession::new(path)?;

    while let Some(input) = read_repl_line()? {
        if input.is_empty() {
            continue;
        }
        if process_repl_input(&input, &mut session, config)? {
            break;
        }
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

    while let Some(input) = read_repl_line()? {
        if input.is_empty() {
            continue;
        }
        if input.starts_with('/') {
            match handle_command(&input, &mut session.history)? {
                CommandResult::Continue => continue,
                CommandResult::Quit => break,
            }
        }
        generate_and_print_fallback(&mut session, &input, config);
    }

    println!("{}", "Goodbye!".cyan());
    Ok(())
}

/// Generate a response, update history, and print (fallback mode).
#[cfg(not(feature = "inference"))]
fn generate_and_print_fallback(session: &mut ChatSession, input: &str, config: &ChatConfig) {
    let response = session.generate(input, config);
    session.history.push(format!("user: {input}"));
    session.history.push(format!("assistant: {response}"));
    println!("{} {}", "Assistant:".blue().bold(), response);
    if config.inspect {
        print_inspection_info(session);
    }
    println!();
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
