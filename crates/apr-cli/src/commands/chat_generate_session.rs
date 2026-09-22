
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

/// Render one assistant turn from a generation result — and the ONE place a generation
/// failure is recorded on the session.
///
/// #3367: `echo hey | apr chat model.gguf` hit a generate error, printed it as an
/// assistant turn (`Assistant: [Error: GGUF generate failed: ...]`) and exited **0**, so
/// a gate keyed on the return code read a failed generation as a pass. The error text is
/// still returned unchanged — an interactive user must keep seeing it — but the failure
/// is no longer invisible to the caller: `had_generate_error` is what
/// [`session_exit_result`] turns into a non-zero exit at session end.
///
/// Mid-session behaviour is deliberately unchanged. The REPL's existing convention is
/// that only I/O errors abort the loop (`read_repl_line()?`); a generation error prints
/// and the conversation continues. This records the failure instead of aborting on it.
#[cfg_attr(not(feature = "inference"), allow(dead_code))]
fn render_assistant_turn(result: Result<String, String>, had_generate_error: &mut bool) -> String {
    match result {
        Ok(text) => text,
        Err(e) => {
            *had_generate_error = true;
            format!("[Error: {}]", e)
        }
    }
}

/// The exit-code decision at session end (#3367).
///
/// A session in which any turn failed to generate exits non-zero, with
/// [`CliError::InferenceFailed`] — exit **8** per `error.rs`, the code `main` already
/// maps a command `Err` to. Nothing else about the session changes.
#[cfg_attr(not(feature = "inference"), allow(dead_code))]
fn session_exit_result(had_generate_error: bool) -> Result<(), CliError> {
    if had_generate_error {
        return Err(CliError::InferenceFailed(
            "chat session ended after a failed generation (#3367); the [Error: ...] turn(s) \
             above are not model output"
                .to_string(),
        ));
    }
    Ok(())
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
    let mut session = ChatSession::new(path, config.force_cpu)?;

    while let Some(input) = read_repl_line()? {
        if input.is_empty() {
            continue;
        }
        if process_repl_input(&input, &mut session, config)? {
            break;
        }
    }

    println!("{}", "Goodbye!".cyan());
    if config.json {
        println!("{}", chat_backend_report(config, session.generated_on_gpu()));
    }
    // #3367: the session's verdict, not the loop's. Every turn was printed as it
    // happened; this is only the exit code catching up with what was printed.
    session_exit_result(session.had_generate_error())
}

/// #3794: the machine-readable backend line, mirroring `apr run --format json`'s
/// `backend: {requested, ran, fell_back}`.
///
/// `ran` is what ACTUALLY answered — recorded by the generate branches
/// themselves, not inferred from the flags — so a harness can hold `apr chat` to
/// its lane the way the `run` cells already hold `apr run`. Pure and
/// string-returning so the shape is unit-testable without a model or a GPU.
#[cfg(feature = "inference")]
pub(crate) fn chat_backend_report(config: &ChatConfig, generated_on_gpu: bool) -> String {
    let requested = if config.force_cpu { "cpu" } else { "default" };
    let ran = if generated_on_gpu { "gpu" } else { "cpu" };
    // A fallback is a request for an accelerator that CPU answered. Asking for
    // cpu and getting cpu is not a fallback, and neither is never asking.
    let fell_back = !config.force_cpu && !generated_on_gpu;
    format!(
        r#"{{"backend":{{"requested":"{requested}","ran":"{ran}","fell_back":{fell_back}}}}}"#
    )
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

/// Process a single REPL input line (fallback). Returns `true` to quit, `false` to continue.
#[cfg(not(feature = "inference"))]
fn process_repl_input_fallback(
    input: &str,
    session: &mut ChatSession,
    config: &ChatConfig,
) -> Result<bool, CliError> {
    let _ = (input, session, config);
    unreachable!("ChatSession::new always returns Err without inference feature")
}

#[cfg(not(feature = "inference"))]
fn run_repl(path: &Path, config: &ChatConfig) -> Result<(), CliError> {
    let mut session = ChatSession::new(path, config.force_cpu)?;

    while let Some(input) = read_repl_line()? {
        if input.is_empty() {
            continue;
        }
        if process_repl_input_fallback(&input, &mut session, config)? {
            break;
        }
    }

    println!("{}", "Goodbye!".cyan());
    Ok(())
}

/// Generate a response, update history, and print (fallback mode).
#[cfg(not(feature = "inference"))]
fn generate_and_print_fallback(_session: &mut ChatSession, _input: &str, _config: &ChatConfig) {
    unreachable!("ChatSession::new always returns Err without inference feature")
}

enum CommandResult {
    Continue,
    Quit,
}

#[cfg(not(feature = "inference"))]
fn handle_command(_input: &str, _history: &mut Vec<String>) -> Result<CommandResult, CliError> {
    unreachable!("ChatSession::new always returns Err without inference feature")
}

#[cfg(not(feature = "inference"))]
fn print_inspection_info(_session: &ChatSession) {
    unreachable!("ChatSession::new always returns Err without inference feature")
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

/// #3794: the case table for `chat_backend_report`.
///
/// `fell_back` is the field a harness acts on, and it is the one easiest to get
/// backwards: asking for CPU and getting CPU is not a fallback, and neither is
/// never asking. Only an unasked-for demotion is.
#[cfg(all(test, feature = "inference"))]
mod pmat3794_chat_backend_report {
    use super::{chat_backend_report, ChatConfig};

    fn report(force_cpu: bool, on_gpu: bool) -> String {
        let config = ChatConfig {
            force_cpu,
            json: true,
            ..Default::default()
        };
        chat_backend_report(&config, on_gpu)
    }

    /// `(force_cpu, generated_on_gpu, expected, what this invocation is)`
    const CASES: &[(bool, bool, &str, &str)] = &[
        (
            true, false,
            r#"{"backend":{"requested":"cpu","ran":"cpu","fell_back":false}}"#,
            "`apr chat --no-gpu` — the #3794 invocation. Asked for CPU, got CPU: \
             NOT a fallback",
        ),
        (
            false, true,
            r#"{"backend":{"requested":"default","ran":"gpu","fell_back":false}}"#,
            "`apr chat` on a CUDA build — the accelerator answered",
        ),
        (
            false, false,
            r#"{"backend":{"requested":"default","ran":"cpu","fell_back":true}}"#,
            "`apr chat` where the accelerator did NOT answer — the only real \
             fallback, and the case a harness needs to see",
        ),
        (
            true, true,
            r#"{"backend":{"requested":"cpu","ran":"gpu","fell_back":false}}"#,
            "force_cpu yet the GPU answered — impossible after #3794's gate, and \
             reported honestly rather than smoothed over, so the contradiction is \
             visible if the gate ever regresses",
        ),
    ];

    #[test]
    fn the_whole_report_table_holds() {
        let wrong: Vec<String> = CASES
            .iter()
            .filter(|(f, g, want, _)| report(*f, *g) != **want)
            .map(|(f, g, want, why)| {
                format!(
                    "\n  - force_cpu={f} generated_on_gpu={g}: {why}\n      expected {want}\n      got      {}",
                    report(*f, *g)
                )
            })
            .collect();
        assert!(
            wrong.is_empty(),
            "{} of {} chat backend reports are wrong:{}",
            wrong.len(),
            CASES.len(),
            wrong.join("")
        );
    }

    /// The field a harness acts on, asserted on its own so a regression in it
    /// cannot hide inside a whole-string diff.
    #[test]
    fn only_an_unasked_for_demotion_is_a_fallback() {
        assert!(
            report(false, false).contains(r#""fell_back":true"#),
            "#3794: a default chat answered by the CPU is a fallback and must say so"
        );
        for (f, g) in [(true, false), (false, true), (true, true)] {
            assert!(
                report(f, g).contains(r#""fell_back":false"#),
                "#3794: force_cpu={f} generated_on_gpu={g} is not a fallback"
            );
        }
    }
}
