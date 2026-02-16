
/// Publish model to Hugging Face Hub (GH-100)
fn cmd_publish(model_path: &str, repo_id: &str, commit_msg: &str, create: bool, private: bool) {
    use aprender::format::model_card::{ModelCard, TrainingDataInfo};

    let path = expand_path(model_path);

    // Load model to get metadata
    let model = match MarkovModel::load(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Failed to load model: {e}");
            std::process::exit(1);
        }
    };

    // Read model file bytes
    let model_bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("❌ Failed to read model file: {e}");
            std::process::exit(1);
        }
    };

    // Generate model card
    let model_id = format!(
        "aprender-shell-markov-{}gram-{}",
        model.ngram_size(),
        chrono_lite_date()
    );

    let card = ModelCard::new(&model_id, "1.0.0")
        .with_name("Shell Completion Model")
        .with_description(
            "Markov chain model trained on shell command history for intelligent tab completion",
        )
        .with_architecture("MarkovModel")
        .with_license("MIT")
        .with_param_count(model.ngram_count() as u64)
        .with_hyperparameter("ngram_size", model.ngram_size())
        .with_metric("vocab_size", model.vocab_size())
        .with_metric("ngram_count", model.ngram_count())
        .with_training_data(
            TrainingDataInfo::new("shell_history").with_samples(model.total_commands() as u64),
        );

    println!("📤 Publishing to Hugging Face Hub...\n");
    println!("  Repository: {repo_id}");
    println!("  Model:      {}", path.display());
    println!("  Size:       {} bytes", model_bytes.len());
    println!("  N-gram:     {}", model.ngram_size());
    println!("  Vocab:      {} commands", model.vocab_size());
    println!();

    // Check for HF_TOKEN
    if std::env::var("HF_TOKEN").is_err() {
        eprintln!("⚠️  HF_TOKEN environment variable not set.\n");
        eprintln!("To publish to Hugging Face Hub:");
        eprintln!("  1. Create a token at https://huggingface.co/settings/tokens");
        eprintln!("  2. Export it: export HF_TOKEN=hf_xxxxx");
        eprintln!();
        eprintln!("📁 Saving model card locally instead...");

        // Save locally
        let local_dir = path.parent().unwrap_or(std::path::Path::new("."));
        let readme_path = local_dir.join("README.md");
        let card_content = card.to_huggingface();

        if let Err(e) = std::fs::write(&readme_path, &card_content) {
            eprintln!("❌ Failed to write README.md: {e}");
            std::process::exit(1);
        }

        println!("✅ Model card saved to: {}", readme_path.display());
        println!();
        println!("💡 Upload manually with:");
        println!(
            "   huggingface-cli upload {repo_id} {} model.apr",
            path.display()
        );
        println!(
            "   huggingface-cli upload {repo_id} {} README.md",
            readme_path.display()
        );
        return;
    }

    // Prepare for HF Hub upload
    println!("🔑 Using HF_TOKEN for authentication");
    println!("  Create repo: {create}");
    println!("  Private:     {private}");
    println!("  Commit:      {commit_msg}");
    println!();

    // Note: Full HTTP upload requires additional implementation
    // For now, prepare files and show instructions
    let local_dir = path.parent().unwrap_or(std::path::Path::new("."));
    let readme_path = local_dir.join("README.md");
    let card_content = card.to_huggingface();

    if let Err(e) = std::fs::write(&readme_path, &card_content) {
        eprintln!("❌ Failed to write README.md: {e}");
        std::process::exit(1);
    }

    println!("✅ Model card generated: {}", readme_path.display());
    println!();
    println!("📋 Model Card Preview:");
    println!("───────────────────────────────────────────");
    for line in card_content.lines().take(20) {
        println!("  {line}");
    }
    println!("  ...");
    println!("───────────────────────────────────────────");
    println!();
    println!("🚀 Upload with huggingface-cli:");
    println!(
        "   huggingface-cli repo create {repo_id} --type model{}",
        if private { " --private" } else { "" }
    );
    println!(
        "   huggingface-cli upload {repo_id} {} model.apr --commit-message \"{commit_msg}\"",
        path.display()
    );
    println!(
        "   huggingface-cli upload {repo_id} {} README.md",
        readme_path.display()
    );
}

// =============================================================================
// Daemon/Stream Mode Commands (GH-95)
// =============================================================================

/// Stream mode: read prefixes from stdin, output suggestions to stdout
///
/// Model is loaded once and kept in memory for sub-millisecond latency.
/// Each line of input is treated as a prefix, with suggestions output immediately.
///
/// # Protocol
/// - Input: One prefix per line (UTF-8)
/// - Output: Suggestions in specified format, followed by empty line
/// - Special: Empty line or "QUIT" terminates
fn cmd_stream(model_path: &str, count: usize, format: &str, use_password: bool) {
    use std::io::{BufRead, Write};

    let path = expand_path(model_path);

    // Load model once
    let model = if use_password {
        let password =
            rpassword::prompt_password("🔐 Model password: ").unwrap_or_else(|_| String::new());
        match MarkovModel::load_encrypted(&path, &password) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("❌ Failed to load encrypted model: {e}");
                std::process::exit(1);
            }
        }
    } else {
        match load_model_graceful(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    };

    eprintln!(
        "🚀 Stream mode ready (model: {} commands)",
        model.total_commands()
    );
    eprintln!("   Enter prefixes, one per line. Empty line or 'QUIT' to exit.");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let prefix = match line {
            Ok(p) => p,
            Err(_) => break,
        };

        // Exit conditions
        if prefix.is_empty() || prefix.eq_ignore_ascii_case("QUIT") {
            break;
        }

        // Validate and sanitize prefix
        let prefix = match sanitize_prefix(&prefix) {
            Ok(p) => p,
            Err(_) => {
                writeln!(stdout).ok();
                continue;
            }
        };

        // Get suggestions
        let suggestions = model.suggest(&prefix, count);
        let filtered = filter_sensitive_suggestions(suggestions);

        // Output in requested format
        match format {
            "json" => {
                let json_suggestions: Vec<_> = filtered
                    .iter()
                    .map(|(s, score)| format!(r#"{{"suggestion":"{}","score":{:.4}}}"#, s, score))
                    .collect();
                writeln!(stdout, "[{}]", json_suggestions.join(",")).ok();
            }
            "tab" => {
                let tab_line: Vec<_> = filtered.iter().map(|(s, _)| s.as_str()).collect();
                writeln!(stdout, "{}", tab_line.join("\t")).ok();
            }
            _ => {
                // "lines" format (default)
                for (suggestion, _) in &filtered {
                    writeln!(stdout, "{suggestion}").ok();
                }
            }
        }

        // Empty line delimiter for batch processing
        writeln!(stdout).ok();
        stdout.flush().ok();
    }

    eprintln!("👋 Stream mode exiting");
}

/// Load a model for daemon/stream use, handling password and error exits.
#[cfg(unix)]
fn load_daemon_model(path: &std::path::Path, use_password: bool) -> MarkovModel {
    if use_password {
        let password =
            rpassword::prompt_password("🔐 Model password: ").unwrap_or_else(|_| String::new());
        match MarkovModel::load_encrypted(path, &password) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("❌ Failed to load encrypted model: {e}");
                std::process::exit(1);
            }
        }
    } else {
        match load_model_graceful(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }
}

/// Result of handling a single daemon request.
#[cfg(unix)]
enum DaemonAction {
    /// Continue accepting connections.
    Continue,
    /// Shut down the daemon.
    Shutdown,
}

/// Handle a single daemon request (special commands or suggestion generation).
#[cfg(unix)]
fn handle_daemon_request(
    prefix: &str,
    stream: &mut std::os::unix::net::UnixStream,
    model: &MarkovModel,
    count: usize,
    start_time: std::time::Instant,
    request_count: u64,
) -> DaemonAction {
    use std::io::Write;

    match prefix.to_uppercase().as_str() {
        "PING" => {
            writeln!(stream, "PONG").ok();
            writeln!(stream).ok();
            return DaemonAction::Continue;
        }
        "QUIT" | "SHUTDOWN" => {
            writeln!(stream, "OK").ok();
            eprintln!("👋 Daemon shutting down (received QUIT)");
            return DaemonAction::Shutdown;
        }
        "STATS" => {
            let uptime = start_time.elapsed().as_secs();
            writeln!(stream, "requests: {request_count}").ok();
            writeln!(stream, "uptime_secs: {uptime}").ok();
            writeln!(stream, "model_commands: {}", model.total_commands()).ok();
            writeln!(stream, "model_ngrams: {}", model.ngram_count()).ok();
            writeln!(stream).ok();
            return DaemonAction::Continue;
        }
        "" => {
            writeln!(stream).ok();
            return DaemonAction::Continue;
        }
        _ => {}
    }

    // Validate and get suggestions
    let suggestions = match sanitize_prefix(prefix) {
        Ok(p) => {
            let raw = model.suggest(&p, count);
            filter_sensitive_suggestions(raw)
        }
        Err(_) => vec![],
    };

    // Send suggestions
    for (suggestion, _) in &suggestions {
        writeln!(stream, "{suggestion}").ok();
    }
    writeln!(stream).ok(); // Empty line terminates response

    DaemonAction::Continue
}

/// Daemon mode: Unix socket server for sub-ms suggestions
///
/// Starts a server that listens on a Unix socket and responds to suggestion requests.
/// Model is loaded once at startup for maximum performance.
///
/// # Protocol (line-based)
/// - Client sends: prefix\n
/// - Server responds: suggestion1\nsuggestion2\n...\n\n (empty line terminates)
/// - Special commands: PING, QUIT, STATS
#[cfg(unix)]
fn cmd_daemon(
    model_path: &str,
    socket_path: &std::path::Path,
    count: usize,
    use_password: bool,
    foreground: bool,
) {
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixListener;

    let path = expand_path(model_path);

    // Remove stale socket if exists
    if socket_path.exists() {
        if let Err(e) = std::fs::remove_file(socket_path) {
            eprintln!("⚠️  Could not remove stale socket: {e}");
        }
    }

    // Load model
    let model = load_daemon_model(&path, use_password);

    // Bind socket
    let listener = match UnixListener::bind(socket_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("❌ Failed to bind socket '{}': {e}", socket_path.display());
            eprintln!("   Hint: Check permissions or use a different path");
            std::process::exit(1);
        }
    };

    if foreground {
        eprintln!("🚀 Daemon running in foreground");
    } else {
        println!("🚀 Daemon started");
    }
    println!("   Socket: {}", socket_path.display());
    println!("   Model:  {} commands", model.total_commands());
    println!("   PID:    {}", std::process::id());

    // Write PID file for daemon management
    let pid_path = socket_path.with_extension("pid");
    if let Err(e) = std::fs::write(&pid_path, std::process::id().to_string()) {
        eprintln!("⚠️  Could not write PID file: {e}");
    }

    let mut request_count = 0u64;
    let start_time = std::time::Instant::now();

    // Accept connections
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("⚠️  Connection error: {e}");
                continue;
            }
        };

        let mut reader = BufReader::new(stream.try_clone().expect("clone stream for reader"));

        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            continue;
        }

        let prefix = line.trim();
        request_count += 1;

        match handle_daemon_request(prefix, &mut stream, &model, count, start_time, request_count)
        {
            DaemonAction::Shutdown => break,
            DaemonAction::Continue => {}
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(socket_path);
    let _ = std::fs::remove_file(&pid_path);
}

#[cfg(not(unix))]
fn cmd_daemon(
    _model_path: &str,
    _socket_path: &std::path::Path,
    _count: usize,
    _use_password: bool,
    _foreground: bool,
) {
    eprintln!("❌ Daemon mode is only supported on Unix systems");
    eprintln!("   Use 'aprender-shell stream' for cross-platform streaming mode");
    std::process::exit(1);
}

/// Stop the running daemon
fn cmd_daemon_stop(socket_path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixStream;

        if !socket_path.exists() {
            eprintln!("❌ Daemon not running (socket not found)");
            std::process::exit(1);
        }

        let mut stream = match UnixStream::connect(socket_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("❌ Could not connect to daemon: {e}");
                std::process::exit(1);
            }
        };

        writeln!(stream, "QUIT").ok();
        stream.flush().ok();

        let mut reader = BufReader::new(&stream);
        let mut response = String::new();
        reader.read_line(&mut response).ok();

        if response.trim() == "OK" {
            println!("✅ Daemon stopped");
        } else {
            eprintln!("⚠️  Unexpected response: {response}");
        }
    }

    #[cfg(not(unix))]
    {
        let _ = socket_path;
        eprintln!("❌ Daemon mode is only supported on Unix systems");
        std::process::exit(1);
    }
}
