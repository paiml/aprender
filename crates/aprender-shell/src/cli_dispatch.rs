
fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            history,
            output,
            ngram,
            memory_limit,
            password,
        } => {
            cmd_train(history, &output, ngram, memory_limit, password);
        }
        Commands::Update {
            history,
            model,
            quiet,
            password,
        } => {
            cmd_update(history, &model, quiet, password);
        }
        Commands::Suggest {
            prefix,
            model,
            count,
            memory_limit,
            password,
        } => {
            cmd_suggest(&prefix, &model, count, memory_limit, password);
        }
        Commands::Stats {
            model,
            memory_limit,
            password,
        } => {
            cmd_stats(&model, memory_limit, password);
        }
        Commands::Export { model, output } => {
            cmd_export(&model, &output);
        }
        Commands::Import { input, output } => {
            cmd_import(&input, &output);
        }
        Commands::ZshWidget => {
            cmd_zsh_widget();
        }
        Commands::BashWidget => {
            cmd_bash_widget();
        }
        Commands::FishWidget => {
            cmd_fish_widget();
        }
        Commands::Uninstall {
            zsh,
            bash,
            fish,
            keep_model,
            dry_run,
        } => {
            cmd_uninstall(zsh, bash, fish, keep_model, dry_run);
        }
        Commands::Validate {
            history,
            ngram,
            ratio,
        } => {
            cmd_validate(history, ngram, ratio);
        }
        Commands::Augment {
            history,
            output,
            ngram,
            augmentation_ratio,
            quality_threshold,
            monitor_diversity,
            use_code_eda,
        } => {
            cmd_augment(
                history,
                &output,
                ngram,
                augmentation_ratio,
                quality_threshold,
                monitor_diversity,
                use_code_eda,
            );
        }
        Commands::Analyze { history, top } => {
            cmd_analyze(history, top);
        }
        Commands::Tune {
            history,
            trials,
            ratio,
        } => {
            cmd_tune(history, trials, ratio);
        }
        Commands::Inspect {
            model,
            format,
            password,
        } => {
            cmd_inspect(&model, &format, password);
        }
        Commands::Publish {
            model,
            repo,
            commit,
            create,
            private,
        } => {
            cmd_publish(&model, &repo, &commit, create, private);
        }
        Commands::Stream {
            model,
            count,
            format,
            password,
        } => {
            cmd_stream(&model, count, &format, password);
        }
        Commands::Daemon {
            model,
            socket,
            count,
            password,
            foreground,
        } => {
            cmd_daemon(&model, &socket, count, password, foreground);
        }
        Commands::DaemonStop { socket } => {
            cmd_daemon_stop(&socket);
        }
        Commands::DaemonStatus { socket } => {
            cmd_daemon_status(&socket);
        }
    }
}

/// Helper: Find and validate history file with graceful error handling (QA 2.4, 8.3)
fn find_history_file_graceful(history_path: Option<PathBuf>) -> PathBuf {
    match history_path {
        Some(path) => {
            if !path.exists() {
                eprintln!("❌ History file not found: {}", path.display());
                eprintln!("   Hint: Check the path or use -f to specify a different file");
                std::process::exit(1);
            }
            path
        }
        None => match HistoryParser::find_history_file() {
            Some(path) => path,
            None => {
                eprintln!("❌ Could not find shell history file");
                eprintln!("   Hint: Use -f to specify a history file manually");
                std::process::exit(1);
            }
        },
    }
}

/// Helper: Parse history file with graceful error handling (QA 8.3)
fn parse_history_graceful(history_file: &PathBuf) -> Vec<String> {
    let parser = HistoryParser::new();
    match parser.parse_file(history_file) {
        Ok(cmds) => cmds,
        Err(e) => {
            eprintln!("❌ Failed to read history file: {e}");
            if e.to_string().contains("ermission") {
                eprintln!(
                    "   Hint: Check file permissions with 'ls -la {}'",
                    history_file.display()
                );
            }
            std::process::exit(1);
        }
    }
}

fn validate_ngram(n: usize) {
    if !(2..=5).contains(&n) {
        eprintln!("❌ Error: N-gram size must be between 2 and 5 (got {})", n);
        std::process::exit(1);
    }
}

fn cmd_train(
    history_path: Option<PathBuf>,
    output: &str,
    ngram: usize,
    memory_limit: Option<usize>,
    use_password: bool,
) {
    validate_ngram(ngram);
    print_train_header(use_password, memory_limit.is_some());

    let commands = load_history_commands(history_path);
    let password = get_train_password(use_password);
    let output_path = expand_path(output);

    if let Some(mem_mb) = memory_limit {
        train_paged_model(&commands, &output_path, ngram, mem_mb, use_password);
    } else {
        train_standard_model(&commands, &output_path, ngram, password.as_deref());
    }
}

fn print_train_header(use_password: bool, paged: bool) {
    let encrypted_str = if use_password { " encrypted" } else { "" };
    let mode_str = if paged { "paged" } else { "standard" };
    println!("🚀 aprender-shell: Training{encrypted_str} {mode_str} model...\n");
}

fn load_history_commands(history_path: Option<PathBuf>) -> Vec<String> {
    let history_file = find_history_file_graceful(history_path);
    println!("📂 History file: {}", history_file.display());

    let commands = parse_history_graceful(&history_file);
    println!("📊 Commands loaded: {}", commands.len());

    if commands.is_empty() {
        eprintln!("❌ No commands found in history file");
        std::process::exit(1);
    }
    commands
}

fn get_train_password(use_password: bool) -> Option<String> {
    if !use_password {
        return None;
    }
    println!("🔐 Encrypting model with AES-256-GCM");
    let pwd = prompt_password_or_exit("   Enter password: ");
    let confirm = prompt_password_or_exit("   Confirm password: ");
    if pwd != confirm {
        eprintln!("❌ Passwords do not match");
        std::process::exit(1);
    }
    if pwd.len() < 8 {
        eprintln!("❌ Password must be at least 8 characters");
        std::process::exit(1);
    }
    Some(pwd)
}

fn prompt_password_or_exit(prompt: &str) -> String {
    rpassword::prompt_password(prompt).unwrap_or_else(|e| {
        eprintln!("❌ Failed to read password: {e}");
        std::process::exit(1);
    })
}

fn train_paged_model(
    commands: &[String],
    output_path: &Path,
    ngram: usize,
    mem_mb: usize,
    use_password: bool,
) {
    if use_password {
        eprintln!("⚠️  Encryption not yet supported for paged models. Creating unencrypted model.");
    }
    let output_path = output_path.with_extension("apbundle");
    print!(
        "🧠 Training {ngram}-gram paged model ({}MB limit)... ",
        mem_mb
    );
    let mut model = PagedMarkovModel::new(ngram, mem_mb);
    model.train(commands);
    println!("done!");

    save_model_or_exit(|| model.save(&output_path), &output_path, "paged model");

    let stats = model.stats();
    println!("\n✅ Paged model saved to: {}", output_path.display());
    println!("\n📈 Model Statistics:");
    println!("   Segments:        {}", stats.total_segments);
    println!("   Vocabulary size: {}", stats.vocab_size);
    println!("   Memory limit:    {} MB", mem_mb);
    println!("\n💡 Next steps:");
    println!("   1. Test: aprender-shell suggest \"git \" --memory-limit {mem_mb}");
    println!("   2. Stats: aprender-shell stats --memory-limit {mem_mb}");
}

fn train_standard_model(
    commands: &[String],
    output_path: &Path,
    ngram: usize,
    password: Option<&str>,
) {
    print!("🧠 Training {ngram}-gram model... ");
    let mut model = MarkovModel::new(ngram);
    model.train(commands);
    println!("done!");

    if let Some(pwd) = password {
        save_model_or_exit(
            || model.save_encrypted(output_path, pwd),
            output_path,
            "encrypted model",
        );
        println!("\n🔒 Encrypted model saved to: {}", output_path.display());
    } else {
        save_model_or_exit(|| model.save(output_path), output_path, "model");
        println!("\n✅ Model saved to: {}", output_path.display());
    }

    print_standard_model_stats(&model, password.is_some());
}

fn save_model_or_exit<F, E: std::fmt::Display>(save_fn: F, path: &Path, model_type: &str)
where
    F: FnOnce() -> Result<(), E>,
{
    if let Err(e) = save_fn() {
        eprintln!("❌ Failed to save {model_type}: {e}");
        if e.to_string().contains("ermission") {
            eprintln!("   Hint: Check write permissions for '{}'", path.display());
        }
        std::process::exit(1);
    }
}

fn print_standard_model_stats(model: &MarkovModel, encrypted: bool) {
    println!("\n📈 Model Statistics:");
    println!("   Unique n-grams: {}", model.ngram_count());
    println!("   Vocabulary size: {}", model.vocab_size());
    println!(
        "   Model size: {:.1} KB",
        model.size_bytes() as f64 / 1024.0
    );
    if encrypted {
        println!("   Encryption: AES-256-GCM (Argon2id KDF)");
    }

    println!("\n💡 Next steps:");
    if encrypted {
        println!("   1. Test: aprender-shell suggest \"git \" --password");
        println!("   2. Stats: aprender-shell stats --password");
    } else {
        println!("   1. Test: aprender-shell suggest \"git \"");
        println!("   2. Install: aprender-shell zsh-widget >> ~/.zshrc");
    }
}

/// Load an existing model from disk, handling encryption and error diagnostics.
fn load_existing_model(path: &std::path::Path, password: Option<&str>) -> MarkovModel {
    if let Some(pwd) = password {
        return MarkovModel::load_encrypted(path, pwd).unwrap_or_else(|e| {
            eprintln!("❌ Failed to load encrypted model: {e}");
            eprintln!("   Hint: Check password or try without --password flag");
            std::process::exit(1);
        });
    }
    match MarkovModel::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Failed to load model '{}': {e}", path.display());
            print_model_load_hint(&e);
            std::process::exit(1);
        }
    }
}

/// Print a diagnostic hint based on the model load error.
fn print_model_load_hint(e: &std::io::Error) {
    let msg = e.to_string();
    if msg.contains("Checksum mismatch") {
        eprintln!("   Hint: The model file may be corrupted. Run 'aprender-shell train' to rebuild.");
    } else if msg.contains("magic") || msg.contains("invalid") {
        eprintln!("   Hint: The file may not be a valid aprender model.");
    }
}

/// Save a model to disk, preserving encryption status.
fn save_model_to_disk(model: &MarkovModel, path: &std::path::Path, password: Option<&str>) {
    if let Some(pwd) = password {
        if let Err(e) = model.save_encrypted(path, pwd) {
            eprintln!("❌ Failed to save encrypted model: {e}");
            std::process::exit(1);
        }
    } else if let Err(e) = model.save(path) {
        eprintln!("❌ Failed to save model: {e}");
        std::process::exit(1);
    }
}

fn cmd_update(history_path: Option<PathBuf>, model_path: &str, quiet: bool, use_password: bool) {
    let path = expand_path(model_path);

    let password = if use_password {
        Some(
            rpassword::prompt_password("Enter password: ").unwrap_or_else(|e| {
                eprintln!("❌ Failed to read password: {e}");
                std::process::exit(1);
            }),
        )
    } else {
        None
    };

    let mut model = if path.exists() {
        load_existing_model(&path, password.as_deref())
    } else {
        if !quiet {
            println!("📝 No existing model, creating new one...");
        }
        MarkovModel::new(3)
    };

    let history_file = find_history_file_graceful(history_path);
    let all_commands = parse_history_graceful(&history_file);

    let last_pos = model.last_trained_position();
    let new_commands: Vec<String> = all_commands.into_iter().skip(last_pos).collect();

    if new_commands.is_empty() {
        if !quiet {
            println!("✓ Model is up to date (no new commands)");
        }
        return;
    }

    if !quiet {
        println!("📊 Found {} new commands", new_commands.len());
    }

    model.train_incremental(&new_commands);
    save_model_to_disk(&model, &path, password.as_deref());

    if !quiet {
        println!(
            "✅ Model updated ({} total commands)",
            model.total_commands()
        );
    }
}

/// Get password from environment or prompt.
fn get_password_or_prompt(use_password: bool, error_prefix: &str) -> Option<String> {
    if !use_password {
        return None;
    }
    std::env::var("APRENDER_PASSWORD").ok().or_else(|| {
        Some(
            rpassword::prompt_password("Enter password: ").unwrap_or_else(|e| {
                eprintln!("{error_prefix}Failed to read password: {e}");
                std::process::exit(1);
            }),
        )
    })
}
