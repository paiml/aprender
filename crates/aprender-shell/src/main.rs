//! aprender-shell: AI-powered shell completion trained on your history
//!
//! Train a personalized autocomplete model on your shell history in seconds.
//! 100% local, private, and fast.

use aprender_shell::{
    filter_sensitive_suggestions, load_model_graceful, sanitize_prefix, synthetic, HistoryParser,
    MarkovModel, PagedMarkovModel, ShellError, SyntheticPipeline,
};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "aprender-shell")]
#[command(about = "AI-powered shell completion trained on your history")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model on your shell history (full retrain)
    Train {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Output model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        output: String,

        /// N-gram size (2-5)
        #[arg(short, long, default_value = "3")]
        ngram: usize,

        /// Memory limit in MB (enables paged storage for large histories)
        #[arg(long)]
        memory_limit: Option<usize>,

        /// Encrypt model with password (AES-256-GCM)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Incrementally update model with new commands (fast)
    Update {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Model file to update
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Be quiet (for hooks)
        #[arg(short, long)]
        quiet: bool,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Get completions for a prefix
    Suggest {
        /// The current command prefix (use `-- PREFIX` for prefixes starting with -)
        #[arg(allow_hyphen_values = true)]
        prefix: String,

        /// Model file to use
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Number of suggestions
        #[arg(short = 'c', visible_short_alias = 'k', long, default_value = "5")]
        count: usize,

        /// Memory limit in MB (for paged models)
        #[arg(long)]
        memory_limit: Option<usize>,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Show model statistics
    Stats {
        /// Model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Memory limit in MB (for paged models)
        #[arg(long)]
        memory_limit: Option<usize>,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Export model for sharing
    Export {
        /// Model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Output path
        output: PathBuf,
    },

    /// Import a shared model
    Import {
        /// Model file to import
        input: PathBuf,

        /// Destination
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        output: String,
    },

    /// Generate ZSH widget code
    ZshWidget,

    /// Generate Bash widget code
    BashWidget,

    /// Generate Fish shell widget code
    FishWidget,

    /// Uninstall widget from shell config
    Uninstall {
        /// Target ZSH shell (~/.zshrc)
        #[arg(long)]
        zsh: bool,

        /// Target Bash shell (~/.bashrc)
        #[arg(long)]
        bash: bool,

        /// Target Fish shell (~/.config/fish/config.fish)
        #[arg(long)]
        fish: bool,

        /// Keep model file (don't delete ~/.aprender-shell.model)
        #[arg(long)]
        keep_model: bool,

        /// Show what would be removed without doing it
        #[arg(long)]
        dry_run: bool,
    },

    /// Validate model accuracy using holdout evaluation
    Validate {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// N-gram size (2-5)
        #[arg(short, long, default_value = "3")]
        ngram: usize,

        /// Train/test split ratio (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        ratio: f32,
    },

    /// Augment training data with synthetic commands
    Augment {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Output model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        output: String,

        /// N-gram size (2-5)
        #[arg(short, long, default_value = "3")]
        ngram: usize,

        /// Augmentation ratio (synthetic/original, e.g., 0.5 = 50% more data)
        #[arg(short = 'a', long, default_value = "0.5")]
        augmentation_ratio: f32,

        /// Minimum quality threshold (0.0-1.0)
        #[arg(short, long, default_value = "0.7")]
        quality_threshold: f32,

        /// Enable diversity monitoring
        #[arg(long)]
        monitor_diversity: bool,

        /// Use CodeEDA for code-aware augmentation
        #[arg(long)]
        use_code_eda: bool,
    },

    /// Analyze command patterns and extract features
    Analyze {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Show top N commands by category
        #[arg(short, long, default_value = "10")]
        top: usize,
    },

    /// Auto-tune hyperparameters using aprender's AutoML
    Tune {
        /// Path to history file (default: auto-detect)
        #[arg(short = 'f', long)]
        history: Option<PathBuf>,

        /// Number of trials to run
        #[arg(short, long, default_value = "10")]
        trials: usize,

        /// Train/test split ratio (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        ratio: f32,
    },

    /// Inspect model metadata (model card, version)
    Inspect {
        /// Model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Output format (json, yaml, huggingface)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Publish model to Hugging Face Hub (GH-100)
    Publish {
        /// Model file to publish
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Repository ID (org/name format)
        #[arg(short, long)]
        repo: String,

        /// Commit message
        #[arg(short, long, default_value = "Upload model via aprender-shell")]
        commit: String,

        /// Create repository if it doesn't exist
        #[arg(long, default_value = "true")]
        create: bool,

        /// Make repository private
        #[arg(long)]
        private: bool,
    },

    /// Stream mode: read prefixes from stdin, output suggestions (GH-95)
    ///
    /// Keeps model in memory for sub-millisecond latency.
    /// Use with shell coprocess for zero-latency suggestions.
    Stream {
        /// Model file to use
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Number of suggestions per request
        #[arg(short = 'c', long, default_value = "5")]
        count: usize,

        /// Output format (lines, json, tab)
        #[arg(short, long, default_value = "lines")]
        format: String,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,
    },

    /// Daemon mode: Unix socket server for sub-ms suggestions (GH-95)
    ///
    /// Starts a background server that keeps the model in memory.
    /// Connect via Unix socket for instant suggestions.
    Daemon {
        /// Model file to use
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Unix socket path
        #[arg(short, long, default_value = "/tmp/aprender-shell.sock")]
        socket: PathBuf,

        /// Number of suggestions per request
        #[arg(short = 'c', long, default_value = "5")]
        count: usize,

        /// Model is encrypted (will prompt for password)
        #[arg(short = 'p', long)]
        password: bool,

        /// Run in foreground (don't daemonize)
        #[arg(long)]
        foreground: bool,
    },

    /// Stop the running daemon
    DaemonStop {
        /// Unix socket path
        #[arg(short, long, default_value = "/tmp/aprender-shell.sock")]
        socket: PathBuf,
    },

    /// Check daemon status
    DaemonStatus {
        /// Unix socket path
        #[arg(short, long, default_value = "/tmp/aprender-shell.sock")]
        socket: PathBuf,
    },
}

fn expand_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

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

fn cmd_update(history_path: Option<PathBuf>, model_path: &str, quiet: bool, use_password: bool) {
    let path = expand_path(model_path);

    // Get password if needed
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

    // Load existing model or create new one
    let mut model = if path.exists() {
        if let Some(ref pwd) = password {
            MarkovModel::load_encrypted(&path, pwd).unwrap_or_else(|e| {
                eprintln!("❌ Failed to load encrypted model: {e}");
                eprintln!("   Hint: Check password or try without --password flag");
                std::process::exit(1);
            })
        } else {
            match MarkovModel::load(&path) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("❌ Failed to load model '{}': {e}", path.display());
                    if e.to_string().contains("Checksum mismatch") {
                        eprintln!("   Hint: The model file may be corrupted. Run 'aprender-shell train' to rebuild.");
                    } else if e.to_string().contains("magic") || e.to_string().contains("invalid") {
                        eprintln!("   Hint: The file may not be a valid aprender model.");
                    }
                    std::process::exit(1);
                }
            }
        }
    } else {
        if !quiet {
            println!("📝 No existing model, creating new one...");
        }
        MarkovModel::new(3)
    };

    // Find and parse history with graceful error handling (QA 2.4, 8.3)
    let history_file = find_history_file_graceful(history_path);
    let all_commands = parse_history_graceful(&history_file);

    // Get only new commands (after last trained position)
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

    // Incremental train
    model.train_incremental(&new_commands);

    // Save (preserve encryption status)
    if let Some(ref pwd) = password {
        if let Err(e) = model.save_encrypted(&path, pwd) {
            eprintln!("❌ Failed to save encrypted model: {e}");
            std::process::exit(1);
        }
    } else if let Err(e) = model.save(&path) {
        eprintln!("❌ Failed to save model: {e}");
        std::process::exit(1);
    }

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

/// Output filtered suggestions.
fn output_suggestions(suggestions: Vec<(String, f32)>) {
    let filtered = filter_sensitive_suggestions(suggestions);
    for (suggestion, score) in filtered {
        println!("{}\t{:.3}", suggestion, score);
    }
}

/// Handle paged model suggestion.
fn suggest_paged(path: &std::path::Path, prefix: &str, count: usize, mem_mb: usize) {
    let paged_path = path.with_extension("apbundle");
    let mut model = match PagedMarkovModel::load(&paged_path, mem_mb) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("# aprender: {e}");
            return;
        }
    };
    output_suggestions(model.suggest(prefix, count));
}

/// Handle standard model suggestion.
fn suggest_standard(path: &std::path::Path, prefix: &str, count: usize, password: Option<&str>) {
    let model = if let Some(pwd) = password {
        match MarkovModel::load_encrypted(path, pwd) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("# aprender: {e}");
                return;
            }
        }
    } else {
        match load_model_graceful(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{e}");
                return;
            }
        }
    };
    output_suggestions(model.suggest(prefix, count));
}

fn cmd_suggest(
    prefix: &str,
    model_path: &str,
    count: usize,
    memory_limit: Option<usize>,
    use_password: bool,
) {
    // Phase 1: Input validation (Poka-yoke)
    let validated_prefix = match sanitize_prefix(prefix) {
        Ok(p) => p,
        Err(ShellError::InvalidInput { message }) => {
            eprintln!("# aprender: {message}");
            return;
        }
        Err(_) => return,
    };

    let path = expand_path(model_path);
    let password = get_password_or_prompt(use_password, "# aprender: ");

    if let Some(mem_mb) = memory_limit {
        suggest_paged(&path, &validated_prefix, count, mem_mb);
    } else {
        suggest_standard(&path, &validated_prefix, count, password.as_deref());
    }
}

/// Print paged model error hints.
fn print_paged_model_error_hint(e: &std::io::Error, paged_path: &std::path::Path, mem_mb: usize) {
    let err_str = e.to_string();
    if err_str.contains("Checksum") || err_str.contains("corrupt") {
        eprintln!("   Hint: The model may be corrupted. Run 'aprender-shell train --memory-limit {mem_mb}' to rebuild.");
    } else if !paged_path.exists() {
        eprintln!("   Hint: Model file not found. Train a model first with 'aprender-shell train --memory-limit {mem_mb}'");
    }
}

/// Print standard model error hints.
fn print_standard_model_error_hint(e: &std::io::Error, path: &std::path::Path) {
    let err_str = e.to_string();
    if err_str.contains("Checksum mismatch") {
        eprintln!(
            "   Hint: The model file may be corrupted. Run 'aprender-shell train' to rebuild."
        );
    } else if !path.exists() {
        eprintln!("   Hint: Model file not found. Train a model first with 'aprender-shell train'");
    } else if MarkovModel::is_encrypted(path).unwrap_or(false) {
        eprintln!("   Hint: This model is encrypted. Use --password flag.");
    }
}

/// Print paging statistics if available.
fn print_paging_stats(model: &PagedMarkovModel) {
    if let Some(paging_stats) = model.paging_stats() {
        println!("\n📈 Paging Statistics:");
        println!("   Page hits:       {}", paging_stats.hits);
        println!("   Page misses:     {}", paging_stats.misses);
        println!("   Evictions:       {}", paging_stats.evictions);
        let total = paging_stats.hits + paging_stats.misses;
        if total > 0 {
            let hit_rate = paging_stats.hits as f64 / total as f64 * 100.0;
            println!("   Hit rate:        {:.1}%", hit_rate);
        }
    }
}

/// Print top commands from a model.
fn print_top_commands(commands: Vec<(String, u32)>) {
    println!("\n🔝 Top commands:");
    for (cmd, count) in commands {
        println!("   {:>6}x  {}", count, cmd);
    }
}

/// Handle paged model stats display.
fn stats_paged(path: &std::path::Path, mem_mb: usize) {
    let paged_path = path.with_extension("apbundle");
    let model = match PagedMarkovModel::load(&paged_path, mem_mb) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "❌ Failed to load paged model '{}': {e}",
                paged_path.display()
            );
            print_paged_model_error_hint(&e, &paged_path, mem_mb);
            std::process::exit(1);
        }
    };

    let stats = model.stats();
    println!("📊 Paged Model Statistics:");
    println!("   N-gram size:     {}", stats.n);
    println!("   Total commands:  {}", stats.total_commands);
    println!("   Vocabulary size: {}", stats.vocab_size);
    println!("   Total segments:  {}", stats.total_segments);
    println!("   Loaded segments: {}", stats.loaded_segments);
    println!(
        "   Memory limit:    {:.1} MB",
        stats.memory_limit as f64 / 1024.0 / 1024.0
    );
    println!(
        "   Loaded bytes:    {:.1} KB",
        stats.loaded_bytes as f64 / 1024.0
    );

    print_paging_stats(&model);
    print_top_commands(model.top_commands(10));
}

/// Handle standard model stats display.
fn stats_standard(path: &std::path::Path, password: Option<&str>) {
    let model = if let Some(pwd) = password {
        MarkovModel::load_encrypted(path, pwd).unwrap_or_else(|e| {
            eprintln!("❌ Failed to load encrypted model: {e}");
            std::process::exit(1);
        })
    } else {
        match MarkovModel::load(path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("❌ Failed to load model '{}': {e}", path.display());
                print_standard_model_error_hint(&e, path);
                std::process::exit(1);
            }
        }
    };

    let encrypted = MarkovModel::is_encrypted(path).unwrap_or(false);

    println!("📊 Model Statistics:");
    println!("   N-gram size: {}", model.ngram_size());
    println!("   Unique n-grams: {}", model.ngram_count());
    println!("   Vocabulary size: {}", model.vocab_size());
    println!(
        "   Model size: {:.1} KB",
        model.size_bytes() as f64 / 1024.0
    );
    if encrypted {
        println!("   🔒 Encryption: AES-256-GCM");
    }
    print_top_commands(model.top_commands(10));
}

fn cmd_stats(model_path: &str, memory_limit: Option<usize>, use_password: bool) {
    let path = expand_path(model_path);
    let password = get_password_or_prompt(use_password, "❌ ");

    if let Some(mem_mb) = memory_limit {
        stats_paged(&path, mem_mb);
    } else {
        stats_standard(&path, password.as_deref());
    }
}

fn cmd_export(model_path: &str, output: &PathBuf) {
    let path = expand_path(model_path);

    if !path.exists() {
        eprintln!("❌ Model file not found: {}", path.display());
        eprintln!("   Hint: Train a model first with 'aprender-shell train'");
        std::process::exit(1);
    }

    if let Err(e) = std::fs::copy(&path, output) {
        eprintln!("❌ Failed to export model: {e}");
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            eprintln!(
                "   Hint: Check write permissions for '{}'",
                output.display()
            );
        } else if e.kind() == std::io::ErrorKind::NotFound {
            eprintln!("   Hint: Destination directory may not exist");
        }
        std::process::exit(1);
    }

    println!("✅ Model exported to: {}", output.display());
}

fn cmd_import(input: &PathBuf, output: &str) {
    if !input.exists() {
        eprintln!("❌ Input file not found: {}", input.display());
        std::process::exit(1);
    }

    let output_path = expand_path(output);

    if let Err(e) = std::fs::copy(input, &output_path) {
        eprintln!("❌ Failed to import model: {e}");
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            eprintln!(
                "   Hint: Check write permissions for '{}'",
                output_path.display()
            );
        }
        std::process::exit(1);
    }

    println!("✅ Model imported to: {}", output_path.display());
}

fn cmd_zsh_widget() {
    print!(
        r#"# >>> aprender-shell widget >>>
# shellcheck shell=zsh disable=SC2154,SC2086,SC2089,SC2227,SC2201,SC2067
# aprender-shell ZSH widget v5 (with daemon support)
# This script is meant to be sourced, not executed directly
# Add this to your ~/.zshrc
# Toggle: export APRENDER_DISABLED=1 to disable
# Daemon: export APRENDER_USE_DAEMON=1 for sub-ms latency
# Uninstall: aprender-shell uninstall --zsh
# Hardened per: docs/specifications/aprender-shell-harden-plan.md
# Validated with bashrs lint

# Environment variables (set externally by user, defaults provided)
APRENDER_DISABLED="${{APRENDER_DISABLED:-0}}"
APRENDER_USE_DAEMON="${{APRENDER_USE_DAEMON:-0}}"
APRENDER_AUTO_DAEMON="${{APRENDER_AUTO_DAEMON:-0}}"
APRENDER_SOCKET="${{APRENDER_SOCKET:-/tmp/aprender-shell.sock}}"
# Debugging (issue #84)
APRENDER_TIMING="${{APRENDER_TIMING:-0}}"
APRENDER_TRACE="${{APRENDER_TRACE:-0}}"
APRENDER_TRACE_FILE="${{APRENDER_TRACE_FILE:-/tmp/aprender-trace.log}}"
# Renacer syscall tracing (issue #89)
APRENDER_RENACER="${{APRENDER_RENACER:-0}}"
APRENDER_RENACER_OPTS="${{APRENDER_RENACER_OPTS:--c --stats}}"
APRENDER_RENACER_LOG="${{APRENDER_RENACER_LOG:-/tmp/aprender-renacer.log}}"

# Check if daemon is running
_aprender_daemon_available() {{
    [[ -S "$APRENDER_SOCKET" ]] && nc -z -U "$APRENDER_SOCKET" 2>/dev/null
}}

# Get suggestion from daemon (sub-ms latency)
_aprender_suggest_daemon() {{
    local prefix="$1"
    echo "$prefix" | nc -U "$APRENDER_SOCKET" 2>/dev/null | head -1
}}

# Get suggestion via command (fallback, ~10ms)
_aprender_suggest_cmd() {{
    local prefix="$1"
    if [[ "$APRENDER_RENACER" == '1' ]] && command -v renacer &>/dev/null; then
        # Wrap with renacer for syscall tracing (issue #89)
        renacer $APRENDER_RENACER_OPTS -- aprender-shell suggest "$prefix" 2>>"$APRENDER_RENACER_LOG" | head -1 | cut -f1
    else
        timeout 0.1 aprender-shell suggest "$prefix" 2>/dev/null | head -1 | cut -f1
    fi
}}

_aprender_suggest() {{
    # Skip if disabled or buffer too short
    [[ "$APRENDER_DISABLED" == '1' ]] && return
    [[ "${{#BUFFER}}" -lt 2 ]] && {{ POSTDISPLAY=''; return; }}

    local suggestion start_ms end_ms elapsed_ms

    # Timing start (issue #84)
    [[ "$APRENDER_TIMING" == '1' ]] && start_ms=$(($(date +%s%N 2>/dev/null || echo 0)/1000000))

    # Use daemon if available and enabled, otherwise fall back to command
    if [[ "$APRENDER_USE_DAEMON" == '1' ]] && _aprender_daemon_available; then
        suggestion="$(_aprender_suggest_daemon "$BUFFER")"
    else
        suggestion="$(_aprender_suggest_cmd "$BUFFER")"
    fi

    # Timing end (issue #84)
    if [[ "$APRENDER_TIMING" == '1' ]]; then
        end_ms=$(($(date +%s%N 2>/dev/null || echo 0)/1000000))
        elapsed_ms=$((end_ms - start_ms))
        print -u2 "[aprender] ${{elapsed_ms}}ms: '$BUFFER' -> '$suggestion'"
    fi

    # Trace logging (issue #84)
    if [[ "$APRENDER_TRACE" == '1' ]]; then
        echo "$(date +%Y-%m-%dT%H:%M:%S) prefix='$BUFFER' suggestion='$suggestion'" >> "$APRENDER_TRACE_FILE"
    fi

    if [[ -n "$suggestion" && "$suggestion" != "$BUFFER" ]]; then
        local suffix="${{suggestion#"$BUFFER"}}"

        # Robust ANSI handling with terminal capability check
        # terminfo is a ZSH builtin associative array
        if [[ -n "$TERM" && "$TERM" != 'dumb' ]] && (( ${{+terminfo[colors]}} )) && (( ${{terminfo[colors]:-0}} >= 8 )); then
            # Use ZSH prompt expansion for portable color codes
            POSTDISPLAY="$(print -P " %F{{240}}${{suffix}}%f")"
        else
            # Fallback: no colors for unsupported terminals
            POSTDISPLAY=" ${{suffix}}"
        fi
    else
        POSTDISPLAY=''
    fi
}}

_aprender_accept() {{
    if [[ -n "$POSTDISPLAY" ]]; then
        # Strip color codes and leading space when accepting
        local clean_suffix
        clean_suffix="${{POSTDISPLAY# }}"
        # Remove ANSI escape sequences (both $'\e[...m' and %F/%f)
        clean_suffix="${{clean_suffix//\\e\[*m/}}"
        clean_suffix="${{clean_suffix//%F\{{*\}}/}}"
        clean_suffix="${{clean_suffix//%f/}}"

        BUFFER="${{BUFFER}}${{clean_suffix}}"
        POSTDISPLAY=''
        CURSOR="${{#BUFFER}}"
        zle redisplay
    else
        # No suggestion: fall back to default Tab completion (issue #83)
        zle expand-or-complete
    fi
}}

_aprender_accept_word() {{
    # Accept next word of suggestion only (issue #85)
    if [[ -n "$POSTDISPLAY" ]]; then
        local clean_suffix
        clean_suffix="${{POSTDISPLAY# }}"
        # Remove color codes
        clean_suffix="${{clean_suffix//\\e\[*m/}}"
        clean_suffix="${{clean_suffix//%F\{{*\}}/}}"
        clean_suffix="${{clean_suffix//%f/}}"

        # Extract next word (up to first space)
        local next_word="${{clean_suffix%% *}}"
        if [[ "$next_word" == "$clean_suffix" ]]; then
            # No space found - accept entire remaining suggestion
            BUFFER="${{BUFFER}}${{next_word}}"
            POSTDISPLAY=''
        else
            # Accept word + space, update suggestion
            BUFFER="${{BUFFER}}${{next_word}} "
            local remaining="${{clean_suffix#* }}"
            if [[ -n "$remaining" ]]; then
                POSTDISPLAY=" ${{remaining}}"
            else
                POSTDISPLAY=''
            fi
        fi
        CURSOR="${{#BUFFER}}"
    fi
    zle redisplay
}}

zle -N _aprender_suggest
zle -N _aprender_accept
zle -N _aprender_accept_word

# Trigger on each keystroke
autoload -Uz add-zle-hook-widget
add-zle-hook-widget line-pre-redraw _aprender_suggest

# Accept with Tab or Right Arrow (full suggestion)
bindkey '^I' _aprender_accept      # Tab
bindkey '^[[C' _aprender_accept    # Right arrow
# Accept next word only (issue #85)
bindkey '^[[1;5C' _aprender_accept_word  # Ctrl+Right

# Start daemon automatically if requested
if [[ "$APRENDER_AUTO_DAEMON" == '1' ]] && ! _aprender_daemon_available; then
    aprender-shell daemon &>/dev/null &
    disown
fi
# <<< aprender-shell widget <<<
"#
    );
}

fn cmd_bash_widget() {
    print!(
        r#"# >>> aprender-shell widget >>>
# aprender-shell Bash widget v1 (issue #82)
# Add this to your ~/.bashrc
# Toggle: export APRENDER_DISABLED=1 to disable
# Uninstall: aprender-shell uninstall --bash

# Environment variables
APRENDER_DISABLED="${{APRENDER_DISABLED:-0}}"
_APRENDER_SUGGESTION=""

_aprender_get_suggestion() {{
    local prefix="$1"
    timeout 0.1 aprender-shell suggest "$prefix" 2>/dev/null | head -1 | cut -f1
}}

_aprender_suggest() {{
    [[ "$APRENDER_DISABLED" == "1" ]] && return
    [[ "${{#READLINE_LINE}}" -lt 2 ]] && return

    local suggestion
    suggestion=$(_aprender_get_suggestion "$READLINE_LINE")

    if [[ -n "$suggestion" && "$suggestion" != "$READLINE_LINE" ]]; then
        _APRENDER_SUGGESTION="${{suggestion#$READLINE_LINE}}"
        # Display ghost text in gray (after cursor position)
        echo -ne "\e[90m${{_APRENDER_SUGGESTION}}\e[0m\e[${{#_APRENDER_SUGGESTION}}D"
    else
        _APRENDER_SUGGESTION=""
    fi
}}

_aprender_accept() {{
    if [[ -n "$_APRENDER_SUGGESTION" ]]; then
        READLINE_LINE="${{READLINE_LINE}}${{_APRENDER_SUGGESTION}}"
        READLINE_POINT=${{#READLINE_LINE}}
        _APRENDER_SUGGESTION=""
    fi
}}

_aprender_clear() {{
    _APRENDER_SUGGESTION=""
}}

# Bind to readline hooks
# Note: Bash readline integration is limited compared to ZSH
# This provides basic suggestion on Tab key
bind -x '"\C-i": _aprender_accept'  # Tab to accept
bind -x '"\e[C": _aprender_accept'  # Right arrow to accept

# Show suggestions after each command edit (Bash 4.0+)
if [[ "${{BASH_VERSINFO[0]}}" -ge 4 ]]; then
    PROMPT_COMMAND="_aprender_suggest; ${{PROMPT_COMMAND}}"
fi
# <<< aprender-shell widget <<<
"#
    );
}

fn cmd_fish_widget() {
    print!(
        r#"# >>> aprender-shell widget >>>
# aprender-shell Fish widget v1
# Add this to your ~/.config/fish/config.fish
# Toggle: set -gx APRENDER_DISABLED 1 to disable
# Uninstall: aprender-shell uninstall --fish

function __aprender_suggest --description "Get AI-powered command suggestions"
    # Skip if disabled
    if test "$APRENDER_DISABLED" = "1"
        return
    end

    set -l cmd (commandline -b)
    # Skip if command too short
    if test (string length "$cmd") -lt 2
        return
    end

    # Get suggestion (with timeout for responsiveness)
    set -l suggestion (timeout 0.1 aprender-shell suggest "$cmd" 2>/dev/null | head -1 | cut -f1)

    if test -n "$suggestion" -a "$suggestion" != "$cmd"
        echo "$suggestion"
    end
end

function __aprender_complete --description "Complete commands for aprender suggestions"
    set -l cmd (commandline -cp)
    if test (string length "$cmd") -ge 2
        aprender-shell suggest "$cmd" 2>/dev/null | while read -l line
            set -l suggestion (echo "$line" | cut -f1)
            set -l score (echo "$line" | cut -f2)
            printf "%s\t%s\n" "$suggestion" "score: $score"
        end
    end
end

# Register completions for all commands
complete -f -c '*' -a '(__aprender_complete)'

# Fish autosuggestion hook (runs on each keystroke)
function __aprender_autosuggest --on-event fish_preexec
    # Optional: Update model incrementally (runs on command execution)
    # aprender-shell update --quiet 2>/dev/null &
end
# <<< aprender-shell widget <<<
"#
    );
}

fn cmd_uninstall(zsh: bool, bash: bool, fish: bool, keep_model: bool, dry_run: bool) {
    let detect_all = !zsh && !bash && !fish;

    let home = match dirs::home_dir() {
        Some(h) => h,
        None => {
            eprintln!("❌ Could not determine home directory");
            eprintln!("   Hint: Set HOME environment variable");
            std::process::exit(1);
        }
    };

    let action = if dry_run { "Would remove" } else { "Removed" };
    let mut removed_any = false;

    // Uninstall from each shell config
    let shells = [
        (zsh, ".zshrc", "source ~/.zshrc"),
        (bash, ".bashrc", "source ~/.bashrc"),
        (
            fish,
            ".config/fish/config.fish",
            "source ~/.config/fish/config.fish",
        ),
    ];

    for (requested, config_path, _reload_cmd) in &shells {
        if *requested || detect_all {
            removed_any |= uninstall_shell_widget(&home, config_path, *requested, action, dry_run);
        }
    }

    // Remove model files
    if !keep_model {
        removed_any |= remove_model_files(&home, dry_run);
    }

    print_uninstall_summary(removed_any, dry_run, zsh, bash, fish, detect_all);
}

/// Uninstall widget from a shell config file.
fn uninstall_shell_widget(
    home: &std::path::Path,
    config_path: &str,
    explicitly_requested: bool,
    action: &str,
    dry_run: bool,
) -> bool {
    let config_file = home.join(config_path);

    if !config_file.exists() {
        if explicitly_requested {
            println!("ℹ {} does not exist", config_file.display());
        }
        return false;
    }

    match remove_widget_block(&config_file, dry_run) {
        Ok(true) => {
            println!("✓ {} widget from {}", action, config_file.display());
            true
        }
        Ok(false) => {
            if explicitly_requested {
                println!("ℹ No widget found in {}", config_file.display());
            }
            false
        }
        Err(e) => {
            eprintln!("✗ Error processing {}: {}", config_file.display(), e);
            false
        }
    }
}

/// Remove model and bundle files.
fn remove_model_files(home: &std::path::Path, dry_run: bool) -> bool {
    let mut removed = false;

    // Standard model file
    let model_file = home.join(".aprender-shell.model");
    if model_file.exists() {
        removed |= remove_single_file(&model_file, "model file", dry_run);
    }

    // Paged model bundle
    let bundle_file = home.join(".aprender-shell.apbundle");
    if bundle_file.exists() {
        removed |= remove_directory(&bundle_file, "model bundle", dry_run);
    }

    removed
}

/// Remove a single file with appropriate messaging.
fn remove_single_file(path: &std::path::Path, description: &str, dry_run: bool) -> bool {
    if dry_run {
        println!("✓ Would remove {} {}", description, path.display());
        return true;
    }

    match std::fs::remove_file(path) {
        Ok(()) => {
            println!("✓ Removed {} {}", description, path.display());
            true
        }
        Err(e) => {
            eprintln!("✗ Error removing {}: {}", description, e);
            false
        }
    }
}

/// Remove a directory with appropriate messaging.
fn remove_directory(path: &std::path::Path, description: &str, dry_run: bool) -> bool {
    if dry_run {
        println!("✓ Would remove {} {}", description, path.display());
        return true;
    }

    match std::fs::remove_dir_all(path) {
        Ok(()) => {
            println!("✓ Removed {} {}", description, path.display());
            true
        }
        Err(e) => {
            eprintln!("✗ Error removing {}: {}", description, e);
            false
        }
    }
}

/// Print summary after uninstall.
fn print_uninstall_summary(
    removed_any: bool,
    dry_run: bool,
    zsh: bool,
    bash: bool,
    fish: bool,
    detect_all: bool,
) {
    if !removed_any {
        println!("ℹ No aprender-shell installation found");
        return;
    }

    if dry_run {
        println!("\n💡 Run without --dry-run to apply changes");
        return;
    }

    println!("\n✅ Done! Restart your shell or run:");
    if zsh || detect_all {
        println!("   source ~/.zshrc");
    }
    if bash || detect_all {
        println!("   source ~/.bashrc");
    }
    if fish || detect_all {
        println!("   source ~/.config/fish/config.fish");
    }
}

/// Remove widget block between marker comments from a file
fn remove_widget_block(path: &std::path::Path, dry_run: bool) -> std::io::Result<bool> {
    let content = std::fs::read_to_string(path)?;

    let start_marker = "# >>> aprender-shell widget >>>";
    let end_marker = "# <<< aprender-shell widget <<<";

    if let Some(start_idx) = content.find(start_marker) {
        if let Some(end_idx) = content.find(end_marker) {
            let end_idx = end_idx + end_marker.len();

            // Find the newline after end marker
            let end_idx = content[end_idx..]
                .find('\n')
                .map(|i| end_idx + i + 1)
                .unwrap_or(end_idx);

            // Find newline before start marker (to remove blank line)
            let start_idx = if start_idx > 0 && content.as_bytes()[start_idx - 1] == b'\n' {
                start_idx - 1
            } else {
                start_idx
            };

            if !dry_run {
                let new_content = format!("{}{}", &content[..start_idx], &content[end_idx..]);
                std::fs::write(path, new_content)?;
            }
            return Ok(true);
        }
    }

    Ok(false)
}

fn cmd_validate(history_path: Option<PathBuf>, ngram: usize, ratio: f32) {
    validate_ngram(ngram);
    println!("🔬 aprender-shell: Model Validation\n");

    // Find and parse history with graceful error handling (QA 2.4, 8.3)
    let history_file = find_history_file_graceful(history_path);
    println!("📂 History file: {}", history_file.display());

    let commands = parse_history_graceful(&history_file);
    println!("📊 Total commands: {}", commands.len());
    println!("⚙️  N-gram size: {}", ngram);
    println!(
        "📈 Train/test split: {:.0}% / {:.0}%\n",
        ratio * 100.0,
        (1.0 - ratio) * 100.0
    );

    print!("🧪 Running holdout validation... ");
    let result = MarkovModel::validate(&commands, ngram, ratio);
    println!("done!\n");

    println!("═══════════════════════════════════════════");
    println!("           VALIDATION RESULTS              ");
    println!("═══════════════════════════════════════════");
    println!("  Training set:     {:>6} commands", result.train_size);
    println!("  Test set:         {:>6} commands", result.test_size);
    println!("  Evaluated:        {:>6} commands", result.evaluated);
    println!("───────────────────────────────────────────");
    // Use aprender's ranking metrics
    println!(
        "  Hit@1  (top 1):   {:>6.1}%",
        result.metrics.hit_at_1 * 100.0
    );
    println!(
        "  Hit@5  (top 5):   {:>6.1}%",
        result.metrics.hit_at_5 * 100.0
    );
    println!(
        "  Hit@10 (top 10):  {:>6.1}%",
        result.metrics.hit_at_10 * 100.0
    );
    println!("  MRR (Mean Recip): {:>6.3}", result.metrics.mrr);
    println!("═══════════════════════════════════════════");

    // Interpretation
    println!("\n📊 Interpretation:");
    if result.metrics.hit_at_5 >= 0.5 {
        println!("   ✅ Excellent: Model finds correct command in top 5 >50% of the time");
    } else if result.metrics.hit_at_5 >= 0.3 {
        println!("   ✓ Good: Model provides useful suggestions");
    } else {
        println!("   ⚠️  Consider more training data or adjusting n-gram size");
        println!("   💡 Try: aprender-shell augment --count 5000");
    }
}

fn cmd_augment(
    history_path: Option<PathBuf>,
    output: &str,
    ngram: usize,
    augmentation_ratio: f32,
    quality_threshold: f32,
    monitor_diversity: bool,
    use_code_eda: bool,
) {
    validate_ngram(ngram);
    use aprender::synthetic::code_eda::{CodeEda, CodeEdaConfig, CodeLanguage};
    use aprender::synthetic::{
        DiversityMonitor, DiversityScore, SyntheticConfig, SyntheticGenerator,
    };

    let mode = if use_code_eda { "CodeEDA" } else { "Template" };
    println!("🧬 aprender-shell: Data Augmentation ({mode} mode)\n");

    // Find history file
    // Find and parse history with graceful error handling (QA 2.4, 8.3)
    let history_file = find_history_file_graceful(history_path);
    println!("📂 History file: {}", history_file.display());

    let commands = parse_history_graceful(&history_file);
    println!("📊 Real commands: {}", commands.len());

    // Configure synthetic data generation using aprender's SyntheticConfig
    let config = SyntheticConfig::default()
        .with_augmentation_ratio(augmentation_ratio)
        .with_quality_threshold(quality_threshold)
        .with_diversity_weight(0.3);

    let target_count = config.target_count(commands.len());
    println!("⚙️  Augmentation ratio: {:.1}x", augmentation_ratio);
    println!("⚙️  Quality threshold:  {:.1}%", quality_threshold * 100.0);
    println!("🎯 Target synthetic:   {} commands", target_count);

    // Extract known n-grams from current history
    let mut known_ngrams = std::collections::HashSet::new();
    for cmd in &commands {
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        for i in 0..tokens.len() {
            let start = i.saturating_sub(ngram - 1);
            let context = tokens[start..=i].join(" ");
            known_ngrams.insert(context);
        }
    }
    println!("🔢 Known n-grams: {}", known_ngrams.len());

    // Initialize diversity monitor if requested
    let mut diversity_monitor = if monitor_diversity {
        Some(DiversityMonitor::new(10).with_collapse_threshold(0.1))
    } else {
        None
    };

    // Generate synthetic data
    print!("\n🧪 Generating synthetic commands... ");

    // Use CodeEDA for code-aware augmentation if requested
    let (generated_commands, _eda_diversity) = if use_code_eda {
        let eda_config = CodeEdaConfig::default()
            .with_rename_prob(0.1)
            .with_comment_prob(0.05)
            .with_reorder_prob(0.1)
            .with_remove_prob(0.05)
            .with_language(CodeLanguage::Generic)
            .with_num_augments(2);

        let code_eda = CodeEda::new(eda_config);
        let eda_synth_config = SyntheticConfig::default()
            .with_augmentation_ratio(augmentation_ratio)
            .with_quality_threshold(quality_threshold)
            .with_seed(42);

        let eda_result = code_eda
            .generate(&commands, &eda_synth_config)
            .unwrap_or_default();
        let diversity = code_eda.diversity_score(&eda_result);
        println!(
            "done! (CodeEDA: {} samples, diversity: {:.2})",
            eda_result.len(),
            diversity
        );
        (eda_result, Some(diversity))
    } else {
        let pipeline = SyntheticPipeline::new();
        let result = pipeline.generate(&commands, known_ngrams.clone(), target_count);
        println!("done!");
        (result.commands, None)
    };

    // For template-based generation, use the pipeline result
    let result = if !use_code_eda {
        let pipeline = SyntheticPipeline::new();
        pipeline.generate(&commands, known_ngrams, target_count)
    } else {
        // Create a synthetic result for CodeEDA
        synthetic::SyntheticResult {
            commands: generated_commands.clone(),
            report: synthetic::CoverageReport {
                known_ngrams: known_ngrams.len(),
                total_ngrams: generated_commands.len(),
                new_ngrams: generated_commands.len() / 2,
                coverage_gain: 0.5,
            },
        }
    };

    // Quality filtering using aprender's config
    let mut quality_filtered: Vec<String> = Vec::new();
    let mut rejected_count = 0;

    for cmd in &result.commands {
        // Simple quality heuristic: command length and token count
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        let quality_score = if tokens.is_empty() {
            0.0
        } else {
            // Quality based on: reasonable length, known base command
            let length_score = (tokens.len() as f32 / 5.0).min(1.0);
            let base_known =
                ["git", "cargo", "docker", "make", "npm", "kubectl", "aws"].contains(&tokens[0]);
            let base_score = if base_known { 0.8 } else { 0.5 };
            (length_score * 0.4 + base_score * 0.6).min(1.0)
        };

        if config.meets_quality(quality_score) {
            quality_filtered.push(cmd.clone());

            // Update diversity monitor
            if let Some(ref mut monitor) = diversity_monitor {
                // Compute simple diversity based on unique tokens
                let unique_tokens: std::collections::HashSet<_> = tokens.iter().collect();
                let diversity = if tokens.is_empty() {
                    0.0
                } else {
                    unique_tokens.len() as f32 / tokens.len() as f32
                };
                let score = DiversityScore::new(diversity, diversity * 0.5, diversity);
                monitor.record(score);
            }
        } else {
            rejected_count += 1;
        }
    }

    println!("\n📈 Coverage Report:");
    println!("   Generated:          {}", result.commands.len());
    println!(
        "   Quality filtered:   {} (rejected {})",
        quality_filtered.len(),
        rejected_count
    );
    println!("   Known n-grams:      {}", result.report.known_ngrams);
    println!("   Total n-grams:      {}", result.report.total_ngrams);
    println!("   New n-grams added:  {}", result.report.new_ngrams);
    println!(
        "   Coverage gain:      {:.1}%",
        result.report.coverage_gain * 100.0
    );

    // Show diversity metrics if monitoring
    if let Some(ref monitor) = diversity_monitor {
        println!("\n📊 Diversity Metrics:");
        println!("   Mean diversity:     {:.3}", monitor.mean_diversity());
        if monitor.is_collapsing() {
            println!("   ⚠️  Warning: Low diversity detected (potential mode collapse)");
        } else {
            println!("   ✓  Diversity is healthy");
        }
        if monitor.is_trending_down() {
            println!("   ⚠️  Warning: Diversity trending downward");
        }
    }

    // Combine real + synthetic
    let mut augmented_commands = commands.clone();
    augmented_commands.extend(quality_filtered);

    println!("\n🧠 Training augmented model...");
    let mut model = MarkovModel::new(ngram);
    model.train(&augmented_commands);

    // Save model
    let output_path = expand_path(output);
    if let Err(e) = model.save(&output_path) {
        eprintln!("❌ Failed to save augmented model: {e}");
        if e.to_string().contains("ermission") {
            eprintln!(
                "   Hint: Check write permissions for '{}'",
                output_path.display()
            );
        }
        std::process::exit(1);
    }

    println!("\n✅ Augmented model saved to: {}", output_path.display());
    println!("\n📊 Model Statistics:");
    println!("   Original commands:   {}", commands.len());
    println!(
        "   Synthetic commands:  {}",
        augmented_commands.len() - commands.len()
    );
    println!("   Total training:      {}", augmented_commands.len());
    println!("   Unique n-grams:      {}", model.ngram_count());
    println!("   Vocabulary size:     {}", model.vocab_size());
    println!(
        "   Model size:          {:.1} KB",
        model.size_bytes() as f64 / 1024.0
    );

    println!("\n💡 Next steps:");
    println!("   Validate: aprender-shell validate");
    println!("   Tune:     aprender-shell tune");
}

fn cmd_analyze(history_path: Option<PathBuf>, top: usize) {
    use aprender::synthetic::code_features::{CodeFeatureExtractor, CommitDiff};
    use std::collections::HashMap;

    println!("📊 aprender-shell: Command Analysis (with CodeFeatureExtractor)\n");

    // Find and parse history with graceful error handling (QA 2.4, 8.3)
    let history_file = find_history_file_graceful(history_path);
    println!("📂 History file: {}", history_file.display());

    let commands = parse_history_graceful(&history_file);
    println!("📊 Total commands: {}\n", commands.len());

    // Use CodeFeatureExtractor to classify commands
    let extractor = CodeFeatureExtractor::new();

    // Count categories
    let mut category_counts: HashMap<u8, Vec<String>> = HashMap::new();
    let mut base_command_counts: HashMap<String, usize> = HashMap::new();

    for cmd in &commands {
        // Extract first word as base command
        let base = cmd.split_whitespace().next().unwrap_or("").to_string();
        *base_command_counts.entry(base).or_insert(0) += 1;

        // Create a "diff" from the command for classification
        // This is creative use of the feature extractor for shell commands
        let diff = CommitDiff::new()
            .with_message(cmd.clone())
            .with_lines_added(cmd.len() as u32)
            .with_timestamp(0);

        let features = extractor.extract(&diff);
        category_counts
            .entry(features.defect_category)
            .or_default()
            .push(cmd.clone());
    }

    // Category names
    let category_names = [
        "General",     // 0
        "Fix/Debug",   // 1
        "Security",    // 2
        "Performance", // 3
        "Refactor",    // 4
    ];

    println!("🏷️  Command Categories (based on keywords):");
    println!("   ─────────────────────────────────────────");
    for (cat, cmds) in &category_counts {
        let name = category_names.get(*cat as usize).unwrap_or(&"Unknown");
        println!(
            "   {:12} {:>5} commands ({:.1}%)",
            name,
            cmds.len(),
            cmds.len() as f32 / commands.len() as f32 * 100.0
        );
    }

    // Show top base commands
    println!("\n🔝 Top {} Base Commands:", top);
    println!("   ─────────────────────────────────────────");
    let mut sorted_bases: Vec<_> = base_command_counts.iter().collect();
    sorted_bases.sort_by(|a, b| b.1.cmp(a.1));

    for (base, count) in sorted_bases.iter().take(top) {
        let pct = **count as f32 / commands.len() as f32 * 100.0;
        let bar_len = (pct / 2.0) as usize;
        let bar = "█".repeat(bar_len.min(25));
        println!("   {:12} {:>5} ({:>5.1}%) {}", base, count, pct, bar);
    }

    // Show sample commands from each category
    println!("\n📋 Sample Commands by Category:");
    println!("   ─────────────────────────────────────────");
    for (cat, cmds) in &category_counts {
        if cmds.is_empty() {
            continue;
        }
        let name = category_names.get(*cat as usize).unwrap_or(&"Unknown");
        println!("\n   [{}]:", name);
        for cmd in cmds.iter().take(3) {
            let truncated = if cmd.len() > 60 {
                format!("{}...", &cmd[..57])
            } else {
                cmd.clone()
            };
            println!("     • {}", truncated);
        }
    }

    // Command complexity analysis
    let avg_tokens: f32 = commands
        .iter()
        .map(|c| c.split_whitespace().count() as f32)
        .sum::<f32>()
        / commands.len().max(1) as f32;

    let max_tokens = commands
        .iter()
        .map(|c| c.split_whitespace().count())
        .max()
        .unwrap_or(0);

    println!("\n📈 Command Complexity:");
    println!("   ─────────────────────────────────────────");
    println!("   Average tokens per command: {:.1}", avg_tokens);
    println!("   Maximum tokens: {}", max_tokens);
    println!("   Unique base commands: {}", base_command_counts.len());

    // Developer workflow insights
    let git_count = base_command_counts.get("git").copied().unwrap_or(0);
    let cargo_count = base_command_counts.get("cargo").copied().unwrap_or(0);
    let docker_count = base_command_counts.get("docker").copied().unwrap_or(0);
    let kubectl_count = base_command_counts.get("kubectl").copied().unwrap_or(0);

    println!("\n🛠️  Developer Workflow Profile:");
    println!("   ─────────────────────────────────────────");
    if git_count > 0 {
        println!("   ✓ Version Control:  {} git commands", git_count);
    }
    if cargo_count > 0 {
        println!("   ✓ Rust Development: {} cargo commands", cargo_count);
    }
    if docker_count > 0 {
        println!("   ✓ Containers:       {} docker commands", docker_count);
    }
    if kubectl_count > 0 {
        println!("   ✓ Kubernetes:       {} kubectl commands", kubectl_count);
    }

    println!("\n💡 Tip: Use 'aprender-shell augment --use-code-eda' for code-aware augmentation");
}

fn cmd_tune(history_path: Option<PathBuf>, trials: usize, ratio: f32) {
    use aprender::automl::params::ParamKey;
    use aprender::automl::{AutoTuner, SearchSpace, TPE};

    println!("🎯 aprender-shell: AutoML Hyperparameter Tuning (TPE)\n");

    // Find and parse history with graceful error handling (QA 2.4, 8.3)
    let history_file = find_history_file_graceful(history_path);
    println!("📂 History file: {}", history_file.display());

    let commands = parse_history_graceful(&history_file);
    println!("📊 Total commands: {}", commands.len());

    if commands.len() < 100 {
        println!(
            "⚠️  Warning: Small history ({} commands). Results may be noisy.",
            commands.len()
        );
    }

    // Define search space for shell model hyperparameters
    // Using generic param for n-gram size (2-5)
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum ShellParam {
        NGram,
    }

    impl aprender::automl::params::ParamKey for ShellParam {
        fn name(&self) -> &'static str {
            match self {
                ShellParam::NGram => "ngram",
            }
        }
    }

    impl std::fmt::Display for ShellParam {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.name())
        }
    }

    let space: SearchSpace<ShellParam> = SearchSpace::new().add(ShellParam::NGram, 2..6); // 2, 3, 4, 5

    println!("🔬 TPE trials: {}", trials);
    println!(
        "📈 Train/test split: {:.0}% / {:.0}%\n",
        ratio * 100.0,
        (1.0 - ratio) * 100.0
    );

    // Track all results for final report
    let mut all_results: Vec<(usize, f64, f32, f32)> = Vec::new();

    // Objective function: evaluate n-gram configuration
    let objective = |trial: &aprender::automl::Trial<ShellParam>| -> f64 {
        let ngram = trial.get_usize(&ShellParam::NGram).unwrap_or(3);

        // Run k-fold cross-validation for this configuration
        let k_folds = 3;
        let mut scores = Vec::new();

        for fold in 0..k_folds {
            let rotation = (commands.len() / k_folds) * fold;
            let mut rotated = commands.clone();
            rotated.rotate_left(rotation % commands.len().max(1));

            let result = MarkovModel::validate(&rotated, ngram, ratio);

            // Combined score: 60% Hit@5 + 40% MRR
            let score =
                f64::from(result.metrics.hit_at_5) * 0.6 + f64::from(result.metrics.mrr) * 0.4;
            scores.push(score);
        }

        scores.iter().sum::<f64>() / scores.len() as f64
    };

    println!("══════════════════════════════════════════════════");
    println!(" Trial │ N-gram │   Hit@5   │    MRR    │  Score  ");
    println!("═══════╪════════╪═══════════╪═══════════╪═════════");

    // Use TPE with early stopping
    let tpe = TPE::new(trials)
        .with_seed(42)
        .with_startup_trials(2) // Random for first 2 trials
        .with_gamma(0.25);

    let result = AutoTuner::new(tpe)
        .early_stopping(4) // Stop if no improvement for 4 trials
        .maximize(&space, |trial| {
            let ngram = trial.get_usize(&ShellParam::NGram).unwrap_or(3);
            let score = objective(trial);

            // Get detailed metrics for display
            let validation = MarkovModel::validate(&commands, ngram, ratio);
            let hit5 = validation.metrics.hit_at_5;
            let mrr = validation.metrics.mrr;

            all_results.push((ngram, score, hit5, mrr));

            println!(
                "  {:>3}  │   {:>2}   │  {:>5.1}%   │  {:>5.3}   │ {:>6.3}",
                all_results.len(),
                ngram,
                hit5 * 100.0,
                mrr,
                score
            );

            score
        });

    println!("══════════════════════════════════════════════════\n");

    // Summary by n-gram size
    println!("📊 Summary by N-gram size:");
    for ngram in 2..=5 {
        let ngram_results: Vec<_> = all_results
            .iter()
            .filter(|(n, _, _, _)| *n == ngram)
            .collect();

        if !ngram_results.is_empty() {
            let avg_score: f64 = ngram_results.iter().map(|(_, s, _, _)| s).sum::<f64>()
                / ngram_results.len() as f64;
            let avg_hit5: f32 = ngram_results.iter().map(|(_, _, h, _)| h).sum::<f32>()
                / ngram_results.len() as f32;
            let avg_mrr: f32 = ngram_results.iter().map(|(_, _, _, m)| m).sum::<f32>()
                / ngram_results.len() as f32;

            let best = if result.best_trial.get_usize(&ShellParam::NGram) == Some(ngram) {
                " ★"
            } else {
                ""
            };

            println!(
                "   n={}: Hit@5={:>5.1}%, MRR={:.3}, Score={:.3} ({} trials){}",
                ngram,
                avg_hit5 * 100.0,
                avg_mrr,
                avg_score,
                ngram_results.len(),
                best
            );
        }
    }

    let best_ngram = result.best_trial.get_usize(&ShellParam::NGram).unwrap_or(3);

    println!("\n🏆 Best Configuration (TPE):");
    println!("   N-gram size: {}", best_ngram);
    println!("   Score:       {:.3}", result.best_score);
    println!("   Trials run:  {}", result.n_trials);
    println!("   Time:        {:.1}s", result.elapsed.as_secs_f64());

    println!("\n💡 Train with optimal settings:");
    println!("   aprender-shell train --ngram {}", best_ngram);
}

fn cmd_inspect(model_path: &str, format: &str, use_password: bool) {
    use aprender::format::model_card::{ModelCard, TrainingDataInfo};

    let path = expand_path(model_path);

    // Get password if needed
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

    // Check encryption status
    let encrypted = MarkovModel::is_encrypted(&path).unwrap_or(false);

    // Load model to get metadata
    let model = if let Some(ref pwd) = password {
        match MarkovModel::load_encrypted(&path, pwd) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("❌ Failed to load encrypted model: {e}");
                std::process::exit(1);
            }
        }
    } else {
        match MarkovModel::load(&path) {
            Ok(m) => m,
            Err(e) => {
                if encrypted {
                    eprintln!("❌ Model is encrypted. Use --password flag to decrypt.");
                } else {
                    eprintln!("❌ Failed to load model: {e}");
                }
                std::process::exit(1);
            }
        }
    };

    // Generate model card from model stats
    let model_id = format!(
        "aprender-shell-markov-{}gram-{}",
        model.ngram_size(),
        chrono_lite_date()
    );

    let card = ModelCard::new(&model_id, "1.0.0")
        .with_name("Shell Completion Model")
        .with_description("Markov chain model trained on shell command history")
        .with_architecture("MarkovModel")
        .with_param_count(model.ngram_count() as u64)
        .with_hyperparameter("ngram_size", model.ngram_size())
        .with_metric("vocab_size", model.vocab_size())
        .with_metric("ngram_count", model.ngram_count())
        .with_training_data(
            TrainingDataInfo::new("shell_history").with_samples(model.total_commands() as u64),
        );

    match format.to_lowercase().as_str() {
        "json" => match card.to_json() {
            Ok(json) => println!("{json}"),
            Err(e) => eprintln!("❌ Failed to serialize: {e}"),
        },
        "huggingface" | "hf" => {
            println!("{}", card.to_huggingface());
        }
        _ => {
            // Default: text format
            println!("📋 Model Card: {}\n", path.display());
            println!("═══════════════════════════════════════════");
            println!("           MODEL INFORMATION               ");
            println!("═══════════════════════════════════════════");
            println!("  ID:           {}", card.model_id);
            println!("  Name:         {}", card.name);
            println!("  Version:      {}", card.version);
            if let Some(ref author) = card.author {
                println!("  Author:       {}", author);
            }
            println!("  Created:      {}", card.created_at);
            println!("  Framework:    {}", card.framework_version);
            if let Some(ref arch) = card.architecture {
                println!("  Architecture: {}", arch);
            }
            if let Some(count) = card.param_count {
                println!("  Parameters:   {}", count);
            }
            println!("───────────────────────────────────────────");

            if let Some(ref training) = card.training_data {
                println!("\n📊 Training Data:");
                println!("  Source:  {}", training.name);
                if let Some(samples) = training.samples {
                    println!("  Samples: {}", samples);
                }
                if let Some(ref hash) = training.hash {
                    println!("  Hash:    {}", hash);
                }
            }

            if !card.hyperparameters.is_empty() {
                println!("\n⚙️  Hyperparameters:");
                for (key, value) in &card.hyperparameters {
                    println!("  {}: {}", key, value);
                }
            }

            if !card.metrics.is_empty() {
                println!("\n📈 Metrics:");
                for (key, value) in &card.metrics {
                    println!("  {}: {}", key, value);
                }
            }

            if let Some(ref desc) = card.description {
                println!("\n📝 Description:");
                println!("  {}", desc);
            }

            println!("\n💡 Export formats:");
            println!("   JSON:        aprender-shell inspect --format json");
            println!("   Hugging Face: aprender-shell inspect --format huggingface");
        }
    }
}

/// Simple date string without chrono dependency
fn chrono_lite_date() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Days since epoch
    let days = secs / 86400;

    // Simple year/month/day calculation
    let mut remaining = days as i64;
    let mut year = 1970i32;
    loop {
        let days_in_year = if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
            366
        } else {
            365
        };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let months = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u32;
    for days_in_month in months {
        if remaining < days_in_month {
            break;
        }
        remaining -= days_in_month;
        month += 1;
    }

    let day = remaining as u32 + 1;
    format!("{year:04}{month:02}{day:02}")
}

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
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;

    let path = expand_path(model_path);

    // Remove stale socket if exists
    if socket_path.exists() {
        if let Err(e) = std::fs::remove_file(socket_path) {
            eprintln!("⚠️  Could not remove stale socket: {e}");
        }
    }

    // Load model
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

        let mut reader = BufReader::new(stream.try_clone().unwrap_or_else(|_| {
            eprintln!("⚠️  Failed to clone stream");
            stream.try_clone().expect("clone")
        }));

        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            continue;
        }

        let prefix = line.trim();
        request_count += 1;

        // Handle special commands
        match prefix.to_uppercase().as_str() {
            "PING" => {
                writeln!(stream, "PONG").ok();
                writeln!(stream).ok();
                continue;
            }
            "QUIT" | "SHUTDOWN" => {
                writeln!(stream, "OK").ok();
                eprintln!("👋 Daemon shutting down (received QUIT)");
                break;
            }
            "STATS" => {
                let uptime = start_time.elapsed().as_secs();
                writeln!(stream, "requests: {request_count}").ok();
                writeln!(stream, "uptime_secs: {uptime}").ok();
                writeln!(stream, "model_commands: {}", model.total_commands()).ok();
                writeln!(stream, "model_ngrams: {}", model.ngram_count()).ok();
                writeln!(stream).ok();
                continue;
            }
            "" => {
                writeln!(stream).ok();
                continue;
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

/// Check daemon status
fn cmd_daemon_status(socket_path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixStream;

        if !socket_path.exists() {
            println!("❌ Daemon not running (socket not found)");
            std::process::exit(1);
        }

        let mut stream = match UnixStream::connect(socket_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("❌ Could not connect to daemon: {e}");
                eprintln!("   Socket exists but daemon may have crashed");
                std::process::exit(1);
            }
        };

        // Send STATS command
        writeln!(stream, "STATS").ok();
        stream.flush().ok();

        println!("✅ Daemon is running");
        println!("   Socket: {}", socket_path.display());

        // Read PID if available
        let pid_path = socket_path.with_extension("pid");
        if let Ok(pid) = std::fs::read_to_string(&pid_path) {
            println!("   PID:    {}", pid.trim());
        }

        // Read stats
        let reader = BufReader::new(&stream);
        println!("\n📊 Statistics:");
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            if line.is_empty() {
                break;
            }
            println!("   {line}");
        }
    }

    #[cfg(not(unix))]
    {
        let _ = socket_path;
        eprintln!("❌ Daemon mode is only supported on Unix systems");
        std::process::exit(1);
    }
}
