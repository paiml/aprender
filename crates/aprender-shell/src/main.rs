//! aprender-shell: AI-powered shell completion trained on your history
//!
//! Train a personalized autocomplete model on your shell history in seconds.
//! 100% local, private, and fast.

use aprender_shell::{synthetic, HistoryParser, MarkovModel, PagedMarkovModel, SyntheticPipeline};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    },

    /// Get completions for a prefix
    Suggest {
        /// The current command prefix
        prefix: String,

        /// Model file to use
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Number of suggestions
        #[arg(short, long, default_value = "5")]
        count: usize,

        /// Memory limit in MB (for paged models)
        #[arg(long)]
        memory_limit: Option<usize>,
    },

    /// Show model statistics
    Stats {
        /// Model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,

        /// Memory limit in MB (for paged models)
        #[arg(long)]
        memory_limit: Option<usize>,
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
        } => {
            cmd_train(history, &output, ngram, memory_limit);
        }
        Commands::Update {
            history,
            model,
            quiet,
        } => {
            cmd_update(history, &model, quiet);
        }
        Commands::Suggest {
            prefix,
            model,
            count,
            memory_limit,
        } => {
            cmd_suggest(&prefix, &model, count, memory_limit);
        }
        Commands::Stats {
            model,
            memory_limit,
        } => {
            cmd_stats(&model, memory_limit);
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
    }
}

fn cmd_train(
    history_path: Option<PathBuf>,
    output: &str,
    ngram: usize,
    memory_limit: Option<usize>,
) {
    let paged = memory_limit.is_some();
    let mode_str = if paged { "paged" } else { "standard" };
    println!("🚀 aprender-shell: Training {mode_str} model...\n");

    // Find history file
    let history_file = history_path.unwrap_or_else(|| {
        HistoryParser::find_history_file().expect("Could not find shell history file")
    });

    println!("📂 History file: {}", history_file.display());

    // Parse history
    let parser = HistoryParser::new();
    let commands = parser
        .parse_file(&history_file)
        .expect("Failed to parse history");

    println!("📊 Commands loaded: {}", commands.len());

    if commands.is_empty() {
        eprintln!("❌ No commands found in history file");
        std::process::exit(1);
    }

    let output_path = expand_path(output);

    if let Some(mem_mb) = memory_limit {
        // Paged model for large histories
        let output_path = output_path.with_extension("apbundle");
        print!(
            "🧠 Training {ngram}-gram paged model ({}MB limit)... ",
            mem_mb
        );
        let mut model = PagedMarkovModel::new(ngram, mem_mb);
        model.train(&commands);
        println!("done!");

        model
            .save(&output_path)
            .expect("Failed to save paged model");

        let stats = model.stats();
        println!("\n✅ Paged model saved to: {}", output_path.display());
        println!("\n📈 Model Statistics:");
        println!("   Segments:        {}", stats.total_segments);
        println!("   Vocabulary size: {}", stats.vocab_size);
        println!("   Memory limit:    {} MB", mem_mb);

        println!("\n💡 Next steps:");
        println!("   1. Test: aprender-shell suggest \"git \" --memory-limit {mem_mb}");
        println!("   2. Stats: aprender-shell stats --memory-limit {mem_mb}");
    } else {
        // Standard in-memory model
        print!("🧠 Training {ngram}-gram model... ");
        let mut model = MarkovModel::new(ngram);
        model.train(&commands);
        println!("done!");

        model.save(&output_path).expect("Failed to save model");

        println!("\n✅ Model saved to: {}", output_path.display());
        println!("\n📈 Model Statistics:");
        println!("   Unique n-grams: {}", model.ngram_count());
        println!("   Vocabulary size: {}", model.vocab_size());
        println!(
            "   Model size: {:.1} KB",
            model.size_bytes() as f64 / 1024.0
        );

        println!("\n💡 Next steps:");
        println!("   1. Test: aprender-shell suggest \"git \"");
        println!("   2. Install: aprender-shell zsh-widget >> ~/.zshrc");
    }
}

fn cmd_update(history_path: Option<PathBuf>, model_path: &str, quiet: bool) {
    let path = expand_path(model_path);

    // Load existing model or create new one
    let mut model = if path.exists() {
        MarkovModel::load(&path).expect("Failed to load model")
    } else {
        if !quiet {
            println!("📝 No existing model, creating new one...");
        }
        MarkovModel::new(3)
    };

    // Find history file
    let history_file = history_path.unwrap_or_else(|| {
        HistoryParser::find_history_file().expect("Could not find shell history file")
    });

    // Parse history
    let parser = HistoryParser::new();
    let all_commands = parser
        .parse_file(&history_file)
        .expect("Failed to parse history");

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

    // Save
    model.save(&path).expect("Failed to save model");

    if !quiet {
        println!(
            "✅ Model updated ({} total commands)",
            model.total_commands()
        );
    }
}

fn cmd_suggest(prefix: &str, model_path: &str, count: usize, memory_limit: Option<usize>) {
    let path = expand_path(model_path);

    if let Some(mem_mb) = memory_limit {
        // Paged model
        let paged_path = path.with_extension("apbundle");
        let mut model =
            PagedMarkovModel::load(&paged_path, mem_mb).expect("Failed to load paged model");

        let suggestions = model.suggest(prefix, count);

        if suggestions.is_empty() {
            return;
        }

        for (suggestion, score) in suggestions {
            println!("{}\t{:.3}", suggestion, score);
        }
    } else {
        // Standard model
        let model = MarkovModel::load(&path).expect("Failed to load model");

        let suggestions = model.suggest(prefix, count);

        if suggestions.is_empty() {
            return;
        }

        for (suggestion, score) in suggestions {
            println!("{}\t{:.3}", suggestion, score);
        }
    }
}

fn cmd_stats(model_path: &str, memory_limit: Option<usize>) {
    let path = expand_path(model_path);

    if let Some(mem_mb) = memory_limit {
        // Paged model stats
        let paged_path = path.with_extension("apbundle");
        let model =
            PagedMarkovModel::load(&paged_path, mem_mb).expect("Failed to load paged model");

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

        if let Some(paging_stats) = model.paging_stats() {
            println!("\n📈 Paging Statistics:");
            println!("   Page hits:       {}", paging_stats.hits);
            println!("   Page misses:     {}", paging_stats.misses);
            println!("   Evictions:       {}", paging_stats.evictions);
            if paging_stats.hits + paging_stats.misses > 0 {
                let hit_rate = paging_stats.hits as f64
                    / (paging_stats.hits + paging_stats.misses) as f64
                    * 100.0;
                println!("   Hit rate:        {:.1}%", hit_rate);
            }
        }

        println!("\n🔝 Top commands:");
        for (cmd, count) in model.top_commands(10) {
            println!("   {:>6}x  {}", count, cmd);
        }
    } else {
        // Standard model stats
        let model = MarkovModel::load(&path).expect("Failed to load model");

        println!("📊 Model Statistics:");
        println!("   N-gram size: {}", model.ngram_size());
        println!("   Unique n-grams: {}", model.ngram_count());
        println!("   Vocabulary size: {}", model.vocab_size());
        println!(
            "   Model size: {:.1} KB",
            model.size_bytes() as f64 / 1024.0
        );
        println!("\n🔝 Top commands:");
        for (cmd, count) in model.top_commands(10) {
            println!("   {:>6}x  {}", count, cmd);
        }
    }
}

fn cmd_export(model_path: &str, output: &PathBuf) {
    let path = expand_path(model_path);
    std::fs::copy(&path, output).expect("Failed to export model");
    println!("✅ Model exported to: {}", output.display());
}

fn cmd_import(input: &PathBuf, output: &str) {
    let output_path = expand_path(output);
    std::fs::copy(input, &output_path).expect("Failed to import model");
    println!("✅ Model imported to: {}", output_path.display());
}

fn cmd_zsh_widget() {
    print!(
        r#"# >>> aprender-shell widget >>>
# aprender-shell ZSH widget v2
# Add this to your ~/.zshrc
# Toggle: export APRENDER_DISABLED=1 to disable
# Uninstall: aprender-shell uninstall --zsh

_aprender_suggest() {{
    # Skip if disabled or buffer too short
    [[ "${{APRENDER_DISABLED:-0}}" == "1" ]] && return
    [[ "${{#BUFFER}}" -lt 2 ]] && {{ POSTDISPLAY=''; return; }}

    local suggestion
    # SC2046: Quote command substitution to prevent word splitting
    suggestion="$(timeout 0.1 aprender-shell suggest "$BUFFER" 2>/dev/null | head -1 | cut -f1)"

    # SC2086: Quote variables in comparisons
    if [[ -n "$suggestion" && "$suggestion" != "$BUFFER" ]]; then
        POSTDISPLAY=" ${{suggestion#"$BUFFER"}}"
        POSTDISPLAY=$'\e[90m'"$POSTDISPLAY"$'\e[0m'
    else
        POSTDISPLAY=""
    fi
}}

_aprender_accept() {{
    if [[ -n "$POSTDISPLAY" ]]; then
        # SC2086: Quote CURSOR variable
        BUFFER="${{BUFFER}}${{POSTDISPLAY# }}"
        POSTDISPLAY=""
        CURSOR="${{#BUFFER}}"
    fi
    zle redisplay
}}

zle -N _aprender_suggest
zle -N _aprender_accept

# Trigger on each keystroke
autoload -Uz add-zle-hook-widget
add-zle-hook-widget line-pre-redraw _aprender_suggest

# Accept with Tab or Right Arrow
bindkey '^I' _aprender_accept      # Tab
bindkey '^[[C' _aprender_accept    # Right arrow
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
    // If no shell specified, try to detect and uninstall from all
    let detect_all = !zsh && !bash && !fish;

    let home = dirs::home_dir().expect("Could not find home directory");
    let mut removed_any = false;

    let action = if dry_run { "Would remove" } else { "Removed" };

    // ZSH
    if zsh || detect_all {
        let zshrc = home.join(".zshrc");
        if zshrc.exists() {
            match remove_widget_block(&zshrc, dry_run) {
                Ok(true) => {
                    println!("✓ {} widget from {}", action, zshrc.display());
                    removed_any = true;
                }
                Ok(false) => {
                    if zsh {
                        println!("ℹ No widget found in {}", zshrc.display());
                    }
                }
                Err(e) => eprintln!("✗ Error processing {}: {}", zshrc.display(), e),
            }
        } else if zsh {
            println!("ℹ {} does not exist", zshrc.display());
        }
    }

    // Bash
    if bash || detect_all {
        let bashrc = home.join(".bashrc");
        if bashrc.exists() {
            match remove_widget_block(&bashrc, dry_run) {
                Ok(true) => {
                    println!("✓ {} widget from {}", action, bashrc.display());
                    removed_any = true;
                }
                Ok(false) => {
                    if bash {
                        println!("ℹ No widget found in {}", bashrc.display());
                    }
                }
                Err(e) => eprintln!("✗ Error processing {}: {}", bashrc.display(), e),
            }
        } else if bash {
            println!("ℹ {} does not exist", bashrc.display());
        }
    }

    // Fish
    if fish || detect_all {
        let fish_config = home.join(".config/fish/config.fish");
        if fish_config.exists() {
            match remove_widget_block(&fish_config, dry_run) {
                Ok(true) => {
                    println!("✓ {} widget from {}", action, fish_config.display());
                    removed_any = true;
                }
                Ok(false) => {
                    if fish {
                        println!("ℹ No widget found in {}", fish_config.display());
                    }
                }
                Err(e) => eprintln!("✗ Error processing {}: {}", fish_config.display(), e),
            }
        } else if fish {
            println!("ℹ {} does not exist", fish_config.display());
        }
    }

    // Model file
    if !keep_model {
        let model_file = home.join(".aprender-shell.model");
        if model_file.exists() {
            if dry_run {
                println!("✓ Would remove model file {}", model_file.display());
            } else {
                match std::fs::remove_file(&model_file) {
                    Ok(()) => println!("✓ Removed model file {}", model_file.display()),
                    Err(e) => eprintln!("✗ Error removing model file: {}", e),
                }
            }
            removed_any = true;
        }

        // Also check for paged model bundle
        let bundle_file = home.join(".aprender-shell.apbundle");
        if bundle_file.exists() {
            if dry_run {
                println!("✓ Would remove model bundle {}", bundle_file.display());
            } else {
                match std::fs::remove_dir_all(&bundle_file) {
                    Ok(()) => println!("✓ Removed model bundle {}", bundle_file.display()),
                    Err(e) => eprintln!("✗ Error removing model bundle: {}", e),
                }
            }
            removed_any = true;
        }
    }

    if removed_any {
        if dry_run {
            println!("\n💡 Run without --dry-run to apply changes");
        } else {
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
    } else {
        println!("ℹ No aprender-shell installation found");
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
    println!("🔬 aprender-shell: Model Validation\n");

    // Find history file
    let history_file = history_path.unwrap_or_else(|| {
        HistoryParser::find_history_file().expect("Could not find shell history file")
    });

    println!("📂 History file: {}", history_file.display());

    // Parse history
    let parser = HistoryParser::new();
    let commands = parser
        .parse_file(&history_file)
        .expect("Failed to parse history");

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
    use aprender::synthetic::code_eda::{CodeEda, CodeEdaConfig, CodeLanguage};
    use aprender::synthetic::{
        DiversityMonitor, DiversityScore, SyntheticConfig, SyntheticGenerator,
    };

    let mode = if use_code_eda { "CodeEDA" } else { "Template" };
    println!("🧬 aprender-shell: Data Augmentation ({mode} mode)\n");

    // Find history file
    let history_file = history_path.unwrap_or_else(|| {
        HistoryParser::find_history_file().expect("Could not find shell history file")
    });

    println!("📂 History file: {}", history_file.display());

    // Parse history
    let parser = HistoryParser::new();
    let commands = parser
        .parse_file(&history_file)
        .expect("Failed to parse history");

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
    model.save(&output_path).expect("Failed to save model");

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

    // Find history file
    let history_file = history_path.unwrap_or_else(|| {
        HistoryParser::find_history_file().expect("Could not find shell history file")
    });

    println!("📂 History file: {}", history_file.display());

    // Parse history
    let parser = HistoryParser::new();
    let commands = parser
        .parse_file(&history_file)
        .expect("Failed to parse history");

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

    // Find history file
    let history_file = history_path.unwrap_or_else(|| {
        HistoryParser::find_history_file().expect("Could not find shell history file")
    });

    println!("📂 History file: {}", history_file.display());

    // Parse history
    let parser = HistoryParser::new();
    let commands = parser
        .parse_file(&history_file)
        .expect("Failed to parse history");

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
