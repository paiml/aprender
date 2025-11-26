//! aprender-shell: AI-powered shell completion trained on your history
//!
//! Train a personalized autocomplete model on your shell history in seconds.
//! 100% local, private, and fast.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod history;
mod model;
mod trie;

use history::HistoryParser;
use model::MarkovModel;

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
    },

    /// Show model statistics
    Stats {
        /// Model file
        #[arg(short, long, default_value = "~/.aprender-shell.model")]
        model: String,
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
}

fn expand_path(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(&path[2..]);
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
        } => {
            cmd_train(history, &output, ngram);
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
        } => {
            cmd_suggest(&prefix, &model, count);
        }
        Commands::Stats { model } => {
            cmd_stats(&model);
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
    }
}

fn cmd_train(history_path: Option<PathBuf>, output: &str, ngram: usize) {
    println!("🚀 aprender-shell: Training model...\n");

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

    // Train model
    print!("🧠 Training {}-gram model... ", ngram);
    let mut model = MarkovModel::new(ngram);
    model.train(&commands);
    println!("done!");

    // Save model
    let output_path = expand_path(output);
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

fn cmd_suggest(prefix: &str, model_path: &str, count: usize) {
    let path = expand_path(model_path);
    let model = MarkovModel::load(&path).expect("Failed to load model");

    let suggestions = model.suggest(prefix, count);

    if suggestions.is_empty() {
        // Silent for shell integration
        return;
    }

    for (suggestion, score) in suggestions {
        println!("{}\t{:.3}", suggestion, score);
    }
}

fn cmd_stats(model_path: &str) {
    let path = expand_path(model_path);
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
        r#"# aprender-shell ZSH widget
# Add this to your ~/.zshrc

_aprender_suggest() {{
    local suggestion
    suggestion=$(aprender-shell suggest "$BUFFER" 2>/dev/null | head -1 | cut -f1)
    if [[ -n "$suggestion" ]]; then
        POSTDISPLAY=" ${{suggestion#$BUFFER}}"
        POSTDISPLAY=$'\e[90m'"$POSTDISPLAY"$'\e[0m'
    else
        POSTDISPLAY=""
    fi
}}

_aprender_accept() {{
    if [[ -n "$POSTDISPLAY" ]]; then
        BUFFER="${{BUFFER}}${{POSTDISPLAY# }}"
        POSTDISPLAY=""
        CURSOR=$#BUFFER
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
"#
    );
}
