//! aprender-shell: AI-powered shell completion trained on your history
//!
//! Train a personalized autocomplete model on your shell history in seconds.
//! 100% local, private, and fast.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod history;
mod model;
mod synthetic;
mod trie;

use history::HistoryParser;
use model::MarkovModel;
use synthetic::SyntheticPipeline;

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

        /// Number of synthetic commands to generate
        #[arg(short, long, default_value = "5000")]
        count: usize,
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
            count,
        } => {
            cmd_augment(history, &output, ngram, count);
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

fn cmd_augment(history_path: Option<PathBuf>, output: &str, ngram: usize, count: usize) {
    println!("🧬 aprender-shell: Data Augmentation\n");

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

    // Generate synthetic data
    print!("\n🧪 Generating synthetic commands... ");
    let pipeline = SyntheticPipeline::new();
    let result = pipeline.generate(&commands, known_ngrams, count);
    println!("done!");

    println!("\n📈 Coverage Report:");
    println!("   Synthetic commands: {}", result.commands.len());
    println!("   Known n-grams:      {}", result.report.known_ngrams);
    println!("   Total n-grams:      {}", result.report.total_ngrams);
    println!("   New n-grams added:  {}", result.report.new_ngrams);
    println!(
        "   Coverage gain:      {:.1}%",
        result.report.coverage_gain * 100.0
    );

    // Combine real + synthetic
    let mut augmented_commands = commands.clone();
    augmented_commands.extend(result.commands);

    println!("\n🧠 Training augmented model...");
    let mut model = MarkovModel::new(ngram);
    model.train(&augmented_commands);

    // Save model
    let output_path = expand_path(output);
    model.save(&output_path).expect("Failed to save model");

    println!("\n✅ Augmented model saved to: {}", output_path.display());
    println!("\n📊 Model Statistics:");
    println!("   Total training commands: {}", augmented_commands.len());
    println!("   Unique n-grams: {}", model.ngram_count());
    println!("   Vocabulary size: {}", model.vocab_size());
    println!(
        "   Model size: {:.1} KB",
        model.size_bytes() as f64 / 1024.0
    );

    println!("\n💡 Validate improvement:");
    println!("   aprender-shell validate");
}

fn cmd_tune(history_path: Option<PathBuf>, trials: usize, ratio: f32) {
    println!("🎯 aprender-shell: AutoML Hyperparameter Tuning\n");

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

    // For small discrete space (n=2,3,4,5), use exhaustive grid search
    // with multiple folds per config to reduce variance
    let ngram_sizes = [2_usize, 3, 4, 5];
    let folds_per_config = trials.max(1);

    println!("🔬 Folds per config: {}", folds_per_config);
    println!(
        "📈 Train/test split: {:.0}% / {:.0}%\n",
        ratio * 100.0,
        (1.0 - ratio) * 100.0
    );

    println!("══════════════════════════════════════════════════");
    println!(" N-gram │   Hit@5      │    MRR       │  Score  ");
    println!("════════╪══════════════╪══════════════╪═════════");

    #[derive(Default)]
    struct Stats {
        hit5_values: Vec<f32>,
        mrr_values: Vec<f32>,
    }

    impl Stats {
        fn mean_std(values: &[f32]) -> (f32, f32) {
            let n = values.len() as f32;
            let mean = values.iter().sum::<f32>() / n;
            let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            (mean, var.sqrt())
        }
    }

    let mut results: Vec<(usize, Stats)> = Vec::new();

    for &ngram in &ngram_sizes {
        let mut stats = Stats::default();

        // Cross-validation with different splits using rotation
        for fold in 0..folds_per_config {
            let rotation = (commands.len() / folds_per_config) * fold;
            let mut rotated = commands.clone();
            rotated.rotate_left(rotation % commands.len().max(1));

            let result = MarkovModel::validate(&rotated, ngram, ratio);

            stats.hit5_values.push(result.metrics.hit_at_5);
            stats.mrr_values.push(result.metrics.mrr);
        }

        results.push((ngram, stats));
    }

    let mut best_ngram = 3_usize;
    let mut best_score = 0.0_f32;

    for (ngram, stats) in &results {
        let (hit5_mean, hit5_std) = Stats::mean_std(&stats.hit5_values);
        let (mrr_mean, mrr_std) = Stats::mean_std(&stats.mrr_values);

        // Score = weighted combination
        let score = hit5_mean * 0.6 + mrr_mean * 0.4;

        let marker = if score > best_score { " ★" } else { "  " };

        println!(
            "   {:>2}   │ {:>5.1}% ±{:>4.1} │ {:>5.3} ±{:>5.3} │ {:>5.3}{}",
            ngram,
            hit5_mean * 100.0,
            hit5_std * 100.0,
            mrr_mean,
            mrr_std,
            score,
            marker
        );

        if score > best_score {
            best_score = score;
            best_ngram = *ngram;
        }
    }

    println!("══════════════════════════════════════════════════\n");

    // Detailed analysis
    println!("📊 Analysis:");
    for (ngram, stats) in &results {
        let (hit5_mean, _) = Stats::mean_std(&stats.hit5_values);
        let (mrr_mean, _) = Stats::mean_std(&stats.mrr_values);
        let score = hit5_mean * 0.6 + mrr_mean * 0.4;

        let recommendation = if *ngram == best_ngram {
            "← BEST"
        } else if score > best_score * 0.95 {
            "(competitive)"
        } else {
            ""
        };

        println!(
            "   n={}: Hit@5={:.1}%, MRR={:.3} {}",
            ngram,
            hit5_mean * 100.0,
            mrr_mean,
            recommendation
        );
    }

    println!("\n🏆 Best Configuration:");
    println!("   N-gram size: {}", best_ngram);
    println!("   Score:       {:.3}", best_score);

    println!("\n💡 Train with optimal settings:");
    println!("   aprender-shell train --ngram {}", best_ngram);
}
