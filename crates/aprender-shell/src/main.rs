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

        /// Augmentation ratio (synthetic/original, e.g., 0.5 = 50% more data)
        #[arg(short = 'a', long, default_value = "0.5")]
        augmentation_ratio: f32,

        /// Minimum quality threshold (0.0-1.0)
        #[arg(short, long, default_value = "0.7")]
        quality_threshold: f32,

        /// Enable diversity monitoring
        #[arg(long)]
        monitor_diversity: bool,
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
            augmentation_ratio,
            quality_threshold,
            monitor_diversity,
        } => {
            cmd_augment(
                history,
                &output,
                ngram,
                augmentation_ratio,
                quality_threshold,
                monitor_diversity,
            );
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

fn cmd_augment(
    history_path: Option<PathBuf>,
    output: &str,
    ngram: usize,
    augmentation_ratio: f32,
    quality_threshold: f32,
    monitor_diversity: bool,
) {
    use aprender::synthetic::{DiversityMonitor, DiversityScore, SyntheticConfig};

    println!("🧬 aprender-shell: Data Augmentation (with aprender synthetic)\n");

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
    let pipeline = SyntheticPipeline::new();
    let result = pipeline.generate(&commands, known_ngrams, target_count);
    println!("done!");

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
