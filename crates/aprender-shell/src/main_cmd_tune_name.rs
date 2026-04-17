
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

/// Load a MarkovModel, optionally with encryption.
fn load_inspect_model(path: &std::path::Path, password: Option<&str>) -> MarkovModel {
    let encrypted = MarkovModel::is_encrypted(path).unwrap_or(false);

    if let Some(pwd) = password {
        match MarkovModel::load_encrypted(path, pwd) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("❌ Failed to load encrypted model: {e}");
                std::process::exit(1);
            }
        }
    } else {
        match MarkovModel::load(path) {
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
    }
}

/// Render a model card in text format to stdout.
fn render_model_card_text(card: &aprender::format::model_card::ModelCard, path: &std::path::Path) {
    print_card_header(card, path);
    print_card_training_data(card);
    print_card_hyperparameters(card);
    print_card_metrics(card);
    print_card_description(card);
    print_card_footer();
}

fn print_card_header(
    card: &aprender::format::model_card::ModelCard,
    path: &std::path::Path,
) {
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
}

fn print_card_training_data(card: &aprender::format::model_card::ModelCard) {
    let Some(ref training) = card.training_data else {
        return;
    };
    println!("\n📊 Training Data:");
    println!("  Source:  {}", training.name);
    if let Some(samples) = training.samples {
        println!("  Samples: {}", samples);
    }
    if let Some(ref hash) = training.hash {
        println!("  Hash:    {}", hash);
    }
}

fn print_card_hyperparameters(card: &aprender::format::model_card::ModelCard) {
    if card.hyperparameters.is_empty() {
        return;
    }
    println!("\n⚙️  Hyperparameters:");
    for (key, value) in &card.hyperparameters {
        println!("  {}: {}", key, value);
    }
}

fn print_card_metrics(card: &aprender::format::model_card::ModelCard) {
    if card.metrics.is_empty() {
        return;
    }
    println!("\n📈 Metrics:");
    for (key, value) in &card.metrics {
        println!("  {}: {}", key, value);
    }
}

fn print_card_description(card: &aprender::format::model_card::ModelCard) {
    let Some(ref desc) = card.description else {
        return;
    };
    println!("\n📝 Description:");
    println!("  {}", desc);
}

fn print_card_footer() {
    println!("\n💡 Export formats:");
    println!("   JSON:        aprender-shell inspect --format json");
    println!("   Hugging Face: aprender-shell inspect --format huggingface");
}

fn cmd_inspect(model_path: &str, format: &str, use_password: bool) {
    use aprender::format::model_card::{ModelCard, TrainingDataInfo};

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

    let model = load_inspect_model(&path, password.as_deref());

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
        _ => render_model_card_text(&card, &path),
    }
}

/// Simple date string without chrono dependency
fn chrono_lite_date() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let days = (secs / 86400) as i64;
    let (year, after_year) = year_and_day_of_year(days);
    let (month, day) = month_and_day(after_year, is_leap_year(year));
    format!("{year:04}{month:02}{day:02}")
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_in_year(year: i32) -> i64 {
    if is_leap_year(year) { 366 } else { 365 }
}

fn year_and_day_of_year(mut remaining: i64) -> (i32, i64) {
    let mut year = 1970i32;
    while remaining >= days_in_year(year) {
        remaining -= days_in_year(year);
        year += 1;
    }
    (year, remaining)
}

fn month_and_day(mut remaining: i64, leap: bool) -> (u32, u32) {
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
    (month, remaining as u32 + 1)
}
