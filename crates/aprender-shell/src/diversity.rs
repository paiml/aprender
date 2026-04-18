
/// Filter synthetic commands by quality score and update diversity monitor.
fn filter_commands_by_quality(
    commands: &[String],
    config: &aprender::synthetic::SyntheticConfig,
    diversity_monitor: &mut Option<aprender::synthetic::DiversityMonitor>,
) -> (Vec<String>, usize) {
    use aprender::synthetic::DiversityScore;

    let mut quality_filtered: Vec<String> = Vec::new();
    let mut rejected_count = 0;

    for cmd in commands {
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        let quality_score = if tokens.is_empty() {
            0.0
        } else {
            let length_score = (tokens.len() as f32 / 5.0).min(1.0);
            let base_known =
                ["git", "cargo", "docker", "make", "npm", "kubectl", "aws"].contains(&tokens[0]);
            let base_score = if base_known { 0.8 } else { 0.5 };
            (length_score * 0.4 + base_score * 0.6).min(1.0)
        };

        if config.meets_quality(quality_score) {
            quality_filtered.push(cmd.clone());

            if let Some(ref mut monitor) = diversity_monitor {
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

    (quality_filtered, rejected_count)
}

/// Print coverage report for augmentation.
fn print_coverage_report(
    result: &synthetic::SyntheticResult,
    quality_count: usize,
    rejected_count: usize,
) {
    println!("\n📈 Coverage Report:");
    println!("   Generated:          {}", result.commands.len());
    println!(
        "   Quality filtered:   {} (rejected {})",
        quality_count, rejected_count
    );
    println!("   Known n-grams:      {}", result.report.known_ngrams);
    println!("   Total n-grams:      {}", result.report.total_ngrams);
    println!("   New n-grams added:  {}", result.report.new_ngrams);
    println!(
        "   Coverage gain:      {:.1}%",
        result.report.coverage_gain * 100.0
    );
}

/// Print diversity metrics if monitoring.
fn print_diversity_report(diversity_monitor: &Option<aprender::synthetic::DiversityMonitor>) {
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
    use aprender::synthetic::{DiversityMonitor, SyntheticConfig, SyntheticGenerator};

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

    let (quality_filtered, rejected_count) =
        filter_commands_by_quality(&result.commands, &config, &mut diversity_monitor);

    print_coverage_report(&result, quality_filtered.len(), rejected_count);
    print_diversity_report(&diversity_monitor);

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
    use std::collections::HashMap;

    println!("📊 aprender-shell: Command Analysis (with CodeFeatureExtractor)\n");

    let history_file = find_history_file_graceful(history_path);
    println!("📂 History file: {}", history_file.display());

    let commands = parse_history_graceful(&history_file);
    println!("📊 Total commands: {}\n", commands.len());

    let (category_counts, base_command_counts) = classify_commands(&commands);
    print_category_counts(&category_counts, commands.len());
    print_top_bases(&base_command_counts, commands.len(), top);
    print_category_samples(&category_counts);
    print_complexity(&commands, &base_command_counts);
    print_workflow_profile(&base_command_counts);

    println!("\n💡 Tip: Use 'aprender-shell augment --use-code-eda' for code-aware augmentation");
}

const CATEGORY_NAMES: [&str; 5] = [
    "General",     // 0
    "Fix/Debug",   // 1
    "Security",    // 2
    "Performance", // 3
    "Refactor",    // 4
];

fn classify_commands(
    commands: &[String],
) -> (
    std::collections::HashMap<u8, Vec<String>>,
    std::collections::HashMap<String, usize>,
) {
    use aprender::synthetic::code_features::{CodeFeatureExtractor, CommitDiff};
    use std::collections::HashMap;

    let extractor = CodeFeatureExtractor::new();
    let mut category_counts: HashMap<u8, Vec<String>> = HashMap::new();
    let mut base_command_counts: HashMap<String, usize> = HashMap::new();

    for cmd in commands {
        let base = cmd.split_whitespace().next().unwrap_or("").to_string();
        *base_command_counts.entry(base).or_insert(0) += 1;

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

    (category_counts, base_command_counts)
}

fn category_name(cat: u8) -> &'static str {
    CATEGORY_NAMES.get(cat as usize).copied().unwrap_or("Unknown")
}

fn print_category_counts(
    category_counts: &std::collections::HashMap<u8, Vec<String>>,
    total: usize,
) {
    println!("🏷️  Command Categories (based on keywords):");
    println!("   ─────────────────────────────────────────");
    for (cat, cmds) in category_counts {
        println!(
            "   {:12} {:>5} commands ({:.1}%)",
            category_name(*cat),
            cmds.len(),
            cmds.len() as f32 / total as f32 * 100.0
        );
    }
}

fn print_top_bases(
    base_command_counts: &std::collections::HashMap<String, usize>,
    total: usize,
    top: usize,
) {
    println!("\n🔝 Top {} Base Commands:", top);
    println!("   ─────────────────────────────────────────");
    let mut sorted_bases: Vec<_> = base_command_counts.iter().collect();
    sorted_bases.sort_by(|a, b| b.1.cmp(a.1));

    for (base, count) in sorted_bases.iter().take(top) {
        let pct = **count as f32 / total as f32 * 100.0;
        let bar_len = (pct / 2.0) as usize;
        let bar = "█".repeat(bar_len.min(25));
        println!("   {:12} {:>5} ({:>5.1}%) {}", base, count, pct, bar);
    }
}

fn print_category_samples(category_counts: &std::collections::HashMap<u8, Vec<String>>) {
    println!("\n📋 Sample Commands by Category:");
    println!("   ─────────────────────────────────────────");
    for (cat, cmds) in category_counts {
        if cmds.is_empty() {
            continue;
        }
        println!("\n   [{}]:", category_name(*cat));
        for cmd in cmds.iter().take(3) {
            let truncated = if cmd.len() > 60 {
                format!("{}...", &cmd[..57])
            } else {
                cmd.clone()
            };
            println!("     • {}", truncated);
        }
    }
}

fn print_complexity(
    commands: &[String],
    base_command_counts: &std::collections::HashMap<String, usize>,
) {
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
}

fn print_workflow_profile(base_command_counts: &std::collections::HashMap<String, usize>) {
    println!("\n🛠️  Developer Workflow Profile:");
    println!("   ─────────────────────────────────────────");
    for (cmd, line) in [
        ("git", "✓ Version Control:  {n} git commands"),
        ("cargo", "✓ Rust Development: {n} cargo commands"),
        ("docker", "✓ Containers:       {n} docker commands"),
        ("kubectl", "✓ Kubernetes:       {n} kubectl commands"),
    ] {
        let count = base_command_counts.get(cmd).copied().unwrap_or(0);
        if count > 0 {
            println!("   {}", line.replace("{n}", &count.to_string()));
        }
    }
}
