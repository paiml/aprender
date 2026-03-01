/// Print the training plan in human-readable text format.
fn print_plan_text(plan: &entrenar::finetune::TrainingPlan) {
    use entrenar::finetune::{CheckStatus, PlanVerdict};

    output::header("apr train plan — Training Pre-Flight");
    println!();

    // ── Verdict banner ────────────────────────────────────────────────
    print_verdict_banner(plan);

    // ── Data audit ────────────────────────────────────────────────────
    print_data_audit(plan);

    // ── Model ─────────────────────────────────────────────────────────
    print_model_section(plan);

    // ── Hyperparameters ───────────────────────────────────────────────
    print_hyperparams_section(plan);

    // ── Resource estimate ─────────────────────────────────────────────
    print_resources_section(plan);

    // ── Pre-flight checks ─────────────────────────────────────────────
    println!("{}", "PRE-FLIGHT CHECKS".white().bold());
    println!("{}", "─".repeat(60));
    for check in &plan.pre_flight {
        let icon = match check.status {
            CheckStatus::Pass => "✓".green(),
            CheckStatus::Warn => "⚠".yellow(),
            CheckStatus::Fail => "✗".red(),
        };
        println!("  {} {}: {}", icon, check.name, check.detail);
    }
    println!();

    // ── Issues ────────────────────────────────────────────────────────
    if !plan.issues.is_empty() {
        println!("{}", "ISSUES".white().bold());
        println!("{}", "─".repeat(60));
        for issue in &plan.issues {
            let icon = match issue.severity {
                CheckStatus::Fail => "!!".red().bold(),
                CheckStatus::Warn => "! ".yellow(),
                CheckStatus::Pass => "i ".blue(),
            };
            println!("  {} [{}] {}", icon, issue.category, issue.message);
            if let Some(ref fix) = issue.fix {
                println!("     {}: {}", "Fix".cyan(), fix);
            }
        }
        println!();
    }

    // ── Next steps ────────────────────────────────────────────────────
    if plan.verdict != PlanVerdict::Blocked {
        println!("{}", "NEXT STEPS".white().bold());
        println!("{}", "─".repeat(60));
        println!(
            "  To apply this plan:  {}",
            "apr train apply --plan <plan.yaml>".cyan()
        );
        println!(
            "  To save this plan:   {}",
            "apr train plan ... --format yaml > plan.yaml".dimmed()
        );
    }
}

fn print_verdict_banner(plan: &entrenar::finetune::TrainingPlan) {
    use entrenar::finetune::PlanVerdict;

    let (pass_count, warn_count, fail_count) = plan.check_counts();
    match plan.verdict {
        PlanVerdict::Ready => {
            println!(
                "  {} {} checks passed",
                "READY".green().bold(),
                pass_count
            );
        }
        PlanVerdict::WarningsPresent => {
            println!(
                "  {} {} passed, {} warnings",
                "WARNINGS".yellow().bold(),
                pass_count,
                warn_count
            );
        }
        PlanVerdict::Blocked => {
            println!(
                "  {} {} passed, {} warnings, {} failures",
                "BLOCKED".red().bold(),
                pass_count,
                warn_count,
                fail_count
            );
        }
    }
    println!();
}

fn print_data_audit(plan: &entrenar::finetune::TrainingPlan) {
    println!("{}", "DATA AUDIT".white().bold());
    println!("{}", "─".repeat(60));
    output::kv("  Source", &plan.data.train_path);
    output::kv("  Samples", plan.data.train_samples.to_string());
    output::kv(
        "  Avg input length",
        format!("{} chars", plan.data.avg_input_len),
    );

    // Class distribution
    println!();
    println!("  {}", "Class Distribution:".dimmed());
    let total = plan.data.train_samples as f64;
    for (i, &count) in plan.data.class_counts.iter().enumerate() {
        let pct = count as f64 / total * 100.0;
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    class {i}: {count:>6}  {pct:>5.1}%  {bar}");
    }
    println!(
        "    Imbalance: {:.1}:1 {}",
        plan.data.imbalance_ratio,
        if plan.data.imbalance_ratio > 5.0 {
            "⚠ SEVERE".yellow().to_string()
        } else if plan.data.imbalance_ratio > 2.0 {
            "(auto-weighted)".dimmed().to_string()
        } else {
            "OK".green().to_string()
        }
    );

    if plan.data.duplicates > 0 {
        println!(
            "    Duplicates: {} ({:.1}%)",
            plan.data.duplicates,
            plan.data.duplicates as f64 / total * 100.0
        );
    }
    if plan.data.preamble_count > 0 {
        println!(
            "    Preambles: {} ({:.0}%)",
            plan.data.preamble_count,
            plan.data.preamble_count as f64 / total * 100.0
        );
    }

    if let Some(val) = plan.data.val_samples {
        output::kv("  Val samples", val.to_string());
    }
    if let Some(test) = plan.data.test_samples {
        output::kv("  Test samples", test.to_string());
    }
    println!();
}

fn print_model_section(plan: &entrenar::finetune::TrainingPlan) {
    println!("{}", "MODEL".white().bold());
    println!("{}", "─".repeat(60));
    output::kv("  Size", &plan.model.size);
    output::kv("  Architecture", &plan.model.architecture);
    output::kv(
        "  Dimensions",
        format!("{}h × {}L", plan.model.hidden_size, plan.model.num_layers),
    );
    output::kv(
        "  LoRA trainable",
        format_params(plan.model.lora_trainable_params),
    );
    output::kv(
        "  Classifier head",
        format_params(plan.model.classifier_params),
    );
    output::kv(
        "  Total trainable",
        format_params(plan.model.lora_trainable_params + plan.model.classifier_params),
    );
    println!();
}

fn print_hyperparams_section(plan: &entrenar::finetune::TrainingPlan) {
    println!("{}", "HYPERPARAMETERS".white().bold());
    println!("{}", "─".repeat(60));
    output::kv("  Strategy", &plan.hyperparameters.strategy);

    if plan.hyperparameters.strategy == "manual" {
        if let Some(ref manual) = plan.hyperparameters.manual {
            output::kv("  Learning rate", format!("{:.2e}", manual.learning_rate));
            output::kv("  LoRA rank", manual.lora_rank.to_string());
            output::kv("  Batch size", manual.batch_size.to_string());
        }
    } else {
        output::kv("  Budget", format!("{} trials", plan.hyperparameters.budget));
        output::kv(
            "  Mode",
            if plan.hyperparameters.scout {
                "scout (1 epoch/trial)"
            } else {
                "full"
            },
        );
        output::kv(
            "  Max epochs",
            plan.hyperparameters.max_epochs.to_string(),
        );
        output::kv(
            "  Search space",
            format!("{} parameters", plan.hyperparameters.search_space_params),
        );

        if !plan.hyperparameters.sample_configs.is_empty() {
            println!();
            println!("  {}", "Sample Configurations:".dimmed());
            for t in &plan.hyperparameters.sample_configs {
                println!(
                    "    Trial {}: lr={:.2e} rank={} alpha={:.1} batch={} warmup={:.2} clip={:.1} wt={} tgt={} lr_min={:.4}",
                    t.trial, t.learning_rate, t.lora_rank, t.lora_alpha, t.batch_size,
                    t.warmup, t.gradient_clip, t.class_weights, t.target_modules, t.lr_min_ratio
                );
            }
        }
    }

    if let Some(ref rec) = plan.hyperparameters.recommendation {
        println!("  {}: {}", "Recommendation".yellow(), rec);
    }
    println!();
}

fn print_resources_section(plan: &entrenar::finetune::TrainingPlan) {
    println!("{}", "RESOURCE ESTIMATE".white().bold());
    println!("{}", "─".repeat(60));
    output::kv(
        "  VRAM",
        format!("{:.1} GB", plan.resources.estimated_vram_gb),
    );
    output::kv(
        "  Steps/epoch",
        plan.resources.steps_per_epoch.to_string(),
    );
    output::kv(
        "  Time/epoch",
        format!("{:.1} min", plan.resources.estimated_minutes_per_epoch),
    );
    output::kv(
        "  Total time",
        format_duration(plan.resources.estimated_total_minutes),
    );
    output::kv(
        "  Checkpoint size",
        format!("{:.1} MB", plan.resources.estimated_checkpoint_mb),
    );
    if let Some(ref gpu) = plan.resources.gpu_device {
        output::kv("  GPU", gpu);
    }
    println!();
}

/// Print apply results (leaderboard, best trial).
fn print_apply_result(result: &entrenar::finetune::TuneResult) {
    use entrenar::finetune::extract_trial_params;

    println!();
    println!("{}", "LEADERBOARD".white().bold());
    println!("{}", "\u{2500}".repeat(60));

    if result.trials.is_empty() {
        println!("  No trials completed.");
        return;
    }

    // Header
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}  Status",
        "Trial", "Val Loss", "Val Acc", "Train Acc", "Epochs", "Time"
    );
    println!("  {}", "\u{2500}".repeat(68));

    for trial in &result.trials {
        let time_str = if trial.time_ms > 60_000 {
            format!("{:.1}m", trial.time_ms as f64 / 60_000.0)
        } else {
            format!("{:.1}s", trial.time_ms as f64 / 1_000.0)
        };

        let is_best = trial.id == result.best_trial_id;
        let marker = if is_best { "*" } else { " " };

        println!(
            " {}{:>4}  {:>10.4}  {:>9.1}%  {:>9.1}%  {:>8}  {:>8}  {}",
            marker,
            trial.id + 1,
            trial.val_loss,
            trial.val_accuracy * 100.0,
            trial.train_accuracy * 100.0,
            trial.epochs_run,
            time_str,
            trial.status,
        );
    }

    println!();

    // Best trial details
    if let Some(best) = result.trials.iter().find(|t| t.id == result.best_trial_id) {
        println!("{}", "BEST TRIAL".green().bold());
        println!("{}", "\u{2500}".repeat(60));

        let (lr, rank, alpha, batch, warmup, clip, weights, targets, lr_min) =
            extract_trial_params(&best.config);

        output::kv("  Trial", (best.id + 1).to_string());
        output::kv("  Val loss", format!("{:.4}", best.val_loss));
        output::kv("  Val accuracy", format!("{:.1}%", best.val_accuracy * 100.0));
        output::kv("  Learning rate", format!("{lr:.2e}"));
        output::kv("  LoRA rank", rank.to_string());
        output::kv("  LoRA alpha", format!("{alpha:.1}"));
        output::kv("  Batch size", batch.to_string());
        output::kv("  Warmup", format!("{warmup:.2}"));
        output::kv("  Gradient clip", format!("{clip:.1}"));
        output::kv("  Class weights", &weights);
        output::kv("  Target modules", &targets);
        output::kv("  LR min ratio", format!("{lr_min:.4}"));
        println!();
    }

    // Summary
    let total_time = if result.total_time_ms > 3_600_000 {
        format!("{:.1} hours", result.total_time_ms as f64 / 3_600_000.0)
    } else if result.total_time_ms > 60_000 {
        format!("{:.1} min", result.total_time_ms as f64 / 60_000.0)
    } else {
        format!("{:.1} sec", result.total_time_ms as f64 / 1_000.0)
    };

    println!("{}", "SUMMARY".white().bold());
    println!("{}", "\u{2500}".repeat(60));
    output::kv("  Strategy", &result.strategy);
    output::kv("  Mode", &result.mode);
    output::kv("  Trials completed", result.trials.len().to_string());
    output::kv("  Total time", total_time);
    println!();

    // Next steps
    println!("{}", "NEXT STEPS".white().bold());
    println!("{}", "\u{2500}".repeat(60));
    println!(
        "  Leaderboard saved: {}",
        "leaderboard.json".cyan()
    );
    println!(
        "  Diagnose best:     {}",
        "apr diagnose <output>/trial_XXX/best/ --data <test.jsonl>".cyan()
    );
    println!(
        "  Evaluate best:     {}",
        "apr eval <output>/trial_XXX/best/ --task classify --data <test.jsonl>".cyan()
    );
}
