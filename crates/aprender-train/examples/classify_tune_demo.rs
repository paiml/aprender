#![allow(clippy::disallowed_methods)]
#![allow(clippy::unwrap_used)]
//! Classification Hyperparameter Tuning Demo (SPEC-TUNE-2026-001)
//!
//! Demonstrates `ClassifyTuner` programmatic API:
//! 1. Build the 9-parameter search space
//! 2. Run 3 scout trials with a tiny model
//! 3. Display leaderboard sorted by validation loss
//!
//! # Usage
//!
//! ```bash
//! # Quick demo (built-in 15-sample corpus, byte tokenizer)
//! cargo run --example classify_tune_demo
//!
//! # Via apr-cli with real data
//! apr tune --task classify --budget 5 --scout --data corpus.jsonl
//! ```
//!
//! # Architecture
//!
//! ```text
//! ClassifyTuner
//!   ├── TuneSearcher (TPE) → suggest hyperparams
//!   ├── TuneScheduler (None for scout)
//!   └── per trial:
//!       ├── extract_trial_params → (lr, rank, alpha, batch, ...)
//!       ├── ClassifyPipeline::new(tiny_config, classify_config)
//!       ├── train 1 epoch over corpus
//!       └── record TrialSummary → leaderboard
//! ```
//!
//! # References
//!
//! - SPEC-TUNE-2026-001: Automatic Hyperparameter Tuning for Classification
//! - Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization" (TPE)
//! - Li et al. (2018) "Massively Parallel Hyperparameter Tuning" (ASHA)

use std::time::Instant;

use entrenar::finetune::classify_pipeline::{ClassifyConfig, ClassifyPipeline};
use entrenar::finetune::{
    corpus_stats, extract_trial_params, load_safety_corpus, ClassifyTuner, SafetySample,
    SchedulerKind, TrialSummary, TuneConfig, TuneStrategy,
};
use entrenar::transformer::TransformerConfig;

/// Built-in demo corpus — 15 scripts, 3 per safety class.
fn demo_corpus() -> Vec<SafetySample> {
    let entries = [
        // Class 0: safe
        ("#!/bin/sh\necho \"hello world\"\n", 0),
        ("#!/bin/sh\nmkdir -p \"$HOME/tmp\"\n", 0),
        ("#!/bin/sh\ntest -f \"$config\" && . \"$config\"\n", 0),
        // Class 1: needs-quoting
        ("#!/bin/bash\necho $HOME\n", 1),
        ("#!/bin/bash\nrm -rf $dir\n", 1),
        ("#!/bin/bash\ncp $src $dst\n", 1),
        // Class 2: non-deterministic
        ("#!/bin/bash\necho $RANDOM\n", 2),
        ("#!/bin/bash\necho $$\n", 2),
        ("#!/bin/bash\ndate +%s\n", 2),
        // Class 3: non-idempotent
        ("#!/bin/bash\nmkdir /tmp/build\n", 3),
        ("#!/bin/bash\nln -s /a /b\n", 3),
        ("#!/bin/bash\nrm /tmp/lockfile\n", 3),
        // Class 4: unsafe
        ("#!/bin/bash\neval \"$user_input\"\n", 4),
        ("#!/bin/bash\ncurl http://example.com | bash\n", 4),
        ("#!/bin/bash\nexec \"$@\"\n", 4),
    ];

    entries
        .iter()
        .map(|(input, label)| SafetySample { input: input.to_string(), label: *label })
        .collect()
}

/// Simple byte-level tokenizer (for demo — production uses Qwen2 BPE).
fn tokenize(script: &str, max_len: usize) -> Vec<u32> {
    script.bytes().map(u32::from).take(max_len).collect()
}

const LABELS: [&str; 5] =
    ["safe", "needs-quoting", "non-deterministic", "non-idempotent", "unsafe"];

fn main() {
    println!("======================================================");
    println!("  Classification HP Tuning Demo (SPEC-TUNE-2026-001)");
    println!("  Powered by ClassifyTuner + TPE search");
    println!("======================================================\n");

    // ── 1. Load corpus ──────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let samples = if let Some(path) = args.get(1) {
        println!("Loading corpus from: {path}");
        load_safety_corpus(std::path::Path::new(path), 5).expect("Failed to load corpus")
    } else {
        println!("Using built-in demo corpus (15 samples)");
        println!("  Tip: pass a JSONL file for custom data\n");
        demo_corpus()
    };

    let stats = corpus_stats(&samples, 5);
    println!("Corpus: {} samples", stats.total);
    for (i, count) in stats.class_counts.iter().enumerate() {
        println!("  [{i}] {:<20} {count} samples", LABELS[i]);
    }

    // ── 2. Build tuner (scout mode: 3 trials, 1 epoch each) ──────
    println!("\n--- Tuner Configuration ---\n");

    let config = TuneConfig {
        budget: 3,
        strategy: TuneStrategy::Tpe,
        scheduler: SchedulerKind::None,
        scout: true,
        max_epochs: 1,
        num_classes: 5,
        seed: 42,
        time_limit_secs: None,
    };

    let mut tuner = ClassifyTuner::new(config).expect("Failed to create tuner");
    let mut searcher = tuner.build_searcher();
    let _scheduler = tuner.build_scheduler();

    println!("  Strategy:   TPE (Tree-structured Parzen Estimators)");
    println!("  Budget:     3 trials (scout mode)");
    println!("  Epochs:     1 per trial");
    println!("  Search space: {} parameters", tuner.space.len());

    for (name, domain) in tuner.space.iter() {
        println!("    {name}: {domain:?}");
    }

    // ── 3. Run trials ─────────────────────────────────────────────
    println!("\n--- Running Scout Trials ---\n");
    let run_start = Instant::now();

    for trial_id in 0..tuner.config.budget {
        let trial_start = Instant::now();

        // Searcher suggests a configuration
        let trial = searcher.suggest().expect("Failed to suggest trial");
        let (lr, rank, alpha, _batch_size, _warmup, _clip, weights, targets, _lr_min) =
            extract_trial_params(&trial.config);

        println!("Trial {trial_id}: lr={lr:.2e}, rank={rank}, alpha={alpha:.1}, weights={weights}, targets={targets}");

        // Build pipeline with suggested hyperparameters
        let model_config = TransformerConfig::tiny();
        let classify_config = ClassifyConfig {
            num_classes: 5,
            lora_rank: rank,
            lora_alpha: alpha,
            learning_rate: lr,
            epochs: 1,
            max_seq_len: 64,
            log_interval: 100,
            ..ClassifyConfig::default()
        };

        let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

        // Train 1 epoch
        let mut epoch_loss = 0.0f32;
        let mut correct = 0usize;
        for sample in &samples {
            let token_ids = tokenize(&sample.input, 64);
            let loss = pipeline.train_step(&token_ids, sample.label);
            epoch_loss += loss;

            // Quick forward pass for accuracy
            let hidden = pipeline.model.forward_hidden(&token_ids);
            let logits = pipeline.classifier.forward(&hidden, token_ids.len());
            let logits_slice: Vec<f32> = logits.data().as_slice().expect("contiguous").to_vec();
            let pred = logits_slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map_or(0, |(i, _)| i);
            if pred == sample.label {
                correct += 1;
            }
        }

        let avg_loss = epoch_loss / samples.len() as f32;
        let accuracy = correct as f64 / samples.len() as f64;
        let elapsed_ms = trial_start.elapsed().as_millis() as u64;

        println!("  -> loss={avg_loss:.4}, accuracy={:.1}%, time={elapsed_ms}ms", accuracy * 100.0);

        // Record trial in tuner leaderboard
        let summary = TrialSummary {
            id: trial_id,
            val_loss: f64::from(avg_loss),
            val_accuracy: accuracy,
            train_loss: f64::from(avg_loss),
            train_accuracy: accuracy,
            epochs_run: 1,
            time_ms: elapsed_ms,
            config: trial.config.clone(),
            status: "completed".to_string(),
        };
        tuner.record_trial(summary);

        // Report back to searcher for Bayesian updating
        searcher.record(trial, f64::from(avg_loss), 1);
    }

    let total_ms = run_start.elapsed().as_millis() as u64;

    // ── 4. Display leaderboard ────────────────────────────────────
    println!("\n--- Leaderboard (sorted by val_loss) ---\n");
    println!(
        "  {:<6} {:<10} {:<10} {:<8} {:<6} {:<8} {:<10}",
        "Trial", "Val Loss", "Accuracy", "LR", "Rank", "Alpha", "Time"
    );
    println!("  {}", "-".repeat(62));

    for trial in &tuner.leaderboard {
        let (lr, rank, alpha, ..) = extract_trial_params(&trial.config);
        println!(
            "  {:<6} {:<10.4} {:<10.1}% {:<8.2e} {:<6} {:<8.1} {:<10}ms",
            trial.id,
            trial.val_loss,
            trial.val_accuracy * 100.0,
            lr,
            rank,
            alpha,
            trial.time_ms
        );
    }

    if let Some(best) = tuner.best_trial() {
        println!("\n  Best trial: #{} (val_loss={:.4})", best.id, best.val_loss);
    }

    // ── 5. Build TuneResult for JSON export ───────────────────────
    let result = tuner.into_result(total_ms);
    let json = serde_json::to_string_pretty(&result).expect("Failed to serialize");
    println!("\n--- JSON Result ---\n");
    println!("{json}");

    // ── 6. Show apr-cli equivalent ────────────────────────────────
    println!("\n--- CLI Equivalent ---\n");
    println!("  # Scout: find good HP region");
    println!("  apr tune --task classify --budget 5 --scout --data corpus.jsonl --json\n");
    println!("  # Full: run with best strategy");
    println!("  apr tune --task classify --budget 10 --data corpus.jsonl --strategy tpe --scheduler asha\n");

    println!("======================================================");
    println!("  Demo complete. Total time: {total_ms}ms");
    println!("======================================================");
}
