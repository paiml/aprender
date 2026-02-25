#![allow(clippy::disallowed_methods)]
//! Shell Safety Classifier Training
//!
//! Trains an MLP classifier to predict shell script safety using the
//! bashrs corpus (17,942 entries) merged with adversarial data (~8,000
//! entries for minority classes).
//!
//! The model classifies scripts into 5 safety categories:
//!
//! - `safe`: passes all checks (lint, deterministic, idempotent)
//! - `needs-quoting`: variable quoting issues
//! - `non-deterministic`: contains $RANDOM, $$, timestamps
//! - `non-idempotent`: missing -p/-f flags
//! - `unsafe`: security rule violations
//!
//! # Usage
//!
//! ```bash
//! # Export corpus + generate adversarial data
//! cd /path/to/bashrs
//! cargo run -- corpus export-dataset --format classification -o /tmp/corpus.jsonl
//! cargo run -- generate-adversarial --verify -o /tmp/adversarial.jsonl
//! { cat /tmp/corpus.jsonl; echo; cat /tmp/adversarial.jsonl; } > /tmp/combined.jsonl
//!
//! # Train the model
//! cd /path/to/aprender
//! cargo run --example shell_safety_training -- /tmp/combined.jsonl
//! ```
//!
//! # Architecture
//!
//! - Tokenization via ShellVocabulary (512 tokens, max_seq_len=64)
//! - MLP: Linear(64→256)→ReLU→Linear(256→128)→ReLU→Linear(128→5)
//! - CrossEntropyLoss with inverse-frequency class weighting
//! - AdamW optimizer with stratified train/val split

use aprender::autograd::Tensor;
use aprender::model_selection::StratifiedKFold;
use aprender::nn::{
    loss::CrossEntropyLoss,
    optim::{Adam, Optimizer},
    serialize::save_model,
    Linear, Module, ReLU, Sequential,
};
use aprender::primitives::Vector;
use aprender::text::shell_vocab::{SafetyClass, ShellVocabulary};

use std::io::BufRead;

/// A single training sample parsed from bashrs corpus JSONL.
struct CorpusSample {
    #[allow(dead_code)]
    id: String,
    input: String,
    label: usize,
}

/// Configuration for the shell safety model.
struct ModelConfig {
    vocab_size: usize,
    embed_dim: usize,
    hidden_dim: usize,
    num_classes: usize,
    max_seq_len: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 512,
            embed_dim: 64,
            hidden_dim: 256,
            num_classes: SafetyClass::num_classes(),
            max_seq_len: 64,
        }
    }
}

/// Training hyperparameters.
struct TrainConfig {
    epochs: usize,
    lr: f32,
    batch_size: usize,
    seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            lr: 0.001,
            batch_size: 256,
            seed: 42,
        }
    }
}

fn main() {
    println!("======================================================");
    println!("  Shell Safety Classifier Training (v2)");
    println!("  Powered by aprender (pure Rust ML)");
    println!("  Stratified split + class-weighted loss");
    println!("======================================================\n");

    // Parse command-line args
    let args: Vec<String> = std::env::args().collect();
    let input_path = args.get(1).map(String::as_str);

    // Load training data
    let samples = match input_path {
        Some(path) => {
            println!("Loading corpus from: {path}");
            load_jsonl(path)
        }
        None => {
            println!("No JSONL file provided. Using built-in demo data.");
            println!("For full training: cargo run --example shell_safety_training -- /tmp/combined.jsonl\n");
            load_demo_data()
        }
    };

    println!("Loaded {} samples", samples.len());

    let (class_counts, class_weights) = compute_class_stats(&samples);
    print_class_distribution(&class_counts, samples.len());
    print_class_weights(&class_weights);

    // Build vocabulary
    let vocab = ShellVocabulary::new();
    println!("\nVocabulary size: {}", vocab.vocab_size());

    let config = ModelConfig {
        vocab_size: vocab.vocab_size() + 1, // +1 for safety
        ..ModelConfig::default()
    };

    let train_config = TrainConfig::default();

    // Tokenize all samples
    println!(
        "Tokenizing {} samples (max_seq_len={})...",
        samples.len(),
        config.max_seq_len
    );
    let encoded: Vec<Vec<usize>> = samples
        .iter()
        .map(|s| vocab.encode(&s.input, config.max_seq_len))
        .collect();

    // Stratified train/val split using StratifiedKFold
    // Use 5 folds: fold 0 = val (20%), folds 1-4 = train (80%)
    let labels_vec: Vec<f32> = samples.iter().map(|s| s.label as f32).collect();
    let labels = Vector::from_vec(labels_vec);
    let skfold = StratifiedKFold::new(5).with_random_state(train_config.seed);
    let splits = skfold.split(&labels);

    // Take first fold split: (train_indices, val_indices)
    let (train_indices, val_indices) = &splits[0];

    // Build train/val sets from indices
    let train_encoded: Vec<Vec<usize>> = train_indices.iter().map(|&i| encoded[i].clone()).collect();
    let train_labels: Vec<usize> = train_indices.iter().map(|&i| samples[i].label).collect();
    let val_encoded: Vec<Vec<usize>> = val_indices.iter().map(|&i| encoded[i].clone()).collect();
    let val_labels: Vec<usize> = val_indices.iter().map(|&i| samples[i].label).collect();

    // Verify stratification
    println!("\nStratified split:");
    println!("  Train: {} samples", train_encoded.len());
    println!("  Val:   {} samples", val_encoded.len());
    print_split_distribution(&train_labels, &val_labels);

    // Build model
    let input_dim = config.max_seq_len;
    let mut model = build_classifier(input_dim, config.hidden_dim, config.num_classes);

    println!("\nModel architecture:");
    println!(
        "  Input: {} (normalized token features per position)",
        input_dim
    );
    println!("  Hidden: {} → {}", config.hidden_dim, config.hidden_dim / 2);
    println!("  Output: {} classes", config.num_classes);

    // Prepare full feature tensors for accuracy evaluation
    let train_x = prepare_features(&train_encoded, config.max_seq_len, config.vocab_size);
    let val_x = prepare_features(&val_encoded, config.max_seq_len, config.vocab_size);

    // Compute per-sample weights for training set
    let sample_weights: Vec<f32> = train_labels.iter().map(|&l| class_weights[l]).collect();

    println!("\nTraining configuration:");
    println!("  Epochs: {}", train_config.epochs);
    println!("  Learning rate: {}", train_config.lr);
    println!("  Batch size: {}", train_config.batch_size);
    println!("  Loss: CrossEntropyLoss (class-weighted)");
    println!("  Optimizer: Adam");
    println!("  Split: Stratified 80/20\n");

    // Run training loop
    train_loop(
        &mut model,
        &train_config,
        &train_encoded,
        &train_labels,
        &sample_weights,
        &config,
        &train_x,
        &val_x,
        &val_labels,
    );

    // Final per-class accuracy on validation set
    println!("\n  Per-class validation accuracy:");
    print_per_class_accuracy(&model, &val_x, &val_labels);

    // Save artifacts
    save_artifacts(&model, &vocab, &config, &class_weights, samples.len());
}

/// Count per-class samples and compute inverse-frequency weights.
fn compute_class_stats(samples: &[CorpusSample]) -> ([usize; 5], Vec<f32>) {
    let mut class_counts = [0usize; 5];
    for sample in samples {
        if sample.label < 5 {
            class_counts[sample.label] += 1;
        }
    }

    let total = samples.len() as f32;
    let num_classes = 5usize;
    let class_weights: Vec<f32> = class_counts
        .iter()
        .map(|&c| {
            if c > 0 {
                total / (num_classes as f32 * c as f32)
            } else {
                1.0
            }
        })
        .collect();

    (class_counts, class_weights)
}

/// Print class distribution table.
fn print_class_distribution(class_counts: &[usize; 5], total: usize) {
    println!("\nClass distribution:");
    for (i, count) in class_counts.iter().enumerate() {
        if let Some(cls) = SafetyClass::from_index(i) {
            let pct = *count as f64 / total as f64 * 100.0;
            println!("  {}: {} samples ({:.1}%)", cls.label(), count, pct);
        }
    }
}

/// Print class weights.
fn print_class_weights(class_weights: &[f32]) {
    println!("\nClass weights (inverse frequency):");
    for (i, w) in class_weights.iter().enumerate() {
        if let Some(cls) = SafetyClass::from_index(i) {
            println!("  {}: {:.3}", cls.label(), w);
        }
    }
}

/// Print per-class distribution for train/val split.
fn print_split_distribution(train_labels: &[usize], val_labels: &[usize]) {
    let mut train_counts = [0usize; 5];
    let mut val_counts = [0usize; 5];
    for &l in train_labels {
        if l < 5 {
            train_counts[l] += 1;
        }
    }
    for &l in val_labels {
        if l < 5 {
            val_counts[l] += 1;
        }
    }
    println!("\n  Train class distribution:");
    for (i, count) in train_counts.iter().enumerate() {
        if let Some(cls) = SafetyClass::from_index(i) {
            let pct = *count as f64 / train_labels.len() as f64 * 100.0;
            println!("    {}: {} ({:.1}%)", cls.label(), count, pct);
        }
    }
    println!("  Val class distribution:");
    for (i, count) in val_counts.iter().enumerate() {
        if let Some(cls) = SafetyClass::from_index(i) {
            let pct = *count as f64 / val_labels.len() as f64 * 100.0;
            println!("    {}: {} ({:.1}%)", cls.label(), count, pct);
        }
    }
}

/// Build a wider MLP classifier for better minority class learning.
fn build_classifier(input_dim: usize, hidden_dim: usize, num_classes: usize) -> Sequential {
    Sequential::new()
        .add(Linear::with_seed(input_dim, hidden_dim, Some(42)))
        .add(ReLU::new())
        .add(Linear::with_seed(hidden_dim, hidden_dim / 2, Some(43)))
        .add(ReLU::new())
        .add(Linear::with_seed(hidden_dim / 2, num_classes, Some(44)))
}

/// Run the mini-batch training loop with class-weighted loss.
fn train_loop(
    model: &mut Sequential,
    train_config: &TrainConfig,
    train_encoded: &[Vec<usize>],
    train_labels: &[usize],
    sample_weights: &[f32],
    config: &ModelConfig,
    train_x: &Tensor,
    val_x: &Tensor,
    val_labels: &[usize],
) {
    let loss_fn = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(model.parameters_mut(), train_config.lr);

    let n_train = train_encoded.len();
    let batch_size = train_config.batch_size.min(n_train);

    println!("  Epoch    Loss       Train Acc   Val Acc");
    println!("  ------------------------------------------------");

    let mut batch_order: Vec<usize> = (0..n_train).collect();

    for epoch in 0..train_config.epochs {
        rotate_indices(&mut batch_order, epoch);

        let (epoch_loss, n_batches) = run_epoch_batches(
            model,
            &loss_fn,
            &mut optimizer,
            &batch_order,
            train_encoded,
            train_labels,
            sample_weights,
            config,
            batch_size,
        );

        if epoch % 10 == 0 || epoch == train_config.epochs - 1 {
            let avg_loss = epoch_loss / n_batches as f32;
            let train_acc = compute_accuracy_from_labels(model, train_x, train_labels);
            let val_acc = compute_accuracy_from_labels(model, val_x, val_labels);

            println!(
                "  {:>5}    {:.6}   {:.1}%        {:.1}%",
                epoch,
                avg_loss,
                train_acc * 100.0,
                val_acc * 100.0,
            );
        }
    }
}

/// Run all mini-batches for a single epoch, returning (total_loss, num_batches).
fn run_epoch_batches(
    model: &mut Sequential,
    loss_fn: &CrossEntropyLoss,
    optimizer: &mut Adam,
    batch_order: &[usize],
    train_encoded: &[Vec<usize>],
    train_labels: &[usize],
    sample_weights: &[f32],
    config: &ModelConfig,
    batch_size: usize,
) -> (f32, usize) {
    let n_train = batch_order.len();
    let mut epoch_loss = 0.0;
    let mut n_batches = 0;
    let mut offset = 0;

    while offset < n_train {
        let end = (offset + batch_size).min(n_train);
        let batch_indices: Vec<usize> = batch_order[offset..end].to_vec();

        let batch_encoded: Vec<Vec<usize>> = batch_indices
            .iter()
            .map(|&i| train_encoded[i].clone())
            .collect();
        let batch_labels: Vec<usize> = batch_indices
            .iter()
            .map(|&i| train_labels[i])
            .collect();
        let batch_weights: Vec<f32> = batch_indices
            .iter()
            .map(|&i| sample_weights[i])
            .collect();

        let batch_x = prepare_features(&batch_encoded, config.max_seq_len, config.vocab_size);
        let batch_y = prepare_labels_from_vec(&batch_labels);

        let logits = model.forward(&batch_x);
        let base_loss = loss_fn.forward(&logits, &batch_y);

        let avg_weight: f32 = batch_weights.iter().sum::<f32>() / batch_weights.len() as f32;
        let weighted_loss = base_loss.mul_scalar(avg_weight);

        let loss_val = weighted_loss.data()[0];
        epoch_loss += loss_val;
        n_batches += 1;

        weighted_loss.backward();

        {
            let mut params = model.parameters_mut();
            optimizer.step_with_params(&mut params);
        }
        optimizer.zero_grad();

        offset = end;
    }

    (epoch_loss, n_batches)
}

/// Save model weights, vocabulary, and config to disk.
fn save_artifacts(
    model: &Sequential,
    vocab: &ShellVocabulary,
    config: &ModelConfig,
    class_weights: &[f32],
    num_samples: usize,
) {
    let output_dir = "/tmp/shell-safety-model";
    std::fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let model_path = format!("{output_dir}/model.safetensors");
    save_model(model, &model_path).expect("Failed to save model");
    println!("\nModel saved to: {model_path}");

    let vocab_path = format!("{output_dir}/vocab.json");
    let vocab_json = vocab.to_json().expect("Failed to serialize vocabulary");
    std::fs::write(&vocab_path, vocab_json).expect("Failed to write vocab.json");
    println!("Vocabulary saved to: {vocab_path}");

    let config_json = serde_json::json!({
        "model_type": "shell-safety-classifier",
        "version": "2.0",
        "vocab_size": config.vocab_size,
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "num_classes": config.num_classes,
        "max_seq_len": config.max_seq_len,
        "labels": SafetyClass::all().iter().map(|c| c.label()).collect::<Vec<_>>(),
        "framework": "aprender",
        "training_data": "bashrs-corpus+adversarial",
        "training_samples": num_samples,
        "class_weights": class_weights,
    });
    let config_path = format!("{output_dir}/config.json");
    std::fs::write(
        &config_path,
        serde_json::to_string_pretty(&config_json).expect("JSON"),
    )
    .expect("Failed to write config.json");
    println!("Config saved to: {config_path}");

    println!("\n======================================================");
    println!("  Training Complete!");
    println!("  Model artifacts in: {output_dir}/");
    println!("    - model.safetensors  (weights)");
    println!("    - vocab.json         (tokenizer)");
    println!("    - config.json        (architecture)");
    println!("======================================================");
}

/// Convert tokenized sequences into feature tensors.
///
/// Normalizes token IDs to [0, 1] range per position.
fn prepare_features(encoded: &[Vec<usize>], max_seq_len: usize, vocab_size: usize) -> Tensor {
    let batch_size = encoded.len();
    let mut data = Vec::with_capacity(batch_size * max_seq_len);

    for seq in encoded {
        for i in 0..max_seq_len {
            let token_id = seq.get(i).copied().unwrap_or(0);
            // Normalize to [0, 1]
            data.push(token_id as f32 / vocab_size as f32);
        }
    }

    Tensor::new(&data, &[batch_size, max_seq_len])
}

/// Prepare label tensor from a label vector.
fn prepare_labels_from_vec(labels: &[usize]) -> Tensor {
    let label_data: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
    Tensor::new(&label_data, &[labels.len()])
}

/// Compute classification accuracy from label indices.
fn compute_accuracy_from_labels(model: &Sequential, x: &Tensor, labels: &[usize]) -> f32 {
    let logits = model.forward(x);
    let batch_size = logits.shape()[0];
    let num_classes = logits.shape()[1];
    let data = logits.data();

    let mut correct = 0;
    for i in 0..batch_size {
        let start = i * num_classes;
        let end = start + num_classes;
        let slice = &data[start..end];

        // argmax
        let predicted = slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if predicted == labels[i] {
            correct += 1;
        }
    }

    correct as f32 / batch_size as f32
}

/// Print per-class accuracy on the given dataset.
fn print_per_class_accuracy(model: &Sequential, x: &Tensor, labels: &[usize]) {
    let logits = model.forward(x);
    let batch_size = logits.shape()[0];
    let num_classes = logits.shape()[1];
    let data = logits.data();

    let mut class_correct = [0usize; 5];
    let mut class_total = [0usize; 5];

    for i in 0..batch_size {
        let start = i * num_classes;
        let end = start + num_classes;
        let slice = &data[start..end];

        let predicted = slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let true_label = labels[i];
        if true_label < 5 {
            class_total[true_label] += 1;
            if predicted == true_label {
                class_correct[true_label] += 1;
            }
        }
    }

    for i in 0..5 {
        if let Some(cls) = SafetyClass::from_index(i) {
            if class_total[i] > 0 {
                let acc = class_correct[i] as f64 / class_total[i] as f64 * 100.0;
                println!(
                    "    {}: {}/{} ({:.1}%)",
                    cls.label(),
                    class_correct[i],
                    class_total[i],
                    acc
                );
            } else {
                println!("    {}: 0/0 (N/A)", cls.label());
            }
        }
    }
}

/// Simple deterministic shuffle by rotation + swap pattern.
fn rotate_indices(indices: &mut [usize], epoch: usize) {
    let n = indices.len();
    if n < 2 {
        return;
    }
    // Rotate by epoch amount
    let rotation = epoch % n;
    indices.rotate_left(rotation);
    // Additional swaps based on epoch for more mixing
    let step = (epoch * 7 + 13) % n;
    if step > 0 {
        for i in (0..n).step_by(step.max(1)) {
            let j = (i + step) % n;
            indices.swap(i, j);
        }
    }
}

/// Load corpus from bashrs JSONL export.
///
/// Supports two formats:
///
/// **v2 classification format** (from `bashrs corpus export-dataset --format classification`):
/// ```json
/// {"input":"#!/bin/sh\necho hello\n","label":0}
/// ```
///
/// **v1/v2 full format** (from `bashrs corpus export-dataset --format jsonl`):
/// ```json
/// {"id":"B-001","safety_index":0,"actual_output":"#!/bin/sh\n...","lint_clean":true,...}
/// ```
///
/// Auto-detects format from the first line.
fn load_jsonl(path: &str) -> Vec<CorpusSample> {
    let file = std::fs::File::open(path).expect("Failed to open JSONL file");
    let reader = std::io::BufReader::new(file);
    let mut samples = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.trim().is_empty() {
            continue;
        }

        let parsed: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if let Some(sample) = parse_jsonl_entry(&parsed, idx) {
            samples.push(sample);
        }
    }

    samples
}

/// Parse a single JSONL entry, auto-detecting classification vs full format.
fn parse_jsonl_entry(parsed: &serde_json::Value, idx: usize) -> Option<CorpusSample> {
    // Auto-detect format: classification JSONL has "input" + "label" only
    if parsed.get("input").is_some() && parsed.get("label").is_some() && parsed.get("id").is_none()
    {
        let input = parsed["input"].as_str().unwrap_or("").to_string();
        let label = parsed["label"].as_u64().unwrap_or(0) as usize;
        if !input.is_empty() && label < 5 {
            return Some(CorpusSample {
                id: format!("C-{idx:05}"),
                input,
                label,
            });
        }
        return None;
    }

    // Full dataset format: use safety_index if present, else derive
    let id = parsed["id"].as_str().unwrap_or("").to_string();
    let input = parsed["actual_output"]
        .as_str()
        .or_else(|| parsed["expected_output"].as_str())
        .unwrap_or("")
        .to_string();

    if input.is_empty() {
        return None;
    }

    let label = if let Some(idx) = parsed["safety_index"].as_u64() {
        idx as usize
    } else {
        derive_safety_label(parsed)
    };

    Some(CorpusSample { id, input, label })
}

/// Derive safety class from corpus JSONL fields.
fn derive_safety_label(entry: &serde_json::Value) -> usize {
    let lint_clean = entry["lint_clean"].as_bool().unwrap_or(false);
    let deterministic = entry["deterministic"].as_bool().unwrap_or(false);
    let transpiled = entry["transpiled"].as_bool().unwrap_or(false);
    let output_correct = entry["output_correct"].as_bool().unwrap_or(false);

    // Check the actual output for safety patterns
    let output = entry["actual_output"].as_str().unwrap_or("");

    // Priority ordering: unsafe > non-deterministic > non-idempotent > needs-quoting > safe
    if !transpiled || !lint_clean {
        return SafetyClass::Unsafe as usize;
    }

    if !deterministic {
        return SafetyClass::NonDeterministic as usize;
    }

    // Check for non-idempotent patterns
    if output.contains("mkdir ") && !output.contains("mkdir -p")
        || output.contains("rm ") && !output.contains("rm -f") && !output.contains("rm -rf")
    {
        return SafetyClass::NonIdempotent as usize;
    }

    // Check for unquoted variables
    if has_unquoted_variables(output) {
        return SafetyClass::NeedsQuoting as usize;
    }

    if output_correct {
        SafetyClass::Safe as usize
    } else {
        SafetyClass::NeedsQuoting as usize
    }
}

/// Simple heuristic to detect unquoted shell variables.
fn has_unquoted_variables(script: &str) -> bool {
    let chars: Vec<char> = script.chars().collect();
    let mut i = 0;
    let mut in_double_quote = false;
    let mut in_single_quote = false;

    while i < chars.len() {
        match chars[i] {
            '"' if !in_single_quote => in_double_quote = !in_double_quote,
            '\'' if !in_double_quote => in_single_quote = !in_single_quote,
            '$' if !in_single_quote && !in_double_quote => {
                // Found unquoted $ — check it's a variable reference
                if i + 1 < chars.len() && (chars[i + 1].is_alphanumeric() || chars[i + 1] == '_') {
                    return true;
                }
            }
            _ => {}
        }
        i += 1;
    }

    false
}

/// Built-in demo data for testing without bashrs corpus.
fn load_demo_data() -> Vec<CorpusSample> {
    let demos = vec![
        // Safe scripts
        (
            "D-001",
            "#!/bin/sh\necho \"hello world\"\n",
            SafetyClass::Safe,
        ),
        (
            "D-002",
            "#!/bin/sh\nmkdir -p \"$HOME/tmp\"\n",
            SafetyClass::Safe,
        ),
        (
            "D-003",
            "#!/bin/sh\nrm -f \"$TMPDIR/cache\"\n",
            SafetyClass::Safe,
        ),
        (
            "D-004",
            "#!/bin/sh\nln -sf \"$src\" \"$dest\"\n",
            SafetyClass::Safe,
        ),
        (
            "D-005",
            "#!/bin/sh\ncp -f \"$input\" \"$output\"\n",
            SafetyClass::Safe,
        ),
        (
            "D-006",
            "#!/bin/sh\nprintf '%s\\n' \"$msg\"\n",
            SafetyClass::Safe,
        ),
        (
            "D-007",
            "#!/bin/sh\ntest -f \"$config\" && . \"$config\"\n",
            SafetyClass::Safe,
        ),
        (
            "D-008",
            "#!/bin/sh\nchmod 755 \"$script\"\n",
            SafetyClass::Safe,
        ),
        // Needs quoting
        (
            "D-010",
            "#!/bin/bash\necho $HOME\n",
            SafetyClass::NeedsQuoting,
        ),
        (
            "D-011",
            "#!/bin/bash\nrm -f $file\n",
            SafetyClass::NeedsQuoting,
        ),
        (
            "D-012",
            "#!/bin/bash\nmkdir -p $dir\n",
            SafetyClass::NeedsQuoting,
        ),
        (
            "D-013",
            "#!/bin/bash\ncp $src $dest\n",
            SafetyClass::NeedsQuoting,
        ),
        (
            "D-014",
            "#!/bin/bash\ncat $input | grep pattern\n",
            SafetyClass::NeedsQuoting,
        ),
        (
            "D-015",
            "#!/bin/bash\nfor f in $files; do echo $f; done\n",
            SafetyClass::NeedsQuoting,
        ),
        (
            "D-016",
            "#!/bin/bash\ntest -d $dir && cd $dir\n",
            SafetyClass::NeedsQuoting,
        ),
        (
            "D-017",
            "#!/bin/bash\n[ -f $config ] && source $config\n",
            SafetyClass::NeedsQuoting,
        ),
        // Non-deterministic
        (
            "D-020",
            "#!/bin/bash\necho $RANDOM\n",
            SafetyClass::NonDeterministic,
        ),
        (
            "D-021",
            "#!/bin/bash\necho $$\n",
            SafetyClass::NonDeterministic,
        ),
        (
            "D-022",
            "#!/bin/bash\ndate +%s > timestamp.txt\n",
            SafetyClass::NonDeterministic,
        ),
        (
            "D-023",
            "#!/bin/bash\nTMP=/tmp/build_$$\n",
            SafetyClass::NonDeterministic,
        ),
        (
            "D-024",
            "#!/bin/bash\nSEED=$RANDOM\necho $SEED\n",
            SafetyClass::NonDeterministic,
        ),
        (
            "D-025",
            "#!/bin/bash\necho $BASHPID\n",
            SafetyClass::NonDeterministic,
        ),
        (
            "D-026",
            "#!/bin/bash\nps aux | grep $$ \n",
            SafetyClass::NonDeterministic,
        ),
        (
            "D-027",
            "#!/bin/bash\nlogfile=\"build_$(date +%s).log\"\n",
            SafetyClass::NonDeterministic,
        ),
        // Non-idempotent
        (
            "D-030",
            "#!/bin/bash\nmkdir /tmp/build\n",
            SafetyClass::NonIdempotent,
        ),
        (
            "D-031",
            "#!/bin/bash\nln -s src dest\n",
            SafetyClass::NonIdempotent,
        ),
        (
            "D-032",
            "#!/bin/bash\nmkdir build && cd build\n",
            SafetyClass::NonIdempotent,
        ),
        (
            "D-033",
            "#!/bin/bash\ntouch /tmp/lock; mkdir /var/data\n",
            SafetyClass::NonIdempotent,
        ),
        (
            "D-034",
            "#!/bin/bash\nmkdir logs; mkdir cache\n",
            SafetyClass::NonIdempotent,
        ),
        (
            "D-035",
            "#!/bin/bash\nln -s /usr/bin/python python3\n",
            SafetyClass::NonIdempotent,
        ),
        (
            "D-036",
            "#!/bin/bash\nmkdir -m 755 /opt/app\n",
            SafetyClass::NonIdempotent,
        ),
        (
            "D-037",
            "#!/bin/bash\nmkdir dist && cp -r src/* dist/\n",
            SafetyClass::NonIdempotent,
        ),
        // Unsafe
        (
            "D-040",
            "#!/bin/bash\neval \"$user_input\"\n",
            SafetyClass::Unsafe,
        ),
        ("D-041", "#!/bin/bash\nrm -rf /\n", SafetyClass::Unsafe),
        (
            "D-042",
            "#!/bin/bash\ncurl $url | bash\n",
            SafetyClass::Unsafe,
        ),
        ("D-043", "#!/bin/bash\nexec \"$cmd\"\n", SafetyClass::Unsafe),
        (
            "D-044",
            "#!/bin/bash\nchmod 777 /etc/passwd\n",
            SafetyClass::Unsafe,
        ),
        (
            "D-045",
            "#!/bin/bash\nsource <(curl -s $url)\n",
            SafetyClass::Unsafe,
        ),
        (
            "D-046",
            "#!/bin/bash\n$(wget -q -O - $url)\n",
            SafetyClass::Unsafe,
        ),
        (
            "D-047",
            "#!/bin/bash\nDD if=/dev/zero of=/dev/sda\n",
            SafetyClass::Unsafe,
        ),
    ];

    demos
        .into_iter()
        .map(|(id, input, class)| CorpusSample {
            id: id.to_string(),
            input: input.to_string(),
            label: class as usize,
        })
        .collect()
}
