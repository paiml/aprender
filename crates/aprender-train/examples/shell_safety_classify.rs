#![allow(clippy::disallowed_methods)]
//! Shell Safety Classification Fine-Tuning Demo
//!
//! Demonstrates the full classification fine-tuning pipeline:
//! Transformer + LoRA adapters + ClassificationHead.
//!
//! Classifies shell scripts into 5 safety categories:
//!
//! | Class | Label            | Example trigger            |
//! |-------|------------------|----------------------------|
//! | 0     | safe             | `echo "hello"`             |
//! | 1     | needs-quoting    | `echo $HOME`               |
//! | 2     | non-deterministic| `echo $RANDOM`             |
//! | 3     | non-idempotent   | `mkdir /tmp/build`         |
//! | 4     | unsafe           | `eval "$user_input"`       |
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────┐
//! │  Transformer (frozen base)     │
//! │  hidden_size=64, layers=2      │
//! │  + LoRA rank=4 on Q/V proj     │
//! ├────────────────────────────────┤
//! │  Mean Pool → Linear(64 → 5)   │
//! │  ClassificationHead (trainable)│
//! └────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```bash
//! # Quick demo with built-in corpus (no files needed)
//! cargo run --example shell_safety_classify
//!
//! # With a JSONL corpus file
//! cargo run --example shell_safety_classify -- /path/to/corpus.jsonl
//!
//! # Via apr-cli (Qwen2-0.5B config)
//! apr finetune --task classify --model-size 0.5B --data corpus.jsonl
//! ```
//!
//! # Corpus Format
//!
//! JSONL with `input` (shell script) and `label` (0-4):
//!
//! ```json
//! {"input": "#!/bin/bash\necho $HOME\n", "label": 1}
//! {"input": "#!/bin/bash\neval \"$x\"\n", "label": 4}
//! ```
//!
//! # Contract
//!
//! See `aprender/contracts/classification-finetune-v1.yaml` for:
//! - F-CLASS-001: Logit shape == num_classes
//! - F-CLASS-002: Label index < num_classes
//! - F-CLASS-004: Classifier weight shape validated
//! - F-CLASS-005: Loss is always finite
//!
//! # References
//!
//! - Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
//! - bashrs shell safety specification: docs/specifications/shell-safety-inference.md

use entrenar::finetune::classify_pipeline::{ClassifyConfig, ClassifyPipeline};
use entrenar::finetune::{corpus_stats, load_safety_corpus, SafetySample};
use entrenar::transformer::TransformerConfig;
use std::path::Path;

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

/// Compute softmax probabilities from logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    logits.iter().map(|&v| (v - max).exp() / exp_sum).collect()
}

const LABELS: [&str; 5] =
    ["safe", "needs-quoting", "non-deterministic", "non-idempotent", "unsafe"];

fn main() {
    println!("======================================================");
    println!("  Shell Safety Classification — Fine-Tuning Demo");
    println!("  Powered by entrenar (training) + aprender (contracts)");
    println!("======================================================\n");

    // ── 1. Load corpus ──────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let samples = if let Some(path) = args.get(1) {
        println!("Loading corpus from: {path}");
        load_safety_corpus(Path::new(path), 5).expect("Failed to load corpus")
    } else {
        println!("Using built-in demo corpus (15 samples)");
        println!("  Tip: pass a JSONL file for custom data\n");
        demo_corpus()
    };

    let stats = corpus_stats(&samples, 5);
    println!("Corpus: {} samples", stats.total);
    println!("  Avg input: {} chars", stats.avg_input_len);
    for (i, count) in stats.class_counts.iter().enumerate() {
        println!("  [{i}] {:<20} {count} samples", LABELS[i]);
    }

    // ── 2. Build pipeline ───────────────────────────────────────────
    println!("\n─── Pipeline Configuration ───\n");

    // Tiny config for demo (64 hidden, 2 layers — runs instantly)
    let model_config = TransformerConfig::tiny();
    let classify_config = ClassifyConfig {
        num_classes: 5,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-3,
        epochs: 10,
        max_seq_len: 64,
        log_interval: 5,
        ..ClassifyConfig::default()
    };

    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    println!("{}\n", pipeline.summary());

    // ── 3. Forward pass: classify each sample ───────────────────────
    println!("─── Classification (untrained model) ───\n");
    println!(
        "  {:<35} {:<8} {:<20} {:>10}",
        "Script (first 30 chars)", "True", "Predicted", "Confidence"
    );
    println!("  {}", "-".repeat(78));

    let mut correct = 0;
    let mut total = 0;

    for sample in &samples {
        let token_ids = tokenize(&sample.input, 64);
        let hidden = pipeline.model.forward_hidden(&token_ids);
        let logits_tensor = pipeline.classifier.forward(&hidden, token_ids.len());
        let logits: Vec<f32> = logits_tensor.data().as_slice().expect("contiguous").to_vec();
        let probs = softmax(&logits);

        let predicted = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("operation should succeed"))
            .map_or(0, |(i, _)| i);
        let confidence = probs[predicted] * 100.0;

        let snippet: String = sample.input.chars().filter(|c| *c != '\n').take(30).collect();
        let marker = if predicted == sample.label {
            correct += 1;
            " "
        } else {
            "x"
        };

        println!(
            "{marker} {:<35} {:<8} {:<20} {:>9.1}%",
            snippet, LABELS[sample.label], LABELS[predicted], confidence
        );
        total += 1;
    }

    println!("\n  Accuracy: {correct}/{total} ({:.0}%)", correct as f32 / total as f32 * 100.0);
    println!("  (Random init — accuracy improves with training)\n");

    // ── 4. Training loop demo ───────────────────────────────────────
    println!("─── Training Loop (10 epochs) ───\n");

    let mut losses: Vec<f32> = Vec::new();
    let epochs = 10;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        for sample in &samples {
            let token_ids = tokenize(&sample.input, 64);
            let loss = pipeline.train_step(&token_ids, sample.label);
            epoch_loss += loss;
        }
        let avg_loss = epoch_loss / samples.len() as f32;
        losses.push(avg_loss);

        if epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 2 == 0 {
            let bar_len = ((avg_loss / losses[0]) * 30.0).min(30.0) as usize;
            let bar: String = "#".repeat(bar_len);
            println!("  Epoch {:>2}/{epochs}  loss={avg_loss:.4}  {bar}", epoch + 1);
        }
    }

    let improvement = ((losses[0] - losses[losses.len() - 1]) / losses[0]) * 100.0;
    println!(
        "\n  Loss: {:.4} -> {:.4} ({improvement:+.1}%)\n",
        losses[0],
        losses[losses.len() - 1]
    );

    // ── 5. Merge adapters ───────────────────────────────────────────
    println!("─── LoRA Adapter Merge ───\n");
    let adapters_before = pipeline.lora_layers.iter().filter(|l| !l.is_merged()).count();
    pipeline.merge_adapters();
    let adapters_after = pipeline.lora_layers.iter().filter(|l| l.is_merged()).count();
    println!("  Merged {adapters_before} adapters into base weights");
    println!("  Merged status: {adapters_after}/{} adapters\n", pipeline.lora_layers.len());

    // ── 6. Qwen2-0.5B plan ─────────────────────────────────────────
    println!("─── Production Config (Qwen2.5-Coder-0.5B) ───\n");
    let qwen_config = TransformerConfig::qwen2_0_5b();
    let prod_classify = ClassifyConfig {
        num_classes: 5,
        lora_rank: 16,
        lora_alpha: 16.0,
        learning_rate: 1e-4,
        epochs: 3,
        max_seq_len: 512,
        log_interval: 100,
        ..ClassifyConfig::default()
    };
    let prod_pipeline = ClassifyPipeline::new(&qwen_config, prod_classify);
    println!("{}\n", prod_pipeline.summary());

    println!("  To run with Qwen2 weights:");
    println!("    apr finetune --task classify --model-size 0.5B \\");
    println!("        --data corpus.jsonl --epochs 3\n");

    println!("======================================================");
    println!("  Demo complete.");
    println!("======================================================");
}
