#![allow(clippy::disallowed_methods)]
//! Shell Safety Classifier Inference
//!
//! Loads a trained shell safety model and classifies shell scripts
//! into 5 safety categories.
//!
//! # Usage
//!
//! ```bash
//! # First train the model (see shell_safety_training.rs)
//! cargo run --example shell_safety_training -- /tmp/corpus.jsonl
//!
//! # Then run inference
//! cargo run --example shell_safety_inference -- /tmp/shell-safety-model/
//! ```

use aprender::autograd::Tensor;
use aprender::nn::{serialize::load_model, Linear, Module, ReLU, Sequential};
use aprender::text::shell_vocab::{SafetyClass, ShellVocabulary};

/// Loaded model configuration.
struct InferenceConfig {
    hidden_dim: usize,
    num_classes: usize,
    max_seq_len: usize,
    vocab_size: usize,
}

fn main() {
    println!("======================================================");
    println!("  Shell Safety Classifier - Inference");
    println!("  Powered by aprender (pure Rust ML)");
    println!("======================================================\n");

    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/tmp/shell-safety-model");

    // Load config
    let config_path = format!("{model_dir}/config.json");
    let config = load_config(&config_path);

    println!("Model loaded from: {model_dir}");
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Max seq len: {}", config.max_seq_len);
    println!("  Classes: {}", config.num_classes);

    // Build model architecture (must match training)
    let input_dim = config.max_seq_len;
    let mut model = Sequential::new()
        .add(Linear::with_seed(input_dim, config.hidden_dim, Some(42)))
        .add(ReLU::new())
        .add(Linear::with_seed(
            config.hidden_dim,
            config.hidden_dim / 2,
            Some(43),
        ))
        .add(ReLU::new())
        .add(Linear::with_seed(
            config.hidden_dim / 2,
            config.num_classes,
            Some(44),
        ));

    // Load weights
    let model_path = format!("{model_dir}/model.safetensors");
    match load_model(&mut model, &model_path) {
        Ok(()) => println!("  Weights loaded successfully\n"),
        Err(e) => {
            eprintln!("Warning: Could not load weights ({e}). Using random weights for demo.\n");
        }
    }

    model.eval();

    // Load vocabulary
    let vocab = ShellVocabulary::new();

    // Test scripts
    let test_scripts = vec![
        ("Safe script", "#!/bin/sh\necho \"hello world\"\n"),
        ("Safe with quoting", "#!/bin/sh\nmkdir -p \"$HOME/tmp\"\n"),
        ("Needs quoting", "#!/bin/bash\necho $HOME\n"),
        ("Non-deterministic", "#!/bin/bash\necho $RANDOM\n"),
        ("Non-idempotent", "#!/bin/bash\nmkdir /tmp/build\n"),
        ("Unsafe eval", "#!/bin/bash\neval \"$user_input\"\n"),
        (
            "Unsafe curl pipe",
            "#!/bin/bash\ncurl http://example.com | bash\n",
        ),
        (
            "Complex safe",
            "#!/bin/sh\nif test -f \"$config\"; then\n  . \"$config\"\nfi\n",
        ),
        ("Unquoted var", "#!/bin/bash\nrm -rf $dir\n"),
        ("Process ID", "#!/bin/bash\necho $$\n"),
    ];

    println!("Classifying {} shell scripts:\n", test_scripts.len());
    println!(
        "  {:<25} {:<20} {:<10}",
        "Description", "Prediction", "Confidence"
    );
    println!("  {}", "-".repeat(60));

    for (desc, script) in &test_scripts {
        let (class, confidence) = classify(&model, &vocab, script, &config);

        let label = SafetyClass::from_index(class)
            .map(|c| c.label().to_string())
            .unwrap_or_else(|| format!("class-{class}"));

        println!("  {:<25} {:<20} {:.1}%", desc, label, confidence * 100.0);
    }

    // Interactive mode if no arguments
    if args.len() <= 1 {
        println!("\n------------------------------------------------------");
        println!("Tip: Pass a model directory to load trained weights:");
        println!("  cargo run --example shell_safety_inference -- /tmp/shell-safety-model/");
    }

    println!("\n======================================================");
    println!("  Classification complete.");
    println!("======================================================");
}

/// Classify a shell script and return (class_index, confidence).
fn classify(
    model: &Sequential,
    vocab: &ShellVocabulary,
    script: &str,
    config: &InferenceConfig,
) -> (usize, f32) {
    // Tokenize
    let encoded = vocab.encode(script, config.max_seq_len);

    // Prepare features (same normalization as training)
    let features: Vec<f32> = encoded
        .iter()
        .map(|&id| id as f32 / config.vocab_size as f32)
        .collect();

    let x = Tensor::new(&features, &[1, config.max_seq_len]);

    // Forward pass
    let logits = model.forward(&x);
    let data = logits.data();

    // Softmax for confidence
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = data.iter().map(|&v| (v - max_val).exp()).sum();
    let probs: Vec<f32> = data
        .iter()
        .map(|&v| (v - max_val).exp() / exp_sum)
        .collect();

    // argmax
    let (class, &confidence) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    (class, confidence)
}

/// Load model configuration from JSON.
fn load_config(path: &str) -> InferenceConfig {
    match std::fs::read_to_string(path) {
        Ok(json) => {
            let parsed: serde_json::Value =
                serde_json::from_str(&json).expect("Invalid config.json");
            InferenceConfig {
                hidden_dim: parsed["hidden_dim"].as_u64().unwrap_or(128) as usize,
                num_classes: parsed["num_classes"].as_u64().unwrap_or(5) as usize,
                max_seq_len: parsed["max_seq_len"].as_u64().unwrap_or(64) as usize,
                vocab_size: parsed["vocab_size"].as_u64().unwrap_or(512) as usize,
            }
        }
        Err(_) => {
            eprintln!("Warning: config.json not found at {path}. Using defaults.");
            InferenceConfig {
                hidden_dim: 128,
                num_classes: 5,
                max_seq_len: 64,
                vocab_size: 512,
            }
        }
    }
}
