#![allow(clippy::disallowed_methods)]
//! Publish Shell Safety Classifier to HuggingFace Hub
//!
//! Uploads the trained shell safety model to HuggingFace as
//! `paiml/shell-safety-classifier`.
//!
//! # Prerequisites
//!
//! 1. Train the model first:
//!    ```bash
//!    cargo run --example shell_safety_training -- /tmp/corpus.jsonl
//!    ```
//!
//! 2. Set HuggingFace token:
//!    ```bash
//!    export HF_TOKEN=hf_xxxxxxxxxxxxx
//!    ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --features hf-hub-integration --example publish_shell_safety -- /tmp/shell-safety-model/
//! ```

use aprender::format::model_card::{ModelCard, TrainingDataInfo};
use aprender::text::shell_vocab::SafetyClass;

fn main() {
    println!("======================================================");
    println!("  Publish Shell Safety Classifier to HuggingFace");
    println!("======================================================\n");

    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/tmp/shell-safety-model");

    let repo_id = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("paiml/shell-safety-classifier");

    // Verify model artifacts exist
    let model_path = format!("{model_dir}/model.safetensors");
    let vocab_path = format!("{model_dir}/vocab.json");
    let config_path = format!("{model_dir}/config.json");

    println!("Model directory: {model_dir}");
    println!("Target repo: {repo_id}\n");

    let mut missing = Vec::new();
    for (name, path) in [
        ("model.safetensors", &model_path),
        ("vocab.json", &vocab_path),
        ("config.json", &config_path),
    ] {
        if std::path::Path::new(path).exists() {
            let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            println!("  [OK] {name} ({} bytes)", size);
        } else {
            println!("  [MISSING] {name}");
            missing.push(name);
        }
    }

    if !missing.is_empty() {
        eprintln!("\nError: Missing model artifacts: {}", missing.join(", "));
        eprintln!("Run training first:");
        eprintln!("  cargo run --example shell_safety_training -- /tmp/corpus.jsonl");
        std::process::exit(1);
    }

    // Generate model card
    let labels: Vec<String> = SafetyClass::all()
        .iter()
        .map(|c| c.label().to_string())
        .collect();

    let model_card = ModelCard::new(repo_id, env!("CARGO_PKG_VERSION"))
        .with_name("Shell Safety Classifier")
        .with_author("paiml")
        .with_description(
            "Classifies bash/shell scripts into 5 safety categories: \
             safe, needs-quoting, non-deterministic, non-idempotent, unsafe. \
             Trained on the bashrs corpus (17,942 entries) using aprender (pure Rust ML).",
        )
        .with_license("MIT")
        .with_architecture("MLP classifier (input -> 128 -> 64 -> 5)")
        .with_training_data(TrainingDataInfo::new("bashrs-corpus").with_samples(17_942))
        .with_hyperparameter("learning_rate", "0.01")
        .with_hyperparameter("epochs", "50")
        .with_hyperparameter("optimizer", "Adam")
        .with_hyperparameter("loss", "CrossEntropyLoss")
        .with_hyperparameter("max_seq_len", "64")
        .with_hyperparameter("hidden_dim", "128")
        .with_metric("num_classes", "5")
        .with_metric("labels", labels.join(", "));

    // Generate README content
    let readme = generate_readme(&model_card, repo_id);

    // Write README to model directory
    let readme_path = format!("{model_dir}/README.md");
    std::fs::write(&readme_path, &readme).expect("Failed to write README.md");
    println!("\n  [OK] README.md generated");

    // Check for HF token
    let hf_token = std::env::var("HF_TOKEN").ok();

    if hf_token.is_none() {
        println!("\n------------------------------------------------------");
        println!("HF_TOKEN not set. Model artifacts ready for manual upload.");
        println!("\nTo publish:");
        println!("  1. export HF_TOKEN=hf_xxxxxxxxxxxxx");
        println!("  2. Re-run this command");
        println!("\nOr upload manually:");
        println!("  huggingface-cli upload {repo_id} {model_dir}");
        println!("------------------------------------------------------");

        // Still save the README so it can be uploaded manually
        println!("\nModel card preview:\n");
        println!("{}", &readme[..readme.len().min(500)]);
        println!("...\n");
        return;
    }

    // Upload using HfHubClient
    println!("\nUploading to HuggingFace Hub...");

    #[cfg(feature = "hf-hub-integration")]
    {
        use aprender::hf_hub::{HfHubClient, PushOptions};

        let client = HfHubClient::new().expect("Failed to create HF client");

        // Upload model weights
        let model_data = std::fs::read(&model_path).expect("Failed to read model");
        let options = PushOptions::new()
            .with_commit_message("Upload shell safety classifier (aprender + bashrs)")
            .with_model_card(model_card)
            .with_filename("model.safetensors");

        match client.push_to_hub(repo_id, &model_data, options) {
            Ok(result) => {
                println!("\nUpload successful!");
                println!("  Repo: {}", result.repo_url);
                println!("  Files: {:?}", result.files_uploaded);
                println!("  Bytes: {}", result.bytes_transferred);
            }
            Err(e) => {
                eprintln!("\nUpload failed: {e}");
                eprintln!("You can upload manually:");
                eprintln!("  huggingface-cli upload {repo_id} {model_dir}");
                std::process::exit(1);
            }
        }

        // Upload vocab.json and config.json
        for (filename, path) in [("vocab.json", &vocab_path), ("config.json", &config_path)] {
            let data = std::fs::read(path).expect("Failed to read file");
            let opts = PushOptions::new()
                .with_commit_message(format!("Upload {filename}"))
                .with_filename(filename);

            if let Err(e) = client.push_to_hub(repo_id, &data, opts) {
                eprintln!("Warning: Failed to upload {filename}: {e}");
            } else {
                println!("  Uploaded: {filename}");
            }
        }
    }

    #[cfg(not(feature = "hf-hub-integration"))]
    {
        println!("\nNote: hf-hub-integration feature not enabled.");
        println!(
            "Rebuild with: cargo run --features hf-hub-integration --example publish_shell_safety"
        );
        println!("\nAlternatively, upload manually:");
        println!("  huggingface-cli upload {repo_id} {model_dir}");
    }

    println!("\n======================================================");
    println!("  Publication complete!");
    println!("  View at: https://huggingface.co/{repo_id}");
    println!("======================================================");
}

/// Generate a HuggingFace-compatible README.md with YAML front matter.
fn generate_readme(card: &ModelCard, _repo_id: &str) -> String {
    // Use the built-in HF export if available, otherwise generate manually
    let hf_readme = card.to_huggingface();

    // If the built-in export works, use it
    if !hf_readme.is_empty() {
        return hf_readme;
    }

    // Fallback: manual generation
    format!(
        r#"---
license: mit
tags:
  - shell
  - bash
  - safety
  - linting
  - aprender
  - bashrs
datasets:
  - paiml/bashrs-corpus
metrics:
  - accuracy
  - f1
library_name: aprender
---

# Shell Safety Classifier

Classifies bash/shell scripts into 5 safety categories.

**Trained with**: [aprender](https://crates.io/crates/aprender) (pure Rust ML)
**Training data**: [bashrs](https://crates.io/crates/bashrs) corpus (17,942 entries)

## Labels

| Class | Label | Description |
|-------|-------|-------------|
| 0 | `safe` | Passes all checks (lint, deterministic, idempotent) |
| 1 | `needs-quoting` | Variable quoting issues |
| 2 | `non-deterministic` | Contains RANDOM, PID, timestamps |
| 3 | `non-idempotent` | Missing -p/-f flags |
| 4 | `unsafe` | Security rule violations (SEC001-008) |

## Usage

```bash
# Classify with bashrs
bashrs lint script.sh
```

## Training

- **Framework**: aprender {version}
- **Architecture**: MLP (64 -> 128 -> 64 -> 5)
- **Optimizer**: Adam (lr=0.01)
- **Loss**: CrossEntropyLoss
- **Epochs**: 50
- **Corpus**: bashrs v6.64.0 ({n_entries} entries)

## Model Card

Generated by [aprender](https://crates.io/crates/aprender), the pure Rust ML framework.
"#,
        version = env!("CARGO_PKG_VERSION"),
        n_entries = "17,942",
    )
}
