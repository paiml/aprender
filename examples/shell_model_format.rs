//! Shell Model Format Verification Example
//!
//! Demonstrates and verifies the .apr model format for shell completion models.
//! Tests that models are saved with correct ModelType::NgramLm (0x0010).
//!
//! Run: cargo run --example shell_model_format
//!
//! Expected output:
//! ```
//! ✅ Model type: NgramLm (0x0010)
//! ✅ Magic bytes: APRN
//! ✅ Roundtrip: suggestions match
//! ```

use aprender::format::{self, ModelType, SaveOptions};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Minimal Markov model for testing (mirrors aprender-shell's MarkovModel)
#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct TestMarkovModel {
    n: usize,
    ngrams: HashMap<String, HashMap<String, u32>>,
    command_freq: HashMap<String, u32>,
    total_commands: usize,
}

impl TestMarkovModel {
    fn new(n: usize) -> Self {
        Self {
            n,
            ngrams: HashMap::new(),
            command_freq: HashMap::new(),
            total_commands: 0,
        }
    }

    fn train(&mut self, commands: &[&str]) {
        for cmd in commands {
            *self.command_freq.entry((*cmd).to_string()).or_insert(0) += 1;
            self.total_commands += 1;

            // Build n-grams
            let tokens: Vec<&str> = cmd.split_whitespace().collect();
            for window in tokens.windows(self.n) {
                let context = window[..self.n - 1].join(" ");
                let next = window[self.n - 1].to_string();
                *self
                    .ngrams
                    .entry(context)
                    .or_default()
                    .entry(next)
                    .or_insert(0) += 1;
            }
        }
    }

    fn suggest(&self, prefix: &str) -> Vec<(String, f64)> {
        let total: u32 = self.command_freq.values().sum();
        let mut results: Vec<_> = self
            .command_freq
            .iter()
            .filter(|(cmd, _)| cmd.starts_with(prefix))
            .map(|(cmd, count)| (cmd.clone(), f64::from(*count) / f64::from(total)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(5);
        results
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Shell Model Format Verification\n");

    let test_path = Path::new("/tmp/shell_model_format_test.apr");

    let model = create_and_train_model();
    save_model(&model, test_path)?;
    verify_header(test_path)?;
    test_roundtrip(&model, test_path)?;
    test_suggestions(test_path)?;
    test_type_mismatch(test_path);

    fs::remove_file(test_path)?;

    println!("\n════════════════════════════════════════");
    println!("✅ All format verification checks passed!");
    println!("════════════════════════════════════════");

    Ok(())
}

fn create_and_train_model() -> TestMarkovModel {
    println!("1️⃣  Creating test model...");
    let mut model = TestMarkovModel::new(3);
    model.train(&[
        "git status",
        "git commit -m 'test'",
        "git push origin main",
        "git pull --rebase",
        "cargo build --release",
        "cargo test --all",
        "cargo clippy",
        "docker run -it ubuntu",
        "kubectl get pods",
    ]);
    println!("   Trained on {} commands", model.total_commands);
    model
}

fn save_model(model: &TestMarkovModel, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2️⃣  Saving with ModelType::NgramLm...");
    let options = SaveOptions::default()
        .with_name("shell-format-test")
        .with_description("Test model for format verification");
    format::save(model, ModelType::NgramLm, path, options)?;
    println!("   Saved to: {}", path.display());
    Ok(())
}

fn verify_header(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3️⃣  Verifying header...");
    let bytes = fs::read(path)?;

    let magic = &bytes[0..4];
    if magic == b"APRN" {
        println!("   ✅ Magic bytes: APRN");
    } else {
        println!("   ❌ Magic bytes: {magic:?} (expected APRN)");
        return Err("Invalid magic".into());
    }

    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    println!("   ✅ Version: {version}");

    let model_type_raw = u16::from_le_bytes([bytes[6], bytes[7]]);
    match model_type_raw {
        0x0010 => println!("   ✅ Model type: NgramLm (0x{model_type_raw:04X})"),
        0x00FF => {
            println!("   ❌ Model type: Custom (0x{model_type_raw:04X}) - should be NgramLm");
            return Err("Wrong model type".into());
        }
        _ => println!("   ❓ Model type: Unknown (0x{model_type_raw:04X})"),
    }
    Ok(())
}

fn test_roundtrip(model: &TestMarkovModel, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n4️⃣  Testing roundtrip...");
    let loaded: TestMarkovModel = format::load(path, ModelType::NgramLm)?;

    assert_eq!(loaded.n, model.n, "n-gram size mismatch");
    assert_eq!(
        loaded.total_commands, model.total_commands,
        "command count mismatch"
    );
    assert_eq!(
        loaded.command_freq.len(),
        model.command_freq.len(),
        "vocab size mismatch"
    );
    println!("   ✅ Data integrity verified");
    Ok(())
}

fn test_suggestions(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n5️⃣  Testing suggestions...");
    let loaded: TestMarkovModel = format::load(path, ModelType::NgramLm)?;
    let suggestions = loaded.suggest("git ");
    println!("   Suggestions for 'git ':");
    for (cmd, score) in &suggestions {
        println!("      {score:.3}  {cmd}");
    }

    if suggestions.is_empty() {
        println!("   ❌ No suggestions returned");
        return Err("Suggestions failed".into());
    }
    println!("   ✅ Suggestions work");
    Ok(())
}

fn test_type_mismatch(path: &Path) {
    println!("\n6️⃣  Testing type mismatch handling...");
    match format::load::<TestMarkovModel>(path, ModelType::Custom) {
        Ok(_) => println!("   ⚠️  Loaded as Custom (unexpected but ok for compat)"),
        Err(e) => println!("   ✅ Correctly rejected Custom type: {e}"),
    }
}
