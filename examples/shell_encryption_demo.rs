//! Shell Model Encryption Demo
//!
//! Demonstrates both encrypted and unencrypted model formats in aprender-shell.
//!
//! Run: cargo run --example shell_encryption_demo --features format-encryption
//!
//! This example shows:
//! 1. Creating and training a shell completion model
//! 2. Saving as unencrypted .apr file
//! 3. Saving as encrypted .apr file (AES-256-GCM with Argon2id)
//! 4. Loading both formats
//! 5. Verifying suggestions work identically

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
            *self.command_freq.entry(cmd.to_string()).or_insert(0) += 1;
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
            .map(|(cmd, count)| (cmd.clone(), *count as f64 / total as f64))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(5);
        results
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let options = SaveOptions::default()
            .with_name("shell-encryption-demo")
            .with_description("Demo model for encryption testing");
        format::save(self, ModelType::NgramLm, path, options)?;
        Ok(())
    }

    fn save_encrypted(
        &self,
        path: &Path,
        password: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let options = SaveOptions::default()
            .with_name("shell-encryption-demo")
            .with_description("Demo model for encryption testing (encrypted)");
        format::save_encrypted(self, ModelType::NgramLm, path, options, password)?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(format::load(path, ModelType::NgramLm)?)
    }

    fn load_encrypted(path: &Path, password: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(format::load_encrypted(path, ModelType::NgramLm, password)?)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 Shell Model Encryption Demo\n");
    println!("This example demonstrates BOTH encrypted and unencrypted model formats.\n");

    let unencrypted_path = Path::new("/tmp/shell_demo_unencrypted.apr");
    let encrypted_path = Path::new("/tmp/shell_demo_encrypted.apr");
    let password = "demo_password_123";

    // 1. Create and train model
    println!("1️⃣  Creating and training model...");
    let mut model = TestMarkovModel::new(3);
    model.train(&[
        "git status",
        "git commit -m 'test'",
        "git push origin main",
        "git pull --rebase",
        "git log --oneline",
        "cargo build --release",
        "cargo test --all",
        "cargo clippy -- -D warnings",
        "docker run -it ubuntu",
        "docker ps -a",
        "kubectl get pods",
        "kubectl logs -f deployment/app",
    ]);
    println!("   Trained on {} commands\n", model.total_commands);

    // 2. Save as unencrypted
    println!("2️⃣  Saving UNENCRYPTED model...");
    model.save(unencrypted_path)?;
    let unenc_size = fs::metadata(unencrypted_path)?.len();
    println!("   ✅ Saved to: {}", unencrypted_path.display());
    println!("   📦 Size: {} bytes\n", unenc_size);

    // 3. Save as encrypted
    println!("3️⃣  Saving ENCRYPTED model (AES-256-GCM)...");
    model.save_encrypted(encrypted_path, password)?;
    let enc_size = fs::metadata(encrypted_path)?.len();
    println!("   ✅ Saved to: {}", encrypted_path.display());
    println!("   📦 Size: {} bytes", enc_size);
    println!("   🔒 Encryption: AES-256-GCM with Argon2id KDF\n");

    // 4. Verify headers
    println!("4️⃣  Verifying file headers...");

    // Unencrypted header
    let unenc_bytes = fs::read(unencrypted_path)?;
    let unenc_magic = &unenc_bytes[0..4];
    let unenc_type = u16::from_le_bytes([unenc_bytes[6], unenc_bytes[7]]);
    println!("   Unencrypted:");
    println!(
        "      Magic: {:?} ({})",
        unenc_magic,
        String::from_utf8_lossy(unenc_magic)
    );
    println!("      Type:  0x{:04X} (NgramLm)", unenc_type);

    // Encrypted header
    let enc_bytes = fs::read(encrypted_path)?;
    let enc_magic = &enc_bytes[0..4];
    let enc_type = u16::from_le_bytes([enc_bytes[6], enc_bytes[7]]);
    // Check encryption flag (in header flags at offset 8)
    let enc_flags = u32::from_le_bytes([enc_bytes[8], enc_bytes[9], enc_bytes[10], enc_bytes[11]]);
    let is_encrypted = (enc_flags & 0x01) != 0;
    println!("   Encrypted:");
    println!(
        "      Magic: {:?} ({})",
        enc_magic,
        String::from_utf8_lossy(enc_magic)
    );
    println!("      Type:  0x{:04X} (NgramLm)", enc_type);
    println!(
        "      Flags: 0x{:08X} (encrypted={})\n",
        enc_flags, is_encrypted
    );

    // 5. Load unencrypted
    println!("5️⃣  Loading UNENCRYPTED model...");
    let loaded_unenc = TestMarkovModel::load(unencrypted_path)?;
    println!(
        "   ✅ Loaded {} commands, {} n-grams\n",
        loaded_unenc.total_commands,
        loaded_unenc.ngrams.len()
    );

    // 6. Load encrypted
    println!("6️⃣  Loading ENCRYPTED model...");
    let loaded_enc = TestMarkovModel::load_encrypted(encrypted_path, password)?;
    println!(
        "   ✅ Loaded {} commands, {} n-grams\n",
        loaded_enc.total_commands,
        loaded_enc.ngrams.len()
    );

    // 7. Verify suggestions match
    println!("7️⃣  Verifying suggestions are identical...");
    let prefixes = ["git ", "cargo ", "docker "];
    for prefix in prefixes {
        let unenc_suggestions = loaded_unenc.suggest(prefix);
        let enc_suggestions = loaded_enc.suggest(prefix);

        // Compare as sets (HashMap ordering may differ)
        let unenc_set: std::collections::HashSet<_> = unenc_suggestions
            .iter()
            .map(|(cmd, _)| cmd.as_str())
            .collect();
        let enc_set: std::collections::HashSet<_> = enc_suggestions
            .iter()
            .map(|(cmd, _)| cmd.as_str())
            .collect();

        if unenc_set == enc_set {
            println!(
                "   ✅ '{}' - {} suggestions match",
                prefix,
                unenc_suggestions.len()
            );
        } else {
            println!("   ❌ '{}' - suggestions differ!", prefix);
            println!("      Unencrypted: {:?}", unenc_suggestions);
            println!("      Encrypted:   {:?}", enc_suggestions);
        }
    }

    // 8. Try loading encrypted without password (should fail)
    println!("\n8️⃣  Testing wrong password handling...");
    match TestMarkovModel::load_encrypted(encrypted_path, "wrong_password") {
        Ok(_) => println!("   ⚠️  Unexpected: Model loaded with wrong password!"),
        Err(e) => println!("   ✅ Correctly rejected wrong password: {}", e),
    }

    // 9. Try loading encrypted as unencrypted (should fail)
    println!("\n9️⃣  Testing encrypted model without password...");
    match TestMarkovModel::load(encrypted_path) {
        Ok(_) => println!("   ⚠️  Unexpected: Encrypted model loaded without password!"),
        Err(e) => println!("   ✅ Correctly rejected: {}", e),
    }

    // Cleanup
    fs::remove_file(unencrypted_path)?;
    fs::remove_file(encrypted_path)?;

    println!("\n════════════════════════════════════════════════════════");
    println!("✅ Encryption Demo Complete!");
    println!("════════════════════════════════════════════════════════");
    println!("\n📚 Usage in aprender-shell:");
    println!("   # Train with encryption");
    println!("   aprender-shell train --password");
    println!("");
    println!("   # Load encrypted model for suggestions");
    println!("   aprender-shell suggest \"git \" --password");
    println!("");
    println!("   # Or use environment variable");
    println!("   export APRENDER_PASSWORD=your_password");
    println!("   aprender-shell suggest \"git \" --password");
    println!("");
    println!("   # View encrypted model stats");
    println!("   aprender-shell stats --password");

    Ok(())
}
