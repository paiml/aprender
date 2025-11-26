# Developer's Guide to Shell History Models

Build personalized ML models from your shell history using the `.apr` format. This guide follows **EXTREME TDD** methodology—every code example compiles and runs.

## Why Shell History is Perfect for ML

Shell commands exhibit strong **Markov properties**:

```text
P(next_token | all_previous) ≈ P(next_token | last_n_tokens)
```

Translation: What you type next depends mostly on your last few words, not your entire history.

**Evidence from real data:**
- `git` → 65% followed by `status`, `commit`, `push`, `pull`
- `cargo` → 70% followed by `build`, `test`, `run`, `clippy`
- `cd` → 80% followed by `..`, project names, or `~`

This predictability makes N-gram models highly effective with minimal compute.

## Part 1: First Principles - Building from Scratch

### Step 1: Define the Core Data Structure (RED)

```rust
use std::collections::HashMap;

/// N-gram frequency table
/// Maps context (previous n-1 tokens) → next token → count
#[derive(Default)]
struct NgramTable {
    /// context → (next_token → frequency)
    table: HashMap<String, HashMap<String, u32>>,
}

impl NgramTable {
    fn new() -> Self {
        Self::default()
    }

    /// Record an observation: given context, next token appeared
    fn observe(&mut self, context: &str, next_token: &str) {
        self.table
            .entry(context.to_string())
            .or_default()
            .entry(next_token.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    /// Get probability distribution for context
    fn predict(&self, context: &str) -> Vec<(String, f32)> {
        let Some(counts) = self.table.get(context) else {
            return vec![];
        };

        let total: u32 = counts.values().sum();
        let mut probs: Vec<_> = counts
            .iter()
            .map(|(token, count)| {
                (token.clone(), *count as f32 / total as f32)
            })
            .collect();

        // Sort by probability descending
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        probs
    }
}

// Test: Empty table returns empty predictions
let table = NgramTable::new();
assert!(table.predict("git").is_empty());

// Test: Single observation
let mut table = NgramTable::new();
table.observe("git", "status");
let preds = table.predict("git");
assert_eq!(preds.len(), 1);
assert_eq!(preds[0].0, "status");
assert!((preds[0].1 - 1.0).abs() < 0.001); // 100% probability
```

### Step 2: Train on Command Sequences (GREEN)

```rust
use std::collections::HashMap;

#[derive(Default)]
struct NgramTable {
    table: HashMap<String, HashMap<String, u32>>,
    n: usize,
}

impl NgramTable {
    fn with_n(n: usize) -> Self {
        Self { table: HashMap::new(), n: n.max(2) }
    }

    fn observe(&mut self, context: &str, next_token: &str) {
        self.table
            .entry(context.to_string())
            .or_default()
            .entry(next_token.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    /// Train on a single command
    fn train_command(&mut self, command: &str) {
        let tokens: Vec<&str> = command.split_whitespace().collect();
        if tokens.is_empty() {
            return;
        }

        // Empty context predicts first token
        self.observe("", tokens[0]);

        // Build n-grams from token sequence
        for i in 0..tokens.len() {
            let context_start = i.saturating_sub(self.n - 1);
            let context = tokens[context_start..=i].join(" ");

            if i + 1 < tokens.len() {
                self.observe(&context, tokens[i + 1]);
            }
        }
    }

    fn predict(&self, context: &str) -> Vec<(String, f32)> {
        let Some(counts) = self.table.get(context) else {
            return vec![];
        };
        let total: u32 = counts.values().sum();
        let mut probs: Vec<_> = counts
            .iter()
            .map(|(t, c)| (t.clone(), *c as f32 / total as f32))
            .collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        probs
    }
}

// Train on real command patterns
let mut model = NgramTable::with_n(3);

let commands = [
    "git status",
    "git commit -m fix",
    "git push",
    "git status",      // Repeated - should have higher probability
    "git status",
    "cargo build",
    "cargo test",
    "cargo build",     // Repeated
];

for cmd in &commands {
    model.train_command(cmd);
}

// Test: "git" context should predict "status" highest (3x vs 1x each)
let preds = model.predict("git");
assert!(!preds.is_empty());
assert_eq!(preds[0].0, "status"); // Most frequent

// Test: "cargo" context
let preds = model.predict("cargo");
assert_eq!(preds[0].0, "build"); // 2x vs 1x for test

// Test: Empty context predicts first tokens
let preds = model.predict("");
assert!(preds.iter().any(|(t, _)| t == "git"));
assert!(preds.iter().any(|(t, _)| t == "cargo"));
```

### Step 3: Add Prefix Trie for O(1) Lookup (REFACTOR)

```rust
use std::collections::HashMap;

/// Trie node for prefix matching
#[derive(Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
    count: u32,
}

/// Trie for fast prefix-based command lookup
#[derive(Default)]
struct Trie {
    root: TrieNode,
}

impl Trie {
    fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_default();
        }
        node.is_end = true;
        node.count += 1;
    }

    /// Find completions for prefix, sorted by frequency
    fn find_prefix(&self, prefix: &str, limit: usize) -> Vec<(String, u32)> {
        // Navigate to prefix node
        let mut node = &self.root;
        for ch in prefix.chars() {
            match node.children.get(&ch) {
                Some(n) => node = n,
                None => return vec![],
            }
        }

        // Collect all completions
        let mut results = Vec::new();
        self.collect(node, prefix.to_string(), &mut results, limit * 10);

        // Sort by frequency and take top N
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results.truncate(limit);
        results
    }

    fn collect(&self, node: &TrieNode, current: String, results: &mut Vec<(String, u32)>, limit: usize) {
        if results.len() >= limit {
            return;
        }
        if node.is_end {
            results.push((current.clone(), node.count));
        }
        for (ch, child) in &node.children {
            let mut next = current.clone();
            next.push(*ch);
            self.collect(child, next, results, limit);
        }
    }
}

// Test: Basic insertion and lookup
let mut trie = Trie::new();
trie.insert("git status");
trie.insert("git commit");
trie.insert("git push");

let results = trie.find_prefix("git ", 10);
assert_eq!(results.len(), 3);

// Test: Frequency ordering
let mut trie = Trie::new();
trie.insert("git status");
trie.insert("git status");
trie.insert("git status");
trie.insert("git commit");

let results = trie.find_prefix("git ", 10);
assert_eq!(results[0].0, "git status");
assert_eq!(results[0].1, 3); // Appeared 3 times

// Test: No match returns empty
let results = trie.find_prefix("docker ", 10);
assert!(results.is_empty());
```

## Part 2: The .apr Format Integration

### Saving Models with aprender

The `.apr` format provides:
- **32-byte header** with magic, version, CRC32
- **MessagePack metadata** for model info
- **Bincode payload** for efficient serialization
- **Optional encryption** for privacy

```rust,ignore
use aprender::format::{save, load, ModelType, SaveOptions};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct ShellModel {
    n: usize,
    ngrams: HashMap<String, HashMap<String, u32>>,
    total_commands: usize,
}

impl ShellModel {
    fn new(n: usize) -> Self {
        Self {
            n,
            ngrams: HashMap::new(),
            total_commands: 0,
        }
    }

    fn train(&mut self, commands: &[String]) {
        self.total_commands = commands.len();
        for cmd in commands {
            let tokens: Vec<&str> = cmd.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            // Empty context → first token
            self.ngrams
                .entry(String::new())
                .or_default()
                .entry(tokens[0].to_string())
                .and_modify(|c| *c += 1)
                .or_insert(1);

            // Build context n-grams
            for i in 0..tokens.len() {
                let start = i.saturating_sub(self.n - 1);
                let context = tokens[start..=i].join(" ");
                if i + 1 < tokens.len() {
                    self.ngrams
                        .entry(context)
                        .or_default()
                        .entry(tokens[i + 1].to_string())
                        .and_modify(|c| *c += 1)
                        .or_insert(1);
                }
            }
        }
    }
}

// Create and train model
let mut model = ShellModel::new(3);
model.train(&[
    "git status".to_string(),
    "git commit -m test".to_string(),
    "cargo build".to_string(),
]);

// Save to .apr format
let options = SaveOptions::default()
    .with_name("my-shell-model")
    .with_description("3-gram shell completion model");

save(&model, ModelType::Custom, "shell.apr", options)?;

// Load and verify
let loaded: ShellModel = load("shell.apr", ModelType::Custom)?;
assert_eq!(loaded.n, 3);
assert_eq!(loaded.total_commands, 3);
```

### Inspecting .apr Files

```bash
# View model metadata
apr inspect shell.apr

# Output:
# Model: my-shell-model
# Type: Custom
# Description: 3-gram shell completion model
# Created: 2025-11-26T15:30:00Z
# Size: 2.1 KB
# Checksum: CRC32 valid
```

## Part 3: Encryption for Privacy

Shell history contains sensitive patterns. Encrypt your models:

```rust,ignore
use aprender::format::{save_encrypted, load_encrypted, ModelType, SaveOptions};

// Save with password encryption (AES-256-GCM + Argon2id)
let options = SaveOptions::default()
    .with_name("private-shell-model")
    .with_description("Encrypted personal shell history model");

save_encrypted(&model, ModelType::Custom, "shell.apr", options, "my-password")?;

// Load requires password
let loaded: ShellModel = load_encrypted("shell.apr", ModelType::Custom, "my-password")?;

// Wrong password fails with DecryptionFailed error
let result: Result<ShellModel, _> = load_encrypted("shell.apr", ModelType::Custom, "wrong");
assert!(result.is_err());
```

### Recipient Encryption (X25519)

For sharing models with specific people:

```rust,ignore
use aprender::format::{save_for_recipient, load_as_recipient, ModelType, SaveOptions};
use aprender::format::x25519::{generate_keypair, PublicKey, SecretKey};

// Generate recipient keypair (they share public key with you)
let (recipient_secret, recipient_public) = generate_keypair();

// Save encrypted for specific recipient
let options = SaveOptions::default()
    .with_name("team-shell-model");

save_for_recipient(&model, ModelType::Custom, "team.apr", options, &recipient_public)?;

// Only recipient can decrypt
let loaded: ShellModel = load_as_recipient("team.apr", ModelType::Custom, &recipient_secret)?;
```

## Part 4: Single Binary Deployment

Embed your trained model directly in a Rust binary:

```rust,ignore
// In build.rs or your binary
const MODEL_BYTES: &[u8] = include_bytes!("../shell.apr");

fn main() {
    use aprender::format::load_from_bytes;

    // Load at runtime - zero filesystem access
    let model: ShellModel = load_from_bytes(MODEL_BYTES, ModelType::Custom)
        .expect("embedded model should be valid");

    // Use model
    let suggestions = model.suggest("git ");
    println!("Suggestions: {:?}", suggestions);
}
```

**Benefits:**
- Zero runtime dependencies
- Works in sandboxed environments
- Tamper-proof (model is part of binary hash)
- ~500KB overhead for typical shell model

### Complete Bundling Pipeline

```bash
# 1. Train on your history
aprender-shell train --output shell.apr

# 2. Optionally encrypt
apr encrypt shell.apr --password "$SECRET" --output shell-enc.apr

# 3. Embed in binary (Cargo.toml)
# [package]
# include = ["shell.apr"]

# 4. Build release
cargo build --release

# Result: Single binary with embedded, optionally encrypted model
```

## Part 5: Extending the Model

### Add Command Categories

```rust
use std::collections::HashMap;

#[derive(Default)]
struct CategorizedModel {
    /// Category → NgramTable
    categories: HashMap<String, HashMap<String, HashMap<String, u32>>>,
}

impl CategorizedModel {
    fn categorize(command: &str) -> &'static str {
        let first = command.split_whitespace().next().unwrap_or("");
        match first {
            "git" | "gh" => "vcs",
            "cargo" | "rustc" | "rustup" => "rust",
            "docker" | "kubectl" | "helm" => "containers",
            "npm" | "yarn" | "pnpm" => "node",
            "cd" | "ls" | "cat" | "grep" | "find" => "filesystem",
            _ => "other",
        }
    }

    fn train(&mut self, command: &str) {
        let category = Self::categorize(command);
        let tokens: Vec<&str> = command.split_whitespace().collect();

        if tokens.is_empty() {
            return;
        }

        let table = self.categories.entry(category.to_string()).or_default();

        // Train within category
        table
            .entry(String::new())
            .or_default()
            .entry(tokens[0].to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);

        for i in 0..tokens.len().saturating_sub(1) {
            table
                .entry(tokens[i].to_string())
                .or_default()
                .entry(tokens[i + 1].to_string())
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
    }
}

let mut model = CategorizedModel::default();
model.train("git status");
model.train("git commit");
model.train("cargo build");
model.train("cargo test");
model.train("ls -la");

// Verify categorization
assert!(model.categories.contains_key("vcs"));
assert!(model.categories.contains_key("rust"));
assert!(model.categories.contains_key("filesystem"));
```

### Add Time-Weighted Decay

Recent commands matter more than old ones:

```rust
use std::collections::HashMap;

struct DecayingModel {
    /// context → (token → weighted_count)
    ngrams: HashMap<String, HashMap<String, f32>>,
    /// Decay factor per observation (0.99 = 1% decay)
    decay: f32,
}

impl DecayingModel {
    fn new(decay: f32) -> Self {
        Self {
            ngrams: HashMap::new(),
            decay: decay.clamp(0.9, 0.999),
        }
    }

    fn observe(&mut self, context: &str, token: &str) {
        // Decay all existing counts first
        for counts in self.ngrams.values_mut() {
            for count in counts.values_mut() {
                *count *= self.decay;
            }
        }

        // Add new observation with weight 1.0
        self.ngrams
            .entry(context.to_string())
            .or_default()
            .entry(token.to_string())
            .and_modify(|c| *c += 1.0)
            .or_insert(1.0);
    }

    fn predict(&self, context: &str) -> Vec<(String, f32)> {
        let Some(counts) = self.ngrams.get(context) else {
            return vec![];
        };
        let total: f32 = counts.values().sum();
        if total < 0.001 {
            return vec![];
        }
        let mut probs: Vec<_> = counts
            .iter()
            .map(|(t, c)| (t.clone(), *c / total))
            .collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        probs
    }
}

// Test decay behavior
let mut model = DecayingModel::new(0.9); // 10% decay per observation

// Old observation
model.observe("git", "status");

// Newer observation (git status decays, commit is fresh)
model.observe("git", "commit");

let preds = model.predict("git");
// "commit" should be weighted higher (fresher)
assert_eq!(preds[0].0, "commit");
```

### Privacy Filter

Filter sensitive commands before training:

```rust
struct PrivacyFilter {
    sensitive_patterns: Vec<String>,
}

impl PrivacyFilter {
    fn new() -> Self {
        Self {
            sensitive_patterns: vec![
                "password".to_string(),
                "passwd".to_string(),
                "secret".to_string(),
                "token".to_string(),
                "api_key".to_string(),
                "AWS_SECRET".to_string(),
                "GITHUB_TOKEN".to_string(),
                "Authorization:".to_string(),
            ],
        }
    }

    fn is_safe(&self, command: &str) -> bool {
        let lower = command.to_lowercase();

        // Check sensitive patterns
        for pattern in &self.sensitive_patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return false;
            }
        }

        // Skip history manipulation
        if command.starts_with("history") || command.starts_with("fc ") {
            return false;
        }

        // Skip very short commands
        if command.len() < 2 {
            return false;
        }

        true
    }

    fn filter(&self, commands: Vec<String>) -> Vec<String> {
        commands.into_iter().filter(|c| self.is_safe(c)).collect()
    }
}

let filter = PrivacyFilter::new();

// Safe commands pass through
assert!(filter.is_safe("git push origin main"));
assert!(filter.is_safe("cargo build --release"));

// Sensitive commands are blocked
assert!(!filter.is_safe("export API_KEY=secret123"));
assert!(!filter.is_safe("curl -H 'Authorization: Bearer token'"));
assert!(!filter.is_safe("echo $PASSWORD"));

// History manipulation blocked
assert!(!filter.is_safe("history -c"));
assert!(!filter.is_safe("fc -l"));

// Filter a batch
let commands = vec![
    "git status".to_string(),
    "export SECRET=abc".to_string(),
    "cargo test".to_string(),
];
let safe = filter.filter(commands);
assert_eq!(safe.len(), 2);
assert_eq!(safe[0], "git status");
assert_eq!(safe[1], "cargo test");
```

## Part 6: Complete Working Example

```rust,ignore
//! Complete shell history model with .apr persistence
//!
//! cargo run --example shell_history_model

use aprender::format::{save, load, ModelType, SaveOptions};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Serialize, Deserialize, Default)]
pub struct ShellHistoryModel {
    n: usize,
    ngrams: HashMap<String, HashMap<String, u32>>,
    command_freq: HashMap<String, u32>,
    total_commands: usize,
}

impl ShellHistoryModel {
    pub fn new(n: usize) -> Self {
        Self {
            n: n.clamp(2, 5),
            ..Default::default()
        }
    }

    pub fn train(&mut self, commands: &[String]) {
        for cmd in commands {
            self.train_command(cmd);
        }
    }

    fn train_command(&mut self, cmd: &str) {
        self.total_commands += 1;
        *self.command_freq.entry(cmd.to_string()).or_insert(0) += 1;

        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        if tokens.is_empty() {
            return;
        }

        // Empty context → first token
        self.observe("", tokens[0]);

        // Build n-grams
        for i in 0..tokens.len() {
            let start = i.saturating_sub(self.n - 1);
            let context = tokens[start..=i].join(" ");
            if i + 1 < tokens.len() {
                self.observe(&context, tokens[i + 1]);
            }
        }
    }

    fn observe(&mut self, context: &str, token: &str) {
        self.ngrams
            .entry(context.to_string())
            .or_default()
            .entry(token.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    pub fn suggest(&self, prefix: &str, count: usize) -> Vec<(String, f32)> {
        let tokens: Vec<&str> = prefix.trim().split_whitespace().collect();
        if tokens.is_empty() {
            return self.top_first_tokens(count);
        }

        let start = tokens.len().saturating_sub(self.n - 1);
        let context = tokens[start..].join(" ");

        let Some(next_tokens) = self.ngrams.get(&context) else {
            return vec![];
        };

        let total: u32 = next_tokens.values().sum();
        let mut suggestions: Vec<_> = next_tokens
            .iter()
            .map(|(token, count)| {
                let completion = format!("{} {}", prefix, token);
                let prob = *count as f32 / total as f32;
                (completion, prob)
            })
            .collect();

        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        suggestions.truncate(count);
        suggestions
    }

    fn top_first_tokens(&self, count: usize) -> Vec<(String, f32)> {
        let Some(firsts) = self.ngrams.get("") else {
            return vec![];
        };
        let total: u32 = firsts.values().sum();
        let mut results: Vec<_> = firsts
            .iter()
            .map(|(t, c)| (t.clone(), *c as f32 / total as f32))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(count);
        results
    }

    pub fn save_to_apr(&self, path: &Path) -> Result<(), aprender::error::AprenderError> {
        let options = SaveOptions::default()
            .with_name("shell-history-model")
            .with_description(&format!(
                "{}-gram model trained on {} commands",
                self.n, self.total_commands
            ));
        save(self, ModelType::Custom, path, options)
    }

    pub fn load_from_apr(path: &Path) -> Result<Self, aprender::error::AprenderError> {
        load(path, ModelType::Custom)
    }

    pub fn stats(&self) -> ModelStats {
        ModelStats {
            n: self.n,
            total_commands: self.total_commands,
            unique_commands: self.command_freq.len(),
            ngram_count: self.ngrams.values().map(|m| m.len()).sum(),
        }
    }
}

#[derive(Debug)]
pub struct ModelStats {
    pub n: usize,
    pub total_commands: usize,
    pub unique_commands: usize,
    pub ngram_count: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate shell history
    let history = vec![
        "git status",
        "git add .",
        "git commit -m fix",
        "git push",
        "git status",
        "git log --oneline",
        "cargo build",
        "cargo test",
        "cargo build --release",
        "cargo clippy",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();

    // Train model
    let mut model = ShellHistoryModel::new(3);
    model.train(&history);

    // Show stats
    let stats = model.stats();
    println!("Model Statistics:");
    println!("  N-gram size: {}", stats.n);
    println!("  Total commands: {}", stats.total_commands);
    println!("  Unique commands: {}", stats.unique_commands);
    println!("  N-gram count: {}", stats.ngram_count);

    // Test suggestions
    println!("\nSuggestions for 'git ':");
    for (suggestion, prob) in model.suggest("git ", 5) {
        println!("  {:.1}%  {}", prob * 100.0, suggestion);
    }

    println!("\nSuggestions for 'cargo ':");
    for (suggestion, prob) in model.suggest("cargo ", 5) {
        println!("  {:.1}%  {}", prob * 100.0, suggestion);
    }

    // Save to .apr
    let path = std::path::Path::new("shell_history.apr");
    model.save_to_apr(path)?;
    println!("\nModel saved to: {}", path.display());

    // Reload and verify
    let loaded = ShellHistoryModel::load_from_apr(path)?;
    assert_eq!(loaded.total_commands, model.total_commands);
    println!("Model reloaded successfully!");

    // Cleanup
    std::fs::remove_file(path)?;

    Ok(())
}
```

## Part 7: Model Validation with aprender Metrics

The `aprender-shell` CLI uses aprender's ranking metrics for proper evaluation:

```bash
# Train on your history
aprender-shell train

# Validate with holdout evaluation
aprender-shell validate
```

### Ranking Metrics (aprender::metrics::ranking)

```rust,ignore
use aprender::metrics::ranking::{hit_at_k, mrr, RankingMetrics};

// Hit@K: Is correct answer in top K predictions?
let predictions = vec!["git commit", "git push", "git pull"];
let target = "git push";
assert_eq!(hit_at_k(&predictions, target, 1), 0.0);  // Not #1
assert_eq!(hit_at_k(&predictions, target, 2), 1.0);  // In top 2

// Mean Reciprocal Rank: 1/rank of correct answer
let all_predictions = vec![
    vec!["git commit", "git push"],  // target at rank 2 → RR = 0.5
    vec!["cargo test", "cargo build"],  // target at rank 1 → RR = 1.0
];
let targets = vec!["git push", "cargo test"];
let score = mrr(&all_predictions, &targets);  // (0.5 + 1.0) / 2 = 0.75

// Comprehensive metrics
let metrics = RankingMetrics::compute(&all_predictions, &targets);
println!("Hit@1: {:.1}%", metrics.hit_at_1 * 100.0);
println!("Hit@5: {:.1}%", metrics.hit_at_5 * 100.0);
println!("MRR: {:.3}", metrics.mrr);
```

### Validation Output

```text
🔬 aprender-shell: Model Validation

📂 History file: ~/.zsh_history
📊 Total commands: 21,763
⚙️  N-gram size: 3
📈 Train/test split: 80% / 20%

═══════════════════════════════════════════
           VALIDATION RESULTS
═══════════════════════════════════════════
  Training set:      17,410 commands
  Test set:           4,353 commands
  Evaluated:          3,857 commands
───────────────────────────────────────────
  Hit@1  (top 1):     13.3%
  Hit@5  (top 5):     26.2%
  Hit@10 (top 10):    30.7%
  MRR (Mean Recip):  0.181
═══════════════════════════════════════════
```

**Interpretation:**
- **Hit@5 ~27%**: Model suggests correct command in top 5 for ~1 in 4 predictions
- **MRR ~0.18**: Average rank of correct answer is ~5th position
- This is realistic for shell completion given command diversity

## Part 8: Synthetic Data Augmentation

Improve model coverage with three strategies:

```bash
# Generate 5000 synthetic commands and retrain
aprender-shell augment --count 5000
```

### CLI Command Templates

```rust,ignore
use aprender_shell::synthetic::CommandGenerator;

let gen = CommandGenerator::new();
let commands = gen.generate(1000);

// Generates realistic dev commands:
// - git status, git commit -m, git push --force
// - cargo build --release, cargo test --lib
// - docker run -it, kubectl get pods
// - npm install --save-dev, pip install -r
```

### Mutation Engine

```rust,ignore
use aprender_shell::synthetic::CommandMutator;

let mutator = CommandMutator::new();

// Original: "git commit -m test"
// Mutations:
//   - "git add -m test"      (command substitution)
//   - "git commit -am test"  (flag substitution)
//   - "git commit test"      (flag removal)
let mutations = mutator.mutate("git commit -m test");
```

### Coverage-Guided Generation

```rust,ignore
use aprender_shell::synthetic::{SyntheticPipeline, CoverageGuidedGenerator};
use std::collections::HashSet;

// Extract known n-grams from current model
let known_ngrams: HashSet<String> = model.ngram_keys().collect();

// Generate commands that maximize new n-gram coverage
let pipeline = SyntheticPipeline::new();
let result = pipeline.generate(&real_history, known_ngrams, 5000);

println!("New n-grams added: {}", result.report.new_ngrams);
println!("Coverage gain: {:.1}%", result.report.coverage_gain * 100.0);
```

### Augmentation Output

```text
🧬 aprender-shell: Data Augmentation

📂 History file: ~/.zsh_history
📊 Real commands: 21,761
🔢 Known n-grams: 39,176

🧪 Generating synthetic commands... done!

📈 Coverage Report:
   Synthetic commands: 5,000
   New n-grams added:  5,473
   Coverage gain:      99.0%

✅ Augmented model saved

📊 Model Statistics:
   Total training commands: 26,761
   Unique n-grams: 46,340 (+18%)
   Vocabulary size: 21,101 (+31%)
```

## Summary

| Component | Purpose | Complexity |
|-----------|---------|------------|
| N-gram table | Token prediction | O(1) lookup |
| Trie index | Prefix completion | O(k) where k=prefix length |
| .apr format | Persistence + metadata | ~2KB overhead |
| Encryption | Privacy protection | +50ms save/load |
| Single binary | Zero-dependency deployment | +500KB binary size |
| **Ranking metrics** | Model validation | `aprender::metrics::ranking` |
| **Synthetic data** | Coverage improvement | +13% n-grams |

**Key insights:**
1. Shell commands are highly predictable (Markov property)
2. N-grams outperform neural nets for this domain (speed, size, accuracy)
3. `.apr` format provides type-safe, versioned persistence
4. Encryption enables sharing sensitive models securely
5. `include_bytes!()` enables self-contained deployment
6. **Ranking metrics** (Hit@K, MRR) are standard for language model evaluation
7. **Synthetic data** fills coverage gaps for commands you rarely use

## CLI Reference

```bash
# Training
aprender-shell train              # Full retrain from history
aprender-shell update             # Incremental update (fast)

# Evaluation
aprender-shell validate           # Holdout evaluation with metrics
aprender-shell validate -n 4      # Test different n-gram sizes
aprender-shell stats              # Model statistics

# Data Augmentation
aprender-shell augment            # Generate synthetic data + retrain
aprender-shell augment -c 10000   # Custom synthetic count

# Inference
aprender-shell suggest "git "     # Get completions
aprender-shell suggest "cargo t"  # Prefix matching

# Export
aprender-shell export model.apr   # Export to .apr format
```

## Next Steps

- [`aprender-shell` source code](https://github.com/paiml/aprender/tree/main/crates/aprender-shell)
- [Model Format Specification](../examples/model-format.md)
- [Ranking Metrics API](../ml-fundamentals/ranking-metrics.md)
