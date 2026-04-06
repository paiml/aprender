//! Shell Command Autocomplete Demo
//!
//! Real WASM implementation using the trained aprender-shell-base.apr model.
//! Uses N-gram Markov model for command prediction.
//!
//! Spec: docs/specifications/showcase-demo-aprender-shell-apr.md

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// APR format header size
const HEADER_SIZE: usize = 32;

/// Shell command autocomplete using N-gram Markov model
#[derive(Debug)]
pub struct ShellAutocomplete {
    /// N-gram size (typically 3)
    n: usize,
    /// N-gram counts: context -> (next_token -> count)
    ngrams: HashMap<String, HashMap<String, u32>>,
    /// Command frequency for ranking
    command_freq: HashMap<String, u32>,
    /// Prefix trie for fast lookup
    trie: Trie,
    /// Total commands in training data
    total_commands: usize,
}

/// Simple trie for prefix matching
#[derive(Debug, Default)]
struct Trie {
    children: HashMap<char, Trie>,
    is_end: bool,
    command: Option<String>,
}

impl Trie {
    fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, word: &str) {
        let mut node = self;
        for c in word.chars() {
            node = node.children.entry(c).or_default();
        }
        node.is_end = true;
        node.command = Some(word.to_string());
    }

    fn find_prefix(&self, prefix: &str, limit: usize) -> Vec<String> {
        let mut results = Vec::new();
        let mut node = self;

        // Navigate to prefix node
        for c in prefix.chars() {
            match node.children.get(&c) {
                Some(child) => node = child,
                None => return results,
            }
        }

        // Collect all commands under this prefix
        Self::collect_commands_recursive(node, &mut results, limit);
        results
    }

    fn collect_commands_recursive(node: &Trie, results: &mut Vec<String>, limit: usize) {
        if results.len() >= limit {
            return;
        }
        if let Some(ref cmd) = node.command {
            results.push(cmd.clone());
        }
        for child in node.children.values() {
            Self::collect_commands_recursive(child, results, limit);
            if results.len() >= limit {
                return;
            }
        }
    }
}

/// Serialized model format (bincode)
#[derive(Debug, Serialize, Deserialize)]
struct MarkovModelData {
    n: usize,
    ngrams: HashMap<String, HashMap<String, u32>>,
    command_freq: HashMap<String, u32>,
    total_commands: usize,
    #[serde(default)]
    last_trained_pos: usize,
}

/// Embedded model for testing and convenience
const SHELL_MODEL_BYTES: &[u8] = include_bytes!("../../assets/aprender-shell-base.apr");

impl ShellAutocomplete {
    /// Create a new ShellAutocomplete with the embedded model.
    ///
    /// This is a convenience method for testing and demos that loads
    /// the model compiled into the binary.
    pub fn new() -> Result<Self, String> {
        Self::load_from_bytes(SHELL_MODEL_BYTES)
    }

    /// Load ShellAutocomplete from raw .apr bytes.
    /// This is the primary method for loading the model.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, String> {
        // Verify magic bytes and minimum size
        if bytes.len() < HEADER_SIZE {
            return Err("Model file too small".to_string());
        }
        if &bytes[0..4] != b"APRN" {
            return Err(format!("Invalid magic bytes: {:?}", &bytes[0..4]));
        }

        // Parse 32-byte APR header
        // Offset 0-3: Magic "APRN"
        // Offset 4-5: Version (major, minor)
        // Offset 6-7: Model type (u16 LE)
        // Offset 8-11: Metadata size (u32 LE)
        // Offset 12-15: Payload size (u32 LE)
        // Offset 16-19: Uncompressed size (u32 LE)
        // Offset 20: Compression type
        // Offset 21: Flags
        // Offset 22-31: Reserved

        let metadata_size = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let payload_size =
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let compression = bytes[20];

        // Calculate offsets
        let metadata_start = HEADER_SIZE;
        let metadata_end = metadata_start + metadata_size;
        let payload_start = metadata_end;
        let payload_end = payload_start + payload_size;

        if payload_end > bytes.len() {
            return Err(format!(
                "Payload extends beyond file: {} > {}",
                payload_end,
                bytes.len()
            ));
        }

        let payload_compressed = &bytes[payload_start..payload_end];

        // Decompress payload if needed
        let payload_decompressed: Vec<u8> = match compression {
            0x00 => payload_compressed.to_vec(), // No compression
            #[cfg(feature = "shell-autocomplete")]
            0x01 | 0x02 => {
                // Zstd compression
                zstd::decode_all(payload_compressed)
                    .map_err(|e| format!("Failed to decompress: {}", e))?
            }
            #[cfg(not(feature = "shell-autocomplete"))]
            0x01 | 0x02 => {
                return Err(
                    "Zstd compression requires the 'shell-autocomplete' feature".to_string()
                );
            }
            _ => return Err(format!("Unknown compression type: 0x{:02X}", compression)),
        };

        // Deserialize the model data with bincode
        let model_data: MarkovModelData = bincode::deserialize(&payload_decompressed)
            .map_err(|e| format!("Failed to deserialize model: {}", e))?;

        // Build trie from commands
        let mut trie = Trie::new();
        for cmd in model_data.command_freq.keys() {
            trie.insert(cmd);
        }

        Ok(Self {
            n: model_data.n,
            ngrams: model_data.ngrams,
            command_freq: model_data.command_freq,
            trie,
            total_commands: model_data.total_commands,
        })
    }

    /// Suggest completions for a prefix
    pub fn suggest(&self, prefix: &str, count: usize) -> Vec<(String, f32)> {
        let prefix = prefix.trim();
        let tokens: Vec<&str> = prefix.split_whitespace().collect();
        let ends_with_space = prefix.is_empty() || prefix.ends_with(' ');

        let capacity = count * 4;
        let mut suggestions = Vec::with_capacity(capacity);
        let mut seen = std::collections::HashSet::with_capacity(capacity);

        // Strategy 1: Trie prefix match for exact commands
        for cmd in self.trie.find_prefix(prefix, capacity) {
            if Self::is_corrupted_command(&cmd) {
                continue;
            }
            let freq = self.command_freq.get(&cmd).copied().unwrap_or(1);
            let score = freq as f32 / self.total_commands.max(1) as f32;
            seen.insert(cmd.clone());
            suggestions.push((cmd, score));
        }

        // Strategy 2: N-gram prediction for next token (only when prefix ends with space)
        if !tokens.is_empty() && ends_with_space {
            let context_start = tokens.len().saturating_sub(self.n - 1);
            let context = tokens[context_start..].join(" ");
            let prefix_trimmed = prefix.trim();

            if let Some(next_tokens) = self.ngrams.get(&context) {
                let total: u32 = next_tokens.values().sum();
                let mut completion = String::with_capacity(prefix_trimmed.len() + 32);

                for (token, ngram_count) in next_tokens {
                    completion.clear();
                    completion.push_str(prefix_trimmed);
                    completion.push(' ');
                    completion.push_str(token);

                    let score = *ngram_count as f32 / total as f32;

                    if !seen.contains(&completion) {
                        seen.insert(completion.clone());
                        suggestions.push((completion.clone(), score * 0.8));
                    }
                }
            }
        }

        // Strategy 3: N-gram prediction with partial token filter
        if !tokens.is_empty() && !ends_with_space && tokens.len() >= 2 {
            let partial_token = tokens.last().unwrap_or(&"");
            let context_tokens = &tokens[..tokens.len() - 1];
            let context_start = context_tokens.len().saturating_sub(self.n - 1);
            let context = context_tokens[context_start..].join(" ");
            let context_prefix = context_tokens.join(" ");

            if let Some(next_tokens) = self.ngrams.get(&context) {
                let total: u32 = next_tokens.values().sum();
                let mut completion = String::with_capacity(context_prefix.len() + 32);

                for (token, ngram_count) in next_tokens {
                    if token.starts_with(partial_token) && !Self::is_corrupted_token(token) {
                        completion.clear();
                        completion.push_str(&context_prefix);
                        completion.push(' ');
                        completion.push_str(token);

                        let score = *ngram_count as f32 / total as f32;

                        if !seen.contains(&completion) {
                            seen.insert(completion.clone());
                            suggestions.push((completion.clone(), score * 0.9));
                        }
                    }
                }
            }
        }

        // If no prefix and no suggestions, return top commands
        if prefix.is_empty() && suggestions.is_empty() {
            let mut top_cmds: Vec<_> = self
                .command_freq
                .iter()
                .map(|(k, v)| (k.clone(), *v as f32 / self.total_commands.max(1) as f32))
                .collect();
            top_cmds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            suggestions = top_cmds;
        }

        // Sort by score and truncate
        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(count);

        suggestions
    }

    /// Detect corrupted commands
    fn is_corrupted_command(cmd: &str) -> bool {
        if cmd.contains("  ") {
            return true;
        }
        if cmd.trim_end().ends_with('\\') {
            return true;
        }
        cmd.split_whitespace().any(Self::is_corrupted_token)
    }

    /// Detect corrupted tokens
    fn is_corrupted_token(token: &str) -> bool {
        if let Some(dash_pos) = token.find('-') {
            if dash_pos > 0 && dash_pos < token.len() - 1 {
                let before = &token[..dash_pos];
                let after = &token[dash_pos + 1..];
                let subcommands = [
                    "commit", "checkout", "clone", "push", "pull", "merge", "rebase", "status",
                    "add", "build", "run", "test", "install",
                ];
                if subcommands.contains(&before) && (after.len() <= 2 || after.starts_with('-')) {
                    return true;
                }
            }
        }
        false
    }

    /// Get JSON-formatted suggestions (for WASM interop)
    pub fn suggest_json(&self, prefix: &str, count: usize) -> String {
        let suggestions = self.suggest(prefix, count);
        let items: Vec<_> = suggestions
            .iter()
            .map(|(text, score)| {
                format!(
                    r#"{{"text":"{}","score":{:.4}}}"#,
                    text.replace('"', "\\\""),
                    score
                )
            })
            .collect();
        format!(r#"{{"suggestions":[{}]}}"#, items.join(","))
    }

    /// Get model info as JSON
    pub fn model_info_json(&self) -> String {
        format!(
            r#"{{"model_name":"aprender-shell-base","model_type":"ngram_lm","vocab_size":{},"ngram_size":{},"ngram_count":{},"total_commands":{}}}"#,
            self.vocab_size(),
            self.n,
            self.ngram_count(),
            self.total_commands
        )
    }

    /// Vocabulary size (unique commands)
    pub fn vocab_size(&self) -> usize {
        self.command_freq.len()
    }

    /// N-gram count
    pub fn ngram_count(&self) -> usize {
        self.ngrams.values().map(HashMap::len).sum()
    }

    /// N-gram size
    pub fn ngram_size(&self) -> usize {
        self.n
    }

    /// Estimated memory usage
    pub fn estimated_memory_bytes(&self) -> usize {
        let ngram_size: usize = self
            .ngrams
            .iter()
            .map(|(k, v)| k.len() + v.keys().map(|k2| k2.len() + 4).sum::<usize>())
            .sum();
        let vocab_size: usize = self.command_freq.keys().map(|k| k.len() + 4).sum();
        ngram_size + vocab_size + std::mem::size_of::<Self>()
    }
}

// ============================================================================ //
// WASM EXPORTS - Browser-accessible API //
// ============================================================================ //

/// WASM-exported shell autocomplete demo
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct ShellAutocompleteDemo {
    inner: ShellAutocomplete,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl ShellAutocompleteDemo {
    /// Create a new ShellAutocompleteDemo from bytes fetched by JavaScript.
    ///
    /// This is the preferred constructor for dynamic model loading:
    /// ```js
    /// const response = await fetch('./models/shell.apr');
    /// const bytes = new Uint8Array(await response.arrayBuffer());
    /// const demo = ShellAutocompleteDemo.from_bytes(bytes);
    /// ```
    #[wasm_bindgen(js_name = "fromBytes")]
    pub fn from_bytes(bytes: &[u8]) -> Result<ShellAutocompleteDemo, JsValue> {
        console_error_panic_hook::set_once();

        let inner =
            ShellAutocomplete::load_from_bytes(bytes).map_err(|e| JsValue::from_str(e.as_str()))?;

        web_sys::console::log_1(
            &format!(
                "ShellAutocomplete loaded from bytes: {} commands, {} n-grams",
                inner.vocab_size(),
                inner.ngram_count()
            )
            .into(),
        );

        Ok(Self { inner })
    }

    /// Create with embedded model (for demos/testing).
    ///
    /// Uses the model compiled into the WASM binary.
    /// This constructor is primarily for testing and quick demos where the model
    /// is hardcoded into the WASM bundle via `include_bytes!`.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<ShellAutocompleteDemo, JsValue> {
        console_error_panic_hook::set_once();

        let inner = ShellAutocomplete::new().map_err(|e| JsValue::from_str(&e))?;

        web_sys::console::log_1(
            &format!(
                "ShellAutocomplete loaded (embedded): {} commands, {} n-grams",
                inner.vocab_size(),
                inner.ngram_count()
            )
            .into(),
        );

        Ok(Self { inner })
    }

    /// Get suggestions for a prefix (returns JSON)
    #[wasm_bindgen]
    pub fn suggest(&self, prefix: &str, count: usize) -> String {
        self.inner.suggest_json(prefix, count)
    }

    /// Get model info as JSON
    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        self.inner.model_info_json()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Get n-gram count
    pub fn ngram_count(&self) -> usize {
        self.inner.ngram_count()
    }

    /// Get n-gram size (n value)
    pub fn ngram_size(&self) -> usize {
        self.inner.ngram_size()
    }

    /// Get estimated memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.inner.estimated_memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_basic() {
        let mut trie = Trie::new();
        trie.insert("git status");
        trie.insert("git commit");
        trie.insert("cargo build");

        let results = trie.find_prefix("git", 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_corrupted_detection() {
        assert!(ShellAutocomplete::is_corrupted_command("git commit-m"));
        assert!(!ShellAutocomplete::is_corrupted_command("git commit -m"));
        assert!(!ShellAutocomplete::is_corrupted_command(
            "git checkout feature-branch"
        ));
    }
}
