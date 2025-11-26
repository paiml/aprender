//! Memory-Paged Markov Model for Large Shell Histories
//!
//! Uses aprender's bundle module for efficient memory management when
//! dealing with large shell histories that exceed available RAM.

use aprender::bundle::{BundleBuilder, BundleConfig, PagedBundle, PagingConfig, PagingStats};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::trie::Trie;

/// Minimum memory limit (1MB)
const MIN_MEMORY_LIMIT: usize = 1024 * 1024;

/// N-gram segment for paged storage.
///
/// Each segment contains n-grams for a specific context prefix,
/// allowing on-demand loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramSegment {
    /// Context prefix this segment covers (e.g., "git", "cargo")
    pub prefix: String,
    /// N-gram data: context -> (next_token -> count)
    pub ngrams: HashMap<String, HashMap<String, u32>>,
    /// Size estimate in bytes
    pub size_bytes: usize,
}

impl NgramSegment {
    /// Create a new empty segment.
    #[must_use]
    pub fn new(prefix: String) -> Self {
        Self {
            prefix,
            ngrams: HashMap::new(),
            size_bytes: 0,
        }
    }

    /// Add an n-gram to this segment.
    pub fn add(&mut self, context: String, next_token: String, count: u32) {
        let entry = self.ngrams.entry(context).or_default();
        *entry.entry(next_token).or_insert(0) += count;
        self.update_size();
    }

    /// Update size estimate.
    fn update_size(&mut self) {
        self.size_bytes = self
            .ngrams
            .iter()
            .map(|(k, v)| k.len() + v.keys().map(|k2| k2.len() + 4).sum::<usize>())
            .sum();
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Simple binary format: prefix_len(4) + prefix + ngram_count(4) + ngrams
        let mut bytes = Vec::new();

        // Prefix
        let prefix_bytes = self.prefix.as_bytes();
        bytes.extend(&(prefix_bytes.len() as u32).to_le_bytes());
        bytes.extend(prefix_bytes);

        // N-gram count
        bytes.extend(&(self.ngrams.len() as u32).to_le_bytes());

        for (context, next_tokens) in &self.ngrams {
            // Context
            let ctx_bytes = context.as_bytes();
            bytes.extend(&(ctx_bytes.len() as u32).to_le_bytes());
            bytes.extend(ctx_bytes);

            // Next tokens count
            bytes.extend(&(next_tokens.len() as u32).to_le_bytes());

            for (token, count) in next_tokens {
                // Token
                let tok_bytes = token.as_bytes();
                bytes.extend(&(tok_bytes.len() as u32).to_le_bytes());
                bytes.extend(tok_bytes);
                // Count
                bytes.extend(&count.to_le_bytes());
            }
        }

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let mut pos = 0;

        // Helper to read 4 bytes as u32
        let read_u32 = |data: &[u8], offset: usize| -> std::io::Result<u32> {
            let slice = data
                .get(offset..offset + 4)
                .ok_or_else(|| std::io::Error::other("Truncated segment data"))?;
            let arr: [u8; 4] = slice
                .try_into()
                .map_err(|_| std::io::Error::other("Invalid byte slice"))?;
            Ok(u32::from_le_bytes(arr))
        };

        // Read prefix
        let prefix_len = read_u32(bytes, pos)? as usize;
        pos += 4;

        if bytes.len() < pos + prefix_len {
            return Err(std::io::Error::other("Truncated prefix"));
        }
        let prefix = String::from_utf8_lossy(&bytes[pos..pos + prefix_len]).to_string();
        pos += prefix_len;

        // Read n-gram count
        let ngram_count = read_u32(bytes, pos)? as usize;
        pos += 4;

        let mut ngrams = HashMap::with_capacity(ngram_count);

        for _ in 0..ngram_count {
            // Read context
            let ctx_len = read_u32(bytes, pos)? as usize;
            pos += 4;

            if bytes.len() < pos + ctx_len {
                return Err(std::io::Error::other("Truncated context"));
            }
            let context = String::from_utf8_lossy(&bytes[pos..pos + ctx_len]).to_string();
            pos += ctx_len;

            // Read next tokens count
            let token_count = read_u32(bytes, pos)? as usize;
            pos += 4;

            let mut next_tokens = HashMap::with_capacity(token_count);

            for _ in 0..token_count {
                // Read token
                let tok_len = read_u32(bytes, pos)? as usize;
                pos += 4;

                if bytes.len() < pos + tok_len {
                    return Err(std::io::Error::other("Truncated token"));
                }
                let token = String::from_utf8_lossy(&bytes[pos..pos + tok_len]).to_string();
                pos += tok_len;

                // Read count
                let count = read_u32(bytes, pos)?;
                pos += 4;

                next_tokens.insert(token, count);
            }

            ngrams.insert(context, next_tokens);
        }

        let mut segment = Self {
            prefix,
            ngrams,
            size_bytes: 0,
        };
        segment.update_size();
        Ok(segment)
    }
}

/// Model metadata stored in the bundle manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedModelMetadata {
    /// N-gram size
    pub n: usize,
    /// Total commands trained on
    pub total_commands: usize,
    /// Number of segments
    pub segment_count: usize,
    /// Command frequency map (kept in memory - relatively small)
    pub command_freq: HashMap<String, u32>,
    /// Segment prefixes for index lookup
    pub segment_prefixes: Vec<String>,
}

/// Memory-paged Markov model for shell command prediction.
///
/// Uses aprender's bundle module to store n-gram data on disk and
/// load segments on-demand, enabling handling of large shell histories
/// without exhausting RAM.
pub struct PagedMarkovModel {
    /// N-gram size
    n: usize,
    /// Memory limit in bytes
    memory_limit: usize,
    /// Metadata
    metadata: PagedModelMetadata,
    /// Paged bundle (when loaded from file)
    bundle: Option<PagedBundle>,
    /// In-memory segments (for training/small models)
    segments: HashMap<String, NgramSegment>,
    /// Prefix trie for fast lookup
    trie: Option<Trie>,
    /// Path to bundle file (if loaded)
    bundle_path: Option<std::path::PathBuf>,
}

impl PagedMarkovModel {
    /// Create a new paged model with given n-gram size and memory limit.
    ///
    /// # Arguments
    /// * `n` - N-gram size (2-5)
    /// * `memory_limit_mb` - Maximum memory usage in megabytes
    #[must_use]
    pub fn new(n: usize, memory_limit_mb: usize) -> Self {
        let memory_limit = (memory_limit_mb * 1024 * 1024).max(MIN_MEMORY_LIMIT);
        Self {
            n: n.clamp(2, 5),
            memory_limit,
            metadata: PagedModelMetadata {
                n,
                total_commands: 0,
                segment_count: 0,
                command_freq: HashMap::new(),
                segment_prefixes: Vec::new(),
            },
            bundle: None,
            segments: HashMap::new(),
            trie: Some(Trie::new()),
            bundle_path: None,
        }
    }

    /// Get memory limit in bytes.
    #[must_use]
    #[allow(dead_code)]
    pub fn memory_limit(&self) -> usize {
        self.memory_limit
    }

    /// Train on a list of commands.
    pub fn train(&mut self, commands: &[String]) {
        self.metadata.total_commands = commands.len();

        for cmd in commands {
            // Track command frequency
            *self.metadata.command_freq.entry(cmd.clone()).or_insert(0) += 1;

            // Add to trie
            if let Some(ref mut trie) = self.trie {
                trie.insert(cmd);
            }

            // Tokenize command
            let tokens: Vec<&str> = cmd.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            // Determine segment prefix (first token)
            let prefix = tokens[0].to_string();

            // Get or create segment
            let segment = self
                .segments
                .entry(prefix.clone())
                .or_insert_with(|| NgramSegment::new(prefix));

            // Empty context predicts first token
            segment.add(String::new(), tokens[0].to_string(), 1);

            // Build context n-grams
            for i in 0..tokens.len() {
                let context_start = i.saturating_sub(self.n - 1);
                let context: String = tokens[context_start..=i].join(" ");

                if i + 1 < tokens.len() {
                    segment.add(context, tokens[i + 1].to_string(), 1);
                }
            }
        }

        // Update metadata
        self.metadata.segment_count = self.segments.len();
        self.metadata.segment_prefixes = self.segments.keys().cloned().collect();
    }

    /// Save model to a paged bundle file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let path_str = path.to_string_lossy().to_string();

        // Add metadata as first model
        let metadata_bytes = serde_json::to_vec(&self.metadata)
            .map_err(|e| std::io::Error::other(format!("Failed to serialize metadata: {e}")))?;

        let mut builder = BundleBuilder::new(&path_str)
            .with_config(BundleConfig::new().with_compression(false))
            .add_model("metadata", metadata_bytes);

        // Add each segment as a separate model
        for (prefix, segment) in &self.segments {
            let segment_bytes = segment.to_bytes();
            builder = builder.add_model(format!("segment_{prefix}"), segment_bytes);
        }

        // Build and save
        builder
            .build()
            .map_err(|e| std::io::Error::other(format!("Failed to build bundle: {e}")))?;

        Ok(())
    }

    /// Load model from a paged bundle file with memory limit.
    pub fn load(path: &Path, memory_limit_mb: usize) -> std::io::Result<Self> {
        let memory_limit = (memory_limit_mb * 1024 * 1024).max(MIN_MEMORY_LIMIT);

        // Open as paged bundle
        let paging_config = PagingConfig::new()
            .with_max_memory(memory_limit)
            .with_prefetch(true);

        let mut bundle = PagedBundle::open(path, paging_config)
            .map_err(|e| std::io::Error::other(format!("Failed to open bundle: {e}")))?;

        // Load metadata (always in memory)
        let metadata_bytes = bundle
            .get_model("metadata")
            .map_err(|e| std::io::Error::other(format!("Failed to read metadata: {e}")))?;

        let metadata: PagedModelMetadata = serde_json::from_slice(metadata_bytes)
            .map_err(|e| std::io::Error::other(format!("Failed to parse metadata: {e}")))?;

        // Rebuild trie from command_freq
        let mut trie = Trie::new();
        for cmd in metadata.command_freq.keys() {
            trie.insert(cmd);
        }

        Ok(Self {
            n: metadata.n,
            memory_limit,
            metadata,
            bundle: Some(bundle),
            segments: HashMap::new(), // Loaded on demand
            trie: Some(trie),
            bundle_path: Some(path.to_path_buf()),
        })
    }

    /// Load a specific segment on demand.
    fn load_segment(&mut self, prefix: &str) -> std::io::Result<Option<NgramSegment>> {
        if let Some(segment) = self.segments.get(prefix) {
            return Ok(Some(segment.clone()));
        }

        if let Some(ref mut bundle) = self.bundle {
            let model_name = format!("segment_{prefix}");
            // Check if model exists by looking at model names
            if bundle.model_names().iter().any(|n| *n == model_name) {
                let bytes = bundle.get_model(&model_name).map_err(|e| {
                    std::io::Error::other(format!("Failed to read segment '{prefix}': {e}"))
                })?;
                let segment = NgramSegment::from_bytes(bytes)?;
                self.segments.insert(prefix.to_string(), segment.clone());
                return Ok(Some(segment));
            }
        }

        Ok(None)
    }

    /// Suggest completions for a prefix.
    pub fn suggest(&mut self, prefix: &str, count: usize) -> Vec<(String, f32)> {
        // Check for trailing space BEFORE trimming
        let ends_with_space = prefix.is_empty() || prefix.ends_with(' ');
        let prefix = prefix.trim();
        let tokens: Vec<&str> = prefix.split_whitespace().collect();

        let mut suggestions = Vec::new();

        // Strategy 1: Trie prefix match for exact commands
        if let Some(ref trie) = self.trie {
            for cmd in trie.find_prefix(prefix, count * 4) {
                let freq = self.metadata.command_freq.get(&cmd).copied().unwrap_or(1);
                let score = freq as f32 / self.metadata.total_commands.max(1) as f32;
                suggestions.push((cmd, score));
            }
        }

        // Strategy 2: N-gram prediction (load segment on demand)
        if !tokens.is_empty() && ends_with_space {
            let segment_prefix = tokens[0];

            // Load segment on demand
            if let Ok(Some(segment)) = self.load_segment(segment_prefix) {
                let context_start = tokens.len().saturating_sub(self.n - 1);
                let context = tokens[context_start..].join(" ");

                if let Some(next_tokens) = segment.ngrams.get(&context) {
                    let total: u32 = next_tokens.values().sum();

                    for (token, ngram_count) in next_tokens {
                        let completion = format!("{} {}", prefix.trim(), token);
                        let score = *ngram_count as f32 / total as f32;

                        if !suggestions.iter().any(|(s, _)| s == &completion) {
                            suggestions.push((completion, score * 0.8));
                        }
                    }
                }
            }
        }

        // Sort by score and take top count
        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(count);

        suggestions
    }

    /// Get model statistics.
    #[must_use]
    pub fn stats(&self) -> PagedModelStats {
        let loaded_segments = self.segments.len();
        let total_segments = self.metadata.segment_count;
        let loaded_bytes: usize = self.segments.values().map(|s| s.size_bytes).sum();

        PagedModelStats {
            n: self.n,
            total_commands: self.metadata.total_commands,
            vocab_size: self.metadata.command_freq.len(),
            total_segments,
            loaded_segments,
            memory_limit: self.memory_limit,
            loaded_bytes,
            bundle_path: self.bundle_path.clone(),
        }
    }

    /// Get paging statistics from the bundle.
    pub fn paging_stats(&self) -> Option<PagingStats> {
        self.bundle.as_ref().map(|b| b.stats().clone())
    }

    /// Hint that a segment will be needed soon (for prefetching).
    #[allow(dead_code)]
    pub fn prefetch_hint(&mut self, prefix: &str) {
        if let Some(ref mut bundle) = self.bundle {
            let _ = bundle.prefetch_hint(&format!("segment_{prefix}"));
        }
    }

    /// Total commands trained on.
    #[must_use]
    #[allow(dead_code)]
    pub fn total_commands(&self) -> usize {
        self.metadata.total_commands
    }

    /// N-gram size.
    #[must_use]
    #[allow(dead_code)]
    pub fn ngram_size(&self) -> usize {
        self.n
    }

    /// Vocabulary size.
    #[must_use]
    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        self.metadata.command_freq.len()
    }

    /// Top commands by frequency.
    #[must_use]
    pub fn top_commands(&self, count: usize) -> Vec<(String, u32)> {
        let mut cmds: Vec<_> = self
            .metadata
            .command_freq
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        cmds.sort_by(|a, b| b.1.cmp(&a.1));
        cmds.truncate(count);
        cmds
    }
}

/// Statistics for a paged model.
#[derive(Debug, Clone)]
pub struct PagedModelStats {
    /// N-gram size
    pub n: usize,
    /// Total commands trained on
    pub total_commands: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Total number of segments
    pub total_segments: usize,
    /// Number of loaded segments
    pub loaded_segments: usize,
    /// Memory limit in bytes
    pub memory_limit: usize,
    /// Currently loaded bytes
    pub loaded_bytes: usize,
    /// Path to bundle file (if loaded from file)
    pub bundle_path: Option<std::path::PathBuf>,
}

impl std::fmt::Display for PagedModelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Paged Model Statistics:")?;
        writeln!(f, "  N-gram size:      {}", self.n)?;
        writeln!(f, "  Total commands:   {}", self.total_commands)?;
        writeln!(f, "  Vocabulary size:  {}", self.vocab_size)?;
        writeln!(
            f,
            "  Segments:         {}/{} loaded",
            self.loaded_segments, self.total_segments
        )?;
        writeln!(
            f,
            "  Memory limit:     {:.1} MB",
            self.memory_limit as f64 / 1024.0 / 1024.0
        )?;
        writeln!(
            f,
            "  Loaded bytes:     {:.1} KB",
            self.loaded_bytes as f64 / 1024.0
        )?;
        if let Some(ref path) = self.bundle_path {
            writeln!(f, "  Bundle path:      {}", path.display())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_ngram_segment_new() {
        let segment = NgramSegment::new("git".to_string());
        assert_eq!(segment.prefix, "git");
        assert!(segment.ngrams.is_empty());
        assert_eq!(segment.size_bytes, 0);
    }

    #[test]
    fn test_ngram_segment_add() {
        let mut segment = NgramSegment::new("git".to_string());
        segment.add("git".to_string(), "commit".to_string(), 1);
        segment.add("git".to_string(), "commit".to_string(), 1);
        segment.add("git".to_string(), "push".to_string(), 1);

        assert_eq!(segment.ngrams.len(), 1);
        let git_nexts = segment.ngrams.get("git").unwrap();
        assert_eq!(git_nexts.get("commit"), Some(&2));
        assert_eq!(git_nexts.get("push"), Some(&1));
    }

    #[test]
    fn test_ngram_segment_serialization() {
        let mut segment = NgramSegment::new("cargo".to_string());
        segment.add("cargo".to_string(), "build".to_string(), 5);
        segment.add("cargo".to_string(), "test".to_string(), 3);
        segment.add("cargo build".to_string(), "--release".to_string(), 2);

        let bytes = segment.to_bytes();
        let restored = NgramSegment::from_bytes(&bytes).unwrap();

        assert_eq!(restored.prefix, "cargo");
        assert_eq!(restored.ngrams.len(), 2);
        assert_eq!(restored.ngrams.get("cargo").unwrap().get("build"), Some(&5));
        assert_eq!(restored.ngrams.get("cargo").unwrap().get("test"), Some(&3));
        assert_eq!(
            restored.ngrams.get("cargo build").unwrap().get("--release"),
            Some(&2)
        );
    }

    #[test]
    fn test_paged_model_new() {
        let model = PagedMarkovModel::new(3, 10);
        assert_eq!(model.ngram_size(), 3);
        assert!(model.memory_limit() >= MIN_MEMORY_LIMIT);
    }

    #[test]
    fn test_paged_model_train() {
        let commands = vec![
            "git status".to_string(),
            "git commit -m test".to_string(),
            "git push".to_string(),
            "cargo build".to_string(),
            "cargo test".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);

        assert_eq!(model.total_commands(), 5);
        assert_eq!(model.vocab_size(), 5);

        // Should have git and cargo segments
        assert!(model.segments.contains_key("git"));
        assert!(model.segments.contains_key("cargo"));
    }

    #[test]
    fn test_paged_model_suggest() {
        let commands = vec![
            "git status".to_string(),
            "git status".to_string(),
            "git commit -m fix".to_string(),
            "git push".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);

        let suggestions = model.suggest("git ", 3);
        assert!(!suggestions.is_empty());

        // status appears twice, should be suggested
        let has_status = suggestions.iter().any(|(s, _)| s.contains("status"));
        assert!(has_status);
    }

    #[test]
    fn test_paged_model_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.apbundle");

        // Create and train model
        let commands = vec![
            "git status".to_string(),
            "git commit".to_string(),
            "cargo build".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);
        model.save(&path).unwrap();

        // Load model
        let mut loaded = PagedMarkovModel::load(&path, 10).unwrap();

        assert_eq!(loaded.total_commands(), 3);
        assert_eq!(loaded.vocab_size(), 3);
        assert_eq!(loaded.ngram_size(), 3);

        // Test suggestions work after load
        let suggestions = loaded.suggest("git ", 3);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_paged_model_stats() {
        let commands = vec![
            "git status".to_string(),
            "cargo build".to_string(),
            "docker run".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);

        let stats = model.stats();
        assert_eq!(stats.n, 3);
        assert_eq!(stats.total_commands, 3);
        assert_eq!(stats.vocab_size, 3);
        assert_eq!(stats.total_segments, 3); // git, cargo, docker
    }

    #[test]
    fn test_paged_model_on_demand_loading() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ondemand.apbundle");

        // Create model with many segments
        let commands = vec![
            "git status".to_string(),
            "git commit".to_string(),
            "cargo build".to_string(),
            "cargo test".to_string(),
            "docker run".to_string(),
            "kubectl get pods".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);
        model.save(&path).unwrap();

        // Load with small memory limit
        let mut loaded = PagedMarkovModel::load(&path, 1).unwrap();

        // Initially no segments loaded
        assert_eq!(loaded.stats().loaded_segments, 0);

        // Query git commands - should load git segment
        let _ = loaded.suggest("git ", 3);
        assert!(loaded.segments.contains_key("git"));

        // Query cargo commands - should load cargo segment
        let _ = loaded.suggest("cargo ", 3);
        assert!(loaded.segments.contains_key("cargo"));
    }

    #[test]
    fn test_paged_model_prefetch_hint() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("prefetch.apbundle");

        let commands = vec!["git status".to_string(), "cargo build".to_string()];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);
        model.save(&path).unwrap();

        let mut loaded = PagedMarkovModel::load(&path, 10).unwrap();

        // Hint that we'll need git segment
        loaded.prefetch_hint("git");

        // Should still work after hint
        let suggestions = loaded.suggest("git ", 3);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_paged_model_top_commands() {
        let commands = vec![
            "git status".to_string(),
            "git status".to_string(),
            "git status".to_string(),
            "cargo build".to_string(),
            "cargo build".to_string(),
            "docker run".to_string(),
        ];

        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&commands);

        let top = model.top_commands(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "git status");
        assert_eq!(top[0].1, 3);
        assert_eq!(top[1].0, "cargo build");
        assert_eq!(top[1].1, 2);
    }

    #[test]
    fn test_paged_model_empty_commands() {
        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&[]);

        assert_eq!(model.total_commands(), 0);
        assert_eq!(model.vocab_size(), 0);

        let suggestions = model.suggest("git ", 3);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_ngram_segment_empty_bytes() {
        let result = NgramSegment::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_paged_model_stats_display() {
        let mut model = PagedMarkovModel::new(3, 10);
        model.train(&["git status".to_string()]);

        let stats = model.stats();
        let display = format!("{stats}");

        assert!(display.contains("N-gram size:"));
        assert!(display.contains("Total commands:"));
        assert!(display.contains("Memory limit:"));
    }
}
