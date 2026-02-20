
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
    pub fn prefetch_hint(&mut self, prefix: &str) {
        if let Some(ref mut bundle) = self.bundle {
            let _ = bundle.prefetch_hint(&format!("segment_{prefix}"));
        }
    }

    /// Total commands trained on.
    #[must_use]
    pub fn total_commands(&self) -> usize {
        self.metadata.total_commands
    }

    /// N-gram size.
    #[must_use]
    pub fn ngram_size(&self) -> usize {
        self.n
    }

    /// Vocabulary size.
    #[must_use]
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
