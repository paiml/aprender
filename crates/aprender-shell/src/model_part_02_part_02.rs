impl MarkovModel {
impl MarkovModel {
    /// Create a new model with given n-gram size
    pub fn new(n: usize) -> Self {
        Self {
            n: n.clamp(2, 5),
            ngrams: HashMap::new(),
            command_freq: HashMap::new(),
            trie: Some(Trie::new()),
            total_commands: 0,
            last_trained_pos: 0,
        }
    }

    /// Train on a list of commands
    pub fn train(&mut self, commands: &[String]) {
        self.total_commands = commands.len();

        for cmd in commands {
            // Track command frequency
            *self.command_freq.entry(cmd.clone()).or_insert(0) += 1;

            // Add to trie
            if let Some(ref mut trie) = self.trie {
                trie.insert(cmd);
            }

            // Tokenize command
            let tokens: Vec<&str> = cmd.split_whitespace().collect();

            if tokens.is_empty() {
                continue;
            }

            // Build n-grams
            // For "git commit -m", with n=3:
            //   "" -> "git"
            //   "git" -> "commit"
            //   "git commit" -> "-m"

            // Empty context predicts first token
            self.ngrams
                .entry(String::new())
                .or_default()
                .entry(tokens[0].to_string())
                .and_modify(|c| *c += 1)
                .or_insert(1);

            // Build context n-grams
            for i in 0..tokens.len() {
                // Context is up to n-1 previous tokens
                let context_start = i.saturating_sub(self.n - 1);
                let context: String = tokens[context_start..=i].join(" ");

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

        self.last_trained_pos = self.total_commands;
    }

    /// Incrementally train on new commands (appends to existing model)
    pub fn train_incremental(&mut self, commands: &[String]) {
        for cmd in commands {
            self.total_commands += 1;

            // Track command frequency
            *self.command_freq.entry(cmd.clone()).or_insert(0) += 1;

            // Add to trie
            if let Some(ref mut trie) = self.trie {
                trie.insert(cmd);
            }

            // Tokenize command
            let tokens: Vec<&str> = cmd.split_whitespace().collect();

            if tokens.is_empty() {
                continue;
            }

            // Empty context predicts first token
            self.ngrams
                .entry(String::new())
                .or_default()
                .entry(tokens[0].to_string())
                .and_modify(|c| *c += 1)
                .or_insert(1);

            // Build context n-grams
            for i in 0..tokens.len() {
                let context_start = i.saturating_sub(self.n - 1);
                let context: String = tokens[context_start..=i].join(" ");

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

        self.last_trained_pos = self.total_commands;
    }

    /// Get the last trained position in history
    pub fn last_trained_position(&self) -> usize {
        self.last_trained_pos
    }

    /// Get total commands trained on
    pub fn total_commands(&self) -> usize {
        self.total_commands
    }

    /// Suggest completions for a prefix
    ///
    /// Optimized for minimal allocations (Issue #93):
    /// - Pre-allocated vectors with capacity
    /// - HashSet for O(1) duplicate detection
    /// - Reused string buffers where possible
    pub fn suggest(&self, prefix: &str, count: usize) -> Vec<(String, f32)> {
        let prefix = prefix.trim();
        let tokens: Vec<&str> = prefix.split_whitespace().collect();
        let ends_with_space = prefix.is_empty() || prefix.ends_with(' ');

        // Pre-allocate with expected capacity (reduces brk syscalls)
        let capacity = count * 4;
        let mut suggestions = Vec::with_capacity(capacity);
        let mut seen = std::collections::HashSet::with_capacity(capacity);

        // Strategy 1: Trie prefix match for exact commands
        if let Some(ref trie) = self.trie {
            for cmd in trie.find_prefix(prefix, capacity) {
                // Filter corrupted commands
                if Self::is_corrupted_command(&cmd) {
                    continue;
                }

                let freq = self.command_freq.get(&cmd).copied().unwrap_or(1);
                let score = freq as f32 / self.total_commands.max(1) as f32;
                seen.insert(cmd.clone());
                suggestions.push((cmd, score));
            }
        }

        // Strategy 2: N-gram prediction for next token (only when prefix ends with space)
        if !tokens.is_empty() && ends_with_space {
            let context_start = tokens.len().saturating_sub(self.n - 1);
            // Pre-compute context string once
            let context = tokens[context_start..].join(" ");
            let prefix_trimmed = prefix.trim();

            if let Some(next_tokens) = self.ngrams.get(&context) {
                let total: u32 = next_tokens.values().sum();

                // Pre-allocate completion buffer
                let mut completion = String::with_capacity(prefix_trimmed.len() + 32);

                for (token, ngram_count) in next_tokens {
                    // Reuse buffer instead of format!()
                    completion.clear();
                    completion.push_str(prefix_trimmed);
                    completion.push(' ');
                    completion.push_str(token);

                    let score = *ngram_count as f32 / total as f32;

                    // O(1) duplicate check with HashSet
                    if !seen.contains(&completion) {
                        seen.insert(completion.clone());
                        suggestions.push((completion.clone(), score * 0.8));
                    }
                }
            }
        }

        // Strategy 3: N-gram prediction with partial token filter (when NOT ending with space)
        if !tokens.is_empty() && !ends_with_space && tokens.len() >= 2 {
            let partial_token = tokens.last().unwrap_or(&"");
            let context_tokens = &tokens[..tokens.len() - 1];
            let context_start = context_tokens.len().saturating_sub(self.n - 1);
            // Pre-compute context strings once
            let context = context_tokens[context_start..].join(" ");
            let context_prefix = context_tokens.join(" ");

            if let Some(next_tokens) = self.ngrams.get(&context) {
                let total: u32 = next_tokens.values().sum();

                // Pre-allocate completion buffer
                let mut completion = String::with_capacity(context_prefix.len() + 32);

                for (token, ngram_count) in next_tokens {
                    // Only include tokens that start with the partial input
                    // AND are not corrupted tokens
                    if token.starts_with(partial_token) && !Self::is_corrupted_token(token) {
                        // Reuse buffer instead of format!()
                        completion.clear();
                        completion.push_str(&context_prefix);
                        completion.push(' ');
                        completion.push_str(token);

                        let score = *ngram_count as f32 / total as f32;

                        // O(1) duplicate check with HashSet
                        if !seen.contains(&completion) {
                            seen.insert(completion.clone());
                            suggestions.push((completion.clone(), score * 0.9));
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

    /// Detect corrupted commands that shouldn't be suggested.
    ///
    /// Common patterns (Issue #92):
    /// - "git commit-m" (missing space before flag)
    /// - "cargo build-r" (missing space before flag)
    /// - "gitr push" (typo in command - space merged with next word)
    /// - "cargo-lambda  help" (double spaces)
    /// - "git rm -r --cached vendor/\\" (trailing backslash)
    fn is_corrupted_command(cmd: &str) -> bool {
        // Check for double spaces
        if cmd.contains("  ") {
            return true;
        }

        // Check for trailing backslash (incomplete multiline)
        if cmd.trim_end().ends_with('\\') {
            return true;
        }

        // Check for trailing escape sequences
        if cmd.trim_end().ends_with("\\\\") {
            return true;
        }

        // Check first word is a valid command (Issue #92)
        // Valid commands start with letter and contain only alphanumeric + underscore + hyphen
        if let Some(first) = cmd.split_whitespace().next() {
            // Reject if first word is a common command with extra chars (typos)
            // e.g., "gitr", "giti", "gits", "cargoo"
            let typo_patterns = [
                ("git", &["gitr", "giti", "gits", "gitl", "gitp"][..]),
                ("cargo", &["cargoo", "cargos", "cargob"][..]),
                ("docker", &["dockerr", "dockers"][..]),
                ("npm", &["npmi", "npmr"][..]),
            ];

            for (valid, typos) in typo_patterns {
                if typos.contains(&first) {
                    return true;
                }
                // Also catch typos where space merged (e.g., "gi tpush" from "git push")
                if first.len() < valid.len() && valid.starts_with(first) {
                    // "gi" is valid prefix, but check second word for merged space
                    let tokens: Vec<&str> = cmd.split_whitespace().collect();
                    if tokens.len() >= 2 {
                        let second = tokens[1];
                        // Check if second word looks like "tpush" (merged space)
                        if second.len() > 1
                            && !second.starts_with('-')
                            && valid.ends_with(&first[first.len().saturating_sub(1)..])
                        {
                            let expected_start = &valid[first.len()..];
                            if second.starts_with(expected_start) && expected_start.len() == 1 {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        cmd.split_whitespace().any(Self::is_corrupted_token)
    }

    /// Detect corrupted individual tokens.
    ///
    /// Checks for patterns like "commit-m", "add-A" where a subcommand
    /// has a flag incorrectly attached without a space.
    fn is_corrupted_token(token: &str) -> bool {
        // Check for pattern: word-singlechar or word--word
        if let Some(dash_pos) = token.find('-') {
            if dash_pos > 0 && dash_pos < token.len() - 1 {
                let before = &token[..dash_pos];
                let after = &token[dash_pos + 1..];

                // Common git/cargo subcommands that shouldn't have flags attached
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

    /// Save model to .apr file
    ///
    /// Uses `ModelType::NgramLm` (0x10) for proper classification (QA report fix).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let options = SaveOptions::default()
            .with_name("aprender-shell")
            .with_description(format!(
                "{}-gram shell completion model ({} commands)",
                self.n, self.total_commands
            ));

        // Use NgramLm type for Markov n-gram models (QA report: was 0xFF Custom, now 0x10 NgramLm)
        format::save(self, ModelType::NgramLm, path, options)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Load model from .apr file using memory-mapped I/O
    ///
    /// Uses mmap for zero-copy loading, reducing syscalls from ~970 to <50
    /// (see bundle-mmap-spec.md Section 8).
    ///
    /// Supports both `NgramLm` (new) and `Custom` (legacy) model types for backward compatibility.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        // Try NgramLm first (new format), fall back to Custom (legacy) for backward compatibility
        let mut model: Self = format::load_mmap(path, ModelType::NgramLm)
            .or_else(|_| format::load_mmap(path, ModelType::Custom))
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        // Rebuild trie (not serialized)
        let mut trie = Trie::new();
        for cmd in model.command_freq.keys() {
            trie.insert(cmd);
        }
        model.trie = Some(trie);

        Ok(model)
    }

    /// Save model with AES-256-GCM encryption (spec §4.1.2)
    ///
    /// Uses Argon2id for key derivation from password.
    /// The model can only be loaded with the correct password.
    pub fn save_encrypted(&self, path: &Path, password: &str) -> std::io::Result<()> {
        let options = SaveOptions::default()
            .with_name("aprender-shell")
            .with_description(format!(
                "{}-gram encrypted shell completion model ({} commands)",
                self.n, self.total_commands
            ));

        format::save_encrypted(self, ModelType::NgramLm, path, options, password)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Load encrypted model from .apr file (spec §4.1.2)
    ///
    /// Requires the same password used during encryption.
    /// Returns an error if the password is incorrect.
    pub fn load_encrypted(path: &Path, password: &str) -> std::io::Result<Self> {
        let mut model: Self = format::load_encrypted(path, ModelType::NgramLm, password)
            .or_else(|_| format::load_encrypted(path, ModelType::Custom, password))
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        // Rebuild trie (not serialized)
        let mut trie = Trie::new();
        for cmd in model.command_freq.keys() {
            trie.insert(cmd);
        }
        model.trie = Some(trie);

        Ok(model)
    }

    /// Check if a model file is encrypted
    pub fn is_encrypted(path: &Path) -> std::io::Result<bool> {
        let info = format::inspect(path).map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(info.encrypted)
    }

    /// Save model with zstd compression (Tier 2)
    ///
    /// Achieves ~14x size reduction with minimal decompression overhead (~10-20ms).
    /// Actually faster in practice due to reduced I/O.
    #[cfg(feature = "format-compression")]
    pub fn save_compressed(&self, path: &Path) -> std::io::Result<()> {
        use aprender::format::Compression;

        let options = SaveOptions::default()
            .with_name("aprender-shell")
            .with_description(format!(
                "{}-gram compressed shell completion model ({} commands)",
                self.n, self.total_commands
            ))
            .with_compression(Compression::ZstdDefault);

        format::save(self, ModelType::NgramLm, path, options)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Save model with both compression and encryption (Tier 2+3)
    ///
    /// Best of both worlds: small size and protection.
    #[cfg(all(feature = "format-compression", feature = "format-encryption"))]
    pub fn save_compressed_encrypted(&self, path: &Path, password: &str) -> std::io::Result<()> {
        use aprender::format::Compression;

        let options = SaveOptions::default()
            .with_name("aprender-shell")
            .with_description(format!(
                "{}-gram compressed+encrypted shell completion model ({} commands)",
                self.n, self.total_commands
            ))
            .with_compression(Compression::ZstdDefault);

        format::save_encrypted(self, ModelType::NgramLm, path, options, password)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    /// Check if a model file is compressed
    ///
    /// Returns true if payload_size < uncompressed_size (compression was applied)
    #[cfg(feature = "format-compression")]
    pub fn is_compressed(path: &Path) -> std::io::Result<bool> {
        let info = format::inspect(path).map_err(|e| std::io::Error::other(e.to_string()))?;
        // If payload is smaller than uncompressed size, compression was used
        Ok(info.payload_size < info.uncompressed_size)
    }
}
}
