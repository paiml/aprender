
impl CodeEda {
    /// Create a new code EDA generator with the given configuration.
    #[must_use]
    pub fn new(config: CodeEdaConfig) -> Self {
        let reserved = Self::get_reserved_keywords(config.language);
        Self {
            config,
            synonyms: VariableSynonyms::default(),
            reserved,
        }
    }

    /// Get reserved keywords for a language.
    fn get_reserved_keywords(lang: CodeLanguage) -> HashSet<String> {
        let keywords: &[&str] = match lang {
            CodeLanguage::Rust => &[
                "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else",
                "enum", "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match",
                "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
                "super", "trait", "true", "type", "unsafe", "use", "where", "while", "abstract",
                "become", "box", "do", "final", "macro", "override", "priv", "try", "typeof",
                "unsized", "virtual", "yield",
            ],
            CodeLanguage::Python => &[
                "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
                "continue", "def", "del", "elif", "else", "except", "finally", "for", "from",
                "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
                "raise", "return", "try", "while", "with", "yield",
            ],
            CodeLanguage::Generic => &[],
        };
        keywords.iter().map(|s| (*s).to_string()).collect()
    }

    /// Augment a single code sample.
    ///
    /// # Arguments
    ///
    /// * `code` - Source code to augment
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Augmented code string.
    #[must_use]
    pub fn augment(&self, code: &str, seed: u64) -> String {
        let tokens = self.tokenize(code);
        if tokens.len() < self.config.min_tokens {
            return code.to_string();
        }

        let mut result_tokens = tokens.clone();
        let mut rng_state = seed;

        // Apply operations based on probability
        if self.random_f32(&mut rng_state) < self.config.rename_prob {
            result_tokens = self.apply_variable_rename(&result_tokens, &mut rng_state);
        }

        if self.random_f32(&mut rng_state) < self.config.comment_prob {
            result_tokens = self.apply_comment_insertion(&result_tokens, &mut rng_state);
        }

        if self.random_f32(&mut rng_state) < self.config.reorder_prob {
            result_tokens = self.apply_statement_reorder(&result_tokens, &mut rng_state);
        }

        if self.random_f32(&mut rng_state) < self.config.remove_prob {
            result_tokens = self.apply_dead_code_removal(&result_tokens);
        }

        result_tokens.join("")
    }

    /// Simple tokenization preserving whitespace and structure.
    #[allow(clippy::unused_self)]
    fn tokenize(&self, code: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in code.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(ch.to_string());
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Apply variable renaming operation.
    fn apply_variable_rename(&self, tokens: &[String], rng: &mut u64) -> Vec<String> {
        let mut result = Vec::with_capacity(tokens.len());
        let mut rename_map: HashMap<String, String> = HashMap::new();

        for token in tokens {
            // Check if token is an identifier (not reserved, alphanumeric start)
            if self.is_identifier(token) && !self.reserved.contains(token) {
                if let Some(synonyms) = self.synonyms.get(token) {
                    // Use cached rename or pick new one
                    let renamed = rename_map.entry(token.clone()).or_insert_with(|| {
                        let idx = (self.random_u64(rng) as usize) % synonyms.len();
                        synonyms[idx].clone()
                    });
                    result.push(renamed.clone());
                } else {
                    result.push(token.clone());
                }
            } else {
                result.push(token.clone());
            }
        }

        result
    }

    /// Apply comment insertion operation.
    fn apply_comment_insertion(&self, tokens: &[String], rng: &mut u64) -> Vec<String> {
        let mut result = Vec::with_capacity(tokens.len() + 2);

        let comments: &[&str] = match self.config.language {
            CodeLanguage::Rust => &["// REVIEW: pending", "// SAFETY: checked", "/* temp */"],
            CodeLanguage::Python => &["# REVIEW: pending", "# NOTE: temp", "# type: ignore"],
            CodeLanguage::Generic => &["/* comment */"],
        };

        // Find a newline to insert comment after
        let mut inserted = false;
        for token in tokens {
            result.push(token.clone());
            if token == "\n" && !inserted && self.random_f32(rng) < 0.5 {
                let idx = (self.random_u64(rng) as usize) % comments.len();
                result.push(comments[idx].to_string());
                result.push("\n".to_string());
                inserted = true;
            }
        }

        result
    }

    /// Apply statement reorder operation (swap adjacent statements).
    fn apply_statement_reorder(&self, tokens: &[String], rng: &mut u64) -> Vec<String> {
        // Find statement boundaries (semicolons for Rust, newlines for Python)
        let delimiter = match self.config.language {
            CodeLanguage::Rust => ";",
            CodeLanguage::Python | CodeLanguage::Generic => "\n",
        };

        // Split into statements
        let mut statements: Vec<Vec<String>> = Vec::new();
        let mut current_stmt: Vec<String> = Vec::new();

        for token in tokens {
            current_stmt.push(token.clone());
            if token == delimiter {
                statements.push(current_stmt.clone());
                current_stmt.clear();
            }
        }
        if !current_stmt.is_empty() {
            statements.push(current_stmt);
        }

        // Swap two adjacent statements if we have enough
        if statements.len() >= 2 {
            let idx = (self.random_u64(rng) as usize) % (statements.len() - 1);
            statements.swap(idx, idx + 1);
        }

        statements.into_iter().flatten().collect()
    }

    /// Apply dead code removal (remove comments and extra whitespace).
    #[allow(clippy::unused_self)]
    fn apply_dead_code_removal(&self, tokens: &[String]) -> Vec<String> {
        let mut result = Vec::with_capacity(tokens.len());
        let mut in_comment = false;
        let mut prev_was_whitespace = false;
        let mut prev_was_slash = false;

        for token in tokens {
            // Detect // comment start (two consecutive slashes)
            if token == "/" {
                if prev_was_slash {
                    // This is the second slash, start comment
                    in_comment = true;
                    prev_was_slash = false;
                    // Remove the first slash we already added
                    if result.last() == Some(&"/".to_string()) {
                        result.pop();
                    }
                    continue;
                }
                prev_was_slash = true;
                if !in_comment {
                    result.push(token.clone());
                }
                continue;
            }

            // Reset slash tracking for non-slash tokens
            prev_was_slash = false;

            // Detect # comment start (Python)
            if token == "#" {
                in_comment = true;
                continue;
            }

            // End single-line comment on newline
            if in_comment && token == "\n" {
                in_comment = false;
                result.push(token.clone());
                continue;
            }

            if in_comment {
                continue;
            }

            // Collapse multiple whitespace
            let is_whitespace = token.chars().all(char::is_whitespace);
            if is_whitespace {
                if !prev_was_whitespace {
                    result.push(token.clone());
                }
                prev_was_whitespace = true;
            } else {
                result.push(token.clone());
                prev_was_whitespace = false;
            }
        }

        result
    }

    /// Check if token is a valid identifier.
    #[allow(clippy::unused_self)]
    fn is_identifier(&self, token: &str) -> bool {
        if token.is_empty() {
            return false;
        }
        let mut chars = token.chars();
        let first = chars.next().unwrap_or('0');
        (first.is_alphabetic() || first == '_') && chars.all(|c| c.is_alphanumeric() || c == '_')
    }

    /// Simple PRNG for reproducibility.
    #[allow(clippy::unused_self)]
    fn random_u64(&self, state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *state
    }

    /// Random f32 in [0, 1).
    fn random_f32(&self, state: &mut u64) -> f32 {
        (self.random_u64(state) as f32) / (u64::MAX as f32)
    }

    /// Calculate token overlap between two code strings.
    #[must_use]
    pub fn token_overlap(&self, a: &str, b: &str) -> f32 {
        let tokens_a: HashSet<_> = self.tokenize(a).into_iter().collect();
        let tokens_b: HashSet<_> = self.tokenize(b).into_iter().collect();

        if tokens_a.is_empty() || tokens_b.is_empty() {
            return 0.0;
        }

        let intersection = tokens_a.intersection(&tokens_b).count();
        let union = tokens_a.union(&tokens_b).count();

        intersection as f32 / union as f32
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &CodeEdaConfig {
        &self.config
    }
}

impl SyntheticGenerator for CodeEda {
    type Input = String;
    type Output = String;

    fn generate(
        &self,
        seeds: &[Self::Input],
        config: &SyntheticConfig,
    ) -> Result<Vec<Self::Output>> {
        let target_count = ((seeds.len() as f32) * config.augmentation_ratio).ceil() as usize;
        let mut results = Vec::with_capacity(target_count);
        let seed = config.seed;

        for (idx, code) in seeds.iter().enumerate() {
            let num_augments = (target_count / seeds.len().max(1)).max(1);
            for aug_idx in 0..num_augments {
                let aug_seed = seed.wrapping_add((idx * 1000 + aug_idx) as u64);
                let augmented = self.augment(code, aug_seed);

                // Check quality threshold
                let quality = self.quality_score(&augmented, code);
                if quality >= config.quality_threshold {
                    results.push(augmented);
                }

                if results.len() >= target_count {
                    break;
                }
            }
            if results.len() >= target_count {
                break;
            }
        }

        Ok(results)
    }

    fn quality_score(&self, generated: &Self::Output, seed: &Self::Input) -> f32 {
        // Quality based on token overlap (semantic preservation)
        let overlap = self.token_overlap(generated, seed);

        // Penalize if too similar (no augmentation) or too different (corrupted)
        // Ideal is 0.6-0.9 overlap
        if overlap > 0.95 {
            0.5 // Too similar, little augmentation
        } else if overlap < 0.3 {
            0.3 // Too different, likely corrupted
        } else {
            overlap
        }
    }

    fn diversity_score(&self, batch: &[Self::Output]) -> f32 {
        if batch.len() <= 1 {
            return 1.0;
        }

        // Calculate pairwise token overlap
        let mut total_overlap = 0.0;
        let mut pairs = 0;

        for i in 0..batch.len() {
            for j in (i + 1)..batch.len() {
                total_overlap += self.token_overlap(&batch[i], &batch[j]);
                pairs += 1;
            }
        }

        if pairs == 0 {
            return 1.0;
        }

        // Diversity is inverse of average overlap
        1.0 - (total_overlap / pairs as f32)
    }
}
