
impl EdaGenerator {
    /// Create a new EDA generator with configuration.
    #[must_use]
    pub fn new(config: EdaConfig) -> Self {
        Self {
            config,
            synonyms: SynonymDict::default(),
        }
    }

    /// Create a new EDA generator with custom synonym dictionary.
    #[must_use]
    pub fn with_synonyms(config: EdaConfig, synonyms: SynonymDict) -> Self {
        Self { config, synonyms }
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &EdaConfig {
        &self.config
    }

    /// Get the synonym dictionary.
    #[must_use]
    pub fn synonyms(&self) -> &SynonymDict {
        &self.synonyms
    }

    /// Augment a single text input.
    ///
    /// Returns multiple augmented versions based on `config.num_augments`.
    #[must_use]
    pub fn augment(&self, text: &str, seed: u64) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();

        if words.len() < self.config.min_words {
            return vec![text.to_string()];
        }

        let mut results = Vec::with_capacity(self.config.num_augments);
        let mut rng = SimpleRng::new(seed);

        for _ in 0..self.config.num_augments {
            let mut augmented = words.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();

            // Apply operations with configured probabilities
            if rng.next_f32() < self.config.synonym_prob {
                augmented = self.synonym_replacement(&augmented, &mut rng);
            }
            if rng.next_f32() < self.config.insert_prob {
                augmented = self.random_insertion(&augmented, &mut rng);
            }
            if rng.next_f32() < self.config.swap_prob {
                augmented = self.random_swap(&augmented, &mut rng);
            }
            if rng.next_f32() < self.config.delete_prob {
                augmented = self.random_deletion(&augmented, &mut rng);
            }

            let result = augmented.join(" ");
            if !result.is_empty() && result != text {
                results.push(result);
            }
        }

        // Ensure we return at least the original if no augmentations applied
        if results.is_empty() {
            results.push(text.to_string());
        }

        results
    }

    /// Synonym replacement: Replace n words with synonyms.
    fn synonym_replacement(&self, words: &[String], rng: &mut SimpleRng) -> Vec<String> {
        let mut result = words.to_vec();
        let n = (words.len() as f32 * self.config.synonym_prob).ceil() as usize;

        for _ in 0..n {
            if result.is_empty() {
                break;
            }
            let idx = rng.next_usize(result.len());
            if let Some(syn) = self.synonyms.random_synonym(&result[idx], rng.next()) {
                result[idx] = syn.to_string();
            }
        }

        result
    }

    /// Random insertion: Insert n random synonyms.
    fn random_insertion(&self, words: &[String], rng: &mut SimpleRng) -> Vec<String> {
        let mut result = words.to_vec();
        let n = (words.len() as f32 * self.config.insert_prob).ceil() as usize;

        let dict_words = self.synonyms.words();
        if dict_words.is_empty() {
            return result;
        }

        for _ in 0..n {
            // Pick a random word from dictionary
            let word_idx = rng.next_usize(dict_words.len());
            let word = dict_words[word_idx];

            // Get a random synonym
            if let Some(syn) = self.synonyms.random_synonym(word, rng.next()) {
                // Insert at random position
                let pos = rng.next_usize(result.len() + 1);
                result.insert(pos, syn.to_string());
            }
        }

        result
    }

    /// Random swap: Swap n pairs of words.
    fn random_swap(&self, words: &[String], rng: &mut SimpleRng) -> Vec<String> {
        if words.len() < 2 {
            return words.to_vec();
        }

        let mut result = words.to_vec();
        let n = (words.len() as f32 * self.config.swap_prob).ceil() as usize;

        for _ in 0..n.max(1) {
            let i = rng.next_usize(result.len());
            let j = rng.next_usize(result.len());
            if i != j {
                result.swap(i, j);
            }
        }

        result
    }

    /// Random deletion: Delete words with probability p.
    fn random_deletion(&self, words: &[String], rng: &mut SimpleRng) -> Vec<String> {
        if words.len() <= 1 {
            return words.to_vec();
        }

        let result: Vec<String> = words
            .iter()
            .filter(|_| rng.next_f32() > self.config.delete_prob)
            .cloned()
            .collect();

        // Ensure at least one word remains
        if result.is_empty() {
            let idx = rng.next_usize(words.len());
            return vec![words[idx].clone()];
        }

        result
    }

    /// Calculate similarity between original and augmented text.
    #[must_use]
    pub fn similarity(&self, original: &str, augmented: &str) -> f32 {
        let orig_words: std::collections::HashSet<_> = original.split_whitespace().collect();
        let aug_words: std::collections::HashSet<_> = augmented.split_whitespace().collect();

        if orig_words.is_empty() && aug_words.is_empty() {
            return 1.0;
        }
        if orig_words.is_empty() || aug_words.is_empty() {
            return 0.0;
        }

        let intersection = orig_words.intersection(&aug_words).count();
        let union = orig_words.union(&aug_words).count();

        intersection as f32 / union as f32
    }
}

impl SyntheticGenerator for EdaGenerator {
    type Input = String;
    type Output = String;

    fn generate(&self, seeds: &[String], config: &SyntheticConfig) -> Result<Vec<String>> {
        let target = config.target_count(seeds.len());
        let mut results = Vec::with_capacity(target);

        for (seed_idx, seed_text) in seeds.iter().enumerate() {
            let augmented = self.augment(seed_text, seed_idx as u64);
            for aug in augmented {
                if self.quality_score(&aug, seed_text) >= config.quality_threshold {
                    results.push(aug);
                }
                if results.len() >= target {
                    return Ok(results);
                }
            }
        }

        Ok(results)
    }

    fn quality_score(&self, generated: &String, seed: &String) -> f32 {
        // Quality = similarity (not too different) + length preservation
        let similarity = self.similarity(seed, generated);
        let len_ratio = generated.len() as f32 / seed.len().max(1) as f32;
        let len_score = if (0.5..=2.0).contains(&len_ratio) {
            1.0
        } else {
            0.5
        };

        0.7 * similarity + 0.3 * len_score
    }

    fn diversity_score(&self, batch: &[String]) -> f32 {
        if batch.len() < 2 {
            return 1.0;
        }

        // Compute pairwise Jaccard distances
        let mut total_dist = 0.0;
        let mut count = 0;

        for i in 0..batch.len() {
            for j in (i + 1)..batch.len() {
                let sim = self.similarity(&batch[i], &batch[j]);
                total_dist += 1.0 - sim;
                count += 1;
            }
        }

        if count == 0 {
            1.0
        } else {
            total_dist / count as f32
        }
    }
}
