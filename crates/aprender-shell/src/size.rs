impl MarkovModel {

    /// Number of unique n-grams
    pub fn ngram_count(&self) -> usize {
        self.ngrams.values().map(|m| m.len()).sum()
    }

    /// Vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.command_freq.len()
    }

    /// N-gram size
    pub fn ngram_size(&self) -> usize {
        self.n
    }

    /// Approximate model size in bytes
    pub fn size_bytes(&self) -> usize {
        // Rough estimate
        let ngram_size: usize = self
            .ngrams
            .iter()
            .map(|(k, v)| k.len() + v.keys().map(|k2| k2.len() + 4).sum::<usize>())
            .sum();
        let vocab_size: usize = self.command_freq.keys().map(|k| k.len() + 4).sum();
        ngram_size + vocab_size
    }

    /// Top commands by frequency
    ///
    /// Optimized to reduce allocations:
    /// - Pre-allocated result vector
    /// - Uses sort_unstable for better cache locality
    pub fn top_commands(&self, count: usize) -> Vec<(String, u32)> {
        let mut cmds: Vec<_> = Vec::with_capacity(self.command_freq.len());
        cmds.extend(self.command_freq.iter().map(|(k, v)| (k.clone(), *v)));
        cmds.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        cmds.truncate(count);
        cmds
    }

    /// Validate model using holdout evaluation with aprender's ranking metrics.
    ///
    /// Uses `aprender::metrics::ranking` for Hit@K and MRR with prefix matching
    /// (appropriate for command completion where partial matches count).
    pub fn validate(commands: &[String], ngram_size: usize, train_ratio: f32) -> ValidationResult {
        let split_idx = (commands.len() as f32 * train_ratio) as usize;
        let (train, test) = commands.split_at(split_idx);

        // Train model
        let mut model = Self::new(ngram_size);
        model.train(train);

        let mut hit_1_sum = 0.0_f32;
        let mut hit_5_sum = 0.0_f32;
        let mut hit_10_sum = 0.0_f32;
        let mut rr_sum = 0.0_f32;
        let mut evaluated = 0;

        for cmd in test {
            let tokens: Vec<&str> = cmd.split_whitespace().collect();
            if tokens.len() < 2 {
                continue;
            }

            evaluated += 1;

            let prefix = tokens[0];
            let suggestions = model.suggest(prefix, 10);

            // For command completion, check if target starts with any suggestion
            // (e.g., "git commit -m" matches suggestion "git commit")
            let mut found_rank: Option<usize> = None;
            for (rank, (suggestion, _)) in suggestions.iter().enumerate() {
                if cmd.starts_with(suggestion.as_str()) || suggestion.starts_with(cmd) {
                    found_rank = Some(rank);
                    break;
                }
            }

            if let Some(rank) = found_rank {
                if rank == 0 {
                    hit_1_sum += 1.0;
                }
                if rank < 5 {
                    hit_5_sum += 1.0;
                }
                if rank < 10 {
                    hit_10_sum += 1.0;
                }
                rr_sum += 1.0 / (rank + 1) as f32;
            }
        }

        let n = evaluated.max(1) as f32;
        let metrics = RankingMetrics {
            hit_at_1: hit_1_sum / n,
            hit_at_5: hit_5_sum / n,
            hit_at_10: hit_10_sum / n,
            mrr: rr_sum / n,
            n_samples: evaluated,
        };

        ValidationResult {
            train_size: train.len(),
            test_size: test.len(),
            evaluated,
            metrics,
        }
    }
}
