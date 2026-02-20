
impl ShellSyntheticGenerator {
    /// Create a new generator with default settings.
    #[must_use]
    pub fn new() -> Self {
        let mut gen = Self {
            grammar: ShellGrammar::common_commands(),
            config: ShellGeneratorConfig::default(),
            substitutions: HashMap::new(),
        };
        gen.init_default_substitutions();
        gen
    }

    /// Create with custom grammar.
    #[must_use]
    pub fn with_grammar(mut self, grammar: ShellGrammar) -> Self {
        self.grammar = grammar;
        self
    }

    /// Create with custom config.
    #[must_use]
    pub fn with_config(mut self, config: ShellGeneratorConfig) -> Self {
        self.config = config;
        self
    }

    /// Add argument substitutions.
    pub fn add_substitution(&mut self, arg: &str, variants: &[&str]) {
        self.substitutions.insert(
            arg.to_string(),
            variants.iter().map(|s| (*s).to_string()).collect(),
        );
    }

    /// Initialize default substitutions for common arguments.
    fn init_default_substitutions(&mut self) {
        // Git branches
        self.add_substitution("main", &["master", "develop", "feature"]);
        self.add_substitution("master", &["main", "develop", "feature"]);

        // Build targets
        self.add_substitution("--release", &["--debug", ""]);
        self.add_substitution("--debug", &["--release", ""]);

        // Common paths
        self.add_substitution(".", &["./", "../", "src/"]);
        self.add_substitution("src/", &["lib/", "tests/", "examples/"]);

        // Commit message patterns
        self.add_substitution("fix:", &["feat:", "docs:", "refactor:", "test:"]);
        self.add_substitution("feat:", &["fix:", "docs:", "refactor:", "chore:"]);
    }

    /// Generate via template substitution.
    fn generate_from_template(&self, seed: &ShellSample, rng_seed: u64) -> Vec<ShellSample> {
        if !self.config.enable_template {
            return Vec::new();
        }

        let mut results = Vec::new();
        let args = seed.arguments();

        for (i, arg) in args.iter().enumerate() {
            if let Some(variants) = self.substitutions.get(*arg) {
                for (vi, variant) in variants.iter().enumerate() {
                    // Use deterministic selection based on rng_seed
                    if (rng_seed + i as u64 + vi as u64) % 3 == 0 {
                        let mut new_args = args.clone();
                        if variant.is_empty() {
                            new_args.remove(i);
                        } else {
                            new_args[i] = variant.as_str();
                        }

                        if let Some(cmd) = seed.command_name() {
                            let new_completion = if new_args.is_empty() {
                                cmd.to_string()
                            } else {
                                format!("{} {}", cmd, new_args.join(" "))
                            };

                            // Generate matching prefix
                            let prefix_len = seed.prefix().len().min(new_completion.len());
                            let new_prefix = &new_completion[..prefix_len];

                            results.push(
                                ShellSample::new(new_prefix, &new_completion)
                                    .with_history(seed.history().to_vec())
                                    .with_cwd(seed.cwd()),
                            );
                        }
                    }
                }
            }
        }

        results
    }

    /// Generate via argument permutation.
    fn permute_arguments(&self, seed: &ShellSample, rng_seed: u64) -> Vec<ShellSample> {
        if !self.config.enable_permutation {
            return Vec::new();
        }

        let mut results = Vec::new();
        let args = seed.arguments();

        if args.len() < 2 || args.len() > self.config.max_permute_args {
            return results;
        }

        // Generate one permutation by swapping first two args
        let mut new_args: Vec<&str> = args.clone();
        new_args.swap(0, 1);

        if let Some(cmd) = seed.command_name() {
            let new_completion = format!("{} {}", cmd, new_args.join(" "));

            // Only use if different from original
            if new_completion != seed.completion() {
                // Add common option based on rng
                let final_completion = if rng_seed % 2 == 0 {
                    format!("{new_completion} --verbose")
                } else {
                    new_completion
                };

                let prefix_len = seed.prefix().len().min(final_completion.len());
                let new_prefix = final_completion[..prefix_len].to_string();

                results.push(
                    ShellSample::new(new_prefix, &final_completion)
                        .with_history(seed.history().to_vec())
                        .with_cwd(seed.cwd()),
                );
            }
        }

        results
    }

    /// Generate via context variation.
    fn vary_context(&self, seed: &ShellSample, rng_seed: u64) -> Vec<ShellSample> {
        if !self.config.enable_context_variation {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Vary working directory
        let cwd_variants = ["/home/user/project", "/tmp", "./src", "/var/log"];
        let cwd_idx = (rng_seed as usize) % cwd_variants.len();
        let new_cwd = cwd_variants[cwd_idx];

        if new_cwd != seed.cwd() {
            results.push(
                ShellSample::new(seed.prefix(), seed.completion())
                    .with_history(seed.history().to_vec())
                    .with_cwd(new_cwd),
            );
        }

        // Vary history
        let history_additions = ["ls -la", "pwd", "cd ..", "clear"];
        let hist_idx = (rng_seed as usize + 1) % history_additions.len();

        let mut new_history = seed.history().to_vec();
        new_history.push(history_additions[hist_idx].to_string());

        results.push(
            ShellSample::new(seed.prefix(), seed.completion())
                .with_history(new_history)
                .with_cwd(seed.cwd()),
        );

        results
    }

    /// Calculate semantic similarity between two commands (Jaccard similarity).
    fn semantic_similarity(a: &str, b: &str) -> f32 {
        let tokens_a: HashSet<&str> = a.split_whitespace().collect();
        let tokens_b: HashSet<&str> = b.split_whitespace().collect();

        if tokens_a.is_empty() && tokens_b.is_empty() {
            return 1.0;
        }
        if tokens_a.is_empty() || tokens_b.is_empty() {
            return 0.0;
        }

        let intersection = tokens_a.intersection(&tokens_b).count();
        let union = tokens_a.union(&tokens_b).count();

        intersection as f32 / union as f32
    }

    /// Calculate context coherence score.
    fn context_coherence(sample: &ShellSample) -> f32 {
        let mut score: f32 = 0.5; // Base score

        // Bonus for having history
        if !sample.history().is_empty() {
            score += 0.2;
        }

        // Bonus for having cwd
        if !sample.cwd().is_empty() {
            score += 0.2;
        }

        // Bonus for valid completion relationship
        if sample.is_valid_completion() {
            score += 0.1;
        }

        score.min(1.0)
    }
}

impl Default for ShellSyntheticGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticGenerator for ShellSyntheticGenerator {
    type Input = ShellSample;
    type Output = ShellSample;

    fn generate(
        &self,
        seeds: &[ShellSample],
        config: &SyntheticConfig,
    ) -> Result<Vec<ShellSample>> {
        let target = config.target_count(seeds.len());
        let mut results = Vec::with_capacity(target);
        let mut seen: HashSet<String> = HashSet::new();

        for (seed_idx, seed) in seeds.iter().enumerate() {
            let rng_seed = config.seed.wrapping_add(seed_idx as u64);

            // Strategy 1: Template substitution
            let template_samples = self.generate_from_template(seed, rng_seed);

            // Strategy 2: Argument permutation
            let permuted = self.permute_arguments(seed, rng_seed);

            // Strategy 3: Context variation
            let context_varied = self.vary_context(seed, rng_seed);

            // Combine and filter
            for sample in template_samples
                .into_iter()
                .chain(permuted)
                .chain(context_varied)
            {
                // Deduplicate
                if seen.contains(sample.completion()) {
                    continue;
                }

                // Quality filter
                let quality = self.quality_score(&sample, seed);
                if config.meets_quality(quality) {
                    // Grammar validation
                    if self.grammar.is_valid_command(sample.completion()) {
                        seen.insert(sample.completion().to_string());
                        results.push(sample);

                        if results.len() >= target {
                            return Ok(results);
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    fn quality_score(&self, generated: &ShellSample, seed: &ShellSample) -> f32 {
        // Semantic similarity (Jaccard)
        let semantic_sim = Self::semantic_similarity(generated.completion(), seed.completion());

        // Grammar validity (binary)
        let grammar_valid = if self.grammar.is_valid_command(generated.completion()) {
            1.0
        } else {
            0.0
        };

        // Context coherence
        let context_coherent = Self::context_coherence(generated);

        // Weighted combination (spec: 0.4 semantic + 0.4 grammar + 0.2 context)
        0.4 * semantic_sim + 0.4 * grammar_valid + 0.2 * context_coherent
    }

    fn diversity_score(&self, batch: &[ShellSample]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        // Count unique command patterns
        let unique_commands: HashSet<_> = batch.iter().filter_map(|s| s.command_name()).collect();

        let unique_completions: HashSet<_> = batch.iter().map(ShellSample::completion).collect();

        let cmd_diversity = unique_commands.len() as f32 / batch.len() as f32;
        let completion_diversity = unique_completions.len() as f32 / batch.len() as f32;

        (cmd_diversity + completion_diversity) / 2.0
    }
}
