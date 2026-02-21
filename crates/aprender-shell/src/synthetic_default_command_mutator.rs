
impl Default for CommandGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Mutation engine for shell commands
pub struct CommandMutator {
    /// Flag substitutions
    flag_subs: HashMap<&'static str, Vec<&'static str>>,
    /// Command substitutions
    cmd_subs: HashMap<&'static str, Vec<&'static str>>,
}

impl CommandMutator {
    /// Create new mutator with default rules
    #[must_use]
    pub fn new() -> Self {
        let mut flag_subs = HashMap::new();
        flag_subs.insert("-m", vec!["-am", "--message", "-m"]);
        flag_subs.insert("--release", vec!["--debug", ""]);
        flag_subs.insert("--lib", vec!["--bin", "--doc", "--test", ""]);
        flag_subs.insert("-v", vec!["-vv", "-vvv", "--verbose", ""]);
        flag_subs.insert("-a", vec!["--all", ""]);
        flag_subs.insert("-f", vec!["--force", ""]);
        flag_subs.insert("-n", vec!["--dry-run", ""]);
        flag_subs.insert("-i", vec!["--interactive", ""]);
        flag_subs.insert("-r", vec!["-R", "--recursive", ""]);

        let mut cmd_subs = HashMap::new();
        cmd_subs.insert("commit", vec!["add", "status", "diff", "log"]);
        cmd_subs.insert("push", vec!["pull", "fetch"]);
        cmd_subs.insert("test", vec!["build", "check", "run", "bench"]);
        cmd_subs.insert("install", vec!["uninstall", "update", "add", "remove"]);
        cmd_subs.insert("start", vec!["stop", "restart", "status"]);
        cmd_subs.insert("up", vec!["down", "restart", "logs"]);
        cmd_subs.insert("create", vec!["delete", "update", "get", "describe"]);

        Self {
            flag_subs,
            cmd_subs,
        }
    }

    /// Generate mutations of a command
    pub fn mutate(&self, command: &str) -> Vec<String> {
        let mut mutations = Vec::new();
        let parts: Vec<&str> = command.split_whitespace().collect();

        if parts.is_empty() {
            return mutations;
        }

        // Mutate subcommand (second token usually)
        if parts.len() >= 2 {
            if let Some(subs) = self.cmd_subs.get(parts[1]) {
                for sub in subs {
                    let mut new_parts = parts.clone();
                    new_parts[1] = sub;
                    mutations.push(new_parts.join(" "));
                }
            }
        }

        // Mutate flags
        for (i, part) in parts.iter().enumerate() {
            if let Some(subs) = self.flag_subs.get(*part) {
                for sub in subs {
                    let mut new_parts: Vec<&str> = parts.clone();
                    if sub.is_empty() {
                        new_parts.remove(i);
                    } else {
                        new_parts[i] = sub;
                    }
                    let new_cmd = new_parts.join(" ");
                    if !new_cmd.is_empty() && new_cmd != command {
                        mutations.push(new_cmd);
                    }
                }
            }
        }

        // Add common flag variations
        if !command.contains("--") {
            if command.starts_with("git ") {
                mutations.push(format!("{} --verbose", command));
            }
            if command.starts_with("cargo ") {
                mutations.push(format!("{} --release", command));
                mutations.push(format!("{} --all-features", command));
            }
        }

        // Remove duplicates
        mutations.sort();
        mutations.dedup();
        mutations
    }

    /// Mutate a batch of commands
    pub fn mutate_batch(&self, commands: &[String]) -> Vec<String> {
        let mut all_mutations = Vec::new();
        let mut seen = HashSet::new();

        for cmd in commands {
            if seen.insert(cmd.clone()) {
                all_mutations.push(cmd.clone());
            }
            for mutation in self.mutate(cmd) {
                if seen.insert(mutation.clone()) {
                    all_mutations.push(mutation);
                }
            }
        }

        all_mutations
    }
}

impl Default for CommandMutator {
    fn default() -> Self {
        Self::new()
    }
}

/// Coverage-guided generator that fills gaps in n-gram model
pub struct CoverageGuidedGenerator {
    /// Known n-grams from training
    known_ngrams: HashSet<String>,
    /// Target n-gram size
    n: usize,
}

impl CoverageGuidedGenerator {
    /// Create from existing n-gram counts
    pub fn new(known_ngrams: HashSet<String>, n: usize) -> Self {
        Self { known_ngrams, n }
    }

    /// Generate commands that exercise underrepresented n-grams
    pub fn generate(&self, base_commands: &[String], count: usize) -> Vec<String> {
        let mut generated = Vec::new();
        let mut new_ngrams_added = HashSet::new();

        // Find which commands introduce new n-grams
        for cmd in base_commands {
            let ngrams = self.extract_ngrams(cmd);
            let new_count = ngrams
                .iter()
                .filter(|ng| !self.known_ngrams.contains(*ng))
                .count();

            if new_count > 0 {
                generated.push((cmd.clone(), new_count));
                for ng in ngrams {
                    if !self.known_ngrams.contains(&ng) {
                        new_ngrams_added.insert(ng);
                    }
                }
            }

            if generated.len() >= count * 2 {
                break;
            }
        }

        // Sort by coverage gain (descending)
        generated.sort_by(|a, b| b.1.cmp(&a.1));

        // Return top commands
        generated
            .into_iter()
            .take(count)
            .map(|(cmd, _)| cmd)
            .collect()
    }

    fn extract_ngrams(&self, command: &str) -> Vec<String> {
        let tokens: Vec<&str> = command.split_whitespace().collect();
        let mut ngrams = Vec::new();

        // First token as context
        if !tokens.is_empty() {
            ngrams.push(tokens[0].to_string());
        }

        // Build n-grams
        for i in 0..tokens.len() {
            let start = i.saturating_sub(self.n - 1);
            let context = tokens[start..=i].join(" ");
            ngrams.push(context);
        }

        ngrams
    }

    /// Report coverage stats
    pub fn coverage_report(&self, commands: &[String]) -> CoverageReport {
        let mut total_ngrams = HashSet::new();
        let mut new_ngrams = HashSet::new();

        for cmd in commands {
            for ng in self.extract_ngrams(cmd) {
                total_ngrams.insert(ng.clone());
                if !self.known_ngrams.contains(&ng) {
                    new_ngrams.insert(ng);
                }
            }
        }

        CoverageReport {
            known_ngrams: self.known_ngrams.len(),
            total_ngrams: total_ngrams.len(),
            new_ngrams: new_ngrams.len(),
            coverage_gain: if total_ngrams.is_empty() {
                0.0
            } else {
                new_ngrams.len() as f32 / total_ngrams.len() as f32
            },
        }
    }
}

/// Coverage statistics
#[derive(Debug, Clone)]
pub struct CoverageReport {
    /// N-grams already in model
    pub known_ngrams: usize,
    /// Total n-grams in synthetic data
    pub total_ngrams: usize,
    /// New n-grams from synthetic data
    pub new_ngrams: usize,
    /// Percentage of synthetic data that's new
    pub coverage_gain: f32,
}

/// Combined synthetic data pipeline
pub struct SyntheticPipeline {
    generator: CommandGenerator,
    mutator: CommandMutator,
}

impl SyntheticPipeline {
    /// Create new pipeline
    #[must_use]
    pub fn new() -> Self {
        Self {
            generator: CommandGenerator::new(),
            mutator: CommandMutator::new(),
        }
    }

    /// Generate synthetic training data
    ///
    /// 1. Generate base commands from templates
    /// 2. Mutate real history for variations
    /// 3. Use coverage-guided selection
    pub fn generate(
        &self,
        real_history: &[String],
        known_ngrams: HashSet<String>,
        count: usize,
    ) -> SyntheticResult {
        // Step 1: Generate template commands
        let template_commands = self.generator.generate(count * 2);

        // Step 2: Mutate real history
        let mutated_commands = self.mutator.mutate_batch(real_history);

        // Step 3: Combine all candidates
        let mut all_candidates: Vec<String> = template_commands;
        all_candidates.extend(mutated_commands);

        // Step 4: Coverage-guided selection
        let coverage_gen = CoverageGuidedGenerator::new(known_ngrams.clone(), 3);
        let selected = coverage_gen.generate(&all_candidates, count);

        // Step 5: Generate report
        let report = coverage_gen.coverage_report(&selected);

        SyntheticResult {
            commands: selected,
            report,
        }
    }
}

impl Default for SyntheticPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of synthetic data generation
#[derive(Debug)]
pub struct SyntheticResult {
    /// Generated commands
    pub commands: Vec<String>,
    /// Coverage report
    pub report: CoverageReport,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_generator_creates_commands() {
        let gen = CommandGenerator::new();
        let commands = gen.generate(1000);
        assert!(commands.len() >= 500);
        assert!(commands.iter().any(|c| c.starts_with("git")));
        assert!(commands.iter().any(|c| c.starts_with("cargo")));
    }

    #[test]
    fn test_command_generator_no_duplicates() {
        let gen = CommandGenerator::new();
        let commands = gen.generate(1000);
        let unique: HashSet<_> = commands.iter().collect();
        assert_eq!(commands.len(), unique.len());
    }

    #[test]
    fn test_mutator_creates_variations() {
        let mutator = CommandMutator::new();
        let mutations = mutator.mutate("git commit -m test");
        assert!(!mutations.is_empty());
        assert!(mutations
            .iter()
            .any(|m| m.contains("add") || m.contains("status")));
    }

    #[test]
    fn test_mutator_flag_substitution() {
        let mutator = CommandMutator::new();
        let mutations = mutator.mutate("cargo build --release");
        assert!(mutations
            .iter()
            .any(|m| !m.contains("--release") || m.contains("--debug")));
    }

    #[test]
    fn test_coverage_guided_prioritizes_new_ngrams() {
        let known: HashSet<String> = vec!["git".to_string(), "git commit".to_string()]
            .into_iter()
            .collect();
        let gen = CoverageGuidedGenerator::new(known, 3);

        let candidates = vec![
            "git commit".to_string(), // Known
            "cargo test".to_string(), // New
        ];

        let selected = gen.generate(&candidates, 1);
        assert_eq!(selected.len(), 1);
        assert!(selected[0].contains("cargo")); // Should prefer new
    }

    #[test]
    fn test_pipeline_generates_diverse_data() {
        let pipeline = SyntheticPipeline::new();
        let history = vec!["git status".to_string(), "cargo test".to_string()];
        let known = HashSet::new();

        let result = pipeline.generate(&history, known, 50);
        assert!(!result.commands.is_empty());
        assert!(result.report.new_ngrams > 0);
    }

    #[test]
    fn test_coverage_report_accuracy() {
        let known: HashSet<String> = vec!["git".to_string()].into_iter().collect();
        let gen = CoverageGuidedGenerator::new(known, 2);

        let commands = vec!["git status".to_string(), "cargo test".to_string()];
        let report = gen.coverage_report(&commands);

        assert_eq!(report.known_ngrams, 1);
        assert!(report.new_ngrams > 0);
        assert!(report.coverage_gain > 0.0);
    }
}
