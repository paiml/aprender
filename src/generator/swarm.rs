//! Swarm testing for grammar-based generation
//!
//! Implements swarm testing (Groce et al. 2012) which randomly enables/disables
//! feature subsets per generation batch. This helps find bugs that only occur
//! with specific feature combinations.
//!
//! # References
//!
//! - Groce, A., et al. "Swarm testing." ISSTA 2012.

use rand::prelude::*;
use std::collections::HashSet;

use super::python_enum::PythonEnumerator;
use super::GeneratedCode;

/// Feature categories for swarm testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Feature {
    /// Integer literals
    IntLiterals,
    /// Float literals
    FloatLiterals,
    /// String literals
    StringLiterals,
    /// Boolean literals
    BoolLiterals,
    /// None literal
    NoneLiteral,
    /// Variable references
    Variables,
    /// Assignment statements
    Assignments,
    /// Arithmetic operators (+, -, *, /, %, //, **)
    ArithmeticOps,
    /// Logical operators (and, or)
    LogicalOps,
    /// Unary operators (not, -, +)
    UnaryOps,
    /// Comparison operators (<, >, ==, !=, <=, >=)
    Comparisons,
    /// If statements
    IfStatements,
    /// While loops
    WhileLoops,
    /// For loops
    ForLoops,
    /// Function definitions
    Functions,
    /// Function calls
    FunctionCalls,
    /// Return statements
    Returns,
    /// List literals
    Lists,
    /// Control flow (break, continue, pass)
    ControlFlow,
}

impl Feature {
    /// Get all available features
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::IntLiterals,
            Self::FloatLiterals,
            Self::StringLiterals,
            Self::BoolLiterals,
            Self::NoneLiteral,
            Self::Variables,
            Self::Assignments,
            Self::ArithmeticOps,
            Self::LogicalOps,
            Self::UnaryOps,
            Self::Comparisons,
            Self::IfStatements,
            Self::WhileLoops,
            Self::ForLoops,
            Self::Functions,
            Self::FunctionCalls,
            Self::Returns,
            Self::Lists,
            Self::ControlFlow,
        ]
    }

    /// Get the core features that should always be enabled
    /// (needed for minimal valid programs)
    #[must_use]
    pub fn core() -> Vec<Self> {
        vec![Self::IntLiterals, Self::Variables, Self::Assignments]
    }
}

/// Configuration for a swarm testing batch
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Enabled features for this batch
    pub enabled_features: HashSet<Feature>,
    /// Random seed used to generate this config
    pub seed: u64,
    /// Batch identifier
    pub batch_id: usize,
}

impl SwarmConfig {
    /// Create a random swarm configuration
    #[must_use]
    pub fn random(seed: u64, features_per_batch: usize, batch_id: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(batch_id as u64));
        let all_features = Feature::all();

        // Always include core features
        let mut enabled: HashSet<Feature> = Feature::core().into_iter().collect();

        // Randomly select additional features
        let optional_features: Vec<Feature> = all_features
            .into_iter()
            .filter(|f| !enabled.contains(f))
            .collect();

        // Select up to features_per_batch additional features
        let to_select = features_per_batch.saturating_sub(enabled.len());
        let selected: Vec<&Feature> = optional_features
            .choose_multiple(&mut rng, to_select)
            .collect();

        for feature in selected {
            enabled.insert(*feature);
        }

        Self {
            enabled_features: enabled,
            seed,
            batch_id,
        }
    }

    /// Check if a feature is enabled
    #[must_use]
    pub fn is_enabled(&self, feature: Feature) -> bool {
        self.enabled_features.contains(&feature)
    }

    /// Get the number of enabled features
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.enabled_features.len()
    }
}

/// Swarm testing generator
///
/// Generates programs using random feature subsets per batch,
/// implementing the swarm testing strategy from Groce et al.
#[derive(Debug)]
pub struct SwarmGenerator {
    /// Maximum AST depth
    max_depth: usize,
    /// Random seed
    seed: u64,
    /// Features per batch
    features_per_batch: usize,
    /// Current batch counter
    current_batch: usize,
    /// Statistics on generated programs
    stats: SwarmStats,
}

/// Statistics from swarm generation
#[derive(Debug, Clone, Default)]
pub struct SwarmStats {
    /// Number of batches generated
    pub batches_generated: usize,
    /// Total programs generated
    pub programs_generated: usize,
    /// Feature coverage (features used at least once)
    pub feature_coverage: HashSet<Feature>,
    /// Programs per feature
    pub programs_per_feature: Vec<(Feature, usize)>,
}

impl SwarmStats {
    /// Get feature coverage percentage
    #[must_use]
    pub fn coverage_percentage(&self) -> f64 {
        let total = Feature::all().len();
        if total == 0 {
            return 0.0;
        }
        (self.feature_coverage.len() as f64 / total as f64) * 100.0
    }
}

impl SwarmGenerator {
    /// Create a new swarm generator
    #[must_use]
    pub fn new(max_depth: usize, features_per_batch: usize) -> Self {
        Self {
            max_depth,
            seed: 42,
            features_per_batch,
            current_batch: 0,
            stats: SwarmStats::default(),
        }
    }

    /// Set the random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate a batch of programs with random feature subset
    pub fn generate_batch(&mut self, batch_size: usize) -> Vec<GeneratedCode> {
        let config = SwarmConfig::random(self.seed, self.features_per_batch, self.current_batch);
        self.current_batch += 1;
        self.stats.batches_generated += 1;

        // Track feature coverage
        for feature in &config.enabled_features {
            self.stats.feature_coverage.insert(*feature);
        }

        // Generate programs using only enabled features
        let programs = self.generate_with_config(&config, batch_size);
        self.stats.programs_generated += programs.len();

        programs
    }

    /// Generate programs with a specific swarm configuration
    fn generate_with_config(&self, config: &SwarmConfig, count: usize) -> Vec<GeneratedCode> {
        let enumerator = PythonEnumerator::new(self.max_depth);
        let all_programs = enumerator.enumerate_programs();

        // Filter programs to those using only enabled features
        let filtered: Vec<GeneratedCode> = all_programs
            .into_iter()
            .filter(|prog| self.matches_config(prog, config))
            .take(count)
            .map(|mut prog| {
                // Add swarm metadata to features
                prog.features
                    .push(format!("swarm_batch_{}", config.batch_id));
                prog.features
                    .push(format!("swarm_features_{}", config.feature_count()));
                prog
            })
            .collect();

        filtered
    }

    /// Check if a program matches the swarm configuration
    fn matches_config(&self, prog: &GeneratedCode, config: &SwarmConfig) -> bool {
        // Parse the features used by this program
        let used_features = self.extract_features(&prog.code);

        // Check that all used features are enabled
        for feature in &used_features {
            if !config.is_enabled(*feature) {
                return false;
            }
        }

        true
    }

    /// Extract features used in a code snippet
    fn extract_features(&self, code: &str) -> HashSet<Feature> {
        let mut features = HashSet::new();

        // Detect literals
        if code.chars().any(|c| c.is_ascii_digit()) {
            features.insert(Feature::IntLiterals);
        }
        if code.contains('.') && code.chars().any(|c| c.is_ascii_digit()) {
            // Could be float
            if code
                .split_whitespace()
                .any(|s| s.parse::<f64>().is_ok() && s.contains('.'))
            {
                features.insert(Feature::FloatLiterals);
            }
        }
        if code.contains('"') || code.contains('\'') {
            features.insert(Feature::StringLiterals);
        }
        if code.contains("True") || code.contains("False") {
            features.insert(Feature::BoolLiterals);
        }
        if code.contains("None") {
            features.insert(Feature::NoneLiteral);
        }

        // Detect operators
        for op in ['+', '-', '*', '/', '%'] {
            if code.contains(op) {
                features.insert(Feature::ArithmeticOps);
                break;
            }
        }
        if code.contains("**") || code.contains("//") {
            features.insert(Feature::ArithmeticOps);
        }
        if code.contains(" and ") || code.contains(" or ") {
            features.insert(Feature::LogicalOps);
        }
        if code.contains("not ") {
            features.insert(Feature::UnaryOps);
        }

        // Detect comparisons
        for op in ["==", "!=", "<=", ">=", " < ", " > "] {
            if code.contains(op) {
                features.insert(Feature::Comparisons);
                break;
            }
        }

        // Detect control flow
        if code.contains("if ") {
            features.insert(Feature::IfStatements);
        }
        if code.contains("while ") {
            features.insert(Feature::WhileLoops);
        }
        if code.contains("for ") {
            features.insert(Feature::ForLoops);
        }
        if code.contains("def ") {
            features.insert(Feature::Functions);
        }
        if code.contains("return") {
            features.insert(Feature::Returns);
        }
        if code.contains("break") || code.contains("continue") || code.contains("pass") {
            features.insert(Feature::ControlFlow);
        }

        // Detect lists
        if code.contains('[') && code.contains(']') {
            features.insert(Feature::Lists);
        }

        // Detect function calls (simplified: look for `name(`)
        if code.contains("print(") || code.contains("len(") || code.contains("range(") {
            features.insert(Feature::FunctionCalls);
        }

        // Variables and assignments are almost always present
        if code.contains(" = ") {
            features.insert(Feature::Assignments);
            features.insert(Feature::Variables);
        }

        features
    }

    /// Generate multiple batches worth of programs
    pub fn generate(&mut self, total_count: usize, batch_size: usize) -> Vec<GeneratedCode> {
        let mut all_programs = Vec::with_capacity(total_count);
        let num_batches = (total_count + batch_size - 1) / batch_size;

        for _ in 0..num_batches {
            let remaining = total_count - all_programs.len();
            let this_batch_size = remaining.min(batch_size);
            let batch = self.generate_batch(this_batch_size);
            all_programs.extend(batch);

            if all_programs.len() >= total_count {
                break;
            }
        }

        all_programs.truncate(total_count);
        all_programs
    }

    /// Get generation statistics
    #[must_use]
    pub fn stats(&self) -> &SwarmStats {
        &self.stats
    }

    /// Reset the generator state
    pub fn reset(&mut self) {
        self.current_batch = 0;
        self.stats = SwarmStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_all() {
        let features = Feature::all();
        assert!(features.len() >= 15, "Should have many features");
    }

    #[test]
    fn test_feature_core() {
        let core = Feature::core();
        assert!(core.contains(&Feature::IntLiterals));
        assert!(core.contains(&Feature::Variables));
        assert!(core.contains(&Feature::Assignments));
    }

    #[test]
    fn test_swarm_config_random() {
        let config = SwarmConfig::random(42, 8, 0);
        assert!(config.feature_count() >= 3, "Should have core features");
        assert!(config.is_enabled(Feature::IntLiterals));
    }

    #[test]
    fn test_swarm_config_different_batches() {
        let config1 = SwarmConfig::random(42, 8, 0);
        let config2 = SwarmConfig::random(42, 8, 1);
        // Different batches should have different feature sets (usually)
        assert_ne!(config1.enabled_features, config2.enabled_features);
    }

    #[test]
    fn test_swarm_generator_new() {
        let gen = SwarmGenerator::new(3, 8);
        assert_eq!(gen.max_depth, 3);
        assert_eq!(gen.features_per_batch, 8);
    }

    #[test]
    fn test_swarm_generator_with_seed() {
        let gen = SwarmGenerator::new(3, 8).with_seed(123);
        assert_eq!(gen.seed, 123);
    }

    #[test]
    fn test_swarm_generator_generate_batch() {
        let mut gen = SwarmGenerator::new(2, 5).with_seed(42);
        let programs = gen.generate_batch(10);
        assert!(!programs.is_empty(), "Should generate some programs");

        // Check that programs have swarm metadata
        for prog in &programs {
            assert!(
                prog.features.iter().any(|f| f.starts_with("swarm_")),
                "Should have swarm metadata"
            );
        }
    }

    #[test]
    fn test_swarm_generator_stats() {
        let mut gen = SwarmGenerator::new(2, 5).with_seed(42);
        gen.generate_batch(10);

        let stats = gen.stats();
        assert_eq!(stats.batches_generated, 1);
        assert!(stats.programs_generated > 0);
        assert!(!stats.feature_coverage.is_empty());
    }

    #[test]
    fn test_swarm_generator_multiple_batches() {
        let mut gen = SwarmGenerator::new(2, 6).with_seed(42);

        gen.generate_batch(5);
        gen.generate_batch(5);
        gen.generate_batch(5);

        let stats = gen.stats();
        assert_eq!(stats.batches_generated, 3);
        // Multiple batches should cover more features
        assert!(
            stats.coverage_percentage() > 20.0,
            "Should have decent coverage"
        );
    }

    #[test]
    fn test_swarm_generator_generate() {
        let mut gen = SwarmGenerator::new(2, 6).with_seed(42);
        let programs = gen.generate(20, 5);

        // Should generate programs across multiple batches
        assert!(!programs.is_empty());
        let stats = gen.stats();
        assert!(stats.batches_generated >= 1);
    }

    #[test]
    fn test_swarm_generator_reset() {
        let mut gen = SwarmGenerator::new(2, 5).with_seed(42);
        gen.generate_batch(10);

        assert!(gen.stats().batches_generated > 0);

        gen.reset();
        assert_eq!(gen.stats().batches_generated, 0);
        assert_eq!(gen.stats().programs_generated, 0);
    }

    #[test]
    fn test_swarm_stats_coverage_percentage() {
        let mut stats = SwarmStats::default();
        assert!((stats.coverage_percentage() - 0.0).abs() < 0.001);

        stats.feature_coverage.insert(Feature::IntLiterals);
        stats.feature_coverage.insert(Feature::Assignments);
        assert!(stats.coverage_percentage() > 0.0);
    }

    #[test]
    fn test_swarm_stats_debug() {
        let stats = SwarmStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("SwarmStats"));
    }

    #[test]
    fn test_swarm_config_debug() {
        let config = SwarmConfig::random(42, 5, 0);
        let debug = format!("{:?}", config);
        assert!(debug.contains("SwarmConfig"));
    }

    #[test]
    fn test_extract_features_arithmetic() {
        let gen = SwarmGenerator::new(2, 5);
        let features = gen.extract_features("x = 1 + 2");
        assert!(features.contains(&Feature::ArithmeticOps));
        assert!(features.contains(&Feature::IntLiterals));
        assert!(features.contains(&Feature::Assignments));
    }

    #[test]
    fn test_extract_features_control_flow() {
        let gen = SwarmGenerator::new(2, 5);
        let features = gen.extract_features("if x > 0:\n    pass");
        assert!(features.contains(&Feature::IfStatements));
        assert!(features.contains(&Feature::Comparisons));
        assert!(features.contains(&Feature::ControlFlow));
    }

    #[test]
    fn test_extract_features_loops() {
        let gen = SwarmGenerator::new(2, 5);

        let features = gen.extract_features("while x > 0:\n    x = x - 1");
        assert!(features.contains(&Feature::WhileLoops));

        let features = gen.extract_features("for i in range(10):\n    pass");
        assert!(features.contains(&Feature::ForLoops));
        assert!(features.contains(&Feature::FunctionCalls));
    }

    #[test]
    fn test_extract_features_functions() {
        let gen = SwarmGenerator::new(2, 5);
        let features = gen.extract_features("def foo():\n    return 1");
        assert!(features.contains(&Feature::Functions));
        assert!(features.contains(&Feature::Returns));
    }

    #[test]
    fn test_extract_features_logical() {
        let gen = SwarmGenerator::new(2, 5);
        let features = gen.extract_features("x = True and False");
        assert!(features.contains(&Feature::LogicalOps));
        assert!(features.contains(&Feature::BoolLiterals));
    }

    #[test]
    fn test_extract_features_lists() {
        let gen = SwarmGenerator::new(2, 5);
        let features = gen.extract_features("x = [1, 2, 3]");
        assert!(features.contains(&Feature::Lists));
    }

    #[test]
    fn test_extract_features_none() {
        let gen = SwarmGenerator::new(2, 5);
        let features = gen.extract_features("x = None");
        assert!(features.contains(&Feature::NoneLiteral));
    }
}
