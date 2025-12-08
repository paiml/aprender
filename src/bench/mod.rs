//! Model Evaluation and Benchmarking Framework (`aprender::bench`)
//!
//! Provides multi-model comparison for evaluating `.apr` models on custom tasks.
//! Unlike QA (single-model validation), this module compares multiple models
//! to find the **smallest model that meets a performance threshold**.
//!
//! # Toyota Way Alignment
//! - **Pull Systems (P3)**: Pareto frontier pulls smallest viable model
//! - **Muda Elimination**: Avoid overprovisioning with right-sized models
//!
//! # References
//! - Deb et al. (2002) "NSGA-II" for Pareto optimization
//!
//! # Example
//! ```
//! use aprender::bench::{EvalResult, ModelComparison};
//!
//! let comparison = ModelComparison::new("python-to-rust");
//! assert!(comparison.results.is_empty());
//! ```

pub mod pareto;
pub mod py2rs;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Custom evaluation task trait
pub trait EvalTask: Send + Sync {
    /// Unique task identifier
    fn id(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// Input examples to evaluate
    fn examples(&self) -> &[Example];

    /// Maximum turns before declaring failure
    fn max_turns(&self) -> u32 {
        5
    }

    /// Timeout per turn
    fn turn_timeout(&self) -> Duration {
        Duration::from_secs(60)
    }
}

/// Example input for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Unique example ID
    pub id: String,
    /// Input prompt/code
    pub input: String,
    /// Expected output or behavior
    pub expected: String,
    /// Difficulty tier
    pub difficulty: Difficulty,
    /// Tags for filtering
    pub tags: Vec<String>,
}

impl Example {
    /// Create a new example
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        input: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            input: input.into(),
            expected: expected.into(),
            difficulty: Difficulty::Medium,
            tags: Vec::new(),
        }
    }

    /// Set difficulty
    #[must_use]
    pub fn with_difficulty(mut self, difficulty: Difficulty) -> Self {
        self.difficulty = difficulty;
        self
    }

    /// Add tags
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Difficulty tier for stratified analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Difficulty {
    /// 1-liner, obvious translation
    Trivial,
    /// Simple logic, standard patterns
    Easy,
    /// Multiple functions, error handling
    Medium,
    /// Complex algorithms, unsafe/FFI
    Hard,
    /// Requires deep language knowledge
    Expert,
}

impl Difficulty {
    /// Get numeric level (1-5)
    #[must_use]
    pub const fn level(&self) -> u8 {
        match self {
            Self::Trivial => 1,
            Self::Easy => 2,
            Self::Medium => 3,
            Self::Hard => 4,
            Self::Expert => 5,
        }
    }

    /// Get display name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Trivial => "Trivial",
            Self::Easy => "Easy",
            Self::Medium => "Medium",
            Self::Hard => "Hard",
            Self::Expert => "Expert",
        }
    }

    /// All difficulties in order
    #[must_use]
    pub const fn all() -> [Self; 5] {
        [
            Self::Trivial,
            Self::Easy,
            Self::Medium,
            Self::Hard,
            Self::Expert,
        ]
    }
}

/// Result of evaluating a single model on a single task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Model identifier
    pub model_id: String,
    /// Model size in bytes
    pub model_size_bytes: u64,
    /// Model parameter count (if known)
    pub model_params: Option<u64>,
    /// Task evaluated
    pub task_id: String,
    /// Per-example results
    pub example_results: Vec<ExampleResult>,
    /// Success rate by turn (cumulative)
    /// e.g., [0.60, 0.85, 0.95] = 60% turn 1, 85% by turn 2, 95% by turn 3
    pub success_by_turn: Vec<f64>,
    /// Average turns to success (for successful examples)
    pub avg_turns_to_success: f64,
    /// Overall success rate (any turn)
    pub overall_success_rate: f64,
    /// Total tokens consumed
    pub total_tokens: u64,
    /// Total latency
    pub total_latency: Duration,
}

impl EvalResult {
    /// Create a new evaluation result
    #[must_use]
    pub fn new(
        model_id: impl Into<String>,
        task_id: impl Into<String>,
        model_size_bytes: u64,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model_size_bytes,
            model_params: None,
            task_id: task_id.into(),
            example_results: Vec::new(),
            success_by_turn: Vec::new(),
            avg_turns_to_success: 0.0,
            overall_success_rate: 0.0,
            total_tokens: 0,
            total_latency: Duration::ZERO,
        }
    }

    /// Add an example result
    pub fn add_example(&mut self, result: ExampleResult) {
        self.total_tokens += result.tokens_per_turn.iter().sum::<u64>();
        self.total_latency += result.latency_per_turn.iter().sum::<Duration>();
        self.example_results.push(result);
    }

    /// Finalize the evaluation (compute aggregate metrics)
    pub fn finalize(&mut self, max_turns: u32) {
        let total = self.example_results.len();
        if total == 0 {
            return;
        }

        // Compute success by turn (cumulative)
        self.success_by_turn = Vec::with_capacity(max_turns as usize);
        for turn in 1..=max_turns {
            let solved = self
                .example_results
                .iter()
                .filter(|r| matches!(r.status, ExampleStatus::Solved { turn: t } if t <= turn))
                .count();
            self.success_by_turn.push(solved as f64 / total as f64);
        }

        // Compute average turns to success
        let solved_examples: Vec<_> = self
            .example_results
            .iter()
            .filter_map(|r| match r.status {
                ExampleStatus::Solved { turn } => Some(turn),
                _ => None,
            })
            .collect();

        if !solved_examples.is_empty() {
            self.avg_turns_to_success =
                f64::from(solved_examples.iter().sum::<u32>()) / solved_examples.len() as f64;
        }

        // Overall success rate
        self.overall_success_rate = solved_examples.len() as f64 / total as f64;
    }

    /// Get success rate at a specific turn
    #[must_use]
    pub fn success_at_turn(&self, turn: u32) -> f64 {
        self.success_by_turn
            .get((turn - 1) as usize)
            .copied()
            .unwrap_or(0.0)
    }
}

/// Result for a single example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleResult {
    /// Example ID
    pub example_id: String,
    /// Difficulty
    pub difficulty: Difficulty,
    /// Which turn solved it (None = failed all turns)
    pub solved_at_turn: Option<u32>,
    /// Tokens per turn
    pub tokens_per_turn: Vec<u64>,
    /// Latency per turn
    pub latency_per_turn: Vec<Duration>,
    /// Final status
    pub status: ExampleStatus,
}

impl ExampleResult {
    /// Create a solved example result
    #[must_use]
    pub fn solved(
        example_id: impl Into<String>,
        difficulty: Difficulty,
        turn: u32,
        tokens: Vec<u64>,
        latencies: Vec<Duration>,
    ) -> Self {
        Self {
            example_id: example_id.into(),
            difficulty,
            solved_at_turn: Some(turn),
            tokens_per_turn: tokens,
            latency_per_turn: latencies,
            status: ExampleStatus::Solved { turn },
        }
    }

    /// Create a failed example result
    #[must_use]
    pub fn failed(
        example_id: impl Into<String>,
        difficulty: Difficulty,
        attempts: u32,
        last_error: impl Into<String>,
        tokens: Vec<u64>,
        latencies: Vec<Duration>,
    ) -> Self {
        Self {
            example_id: example_id.into(),
            difficulty,
            solved_at_turn: None,
            tokens_per_turn: tokens,
            latency_per_turn: latencies,
            status: ExampleStatus::Failed {
                attempts,
                last_error: last_error.into(),
            },
        }
    }
}

/// Status of an example evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExampleStatus {
    /// Solved within max_turns
    Solved {
        /// Turn on which it was solved
        turn: u32,
    },
    /// Failed all turns
    Failed {
        /// Number of attempts
        attempts: u32,
        /// Last error message
        last_error: String,
    },
    /// Timed out
    Timeout {
        /// Turn on which it timed out
        turn: u32,
    },
    /// Skipped
    Skipped {
        /// Reason for skipping
        reason: String,
    },
}

/// Compare multiple models on the same task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Task being evaluated
    pub task_id: String,
    /// Results per model
    pub results: Vec<EvalResult>,
    /// Pareto-optimal models
    pub pareto_frontier: Vec<ParetoPoint>,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
}

impl ModelComparison {
    /// Create a new model comparison
    #[must_use]
    pub fn new(task_id: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            results: Vec::new(),
            pareto_frontier: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Add an evaluation result
    pub fn add_result(&mut self, result: EvalResult) {
        self.results.push(result);
    }

    /// Find smallest model meeting a success threshold
    #[must_use]
    pub fn smallest_meeting_threshold(&self, min_success: f64) -> Option<&EvalResult> {
        self.results
            .iter()
            .filter(|r| r.overall_success_rate >= min_success)
            .min_by_key(|r| r.model_size_bytes)
    }

    /// Find fastest model meeting a success threshold
    #[must_use]
    pub fn fastest_meeting_threshold(&self, min_success: f64) -> Option<&EvalResult> {
        self.results
            .iter()
            .filter(|r| r.overall_success_rate >= min_success)
            .min_by(|a, b| {
                a.avg_turns_to_success
                    .partial_cmp(&b.avg_turns_to_success)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Compute Pareto frontier (size vs accuracy)
    pub fn compute_pareto_frontier(&mut self) {
        self.pareto_frontier = pareto::compute_pareto_frontier(&self.results);
    }

    /// Generate recommendations based on results
    pub fn generate_recommendations(&mut self) {
        self.recommendations.clear();

        // Smallest overall
        if let Some(smallest) = self.results.iter().min_by_key(|r| r.model_size_bytes) {
            self.recommendations.push(Recommendation {
                scenario: "Minimum footprint".to_string(),
                model_id: smallest.model_id.clone(),
                rationale: format!(
                    "Smallest model at {} bytes, {:.1}% success",
                    smallest.model_size_bytes,
                    smallest.overall_success_rate * 100.0
                ),
            });
        }

        // Best accuracy
        if let Some(best) = self.results.iter().max_by(|a, b| {
            a.overall_success_rate
                .partial_cmp(&b.overall_success_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.recommendations.push(Recommendation {
                scenario: "Maximum accuracy".to_string(),
                model_id: best.model_id.clone(),
                rationale: format!(
                    "Highest success rate at {:.1}%",
                    best.overall_success_rate * 100.0
                ),
            });
        }

        // Best balance (on Pareto frontier, closest to ideal)
        if let Some(best) = self.pareto_frontier.iter().max_by(|a, b| {
            // Score: success_rate - normalized_size
            let score_a = a.success_rate - (a.size_bytes as f64 / 1e9);
            let score_b = b.success_rate - (b.size_bytes as f64 / 1e9);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.recommendations.push(Recommendation {
                scenario: "Best balance".to_string(),
                model_id: best.model_id.clone(),
                rationale: format!(
                    "Pareto optimal: {:.1}% success at {} bytes",
                    best.success_rate * 100.0,
                    best.size_bytes
                ),
            });
        }
    }

    /// Get stratified success rates by difficulty
    #[must_use]
    pub fn stratified_by_difficulty(&self) -> HashMap<String, HashMap<Difficulty, f64>> {
        let mut result = HashMap::new();

        for eval in &self.results {
            let mut by_difficulty: HashMap<Difficulty, (usize, usize)> = HashMap::new();

            for example in &eval.example_results {
                let entry = by_difficulty.entry(example.difficulty).or_insert((0, 0));
                entry.1 += 1; // total
                if example.solved_at_turn.is_some() {
                    entry.0 += 1; // solved
                }
            }

            let rates: HashMap<Difficulty, f64> = by_difficulty
                .into_iter()
                .map(|(d, (solved, total))| {
                    let rate = if total > 0 {
                        solved as f64 / total as f64
                    } else {
                        0.0
                    };
                    (d, rate)
                })
                .collect();

            result.insert(eval.model_id.clone(), rates);
        }

        result
    }
}

/// Point on the Pareto frontier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    /// Model ID
    pub model_id: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Success rate
    pub success_rate: f64,
    /// Average turns
    pub avg_turns: f64,
    /// Is Pareto optimal
    pub is_pareto_optimal: bool,
}

/// Recommendation for a specific scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Scenario description
    pub scenario: String,
    /// Recommended model ID
    pub model_id: String,
    /// Rationale
    pub rationale: String,
}

/// Evaluation suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSuiteConfig {
    /// Suite ID
    pub id: String,
    /// Description
    pub description: String,
    /// Maximum turns
    pub max_turns: u32,
    /// Turn timeout
    pub turn_timeout_secs: u64,
    /// Examples path
    pub examples_path: PathBuf,
    /// Success thresholds for recommendations
    pub success_thresholds: Vec<f64>,
}

impl Default for EvalSuiteConfig {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            description: "Default evaluation suite".to_string(),
            max_turns: 5,
            turn_timeout_secs: 60,
            examples_path: PathBuf::from("./examples"),
            success_thresholds: vec![0.80, 0.90, 0.95],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_creation() {
        let example = Example::new("ex1", "print('hello')", "println!(\"hello\");")
            .with_difficulty(Difficulty::Trivial)
            .with_tags(vec!["hello".to_string()]);

        assert_eq!(example.id, "ex1");
        assert_eq!(example.difficulty, Difficulty::Trivial);
        assert_eq!(example.tags.len(), 1);
    }

    #[test]
    fn test_difficulty_levels() {
        assert_eq!(Difficulty::Trivial.level(), 1);
        assert_eq!(Difficulty::Expert.level(), 5);
        assert_eq!(Difficulty::all().len(), 5);
    }

    #[test]
    fn test_eval_result_creation() {
        let mut result = EvalResult::new("model1", "py2rs", 1_000_000);
        assert_eq!(result.model_id, "model1");
        assert_eq!(result.model_size_bytes, 1_000_000);
        assert!(result.example_results.is_empty());

        // Add some example results
        result.add_example(ExampleResult::solved(
            "ex1",
            Difficulty::Easy,
            1,
            vec![100],
            vec![Duration::from_millis(50)],
        ));
        result.add_example(ExampleResult::solved(
            "ex2",
            Difficulty::Medium,
            2,
            vec![100, 150],
            vec![Duration::from_millis(50), Duration::from_millis(75)],
        ));
        result.add_example(ExampleResult::failed(
            "ex3",
            Difficulty::Hard,
            3,
            "Compile error",
            vec![100, 150, 200],
            vec![Duration::from_millis(50); 3],
        ));

        result.finalize(3);

        assert_eq!(result.example_results.len(), 3);
        assert!((result.overall_success_rate - 2.0 / 3.0).abs() < 0.01);
        assert!(result.avg_turns_to_success > 0.0);
    }

    #[test]
    fn test_eval_result_success_by_turn() {
        let mut result = EvalResult::new("model1", "task1", 1000);

        // 3 examples: turn 1, turn 2, failed
        result.add_example(ExampleResult::solved(
            "ex1",
            Difficulty::Easy,
            1,
            vec![100],
            vec![Duration::from_millis(10)],
        ));
        result.add_example(ExampleResult::solved(
            "ex2",
            Difficulty::Medium,
            2,
            vec![100, 100],
            vec![Duration::from_millis(10); 2],
        ));
        result.add_example(ExampleResult::failed(
            "ex3",
            Difficulty::Hard,
            3,
            "Failed",
            vec![100; 3],
            vec![Duration::from_millis(10); 3],
        ));

        result.finalize(3);

        // Turn 1: 1/3 solved
        assert!((result.success_at_turn(1) - 1.0 / 3.0).abs() < 0.01);
        // Turn 2: 2/3 solved
        assert!((result.success_at_turn(2) - 2.0 / 3.0).abs() < 0.01);
        // Turn 3: still 2/3 (failed didn't solve)
        assert!((result.success_at_turn(3) - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_model_comparison() {
        let mut comparison = ModelComparison::new("py2rs");

        // Add two models
        let mut model1 = EvalResult::new("small", "py2rs", 1_000_000);
        model1.add_example(ExampleResult::solved(
            "ex1",
            Difficulty::Easy,
            1,
            vec![100],
            vec![Duration::from_millis(10)],
        ));
        model1.finalize(3);

        let mut model2 = EvalResult::new("large", "py2rs", 10_000_000);
        model2.add_example(ExampleResult::solved(
            "ex1",
            Difficulty::Easy,
            1,
            vec![100],
            vec![Duration::from_millis(10)],
        ));
        model2.finalize(3);

        comparison.add_result(model1);
        comparison.add_result(model2);

        assert_eq!(comparison.results.len(), 2);

        // Both have 100% success
        let smallest = comparison.smallest_meeting_threshold(0.9);
        assert!(smallest.is_some());
        assert_eq!(smallest.unwrap().model_id, "small");
    }

    #[test]
    fn test_example_status() {
        let solved = ExampleStatus::Solved { turn: 2 };
        let failed = ExampleStatus::Failed {
            attempts: 3,
            last_error: "Error".to_string(),
        };
        let timeout = ExampleStatus::Timeout { turn: 1 };
        let skipped = ExampleStatus::Skipped {
            reason: "No deps".to_string(),
        };

        // Just verify they serialize
        assert!(serde_json::to_string(&solved).is_ok());
        assert!(serde_json::to_string(&failed).is_ok());
        assert!(serde_json::to_string(&timeout).is_ok());
        assert!(serde_json::to_string(&skipped).is_ok());
    }

    #[test]
    fn test_eval_suite_config_default() {
        let config = EvalSuiteConfig::default();
        assert_eq!(config.max_turns, 5);
        assert_eq!(config.turn_timeout_secs, 60);
        assert!(config.success_thresholds.contains(&0.9));
    }

    #[test]
    fn test_stratified_by_difficulty() {
        let mut comparison = ModelComparison::new("test");

        let mut result = EvalResult::new("model1", "test", 1000);
        result.add_example(ExampleResult::solved(
            "e1",
            Difficulty::Easy,
            1,
            vec![10],
            vec![Duration::ZERO],
        ));
        result.add_example(ExampleResult::solved(
            "e2",
            Difficulty::Easy,
            1,
            vec![10],
            vec![Duration::ZERO],
        ));
        result.add_example(ExampleResult::failed(
            "e3",
            Difficulty::Hard,
            3,
            "err",
            vec![10; 3],
            vec![Duration::ZERO; 3],
        ));
        result.finalize(3);
        comparison.add_result(result);

        let stratified = comparison.stratified_by_difficulty();
        let model1 = stratified.get("model1").unwrap();

        // Easy: 2/2 = 100%
        assert!((model1.get(&Difficulty::Easy).unwrap() - 1.0).abs() < 0.01);
        // Hard: 0/1 = 0%
        assert!((model1.get(&Difficulty::Hard).unwrap() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_generate_recommendations() {
        let mut comparison = ModelComparison::new("test");

        let mut small = EvalResult::new("small", "test", 1000);
        small.overall_success_rate = 0.8;
        comparison.add_result(small);

        let mut large = EvalResult::new("large", "test", 10000);
        large.overall_success_rate = 0.95;
        comparison.add_result(large);

        comparison.generate_recommendations();

        assert!(comparison.recommendations.len() >= 2);
        assert!(comparison
            .recommendations
            .iter()
            .any(|r| r.scenario.contains("footprint")));
        assert!(comparison
            .recommendations
            .iter()
            .any(|r| r.scenario.contains("accuracy")));
    }
}
