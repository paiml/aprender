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
    /// Solved within `max_turns`
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

include!("mod_part_02.rs");
include!("test_task.rs");
