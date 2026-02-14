//! Python to Rust Single-Shot Compile Benchmark (10 Levels)
//!
//! Canonical benchmark for code translation model evaluation.
//! Measures success rate by turn and finds smallest model meeting thresholds.
//!
//! # Levels
//! 1. Hello World
//! 2. Variables & Arithmetic
//! 3. Functions & Ownership
//! 4. Collections & Iterators
//! 5. Control Flow & Borrowing
//! 6. Error Handling (Result)
//! 7. OOP → Traits
//! 8. Concurrency (async/rayon)
//! 9. FFI/Unsafe
//! 10. Metaprogramming (proc macros)

use super::{Difficulty, EvalResult, Example, ExampleResult, ModelComparison};
use serde::{Deserialize, Serialize};
use std::fmt::Write;
use std::time::Duration;

/// Python→Rust benchmark level (1-10)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Py2RsLevel {
    /// Level 1: Hello World
    Hello = 1,
    /// Level 2: Variables & Arithmetic
    Variables = 2,
    /// Level 3: Functions & Ownership
    Functions = 3,
    /// Level 4: Collections & Iterators
    Collections = 4,
    /// Level 5: Control Flow & Borrowing
    ControlFlow = 5,
    /// Level 6: Error Handling
    ErrorHandling = 6,
    /// Level 7: OOP → Traits
    OopTraits = 7,
    /// Level 8: Concurrency
    Concurrency = 8,
    /// Level 9: FFI/Unsafe
    FfiUnsafe = 9,
    /// Level 10: Metaprogramming
    Metaprogramming = 10,
}

impl Py2RsLevel {
    /// Get all levels in order
    #[must_use]
    pub const fn all() -> [Self; 10] {
        [
            Self::Hello,
            Self::Variables,
            Self::Functions,
            Self::Collections,
            Self::ControlFlow,
            Self::ErrorHandling,
            Self::OopTraits,
            Self::Concurrency,
            Self::FfiUnsafe,
            Self::Metaprogramming,
        ]
    }

    /// Get level number (1-10)
    #[must_use]
    pub const fn number(&self) -> u8 {
        *self as u8
    }

    /// Get level name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Hello => "Hello",
            Self::Variables => "Variables",
            Self::Functions => "Functions",
            Self::Collections => "Collections",
            Self::ControlFlow => "ControlFlow",
            Self::ErrorHandling => "ErrorHandling",
            Self::OopTraits => "OOP→Traits",
            Self::Concurrency => "Concurrency",
            Self::FfiUnsafe => "FFI/Unsafe",
            Self::Metaprogramming => "Metaprogramming",
        }
    }

    /// Get difficulty for this level
    #[must_use]
    pub const fn difficulty(&self) -> Difficulty {
        match self {
            Self::Hello | Self::Variables => Difficulty::Trivial,
            Self::Functions | Self::Collections => Difficulty::Easy,
            Self::ControlFlow | Self::ErrorHandling => Difficulty::Medium,
            Self::OopTraits | Self::Concurrency => Difficulty::Hard,
            Self::FfiUnsafe | Self::Metaprogramming => Difficulty::Expert,
        }
    }

    /// Get scoring weight (higher levels worth more)
    #[must_use]
    pub const fn weight(&self) -> f32 {
        match self {
            Self::Hello => 1.0,
            Self::Variables => 1.5,
            Self::Functions => 2.0,
            Self::Collections => 3.0,
            Self::ControlFlow => 4.0,
            Self::ErrorHandling => 5.0,
            Self::OopTraits => 7.0,
            Self::Concurrency => 10.0,
            Self::FfiUnsafe => 15.0,
            Self::Metaprogramming => 20.0,
        }
    }

    /// Get canonical Python example
    #[must_use]
    pub fn python_example(&self) -> &'static str {
        match self {
            Self::Hello => r#"print("hello world")"#,
            Self::Variables => {
                r#"x = 42
y = x * 2 + 1
print(f"Result: {y}")"#
            }
            Self::Functions => {
                r"def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"
            }
            Self::Collections => {
                r#"squares = [x**2 for x in range(10) if x % 2 == 0]
counts = {word: len(word) for word in ["hello", "world"]}"#
            }
            Self::ControlFlow => {
                r"def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"
            }
            Self::ErrorHandling => {
                r#"def read_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")"#
            }
            Self::OopTraits => {
                r"class Shape:
    def area(self): raise NotImplementedError

class Circle(Shape):
    def __init__(self, radius): self.radius = radius
    def area(self): return 3.14159 * self.radius ** 2"
            }
            Self::Concurrency => {
                r"import asyncio
async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[fetch(session, u) for u in urls])"
            }
            Self::FfiUnsafe => {
                r#"import ctypes
lib = ctypes.CDLL("libcrypto.so")
lib.SHA256_Init.argtypes = [ctypes.POINTER(SHA256_CTX)]"#
            }
            Self::Metaprogramming => {
                r"@dataclass
class Point:
    x: float
    y: float
    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5"
            }
        }
    }
}

/// Score for Python→Rust benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Py2RsScore {
    /// Model identifier
    pub model_id: String,
    /// Highest level passed (1-10)
    pub max_level: u8,
    /// Levels passed on turn 1 (single-shot)
    pub single_shot_levels: Vec<u8>,
    /// Average turns per level (lower = better)
    pub avg_turns_by_level: [f32; 10],
    /// Composite score (0-100)
    pub composite: f32,
    /// Per-level results
    pub level_results: Vec<LevelResult>,
}

impl Py2RsScore {
    /// Create a new score
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            max_level: 0,
            single_shot_levels: Vec::new(),
            avg_turns_by_level: [0.0; 10],
            composite: 0.0,
            level_results: Vec::new(),
        }
    }

    /// Add a level result
    pub fn add_level(&mut self, result: LevelResult) {
        if result.passed {
            if result.level > self.max_level {
                self.max_level = result.level;
            }
            if result.turn == 1 {
                self.single_shot_levels.push(result.level);
            }
        }
        self.avg_turns_by_level[(result.level - 1) as usize] = result.turn as f32;
        self.level_results.push(result);
    }

    /// Compute composite score
    pub fn finalize(&mut self) {
        self.composite = self.compute_composite_score();
    }

    /// Compute weighted composite score (0-100)
    fn compute_composite_score(&self) -> f32 {
        let weights = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0];
        let max_possible: f32 = weights.iter().sum(); // 68.5

        let earned: f32 = self
            .single_shot_levels
            .iter()
            .map(|&l| weights[(l - 1) as usize])
            .sum();

        (earned / max_possible) * 100.0
    }

    /// Get display symbol for level result
    #[must_use]
    pub fn level_symbol(&self, level: u8) -> char {
        if let Some(result) = self.level_results.iter().find(|r| r.level == level) {
            if result.passed {
                if result.turn == 1 {
                    '●' // Pass Turn 1
                } else {
                    '◐' // Pass Turn 2+
                }
            } else {
                '○' // Failed
            }
        } else {
            '○'
        }
    }

    /// Format results as visual string
    #[must_use]
    pub fn visual_summary(&self) -> String {
        (1..=10).map(|l| self.level_symbol(l)).collect()
    }
}

/// Result for a single level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelResult {
    /// Level number (1-10)
    pub level: u8,
    /// Level name
    pub name: String,
    /// Whether level was passed
    pub passed: bool,
    /// Turn on which it was solved (1-based)
    pub turn: u32,
    /// Error message if failed
    pub error: Option<String>,
    /// Latency
    pub latency: Duration,
}

impl LevelResult {
    /// Create a passed level result
    #[must_use]
    pub fn passed(level: Py2RsLevel, turn: u32, latency: Duration) -> Self {
        Self {
            level: level.number(),
            name: level.name().to_string(),
            passed: true,
            turn,
            error: None,
            latency,
        }
    }

    /// Create a failed level result
    #[must_use]
    pub fn failed(
        level: Py2RsLevel,
        attempts: u32,
        error: impl Into<String>,
        latency: Duration,
    ) -> Self {
        Self {
            level: level.number(),
            name: level.name().to_string(),
            passed: false,
            turn: attempts,
            error: Some(error.into()),
            latency,
        }
    }
}

/// Generate canonical `Py2Rs` examples
#[must_use]
pub fn generate_canonical_examples() -> Vec<Example> {
    Py2RsLevel::all()
        .iter()
        .map(|level| {
            Example::new(
                format!("py2rs-L{}", level.number()),
                level.python_example(),
                format!("Compile to valid Rust (Level {})", level.number()),
            )
            .with_difficulty(level.difficulty())
            .with_tags(vec![
                "py2rs".to_string(),
                format!("level-{}", level.number()),
                level.name().to_string(),
            ])
        })
        .collect()
}

/// Run `Py2Rs` benchmark on a model (mock implementation)
#[must_use]
pub fn run_benchmark(model_id: &str, max_turns: u32) -> Py2RsScore {
    let mut score = Py2RsScore::new(model_id);

    // Mock results - in real implementation, would run actual model
    for level in Py2RsLevel::all() {
        let (passed, turn) = mock_model_result(level, model_id);

        let result = if passed {
            LevelResult::passed(
                level,
                turn,
                Duration::from_millis(100 * u64::from(level.number())),
            )
        } else {
            LevelResult::failed(
                level,
                max_turns,
                "Compile error",
                Duration::from_millis(100 * u64::from(level.number()) * u64::from(max_turns)),
            )
        };

        score.add_level(result);
    }

    score.finalize();
    score
}

/// Mock model results (for testing)
fn mock_model_result(level: Py2RsLevel, model_id: &str) -> (bool, u32) {
    // Simulate different model capabilities
    let model_capability = match model_id {
        m if m.contains("16b") || m.contains("large") => 9,
        m if m.contains("6b") || m.contains("medium") => 7,
        m if m.contains("2b") || m.contains("small") => 5,
        _ => 6,
    };

    let level_num = level.number();

    if level_num <= model_capability {
        // Pass with 1-2 turns depending on difficulty
        let turn = if level_num <= model_capability - 2 {
            1
        } else {
            2
        };
        (true, turn)
    } else {
        (false, 5)
    }
}

/// Compare multiple models on `Py2Rs` benchmark
#[must_use]
pub fn compare_models(model_ids: &[(&str, u64)], max_turns: u32) -> ModelComparison {
    let mut comparison = ModelComparison::new("py2rs-canonical");

    for &(model_id, size) in model_ids {
        let score = run_benchmark(model_id, max_turns);

        // Convert Py2RsScore to EvalResult
        let mut result = EvalResult::new(model_id, "py2rs-canonical", size);

        for level_result in &score.level_results {
            let level = Py2RsLevel::all()[(level_result.level - 1) as usize];
            let example_result = if level_result.passed {
                ExampleResult::solved(
                    format!("L{}", level_result.level),
                    level.difficulty(),
                    level_result.turn,
                    vec![100; level_result.turn as usize],
                    vec![Duration::from_millis(50); level_result.turn as usize],
                )
            } else {
                ExampleResult::failed(
                    format!("L{}", level_result.level),
                    level.difficulty(),
                    max_turns,
                    {
                        #[allow(clippy::unwrap_or_default)]
                        level_result.error.clone().unwrap_or(String::new())
                    },
                    vec![100; max_turns as usize],
                    vec![Duration::from_millis(50); max_turns as usize],
                )
            };
            result.add_example(example_result);
        }

        result.finalize(max_turns);
        comparison.add_result(result);
    }

    comparison.compute_pareto_frontier();
    comparison.generate_recommendations();
    comparison
}

/// Format comparison as table
#[must_use]
pub fn format_comparison_table(_comparison: &ModelComparison, scores: &[Py2RsScore]) -> String {
    let mut output = String::new();

    output.push_str("┌────────────────────────────────────────────────────────────────┐\n");
    output.push_str("│ Benchmark: py2rs-canonical (10 levels)                         │\n");
    output.push_str("├──────────────┬───────┬────────────────────────────────┬────────┤\n");
    output.push_str("│ Model        │ Score │ Levels (● = T1, ◐ = T2+, ○ = fail) │ Max  │\n");
    output.push_str("├──────────────┼───────┼────────────────────────────────┼────────┤\n");

    for score in scores {
        let visual = score.visual_summary();
        let _ = writeln!(
            output,
            "│ {:12} │ {:5.1} │ {:30} │ L{:<4} │",
            &score.model_id[..score.model_id.len().min(12)],
            score.composite,
            visual,
            score.max_level
        );
    }

    output.push_str("├──────────────┴───────┴────────────────────────────────┴────────┤\n");
    output.push_str("│ Legend: ● Pass Turn 1 | ◐ Pass Turn 2+ | ○ Failed              │\n");
    output.push_str("└────────────────────────────────────────────────────────────────┘\n");

    output
}

#[cfg(test)]
#[path = "py2rs_tests.rs"]
mod tests;
