
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
