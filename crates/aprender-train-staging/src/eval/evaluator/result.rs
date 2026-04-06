//! Evaluation result structure

use super::metric::Metric;
use std::collections::HashMap;
use std::fmt;

/// Model evaluation results
#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Name of the model
    pub model_name: String,
    /// Computed metric scores
    pub scores: HashMap<Metric, f64>,
    /// Cross-validation scores per fold (if CV enabled)
    pub cv_scores: Option<Vec<f64>>,
    /// Mean CV score
    pub cv_mean: Option<f64>,
    /// CV score standard deviation
    pub cv_std: Option<f64>,
    /// Inference time in milliseconds
    pub inference_time_ms: f64,
    /// Optional trace ID for observability
    pub trace_id: Option<String>,
}

impl EvalResult {
    /// Create new eval result
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            scores: HashMap::new(),
            cv_scores: None,
            cv_mean: None,
            cv_std: None,
            inference_time_ms: 0.0,
            trace_id: None,
        }
    }

    /// Get score for a specific metric
    pub fn get_score(&self, metric: Metric) -> Option<f64> {
        self.scores.get(&metric).copied()
    }

    /// Add a score
    pub fn add_score(&mut self, metric: Metric, score: f64) {
        self.scores.insert(metric, score);
    }
}

impl fmt::Display for EvalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model: {}", self.model_name)?;
        writeln!(f, "Metrics:")?;
        for (metric, score) in &self.scores {
            writeln!(f, "  {metric}: {score:.4}")?;
        }
        writeln!(f, "Inference time: {:.2}ms", self.inference_time_ms)?;
        Ok(())
    }
}
