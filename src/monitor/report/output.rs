//! Post-training report structure.

use super::types::{MetricSummary, TrainingIssue};
use crate::monitor::Metric;
use std::collections::HashMap;

/// Post-training analysis report (Hansei)
#[derive(Debug, Clone)]
pub struct PostTrainingReport {
    pub training_id: String,
    pub duration_secs: f64,
    pub total_steps: u64,
    pub final_metrics: HashMap<Metric, f64>,
    pub metric_summaries: HashMap<Metric, MetricSummary>,
    pub issues: Vec<TrainingIssue>,
    pub recommendations: Vec<String>,
}
