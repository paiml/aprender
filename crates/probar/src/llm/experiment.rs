//! ML experiment tracking with pre-flight data audits, budget gates,
//! early stopping, and kill criteria evaluation.
//!
//! Provides structured experiment lifecycle management to prevent
//! wasted GPU time on doomed training runs.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Status of an experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    /// Experiment created but not started.
    Created,
    /// Currently running.
    Running,
    /// Completed successfully.
    Completed,
    /// Stopped by early stopping criteria.
    EarlyStopped,
    /// Killed by kill criteria.
    Killed,
    /// Failed with error.
    Failed,
}

/// A single metric snapshot at a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    /// Checkpoint number (epoch or step).
    pub checkpoint: u64,
    /// Metric name → value.
    pub metrics: BTreeMap<String, f64>,
    /// Wall-clock time since experiment start.
    pub elapsed_secs: f64,
    /// Estimated GPU-hours consumed so far.
    pub gpu_hours: f64,
}

/// Kill criterion: stop training if a metric crosses a threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillCriterion {
    /// Metric to monitor (e.g., "eval_accuracy").
    pub metric: String,
    /// Checkpoint at which to evaluate.
    pub at_checkpoint: u64,
    /// Minimum acceptable value (kill if below).
    pub min_value: Option<f64>,
    /// Maximum acceptable value (kill if above, e.g., for loss).
    pub max_value: Option<f64>,
}

impl KillCriterion {
    /// Check if a metric snapshot triggers this kill criterion.
    #[must_use]
    pub fn evaluate(&self, snapshot: &MetricSnapshot) -> Option<String> {
        if snapshot.checkpoint != self.at_checkpoint {
            return None;
        }
        let value = snapshot.metrics.get(&self.metric)?;
        if let Some(min) = self.min_value {
            if *value < min {
                return Some(format!(
                    "Kill criterion triggered: {} = {:.4} < {:.4} at checkpoint {}",
                    self.metric, value, min, self.at_checkpoint
                ));
            }
        }
        if let Some(max) = self.max_value {
            if *value > max {
                return Some(format!(
                    "Kill criterion triggered: {} = {:.4} > {:.4} at checkpoint {}",
                    self.metric, value, max, self.at_checkpoint
                ));
            }
        }
        None
    }
}

/// Early stopping configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Metric to monitor (e.g., "eval_loss").
    pub metric: String,
    /// Number of checkpoints without improvement before stopping.
    pub patience: u32,
    /// Minimum improvement to count as progress.
    pub min_delta: f64,
    /// Whether lower is better (true for loss, false for accuracy).
    pub lower_is_better: bool,
}

/// Early stopping state tracker.
#[derive(Debug, Clone)]
pub struct EarlyStoppingState {
    config: EarlyStoppingConfig,
    best_value: Option<f64>,
    best_checkpoint: u64,
    stale_count: u32,
}

impl EarlyStoppingState {
    /// Create a new early stopping tracker.
    #[must_use]
    pub fn new(config: EarlyStoppingConfig) -> Self {
        Self {
            config,
            best_value: None,
            best_checkpoint: 0,
            stale_count: 0,
        }
    }

    /// Update with a new metric snapshot. Returns true if training should stop.
    pub fn update(&mut self, snapshot: &MetricSnapshot) -> bool {
        let value = match snapshot.metrics.get(&self.config.metric) {
            Some(v) => *v,
            None => return false,
        };

        let improved = match self.best_value {
            None => true,
            Some(best) => {
                if self.config.lower_is_better {
                    value < best - self.config.min_delta
                } else {
                    value > best + self.config.min_delta
                }
            }
        };

        if improved {
            self.best_value = Some(value);
            self.best_checkpoint = snapshot.checkpoint;
            self.stale_count = 0;
            false
        } else {
            self.stale_count += 1;
            self.stale_count >= self.config.patience
        }
    }

    /// Current stale count.
    #[must_use]
    pub fn stale_count(&self) -> u32 {
        self.stale_count
    }

    /// Best value seen so far.
    #[must_use]
    pub fn best_value(&self) -> Option<f64> {
        self.best_value
    }
}

/// Budget configuration for GPU-hour or cost limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum GPU-hours allowed.
    pub max_gpu_hours: Option<f64>,
    /// Maximum cost in USD (at given rate).
    pub max_cost_usd: Option<f64>,
    /// Cost per GPU-hour (for cost estimation).
    pub cost_per_gpu_hour: f64,
}

impl BudgetConfig {
    /// Check if a given GPU-hours usage exceeds the budget.
    #[must_use]
    pub fn exceeds_budget(&self, gpu_hours: f64) -> Option<String> {
        if let Some(max) = self.max_gpu_hours {
            if gpu_hours > max {
                return Some(format!(
                    "GPU-hour budget exceeded: {gpu_hours:.2}h > {max:.2}h"
                ));
            }
        }
        if let Some(max_cost) = self.max_cost_usd {
            let cost = gpu_hours * self.cost_per_gpu_hour;
            if cost > max_cost {
                return Some(format!("Cost budget exceeded: ${cost:.2} > ${max_cost:.2}"));
            }
        }
        None
    }

    /// Estimated cost for given GPU-hours.
    #[must_use]
    pub fn estimated_cost(&self, gpu_hours: f64) -> f64 {
        gpu_hours * self.cost_per_gpu_hour
    }
}

/// Result of a pre-flight data audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAuditResult {
    /// Total number of samples.
    pub total_samples: usize,
    /// Label distribution (label → count).
    pub label_distribution: BTreeMap<String, usize>,
    /// Maximum class imbalance ratio (majority / minority).
    pub imbalance_ratio: f64,
    /// Whether the audit passed (imbalance_ratio <= threshold).
    pub passed: bool,
    /// Detected issues.
    pub issues: Vec<String>,
}

impl DataAuditResult {
    /// Run a data audit on label counts.
    #[must_use]
    pub fn from_labels(labels: &[String], max_imbalance: f64) -> Self {
        let mut distribution = BTreeMap::new();
        for label in labels {
            *distribution.entry(label.clone()).or_insert(0usize) += 1;
        }

        let counts: Vec<usize> = distribution.values().copied().collect();
        let min_count = counts.iter().copied().min().unwrap_or(0);
        let max_count = counts.iter().copied().max().unwrap_or(0);

        let imbalance_ratio = if min_count == 0 {
            f64::INFINITY
        } else {
            max_count as f64 / min_count as f64
        };

        let mut issues = Vec::new();

        if imbalance_ratio > max_imbalance {
            issues.push(format!(
                "Class imbalance {imbalance_ratio:.1}:1 exceeds threshold {max_imbalance:.1}:1"
            ));
        }

        if labels.is_empty() {
            issues.push("Dataset is empty".to_string());
        }

        if distribution.len() < 2 {
            issues.push(format!(
                "Only {} class(es) found — need at least 2",
                distribution.len()
            ));
        }

        let passed = issues.is_empty();

        Self {
            total_samples: labels.len(),
            label_distribution: distribution,
            imbalance_ratio,
            passed,
            issues,
        }
    }
}

/// A single experiment run with hyperparameters and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRun {
    /// Run identifier (e.g., "run-1", "run-6").
    pub id: String,
    /// Hyperparameters used.
    pub hyperparameters: BTreeMap<String, serde_json::Value>,
    /// Metric snapshots at each checkpoint.
    pub snapshots: Vec<MetricSnapshot>,
    /// Final status.
    pub status: ExperimentStatus,
    /// Why the run ended (if not Completed).
    pub stop_reason: Option<String>,
    /// Total GPU-hours consumed.
    pub total_gpu_hours: f64,
    /// Wall-clock duration.
    pub wall_clock_secs: f64,
}

impl ExperimentRun {
    /// Create a new run.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            hyperparameters: BTreeMap::new(),
            snapshots: Vec::new(),
            status: ExperimentStatus::Created,
            total_gpu_hours: 0.0,
            wall_clock_secs: 0.0,
            stop_reason: None,
        }
    }

    /// Get the last snapshot's value for a metric.
    #[must_use]
    pub fn last_metric(&self, name: &str) -> Option<f64> {
        self.snapshots
            .last()
            .and_then(|s| s.metrics.get(name).copied())
    }

    /// Get the best value for a metric across all snapshots.
    #[must_use]
    pub fn best_metric(&self, name: &str, lower_is_better: bool) -> Option<f64> {
        self.snapshots
            .iter()
            .filter_map(|s| s.metrics.get(name).copied())
            .reduce(|a, b| if lower_is_better { a.min(b) } else { a.max(b) })
    }
}

/// Top-level experiment container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Experiment name.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// Creation timestamp (ISO 8601).
    pub created: String,
    /// Data audit result (if run).
    pub data_audit: Option<DataAuditResult>,
    /// Budget configuration.
    pub budget: Option<BudgetConfig>,
    /// Early stopping configuration.
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Kill criteria.
    pub kill_criteria: Vec<KillCriterion>,
    /// All runs in this experiment.
    pub runs: Vec<ExperimentRun>,
}

impl Experiment {
    /// Create a new experiment.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            created: chrono::Utc::now().to_rfc3339(),
            data_audit: None,
            budget: None,
            early_stopping: None,
            kill_criteria: Vec::new(),
            runs: Vec::new(),
        }
    }

    /// Add a run to the experiment.
    pub fn add_run(&mut self, run: ExperimentRun) {
        self.runs.push(run);
    }

    /// Total GPU-hours across all runs.
    #[must_use]
    pub fn total_gpu_hours(&self) -> f64 {
        self.runs.iter().map(|r| r.total_gpu_hours).sum()
    }

    /// Estimated total cost.
    #[must_use]
    pub fn total_cost(&self) -> Option<f64> {
        self.budget
            .as_ref()
            .map(|b| b.estimated_cost(self.total_gpu_hours()))
    }

    /// Save experiment state to a JSON file.
    ///
    /// # Errors
    /// Returns error if serialization or file write fails.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("Serialize error: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("Write error: {e}"))
    }

    /// Load experiment state from a JSON file.
    ///
    /// # Errors
    /// Returns error if file read or deserialization fails.
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("Read error: {e}"))?;
        serde_json::from_str(&data).map_err(|e| format!("Parse error: {e}"))
    }

    /// Compare two runs by metric name.
    #[must_use]
    pub fn compare_runs(
        &self,
        run_a: &str,
        run_b: &str,
        metric: &str,
        lower_is_better: bool,
    ) -> Option<RunComparison> {
        let a = self.runs.iter().find(|r| r.id == run_a)?;
        let b = self.runs.iter().find(|r| r.id == run_b)?;
        let best_a = a.best_metric(metric, lower_is_better)?;
        let best_b = b.best_metric(metric, lower_is_better)?;
        let diff = best_b - best_a;
        let pct = if best_a.abs() > f64::EPSILON {
            (diff / best_a) * 100.0
        } else {
            0.0
        };
        Some(RunComparison {
            metric: metric.to_string(),
            run_a: run_a.to_string(),
            run_b: run_b.to_string(),
            value_a: best_a,
            value_b: best_b,
            diff,
            diff_pct: pct,
        })
    }
}

/// Result of comparing two runs on a metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunComparison {
    /// Metric name.
    pub metric: String,
    /// First run ID.
    pub run_a: String,
    /// Second run ID.
    pub run_b: String,
    /// Best value from run A.
    pub value_a: f64,
    /// Best value from run B.
    pub value_b: f64,
    /// Absolute difference (B - A).
    pub diff: f64,
    /// Percentage difference.
    pub diff_pct: f64,
}

/// Audit a JSONL data file for label distribution.
///
/// Expects each line to be a JSON object with a `label` field.
///
/// # Errors
/// Returns error if the file cannot be read or parsed.
pub fn audit_jsonl_file(path: &Path, max_imbalance: f64) -> Result<DataAuditResult, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("Read error: {e}"))?;
    let mut labels = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obj: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("Line {}: parse error: {e}", i + 1))?;
        let label = obj
            .get("label")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("Line {}: missing 'label' field", i + 1))?;
        labels.push(label.to_string());
    }
    Ok(DataAuditResult::from_labels(&labels, max_imbalance))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_data_audit_balanced() {
        let labels: Vec<String> = vec!["A", "A", "B", "B", "C", "C"]
            .into_iter()
            .map(String::from)
            .collect();
        let result = DataAuditResult::from_labels(&labels, 3.0);
        assert!(result.passed);
        assert_eq!(result.total_samples, 6);
        assert!((result.imbalance_ratio - 1.0).abs() < f64::EPSILON);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_data_audit_imbalanced() {
        let labels: Vec<String> = vec!["A", "A", "A", "A", "B"]
            .into_iter()
            .map(String::from)
            .collect();
        let result = DataAuditResult::from_labels(&labels, 3.0);
        assert!(!result.passed);
        assert!((result.imbalance_ratio - 4.0).abs() < f64::EPSILON);
        assert!(result.issues[0].contains("imbalance"));
    }

    #[test]
    fn test_data_audit_empty() {
        let labels: Vec<String> = vec![];
        let result = DataAuditResult::from_labels(&labels, 3.0);
        assert!(!result.passed);
        assert!(result.issues.iter().any(|i| i.contains("empty")));
    }

    #[test]
    fn test_data_audit_single_class() {
        let labels: Vec<String> = vec!["A", "A", "A"].into_iter().map(String::from).collect();
        let result = DataAuditResult::from_labels(&labels, 3.0);
        assert!(!result.passed);
        assert!(result.issues.iter().any(|i| i.contains("1 class")));
    }

    #[test]
    fn test_kill_criterion_triggers() {
        let criterion = KillCriterion {
            metric: "eval_accuracy".to_string(),
            at_checkpoint: 3,
            min_value: Some(0.5),
            max_value: None,
        };

        let mut metrics = BTreeMap::new();
        metrics.insert("eval_accuracy".to_string(), 0.32);

        let snapshot = MetricSnapshot {
            checkpoint: 3,
            metrics,
            elapsed_secs: 100.0,
            gpu_hours: 0.5,
        };

        let result = criterion.evaluate(&snapshot);
        assert!(result.is_some());
        assert!(result.unwrap().contains("Kill criterion triggered"));
    }

    #[test]
    fn test_kill_criterion_wrong_checkpoint() {
        let criterion = KillCriterion {
            metric: "eval_accuracy".to_string(),
            at_checkpoint: 3,
            min_value: Some(0.5),
            max_value: None,
        };

        let mut metrics = BTreeMap::new();
        metrics.insert("eval_accuracy".to_string(), 0.32);

        let snapshot = MetricSnapshot {
            checkpoint: 2,
            metrics,
            elapsed_secs: 100.0,
            gpu_hours: 0.5,
        };

        assert!(criterion.evaluate(&snapshot).is_none());
    }

    #[test]
    fn test_kill_criterion_passes() {
        let criterion = KillCriterion {
            metric: "eval_accuracy".to_string(),
            at_checkpoint: 3,
            min_value: Some(0.5),
            max_value: None,
        };

        let mut metrics = BTreeMap::new();
        metrics.insert("eval_accuracy".to_string(), 0.72);

        let snapshot = MetricSnapshot {
            checkpoint: 3,
            metrics,
            elapsed_secs: 100.0,
            gpu_hours: 0.5,
        };

        assert!(criterion.evaluate(&snapshot).is_none());
    }

    #[test]
    fn test_early_stopping_loss() {
        let config = EarlyStoppingConfig {
            metric: "eval_loss".to_string(),
            patience: 3,
            min_delta: 0.01,
            lower_is_better: true,
        };
        let mut state = EarlyStoppingState::new(config);

        // Loss decreasing — no stop
        for (i, loss) in [1.0, 0.8, 0.6, 0.4].iter().enumerate() {
            let mut metrics = BTreeMap::new();
            metrics.insert("eval_loss".to_string(), *loss);
            let snapshot = MetricSnapshot {
                checkpoint: i as u64,
                metrics,
                elapsed_secs: 0.0,
                gpu_hours: 0.0,
            };
            assert!(!state.update(&snapshot));
        }

        // Loss plateaus — stop after patience=3
        for i in 4..7 {
            let mut metrics = BTreeMap::new();
            metrics.insert("eval_loss".to_string(), 0.4);
            let snapshot = MetricSnapshot {
                checkpoint: i,
                metrics,
                elapsed_secs: 0.0,
                gpu_hours: 0.0,
            };
            let should_stop = state.update(&snapshot);
            if i == 6 {
                assert!(should_stop);
            } else {
                assert!(!should_stop);
            }
        }
    }

    #[test]
    fn test_early_stopping_accuracy() {
        let config = EarlyStoppingConfig {
            metric: "eval_accuracy".to_string(),
            patience: 2,
            min_delta: 0.01,
            lower_is_better: false,
        };
        let mut state = EarlyStoppingState::new(config);

        let mut metrics = BTreeMap::new();
        metrics.insert("eval_accuracy".to_string(), 0.5);
        assert!(!state.update(&MetricSnapshot {
            checkpoint: 0,
            metrics,
            elapsed_secs: 0.0,
            gpu_hours: 0.0,
        }));

        // No improvement
        let mut metrics = BTreeMap::new();
        metrics.insert("eval_accuracy".to_string(), 0.5);
        assert!(!state.update(&MetricSnapshot {
            checkpoint: 1,
            metrics,
            elapsed_secs: 0.0,
            gpu_hours: 0.0,
        }));

        // Still no improvement — patience=2 hit
        let mut metrics = BTreeMap::new();
        metrics.insert("eval_accuracy".to_string(), 0.5);
        assert!(state.update(&MetricSnapshot {
            checkpoint: 2,
            metrics,
            elapsed_secs: 0.0,
            gpu_hours: 0.0,
        }));
    }

    #[test]
    fn test_budget_gpu_hours() {
        let budget = BudgetConfig {
            max_gpu_hours: Some(2.0),
            max_cost_usd: None,
            cost_per_gpu_hour: 3.50,
        };
        assert!(budget.exceeds_budget(1.5).is_none());
        assert!(budget.exceeds_budget(2.5).is_some());
    }

    #[test]
    fn test_budget_cost() {
        let budget = BudgetConfig {
            max_gpu_hours: None,
            max_cost_usd: Some(10.0),
            cost_per_gpu_hour: 3.50,
        };
        assert!(budget.exceeds_budget(2.0).is_none()); // $7
        assert!(budget.exceeds_budget(3.0).is_some()); // $10.50
    }

    #[test]
    fn test_experiment_run_best_metric() {
        let mut run = ExperimentRun::new("run-1");
        for (i, acc) in [0.3, 0.5, 0.45, 0.52].iter().enumerate() {
            let mut metrics = BTreeMap::new();
            metrics.insert("accuracy".to_string(), *acc);
            run.snapshots.push(MetricSnapshot {
                checkpoint: i as u64,
                metrics,
                elapsed_secs: 0.0,
                gpu_hours: 0.0,
            });
        }
        assert!((run.best_metric("accuracy", false).unwrap() - 0.52).abs() < f64::EPSILON);
        assert!((run.best_metric("accuracy", true).unwrap() - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_experiment_compare_runs() {
        let mut exp = Experiment::new("test");

        let mut run_a = ExperimentRun::new("run-1");
        let mut m = BTreeMap::new();
        m.insert("accuracy".to_string(), 0.32);
        run_a.snapshots.push(MetricSnapshot {
            checkpoint: 0,
            metrics: m,
            elapsed_secs: 0.0,
            gpu_hours: 1.0,
        });
        run_a.total_gpu_hours = 1.0;

        let mut run_b = ExperimentRun::new("run-6");
        let mut m = BTreeMap::new();
        m.insert("accuracy".to_string(), 0.48);
        run_b.snapshots.push(MetricSnapshot {
            checkpoint: 0,
            metrics: m,
            elapsed_secs: 0.0,
            gpu_hours: 2.0,
        });
        run_b.total_gpu_hours = 2.0;

        exp.add_run(run_a);
        exp.add_run(run_b);

        let cmp = exp
            .compare_runs("run-1", "run-6", "accuracy", false)
            .unwrap();
        assert!((cmp.value_a - 0.32).abs() < f64::EPSILON);
        assert!((cmp.value_b - 0.48).abs() < f64::EPSILON);
        assert!(cmp.diff_pct > 0.0);
    }

    #[test]
    fn test_experiment_total_gpu_hours() {
        let mut exp = Experiment::new("test");
        let mut r1 = ExperimentRun::new("r1");
        r1.total_gpu_hours = 1.5;
        let mut r2 = ExperimentRun::new("r2");
        r2.total_gpu_hours = 2.5;
        exp.add_run(r1);
        exp.add_run(r2);
        assert!((exp.total_gpu_hours() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_experiment_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("experiment.json");

        let mut exp = Experiment::new("roundtrip-test");
        exp.description = Some("Test save/load".to_string());
        exp.budget = Some(BudgetConfig {
            max_gpu_hours: Some(5.0),
            max_cost_usd: Some(20.0),
            cost_per_gpu_hour: 3.50,
        });

        let mut run = ExperimentRun::new("run-1");
        run.total_gpu_hours = 1.0;
        run.status = ExperimentStatus::Completed;
        exp.add_run(run);

        exp.save(&path).unwrap();
        let loaded = Experiment::load(&path).unwrap();

        assert_eq!(loaded.name, "roundtrip-test");
        assert_eq!(loaded.runs.len(), 1);
        assert_eq!(loaded.runs[0].status, ExperimentStatus::Completed);
    }

    #[test]
    fn test_audit_jsonl_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl");
        std::fs::write(
            &path,
            r#"{"label": "safe", "text": "hello"}
{"label": "safe", "text": "world"}
{"label": "unsafe", "text": "bad"}
{"label": "safe", "text": "ok"}
"#,
        )
        .unwrap();

        let result = audit_jsonl_file(&path, 3.0).unwrap();
        assert!(result.passed);
        assert_eq!(result.total_samples, 4);
        assert_eq!(result.label_distribution["safe"], 3);
        assert_eq!(result.label_distribution["unsafe"], 1);
    }

    #[test]
    fn test_audit_jsonl_imbalanced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl");
        let mut lines = String::new();
        for _ in 0..10 {
            lines.push_str("{\"label\": \"A\", \"text\": \"x\"}\n");
        }
        lines.push_str("{\"label\": \"B\", \"text\": \"y\"}\n");
        std::fs::write(&path, lines).unwrap();

        let result = audit_jsonl_file(&path, 3.0).unwrap();
        assert!(!result.passed);
        assert!((result.imbalance_ratio - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_kill_criterion_max_value() {
        let criterion = KillCriterion {
            metric: "eval_loss".to_string(),
            at_checkpoint: 5,
            min_value: None,
            max_value: Some(2.0),
        };

        let mut metrics = BTreeMap::new();
        metrics.insert("eval_loss".to_string(), 2.5);
        let snapshot = MetricSnapshot {
            checkpoint: 5,
            metrics,
            elapsed_secs: 0.0,
            gpu_hours: 0.0,
        };
        assert!(criterion.evaluate(&snapshot).is_some());
    }

    #[test]
    fn test_experiment_status_serialization() {
        let json = serde_json::to_string(&ExperimentStatus::EarlyStopped).unwrap();
        assert_eq!(json, "\"early_stopped\"");
        let back: ExperimentStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ExperimentStatus::EarlyStopped);
    }

    #[test]
    fn test_early_stopping_state_accessors() {
        let config = EarlyStoppingConfig {
            metric: "loss".to_string(),
            patience: 3,
            min_delta: 0.01,
            lower_is_better: true,
        };
        let mut state = EarlyStoppingState::new(config);
        assert_eq!(state.stale_count(), 0);
        assert!(state.best_value().is_none());

        let mut metrics = BTreeMap::new();
        metrics.insert("loss".to_string(), 1.0);
        state.update(&MetricSnapshot {
            checkpoint: 0,
            metrics,
            elapsed_secs: 0.0,
            gpu_hours: 0.0,
        });
        assert!((state.best_value().unwrap() - 1.0).abs() < f64::EPSILON);
    }
}
