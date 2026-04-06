//! Hyperparameter sweep executor (Kaizen principle).

use entrenar_common::Result;

/// Sweep configuration.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Parameter to sweep
    pub parameter: SweepParameter,
    /// Number of runs per configuration
    pub runs_per_point: usize,
    /// Whether to use early stopping
    pub early_stop: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl SweepConfig {
    /// Create a temperature sweep.
    pub fn temperature(range: std::ops::Range<f32>, step: f32) -> Self {
        Self {
            parameter: SweepParameter::Temperature { start: range.start, end: range.end, step },
            runs_per_point: 1,
            early_stop: false,
            seed: Some(42),
        }
    }

    /// Create an alpha sweep.
    pub fn alpha(range: std::ops::Range<f32>, step: f32) -> Self {
        Self {
            parameter: SweepParameter::Alpha { start: range.start, end: range.end, step },
            runs_per_point: 1,
            early_stop: false,
            seed: Some(42),
        }
    }

    /// Set number of runs per point.
    pub fn with_runs(mut self, runs: usize) -> Self {
        self.runs_per_point = runs;
        self
    }

    /// Enable early stopping.
    pub fn with_early_stop(mut self) -> Self {
        self.early_stop = true;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Parameter being swept.
#[derive(Debug, Clone)]
pub enum SweepParameter {
    /// Temperature parameter
    Temperature { start: f32, end: f32, step: f32 },
    /// Alpha parameter
    Alpha { start: f32, end: f32, step: f32 },
    /// LoRA rank
    Rank { values: Vec<u32> },
    /// Learning rate
    LearningRate { values: Vec<f64> },
}

impl SweepParameter {
    /// Get the values to sweep over.
    pub fn values(&self) -> Vec<f64> {
        match self {
            Self::Temperature { start, end, step } | Self::Alpha { start, end, step } => {
                let mut values = Vec::new();
                let mut v = *start;
                while v <= *end {
                    values.push(f64::from(v));
                    v += step;
                }
                values
            }
            Self::Rank { values } => values.iter().map(|&v| f64::from(v)).collect(),
            Self::LearningRate { values } => values.clone(),
        }
    }

    /// Get parameter name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Temperature { .. } => "temperature",
            Self::Alpha { .. } => "alpha",
            Self::Rank { .. } => "rank",
            Self::LearningRate { .. } => "learning_rate",
        }
    }
}

/// Sweep executor.
pub struct Sweeper {
    config: SweepConfig,
}

impl Sweeper {
    /// Create a new sweeper.
    pub fn new(config: SweepConfig) -> Self {
        Self { config }
    }

    /// Run the sweep.
    pub fn run(&self) -> Result<SweepResult> {
        let values = self.config.parameter.values();
        let mut data_points = Vec::new();

        for value in &values {
            let mut metrics = Vec::new();

            for run in 0..self.config.runs_per_point {
                // Simulate training with this configuration
                let result = self.simulate_training(*value, run);
                metrics.push(result);
            }

            // Aggregate metrics across runs
            let mean_loss = metrics.iter().map(|m| m.loss).sum::<f64>() / metrics.len() as f64;
            let mean_accuracy =
                metrics.iter().map(|m| m.accuracy).sum::<f64>() / metrics.len() as f64;
            let std_loss = self.calculate_std(&metrics.iter().map(|m| m.loss).collect::<Vec<_>>());
            let std_accuracy =
                self.calculate_std(&metrics.iter().map(|m| m.accuracy).collect::<Vec<_>>());

            data_points.push(DataPoint {
                parameter_value: *value,
                mean_loss,
                std_loss,
                mean_accuracy,
                std_accuracy,
                runs: metrics.len(),
            });
        }

        // Find optimal
        let optimal = data_points
            .iter()
            .min_by(|a, b| {
                a.mean_loss.partial_cmp(&b.mean_loss).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        Ok(SweepResult {
            parameter_name: self.config.parameter.name().to_string(),
            data_points,
            optimal,
            config: self.config.clone(),
        })
    }

    fn simulate_training(&self, param_value: f64, run: usize) -> TrainingMetrics {
        // Simulated training - in real implementation would run actual training
        // Using a simple model where:
        // - Temperature ~4.0 is optimal
        // - Alpha ~0.7 is optimal

        let seed_offset = self.config.seed.unwrap_or(0) + run as u64;
        let noise = (seed_offset as f64 * 0.1).sin() * 0.05; // Deterministic "randomness"

        let param_name = self.config.parameter.name();

        let (loss, accuracy) = match param_name {
            "temperature" => {
                // Optimal around 4.0
                let deviation = (param_value - 4.0).abs();
                let loss = 0.65 + deviation * 0.1 + noise;
                let accuracy = 0.83 - deviation * 0.02 + noise * 0.5;
                (loss, accuracy.clamp(0.0, 1.0))
            }
            "alpha" => {
                // Optimal around 0.7
                let deviation = (param_value - 0.7).abs();
                let loss = 0.65 + deviation * 0.2 + noise;
                let accuracy = 0.83 - deviation * 0.05 + noise * 0.5;
                (loss, accuracy.clamp(0.0, 1.0))
            }
            _ => (0.8 + noise, 0.75 + noise * 0.5),
        };

        TrainingMetrics {
            loss,
            accuracy,
            throughput: 1200.0 + noise * 100.0,
            duration_secs: 3600.0 + noise * 600.0,
        }
    }

    fn calculate_std(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }
}

/// Training metrics from a single run.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Final loss
    pub loss: f64,
    /// Final accuracy
    pub accuracy: f64,
    /// Training throughput (samples/sec)
    pub throughput: f64,
    /// Training duration in seconds
    pub duration_secs: f64,
}

/// A single data point in the sweep.
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Parameter value
    pub parameter_value: f64,
    /// Mean loss across runs
    pub mean_loss: f64,
    /// Standard deviation of loss
    pub std_loss: f64,
    /// Mean accuracy across runs
    pub mean_accuracy: f64,
    /// Standard deviation of accuracy
    pub std_accuracy: f64,
    /// Number of runs
    pub runs: usize,
}

/// Result of a sweep.
#[derive(Debug, Clone)]
pub struct SweepResult {
    /// Parameter name
    pub parameter_name: String,
    /// Data points
    pub data_points: Vec<DataPoint>,
    /// Optimal configuration
    pub optimal: Option<DataPoint>,
    /// Original configuration
    pub config: SweepConfig,
}

impl SweepResult {
    /// Format as ASCII table.
    pub fn to_table(&self) -> String {
        let mut output = format!("{} Sweep Results\n", self.parameter_name);
        output.push_str("┌─────────────┬────────────┬────────────┬────────────┐\n");
        output.push_str("│ Value       │ Loss       │ Accuracy   │ Runs       │\n");
        output.push_str("├─────────────┼────────────┼────────────┼────────────┤\n");

        for point in &self.data_points {
            let optimal_marker = if self.optimal.as_ref().map(|o| o.parameter_value)
                == Some(point.parameter_value)
            {
                " ★"
            } else {
                ""
            };

            output.push_str(&format!(
                "│ {:>10.2} │ {:>10.4} │ {:>9.1}% │ {:>10}{} │\n",
                point.parameter_value,
                point.mean_loss,
                point.mean_accuracy * 100.0,
                point.runs,
                optimal_marker
            ));
        }

        output.push_str("└─────────────┴────────────┴────────────┴────────────┘\n");

        if let Some(optimal) = &self.optimal {
            output.push_str(&format!(
                "\nOptimal: {} = {:.2} (loss={:.4}, accuracy={:.1}%)\n",
                self.parameter_name,
                optimal.parameter_value,
                optimal.mean_loss,
                optimal.mean_accuracy * 100.0
            ));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sweep_config_temperature() {
        let config = SweepConfig::temperature(1.0..5.0, 1.0);
        assert_eq!(config.parameter.name(), "temperature");

        let values = config.parameter.values();
        assert_eq!(values.len(), 5); // 1, 2, 3, 4, 5
    }

    #[test]
    fn test_sweep_config_alpha() {
        let config = SweepConfig::alpha(0.1..0.9, 0.1);
        assert_eq!(config.parameter.name(), "alpha");
    }

    #[test]
    fn test_sweeper_runs() {
        let config = SweepConfig::temperature(1.0..3.0, 1.0).with_runs(2);
        let sweeper = Sweeper::new(config);
        let result = sweeper.run().expect("operation should succeed");

        assert!(!result.data_points.is_empty());
        assert!(result.optimal.is_some());
    }

    #[test]
    fn test_sweeper_finds_optimal_temperature() {
        let config = SweepConfig::temperature(2.0..6.0, 1.0).with_runs(1);
        let sweeper = Sweeper::new(config);
        let result = sweeper.run().expect("operation should succeed");

        // Optimal should be around 4.0
        let optimal = result.optimal.expect("operation should succeed");
        assert!((optimal.parameter_value - 4.0).abs() < 1.5);
    }

    #[test]
    fn test_sweep_result_table() {
        let config = SweepConfig::temperature(1.0..3.0, 1.0);
        let sweeper = Sweeper::new(config);
        let result = sweeper.run().expect("operation should succeed");

        let table = result.to_table();
        assert!(table.contains("temperature"));
        assert!(table.contains("Loss"));
        assert!(table.contains("Accuracy"));
    }

    #[test]
    fn test_std_calculation() {
        let sweeper = Sweeper::new(SweepConfig::temperature(1.0..2.0, 1.0));

        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = sweeper.calculate_std(&values);
        assert!((std - 1.58).abs() < 0.1); // sqrt(2.5) ≈ 1.58
    }

    #[test]
    fn test_std_calculation_single_value() {
        let sweeper = Sweeper::new(SweepConfig::temperature(1.0..2.0, 1.0));

        let values = vec![5.0];
        let std = sweeper.calculate_std(&values);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_std_calculation_empty() {
        let sweeper = Sweeper::new(SweepConfig::temperature(1.0..2.0, 1.0));

        let values: Vec<f64> = vec![];
        let std = sweeper.calculate_std(&values);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_sweep_config_with_seed() {
        let config = SweepConfig::temperature(1.0..5.0, 1.0).with_seed(123);
        assert_eq!(config.seed, Some(123));
    }

    #[test]
    fn test_sweep_config_with_early_stop() {
        let config = SweepConfig::temperature(1.0..5.0, 1.0).with_early_stop();
        assert!(config.early_stop);
    }

    #[test]
    fn test_sweep_config_with_runs() {
        let config = SweepConfig::temperature(1.0..5.0, 1.0).with_runs(10);
        assert_eq!(config.runs_per_point, 10);
    }

    #[test]
    fn test_sweep_parameter_rank() {
        let param = SweepParameter::Rank { values: vec![8, 16, 32, 64] };
        let values = param.values();
        assert_eq!(values, vec![8.0, 16.0, 32.0, 64.0]);
        assert_eq!(param.name(), "rank");
    }

    #[test]
    fn test_sweep_parameter_learning_rate() {
        let param = SweepParameter::LearningRate { values: vec![1e-5, 1e-4, 1e-3] };
        let values = param.values();
        assert_eq!(values, vec![1e-5, 1e-4, 1e-3]);
        assert_eq!(param.name(), "learning_rate");
    }

    #[test]
    fn test_sweep_result_fields() {
        let config = SweepConfig::temperature(1.0..3.0, 1.0);
        let sweeper = Sweeper::new(config);
        let result = sweeper.run().expect("operation should succeed");

        assert_eq!(result.parameter_name, "temperature");
        assert!(!result.data_points.is_empty());
    }

    #[test]
    fn test_data_point_fields() {
        let point = DataPoint {
            parameter_value: 4.0,
            mean_loss: 0.65,
            std_loss: 0.02,
            mean_accuracy: 0.83,
            std_accuracy: 0.01,
            runs: 5,
        };

        assert_eq!(point.parameter_value, 4.0);
        assert_eq!(point.runs, 5);
    }

    #[test]
    fn test_training_metrics_fields() {
        let metrics = TrainingMetrics {
            loss: 0.75,
            accuracy: 0.82,
            throughput: 1200.0,
            duration_secs: 3600.0,
        };

        assert_eq!(metrics.loss, 0.75);
        assert_eq!(metrics.throughput, 1200.0);
    }

    #[test]
    fn test_sweep_result_table_optimal() {
        let config = SweepConfig::temperature(3.0..5.0, 1.0);
        let sweeper = Sweeper::new(config);
        let result = sweeper.run().expect("operation should succeed");

        let table = result.to_table();

        // Should contain "Optimal" section
        assert!(table.contains("Optimal"));
        assert!(table.contains('★'));
    }

    #[test]
    fn test_sweep_deterministic() {
        let config = SweepConfig::temperature(1.0..3.0, 1.0).with_seed(42);
        let sweeper = Sweeper::new(config.clone());
        let result1 = sweeper.run().expect("operation should succeed");

        let sweeper2 = Sweeper::new(config);
        let result2 = sweeper2.run().expect("operation should succeed");

        // Same seed should produce same results
        assert_eq!(result1.data_points[0].mean_loss, result2.data_points[0].mean_loss);
    }

    #[test]
    fn test_alpha_sweep_finds_optimal() {
        let config = SweepConfig::alpha(0.3..0.9, 0.2).with_runs(1);
        let sweeper = Sweeper::new(config);
        let result = sweeper.run().expect("operation should succeed");

        // Optimal should be around 0.7
        let optimal = result.optimal.expect("operation should succeed");
        assert!((optimal.parameter_value - 0.7).abs() < 0.3);
    }

    #[test]
    fn test_sweep_multiple_runs() {
        let config = SweepConfig::temperature(3.0..5.0, 1.0).with_runs(3);
        let sweeper = Sweeper::new(config);
        let result = sweeper.run().expect("operation should succeed");

        // Each data point should have 3 runs
        for point in &result.data_points {
            assert_eq!(point.runs, 3);
        }
    }
}
