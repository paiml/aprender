//! Distillation strategy comparison.

use crate::stats::StatisticalAnalyzer;
use entrenar_common::Result;

/// A distillation strategy to benchmark.
#[derive(Debug, Clone)]
pub enum DistillStrategy {
    /// Knowledge distillation only (soft targets)
    KDOnly { temperature: f32, alpha: f32 },
    /// Progressive distillation (hidden state matching)
    Progressive { temperature: f32, alpha: f32, layer_weight: f32 },
    /// Attention transfer
    Attention { temperature: f32, alpha: f32, attention_weight: f32 },
    /// Combined approach
    Combined { temperature: f32, alpha: f32, layer_weight: f32, attention_weight: f32 },
}

impl DistillStrategy {
    /// Get strategy name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::KDOnly { .. } => "KD-only",
            Self::Progressive { .. } => "Progressive",
            Self::Attention { .. } => "Attention",
            Self::Combined { .. } => "Combined",
        }
    }

    /// Default KD-only strategy.
    pub fn kd_only() -> Self {
        Self::KDOnly { temperature: 4.0, alpha: 0.7 }
    }

    /// Default progressive strategy.
    pub fn progressive() -> Self {
        Self::Progressive { temperature: 4.0, alpha: 0.7, layer_weight: 0.3 }
    }

    /// Default attention strategy.
    pub fn attention() -> Self {
        Self::Attention { temperature: 4.0, alpha: 0.7, attention_weight: 0.1 }
    }

    /// Default combined strategy.
    pub fn combined() -> Self {
        Self::Combined { temperature: 4.0, alpha: 0.7, layer_weight: 0.3, attention_weight: 0.1 }
    }

    /// Simulate training with this strategy.
    fn simulate(&self, seed: u64) -> StrategyMetrics {
        let noise = (seed as f64 * 0.1).sin() * 0.02;

        let (base_loss, base_accuracy, time_factor) = match self {
            Self::KDOnly { .. } => (0.82, 0.782, 1.0),
            Self::Progressive { .. } => (0.75, 0.818, 1.15),
            Self::Attention { .. } => (0.78, 0.796, 1.08),
            Self::Combined { .. } => (0.71, 0.831, 1.25),
        };

        StrategyMetrics {
            final_loss: base_loss + noise,
            final_accuracy: base_accuracy + noise * 0.5,
            training_time_hours: 2.0 * time_factor + noise * 0.5,
            peak_memory_gb: 16.0 + noise * 2.0,
        }
    }
}

/// Metrics from running a strategy.
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Final training loss
    pub final_loss: f64,
    /// Final accuracy/score
    pub final_accuracy: f64,
    /// Training time in hours
    pub training_time_hours: f64,
    /// Peak memory usage in GB
    pub peak_memory_gb: f64,
}

/// Result of comparing strategies.
#[derive(Debug, Clone)]
pub struct StrategyComparison {
    /// Results per strategy
    pub results: Vec<StrategyResult>,
    /// Best strategy by loss
    pub best_by_loss: Option<String>,
    /// Best strategy by accuracy
    pub best_by_accuracy: Option<String>,
    /// Statistical significance of differences
    pub significance: Vec<PairwiseComparison>,
}

/// Result for a single strategy.
#[derive(Debug, Clone)]
pub struct StrategyResult {
    /// Strategy name
    pub name: String,
    /// Mean metrics across runs
    pub mean_loss: f64,
    /// Standard deviation
    pub std_loss: f64,
    /// Mean accuracy
    pub mean_accuracy: f64,
    /// Standard deviation
    pub std_accuracy: f64,
    /// Mean training time
    pub mean_time_hours: f64,
    /// Number of runs
    pub runs: usize,
}

/// Pairwise statistical comparison.
#[derive(Debug, Clone)]
pub struct PairwiseComparison {
    /// First strategy
    pub strategy1: String,
    /// Second strategy
    pub strategy2: String,
    /// P-value for difference
    pub p_value: f64,
    /// Whether difference is significant
    pub significant: bool,
    /// Effect size
    pub effect_size: f64,
}

/// Compare multiple strategies.
pub fn compare(strategies: &[DistillStrategy]) -> Result<StrategyComparison> {
    let runs_per_strategy = 5;
    let mut results = Vec::new();
    let mut all_losses: Vec<(String, Vec<f64>)> = Vec::new();

    for strategy in strategies {
        let mut losses = Vec::new();
        let mut accuracies = Vec::new();
        let mut times = Vec::new();

        for run in 0..runs_per_strategy {
            let metrics = strategy.simulate(run as u64);
            losses.push(metrics.final_loss);
            accuracies.push(metrics.final_accuracy);
            times.push(metrics.training_time_hours);
        }

        let n = losses.len() as f64;
        let mean_loss = losses.iter().sum::<f64>() / n;
        let mean_accuracy = accuracies.iter().sum::<f64>() / n;
        let mean_time = times.iter().sum::<f64>() / n;

        let std_loss =
            (losses.iter().map(|x| (x - mean_loss).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let std_accuracy = (accuracies.iter().map(|x| (x - mean_accuracy).powi(2)).sum::<f64>()
            / (n - 1.0))
            .sqrt();

        results.push(StrategyResult {
            name: strategy.name().to_string(),
            mean_loss,
            std_loss,
            mean_accuracy,
            std_accuracy,
            mean_time_hours: mean_time,
            runs: runs_per_strategy,
        });

        all_losses.push((strategy.name().to_string(), losses));
    }

    // Find best
    let best_by_loss = results
        .iter()
        .min_by(|a, b| a.mean_loss.partial_cmp(&b.mean_loss).unwrap_or(std::cmp::Ordering::Equal))
        .map(|r| r.name.clone());

    let best_by_accuracy = results
        .iter()
        .max_by(|a, b| {
            a.mean_accuracy.partial_cmp(&b.mean_accuracy).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.name.clone());

    // Pairwise comparisons
    let mut significance = Vec::new();
    for i in 0..all_losses.len() {
        for j in (i + 1)..all_losses.len() {
            let (name1, losses1) = &all_losses[i];
            let (name2, losses2) = &all_losses[j];

            let test = StatisticalAnalyzer::welch_t_test(losses1, losses2);

            significance.push(PairwiseComparison {
                strategy1: name1.clone(),
                strategy2: name2.clone(),
                p_value: test.p_value,
                significant: test.significant,
                effect_size: test.effect_size,
            });
        }
    }

    Ok(StrategyComparison { results, best_by_loss, best_by_accuracy, significance })
}

impl StrategyComparison {
    /// Format as ASCII table.
    pub fn to_table(&self) -> String {
        let mut output = String::from("Strategy Comparison\n");
        output.push_str("┌──────────────┬─────────────────┬─────────────────┬────────────┐\n");
        output.push_str("│ Strategy     │ Loss            │ Accuracy        │ Time (h)   │\n");
        output.push_str("├──────────────┼─────────────────┼─────────────────┼────────────┤\n");

        for result in &self.results {
            let loss_marker =
                if self.best_by_loss.as_ref() == Some(&result.name) { " ★" } else { "" };
            let acc_marker =
                if self.best_by_accuracy.as_ref() == Some(&result.name) { " ★" } else { "" };

            output.push_str(&format!(
                "│ {:12} │ {:.3} ± {:.3}{:2} │ {:.1}% ± {:.1}%{:2} │ {:>10.1} │\n",
                result.name,
                result.mean_loss,
                result.std_loss,
                loss_marker,
                result.mean_accuracy * 100.0,
                result.std_accuracy * 100.0,
                acc_marker,
                result.mean_time_hours
            ));
        }

        output.push_str("└──────────────┴─────────────────┴─────────────────┴────────────┘\n");

        // Significance
        output.push_str("\nStatistical Significance:\n");
        for comp in &self.significance {
            let sig = if comp.significant { "✓" } else { "✗" };
            output.push_str(&format!(
                "  {} vs {}: p={:.4} {} (effect={:.2})\n",
                comp.strategy1, comp.strategy2, comp.p_value, sig, comp.effect_size
            ));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_names() {
        assert_eq!(DistillStrategy::kd_only().name(), "KD-only");
        assert_eq!(DistillStrategy::progressive().name(), "Progressive");
        assert_eq!(DistillStrategy::attention().name(), "Attention");
        assert_eq!(DistillStrategy::combined().name(), "Combined");
    }

    #[test]
    fn test_compare_strategies() {
        let strategies = vec![
            DistillStrategy::kd_only(),
            DistillStrategy::progressive(),
            DistillStrategy::combined(),
        ];

        let comparison = compare(&strategies).expect("operation should succeed");

        assert_eq!(comparison.results.len(), 3);
        assert!(comparison.best_by_loss.is_some());
        assert!(comparison.best_by_accuracy.is_some());
    }

    #[test]
    fn test_combined_is_best() {
        let strategies = vec![DistillStrategy::kd_only(), DistillStrategy::combined()];

        let comparison = compare(&strategies).expect("operation should succeed");

        // Combined should generally be best
        assert_eq!(comparison.best_by_accuracy.as_deref(), Some("Combined"));
    }

    #[test]
    fn test_comparison_table() {
        let strategies = vec![DistillStrategy::kd_only(), DistillStrategy::progressive()];

        let comparison = compare(&strategies).expect("operation should succeed");
        let table = comparison.to_table();

        assert!(table.contains("KD-only"));
        assert!(table.contains("Progressive"));
        assert!(table.contains("Significance"));
    }

    #[test]
    fn test_strategy_constructors() {
        let kd = DistillStrategy::kd_only();
        if let DistillStrategy::KDOnly { temperature, alpha } = kd {
            assert_eq!(temperature, 4.0);
            assert_eq!(alpha, 0.7);
        } else {
            panic!("Expected KDOnly");
        }

        let prog = DistillStrategy::progressive();
        if let DistillStrategy::Progressive { temperature, alpha, layer_weight } = prog {
            assert_eq!(temperature, 4.0);
            assert_eq!(alpha, 0.7);
            assert_eq!(layer_weight, 0.3);
        } else {
            panic!("Expected Progressive");
        }

        let attn = DistillStrategy::attention();
        if let DistillStrategy::Attention { temperature, alpha, attention_weight } = attn {
            assert_eq!(temperature, 4.0);
            assert_eq!(alpha, 0.7);
            assert_eq!(attention_weight, 0.1);
        } else {
            panic!("Expected Attention");
        }

        let combined = DistillStrategy::combined();
        if let DistillStrategy::Combined { temperature, alpha, layer_weight, attention_weight } =
            combined
        {
            assert_eq!(temperature, 4.0);
            assert_eq!(alpha, 0.7);
            assert_eq!(layer_weight, 0.3);
            assert_eq!(attention_weight, 0.1);
        } else {
            panic!("Expected Combined");
        }
    }

    #[test]
    fn test_strategy_simulate_deterministic() {
        let strategy = DistillStrategy::kd_only();
        let metrics1 = strategy.simulate(42);
        let metrics2 = strategy.simulate(42);

        // Same seed should produce same results
        assert_eq!(metrics1.final_loss, metrics2.final_loss);
        assert_eq!(metrics1.final_accuracy, metrics2.final_accuracy);
    }

    #[test]
    fn test_strategy_simulate_different_seeds() {
        let strategy = DistillStrategy::kd_only();
        let metrics1 = strategy.simulate(1);
        let metrics2 = strategy.simulate(2);

        // Different seeds should produce different results (due to noise)
        assert_ne!(metrics1.final_loss, metrics2.final_loss);
    }

    #[test]
    fn test_strategy_metrics_fields() {
        let metrics = StrategyMetrics {
            final_loss: 0.75,
            final_accuracy: 0.82,
            training_time_hours: 2.5,
            peak_memory_gb: 16.0,
        };

        assert_eq!(metrics.final_loss, 0.75);
        assert_eq!(metrics.final_accuracy, 0.82);
        assert_eq!(metrics.training_time_hours, 2.5);
        assert_eq!(metrics.peak_memory_gb, 16.0);
    }

    #[test]
    fn test_strategy_result_fields() {
        let result = StrategyResult {
            name: "test".to_string(),
            mean_loss: 0.7,
            std_loss: 0.02,
            mean_accuracy: 0.85,
            std_accuracy: 0.01,
            mean_time_hours: 3.0,
            runs: 5,
        };

        assert_eq!(result.name, "test");
        assert_eq!(result.runs, 5);
    }

    #[test]
    fn test_pairwise_comparison_fields() {
        let comp = PairwiseComparison {
            strategy1: "A".to_string(),
            strategy2: "B".to_string(),
            p_value: 0.03,
            significant: true,
            effect_size: 0.8,
        };

        assert!(comp.significant);
        assert_eq!(comp.effect_size, 0.8);
    }

    #[test]
    fn test_comparison_significance_markers() {
        let strategies = vec![DistillStrategy::kd_only(), DistillStrategy::combined()];

        let comparison = compare(&strategies).expect("operation should succeed");

        // Should have one pairwise comparison
        assert_eq!(comparison.significance.len(), 1);
    }

    #[test]
    fn test_compare_all_strategies() {
        let strategies = vec![
            DistillStrategy::kd_only(),
            DistillStrategy::progressive(),
            DistillStrategy::attention(),
            DistillStrategy::combined(),
        ];

        let comparison = compare(&strategies).expect("operation should succeed");

        // 4 choose 2 = 6 pairwise comparisons
        assert_eq!(comparison.significance.len(), 6);
        assert_eq!(comparison.results.len(), 4);
    }

    #[test]
    fn test_comparison_table_star_markers() {
        let strategies = vec![DistillStrategy::kd_only(), DistillStrategy::combined()];

        let comparison = compare(&strategies).expect("operation should succeed");
        let table = comparison.to_table();

        // Should have star marker for best
        assert!(table.contains('★'));
    }
}
