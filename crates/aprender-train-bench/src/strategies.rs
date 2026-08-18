//! Distillation strategy comparison.

use entrenar_common::{EntrenarError, Result};

/// A distillation strategy to benchmark.
#[derive(Debug, Clone)]
pub enum DistillStrategy {
    /// Knowledge distillation only (soft targets)
    KDOnly { temperature: f32, alpha: f32 },
    /// Progressive distillation (hidden state matching)
    Progressive {
        temperature: f32,
        alpha: f32,
        layer_weight: f32,
    },
    /// Attention transfer
    Attention {
        temperature: f32,
        alpha: f32,
        attention_weight: f32,
    },
    /// Combined approach
    Combined {
        temperature: f32,
        alpha: f32,
        layer_weight: f32,
        attention_weight: f32,
    },
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
        Self::KDOnly {
            temperature: 4.0,
            alpha: 0.7,
        }
    }

    /// Default progressive strategy.
    pub fn progressive() -> Self {
        Self::Progressive {
            temperature: 4.0,
            alpha: 0.7,
            layer_weight: 0.3,
        }
    }

    /// Default attention strategy.
    pub fn attention() -> Self {
        Self::Attention {
            temperature: 4.0,
            alpha: 0.7,
            attention_weight: 0.1,
        }
    }

    /// Default combined strategy.
    pub fn combined() -> Self {
        Self::Combined {
            temperature: 4.0,
            alpha: 0.7,
            layer_weight: 0.3,
            attention_weight: 0.1,
        }
    }

    /// Simulate training with this strategy.
    //
    // #2519: retained ONLY for the unit tests that pin its per-variant literal
    // table, so it stays on the record as a lookup rather than a run. Scoped to
    // test builds so no production path can present it as a result again. Note
    // what it ignores: every field of every variant. `KDOnly { alpha: 0.0 }`
    // (no distillation at all) and `KDOnly { alpha: 0.7 }` get byte-identical
    // metrics, which is why the `ablation` subcommand printed `Δ Loss +0.0000`
    // for "+ KD (T=4)" over the CE-only baseline.
    #[cfg(test)]
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
///
/// # Errors
///
/// If `strategies` is empty, and otherwise always: nothing here trains, so
/// there is no honest comparison to return -- see the #2519 note in the body.
pub fn compare(strategies: &[DistillStrategy]) -> Result<StrategyComparison> {
    // Kept: an empty strategy list is a genuine caller mistake with its own
    // distinct diagnosis, and it is still worth naming separately from the
    // refusal below.
    if strategies.is_empty() {
        return Err(EntrenarError::ConfigValue {
            field: "strategies".into(),
            message: "No strategies to compare".into(),
            suggestion: "Pass at least one of: kd, progressive, attention, combined".into(),
        });
    }

    // #2519: this used to read
    //
    //     for run in 0..runs_per_strategy {
    //         let metrics = strategy.simulate(run as u64);
    //
    // -- five "runs" of a per-variant LITERAL TABLE (`Combined -> 0.71/0.831`),
    // plus a sinusoid of the run index standing in for run-to-run variance. It
    // then fed those numbers to a real Welch t-test and printed p-values.
    //
    // Measured before this change, with no model and no data:
    //
    //     Combined  0.714 ± 0.003 ★   83.3% ± 0.2% ★
    //     KD-only vs Combined: p=0.0000 ✓ (effect=35.69)
    //     ✓ Recommendation: Combined for best accuracy
    //
    // The p-value is the sharpest part of the defect: a correct statistical
    // test applied to invented samples reports overwhelming significance,
    // because the "variance" is a deterministic curve. The statistics were
    // never wrong -- their input was fabricated, and the honest-looking
    // machinery around it is what made the output persuasive.
    //
    // Refusing is strictly better than fabricating. Whether this binary should
    // exist at all is tracked in #2519; this change does not prejudge it.
    Err(EntrenarError::ConfigValue {
        field: "strategies".into(),
        message: format!(
            "cannot compare {} distillation strategies: this crate never trains any \
             of them. It previously returned a per-variant literal table, ran a real \
             t-test over it and recommended a winner, which is why it is now an \
             error rather than a plausible-looking comparison",
            strategies.len()
        ),
        suggestion: "Train each strategy for real (`apr distill`) and compare the \
                     metrics those runs report. Tracked in #2519."
            .into(),
    })
}

impl StrategyComparison {
    /// Format as ASCII table.
    pub fn to_table(&self) -> String {
        let mut output = String::from("Strategy Comparison\n");
        output.push_str("┌──────────────┬─────────────────┬─────────────────┬────────────┐\n");
        output.push_str("│ Strategy     │ Loss            │ Accuracy        │ Time (h)   │\n");
        output.push_str("├──────────────┼─────────────────┼─────────────────┼────────────┤\n");

        for result in &self.results {
            let loss_marker = if self.best_by_loss.as_ref() == Some(&result.name) {
                " ★"
            } else {
                ""
            };
            let acc_marker = if self.best_by_accuracy.as_ref() == Some(&result.name) {
                " ★"
            } else {
                ""
            };

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

    /// Build a `StrategyComparison` from values supplied by the caller, so the
    /// formatter tests below exercise `to_table` without a comparison having to
    /// invent the numbers it formats.
    fn comparison_from(entries: &[(&str, f64, f64)]) -> StrategyComparison {
        let results: Vec<StrategyResult> = entries
            .iter()
            .map(|&(name, mean_loss, mean_accuracy)| StrategyResult {
                name: name.to_string(),
                mean_loss,
                std_loss: 0.0,
                mean_accuracy,
                std_accuracy: 0.0,
                mean_time_hours: 2.0,
                runs: 1,
            })
            .collect();

        let best_by_loss = results
            .iter()
            .min_by(|a, b| {
                a.mean_loss
                    .partial_cmp(&b.mean_loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.name.clone());
        let best_by_accuracy = results
            .iter()
            .max_by(|a, b| {
                a.mean_accuracy
                    .partial_cmp(&b.mean_accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.name.clone());

        let significance = results
            .windows(2)
            .map(|pair| PairwiseComparison {
                strategy1: pair[0].name.clone(),
                strategy2: pair[1].name.clone(),
                p_value: 0.5,
                significant: false,
                effect_size: 0.0,
            })
            .collect();

        StrategyComparison {
            results,
            best_by_loss,
            best_by_accuracy,
            significance,
        }
    }

    #[test]
    fn test_strategy_names() {
        assert_eq!(DistillStrategy::kd_only().name(), "KD-only");
        assert_eq!(DistillStrategy::progressive().name(), "Progressive");
        assert_eq!(DistillStrategy::attention().name(), "Attention");
        assert_eq!(DistillStrategy::combined().name(), "Combined");
    }

    // #2519: `test_compare_strategies` and `test_combined_is_best` used to
    // assert `compare()` was Ok and that "Combined" won -- they asserted the
    // literal table, so they would have gone RED on any honest fix. They now
    // pin the refusal.
    #[test]
    fn test_compare_refuses_to_compare() {
        let strategies = vec![
            DistillStrategy::kd_only(),
            DistillStrategy::progressive(),
            DistillStrategy::combined(),
        ];

        let err = compare(&strategies).expect_err("comparing untrained strategies must fail");
        assert!(format!("{err}").contains("never trains"));
    }

    #[test]
    fn test_compare_does_not_name_a_winner() {
        let strategies = vec![DistillStrategy::kd_only(), DistillStrategy::combined()];

        // The old output ended in "Recommendation: Combined for best accuracy",
        // derived from two hardcoded pairs of numbers.
        assert!(compare(&strategies).is_err());
    }

    #[test]
    fn test_comparison_table() {
        let comparison = comparison_from(&[("KD-only", 0.82, 0.782), ("Progressive", 0.75, 0.818)]);
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
        if let DistillStrategy::Progressive {
            temperature,
            alpha,
            layer_weight,
        } = prog
        {
            assert_eq!(temperature, 4.0);
            assert_eq!(alpha, 0.7);
            assert_eq!(layer_weight, 0.3);
        } else {
            panic!("Expected Progressive");
        }

        let attn = DistillStrategy::attention();
        if let DistillStrategy::Attention {
            temperature,
            alpha,
            attention_weight,
        } = attn
        {
            assert_eq!(temperature, 4.0);
            assert_eq!(alpha, 0.7);
            assert_eq!(attention_weight, 0.1);
        } else {
            panic!("Expected Attention");
        }

        let combined = DistillStrategy::combined();
        if let DistillStrategy::Combined {
            temperature,
            alpha,
            layer_weight,
            attention_weight,
        } = combined
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
        let comparison = comparison_from(&[("KD-only", 0.82, 0.782), ("Combined", 0.71, 0.831)]);

        // Should have one pairwise comparison
        assert_eq!(comparison.significance.len(), 1);
    }

    #[test]
    fn test_compare_all_strategies_still_refuses() {
        let strategies = vec![
            DistillStrategy::kd_only(),
            DistillStrategy::progressive(),
            DistillStrategy::attention(),
            DistillStrategy::combined(),
        ];

        // Asking for more strategies does not make any of them run.
        let err = compare(&strategies).expect_err("must refuse");
        assert!(format!("{err}").contains('4'));
    }

    #[test]
    fn test_compare_empty_fails_for_its_own_reason() {
        // Non-vacuity: the refusal above is not a blanket "always Err" -- an
        // empty list is still diagnosed as an empty list.
        let err = compare(&[]).expect_err("an empty strategy list must fail");
        let text = format!("{err}");
        assert!(text.contains("No strategies to compare"), "got: {text}");
        assert!(!text.contains("never trains"), "got: {text}");
    }

    #[test]
    fn test_simulate_ignores_every_strategy_field() {
        // #2519's discriminating symptom: `simulate` matches only on the enum
        // VARIANT, so a run with no distillation at all (alpha = 0.0, T = 1.0)
        // is indistinguishable from one with alpha = 0.7, T = 4.0. That is why
        // `ablation` printed "Δ Loss +0.0000" for adding KD.
        let no_kd = DistillStrategy::KDOnly {
            temperature: 1.0,
            alpha: 0.0,
        };
        let with_kd = DistillStrategy::KDOnly {
            temperature: 4.0,
            alpha: 0.7,
        };

        assert_eq!(no_kd.simulate(0).final_loss, with_kd.simulate(0).final_loss);
    }

    #[test]
    fn test_comparison_table_star_markers() {
        let comparison = comparison_from(&[("KD-only", 0.82, 0.782), ("Combined", 0.71, 0.831)]);
        let table = comparison.to_table();

        // Should have star marker for best
        assert!(table.contains('★'));
    }
}
