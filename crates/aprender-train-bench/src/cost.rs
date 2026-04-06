//! Cost-Performance Analysis (ENT-029)
//!
//! Provides Pareto frontier analysis for balancing training cost vs model performance.

use serde::{Deserialize, Serialize};

/// A single configuration with cost and performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPerformancePoint {
    /// Configuration name or description
    pub name: String,
    /// Training cost in GPU-hours
    pub gpu_hours: f64,
    /// Estimated cloud cost in USD
    pub cost_usd: f64,
    /// Model accuracy (0.0 - 1.0)
    pub accuracy: f64,
    /// Model loss
    pub loss: f64,
    /// Memory usage in GB
    pub memory_gb: f64,
    /// Whether this point is on the Pareto frontier
    pub is_pareto_optimal: bool,
    /// Configuration parameters
    pub config: ConfigParams,
}

/// Configuration parameters for a training run
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfigParams {
    /// LoRA rank
    pub lora_rank: Option<u32>,
    /// Quantization bits (4, 8, 16, 32)
    pub quant_bits: Option<u8>,
    /// Temperature for distillation
    pub temperature: Option<f32>,
    /// Alpha for distillation
    pub alpha: Option<f32>,
    /// Batch size
    pub batch_size: Option<usize>,
    /// Learning rate
    pub learning_rate: Option<f64>,
}

/// Cost model for different GPU types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// GPU type name
    pub gpu_type: String,
    /// Cost per hour in USD
    pub cost_per_hour: f64,
    /// Memory in GB
    pub memory_gb: f64,
    /// Relative performance factor (vs baseline)
    pub performance_factor: f64,
}

impl CostModel {
    /// Create an A100 80GB cost model
    pub fn a100_80gb() -> Self {
        Self {
            gpu_type: "A100-80GB".to_string(),
            cost_per_hour: 2.21,
            memory_gb: 80.0,
            performance_factor: 1.0,
        }
    }

    /// Create an A100 40GB cost model
    pub fn a100_40gb() -> Self {
        Self {
            gpu_type: "A100-40GB".to_string(),
            cost_per_hour: 1.10,
            memory_gb: 40.0,
            performance_factor: 0.9,
        }
    }

    /// Create a V100 cost model
    pub fn v100() -> Self {
        Self {
            gpu_type: "V100".to_string(),
            cost_per_hour: 0.90,
            memory_gb: 16.0,
            performance_factor: 0.5,
        }
    }

    /// Create a T4 cost model
    pub fn t4() -> Self {
        Self {
            gpu_type: "T4".to_string(),
            cost_per_hour: 0.35,
            memory_gb: 16.0,
            performance_factor: 0.25,
        }
    }

    /// Create a custom cost model
    pub fn custom(gpu_type: &str, cost_per_hour: f64, memory_gb: f64) -> Self {
        Self { gpu_type: gpu_type.to_string(), cost_per_hour, memory_gb, performance_factor: 1.0 }
    }
}

/// Constraints for recommendations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Constraints {
    /// Maximum GPU-hours
    pub max_gpu_hours: Option<f64>,
    /// Maximum cost in USD
    pub max_cost_usd: Option<f64>,
    /// Minimum accuracy required
    pub min_accuracy: Option<f64>,
    /// Maximum memory in GB
    pub max_memory_gb: Option<f64>,
    /// Maximum loss
    pub max_loss: Option<f64>,
}

impl Constraints {
    /// Create new empty constraints
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum GPU-hours
    pub fn with_max_gpu_hours(mut self, hours: f64) -> Self {
        self.max_gpu_hours = Some(hours);
        self
    }

    /// Set maximum cost
    pub fn with_max_cost(mut self, cost: f64) -> Self {
        self.max_cost_usd = Some(cost);
        self
    }

    /// Set minimum accuracy
    pub fn with_min_accuracy(mut self, accuracy: f64) -> Self {
        self.min_accuracy = Some(accuracy);
        self
    }

    /// Set maximum memory
    pub fn with_max_memory(mut self, memory_gb: f64) -> Self {
        self.max_memory_gb = Some(memory_gb);
        self
    }

    /// Check if a point satisfies all constraints
    pub fn is_satisfied(&self, point: &CostPerformancePoint) -> bool {
        if let Some(max_hours) = self.max_gpu_hours {
            if point.gpu_hours > max_hours {
                return false;
            }
        }
        if let Some(max_cost) = self.max_cost_usd {
            if point.cost_usd > max_cost {
                return false;
            }
        }
        if let Some(min_acc) = self.min_accuracy {
            if point.accuracy < min_acc {
                return false;
            }
        }
        if let Some(max_mem) = self.max_memory_gb {
            if point.memory_gb > max_mem {
                return false;
            }
        }
        if let Some(max_loss) = self.max_loss {
            if point.loss > max_loss {
                return false;
            }
        }
        true
    }
}

/// Cost-Performance Analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostPerformanceAnalysis {
    /// All data points
    pub points: Vec<CostPerformancePoint>,
    /// Pareto-optimal points
    pub pareto_frontier: Vec<CostPerformancePoint>,
    /// Best by accuracy
    pub best_accuracy: Option<CostPerformancePoint>,
    /// Best by cost efficiency (accuracy per dollar)
    pub best_efficiency: Option<CostPerformancePoint>,
    /// Best by cost (lowest)
    pub lowest_cost: Option<CostPerformancePoint>,
}

impl CostPerformanceAnalysis {
    /// Compute Pareto frontier from data points
    pub fn from_points(mut points: Vec<CostPerformancePoint>) -> Self {
        // Compute Pareto frontier (minimize cost, maximize accuracy)
        let pareto = compute_pareto_frontier(&points);

        // Mark points that are Pareto optimal
        for point in &mut points {
            point.is_pareto_optimal = pareto.iter().any(|p| {
                (p.cost_usd - point.cost_usd).abs() < 1e-6
                    && (p.accuracy - point.accuracy).abs() < 1e-6
            });
        }

        let pareto_frontier = pareto;

        let best_accuracy = points
            .iter()
            .max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap_or(std::cmp::Ordering::Equal))
            .cloned();

        let best_efficiency = points
            .iter()
            .filter(|p| p.cost_usd > 0.0)
            .max_by(|a, b| {
                let eff_a = a.accuracy / a.cost_usd;
                let eff_b = b.accuracy / b.cost_usd;
                eff_a.partial_cmp(&eff_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        let lowest_cost = points
            .iter()
            .min_by(|a, b| a.cost_usd.partial_cmp(&b.cost_usd).unwrap_or(std::cmp::Ordering::Equal))
            .cloned();

        Self { points, pareto_frontier, best_accuracy, best_efficiency, lowest_cost }
    }

    /// Get recommendations based on constraints
    pub fn recommend(&self, constraints: &Constraints) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Filter points that satisfy constraints
        let valid_points: Vec<_> =
            self.points.iter().filter(|p| constraints.is_satisfied(p)).collect();

        if valid_points.is_empty() {
            return recommendations;
        }

        // Best accuracy within constraints
        if let Some(best_acc) = valid_points
            .iter()
            .max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap_or(std::cmp::Ordering::Equal))
        {
            recommendations.push(Recommendation {
                reason: "Best accuracy within constraints".to_string(),
                point: (*best_acc).clone(),
            });
        }

        // Best efficiency within constraints
        if let Some(best_eff) = valid_points.iter().filter(|p| p.cost_usd > 0.0).max_by(|a, b| {
            let eff_a = a.accuracy / a.cost_usd;
            let eff_b = b.accuracy / b.cost_usd;
            eff_a.partial_cmp(&eff_b).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if recommendations.iter().all(|r| r.point.name != best_eff.name) {
                recommendations.push(Recommendation {
                    reason: "Best accuracy per dollar within constraints".to_string(),
                    point: (*best_eff).clone(),
                });
            }
        }

        // Pareto-optimal within constraints
        for point in &self.pareto_frontier {
            if constraints.is_satisfied(point)
                && recommendations.iter().all(|r| r.point.name != point.name)
            {
                recommendations.push(Recommendation {
                    reason: "Pareto-optimal configuration".to_string(),
                    point: point.clone(),
                });
            }
        }

        recommendations
    }

    /// Generate ASCII table for display
    pub fn to_table(&self) -> String {
        let mut table = String::new();
        table.push_str("Cost-Performance Analysis\n");
        table.push_str(
            "┌────────────────────────┬───────────┬───────────┬──────────┬─────────┬─────────┐\n",
        );
        table.push_str(
            "│ Configuration          │ GPU Hours │ Cost (USD)│ Accuracy │   Loss  │ Pareto? │\n",
        );
        table.push_str(
            "├────────────────────────┼───────────┼───────────┼──────────┼─────────┼─────────┤\n",
        );

        for point in &self.points {
            let pareto_mark = if point.is_pareto_optimal { "★" } else { " " };
            table.push_str(&format!(
                "│ {:22} │ {:>9.2} │ {:>9.2} │ {:>7.1}% │ {:>7.4} │    {}    │\n",
                truncate(&point.name, 22),
                point.gpu_hours,
                point.cost_usd,
                point.accuracy * 100.0,
                point.loss,
                pareto_mark
            ));
        }

        table.push_str(
            "└────────────────────────┴───────────┴───────────┴──────────┴─────────┴─────────┘\n",
        );
        table.push_str(
            "\n★ = Pareto-optimal (no configuration is both cheaper AND more accurate)\n",
        );

        table
    }
}

/// A recommendation with reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Reason for the recommendation
    pub reason: String,
    /// The recommended configuration
    pub point: CostPerformancePoint,
}

/// Compute Pareto frontier for minimizing cost and maximizing accuracy
fn compute_pareto_frontier(points: &[CostPerformancePoint]) -> Vec<CostPerformancePoint> {
    let mut frontier = Vec::new();

    for point in points {
        // Check if this point is dominated by any other
        let is_dominated = points.iter().any(|other| {
            // Other dominates point if:
            // - Other has lower or equal cost AND higher or equal accuracy
            // - AND at least one is strictly better
            other.cost_usd <= point.cost_usd
                && other.accuracy >= point.accuracy
                && (other.cost_usd < point.cost_usd || other.accuracy > point.accuracy)
        });

        if !is_dominated {
            frontier.push(point.clone());
        }
    }

    // Sort by cost
    frontier
        .sort_by(|a, b| a.cost_usd.partial_cmp(&b.cost_usd).unwrap_or(std::cmp::Ordering::Equal));
    frontier
}

/// Truncate string to max length
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        format!("{s:max_len$}")
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Generate sample data points for testing/demo
pub fn generate_sample_points(cost_model: &CostModel) -> Vec<CostPerformancePoint> {
    // Sample configurations representing different trade-offs
    vec![
        // Full fine-tuning (expensive, high accuracy)
        CostPerformancePoint {
            name: "Full Fine-Tuning (7B)".to_string(),
            gpu_hours: 120.0,
            cost_usd: 120.0 * cost_model.cost_per_hour,
            accuracy: 0.92,
            loss: 0.25,
            memory_gb: 56.0,
            is_pareto_optimal: false,
            config: ConfigParams {
                lora_rank: None,
                quant_bits: Some(16),
                batch_size: Some(8),
                learning_rate: Some(5e-5),
                ..Default::default()
            },
        },
        // LoRA (moderate cost, good accuracy)
        CostPerformancePoint {
            name: "LoRA r=64".to_string(),
            gpu_hours: 24.0,
            cost_usd: 24.0 * cost_model.cost_per_hour,
            accuracy: 0.89,
            loss: 0.30,
            memory_gb: 28.0,
            is_pareto_optimal: false,
            config: ConfigParams {
                lora_rank: Some(64),
                quant_bits: Some(16),
                batch_size: Some(16),
                learning_rate: Some(2e-4),
                ..Default::default()
            },
        },
        // LoRA r=32 (cheaper, slightly lower accuracy)
        CostPerformancePoint {
            name: "LoRA r=32".to_string(),
            gpu_hours: 18.0,
            cost_usd: 18.0 * cost_model.cost_per_hour,
            accuracy: 0.87,
            loss: 0.33,
            memory_gb: 24.0,
            is_pareto_optimal: false,
            config: ConfigParams {
                lora_rank: Some(32),
                quant_bits: Some(16),
                batch_size: Some(16),
                learning_rate: Some(2e-4),
                ..Default::default()
            },
        },
        // QLoRA 4-bit (low cost, good accuracy)
        CostPerformancePoint {
            name: "QLoRA 4-bit r=64".to_string(),
            gpu_hours: 20.0,
            cost_usd: 20.0 * cost_model.cost_per_hour,
            accuracy: 0.86,
            loss: 0.35,
            memory_gb: 12.0,
            is_pareto_optimal: false,
            config: ConfigParams {
                lora_rank: Some(64),
                quant_bits: Some(4),
                batch_size: Some(32),
                learning_rate: Some(3e-4),
                ..Default::default()
            },
        },
        // Distillation (moderate cost, moderate accuracy)
        CostPerformancePoint {
            name: "Distillation T=4".to_string(),
            gpu_hours: 36.0,
            cost_usd: 36.0 * cost_model.cost_per_hour,
            accuracy: 0.84,
            loss: 0.38,
            memory_gb: 32.0,
            is_pareto_optimal: false,
            config: ConfigParams {
                temperature: Some(4.0),
                alpha: Some(0.7),
                batch_size: Some(16),
                learning_rate: Some(1e-4),
                ..Default::default()
            },
        },
        // LoRA + Distillation (balanced)
        CostPerformancePoint {
            name: "LoRA + Distillation".to_string(),
            gpu_hours: 32.0,
            cost_usd: 32.0 * cost_model.cost_per_hour,
            accuracy: 0.88,
            loss: 0.31,
            memory_gb: 26.0,
            is_pareto_optimal: false,
            config: ConfigParams {
                lora_rank: Some(32),
                temperature: Some(4.0),
                alpha: Some(0.5),
                batch_size: Some(16),
                learning_rate: Some(2e-4),
                ..Default::default()
            },
        },
        // QLoRA 8-bit (moderate everything)
        CostPerformancePoint {
            name: "QLoRA 8-bit r=32".to_string(),
            gpu_hours: 16.0,
            cost_usd: 16.0 * cost_model.cost_per_hour,
            accuracy: 0.85,
            loss: 0.36,
            memory_gb: 16.0,
            is_pareto_optimal: false,
            config: ConfigParams {
                lora_rank: Some(32),
                quant_bits: Some(8),
                batch_size: Some(32),
                learning_rate: Some(2e-4),
                ..Default::default()
            },
        },
        // Minimal LoRA (very cheap, lower accuracy)
        CostPerformancePoint {
            name: "LoRA r=8".to_string(),
            gpu_hours: 8.0,
            cost_usd: 8.0 * cost_model.cost_per_hour,
            accuracy: 0.81,
            loss: 0.42,
            memory_gb: 18.0,
            is_pareto_optimal: false,
            config: ConfigParams {
                lora_rank: Some(8),
                quant_bits: Some(16),
                batch_size: Some(32),
                learning_rate: Some(5e-4),
                ..Default::default()
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_frontier() {
        let points = vec![
            CostPerformancePoint {
                name: "A".to_string(),
                gpu_hours: 10.0,
                cost_usd: 10.0,
                accuracy: 0.8,
                loss: 0.3,
                memory_gb: 16.0,
                is_pareto_optimal: false,
                config: Default::default(),
            },
            CostPerformancePoint {
                name: "B".to_string(),
                gpu_hours: 20.0,
                cost_usd: 20.0,
                accuracy: 0.9,
                loss: 0.2,
                memory_gb: 24.0,
                is_pareto_optimal: false,
                config: Default::default(),
            },
            CostPerformancePoint {
                name: "C".to_string(), // Dominated by B
                gpu_hours: 25.0,
                cost_usd: 25.0,
                accuracy: 0.85,
                loss: 0.25,
                memory_gb: 24.0,
                is_pareto_optimal: false,
                config: Default::default(),
            },
        ];

        let frontier = compute_pareto_frontier(&points);
        assert_eq!(frontier.len(), 2); // A and B are Pareto optimal
        assert!(frontier.iter().any(|p| p.name == "A"));
        assert!(frontier.iter().any(|p| p.name == "B"));
        assert!(!frontier.iter().any(|p| p.name == "C"));
    }

    #[test]
    fn test_constraints() {
        let constraints = Constraints::new().with_max_cost(50.0).with_min_accuracy(0.85);

        let point_good = CostPerformancePoint {
            name: "Good".to_string(),
            gpu_hours: 20.0,
            cost_usd: 40.0,
            accuracy: 0.90,
            loss: 0.25,
            memory_gb: 16.0,
            is_pareto_optimal: false,
            config: Default::default(),
        };

        let point_expensive = CostPerformancePoint {
            name: "Expensive".to_string(),
            gpu_hours: 30.0,
            cost_usd: 60.0,
            accuracy: 0.95,
            loss: 0.20,
            memory_gb: 16.0,
            is_pareto_optimal: false,
            config: Default::default(),
        };

        let point_low_acc = CostPerformancePoint {
            name: "LowAcc".to_string(),
            gpu_hours: 10.0,
            cost_usd: 20.0,
            accuracy: 0.80,
            loss: 0.35,
            memory_gb: 16.0,
            is_pareto_optimal: false,
            config: Default::default(),
        };

        assert!(constraints.is_satisfied(&point_good));
        assert!(!constraints.is_satisfied(&point_expensive)); // Too expensive
        assert!(!constraints.is_satisfied(&point_low_acc)); // Too low accuracy
    }

    #[test]
    fn test_analysis_recommendations() {
        let cost_model = CostModel::a100_80gb();
        let points = generate_sample_points(&cost_model);
        let analysis = CostPerformanceAnalysis::from_points(points);

        assert!(!analysis.pareto_frontier.is_empty());
        assert!(analysis.best_accuracy.is_some());
        assert!(analysis.best_efficiency.is_some());

        let constraints = Constraints::new().with_max_cost(50.0);
        let recommendations = analysis.recommend(&constraints);
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_cost_models() {
        let a100 = CostModel::a100_80gb();
        assert_eq!(a100.gpu_type, "A100-80GB");
        assert!(a100.cost_per_hour > 0.0);

        let v100 = CostModel::v100();
        assert!(v100.cost_per_hour < a100.cost_per_hour);
    }
}
