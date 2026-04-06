//! Benchmark statistics summary

/// Benchmark statistics summary
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStatistics {
    /// Number of entries
    pub count: usize,
    /// Minimum quality score
    pub quality_min: f64,
    /// Maximum quality score
    pub quality_max: f64,
    /// Average quality score
    pub quality_avg: f64,
    /// Minimum cost
    pub cost_min: f64,
    /// Maximum cost
    pub cost_max: f64,
    /// Average cost
    pub cost_avg: f64,
    /// Minimum energy (joules)
    pub energy_min: f64,
    /// Maximum energy (joules)
    pub energy_max: f64,
    /// Average energy (joules)
    pub energy_avg: f64,
    /// Number of Pareto-optimal entries
    pub pareto_count: usize,
}
