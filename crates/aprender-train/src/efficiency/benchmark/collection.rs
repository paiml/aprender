//! Cost-performance benchmark collection

use super::entry::BenchmarkEntry;
use super::statistics::BenchmarkStatistics;
use crate::efficiency::{ComputeDevice, ModelParadigm};
use serde::{Deserialize, Serialize};

/// Cost-performance benchmark collection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostPerformanceBenchmark {
    /// Benchmark entries
    pub entries: Vec<BenchmarkEntry>,
}

impl CostPerformanceBenchmark {
    /// Create a new empty benchmark
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a benchmark entry
    pub fn add(&mut self, entry: BenchmarkEntry) {
        self.entries.push(entry);
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find the Pareto frontier (non-dominated entries)
    ///
    /// Returns entries where no other entry is better in both quality AND cost.
    pub fn pareto_frontier(&self) -> Vec<&BenchmarkEntry> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let mut frontier = Vec::new();

        for entry in &self.entries {
            let is_dominated = self
                .entries
                .iter()
                .any(|other| !std::ptr::eq(entry, other) && other.dominates(entry));

            if !is_dominated {
                frontier.push(entry);
            }
        }

        // Sort by quality (descending)
        frontier.sort_by(|a, b| {
            b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        frontier
    }

    /// Find best entry within a budget
    pub fn best_for_budget(&self, max_usd: f64) -> Option<&BenchmarkEntry> {
        self.entries.iter().filter(|e| e.cost.total_cost_usd <= max_usd).max_by(|a, b| {
            a.quality_score.partial_cmp(&b.quality_score).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find cheapest entry that meets quality threshold
    pub fn cheapest_for_quality(&self, min_quality: f64) -> Option<&BenchmarkEntry> {
        self.entries.iter().filter(|e| e.quality_score >= min_quality).min_by(|a, b| {
            a.cost
                .total_cost_usd
                .partial_cmp(&b.cost.total_cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Calculate efficiency score for an entry
    pub fn efficiency_score(&self, entry: &BenchmarkEntry) -> f64 {
        entry.efficiency_score()
    }

    /// Find most efficient entry (highest quality/cost ratio)
    pub fn most_efficient(&self) -> Option<&BenchmarkEntry> {
        self.entries.iter().max_by(|a, b| {
            a.efficiency_score()
                .partial_cmp(&b.efficiency_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find greenest entry (highest quality/energy ratio)
    pub fn greenest(&self) -> Option<&BenchmarkEntry> {
        self.entries.iter().max_by(|a, b| {
            a.energy_efficiency()
                .partial_cmp(&b.energy_efficiency())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get best entry by quality
    pub fn best_quality(&self) -> Option<&BenchmarkEntry> {
        self.entries.iter().max_by(|a, b| {
            a.quality_score.partial_cmp(&b.quality_score).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get cheapest entry
    pub fn cheapest(&self) -> Option<&BenchmarkEntry> {
        self.entries.iter().min_by(|a, b| {
            a.cost
                .total_cost_usd
                .partial_cmp(&b.cost.total_cost_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Filter entries by paradigm
    pub fn filter_by_paradigm(&self, paradigm: &ModelParadigm) -> Vec<&BenchmarkEntry> {
        self.entries
            .iter()
            .filter(|e| std::mem::discriminant(&e.paradigm) == std::mem::discriminant(paradigm))
            .collect()
    }

    /// Filter entries by device type
    pub fn filter_by_device_type<F>(&self, predicate: F) -> Vec<&BenchmarkEntry>
    where
        F: Fn(&ComputeDevice) -> bool,
    {
        self.entries.iter().filter(|e| predicate(&e.device)).collect()
    }

    /// Get statistics summary
    pub fn statistics(&self) -> BenchmarkStatistics {
        contract_pre_statistics!();
        if self.entries.is_empty() {
            return BenchmarkStatistics::default();
        }

        let qualities: Vec<f64> = self.entries.iter().map(|e| e.quality_score).collect();
        let costs: Vec<f64> = self.entries.iter().map(|e| e.cost.total_cost_usd).collect();
        let energies: Vec<f64> = self.entries.iter().map(|e| e.energy.joules_total).collect();

        BenchmarkStatistics {
            count: self.entries.len(),
            quality_min: qualities.iter().copied().fold(f64::INFINITY, f64::min),
            quality_max: qualities.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            quality_avg: qualities.iter().sum::<f64>() / qualities.len().max(1) as f64,
            cost_min: costs.iter().copied().fold(f64::INFINITY, f64::min),
            cost_max: costs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            cost_avg: costs.iter().sum::<f64>() / costs.len().max(1) as f64,
            energy_min: energies.iter().copied().fold(f64::INFINITY, f64::min),
            energy_max: energies.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            energy_avg: energies.iter().sum::<f64>() / energies.len().max(1) as f64,
            pareto_count: self.pareto_frontier().len(),
        }
    }

    /// Generate a comparison report
    pub fn comparison_report(&self) -> String {
        let stats = self.statistics();
        let frontier = self.pareto_frontier();

        let mut report = String::new();
        report.push_str(&format!("=== Benchmark Report ({} entries) ===\n\n", stats.count));

        report.push_str("Quality Scores:\n");
        report.push_str(&format!(
            "  Min: {:.4}  Max: {:.4}  Avg: {:.4}\n\n",
            stats.quality_min, stats.quality_max, stats.quality_avg
        ));

        report.push_str("Costs (USD):\n");
        report.push_str(&format!(
            "  Min: ${:.2}  Max: ${:.2}  Avg: ${:.2}\n\n",
            stats.cost_min, stats.cost_max, stats.cost_avg
        ));

        report.push_str(&format!("Pareto Frontier ({} entries):\n", frontier.len()));
        for entry in frontier.iter().take(5) {
            report.push_str(&format!(
                "  - {} ({}): quality={:.4}, cost=${:.2}\n",
                entry.run_id, entry.paradigm, entry.quality_score, entry.cost.total_cost_usd
            ));
        }

        if let Some(best) = self.most_efficient() {
            report.push_str(&format!(
                "\nMost Efficient: {} (score={:.2})\n",
                best.run_id,
                best.efficiency_score()
            ));
        }

        report
    }
}
