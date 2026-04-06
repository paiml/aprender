//! Benchmark entry type

use crate::efficiency::{ComputeDevice, CostMetrics, EnergyMetrics, ModelParadigm};
use serde::{Deserialize, Serialize};

/// A single benchmark entry
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    /// Unique run identifier
    pub run_id: String,
    /// Model paradigm used
    pub paradigm: ModelParadigm,
    /// Compute device used
    pub device: ComputeDevice,
    /// Quality score achieved (accuracy, F1, etc.)
    pub quality_score: f64,
    /// Cost metrics
    pub cost: CostMetrics,
    /// Energy metrics
    pub energy: EnergyMetrics,
}

impl BenchmarkEntry {
    /// Create a new benchmark entry
    pub fn new(
        run_id: impl Into<String>,
        paradigm: ModelParadigm,
        device: ComputeDevice,
        quality_score: f64,
        cost: CostMetrics,
        energy: EnergyMetrics,
    ) -> Self {
        Self { run_id: run_id.into(), paradigm, device, quality_score, cost, energy }
    }

    /// Get efficiency score (quality per dollar)
    pub fn efficiency_score(&self) -> f64 {
        if self.cost.total_cost_usd > 0.0 {
            self.quality_score / self.cost.total_cost_usd
        } else {
            f64::INFINITY
        }
    }

    /// Get energy efficiency (quality per kWh)
    pub fn energy_efficiency(&self) -> f64 {
        let kwh = self.energy.kwh();
        if kwh > 0.0 {
            self.quality_score / kwh
        } else {
            f64::INFINITY
        }
    }

    /// Get carbon efficiency (quality per kg CO2)
    pub fn carbon_efficiency(&self) -> f64 {
        if self.energy.carbon_kg > 0.0 {
            self.quality_score / self.energy.carbon_kg
        } else {
            f64::INFINITY
        }
    }

    /// Check if this entry dominates another (better in all metrics)
    pub fn dominates(&self, other: &Self) -> bool {
        self.quality_score >= other.quality_score
            && self.cost.total_cost_usd <= other.cost.total_cost_usd
            && self.energy.joules_total <= other.energy.joules_total
            && (self.quality_score > other.quality_score
                || self.cost.total_cost_usd < other.cost.total_cost_usd
                || self.energy.joules_total < other.energy.joules_total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::efficiency::device::CpuInfo;

    fn make_entry(quality: f64, cost_usd: f64, joules: f64, carbon: f64) -> BenchmarkEntry {
        let mut energy = EnergyMetrics::new(100.0, joules, 1000);
        energy.carbon_kg = carbon;
        BenchmarkEntry::new(
            "test-run",
            ModelParadigm::DeepLearning,
            ComputeDevice::Cpu(CpuInfo::detect()),
            quality,
            CostMetrics {
                total_cost_usd: cost_usd,
                cost_per_sample_usd: cost_usd / 1000.0,
                cost_per_epoch_usd: cost_usd / 10.0,
                device_hours: 1.0,
                rate_per_hour_usd: cost_usd,
            },
            energy,
        )
    }

    #[test]
    fn test_benchmark_entry_new() {
        let entry = make_entry(0.95, 10.0, 36000.0, 0.5);
        assert_eq!(entry.run_id, "test-run");
        assert_eq!(entry.paradigm, ModelParadigm::DeepLearning);
        assert!((entry.quality_score - 0.95).abs() < 1e-9);
        assert!((entry.cost.total_cost_usd - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_efficiency_score_normal() {
        let entry = make_entry(0.9, 10.0, 36000.0, 0.5);
        assert!((entry.efficiency_score() - 0.09).abs() < 1e-9);
    }

    #[test]
    fn test_efficiency_score_zero_cost() {
        let entry = make_entry(0.9, 0.0, 36000.0, 0.5);
        assert!(entry.efficiency_score().is_infinite());
    }

    #[test]
    fn test_energy_efficiency_normal() {
        let entry = make_entry(0.9, 10.0, 3600000.0, 0.5);
        // 3600000 joules = 1 kWh, so efficiency = 0.9 / 1.0 = 0.9
        assert!((entry.energy_efficiency() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_energy_efficiency_zero_energy() {
        let entry = make_entry(0.9, 10.0, 0.0, 0.5);
        assert!(entry.energy_efficiency().is_infinite());
    }

    #[test]
    fn test_carbon_efficiency_normal() {
        let entry = make_entry(0.9, 10.0, 36000.0, 0.5);
        // efficiency = 0.9 / 0.5 = 1.8
        assert!((entry.carbon_efficiency() - 1.8).abs() < 1e-9);
    }

    #[test]
    fn test_carbon_efficiency_zero_carbon() {
        let entry = make_entry(0.9, 10.0, 36000.0, 0.0);
        assert!(entry.carbon_efficiency().is_infinite());
    }

    #[test]
    fn test_dominates_better_quality() {
        let better = make_entry(0.95, 10.0, 36000.0, 0.5);
        let worse = make_entry(0.90, 10.0, 36000.0, 0.5);
        assert!(better.dominates(&worse));
        assert!(!worse.dominates(&better));
    }

    #[test]
    fn test_dominates_lower_cost() {
        let better = make_entry(0.9, 5.0, 36000.0, 0.5);
        let worse = make_entry(0.9, 10.0, 36000.0, 0.5);
        assert!(better.dominates(&worse));
        assert!(!worse.dominates(&better));
    }

    #[test]
    fn test_dominates_lower_energy() {
        let better = make_entry(0.9, 10.0, 18000.0, 0.5);
        let worse = make_entry(0.9, 10.0, 36000.0, 0.5);
        assert!(better.dominates(&worse));
        assert!(!worse.dominates(&better));
    }

    #[test]
    fn test_dominates_equal_entries() {
        let entry1 = make_entry(0.9, 10.0, 36000.0, 0.5);
        let entry2 = make_entry(0.9, 10.0, 36000.0, 0.5);
        // Equal entries don't dominate each other (need strict improvement in at least one)
        assert!(!entry1.dominates(&entry2));
        assert!(!entry2.dominates(&entry1));
    }

    #[test]
    fn test_dominates_tradeoff() {
        // Better quality but worse cost - neither dominates
        let entry1 = make_entry(0.95, 20.0, 36000.0, 0.5);
        let entry2 = make_entry(0.90, 10.0, 36000.0, 0.5);
        assert!(!entry1.dominates(&entry2));
        assert!(!entry2.dominates(&entry1));
    }

    #[test]
    fn test_benchmark_entry_clone() {
        let entry = make_entry(0.9, 10.0, 36000.0, 0.5);
        let cloned = entry.clone();
        assert_eq!(entry, cloned);
    }

    #[test]
    fn test_benchmark_entry_serde() {
        let entry = make_entry(0.9, 10.0, 36000.0, 0.5);
        let json = serde_json::to_string(&entry).expect("JSON serialization should succeed");
        let deserialized: BenchmarkEntry =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(entry.run_id, deserialized.run_id);
        assert!((entry.quality_score - deserialized.quality_score).abs() < 1e-9);
    }

    #[test]
    fn test_benchmark_entry_debug() {
        let entry = make_entry(0.9, 10.0, 36000.0, 0.5);
        let debug_str = format!("{entry:?}");
        assert!(debug_str.contains("BenchmarkEntry"));
        assert!(debug_str.contains("test-run"));
    }
}
