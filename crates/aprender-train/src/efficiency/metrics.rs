//! Energy and Cost Metrics (ENT-009)
//!
//! Provides tracking for energy consumption and training costs.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Energy consumption metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyMetrics {
    /// Average power consumption in watts
    pub watts_avg: f64,
    /// Total energy consumed in joules
    pub joules_total: f64,
    /// Estimated carbon emissions in kg CO2
    pub carbon_kg: f64,
    /// Training efficiency: samples processed per joule
    pub efficiency_samples_per_joule: f64,
}

impl EnergyMetrics {
    /// Create new energy metrics
    pub fn new(watts_avg: f64, joules_total: f64, samples: u64) -> Self {
        let efficiency = if joules_total > 0.0 { samples as f64 / joules_total } else { 0.0 };

        Self { watts_avg, joules_total, carbon_kg: 0.0, efficiency_samples_per_joule: efficiency }
    }

    /// Create energy metrics from power readings over time
    ///
    /// # Arguments
    ///
    /// * `readings` - Slice of (timestamp, watts) readings
    /// * `samples` - Total samples processed during this period
    pub fn from_power_readings(readings: &[(Instant, f64)], samples: u64) -> Self {
        if readings.len() < 2 {
            return Self::new(0.0, 0.0, samples);
        }

        // Calculate average power and total energy
        let mut total_joules = 0.0;
        let mut total_watts = 0.0;

        for i in 1..readings.len() {
            let (t1, w1) = readings[i - 1];
            let (t2, w2) = readings[i];

            let duration_secs = t2.duration_since(t1).as_secs_f64();
            let avg_watts = f64::midpoint(w1, w2);

            total_joules += avg_watts * duration_secs;
            total_watts += avg_watts;
        }

        let watts_avg = total_watts / (readings.len() - 1).max(1) as f64;
        Self::new(watts_avg, total_joules, samples)
    }

    /// Set carbon emissions based on grid carbon intensity
    ///
    /// # Arguments
    ///
    /// * `kg_co2_per_kwh` - Carbon intensity (kg CO2 per kWh)
    ///   - US average: ~0.4
    ///   - EU average: ~0.3
    ///   - France (nuclear): ~0.05
    ///   - Coal-heavy: ~0.8
    pub fn with_carbon_intensity(mut self, kg_co2_per_kwh: f64) -> Self {
        let kwh = self.joules_total / 3_600_000.0; // Joules to kWh
        self.carbon_kg = kwh * kg_co2_per_kwh;
        self
    }

    /// Get energy in kWh
    pub fn kwh(&self) -> f64 {
        self.joules_total / 3_600_000.0
    }

    /// Get energy in Wh
    pub fn wh(&self) -> f64 {
        self.joules_total / 3_600.0
    }

    /// Estimate cost at given electricity rate
    pub fn estimated_cost_usd(&self, usd_per_kwh: f64) -> f64 {
        self.kwh() * usd_per_kwh
    }

    /// Create zero energy metrics
    pub fn zero() -> Self {
        Self {
            watts_avg: 0.0,
            joules_total: 0.0,
            carbon_kg: 0.0,
            efficiency_samples_per_joule: 0.0,
        }
    }

    /// Add two energy metrics (for aggregation)
    pub fn add(&self, other: &Self) -> Self {
        let total_joules = self.joules_total + other.joules_total;
        let weighted_watts = if total_joules > 0.0 {
            (self.watts_avg * self.joules_total + other.watts_avg * other.joules_total)
                / total_joules
        } else {
            0.0
        };

        Self {
            watts_avg: weighted_watts,
            joules_total: total_joules,
            carbon_kg: self.carbon_kg + other.carbon_kg,
            efficiency_samples_per_joule: f64::midpoint(
                self.efficiency_samples_per_joule,
                other.efficiency_samples_per_joule,
            ),
        }
    }
}

impl Default for EnergyMetrics {
    fn default() -> Self {
        Self::zero()
    }
}

/// Cost metrics for training
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Cost per sample in USD
    pub cost_per_sample_usd: f64,
    /// Cost per epoch in USD
    pub cost_per_epoch_usd: f64,
    /// Total cost in USD
    pub total_cost_usd: f64,
    /// Total device hours used
    pub device_hours: f64,
    /// Rate per hour in USD
    pub rate_per_hour_usd: f64,
}

/// Common cloud GPU pricing constants (approximate hourly rates)
pub mod pricing {
    /// NVIDIA A100 (40GB) spot price ~$1.00/hr
    pub const A100_SPOT: f64 = 1.00;
    /// NVIDIA A100 on-demand ~$3.00/hr
    pub const A100_ONDEMAND: f64 = 3.00;
    /// NVIDIA V100 spot ~$0.50/hr
    pub const V100_SPOT: f64 = 0.50;
    /// NVIDIA T4 spot ~$0.15/hr
    pub const T4_SPOT: f64 = 0.15;
    /// CPU instance (8 cores) ~$0.20/hr
    pub const CPU_8CORE: f64 = 0.20;
    /// Apple M2 Mac Studio (amortized) ~$0.05/hr
    pub const M2_AMORTIZED: f64 = 0.05;
}

impl CostMetrics {
    /// Create new cost metrics
    ///
    /// # Arguments
    ///
    /// * `device_hours` - Total compute time in hours
    /// * `rate_per_hour_usd` - Cost rate per device hour
    /// * `samples` - Total samples processed
    /// * `epochs` - Number of training epochs
    pub fn new(device_hours: f64, rate_per_hour_usd: f64, samples: u64, epochs: u32) -> Self {
        let total_cost = device_hours * rate_per_hour_usd;
        let cost_per_sample = if samples > 0 { total_cost / samples as f64 } else { 0.0 };
        let cost_per_epoch = if epochs > 0 { total_cost / f64::from(epochs) } else { 0.0 };

        Self {
            cost_per_sample_usd: cost_per_sample,
            cost_per_epoch_usd: cost_per_epoch,
            total_cost_usd: total_cost,
            device_hours,
            rate_per_hour_usd,
        }
    }

    /// Create cost metrics from duration
    pub fn from_duration(
        duration: Duration,
        rate_per_hour_usd: f64,
        samples: u64,
        epochs: u32,
    ) -> Self {
        let device_hours = duration.as_secs_f64() / 3600.0;
        Self::new(device_hours, rate_per_hour_usd, samples, epochs)
    }

    /// Create zero cost metrics
    pub fn zero() -> Self {
        Self {
            cost_per_sample_usd: 0.0,
            cost_per_epoch_usd: 0.0,
            total_cost_usd: 0.0,
            device_hours: 0.0,
            rate_per_hour_usd: 0.0,
        }
    }

    /// Add two cost metrics (for aggregation)
    pub fn add(&self, other: &Self) -> Self {
        let total_hours = self.device_hours + other.device_hours;
        let weighted_rate = if total_hours > 0.0 {
            (self.rate_per_hour_usd * self.device_hours
                + other.rate_per_hour_usd * other.device_hours)
                / total_hours
        } else {
            0.0
        };

        Self {
            cost_per_sample_usd: self.cost_per_sample_usd + other.cost_per_sample_usd,
            cost_per_epoch_usd: self.cost_per_epoch_usd + other.cost_per_epoch_usd,
            total_cost_usd: self.total_cost_usd + other.total_cost_usd,
            device_hours: total_hours,
            rate_per_hour_usd: weighted_rate,
        }
    }

    /// Get cost efficiency (samples per dollar)
    pub fn samples_per_dollar(&self, samples: u64) -> f64 {
        if self.total_cost_usd > 0.0 {
            samples as f64 / self.total_cost_usd
        } else {
            0.0
        }
    }

    /// Estimate cost for additional training
    pub fn estimate_additional(&self, additional_hours: f64) -> f64 {
        additional_hours * self.rate_per_hour_usd
    }
}

impl Default for CostMetrics {
    fn default() -> Self {
        Self::zero()
    }
}

/// Combined efficiency metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Energy metrics
    pub energy: EnergyMetrics,
    /// Cost metrics
    pub cost: CostMetrics,
    /// Quality score achieved (accuracy, F1, etc.)
    pub quality_score: f64,
}

impl EfficiencyMetrics {
    /// Create new efficiency metrics
    pub fn new(energy: EnergyMetrics, cost: CostMetrics, quality_score: f64) -> Self {
        Self { energy, cost, quality_score }
    }

    /// Calculate quality per dollar
    pub fn quality_per_dollar(&self) -> f64 {
        if self.cost.total_cost_usd > 0.0 {
            self.quality_score / self.cost.total_cost_usd
        } else {
            0.0
        }
    }

    /// Calculate quality per kWh
    pub fn quality_per_kwh(&self) -> f64 {
        let kwh = self.energy.kwh();
        if kwh > 0.0 {
            self.quality_score / kwh
        } else {
            0.0
        }
    }

    /// Calculate quality per kg CO2
    pub fn quality_per_carbon(&self) -> f64 {
        if self.energy.carbon_kg > 0.0 {
            self.quality_score / self.energy.carbon_kg
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_metrics_new() {
        let metrics = EnergyMetrics::new(200.0, 720_000.0, 1000);

        assert!((metrics.watts_avg - 200.0).abs() < f64::EPSILON);
        assert!((metrics.joules_total - 720_000.0).abs() < f64::EPSILON);
        assert!((metrics.efficiency_samples_per_joule - 1000.0 / 720_000.0).abs() < 0.0001);
    }

    #[test]
    fn test_energy_metrics_from_power_readings() {
        let start = Instant::now();
        let readings = vec![
            (start, 100.0),
            (start + Duration::from_secs(1), 150.0),
            (start + Duration::from_secs(2), 200.0),
        ];

        let metrics = EnergyMetrics::from_power_readings(&readings, 100);

        // Average power: (100+150)/2 + (150+200)/2 = 125 + 175 = 300 / 2 = 150
        assert!((metrics.watts_avg - 150.0).abs() < 1.0);
        // Joules: 125 * 1 + 175 * 1 = 300
        assert!((metrics.joules_total - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_energy_metrics_with_carbon() {
        let metrics = EnergyMetrics::new(200.0, 3_600_000.0, 1000) // 1 kWh
            .with_carbon_intensity(0.4); // US average

        assert!((metrics.kwh() - 1.0).abs() < 0.01);
        assert!((metrics.carbon_kg - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_energy_metrics_kwh() {
        let metrics = EnergyMetrics::new(200.0, 7_200_000.0, 1000); // 2 kWh
        assert!((metrics.kwh() - 2.0).abs() < 0.01);
        assert!((metrics.wh() - 2000.0).abs() < 0.1);
    }

    #[test]
    fn test_energy_metrics_cost() {
        let metrics = EnergyMetrics::new(200.0, 3_600_000.0, 1000); // 1 kWh
        let cost = metrics.estimated_cost_usd(0.15); // $0.15/kWh
        assert!((cost - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_energy_metrics_add() {
        let m1 = EnergyMetrics::new(100.0, 1000.0, 100);
        let m2 = EnergyMetrics::new(200.0, 2000.0, 200);

        let combined = m1.add(&m2);
        assert!((combined.joules_total - 3000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_energy_metrics_zero() {
        let zero = EnergyMetrics::zero();
        assert!((zero.watts_avg - 0.0).abs() < f64::EPSILON);
        assert!((zero.joules_total - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_metrics_new() {
        let metrics = CostMetrics::new(2.0, 1.50, 10000, 5);

        assert!((metrics.device_hours - 2.0).abs() < f64::EPSILON);
        assert!((metrics.rate_per_hour_usd - 1.50).abs() < f64::EPSILON);
        assert!((metrics.total_cost_usd - 3.0).abs() < 0.01);
        assert!((metrics.cost_per_sample_usd - 0.0003).abs() < 0.0001);
        assert!((metrics.cost_per_epoch_usd - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_cost_metrics_from_duration() {
        let duration = Duration::from_secs(7200); // 2 hours
        let metrics = CostMetrics::from_duration(duration, 1.0, 1000, 10);

        assert!((metrics.device_hours - 2.0).abs() < 0.01);
        assert!((metrics.total_cost_usd - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_metrics_samples_per_dollar() {
        let metrics = CostMetrics::new(1.0, 1.0, 1000, 1);
        assert!((metrics.samples_per_dollar(1000) - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_metrics_estimate_additional() {
        let metrics = CostMetrics::new(1.0, 2.50, 1000, 1);
        let additional = metrics.estimate_additional(4.0);
        assert!((additional - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_metrics_add() {
        let m1 = CostMetrics::new(1.0, 1.0, 500, 1);
        let m2 = CostMetrics::new(2.0, 2.0, 1000, 2);

        let combined = m1.add(&m2);
        assert!((combined.device_hours - 3.0).abs() < f64::EPSILON);
        assert!((combined.total_cost_usd - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_metrics_pricing_constants() {
        assert!(pricing::A100_SPOT > 0.0);
        assert!(pricing::A100_ONDEMAND > pricing::A100_SPOT);
        assert!(pricing::T4_SPOT < pricing::V100_SPOT);
    }

    #[test]
    fn test_efficiency_metrics() {
        let energy = EnergyMetrics::new(200.0, 3_600_000.0, 1000);
        let cost = CostMetrics::new(1.0, 2.0, 1000, 10);
        let efficiency = EfficiencyMetrics::new(energy, cost, 0.95);

        assert!((efficiency.quality_score - 0.95).abs() < f64::EPSILON);
        assert!(efficiency.quality_per_dollar() > 0.0);
        assert!(efficiency.quality_per_kwh() > 0.0);
    }

    #[test]
    fn test_efficiency_metrics_quality_per_carbon() {
        let energy = EnergyMetrics::new(200.0, 3_600_000.0, 1000).with_carbon_intensity(0.4);
        let cost = CostMetrics::new(1.0, 2.0, 1000, 10);
        let efficiency = EfficiencyMetrics::new(energy, cost, 0.95);

        assert!(efficiency.quality_per_carbon() > 0.0);
    }

    #[test]
    fn test_energy_metrics_serialization() {
        let metrics = EnergyMetrics::new(200.0, 720_000.0, 1000);
        let json = serde_json::to_string(&metrics).expect("JSON serialization should succeed");
        let parsed: EnergyMetrics =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert!((parsed.watts_avg - metrics.watts_avg).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cost_metrics_serialization() {
        let metrics = CostMetrics::new(2.0, 1.50, 10000, 5);
        let json = serde_json::to_string(&metrics).expect("JSON serialization should succeed");
        let parsed: CostMetrics =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert!((parsed.total_cost_usd - metrics.total_cost_usd).abs() < f64::EPSILON);
    }
}
