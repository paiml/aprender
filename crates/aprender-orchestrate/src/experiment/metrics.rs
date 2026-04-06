//! Energy and cost metrics for experiment tracking.
//!
//! This module provides types for tracking energy consumption, cost, and
//! efficiency metrics during ML experiments.

use serde::{Deserialize, Serialize};

/// Energy consumption metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyMetrics {
    /// Total energy consumed in joules
    pub total_joules: f64,
    /// Average power draw in watts
    pub average_power_watts: f64,
    /// Peak power draw in watts
    pub peak_power_watts: f64,
    /// Duration of measurement in seconds
    pub duration_seconds: f64,
    /// CO2 equivalent emissions in grams (based on grid carbon intensity)
    pub co2_grams: Option<f64>,
    /// Power Usage Effectiveness (datacenter overhead)
    pub pue: f64,
}

impl EnergyMetrics {
    /// Create new energy metrics
    pub fn new(
        total_joules: f64,
        average_power_watts: f64,
        peak_power_watts: f64,
        duration_seconds: f64,
    ) -> Self {
        Self {
            total_joules,
            average_power_watts,
            peak_power_watts,
            duration_seconds,
            co2_grams: None,
            pue: 1.0,
        }
    }

    /// Calculate CO2 emissions based on carbon intensity (g CO2/kWh)
    pub fn with_carbon_intensity(mut self, carbon_intensity_g_per_kwh: f64) -> Self {
        let kwh = self.total_joules / 3_600_000.0;
        self.co2_grams = Some(kwh * carbon_intensity_g_per_kwh * self.pue);
        self
    }

    /// Set the Power Usage Effectiveness factor
    pub fn with_pue(mut self, pue: f64) -> Self {
        let old_pue = self.pue;
        self.pue = pue;
        // Recalculate CO2 if already set (scale by new PUE / old PUE)
        if let Some(co2) = self.co2_grams {
            self.co2_grams = Some(co2 / old_pue * pue);
        }
        self
    }

    /// Calculate energy efficiency in FLOPS per watt
    pub fn flops_per_watt(&self, total_flops: f64) -> f64 {
        if self.average_power_watts > 0.0 {
            total_flops / self.average_power_watts
        } else {
            0.0
        }
    }
}

/// Cost metrics for experiments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Compute cost in USD
    pub compute_cost_usd: f64,
    /// Storage cost in USD
    pub storage_cost_usd: f64,
    /// Network transfer cost in USD
    pub network_cost_usd: f64,
    /// Total cost in USD
    pub total_cost_usd: f64,
    /// Cost per FLOP in USD
    pub cost_per_flop: Option<f64>,
    /// Cost per sample processed
    pub cost_per_sample: Option<f64>,
    /// Currency (default USD)
    pub currency: String,
}

impl CostMetrics {
    /// Create new cost metrics
    pub fn new(compute_cost: f64, storage_cost: f64, network_cost: f64) -> Self {
        Self {
            compute_cost_usd: compute_cost,
            storage_cost_usd: storage_cost,
            network_cost_usd: network_cost,
            total_cost_usd: compute_cost + storage_cost + network_cost,
            cost_per_flop: None,
            cost_per_sample: None,
            currency: "USD".to_string(),
        }
    }

    /// Add FLOP-based cost calculation
    pub fn with_flops(mut self, total_flops: f64) -> Self {
        if total_flops > 0.0 {
            self.cost_per_flop = Some(self.total_cost_usd / total_flops);
        }
        self
    }

    /// Add sample-based cost calculation
    pub fn with_samples(mut self, total_samples: u64) -> Self {
        if total_samples > 0 {
            self.cost_per_sample = Some(self.total_cost_usd / total_samples as f64);
        }
        self
    }
}
