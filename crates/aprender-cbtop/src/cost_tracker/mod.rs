//! Cost and Energy Efficiency Tracker (PMAT-042)
//!
//! Track inference cost per token and energy consumption per operation.
//!
//! # Features
//!
//! - Energy consumption tracking (joules, kWh)
//! - Cost calculation with provider pricing
//! - Cost per token and throughput metrics
//! - Carbon emissions estimation
//!
//! # Falsification Criteria (F1341-F1350)
//!
//! See `tests/cost_tracker_f1341.rs` for falsification tests.

use std::collections::HashMap;

/// Joules per kWh
pub const JOULES_PER_KWH: f64 = 3_600_000.0;

/// Default grid carbon intensity (gCO2/kWh)
pub const DEFAULT_CARBON_INTENSITY: f64 = 400.0;

/// Cloud provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloudProvider {
    /// Amazon Web Services
    Aws,
    /// Google Cloud Platform
    Gcp,
    /// Microsoft Azure
    Azure,
    /// On-premise / Self-hosted
    OnPrem,
}

impl CloudProvider {
    /// Get provider name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Aws => "AWS",
            Self::Gcp => "GCP",
            Self::Azure => "Azure",
            Self::OnPrem => "On-Premise",
        }
    }
}

/// GPU pricing tier
#[derive(Debug, Clone)]
pub struct GpuPricing {
    /// Provider
    pub provider: CloudProvider,
    /// GPU type (e.g., "A100", "H100")
    pub gpu_type: String,
    /// Price per hour (USD)
    pub price_per_hour: f64,
    /// Energy consumption (watts)
    pub power_watts: f64,
}

impl GpuPricing {
    /// Create new pricing
    pub fn new(
        provider: CloudProvider,
        gpu_type: &str,
        price_per_hour: f64,
        power_watts: f64,
    ) -> Self {
        Self { provider, gpu_type: gpu_type.to_string(), price_per_hour, power_watts }
    }

    /// Price per second
    pub fn price_per_second(&self) -> f64 {
        self.price_per_hour / 3600.0
    }

    /// Energy per second (joules)
    pub fn joules_per_second(&self) -> f64 {
        self.power_watts
    }
}

/// Pre-defined GPU pricing (approximate, January 2026)
pub fn default_gpu_pricing() -> Vec<GpuPricing> {
    vec![
        // AWS
        GpuPricing::new(CloudProvider::Aws, "A100-40GB", 4.10, 400.0),
        GpuPricing::new(CloudProvider::Aws, "A100-80GB", 5.12, 400.0),
        GpuPricing::new(CloudProvider::Aws, "H100", 8.22, 700.0),
        // GCP
        GpuPricing::new(CloudProvider::Gcp, "A100-40GB", 3.67, 400.0),
        GpuPricing::new(CloudProvider::Gcp, "A100-80GB", 4.87, 400.0),
        GpuPricing::new(CloudProvider::Gcp, "H100", 7.65, 700.0),
        // Azure
        GpuPricing::new(CloudProvider::Azure, "A100-40GB", 3.85, 400.0),
        GpuPricing::new(CloudProvider::Azure, "A100-80GB", 4.95, 400.0),
        GpuPricing::new(CloudProvider::Azure, "H100", 8.00, 700.0),
        // On-prem (electricity only, ~$0.10/kWh)
        GpuPricing::new(CloudProvider::OnPrem, "A100-40GB", 0.04, 400.0),
        GpuPricing::new(CloudProvider::OnPrem, "H100", 0.07, 700.0),
    ]
}

/// Energy measurement
#[derive(Debug, Clone, Default)]
pub struct EnergyMeasurement {
    /// Joules consumed
    pub joules: f64,
    /// Duration in seconds
    pub duration_sec: f64,
    /// Power in watts
    pub power_watts: f64,
}

impl EnergyMeasurement {
    /// Create from power and duration
    pub fn from_power_duration(power_watts: f64, duration_sec: f64) -> Self {
        Self { joules: power_watts * duration_sec, duration_sec, power_watts }
    }

    /// Create from joules and duration
    pub fn from_joules_duration(joules: f64, duration_sec: f64) -> Self {
        let power_watts = if duration_sec > 0.0 { joules / duration_sec } else { 0.0 };
        Self { joules, duration_sec, power_watts }
    }

    /// Get kWh
    pub fn kwh(&self) -> f64 {
        self.joules / JOULES_PER_KWH
    }
}

/// Cost calculation result
#[derive(Debug, Clone)]
pub struct CostResult {
    /// Total cost (USD)
    pub total_cost: f64,
    /// Cost per token
    pub cost_per_token: f64,
    /// Cost per million tokens
    pub cost_per_million_tokens: f64,
    /// Energy consumed (joules)
    pub energy_joules: f64,
    /// Energy consumed (kWh)
    pub energy_kwh: f64,
    /// Carbon emissions (gCO2)
    pub carbon_g: f64,
    /// Duration (seconds)
    pub duration_sec: f64,
    /// Token count
    pub token_count: u64,
}

impl CostResult {
    /// Create from components
    pub fn new(
        cost: f64,
        energy_joules: f64,
        carbon_g: f64,
        duration_sec: f64,
        token_count: u64,
    ) -> Self {
        let cost_per_token = if token_count > 0 { cost / token_count as f64 } else { 0.0 };

        Self {
            total_cost: cost,
            cost_per_token,
            cost_per_million_tokens: cost_per_token * 1_000_000.0,
            energy_joules,
            energy_kwh: energy_joules / JOULES_PER_KWH,
            carbon_g,
            duration_sec,
            token_count,
        }
    }

    /// Format as JSON
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"total_cost":{:.6},"cost_per_million_tokens":{:.4},"energy_kwh":{:.6},"carbon_g":{:.2},"duration_sec":{:.2},"token_count":{}}}"#,
            self.total_cost,
            self.cost_per_million_tokens,
            self.energy_kwh,
            self.carbon_g,
            self.duration_sec,
            self.token_count
        )
    }
}

/// Cost comparison between baseline and current
#[derive(Debug, Clone)]
pub struct CostComparison {
    /// Baseline cost
    pub baseline: CostResult,
    /// Current cost
    pub current: CostResult,
    /// Cost change percent
    pub cost_change_percent: f64,
    /// Energy change percent
    pub energy_change_percent: f64,
    /// Is regression (cost increased)
    pub is_regression: bool,
}

impl CostComparison {
    /// Create comparison
    pub fn new(baseline: CostResult, current: CostResult) -> Self {
        let cost_change_percent = if baseline.total_cost > 0.0 {
            ((current.total_cost - baseline.total_cost) / baseline.total_cost) * 100.0
        } else {
            0.0
        };

        let energy_change_percent = if baseline.energy_joules > 0.0 {
            ((current.energy_joules - baseline.energy_joules) / baseline.energy_joules) * 100.0
        } else {
            0.0
        };

        Self {
            is_regression: cost_change_percent > 5.0, // >5% cost increase is regression
            baseline,
            current,
            cost_change_percent,
            energy_change_percent,
        }
    }
}

/// Budget alert
#[derive(Debug, Clone)]
pub struct BudgetAlert {
    /// Alert message
    pub message: String,
    /// Current spend
    pub current_spend: f64,
    /// Budget limit
    pub budget_limit: f64,
    /// Percent used
    pub percent_used: f64,
}

/// Cost tracker
#[derive(Debug)]
pub struct CostTracker {
    /// GPU pricing database
    pricing: HashMap<String, GpuPricing>,
    /// Current GPU type
    current_gpu: String,
    /// Current provider
    current_provider: CloudProvider,
    /// Carbon intensity (gCO2/kWh)
    carbon_intensity: f64,
    /// Historical cost records
    history: Vec<CostResult>,
    /// Max history size
    max_history: usize,
    /// Budget limit (USD)
    budget_limit: Option<f64>,
    /// Total spend
    total_spend: f64,
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CostTracker {
    /// Create new tracker
    pub fn new() -> Self {
        let pricing: HashMap<String, GpuPricing> = default_gpu_pricing()
            .into_iter()
            .map(|p| (format!("{}-{}", p.provider.name(), p.gpu_type), p))
            .collect();

        Self {
            pricing,
            current_gpu: "A100-40GB".to_string(),
            current_provider: CloudProvider::Aws,
            carbon_intensity: DEFAULT_CARBON_INTENSITY,
            history: Vec::new(),
            max_history: 1000,
            budget_limit: None,
            total_spend: 0.0,
        }
    }

    /// Set current GPU
    pub fn with_gpu(mut self, provider: CloudProvider, gpu_type: &str) -> Self {
        self.current_provider = provider;
        self.current_gpu = gpu_type.to_string();
        self
    }

    /// Set carbon intensity
    pub fn with_carbon_intensity(mut self, intensity: f64) -> Self {
        self.carbon_intensity = intensity;
        self
    }

    /// Set budget limit
    pub fn with_budget(mut self, limit: f64) -> Self {
        self.budget_limit = Some(limit);
        self
    }

    /// Get current pricing
    fn current_pricing(&self) -> Option<&GpuPricing> {
        let key = format!("{}-{}", self.current_provider.name(), self.current_gpu);
        self.pricing.get(&key)
    }

    /// Calculate cost for duration and tokens
    pub fn calculate_cost(&mut self, duration_sec: f64, token_count: u64) -> CostResult {
        let pricing = self.current_pricing().cloned().unwrap_or_else(|| {
            GpuPricing::new(self.current_provider, &self.current_gpu, 5.0, 400.0)
        });

        let cost = pricing.price_per_second() * duration_sec;
        let energy_joules = pricing.joules_per_second() * duration_sec;
        let energy_kwh = energy_joules / JOULES_PER_KWH;
        let carbon_g = energy_kwh * self.carbon_intensity;

        let result = CostResult::new(cost, energy_joules, carbon_g, duration_sec, token_count);

        // Track spend
        self.total_spend += cost;

        // Store in history
        self.history.push(result.clone());
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }

        result
    }

    /// Calculate cost from energy measurement
    pub fn calculate_from_energy(
        &mut self,
        energy: &EnergyMeasurement,
        token_count: u64,
    ) -> CostResult {
        let pricing = self.current_pricing().cloned().unwrap_or_else(|| {
            GpuPricing::new(self.current_provider, &self.current_gpu, 5.0, 400.0)
        });

        let cost = pricing.price_per_second() * energy.duration_sec;
        let energy_kwh = energy.kwh();
        let carbon_g = energy_kwh * self.carbon_intensity;

        let result =
            CostResult::new(cost, energy.joules, carbon_g, energy.duration_sec, token_count);

        self.total_spend += cost;

        self.history.push(result.clone());
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }

        result
    }

    /// Get total spend
    pub fn total_spend(&self) -> f64 {
        self.total_spend
    }

    /// Check budget
    pub fn check_budget(&self) -> Option<BudgetAlert> {
        let limit = self.budget_limit?;

        let percent_used = (self.total_spend / limit) * 100.0;

        if percent_used >= 80.0 {
            Some(BudgetAlert {
                message: format!(
                    "Budget alert: {:.1}% used (${:.2} of ${:.2})",
                    percent_used, self.total_spend, limit
                ),
                current_spend: self.total_spend,
                budget_limit: limit,
                percent_used,
            })
        } else {
            None
        }
    }

    /// Detect cost creep (trend analysis)
    pub fn detect_cost_creep(&self) -> Option<f64> {
        if self.history.len() < 10 {
            return None;
        }

        // Compare last 10 to previous 10
        let recent: f64 =
            self.history.iter().rev().take(10).map(|r| r.cost_per_million_tokens).sum::<f64>()
                / 10.0;

        let older_start = self.history.len().saturating_sub(20);
        let older: f64 = self.history[older_start..older_start + 10.min(self.history.len() - 10)]
            .iter()
            .map(|r| r.cost_per_million_tokens)
            .sum::<f64>()
            / 10.0;

        if older > 0.0 {
            let change = ((recent - older) / older) * 100.0;
            if change > 10.0 {
                return Some(change);
            }
        }

        None
    }

    /// Get cost history
    pub fn history(&self) -> &[CostResult] {
        &self.history
    }

    /// Export history to CSV
    pub fn export_csv(&self) -> String {
        let mut lines =
            vec!["duration_sec,token_count,total_cost,cost_per_million,energy_kwh,carbon_g"
                .to_string()];

        for result in &self.history {
            lines.push(format!(
                "{:.2},{},{:.6},{:.4},{:.6},{:.2}",
                result.duration_sec,
                result.token_count,
                result.total_cost,
                result.cost_per_million_tokens,
                result.energy_kwh,
                result.carbon_g
            ));
        }

        lines.join("\n")
    }

    /// Export history to JSON
    pub fn export_json(&self) -> String {
        let entries: Vec<String> = self.history.iter().map(|r| r.to_json()).collect();
        format!("[{}]", entries.join(","))
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.total_spend = 0.0;
    }
}

#[cfg(test)]
mod tests;
