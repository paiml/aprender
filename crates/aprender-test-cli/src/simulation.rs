//! Simulation Playback Module (PROBAR-SPEC-006 Section K)
//!
//! Implements deterministic replay, parameterized replay, Monte Carlo
//! simulation, and chaos engineering for load test analysis.
//!
//! Based on research:
//! - [C11] Wasm-R3 record-reduce-replay methodology

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::format_push_string)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cloned_instead_of_copied)]
#![allow(clippy::useless_format)]
#![allow(clippy::single_char_add_str)]
#![allow(clippy::useless_vec)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// =============================================================================
// K.2 Simulation Modes
// =============================================================================

/// Simulation mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimulationMode {
    /// Exact reproduction of recorded session
    DeterministicReplay,
    /// Replay with modified parameters
    Parameterized {
        /// Parameter multipliers
        multipliers: HashMap<String, f64>,
    },
    /// Monte Carlo randomized variations
    MonteCarlo {
        /// Number of iterations
        iterations: u32,
        /// Random seed (for reproducibility)
        seed: Option<u64>,
    },
    /// Chaos engineering with failure injection
    Chaos {
        /// Failures to inject
        injections: Vec<FailureInjection>,
    },
}

/// Failure injection for chaos engineering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureInjection {
    /// Injection type
    pub injection_type: InjectionType,
    /// Probability (0.0 - 1.0)
    pub probability: f64,
    /// Target component
    pub target: String,
    /// Duration in milliseconds (for delays)
    pub duration_ms: Option<u64>,
}

impl FailureInjection {
    /// Create a latency injection
    pub fn latency(target: &str, probability: f64, delay_ms: u64) -> Self {
        Self {
            injection_type: InjectionType::Latency,
            probability,
            target: target.to_string(),
            duration_ms: Some(delay_ms),
        }
    }

    /// Create a packet loss injection
    pub fn packet_loss(target: &str, probability: f64) -> Self {
        Self {
            injection_type: InjectionType::PacketLoss,
            probability,
            target: target.to_string(),
            duration_ms: None,
        }
    }

    /// Create an error injection
    pub fn error(target: &str, probability: f64) -> Self {
        Self {
            injection_type: InjectionType::Error,
            probability,
            target: target.to_string(),
            duration_ms: None,
        }
    }
}

/// Types of failure injection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InjectionType {
    /// Add latency
    Latency,
    /// Drop packets
    PacketLoss,
    /// Return errors
    Error,
    /// Timeout
    Timeout,
    /// CPU throttle
    CpuThrottle,
    /// Memory pressure
    MemoryPressure,
}

impl InjectionType {
    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Latency => "latency",
            Self::PacketLoss => "packet_loss",
            Self::Error => "error",
            Self::Timeout => "timeout",
            Self::CpuThrottle => "cpu_throttle",
            Self::MemoryPressure => "memory_pressure",
        }
    }
}

// =============================================================================
// K.1 Simulation Configuration
// =============================================================================

/// Simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Base session file path
    pub base_session: PathBuf,
    /// Simulation mode
    pub mode: SimulationMode,
    /// Parameter variations for Monte Carlo
    pub parameter_variations: HashMap<String, ParameterVariation>,
    /// Output configuration
    pub output: SimulationOutput,
}

impl SimulationConfig {
    /// Create deterministic replay config
    pub fn deterministic(session_path: PathBuf) -> Self {
        Self {
            base_session: session_path,
            mode: SimulationMode::DeterministicReplay,
            parameter_variations: HashMap::new(),
            output: SimulationOutput::default(),
        }
    }

    /// Create Monte Carlo config
    pub fn monte_carlo(session_path: PathBuf, iterations: u32) -> Self {
        Self {
            base_session: session_path,
            mode: SimulationMode::MonteCarlo {
                iterations,
                seed: None,
            },
            parameter_variations: HashMap::new(),
            output: SimulationOutput::default(),
        }
    }

    /// Add parameter variation
    pub fn with_variation(mut self, name: &str, variation: ParameterVariation) -> Self {
        self.parameter_variations
            .insert(name.to_string(), variation);
        self
    }
}

/// Parameter variation definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterVariation {
    /// Distribution type
    pub distribution: Distribution,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Base value (center for normal distribution)
    pub base: f64,
}

impl ParameterVariation {
    /// Create uniform variation
    pub fn uniform(min: f64, max: f64) -> Self {
        Self {
            distribution: Distribution::Uniform,
            min,
            max,
            base: (min + max) / 2.0,
        }
    }

    /// Create normal variation
    pub fn normal(mean: f64, std_dev: f64) -> Self {
        Self {
            distribution: Distribution::Normal { mean, std_dev },
            min: mean - 3.0 * std_dev,
            max: mean + 3.0 * std_dev,
            base: mean,
        }
    }

    /// Sample a value (simple implementation without external RNG)
    pub fn sample(&self, random_value: f64) -> f64 {
        match &self.distribution {
            Distribution::Uniform => self.min + random_value * (self.max - self.min),
            Distribution::Normal { mean, std_dev } => {
                // Box-Muller approximation (using single random value)
                let z = (random_value - 0.5) * 6.0; // Rough approximation
                mean + z * std_dev
            }
            Distribution::Exponential { lambda } => -random_value.ln() / lambda,
            Distribution::Poisson { lambda } => {
                // Approximation for Poisson
                (random_value * lambda * 2.0).round()
            }
        }
    }
}

/// Statistical distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    /// Uniform distribution
    Uniform,
    /// Normal (Gaussian) distribution
    Normal {
        /// Mean
        mean: f64,
        /// Standard deviation
        std_dev: f64,
    },
    /// Exponential distribution
    Exponential {
        /// Rate parameter
        lambda: f64,
    },
    /// Poisson distribution
    Poisson {
        /// Rate parameter
        lambda: f64,
    },
}

/// Simulation output configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulationOutput {
    /// Output directory
    pub directory: Option<PathBuf>,
    /// Save individual iteration results
    pub save_iterations: bool,
    /// Generate summary report
    pub generate_summary: bool,
}

// =============================================================================
// K.3 Monte Carlo Results
// =============================================================================

/// Monte Carlo simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    /// Number of iterations completed
    pub iterations: u32,
    /// Latency distribution (p95 across iterations)
    pub latency_distribution: LatencyDistribution,
    /// SLA probability results
    pub sla_probabilities: Vec<SlaProbability>,
    /// Sensitivity analysis
    pub sensitivity_analysis: Vec<SensitivityFactor>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl MonteCarloResult {
    /// Create new result
    pub fn new(iterations: u32) -> Self {
        Self {
            iterations,
            latency_distribution: LatencyDistribution::default(),
            sla_probabilities: Vec::new(),
            sensitivity_analysis: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Add SLA probability
    pub fn add_sla(&mut self, sla: SlaProbability) {
        self.sla_probabilities.push(sla);
    }

    /// Add sensitivity factor
    pub fn add_sensitivity(&mut self, factor: SensitivityFactor) {
        self.sensitivity_analysis.push(factor);
    }

    /// Add recommendation
    pub fn add_recommendation(&mut self, rec: &str) {
        self.recommendations.push(rec.to_string());
    }
}

/// Latency distribution from Monte Carlo
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyDistribution {
    /// Mean latency
    pub mean_ms: f64,
    /// Standard deviation
    pub std_dev_ms: f64,
    /// 95% confidence interval lower bound
    pub ci_lower_ms: f64,
    /// 95% confidence interval upper bound
    pub ci_upper_ms: f64,
    /// Histogram buckets (for visualization)
    pub histogram: Vec<(f64, u32)>, // (latency_ms, count)
}

impl LatencyDistribution {
    /// Create from samples
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // 95% CI = mean ± 1.96 * std_dev / sqrt(n)
        let margin = 1.96 * std_dev / n.sqrt();

        // Build histogram (10 buckets)
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bucket_size = (max - min) / 10.0;

        let mut buckets = vec![0u32; 10];
        for &sample in samples {
            let idx = ((sample - min) / bucket_size) as usize;
            let idx = idx.min(9);
            buckets[idx] += 1;
        }

        let histogram: Vec<(f64, u32)> = buckets
            .iter()
            .enumerate()
            .map(|(i, &count)| (min + (i as f64 + 0.5) * bucket_size, count))
            .collect();

        Self {
            mean_ms: mean,
            std_dev_ms: std_dev,
            ci_lower_ms: mean - margin,
            ci_upper_ms: mean + margin,
            histogram,
        }
    }
}

/// SLA probability result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaProbability {
    /// SLA target description
    pub target: String,
    /// Probability of meeting target (0.0 - 1.0)
    pub probability: f64,
    /// Risk level
    pub risk: RiskLevel,
}

impl SlaProbability {
    /// Create new SLA probability
    pub fn new(target: &str, probability: f64) -> Self {
        let risk = if probability >= 0.95 {
            RiskLevel::Minimal
        } else if probability >= 0.80 {
            RiskLevel::Low
        } else if probability >= 0.50 {
            RiskLevel::Medium
        } else {
            RiskLevel::High
        };

        Self {
            target: target.to_string(),
            probability,
            risk,
        }
    }
}

/// Risk level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskLevel {
    /// < 5% chance of missing
    Minimal,
    /// 5-20% chance of missing
    Low,
    /// 20-50% chance of missing
    Medium,
    /// > 50% chance of missing
    High,
}

impl RiskLevel {
    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Minimal => "MINIMAL",
            Self::Low => "LOW",
            Self::Medium => "MEDIUM",
            Self::High => "HIGH",
        }
    }

    /// Get bar representation
    pub fn bar(&self) -> &'static str {
        match self {
            Self::Minimal => "█",
            Self::Low => "████",
            Self::Medium => "██████████",
            Self::High => "███████████████",
        }
    }
}

/// Sensitivity factor from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityFactor {
    /// Parameter name
    pub parameter: String,
    /// Correlation coefficient with target metric
    pub correlation: f64,
    /// Impact level
    pub impact: ImpactLevel,
}

impl SensitivityFactor {
    /// Create new factor
    pub fn new(parameter: &str, correlation: f64) -> Self {
        let impact = if correlation.abs() >= 0.7 {
            ImpactLevel::High
        } else if correlation.abs() >= 0.4 {
            ImpactLevel::Medium
        } else {
            ImpactLevel::Low
        };

        Self {
            parameter: parameter.to_string(),
            correlation,
            impact,
        }
    }
}

/// Impact level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImpactLevel {
    /// Low impact
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
}

impl ImpactLevel {
    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "LOW",
            Self::Medium => "MEDIUM",
            Self::High => "HIGH",
        }
    }

    /// Get bar representation
    pub fn bar(&self) -> &'static str {
        match self {
            Self::Low => "███",
            Self::Medium => "████████████",
            Self::High => "████████████████████",
        }
    }
}

// =============================================================================
// K.4 Chaos Engineering Results
// =============================================================================

/// Chaos experiment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosResult {
    /// Experiment name
    pub experiment_name: String,
    /// Injections applied
    pub injections: Vec<FailureInjection>,
    /// System behavior observations
    pub observations: Vec<ChaosObservation>,
    /// Did system degrade gracefully?
    pub graceful_degradation: bool,
    /// Recovery time in milliseconds
    pub recovery_time_ms: Option<u64>,
}

impl ChaosResult {
    /// Create new result
    pub fn new(name: &str) -> Self {
        Self {
            experiment_name: name.to_string(),
            injections: Vec::new(),
            observations: Vec::new(),
            graceful_degradation: true,
            recovery_time_ms: None,
        }
    }

    /// Add observation
    pub fn add_observation(&mut self, obs: ChaosObservation) {
        // Check if any observation indicates non-graceful degradation
        if matches!(obs.severity, ObservationSeverity::Critical) {
            self.graceful_degradation = false;
        }
        self.observations.push(obs);
    }
}

/// Observation during chaos experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosObservation {
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Component affected
    pub component: String,
    /// Observation description
    pub description: String,
    /// Severity
    pub severity: ObservationSeverity,
}

impl ChaosObservation {
    /// Create new observation
    pub fn new(
        timestamp_ms: u64,
        component: &str,
        description: &str,
        severity: ObservationSeverity,
    ) -> Self {
        Self {
            timestamp_ms,
            component: component.to_string(),
            description: description.to_string(),
            severity,
        }
    }
}

/// Observation severity
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObservationSeverity {
    /// Informational
    Info,
    /// Warning - degraded but functional
    Warning,
    /// Error - partial failure
    Error,
    /// Critical - complete failure
    Critical,
}

impl ObservationSeverity {
    /// Get symbol
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Info => "ℹ",
            Self::Warning => "⚠",
            Self::Error => "✗",
            Self::Critical => "💀",
        }
    }
}

// =============================================================================
// Rendering
// =============================================================================

/// Render Monte Carlo results as TUI
pub fn render_monte_carlo_report(result: &MonteCarloResult) -> String {
    let mut out = String::new();

    out.push_str("MONTE CARLO SIMULATION\n");
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    out.push_str(&format!("Iterations: {}\n\n", result.iterations));

    // Latency distribution
    out.push_str("LATENCY DISTRIBUTION (p95 across iterations)\n");
    out.push_str("┌─────────────────────────────────────────────────────────────────────────┐\n");

    // Histogram visualization
    if !result.latency_distribution.histogram.is_empty() {
        let max_count = result
            .latency_distribution
            .histogram
            .iter()
            .map(|(_, c)| *c)
            .max()
            .unwrap_or(1);
        for (latency, count) in &result.latency_distribution.histogram {
            let bar_len = (*count as f64 / max_count as f64 * 30.0) as usize;
            let bar: String = "█".repeat(bar_len.max(1));
            out.push_str(&format!(
                "│ {:>6.0}ms │ {:30} │ {:>4}\n",
                latency, bar, count
            ));
        }
    }

    out.push_str(&format!(
        "│                                                                         │\n"
    ));
    out.push_str(&format!(
        "│  Mean: {:.0}ms    StdDev: {:.0}ms    95%% CI: [{:.0}ms, {:.0}ms]         │\n",
        result.latency_distribution.mean_ms,
        result.latency_distribution.std_dev_ms,
        result.latency_distribution.ci_lower_ms,
        result.latency_distribution.ci_upper_ms
    ));
    out.push_str("└─────────────────────────────────────────────────────────────────────────┘\n\n");

    // SLA probabilities
    if !result.sla_probabilities.is_empty() {
        out.push_str("FAILURE PROBABILITY\n");
        out.push_str(
            "┌─────────────────────────┬────────────────────────┬─────────────────────────┐\n",
        );
        out.push_str(
            "│ SLA Target              │ Probability of Meeting │ Risk Level              │\n",
        );
        out.push_str(
            "├─────────────────────────┼────────────────────────┼─────────────────────────┤\n",
        );

        for sla in &result.sla_probabilities {
            out.push_str(&format!(
                "│ {:<23} │ {:>20.1}% │ {:15} {:>7} │\n",
                truncate(&sla.target, 23),
                sla.probability * 100.0,
                sla.risk.bar(),
                sla.risk.as_str()
            ));
        }
        out.push_str(
            "└─────────────────────────┴────────────────────────┴─────────────────────────┘\n\n",
        );
    }

    // Sensitivity analysis
    if !result.sensitivity_analysis.is_empty() {
        out.push_str("SENSITIVITY ANALYSIS\n");
        out.push_str(
            "┌─────────────────────────┬──────────────────────┬───────────────────────────┐\n",
        );
        out.push_str(
            "│ Parameter               │ Correlation with p95 │ Impact                    │\n",
        );
        out.push_str(
            "├─────────────────────────┼──────────────────────┼───────────────────────────┤\n",
        );

        for factor in &result.sensitivity_analysis {
            out.push_str(&format!(
                "│ {:<23} │ r = {:>16.2} │ {:20} {:>5} │\n",
                truncate(&factor.parameter, 23),
                factor.correlation,
                factor.impact.bar(),
                factor.impact.as_str()
            ));
        }
        out.push_str(
            "└─────────────────────────┴──────────────────────┴───────────────────────────┘\n\n",
        );
    }

    // Recommendations
    if !result.recommendations.is_empty() {
        out.push_str("RECOMMENDATIONS\n");
        for (i, rec) in result.recommendations.iter().enumerate() {
            out.push_str(&format!("{}. {}\n", i + 1, rec));
        }
    }

    out
}

/// Render chaos result as TUI
pub fn render_chaos_report(result: &ChaosResult) -> String {
    let mut out = String::new();

    out.push_str(&format!("CHAOS EXPERIMENT: {}\n", result.experiment_name));
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    // Injections
    out.push_str("INJECTIONS APPLIED\n");
    for injection in &result.injections {
        out.push_str(&format!(
            "  • {} on '{}' (probability: {:.0}%)\n",
            injection.injection_type.name(),
            injection.target,
            injection.probability * 100.0
        ));
    }
    out.push_str("\n");

    // Observations
    out.push_str("OBSERVATIONS\n");
    out.push_str("┌──────────┬─────────────┬─────────────────────────────────────────────────┐\n");
    out.push_str("│ Time     │ Component   │ Observation                                     │\n");
    out.push_str("├──────────┼─────────────┼─────────────────────────────────────────────────┤\n");

    for obs in &result.observations {
        out.push_str(&format!(
            "│ {:>6}ms │ {:<11} │ {} {:<45} │\n",
            obs.timestamp_ms,
            truncate(&obs.component, 11),
            obs.severity.symbol(),
            truncate(&obs.description, 45)
        ));
    }
    out.push_str(
        "└──────────┴─────────────┴─────────────────────────────────────────────────┘\n\n",
    );

    // Verdict
    let verdict = if result.graceful_degradation {
        "✓ System degraded gracefully"
    } else {
        "✗ System did NOT degrade gracefully"
    };
    out.push_str(&format!("VERDICT: {}\n", verdict));

    if let Some(recovery) = result.recovery_time_ms {
        out.push_str(&format!("Recovery Time: {}ms\n", recovery));
    }

    out
}

/// Render as JSON
pub fn render_monte_carlo_json(result: &MonteCarloResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
}

/// Truncate string
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_injection() {
        let latency = FailureInjection::latency("network", 0.1, 100);
        assert_eq!(latency.injection_type, InjectionType::Latency);
        assert_eq!(latency.probability, 0.1);
        assert_eq!(latency.duration_ms, Some(100));

        let packet_loss = FailureInjection::packet_loss("network", 0.05);
        assert_eq!(packet_loss.injection_type, InjectionType::PacketLoss);
    }

    #[test]
    fn test_injection_type() {
        assert_eq!(InjectionType::Latency.name(), "latency");
        assert_eq!(InjectionType::PacketLoss.name(), "packet_loss");
    }

    #[test]
    fn test_simulation_config() {
        let config = SimulationConfig::deterministic(PathBuf::from("session.simular"));
        assert!(matches!(config.mode, SimulationMode::DeterministicReplay));

        let mc_config = SimulationConfig::monte_carlo(PathBuf::from("session.simular"), 1000);
        assert!(matches!(
            mc_config.mode,
            SimulationMode::MonteCarlo {
                iterations: 1000,
                ..
            }
        ));
    }

    #[test]
    fn test_parameter_variation_uniform() {
        let var = ParameterVariation::uniform(10.0, 20.0);
        let sample = var.sample(0.5);
        assert!((10.0..=20.0).contains(&sample));
    }

    #[test]
    fn test_parameter_variation_normal() {
        let var = ParameterVariation::normal(100.0, 10.0);
        let sample = var.sample(0.5);
        // Should be close to mean when random = 0.5
        assert!((sample - 100.0).abs() < 50.0);
    }

    #[test]
    fn test_latency_distribution() {
        let samples = vec![100.0, 110.0, 120.0, 130.0, 140.0];
        let dist = LatencyDistribution::from_samples(&samples);

        assert!((dist.mean_ms - 120.0).abs() < 0.001);
        assert!(dist.std_dev_ms > 0.0);
    }

    #[test]
    fn test_sla_probability() {
        let sla = SlaProbability::new("p95 < 200ms", 0.95);
        assert_eq!(sla.risk, RiskLevel::Minimal);

        let sla_high = SlaProbability::new("p95 < 100ms", 0.30);
        assert_eq!(sla_high.risk, RiskLevel::High);
    }

    #[test]
    fn test_risk_level() {
        assert_eq!(RiskLevel::Minimal.as_str(), "MINIMAL");
        assert!(RiskLevel::High.bar().len() > RiskLevel::Minimal.bar().len());
    }

    #[test]
    fn test_sensitivity_factor() {
        let high = SensitivityFactor::new("network_latency", 0.85);
        assert_eq!(high.impact, ImpactLevel::High);

        let low = SensitivityFactor::new("cache_size", 0.15);
        assert_eq!(low.impact, ImpactLevel::Low);
    }

    #[test]
    fn test_monte_carlo_result() {
        let mut result = MonteCarloResult::new(1000);
        result.add_sla(SlaProbability::new("p95 < 200ms", 0.89));
        result.add_sensitivity(SensitivityFactor::new("network", 0.82));
        result.add_recommendation("Use CDN for WASM assets");

        assert_eq!(result.iterations, 1000);
        assert_eq!(result.sla_probabilities.len(), 1);
        assert_eq!(result.sensitivity_analysis.len(), 1);
        assert_eq!(result.recommendations.len(), 1);
    }

    #[test]
    fn test_chaos_result() {
        let mut result = ChaosResult::new("Network Partition");
        result
            .injections
            .push(FailureInjection::packet_loss("network", 0.5));
        result.add_observation(ChaosObservation::new(
            1000,
            "frontend",
            "Requests timing out",
            ObservationSeverity::Warning,
        ));

        assert!(result.graceful_degradation);

        result.add_observation(ChaosObservation::new(
            2000,
            "backend",
            "Complete failure",
            ObservationSeverity::Critical,
        ));
        assert!(!result.graceful_degradation);
    }

    #[test]
    fn test_observation_severity() {
        assert_eq!(ObservationSeverity::Info.symbol(), "ℹ");
        assert_eq!(ObservationSeverity::Critical.symbol(), "💀");
    }

    #[test]
    fn test_render_monte_carlo_report() {
        let mut result = MonteCarloResult::new(100);
        result.latency_distribution = LatencyDistribution::from_samples(&[100.0, 200.0, 150.0]);
        result.add_sla(SlaProbability::new("p95 < 300ms", 0.95));

        let report = render_monte_carlo_report(&result);
        assert!(report.contains("MONTE CARLO"));
        assert!(report.contains("100"));
    }

    #[test]
    fn test_render_chaos_report() {
        let mut result = ChaosResult::new("Test Chaos");
        result
            .injections
            .push(FailureInjection::latency("api", 0.1, 50));

        let report = render_chaos_report(&result);
        assert!(report.contains("Test Chaos"));
        assert!(report.contains("INJECTIONS"));
    }

    #[test]
    fn test_failure_injection_error() {
        let injection = FailureInjection::error("database", 0.3);
        assert_eq!(injection.injection_type, InjectionType::Error);
        assert_eq!(injection.probability, 0.3);
        assert_eq!(injection.target, "database");
        assert!(injection.duration_ms.is_none());
    }

    #[test]
    fn test_injection_type_all_names() {
        // Cover all InjectionType::name() branches
        assert_eq!(InjectionType::Latency.name(), "latency");
        assert_eq!(InjectionType::PacketLoss.name(), "packet_loss");
        assert_eq!(InjectionType::Error.name(), "error");
        assert_eq!(InjectionType::Timeout.name(), "timeout");
        assert_eq!(InjectionType::CpuThrottle.name(), "cpu_throttle");
        assert_eq!(InjectionType::MemoryPressure.name(), "memory_pressure");
    }

    #[test]
    fn test_simulation_config_with_variation() {
        let config = SimulationConfig::deterministic(PathBuf::from("session.json")).with_variation(
            "latency",
            ParameterVariation {
                distribution: Distribution::Normal {
                    mean: 50.0,
                    std_dev: 10.0,
                },
                min: 10.0,
                max: 100.0,
                base: 50.0,
            },
        );

        assert!(config.parameter_variations.contains_key("latency"));
        let variation = config.parameter_variations.get("latency").unwrap();
        assert_eq!(variation.min, 10.0);
        assert_eq!(variation.max, 100.0);
        assert_eq!(variation.base, 50.0);
    }

    #[test]
    fn test_parameter_variation_exponential() {
        let var = ParameterVariation {
            distribution: Distribution::Exponential { lambda: 1.0 },
            min: 0.0,
            max: 10.0,
            base: 1.0,
        };
        // ln(0.5) / 1.0 ≈ 0.693
        let sample = var.sample(0.5);
        assert!(sample > 0.0);
    }

    #[test]
    fn test_parameter_variation_poisson() {
        let var = ParameterVariation {
            distribution: Distribution::Poisson { lambda: 5.0 },
            min: 0.0,
            max: 20.0,
            base: 5.0,
        };
        let sample = var.sample(0.5);
        // Should be around 5.0 (random=0.5 * lambda=5.0 * 2.0 = 5.0)
        assert!((sample - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_sla_probability_low_risk() {
        let sla = SlaProbability::new("p95 < 150ms", 0.85);
        assert_eq!(sla.risk, RiskLevel::Low);
    }

    #[test]
    fn test_sla_probability_medium_risk() {
        let sla = SlaProbability::new("p95 < 100ms", 0.60);
        assert_eq!(sla.risk, RiskLevel::Medium);
    }

    #[test]
    fn test_risk_level_all_bars() {
        assert_eq!(RiskLevel::Minimal.bar(), "█");
        assert_eq!(RiskLevel::Low.bar(), "████");
        assert_eq!(RiskLevel::Medium.bar(), "██████████");
        assert_eq!(RiskLevel::High.bar(), "███████████████");
    }

    #[test]
    fn test_risk_level_all_strings() {
        assert_eq!(RiskLevel::Minimal.as_str(), "MINIMAL");
        assert_eq!(RiskLevel::Low.as_str(), "LOW");
        assert_eq!(RiskLevel::Medium.as_str(), "MEDIUM");
        assert_eq!(RiskLevel::High.as_str(), "HIGH");
    }

    #[test]
    fn test_impact_level_all_strings() {
        assert_eq!(ImpactLevel::Low.as_str(), "LOW");
        assert_eq!(ImpactLevel::Medium.as_str(), "MEDIUM");
        assert_eq!(ImpactLevel::High.as_str(), "HIGH");
    }

    #[test]
    fn test_impact_level_all_bars() {
        assert_eq!(ImpactLevel::Low.bar(), "███");
        assert_eq!(ImpactLevel::Medium.bar(), "████████████");
        assert_eq!(ImpactLevel::High.bar(), "████████████████████");
    }

    #[test]
    fn test_sensitivity_factor_medium_impact() {
        let factor = SensitivityFactor::new("cache_hit_rate", 0.55);
        assert_eq!(factor.impact, ImpactLevel::Medium);
    }

    #[test]
    fn test_observation_severity_all_symbols() {
        assert_eq!(ObservationSeverity::Info.symbol(), "ℹ");
        assert_eq!(ObservationSeverity::Warning.symbol(), "⚠");
        assert_eq!(ObservationSeverity::Error.symbol(), "✗");
        assert_eq!(ObservationSeverity::Critical.symbol(), "💀");
    }

    #[test]
    fn test_latency_distribution_empty() {
        let dist = LatencyDistribution::from_samples(&[]);
        assert_eq!(dist.mean_ms, 0.0);
        assert_eq!(dist.std_dev_ms, 0.0);
    }

    #[test]
    fn test_chaos_result_with_recovery_time() {
        let mut result = ChaosResult::new("Recovery Test");
        result.recovery_time_ms = Some(5000);
        assert_eq!(result.recovery_time_ms, Some(5000));
    }

    #[test]
    fn test_chaos_observation_fields() {
        let obs = ChaosObservation::new(
            1500,
            "api_gateway",
            "High latency detected",
            ObservationSeverity::Error,
        );
        assert_eq!(obs.timestamp_ms, 1500);
        assert_eq!(obs.component, "api_gateway");
        assert_eq!(obs.description, "High latency detected");
        assert_eq!(obs.severity, ObservationSeverity::Error);
    }

    #[test]
    fn test_monte_carlo_result_full_workflow() {
        let mut result = MonteCarloResult::new(500);

        // Add distribution
        result.latency_distribution = LatencyDistribution::from_samples(&[
            50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
        ]);

        // Add multiple SLAs with different risk levels
        result.add_sla(SlaProbability::new("p95 < 200ms", 0.98));
        result.add_sla(SlaProbability::new("p99 < 300ms", 0.75));
        result.add_sla(SlaProbability::new("p99.9 < 500ms", 0.45));

        // Add multiple sensitivity factors
        result.add_sensitivity(SensitivityFactor::new("network_latency", 0.88));
        result.add_sensitivity(SensitivityFactor::new("db_connection_pool", 0.52));
        result.add_sensitivity(SensitivityFactor::new("cache_ttl", 0.15));

        // Add multiple recommendations
        result.add_recommendation("Consider using connection pooling");
        result.add_recommendation("Increase cache TTL for static assets");

        assert_eq!(result.iterations, 500);
        assert_eq!(result.sla_probabilities.len(), 3);
        assert_eq!(result.sensitivity_analysis.len(), 3);
        assert_eq!(result.recommendations.len(), 2);
    }

    #[test]
    fn test_simulation_mode_parameterized() {
        let mut multipliers = HashMap::new();
        multipliers.insert("latency".to_string(), 1.5);
        multipliers.insert("throughput".to_string(), 0.8);

        let mode = SimulationMode::Parameterized { multipliers };
        if let SimulationMode::Parameterized { multipliers: m } = mode {
            assert_eq!(m.len(), 2);
            assert_eq!(m.get("latency"), Some(&1.5));
        } else {
            panic!("Expected Parameterized mode");
        }
    }

    #[test]
    fn test_simulation_mode_chaos() {
        let injections = vec![
            FailureInjection::latency("network", 0.1, 100),
            FailureInjection::packet_loss("network", 0.05),
        ];

        let mode = SimulationMode::Chaos { injections };
        if let SimulationMode::Chaos { injections: i } = mode {
            assert_eq!(i.len(), 2);
        } else {
            panic!("Expected Chaos mode");
        }
    }

    #[test]
    fn test_simulation_output_default() {
        let output = SimulationOutput::default();
        assert!(output.directory.is_none());
        assert!(!output.save_iterations);
        assert!(!output.generate_summary);
    }
}
