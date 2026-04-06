//! Variance Source Analysis Module (PMAT-027)
//!
//! Analyzes sources of performance variance to identify and mitigate
//! benchmark instability per PERF-003 (CV 5-8% vs target <5%).
//!
//! # Motivation
//!
//! F605 (Results reproducible) is PARTIAL with CV 5-8%. Need systematic
//! variance attribution to identify and mitigate sources.
//!
//! # Components
//!
//! | Component | Detection Method | Mitigation |
//! |-----------|-----------------|------------|
//! | Frequency Variance | std_dev(CPU MHz samples) | Pin frequency |
//! | Thermal Drift | Correlation(temp, latency) | Cooldown periods |
//! | Cache Noise | First-run vs warm-run delta | Warmup iterations |
//! | System Noise | Residual after above | Isolation/shielding |

/// Source of performance variance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceSource {
    /// CPU frequency scaling (turbo boost variance)
    FrequencyScaling,
    /// Thermal throttling effects
    ThermalThrottling,
    /// Cache state variance (cold vs warm)
    CacheState,
    /// Background system activity
    SystemNoise,
    /// Unknown or unattributed
    Unknown,
}

impl VarianceSource {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            VarianceSource::FrequencyScaling => "CPU frequency scaling",
            VarianceSource::ThermalThrottling => "thermal throttling",
            VarianceSource::CacheState => "cache state variance",
            VarianceSource::SystemNoise => "system noise",
            VarianceSource::Unknown => "unknown",
        }
    }

    /// Get mitigation recommendation
    pub fn mitigation(&self) -> &'static str {
        match self {
            VarianceSource::FrequencyScaling => {
                "Pin CPU frequency with: cpupower frequency-set -g performance"
            }
            VarianceSource::ThermalThrottling => {
                "Add cooldown periods between runs or improve cooling"
            }
            VarianceSource::CacheState => "Increase warmup iterations before measurement",
            VarianceSource::SystemNoise => {
                "Run with CPU isolation (isolcpus) or reduce background tasks"
            }
            VarianceSource::Unknown => "Profile with renacer for deeper analysis",
        }
    }
}

/// Variance analysis result
#[derive(Debug, Clone)]
pub struct VarianceAnalysis {
    /// Total coefficient of variation (%)
    pub total_cv_percent: f64,
    /// Estimated frequency scaling contribution (%)
    pub frequency_contribution: f64,
    /// Estimated thermal contribution (%)
    pub thermal_contribution: f64,
    /// Estimated cache state contribution (%)
    pub cache_contribution: f64,
    /// Residual unexplained noise (%)
    pub residual_noise: f64,
    /// Dominant source of variance
    pub dominant_source: VarianceSource,
    /// Mitigation recommendations
    pub recommendations: Vec<String>,
    /// Whether variance budget is met (CV < 5%)
    pub budget_met: bool,
    /// Sample statistics
    pub sample_count: usize,
    /// Warmup effect ratio (cold/warm performance)
    pub warmup_effect: f64,
    /// Trend coefficient (positive = increasing latency)
    pub trend_coefficient: f64,
}

/// Input data for variance analysis
#[derive(Debug, Clone)]
pub struct VarianceInput {
    /// Latency samples (µs)
    pub latencies: Vec<f64>,
    /// CPU frequency samples (MHz), if available
    pub frequencies: Option<Vec<f64>>,
    /// Temperature samples (°C), if available
    pub temperatures: Option<Vec<f64>>,
    /// Number of warmup iterations
    pub warmup_count: usize,
}

impl VarianceAnalysis {
    /// Analyze variance sources from input data
    pub fn analyze(input: &VarianceInput) -> Option<Self> {
        if input.latencies.is_empty() {
            return None;
        }

        let n = input.latencies.len();
        let mean = input.latencies.iter().sum::<f64>() / n as f64;

        // Calculate total CV
        let variance = if n > 1 {
            input.latencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let total_cv_percent = if mean > 0.0 { (std_dev / mean) * 100.0 } else { 0.0 };

        // Estimate frequency contribution
        let frequency_contribution = if let Some(ref freqs) = input.frequencies {
            estimate_frequency_contribution(freqs, &input.latencies)
        } else {
            0.0
        };

        // Estimate thermal contribution
        let thermal_contribution = if let Some(ref temps) = input.temperatures {
            estimate_thermal_contribution(temps, &input.latencies)
        } else {
            0.0
        };

        // Estimate cache state contribution
        let (cache_contribution, warmup_effect) =
            estimate_cache_contribution(&input.latencies, input.warmup_count);

        // Calculate residual
        let attributed = frequency_contribution + thermal_contribution + cache_contribution;
        let residual_noise = (total_cv_percent - attributed).max(0.0);

        // Identify dominant source
        let dominant_source = identify_dominant_source(
            frequency_contribution,
            thermal_contribution,
            cache_contribution,
            residual_noise,
        );

        // Generate recommendations
        let recommendations = generate_recommendations(
            total_cv_percent,
            frequency_contribution,
            thermal_contribution,
            cache_contribution,
            residual_noise,
        );

        // Calculate trend
        let trend_coefficient = calculate_trend(&input.latencies);

        let budget_met = total_cv_percent < 5.0;

        Some(Self {
            total_cv_percent,
            frequency_contribution,
            thermal_contribution,
            cache_contribution,
            residual_noise,
            dominant_source,
            recommendations,
            budget_met,
            sample_count: n,
            warmup_effect,
            trend_coefficient,
        })
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "CV={:.1}% (freq={:.1}% therm={:.1}% cache={:.1}% noise={:.1}%) dominant={}",
            self.total_cv_percent,
            self.frequency_contribution,
            self.thermal_contribution,
            self.cache_contribution,
            self.residual_noise,
            self.dominant_source.name()
        )
    }

    /// Check if any single source dominates (>50% of variance)
    pub fn has_dominant_source(&self) -> bool {
        let max = self
            .frequency_contribution
            .max(self.thermal_contribution)
            .max(self.cache_contribution)
            .max(self.residual_noise);
        max > self.total_cv_percent * 0.5
    }
}

/// Estimate frequency scaling contribution to variance
fn estimate_frequency_contribution(frequencies: &[f64], latencies: &[f64]) -> f64 {
    if frequencies.len() < 2 || latencies.len() < 2 {
        return 0.0;
    }

    // Calculate correlation between frequency and latency
    let correlation = calculate_correlation(frequencies, latencies);

    // Frequency variance
    let freq_mean = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
    let freq_variance = frequencies.iter().map(|f| (f - freq_mean).powi(2)).sum::<f64>()
        / (frequencies.len() - 1) as f64;
    let freq_cv = if freq_mean > 0.0 { freq_variance.sqrt() / freq_mean * 100.0 } else { 0.0 };

    // Contribution is correlation × frequency CV (simplified model)
    correlation.abs() * freq_cv
}

/// Estimate thermal throttling contribution
fn estimate_thermal_contribution(temperatures: &[f64], latencies: &[f64]) -> f64 {
    if temperatures.len() < 2 || latencies.len() < 2 {
        return 0.0;
    }

    // Calculate correlation between temperature and latency
    let correlation = calculate_correlation(temperatures, latencies);

    // Temperature variance
    let temp_mean = temperatures.iter().sum::<f64>() / temperatures.len() as f64;
    let temp_variance = temperatures.iter().map(|t| (t - temp_mean).powi(2)).sum::<f64>()
        / (temperatures.len() - 1) as f64;
    let temp_cv = if temp_mean > 0.0 { temp_variance.sqrt() / temp_mean * 100.0 } else { 0.0 };

    // Positive correlation threshold: higher temperature correlates with higher latency
    if correlation > 0.3 {
        correlation * temp_cv
    } else {
        0.0
    }
}

/// Estimate cache state contribution
fn estimate_cache_contribution(latencies: &[f64], warmup_count: usize) -> (f64, f64) {
    if latencies.len() <= warmup_count || warmup_count == 0 {
        return (0.0, 1.0);
    }

    // Split into cold (early) and warm (later) samples
    let cold_samples: Vec<f64> = latencies.iter().take(warmup_count).cloned().collect();
    let warm_samples: Vec<f64> = latencies.iter().skip(warmup_count).cloned().collect();

    if cold_samples.is_empty() || warm_samples.is_empty() {
        return (0.0, 1.0);
    }

    let cold_mean = cold_samples.iter().sum::<f64>() / cold_samples.len() as f64;
    let warm_mean = warm_samples.iter().sum::<f64>() / warm_samples.len() as f64;

    // Warmup effect ratio
    let warmup_effect = if warm_mean > 0.0 { cold_mean / warm_mean } else { 1.0 };

    // Cache contribution is the difference between cold and warm CV
    let cold_cv = calculate_cv(&cold_samples);
    let warm_cv = calculate_cv(&warm_samples);

    let cache_contribution = (cold_cv - warm_cv).max(0.0);

    (cache_contribution, warmup_effect)
}

/// Calculate coefficient of variation
fn calculate_cv(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }

    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;

    (variance.sqrt() / mean) * 100.0
}

/// Calculate Pearson correlation coefficient
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let x_mean = x.iter().take(n).sum::<f64>() / n as f64;
    let y_mean = y.iter().take(n).sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        numerator += dx * dy;
        x_var += dx * dx;
        y_var += dy * dy;
    }

    let denominator = (x_var * y_var).sqrt();
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Calculate trend coefficient (slope of linear regression)
fn calculate_trend(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let n = samples.len() as f64;
    let x_mean = (n - 1.0) / 2.0; // Mean of 0, 1, 2, ..., n-1
    let y_mean = samples.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in samples.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Identify the dominant source of variance
fn identify_dominant_source(freq: f64, thermal: f64, cache: f64, residual: f64) -> VarianceSource {
    let max = freq.max(thermal).max(cache).max(residual);

    if max < 0.5 {
        VarianceSource::Unknown
    } else if max == freq {
        VarianceSource::FrequencyScaling
    } else if max == thermal {
        VarianceSource::ThermalThrottling
    } else if max == cache {
        VarianceSource::CacheState
    } else {
        VarianceSource::SystemNoise
    }
}

/// Generate mitigation recommendations based on variance sources
fn generate_recommendations(
    total_cv: f64,
    freq: f64,
    thermal: f64,
    cache: f64,
    residual: f64,
) -> Vec<String> {
    let mut recs = Vec::new();

    if total_cv >= 5.0 {
        recs.push(format!("CV {:.1}% exceeds 5% target. Mitigation needed.", total_cv));
    }

    if freq > 1.0 {
        recs.push(format!(
            "Frequency variance ({:.1}%): {}",
            freq,
            VarianceSource::FrequencyScaling.mitigation()
        ));
    }

    if thermal > 1.0 {
        recs.push(format!(
            "Thermal variance ({:.1}%): {}",
            thermal,
            VarianceSource::ThermalThrottling.mitigation()
        ));
    }

    if cache > 1.0 {
        recs.push(format!(
            "Cache variance ({:.1}%): {}",
            cache,
            VarianceSource::CacheState.mitigation()
        ));
    }

    if residual > 2.0 {
        recs.push(format!(
            "Residual noise ({:.1}%): {}",
            residual,
            VarianceSource::SystemNoise.mitigation()
        ));
    }

    if recs.is_empty() {
        recs.push("Variance within acceptable limits.".to_string());
    }

    recs
}

#[cfg(test)]
mod tests;
