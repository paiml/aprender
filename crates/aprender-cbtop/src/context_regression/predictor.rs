//! Context-aware regression predictor with adaptive thresholds.

use std::collections::HashMap;

use super::{
    BaselineEntry, RegressionCheck, RegressionThreshold, SystemContext, Trend,
    DEFAULT_COLD_START_MARGIN, DEFAULT_STALENESS_SEC, MIN_SAMPLES_FOR_CONTEXT,
};

/// Context-aware regression predictor
#[derive(Debug)]
pub struct ContextRegressionPredictor {
    /// Historical baselines by metric
    baselines: HashMap<String, Vec<BaselineEntry>>,
    /// Maximum history size
    max_history: usize,
    /// Cold start margin (%)
    cold_start_margin: f64,
    /// Minimum learned margin (%)
    min_margin: f64,
    /// Context staleness threshold (seconds)
    staleness_sec: u64,
    /// Temperature variance factor (% per 10°C)
    temp_factor: f64,
    /// Memory variance factor (% per 10% utilization)
    memory_factor: f64,
    /// Frequency variance factor (% per 10% reduction)
    freq_factor: f64,
    /// Cache cold penalty (%)
    cache_cold_penalty: f64,
}

impl Default for ContextRegressionPredictor {
    fn default() -> Self {
        Self {
            baselines: HashMap::new(),
            max_history: 100,
            cold_start_margin: DEFAULT_COLD_START_MARGIN,
            min_margin: 3.0,
            staleness_sec: DEFAULT_STALENESS_SEC,
            temp_factor: 2.0,
            memory_factor: 1.0,
            freq_factor: 5.0,
            cache_cold_penalty: 10.0,
        }
    }
}

/// Compute the mean of values extracted from baseline entries via a field accessor.
fn entries_mean(entries: &[BaselineEntry], field: impl Fn(&BaselineEntry) -> f64) -> f64 {
    let sum: f64 = entries.iter().map(&field).sum();
    sum / entries.len() as f64
}

/// Compute a context adjustment: scale the difference between current and historical
/// average by a divisor and factor, optionally clamping negative values to zero.
fn context_adjustment(
    current: f64,
    historical_avg: f64,
    scale: f64,
    factor: f64,
    clamp_positive: bool,
) -> f64 {
    let diff = current - historical_avg;
    let scaled = diff * scale;
    if clamp_positive {
        scaled.max(0.0) * factor
    } else {
        scaled * factor
    }
}

/// Simple linear regression result.
struct LinearFit {
    slope: f64,
    r_squared: f64,
}

/// Compute simple linear regression (slope and R-squared) for (x, y) pairs
/// extracted from baseline entries. Returns `None` if the data is degenerate.
fn linear_regression(
    entries: &[BaselineEntry],
    x_fn: impl Fn(&BaselineEntry) -> f64,
    y_fn: impl Fn(&BaselineEntry) -> f64,
) -> Option<LinearFit> {
    let n = entries.len() as f64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xy = 0.0_f64;
    let mut sum_xx = 0.0_f64;

    for entry in entries {
        let x = x_fn(entry);
        let y = y_fn(entry);
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    // Compute R-squared
    let mean_y = sum_y / n;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for entry in entries {
        let x = x_fn(entry);
        let y_pred = slope * x + intercept;
        let y = y_fn(entry);
        ss_res += (y - y_pred).powi(2);
        ss_tot += (y - mean_y).powi(2);
    }
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Some(LinearFit { slope, r_squared })
}

/// Compute the coefficient of variation (%) of a slice of f64 values.
/// Returns a default CV if the mean is near zero.
fn coefficient_of_variation(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 5.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    if mean.abs() <= 1e-10 {
        return 5.0;
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    (variance.sqrt() / mean.abs()) * 100.0
}

impl ContextRegressionPredictor {
    /// Create new predictor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set cold start margin
    pub fn with_cold_start_margin(mut self, margin: f64) -> Self {
        self.cold_start_margin = margin.max(5.0);
        self
    }

    /// Set minimum margin
    pub fn with_min_margin(mut self, margin: f64) -> Self {
        self.min_margin = margin.max(1.0);
        self
    }

    /// Set temperature factor
    pub fn with_temp_factor(mut self, factor: f64) -> Self {
        self.temp_factor = factor.max(0.0);
        self
    }

    /// Set staleness threshold
    pub fn with_staleness(mut self, sec: u64) -> Self {
        self.staleness_sec = sec;
        self
    }

    /// Add baseline entry
    pub fn add_baseline(&mut self, metric: &str, value: f64, context: SystemContext) {
        let entry = BaselineEntry::new(metric, value, context);

        self.baselines
            .entry(metric.to_string())
            .or_default()
            .push(entry);

        // Trim old entries
        if let Some(entries) = self.baselines.get_mut(metric) {
            while entries.len() > self.max_history {
                entries.remove(0);
            }
        }
    }

    /// Get baseline count for metric
    pub fn baseline_count(&self, metric: &str) -> usize {
        self.baselines.get(metric).map(|e| e.len()).unwrap_or(0)
    }

    /// Check if sufficient history
    pub fn has_sufficient_history(&self, metric: &str) -> bool {
        self.baseline_count(metric) >= MIN_SAMPLES_FOR_CONTEXT
    }

    /// Compute context-aware threshold
    pub fn compute_threshold(
        &self,
        metric: &str,
        current_context: &SystemContext,
    ) -> RegressionThreshold {
        let sample_count = self.baseline_count(metric);

        // Cold start: use conservative margin
        if sample_count < MIN_SAMPLES_FOR_CONTEXT {
            return RegressionThreshold {
                base_percent: self.cold_start_margin,
                temp_adjustment: 0.0,
                memory_adjustment: 0.0,
                freq_adjustment: 0.0,
                cache_adjustment: 0.0,
                final_percent: self.cold_start_margin,
                confidence: 0.1,
                sample_count,
            };
        }

        let entries = self
            .baselines
            .get(metric)
            .expect("metric should exist in baselines after sufficient history check");

        // Compute base threshold from historical variance (coefficient of variation)
        let values: Vec<f64> = entries.iter().map(|e| e.value).collect();
        let cv = coefficient_of_variation(&values);

        // Base threshold: 2*CV + min margin
        let base_percent = (cv * 2.0).max(self.min_margin);

        // Temperature adjustment: warmer = more variance expected
        let avg_temp = entries_mean(entries, |e| e.context.cpu_temp_c);
        let temp_adjustment = context_adjustment(
            current_context.cpu_temp_c,
            avg_temp,
            1.0 / 10.0,
            self.temp_factor,
            false,
        );

        // Memory adjustment: higher pressure = more variance
        let avg_mem = entries_mean(entries, |e| e.context.memory_percent);
        let memory_adjustment = context_adjustment(
            current_context.memory_percent,
            avg_mem,
            1.0 / 10.0,
            self.memory_factor,
            true,
        );

        // Frequency adjustment: lower frequency = expect slower
        // Direction is reversed (historical - current), so we swap current/historical
        let avg_freq_util = entries_mean(entries, |e| e.context.freq_utilization());
        let freq_adjustment = context_adjustment(
            avg_freq_util,
            current_context.freq_utilization(),
            10.0,
            self.freq_factor,
            true,
        );

        // Cache adjustment: cold cache = expect slower
        let cache_adjustment = if !current_context.cache_warm {
            self.cache_cold_penalty
        } else {
            0.0
        };

        // Final threshold
        let final_percent = (base_percent
            + temp_adjustment
            + memory_adjustment
            + freq_adjustment
            + cache_adjustment)
            .max(self.min_margin);

        // Confidence increases with more samples
        let confidence = (sample_count as f64 / 50.0).min(1.0);

        RegressionThreshold {
            base_percent,
            temp_adjustment,
            memory_adjustment,
            freq_adjustment,
            cache_adjustment,
            final_percent,
            confidence,
            sample_count,
        }
    }

    /// Detect trend in baselines
    pub fn detect_trend(&self, metric: &str) -> Option<Trend> {
        let entries = self.baselines.get(metric)?;
        if entries.len() < MIN_SAMPLES_FOR_CONTEXT {
            return None;
        }

        let base_time = entries.first()?.context.timestamp;
        let fit = linear_regression(
            entries,
            |e| (e.context.timestamp - base_time) as f64 / 86400.0,
            |e| e.value,
        )?;

        let direction = if fit.slope > 0.1 {
            "increasing"
        } else if fit.slope < -0.1 {
            "decreasing"
        } else {
            "stable"
        };

        Some(Trend {
            slope_per_day: fit.slope,
            r_squared: fit.r_squared,
            direction,
        })
    }

    /// Check for regression
    pub fn check_regression(
        &self,
        metric: &str,
        current_value: f64,
        context: &SystemContext,
    ) -> RegressionCheck {
        let threshold = self.compute_threshold(metric, context);

        let entries = self.baselines.get(metric);
        let baseline_mean = entries
            .map(|e| entries_mean(e, |x| x.value))
            .unwrap_or(current_value);

        let percent_change = if baseline_mean.abs() > 1e-10 {
            ((current_value - baseline_mean) / baseline_mean) * 100.0
        } else {
            0.0
        };

        let is_regression = threshold.is_regression(percent_change);
        let trend = self.detect_trend(metric);

        RegressionCheck {
            metric: metric.to_string(),
            current_value,
            baseline_mean,
            percent_change,
            threshold,
            is_regression,
            trend,
        }
    }

    /// Clear history for metric
    pub fn clear(&mut self, metric: &str) {
        self.baselines.remove(metric);
    }

    /// Clear all history
    pub fn clear_all(&mut self) {
        self.baselines.clear();
    }

    /// Export baselines to JSON
    pub fn export_json(&self, metric: &str) -> Option<String> {
        let entries = self.baselines.get(metric)?;
        let entries_json: Vec<String> = entries
            .iter()
            .map(|e| {
                format!(
                    r#"{{"value":{},"context":{}}}"#,
                    e.value,
                    e.context.to_json()
                )
            })
            .collect();
        Some(format!(
            r#"{{"metric":"{}","entries":[{}]}}"#,
            metric,
            entries_json.join(",")
        ))
    }
}
