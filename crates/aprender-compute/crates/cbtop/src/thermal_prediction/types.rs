//! Thermal prediction data types and assessment structures.

/// Thermal sample with timestamp
#[derive(Debug, Clone, Copy)]
pub struct ThermalSample {
    /// Temperature in Celsius
    pub temperature_c: f64,
    /// Timestamp in seconds (relative to start)
    pub timestamp_sec: f64,
    /// Optional latency measurement at this time
    pub latency_us: Option<f64>,
}

impl ThermalSample {
    /// Create new sample
    pub fn new(temperature_c: f64, timestamp_sec: f64) -> Self {
        Self { temperature_c, timestamp_sec, latency_us: None }
    }

    /// Create sample with latency
    pub fn with_latency(temperature_c: f64, timestamp_sec: f64, latency_us: f64) -> Self {
        Self { temperature_c, timestamp_sec, latency_us: Some(latency_us) }
    }
}

/// Thermal trend prediction result
#[derive(Debug, Clone)]
pub struct ThermalPrediction {
    /// Predicted temperature at horizon
    pub predicted_temp_c: f64,
    /// Prediction horizon in seconds
    pub horizon_sec: f64,
    /// Trend slope (degrees C/second)
    pub trend_slope: f64,
    /// Confidence (0.0 - 1.0) based on R-squared fit
    pub confidence: f64,
    /// Number of samples used
    pub sample_count: usize,
}

impl ThermalPrediction {
    /// Check if temperature is predicted to exceed threshold
    pub fn will_throttle(&self, threshold_c: f64) -> bool {
        self.predicted_temp_c >= threshold_c
    }

    /// Get time until throttle (if trending up)
    pub fn time_to_throttle(&self, current_temp: f64, threshold_c: f64) -> Option<f64> {
        if self.trend_slope <= 0.0 {
            return None; // Not trending up
        }

        let delta = threshold_c - current_temp;
        if delta <= 0.0 {
            return Some(0.0); // Already at or above threshold
        }

        Some(delta / self.trend_slope)
    }
}

/// Throttle risk assessment
#[derive(Debug, Clone)]
pub struct ThrottleRisk {
    /// Risk probability (0.0 - 1.0)
    pub probability: f64,
    /// Current temperature
    pub current_temp_c: f64,
    /// Throttle threshold
    pub threshold_c: f64,
    /// Temperature margin to threshold
    pub margin_c: f64,
    /// Risk category
    pub category: RiskCategory,
}

impl ThrottleRisk {
    /// Create from temperature and trend
    pub fn assess(current_temp: f64, threshold: f64, trend_slope: f64, horizon_sec: f64) -> Self {
        let margin = threshold - current_temp;
        let _predicted_temp = current_temp + trend_slope * horizon_sec;

        // Calculate risk based on:
        // 1. How close we are to threshold
        // 2. Whether we're trending toward it
        // 3. How fast we're approaching it

        let proximity_risk =
            if margin <= 0.0 { 1.0 } else { 1.0 - (margin / threshold).clamp(0.0, 1.0) };

        let trend_risk = if trend_slope > 0.0 {
            // Approaching threshold
            let time_to_threshold = margin / trend_slope;
            if time_to_threshold <= horizon_sec {
                1.0
            } else {
                (horizon_sec / time_to_threshold).clamp(0.0, 1.0)
            }
        } else {
            0.0 // Cooling down
        };

        // Combined risk (weighted average)
        let probability = (0.4 * proximity_risk + 0.6 * trend_risk).clamp(0.0, 1.0);

        let category = RiskCategory::from_probability(probability);

        Self {
            probability,
            current_temp_c: current_temp,
            threshold_c: threshold,
            margin_c: margin,
            category,
        }
    }
}

/// Risk category classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCategory {
    /// Low risk (< 25%)
    Low,
    /// Moderate risk (25% - 50%)
    Moderate,
    /// High risk (50% - 75%)
    High,
    /// Critical risk (> 75%)
    Critical,
}

impl RiskCategory {
    /// Create from probability
    pub fn from_probability(prob: f64) -> Self {
        if prob < 0.25 {
            Self::Low
        } else if prob < 0.50 {
            Self::Moderate
        } else if prob < 0.75 {
            Self::High
        } else {
            Self::Critical
        }
    }

    /// Get category name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Moderate => "moderate",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// Cooldown recommendation
#[derive(Debug, Clone)]
pub struct CooldownRecommendation {
    /// Recommended cooldown time in seconds
    pub duration_sec: f64,
    /// Target temperature after cooldown
    pub target_temp_c: f64,
    /// Current temperature
    pub current_temp_c: f64,
    /// Cooling rate used (degrees C/second)
    pub cooling_rate: f64,
}

impl CooldownRecommendation {
    /// Calculate cooldown time needed
    pub fn calculate(current_temp: f64, target_temp: f64, cooling_rate: f64) -> Self {
        let temp_delta = current_temp - target_temp;
        let duration =
            if temp_delta > 0.0 && cooling_rate > 0.0 { temp_delta / cooling_rate } else { 0.0 };

        Self {
            duration_sec: duration.max(0.0),
            target_temp_c: target_temp,
            current_temp_c: current_temp,
            cooling_rate,
        }
    }

    /// Check if cooldown is needed
    pub fn is_needed(&self) -> bool {
        self.duration_sec > 0.0
    }
}

/// Thermal-latency correlation result
#[derive(Debug, Clone)]
pub struct ThermalCorrelation {
    /// Pearson correlation coefficient (-1.0 to 1.0)
    pub pearson_r: f64,
    /// Number of paired samples
    pub sample_count: usize,
    /// Is correlation significant?
    pub is_significant: bool,
    /// Estimated latency increase per degree (us/degrees C)
    pub latency_per_degree: f64,
}

impl ThermalCorrelation {
    /// Check if there's positive correlation (hotter = slower)
    pub fn has_thermal_impact(&self) -> bool {
        self.pearson_r > 0.3 && self.is_significant
    }
}

/// Thermal variance contribution
#[derive(Debug, Clone)]
pub struct ThermalVariance {
    /// Percentage of total variance explained by thermal
    pub contribution_percent: f64,
    /// Temperature range during measurement
    pub temp_range_c: f64,
    /// Average temperature
    pub avg_temp_c: f64,
}
