//! Core types for adaptive ML thresholds.

/// Result type for ML threshold operations
pub type MlThresholdResult<T> = Result<T, MlThresholdError>;

/// Errors in ML threshold operations
#[derive(Debug, Clone, PartialEq)]
pub enum MlThresholdError {
    /// Insufficient training data
    InsufficientData { have: usize, need: usize },
    /// Workload not recognized
    UnknownWorkload { name: String },
    /// Model not trained
    ModelNotTrained,
    /// Feature extraction failed
    FeatureExtractionFailed { reason: String },
    /// Confidence too low
    LowConfidence { confidence: f64, threshold: f64 },
    /// Drift detected, re-calibration needed
    DriftDetected { metric: String, drift_score: f64 },
}

impl std::fmt::Display for MlThresholdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData { have, need } => {
                write!(f, "Insufficient data: have {}, need {}", have, need)
            }
            Self::UnknownWorkload { name } => write!(f, "Unknown workload: {}", name),
            Self::ModelNotTrained => write!(f, "Model not trained"),
            Self::FeatureExtractionFailed { reason } => {
                write!(f, "Feature extraction failed: {}", reason)
            }
            Self::LowConfidence {
                confidence,
                threshold,
            } => {
                write!(f, "Low confidence {} < {}", confidence, threshold)
            }
            Self::DriftDetected {
                metric,
                drift_score,
            } => {
                write!(f, "Drift detected in {}: score {:.2}", metric, drift_score)
            }
        }
    }
}

impl std::error::Error for MlThresholdError {}

/// Workload type for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadClass {
    /// FFN/MLP operations
    Ffn,
    /// Matrix multiplication
    Matmul,
    /// Attention operations
    Attention,
    /// Quantization/dequantization
    Quantize,
    /// Memory-bound operations
    MemoryBound,
    /// Compute-bound operations
    ComputeBound,
    /// Unknown workload
    Unknown,
}

impl WorkloadClass {
    /// Get default CV threshold for this workload class
    pub fn default_cv_threshold(&self) -> f64 {
        match self {
            Self::Ffn => 18.0,         // FFN naturally has higher variance
            Self::Matmul => 10.0,      // Matmul is very consistent
            Self::Attention => 15.0,   // Attention has moderate variance
            Self::Quantize => 12.0,    // Quantize is fairly consistent
            Self::MemoryBound => 20.0, // Memory-bound is highly variable
            Self::ComputeBound => 8.0, // Compute-bound is very consistent
            Self::Unknown => 15.0,     // Conservative default
        }
    }

    /// Get name as string
    pub fn name(&self) -> &'static str {
        match self {
            Self::Ffn => "FFN",
            Self::Matmul => "Matmul",
            Self::Attention => "Attention",
            Self::Quantize => "Quantize",
            Self::MemoryBound => "MemoryBound",
            Self::ComputeBound => "ComputeBound",
            Self::Unknown => "Unknown",
        }
    }

    /// Parse from name string (used by import_state)
    pub(super) fn from_name(name: &str) -> Option<Self> {
        match name {
            "FFN" => Some(WorkloadClass::Ffn),
            "Matmul" => Some(WorkloadClass::Matmul),
            "Attention" => Some(WorkloadClass::Attention),
            "Quantize" => Some(WorkloadClass::Quantize),
            "MemoryBound" => Some(WorkloadClass::MemoryBound),
            "ComputeBound" => Some(WorkloadClass::ComputeBound),
            _ => None,
        }
    }
}

/// Features extracted from a time series
#[derive(Debug, Clone)]
pub struct TimeSeriesFeatures {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Coefficient of variation (CV)
    pub cv: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Autocorrelation at lag 1
    pub autocorr_lag1: f64,
    /// Trend slope (linear fit)
    pub trend_slope: f64,
    /// Number of samples
    pub sample_count: usize,
}

impl TimeSeriesFeatures {
    /// Extract features from sample values
    pub fn extract(values: &[f64]) -> Option<Self> {
        if values.len() < 10 {
            return None;
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let cv = if mean.abs() > 1e-10 {
            (std_dev / mean) * 100.0
        } else {
            0.0
        };

        // Skewness
        let skewness = if std_dev > 1e-10 {
            let m3 = values
                .iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / n;
            m3
        } else {
            0.0
        };

        // Kurtosis
        let kurtosis = if std_dev > 1e-10 {
            let m4 = values
                .iter()
                .map(|x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / n;
            m4 - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        // Autocorrelation at lag 1
        let autocorr_lag1 = if values.len() > 1 && std_dev > 1e-10 {
            let mut sum = 0.0;
            for i in 0..values.len() - 1 {
                sum += (values[i] - mean) * (values[i + 1] - mean);
            }
            sum / ((values.len() - 1) as f64 * variance)
        } else {
            0.0
        };

        // Trend slope (simple linear regression)
        let trend_slope = {
            let x_mean = (values.len() as f64 - 1.0) / 2.0;
            let mut num = 0.0;
            let mut den = 0.0;
            for (i, &y) in values.iter().enumerate() {
                let x = i as f64;
                num += (x - x_mean) * (y - mean);
                den += (x - x_mean).powi(2);
            }
            if den > 1e-10 {
                num / den
            } else {
                0.0
            }
        };

        Some(Self {
            mean,
            std_dev,
            cv,
            skewness,
            kurtosis,
            autocorr_lag1,
            trend_slope,
            sample_count: values.len(),
        })
    }

    /// Convert features to vector for model input
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.cv,
            self.skewness,
            self.kurtosis,
            self.autocorr_lag1,
            self.trend_slope,
        ]
    }
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Whether this is an anomaly
    pub is_anomaly: bool,
    /// Anomaly score (higher = more anomalous)
    pub score: f64,
    /// Threshold used
    pub threshold: f64,
    /// Confidence in the result
    pub confidence: f64,
    /// Workload class
    pub workload_class: WorkloadClass,
    /// Reason for classification
    pub reason: String,
}

/// Classification precision/recall metrics
#[derive(Debug, Clone, Default)]
pub struct ClassificationMetrics {
    /// True positives
    pub true_positives: usize,
    /// False positives
    pub false_positives: usize,
    /// True negatives
    pub true_negatives: usize,
    /// False negatives
    pub false_negatives: usize,
}

impl ClassificationMetrics {
    /// Calculate precision
    pub fn precision(&self) -> f64 {
        let total = self.true_positives + self.false_positives;
        if total == 0 {
            0.0
        } else {
            self.true_positives as f64 / total as f64
        }
    }

    /// Calculate recall
    pub fn recall(&self) -> f64 {
        let total = self.true_positives + self.false_negatives;
        if total == 0 {
            0.0
        } else {
            self.true_positives as f64 / total as f64
        }
    }

    /// Calculate F1 score
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Calculate false positive rate
    pub fn false_positive_rate(&self) -> f64 {
        let total = self.false_positives + self.true_negatives;
        if total == 0 {
            0.0
        } else {
            self.false_positives as f64 / total as f64
        }
    }
}
