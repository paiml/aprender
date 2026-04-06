//! Types for the performance prediction model.

/// Minimum samples required for fitting
pub const MIN_SAMPLES_FOR_FIT: usize = 5;

/// Performance data point
#[derive(Debug, Clone, Copy)]
pub struct DataPoint {
    /// Problem size (elements)
    pub size: usize,
    /// Performance metric (GFLOP/s, throughput, etc.)
    pub performance: f64,
    /// Latency (microseconds)
    pub latency_us: f64,
}

impl DataPoint {
    /// Create new data point
    pub fn new(size: usize, performance: f64, latency_us: f64) -> Self {
        Self {
            size,
            performance,
            latency_us,
        }
    }
}

/// Model type for curve fitting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// Linear: y = a*x + b
    Linear,
    /// Polynomial: y = a*x^2 + b*x + c
    Polynomial,
    /// Exponential decay: y = a * exp(-b*x) + c
    ExponentialDecay,
    /// Logarithmic: y = a * log(x) + b
    Logarithmic,
    /// Roofline: y = min(peak, bandwidth * intensity)
    Roofline,
}

impl ModelType {
    /// Get model name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Polynomial => "polynomial",
            Self::ExponentialDecay => "exponential_decay",
            Self::Logarithmic => "logarithmic",
            Self::Roofline => "roofline",
        }
    }
}

/// Fitted model parameters
#[derive(Debug, Clone)]
pub struct FittedModel {
    /// Model type
    pub model_type: ModelType,
    /// Model coefficients
    pub coefficients: Vec<f64>,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Residual sum of squares
    pub rss: f64,
    /// Number of data points used
    pub sample_count: usize,
}

impl FittedModel {
    /// Predict performance at given size
    pub fn predict(&self, size: usize) -> f64 {
        let x = size as f64;
        match self.model_type {
            ModelType::Linear => {
                // y = a*x + b
                let a = self.coefficients.first().copied().unwrap_or(0.0);
                let b = self.coefficients.get(1).copied().unwrap_or(0.0);
                a * x + b
            }
            ModelType::Polynomial => {
                // y = a*x^2 + b*x + c
                let a = self.coefficients.first().copied().unwrap_or(0.0);
                let b = self.coefficients.get(1).copied().unwrap_or(0.0);
                let c = self.coefficients.get(2).copied().unwrap_or(0.0);
                a * x * x + b * x + c
            }
            ModelType::ExponentialDecay => {
                // y = a * exp(-b*x) + c
                let a = self.coefficients.first().copied().unwrap_or(0.0);
                let b = self.coefficients.get(1).copied().unwrap_or(0.0);
                let c = self.coefficients.get(2).copied().unwrap_or(0.0);
                a * (-b * x).exp() + c
            }
            ModelType::Logarithmic => {
                // y = a * log(x) + b
                let a = self.coefficients.first().copied().unwrap_or(0.0);
                let b = self.coefficients.get(1).copied().unwrap_or(0.0);
                if x > 0.0 {
                    a * x.ln() + b
                } else {
                    b
                }
            }
            ModelType::Roofline => {
                // y = min(peak, bandwidth * intensity)
                // Simplified: use polynomial approximation
                let peak = self.coefficients.first().copied().unwrap_or(100.0);
                let _knee = self.coefficients.get(1).copied().unwrap_or(1000.0);
                let slope = self.coefficients.get(2).copied().unwrap_or(1.0);

                let linear = slope * x;
                linear.min(peak).max(0.0)
            }
        }
    }

    /// Check if model has good fit (R-squared > 0.9)
    pub fn is_good_fit(&self) -> bool {
        self.r_squared > 0.9
    }
}

/// Prediction result with confidence bounds
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted size
    pub size: usize,
    /// Predicted performance
    pub predicted: f64,
    /// Lower bound (confidence interval)
    pub lower_bound: f64,
    /// Upper bound (confidence interval)
    pub upper_bound: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Model used for prediction
    pub model_type: ModelType,
    /// Is this extrapolation (outside training range)?
    pub is_extrapolation: bool,
}

impl Prediction {
    /// Check if prediction is reasonable
    pub fn is_reasonable(&self) -> bool {
        self.predicted > 0.0
            && self.lower_bound >= 0.0
            && self.upper_bound >= self.predicted
            && !self.predicted.is_nan()
            && !self.predicted.is_infinite()
    }

    /// Get prediction range width
    pub fn range_width(&self) -> f64 {
        self.upper_bound - self.lower_bound
    }

    /// Get prediction uncertainty (%)
    pub fn uncertainty_percent(&self) -> f64 {
        if self.predicted > 0.0 {
            (self.range_width() / self.predicted) * 100.0 / 2.0
        } else {
            100.0
        }
    }
}
