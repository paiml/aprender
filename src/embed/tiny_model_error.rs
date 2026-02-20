
/// Errors specific to tiny model operations
#[derive(Debug, Clone)]
pub enum TinyModelError {
    /// Model has no parameters
    EmptyModel,
    /// Invalid coefficient value
    InvalidCoefficient { index: usize, value: f32 },
    /// Invalid variance (must be positive)
    InvalidVariance {
        class: usize,
        feature: usize,
        value: f32,
    },
    /// Shape mismatch in model components
    ShapeMismatch { message: String },
    /// Invalid k for KNN
    InvalidK { k: u32, n_samples: usize },
    /// Feature dimension mismatch
    FeatureMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for TinyModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModel => write!(f, "Model has no parameters"),
            Self::InvalidCoefficient { index, value } => {
                write!(f, "Invalid coefficient at index {index}: {value}")
            }
            Self::InvalidVariance {
                class,
                feature,
                value,
            } => {
                write!(
                    f,
                    "Invalid variance for class {class}, feature {feature}: {value}"
                )
            }
            Self::ShapeMismatch { message } => write!(f, "Shape mismatch: {message}"),
            Self::InvalidK { k, n_samples } => {
                write!(f, "Invalid k={k} for {n_samples} samples")
            }
            Self::FeatureMismatch { expected, got } => {
                write!(f, "Expected {expected} features, got {got}")
            }
        }
    }
}

impl std::error::Error for TinyModelError {}
