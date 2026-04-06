//! Simple Linear Autoencoder for Anomaly Detection (Sprint 23)
//!
//! Implements a 3-layer linear autoencoder for unsupervised anomaly detection
//! using reconstruction error as the anomaly metric.
//!
//! # Architecture
//!
//! Input Layer (n features) → Hidden Layer (compressed) → Output Layer (n features)
//!
//! # Algorithm
//!
//! Training: Gradient descent minimizing Mean Squared Error (MSE)
//! Inference: High reconstruction error = anomaly
//!
//! # References
//!
//! Goodfellow, I., Bengio, Y., & Courville, A. (2016).
//! Deep Learning. MIT Press. Chapter 14: Autoencoders.

use std::collections::HashMap;

/// Simple linear autoencoder with one hidden layer
#[derive(Debug, Clone)]
pub struct Autoencoder {
    /// Encoder weights: input → hidden
    encoder_weights: Vec<Vec<f64>>,
    /// Encoder bias
    encoder_bias: Vec<f64>,
    /// Decoder weights: hidden → output
    decoder_weights: Vec<Vec<f64>>,
    /// Decoder bias
    decoder_bias: Vec<f64>,
    /// Input dimension
    input_dim: usize,
    /// Hidden layer dimension
    hidden_dim: usize,
}

impl Autoencoder {
    /// Create a new autoencoder with random initialization
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Xavier initialization for better convergence
        let encoder_scale = (2.0 / (input_dim as f64 + hidden_dim as f64)).sqrt();
        let decoder_scale = (2.0 / (hidden_dim as f64 + input_dim as f64)).sqrt();

        // Initialize encoder weights (input_dim x hidden_dim)
        let encoder_weights: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| {
                (0..hidden_dim).map(|_| rng.gen_range(-encoder_scale..encoder_scale)).collect()
            })
            .collect();

        let encoder_bias: Vec<f64> = (0..hidden_dim).map(|_| 0.0).collect();

        // Initialize decoder weights (hidden_dim x input_dim)
        let decoder_weights: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| (0..input_dim).map(|_| rng.gen_range(-decoder_scale..decoder_scale)).collect())
            .collect();

        let decoder_bias: Vec<f64> = (0..input_dim).map(|_| 0.0).collect();

        Autoencoder {
            encoder_weights,
            encoder_bias,
            decoder_weights,
            decoder_bias,
            input_dim,
            hidden_dim,
        }
    }

    /// Encode input to hidden representation
    fn encode(&self, input: &[f64]) -> Vec<f64> {
        let mut hidden = self.encoder_bias.clone();

        for (i, h) in hidden.iter_mut().enumerate() {
            for (j, &x) in input.iter().enumerate() {
                *h += self.encoder_weights[j][i] * x;
            }
        }

        // ReLU activation
        hidden.iter().map(|&h| h.max(0.0)).collect()
    }

    /// Decode hidden representation to output
    fn decode(&self, hidden: &[f64]) -> Vec<f64> {
        let mut output = self.decoder_bias.clone();

        for (i, o) in output.iter_mut().enumerate() {
            for (j, &h) in hidden.iter().enumerate() {
                *o += self.decoder_weights[j][i] * h;
            }
        }

        output
    }

    /// Forward pass: input → hidden → output
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let hidden = self.encode(input);
        self.decode(&hidden)
    }

    /// Calculate reconstruction error (MSE)
    pub fn reconstruction_error(&self, input: &[f64]) -> f64 {
        let output = self.forward(input);
        let mut mse = 0.0;

        for (i, &x) in input.iter().enumerate() {
            let diff = x - output[i];
            mse += diff * diff;
        }

        mse / input.len() as f64
    }

    /// Train the autoencoder using gradient descent
    pub fn train(&mut self, samples: &[Vec<f64>], epochs: usize, learning_rate: f64) {
        for _epoch in 0..epochs {
            for sample in samples {
                // Forward pass
                let hidden = self.encode(sample);
                let output = self.decode(&hidden);

                // Calculate output error
                let output_error: Vec<f64> = sample
                    .iter()
                    .zip(output.iter())
                    .map(|(&target, &pred)| target - pred)
                    .collect();

                // Backpropagate through decoder
                let mut hidden_error = vec![0.0; self.hidden_dim];
                for (i, h_err) in hidden_error.iter_mut().enumerate() {
                    for (j, &o_err) in output_error.iter().enumerate() {
                        *h_err += o_err * self.decoder_weights[i][j];
                    }
                }

                // Apply ReLU derivative (gradient is 0 if hidden <= 0)
                for (i, &h) in hidden.iter().enumerate() {
                    if h <= 0.0 {
                        hidden_error[i] = 0.0;
                    }
                }

                // Update decoder weights and bias
                for (i, &h) in hidden.iter().enumerate().take(self.hidden_dim) {
                    for (j, &o_err) in output_error.iter().enumerate().take(self.input_dim) {
                        self.decoder_weights[i][j] += learning_rate * o_err * h;
                    }
                }
                for (j, &o_err) in output_error.iter().enumerate() {
                    self.decoder_bias[j] += learning_rate * o_err;
                }

                // Update encoder weights and bias
                for (i, &s) in sample.iter().enumerate().take(self.input_dim) {
                    for (j, &h_err) in hidden_error.iter().enumerate().take(self.hidden_dim) {
                        self.encoder_weights[i][j] += learning_rate * h_err * s;
                    }
                }
                for (j, &h_err) in hidden_error.iter().enumerate() {
                    self.encoder_bias[j] += learning_rate * h_err;
                }
            }
        }
    }

    /// Predict if sample is anomalous based on reconstruction error threshold
    pub fn predict(&self, sample: &[f64], threshold: f64) -> bool {
        self.reconstruction_error(sample) > threshold
    }
}

/// Anomaly detected by Autoencoder
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub syscall: String,
    pub reconstruction_error: f64,
    pub avg_duration_us: f64,
    pub call_count: u64,
    pub feature_contributions: Vec<(String, f64)>,
}

/// Result of Autoencoder analysis
#[derive(Debug, Clone)]
pub struct AutoencoderReport {
    pub anomalies: Vec<Anomaly>,
    pub total_samples: usize,
    pub threshold: f64,
    pub hidden_size: usize,
    pub epochs: usize,
}

/// Extract features from syscall statistics (reuse `isolation_forest` approach)
fn extract_features(syscall_data: &HashMap<String, (u64, u64)>) -> (Vec<String>, Vec<Vec<f64>>) {
    let mut syscall_names = Vec::new();
    let mut features = Vec::new();

    for (name, (count, total_time_ns)) in syscall_data {
        if *count == 0 {
            continue;
        }

        let total_time_us = *total_time_ns as f64 / 1000.0;
        let avg_time_us = total_time_us / *count as f64;

        syscall_names.push(name.clone());

        // Feature vector: [avg_duration, call_count, total_duration]
        features.push(vec![
            avg_time_us,
            (*count as f64).ln().max(0.0), // Log scale for count
            total_time_us.ln().max(0.0),   // Log scale for total time
        ]);
    }

    (syscall_names, features)
}

/// Normalize features to [0, 1] range for better training
fn normalize_features(features: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    if features.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let num_features = features[0].len();
    let mut min_vals = vec![f64::MAX; num_features];
    let mut max_vals = vec![f64::MIN; num_features];

    // Find min/max for each feature
    for sample in features {
        for (i, &val) in sample.iter().enumerate() {
            min_vals[i] = min_vals[i].min(val);
            max_vals[i] = max_vals[i].max(val);
        }
    }

    // Normalize
    let normalized: Vec<Vec<f64>> = features
        .iter()
        .map(|sample| {
            sample
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let range = max_vals[i] - min_vals[i];
                    if range < f64::EPSILON {
                        0.5 // All values are the same
                    } else {
                        (val - min_vals[i]) / range
                    }
                })
                .collect()
        })
        .collect();

    (normalized, min_vals, max_vals)
}

/// Calculate feature contributions for explainability (XAI)
fn calculate_feature_contributions(original: &[f64], reconstructed: &[f64]) -> Vec<(String, f64)> {
    let feature_names = ["avg_duration", "call_frequency", "total_duration"];

    let total_error: f64 =
        original.iter().zip(reconstructed.iter()).map(|(&o, &r)| (o - r).abs()).sum();

    feature_names
        .iter()
        .zip(original.iter().zip(reconstructed.iter()))
        .map(|(name, (&o, &r))| {
            let contribution =
                if total_error > 0.0 { ((o - r).abs() / total_error) * 100.0 } else { 0.0 };
            ((*name).to_string(), contribution)
        })
        .collect()
}

/// Analyze syscall data for anomalies using Autoencoder
pub fn analyze_anomalies(
    syscall_data: &HashMap<String, (u64, u64)>,
    hidden_size: usize,
    epochs: usize,
    threshold: f64,
    explain: bool,
) -> AutoencoderReport {
    // Extract features
    let (syscall_names, features) = extract_features(syscall_data);

    if features.len() < 5 {
        // Insufficient data for meaningful training
        return AutoencoderReport {
            anomalies: Vec::new(),
            total_samples: features.len(),
            threshold,
            hidden_size,
            epochs,
        };
    }

    let input_dim = features[0].len();

    // Normalize features
    let (normalized_features, _min_vals, _max_vals) = normalize_features(&features);

    // Train autoencoder
    let mut autoencoder = Autoencoder::new(input_dim, hidden_size);
    autoencoder.train(&normalized_features, epochs, 0.01);

    // Calculate baseline threshold if not provided
    let reconstruction_errors: Vec<f64> =
        normalized_features.iter().map(|f| autoencoder.reconstruction_error(f)).collect();

    let mean_error: f64 =
        reconstruction_errors.iter().sum::<f64>() / reconstruction_errors.len() as f64;
    let std_error: f64 = {
        let variance: f64 =
            reconstruction_errors.iter().map(|&e| (e - mean_error).powi(2)).sum::<f64>()
                / reconstruction_errors.len() as f64;
        variance.sqrt()
    };

    // Adaptive threshold: mean + threshold * std_dev
    let adaptive_threshold = mean_error + (threshold * std_error);

    // Detect anomalies
    let mut anomalies = Vec::new();

    for (name, feature_vec) in syscall_names.iter().zip(normalized_features.iter()) {
        let error = autoencoder.reconstruction_error(feature_vec);

        if error > adaptive_threshold {
            let (count, total_time_ns) = syscall_data[name];
            let avg_duration_us = total_time_ns as f64 / 1000.0 / count as f64;

            let feature_contributions = if explain {
                let reconstructed = autoencoder.forward(feature_vec);
                calculate_feature_contributions(feature_vec, &reconstructed)
            } else {
                Vec::new()
            };

            anomalies.push(Anomaly {
                syscall: name.clone(),
                reconstruction_error: error,
                avg_duration_us,
                call_count: count,
                feature_contributions,
            });
        }
    }

    // Sort by reconstruction error (highest first, handle NaN gracefully)
    anomalies.sort_by(|a, b| {
        b.reconstruction_error
            .partial_cmp(&a.reconstruction_error)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    AutoencoderReport {
        anomalies,
        total_samples: features.len(),
        threshold: adaptive_threshold,
        hidden_size,
        epochs,
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(Autoencoder: Send, Sync);
static_assertions::assert_impl_all!(AutoencoderReport: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoencoder_creation() {
        let ae = Autoencoder::new(3, 2);
        assert_eq!(ae.input_dim, 3);
        assert_eq!(ae.hidden_dim, 2);
        assert_eq!(ae.encoder_weights.len(), 3);
        assert_eq!(ae.encoder_weights[0].len(), 2);
    }

    #[test]
    fn test_forward_pass() {
        let ae = Autoencoder::new(3, 2);
        let input = vec![1.0, 2.0, 3.0];
        let output = ae.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_reconstruction_error() {
        let ae = Autoencoder::new(3, 2);
        let input = vec![1.0, 2.0, 3.0];
        let error = ae.reconstruction_error(&input);
        assert!(error >= 0.0); // Error should be non-negative
    }

    #[test]
    fn test_training_reduces_error() {
        let mut ae = Autoencoder::new(3, 2);
        let samples = vec![vec![1.0, 2.0, 3.0], vec![1.1, 2.1, 3.1], vec![0.9, 1.9, 2.9]];

        let initial_error = ae.reconstruction_error(&samples[0]);
        ae.train(&samples, 100, 0.01);
        let final_error = ae.reconstruction_error(&samples[0]);

        // Training should reduce reconstruction error
        assert!(
            final_error < initial_error,
            "Training should reduce error: {} -> {}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_anomaly_detection() {
        let mut ae = Autoencoder::new(3, 2);

        // Normal samples
        let normal_samples = vec![vec![1.0, 2.0, 3.0], vec![1.1, 2.1, 3.1], vec![0.9, 1.9, 2.9]];

        ae.train(&normal_samples, 100, 0.01);

        // Anomalous sample (very different)
        let anomaly = vec![10.0, 20.0, 30.0];

        let normal_error = ae.reconstruction_error(&normal_samples[0]);
        let anomaly_error = ae.reconstruction_error(&anomaly);

        // Anomaly should have higher reconstruction error
        assert!(
            anomaly_error > normal_error,
            "Anomaly error ({}) should be > normal error ({})",
            anomaly_error,
            normal_error
        );
    }

    #[test]
    fn test_feature_extraction() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (100, 1_000_000));
        data.insert("read".to_string(), (10, 10_000_000));

        let (names, features) = extract_features(&data);

        assert_eq!(names.len(), 2);
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].len(), 3);
    }

    #[test]
    fn test_feature_normalization() {
        let features = vec![vec![1.0, 10.0, 100.0], vec![2.0, 20.0, 200.0], vec![3.0, 30.0, 300.0]];

        let (normalized, _min_vals, _max_vals) = normalize_features(&features);

        assert_eq!(normalized.len(), 3);
        // First sample should be all zeros (minimum values)
        assert!((normalized[0][0] - 0.0).abs() < 1e-10);
        // Last sample should be all ones (maximum values)
        assert!((normalized[2][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyze_anomalies() {
        let mut data = HashMap::new();
        // Normal syscalls with similar patterns
        data.insert("write".to_string(), (100, 1_000_000));
        data.insert("read".to_string(), (100, 1_000_000));
        data.insert("open".to_string(), (90, 900_000));
        data.insert("close".to_string(), (95, 950_000));
        data.insert("stat".to_string(), (98, 980_000));
        data.insert("lseek".to_string(), (102, 1_020_000));
        // Anomaly - very slow (100x slower)
        data.insert("slow_syscall".to_string(), (10, 100_000_000));

        let report = analyze_anomalies(&data, 2, 100, 1.5, false);

        assert!(!report.anomalies.is_empty(), "Should detect at least one anomaly");
        assert_eq!(report.total_samples, 7);
    }

    #[test]
    fn test_insufficient_data() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (1, 1000));

        let report = analyze_anomalies(&data, 2, 50, 2.0, false);
        assert_eq!(report.anomalies.len(), 0);
        assert_eq!(report.total_samples, 1);
    }

    #[test]
    fn test_feature_contributions() {
        let original = vec![1.0, 2.0, 3.0];
        let reconstructed = vec![0.9, 2.1, 2.8];

        let contributions = calculate_feature_contributions(&original, &reconstructed);

        assert_eq!(contributions.len(), 3);
        // Sum should be ~100%
        let total: f64 = contributions.iter().map(|(_, v)| v).sum();
        assert!((total - 100.0).abs() < 0.1);
    }
}
