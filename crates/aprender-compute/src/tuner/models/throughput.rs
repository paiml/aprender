//! Throughput regressor model for ML tuner.

use serde::{Deserialize, Serialize};

use super::super::error::TunerError;
use super::super::features::TunerFeatures;
use super::ThroughputPrediction;

/// Simple linear regression model for throughput prediction.
///
/// Uses closed-form solution: w = (X^T X)^-1 X^T y. The `ml-tuner` showcase
/// (SHOWCASE-BRICK-001) feeds these features into `aprender::RandomForestRegressor`
/// from `examples/ml_tuner_demo.rs` (aprender is a dev-dependency — the SIMD foundation
/// must not depend on the ML layer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputRegressor {
    /// Model weights (one per feature + bias)
    pub(crate) weights: Vec<f32>,
    /// Feature importance scores
    pub(crate) feature_importance: Vec<(String, f32)>,
    /// Training sample count
    pub(crate) sample_count: usize,
    /// Mean absolute percentage error on validation
    pub(crate) mape: f32,
}

impl Default for ThroughputRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl ThroughputRegressor {
    /// Create a new regressor with default weights
    pub fn new() -> Self {
        // Initialize with heuristic-based weights
        // These encode domain knowledge from SHOWCASE-BRICK-001
        let mut weights = vec![0.0; TunerFeatures::DIM + 1]; // +1 for bias

        // Bias: baseline throughput ~200 tok/s normalized
        weights[0] = 0.4;

        // Batch size has largest positive impact (index 6)
        weights[7] = 0.3; // batch_size_norm

        // CUDA graphs help (index 8)
        weights[9] = 0.1; // cuda_graphs

        // GPU memory bandwidth matters (index 35)
        weights[36] = 0.15; // gpu_mem_bw_norm

        // GPU SM count matters (index 37)
        weights[38] = 0.1; // gpu_sm_norm

        // Larger models are slower (negative impact)
        weights[1] = -0.15; // model_params_b

        // Longer sequences slower for decode
        weights[8] = -0.05; // seq_len_log

        Self {
            weights,
            feature_importance: Self::default_feature_importance(),
            sample_count: 0,
            mape: 0.15, // 15% default MAPE
        }
    }

    fn default_feature_importance() -> Vec<(String, f32)> {
        vec![
            ("batch_size".into(), 0.25),
            ("gpu_mem_bw".into(), 0.20),
            ("model_params".into(), 0.15),
            ("cuda_graphs".into(), 0.10),
            ("gpu_sm_count".into(), 0.10),
            ("hidden_dim".into(), 0.08),
            ("quant_type".into(), 0.07),
            ("seq_len".into(), 0.05),
        ]
    }

    /// Train the model on labeled data
    pub fn train(&mut self, data: &[(TunerFeatures, f32)]) -> Result<(), TunerError> {
        if data.len() < 10 {
            return Err(TunerError::InsufficientData(data.len()));
        }

        // Simple gradient descent (in production: aprender's GBDT)
        let learning_rate = 0.01;
        let epochs = 100;

        for _ in 0..epochs {
            let mut gradients = vec![0.0; self.weights.len()];

            for (features, target) in data {
                let x = features.to_vector();
                let predicted = self.predict_raw(&x);
                let error = predicted - target;

                // Gradient for bias
                gradients[0] += error;

                // Gradient for features
                for (i, xi) in x.iter().enumerate() {
                    gradients[i + 1] += error * xi;
                }
            }

            // Update weights
            let n = data.len().max(1) as f32;
            for (i, g) in gradients.iter().enumerate() {
                self.weights[i] -= learning_rate * g / n;
            }
        }

        // Calculate MAPE on training data
        let mut total_ape = 0.0;
        for (features, target) in data {
            let predicted = self.predict_raw(&features.to_vector());
            total_ape += ((predicted - target) / target.max(1.0)).abs();
        }
        self.mape = total_ape / data.len().max(1) as f32;
        self.sample_count = data.len();

        Ok(())
    }

    pub(crate) fn predict_raw(&self, x: &[f32]) -> f32 {
        let mut result = self.weights[0]; // bias
        for (i, xi) in x.iter().enumerate() {
            if i + 1 < self.weights.len() {
                result += self.weights[i + 1] * xi;
            }
        }
        // Convert from normalized to tok/s (scale ~1000)
        (result * 1000.0).max(1.0)
    }

    /// Predict throughput for features using the linear model.
    pub fn predict(&self, features: &TunerFeatures) -> ThroughputPrediction {
        let x = features.to_vector();
        let raw_predicted_tps = self.predict_raw(&x);

        // v1.1.0: Roofline clamping - predictions must not exceed theoretical maximum
        let theoretical_max_tps = Self::compute_roofline_bound(features);
        let predicted_tps = raw_predicted_tps.min(theoretical_max_tps);

        // Confidence based on training MAPE and feature validity
        // Lower confidence if we hit the roofline cap
        let roofline_penalty = if raw_predicted_tps > theoretical_max_tps {
            0.9 // 10% confidence penalty for capped predictions
        } else {
            1.0
        };
        let confidence = (1.0 - self.mape).max(0.5) * roofline_penalty;

        ThroughputPrediction {
            predicted_tps,
            confidence,
            top_features: self.feature_importance.iter().take(5).cloned().collect(),
        }
    }

    /// Compute theoretical maximum throughput based on roofline model (v1.1.0)
    ///
    /// For memory-bound LLM inference (decode phase):
    /// max_tps = memory_bw_bytes_per_sec / bytes_per_token
    /// bytes_per_token = model_params x bytes_per_param / batch_size
    pub fn compute_roofline_bound(features: &TunerFeatures) -> f32 {
        // Denormalize model params: normalized = (log10(b) + 1) / 3
        // log10(b) = normalized * 3 - 1
        // b = 10^(normalized * 3 - 1)
        let model_params_b = 10.0_f32.powf(features.model_params_b * 3.0 - 1.0);

        // Get bytes per param from quant type one-hot encoding
        let bytes_per_param = Self::bytes_per_param_from_onehot(&features.quant_type_onehot);

        // Denormalize memory bandwidth: normalized = bw / 3000 GB/s
        let gpu_mem_bw_gbs = features.gpu_mem_bw_norm * 3000.0;

        // Denormalize batch size: normalized = batch_size / 64
        let batch_size = (features.batch_size_norm * 64.0).max(1.0);

        // Roofline calculation:
        // model_bytes = model_params_b * bytes_per_param * 1e9
        // bytes_per_token = model_bytes / batch_size
        // max_tps = (gpu_mem_bw_gbs * 1e9) / bytes_per_token
        //         = (gpu_mem_bw_gbs * 1e9 * batch_size) / (model_params_b * bytes_per_param * 1e9)
        //         = (gpu_mem_bw_gbs * batch_size) / (model_params_b * bytes_per_param)
        let theoretical_max = (gpu_mem_bw_gbs * batch_size) / (model_params_b * bytes_per_param);

        // Clamp to reasonable range (1 tok/s to 10000 tok/s)
        theoretical_max.clamp(1.0, 10000.0)
    }

    /// Extract bytes per param from quant type one-hot encoding
    pub fn bytes_per_param_from_onehot(onehot: &[f32; 8]) -> f32 {
        // One-hot indices map to QuantType variants
        // 0: Q4_0, 1: Q4_1, 2: Q4K, 3: Q5K, 4: Q6K, 5: Q8_0, 6: F16, 7: F32
        let bytes_per_param = [0.5625, 0.5625, 0.5625, 0.6875, 0.8125, 1.0, 2.0, 4.0];

        // Find the active index (max value in one-hot)
        let idx = onehot
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            // N-07 (Meyer DbC): default to Q4_0 (idx 0) if ambiguous, not Q4K.
            .unwrap_or(0);

        bytes_per_param[idx]
    }
}
