use serde::{Deserialize, Serialize};

use crate::engine::rng::SimRng;
use crate::error::SimResult;

// ============================================================================
// Prediction Simulation
// ============================================================================

/// Prediction state for replay and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionState {
    /// Input features.
    pub input: Vec<f64>,
    /// Model output.
    pub output: Vec<f64>,
    /// Uncertainty estimate (if available).
    pub uncertainty: Option<f64>,
    /// Inference latency in microseconds (simulated).
    pub latency_us: u64,
    /// Sequence number.
    pub sequence: u64,
}

/// Inference configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for inference.
    pub batch_size: usize,
    /// Temperature for probabilistic outputs.
    pub temperature: f64,
    /// Top-k sampling (0 = greedy).
    pub top_k: usize,
    /// Enable uncertainty quantification.
    pub uncertainty: bool,
    /// Simulated latency base (microseconds).
    pub base_latency_us: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            temperature: 1.0,
            top_k: 0,
            uncertainty: false,
            base_latency_us: 1000,
        }
    }
}

/// Simulated inference scenario for reproducible prediction testing.
pub struct PredictionSimulation {
    /// Inference configuration.
    config: InferenceConfig,
    /// Deterministic RNG for stochastic models.
    rng: SimRng,
    /// Prediction sequence counter.
    sequence: u64,
    /// Prediction history.
    history: Vec<PredictionState>,
}

impl PredictionSimulation {
    /// Create new prediction simulation with deterministic seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            config: InferenceConfig::default(),
            rng: SimRng::new(seed),
            sequence: 0,
            history: Vec::new(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(seed: u64, config: InferenceConfig) -> Self {
        Self {
            config,
            rng: SimRng::new(seed),
            sequence: 0,
            history: Vec::new(),
        }
    }

    /// Get inference configuration.
    #[must_use]
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Simulate single prediction using a model function.
    ///
    /// The `model_fn` takes input and returns output vector.
    ///
    /// # Errors
    ///
    /// Returns error if model prediction fails.
    pub fn predict<F>(&mut self, input: &[f64], model_fn: F) -> SimResult<PredictionState>
    where
        F: FnOnce(&[f64]) -> Vec<f64>,
    {
        // Simulate inference
        let mut output = model_fn(input);

        // Apply temperature scaling if not 1.0
        if (self.config.temperature - 1.0).abs() > 1e-10 {
            output = self.apply_temperature(&output, self.config.temperature);
        }

        // Apply top-k sampling if configured
        if self.config.top_k > 0 {
            output = self.sample_top_k(&output, self.config.top_k);
        }

        // Compute uncertainty if enabled (simplified: variance of output)
        let uncertainty = if self.config.uncertainty {
            Some(self.compute_uncertainty(&output))
        } else {
            None
        };

        // Simulate latency with noise
        let latency_noise = (self.rng.gen_f64() * 0.2 - 0.1) * self.config.base_latency_us as f64;
        let latency_us = (self.config.base_latency_us as f64 + latency_noise).max(1.0) as u64;

        let state = PredictionState {
            input: input.to_vec(),
            output,
            uncertainty,
            latency_us,
            sequence: self.sequence,
        };

        self.sequence += 1;
        self.history.push(state.clone());

        Ok(state)
    }

    /// Simulate batch prediction.
    ///
    /// # Errors
    ///
    /// Returns error if any prediction fails.
    pub fn predict_batch<F>(
        &mut self,
        inputs: &[Vec<f64>],
        model_fn: F,
    ) -> SimResult<Vec<PredictionState>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        inputs
            .iter()
            .map(|input| self.predict(input, &model_fn))
            .collect()
    }

    /// Apply temperature scaling to logits.
    #[allow(clippy::unused_self)]
    fn apply_temperature(&self, logits: &[f64], temperature: f64) -> Vec<f64> {
        if temperature <= 0.0 {
            return logits.to_vec();
        }
        logits.iter().map(|x| x / temperature).collect()
    }

    /// Sample top-k values, zeroing out the rest.
    #[allow(clippy::unused_self)]
    fn sample_top_k(&self, values: &[f64], k: usize) -> Vec<f64> {
        if k >= values.len() {
            return values.to_vec();
        }

        // Find k-th largest value
        let mut sorted: Vec<f64> = values.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted.get(k - 1).copied().unwrap_or(f64::NEG_INFINITY);

        // Zero out values below threshold
        values
            .iter()
            .map(|&v| if v >= threshold { v } else { 0.0 })
            .collect()
    }

    /// Compute simplified uncertainty estimate.
    #[allow(clippy::unused_self)]
    fn compute_uncertainty(&self, output: &[f64]) -> f64 {
        if output.is_empty() {
            return 0.0;
        }
        let mean = output.iter().sum::<f64>() / output.len() as f64;
        let variance = output.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / output.len() as f64;
        variance.sqrt()
    }

    /// Get prediction history.
    #[must_use]
    pub fn history(&self) -> &[PredictionState] {
        &self.history
    }

    /// Reset simulation state.
    pub fn reset(&mut self, seed: u64) {
        self.rng = SimRng::new(seed);
        self.sequence = 0;
        self.history.clear();
    }
}
