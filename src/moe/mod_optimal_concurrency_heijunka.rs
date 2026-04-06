
impl HeijunkaController {
    /// Create a new Heijunka controller
    #[must_use]
    pub fn new(config: HeijunkaConfig) -> Self {
        Self { config }
    }

    /// Calculate optimal concurrency using Little's Law
    ///
    /// # Arguments
    ///
    /// * `arrival_rate` - Requests per second
    /// * `latency_ms` - Average latency in milliseconds
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn optimal_concurrency(&self, arrival_rate: f64, latency_ms: f64) -> usize {
        // Little's Law: L = lambda * W
        let optimal = (arrival_rate * latency_ms / 1000.0).ceil() as usize;
        optimal.clamp(1, self.config.max_concurrency)
    }

    /// Determine if load should be shed based on current state
    ///
    /// # Arguments
    ///
    /// * `current_latency_ms` - Current observed latency
    /// * `current_concurrency` - Current number of concurrent requests
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    pub fn should_shed_load(
        &self,
        current_latency_ms: f64,
        current_concurrency: usize,
    ) -> LoadSheddingDecision {
        let should_shed = current_latency_ms > self.config.target_latency_ms
            && current_concurrency >= self.config.max_concurrency;

        // Calculate recommended concurrency to meet target latency
        // If latency is 2x target, we need to halve concurrency
        let ratio = self.config.target_latency_ms / current_latency_ms;
        let concurrency_f64: f64 = current_concurrency as f64;
        let recommended = (concurrency_f64 * ratio).ceil() as usize;

        LoadSheddingDecision {
            shed_load: should_shed,
            recommended_concurrency: recommended.clamp(1, self.config.max_concurrency),
        }
    }

    /// Get the target latency
    #[must_use]
    pub fn target_latency_ms(&self) -> f64 {
        self.config.target_latency_ms
    }
}

/// Andon trigger types per Toyota Production System (Jidoka)
#[derive(Debug, Clone, PartialEq)]
pub enum AndonTrigger {
    /// Model checksum mismatch - corrupted model
    ModelChecksumMismatch {
        /// ID of the corrupted model
        model_id: String,
    },
    /// Latency P99 exceeded threshold
    LatencyExceeded {
        /// Observed P99 latency in milliseconds
        p99_ms: f64,
        /// Threshold that was exceeded
        threshold_ms: f64,
    },
    /// Error rate above threshold
    ErrorRateThreshold {
        /// Observed error rate (0.0 - 1.0)
        rate: f64,
        /// Threshold that was exceeded
        threshold: f64,
    },
    /// Expert load imbalance detected
    ExpertImbalance {
        /// Ratio of max/min expert utilization
        imbalance_ratio: f64,
    },
}

/// Response action for Andon triggers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AndonResponse {
    /// Automatically rollback to previous known-good state
    Rollback,
    /// Notify operators but continue serving
    Notify,
    /// Quarantine the failing expert (stop routing to it)
    Quarantine,
}

impl AndonTrigger {
    /// Determine appropriate response for this trigger
    #[must_use]
    pub fn response(&self) -> AndonResponse {
        match self {
            Self::ModelChecksumMismatch { .. } => AndonResponse::Rollback,
            Self::ErrorRateThreshold { rate, threshold } => {
                if *rate > threshold * 2.0 {
                    AndonResponse::Quarantine
                } else {
                    AndonResponse::Notify
                }
            },
            Self::LatencyExceeded { .. } | Self::ExpertImbalance { .. } => AndonResponse::Notify,
        }
    }

    /// Check if this trigger is critical (requires immediate action)
    #[must_use]
    pub fn is_critical(&self) -> bool {
        matches!(
            self.response(),
            AndonResponse::Rollback | AndonResponse::Quarantine
        )
    }
}

#[cfg(test)]
mod tests;
