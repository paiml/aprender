//! Batuta client for pricing and queue services.

use std::time::Duration;

use super::error::BatutaError;
use super::pricing::{FallbackPricing, GpuPricing};
use super::queue::QueueState;

/// Client for interacting with Batuta pricing and queue services.
#[derive(Debug, Clone)]
pub struct BatutaClient {
    /// Base URL for Batuta API (None if using fallback only)
    base_url: Option<String>,
    /// Fallback pricing when Batuta is unavailable
    fallback: FallbackPricing,
    /// Connection timeout
    timeout: Duration,
    /// Whether Batuta service is available
    service_available: bool,
}

impl Default for BatutaClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BatutaClient {
    /// Create a new Batuta client with fallback pricing only.
    pub fn new() -> Self {
        Self {
            base_url: None,
            fallback: FallbackPricing::new(),
            timeout: Duration::from_secs(5),
            service_available: false,
        }
    }

    /// Create a client connected to a Batuta instance.
    pub fn with_url(url: impl Into<String>) -> Self {
        Self {
            base_url: Some(url.into()),
            fallback: FallbackPricing::new(),
            timeout: Duration::from_secs(5),
            service_available: true,
        }
    }

    /// Set connection timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set custom fallback pricing.
    pub fn with_fallback(mut self, fallback: FallbackPricing) -> Self {
        self.fallback = fallback;
        self
    }

    /// Check if connected to a live Batuta service.
    pub fn is_connected(&self) -> bool {
        self.base_url.is_some() && self.service_available
    }

    /// Get hourly rate for a GPU type.
    ///
    /// Returns live pricing from Batuta if available, otherwise fallback pricing.
    pub fn get_hourly_rate(&self, gpu_type: &str) -> Result<GpuPricing, BatutaError> {
        // If we have a live connection, try to fetch from Batuta
        if let Some(_url) = &self.base_url {
            // In a real implementation, this would make an HTTP request
            // For now, we simulate by returning fallback
            // FUTURE(batuta-api): HTTP client integration pending API finalization
        }

        // Use fallback pricing
        self.fallback
            .get_rate(gpu_type)
            .cloned()
            .ok_or_else(|| BatutaError::UnknownGpuType(gpu_type.to_string()))
    }

    /// Get current queue depth.
    ///
    /// Returns queue state from Batuta if available, otherwise returns
    /// an optimistic default (no queue).
    pub fn get_queue_depth(&self, gpu_type: &str) -> Result<QueueState, BatutaError> {
        // Validate GPU type exists
        if self.fallback.get_rate(gpu_type).is_none() {
            return Err(BatutaError::UnknownGpuType(gpu_type.to_string()));
        }

        // If we have a live connection, try to fetch from Batuta
        if let Some(_url) = &self.base_url {
            // In a real implementation, this would make an HTTP request
            // FUTURE(batuta-api): HTTP client integration pending API finalization
        }

        // Return optimistic default (no queue)
        Ok(QueueState::new(0, 4, 4))
    }

    /// Get both pricing and queue state in one call.
    pub fn get_status(&self, gpu_type: &str) -> Result<(GpuPricing, QueueState), BatutaError> {
        let pricing = self.get_hourly_rate(gpu_type)?;
        let queue = self.get_queue_depth(gpu_type)?;
        Ok((pricing, queue))
    }

    /// Estimate total cost for a job.
    pub fn estimate_cost(&self, gpu_type: &str, hours: f64) -> Result<f64, BatutaError> {
        let pricing = self.get_hourly_rate(gpu_type)?;
        Ok(pricing.hourly_rate * hours)
    }

    /// Get the cheapest GPU that meets memory requirements.
    pub fn cheapest_gpu(&self, min_memory_gb: u32) -> Option<&GpuPricing> {
        self.fallback.all_pricing().iter().filter(|p| p.memory_gb >= min_memory_gb).min_by(
            |a, b| a.hourly_rate.partial_cmp(&b.hourly_rate).unwrap_or(std::cmp::Ordering::Equal),
        )
    }
}
