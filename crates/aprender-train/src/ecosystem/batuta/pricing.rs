//! GPU pricing types and fallback pricing.

use serde::{Deserialize, Serialize};

/// GPU pricing information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuPricing {
    /// GPU type identifier (e.g., "a100-80gb", "v100", "t4")
    pub gpu_type: String,
    /// Hourly rate in USD
    pub hourly_rate: f64,
    /// GPU memory in GB
    pub memory_gb: u32,
    /// Whether this is spot/preemptible pricing
    pub is_spot: bool,
    /// Provider name (e.g., "aws", "gcp", "azure")
    pub provider: String,
    /// Region identifier
    pub region: String,
}

impl GpuPricing {
    /// Create a new GPU pricing entry.
    pub fn new(gpu_type: impl Into<String>, hourly_rate: f64, memory_gb: u32) -> Self {
        Self {
            gpu_type: gpu_type.into(),
            hourly_rate,
            memory_gb,
            is_spot: false,
            provider: "unknown".to_string(),
            region: "unknown".to_string(),
        }
    }

    /// Set spot pricing flag.
    pub fn with_spot(mut self, is_spot: bool) -> Self {
        self.is_spot = is_spot;
        self
    }

    /// Set provider.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// Set region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }
}

/// Fallback pricing when Batuta is unavailable.
///
/// Uses conservative estimates based on typical cloud provider pricing.
#[derive(Debug, Clone)]
pub struct FallbackPricing {
    /// Default pricing for known GPU types
    pricing: Vec<GpuPricing>,
}

impl Default for FallbackPricing {
    fn default() -> Self {
        Self::new()
    }
}

impl FallbackPricing {
    /// Create fallback pricing with typical cloud rates.
    pub fn new() -> Self {
        Self {
            pricing: vec![
                GpuPricing::new("a100-80gb", 3.00, 80)
                    .with_provider("generic")
                    .with_region("us-east-1"),
                GpuPricing::new("a100-40gb", 2.50, 40)
                    .with_provider("generic")
                    .with_region("us-east-1"),
                GpuPricing::new("v100", 2.00, 16).with_provider("generic").with_region("us-east-1"),
                GpuPricing::new("t4", 0.50, 16).with_provider("generic").with_region("us-east-1"),
                GpuPricing::new("l4", 0.75, 24).with_provider("generic").with_region("us-east-1"),
                GpuPricing::new("a10g", 1.00, 24).with_provider("generic").with_region("us-east-1"),
                GpuPricing::new("h100-80gb", 4.50, 80)
                    .with_provider("generic")
                    .with_region("us-east-1"),
            ],
        }
    }

    /// Get pricing for a GPU type.
    pub fn get_rate(&self, gpu_type: &str) -> Option<&GpuPricing> {
        let normalized = gpu_type.to_lowercase().replace(['-', '_'], "");
        self.pricing.iter().find(|p| {
            let p_normalized = p.gpu_type.to_lowercase().replace(['-', '_'], "");
            p_normalized == normalized
        })
    }

    /// Get all available pricing.
    pub fn all_pricing(&self) -> &[GpuPricing] {
        &self.pricing
    }

    /// Add or update pricing for a GPU type.
    pub fn set_rate(&mut self, pricing: GpuPricing) {
        let normalized = pricing.gpu_type.to_lowercase().replace(['-', '_'], "");
        if let Some(existing) = self.pricing.iter_mut().find(|p| {
            let p_normalized = p.gpu_type.to_lowercase().replace(['-', '_'], "");
            p_normalized == normalized
        }) {
            *existing = pricing;
        } else {
            self.pricing.push(pricing);
        }
    }
}
