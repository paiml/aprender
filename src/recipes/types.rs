//! Recipe types and configuration structures.

use crate::experiment::{ComputeDevice, CpuArchitecture, ModelParadigm, PlatformEfficiency};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Recipe execution result
#[derive(Debug, Clone)]
pub struct RecipeResult {
    /// Recipe name
    pub recipe_name: String,
    /// Whether the recipe succeeded
    pub success: bool,
    /// Output artifacts
    pub artifacts: Vec<String>,
    /// Metrics collected
    pub metrics: HashMap<String, f64>,
    /// Error message if failed
    pub error: Option<String>,
}

impl RecipeResult {
    /// Create a successful result
    pub fn success(name: impl Into<String>) -> Self {
        Self {
            recipe_name: name.into(),
            success: true,
            artifacts: Vec::new(),
            metrics: HashMap::new(),
            error: None,
        }
    }

    /// Create a failed result
    pub fn failure(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            recipe_name: name.into(),
            success: false,
            artifacts: Vec::new(),
            metrics: HashMap::new(),
            error: Some(error.into()),
        }
    }

    /// Add an artifact
    pub fn with_artifact(mut self, artifact: impl Into<String>) -> Self {
        self.artifacts.push(artifact.into());
        self
    }

    /// Add a metric
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }
}

/// Experiment tracking recipe configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTrackingConfig {
    /// Experiment name
    pub experiment_name: String,
    /// Model paradigm
    pub paradigm: ModelParadigm,
    /// Compute device configuration
    pub device: ComputeDevice,
    /// Platform type
    pub platform: PlatformEfficiency,
    /// Enable energy tracking
    pub track_energy: bool,
    /// Enable cost tracking
    pub track_cost: bool,
    /// Carbon intensity for CO2 calculation (g/kWh)
    pub carbon_intensity: Option<f64>,
    /// Tags for organization
    pub tags: Vec<String>,
}

impl Default for ExperimentTrackingConfig {
    fn default() -> Self {
        Self {
            experiment_name: "default-experiment".to_string(),
            paradigm: ModelParadigm::DeepLearning,
            device: ComputeDevice::Cpu {
                cores: 8,
                threads_per_core: 2,
                architecture: CpuArchitecture::X86_64,
            },
            platform: PlatformEfficiency::Server,
            track_energy: true,
            track_cost: true,
            carbon_intensity: Some(400.0), // Global average
            tags: Vec::new(),
        }
    }
}
