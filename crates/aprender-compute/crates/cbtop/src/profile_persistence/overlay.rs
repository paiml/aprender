//! CLI overlay and profile templates.

use super::config::{default_threads, BackendConfig, ProfileConfig, WorkloadConfig};

/// CLI overlay for merging profile with command-line arguments
#[derive(Debug, Clone, Default)]
pub struct ProfileOverlay {
    /// Override refresh rate
    pub refresh_ms: Option<u64>,
    /// Override device index
    pub device_index: Option<u32>,
    /// Override backend
    pub backend: Option<BackendConfig>,
    /// Override load intensity
    pub load_intensity: Option<f64>,
    /// Override workload
    pub workload: Option<WorkloadConfig>,
    /// Override problem size
    pub problem_size: Option<usize>,
    /// Override threads
    pub threads: Option<usize>,
    /// Override deterministic
    pub deterministic: Option<bool>,
}

impl ProfileOverlay {
    /// Create empty overlay
    pub fn new() -> Self {
        Self::default()
    }

    /// Set refresh rate override
    pub fn refresh_ms(mut self, ms: u64) -> Self {
        self.refresh_ms = Some(ms);
        self
    }

    /// Set backend override
    pub fn backend(mut self, backend: BackendConfig) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set workload override
    pub fn workload(mut self, workload: WorkloadConfig) -> Self {
        self.workload = Some(workload);
        self
    }

    /// Set problem size override
    pub fn problem_size(mut self, size: usize) -> Self {
        self.problem_size = Some(size);
        self
    }

    /// Apply overlay to profile (CLI > profile > default)
    pub fn apply(&self, mut profile: ProfileConfig) -> ProfileConfig {
        if let Some(v) = self.refresh_ms {
            profile.refresh_ms = v;
        }
        if let Some(v) = self.device_index {
            profile.device_index = v;
        }
        if let Some(v) = self.backend {
            profile.backend = v;
        }
        if let Some(v) = self.load_intensity {
            profile.load_intensity = v;
        }
        if let Some(v) = self.workload {
            profile.workload = v;
        }
        if let Some(v) = self.problem_size {
            profile.problem_size = v;
        }
        if let Some(v) = self.threads {
            profile.threads = v;
        }
        if let Some(v) = self.deterministic {
            profile.deterministic = v;
        }
        profile
    }

    /// Check if any overrides are set
    pub fn has_overrides(&self) -> bool {
        self.refresh_ms.is_some()
            || self.device_index.is_some()
            || self.backend.is_some()
            || self.load_intensity.is_some()
            || self.workload.is_some()
            || self.problem_size.is_some()
            || self.threads.is_some()
            || self.deterministic.is_some()
    }
}

/// Pre-defined profile templates
pub mod templates {
    use super::*;

    /// Create ML training profile
    pub fn ml_training() -> ProfileConfig {
        ProfileConfig {
            name: "ml_training".to_string(),
            description: "Optimized for ML training workloads".to_string(),
            version: "1.0".to_string(),
            refresh_ms: 200,
            device_index: 0,
            backend: BackendConfig::Cuda,
            load_intensity: 0.75,
            workload: WorkloadConfig::Gemm,
            problem_size: 4_194_304,
            threads: default_threads(),
            deterministic: false,
            metadata: [("use_case".to_string(), "training".to_string())].into_iter().collect(),
        }
    }

    /// Create inference profile
    pub fn inference() -> ProfileConfig {
        ProfileConfig {
            name: "inference".to_string(),
            description: "Optimized for inference workloads".to_string(),
            version: "1.0".to_string(),
            refresh_ms: 50,
            device_index: 0,
            backend: BackendConfig::Cuda,
            load_intensity: 0.5,
            workload: WorkloadConfig::Attention,
            problem_size: 1_048_576,
            threads: default_threads(),
            deterministic: true,
            metadata: [("use_case".to_string(), "inference".to_string())].into_iter().collect(),
        }
    }

    /// Create stress test profile
    pub fn stress_test() -> ProfileConfig {
        ProfileConfig {
            name: "stress_test".to_string(),
            description: "Maximum stress for stability testing".to_string(),
            version: "1.0".to_string(),
            refresh_ms: 100,
            device_index: 0,
            backend: BackendConfig::All,
            load_intensity: 1.0,
            workload: WorkloadConfig::All,
            problem_size: 16_777_216,
            threads: default_threads(),
            deterministic: false,
            metadata: [("use_case".to_string(), "stress".to_string())].into_iter().collect(),
        }
    }

    /// Create SIMD-only profile
    pub fn simd_only() -> ProfileConfig {
        ProfileConfig {
            name: "simd_only".to_string(),
            description: "CPU SIMD operations only".to_string(),
            version: "1.0".to_string(),
            refresh_ms: 100,
            device_index: 0,
            backend: BackendConfig::Simd,
            load_intensity: 0.5,
            workload: WorkloadConfig::Elementwise,
            problem_size: 1_048_576,
            threads: default_threads(),
            deterministic: false,
            metadata: [("use_case".to_string(), "cpu".to_string())].into_iter().collect(),
        }
    }
}
