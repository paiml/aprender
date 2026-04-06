//! Stress test configuration types (TRUENO-SPEC-025 Section 7.3)
//!
//! Contains [`StressTestConfig`], [`StressTarget`], and [`ChaosPreset`].

use std::time::Duration;

use super::super::device::DeviceId;

// ============================================================================
// Stress Test Configuration (TRUENO-SPEC-025 Section 7.3)
// ============================================================================

/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Target resource(s) to stress
    pub target: StressTarget,
    /// Test duration
    pub duration: Duration,
    /// Intensity (0.0-1.0, where 1.0 = maximum load)
    pub intensity: f64,
    /// Ramp-up duration (gradual increase)
    pub ramp_up: Duration,
    /// Chaos engineering preset (optional)
    pub chaos_preset: Option<ChaosPreset>,
    /// Whether to collect detailed metrics
    pub collect_metrics: bool,
    /// Whether to export report on completion
    pub export_report: bool,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            target: StressTarget::All,
            duration: Duration::from_secs(60),
            intensity: 1.0,
            ramp_up: Duration::from_secs(5),
            chaos_preset: None,
            collect_metrics: true,
            export_report: true,
        }
    }
}

impl StressTestConfig {
    /// Create a new stress test configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target
    #[must_use]
    pub fn with_target(mut self, target: StressTarget) -> Self {
        self.target = target;
        self
    }

    /// Set duration
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set intensity
    #[must_use]
    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set ramp-up duration
    #[must_use]
    pub fn with_ramp_up(mut self, ramp_up: Duration) -> Self {
        self.ramp_up = ramp_up;
        self
    }

    /// Set chaos preset
    #[must_use]
    pub fn with_chaos(mut self, preset: ChaosPreset) -> Self {
        self.chaos_preset = Some(preset);
        self
    }

    /// Parse duration from string (e.g., "60s", "5m", "1h")
    #[must_use]
    pub fn parse_duration(s: &str) -> Option<Duration> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        let (num, unit) = s.split_at(s.len() - 1);
        let value: u64 = num.parse().ok()?;

        match unit {
            "s" => Some(Duration::from_secs(value)),
            "m" => Some(Duration::from_secs(value * 60)),
            "h" => Some(Duration::from_secs(value * 3600)),
            _ => None,
        }
    }
}

/// Stress target
#[derive(Debug, Clone, PartialEq)]
pub enum StressTarget {
    /// Stress all resources
    All,
    /// Stress CPU only
    Cpu,
    /// Stress GPU (optionally specific device)
    Gpu(Option<DeviceId>),
    /// Stress memory (RAM + VRAM)
    Memory,
    /// Stress PCIe bandwidth
    Pcie,
    /// Custom combination
    Custom(Vec<StressTarget>),
}

impl StressTarget {
    /// Parse from string (e.g., "cpu", "gpu", "gpu:0", "memory", "pcie")
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_lowercase();

        if s == "all" {
            return Some(Self::All);
        }
        if s == "cpu" {
            return Some(Self::Cpu);
        }
        if s == "memory" {
            return Some(Self::Memory);
        }
        if s == "pcie" {
            return Some(Self::Pcie);
        }
        if s == "gpu" {
            return Some(Self::Gpu(None));
        }
        if let Some(idx_str) = s.strip_prefix("gpu:") {
            let idx: u32 = idx_str.parse().ok()?;
            return Some(Self::Gpu(Some(DeviceId::nvidia(idx))));
        }

        None
    }

    /// Check if target includes CPU
    #[must_use]
    pub fn includes_cpu(&self) -> bool {
        match self {
            Self::All | Self::Cpu => true,
            Self::Custom(targets) => targets.iter().any(|t| t.includes_cpu()),
            _ => false,
        }
    }

    /// Check if target includes GPU
    #[must_use]
    pub fn includes_gpu(&self) -> bool {
        match self {
            Self::All | Self::Gpu(_) => true,
            Self::Custom(targets) => targets.iter().any(|t| t.includes_gpu()),
            _ => false,
        }
    }

    /// Check if target includes memory
    #[must_use]
    pub fn includes_memory(&self) -> bool {
        match self {
            Self::All | Self::Memory => true,
            Self::Custom(targets) => targets.iter().any(|t| t.includes_memory()),
            _ => false,
        }
    }

    /// Check if target includes PCIe
    #[must_use]
    pub fn includes_pcie(&self) -> bool {
        match self {
            Self::All | Self::Pcie => true,
            Self::Custom(targets) => targets.iter().any(|t| t.includes_pcie()),
            _ => false,
        }
    }
}

// ============================================================================
// Chaos Engineering Presets (from renacer)
// ============================================================================

/// Chaos engineering preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChaosPreset {
    /// Gentle chaos (minimal impact)
    Gentle,
    /// Moderate chaos
    Moderate,
    /// Aggressive chaos
    Aggressive,
    /// Extreme chaos (maximum stress)
    Extreme,
}

impl ChaosPreset {
    /// Parse from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "gentle" => Some(Self::Gentle),
            "moderate" => Some(Self::Moderate),
            "aggressive" => Some(Self::Aggressive),
            "extreme" => Some(Self::Extreme),
            _ => None,
        }
    }

    /// Get memory limit factor (0.0-1.0)
    #[must_use]
    pub fn memory_limit_factor(&self) -> f64 {
        match self {
            Self::Gentle => 0.9,     // 90% of available
            Self::Moderate => 0.75,  // 75%
            Self::Aggressive => 0.5, // 50%
            Self::Extreme => 0.25,   // 25%
        }
    }

    /// Get CPU throttle factor (0.0-1.0)
    #[must_use]
    pub fn cpu_throttle_factor(&self) -> f64 {
        match self {
            Self::Gentle => 1.0,     // No throttle
            Self::Moderate => 0.9,   // 90%
            Self::Aggressive => 0.7, // 70%
            Self::Extreme => 0.5,    // 50%
        }
    }

    /// Get network latency injection in milliseconds
    #[must_use]
    pub fn network_latency_ms(&self) -> u32 {
        match self {
            Self::Gentle => 0,
            Self::Moderate => 10,
            Self::Aggressive => 50,
            Self::Extreme => 200,
        }
    }

    /// Get failure injection rate (0.0-1.0)
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        match self {
            Self::Gentle => 0.0,
            Self::Moderate => 0.01,
            Self::Aggressive => 0.05,
            Self::Extreme => 0.10,
        }
    }
}
