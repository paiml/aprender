//! Experiment Tracking Integration Module
//!
//! Integrates with Entrenar Experiment Tracking Spec v1.8.0 for tracking
//! synthetic data generation experiments with cost, energy, and quality metrics.
//!
//! # Features
//! - `ComputeDevice` abstraction (CPU/GPU/TPU/AppleSilicon)
//! - `EnergyMetrics` and `CostMetrics` for efficiency tracking
//! - `GenerationExperiment` for tracking data generation runs
//! - CO2 emissions estimation based on grid carbon intensity
//!
//! # Example
//! ```
//! use verificar::ml::{ComputeDevice, CpuArchitecture, GenerationExperiment};
//!
//! let device = ComputeDevice::Cpu {
//!     cores: 8,
//!     threads_per_core: 2,
//!     architecture: CpuArchitecture::X86_64,
//! };
//!
//! let mut experiment = GenerationExperiment::new("depyler-corpus-v1", device);
//! experiment.record_samples(1000, std::time::Duration::from_secs(60));
//! let metrics = experiment.finalize();
//! println!("Cost per sample: ${:.6}", metrics.cost_per_sample());
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Compute device abstraction for heterogeneous hardware
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputeDevice {
    /// Standard CPU execution
    Cpu {
        /// Number of physical cores
        cores: u32,
        /// Threads per core (hyperthreading)
        threads_per_core: u32,
        /// CPU architecture
        architecture: CpuArchitecture,
    },
    /// NVIDIA/AMD GPU acceleration
    Gpu {
        /// GPU model name
        name: String,
        /// GPU memory in GB
        memory_gb: f32,
        /// Compute capability (e.g., "8.6" for Ampere)
        compute_capability: Option<String>,
        /// GPU vendor
        vendor: GpuVendor,
    },
    /// Google TPU accelerator
    Tpu {
        /// TPU version
        version: TpuVersion,
        /// Number of TPU cores
        cores: u32,
    },
    /// Apple Silicon unified memory
    AppleSilicon {
        /// Apple chip model
        chip: AppleChip,
        /// Neural engine cores
        neural_engine_cores: u32,
        /// GPU cores
        gpu_cores: u32,
        /// Unified memory in GB
        memory_gb: u32,
    },
    /// Edge/embedded devices
    Edge {
        /// Device name
        name: String,
        /// Power budget in watts
        power_budget_watts: f32,
    },
}

/// CPU architecture variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuArchitecture {
    /// x86-64 (Intel/AMD)
    X86_64,
    /// ARM 64-bit
    Aarch64,
    /// RISC-V 64-bit
    Riscv64,
    /// WebAssembly 32-bit
    Wasm32,
}

/// GPU vendor identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendor {
    /// NVIDIA GPUs
    Nvidia,
    /// AMD GPUs
    Amd,
    /// Intel GPUs
    Intel,
    /// Apple GPUs
    Apple,
}

/// TPU version variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpuVersion {
    /// TPU v2
    V2,
    /// TPU v3
    V3,
    /// TPU v4
    V4,
    /// TPU v5e (efficiency)
    V5e,
    /// TPU v5p (performance)
    V5p,
}

/// Apple Silicon chip variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppleChip {
    /// M1 base
    M1,
    /// M1 Pro
    M1Pro,
    /// M1 Max
    M1Max,
    /// M1 Ultra
    M1Ultra,
    /// M2 base
    M2,
    /// M2 Pro
    M2Pro,
    /// M2 Max
    M2Max,
    /// M2 Ultra
    M2Ultra,
    /// M3 base
    M3,
    /// M3 Pro
    M3Pro,
    /// M3 Max
    M3Max,
    /// M4 base
    M4,
    /// M4 Pro
    M4Pro,
    /// M4 Max
    M4Max,
}

impl ComputeDevice {
    /// Calculate theoretical FLOPS for the device
    #[must_use]
    pub fn theoretical_flops(&self) -> f64 {
        match self {
            ComputeDevice::Cpu {
                cores,
                threads_per_core,
                architecture,
            } => {
                let base_flops = match architecture {
                    CpuArchitecture::X86_64 => 32.0,  // AVX2: 8 FP32 * 4 ops
                    CpuArchitecture::Aarch64 => 16.0, // NEON: 4 FP32 * 4 ops
                    CpuArchitecture::Riscv64 => 8.0,
                    CpuArchitecture::Wasm32 => 4.0,
                };
                f64::from(*cores) * f64::from(*threads_per_core) * base_flops * 1e9
            }
            ComputeDevice::Gpu {
                memory_gb, vendor, ..
            } => {
                // Rough estimate based on memory bandwidth
                let bandwidth_factor = match vendor {
                    GpuVendor::Nvidia => 15.0,
                    GpuVendor::Amd => 12.0,
                    GpuVendor::Intel => 8.0,
                    GpuVendor::Apple => 10.0,
                };
                f64::from(*memory_gb) * bandwidth_factor * 1e12
            }
            ComputeDevice::Tpu { version, cores } => {
                let flops_per_core = match version {
                    TpuVersion::V2 => 45e12,
                    TpuVersion::V3 => 90e12,
                    TpuVersion::V4 => 275e12,
                    TpuVersion::V5e => 197e12,
                    TpuVersion::V5p => 459e12,
                };
                f64::from(*cores) * flops_per_core
            }
            ComputeDevice::AppleSilicon {
                chip, gpu_cores, ..
            } => {
                let flops_per_gpu_core = match chip {
                    AppleChip::M1 | AppleChip::M1Pro | AppleChip::M1Max | AppleChip::M1Ultra => {
                        128e9
                    }
                    AppleChip::M2 | AppleChip::M2Pro | AppleChip::M2Max | AppleChip::M2Ultra => {
                        150e9
                    }
                    AppleChip::M3 | AppleChip::M3Pro | AppleChip::M3Max => 180e9,
                    AppleChip::M4 | AppleChip::M4Pro | AppleChip::M4Max => 200e9,
                };
                f64::from(*gpu_cores) * flops_per_gpu_core
            }
            ComputeDevice::Edge {
                power_budget_watts, ..
            } => {
                // Assume ~10 GFLOPS per watt for edge devices
                f64::from(*power_budget_watts) * 10e9
            }
        }
    }

    /// Estimate power consumption in watts
    #[must_use]
    pub fn estimated_power_watts(&self) -> f32 {
        match self {
            ComputeDevice::Cpu { cores, .. } => (*cores as f32) * 15.0,
            ComputeDevice::Gpu {
                memory_gb, vendor, ..
            } => {
                let base = match vendor {
                    GpuVendor::Nvidia => 30.0,
                    GpuVendor::Amd => 35.0,
                    GpuVendor::Intel => 25.0,
                    GpuVendor::Apple => 20.0,
                };
                *memory_gb * base
            }
            ComputeDevice::Tpu { version, cores } => {
                let per_core = match version {
                    TpuVersion::V2 => 40.0,
                    TpuVersion::V3 => 50.0,
                    TpuVersion::V4 => 60.0,
                    TpuVersion::V5e => 45.0,
                    TpuVersion::V5p => 70.0,
                };
                (*cores as f32) * per_core
            }
            ComputeDevice::AppleSilicon { chip, .. } => match chip {
                AppleChip::M1 => 20.0,
                AppleChip::M1Pro => 30.0,
                AppleChip::M1Max => 40.0,
                AppleChip::M1Ultra => 60.0,
                AppleChip::M2 => 22.0,
                AppleChip::M2Pro => 32.0,
                AppleChip::M2Max => 45.0,
                AppleChip::M2Ultra => 65.0,
                AppleChip::M3 => 24.0,
                AppleChip::M3Pro => 35.0,
                AppleChip::M3Max => 50.0,
                AppleChip::M4 => 25.0,
                AppleChip::M4Pro => 38.0,
                AppleChip::M4Max => 55.0,
            },
            ComputeDevice::Edge {
                power_budget_watts, ..
            } => *power_budget_watts,
        }
    }

    /// Create a default CPU device based on current system
    #[must_use]
    pub fn default_cpu() -> Self {
        ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            #[cfg(target_arch = "x86_64")]
            architecture: CpuArchitecture::X86_64,
            #[cfg(target_arch = "aarch64")]
            architecture: CpuArchitecture::Aarch64,
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            architecture: CpuArchitecture::X86_64,
        }
    }
}

impl Default for ComputeDevice {
    fn default() -> Self {
        Self::default_cpu()
    }
}

/// Energy consumption metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyMetrics {
    /// Total energy consumed in joules
    pub total_joules: f64,
    /// Average power draw in watts
    pub average_power_watts: f64,
    /// Peak power draw in watts
    pub peak_power_watts: f64,
    /// Duration of measurement in seconds
    pub duration_seconds: f64,
    /// CO2 equivalent emissions in grams (based on grid carbon intensity)
    pub co2_grams: Option<f64>,
    /// Power Usage Effectiveness (datacenter overhead)
    pub pue: f64,
}

impl EnergyMetrics {
    /// Create new energy metrics
    #[must_use]
    pub fn new(
        total_joules: f64,
        average_power_watts: f64,
        peak_power_watts: f64,
        duration_seconds: f64,
    ) -> Self {
        Self {
            total_joules,
            average_power_watts,
            peak_power_watts,
            duration_seconds,
            co2_grams: None,
            pue: 1.0,
        }
    }

    /// Calculate CO2 emissions based on carbon intensity (g CO2/kWh)
    ///
    /// Default grid intensity values:
    /// - US Average: 386 g/kWh
    /// - EU Average: 231 g/kWh
    /// - Renewable: ~20 g/kWh
    #[must_use]
    pub fn with_carbon_intensity(mut self, carbon_intensity_g_per_kwh: f64) -> Self {
        let kwh = self.total_joules / 3_600_000.0;
        self.co2_grams = Some(kwh * carbon_intensity_g_per_kwh * self.pue);
        self
    }

    /// Set the Power Usage Effectiveness factor
    ///
    /// PUE represents datacenter overhead:
    /// - 1.0 = no overhead (local machine)
    /// - 1.2 = efficient datacenter
    /// - 1.5 = average datacenter
    /// - 2.0 = inefficient datacenter
    #[must_use]
    pub fn with_pue(mut self, pue: f64) -> Self {
        let old_pue = self.pue;
        self.pue = pue;
        // Recalculate CO2 if already set
        if let Some(co2) = self.co2_grams {
            self.co2_grams = Some(co2 / old_pue * pue);
        }
        self
    }

    /// Calculate energy efficiency in FLOPS per watt
    #[must_use]
    pub fn flops_per_watt(&self, total_flops: f64) -> f64 {
        if self.average_power_watts > 0.0 {
            total_flops / self.average_power_watts
        } else {
            0.0
        }
    }
}

impl Default for EnergyMetrics {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

/// Cost metrics for experiments
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Compute cost in USD
    pub compute_cost_usd: f64,
    /// Storage cost in USD
    pub storage_cost_usd: f64,
    /// Network transfer cost in USD
    pub network_cost_usd: f64,
    /// Total cost in USD
    pub total_cost_usd: f64,
    /// Cost per sample processed
    pub cost_per_sample: Option<f64>,
    /// Currency (default USD)
    pub currency: String,
}

impl CostMetrics {
    /// Create new cost metrics
    #[must_use]
    pub fn new(compute_cost: f64, storage_cost: f64, network_cost: f64) -> Self {
        Self {
            compute_cost_usd: compute_cost,
            storage_cost_usd: storage_cost,
            network_cost_usd: network_cost,
            total_cost_usd: compute_cost + storage_cost + network_cost,
            cost_per_sample: None,
            currency: "USD".to_string(),
        }
    }

    /// Add sample-based cost calculation
    #[must_use]
    pub fn with_samples(mut self, total_samples: u64) -> Self {
        if total_samples > 0 {
            self.cost_per_sample = Some(self.total_cost_usd / total_samples as f64);
        }
        self
    }

    /// Get cost per sample (or 0 if not calculated)
    #[must_use]
    pub fn cost_per_sample(&self) -> f64 {
        self.cost_per_sample.unwrap_or(0.0)
    }
}

impl Default for CostMetrics {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// Generation experiment tracker
///
/// Tracks synthetic data generation runs with timing, energy, and cost metrics.
#[derive(Debug, Clone)]
pub struct GenerationExperiment {
    /// Experiment name/ID
    pub name: String,
    /// Compute device used
    pub device: ComputeDevice,
    /// Start time
    start_time: Option<Instant>,
    /// Total samples generated
    pub samples_generated: u64,
    /// Total duration
    pub total_duration: Duration,
    /// Hourly compute rate in USD
    pub hourly_rate_usd: f64,
    /// Carbon intensity (g CO2/kWh)
    pub carbon_intensity: f64,
}

impl GenerationExperiment {
    /// Create a new generation experiment
    #[must_use]
    pub fn new(name: &str, device: ComputeDevice) -> Self {
        Self {
            name: name.to_string(),
            device,
            start_time: None,
            samples_generated: 0,
            total_duration: Duration::ZERO,
            hourly_rate_usd: 0.10,   // Default: $0.10/hour for CPU
            carbon_intensity: 386.0, // US average
        }
    }

    /// Set hourly compute rate
    #[must_use]
    pub fn with_hourly_rate(mut self, rate_usd: f64) -> Self {
        self.hourly_rate_usd = rate_usd;
        self
    }

    /// Set carbon intensity for CO2 calculation
    #[must_use]
    pub fn with_carbon_intensity(mut self, g_per_kwh: f64) -> Self {
        self.carbon_intensity = g_per_kwh;
        self
    }

    /// Start timing the experiment
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Record samples generated with duration
    pub fn record_samples(&mut self, count: u64, duration: Duration) {
        self.samples_generated += count;
        self.total_duration += duration;
    }

    /// Stop timing and record elapsed
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.total_duration += start.elapsed();
        }
    }

    /// Finalize and compute all metrics
    #[must_use]
    pub fn finalize(&self) -> ExperimentMetrics {
        let duration_secs = self.total_duration.as_secs_f64();
        let power_watts = f64::from(self.device.estimated_power_watts());

        // Energy: P * t (joules)
        let total_joules = power_watts * duration_secs;

        let energy =
            EnergyMetrics::new(total_joules, power_watts, power_watts * 1.2, duration_secs)
                .with_carbon_intensity(self.carbon_intensity);

        // Cost: hourly rate * hours
        let hours = duration_secs / 3600.0;
        let compute_cost = self.hourly_rate_usd * hours;
        let cost = CostMetrics::new(compute_cost, 0.0, 0.0).with_samples(self.samples_generated);

        ExperimentMetrics {
            name: self.name.clone(),
            samples_generated: self.samples_generated,
            duration: self.total_duration,
            energy,
            cost,
            samples_per_second: if duration_secs > 0.0 {
                self.samples_generated as f64 / duration_secs
            } else {
                0.0
            },
        }
    }
}

/// Final experiment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    /// Experiment name
    pub name: String,
    /// Total samples generated
    pub samples_generated: u64,
    /// Total duration
    #[serde(with = "duration_serde")]
    pub duration: Duration,
    /// Energy metrics
    pub energy: EnergyMetrics,
    /// Cost metrics
    pub cost: CostMetrics,
    /// Throughput (samples/second)
    pub samples_per_second: f64,
}

impl ExperimentMetrics {
    /// Get cost per sample
    #[must_use]
    pub fn cost_per_sample(&self) -> f64 {
        self.cost.cost_per_sample()
    }

    /// Get CO2 per sample in grams
    #[must_use]
    pub fn co2_per_sample(&self) -> f64 {
        if self.samples_generated > 0 {
            self.energy.co2_grams.unwrap_or(0.0) / self.samples_generated as f64
        } else {
            0.0
        }
    }
}

/// Serde helper for Duration
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub(super) fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_device_cpu() {
        let device = ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        };
        assert!(device.theoretical_flops() > 0.0);
        assert!(device.estimated_power_watts() > 0.0);
    }

    #[test]
    fn test_compute_device_gpu() {
        let device = ComputeDevice::Gpu {
            name: "RTX 4090".to_string(),
            memory_gb: 24.0,
            compute_capability: Some("8.9".to_string()),
            vendor: GpuVendor::Nvidia,
        };
        assert!(device.theoretical_flops() > 1e12);
        assert!(device.estimated_power_watts() > 100.0);
    }

    #[test]
    fn test_compute_device_apple_silicon() {
        let device = ComputeDevice::AppleSilicon {
            chip: AppleChip::M3Max,
            neural_engine_cores: 16,
            gpu_cores: 40,
            memory_gb: 64,
        };
        assert!(device.theoretical_flops() > 1e12);
        assert_eq!(device.estimated_power_watts(), 50.0);
    }

    #[test]
    fn test_energy_metrics() {
        let energy = EnergyMetrics::new(3600.0, 100.0, 120.0, 36.0)
            .with_carbon_intensity(386.0)
            .with_pue(1.2);

        assert!(energy.co2_grams.is_some());
        assert!(energy.pue > 1.0);
    }

    #[test]
    fn test_cost_metrics() {
        let cost = CostMetrics::new(1.0, 0.1, 0.05).with_samples(1000);
        // Use approximate comparison for floating point
        assert!((cost.total_cost_usd - 1.15).abs() < 0.0001);
        assert!((cost.cost_per_sample() - 0.00115).abs() < 0.0001);
    }

    #[test]
    fn test_generation_experiment() {
        let device = ComputeDevice::default_cpu();
        let mut experiment = GenerationExperiment::new("test-run", device)
            .with_hourly_rate(0.50)
            .with_carbon_intensity(200.0);

        experiment.record_samples(1000, Duration::from_secs(60));
        let metrics = experiment.finalize();

        assert_eq!(metrics.samples_generated, 1000);
        assert!(metrics.samples_per_second > 10.0);
        assert!(metrics.cost_per_sample() > 0.0);
    }

    #[test]
    fn test_experiment_start_stop() {
        let device = ComputeDevice::default();
        let mut experiment = GenerationExperiment::new("timed-run", device);

        experiment.start();
        std::thread::sleep(Duration::from_millis(10));
        experiment.stop();

        assert!(experiment.total_duration.as_millis() >= 10);
    }

    #[test]
    fn test_compute_device_default() {
        let device = ComputeDevice::default();
        match device {
            ComputeDevice::Cpu { cores, .. } => assert!(cores > 0),
            _ => panic!("Expected CPU device"),
        }
    }

    #[test]
    fn test_energy_metrics_default() {
        let energy = EnergyMetrics::default();
        assert_eq!(energy.total_joules, 0.0);
        assert_eq!(energy.pue, 1.0);
    }

    #[test]
    fn test_cost_metrics_default() {
        let cost = CostMetrics::default();
        assert_eq!(cost.total_cost_usd, 0.0);
        assert_eq!(cost.currency, "USD");
    }

    #[test]
    fn test_tpu_device() {
        let device = ComputeDevice::Tpu {
            version: TpuVersion::V4,
            cores: 4,
        };
        assert!(device.theoretical_flops() > 1e15);
    }

    #[test]
    fn test_edge_device() {
        let device = ComputeDevice::Edge {
            name: "Jetson Nano".to_string(),
            power_budget_watts: 10.0,
        };
        assert_eq!(device.estimated_power_watts(), 10.0);
    }

    #[test]
    fn test_experiment_metrics_co2_per_sample() {
        let device = ComputeDevice::default_cpu();
        let mut experiment = GenerationExperiment::new("co2-test", device);
        experiment.record_samples(100, Duration::from_secs(10));
        let metrics = experiment.finalize();

        assert!(metrics.co2_per_sample() >= 0.0);
    }

    #[test]
    fn test_experiment_metrics_serialization() {
        let device = ComputeDevice::default_cpu();
        let mut experiment = GenerationExperiment::new("serial-test", device);
        experiment.record_samples(50, Duration::from_secs(5));
        let metrics = experiment.finalize();

        let json = serde_json::to_string(&metrics).expect("serialization");
        assert!(json.contains("serial-test"));

        let parsed: ExperimentMetrics = serde_json::from_str(&json).expect("deserialization");
        assert_eq!(parsed.samples_generated, 50);
    }
}
