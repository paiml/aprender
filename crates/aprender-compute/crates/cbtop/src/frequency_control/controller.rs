use std::path::PathBuf;

use super::governor::CpuGovernor;
use super::info::CpuFrequencyInfo;
use super::reading::FrequencyReading;
use super::variance::FrequencyVariance;

/// Frequency controller for reading and managing CPU frequencies
#[derive(Debug)]
pub struct FrequencyController {
    /// Sysfs base path
    pub(crate) sysfs_path: PathBuf,
    /// Number of CPUs
    cpu_count: usize,
    /// Mock mode for testing
    pub(crate) mock_mode: bool,
    /// Mock frequency value
    mock_frequency: u64,
    /// Mock governor
    mock_governor: CpuGovernor,
}

impl Default for FrequencyController {
    fn default() -> Self {
        Self::new()
    }
}

impl FrequencyController {
    /// Create new controller
    pub fn new() -> Self {
        let cpu_count = super::num_cpus();
        Self {
            sysfs_path: PathBuf::from("/sys/devices/system/cpu"),
            cpu_count,
            mock_mode: false,
            mock_frequency: 3_000_000, // 3 GHz default
            mock_governor: CpuGovernor::Performance,
        }
    }

    /// Enable mock mode for testing
    pub fn with_mock(mut self, frequency_khz: u64, governor: CpuGovernor) -> Self {
        self.mock_mode = true;
        self.mock_frequency = frequency_khz;
        self.mock_governor = governor;
        self
    }

    /// Get CPU count
    pub fn cpu_count(&self) -> usize {
        self.cpu_count
    }

    /// Read frequency for single CPU
    pub fn read_cpu_frequency(&self, cpu_id: usize) -> Option<CpuFrequencyInfo> {
        Some(if self.mock_mode {
            CpuFrequencyInfo::mock(cpu_id, self.mock_frequency, self.mock_governor)
        } else {
            CpuFrequencyInfo::from_sysfs(cpu_id, &self.sysfs_path)
        })
    }

    /// Read all CPU frequencies
    pub fn read_all_frequencies(&self) -> FrequencyReading {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let cpus: Vec<CpuFrequencyInfo> =
            (0..self.cpu_count).filter_map(|id| self.read_cpu_frequency(id)).collect();

        FrequencyReading { cpus, timestamp_ns }
    }

    /// Detect current governor
    pub fn detect_governor(&self) -> CpuGovernor {
        self.read_all_frequencies().common_governor()
    }

    /// Check if frequency can be controlled
    pub fn can_control(&self) -> bool {
        if self.mock_mode {
            return true;
        }

        let reading = self.read_all_frequencies();
        if reading.cpus.is_empty() {
            return false;
        }

        let govs = &reading.cpus[0].available_governors;
        govs.contains(&CpuGovernor::Userspace) || govs.contains(&CpuGovernor::Performance)
    }

    /// Measure frequency variance over time
    pub fn measure_variance(&self, samples: usize, interval_ms: u64) -> FrequencyVariance {
        let mut readings = Vec::with_capacity(samples);
        for _ in 0..samples {
            readings.push(self.read_all_frequencies().average_mhz());
            std::thread::sleep(std::time::Duration::from_millis(interval_ms));
        }
        FrequencyVariance::from_samples(&readings)
    }
}
