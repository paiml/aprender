use super::governor::CpuGovernor;
use super::sysfs::{read_sysfs_string, read_sysfs_value};

/// CPU frequency info for a single CPU
#[derive(Debug, Clone)]
pub struct CpuFrequencyInfo {
    /// CPU ID
    pub cpu_id: usize,
    /// Current frequency in kHz
    pub current_khz: u64,
    /// Minimum frequency in kHz
    pub min_khz: u64,
    /// Maximum frequency in kHz
    pub max_khz: u64,
    /// Current governor
    pub governor: CpuGovernor,
    /// Available governors
    pub available_governors: Vec<CpuGovernor>,
}

impl CpuFrequencyInfo {
    /// Convert kHz to MHz.
    fn khz_to_mhz(khz: u64) -> f64 {
        khz as f64 / 1_000.0
    }

    /// Get current frequency in MHz
    pub fn current_mhz(&self) -> f64 {
        Self::khz_to_mhz(self.current_khz)
    }

    /// Get current frequency in GHz
    pub fn current_ghz(&self) -> f64 {
        Self::khz_to_mhz(self.current_khz) / 1_000.0
    }

    /// Get frequency utilization (current / max)
    pub fn utilization(&self) -> f64 {
        if self.max_khz > 0 {
            self.current_khz as f64 / self.max_khz as f64
        } else {
            1.0
        }
    }

    /// Build a mock CpuFrequencyInfo for testing.
    pub(crate) fn mock(cpu_id: usize, frequency_khz: u64, governor: CpuGovernor) -> Self {
        Self {
            cpu_id,
            current_khz: frequency_khz,
            min_khz: 800_000,
            max_khz: 4_000_000,
            governor,
            available_governors: vec![CpuGovernor::Performance, CpuGovernor::Powersave],
        }
    }

    /// Read CpuFrequencyInfo from sysfs for the given CPU.
    pub(crate) fn from_sysfs(cpu_id: usize, sysfs_path: &std::path::Path) -> Self {
        let cpu_path = sysfs_path.join(format!("cpu{cpu_id}/cpufreq"));

        let read_khz = |name: &str| read_sysfs_value(&cpu_path.join(name)).unwrap_or(0);

        let governor = read_sysfs_string(&cpu_path.join("scaling_governor"))
            .map(|s| CpuGovernor::parse(&s))
            .unwrap_or(CpuGovernor::Unknown);

        let available_governors: Vec<CpuGovernor> =
            read_sysfs_string(&cpu_path.join("scaling_available_governors"))
                .unwrap_or_default()
                .split_whitespace()
                .map(CpuGovernor::parse)
                .collect();

        Self {
            cpu_id,
            current_khz: read_khz("scaling_cur_freq"),
            min_khz: read_khz("scaling_min_freq"),
            max_khz: read_khz("scaling_max_freq"),
            governor,
            available_governors,
        }
    }
}
