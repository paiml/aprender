use super::governor::CpuGovernor;
use super::info::CpuFrequencyInfo;

/// Frequency reading result
#[derive(Debug, Clone)]
pub struct FrequencyReading {
    /// CPU info for each core
    pub cpus: Vec<CpuFrequencyInfo>,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl FrequencyReading {
    /// Get average frequency in MHz
    pub fn average_mhz(&self) -> f64 {
        if self.cpus.is_empty() {
            return 0.0;
        }
        let total: f64 = self.cpus.iter().map(|c| c.current_mhz()).sum();
        total / self.cpus.len() as f64
    }

    /// Get min frequency across cores in MHz
    pub fn min_mhz(&self) -> f64 {
        self.extremum_mhz(f64::min)
    }

    /// Get max frequency across cores in MHz
    pub fn max_mhz(&self) -> f64 {
        self.extremum_mhz(f64::max)
    }

    /// Get frequency variance in MHz
    pub fn variance_mhz(&self) -> f64 {
        self.max_mhz() - self.min_mhz()
    }

    /// Shared extremum helper (min or max across core frequencies).
    fn extremum_mhz(&self, cmp: fn(f64, f64) -> f64) -> f64 {
        self.cpus
            .iter()
            .map(|c| c.current_mhz())
            .reduce(cmp)
            .unwrap_or(0.0)
    }

    /// Check if all cores have same governor
    pub fn uniform_governor(&self) -> bool {
        if self.cpus.is_empty() {
            return true;
        }
        let first = self.cpus[0].governor;
        self.cpus.iter().all(|c| c.governor == first)
    }

    /// Get most common governor
    pub fn common_governor(&self) -> CpuGovernor {
        self.cpus
            .first()
            .map(|c| c.governor)
            .unwrap_or(CpuGovernor::Unknown)
    }
}
