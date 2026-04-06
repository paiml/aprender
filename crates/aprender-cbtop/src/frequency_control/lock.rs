//! RAII frequency lock guard for deterministic benchmarks.

use std::path::PathBuf;

use super::controller::FrequencyController;
use super::governor::CpuGovernor;
use super::sysfs::governor_path;

/// RAII frequency lock guard
#[derive(Debug)]
pub struct FrequencyLock {
    /// Original governor per CPU
    original_governors: Vec<(usize, CpuGovernor)>,
    /// Original frequencies per CPU
    original_frequencies: Vec<(usize, u64)>,
    /// Controller reference path
    sysfs_path: PathBuf,
    /// Was lock successful
    locked: bool,
    /// Mock mode
    mock_mode: bool,
}

impl FrequencyLock {
    /// Try to lock frequency to maximum
    pub fn try_lock(controller: &FrequencyController) -> Self {
        let reading = controller.read_all_frequencies();

        let (original_governors, original_frequencies): (Vec<_>, Vec<_>) = reading
            .cpus
            .iter()
            .map(|c| ((c.cpu_id, c.governor), (c.cpu_id, c.current_khz)))
            .unzip();

        let mut lock = Self {
            original_governors,
            original_frequencies,
            sysfs_path: controller.sysfs_path.clone(),
            locked: false,
            mock_mode: controller.mock_mode,
        };

        if controller.mock_mode {
            lock.locked = true;
        } else {
            lock.locked = lock.try_set_performance();
        }

        lock
    }

    /// Try to set performance governor on all CPUs
    fn try_set_performance(&self) -> bool {
        for (cpu_id, _) in &self.original_governors {
            if std::fs::write(governor_path(&self.sysfs_path, *cpu_id), "performance").is_err() {
                return false;
            }
        }
        true
    }

    /// Check if lock was successful
    pub fn is_locked(&self) -> bool {
        self.locked
    }
}

impl Drop for FrequencyLock {
    fn drop(&mut self) {
        if !self.locked || self.mock_mode {
            return;
        }

        // Restore original governors
        for (cpu_id, governor) in &self.original_governors {
            let _ = std::fs::write(governor_path(&self.sysfs_path, *cpu_id), governor.name());
        }
    }
}
