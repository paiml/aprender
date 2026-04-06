//! Disk monitoring panel.
//!
//! Displays disk I/O metrics and mount point usage.

use crate::monitor::collectors::DiskCollector;

/// Panel for disk metrics visualization.
#[derive(Debug)]
pub struct DiskPanel {
    /// Disk collector.
    pub collector: DiskCollector,
}

impl DiskPanel {
    /// Creates a new disk panel.
    #[must_use]
    pub fn new() -> Self {
        Self { collector: DiskCollector::new() }
    }
}

impl Default for DiskPanel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disk_panel_new() {
        let panel = DiskPanel::new();
        assert!(panel.collector.mounts().is_empty());
    }

    #[test]
    fn test_disk_panel_default() {
        let panel = DiskPanel::default();
        assert!(panel.collector.mounts().is_empty());
    }
}
