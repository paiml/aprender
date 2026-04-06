//! Network monitoring panel.
//!
//! Displays network throughput and interface statistics.

use crate::monitor::collectors::NetworkCollector;

/// Panel for network metrics visualization.
#[derive(Debug)]
pub struct NetworkPanel {
    /// Network collector.
    pub collector: NetworkCollector,
}

impl NetworkPanel {
    /// Creates a new network panel.
    #[must_use]
    pub fn new() -> Self {
        Self { collector: NetworkCollector::new() }
    }
}

impl Default for NetworkPanel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_panel_new() {
        let panel = NetworkPanel::new();
        assert!(panel.collector.interfaces().is_empty());
    }

    #[test]
    fn test_network_panel_default() {
        let panel = NetworkPanel::default();
        assert!(panel.collector.interfaces().is_empty());
    }
}
