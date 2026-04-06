//! Panel implementations for renacer visualize
//!
//! Each panel provides a specific view of tracing data:
//! - syscall_heatmap: Syscall category activity over time
//! - anomaly_timeline: Z-score anomaly visualization
//! - ml_scatter: ML clustering visualization (braille scatter)
//! - trace_waterfall: OTLP span Gantt chart
//! - process_syscalls: Per-process syscall breakdown
//! - stats_summary: Aggregate statistics
//! - metrics: Counter/gauge/histogram visualization (Sprint 56)
//! - alerts: Real-time alerting panel (Sprint 56)

pub mod alerts;
pub mod anomaly_timeline;
pub mod metrics;
pub mod ml_scatter;
pub mod process_syscalls;
pub mod stats_summary;
pub mod syscall_heatmap;
pub mod trace_waterfall;

/// Panel identifier for type-safe operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanelId {
    SyscallHeatmap,
    AnomalyTimeline,
    MlScatter,
    TraceWaterfall,
    ProcessSyscalls,
    StatsSummary,
    Metrics,
    Alerts,
}

impl PanelId {
    /// Get display name for panel
    pub fn name(&self) -> &'static str {
        match self {
            Self::SyscallHeatmap => "Syscalls",
            Self::AnomalyTimeline => "Anomalies",
            Self::MlScatter => "ML Clusters",
            Self::TraceWaterfall => "Trace",
            Self::ProcessSyscalls => "Processes",
            Self::StatsSummary => "Stats",
            Self::Metrics => "Metrics",
            Self::Alerts => "Alerts",
        }
    }

    /// Get keyboard shortcut
    pub fn shortcut(&self) -> char {
        match self {
            Self::SyscallHeatmap => '1',
            Self::AnomalyTimeline => '2',
            Self::MlScatter => '3',
            Self::TraceWaterfall => '4',
            Self::ProcessSyscalls => '5',
            Self::StatsSummary => '6',
            Self::Metrics => '7',
            Self::Alerts => '8',
        }
    }

    /// Get all panel IDs
    pub fn all() -> &'static [PanelId] {
        &[
            Self::SyscallHeatmap,
            Self::AnomalyTimeline,
            Self::MlScatter,
            Self::TraceWaterfall,
            Self::ProcessSyscalls,
            Self::StatsSummary,
            Self::Metrics,
            Self::Alerts,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_id_names() {
        assert_eq!(PanelId::SyscallHeatmap.name(), "Syscalls");
        assert_eq!(PanelId::AnomalyTimeline.name(), "Anomalies");
        assert_eq!(PanelId::MlScatter.name(), "ML Clusters");
        assert_eq!(PanelId::TraceWaterfall.name(), "Trace");
        assert_eq!(PanelId::ProcessSyscalls.name(), "Processes");
        assert_eq!(PanelId::StatsSummary.name(), "Stats");
        assert_eq!(PanelId::Metrics.name(), "Metrics");
        assert_eq!(PanelId::Alerts.name(), "Alerts");
    }

    #[test]
    fn test_panel_id_shortcuts() {
        assert_eq!(PanelId::SyscallHeatmap.shortcut(), '1');
        assert_eq!(PanelId::AnomalyTimeline.shortcut(), '2');
        assert_eq!(PanelId::MlScatter.shortcut(), '3');
        assert_eq!(PanelId::TraceWaterfall.shortcut(), '4');
        assert_eq!(PanelId::ProcessSyscalls.shortcut(), '5');
        assert_eq!(PanelId::StatsSummary.shortcut(), '6');
        assert_eq!(PanelId::Metrics.shortcut(), '7');
        assert_eq!(PanelId::Alerts.shortcut(), '8');
    }

    #[test]
    fn test_panel_id_all() {
        let all = PanelId::all();
        assert_eq!(all.len(), 8);
        assert!(all.contains(&PanelId::SyscallHeatmap));
        assert!(all.contains(&PanelId::Metrics));
        assert!(all.contains(&PanelId::Alerts));
    }

    #[test]
    fn test_panel_id_eq_hash() {
        use std::collections::HashSet;

        let mut set: HashSet<PanelId> = HashSet::new();
        set.insert(PanelId::SyscallHeatmap);
        set.insert(PanelId::Metrics);
        set.insert(PanelId::Alerts);

        assert!(set.contains(&PanelId::SyscallHeatmap));
        assert!(set.contains(&PanelId::Metrics));
        assert!(!set.contains(&PanelId::MlScatter));
    }

    #[test]
    fn test_panel_id_clone() {
        let panel = PanelId::AnomalyTimeline;
        let cloned = panel;
        assert_eq!(panel, cloned);
    }

    #[test]
    fn test_panel_id_debug() {
        let panel = PanelId::Metrics;
        let debug_str = format!("{:?}", panel);
        assert!(debug_str.contains("Metrics"));
    }
}
