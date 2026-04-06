//! SPEC-024: Interface-Defining Tests for ptop App Module
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**
//!
//! These tests define the REQUIRED interface for ptop's core types.
//! If a test doesn't compile, the interface is missing.
//! If a test fails, the implementation is wrong.

#![cfg(feature = "ptop")]

use presentar_terminal::ptop::app::MetricsSnapshot;
use presentar_terminal::ptop::{App, PanelType};
use presentar_terminal::Snapshot;

// =============================================================================
// SECTION 1: MetricsSnapshot Interface
// =============================================================================

mod metrics_snapshot {
    use super::*;

    /// MetricsSnapshot MUST have cpu_avg field
    #[test]
    fn has_cpu_avg() {
        let s = MetricsSnapshot::empty();
        let _: f64 = s.cpu_avg;
    }

    /// MetricsSnapshot MUST have per_core_percent field
    #[test]
    fn has_per_core_percent() {
        let s = MetricsSnapshot::empty();
        let _: &Vec<f64> = &s.per_core_percent;
    }

    /// MetricsSnapshot MUST have per_core_freq field (SPEC-024 async requirement)
    #[test]
    fn has_per_core_freq() {
        let s = MetricsSnapshot::empty();
        let _: &Vec<u64> = &s.per_core_freq;
    }

    /// MetricsSnapshot MUST have per_core_temp field (SPEC-024 async requirement)
    #[test]
    fn has_per_core_temp() {
        let s = MetricsSnapshot::empty();
        let _: &Vec<f32> = &s.per_core_temp;
    }

    /// MetricsSnapshot MUST have memory fields
    #[test]
    fn has_memory_fields() {
        let s = MetricsSnapshot::empty();
        let _: u64 = s.mem_total;
        let _: u64 = s.mem_used;
        let _: u64 = s.mem_available;
        let _: u64 = s.mem_cached;
        let _: u64 = s.swap_total;
        let _: u64 = s.swap_used;
    }

    /// MetricsSnapshot MUST have load_avg field
    #[test]
    fn has_load_avg() {
        let s = MetricsSnapshot::empty();
        let _: f64 = s.load_avg.one;
        let _: f64 = s.load_avg.five;
        let _: f64 = s.load_avg.fifteen;
    }

    /// MetricsSnapshot MUST implement Snapshot trait
    #[test]
    fn implements_snapshot_trait() {
        fn assert_snapshot<T: Snapshot>() {}
        assert_snapshot::<MetricsSnapshot>();
    }

    /// MetricsSnapshot::empty() MUST return valid empty state
    #[test]
    fn empty_is_valid() {
        let s = MetricsSnapshot::empty();
        assert!(s.per_core_percent.is_empty() || s.per_core_percent.iter().all(|&v| v == 0.0));
        assert!(s.per_core_freq.is_empty() || s.per_core_freq.iter().all(|&v| v == 0));
        assert!(s.per_core_temp.is_empty() || s.per_core_temp.iter().all(|&v| v == 0.0));
    }
}

// =============================================================================
// SECTION 2: App Interface
// =============================================================================

mod app {
    use super::*;

    /// App MUST be constructable with config
    #[test]
    fn constructable() {
        let _app = App::with_config(true, Default::default());
    }

    /// App MUST have per_core_freq field (mirrors MetricsSnapshot)
    #[test]
    fn has_per_core_freq() {
        let app = App::with_config(true, Default::default());
        let _: &Vec<u64> = &app.per_core_freq;
    }

    /// App MUST have per_core_temp field (mirrors MetricsSnapshot)
    #[test]
    fn has_per_core_temp() {
        let app = App::with_config(true, Default::default());
        let _: &Vec<f32> = &app.per_core_temp;
    }

    /// App MUST have per_core_percent field
    #[test]
    fn has_per_core_percent() {
        let app = App::with_config(true, Default::default());
        let _: &Vec<f64> = &app.per_core_percent;
    }

    /// App MUST have exploded_panel field for panel expansion
    #[test]
    fn has_exploded_panel() {
        let mut app = App::with_config(true, Default::default());
        app.exploded_panel = Some(PanelType::Cpu);
        assert_eq!(app.exploded_panel, Some(PanelType::Cpu));
    }

    /// App MUST have apply_snapshot method
    #[test]
    fn has_apply_snapshot() {
        let mut app = App::with_config(true, Default::default());
        let snapshot = MetricsSnapshot::empty();
        app.apply_snapshot(snapshot);
    }

    /// App MUST have collect_metrics method
    #[test]
    fn has_collect_metrics() {
        let mut app = App::with_config(true, Default::default());
        app.collect_metrics();
    }

    /// App MUST have deterministic field
    #[test]
    fn has_deterministic() {
        let app = App::with_config(true, Default::default());
        assert!(app.deterministic);
    }
}

// =============================================================================
// SECTION 3: Async Data Flow Interface
// =============================================================================

mod async_data_flow {
    use super::*;

    /// apply_snapshot MUST transfer per_core_freq
    #[test]
    fn transfers_per_core_freq() {
        let mut app = App::with_config(true, Default::default());

        let mut snapshot = MetricsSnapshot::empty();
        snapshot.per_core_freq = vec![4500, 4600, 4700];

        app.apply_snapshot(snapshot);

        assert_eq!(app.per_core_freq, vec![4500, 4600, 4700]);
    }

    /// apply_snapshot MUST transfer per_core_temp
    #[test]
    fn transfers_per_core_temp() {
        let mut app = App::with_config(true, Default::default());

        let mut snapshot = MetricsSnapshot::empty();
        snapshot.per_core_temp = vec![65.0, 68.0, 70.0];

        app.apply_snapshot(snapshot);

        assert!((app.per_core_temp[0] - 65.0).abs() < 0.1);
        assert!((app.per_core_temp[1] - 68.0).abs() < 0.1);
        assert!((app.per_core_temp[2] - 70.0).abs() < 0.1);
    }

    /// apply_snapshot MUST transfer cpu_avg
    #[test]
    fn transfers_cpu_avg() {
        let mut app = App::with_config(true, Default::default());

        let mut snapshot = MetricsSnapshot::empty();
        snapshot.cpu_avg = 0.75;

        app.apply_snapshot(snapshot);

        // cpu_avg is pushed to history, check history has data
        // RingBuffer uses as_slice() to access data
        assert!(!app.cpu_history.as_slice().is_empty());
    }

    /// apply_snapshot MUST transfer memory fields
    #[test]
    fn transfers_memory() {
        let mut app = App::with_config(true, Default::default());

        let mut snapshot = MetricsSnapshot::empty();
        snapshot.mem_total = 16_000_000_000;
        snapshot.mem_used = 8_000_000_000;

        app.apply_snapshot(snapshot);

        assert_eq!(app.mem_total, 16_000_000_000);
        assert_eq!(app.mem_used, 8_000_000_000);
    }

    /// Values MUST change between async updates
    #[test]
    fn values_change_between_updates() {
        let mut app = App::with_config(true, Default::default());

        let mut s1 = MetricsSnapshot::empty();
        s1.per_core_freq = vec![4500];
        app.apply_snapshot(s1);
        let v1 = app.per_core_freq[0];

        let mut s2 = MetricsSnapshot::empty();
        s2.per_core_freq = vec![4800];
        app.apply_snapshot(s2);
        let v2 = app.per_core_freq[0];

        assert_ne!(v1, v2);
    }
}

// =============================================================================
// SECTION 4: Panel Types Interface
// =============================================================================

mod panel_types {
    use super::*;

    /// PanelType MUST have all required variants
    #[test]
    fn has_all_variants() {
        let panels = [
            PanelType::Cpu,
            PanelType::Memory,
            PanelType::Disk,
            PanelType::Network,
            PanelType::Process,
            PanelType::Gpu,
            PanelType::Sensors,
            PanelType::Connections,
            PanelType::Psi,
            PanelType::Files,
            PanelType::Battery,
            PanelType::Containers,
        ];
        assert_eq!(panels.len(), 12);
    }

    /// PanelType MUST implement required traits
    #[test]
    fn implements_traits() {
        fn assert_traits<T: Clone + Copy + PartialEq + Eq + std::fmt::Debug>() {}
        assert_traits::<PanelType>();
    }
}

// =============================================================================
// SECTION 5: MetricsCollector Interface
// =============================================================================

mod metrics_collector {
    use super::*;
    use presentar_terminal::ptop::app::MetricsCollector;
    use presentar_terminal::AsyncCollector;

    /// MetricsCollector MUST be constructable
    #[test]
    fn constructable() {
        let _collector = MetricsCollector::new(true); // deterministic
    }

    /// MetricsCollector MUST implement AsyncCollector trait
    #[test]
    fn implements_async_collector() {
        fn assert_async_collector<T: AsyncCollector<Snapshot = MetricsSnapshot>>() {}
        assert_async_collector::<MetricsCollector>();
    }

    /// MetricsCollector::collect MUST return MetricsSnapshot
    #[test]
    fn collect_returns_snapshot() {
        let mut collector = MetricsCollector::new(true);
        let snapshot: MetricsSnapshot = collector.collect();
        let _ = snapshot.cpu_avg; // Verify it's a real MetricsSnapshot
    }

    /// MetricsCollector MUST populate per_core_freq in non-deterministic mode
    #[test]
    fn populates_freq_in_real_mode() {
        let mut collector = MetricsCollector::new(false); // real mode
        let snapshot = collector.collect();

        // On real systems, should have frequency data
        // (empty is ok for CI/sandboxed environments)
        if !snapshot.per_core_freq.is_empty() {
            assert!(snapshot.per_core_freq.iter().any(|&f| f > 0));
        }
    }
}
