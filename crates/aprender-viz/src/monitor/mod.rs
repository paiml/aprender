//! TUI Monitoring System for trueno-viz.
//!
//! This module provides a btop-like terminal user interface for real-time system
//! and ML workload monitoring. It integrates with the Sovereign AI Stack components
//! (realizar, entrenar, trueno-zram, repartir) while also providing standard system
//! metrics (CPU, memory, disk, network, processes).
//!
//! # Features
//!
//! - **Pure Rust**: No C/C++ dependencies, WASM-compatible core
//! - **YAML Configuration**: Declarative layout, theming, and metric selection
//! - **Multi-System**: Distributed monitoring over TCP/TLS with MessagePack
//! - **Stack Integration**: LLM inference, training, ZRAM compression metrics
//!
//! # Feature Flags
//!
//! - `monitor`: Core TUI with system metrics (CPU, memory, process)
//! - `monitor-nvidia`: NVIDIA GPU metrics via NVML
//! - `monitor-remote`: Multi-system monitoring with TCP/MessagePack
//! - `monitor-tls`: TLS encryption for remote connections
//! - `monitor-stack`: Sovereign AI Stack integration
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use trueno_viz::monitor::{App, Config};
//!
//! let config = Config::load("~/.config/trueno-monitor/config.yaml")?;
//! let mut app = App::new(config)?;
//! app.run()?;
//! ```
//!
//! # Performance Targets
//!
//! - Frame time: <16ms (60fps)
//! - Memory: <10MB total
//! - CPU: <5% at 1Hz refresh
//!
//! # References
//!
//! - Specification: `docs/specifications/tui-monitoring-spec.md`
//! - Falsification checklist: 100 testable criteria in specification

// ============================================================================
// Error Types
// ============================================================================

pub mod error;
pub use error::{MonitorError, Result};

// ============================================================================
// Core Types
// ============================================================================

pub mod debug;
pub mod ring_buffer;
pub mod simd;
pub mod subprocess;
pub mod types;

pub use ring_buffer::RingBuffer;
pub use simd::{SimdRingBuffer, SimdStats};
pub use subprocess::{run_with_timeout, run_with_timeout_stdout, SubprocessResult};
pub use types::{Collector, MetricValue, Metrics};

// ============================================================================
// Re-export ratatui for downstream crates
// This ensures trait compatibility when using widgets
// ============================================================================

pub use ratatui;

// ============================================================================
// Widgets
// ============================================================================

pub mod widgets;

// ============================================================================
// Collectors
// ============================================================================

pub mod collectors;

// ============================================================================
// Panels
// ============================================================================

pub mod panels;

// ============================================================================
// Configuration
// ============================================================================

pub mod config;
pub mod theme;

pub use config::Config;
pub use theme::Theme;

// ============================================================================
// Application
// ============================================================================

pub mod app;
pub mod input;
pub mod layout;
pub mod presets;
pub mod state;

pub use app::App;

// ============================================================================
// FFI - Native Platform Integration (Feature-Gated)
// ============================================================================

/// Native platform integration for GPU/accelerator monitoring.
///
/// Provides:
/// - WGPU: Pure safe Rust multi-GPU monitoring (recommended)
/// - IOKit: macOS GPU monitoring (uses unsafe FFI)
/// - Afterburner: Apple FPGA monitoring (uses unsafe FFI)
pub mod ffi;

// ============================================================================
// Multi-System Support (Feature-Gated)
// ============================================================================

#[cfg(feature = "monitor-remote")]
#[cfg_attr(docsrs, doc(cfg(feature = "monitor-remote")))]
pub mod remote;

// ============================================================================
// Prelude
// ============================================================================

/// Commonly used types for monitor functionality.
pub mod prelude {
    pub use super::app::App;
    pub use super::config::Config;
    pub use super::error::{MonitorError, Result};
    pub use super::ring_buffer::RingBuffer;
    pub use super::simd::{SimdRingBuffer, SimdStats};
    pub use super::theme::Theme;
    pub use super::types::{Collector, MetricValue, Metrics};

    // SIMD-accelerated collectors
    pub use super::collectors::{
        GpuMetricsSoA, SimdBatterySensorsCollector, SimdCpuCollector, SimdDiskCollector,
        SimdGpuHistory, SimdMemoryCollector, SimdNetworkCollector, SimdProcessCollector,
    };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    /// Smoke test to verify the monitor module compiles with feature flag.
    #[test]
    fn test_monitor_feature_compiles() {
        assert!(true, "Monitor module should compile");
    }

    /// Verify all public types are accessible.
    #[test]
    fn test_prelude_exports() {
        use super::prelude::*;

        // Just verify types exist - actual tests are in submodules
        let _ = std::any::type_name::<MonitorError>();
        let _ = std::any::type_name::<RingBuffer<f64>>();
    }
}
