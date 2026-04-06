//! GPU Monitoring Module (MLOPS-005)
//!
//! btop-inspired GPU monitoring for terminal training dashboard.
//!
//! # Toyota Way: Andon
//!
//! Visual alerting system for immediate problem detection.
//! Thermal throttling, memory pressure, and power limits trigger alerts.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::monitor::gpu::{GpuMonitor, GpuMetrics, GpuAlert};
//!
//! let monitor = GpuMonitor::new()?;
//! let metrics = monitor.sample();
//! for m in &metrics {
//!     println!("GPU {}: {}°C, {}% util", m.device_id, m.temperature_celsius, m.utilization_percent);
//! }
//! ```

mod alert;
mod buffer;
mod monitor;
mod render;
mod types;

pub use alert::{AndonThresholds, GpuAlert, GpuAndonSystem};
pub use buffer::GpuMetricsBuffer;
pub use monitor::GpuMonitor;
pub use render::{format_gpu_panel, render_progress_bar, render_sparkline};
pub use types::{GpuMetrics, GpuProcess};
