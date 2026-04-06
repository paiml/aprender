//! Detached TUI Monitor (SPEC-FT-001 Section 10)
//!
//! Implements the producer-consumer pattern for real-time training visualization.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐          ┌──────────────┐
//! │  Trainer     │──write──▶│  Metric      │◀──read──│  TUI Monitor │
//! │  (Producer)  │          │  Store (IPC) │          │  (Consumer)  │
//! └──────────────┘          └──────────────┘          └──────────────┘
//! ```
//!
//! The TUI runs in a separate process/shell, reading state without blocking training.
//!
//! # Usage
//!
//! ```bash
//! # Shell 1: Start training (writes to metric store)
//! cargo run --example finetune_test_gen -- --output ./experiments/ft-001
//!
//! # Shell 2: Attach TUI monitor (reads from metric store)
//! cargo run --example finetune_test_gen -- --monitor --experiment ./experiments/ft-001
//! ```
//!
//! # Toyota Way: Andon (アンドン)
//!
//! Visual alerting system for immediate problem detection.
//! Loss spikes, OOM warnings, and gradient explosions trigger visual alerts.

pub mod app;
pub mod color;
pub mod dashboard;
pub mod headless;
pub mod panel;
pub mod render;
pub mod state;

pub use app::{TrainingStateWriter, TuiMonitor, TuiMonitorConfig};
pub use color::{colored_bar, colored_value, ColorMode, Rgb, Styled, TrainingPalette};
pub use dashboard::TrainingDashboard;
pub use headless::{
    HeadlessGpu, HeadlessMonitor, HeadlessOutput, HeadlessSample, HeadlessWriter, OutputFormat,
};
pub use panel::{
    layout_can_render, verify_layout, GpuPanel, LossCurvePanel, MetricsPanel, Panel,
    PanelVerification, ProcessPanel, SamplePanel,
};
pub use render::{
    render_braille_chart, render_gauge, render_layout, render_layout_colored, BrailleChart,
};
pub use state::{
    GpuProcessInfo, GpuTelemetry, LossTrend, SamplePeek, TrainingSnapshot, TrainingState,
    TrainingStatus,
};
