//! TUI Layout Specification (TRUENO-SPEC-024)
//!
//! Terminal user interface layout and widget definitions for
//! real-time compute monitoring.
//!
//! # Layout
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ TRUENO Compute Monitor │ CPU: ... │ GPU: ... │ F1 Help                     │
//! ├────────────────────────────────────────────────────────────────────────────┤
//! │ [COMPUTE] CPU/GPU utilization gauges + sparklines                          │
//! │ [MEMORY] RAM/SWAP/VRAM bars                                                 │
//! │ [DATA FLOW] PCIe TX/RX + transfers                                          │
//! │ [KERNELS] Active kernel list                                                │
//! ├────────────────────────────────────────────────────────────────────────────┤
//! │ q:Quit r:Refresh s:Stress Tab:Focus │ Refresh: 100ms                        │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - [Wang2004] SSIM for visual quality
//! - Viridis colormap for colorblind accessibility

mod colors;
mod config;
mod controls;
mod render_state;
mod widgets;

#[cfg(test)]
mod tests;

pub use colors::{ColorScheme, RgbColor};
pub use config::{Section, SizeCheck, TuiLayout};
pub use controls::KeyAction;
pub use render_state::{
    DataFlowRenderState, DeviceRenderState, KernelRenderState, MemoryRenderState, TuiRenderState,
};
pub use widgets::{
    GaugeColor, GaugeWidget, ProgressBarWidget, SparklineWidget, TableWidget, TextStyle,
    TextWidget, Widget,
};
