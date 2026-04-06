//! High-level panel components for the monitoring TUI.
//!
//! Each panel combines widgets with collectors to display a specific
//! category of metrics.

pub mod cpu;
pub mod disk;
pub mod memory;
pub mod network;
pub mod process;

pub use cpu::CpuPanel;
pub use disk::DiskPanel;
pub use memory::MemoryPanel;
pub use network::NetworkPanel;
pub use process::ProcessPanel;
