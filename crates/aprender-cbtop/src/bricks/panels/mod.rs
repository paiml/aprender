//! Panel bricks (Layer 3)
//!
//! Rendering using presentar-terminal widgets.

pub mod config;
pub mod cpu;
pub mod gpu;
pub mod help;
pub mod load;
pub mod memory;
pub mod overview;
pub mod pcie;
pub mod thermal;

pub use config::ConfigPanelBrick;
pub use cpu::CpuPanelBrick;
pub use gpu::GpuPanelBrick;
pub use help::HelpPanelBrick;
pub use load::LoadControlPanelBrick;
pub use memory::MemoryPanelBrick;
pub use overview::OverviewPanelBrick;
pub use pcie::PciePanelBrick;
pub use thermal::ThermalPanelBrick;
