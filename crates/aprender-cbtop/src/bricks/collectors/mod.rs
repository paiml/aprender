//! Collector bricks (Layer 1)
//!
//! Data collection from hardware (Genchi Genbutsu).

pub mod cpu;
pub mod gpu;
pub mod memory;
pub mod pcie;
pub mod pepita;
pub mod thermal;
pub mod wos;
pub mod zram;

pub use cpu::CpuCollectorBrick;
pub use gpu::GpuCollectorBrick;
pub use memory::MemoryCollectorBrick;
pub use pcie::PcieCollectorBrick;
pub use pepita::{IoUringMetrics, PepitaCollectorBrick};
pub use thermal::ThermalCollectorBrick;
pub use wos::{JidokaHealthStatus, WosCollectorBrick, WosKernelMetrics};
pub use zram::{ZramAlgorithm, ZramCollectorBrick, ZramMetrics};
