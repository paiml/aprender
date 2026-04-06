//! Compute Device Abstraction (ENT-008)
//!
//! Provides hardware device detection and abstraction for CPU, GPU, TPU,
//! and Apple Silicon devices.

mod apple;
mod compute;
mod cpu;
mod gpu;
mod simd;
mod tpu;

#[cfg(test)]
mod tests;

// Re-export all public types for API compatibility
pub use apple::AppleSiliconInfo;
pub use compute::ComputeDevice;
pub use cpu::CpuInfo;
pub use gpu::GpuInfo;
pub use simd::SimdCapability;
pub use tpu::TpuInfo;
