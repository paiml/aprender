//! Load generator bricks (Layer 4)
//!
//! Generate controlled compute workloads.

mod cuda;
mod memory;
mod simd;
mod wgpu;

pub use cuda::CudaLoadBrick;
pub use memory::MemBandwidthBrick;
pub use simd::SimdLoadBrick;
pub use wgpu::WgpuLoadBrick;
