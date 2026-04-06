//! Backend profilers for all compute modalities.
//! Wraps ncu, nsys, CUPTI, perf stat, wgpu timestamps, wasmtime, and more.

pub mod cuda;
pub mod neon;
pub mod quant;
pub mod rayon_parallel;
pub mod scalar;
pub mod simd;
pub mod system;
pub mod wasm;
pub mod wgpu_profiler;
