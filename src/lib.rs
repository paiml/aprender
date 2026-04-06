//! # trueno-zram
//!
//! SIMD-accelerated LZ4/ZSTD compression engine for Linux zram devices.
//!
//! This is the workspace root crate that re-exports core functionality.
//! For direct usage, depend on individual sub-crates:
//!
//! - [`trueno-zram-core`] - Compression algorithms (LZ4, ZSTD, SIMD kernels)
//! - [`trueno-zram-adaptive`] - ML-driven algorithm selection
//! - [`trueno-zram-generator`] - systemd generator for zram configuration
//! - [`trueno-zram-cli`] - CLI tool (`trueno-zram` binary)
//!
//! ## Feature Flags
//!
//! - `std` (default) - Standard library support
//! - `cuda` - GPU-accelerated compression via CUDA
//! - `adaptive` - ML-driven compression selection

#[allow(clippy::doc_markdown)]
pub mod verification_specs;
