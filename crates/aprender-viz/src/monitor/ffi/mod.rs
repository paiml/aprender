//! FFI Module - Native Platform Integration
//!
//! This module provides safe wrappers around platform-specific APIs:
//! - WGPU: Pure safe Rust multi-GPU monitoring (RECOMMENDED)
//! - IOKit: macOS GPU/Afterburner monitoring (requires unsafe)
//!
//! ## Safety Philosophy
//!
//! Following Toyota Production System principles:
//! - **Jidoka**: Errors convert to Result, never panic
//! - **Poka-Yoke**: Type-safe wrappers prevent invalid states
//! - **Muda**: Zero-copy where possible
//!
//! ## Module Organization
//!
//! ```text
//! ffi/
//! ├── mod.rs          # This file - module root
//! ├── wgpu.rs         # WGPU multi-GPU (100% safe Rust)
//! ├── iokit.rs        # macOS IOKit GPU (unsafe, isolated)
//! └── afterburner.rs  # Apple Afterburner FPGA (unsafe, isolated)
//! ```

// WGPU is 100% safe Rust - no unsafe allowed
#[cfg(feature = "gpu-wgpu")]
pub mod wgpu;

// IOKit requires unsafe FFI - isolated here
#[cfg(all(feature = "gpu-iokit", target_os = "macos"))]
pub mod iokit;

// Afterburner requires unsafe FFI - isolated here
#[cfg(all(feature = "afterburner", target_os = "macos"))]
pub mod afterburner;

// Re-exports for convenience
#[cfg(feature = "gpu-wgpu")]
pub use self::wgpu::{AdapterLimits, GpuAdapterInfo, WgpuBackendType, WgpuMetrics, WgpuMonitor};
