//! wgpu-accelerated quantized model — re-export shim.
//!
//! Mirrors `cuda_model.rs` for the wgpu backend. Provides
//! GPU-accelerated inference for quantized models on non-NVIDIA
//! hardware (Apple Silicon Metal, AMD via Vulkan, Intel ARC via
//! Vulkan, etc) per CLAUDE.md backend-agnostic mandate.
//!
//! See `wgpu_backend/mod.rs` for the concrete struct definition;
//! this file is the thin public-API re-export so consumers can
//! `use realizar::gguf::OwnedQuantizedModelWgpu;`.

#[cfg(feature = "gpu")]
pub use super::wgpu_backend::OwnedQuantizedModelWgpu;
