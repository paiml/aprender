//! SIMD-accelerated parameter update operations via Trueno
//!
//! This module provides vectorized implementations of common optimizer update
//! operations using Trueno's multi-backend SIMD support. These functions can
//! provide significant speedup for large parameter tensors.

mod adam;
mod adamw;
mod axpy;

#[cfg(test)]
mod tests;

// Re-export all public functions
pub use adam::simd_adam_update;
pub use adamw::simd_adamw_update;
pub use axpy::simd_axpy;
