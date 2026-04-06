//! TDD Tests for GPU-Resident Tensor Architecture (WAPR-PERF-004)
//!
//! These tests are written FIRST (Test-Driven Development) to define the API
//! before implementation exists. Each test documents a specific requirement.
//!
//! ## Problem Statement
//!
//! Current approach: ~150 host↔device transfers per encoder forward pass
//! Target: 2 transfers total (upload weights once, download output once)
//!
//! ## Root Cause (Five Whys)
//!
//! 1. Why is GPU encoder slower than CPU? → 0.76x speedup (actually slower)
//! 2. Why is CUDA gemm not helping? → ~150 host↔device transfers per pass
//! 3. Why so many transfers? → Data ping-pongs: CPU→GPU for gemm, GPU→CPU for softmax
//! 4. Why not keep data on GPU? → No GPU-resident tensor abstraction
//! 5. What's the fix? → Build GpuResidentTensor into trueno-gpu
//!
//! ## Citations
//!
//! - [Dao2022] FlashAttention: Fast and Memory-Efficient Exact Attention
//! - [Kwon2023] PagedAttention for LLM Serving with vLLM
//! - [Popper1934] The Logic of Scientific Discovery - Falsificationism

#![allow(unused_imports)]
#![allow(dead_code)]

mod attention;
mod core_api;
mod debug_isolate;
mod pipeline;
mod softmax;

// ============================================================================
// Marker module for feature gate
// ============================================================================

#[cfg(test)]
mod test_helpers {
    /// Helper to skip tests when CUDA is not available
    pub fn skip_if_no_cuda() -> bool {
        // Check if CUDA is available
        // Return true to skip, false to run
        std::env::var("SKIP_CUDA_TESTS").is_ok()
    }
}
