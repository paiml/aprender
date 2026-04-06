//! ArgMax Kernel Falsification Tests (F114)
//!
//! CUDA-TDG 100-Point Popper Falsification Protocol
//! Category A: Falsifiability & Testability (25 points)
//!
//! Tests apply Karl Popper's falsificationist methodology to verify:
//! - PARITY-114: Barrier safety (all threads reach bar.sync)
//! - PAR-002: Bounds checking (no illegal memory access)
//! - PAR-062: GPU argmax correctness
//!
//! Reference: Popper, K. R. (1959). The Logic of Scientific Discovery.

#[cfg(feature = "cuda")]
mod barrier_bounds;
#[cfg(feature = "cuda")]
mod gpu_correctness;
#[cfg(feature = "cuda")]
mod ptx_analysis;
