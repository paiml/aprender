//! Memory Fuzz Tests (PMAT-018)
//!
//! Stress testing for GPU memory management.
//!
//! # Falsification Strategy
//! - **Scarcity**: Force OOM conditions
//! - **Degeneracy**: Zero-sized buffers, unaligned copies
//! - **Concurrency**: Stream overlap (simulated)

use super::context::CudaContext;
use super::memory::{device_bytes_outstanding, device_memory_exclusive, GpuBuffer};
use crate::GpuError;
use proptest::prelude::*;

mod adversarial;
mod async_and_stress;
mod basic_operations;

proptest! {
    #[test]
    fn test_buffer_roundtrip_fuzz(
        len in 1usize..100_000usize,
        val in any::<f32>()
    ) {
        // Setup context locally per test (expensive but safe for proptest)
        // Note: In real life, use a lazy_static context or run this test single-threaded
        // For now, we assume single-threaded execution via Makefile
        if let Ok(ctx) = CudaContext::new(0) {
             let mut buf = GpuBuffer::<f32>::new(&ctx, len).unwrap();

             let data = vec![val; len];
             buf.copy_from_host(&data).unwrap();

             let mut out = vec![0.0; len];
             buf.copy_to_host(&mut out).unwrap();

             // Check first, middle, last
             prop_assert_eq!(data[0], out[0]);
             prop_assert_eq!(data[len/2], out[len/2]);
             prop_assert_eq!(data[len-1], out[len-1]);
        }
    }
}
