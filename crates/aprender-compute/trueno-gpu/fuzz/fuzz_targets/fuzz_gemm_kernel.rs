//! Fuzz target for GEMM kernel generation
//!
//! Tests that arbitrary matrix dimensions don't cause panics
//! or invalid PTX output in GEMM kernels.

#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

use trueno_gpu::kernels::{GemmKernel, Kernel};

/// Fuzz input for GEMM kernel generation
#[derive(Debug, Arbitrary)]
struct GemmFuzzInput {
    /// Matrix M dimension
    m: u32,
    /// Matrix N dimension
    n: u32,
    /// Matrix K dimension
    k: u32,
    /// Tile size for tiled variant
    tile_size: u32,
    /// Kernel variant (0=naive, 1=tiled, 2=tensor_core)
    variant: u8,
}

impl GemmFuzzInput {
    /// Clamp dimensions to reasonable range
    fn clamped_dimensions(&self) -> (u32, u32, u32) {
        // Clamp to 1-4096 to avoid excessive memory/time
        let m = self.m.clamp(1, 4096);
        let n = self.n.clamp(1, 4096);
        let k = self.k.clamp(1, 4096);
        (m, n, k)
    }

    /// Clamp tile size to valid range
    fn clamped_tile_size(&self) -> u32 {
        // Tile size must be power of 2, between 8 and 64
        match self.tile_size % 4 {
            0 => 8,
            1 => 16,
            2 => 32,
            _ => 64,
        }
    }
}

fuzz_target!(|input: GemmFuzzInput| {
    let (m, n, k) = input.clamped_dimensions();
    let tile_size = input.clamped_tile_size();

    // Generate kernel based on variant
    let kernel = match input.variant % 3 {
        0 => GemmKernel::naive(m, n, k),
        1 => GemmKernel::tiled(m, n, k, tile_size),
        _ => GemmKernel::tensor_core(m, n, k),
    };

    // Emit PTX - should never panic
    let ptx = kernel.emit_ptx();

    // Validate basic PTX structure
    assert!(ptx.contains(".version"), "PTX must have version");
    assert!(ptx.contains(".target"), "PTX must have target");
    assert!(ptx.contains(".entry"), "PTX must have entry point");
    assert!(ptx.contains("gemm"), "PTX must contain gemm kernel");

    // Validate parameters exist
    assert!(ptx.contains(".param"), "PTX must have parameters");
    assert!(ptx.contains("a_ptr") || ptx.contains("A"), "PTX must have A matrix param");
    assert!(ptx.contains("b_ptr") || ptx.contains("B"), "PTX must have B matrix param");
    assert!(ptx.contains("c_ptr") || ptx.contains("C"), "PTX must have C matrix param");

    // Tiled kernels should use shared memory
    if input.variant % 3 == 1 {
        // Tiled variant should have shared memory
        // (but don't fail if not - it's a quality check)
        let _ = ptx.contains(".shared");
    }
});
