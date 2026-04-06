//! GPU LZ4 decompression kernel (warp-cooperative)
//!
//! Decompresses LZ4 blocks using warp-cooperative parallelism.

use super::PAGE_SIZE;
use crate::kernels::Kernel;
use crate::ptx::builder::PtxControl;
use crate::ptx::{PtxKernel, PtxType};

/// GPU LZ4 decompression kernel (warp-cooperative)
#[derive(Debug, Clone)]
pub struct Lz4WarpDecompressKernel {
    batch_size: u32,
}

impl Lz4WarpDecompressKernel {
    /// Create a new LZ4 warp-cooperative decompression kernel
    #[must_use]
    pub fn new(batch_size: u32) -> Self {
        Self { batch_size }
    }

    /// Get the batch size
    #[must_use]
    pub fn batch_size(&self) -> u32 {
        self.batch_size
    }
}

impl Kernel for Lz4WarpDecompressKernel {
    fn name(&self) -> &str {
        "lz4_decompress_warp"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new(self.name())
            .param(PtxType::U64, "input_batch")
            .param(PtxType::U64, "input_sizes")
            .param(PtxType::U64, "output_batch")
            .param(PtxType::U32, "batch_size")
            .shared_memory(PAGE_SIZE as usize * 2)
            .build(|ctx| {
                // Decompression kernel body (stub — exits immediately)
                ctx.label("L_exit");
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f057_decompress_kernel_exists() {
        let kernel = Lz4WarpDecompressKernel::new(100);
        assert_eq!(kernel.name(), "lz4_decompress_warp");
    }

    #[test]
    fn test_decompress_kernel_batch_size() {
        let kernel = Lz4WarpDecompressKernel::new(500);
        assert_eq!(kernel.batch_size(), 500);
    }

    #[test]
    fn test_decompress_kernel_ptx_generation() {
        let kernel = Lz4WarpDecompressKernel::new(100);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".version"), "Missing PTX version");
        assert!(ptx.contains(".target"), "Missing PTX target");
        assert!(ptx.contains(".entry"), "Missing entry point");
        assert!(ptx.contains("lz4_decompress_warp"), "Missing kernel name");
    }

    #[test]
    fn test_decompress_kernel_has_parameters() {
        let kernel = Lz4WarpDecompressKernel::new(100);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("input_batch"));
        assert!(ptx.contains("input_sizes"));
        assert!(ptx.contains("output_batch"));
        assert!(ptx.contains("batch_size"));
    }

    #[test]
    fn test_decompress_kernel_has_shared_memory() {
        let kernel = Lz4WarpDecompressKernel::new(100);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".shared"));
    }
}
