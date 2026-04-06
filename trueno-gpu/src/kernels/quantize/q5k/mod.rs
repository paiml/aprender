//! Q5_K Quantization Kernels
//!
//! Implements Q5_K quantized GEMM and GEMV operations.
//!
//! ## Q5_K Super-block Layout (176 bytes for 256 values)
//!
//! - Offset 0-1: d (f16 super-block scale)
//! - Offset 2-3: dmin (f16 super-block min)
//! - Offset 4-15: scales (12 bytes, packed 6-bit scale+min x 8 sub-blocks)
//! - Offset 16-143: qs (128 bytes, 256 x 4-bit low values packed)
//! - Offset 144-175: qh (32 bytes, 256 x 1-bit high values packed)
//!
//! Dequantization: val = d * scale_b * (ql + 16*qh) - dmin * min_b
//! Where ql is 4-bit (0-15), qh is 1-bit (0 or 1), giving 5-bit range (0-31)
//!
//! ## Kernels
//!
//! - [`Q5KKernel`]: Q5_K GEMM kernel (PARITY-116)
//! - [`Q5KGemvKernel`]: Q5_K GEMV kernel for M=1 decode throughput (PAR-003)

mod gemm;
mod gemv;

pub use gemm::Q5KKernel;
pub use gemv::Q5KGemvKernel;
