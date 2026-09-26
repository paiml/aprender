//! Q5_K Quantization Kernels
//!
//! Implements Q5_K quantized GEMM and GEMV operations.
//!
//! ## Q5_K Super-block Layout (176 bytes for 256 values)
//!
//! - Offset 0-1: d (f16 super-block scale)
//! - Offset 2-3: dmin (f16 super-block min)
//! - Offset 4-15: scales (12 bytes, packed 6-bit scale+min x 8 sub-blocks, get_scale_min_k4)
//! - Offset 16-47: qh (32 bytes; bit s of qh[l] is the fifth bit of value l of sub-block s)
//! - Offset 48-175: qs (128 bytes; sub-blocks 2c and 2c+1 share qs[32c..32c+32], low then high nibble)
//!
//! Dequantization: val = d * scale_b * (ql + 16*qh) - dmin * min_b
//! Where ql is 4-bit (0-15), qh is 1-bit (0 or 1), giving 5-bit range (0-31)
//!
//! ## Kernels
//!
//! - [`Q5KKernel`]: Q5_K GEMM kernel (PARITY-116)
//! - [`Q5KGemvKernel`]: Q5_K GEMV kernel for M=1 decode throughput (PAR-003)

mod dequant;
mod gemm;
mod gemv;

pub use dequant::Q5KDequantKernel;
pub use gemm::Q5KKernel;
pub use gemv::Q5KGemvKernel;
