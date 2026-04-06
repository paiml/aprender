use super::*;
use crate::kernels::gemm::basic::{GemmConfig, GemmKernel};
use crate::kernels::gemm::batched::{BatchedGemmConfig, BatchedGemmKernel};

mod basic_gemm;
mod batched_and_4d;
mod parity_114;
