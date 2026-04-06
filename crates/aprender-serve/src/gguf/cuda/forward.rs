//! Forward pass methods for CUDA-accelerated inference
//!
//! This module contains all forward pass implementations:
//! - `forward_cuda`: Basic forward pass
//! - `forward_single_cuda_with_cache`: Single token with KV cache
//! - `forward_single_full_cuda_with_cache`: Full GPU forward with cache
//! - `forward_gpu_resident`: GPU-resident forward (minimal CPU↔GPU transfers)
//! - Internal helpers: `fused_matmul_cuda`, `qkv_matmul_cuda`, `cuda_attention_with_cache`

use super::{
    OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedModelCuda, OwnedQuantizedTensor,
};
use crate::error::{RealizarError, Result};
use crate::gguf::ops;

include!("uses.rs");
