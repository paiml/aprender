//! Quantized matrix operations for OwnedQuantizedModel
//!
//! Contains embed, fused_matmul, qkv_matmul methods with real implementations
//! for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K quantization formats.

use crate::error::{RealizarError, Result};
use crate::gguf::types::{
    GGUF_TYPE_BF16, GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K,
    GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
};
use crate::gguf::{ops, OwnedQKVWeights, OwnedQuantizedModel, OwnedQuantizedTensor};

include!("fused_matmul_into.rs");
