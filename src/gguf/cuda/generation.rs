//! Token generation methods for CUDA-accelerated inference
//!
//! This module contains all generation loop implementations:
//! - `generate_cuda`: Basic CUDA generation
//! - `generate_cuda_with_cache`: Generation with KV cache
//! - `generate_full_cuda_with_cache`: Full GPU generation with cache
//! - `generate_gpu_resident`: GPU-resident generation (minimal transfers)
//! - `generate_gpu_resident_streaming`: Streaming generation with callback
//! - `generate_batch_gpu_resident`: Batch generation for multiple prompts

use super::super::model::OwnedQuantizedModel;
use super::{OwnedQuantizedKVCache, OwnedQuantizedModelCuda, QuantizedGenerateConfig};
use crate::error::{RealizarError, Result};

include!("generate_1.rs");
include!("generate_batched_streaming.rs");
