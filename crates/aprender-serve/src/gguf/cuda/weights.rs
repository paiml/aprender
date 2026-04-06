//! Weight management methods for CUDA-accelerated inference
//!
//! This module contains weight upload and caching implementations:
//! - `pre_cache_weights_for_batch`: Pre-cache weights for batched forward pass
//! - `preload_weights_gpu`: Upload all layer weights to GPU with indexed lookup
//! - `clear_decode_graph`: Clear CUDA graph state
//! - `supports_gpu_resident`: Check if model supports GPU-resident path

use super::{OwnedQKVWeights, OwnedQuantizedModelCuda};
use crate::error::{RealizarError, Result};

include!("weights_preload_gpu.rs");
