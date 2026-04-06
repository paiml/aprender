//! GGUF model loading and parsing
//!
//! Contains GGUFModel and GGUFTransformer parsing implementations extracted from monolith.
//! This handles the binary format parsing and tensor info extraction.

use crate::error::{RealizarError, Result};
use crate::gguf::utils::gpt2_unicode_to_byte;
use crate::gguf::{
    GGUFConfig, GGUFHeader, GGUFModel, GGUFTransformer, GGUFTransformerLayer, GGUFValue,
    TensorInfo, ValidatedModelConfig, GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32,
    GGUF_TYPE_Q2_K, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1,
    GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0, GGUF_VERSION_V2, GGUF_VERSION_V3,
};
use std::collections::HashMap;
use std::io::{Cursor, Read};

include!("token.rs");
include!("transformer_loader.rs");
include!("dtype.rs");
include!("loader_vocab_tests.rs");
