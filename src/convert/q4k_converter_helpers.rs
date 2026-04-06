
impl GgufToAprQ4KConverter {
    /// Calculate byte size for a GGML tensor based on quantization type and element count.
    fn ggml_tensor_byte_size_h(qtype: u32, num_elements: usize) -> usize {
        match qtype {
            0 => num_elements * 4,                     // F32
            1 => num_elements * 2,                     // F16
            2 => num_elements.div_ceil(32) * 18,       // Q4_0
            3 => num_elements.div_ceil(32) * 20,       // Q4_1
            6 => num_elements.div_ceil(32) * 22,       // Q5_0
            7 => num_elements.div_ceil(32) * 24,       // Q5_1
            8 => num_elements.div_ceil(32) * 34,       // Q8_0
            12 => num_elements.div_ceil(256) * 144,    // Q4_K
            13 => num_elements.div_ceil(256) * 176,    // Q5_K
            14 => num_elements.div_ceil(256) * 210,    // Q6_K
            _ => num_elements * 4,                     // Default to F32
        }
    }

    /// Helper to extract string from GGUF metadata
    fn get_string(
        metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
        key: &str,
    ) -> Option<String> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::String(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Helper to extract u32 from GGUF metadata
    fn get_u32(
        metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
        key: &str,
    ) -> Option<u32> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::UInt32(v)) => Some(*v),
            Some(crate::gguf::GGUFValue::Int32(v)) => Some(*v as u32),
            Some(crate::gguf::GGUFValue::UInt64(v)) => Some(*v as u32),
            _ => None,
        }
    }

    /// Helper to extract string array from GGUF metadata (GH-86: tokenizer vocab/merges)
    fn get_string_array(
        metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
        key: &str,
    ) -> Option<Vec<String>> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::Array(arr)) => {
                let strings: Vec<String> = arr
                    .iter()
                    .filter_map(|v| match v {
                        crate::gguf::GGUFValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
                if strings.is_empty() { None } else { Some(strings) }
            }
            _ => None,
        }
    }

    /// Helper to extract f32 from GGUF metadata
    fn get_f32(
        metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
        key: &str,
    ) -> Option<f32> {
        match metadata.get(key) {
            Some(crate::gguf::GGUFValue::Float32(v)) => Some(*v),
            Some(crate::gguf::GGUFValue::Float64(v)) => Some(*v as f32),
            _ => None,
        }
    }

    /// GH-86 / F-APR-SELF-CONTAINED-001: Embed tokenizer from GGUF into APR metadata.
    ///
    /// Without this, APR serve falls back to character-level tokenizer producing garbage.
    /// Keys match what `load_embedded_bpe_tokenizer()` expects in `realizar/src/apr/loading.rs`.
    fn embed_tokenizer_metadata(
        metadata: &mut serde_json::Value,
        gguf_metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
    ) {
        use crate::gguf::keys;
        let obj = match metadata.as_object_mut() {
            Some(obj) => obj,
            None => return,
        };

        if let Some(vocab) = Self::get_string_array(gguf_metadata, keys::TOKENIZER_TOKENS) {
            eprintln!("[GH-86] Embedding {} vocabulary tokens into APR metadata", vocab.len());
            obj.insert("tokenizer.vocab_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(vocab.len())));
            obj.insert("tokenizer.vocabulary".to_string(),
                serde_json::Value::Array(vocab.into_iter().map(serde_json::Value::String).collect()));
        }
        if let Some(merges) = Self::get_string_array(gguf_metadata, "tokenizer.ggml.merges") {
            eprintln!("[GH-86] Embedding {} BPE merge rules into APR metadata", merges.len());
            obj.insert("tokenizer.merges".to_string(),
                serde_json::Value::Array(merges.into_iter().map(serde_json::Value::String).collect()));
        }
        if let Some(bos) = Self::get_u32(gguf_metadata, keys::TOKENIZER_BOS_ID) {
            obj.insert("tokenizer.bos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(bos)));
        }
        if let Some(eos) = Self::get_u32(gguf_metadata, keys::TOKENIZER_EOS_ID) {
            obj.insert("tokenizer.eos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(eos)));
        }
        if let Some(tmpl) = Self::get_string(gguf_metadata, "tokenizer.chat_template") {
            obj.insert("chat_template".to_string(), serde_json::Value::String(tmpl));
        }
        if let Some(model) = Self::get_string(gguf_metadata, keys::TOKENIZER_MODEL) {
            obj.insert("tokenizer.model".to_string(), serde_json::Value::String(model));
        }
    }

    /// GH-329: Infer rope_type from architecture, checking metadata first.
    ///
    /// Delegates to shared `crate::gguf::infer_rope_type()` for architecture inference.
    fn infer_rope_type(
        architecture: &str,
        metadata: &std::collections::HashMap<String, crate::gguf::GGUFValue>,
    ) -> u32 {
        // First check for explicit rope.scaling.type in metadata
        let scaling_key = crate::gguf::keys::arch_key(architecture, crate::gguf::keys::ROPE_SCALING_TYPE);
        if let Some(crate::gguf::GGUFValue::String(s)) = metadata.get(&scaling_key) {
            match s.as_str() {
                "none" | "linear" => return 0, // NORM style
                "yarn" | "neox" => return 2,   // NEOX style
                _ => {},
            }
        }
        // GH-329: Use shared inference function (single source of truth)
        crate::gguf::infer_rope_type(architecture)
    }

    /// Convert GGUF file to APR v2 with preserved Q4K quantization
    ///
    /// # Arguments
    ///
    /// * `gguf_path` - Path to GGUF file
    /// * `output_path` - Path to write APR v2 file
    ///
    /// # Returns
    ///
    /// Statistics about the conversion
    // serde_json::json!() uses infallible unwrap
    #[allow(clippy::disallowed_methods)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn convert(
        gguf_path: &std::path::Path,
        output_path: &std::path::Path,
    ) -> Result<Q4KConversionStats> {
        use std::io::Write;

        // Load GGUF with raw quantized tensors
        let gguf_data = std::fs::read(gguf_path).map_err(|e| RealizarError::IoError {
            message: format!("Failed to read GGUF: {e}"),
        })?;

        let gguf_model = crate::gguf::GGUFModel::from_bytes(&gguf_data)?;

        // Extract model config from metadata
        use crate::gguf::keys;
        let architecture = Self::get_string(&gguf_model.metadata, keys::GENERAL_ARCHITECTURE)
            .unwrap_or_else(|| "unknown".to_string());
        let hidden_size = Self::get_u32(
            &gguf_model.metadata,
            &keys::arch_key(&architecture, keys::EMBEDDING_LENGTH),
        )
        .unwrap_or(0);
        let num_layers =
            Self::get_u32(&gguf_model.metadata, &keys::arch_key(&architecture, keys::BLOCK_COUNT))
                .unwrap_or(0);
        let num_heads = Self::get_u32(
            &gguf_model.metadata,
            &keys::arch_key(&architecture, keys::ATTENTION_HEAD_COUNT),
        )
        .unwrap_or(0);
        let num_kv_heads = Self::get_u32(
            &gguf_model.metadata,
            &keys::arch_key(&architecture, keys::ATTENTION_HEAD_COUNT_KV),
        )
        .unwrap_or(num_heads);
        let vocab_size = Self::get_u32(&gguf_model.metadata, &keys::arch_key(&architecture, keys::VOCAB_SIZE))
            .or_else(|| Self::get_u32(&gguf_model.metadata, keys::TOKENIZER_VOCAB_SIZE))
            .unwrap_or_else(|| {
                // Infer from embedding tensor shape if metadata not available
                gguf_model
                    .tensors
                    .iter()
                    .find(|t| {
                        t.name.contains("token_embd")
                            || t.name.contains("embed_tokens")
                            || t.name.contains("tok_embeddings")
                    })
                    .and_then(|t| t.dims.first().copied().map(|d| d as u32))
                    .unwrap_or(0)
            }) as usize;
        let intermediate_size = Self::get_u32(
            &gguf_model.metadata,
            &keys::arch_key(&architecture, keys::FEED_FORWARD_LENGTH),
        )
        .unwrap_or(0);
        let context_length = Self::get_u32(
            &gguf_model.metadata,
            &keys::arch_key(&architecture, keys::CONTEXT_LENGTH),
        )
        .unwrap_or(0);
        // R-02 (Meyer DbC): rope_theta from GGUF, or architecture-specific default.
        let rope_theta = Self::get_f32(
            &gguf_model.metadata,
            &keys::arch_key(&architecture, keys::ROPE_FREQ_BASE),
        )
        .unwrap_or_else(|| crate::gguf::default_rope_theta_for_architecture(&architecture));
        let eps = Self::get_f32(
            &gguf_model.metadata,
            &keys::arch_key(&architecture, keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
        )
        .unwrap_or(1e-5);

        // PMAT-107: Infer rope_type from architecture (matches llama.cpp llama-model.cpp:7763-7811)
        // NEOX style (type 2) uses split-halves, NORM style (type 0) uses adjacent pairs
        let rope_type = Self::infer_rope_type(&architecture, &gguf_model.metadata);

        // Build metadata JSON
        // F-REGR-231 FIX: Use field names consistent with AprTransformer::from_apr_bytes
        // BUG-APR-044 FIX: Use QuantizationMetadata struct format expected by aprender
        let mut metadata = serde_json::json!({
            "model_type": "transformer_lm_q4k",
            "architecture": architecture,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,  // Loader checks num_hidden_layers first
            "num_attention_heads": num_heads, // Loader checks num_attention_heads first
            "num_key_value_heads": num_kv_heads, // Loader checks num_key_value_heads first
            "vocab_size": vocab_size,
            "intermediate_size": intermediate_size, // Loader checks intermediate_size first
            "max_position_embeddings": context_length, // Loader checks max_position_embeddings
            "rope_theta": rope_theta,
            "rope_type": rope_type,
            "rms_norm_eps": eps,  // F-REGR-231: Was "eps", loader reads "rms_norm_eps"
            "quantization": {
                "quant_type": "Q4_K",
                "bits": 4,
                "block_size": 256,
                "symmetric": true
            },
        });

        // GH-86 / F-APR-SELF-CONTAINED-001: Embed tokenizer for standalone inference
        Self::embed_tokenizer_metadata(&mut metadata, &gguf_model.metadata);
        let metadata_bytes =
            serde_json::to_vec(&metadata).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to serialize metadata: {e}"),
            })?;
        let metadata_padded_len = metadata_bytes.len().div_ceil(ALIGNMENT) * ALIGNMENT;

        // Extract raw tensors from GGUF
        let mut raw_tensors: Vec<RawTensor> = Vec::new();
        let mut q4k_count = 0usize;
        let mut total_bytes = 0usize;

        for tensor_meta in &gguf_model.tensors {
            let name = tensor_meta.name.clone();
            let shape: Vec<usize> = tensor_meta.dims.iter().map(|&d| d as usize).collect();
            let num_elements: usize = shape.iter().product();
            let qtype = tensor_meta.qtype;

            // Calculate byte size based on qtype (GGML dtype)
            // GGML types: 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
            let byte_size = Self::ggml_tensor_byte_size_h(qtype, num_elements);

            // Extract raw bytes
            let tensor_start = gguf_model.tensor_data_start + tensor_meta.offset as usize;
            if tensor_start + byte_size > gguf_data.len() {
                return Err(RealizarError::FormatError {
                    reason: format!(
                        "Tensor '{}' exceeds file bounds (start={}, size={}, file_len={})",
                        name,
                        tensor_start,
                        byte_size,
                        gguf_data.len()
                    ),
                });
            }

            let data = gguf_data[tensor_start..tensor_start + byte_size].to_vec();

            // Q4_K is GGML type 12
            if qtype == 12 {
                q4k_count += 1;
            }
            total_bytes += byte_size;

            raw_tensors.push(RawTensor {
                name,
                data,
                shape,
                dtype: qtype,
            });
        }

        // BUG-APR-044 FIX: Sort tensors by name (APR v2 format requires sorted tensor index)
        raw_tensors.sort_by(|a, b| a.name.cmp(&b.name));

        // Build binary tensor index
        let mut tensor_index_bytes: Vec<u8> = Vec::new();
        let mut current_offset = 0u64;

        for tensor in &raw_tensors {
            // name_len (2 bytes) + name
            let name_bytes = tensor.name.as_bytes();
            tensor_index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            tensor_index_bytes.extend_from_slice(name_bytes);

            // dtype (1 byte) - write GGML dtype directly
            // The APR v2 TensorEntry::from_binary reader handles these values:
            //   0=F32, 1=F16, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
            // GH-191 FIX: Previously used wrong APR-specific dtype codes (8=Q4_K, 9=Q6_K, 10=Q8_0)
            // that didn't match the reader's mapping, causing all tensors to load as F32.
            let apr_dtype = match tensor.dtype {
                0 => 0u8,   // F32
                1 => 1u8,   // F16
                2 => 2u8,   // Q4_0 (GGML type 2)
                3 => 3u8,   // Q4_1 (GGML type 3)
                6 => 6u8,   // Q5_0 (GGML type 6) — GH-88 FIX
                7 => 7u8,   // Q5_1 (GGML type 7) — GH-88 FIX
                8 => 8u8,   // Q8_0 (GGML type 8)
                12 => 12u8, // Q4_K (GGML type 12)
                13 => 13u8, // Q5_K (GGML type 13)
                14 => 14u8, // Q6_K (GGML type 14)
                other => {
                    eprintln!(
                        "WARN: Unknown GGML dtype {other} for tensor '{}', writing as F32",
                        tensor.name
                    );
                    0u8
                },
            };
            tensor_index_bytes.push(apr_dtype);

            // ndim (1 byte) + dims (8 bytes each)
            tensor_index_bytes.push(tensor.shape.len() as u8);
            for &dim in &tensor.shape {
                tensor_index_bytes.extend_from_slice(&(dim as u64).to_le_bytes());
            }

            // offset (8 bytes)
            tensor_index_bytes.extend_from_slice(&current_offset.to_le_bytes());

            // size (8 bytes)
            let size = tensor.data.len() as u64;
            tensor_index_bytes.extend_from_slice(&size.to_le_bytes());

            // Align next tensor to 64 bytes
            current_offset += size;
            let aligned = current_offset.div_ceil(ALIGNMENT as u64) * ALIGNMENT as u64;
            current_offset = aligned;
        }

        // Calculate offsets
        let metadata_offset = HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + metadata_padded_len as u64;
        let data_offset = tensor_index_offset + tensor_index_bytes.len() as u64;
        // Align data offset
        let data_offset_aligned = data_offset.div_ceil(ALIGNMENT as u64) * ALIGNMENT as u64;

        // Build header (64 bytes)
        let mut header = vec![0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4] = 2; // version major
        header[5] = 0; // version minor
        header[6..8].copy_from_slice(&0x0020u16.to_le_bytes()); // flags: QUANTIZED=0x0020
        header[8..12].copy_from_slice(&(raw_tensors.len() as u32).to_le_bytes());
        header[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
        header[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        header[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        header[32..40].copy_from_slice(&data_offset_aligned.to_le_bytes());

        // BUG-APR-044 FIX: Compute and write CRC32 checksum for APR v2 compatibility
        // Checksum is over bytes [0..40] + [44..64], excluding checksum field itself
        let checksum = compute_apr_header_checksum(&header);
        header[40..44].copy_from_slice(&checksum.to_le_bytes());

        // Write file
        let mut file = std::fs::File::create(output_path).map_err(|e| RealizarError::IoError {
            message: format!("Failed to create output file: {e}"),
        })?;

        // Header
        file.write_all(&header)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to write header: {e}"),
            })?;

        // Metadata (padded)
        file.write_all(&metadata_bytes)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to write metadata: {e}"),
            })?;
        let padding = metadata_padded_len - metadata_bytes.len();
        if padding > 0 {
            file.write_all(&vec![0u8; padding])
                .map_err(|e| RealizarError::IoError {
                    message: format!("Failed to write padding: {e}"),
                })?;
        }

        // Tensor index
        file.write_all(&tensor_index_bytes)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to write tensor index: {e}"),
            })?;

        // Alignment padding before data
        let pre_data_padding = (data_offset_aligned - data_offset) as usize;
        if pre_data_padding > 0 {
            file.write_all(&vec![0u8; pre_data_padding])
                .map_err(|e| RealizarError::IoError {
                    message: format!("Failed to write data alignment: {e}"),
                })?;
        }

        // Tensor data (with alignment)
        for tensor in &raw_tensors {
            file.write_all(&tensor.data)
                .map_err(|e| RealizarError::IoError {
                    message: format!("Failed to write tensor '{}': {e}", tensor.name),
                })?;

            // Align to 64 bytes
            let pad = (ALIGNMENT - (tensor.data.len() % ALIGNMENT)) % ALIGNMENT;
            if pad > 0 {
                file.write_all(&vec![0u8; pad])
                    .map_err(|e| RealizarError::IoError {
                        message: format!("Failed to write tensor padding: {e}"),
                    })?;
            }
        }

        Ok(Q4KConversionStats {
            tensor_count: raw_tensors.len(),
            q4k_tensor_count: q4k_count,
            total_bytes,
            architecture: architecture.clone(),
            num_layers: num_layers as usize,
            hidden_size: hidden_size as usize,
        })
    }
}

/// Statistics from Q4K conversion
#[derive(Debug, Clone)]
pub struct Q4KConversionStats {
    /// Total number of tensors
    pub tensor_count: usize,
    /// Number of Q4K quantized tensors
    pub q4k_tensor_count: usize,
    /// Total bytes written
    pub total_bytes: usize,
    /// Model architecture
    pub architecture: String,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden size
    pub hidden_size: usize,
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod convert_tests;

// T-COV-95 Coverage Bridge tests (Part 02 - B5)
#[cfg(test)]
#[path = "tests_gguf_roundtrip.rs"]
mod convert_tests_gguf_roundtrip;
