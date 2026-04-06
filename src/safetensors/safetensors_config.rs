
impl SafetensorsModel {
    /// Parse Safetensors file from bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw Safetensors file bytes
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Invalid header
    /// - Malformed JSON metadata
    /// - Invalid data offsets
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let data = std::fs::read("model.safetensors")?;
    /// let model = SafetensorsModel::from_bytes(&data)?;
    /// println!("Loaded {} tensors", model.tensors.len());
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Parse header (8-byte metadata length)
        let metadata_len = Self::parse_header(&mut cursor)?;

        // Parse JSON metadata
        let tensors = Self::parse_metadata(&mut cursor, metadata_len)?;

        // Store remaining data
        let data_start =
            usize::try_from(8 + metadata_len).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "convert_data_offset".to_string(),
                reason: format!(
                    "Data offset {} exceeds platform usize limit",
                    8 + metadata_len
                ),
            })?;
        let data = data[data_start..].to_vec();

        Ok(Self { tensors, data })
    }

    /// Extract F32 tensor data by name
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Tensor not found
    /// - Tensor dtype is not F32
    /// - Data offsets are invalid
    ///
    /// # Panics
    ///
    /// Never panics. The `unwrap()` in byte conversion is safe because
    /// `chunks_exact(4)` guarantees exactly 4 bytes per chunk.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let model = SafetensorsModel::from_bytes(&data)?;
    /// let weights = model.get_tensor_f32("layer1.weight")?;
    /// println!("Weights: {:?}", weights);
    /// ```
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        // Find tensor metadata
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        // Verify dtype is F32
        if tensor.dtype != SafetensorsDtype::F32 {
            let dtype = &tensor.dtype;
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' has dtype {dtype:?}, expected F32"),
            });
        }

        // Extract data slice
        let [start, end] = tensor.data_offsets;
        if end > self.data.len() {
            let data_len = self.data.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Data offset {end} exceeds data size {data_len}"),
            });
        }

        let bytes = &self.data[start..end];

        // Convert bytes to f32 vector
        if !bytes.len().is_multiple_of(4) {
            let len = bytes.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Data size {len} is not a multiple of 4"),
            });
        }

        let values = bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(4) guarantees 4-byte slices"),
                )
            })
            .collect();

        Ok(values)
    }

    /// Parse header (8-byte metadata length)
    fn parse_header(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
        let mut buf = [0u8; 8];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_metadata_len".to_string(),
                reason: e.to_string(),
            })?;

        Ok(u64::from_le_bytes(buf))
    }

    /// Parse JSON metadata
    fn parse_metadata(
        cursor: &mut Cursor<&[u8]>,
        len: u64,
    ) -> Result<HashMap<String, SafetensorsTensorInfo>> {
        // Read JSON bytes
        let len_usize = usize::try_from(len).map_err(|_| RealizarError::UnsupportedOperation {
            operation: "convert_metadata_len".to_string(),
            reason: format!("Metadata length {len} exceeds platform usize limit"),
        })?;

        let mut json_bytes = vec![0u8; len_usize];
        cursor
            .read_exact(&mut json_bytes)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_metadata_json".to_string(),
                reason: e.to_string(),
            })?;

        // Parse JSON as generic Value first to handle __metadata__ and other special keys
        let json_value: serde_json::Value = serde_json::from_slice(&json_bytes).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "parse_json".to_string(),
                reason: e.to_string(),
            }
        })?;

        let json_map =
            json_value
                .as_object()
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "parse_json".to_string(),
                    reason: "Expected JSON object".to_string(),
                })?;

        // Convert to SafetensorsTensorInfo, skipping special keys like __metadata__
        let mut tensors = HashMap::new();
        for (name, value) in json_map {
            // Skip metadata keys (start with __)
            if name.starts_with("__") {
                continue;
            }

            // Parse tensor metadata
            let meta: TensorMetadata = serde_json::from_value(value.clone()).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "parse_tensor_metadata".to_string(),
                    reason: format!("Failed to parse tensor '{name}': {e}"),
                }
            })?;

            tensors.insert(
                name.clone(),
                SafetensorsTensorInfo {
                    name: name.clone(),
                    dtype: meta.dtype,
                    shape: meta.shape,
                    data_offsets: meta.data_offsets,
                },
            );
        }

        Ok(tensors)
    }

    /// Get tensor data as F16 values (converts to F32)
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not F16
    pub fn get_tensor_f16_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_as_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::F16 {
            let dtype = &tensor.dtype;
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_as_f32".to_string(),
                reason: format!("Tensor '{name}' has dtype {dtype:?}, expected F16"),
            });
        }

        let [start, end] = tensor.data_offsets;
        if end > self.data.len() {
            let data_len = self.data.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_as_f32".to_string(),
                reason: format!("Data offset {end} exceeds data size {data_len}"),
            });
        }

        let bytes = &self.data[start..end];

        // Convert F16 bytes to F32
        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect();

        Ok(values)
    }

    /// Get tensor data as BF16 values (converts to F32)
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not BF16
    pub fn get_tensor_bf16_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_as_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::BF16 {
            let dtype = &tensor.dtype;
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_as_f32".to_string(),
                reason: format!("Tensor '{name}' has dtype {dtype:?}, expected BF16"),
            });
        }

        let [start, end] = tensor.data_offsets;
        if end > self.data.len() {
            let data_len = self.data.len();
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_as_f32".to_string(),
                reason: format!("Data offset {end} exceeds data size {data_len}"),
            });
        }

        let bytes = &self.data[start..end];

        // Convert BF16 bytes to F32 using SIMD-accelerated conversion
        // This provides 3-4x speedup over scalar conversion
        let values = simd_bf16_to_f32(bytes);

        Ok(values)
    }

    /// Get tensor as F32 with automatic dtype conversion
    ///
    /// Supports F32, F16, and BF16 dtypes with automatic conversion to F32.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name to extract
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not supported
    pub fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_auto".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        match tensor.dtype {
            SafetensorsDtype::F32 => self.get_tensor_f32(name),
            SafetensorsDtype::F16 => self.get_tensor_f16_as_f32(name),
            SafetensorsDtype::BF16 => self.get_tensor_bf16_as_f32(name),
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_auto".to_string(),
                reason: format!("Unsupported dtype {:?} for tensor '{name}'", tensor.dtype),
            }),
        }
    }

    /// Get list of tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Get tensor info by name
    #[must_use]
    pub fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        self.tensors.get(name)
    }

    /// Check if model has a tensor with given name
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

// ============================================================================
// SafeTensors Config loader (for sibling config.json)
// ============================================================================

/// Model configuration from config.json
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SafetensorsConfig {
    /// Hidden dimension
    #[serde(alias = "n_embd", alias = "d_model")]
    pub hidden_size: Option<usize>,
    /// Number of transformer layers
    #[serde(alias = "n_layer", alias = "num_layers")]
    pub num_hidden_layers: Option<usize>,
    /// Number of attention heads
    #[serde(alias = "n_head")]
    pub num_attention_heads: Option<usize>,
    /// Number of key-value heads (for GQA)
    pub num_key_value_heads: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Intermediate/FFN dimension
    #[serde(alias = "n_inner")]
    pub intermediate_size: Option<usize>,
    /// Maximum sequence length
    #[serde(alias = "n_positions", alias = "n_ctx")]
    pub max_position_embeddings: Option<usize>,
    /// RMSNorm epsilon
    pub rms_norm_eps: Option<f32>,
    /// RoPE theta
    pub rope_theta: Option<f32>,
    /// Model architecture name
    pub architectures: Option<Vec<String>>,
    /// Model type
    pub model_type: Option<String>,
    /// BOS token ID
    pub bos_token_id: Option<u32>,
    /// EOS token ID
    pub eos_token_id: Option<u32>,
    /// Whether to tie input/output embeddings (lm_head = embed_tokens)
    pub tie_word_embeddings: Option<bool>,
    /// GH-278: Explicit head dimension (Qwen3.5 uses 256 vs Qwen2's 128)
    /// When present, overrides hidden_size / num_attention_heads calculation
    pub head_dim: Option<usize>,
    /// GH-278: Whether attention layers use bias (Qwen3.5=false, Qwen2=true)
    pub attention_bias: Option<bool>,
    /// GH-278: Per-layer attention type for hybrid models (Qwen3.5)
    /// Values: "attention" (standard softmax) or "linear" (linear attention)
    pub layer_types: Option<Vec<String>>,
    /// GH-278: Conv1D kernel size for linear attention layers (Qwen3.5 default: 4)
    pub linear_conv_kernel_dim: Option<usize>,
    /// GH-278: Key head dimension for linear attention (Qwen3.5 default: 128)
    pub linear_key_head_dim: Option<usize>,
    /// GH-278: Value head dimension for linear attention (Qwen3.5 default: 128)
    pub linear_value_head_dim: Option<usize>,
    /// GH-278: Number of key heads for linear attention (Qwen3.5 default: 16)
    pub linear_num_key_heads: Option<usize>,
    /// GH-278: Number of value heads for linear attention (Qwen3.5 default: 32)
    pub linear_num_value_heads: Option<usize>,
    /// ALB-010: Number of MoE experts (Qwen3.5-35B-A3B: 256)
    pub num_experts: Option<usize>,
    /// ALB-010: Number of experts selected per token (Qwen3.5-35B-A3B: 8)
    pub num_experts_per_tok: Option<usize>,
    /// ALB-010: MoE expert intermediate/FFN dimension (Qwen3.5-35B-A3B: 512)
    pub moe_intermediate_size: Option<usize>,
    /// ALB-010: Shared expert intermediate size (Qwen3.5-35B-A3B: 512)
    pub shared_expert_intermediate_size: Option<usize>,
}

impl SafetensorsConfig {
    /// Load config from sibling config.json file
    ///
    /// GAP-UX-002: Tries hash-prefixed companion first (`{stem}.config.json`),
    /// then falls back to non-prefixed (`config.json`) for backwards compatibility.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file (config.json will be loaded from same directory)
    ///
    /// # Returns
    ///
    /// Config if found and parsed, None otherwise
    pub fn load_from_sibling(model_path: &std::path::Path) -> Option<Self> {
        let config_path = find_sibling_file(model_path, "config.json")?;
        let content = std::fs::read_to_string(&config_path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Get number of key-value heads (defaults to num_attention_heads for MHA)
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
            .or(self.num_attention_heads)
            .unwrap_or(1)
    }

    /// Get model architecture string
    #[must_use]
    pub fn architecture(&self) -> String {
        self.architectures
            .as_ref()
            .and_then(|a| a.first())
            .cloned()
            .or_else(|| self.model_type.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// GH-278: Get effective head dimension
    /// Uses explicit head_dim if present, otherwise computes from hidden_size / num_attention_heads
    #[must_use]
    pub fn effective_head_dim(&self) -> Option<usize> {
        self.head_dim.or_else(|| {
            let hidden = self.hidden_size?;
            let heads = self.num_attention_heads?;
            if heads > 0 { Some(hidden / heads) } else { None }
        })
    }

    /// GH-278: Check if model uses hybrid attention (has layer_types with both types)
    #[must_use]
    pub fn is_hybrid_attention(&self) -> bool {
        contract_pre_hybrid_block!();
        self.layer_types.as_ref().is_some_and(|types| {
            let has_attn = types.iter().any(|t| t == "attention");
            let has_linear = types.iter().any(|t| t == "linear");
            has_attn && has_linear
        })
    }
}
