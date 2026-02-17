
impl Architecture {
    /// Map a source tensor name to APR canonical name
    #[must_use]
    pub fn map_name(&self, source_name: &str) -> String {
        match self {
            Self::Auto => Self::auto_map_name(source_name),
            Self::Whisper => Self::whisper_map_name(source_name),
            Self::Llama => Self::llama_map_name(source_name),
            Self::Bert => Self::bert_map_name(source_name),
            Self::Qwen2 => Self::qwen2_map_name(source_name),
            Self::Qwen3 => Self::qwen2_map_name(source_name), // Qwen3 uses same GGUF naming as Qwen2
            Self::Qwen3_5 => Self::qwen2_map_name(source_name), // Qwen3.5 uses same tensor naming as Qwen2
            Self::Gpt2 => Self::gpt2_map_name(source_name),
            Self::Phi => Self::llama_map_name(source_name), // Phi uses HuggingFace model.layers naming
        }
    }

    /// PMAT-224: Check if this architecture has verified inference support.
    ///
    /// Returns true only for architectures with tested tensor name mapping
    /// and confirmed realizar inference compatibility.
    #[must_use]
    pub fn is_inference_verified(&self) -> bool {
        matches!(self, Self::Qwen2 | Self::Qwen3 | Self::Qwen3_5 | Self::Llama | Self::Phi)
    }

    /// PMAT-224: Get a human-readable name for warning messages.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Auto => "auto-detected",
            Self::Whisper => "Whisper",
            Self::Llama => "LLaMA",
            Self::Bert => "BERT",
            Self::Qwen2 => "Qwen2",
            Self::Qwen3 => "Qwen3",
            Self::Qwen3_5 => "Qwen3.5",
            Self::Gpt2 => "GPT-2",
            Self::Phi => "Phi",
        }
    }

    /// Parse a `model_type` string (from config.json or GGUF metadata) into an Architecture.
    ///
    /// Returns None for unrecognized types. Centralizes the mapping used by
    /// `infer_architecture()` (import.rs) and `detect_gguf_architecture()` (export.rs).
    #[must_use]
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type.to_lowercase().as_str() {
            "qwen2" | "qwen" | "qwen2.5" => Some(Self::Qwen2),
            "qwen3" => Some(Self::Qwen3),
            "qwen3_5" | "qwen3.5" => Some(Self::Qwen3_5),
            "llama" | "llama2" | "llama3" => Some(Self::Llama),
            "whisper" => Some(Self::Whisper),
            "bert" => Some(Self::Bert),
            "gpt2" => Some(Self::Gpt2),
            "phi" | "phi3" | "phi4" => Some(Self::Phi),
            // LLaMA derivatives
            "smollm" | "smollm2" | "granite" | "granite3" | "nemotron" | "mistral" | "gemma"
            | "gemma2" | "gemma3" => Some(Self::Llama),
            _ => None,
        }
    }

    fn auto_map_name(name: &str) -> String {
        // PMAT-099: Preserve original tensor names for AprTransformer compatibility
        // AprTransformer::from_apr_bytes expects model.* prefixes for HuggingFace models
        name.to_string()
    }

    fn whisper_map_name(name: &str) -> String {
        // PMAT-099: Preserve model. prefix for Whisper
        name.to_string()
    }

    fn llama_map_name(name: &str) -> String {
        // PMAT-099: Preserve model. prefix for LLaMA
        name.to_string()
    }

    fn bert_map_name(name: &str) -> String {
        // BERT uses "bert." prefix - preserve it
        name.to_string()
    }

    fn qwen2_map_name(name: &str) -> String {
        // PMAT-205 FIX (GH-190): Map GGUF tensor names to APR canonical format.
        // APR uses BARE names WITHOUT "model." prefix to match the Qwen2 loader
        // contract (models/qwen2/mod.rs:1046-1131).
        //
        // GGUF: blk.N.attn_q.weight → APR: layers.N.self_attn.q_proj.weight
        //
        // PMAT-113 originally added "model." prefix, but the loader expects bare
        // names. This mismatch caused GH-190: 196 tensors unfindable → garbage.

        // Handle layer-specific tensors (blk.N.*)
        if let Some(rest) = name.strip_prefix("blk.") {
            if let Some(dot_pos) = rest.find('.') {
                let layer_num = &rest[..dot_pos];
                let suffix = &rest[dot_pos + 1..];

                // Map GGUF tensor suffixes to APR canonical names
                let apr_suffix = match suffix {
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_q.bias" => "self_attn.q_proj.bias",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_k.bias" => "self_attn.k_proj.bias",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_v.bias" => "self_attn.v_proj.bias",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "attn_output.bias" => "self_attn.o_proj.bias",
                    "attn_norm.weight" => "input_layernorm.weight",
                    // GH-279: Qwen3 QK normalization tensors
                    "attn_q_norm.weight" => "self_attn.q_norm.weight",
                    "attn_k_norm.weight" => "self_attn.k_norm.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    "ffn_norm.weight" => "post_attention_layernorm.weight",
                    other => other, // Preserve unknown suffixes
                };

                // PMAT-222 FIX: Add "model." prefix to match SafeTensors convention
                // GH-190 was wrong - realizar DOES expect "model.layers.N.suffix"
                return format!("model.layers.{layer_num}.{apr_suffix}");
            }
        }

        // PMAT-222 FIX: Handle non-layer tensors with "model." prefix to match SafeTensors
        // Realizar's AprTransformer looks for "model.embed_tokens.weight" not "embed_tokens.weight"
        match name {
            "token_embd.weight" => "model.embed_tokens.weight".to_string(),
            "output.weight" => "lm_head.weight".to_string(),
            "output_norm.weight" => "model.norm.weight".to_string(),
            _ => name.to_string(), // Preserve unknown names
        }
    }

    /// GH-233: Map GPT-2 tensor names to APR canonical format.
    ///
    /// GPT-2 uses `transformer.h.N.*` naming. The fused `c_attn` tensor is
    /// preserved here and split by `split_gpt2_fused_qkv()` after mapping.
    fn gpt2_map_name(name: &str) -> String {
        // GH-255: Handle both "transformer.h.N.*" (PyTorch) and "h.N.*" (SafeTensors) patterns
        let layer_rest = name
            .strip_prefix("transformer.h.")
            .or_else(|| name.strip_prefix("h."));

        if let Some(rest) = layer_rest {
            if let Some(dot_pos) = rest.find('.') {
                let layer_num = &rest[..dot_pos];
                let suffix = &rest[dot_pos + 1..];

                let apr_suffix = match suffix {
                    "ln_1.weight" => "input_layernorm.weight",
                    "ln_1.bias" => "input_layernorm.bias",
                    "ln_2.weight" => "post_attention_layernorm.weight",
                    "ln_2.bias" => "post_attention_layernorm.bias",
                    "attn.c_attn.weight" => "self_attn.c_attn.weight",
                    "attn.c_attn.bias" => "self_attn.c_attn.bias",
                    "attn.c_proj.weight" => "self_attn.o_proj.weight",
                    "attn.c_proj.bias" => "self_attn.o_proj.bias",
                    "mlp.c_fc.weight" => "mlp.up_proj.weight",
                    "mlp.c_fc.bias" => "mlp.up_proj.bias",
                    "mlp.c_proj.weight" => "mlp.down_proj.weight",
                    "mlp.c_proj.bias" => "mlp.down_proj.bias",
                    other => other,
                };

                return format!("model.layers.{layer_num}.{apr_suffix}");
            }
        }

        // Non-layer tensors: handle with/without "transformer." prefix
        let base_name = name.strip_prefix("transformer.").unwrap_or(name);
        match base_name {
            "wte.weight" => "model.embed_tokens.weight".to_string(),
            "wpe.weight" => "model.position_embedding.weight".to_string(),
            "ln_f.weight" => "model.norm.weight".to_string(),
            "ln_f.bias" => "model.norm.bias".to_string(),
            _ => name.to_string(),
        }
    }

    /// GH-233/GH-255: Split GPT-2 fused QKV tensors into separate Q, K, V projections.
    ///
    /// GPT-2's `c_attn` can have shape `[hidden, 3*hidden]` (SafeTensors/HF) or
    /// `[3*hidden, hidden]` (GGUF). Detects fused dimension automatically.
    /// Call this AFTER `map_tensor_names()` when architecture is `Gpt2`.
    pub fn split_gpt2_fused_qkv(tensors: &mut BTreeMap<String, (Vec<f32>, Vec<usize>)>) {
        // Collect fused c_attn tensor names
        let fused_keys: Vec<String> = tensors
            .keys()
            .filter(|k| k.contains("self_attn.c_attn."))
            .cloned()
            .collect();

        for fused_name in fused_keys {
            let (data, shape) = match tensors.remove(&fused_name) {
                Some(v) => v,
                None => continue,
            };

            let is_bias = fused_name
                .rsplit_once('.')
                .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("bias"));

            if is_bias {
                // Bias: 1D tensor of shape [3*hidden] — split into 3 equal parts
                if data.len() % 3 != 0 {
                    // Can't split evenly, put it back
                    tensors.insert(fused_name, (data, shape));
                    continue;
                }
                let chunk = data.len() / 3;
                let base = fused_name.replace("self_attn.c_attn.bias", "");

                tensors.insert(
                    format!("{base}self_attn.q_proj.bias"),
                    (data[..chunk].to_vec(), vec![chunk]),
                );
                tensors.insert(
                    format!("{base}self_attn.k_proj.bias"),
                    (data[chunk..2 * chunk].to_vec(), vec![chunk]),
                );
                tensors.insert(
                    format!("{base}self_attn.v_proj.bias"),
                    (data[2 * chunk..].to_vec(), vec![chunk]),
                );
            } else {
                // Weight: 2D tensor — detect fused dimension
                // SafeTensors/HF: [hidden, 3*hidden] → split columns (dim 1)
                // GGUF:           [3*hidden, hidden] → split rows (dim 0)
                if shape.len() != 2 {
                    tensors.insert(fused_name, (data, shape));
                    continue;
                }

                let base = fused_name.replace("self_attn.c_attn.weight", "");

                if shape[1] == 3 * shape[0] {
                    // GH-255: SafeTensors shape [hidden, 3*hidden] — split columns
                    let rows = shape[0];
                    let cols_per_proj = shape[0]; // hidden
                    let total_cols = shape[1]; // 3*hidden

                    let mut q_data = Vec::with_capacity(rows * cols_per_proj);
                    let mut k_data = Vec::with_capacity(rows * cols_per_proj);
                    let mut v_data = Vec::with_capacity(rows * cols_per_proj);

                    for row in 0..rows {
                        let row_start = row * total_cols;
                        q_data.extend_from_slice(&data[row_start..row_start + cols_per_proj]);
                        k_data.extend_from_slice(
                            &data[row_start + cols_per_proj..row_start + 2 * cols_per_proj],
                        );
                        v_data.extend_from_slice(
                            &data[row_start + 2 * cols_per_proj..row_start + total_cols],
                        );
                    }

                    tensors.insert(
                        format!("{base}self_attn.q_proj.weight"),
                        (q_data, vec![rows, cols_per_proj]),
                    );
                    tensors.insert(
                        format!("{base}self_attn.k_proj.weight"),
                        (k_data, vec![rows, cols_per_proj]),
                    );
                    tensors.insert(
                        format!("{base}self_attn.v_proj.weight"),
                        (v_data, vec![rows, cols_per_proj]),
                    );
                } else if shape[0] % 3 == 0 {
                    // Original path: [3*hidden, hidden] — split rows (dim 0)
                    let rows_per_proj = shape[0] / 3;
                    let cols = shape[1];
                    let chunk = rows_per_proj * cols;

                    tensors.insert(
                        format!("{base}self_attn.q_proj.weight"),
                        (data[..chunk].to_vec(), vec![rows_per_proj, cols]),
                    );
                    tensors.insert(
                        format!("{base}self_attn.k_proj.weight"),
                        (data[chunk..2 * chunk].to_vec(), vec![rows_per_proj, cols]),
                    );
                    tensors.insert(
                        format!("{base}self_attn.v_proj.weight"),
                        (data[2 * chunk..].to_vec(), vec![rows_per_proj, cols]),
                    );
                } else {
                    // Can't split — put it back
                    tensors.insert(fused_name, (data, shape));
                    continue;
                }
            }

            eprintln!(
                "[GH-233] Split fused c_attn tensor: {} → q_proj + k_proj + v_proj",
                fused_name
            );
        }
    }

    /// GH-241: Split GPT-2 fused QKV tensors (raw/quantized version).
    ///
    /// Like `split_gpt2_fused_qkv()` but works with raw quantized bytes
    /// (`GgufRawTensor`) instead of f32 data. Splits by dividing raw bytes
    /// into 3 equal parts — valid because GGUF row-major storage means
    /// each projection's quantization blocks are contiguous.
    pub fn split_gpt2_fused_qkv_raw(
        tensors: &mut BTreeMap<String, crate::format::gguf::GgufRawTensor>,
    ) {
        let fused_keys: Vec<String> = tensors
            .keys()
            .filter(|k| k.contains("self_attn.c_attn."))
            .cloned()
            .collect();

        for fused_name in fused_keys {
            let tensor = match tensors.remove(&fused_name) {
                Some(v) => v,
                None => continue,
            };

            let is_bias = fused_name
                .rsplit_once('.')
                .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("bias"));

            if is_bias {
                // Bias: 1D shape [3*hidden] — split bytes into 3 equal parts
                if tensor.data.len() % 3 != 0 || tensor.shape.len() != 1 || tensor.shape[0] % 3 != 0
                {
                    tensors.insert(fused_name, tensor);
                    continue;
                }
                let byte_chunk = tensor.data.len() / 3;
                let elem_chunk = tensor.shape[0] / 3;
                let base = fused_name.replace("self_attn.c_attn.bias", "");

                tensors.insert(
                    format!("{base}self_attn.q_proj.bias"),
                    crate::format::gguf::GgufRawTensor {
                        data: tensor.data[..byte_chunk].to_vec(),
                        shape: vec![elem_chunk],
                        dtype: tensor.dtype,
                    },
                );
                tensors.insert(
                    format!("{base}self_attn.k_proj.bias"),
                    crate::format::gguf::GgufRawTensor {
                        data: tensor.data[byte_chunk..2 * byte_chunk].to_vec(),
                        shape: vec![elem_chunk],
                        dtype: tensor.dtype,
                    },
                );
                tensors.insert(
                    format!("{base}self_attn.v_proj.bias"),
                    crate::format::gguf::GgufRawTensor {
                        data: tensor.data[2 * byte_chunk..].to_vec(),
                        shape: vec![elem_chunk],
                        dtype: tensor.dtype,
                    },
                );
            } else {
                // Weight: 2D shape [3*hidden, hidden] — split dim 0
                if tensor.shape.len() != 2 || tensor.shape[0] % 3 != 0 || tensor.data.len() % 3 != 0
                {
                    tensors.insert(fused_name, tensor);
                    continue;
                }
                let rows_per_proj = tensor.shape[0] / 3;
                let cols = tensor.shape[1];
                let byte_chunk = tensor.data.len() / 3;
                let base = fused_name.replace("self_attn.c_attn.weight", "");

                tensors.insert(
                    format!("{base}self_attn.q_proj.weight"),
                    crate::format::gguf::GgufRawTensor {
                        data: tensor.data[..byte_chunk].to_vec(),
                        shape: vec![rows_per_proj, cols],
                        dtype: tensor.dtype,
                    },
                );
                tensors.insert(
                    format!("{base}self_attn.k_proj.weight"),
                    crate::format::gguf::GgufRawTensor {
                        data: tensor.data[byte_chunk..2 * byte_chunk].to_vec(),
                        shape: vec![rows_per_proj, cols],
                        dtype: tensor.dtype,
                    },
                );
                tensors.insert(
                    format!("{base}self_attn.v_proj.weight"),
                    crate::format::gguf::GgufRawTensor {
                        data: tensor.data[2 * byte_chunk..].to_vec(),
                        shape: vec![rows_per_proj, cols],
                        dtype: tensor.dtype,
                    },
                );
            }

            eprintln!(
                "[GH-241] Split fused c_attn tensor (raw): {} → q_proj + k_proj + v_proj",
                fused_name
            );
        }
    }
}

// ============================================================================
// Tensor Expectations
// ============================================================================

/// Expected statistics for a tensor type
#[derive(Debug, Clone)]
pub struct TensorExpectation {
    /// Expected mean range (min, max)
    pub mean_range: (f32, f32),
    /// Expected std range (min, max)
    pub std_range: Option<(f32, f32)>,
    /// Description for error messages
    pub description: &'static str,
}
