
/// Tensor lookup helper for APR v2 format parsing.
///
/// Bundles raw file data and parsed tensor index for clean access patterns.
/// Replaces closure-based lookups with method calls.
struct AprTensorLookup<'a> {
    data: &'a [u8],
    tensors: &'a std::collections::BTreeMap<String, (usize, usize, Vec<usize>, u8)>,
    /// ALB-106: entrenar checkpoints store linear weights in cuBLAS column-major
    /// convention. When true, 2D linear weight tensors are transposed on load.
    transpose_cublas_weights: bool,
}

/// Dequantize Q4K/Q5K tensor data, handling per-row padding for 2D tensors (GH-202).
fn dequant_q4k_tensor(tensor_data: &[u8], dims: &[usize]) -> Vec<f32> {
    if dims.len() == 2 && !dims[1].is_multiple_of(256) {
        dequant_perrow(tensor_data, dims, 256, 144, |block, out| {
            dequant_q4k_block(block, out);
        })
    } else {
        let num_elements: usize = dims.iter().product();
        dequantize_q4_k_apr(tensor_data, num_elements)
    }
}

/// Dequantize Q6K tensor data, handling per-row padding for 2D tensors (GH-202).
fn dequant_q6k_tensor(tensor_data: &[u8], dims: &[usize]) -> Vec<f32> {
    if dims.len() == 2 && !dims[1].is_multiple_of(256) {
        dequant_perrow(tensor_data, dims, 256, 210, |block, out| {
            dequant_q6k_block(block, out);
        })
    } else {
        let num_elements: usize = dims.iter().product();
        dequantize_q6_k_apr(tensor_data, num_elements)
    }
}

/// Dequantize a single tensor's raw bytes based on GGML dtype.
/// GH-191: Match on GGML dtype values written by converter.
/// GH-88: Added Q4_0, Q4_1, Q5_0, Q5_1 support for mixed-quantization GGUF imports.
fn dequant_by_dtype(tensor_data: &[u8], dims: &[usize], dtype: u8) -> Vec<f32> {
    match dtype {
        // GH-88: Q4_0 (GGML type 2) — 32-element blocks, 18 bytes each
        2 => crate::quantize::dequantize_q4_0(tensor_data)
            .unwrap_or_else(|_| vec![0.0; dims.iter().product()]),
        // GH-88: Q4_1 (GGML type 3) — 32-element blocks, 20 bytes each
        3 => crate::quantize::dequantize_q4_1(tensor_data)
            .unwrap_or_else(|_| vec![0.0; dims.iter().product()]),
        // GH-88: Q5_0 (GGML type 6) — 32-element blocks, 22 bytes each
        6 => crate::quantize::dequantize_q5_0(tensor_data)
            .unwrap_or_else(|_| vec![0.0; dims.iter().product()]),
        // GH-88: Q5_1 (GGML type 7) — 32-element blocks, 24 bytes each
        7 => crate::quantize::dequantize_q5_1(tensor_data)
            .unwrap_or_else(|_| vec![0.0; dims.iter().product()]),
        12 | 13 => dequant_q4k_tensor(tensor_data, dims),
        14 => dequant_q6k_tensor(tensor_data, dims),
        // GH-239: dtype=8 is ambiguous — either GGML Q8_0 or APR Q4 native.
        8 => {
            let num_elements: usize = dims.iter().product();
            let num_blocks = num_elements.div_ceil(32);
            if tensor_data.len() == num_blocks * 34 {
                dequantize_q8_0_apr(tensor_data, num_elements)
            } else {
                dequantize_apr_q4_native(tensor_data, num_elements)
            }
        },
        9 => {
            let num_elements: usize = dims.iter().product();
            dequantize_apr_q8_native(tensor_data, num_elements)
        },
        1 => tensor_data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        // GH-369: BF16 (GGML type 30) — 2 bytes per element, upper 16 bits of F32
        30 => tensor_data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        _ => tensor_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

impl AprTensorLookup<'_> {
    /// Retrieve the raw byte slice for a named tensor, or None if missing/out-of-bounds.
    fn raw_bytes(&self, name: &str) -> Option<(&[u8], &Vec<usize>, u8)> {
        self.tensors.get(name).and_then(|(offset, size, dims, dtype)| {
            let end = offset + size;
            if end > self.data.len() {
                return None;
            }
            Some((&self.data[*offset..end], dims, *dtype))
        })
    }

    /// Extract tensor as f32 values (with dequantization for Q4K/Q5K/Q6K/Q8_0/APR Q4/Q8).
    /// ALB-106: When `transpose_cublas_weights` is set, 2D linear weight tensors
    /// are transposed from cuBLAS convention [K,N] row-major to HF [N,K] row-major.
    fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        self.raw_bytes(name)
            .map(|(tensor_data, dims, dtype)| {
                let mut data = dequant_by_dtype(tensor_data, dims, dtype);
                if self.transpose_cublas_weights
                    && dims.len() == 2
                    && Self::is_linear_weight(name)
                {
                    data = Self::transpose_2d(&data, dims[0], dims[1]);
                }
                data
            })
    }

    /// Check if a tensor name refers to a linear weight that needs transposing.
    fn is_linear_weight(name: &str) -> bool {
        name.contains("_proj.weight")
            || name.contains("gate_proj.weight")
            || name.contains("up_proj.weight")
            || name.contains("down_proj.weight")
            || name == "lm_head.weight"
    }

    /// Transpose a 2D matrix from [cols, rows] row-major to [rows, cols] row-major.
    /// Used to convert cuBLAS convention [K, N] to HF convention [N, K].
    fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; data.len()];
        for c in 0..cols {
            for r in 0..rows {
                out[r * cols + c] = data[c * rows + r];
            }
        }
        out
    }

    /// Extract raw Q4K bytes (no dequantization) for fused kernel.
    /// GH-191 FIX: Use GGML dtype values (12=Q4_K, 13=Q5_K).
    fn get_q4k(&self, name: &str) -> Option<Vec<u8>> {
        self.raw_bytes(name).and_then(|(data, _, dtype)| {
            if dtype != 12 && dtype != 13 { return None; }
            Some(data.to_vec())
        })
    }

    /// Extract raw Q6K bytes (no dequantization) for fused kernel.
    /// GH-191 FIX: Use GGML dtype value 14=Q6_K.
    fn get_q6k(&self, name: &str) -> Option<Vec<u8>> {
        self.raw_bytes(name).and_then(|(data, _, dtype)| {
            if dtype != 14 { return None; }
            Some(data.to_vec())
        })
    }
}

/// HF and GGUF naming prefixes for a single layer.
struct LayerPrefixes {
    hf: String,
    gguf: String,
}

impl LayerPrefixes {
    fn new(layer_idx: usize) -> Self {
        Self {
            hf: format!("model.layers.{layer_idx}"),
            gguf: format!("blk.{layer_idx}"),
        }
    }
}

impl AprTensorLookup<'_> {
    /// Look up a tensor by HF name first, then GGUF name, returning f32 values.
    fn get_hf_or_gguf(&self, hf_name: &str, gguf_name: &str) -> Option<Vec<f32>> {
        self.get_f32(hf_name).or_else(|| self.get_f32(gguf_name))
    }

    /// Load fused or separate QKV weight for a layer.
    fn load_qkv_weight(
        &self,
        pfx: &LayerPrefixes,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Vec<f32> {
        if let Some(qkv) = self.get_f32(&format!("{}.self_attn.qkv_proj.weight", pfx.hf)) {
            return qkv;
        }
        let q = self.get_hf_or_gguf(
            &format!("{}.self_attn.q_proj.weight", pfx.hf),
            &format!("{}.attn_q.weight", pfx.gguf),
        ).unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
        let k = self.get_hf_or_gguf(
            &format!("{}.self_attn.k_proj.weight", pfx.hf),
            &format!("{}.attn_k.weight", pfx.gguf),
        ).unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);
        let v = self.get_hf_or_gguf(
            &format!("{}.self_attn.v_proj.weight", pfx.hf),
            &format!("{}.attn_v.weight", pfx.gguf),
        ).unwrap_or_else(|| vec![0.0; hidden_dim * kv_dim]);
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(&q);
        qkv.extend_from_slice(&k);
        qkv.extend_from_slice(&v);
        qkv
    }

    /// Load fused or separate QKV bias for a layer (optional, for Qwen models).
    fn load_qkv_bias(&self, pfx: &LayerPrefixes) -> Option<Vec<f32>> {
        if let Some(fused) = self.get_f32(&format!("{}.self_attn.qkv_proj.bias", pfx.hf)) {
            return Some(fused);
        }
        let q = self.get_hf_or_gguf(
            &format!("{}.self_attn.q_proj.bias", pfx.hf),
            &format!("{}.attn_q.bias", pfx.gguf),
        );
        let k = self.get_hf_or_gguf(
            &format!("{}.self_attn.k_proj.bias", pfx.hf),
            &format!("{}.attn_k.bias", pfx.gguf),
        );
        let v = self.get_hf_or_gguf(
            &format!("{}.self_attn.v_proj.bias", pfx.hf),
            &format!("{}.attn_v.bias", pfx.gguf),
        );
        match (&q, &k, &v) {
            (Some(q), Some(k), Some(v)) => {
                let mut bias = Vec::with_capacity(q.len() + k.len() + v.len());
                bias.extend_from_slice(q);
                bias.extend_from_slice(k);
                bias.extend_from_slice(v);
                Some(bias)
            },
            _ => None,
        }
    }

    /// Load Q4K/Q6K raw bytes for all layer weights (PMAT-103).
    fn load_quantized_layer_weights(&self, pfx: &LayerPrefixes) -> Q4KLayerWeights {
        let get_q4k_hf_or_gguf = |hf: &str, gguf: &str| -> Option<Vec<u8>> {
            self.get_q4k(hf).or_else(|| self.get_q4k(gguf))
        };
        let get_q6k_hf_or_gguf = |hf: &str, gguf: &str| -> Option<Vec<u8>> {
            self.get_q6k(hf).or_else(|| self.get_q6k(gguf))
        };
        Q4KLayerWeights {
            qkv_weight: None,
            attn_q_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.q_proj.weight", pfx.hf),
                &format!("{}.attn_q.weight", pfx.gguf),
            ),
            attn_k_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.k_proj.weight", pfx.hf),
                &format!("{}.attn_k.weight", pfx.gguf),
            ),
            attn_v_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.v_proj.weight", pfx.hf),
                &format!("{}.attn_v.weight", pfx.gguf),
            ),
            attn_v_weight_q6k: get_q6k_hf_or_gguf(
                &format!("{}.self_attn.v_proj.weight", pfx.hf),
                &format!("{}.attn_v.weight", pfx.gguf),
            ),
            attn_output_weight: get_q4k_hf_or_gguf(
                &format!("{}.self_attn.o_proj.weight", pfx.hf),
                &format!("{}.attn_output.weight", pfx.gguf),
            ),
            ffn_gate_weight: get_q4k_hf_or_gguf(
                &format!("{}.mlp.gate_proj.weight", pfx.hf),
                &format!("{}.ffn_gate.weight", pfx.gguf),
            ),
            ffn_up_weight: get_q4k_hf_or_gguf(
                &format!("{}.mlp.up_proj.weight", pfx.hf),
                &format!("{}.ffn_up.weight", pfx.gguf),
            ),
            ffn_down_weight: get_q4k_hf_or_gguf(
                &format!("{}.mlp.down_proj.weight", pfx.hf),
                &format!("{}.ffn_down.weight", pfx.gguf),
            ),
            ffn_down_weight_q6k: get_q6k_hf_or_gguf(
                &format!("{}.mlp.down_proj.weight", pfx.hf),
                &format!("{}.ffn_down.weight", pfx.gguf),
            ),
            ffn_up_weight_q6k: get_q6k_hf_or_gguf(
                &format!("{}.mlp.up_proj.weight", pfx.hf),
                &format!("{}.ffn_up.weight", pfx.gguf),
            ),
        }
    }
}

/// Check if a `Q4KLayerWeights` has any quantized data.
fn has_quantized_data(w: &Q4KLayerWeights) -> bool {
    w.attn_q_weight.is_some()
        || w.attn_k_weight.is_some()
        || w.attn_output_weight.is_some()
        || w.ffn_gate_weight.is_some()
        || w.ffn_up_weight.is_some()
        || w.ffn_down_weight.is_some()
        || w.ffn_down_weight_q6k.is_some()
        || w.ffn_up_weight_q6k.is_some()
        || w.attn_v_weight_q6k.is_some()
}

impl AprTransformer {
    /// Build transformer layers from APR tensor data.
    ///
    /// Loads Q/K/V weights, attention output, FFN weights, and norms for each layer.
    /// Also extracts Q4K/Q6K raw bytes for fused kernel inference.
    fn build_apr_layers(
        lookup: &AprTensorLookup<'_>,
        num_layers: usize,
        hidden_dim: usize,
        kv_dim: usize,
        intermediate_dim: usize,
        num_experts: Option<usize>,
        debug_enabled: bool,
    ) -> (Vec<AprTransformerLayer>, Option<Vec<Q4KLayerWeights>>) {
        let mut layers = Vec::with_capacity(num_layers);
        let mut q4k_layer_weights: Vec<Q4KLayerWeights> = Vec::with_capacity(num_layers);
        let mut has_any_q4k = false;

        for i in 0..num_layers {
            let pfx = LayerPrefixes::new(i);

            let qkv_weight = lookup.load_qkv_weight(&pfx, hidden_dim, kv_dim);
            let qkv_bias = lookup.load_qkv_bias(&pfx);
            let attn_output = lookup.get_hf_or_gguf(
                &format!("{}.self_attn.o_proj.weight", pfx.hf),
                &format!("{}.attn_output.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![0.0; hidden_dim * hidden_dim]);
            let attn_norm = lookup.get_hf_or_gguf(
                &format!("{}.input_layernorm.weight", pfx.hf),
                &format!("{}.attn_norm.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![1.0; hidden_dim]);
            let ffn_norm = lookup.get_hf_or_gguf(
                &format!("{}.post_attention_layernorm.weight", pfx.hf),
                &format!("{}.ffn_norm.weight", pfx.gguf),
            );
            let ffn_gate = lookup.get_hf_or_gguf(
                &format!("{}.mlp.gate_proj.weight", pfx.hf),
                &format!("{}.ffn_gate.weight", pfx.gguf),
            );
            let ffn_up = lookup.get_hf_or_gguf(
                &format!("{}.mlp.up_proj.weight", pfx.hf),
                &format!("{}.ffn_up.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![0.0; hidden_dim * intermediate_dim]);
            let ffn_down = lookup.get_hf_or_gguf(
                &format!("{}.mlp.down_proj.weight", pfx.hf),
                &format!("{}.ffn_down.weight", pfx.gguf),
            ).unwrap_or_else(|| vec![0.0; intermediate_dim * hidden_dim]);

            let quant_weights = lookup.load_quantized_layer_weights(&pfx);
            if has_quantized_data(&quant_weights) {
                has_any_q4k = true;
            }
            q4k_layer_weights.push(quant_weights);

            // ALB-094: Load MoE weights if architecture supports it
            let moe_gate_weight = lookup.get_f32(&format!("{}.mlp.gate.weight", pfx.hf));

            let (moe_expert_gate_up, moe_expert_down, has_moe) = if let Some(n_experts) = num_experts {
                let mut all_gate_up = Vec::new();
                let mut all_down = Vec::new();
                for e in 0..n_experts {
                    let gate = lookup.get_f32(&format!("{}.mlp.experts.{e}.gate_proj.weight", pfx.hf))
                        .unwrap_or_default();
                    let up = lookup.get_f32(&format!("{}.mlp.experts.{e}.up_proj.weight", pfx.hf))
                        .unwrap_or_default();
                    let down = lookup.get_f32(&format!("{}.mlp.experts.{e}.down_proj.weight", pfx.hf))
                        .unwrap_or_default();
                    // Concatenate gate + up for fused gate_up
                    all_gate_up.extend_from_slice(&gate);
                    all_gate_up.extend_from_slice(&up);
                    all_down.extend_from_slice(&down);
                }
                let has = !all_gate_up.is_empty();
                (
                    if has { Some(all_gate_up) } else { None },
                    if has { Some(all_down) } else { None },
                    has,
                )
            } else {
                (None, None, false)
            };

            // ALB-094: Load shared expert weights (Qwen3.5 MoE)
            let moe_shared_gate = lookup.get_f32(&format!("{}.mlp.shared_expert.gate_proj.weight", pfx.hf));
            let moe_shared_up = lookup.get_f32(&format!("{}.mlp.shared_expert.up_proj.weight", pfx.hf));
            let moe_shared_down = lookup.get_f32(&format!("{}.mlp.shared_expert.down_proj.weight", pfx.hf));
            let moe_shared_expert_gate_weight = lookup.get_f32(&format!("{}.mlp.shared_expert_gate.weight", pfx.hf));

            // When MoE is active, standard FFN is replaced by experts
            let (final_ffn_gate, final_ffn_up, final_ffn_down) = if has_moe {
                (None, vec![0.0f32; 0], vec![0.0f32; 0])
            } else {
                (ffn_gate, ffn_up, ffn_down)
            };

            layers.push(AprTransformerLayer {
                attn_norm_weight: attn_norm,
                attn_norm_bias: None,
                qkv_weight,
                qkv_bias,
                attn_output_weight: attn_output,
                attn_output_bias: None,
                ffn_gate_weight: final_ffn_gate,
                ffn_gate_bias: None,
                ffn_up_weight: final_ffn_up,
                ffn_up_bias: None,
                ffn_down_weight: final_ffn_down,
                ffn_down_bias: None,
                ffn_norm_weight: ffn_norm,
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
                linear_attn_z_weight: None,
                linear_attn_b_weight: None,
                linear_attn_a_weight: None,
                linear_attn_conv1d_weight: None,
                linear_attn_a_log: None,
                linear_attn_dt_bias: None,
                linear_attn_norm_weight: None,
                moe_gate_weight,
                moe_expert_gate_up,
                moe_expert_down,
                moe_shared_gate,
                moe_shared_up,
                moe_shared_down,
                moe_shared_expert_gate_weight,
            });
        }

        let q4k_layers = if has_any_q4k {
            if debug_enabled {
                eprintln!("[DEBUG] Loaded Q4K raw bytes for fused kernel inference");
            }
            Some(q4k_layer_weights)
        } else {
            None
        };

        (layers, q4k_layers)
    }
}

// ============================================================================
// GH-369: BF16 dequant_by_dtype contract tests (Popperian falsification)
// ============================================================================
#[cfg(test)]
mod tests_dequant_bf16 {
    use super::dequant_by_dtype;

    /// Helper: encode f32 as BF16 bytes (truncate lower 16 bits).
    fn f32_to_bf16_bytes(val: f32) -> [u8; 2] {
        let bits = val.to_bits();
        let bf16 = (bits >> 16) as u16;
        bf16.to_le_bytes()
    }

    /// Falsify: BF16 element count matches dims, not half (old bug: 4 bytes/elem → wrong count).
    #[test]
    fn falsify_bf16_dequant_001_element_count() {
        let n = 8;
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.extend_from_slice(&f32_to_bf16_bytes(i as f32));
        }
        let result = dequant_by_dtype(&data, &[n], 30);
        assert_eq!(result.len(), n, "BF16 must produce exactly {n} elements from {n}*2 bytes");
    }

    /// Falsify: BF16 zeros produce f32 zeros (not garbage from misaligned reads).
    #[test]
    fn falsify_bf16_dequant_002_zeros() {
        let data = vec![0u8; 16]; // 8 BF16 zeros
        let result = dequant_by_dtype(&data, &[8], 30);
        assert!(result.iter().all(|&x| x == 0.0), "BF16 zeros must decode to f32 zeros");
    }

    /// Falsify: BF16 round-trip preserves BF16-representable values.
    #[test]
    fn falsify_bf16_dequant_003_round_trip() {
        let test_values: Vec<f32> = vec![1.0, -1.0, 0.5, 42.0, -128.0, 3.14159];
        let mut data = Vec::new();
        for &v in &test_values {
            data.extend_from_slice(&f32_to_bf16_bytes(v));
        }
        let result = dequant_by_dtype(&data, &[test_values.len()], 30);
        for (i, (&original, &decoded)) in test_values.iter().zip(result.iter()).enumerate() {
            // BF16 truncates lower 16 bits — tolerance is relative
            let expected_bf16 = f32::from_bits(original.to_bits() & 0xFFFF_0000);
            assert!(
                (decoded - expected_bf16).abs() < 1e-10,
                "BF16 round-trip mismatch at [{i}]: original={original}, expected_bf16={expected_bf16}, got={decoded}"
            );
        }
    }

    /// Falsify: BF16 1.0 decodes to exactly 1.0 (1.0 is BF16-exact).
    #[test]
    fn falsify_bf16_dequant_004_one() {
        let data = f32_to_bf16_bytes(1.0);
        let result = dequant_by_dtype(&data, &[1], 30);
        assert_eq!(result[0], 1.0, "BF16 1.0 must decode to exactly f32 1.0");
    }

    /// Falsify: F16 (dtype=1) and BF16 (dtype=30) produce DIFFERENT results for same bytes.
    /// This proves they're not aliased (old default arm treated both as F32).
    #[test]
    fn falsify_bf16_dequant_005_not_f16() {
        // 0x3F80 as F16 = 1.875, as BF16 = some other value
        let data: [u8; 2] = [0x80, 0x3F];
        let bf16_result = dequant_by_dtype(&data, &[1], 30);
        let f16_result = dequant_by_dtype(&data, &[1], 1);
        assert_ne!(
            bf16_result[0], f16_result[0],
            "BF16 and F16 must not decode identically — they're different formats"
        );
    }

    /// Falsify: BF16 dtype 30 is NOT treated as F32 (old default arm bug).
    /// F32 needs 4 bytes/element; BF16 needs 2. If treated as F32, 8 bytes → 2 elements.
    /// Correct BF16: 8 bytes → 4 elements.
    #[test]
    fn falsify_bf16_dequant_006_not_f32_default() {
        let data = vec![0u8; 8]; // 4 BF16 elements OR 2 F32 elements
        let bf16_result = dequant_by_dtype(&data, &[4], 30);
        let f32_result = dequant_by_dtype(&data, &[2], 0); // dtype 0 hits default (F32)
        assert_eq!(bf16_result.len(), 4, "BF16: 8 bytes → 4 elements");
        assert_eq!(f32_result.len(), 2, "F32: 8 bytes → 2 elements");
    }

    /// Falsify: 2D tensor dims work correctly (embedding matrix shape).
    #[test]
    fn falsify_bf16_dequant_007_2d_embedding() {
        let rows = 4;
        let cols = 8;
        let n = rows * cols;
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.extend_from_slice(&f32_to_bf16_bytes(i as f32));
        }
        let result = dequant_by_dtype(&data, &[rows, cols], 30);
        assert_eq!(result.len(), n, "BF16 2D tensor must produce rows*cols elements");
        // Verify first and last values
        assert_eq!(result[0], 0.0);
        let expected_last = f32::from_bits(((n - 1) as f32).to_bits() & 0xFFFF_0000);
        assert!((result[n - 1] - expected_last).abs() < 0.1);
    }
}
