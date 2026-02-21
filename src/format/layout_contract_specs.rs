
impl LayoutContract {
    /// Create a new layout contract with all tensor specifications.
    ///
    /// This is the AUTHORITATIVE source of truth for tensor layouts.
    /// DO NOT modify these values without updating `contracts/tensor-layout-v1.yaml`.
    #[must_use]
    pub fn new() -> Self {
        let contracts = vec![
            // Embedding layer
            TensorContract {
                gguf_name: "token_embd.weight",
                apr_name: "model.embed_tokens.weight",
                gguf_shape_formula: "[hidden, vocab]",
                apr_shape_formula: "[vocab, hidden]",
                should_transpose: true,
                kernel_signature: "lookup (row = token embedding, not matmul)",
                kernel_out_dim: "vocab_size",
                kernel_in_dim: "hidden_dim",
                is_critical: false,
                notes: "Embedding lookup table - row = token embedding",
            },
            // LM Head (CRITICAL - GH-202 root cause)
            TensorContract {
                gguf_name: "output.weight",
                apr_name: "lm_head.weight",
                gguf_shape_formula: "[hidden, vocab]",
                apr_shape_formula: "[vocab, hidden]",
                should_transpose: true,
                kernel_signature: "matmul_q*k_rowmajor(W, x, vocab_size, hidden_dim)",
                kernel_out_dim: "vocab_size",
                kernel_in_dim: "hidden_dim",
                is_critical: true,
                notes: "GH-202 root cause - wrong shape caused [PAD] garbage output",
            },
            // Q projection
            TensorContract {
                gguf_name: "blk.{n}.attn_q.weight",
                apr_name: "model.layers.{n}.self_attn.q_proj.weight",
                gguf_shape_formula: "[hidden, heads*head_dim]",
                apr_shape_formula: "[heads*head_dim, hidden]",
                should_transpose: true,
                kernel_signature: "matmul_q*k_rowmajor(W, x, num_heads*head_dim, hidden_dim)",
                kernel_out_dim: "num_heads * head_dim",
                kernel_in_dim: "hidden_dim",
                is_critical: false,
                notes: "Query projection in attention",
            },
            // K projection
            TensorContract {
                gguf_name: "blk.{n}.attn_k.weight",
                apr_name: "model.layers.{n}.self_attn.k_proj.weight",
                gguf_shape_formula: "[hidden, kv_heads*head_dim]",
                apr_shape_formula: "[kv_heads*head_dim, hidden]",
                should_transpose: true,
                kernel_signature: "matmul_q*k_rowmajor(W, x, num_kv_heads*head_dim, hidden_dim)",
                kernel_out_dim: "num_kv_heads * head_dim",
                kernel_in_dim: "hidden_dim",
                is_critical: false,
                notes: "Key projection in attention",
            },
            // V projection
            TensorContract {
                gguf_name: "blk.{n}.attn_v.weight",
                apr_name: "model.layers.{n}.self_attn.v_proj.weight",
                gguf_shape_formula: "[hidden, kv_heads*head_dim]",
                apr_shape_formula: "[kv_heads*head_dim, hidden]",
                should_transpose: true,
                kernel_signature: "matmul_q*k_rowmajor(W, x, num_kv_heads*head_dim, hidden_dim)",
                kernel_out_dim: "num_kv_heads * head_dim",
                kernel_in_dim: "hidden_dim",
                is_critical: false,
                notes: "Value projection in attention",
            },
            // O projection
            TensorContract {
                gguf_name: "blk.{n}.attn_output.weight",
                apr_name: "model.layers.{n}.self_attn.o_proj.weight",
                gguf_shape_formula: "[heads*head_dim, hidden]",
                apr_shape_formula: "[hidden, heads*head_dim]",
                should_transpose: true,
                kernel_signature: "matmul_q*k_rowmajor(W, x, hidden_dim, num_heads*head_dim)",
                kernel_out_dim: "hidden_dim",
                kernel_in_dim: "num_heads * head_dim",
                is_critical: false,
                notes: "Output projection in attention",
            },
            // Gate projection (MLP)
            TensorContract {
                gguf_name: "blk.{n}.ffn_gate.weight",
                apr_name: "model.layers.{n}.mlp.gate_proj.weight",
                gguf_shape_formula: "[hidden, intermediate]",
                apr_shape_formula: "[intermediate, hidden]",
                should_transpose: true,
                kernel_signature: "matmul_q*k_rowmajor(W, x, intermediate_dim, hidden_dim)",
                kernel_out_dim: "intermediate_dim",
                kernel_in_dim: "hidden_dim",
                is_critical: false,
                notes: "Gate projection in SwiGLU MLP",
            },
            // Up projection (MLP)
            TensorContract {
                gguf_name: "blk.{n}.ffn_up.weight",
                apr_name: "model.layers.{n}.mlp.up_proj.weight",
                gguf_shape_formula: "[hidden, intermediate]",
                apr_shape_formula: "[intermediate, hidden]",
                should_transpose: true,
                kernel_signature: "matmul_q*k_rowmajor(W, x, intermediate_dim, hidden_dim)",
                kernel_out_dim: "intermediate_dim",
                kernel_in_dim: "hidden_dim",
                is_critical: false,
                notes: "Up projection in SwiGLU MLP",
            },
            // Down projection (MLP)
            TensorContract {
                gguf_name: "blk.{n}.ffn_down.weight",
                apr_name: "model.layers.{n}.mlp.down_proj.weight",
                gguf_shape_formula: "[intermediate, hidden]",
                apr_shape_formula: "[hidden, intermediate]",
                should_transpose: true,
                kernel_signature: "matmul_q*k_rowmajor(W, x, hidden_dim, intermediate_dim)",
                kernel_out_dim: "hidden_dim",
                kernel_in_dim: "intermediate_dim",
                is_critical: false,
                notes: "Down projection in SwiGLU MLP",
            },
            // Input LayerNorm (1D - no transpose)
            TensorContract {
                gguf_name: "blk.{n}.attn_norm.weight",
                apr_name: "model.layers.{n}.input_layernorm.weight",
                gguf_shape_formula: "[hidden]",
                apr_shape_formula: "[hidden]",
                should_transpose: false,
                kernel_signature: "element-wise multiply",
                kernel_out_dim: "hidden_dim",
                kernel_in_dim: "hidden_dim",
                is_critical: false,
                notes: "RMSNorm before attention - 1D tensor, no transpose",
            },
            // Post-attention LayerNorm (1D - no transpose)
            TensorContract {
                gguf_name: "blk.{n}.ffn_norm.weight",
                apr_name: "model.layers.{n}.post_attention_layernorm.weight",
                gguf_shape_formula: "[hidden]",
                apr_shape_formula: "[hidden]",
                should_transpose: false,
                kernel_signature: "element-wise multiply",
                kernel_out_dim: "hidden_dim",
                kernel_in_dim: "hidden_dim",
                is_critical: false,
                notes: "RMSNorm before MLP - 1D tensor, no transpose",
            },
            // Final LayerNorm (1D - no transpose)
            TensorContract {
                gguf_name: "output_norm.weight",
                apr_name: "model.norm.weight",
                gguf_shape_formula: "[hidden]",
                apr_shape_formula: "[hidden]",
                should_transpose: false,
                kernel_signature: "element-wise multiply",
                kernel_out_dim: "hidden_dim",
                kernel_in_dim: "hidden_dim",
                is_critical: false,
                notes: "Final RMSNorm - 1D tensor, no transpose",
            },
        ];

        let mut gguf_contracts = HashMap::new();
        let mut apr_contracts = HashMap::new();

        for contract in contracts {
            gguf_contracts.insert(contract.gguf_name, contract.clone());
            apr_contracts.insert(contract.apr_name, contract);
        }

        Self {
            gguf_contracts,
            apr_contracts,
        }
    }

    /// Get the contract for a GGUF tensor name.
    ///
    /// Handles both exact matches and pattern matches (e.g., "blk.0.attn_q.weight").
    #[must_use]
    pub fn get_gguf_contract(&self, name: &str) -> Option<&TensorContract> {
        // Try exact match first
        if let Some(contract) = self.gguf_contracts.get(name) {
            return Some(contract);
        }

        // Try pattern match (replace layer numbers with {n})
        let pattern = normalize_layer_pattern(name);
        self.gguf_contracts.get(pattern.as_str())
    }

    /// Get the contract for an APR tensor name.
    ///
    /// Handles both exact matches and pattern matches.
    #[must_use]
    pub fn get_apr_contract(&self, name: &str) -> Option<&TensorContract> {
        // Try exact match first
        if let Some(contract) = self.apr_contracts.get(name) {
            return Some(contract);
        }

        // Try pattern match (replace layer numbers with {n})
        let pattern = normalize_layer_pattern(name);
        self.apr_contracts.get(pattern.as_str())
    }

    /// Check if a tensor should be transposed during GGUF→APR conversion.
    ///
    /// Returns `true` for all 2D weight tensors, `false` for 1D tensors.
    #[must_use]
    pub fn should_transpose_gguf(&self, name: &str) -> bool {
        self.get_gguf_contract(name)
            .map_or(false, |c| c.should_transpose)
    }

    /// Check if a tensor is critical (affects inference correctness).
    #[must_use]
    pub fn is_critical_tensor(&self, name: &str) -> bool {
        self.get_gguf_contract(name)
            .or_else(|| self.get_apr_contract(name))
            .map_or(false, |c| c.is_critical)
    }

    /// Validate that an APR tensor shape matches the contract expectation.
    ///
    /// # Arguments
    ///
    /// * `name` - APR tensor name
    /// * `shape` - Actual tensor shape
    /// * `vocab_size` - Model vocabulary size
    /// * `hidden_dim` - Model hidden dimension
    ///
    /// # Errors
    ///
    /// Returns `ContractError::ShapeMismatch` if the shape doesn't match expectations.
    pub fn validate_apr_shape(
        &self,
        name: &str,
        shape: &[usize],
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<(), ContractError> {
        let Some(contract) = self.get_apr_contract(name) else {
            return Ok(()); // Unknown tensor - skip validation
        };

        // Special validation for critical tensors
        if contract.is_critical {
            // lm_head must be [vocab_size, hidden_dim]
            if name.contains("lm_head") || name.contains("output.weight") {
                if shape.len() != 2 {
                    return Err(ContractError::ShapeMismatch {
                        tensor: name.to_string(),
                        expected: format!("[{}, {}]", vocab_size, hidden_dim),
                        actual: shape.to_vec(),
                    });
                }
                if shape[0] != vocab_size || shape[1] != hidden_dim {
                    return Err(ContractError::ShapeMismatch {
                        tensor: name.to_string(),
                        expected: format!("[{}, {}]", vocab_size, hidden_dim),
                        actual: shape.to_vec(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Calculate the expected byte size for a quantized tensor.
    ///
    /// Uses the formula: `out_dim * ceil(in_dim / QK_K) * block_bytes`
    #[must_use]
    pub fn calculate_q4k_bytes(out_dim: usize, in_dim: usize) -> usize {
        let superblocks = (in_dim + block_sizes::QK_K - 1) / block_sizes::QK_K;
        out_dim * superblocks * block_sizes::Q4_K
    }

    /// Calculate the expected byte size for Q6K quantized tensor.
    #[must_use]
    pub fn calculate_q6k_bytes(out_dim: usize, in_dim: usize) -> usize {
        let superblocks = (in_dim + block_sizes::QK_K - 1) / block_sizes::QK_K;
        out_dim * superblocks * block_sizes::Q6_K
    }

    /// Validate that a quantized tensor has the correct byte size.
    ///
    /// # Errors
    ///
    /// Returns `ContractError::ByteSizeMismatch` if the size doesn't match.
    pub fn validate_q6k_bytes(
        &self,
        name: &str,
        actual_bytes: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(), ContractError> {
        let expected = Self::calculate_q6k_bytes(out_dim, in_dim);
        if actual_bytes != expected {
            return Err(ContractError::ByteSizeMismatch {
                tensor: name.to_string(),
                expected,
                actual: actual_bytes,
            });
        }
        Ok(())
    }

    /// Get all critical tensors from the contract.
    #[must_use]
    pub fn critical_tensors(&self) -> Vec<&TensorContract> {
        self.gguf_contracts
            .values()
            .filter(|c| c.is_critical)
            .collect()
    }

    /// Get all tensors that require transpose.
    #[must_use]
    pub fn transpose_tensors(&self) -> Vec<&TensorContract> {
        self.gguf_contracts
            .values()
            .filter(|c| c.should_transpose)
            .collect()
    }

    /// Get all 1D tensors (no transpose required).
    #[must_use]
    pub fn non_transpose_tensors(&self) -> Vec<&TensorContract> {
        self.gguf_contracts
            .values()
            .filter(|c| !c.should_transpose)
            .collect()
    }
}

/// Normalize a tensor name by replacing layer numbers with `{n}`.
///
/// Examples:
/// - `blk.0.attn_q.weight` → `blk.{n}.attn_q.weight`
/// - `model.layers.5.self_attn.q_proj.weight` → `model.layers.{n}.self_attn.q_proj.weight`
fn normalize_layer_pattern(name: &str) -> String {
    // Replace patterns like "blk.0." or "layers.5." with "blk.{n}." or "layers.{n}."
    let mut result = name.to_string();

    // GGUF pattern: blk.N.
    if let Some(start) = result.find("blk.") {
        let after_blk = start + 4;
        if let Some(dot_pos) = result[after_blk..].find('.') {
            let num_end = after_blk + dot_pos;
            if result[after_blk..num_end]
                .chars()
                .all(|c| c.is_ascii_digit())
            {
                result = format!("{}{{n}}{}", &result[..after_blk], &result[num_end..]);
            }
        }
    }

    // APR pattern: layers.N.
    if let Some(start) = result.find("layers.") {
        let after_layers = start + 7;
        if let Some(dot_pos) = result[after_layers..].find('.') {
            let num_end = after_layers + dot_pos;
            if result[after_layers..num_end]
                .chars()
                .all(|c| c.is_ascii_digit())
            {
                result = format!("{}{{n}}{}", &result[..after_layers], &result[num_end..]);
            }
        }
    }

    result
}

/// Global layout contract instance.
///
/// Use this for all layout validation instead of creating new instances.
/// Note: Using a function instead of LazyLock for MSRV 1.70 compatibility.
/// LazyLock requires Rust 1.80+.
pub fn contract() -> LayoutContract {
    LayoutContract::new()
}

/// Validation rule IDs from the contract.
pub mod validation_rules {
    /// F-LAYOUT-CONTRACT-001: All 2D weights are transposed
    pub const ALL_2D_TRANSPOSED: &str = "F-LAYOUT-CONTRACT-001";
    /// F-LAYOUT-CONTRACT-002: lm_head shape matches kernel expectation
    pub const LM_HEAD_SHAPE: &str = "F-LAYOUT-CONTRACT-002";
    /// F-LAYOUT-CONTRACT-003: 1D tensors unchanged
    pub const TENSORS_1D_UNCHANGED: &str = "F-LAYOUT-CONTRACT-003";
    /// F-LAYOUT-CONTRACT-004: Byte size matches kernel expectation
    pub const BYTE_SIZE_MATCHES: &str = "F-LAYOUT-CONTRACT-004";
    /// F-LAYOUT-CONTRACT-005: No garbage output from lm_head
    pub const NO_GARBAGE_OUTPUT: &str = "F-LAYOUT-CONTRACT-005";
}
