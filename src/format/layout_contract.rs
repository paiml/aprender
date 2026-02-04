//! Tensor Layout Contract - THE SOURCE OF TRUTH
//!
//! This module implements the tensor layout contract from `contracts/tensor-layout-v1.yaml`.
//! ALL tooling that deals with tensor shapes, layouts, or transpose operations MUST use this module.
//!
//! # Contract Source
//!
//! - **YAML Contract**: `contracts/tensor-layout-v1.yaml`
//! - **Specification**: `docs/specifications/qwen2.5-coder-showcase-demo.md` Section E.8
//! - **GitHub Issue**: paiml/apr-model-qa-playbook#4
//!
//! # Key Principles
//!
//! 1. **Kernel Defines Shape**: The kernel function signature defines the expected data shape,
//!    NOT comments in the code. When `matmul_q6k_rowmajor(W, x, out_dim, in_dim)` is called
//!    with `out_dim=vocab`, W must have `vocab` rows.
//!
//! 2. **GGUF is Column-Major**: GGUF stores weights in column-major order with shape `[in_dim, out_dim]`.
//!
//! 3. **APR is Row-Major**: APR uses row-major order with shape `[out_dim, in_dim]`.
//!
//! 4. **All 2D Weights Transpose**: When converting GGUF→APR, ALL 2D weight tensors are transposed.
//!    There are NO exceptions.
//!
//! # Usage
//!
//! ```rust,ignore
//! use aprender::format::layout_contract::{TensorContract, LayoutContract};
//!
//! // Get the contract for a specific tensor
//! let contract = TensorContract::for_tensor("output.weight")?;
//! assert!(contract.should_transpose);
//! assert_eq!(contract.apr_shape_formula, "[vocab, hidden]");
//!
//! // Validate a shape after conversion
//! let is_valid = LayoutContract::validate_apr_shape("lm_head.weight", &[151936, 896], 151936, 896)?;
//! ```
//!
//! # GH-202 Post-Mortem
//!
//! This contract was created after GH-202 where APR inference produced garbage `[PAD151935]` output.
//! The root cause was confusion about which tensors should be transposed. This contract eliminates
//! that confusion by providing a single authoritative source.

use std::collections::HashMap;

/// Block sizes for quantized formats (bytes per super-block)
pub mod block_sizes {
    /// Q4_K super-block size in bytes
    pub const Q4_K: usize = 144;
    /// Q5_K super-block size in bytes
    pub const Q5_K: usize = 176;
    /// Q6_K super-block size in bytes
    pub const Q6_K: usize = 210;
    /// Elements per super-block (QK_K)
    pub const QK_K: usize = 256;
}

/// Tensor contract specifying layout expectations for a single tensor type.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorContract {
    /// GGUF tensor name pattern (e.g., "blk.{n}.attn_q.weight")
    pub gguf_name: &'static str,
    /// APR tensor name pattern (e.g., "model.layers.{n}.self_attn.q_proj.weight")
    pub apr_name: &'static str,
    /// GGUF shape formula (e.g., "[hidden, heads*head_dim]")
    pub gguf_shape_formula: &'static str,
    /// APR shape formula (e.g., "[heads*head_dim, hidden]")
    pub apr_shape_formula: &'static str,
    /// Whether this tensor should be transposed during GGUF→APR conversion
    pub should_transpose: bool,
    /// Kernel signature that consumes this tensor
    pub kernel_signature: &'static str,
    /// Output dimension expression for kernel
    pub kernel_out_dim: &'static str,
    /// Input dimension expression for kernel
    pub kernel_in_dim: &'static str,
    /// Whether this is a critical tensor (affects inference correctness)
    pub is_critical: bool,
    /// Human-readable notes
    pub notes: &'static str,
}

/// Layout contract validation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ContractError {
    /// Tensor name not found in contract
    UnknownTensor(String),
    /// Shape does not match contract expectation
    ShapeMismatch {
        tensor: String,
        expected: String,
        actual: Vec<usize>,
    },
    /// Byte size does not match kernel expectation
    ByteSizeMismatch {
        tensor: String,
        expected: usize,
        actual: usize,
    },
    /// Transpose was not applied correctly
    TransposeError {
        tensor: String,
        message: String,
    },
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContractError::UnknownTensor(name) => {
                write!(f, "Unknown tensor '{}' not in layout contract", name)
            }
            ContractError::ShapeMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "Shape mismatch for '{}': expected {}, got {:?}",
                tensor, expected, actual
            ),
            ContractError::ByteSizeMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "Byte size mismatch for '{}': expected {} bytes, got {} bytes",
                tensor, expected, actual
            ),
            ContractError::TransposeError { tensor, message } => {
                write!(f, "Transpose error for '{}': {}", tensor, message)
            }
        }
    }
}

impl std::error::Error for ContractError {}

/// The complete layout contract with all tensor specifications.
#[derive(Debug)]
pub struct LayoutContract {
    /// Map from GGUF tensor name pattern to contract
    gguf_contracts: HashMap<&'static str, TensorContract>,
    /// Map from APR tensor name pattern to contract
    apr_contracts: HashMap<&'static str, TensorContract>,
}

impl Default for LayoutContract {
    fn default() -> Self {
        Self::new()
    }
}

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
pub static CONTRACT: std::sync::LazyLock<LayoutContract> =
    std::sync::LazyLock::new(LayoutContract::new);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_creation() {
        let contract = LayoutContract::new();

        // Verify all expected tensors are present
        assert!(contract.get_gguf_contract("output.weight").is_some());
        assert!(contract.get_gguf_contract("token_embd.weight").is_some());
        assert!(contract.get_gguf_contract("output_norm.weight").is_some());
    }

    #[test]
    fn test_f_layout_contract_001_all_2d_transposed() {
        // F-LAYOUT-CONTRACT-001: All 2D weights are transposed
        let contract = LayoutContract::new();

        let transpose_tensors = contract.transpose_tensors();
        assert!(!transpose_tensors.is_empty(), "Should have transpose tensors");

        for tensor in transpose_tensors {
            assert!(
                tensor.should_transpose,
                "Tensor {} should be transposed",
                tensor.gguf_name
            );
            // 2D tensors have different GGUF and APR shapes
            assert_ne!(
                tensor.gguf_shape_formula, tensor.apr_shape_formula,
                "Tensor {} should have different GGUF/APR shapes",
                tensor.gguf_name
            );
        }
    }

    #[test]
    fn test_f_layout_contract_002_lm_head_shape() {
        // F-LAYOUT-CONTRACT-002: lm_head shape matches kernel expectation
        let contract = LayoutContract::new();

        let lm_head = contract
            .get_gguf_contract("output.weight")
            .expect("lm_head contract should exist");

        assert!(lm_head.is_critical, "lm_head should be critical");
        assert_eq!(lm_head.apr_shape_formula, "[vocab, hidden]");
        assert_eq!(lm_head.kernel_out_dim, "vocab_size");
        assert_eq!(lm_head.kernel_in_dim, "hidden_dim");

        // Validate actual shape
        let result = contract.validate_apr_shape("lm_head.weight", &[151936, 896], 151936, 896);
        assert!(result.is_ok(), "Valid lm_head shape should pass");

        // Invalid shape should fail
        let result = contract.validate_apr_shape("lm_head.weight", &[896, 151936], 151936, 896);
        assert!(result.is_err(), "Swapped lm_head shape should fail");
    }

    #[test]
    fn test_f_layout_contract_003_1d_unchanged() {
        // F-LAYOUT-CONTRACT-003: 1D tensors unchanged
        let contract = LayoutContract::new();

        let non_transpose = contract.non_transpose_tensors();
        assert!(!non_transpose.is_empty(), "Should have 1D tensors");

        for tensor in non_transpose {
            assert!(
                !tensor.should_transpose,
                "1D tensor {} should NOT be transposed",
                tensor.gguf_name
            );
            // 1D tensors have same GGUF and APR shapes
            assert_eq!(
                tensor.gguf_shape_formula, tensor.apr_shape_formula,
                "1D tensor {} should have same GGUF/APR shapes",
                tensor.gguf_name
            );
        }
    }

    #[test]
    fn test_f_layout_contract_004_byte_size() {
        // F-LAYOUT-CONTRACT-004: Byte size matches kernel expectation

        // Q6K: 210 bytes per super-block, 256 elements per super-block
        // For lm_head [vocab=151936, hidden=896]:
        // Expected = vocab * ceil(hidden/256) * 210 = 151936 * 4 * 210 = 127,626,240
        let expected = LayoutContract::calculate_q6k_bytes(151936, 896);
        assert_eq!(expected, 127_626_240, "Q6K byte calculation should match");

        // Q4K: 144 bytes per super-block
        let expected_q4k = LayoutContract::calculate_q4k_bytes(151936, 896);
        assert_eq!(
            expected_q4k,
            151936 * 4 * 144,
            "Q4K byte calculation should match"
        );
    }

    #[test]
    fn test_pattern_matching() {
        let contract = LayoutContract::new();

        // Test GGUF pattern matching
        assert!(contract.get_gguf_contract("blk.0.attn_q.weight").is_some());
        assert!(contract.get_gguf_contract("blk.15.attn_q.weight").is_some());
        assert!(contract.get_gguf_contract("blk.99.attn_k.weight").is_some());

        // Test APR pattern matching
        assert!(contract
            .get_apr_contract("model.layers.0.self_attn.q_proj.weight")
            .is_some());
        assert!(contract
            .get_apr_contract("model.layers.27.mlp.gate_proj.weight")
            .is_some());
    }

    #[test]
    fn test_normalize_layer_pattern() {
        assert_eq!(
            normalize_layer_pattern("blk.0.attn_q.weight"),
            "blk.{n}.attn_q.weight"
        );
        assert_eq!(
            normalize_layer_pattern("blk.15.ffn_gate.weight"),
            "blk.{n}.ffn_gate.weight"
        );
        assert_eq!(
            normalize_layer_pattern("model.layers.0.self_attn.q_proj.weight"),
            "model.layers.{n}.self_attn.q_proj.weight"
        );
        assert_eq!(
            normalize_layer_pattern("output.weight"),
            "output.weight" // No layer number
        );
    }

    #[test]
    fn test_critical_tensors() {
        let contract = LayoutContract::new();
        let critical = contract.critical_tensors();

        // Only lm_head should be critical
        assert_eq!(critical.len(), 1, "Only lm_head should be critical");
        assert_eq!(critical[0].gguf_name, "output.weight");
    }

    #[test]
    fn test_should_transpose() {
        let contract = LayoutContract::new();

        // 2D tensors should transpose
        assert!(contract.should_transpose_gguf("output.weight"));
        assert!(contract.should_transpose_gguf("token_embd.weight"));
        assert!(contract.should_transpose_gguf("blk.0.attn_q.weight"));
        assert!(contract.should_transpose_gguf("blk.5.ffn_down.weight"));

        // 1D tensors should NOT transpose
        assert!(!contract.should_transpose_gguf("output_norm.weight"));
        assert!(!contract.should_transpose_gguf("blk.0.attn_norm.weight"));
        assert!(!contract.should_transpose_gguf("blk.3.ffn_norm.weight"));
    }

    #[test]
    fn test_global_contract() {
        // Test the global lazy static instance
        assert!(CONTRACT.get_gguf_contract("output.weight").is_some());
        assert!(CONTRACT.should_transpose_gguf("output.weight"));
        assert!(!CONTRACT.should_transpose_gguf("output_norm.weight"));
    }
}
