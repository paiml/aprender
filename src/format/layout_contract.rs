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
    TransposeError { tensor: String, message: String },
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

include!("layout_contract_part_02.rs");
include!("layout_contract_part_03.rs");
include!("layout_contract_part_04.rs");
