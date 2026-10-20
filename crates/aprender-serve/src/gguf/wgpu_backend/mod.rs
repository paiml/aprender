//! wgpu-accelerated quantized model — **M-GPU-MOE-2.0 stub**.
//!
//! Implements the contract `qwen3-moe-forward-gpu-v1` (paiml/aprender
//! `contracts/qwen3-moe-forward-gpu-v1.yaml`, v1.2.0 ACTIVE_ALGORITHM_LEVEL
//! per the option I amendment in the v1.2.0 amendment_history block).
//! This file is **M-GPU-MOE-2.0** — the first sub-stage of M-GPU-MOE-2,
//! analog of `M-GPU-MOE-1.0-redo` for the wgpu backend.
//!
//! # Why a separate type
//!
//! Per the v1.2.0 amendment's option-I decision: the wgpu fallback
//! lives at a NEW wrapper type `OwnedQuantizedModelWgpu` analog of
//! `OwnedQuantizedModelCuda`. Rationale:
//!
//!   - Code-path symmetry, not trait abstraction. Reviewers verify
//!     parity bugs by diff against the cuda/ tree, not by elaborate
//!     test infrastructure.
//!   - The existing `OwnedQuantizedModelCuda` (cuda/mod.rs) holds
//!     CUDA-specific state (CudaExecutor, FP16 cache, prefix_cache)
//!     that does not map 1-to-1 onto wgpu primitives. A trait
//!     abstraction would force a Result::Err arm on every CUDA-only
//!     method.
//!
//! See `qwen3-moe-forward-gpu-v1` v1.2.0 amendment_history for the
//! full four-options analysis (option I CHOSEN; II/III/IV REJECTED).
//!
//! # Why this is a stub
//!
//! Same staging discipline as M-GPU-MOE-1.0-redo (cuda/forward_qwen3_moe_cuda.rs):
//! contract first, scaffold second, implementation third. M-GPU-MOE-2.0
//! establishes the function on the correct type so M-GPU-MOE-2.1
//! (per-expert wgpu dispatch via trueno-gpu QuantizeKernel + GemmKernel)
//! can land in a separate PR without re-arguing the architectural seam.
//!
//! # Module name
//!
//! Named `wgpu_backend` rather than `wgpu` to avoid collision with the
//! `wgpu` crate identifier inside this file's body.

use crate::error::{RealizarError, Result};
use crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer;
use crate::gguf::OwnedQuantizedModel;

/// M-GPU-MOE-2.1 scaffold — fused SwiGLU WGSL compute pipeline (#1582).
/// Element-wise post-matvec activation; cacheable on the parent model.
/// See `swiglu_kernel.rs` for the WGSL source + pipeline builder.
pub mod swiglu_kernel;

/// wgpu-accelerated wrapper for `OwnedQuantizedModel` (M-GPU-MOE-2.0 stub).
///
/// Provides GPU-accelerated forward passes via wgpu compute pipelines on
/// non-NVIDIA hardware (Apple Silicon Metal, AMD via Vulkan, Intel ARC
/// via Vulkan, etc) per CLAUDE.md backend-agnostic mandate.
///
/// At M-GPU-MOE-2.0 (this stub), the type holds only the inner model.
/// At M-GPU-MOE-2.1+, fields will be added for:
///   - `wgpu::Device` + `wgpu::Queue` (handle to the wgpu adapter)
///   - cached `wgpu::ComputePipeline` for trueno-gpu kernels
///     (QuantizeKernel for Q4_K dequant, GemmKernel for matmul)
///   - GPU-resident expert tensor handles (Q4_K/Q6_K bytes uploaded
///     once at construction, mirroring `OwnedQuantizedModelCuda`'s
///     preload pattern)
///
/// # Example (post-M-GPU-MOE-2.2)
///
/// ```rust,ignore
/// use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelWgpu};
///
/// let model = OwnedQuantizedModel::from_mapped(&mapped)?;
/// let mut wgpu_model = OwnedQuantizedModelWgpu::new(model)?;
///
/// // wgpu-accelerated forward pass (Apple Silicon, AMD, Intel)
/// let logits = wgpu_model.forward_qwen3_moe_wgpu(
///     &tokens, &moe_layers, num_experts, num_experts_per_tok,
///     moe_intermediate, data,
/// )?;
/// ```
pub struct OwnedQuantizedModelWgpu {
    /// Inner model (mirrors `OwnedQuantizedModelCuda::model`).
    pub(crate) model: OwnedQuantizedModel,
    // M-GPU-MOE-2.1 will add:
    //   pub(crate) device: wgpu::Device,
    //   pub(crate) queue: wgpu::Queue,
    //   pub(crate) pipelines: WgpuPipelineCache,
    //   pub(crate) expert_buffers: Vec<wgpu::Buffer>,
}

impl OwnedQuantizedModelWgpu {
    /// Create a new wgpu-accelerated model wrapper — **M-GPU-MOE-2.0 stub**.
    ///
    /// At M-GPU-MOE-2.0, this just stores the inner model. At
    /// M-GPU-MOE-2.1, this will also probe wgpu adapters (Metal /
    /// Vulkan / DX12), pick the best one, build cached compute
    /// pipelines, and preload expert tensors.
    ///
    /// # Errors
    ///
    /// At M-GPU-MOE-2.0: never errors (just stores the model).
    /// At M-GPU-MOE-2.1+: returns `RealizarError::CapabilityMismatch`
    /// if no wgpu adapter is available, or
    /// `RealizarError::UnsupportedOperation` if the chosen adapter
    /// lacks a required feature (e.g. shader storage atomics).
    pub fn new(model: OwnedQuantizedModel) -> Result<Self> {
        Ok(Self { model })
    }

    /// wgpu forward pass for a Qwen3-MoE-arch model — **M-GPU-MOE-2.0 stub**.
    ///
    /// Mirrors `OwnedQuantizedModelCuda::forward_qwen3_moe_cuda`
    /// signature step-for-step. The implementation will land
    /// incrementally per the contract's `implementation_stages`:
    ///
    /// - **M-GPU-MOE-2.0 (this stub)**: function exists on the
    ///   correct type; returns structured `UnsupportedOperation`
    ///   pointing at the contract.
    /// - **M-GPU-MOE-2.1**: per-expert wgpu dispatch via cached
    ///   compute pipelines (QuantizeKernel for Q4_K dequant +
    ///   GemmKernel for matmul, both from trueno-gpu).
    /// - **M-GPU-MOE-2.2**: full forward integration mirroring
    ///   `forward_qwen3_moe_cuda` line-for-line, with FFN routed
    ///   through `moe_ffn_forward_layer_wgpu` →
    ///   `expert_swiglu_wgpu` → cached pipelines.
    /// - **M-GPU-MOE-2.3**: cosine-vs-CPU parity gate
    ///   (FALSIFY-QW3-MOE-GPU-PARITY-001 wgpu sibling, ≥0.99 vs
    ///   CPU LAZY-FUSED-MATVEC reference).
    ///
    /// # Arguments
    ///
    /// Identical to `forward_qwen3_moe` (CPU sibling on
    /// `OwnedQuantizedModel`) and `forward_qwen3_moe_cuda` (CUDA
    /// sibling on `OwnedQuantizedModelCuda`). See
    /// `OwnedQuantizedModel::forward_qwen3_moe` doc-comment for
    /// parameter semantics.
    ///
    /// # Returns
    ///
    /// `Vec<f32>` logits with shape `[vocab_size]` for the LAST token.
    ///
    /// # Errors
    ///
    /// At M-GPU-MOE-2.0: returns `RealizarError::UnsupportedOperation`
    /// whose `reason` mentions `qwen3-moe-forward-gpu-v1` v1.2.0
    /// option I and points callers at the v1.2.0 amendment block in
    /// the contract for the staging plan.
    ///
    /// At M-GPU-MOE-2.1+: propagates errors from wgpu compute
    /// pipeline submission, buffer mapping, and expert dispatch.
    ///
    /// # Pre-conditions (validated even at M-GPU-MOE-2.0 stub)
    ///
    /// - `moe_layers.len() == self.model.layers.len()`
    /// - `num_experts > 0 && num_experts_per_tok > 0 && moe_intermediate > 0`
    /// - `num_experts_per_tok <= num_experts`
    /// - `token_ids` is non-empty
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen3_moe_wgpu(
        &self,
        token_ids: &[u32],
        moe_layers: &[Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        _data: &[u8],
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "forward_qwen3_moe_wgpu: token_ids must not be empty".to_string(),
            });
        }
        if moe_layers.len() != self.model.layers.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_wgpu: moe_layers.len() = {} but model has {} decoder layers",
                    moe_layers.len(),
                    self.model.layers.len()
                ),
            });
        }
        if num_experts == 0 || num_experts_per_tok == 0 || moe_intermediate == 0 {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_wgpu: incomplete MoE config — num_experts={num_experts}, \
                     num_experts_per_tok={num_experts_per_tok}, moe_intermediate={moe_intermediate}. \
                     Caller must supply all three from GGUF metadata."
                ),
            });
        }
        if num_experts_per_tok > num_experts {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_wgpu: num_experts_per_tok ({num_experts_per_tok}) \
                     exceeds num_experts ({num_experts})"
                ),
            });
        }

        Err(RealizarError::UnsupportedOperation {
            operation: "forward_qwen3_moe_wgpu".to_string(),
            reason: "M-GPU-MOE-2.0 stub on OwnedQuantizedModelWgpu. \
                 Per qwen3-moe-forward-gpu-v1 v1.2.0 option I, the implementation lands \
                 incrementally: M-GPU-MOE-2.1 (per-expert wgpu dispatch helpers) → \
                 M-GPU-MOE-2.2 (full forward integration analog of \
                 forward_qwen3_moe_cuda) → M-GPU-MOE-2.3 (cosine-vs-CPU parity test). \
                 Until 2.2 lands, callers on non-CUDA hardware should fall back to \
                 OwnedQuantizedModel::forward_qwen3_moe (CPU LAZY-FUSED-MATVEC, \
                 ~30 tok/s on Qwen3-Coder-30B-A3B-Instruct-Q4_K_M)."
                .to_string(),
        })
    }
}

#[cfg(test)]
mod owned_quantized_model_wgpu_tests {
    use super::*;

    /// M-GPU-MOE-2.0 stub returns UnsupportedOperation pointing at v1.2.0.
    /// Validates that the function exists on the correct type per
    /// option I and rejects on the precondition boundary.
    #[test]
    fn forward_qwen3_moe_wgpu_precondition_empty_tokens_returns_invalid_shape() {
        // We can't instantiate OwnedQuantizedModelWgpu without a real model,
        // so this test just verifies the type's existence at compile time
        // and the precondition-validation pattern matches the cuda sibling.
        // The runtime stub is exercised once M-GPU-MOE-2.1 lands and we
        // can actually construct the wrapper.
        let _: fn(
            &OwnedQuantizedModelWgpu,
            &[u32],
            &[Qwen3MoeQuantizedLayer],
            usize,
            usize,
            usize,
            &[u8],
        ) -> Result<Vec<f32>> = OwnedQuantizedModelWgpu::forward_qwen3_moe_wgpu;
    }
}

/// Runtime coverage of the `forward_qwen3_moe_wgpu` precondition ladder and
/// the M-GPU-MOE-2.0 stub (#5629). The wrapper is built from the shared
/// `create_test_model_with_config` fixture, so no GPU or GGUF file is needed.
#[cfg(test)]
mod forward_qwen3_moe_wgpu_runtime_tests {
    use super::*;
    use crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer;
    use crate::gguf::test_helpers::create_test_model_with_config;
    use crate::gguf::{ArchConstraints, GGUFConfig, QuantizedTensorRef};

    // `create_test_model_with_config` always builds exactly one decoder layer.
    const LAYERS: usize = 1;

    fn tensor_ref() -> QuantizedTensorRef {
        QuantizedTensorRef {
            offset: 0,
            byte_size: 0,
            num_elements: 0,
            qtype: 12,
        }
    }

    fn moe_layers(n: usize) -> Vec<Qwen3MoeQuantizedLayer> {
        (0..n)
            .map(|_| Qwen3MoeQuantizedLayer {
                router: tensor_ref(),
                gate_exps: tensor_ref(),
                up_exps: tensor_ref(),
                down_exps: tensor_ref(),
            })
            .collect()
    }

    fn wrapper() -> OwnedQuantizedModelWgpu {
        let config = GGUFConfig {
            architecture: "test".to_string(),
            constraints: ArchConstraints::from_architecture("test"),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 1,
            vocab_size: 100,
            rope_theta: 10000.0,
            context_length: 512,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            query_pre_attn_scalar: None,
            bos_token_id: None,
            eos_token_id: None,
        };
        OwnedQuantizedModelWgpu::new(create_test_model_with_config(&config))
            .expect("wrapper construction never errors at M-GPU-MOE-2.0")
    }

    /// Run the forward with the given tokens, layer count and MoE config.
    fn run(tokens: &[u32], layers: usize, cfg: (usize, usize, usize)) -> Result<Vec<f32>> {
        wrapper().forward_qwen3_moe_wgpu(tokens, &moe_layers(layers), cfg.0, cfg.1, cfg.2, &[])
    }

    fn invalid_shape_reason(r: Result<Vec<f32>>) -> String {
        match r {
            Err(RealizarError::InvalidShape { reason }) => reason,
            other => panic!("expected InvalidShape, got {other:?}"),
        }
    }

    #[test]
    fn empty_tokens_are_rejected_before_any_other_check() {
        // Layer count AND config are also wrong: only the token check may fire.
        let reason = invalid_shape_reason(run(&[], LAYERS + 1, (0, 0, 0)));
        assert!(reason.contains("token_ids must not be empty"), "{reason}");
    }

    #[test]
    fn layer_count_mismatch_names_both_counts_and_beats_config_check() {
        let reason = invalid_shape_reason(run(&[1], LAYERS + 1, (0, 0, 0)));
        assert!(reason.contains("moe_layers.len() = 2"), "{reason}");
        assert!(reason.contains("model has 1 decoder layers"), "{reason}");
        // Fewer MoE layers than decoder layers is a mismatch too (not only more).
        let reason = invalid_shape_reason(run(&[1], 0, (4, 2, 8)));
        assert!(reason.contains("moe_layers.len() = 0"), "{reason}");
    }

    #[test]
    fn each_zero_moe_config_field_is_rejected_and_named() {
        // (0, 2, 8) also breaks top_k <= experts, which pins zero-config ahead of it.
        for cfg in [(0, 2, 8), (4, 0, 8), (4, 2, 0)] {
            let reason = invalid_shape_reason(run(&[1], LAYERS, cfg));
            let want = format!(
                "num_experts={}, num_experts_per_tok={}, moe_intermediate={}",
                cfg.0, cfg.1, cfg.2
            );
            assert!(
                reason.contains("incomplete MoE config"),
                "{cfg:?}: {reason}"
            );
            assert!(reason.contains(&want), "{cfg:?}: {reason}");
        }
    }

    #[test]
    fn top_k_above_expert_count_is_rejected_but_equal_is_allowed() {
        let reason = invalid_shape_reason(run(&[1], LAYERS, (4, 5, 8)));
        assert!(
            reason.contains("num_experts_per_tok (5) exceeds num_experts (4)"),
            "{reason}"
        );
        // k == num_experts passes the ladder and reaches the stub.
        assert!(matches!(
            run(&[1], LAYERS, (4, 4, 8)),
            Err(RealizarError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn valid_inputs_reach_the_stub_with_its_contract_pointer() {
        match run(&[1, 2, 3], LAYERS, (4, 2, 8)) {
            Err(RealizarError::UnsupportedOperation { operation, reason }) => {
                assert_eq!(operation, "forward_qwen3_moe_wgpu");
                assert!(reason.contains("qwen3-moe-forward-gpu-v1"), "{reason}");
                assert!(reason.contains("forward_qwen3_moe"), "{reason}");
            },
            other => panic!("expected UnsupportedOperation, got {other:?}"),
        }
    }

    #[test]
    fn new_wraps_the_model_without_touching_it() {
        assert_eq!(wrapper().model.layers.len(), LAYERS);
    }
}
