//! PMAT-232: Bound weight dispatch — kernel resolved at load time, not inference.
//!
//! This module implements the dispatch-free forward pass architecture.
//! `BoundWeight::gemv()` calls the pre-bound kernel with zero runtime matching.
//!
//! The match in this file is the SINGLE source of truth for quant → kernel mapping.
//! See `contracts/tensor-layout-v1.yaml` `quant_dispatch` section.

use super::CudaExecutor;
use crate::cuda::types::{BoundWeight, GemvKernel};
use trueno_gpu::driver::GpuBuffer;
use trueno_gpu::error::GpuError;

impl CudaExecutor {
    /// Execute the pre-bound GEMV kernel for a weight.
    ///
    /// This is the ONLY dispatch point for bound weights. The kernel was
    /// resolved at model load time via `BoundWeight::bind()`. Adding a new
    /// `GemvKernel` variant produces a compile error HERE and ONLY here.
    ///
    /// The forward pass calls this method — no match statements needed.
    #[inline]
    pub fn bound_gemv(
        &mut self,
        weight: &BoundWeight,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
    ) -> Result<(), GpuError> {
        // PMAT-232: Exhaustive — no catch-all. Compile error on new variant.
        match weight.kernel() {
            GemvKernel::Q4K => {
                self.q4k_gemv_into(weight.ptr, input, output, weight.out_dim, weight.in_dim)
            },
            GemvKernel::Q5K => {
                self.q5k_gemv_into(weight.ptr, input, output, weight.out_dim, weight.in_dim)
            },
            GemvKernel::Q6K => {
                self.q6k_gemv_into(weight.ptr, input, output, weight.out_dim, weight.in_dim)
            },
            GemvKernel::Q8_0 => {
                self.q8_0_gemv_into(weight.ptr, input, output, weight.out_dim, weight.in_dim)
            },
            GemvKernel::Q4_0 => {
                self.q4_0_gemv_into(weight.ptr, input, output, weight.out_dim, weight.in_dim)
            },
            GemvKernel::Q5_0 => {
                self.q5_0_gemv_into(weight.ptr, input, output, weight.out_dim, weight.in_dim)
            },
            GemvKernel::Q4_1 => {
                self.q4_1_gemv_into(weight.ptr, input, output, weight.out_dim, weight.in_dim)
            },
            GemvKernel::F32 => {
                self.f32_gemv_into(weight.ptr, input, output, weight.out_dim, weight.in_dim)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::types::WeightQuantType;

    #[test]
    fn test_bound_weight_kernel_mapping_exhaustive() {
        // Verify every WeightQuantType maps to a distinct GemvKernel
        let types = [
            (WeightQuantType::Q4K, GemvKernel::Q4K),
            (WeightQuantType::Q5K, GemvKernel::Q5K),
            (WeightQuantType::Q6K, GemvKernel::Q6K),
            (WeightQuantType::Q8_0, GemvKernel::Q8_0),
            (WeightQuantType::Q4_0, GemvKernel::Q4_0),
            (WeightQuantType::Q5_0, GemvKernel::Q5_0),
            (WeightQuantType::Q4_1, GemvKernel::Q4_1),
        ];

        for (qtype, expected_kernel) in types {
            let bound = BoundWeight::bind(0x1000, 1024, qtype, 128, 128);
            assert_eq!(
                bound.kernel(),
                expected_kernel,
                "WeightQuantType::{:?} should map to GemvKernel::{:?}",
                qtype,
                expected_kernel
            );
        }
    }

    #[test]
    fn test_bound_weight_preserves_dimensions() {
        let bound = BoundWeight::bind(0xDEAD, 4096, WeightQuantType::Q6K, 512, 256);
        assert_eq!(bound.ptr, 0xDEAD);
        assert_eq!(bound.len, 4096);
        assert_eq!(bound.out_dim, 512);
        assert_eq!(bound.in_dim, 256);
        assert_eq!(bound.kernel(), GemvKernel::Q6K);
    }

    #[test]
    fn test_bound_layer_weights_construction() {
        use crate::cuda::types::{BoundLayerWeights, IndexedLayerWeights};

        let indexed = IndexedLayerWeights {
            attn_q_ptr: 0x1000,
            attn_q_len: 1024,
            attn_q_qtype: WeightQuantType::Q4K,
            attn_k_ptr: 0x2000,
            attn_k_len: 512,
            attn_k_qtype: WeightQuantType::Q4K,
            attn_v_ptr: 0x3000,
            attn_v_len: 768,
            attn_v_qtype: WeightQuantType::Q6K,
            attn_output_ptr: 0x4000,
            attn_output_len: 1024,
            attn_output_qtype: WeightQuantType::Q4K,
            ffn_gate_ptr: 0x5000,
            ffn_gate_len: 2048,
            ffn_gate_qtype: WeightQuantType::Q4K,
            ffn_up_ptr: 0x6000,
            ffn_up_len: 2048,
            ffn_up_qtype: WeightQuantType::Q4K,
            ffn_down_ptr: 0x7000,
            ffn_down_len: 1536,
            ffn_down_qtype: WeightQuantType::Q6K,
            attn_norm_ptr: 0x8000,
            attn_norm_len: 256,
            ffn_norm_ptr: 0x9000,
            ffn_norm_len: 256,
            attn_q_bias_ptr: 0xA000,
            attn_q_bias_len: 128,
            attn_k_bias_ptr: 0xB000,
            attn_k_bias_len: 64,
            attn_v_bias_ptr: 0xC000,
            attn_v_bias_len: 64,
            attn_q_norm_ptr: 0,
            attn_q_norm_len: 0,
            attn_k_norm_ptr: 0,
            attn_k_norm_len: 0,
        };

        // GH-279: Validate before binding (use qwen2 arch since test has bias fields set)
        let arch = crate::gguf::ArchConstraints::from_architecture("qwen2");
        let validated = crate::cuda::types::ValidatedLayerWeights::validate(indexed, &arch, 0)
            .expect("test weights should validate for qwen2 arch");
        let bound = BoundLayerWeights::bind(&validated, 256, 256, 64, 512);

        // Verify kernels were bound correctly based on qtypes
        assert_eq!(bound.q_proj.kernel(), GemvKernel::Q4K);
        assert_eq!(bound.k_proj.kernel(), GemvKernel::Q4K);
        assert_eq!(bound.v_proj.kernel(), GemvKernel::Q6K); // V uses Q6K
        assert_eq!(bound.o_proj.kernel(), GemvKernel::Q4K);
        assert_eq!(bound.ffn_gate.kernel(), GemvKernel::Q4K);
        assert_eq!(bound.ffn_up.kernel(), GemvKernel::Q4K);
        assert_eq!(bound.ffn_down.kernel(), GemvKernel::Q6K); // down uses Q6K

        // Verify dimensions
        assert_eq!(bound.q_proj.out_dim, 256); // q_dim
        assert_eq!(bound.q_proj.in_dim, 256); // hidden_dim
        assert_eq!(bound.v_proj.out_dim, 64); // kv_dim
        assert_eq!(bound.ffn_gate.out_dim, 512); // intermediate_dim
        assert_eq!(bound.ffn_down.out_dim, 256); // hidden_dim
        assert_eq!(bound.ffn_down.in_dim, 512); // intermediate_dim

        // Verify passthrough fields
        assert_eq!(bound.attn_norm_ptr, 0x8000);
        assert_eq!(bound.attn_q_bias_ptr, 0xA000);
        assert_eq!(bound.attn_v_bias_len, 64);
    }
}
