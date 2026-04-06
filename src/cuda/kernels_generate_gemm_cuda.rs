impl CudaKernels {
    /// Generate PTX for GEMM kernels (matrix-matrix multiplication)
    fn generate_gemm_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        let ptx = match kernel_type {
            KernelType::GemmNaive { m, n, k } => GemmKernel::naive(*m, *n, *k).emit_ptx_for_target(target),
            KernelType::GemmTiled { m, n, k, tile_size }
            | KernelType::GemmOptimized {
                m, n, k, tile_size, ..
            } => GemmKernel::tiled(*m, *n, *k, *tile_size).emit_ptx_for_target(target),
            KernelType::GemmTensorCore { m, n, k } => {
                GemmKernel::tensor_core(*m, *n, *k).emit_ptx_for_target(target)
            },
            KernelType::GemmFp16TensorCore { m, n, k } => {
                GemmKernel::wmma_fp16(*m, *n, *k).emit_ptx_for_target(target)
            },
            KernelType::GemmBiasActivation { m, n, k, .. } => {
                GemmKernel::tiled(*m, *n, *k, 32).emit_ptx_for_target(target)
            },
            KernelType::QuantizedGemm { m, n, k } => QuantizeKernel::new(*m, *n, *k).emit_ptx_for_target(target),
            KernelType::QuantizedGemmGgml { m, n, k } => {
                QuantizeKernel::ggml(*m, *n, *k).emit_ptx_for_target(target)
            },
            KernelType::QuantizedGemmGgmlTiled { m, n, k, tile_m } => {
                QuantizeKernel::ggml_tiled(*m, *n, *k, *tile_m).emit_ptx_for_target(target)
            },
            KernelType::Q5KQuantizedGemm { m, n, k } => Q5KKernel::new(*m, *n, *k).emit_ptx_for_target(target),
            KernelType::Q6KQuantizedGemm { m, n, k } => Q6KKernel::new(*m, *n, *k).emit_ptx_for_target(target),
            KernelType::FusedQ4Q8Dot { n } => QuantizeKernel::ggml(1, 1, *n).emit_ptx_for_target(target),
            KernelType::TensorCoreQ4KGemm { m, k, n } => {
                TensorCoreQ4KGemmKernel::new(*m, *k, *n).emit_ptx_for_target(target)
            },
            KernelType::MultiWarpTensorCoreQ4KGemm { m, k, n } => {
                MultiWarpTensorCoreQ4KGemmKernel::new(*m, *k, *n).emit_ptx_for_target(target)
            },
            // GH-143: WMMA kernels only on x86_64
            #[cfg(target_arch = "x86_64")]
            KernelType::InterleavedWmmaQ4KGemm { m, k, n } => {
                InterleavedWmmaQ4KGemmKernel::new(*m, *k, *n).emit_ptx_for_target(target)
            },
            #[cfg(not(target_arch = "x86_64"))]
            KernelType::InterleavedWmmaQ4KGemm { .. } => return None,
            #[cfg(target_arch = "x86_64")]
            KernelType::W4a16WmmaQ4KGemm { m, k, n } => {
                W4a16WmmaQ4KGemmKernel::new(*m, *k, *n).emit_ptx_for_target(target)
            },
            #[cfg(not(target_arch = "x86_64"))]
            KernelType::W4a16WmmaQ4KGemm { .. } => return None,
            KernelType::Dp4aQ4KGemm { m, n, k } => {
                Dp4aQ4KGemmKernel::new(*m, *n, *k).emit_ptx_for_target(target)
            },
            KernelType::Q4KDequant { k, n } => Q4KDequantKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::Q4KDequantFp16 { k, n } => Q4KDequantFp16Kernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::Q6KDequant { k, n } => Q6KDequantKernel::new(*k, *n).emit_ptx_for_target(target),
            _ => return None,
        };
        Some(ptx)
    }

    /// Generate PTX for standard GEMV kernels (matrix-vector, non-Q4K)
    fn generate_gemv_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        let ptx = match kernel_type {
            KernelType::Gemv { k, n } => GemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::CoalescedGemv { k, n } => CoalescedGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::Q5KGemv { k, n } => Q5KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::Q6KGemv { k, n } => Q6KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::CoalescedQ6KGemv { k, n } => CoalescedQ6KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::BatchedQ6KGemv { k, n, m } => {
                BatchedQ6KGemvKernel::new(*k, *n, *m).emit_ptx_for_target(target)
            },
            KernelType::MwvQ6KGemv { k, n, num_warps } => {
                MultiWarpQ6KGemvKernel::with_warps(*k, *n, *num_warps).emit_ptx_for_target(target)
            },
            KernelType::Dp4aQ6KGemv { k, n, num_warps } => {
                Dp4aQ6KGemvKernel::with_warps(*k, *n, *num_warps).emit_ptx_for_target(target)
            },
            KernelType::HwDp4aQ6KGemv { k, n, num_warps: _ } => {
                HalfWarpDp4aQ6KGemvKernel::new(*k, *n).emit_ptx_for_target(target)
            },
            KernelType::Fp16Q4KGemv { k, n } => Fp16Q4KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::Q8_0Gemv { k, n } => Q8_0GemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::Q5_0Gemv { k, n } => generate_q5_0_candle_ptx(*k, *n),
            KernelType::Q4_0Gemv { k, n } => generate_q4_0_candle_ptx(*k, *n),
            KernelType::Q4_1Gemv { k, n } => Q4_1GemvKernel::new(*k, *n).emit_ptx_for_target(target),
            _ => return None,
        };
        Some(ptx)
    }

    /// Generate PTX for Q4K GEMV kernels (quantized 4-bit matrix-vector variants)
    fn generate_q4k_gemv_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        let ptx = match kernel_type {
            KernelType::Q4KGemv { k, n } => Q4KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::TiledQ4KGemv { k, n, outputs_per_block } => {
                TiledQ4KGemvKernel::new(*k, *n).with_outputs_per_block(*outputs_per_block).emit_ptx_for_target(target)
            },
            KernelType::ChunkedTiledQ4KGemv { k, n, outputs_per_block } => {
                ChunkedTiledQ4KGemvKernel::new(*k, *n).with_outputs_per_block(*outputs_per_block).emit_ptx_for_target(target)
            },
            KernelType::CoalescedQ4KGemv { k, n } => CoalescedQ4KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::WideQ4KGemv { k, n } => WideQ4KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::VectorizedQ4KGemv { k, n } => VectorizedQ4KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::MwvQ4KGemv { k, n, num_warps } => {
                let mut kernel = MultiWarpVectorizedQ4KGemvKernel::new(*k, *n);
                kernel.num_warps = *num_warps;
                kernel.emit_ptx_for_target(target)
            },
            KernelType::MwvDp4aQ4KGemv { k, n, num_warps } => {
                let mut kernel = MwvDp4aQ4KGemvKernel::new(*k, *n);
                kernel.num_warps = *num_warps;
                kernel.emit_ptx_for_target(target)
            },
            KernelType::HwDp4aQ4KGemv { k, n, num_warps } => {
                let mut kernel = HalfWarpDp4aQ4KGemvKernel::new(*k, *n);
                kernel.num_warps = *num_warps;
                kernel.emit_ptx_for_target(target)
            },
            KernelType::Dp4aQ4KGemv { k, n } => Dp4aQ4KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::Dp4aSIMDQ4KGemv { k, n } => Dp4aQ4KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::TrueDp4aQ4KGemv { k, n } => TrueDp4aQ4KGemvKernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::BatchedQ4KGemv { m, k, n } => BatchedQ4KGemvKernel::new(*k, *n, *m).emit_ptx_for_target(target),
            KernelType::MultiWarpBatchedQ4KGemv { k, n, warps } => {
                BatchedQ4KGemvKernel::new(*k, *n, *warps * 8).emit_ptx_for_target(target)
            },
            KernelType::BatchedHwDp4aQ4KGemv { k, n, m, num_warps } => {
                let mut kernel = BatchedHwDp4aQ4KGemvKernel::new(*k, *n, *m);
                kernel.num_warps = *num_warps;
                kernel.emit_ptx_for_target(target)
            },
            KernelType::FusedFp32Q4KGemv { k, n, m, num_warps } => {
                let mut kernel = FusedFp32Q4KGemvKernel::new(*k, *n, *m);
                kernel.num_warps = *num_warps;
                kernel.emit_ptx_for_target(target)
            },
            KernelType::InlineQ8Dp4aQ4KGemv { k, n, m, num_warps } => {
                let mut kernel = InlineQ8Dp4aQ4KGemvKernel::new(*k, *n, *m);
                kernel.num_warps = *num_warps;
                kernel.emit_ptx_for_target(target)
            },
            _ => return None,
        };
        Some(ptx)
    }

    /// Generate PTX for simple attention kernels (softmax, layer norm, basic attention)
    fn generate_simple_attention_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        let ptx = match kernel_type {
            KernelType::Softmax { dim } => SoftmaxKernel::new(*dim).emit_ptx_for_target(target),
            KernelType::LayerNorm { hidden_size, epsilon, affine } => {
                let mut kernel = LayerNormKernel::new(*hidden_size);
                if (*epsilon - 1e-5).abs() > f32::EPSILON {
                    kernel = kernel.with_epsilon(*epsilon);
                }
                if !affine {
                    kernel = kernel.without_affine();
                }
                kernel.emit_ptx_for_target(target)
            },
            KernelType::Attention { seq_len, head_dim, causal } => {
                let mut kernel = AttentionKernel::new(*seq_len, *head_dim);
                if *causal {
                    kernel = kernel.with_causal();
                }
                kernel.emit_ptx_for_target(target)
            },
            _ => return None,
        };
        Some(ptx)
    }

    /// Generate PTX for multi-head and incremental attention kernels
    fn generate_advanced_attention_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        let ptx = match kernel_type {
            KernelType::MultiHeadAttention { seq_len, head_dim, n_heads: _, causal } => {
                let max_tile = (48 * 1024) / (head_dim * 12);
                let thread_limit = 1024 / head_dim;
                let tile_size = max_tile.min(64).min(*seq_len).min(thread_limit);
                let mut kernel = AttentionKernel::new(*seq_len, *head_dim).with_tiles(tile_size, tile_size);
                if *causal {
                    kernel = kernel.with_causal();
                }
                kernel.emit_ptx_for_target(target)
            },
            KernelType::AttentionTensorCore { seq_len, head_dim, n_heads: _, causal } => {
                let mut kernel = AttentionKernel::tensor_core(*seq_len, *head_dim);
                if *causal {
                    kernel = kernel.with_causal();
                }
                kernel.emit_ptx_for_target(target)
            },
            KernelType::IncrementalAttention { max_seq_len, head_dim, n_heads, n_kv_heads, indirect } => {
                IncrementalAttentionKernel::with_gqa(*max_seq_len, *head_dim, *n_heads, *n_kv_heads)
                    .with_indirect_seq_len(*indirect)
                    .emit_ptx_for_target(target)
            },
            KernelType::MultiWarpAttention { max_seq_len, head_dim, n_heads, n_kv_heads, num_warps_per_head, indirect } => {
                MultiWarpIncrementalAttentionKernel::new(*max_seq_len, *head_dim, *n_heads, *n_kv_heads, *num_warps_per_head)
                    .with_indirect_seq_len(*indirect)
                    .emit_ptx_for_target(target)
            },
            _ => return None,
        };
        Some(ptx)
    }

    /// Generate PTX for attention kernels (flash attention, incremental, multi-warp)
    fn generate_attention_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        Self::generate_simple_attention_ptx(kernel_type, target)
            .or_else(|| Self::generate_advanced_attention_ptx(kernel_type, target))
    }

    /// Generate PTX for RMSNorm kernels
    fn generate_rmsnorm_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        let ptx = match kernel_type {
            KernelType::RmsNorm { hidden_size, epsilon } => {
                RmsNormKernel::new(*hidden_size).with_epsilon(*epsilon).emit_ptx_for_target(target)
            },
            KernelType::VectorizedRmsNorm { hidden_size, epsilon } => {
                VectorizedRmsNormKernel::new(*hidden_size).with_epsilon(*epsilon).emit_ptx_for_target(target)
            },
            KernelType::BatchedVectorizedRmsNorm { hidden_size, batch_size, epsilon } => {
                BatchedVectorizedRmsNormKernel::new(*hidden_size, *batch_size).with_epsilon(*epsilon).emit_ptx_for_target(target)
            },
            KernelType::PreciseRmsNorm { hidden_size, epsilon } => {
                PreciseRmsNormKernel::new(*hidden_size).with_epsilon(*epsilon).emit_ptx_for_target(target)
            },
            KernelType::PerHeadRmsNorm { head_dim, num_heads, epsilon } => {
                PerHeadRmsNormKernel::new(*head_dim, *num_heads).with_epsilon(*epsilon).emit_ptx_for_target(target)
            },
            // GH-143: BatchedFusedResidualRmsNormKernel only on x86_64
            #[cfg(target_arch = "x86_64")]
            KernelType::BatchedFusedResidualRmsNorm { hidden_size, batch_size, epsilon } => {
                BatchedFusedResidualRmsNormKernel::new(*hidden_size, *batch_size).with_epsilon(*epsilon).emit_ptx_for_target(target)
            },
            #[cfg(not(target_arch = "x86_64"))]
            KernelType::BatchedFusedResidualRmsNorm { .. } => return None,
            KernelType::FusedResidualRmsNorm { hidden_size, epsilon } => {
                FusedResidualRmsNormKernel::new(*hidden_size).with_epsilon(*epsilon).emit_ptx_for_target(target)
            },
            KernelType::FusedRmsNormQ4KGemv { k, n, epsilon } => {
                FusedRmsNormQ4KGemvKernel::new(*k, *n).with_epsilon(*epsilon).emit_ptx_for_target(target)
            },
            KernelType::FusedRmsNormGateUpSwigluQ4K { k, n, epsilon } => {
                FusedRmsNormGateUpSwigluQ4KKernel::new(*k, *n).with_epsilon(*epsilon).emit_ptx_for_target(target)
            },
            _ => return None,
        };
        Some(ptx)
    }

    /// Generate PTX for RoPE and residual kernels
    fn generate_rope_residual_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        let ptx = match kernel_type {
            KernelType::Rope { num_heads, head_dim, theta } => {
                RopeKernel::new(*num_heads, *head_dim, *theta).emit_ptx_for_target(target)
            },
            KernelType::RopeIndirect { num_heads, head_dim, theta } => {
                RopeIndirectKernel::new(*num_heads, *head_dim, *theta).emit_ptx_for_target(target)
            },
            KernelType::RopeNeox { num_heads, head_dim, theta } => {
                RopeNeoxKernel::new(*num_heads, *head_dim, *theta).emit_ptx_for_target(target)
            },
            KernelType::RopeNeoxIndirect { num_heads, head_dim, theta } => {
                RopeNeoxIndirectKernel::new(*num_heads, *head_dim, *theta).emit_ptx_for_target(target)
            },
            KernelType::PreciseRopeNeoxIndirect { num_heads, head_dim, theta } => {
                PreciseRopeIndirectKernel::new(*num_heads, *head_dim, *theta).emit_ptx_for_target(target)
            },
            KernelType::BatchedRope { num_heads, head_dim, batch_size, theta } => {
                BatchedRopeKernel::new(*num_heads, *head_dim, *batch_size, *theta).emit_ptx_for_target(target)
            },
            KernelType::BatchedResidualAdd { n, batch_size } => {
                BatchedResidualAddKernel::new(*n, *batch_size).emit_ptx_for_target(target)
            },
            KernelType::BatchedSwiglu { n, batch_size } => {
                BatchedSwigluKernel::new(*n, *batch_size).emit_ptx_for_target(target)
            },
            KernelType::ResidualAdd { n } => ResidualAddKernel::new(*n).emit_ptx_for_target(target),
            _ => return None,
        };
        Some(ptx)
    }

    /// Generate PTX for normalization and RoPE kernels
    fn generate_norm_rope_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        Self::generate_rmsnorm_ptx(kernel_type, target)
            .or_else(|| Self::generate_rope_residual_ptx(kernel_type, target))
    }

    /// Generate PTX for activation, fusion, and miscellaneous kernels
    fn generate_activation_misc_ptx(kernel_type: &KernelType, target: &str) -> Option<String> {
        let ptx = match kernel_type {
            KernelType::BiasActivation { n, bias_size, activation } => {
                let kernel = BiasActivationKernel::new(*n, *bias_size).with_activation(match activation {
                    1 => Activation::ReLU,
                    2 => Activation::GELU,
                    _ => Activation::None,
                });
                kernel.emit_ptx_for_target(target)
            },
            KernelType::Silu { n } => SiluKernel::new(*n).emit_ptx_for_target(target),
            KernelType::Gelu { n } => GeluKernel::new(*n).emit_ptx_for_target(target),
            KernelType::ElementwiseMul { n } => ElementwiseMulKernel::new(*n).emit_ptx_for_target(target),
            KernelType::FusedSwiglu { n } => FusedSwigluKernel::new(*n).emit_ptx_for_target(target),
            KernelType::FusedQKV { hidden_size, kv_dim } => {
                FusedQKVKernel::new(*hidden_size as usize, *kv_dim as usize).emit_ptx_for_target(target)
            },
            KernelType::FusedGateUp { hidden_size, intermediate_size } => {
                FusedGateUpKernel::new(*hidden_size as usize, *intermediate_size as usize).emit_ptx_for_target(target)
            },
            KernelType::FusedGateUpQ4KGemv { k, n } => {
                FusedGateUpQ4KGemvKernel::new(*k, *n).emit_ptx_for_target(target)
            },
            KernelType::FusedGateUpSwigluHwDp4aQ4KGemv { k, n } => {
                FusedGateUpSwigluHwDp4aQ4KGemvKernel::new(*k, *n).emit_ptx_for_target(target)
            },
            KernelType::KvCacheScatter { num_kv_heads, head_dim, max_len } => {
                KvCacheScatterKernel::new(*num_kv_heads, *head_dim, *max_len).emit_ptx_for_target(target)
            },
            KernelType::KvCacheScatterIndirect { num_kv_heads, head_dim, max_len } => {
                KvCacheScatterIndirectKernel::new(*num_kv_heads, *head_dim, *max_len).emit_ptx_for_target(target)
            },
            KernelType::Q8Quantize { n } => Q8QuantizeKernel { n: *n }.emit_ptx_for_target(target),
            KernelType::Q4KQ8Dot { k, n } => Q4KQ8DotKernel { k: *k, n: *n }.emit_ptx_for_target(target),
            KernelType::PackedDp4aQ4KQ8 { k, n } => PackedDp4aQ4KQ8Kernel::new(*k, *n).emit_ptx_for_target(target),
            KernelType::ArgMax { length } => ArgMaxKernel::new(*length).emit_ptx_for_target(target),
            KernelType::ArgMaxFinal { num_blocks } => ArgMaxFinalKernel::new(*num_blocks).emit_ptx_for_target(target),
            KernelType::Q8Dequant { n } => generate_q8_dequant_ptx(*n),
            _ => return None,
        };
        Some(ptx)
    }
}
