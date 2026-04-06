
impl OwnedQuantizedModel {

    /// Fused matmul into pre-allocated output buffer
    pub(crate) fn fused_matmul_into(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        output: &mut [f32],
    ) -> Result<()> {
        use crate::quantize::{
            fused_q4_0_q8_0_parallel_matvec_into, fused_q4k_parallel_matvec_into,
            fused_q5k_parallel_matvec_into, fused_q6k_parallel_matvec_into,
            fused_q8_0_q8_0_parallel_matvec_into,
        };

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let seq_len = input.len() / in_dim;

        // Only support single-token case for now (most common in generation)
        if seq_len != 1 {
            let result = self.fused_matmul(input, weight)?;
            output[..result.len()].copy_from_slice(&result);
            return Ok(());
        }

        debug_assert!(
            output.len() >= out_dim,
            "Output buffer too small: {} < {}",
            output.len(),
            out_dim
        );

        match weight.qtype {
            GGUF_TYPE_Q4_0 => fused_q4_0_q8_0_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q8_0 => fused_q8_0_q8_0_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec_into(
                &weight.data,
                input,
                in_dim,
                out_dim,
                &mut output[..out_dim],
            ),
            _ => {
                let result = self.fused_matmul(input, weight)?;
                output[..result.len()].copy_from_slice(&result);
                Ok(())
            },
        }
    }

    /// Fused gate+up matmul into pre-allocated output buffers (PMAT-FFN-FUSION)
    ///
    /// Computes both gate and up projections in a single rayon dispatch when both
    /// weights share the same quantization type. Falls back to `rayon::join` with
    /// two separate `fused_matmul_into` calls for mixed types.
    pub(crate) fn fused_gate_up_matmul_into(
        &self,
        input: &[f32],
        gate_weight: &OwnedQuantizedTensor,
        up_weight: &OwnedQuantizedTensor,
        gate_output: &mut [f32],
        up_output: &mut [f32],
    ) -> Result<()> {
        use crate::quantize::{
            fused_gate_up_q4k_into, fused_gate_up_q5k_into, fused_gate_up_q6k_into,
        };

        let in_dim = gate_weight.in_dim;
        let out_dim = gate_weight.out_dim;
        let seq_len = input.len() / in_dim;

        // Only fuse for single-token case with matching types and dimensions
        if seq_len == 1
            && gate_weight.qtype == up_weight.qtype
            && gate_weight.in_dim == up_weight.in_dim
            && gate_weight.out_dim == up_weight.out_dim
        {
            match gate_weight.qtype {
                GGUF_TYPE_Q4_K => {
                    return fused_gate_up_q4k_into(
                        &gate_weight.data,
                        &up_weight.data,
                        input,
                        in_dim,
                        out_dim,
                        gate_output,
                        up_output,
                    );
                }
                GGUF_TYPE_Q5_K => {
                    return fused_gate_up_q5k_into(
                        &gate_weight.data,
                        &up_weight.data,
                        input,
                        in_dim,
                        out_dim,
                        gate_output,
                        up_output,
                    );
                }
                GGUF_TYPE_Q6_K => {
                    return fused_gate_up_q6k_into(
                        &gate_weight.data,
                        &up_weight.data,
                        input,
                        in_dim,
                        out_dim,
                        gate_output,
                        up_output,
                    );
                }
                _ => {} // Fall through to rayon::join fallback
            }
        }

        // Fallback: two separate matmuls via rayon::join
        let (gate_result, up_result) = rayon::join(
            || self.fused_matmul_into(input, gate_weight, gate_output),
            || self.fused_matmul_into(input, up_weight, up_output),
        );
        gate_result?;
        up_result?;
        Ok(())
    }

    /// QKV projection matmul
    pub fn qkv_matmul(&self, input: &[f32], qkv: &OwnedQKVWeights) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.fused_matmul(input, weight),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                let seq_len = input.len() / hidden_dim;

                // P4: Parallel QKV projections — K+V overlap with Q tail
                let (q_out, (k_out, v_out)) = rayon::join(
                    || self.fused_matmul(input, q),
                    || rayon::join(
                        || self.fused_matmul(input, k),
                        || self.fused_matmul(input, v),
                    ),
                );
                let q_out = q_out?;
                let k_out = k_out?;
                let v_out = v_out?;

                // Interleave Q, K, V for each position
                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut output = Vec::with_capacity(seq_len * qkv_dim);
                for s in 0..seq_len {
                    output.extend_from_slice(&q_out[s * q.out_dim..(s + 1) * q.out_dim]);
                    output.extend_from_slice(&k_out[s * k.out_dim..(s + 1) * k.out_dim]);
                    output.extend_from_slice(&v_out[s * v.out_dim..(s + 1) * v.out_dim]);
                }
                Ok(output)
            },
        }
    }

    /// QKV matmul into pre-allocated buffer
    pub fn qkv_matmul_into(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        output: &mut [f32],
    ) -> Result<()> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => self.fused_matmul_into(input, weight, output),
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                let q_dim = q.out_dim;
                let k_dim = k.out_dim;
                let v_dim = v.out_dim;

                // P4: Parallel QKV projections — split output buffer, run concurrently
                let (q_out, kv_out) = output[..q_dim + k_dim + v_dim].split_at_mut(q_dim);
                let (k_out, v_out) = kv_out.split_at_mut(k_dim);

                let (q_res, (k_res, v_res)) = rayon::join(
                    || self.fused_matmul_into(input, q, q_out),
                    || rayon::join(
                        || self.fused_matmul_into(input, k, k_out),
                        || self.fused_matmul_into(input, v, v_out),
                    ),
                );
                q_res?;
                k_res?;
                v_res?;

                Ok(())
            },
        }
    }

    /// Layer normalization
    pub fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        ops::layer_norm(input, weight, bias, eps)
    }

    /// Add bias to activations
    pub fn add_bias(&self, input: &mut [f32], bias: &[f32]) {
        for (x, b) in input.iter_mut().zip(bias.iter()) {
            *x += b;
        }
    }

    /// GELU activation
    ///
    /// ONE PATH: Per-element delegates to `trueno::gelu_scalar` (UCBD §4).
    pub fn gelu(&self, input: &mut [f32]) {
        for x in input.iter_mut() {
            *x = trueno::gelu_scalar(*x);
        }
    }

    /// Fused RMSNorm + matmul helper
    fn fused_rmsnorm_matmul(
        &self,
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
        weight: &OwnedQuantizedTensor,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_rmsnorm_q4_0_matmul;

        // Only use fused path for Q4_0 weights (most common)
        if weight.qtype == GGUF_TYPE_Q4_0 && input.len() == weight.in_dim {
            return fused_rmsnorm_q4_0_matmul(
                input,
                norm_weight,
                eps,
                &weight.data,
                weight.in_dim,
                weight.out_dim,
            );
        }

        // Fallback to separate RMSNorm + matmul for other types
        let normed = ops::rms_norm(input, norm_weight, eps);
        self.fused_matmul(&normed, weight)
    }

    /// Fused RMSNorm + QKV matmul
    pub fn fused_rmsnorm_qkv_matmul(
        &self,
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
        qkv: &OwnedQKVWeights,
    ) -> Result<Vec<f32>> {
        match qkv {
            OwnedQKVWeights::Fused(ref weight) => {
                self.fused_rmsnorm_matmul(input, norm_weight, eps, weight)
            },
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                // For separate Q/K/V, normalize once and reuse
                let normed = ops::rms_norm(input, norm_weight, eps);

                // PMAT-114: Trace K weight to compare with APR
                static ONCE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if std::env::var("APR_TRACE_WEIGHTS").is_ok()
                    && !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    eprintln!(
                        "[PMAT-114-GGUF] K weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
                        k.in_dim,
                        k.out_dim,
                        k.qtype,
                        k.data.len()
                    );
                    // Dequantize first row completely to compare with APR
                    // Q4K: 144 bytes per super-block of 256 values, so first row = in_dim/256 super-blocks
                    let bytes_per_row = (k.in_dim.div_ceil(256)) * 144;
                    use crate::quantize::dequantize_q4_k_parallel;
                    if let Ok(dequant) = dequantize_q4_k_parallel(&k.data[0..bytes_per_row]) {
                        let row_mean: f32 = dequant.iter().sum::<f32>() / dequant.len() as f32;
                        let row_min = dequant.iter().cloned().fold(f32::INFINITY, f32::min);
                        let row_max = dequant.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        eprintln!("[PMAT-114-GGUF] K weight row 0 (dequant): mean={:.6}, min={:.6}, max={:.6}, len={}",
                            row_mean, row_min, row_max, dequant.len());
                        eprintln!(
                            "[PMAT-114-GGUF] K weight row 0 first10={:?}",
                            &dequant[..10.min(dequant.len())]
                        );
                    }
                }

                // P4: Parallel QKV projections — K+V overlap with Q tail
                let (q_out, (k_out, v_out)) = rayon::join(
                    || self.fused_matmul(&normed, q),
                    || rayon::join(
                        || self.fused_matmul(&normed, k),
                        || self.fused_matmul(&normed, v),
                    ),
                );
                let q_out = q_out?;
                let k_out = k_out?;
                let v_out = v_out?;

                let qkv_dim = q.out_dim + k.out_dim + v.out_dim;
                let mut output = Vec::with_capacity(qkv_dim);
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                Ok(output)
            },
        }
    }

    /// Fused RMSNorm + LM head
    pub fn fused_rmsnorm_lm_head(&self, input: &[f32]) -> Result<Vec<f32>> {
        use crate::quantize::fused_rmsnorm_q4_0_matmul;

        // Only use fused path for Q4_0 weights
        if self.lm_head_weight.qtype == GGUF_TYPE_Q4_0 && input.len() == self.lm_head_weight.in_dim
        {
            return fused_rmsnorm_q4_0_matmul(
                input,
                &self.output_norm_weight,
                self.config.eps,
                &self.lm_head_weight.data,
                self.lm_head_weight.in_dim,
                self.lm_head_weight.out_dim,
            );
        }

        // Fallback to separate RMSNorm + matmul for other types
        let normed = ops::rms_norm(input, &self.output_norm_weight, self.config.eps);
        self.fused_matmul(&normed, &self.lm_head_weight)
    }

    /// Fused RMSNorm + FFN up/gate projections for SwiGLU
    pub fn fused_rmsnorm_ffn_up_gate(
        &self,
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
        up_weight: &OwnedQuantizedTensor,
        gate_weight: &OwnedQuantizedTensor,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        use crate::quantize::fused_rmsnorm_ffn_up_gate;

        // Only use fused path for Q4_0 weights
        if up_weight.qtype == GGUF_TYPE_Q4_0
            && gate_weight.qtype == GGUF_TYPE_Q4_0
            && input.len() == up_weight.in_dim
            && up_weight.in_dim == gate_weight.in_dim
            && up_weight.out_dim == gate_weight.out_dim
        {
            return fused_rmsnorm_ffn_up_gate(
                input,
                norm_weight,
                eps,
                &up_weight.data,
                &gate_weight.data,
                up_weight.in_dim,
                up_weight.out_dim,
            );
        }

        // PMAT-309: Shared Q8K for Q4K gate+up FALSIFIED (-3.4%, 31.5 vs 32.6).
        // The Vec allocation overhead for quantize_for_q4k_matvec (3 Vecs)
        // offsets the saving from skipping one Q8K quantize.
        // The per-matmul path with stack-allocated Q8K is faster.
        let normed = ops::rms_norm(input, norm_weight, eps);
        let up_out = self.fused_matmul(&normed, up_weight)?;
        let gate_out = self.fused_matmul(&normed, gate_weight)?;
        Ok((up_out, gate_out))
    }

    /// Q8K QKV matmul into buffer
    ///
    /// Uses pre-quantized Q8K activations for faster matmul with Q4K weights.
    /// Dispatches to `fused_q4k_q8k_parallel_matvec_into` (maddubs-based, 32 vals/instr)
    /// instead of the f32 dequant path (8 vals/instr) for ~3-4x speedup on QKV projections.
    pub fn qkv_matmul_q8k_into(
        &self,
        input: &[f32],
        qkv: &OwnedQKVWeights,
        output: &mut [f32],
        scales: &[f32],
        quants: &[i8],
    ) -> Result<()> {
        use crate::quantize::fused_q4k_q8k_parallel_matvec_into;

        match qkv {
            OwnedQKVWeights::Fused(ref weight) => {
                if weight.qtype == GGUF_TYPE_Q4_K {
                    fused_q4k_q8k_parallel_matvec_into(
                        &weight.data,
                        scales,
                        quants,
                        weight.in_dim,
                        weight.out_dim,
                        output,
                    )
                } else {
                    self.fused_matmul_into(input, weight, output)
                }
            }
            OwnedQKVWeights::Separate {
                ref q,
                ref k,
                ref v,
            } => {
                let q_dim = q.out_dim;
                let k_dim = k.out_dim;
                let v_dim = v.out_dim;

                // P4: Parallel QKV projections — split output buffer, run concurrently
                let (q_out, kv_out) = output[..q_dim + k_dim + v_dim].split_at_mut(q_dim);
                let (k_out, v_out) = kv_out.split_at_mut(k_dim);

                let q_fn = || -> Result<()> {
                    if q.qtype == GGUF_TYPE_Q4_K {
                        fused_q4k_q8k_parallel_matvec_into(
                            &q.data, scales, quants, q.in_dim, q_dim, q_out,
                        )
                    } else {
                        self.fused_matmul_into(input, q, q_out)
                    }
                };
                let k_fn = || -> Result<()> {
                    if k.qtype == GGUF_TYPE_Q4_K {
                        fused_q4k_q8k_parallel_matvec_into(
                            &k.data, scales, quants, k.in_dim, k_dim, k_out,
                        )
                    } else {
                        self.fused_matmul_into(input, k, k_out)
                    }
                };
                let v_fn = || -> Result<()> {
                    if v.qtype == GGUF_TYPE_Q4_K {
                        fused_q4k_q8k_parallel_matvec_into(
                            &v.data, scales, quants, v.in_dim, v_dim, v_out,
                        )
                    } else {
                        self.fused_matmul_into(input, v, v_out)
                    }
                };

                let (q_res, (k_res, v_res)) = rayon::join(q_fn, || rayon::join(k_fn, v_fn));
                q_res?;
                k_res?;
                v_res?;
                Ok(())
            }
        }
    }

    /// Helper to dequantize weights for CUDA GEMM
    #[cfg(feature = "cuda")]
    fn dequantize_weight_for_cuda(&self, weight: &OwnedQuantizedTensor) -> Result<Vec<f32>> {
        use crate::quantize::{
            dequantize_q4_0, dequantize_q4_1, dequantize_q4_k, dequantize_q5_0, dequantize_q5_k,
            dequantize_q6_k, dequantize_q8_0,
        };

        match weight.qtype {
            // GH-242: F32 weights are already dequantized — reinterpret bytes
            GGUF_TYPE_F32 => {
                let floats: Vec<f32> = weight
                    .data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Ok(floats)
            }
            // GH-242: F16 weights — convert to F32
            GGUF_TYPE_F16 => {
                let floats: Vec<f32> = weight
                    .data
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            // GH-368: BF16 weights — convert to F32 (left-shift 16 bits)
            GGUF_TYPE_BF16 => {
                let floats: Vec<f32> = weight
                    .data
                    .chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                Ok(floats)
            }
            GGUF_TYPE_Q4_0 => dequantize_q4_0(&weight.data),
            GGUF_TYPE_Q4_1 => dequantize_q4_1(&weight.data),
            GGUF_TYPE_Q5_0 => dequantize_q5_0(&weight.data),
            GGUF_TYPE_Q8_0 => dequantize_q8_0(&weight.data),
            GGUF_TYPE_Q4_K => dequantize_q4_k(&weight.data),
            GGUF_TYPE_Q5_K => dequantize_q5_k(&weight.data),
            GGUF_TYPE_Q6_K => dequantize_q6_k(&weight.data),
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "dequantize_weight_for_cuda".to_string(),
                reason: format!("Unsupported quantization type: {}", weight.qtype),
            }),
        }
    }
}

include!("matmul_fused.rs");
