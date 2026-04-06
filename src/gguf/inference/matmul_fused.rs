/// CPU matmul for 2-byte-per-element float formats (BF16, F16)
/// Shared by BF16 and F16 paths — same structure, different decode.
fn float16_matmul(
    input: &[f32],
    data: &[u8],
    in_dim: usize,
    out_dim: usize,
    seq_len: usize,
    decode: fn(u16) -> f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let mut all_output = Vec::with_capacity(seq_len * out_dim);
    for s in 0..seq_len {
        let x = &input[s * in_dim..(s + 1) * in_dim];

        let row_output: Vec<f32> = (0..out_dim)
            .into_par_iter()
            .map(|row| {
                let row_byte_start = row * in_dim * 2;
                let mut sum = 0.0f32;
                for col in 0..in_dim {
                    let offset = row_byte_start + col * 2;
                    if offset + 1 < data.len() {
                        let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                        sum += decode(bits) * x[col];
                    }
                }
                sum
            })
            .collect();

        all_output.extend_from_slice(&row_output);
    }
    all_output
}

impl OwnedQuantizedModel {
    /// Look up token embeddings (public for debugging PAR-001)
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let start = (token_id as usize) * hidden_dim;
            let end = start + hidden_dim;
            if end <= self.token_embedding.len() {
                embeddings.extend_from_slice(&self.token_embedding[start..end]);
            } else {
                // N-09: OOB token → zeros. Contract: embedding-lookup-v1.yaml
                eprintln!(
                    "Warning: OwnedQuantizedModel::embed token_id {} OOB (end={end}, len={}). N-09 escape.",
                    token_id, self.token_embedding.len()
                );
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// Look up single token embedding into pre-allocated buffer (IMP-131)
    pub(crate) fn embed_into(&self, token_id: u32, output: &mut [f32]) {
        let hidden_dim = self.config.hidden_dim;
        let start = (token_id as usize) * hidden_dim;
        let end = start + hidden_dim;
        if end <= self.token_embedding.len() {
            output[..hidden_dim].copy_from_slice(&self.token_embedding[start..end]);
        } else {
            // N-09: OOB token → zeros. Contract: embedding-lookup-v1.yaml
            eprintln!(
                "Warning: embed_into token_id {} OOB (end={end}, len={}). N-09 escape.",
                token_id, self.token_embedding.len()
            );
            output[..hidden_dim].iter_mut().for_each(|x| *x = 0.0);
        }
    }

    /// Fused dequantize + matmul for quantized weights
    ///
    /// Supports F32, BF16, F16, Q4_0, Q8_0, Q4_1, Q5_0, Q4_K, Q5_K, Q6_K formats.
    /// Uses SIMD-accelerated implementations for optimal performance.
    pub(crate) fn fused_matmul(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
    ) -> Result<Vec<f32>> {
        use crate::quantize::{dequantize_q4_1, dequantize_q5_0};
        use trueno::{Matrix as TruenoMatrix, Vector as TruenoVector};

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let seq_len = input.len() / in_dim;

        // CUDA path when enabled
        #[cfg(feature = "cuda")]
        if let Some(ref executor_mutex) = self.cuda_executor {
            return self.fused_matmul_cuda(input, weight, executor_mutex);
        }

        // CPU path: F32 weights — rayon parallel dot products (zero-copy on raw bytes)
        if weight.qtype == GGUF_TYPE_F32 {
            return Ok(self.fused_matmul_f32(input, &weight.data, in_dim, out_dim, seq_len));
        }

        // CPU path: BF16 weights — GH-368
        // BF16→F32: f32::from_bits((bits as u32) << 16)
        if weight.qtype == GGUF_TYPE_BF16 {
            return Ok(float16_matmul(
                input, &weight.data, in_dim, out_dim, seq_len,
                |bits| f32::from_bits((bits as u32) << 16),
            ));
        }

        // CPU path: F16 weights
        if weight.qtype == GGUF_TYPE_F16 {
            return Ok(float16_matmul(
                input, &weight.data, in_dim, out_dim, seq_len,
                |bits| half::f16::from_bits(bits).to_f32(),
            ));
        }

        // CPU path: Fused integer SIMD matmul for Q4_0, Q8_0
        if weight.qtype == GGUF_TYPE_Q4_0 || weight.qtype == GGUF_TYPE_Q8_0 {
            return self.fused_matmul_q4_q8(input, weight, in_dim, out_dim, seq_len);
        }

        // CPU path: Dequantize + SIMD matmul for Q4_1, Q5_0
        if weight.qtype == GGUF_TYPE_Q4_1 || weight.qtype == GGUF_TYPE_Q5_0 {
            let weights_f32 = if weight.qtype == GGUF_TYPE_Q4_1 {
                dequantize_q4_1(&weight.data)?
            } else {
                dequantize_q5_0(&weight.data)?
            };
            let label = if weight.qtype == GGUF_TYPE_Q4_1 { "Q4_1" } else { "Q5_0" };

            let weight_matrix = TruenoMatrix::from_vec(out_dim, in_dim, weights_f32)
                .map_err(|_| RealizarError::InvalidShape {
                    reason: format!("Failed to create weight matrix for {label}"),
                })?;

            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let x_vec = TruenoVector::from_slice(x);
                let r = weight_matrix.matvec(&x_vec).map_err(|_| RealizarError::InvalidShape {
                    reason: format!("SIMD matvec failed for {label}"),
                })?;
                output.extend_from_slice(r.as_slice());
            }
            return Ok(output);
        }

        // CPU path: Fused K-quant kernels for Q4_K, Q5_K, Q6_K
        self.fused_matmul_k_quants(input, weight, in_dim, out_dim, seq_len)
    }

    /// F32 zero-copy rayon matmul (extracted for complexity)
    fn fused_matmul_f32(
        &self,
        input: &[f32],
        data: &[u8],
        in_dim: usize,
        out_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        use rayon::prelude::*;

        let mut all_output = Vec::with_capacity(seq_len * out_dim);
        for s in 0..seq_len {
            let x = &input[s * in_dim..(s + 1) * in_dim];

            let row_output: Vec<f32> = (0..out_dim)
                .into_par_iter()
                .map(|row| {
                    let row_byte_start = row * in_dim * 4;
                    let mut sum = 0.0f32;
                    let chunks = in_dim / 4;
                    let remainder = in_dim % 4;
                    for chunk in 0..chunks {
                        let base = row_byte_start + chunk * 16;
                        let w0 = f32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]]);
                        let w1 = f32::from_le_bytes([data[base + 4], data[base + 5], data[base + 6], data[base + 7]]);
                        let w2 = f32::from_le_bytes([data[base + 8], data[base + 9], data[base + 10], data[base + 11]]);
                        let w3 = f32::from_le_bytes([data[base + 12], data[base + 13], data[base + 14], data[base + 15]]);
                        let col = chunk * 4;
                        sum += w0 * x[col] + w1 * x[col + 1] + w2 * x[col + 2] + w3 * x[col + 3];
                    }
                    for i in 0..remainder {
                        let col = chunks * 4 + i;
                        let offset = row_byte_start + col * 4;
                        let w = f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
                        sum += w * x[col];
                    }
                    sum
                })
                .collect();

            all_output.extend_from_slice(&row_output);
        }
        all_output
    }

    /// Fused integer SIMD matmul for Q4_0 and Q8_0
    fn fused_matmul_q4_q8(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        in_dim: usize,
        out_dim: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        use crate::quantize::{fused_q4_0_q8_0_parallel_matvec, fused_q8_0_q8_0_parallel_matvec};

        let matvec_fn = if weight.qtype == GGUF_TYPE_Q4_0 {
            fused_q4_0_q8_0_parallel_matvec
        } else {
            fused_q8_0_q8_0_parallel_matvec
        };

        if seq_len == 1 {
            return matvec_fn(&weight.data, input, in_dim, out_dim);
        }
        let mut output = Vec::with_capacity(seq_len * out_dim);
        for s in 0..seq_len {
            let x = &input[s * in_dim..(s + 1) * in_dim];
            let row_output = matvec_fn(&weight.data, x, in_dim, out_dim)?;
            output.extend_from_slice(&row_output);
        }
        Ok(output)
    }

    /// Fused K-quant kernels for Q4_K, Q5_K, Q6_K
    fn fused_matmul_k_quants(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        in_dim: usize,
        out_dim: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        use crate::quantize::{
            fused_q4k_parallel_matvec, fused_q5k_parallel_matvec, fused_q6k_parallel_matvec,
        };

        if seq_len > 1 {
            let mut output = Vec::with_capacity(seq_len * out_dim);
            for s in 0..seq_len {
                let x = &input[s * in_dim..(s + 1) * in_dim];
                let row_output = match weight.qtype {
                    GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(&weight.data, x, in_dim, out_dim)?,
                    GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec(&weight.data, x, in_dim, out_dim)?,
                    GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(&weight.data, x, in_dim, out_dim)?,
                    _ => {
                        return Err(RealizarError::UnsupportedOperation {
                            operation: "owned_fused_matmul".to_string(),
                            reason: format!(
                                "Fused matmul only supports F32/BF16/F16/Q4_0/Q4_1/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K, got type {}",
                                weight.qtype
                            ),
                        });
                    },
                };
                output.extend_from_slice(&row_output);
            }
            Ok(output)
        } else {
            match weight.qtype {
                GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(&weight.data, input, in_dim, out_dim),
                GGUF_TYPE_Q5_K => fused_q5k_parallel_matvec(&weight.data, input, in_dim, out_dim),
                GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(&weight.data, input, in_dim, out_dim),
                _ => Err(RealizarError::UnsupportedOperation {
                    operation: "owned_fused_matmul".to_string(),
                    reason: format!(
                        "Fused matmul only supports F32/BF16/F16/Q4_0/Q4_1/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K, got type {}",
                        weight.qtype
                    ),
                }),
            }
        }
    }

    /// CUDA path for fused matmul
    #[cfg(feature = "cuda")]
    fn fused_matmul_cuda(
        &self,
        input: &[f32],
        weight: &OwnedQuantizedTensor,
        executor_mutex: &std::sync::Mutex<crate::cuda::CudaExecutor>,
    ) -> Result<Vec<f32>> {
        use tracing::info_span;

        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let seq_len = input.len() / in_dim;
        let gemm_start = std::time::Instant::now();
        let mut output = vec![0.0f32; seq_len * out_dim];

        // Use native quantized GEMV kernels for single-token generation
        if seq_len == 1 {
            let cache_key = format!(
                "{}_{:016x}",
                match weight.qtype {
                    GGUF_TYPE_Q4_K => "q4k",
                    GGUF_TYPE_Q5_K => "q5k",
                    GGUF_TYPE_Q6_K => "q6k",
                    _ => "unknown",
                },
                weight.data.as_ptr() as usize
            );

            if weight.qtype == GGUF_TYPE_Q4_K
                || weight.qtype == GGUF_TYPE_Q5_K
                || weight.qtype == GGUF_TYPE_Q6_K
            {
                let mut executor =
                    executor_mutex
                        .lock()
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_lock".to_string(),
                            reason: format!("Failed to acquire CUDA executor lock: {e}"),
                        })?;

                executor
                    .make_current()
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "cuda_make_current".to_string(),
                        reason: format!("Failed to set CUDA context current: {e}"),
                    })?;

                if !executor.has_quantized_weights(&cache_key) {
                    executor
                        .load_quantized_weights(&cache_key, &weight.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "cuda_cache".to_string(),
                            reason: format!("Failed to cache weights: {e}"),
                        })?;
                }

                let result = match weight.qtype {
                    GGUF_TYPE_Q4_K => executor.q4k_gemv_cached(
                        &cache_key,
                        input,
                        &mut output,
                        out_dim as u32,
                        in_dim as u32,
                    ),
                    GGUF_TYPE_Q5_K => executor.q5k_gemv_cached(
                        &cache_key,
                        input,
                        &mut output,
                        out_dim as u32,
                        in_dim as u32,
                    ),
                    GGUF_TYPE_Q6_K => executor.q6k_gemv_cached(
                        &cache_key,
                        input,
                        &mut output,
                        out_dim as u32,
                        in_dim as u32,
                    ),
                    _ => unreachable!(),
                };

                result.map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_gemv".to_string(),
                    reason: format!("CUDA GEMV failed: {e}"),
                })?;

                let gemm_duration_us = gemm_start.elapsed().as_micros() as u64;
                let _span = info_span!(
                    "gpu_kernel:gemv",
                    gpu.backend = "cuda",
                    gpu.dimensions.n = out_dim,
                    gpu.dimensions.k = in_dim,
                    duration_us = gemm_duration_us,
                )
                .entered();

                self.cuda_kernel_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                return Ok(output);
            }
        }

        // Fallback: Dequantize and use FP32 GEMM
        let dequant_weight = self.dequantize_weight_for_cuda(weight)?;

        {
            let mut executor =
                executor_mutex
                    .lock()
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "cuda_gemm_lock".to_string(),
                        reason: format!("Failed to acquire CUDA executor lock: {e}"),
                    })?;

            executor
                .make_current()
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_make_current".to_string(),
                    reason: format!("Failed to set CUDA context current: {e}"),
                })?;

            executor
                .gemm(
                    input,
                    &dequant_weight,
                    &mut output,
                    seq_len as u32,
                    out_dim as u32,
                    in_dim as u32,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_gemm".to_string(),
                    reason: format!("CUDA GEMM failed: {e}"),
                })?;
        }

        self.cuda_kernel_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(output)
    }
}
