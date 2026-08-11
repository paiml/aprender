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

        // PMAT-809 (c): Gemma scales token embeddings by sqrt(hidden_size) after
        // lookup. Single choke point — ALL forward variants call embed(), so the
        // scaling applies uniformly. `None` for every non-Gemma arch = byte-identical.
        if let Some(scale) = self.config.embed_scale() {
            for v in &mut embeddings {
                *v *= scale;
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
        // PMAT-809 (c): Gemma embedding scaling — mirror embed() for the decode
        // hot path. None for non-Gemma = byte-identical.
        if let Some(scale) = self.config.embed_scale() {
            for v in &mut output[..hidden_dim] {
                *v *= scale;
            }
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

        // #1789 defensive guards: empty / undersized `weight.data` would
        // cause a cryptic `index out of bounds: the len is N but the index
        // is M` panic deep in the parallel matmul kernel. Most-likely cause
        // is a Qwen3-MoE-style per-expert tensor where the parent FFN
        // tensor was registered with empty data because the actual weights
        // live in per-expert slices the loader hasn't wired in. Bail early
        // with an actionable error instead of letting rayon workers crash.
        validate_matmul_weight_shape(weight)?;

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

        // GH-478: APR-native Q4 / Q8 — per-tensor scratch dequant.
        // Storage stays at 4-/8-bit; F32 expansion is bounded to one tensor's
        // working set instead of `4 × num_params` bytes at load time.
        if weight.qtype == APR_TYPE_Q4 || weight.qtype == APR_TYPE_Q8 {
            let num_elements = in_dim * out_dim;
            let weights_f32 = if weight.qtype == APR_TYPE_Q4 {
                crate::apr::dequant::dequantize_apr_q4(&weight.data, num_elements)
            } else {
                crate::apr::dequant::dequantize_apr_q8(&weight.data, num_elements)
            };
            let label = if weight.qtype == APR_TYPE_Q4 { "APR-Q4" } else { "APR-Q8" };

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

/// #1789 defensive guard for matmul: validate the weight buffer is
/// non-empty AND large enough for the declared `(in_dim, out_dim)` shape
/// (plus the F32 byte layout when `qtype == GGUF_TYPE_F32`). Returns
/// `RealizarError::InvalidShape` with an actionable message instead of
/// allowing the matmul kernel to panic with an opaque index-out-of-bounds.
///
/// The empty-data check fires for Qwen3-MoE-style models where the parent
/// FFN tensor is registered with an empty data buffer because the actual
/// weights live in per-expert slices the loader hasn't wired in — without
/// this guard, the panic site is deep in a parallel kernel and gives no
/// indication that the root cause is a tensor-loading issue.
///
/// Extracted as a free function so the validation logic is unit-testable
/// without constructing a full `OwnedQuantizedModel`.
fn validate_matmul_weight_shape(weight: &OwnedQuantizedTensor) -> Result<()> {
    if weight.data.is_empty() {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "matmul weight has EMPTY data buffer (in_dim={}, out_dim={}, qtype={}): \
                 the tensor is declared with a shape but carries 0 bytes. Known causes: \
                 (1) a tied-embedding model (e.g. dense Qwen2/Qwen2.5) whose lm_head.weight \
                 is declared but empty because it was never aliased to the populated token \
                 embedding matrix; (2) a MoE parent FFN tensor whose real weights live in \
                 per-expert slices the loader has not wired in. Run `apr tensors <model>` — \
                 the tensor showing a shape with 0 B of data is the one at fault. \
                 See aprender#1789",
                weight.in_dim, weight.out_dim, weight.qtype
            ),
        });
    }
    if weight.qtype == GGUF_TYPE_F32 {
        let expected_bytes = weight
            .out_dim
            .checked_mul(weight.in_dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!(
                    "F32 matmul: in_dim={} * out_dim={} * 4 overflows usize",
                    weight.in_dim, weight.out_dim
                ),
            })?;
        if weight.data.len() < expected_bytes {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "F32 matmul weight too small: have {} bytes, need {expected_bytes} \
                     (in_dim={}, out_dim={})",
                    weight.data.len(),
                    weight.in_dim,
                    weight.out_dim
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn mk_tensor(data: Vec<u8>, in_dim: usize, out_dim: usize, qtype: u32) -> OwnedQuantizedTensor {
        OwnedQuantizedTensor {
            data,
            in_dim,
            out_dim,
            qtype,
        }
    }

    #[test]
    fn validate_empty_data_fires_with_actionable_message() {
        let t = mk_tensor(vec![], 4096, 4096, GGUF_TYPE_F32);
        let err = validate_matmul_weight_shape(&t).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("EMPTY data buffer"),
            "must call out the empty-data root cause; got: {msg}"
        );
        assert!(
            msg.contains("aprender#1789"),
            "must reference the tracking issue; got: {msg}"
        );
        assert!(
            msg.contains("in_dim=4096"),
            "must include declared dims for diagnostics; got: {msg}"
        );
    }

    /// The message must not diagnose ONE cause it cannot have established.
    ///
    /// On `qwen2.5-coder-0.5b-instruct.apr` — a dense, tied-embedding model
    /// with no experts at all — this guard fired and told the user "likely a
    /// MoE per-expert tensor was registered with len-0 data". The actual state
    /// was `lm_head.weight [151936, 896] 0 B` beside a fully populated
    /// `model.embed_tokens.weight`: tied embeddings that were never resolved.
    /// A confident wrong diagnosis costs more than no diagnosis.
    #[test]
    fn validate_empty_data_does_not_blame_moe_alone() {
        let t = mk_tensor(vec![], 896, 151_936, GGUF_TYPE_F32);
        let msg = format!("{}", validate_matmul_weight_shape(&t).unwrap_err());
        assert!(
            msg.contains("tied-embedding") || msg.contains("lm_head"),
            "a dense tied-embedding model is the other known cause and must be \
             named; got: {msg}"
        );
        assert!(
            !msg.contains("likely a MoE"),
            "must not present MoE as the single likely cause; got: {msg}"
        );
        assert!(
            msg.contains("apr tensors"),
            "must tell the user how to find the offending tensor; got: {msg}"
        );
    }

    #[test]
    fn validate_f32_undersized_fires_with_byte_count() {
        // Declared 16×16 F32 = 16 * 16 * 4 = 1024 bytes needed.
        // Provide only 100 bytes — should error with concrete counts.
        let t = mk_tensor(vec![0u8; 100], 16, 16, GGUF_TYPE_F32);
        let err = validate_matmul_weight_shape(&t).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("F32 matmul weight too small"), "got: {msg}");
        assert!(msg.contains("have 100 bytes"), "got: {msg}");
        assert!(msg.contains("need 1024"), "got: {msg}");
    }

    #[test]
    fn validate_f32_sized_correctly_passes() {
        // 16×16 F32 with exactly 1024 bytes is fine.
        let t = mk_tensor(vec![0u8; 1024], 16, 16, GGUF_TYPE_F32);
        assert!(validate_matmul_weight_shape(&t).is_ok());
    }

    #[test]
    fn validate_f32_oversized_data_passes() {
        // Padding is allowed (some GGUF readers pad to alignment); only
        // undersized fails.
        let t = mk_tensor(vec![0u8; 2048], 16, 16, GGUF_TYPE_F32);
        assert!(validate_matmul_weight_shape(&t).is_ok());
    }

    #[test]
    fn validate_non_f32_only_checks_emptiness() {
        // Quantized formats (Q4_K, etc.) have their own byte layouts that
        // aren't `out_dim * in_dim * 4`. The guard only checks that data
        // isn't empty for non-F32 types; layout validation lives in the
        // dequantize kernels.
        let t = mk_tensor(vec![0u8; 1], 4096, 4096, 12); // qtype=12 = GGUF_TYPE_Q4_K
        assert!(
            validate_matmul_weight_shape(&t).is_ok(),
            "1-byte non-F32 data must pass the early guard (full layout check is downstream)"
        );
    }

    #[test]
    fn validate_overflow_fires_with_message() {
        // usize overflow when in_dim * out_dim * 4 exceeds usize::MAX.
        // On a 64-bit host this requires dims that multiply to >2^62.
        let t = mk_tensor(vec![0u8; 1], usize::MAX / 2, 5, GGUF_TYPE_F32);
        let err = validate_matmul_weight_shape(&t).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("overflows usize"), "got: {msg}");
    }
}
