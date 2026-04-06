//! PMAT-291: KernelDispatch implementation for CudaExecutor.
//!
//! Connects trueno's tensor graph executor to realizr's existing kernel
//! dispatch functions. Each TensorOp is delegated to the corresponding
//! CudaExecutor method (batched_gemv_or_gemm, batched_rmsnorm_ptr_into, etc.).
//!
//! This is the bridge between the ~14-node graph and the actual GPU kernels.

use trueno_gpu::graph::executor::KernelDispatch;
use trueno_gpu::graph::TensorNode;
use trueno_gpu::GpuError;

use super::CudaExecutor;

impl KernelDispatch for CudaExecutor {
    fn dispatch_mul_mat(
        &mut self,
        node: &TensorNode,
        input_ptr: u64,
        output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = node.params.weight_ptr;

        // SAFETY: pointers are valid device allocations from the graph
        let input_buf = unsafe {
            trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(input_ptr, (m * k) as usize)
        };
        let output_buf = unsafe {
            trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(output_ptr, (m * n) as usize)
        };

        // PMAT-295: Use inline Q8 DP4A GEMV when enabled.
        // Single kernel launch (Q8 quantize fused into DP4A) for M=2-4 Q4K.
        // At M>=5, FP8 cuBLASLt fires. At M=1, existing DP4A with Q8 cache works.
        let use_inline_q8 = Self::use_inline_q8_gemv()
            && m >= 2
            && m <= 8
            && self.gpu_profile.q4k == crate::cuda::gpu_profile::Q4kVariant::HwDp4a
            && !self.is_prefilling;

        if use_inline_q8 {
            self.inline_q8_dp4a_q4k_gemv_into(weight_ptr, &input_buf, &output_buf, m, n, k)?;
        } else {
            self.batched_gemv_or_gemm(
                crate::cuda::types::WeightQuantType::Q4K,
                weight_ptr,
                &input_buf,
                &output_buf,
                input_ptr,
                output_ptr,
                m,
                n,
                k,
            )?;
        }

        std::mem::forget(input_buf);
        std::mem::forget(output_buf);
        Ok(())
    }

    fn dispatch_rms_norm(
        &mut self,
        node: &TensorNode,
        input_ptr: u64,
        output_ptr: u64,
        hidden_dim: u32,
        m: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let gamma_ptr = node.params.gamma_ptr;
        let gamma_len = hidden_dim as usize;

        // SAFETY: pointers are valid device allocations
        let input_buf = unsafe {
            trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(
                input_ptr,
                (m * hidden_dim) as usize,
            )
        };
        let output_buf = unsafe {
            trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(
                output_ptr,
                (m * hidden_dim) as usize,
            )
        };

        self.batched_rmsnorm_ptr_into(
            &input_buf,
            gamma_ptr,
            gamma_len,
            &output_buf,
            hidden_dim,
            m,
            epsilon,
        )?;

        // PMAT-294: Invalidate Q8 cache — RMSNorm writes new content to output buffer
        // which the next GEMV will read as input.
        self.q8_activation_valid = false;

        std::mem::forget(input_buf);
        std::mem::forget(output_buf);
        Ok(())
    }

    fn dispatch_add(
        &mut self,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n_elements: usize,
    ) -> Result<(), GpuError> {
        let a_buf =
            unsafe { trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(a_ptr, n_elements) };
        let b_buf =
            unsafe { trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(b_ptr, n_elements) };
        let out_buf =
            unsafe { trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(output_ptr, n_elements) };

        // hidden_dim is n_elements / m, but for residual add it's element-wise
        // Use the batched residual add with m=1 for simplicity
        self.batched_residual_add_into(&a_buf, &b_buf, &out_buf, n_elements as u32, 1)?;

        // PMAT-294: Invalidate Q8 cache — residual add writes new content
        self.q8_activation_valid = false;

        std::mem::forget(a_buf);
        std::mem::forget(b_buf);
        std::mem::forget(out_buf);
        Ok(())
    }

    fn dispatch_rope(
        &mut self,
        _node: &TensorNode,
        _qk_ptr: u64,
        _positions: &[u32],
        _head_dim: u32,
        _num_heads: u32,
    ) -> Result<(), GpuError> {
        // RoPE is handled as part of the compound attention dispatch
        // (dispatch_attention applies RoPE + KV scatter + attention).
        // Standalone RoPE nodes are not used in the current graph.
        Ok(())
    }

    fn dispatch_attention(
        &mut self,
        _node: &TensorNode,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        output_ptr: u64,
        m: u32,
        layer_idx: usize,
    ) -> Result<(), GpuError> {
        // Compound operation: RoPE on Q/K + KV cache scatter + attention.
        // Positions are read from self.graph_dispatch_positions (set before execute_graph).
        let positions = self.graph_dispatch_positions.clone();
        let num_heads = self.kv_num_heads as u32;
        let num_kv_heads = self.kv_num_kv_heads as u32;
        let head_dim = self.kv_head_dim as u32;
        let theta = self.rope_theta;
        let q_dim = (num_heads * head_dim) as usize;
        let kv_dim = (num_kv_heads * head_dim) as usize;

        // SAFETY: pointers are valid device allocations from workspace
        let q_buf = unsafe {
            trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(q_ptr, m as usize * q_dim)
        };
        let k_buf = unsafe {
            trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(k_ptr, m as usize * kv_dim)
        };
        let v_buf = unsafe {
            trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(v_ptr, m as usize * kv_dim)
        };
        let attn_out = unsafe {
            trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(output_ptr, m as usize * q_dim)
        };

        // Upload positions to device
        let positions_buf_ptr = self
            .workspace
            .positions_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-291: positions_buf not initialized".to_string())
            })?
            .as_ptr();
        let mut positions_buf = unsafe {
            trueno_gpu::driver::GpuBuffer::<u32>::from_raw_parts(positions_buf_ptr, m as usize)
        };

        if !self.is_capturing {
            positions_buf
                .copy_from_host(&positions)
                .map_err(|e| GpuError::Transfer(format!("PMAT-291 positions: {e}")))?;
        }

        // RoPE on Q and K
        if self.rope_type == 2 {
            self.batched_rope_neox_into(
                &q_buf,
                &q_buf,
                &positions_buf,
                num_heads,
                head_dim,
                m,
                theta,
            )?;
            self.batched_rope_neox_into(
                &k_buf,
                &k_buf,
                &positions_buf,
                num_kv_heads,
                head_dim,
                m,
                theta,
            )?;
        } else {
            self.batched_rope_into(
                &q_buf,
                &q_buf,
                &positions_buf,
                num_heads,
                head_dim,
                m,
                theta,
            )?;
            self.batched_rope_into(
                &k_buf,
                &k_buf,
                &positions_buf,
                num_kv_heads,
                head_dim,
                m,
                theta,
            )?;
        }

        // Attention (batched incremental or flash decode)
        if self.batched_kv_stride > 0 && self.batched_kv_k_caches.contains_key(&layer_idx) {
            let max_seq_len = self
                .batched_kv_lengths
                .iter()
                .take(m as usize)
                .copied()
                .max()
                .unwrap_or(0);

            if self.flash_decode_enabled && max_seq_len > 1024 {
                self.flash_decoding_attention_into(
                    layer_idx, &q_buf, &k_buf, &v_buf, &attn_out, m as usize, &positions,
                )?;
            } else {
                self.batched_incremental_attention_into(
                    layer_idx, &q_buf, &k_buf, &v_buf, &attn_out, m as usize, &positions,
                )?;
            }
        }

        // PMAT-294: Invalidate Q8 cache — attention writes new content to attn_out
        self.q8_activation_valid = false;

        std::mem::forget(q_buf);
        std::mem::forget(k_buf);
        std::mem::forget(v_buf);
        std::mem::forget(attn_out);
        std::mem::forget(positions_buf);
        Ok(())
    }

    fn dispatch_copy(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        size_bytes: usize,
    ) -> Result<(), GpuError> {
        self.stream.memcpy_dtod_sync(dst_ptr, src_ptr, size_bytes)
    }

    fn dispatch_mul(
        &mut self,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n_elements: usize,
    ) -> Result<(), GpuError> {
        // SwiGLU: output = gate * silu(up)
        // a_ptr = gate projection output, b_ptr = up projection output
        let gate_buf =
            unsafe { trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(a_ptr, n_elements) };
        let up_buf =
            unsafe { trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(b_ptr, n_elements) };
        let out_buf =
            unsafe { trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(output_ptr, n_elements) };

        // batched_swiglu_into expects (gate, up, output, dim, m).
        // For graph dispatch, we pass n_elements as dim with m=1 (flat dispatch).
        self.batched_swiglu_into(&gate_buf, &up_buf, &out_buf, n_elements as u32, 1)?;

        // PMAT-294: Invalidate Q8 cache — SwiGLU writes new content to output
        self.q8_activation_valid = false;

        std::mem::forget(gate_buf);
        std::mem::forget(up_buf);
        std::mem::forget(out_buf);
        Ok(())
    }

    fn dispatch_silu(
        &mut self,
        _input_ptr: u64,
        _output_ptr: u64,
        _n_elements: usize,
    ) -> Result<(), GpuError> {
        // SiLU is handled as part of dispatch_mul (SwiGLU = gate * silu(up)).
        // Standalone SiLU nodes are not used in the current graph.
        Ok(())
    }
}

impl CudaExecutor {
    /// PMAT-295: Check if inline Q8 DP4A GEMV is enabled (INLINE_Q8_GEMV=1).
    /// Cached after first check.
    fn use_inline_q8_gemv() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("INLINE_Q8_GEMV").as_deref() == Ok("1"))
    }
}
