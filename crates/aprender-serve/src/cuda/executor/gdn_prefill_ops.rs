//! PMAT-3596 (#3596): executor wrappers for the Qwen3.5 hybrid's batched prefill.
//!
//! Three groups:
//!
//! 1. **Projections as GEMMs.** [`CudaExecutor::qwen35_project_rows`] dequantizes one
//!    weight into the shared f32 scratch and runs a cuBLAS SGEMM over `rows` activation
//!    rows. This path is f32 end to end — deliberately NOT `cublas_prefill_gemm`, whose
//!    default on sm_89+ is FP8 and whose other legs are FP16/WMMA/DP4A: the hybrid feeds
//!    its projections into a recurrence, which compounds low-precision activation error
//!    (`Qwen35CudaModel::pin_float_gemv` documents the measured DP4A failure). The
//!    handle is `CUBLAS_PEDANTIC_MATH` (no TF32), so SGEMM here is fp32 in and out.
//! 2. **The row-batched Gated `DeltaNet` kernels** (`aprender-gpu` `kernels/gdn`): the
//!    chunk-resident delta-rule scan, conv1d over a chunk, and the row twins of the L2
//!    norm, gates and partial RoPE. Each is bitwise-equal to `T` launches of its
//!    per-token twin (measured, in `aprender-gpu`).
//! 3. **Causal attention over the resident KV cache** for a chunk of queries:
//!    QKᵀ → causal mask + softmax → PV, the dense prefill's cuBLAS pattern
//!    (`prefill_attention_cublas`) applied to the Qwen3.5 cache, whose
//!    `[pos][num_kv_heads * head_dim]` rows are already the packed layout cuBLAS reads
//!    with `lda = kv_dim`. cuBLAS has no `head_dim <= 128` limit, which is what keeps
//!    the 256-wide Qwen3.5 heads off every other prefill attention kernel.
//!
//! None of these launches is recorded for graph replay: prefill never runs under a
//! decode-graph capture, and a recording here would be replayed on every decode step.

use super::*;
use trueno_gpu::kernels::gdn::{
    CausalConv1dSiluSeqKernel, DeltaRuleChunkScanKernel, GdnGatesRowsKernel,
    PartialNeoxRopeRowsKernel, PerHeadL2NormRowsKernel,
};
use trueno_gpu::kernels::{
    Q4KDequantKernel, Q5KDequantKernel, Q6KDequantKernel, Q8_0DequantKernel,
};

impl CudaExecutor {
    /// Compile `kernel` for this device's target once, cached under `key`.
    fn qp_prepare<K: Kernel>(&mut self, key: &str, kernel: &K) -> Result<(), GpuError> {
        if !self.modules.contains_key(key) {
            let ptx = kernel.emit_ptx_for_target(&self.kernels.sm_target);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(key.to_string(), module);
        }
        Ok(())
    }

    /// Launch a prepared kernel. The first `n_ptrs` slots are device pointers and are
    /// checked non-null; the rest are scalars in the low half of their slot, which is
    /// how the driver reads a declared `u32`/`f32` parameter.
    fn qp_launch(
        &mut self,
        key: &str,
        name: &str,
        config: LaunchConfig,
        args: &mut [u64],
        n_ptrs: usize,
    ) -> Result<(), GpuError> {
        for (i, &p) in args.iter().take(n_ptrs).enumerate() {
            validate_device_ptr(p, &format!("{name} arg {i}"))?;
        }
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast::<std::ffi::c_void>())
            .collect();
        let module = self.modules.get_mut(key).expect("module prepared");
        // SAFETY: every pointer slot was checked non-null, the caller sized each buffer
        // for the launch it describes, and the slot order is the kernel's `.param` order.
        unsafe {
            self.stream.launch_kernel(module, name, &config, &mut raw)?;
        }
        Ok(())
    }

    /// Make the shared f32 dequant scratch hold at least `elems` floats.
    fn qp_dequant_scratch(&mut self, elems: usize) -> Result<u64, GpuError> {
        if self.dequant_scratch_size < elems || self.dequant_scratch.is_none() {
            self.dequant_scratch = Some(GpuBuffer::new(&self.context, elems)?);
            self.dequant_scratch_size = elems;
        }
        Ok(self
            .dequant_scratch
            .as_ref()
            .expect("dequant scratch just ensured")
            .as_ptr())
    }

    /// Dequantize the `[n × k]` weight at `w_ptr` into the f32 scratch and return the
    /// scratch pointer. An F32 weight is returned as it is.
    ///
    /// # Errors
    /// A quantization type with no dequant kernel (the prefill then refuses and the
    /// caller keeps the per-token path), or a compile/launch failure.
    pub(crate) fn qwen35_dequant_f32(
        &mut self,
        qtype: WeightQuantType,
        w_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        if qtype == WeightQuantType::F32 {
            return Ok(w_ptr);
        }
        let out = self.qp_dequant_scratch(n as usize * k as usize)?;
        let (key, name, grid) = match qtype {
            WeightQuantType::Q4K => {
                let kern = Q4KDequantKernel::new(k, n);
                let key = format!("qp_q4k_dequant_{k}_{n}");
                self.qp_prepare(&key, &kern)?;
                (key, "q4k_dequant_to_f32", (n, k.div_ceil(256)))
            },
            WeightQuantType::Q5K => {
                let kern = Q5KDequantKernel::new(k, n);
                let key = format!("qp_q5k_dequant_{k}_{n}");
                self.qp_prepare(&key, &kern)?;
                (key, "q5k_dequant_to_f32", (n, k.div_ceil(256)))
            },
            WeightQuantType::Q6K => {
                let kern = Q6KDequantKernel::new(k, n);
                let key = format!("qp_q6k_dequant_{k}_{n}");
                self.qp_prepare(&key, &kern)?;
                (key, "q6k_dequant_to_f32", (n, k.div_ceil(256)))
            },
            WeightQuantType::Q8_0 => {
                let kern = Q8_0DequantKernel::new(k, n);
                let key = format!("qp_q8_0_dequant_{k}_{n}");
                self.qp_prepare(&key, &kern)?;
                (key, "q8_0_dequant_to_f32", (n, k.div_ceil(32)))
            },
            other => {
                return Err(GpuError::InvalidParameter(format!(
                    "qwen35 prefill: no f32 dequant kernel for {other:?}"
                )))
            },
        };
        let config = LaunchConfig::grid_2d(grid.0, grid.1, 32, 1);
        let mut args = [out, w_ptr, u64::from(k), u64::from(n)];
        self.qp_launch(&key, name, config, &mut args, 2)?;
        Ok(out)
    }

    /// `Y[rows × n] = X[rows × k] · Wᵀ` for the quantized `[n × k]` weight `W`, all
    /// row-major; `Y`'s rows are `ldc` floats apart (`ldc >= n`), so the result can land
    /// in a strided destination such as the KV cache.
    ///
    /// # Errors
    /// See [`Self::qwen35_dequant_f32`]; also a cuBLAS failure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen35_project_rows(
        &mut self,
        qtype: WeightQuantType,
        w_ptr: u64,
        x_ptr: u64,
        y_ptr: u64,
        rows: u32,
        n: u32,
        k: u32,
        ldc: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(x_ptr, "qwen35_project_rows x")?;
        validate_device_ptr(y_ptr, "qwen35_project_rows y")?;
        let w_f32 = self.qwen35_dequant_f32(qtype, w_ptr, n, k)?;
        self.ensure_cublas()?;
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        // Column-major view: Yᵀ (n × rows, ld ldc) = W (k × n col-major, op T) · Xᵀ (k × rows).
        handle.gemm_f32(
            trueno_gpu::driver::GemmOp::Trans,
            trueno_gpu::driver::GemmOp::NoTrans,
            n as i32,
            rows as i32,
            k as i32,
            1.0,
            w_f32,
            k as i32,
            x_ptr,
            k as i32,
            0.0,
            y_ptr,
            ldc as i32,
        )
    }

    /// Causal conv1d + SiLU over `rows` rows (the window in `state` is advanced).
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    pub(crate) fn qwen35_conv1d_rows(
        &mut self,
        input: u64,
        state: u64,
        weight: u64,
        output: u64,
        channels: u32,
        kernel_size: u32,
        rows: u32,
    ) -> Result<(), GpuError> {
        let kern = CausalConv1dSiluSeqKernel::new(channels, kernel_size, channels, channels);
        let key = format!("qp_conv_seq_{channels}_{kernel_size}");
        self.qp_prepare(&key, &kern)?;
        let (gx, _, _) = kern.grid();
        let (bx, _, _) = kern.block();
        let mut args = [input, state, weight, output, u64::from(rows)];
        self.qp_launch(
            &key,
            kern.name(),
            LaunchConfig::grid_2d(gx, 1, bx, 1),
            &mut args,
            4,
        )
    }

    /// Per-head L2 norm of `num_heads` heads in each of `rows` rows `row_stride` apart.
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    pub(crate) fn qwen35_l2_norm_rows(
        &mut self,
        x: u64,
        head_dim: u32,
        num_heads: u32,
        eps: f32,
        row_stride: u32,
        rows: u32,
    ) -> Result<(), GpuError> {
        let kern = PerHeadL2NormRowsKernel::new(head_dim, num_heads, eps, row_stride);
        let key = format!(
            "qp_l2_rows_{head_dim}_{num_heads}_{}_{row_stride}",
            eps.to_bits()
        );
        self.qp_prepare(&key, &kern)?;
        let (gx, gy, _) = kern.grid(rows);
        let (bx, _, _) = kern.block();
        let mut args = [x];
        self.qp_launch(
            &key,
            kern.name(),
            LaunchConfig::grid_2d(gx, gy, bx, 1),
            &mut args,
            1,
        )
    }

    /// The `dt`/`beta` gates over `rows` rows of `num_heads` value heads.
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen35_gates_rows(
        &mut self,
        alpha: u64,
        dt_bias: u64,
        a: u64,
        beta_raw: u64,
        dt: u64,
        beta: u64,
        num_heads: u32,
        rows: u32,
    ) -> Result<(), GpuError> {
        let kern = GdnGatesRowsKernel::new(num_heads);
        let key = format!("qp_gates_rows_{num_heads}");
        self.qp_prepare(&key, &kern)?;
        let (gx, _, _) = kern.grid(rows);
        let (bx, _, _) = kern.block();
        let mut args = [
            alpha,
            dt_bias,
            a,
            beta_raw,
            dt,
            beta,
            u64::from(rows * num_heads),
        ];
        self.qp_launch(
            &key,
            kern.name(),
            LaunchConfig::grid_2d(gx, 1, bx, 1),
            &mut args,
            6,
        )
    }

    /// Partial NEOX RoPE over `rows` rows, row `t` at position `pos0 + t`.
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen35_rope_rows(
        &mut self,
        x: u64,
        num_heads: u32,
        head_dim: u32,
        n_rot: u32,
        row_stride: u32,
        rows: u32,
        pos0: u32,
        theta_scale: f32,
    ) -> Result<(), GpuError> {
        let kern = PartialNeoxRopeRowsKernel::new(num_heads, head_dim, n_rot, row_stride);
        let key = format!("qp_rope_rows_{num_heads}_{head_dim}_{n_rot}_{row_stride}");
        self.qp_prepare(&key, &kern)?;
        let (gx, gy, _) = kern.grid(rows);
        let (bx, _, _) = kern.block();
        let mut args = [x, u64::from(pos0), u64::from(theta_scale.to_bits())];
        self.qp_launch(
            &key,
            kern.name(),
            LaunchConfig::grid_2d(gx, gy, bx, 1),
            &mut args,
            1,
        )
    }

    /// The gated delta-rule recurrence over `rows` tokens, state resident on chip.
    ///
    /// `q`/`k`/`v` point at their sections of the `[rows][qkv_row_stride]` conv output;
    /// `beta`/`gate` are `[rows][num_v_heads]`, `output` `[rows][num_v_heads * head_v_dim]`.
    ///
    /// # Errors
    /// Compile/launch failure or a null pointer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen35_delta_rule_scan(
        &mut self,
        q: u64,
        k: u64,
        v: u64,
        beta: u64,
        gate: u64,
        state: u64,
        output: u64,
        dims: (u32, u32, u32, u32),
        qkv_row_stride: u32,
        rows: u32,
    ) -> Result<(), GpuError> {
        let (nk, dk, nv, dv) = dims;
        let kern = DeltaRuleChunkScanKernel::new(nk, dk, nv, dv, qkv_row_stride, nv * dv);
        let key = format!("qp_scan_{nk}_{dk}_{nv}_{dv}_{qkv_row_stride}");
        self.qp_prepare(&key, &kern)?;
        let (gx, _, _) = kern.grid();
        let (bx, _, _) = kern.block();
        let config = LaunchConfig {
            grid: (gx, 1, 1),
            block: (bx, 1, 1),
            shared_mem: 0, // static: the kernel declares its k/q staging buffer
        };
        let mut args = [q, k, v, beta, gate, state, output, u64::from(rows)];
        self.qp_launch(&key, kern.name(), config, &mut args, 7)
    }

    /// Causal attention for `rows` query rows at positions `pos0..pos0+rows` over the
    /// resident KV cache rows `0..pos0+rows`.
    ///
    /// `q` is `[rows][num_heads * head_dim]` (normed and rotated), `k_cache`/`v_cache`
    /// are `[max_len][num_kv_heads * head_dim]` with this chunk's rows already written,
    /// `out` is `[rows][num_heads * head_dim]`. `scores` must hold
    /// `heads_per_kv * rows * (pos0 + rows)` floats: the KV groups are processed one
    /// after another through the same scratch, which is what bounds it.
    ///
    /// Query head `h` reads KV head `h / (num_heads / num_kv_heads)` — the grouping of
    /// the CPU reference and of `DecodeAttention256Kernel`.
    ///
    /// # Errors
    /// cuBLAS or launch failure, or a null pointer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen35_prefill_attention(
        &mut self,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        out: u64,
        scores: u64,
        rows: u32,
        pos0: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        for (p, name) in [
            (q, "q"),
            (k_cache, "k_cache"),
            (v_cache, "v_cache"),
            (out, "out"),
            (scores, "scores"),
        ] {
            validate_device_ptr(p, &format!("qwen35_prefill_attention {name}"))?;
        }
        let hpk = num_heads / num_kv_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let total = pos0 + rows;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let per_head = i64::from(rows) * i64::from(total);
        self.ensure_cublas()?;
        if !self.modules.contains_key("causal_mask_softmax") {
            let module = self.compile_ptx(Self::CAUSAL_MASK_SOFTMAX_PTX)?;
            self.modules
                .insert("causal_mask_softmax".to_string(), module);
        }
        let f = std::mem::size_of::<f32>() as u64;
        for g in 0..num_kv_heads {
            let first = g * hpk;
            let k_g = k_cache + u64::from(g * head_dim) * f;
            let v_g = v_cache + u64::from(g * head_dim) * f;
            let q_g = q + u64::from(first * head_dim) * f;
            let o_g = out + u64::from(first * head_dim) * f;
            let handle = self.cublas_handle.as_ref().expect("cublas initialized");
            // scores[h][t][j] = (q_h[t] · k[j]) / sqrt(d): column-major C (total × rows).
            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::Trans,
                trueno_gpu::driver::GemmOp::NoTrans,
                total as i32,
                rows as i32,
                head_dim as i32,
                scale,
                k_g,
                kv_dim as i32,
                0,
                q_g,
                q_dim as i32,
                i64::from(head_dim),
                0.0,
                scores,
                total as i32,
                per_head,
                hpk as i32,
            )?;
            // Mask keys past pos0 + t and softmax each row, in place.
            let module = self
                .modules
                .get_mut("causal_mask_softmax")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (hpk, rows, 1),
                block: (32, 1, 1),
                shared_mem: 0,
            };
            let (mut s_ptr, mut m_val, mut tl_val, mut base_val, mut nh_val) =
                (scores, rows, total, pos0, hpk);
            // SAFETY: `scores` holds hpk * rows * total floats (the caller's contract),
            // and the scalars match the kernel's u32 params in order.
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "causal_mask_softmax",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut s_ptr).cast::<std::ffi::c_void>(),
                        std::ptr::from_mut(&mut m_val).cast::<std::ffi::c_void>(),
                        std::ptr::from_mut(&mut tl_val).cast::<std::ffi::c_void>(),
                        std::ptr::from_mut(&mut base_val).cast::<std::ffi::c_void>(),
                        std::ptr::from_mut(&mut nh_val).cast::<std::ffi::c_void>(),
                    ],
                )?;
            }
            // out_h[t] = sum_j p[h][t][j] v[j]: column-major C (head_dim × rows, ld q_dim).
            let handle = self.cublas_handle.as_ref().expect("cublas initialized");
            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::NoTrans,
                trueno_gpu::driver::GemmOp::NoTrans,
                head_dim as i32,
                rows as i32,
                total as i32,
                1.0,
                v_g,
                kv_dim as i32,
                0,
                scores,
                total as i32,
                per_head,
                0.0,
                o_g,
                q_dim as i32,
                i64::from(head_dim),
                hpk as i32,
            )?;
        }
        Ok(())
    }
}
