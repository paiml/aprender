//! PMAT-3477 / aprender#3090: executor wrappers for the six Gated `DeltaNet`
//! device kernels in `aprender-gpu/src/kernels/gdn/`.
//!
//! Each wrapper mirrors [`CudaExecutor::per_head_rmsnorm_into`] exactly:
//! `KernelType` arm -> kernel name -> PTX compiled once per shape and cached by
//! key -> `LaunchConfig` from the kernel's own `grid()`/`block()` -> launch ->
//! **the `graph_recording` push**. The push is not optional: the manual decode
//! graph is rebuilt ONLY from `graph_recorded_kernels`, so a kernel that does not
//! record itself is silently dropped from every replayed step (#3413 was exactly
//! that bug, for QK-norm).
//!
//! The CPU reference these reproduce is
//! `gguf/inference/forward/forward_qwen35.rs`; the per-layer parity contract
//! (relative L∞ <= 1e-3 against `forward_deltanet`) is proven in
//! `gguf/cuda/forward_qwen35_cuda_tests.rs`.

use super::*;

impl CudaExecutor {
    /// Compile (once) and fetch the module for a Gated `DeltaNet` kernel.
    ///
    /// Returns the kernel's entry name; the module lives in `self.modules`
    /// under `cache_key`.
    fn gdn_prepare(
        &mut self,
        kernel_type: &KernelType,
        cache_key: &str,
    ) -> Result<&'static str, GpuError> {
        let kernel_name = self.kernels.kernel_name(kernel_type);
        if !self.modules.contains_key(cache_key) {
            let ptx = self.kernels.generate_ptx(kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.to_string(), module);
        }
        Ok(kernel_name)
    }

    /// Launch a Gated `DeltaNet` kernel whose arguments are all device pointers,
    /// and record it for manual graph construction.
    ///
    /// Every GDN kernel folds its dimensions into the PTX as immediates, so the
    /// argument list is pointers only — which is also what makes `arg_data` a
    /// faithful replay record.
    fn gdn_launch(
        &mut self,
        cache_key: &str,
        kernel_name: &'static str,
        config: LaunchConfig,
        ptrs: &[u64],
    ) -> Result<(), GpuError> {
        for (i, &p) in ptrs.iter().enumerate() {
            validate_device_ptr(p, &format!("{kernel_name} arg {i}"))?;
        }
        let mut args: Vec<u64> = ptrs.to_vec();
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast::<std::ffi::c_void>())
            .collect();

        let module = self
            .modules
            .get_mut(cache_key)
            .expect("module just inserted");
        // SAFETY: every pointer was checked non-null above, each buffer is
        // allocated with the length the kernel's immediates encode, and the
        // argument order matches the kernel's `.param` declarations.
        unsafe {
            self.stream
                .launch_kernel(module, kernel_name, &config, &mut raw)?;
        }

        // trueno#243 / #3413: the manual decode graph is rebuilt ONLY from
        // `graph_recorded_kernels` — a kernel that skips this push runs during
        // capture and never again during replay.
        if self.graph_recording {
            let module = self.modules.get_mut(cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: args,
            });
        }
        Ok(())
    }

    /// Launch a Gated `DeltaNet` kernel that takes device pointers FOLLOWED BY
    /// scalar parameters, and record it for manual graph construction.
    ///
    /// The driver reads each argument slot as the parameter's declared width out
    /// of the low half of a `u64`: a `.param(PtxType::U32, …)` is
    /// `u64::from(x)` and a `.param(PtxType::F32, …)` is
    /// `u64::from(x.to_bits())`. Only `ptrs` are validated as device pointers —
    /// a scalar is not one, and `validate_device_ptr` would reject every small
    /// integer.
    fn gdn_launch_mixed(
        &mut self,
        cache_key: &str,
        kernel_name: &'static str,
        config: LaunchConfig,
        ptrs: &[u64],
        scalars: &[u64],
    ) -> Result<(), GpuError> {
        for (i, &p) in ptrs.iter().enumerate() {
            validate_device_ptr(p, &format!("{kernel_name} arg {i}"))?;
        }
        let mut args: Vec<u64> = ptrs.to_vec();
        args.extend_from_slice(scalars);
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast::<std::ffi::c_void>())
            .collect();

        let module = self
            .modules
            .get_mut(cache_key)
            .expect("module just inserted");
        // SAFETY: every pointer was checked non-null above, each buffer is
        // allocated with the length the kernel's immediates encode, the scalars
        // are the widths the kernel declares, and the argument order matches the
        // kernel's `.param` declarations.
        unsafe {
            self.stream
                .launch_kernel(module, kernel_name, &config, &mut raw)?;
        }

        // trueno#243 / #3413: the manual decode graph is rebuilt ONLY from
        // `graph_recorded_kernels` — a kernel that skips this push runs during
        // capture and never again during replay.
        if self.graph_recording {
            let module = self.modules.get_mut(cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: args,
            });
        }
        Ok(())
    }

    /// De-interleave the joint `[q | gate]` attention projection into two
    /// contiguous buffers, one `head_dim` slice of each per head.
    ///
    /// `src` is `[num_heads * 2 * head_dim]`; `q` and `gate` are
    /// `[num_heads * head_dim]`. Pure data movement — the result is
    /// bit-identical to the CPU's two `copy_from_slice`s.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    pub fn gdn_split_interleaved_into(
        &mut self,
        src: &GpuBuffer<f32>,
        q: &GpuBuffer<f32>,
        gate: &GpuBuffer<f32>,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        let kernel = trueno_gpu::kernels::gdn::SplitInterleavedKernel::new(num_heads, head_dim);
        let kernel_type = KernelType::GdnSplitInterleaved {
            num_heads,
            head_dim,
        };
        let cache_key = format!("gdn_split_interleaved_{num_heads}_{head_dim}");
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch(
            &cache_key,
            kernel_name,
            config,
            &[src.as_ptr(), q.as_ptr(), gate.as_ptr()],
        )
    }

    /// Partial NEOX RoPE over the first `n_rot` dimensions of each head of `x`,
    /// in place (`apply_partial_neox_rope`).
    ///
    /// `theta_scale` MUST be [`trueno_gpu::kernels::gdn::PartialNeoxRopeKernel::theta_scale`]
    /// (`freq_base.powf(-2.0 / n_rot)`), computed on the HOST exactly as the CPU
    /// reference computes it: the kernel rebuilds the CPU's iterative `theta` by
    /// multiplying it in, and the error is amplified by `j * theta_j`, so one
    /// ulp of a device-computed scale is already more than the parity budget.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    pub fn gdn_partial_neox_rope_into(
        &mut self,
        x: &GpuBuffer<f32>,
        num_heads: u32,
        head_dim: u32,
        n_rot: u32,
        position: u32,
        theta_scale: f32,
    ) -> Result<(), GpuError> {
        let kernel =
            trueno_gpu::kernels::gdn::PartialNeoxRopeKernel::new(num_heads, head_dim, n_rot);
        let kernel_type = KernelType::GdnPartialNeoxRope {
            num_heads,
            head_dim,
            n_rot,
        };
        let cache_key = format!("gdn_partial_neox_rope_{num_heads}_{head_dim}_{n_rot}");
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch_mixed(
            &cache_key,
            kernel_name,
            config,
            &[x.as_ptr()],
            &[u64::from(position), u64::from(theta_scale.to_bits())],
        )
    }

    /// Single-query decode attention over a resident KV cache, `head_dim <= 256`
    /// (the scores / softmax / value accumulation block of `forward_attention`).
    ///
    /// `k_cache` and `v_cache` are `[max_len][num_kv_heads * head_dim]`;
    /// `seq_len` is the number of valid positions, i.e. `position + 1`, and is a
    /// runtime parameter so one compiled module serves the whole decode.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_decode_attention_into(
        &mut self,
        q: &GpuBuffer<f32>,
        k_cache: &GpuBuffer<f32>,
        v_cache: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
    ) -> Result<(), GpuError> {
        let kernel = trueno_gpu::kernels::gdn::DecodeAttention256Kernel::new(
            num_heads,
            num_kv_heads,
            head_dim,
        );
        let kernel_type = KernelType::GdnDecodeAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        };
        let cache_key = format!("gdn_decode_attention_{num_heads}_{num_kv_heads}_{head_dim}");
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch_mixed(
            &cache_key,
            kernel_name,
            config,
            &[
                q.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                output.as_ptr(),
            ],
            &[u64::from(seq_len)],
        )
    }

    /// Fused causal depthwise conv1d + SiLU for one decode step (`causal_conv1d`
    /// plus the SiLU loop that follows it in `forward_deltanet`).
    ///
    /// `state` is `[channels * (kernel_size - 1)]` and is updated IN PLACE (the
    /// window shifts left and the new sample is appended), exactly as the CPU
    /// reference does. `weight` is `[channels * kernel_size]`, channel-outer.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    pub fn gdn_causal_conv1d_silu_into(
        &mut self,
        input: &GpuBuffer<f32>,
        state: &GpuBuffer<f32>,
        weight: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        channels: u32,
        kernel_size: u32,
    ) -> Result<(), GpuError> {
        let kernel = trueno_gpu::kernels::gdn::CausalConv1dSiluKernel::new(channels, kernel_size);
        let kernel_type = KernelType::GdnCausalConv1dSilu {
            channels,
            kernel_size,
        };
        let cache_key = format!("gdn_causal_conv1d_silu_{channels}_{kernel_size}");
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch(
            &cache_key,
            kernel_name,
            config,
            &[
                input.as_ptr(),
                state.as_ptr(),
                weight.as_ptr(),
                output.as_ptr(),
            ],
        )
    }

    /// Per-head L2 normalisation of `x`, in place (`l2_norm_per_head`).
    ///
    /// Epsilon is added to the SUM of squares, not to a mean — this is
    /// llama.cpp's `build_gdn_l2_norm`, not RMSNorm.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    pub fn gdn_per_head_l2_norm_into(
        &mut self,
        x: &GpuBuffer<f32>,
        head_dim: u32,
        num_heads: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel =
            trueno_gpu::kernels::gdn::PerHeadL2NormKernel::new(head_dim, num_heads, epsilon);
        let kernel_type = KernelType::GdnPerHeadL2Norm {
            head_dim,
            num_heads,
            epsilon,
        };
        // epsilon is an immediate in the PTX, so it belongs in the cache key.
        let cache_key = format!("gdn_per_head_l2_norm_{head_dim}_{num_heads}_{epsilon:e}");
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch(&cache_key, kernel_name, config, &[x.as_ptr()])
    }

    /// The per-head gates: `dt[h] = softplus(alpha[h] + dt_bias[h]) * a[h]` and
    /// `beta[h] = sigmoid(beta_raw[h])`.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gates_into(
        &mut self,
        alpha: &GpuBuffer<f32>,
        dt_bias: &GpuBuffer<f32>,
        a: &GpuBuffer<f32>,
        beta_raw: &GpuBuffer<f32>,
        dt_out: &GpuBuffer<f32>,
        beta_out: &GpuBuffer<f32>,
        num_heads: u32,
    ) -> Result<(), GpuError> {
        let kernel = trueno_gpu::kernels::gdn::GdnGatesKernel::new(num_heads);
        let kernel_type = KernelType::GdnGates { num_heads };
        let cache_key = format!("gdn_gates_{num_heads}");
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch(
            &cache_key,
            kernel_name,
            config,
            &[
                alpha.as_ptr(),
                dt_bias.as_ptr(),
                a.as_ptr(),
                beta_raw.as_ptr(),
                dt_out.as_ptr(),
                beta_out.as_ptr(),
            ],
        )
    }

    /// The gated delta-rule recurrence for one token
    /// (`delta_rule_recurrence`). `state` is `[num_v_heads * D * D]` and is
    /// updated in place; `output` is `[num_v_heads * D]`.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_delta_rule_into(
        &mut self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        beta: &GpuBuffer<f32>,
        gate: &GpuBuffer<f32>,
        state: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        num_v_heads: u32,
        head_v_dim: u32,
    ) -> Result<(), GpuError> {
        let kernel =
            trueno_gpu::kernels::gdn::DeltaRuleRecurrenceKernel::new(num_v_heads, head_v_dim);
        let kernel_type = KernelType::GdnDeltaRule {
            num_v_heads,
            head_v_dim,
        };
        let cache_key = format!("gdn_delta_rule_{num_v_heads}_{head_v_dim}");
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch(
            &cache_key,
            kernel_name,
            config,
            &[
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                beta.as_ptr(),
                gate.as_ptr(),
                state.as_ptr(),
                output.as_ptr(),
            ],
        )
    }

    /// Gated RMSNorm (`gated_rmsnorm`): the INPUT is normalised per head, the
    /// gate is not. `weight` is `[head_dim]`, shared across heads.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gated_rmsnorm_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gate: &GpuBuffer<f32>,
        weight: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        head_dim: u32,
        num_heads: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel =
            trueno_gpu::kernels::gdn::GatedRmsNormKernel::new(head_dim, num_heads, epsilon);
        let kernel_type = KernelType::GdnGatedRmsNorm {
            head_dim,
            num_heads,
            epsilon,
        };
        let cache_key = format!("gdn_gated_rmsnorm_{head_dim}_{num_heads}_{epsilon:e}");
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch(
            &cache_key,
            kernel_name,
            config,
            &[
                input.as_ptr(),
                gate.as_ptr(),
                weight.as_ptr(),
                output.as_ptr(),
            ],
        )
    }

    /// `x[i] *= sigmoid(gate[i])`, in place (`apply_sigmoid_gate`) — the output
    /// gate of Qwen3.5's full-attention layers.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    pub fn gdn_sigmoid_gate_into(
        &mut self,
        x: &GpuBuffer<f32>,
        gate: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<(), GpuError> {
        let kernel = trueno_gpu::kernels::gdn::SigmoidGateKernel::new(n);
        let kernel_type = KernelType::GdnSigmoidGate { n };
        let cache_key = format!("gdn_sigmoid_gate_{n}");
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch(
            &cache_key,
            kernel_name,
            config,
            &[x.as_ptr(), gate.as_ptr()],
        )
    }
}
