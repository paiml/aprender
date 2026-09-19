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
