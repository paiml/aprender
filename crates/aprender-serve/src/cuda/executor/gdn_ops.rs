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

/// The most argument slots any Gated `DeltaNet` kernel takes.
const GDN_MAX_ARGS: usize = 16;

/// Pack `ptrs` then `scalars` into `args`; returns the slot count.
fn gdn_pack_args(
    kernel_name: &str,
    args: &mut [u64; GDN_MAX_ARGS],
    ptrs: &[u64],
    scalars: &[u64],
) -> Result<usize, GpuError> {
    let n = ptrs.len() + scalars.len();
    if n > GDN_MAX_ARGS {
        return Err(GpuError::InvalidParameter(format!(
            "{kernel_name}: {n} arguments exceed {GDN_MAX_ARGS}"
        )));
    }
    args[..ptrs.len()].copy_from_slice(ptrs);
    args[ptrs.len()..n].copy_from_slice(scalars);
    Ok(n)
}

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
            validate_kernel_arg(p, kernel_name, i)?;
            // #4258: any pointer argument may be an output (several GDN kernels
            // write in place), so any of them can stale the Q8 activation.
            self.q8_activation_written(p);
        }
        // #4215: argument slots live on the stack; the `Vec` is built only when
        // a graph is being recorded.
        let mut args = [0u64; GDN_MAX_ARGS];
        let n = gdn_pack_args(kernel_name, &mut args, ptrs, &[])?;
        let mut raw = [std::ptr::null_mut::<std::ffi::c_void>(); GDN_MAX_ARGS];
        for (r, a) in raw.iter_mut().zip(args.iter_mut()) {
            *r = std::ptr::from_mut(a).cast::<std::ffi::c_void>();
        }

        let module = self
            .modules
            .get_mut(cache_key)
            .expect("module just inserted");
        // SAFETY: every pointer was checked non-null above, each buffer is
        // allocated with the length the kernel's immediates encode, and the
        // argument order matches the kernel's `.param` declarations.
        unsafe {
            self.stream
                .launch_kernel(module, kernel_name, &config, &mut raw[..n])?;
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
                arg_data: args[..n].to_vec(),
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
            validate_kernel_arg(p, kernel_name, i)?;
            // #4258: any pointer argument may be an output (several GDN kernels
            // write in place), so any of them can stale the Q8 activation.
            self.q8_activation_written(p);
        }
        // #4215: argument slots live on the stack; the `Vec` is built only when
        // a graph is being recorded.
        let mut args = [0u64; GDN_MAX_ARGS];
        let n = gdn_pack_args(kernel_name, &mut args, ptrs, scalars)?;
        let mut raw = [std::ptr::null_mut::<std::ffi::c_void>(); GDN_MAX_ARGS];
        for (r, a) in raw.iter_mut().zip(args.iter_mut()) {
            *r = std::ptr::from_mut(a).cast::<std::ffi::c_void>();
        }

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
                .launch_kernel(module, kernel_name, &config, &mut raw[..n])?;
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
                arg_data: args[..n].to_vec(),
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
        let cache_key = module_key!(self, "gdn_split_interleaved_{}_{}", num_heads, head_dim);
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
        let cache_key = module_key!(
            self,
            "gdn_partial_neox_rope_{}_{}_{}_b{:08x}",
            num_heads,
            head_dim,
            n_rot,
            theta_scale.to_bits()
        );
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
        // The split kernel scores a position with one warp in 16-byte lanes.
        if head_dim % 128 == 0 && !decode_attention_unsplit() {
            return self.gdn_decode_attention_split_into(
                q,
                k_cache,
                v_cache,
                output,
                num_heads,
                num_kv_heads,
                head_dim,
                seq_len,
            );
        }
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
        let cache_key = module_key!(
            self,
            "gdn_decode_attention_{}_{}_{}",
            num_heads,
            num_kv_heads,
            head_dim
        );
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

    /// aprender#4233: [`Self::gdn_partial_neox_rope_into`] with the position read
    /// from the device `u32` at `pos`, so a captured decode graph replays at
    /// whatever position the host last wrote there. Bit-identical to the direct
    /// kernel (`aprender-gpu` device test).
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    pub fn gdn_partial_neox_rope_indirect_into(
        &mut self,
        x: &GpuBuffer<f32>,
        pos: &GpuBuffer<u32>,
        num_heads: u32,
        head_dim: u32,
        n_rot: u32,
        theta_scale: f32,
    ) -> Result<(), GpuError> {
        let kernel =
            trueno_gpu::kernels::gdn::PartialNeoxRopeKernel::new(num_heads, head_dim, n_rot)
                .indirect();
        let kernel_type = KernelType::GdnPartialNeoxRopeIndirect {
            num_heads,
            head_dim,
            n_rot,
        };
        let cache_key = module_key!(
            self,
            "gdn_partial_neox_rope_indirect_{}_{}_{}",
            num_heads,
            head_dim,
            n_rot
        );
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch_mixed(
            &cache_key,
            kernel_name,
            config,
            &[x.as_ptr(), pos.as_ptr()],
            &[u64::from(theta_scale.to_bits())],
        )
    }

    /// #5279: [`Self::gdn_partial_neox_rope_indirect_into`] over `m` decode
    /// slots in one launch: row `r` of `x` (`num_heads * head_dim` wide) is
    /// rotated at the device position `pos[r]`, bitwise the single kernel on
    /// that row at that position.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_partial_neox_rope_slots_into(
        &mut self,
        x: &GpuBuffer<f32>,
        pos: &GpuBuffer<u32>,
        num_heads: u32,
        head_dim: u32,
        n_rot: u32,
        theta_scale: f32,
        m: u32,
    ) -> Result<(), GpuError> {
        let kernel =
            trueno_gpu::kernels::gdn::PartialNeoxRopeKernel::new(num_heads, head_dim, n_rot)
                .slots();
        let kernel_type = KernelType::GdnPartialNeoxRopeSlots {
            num_heads,
            head_dim,
            n_rot,
        };
        let cache_key = module_key!(
            self,
            "gdn_partial_neox_rope_slots_{}_{}_{}",
            num_heads,
            head_dim,
            n_rot
        );
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, m, bx, 1);
        self.gdn_launch_mixed(
            &cache_key,
            kernel_name,
            config,
            &[x.as_ptr(), pos.as_ptr()],
            &[u64::from(theta_scale.to_bits())],
        )
    }

    /// aprender#4233: [`Self::gdn_decode_attention_into`] attending over
    /// `*pos + 1` positions, read on the device.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_decode_attention_indirect_into(
        &mut self,
        q: &GpuBuffer<f32>,
        k_cache: &GpuBuffer<f32>,
        v_cache: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        pos: &GpuBuffer<u32>,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        let kernel = trueno_gpu::kernels::gdn::DecodeAttention256Kernel::new(
            num_heads,
            num_kv_heads,
            head_dim,
        )
        .indirect();
        let kernel_type = KernelType::GdnDecodeAttentionIndirect {
            num_heads,
            num_kv_heads,
            head_dim,
        };
        let cache_key = module_key!(
            self,
            "gdn_decode_attention_indirect_{}_{}_{}",
            num_heads,
            num_kv_heads,
            head_dim
        );
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
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                output.as_ptr(),
                pos.as_ptr(),
            ],
        )
    }

    /// aprender#4233: append one KV row, `cache[*pos * row ..][..row] = src`,
    /// with the position read on the device — the graph-safe replacement for
    /// writing through a host-computed row view.
    ///
    /// # Errors
    /// PTX compilation or kernel launch failure, or a null device pointer.
    pub fn gdn_kv_row_scatter_indirect_into(
        &mut self,
        src: &GpuBuffer<f32>,
        cache: &GpuBuffer<f32>,
        pos: &GpuBuffer<u32>,
        row: u32,
    ) -> Result<(), GpuError> {
        let kernel = trueno_gpu::kernels::gdn::KvRowScatterIndirectKernel::new(row);
        let kernel_type = KernelType::GdnKvRowScatterIndirect { row };
        let cache_key = module_key!(self, "gdn_kv_row_scatter_indirect_{}", row);
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, 1, bx, 1);
        self.gdn_launch(
            &cache_key,
            kernel_name,
            config,
            &[src.as_ptr(), cache.as_ptr(), pos.as_ptr()],
        )
    }

    /// #5279: [`Self::gpu_argmax`] for `m` rows of `vocab` logits in one pair of
    /// launches, row `r`'s token id written to `out[r]` — and nothing synced or
    /// downloaded, so the caller downloads all `m` ids at once. `vals` and
    /// `idxs` hold `m * argmax_blocks(vocab)` block results.
    pub fn argmax_rows_into(
        &mut self,
        logits: &GpuBuffer<f32>,
        (vals, idxs): (&GpuBuffer<f32>, &GpuBuffer<u32>),
        out: &GpuBuffer<u32>,
        vocab: u32,
        m: u32,
    ) -> Result<(), GpuError> {
        let blocks = Self::argmax_blocks(vocab);
        let first = KernelType::ArgMaxRows { length: vocab };
        let first_key = module_key!(self, "argmax_rows_{}", vocab);
        let first_name = self.gdn_prepare(&first, &first_key)?;
        self.gdn_launch_mixed(
            &first_key,
            first_name,
            LaunchConfig::grid_2d(blocks, m, 256, 1),
            &[logits.as_ptr(), vals.as_ptr(), idxs.as_ptr()],
            &[u64::from(vocab)],
        )?;
        let last = KernelType::ArgMaxFinalRows { num_blocks: blocks };
        let last_key = module_key!(self, "argmax_final_rows_{}", blocks);
        let last_name = self.gdn_prepare(&last, &last_key)?;
        self.gdn_launch_mixed(
            &last_key,
            last_name,
            LaunchConfig::grid_2d(1, m, 256, 1),
            &[vals.as_ptr(), idxs.as_ptr(), out.as_ptr()],
            &[u64::from(blocks)],
        )
    }

    /// The first argmax pass's block count for `vocab` logits.
    #[must_use]
    pub fn argmax_blocks(vocab: u32) -> u32 {
        trueno_gpu::kernels::ArgMaxKernel::new(vocab).num_blocks()
    }

    /// #5279: [`Self::gdn_kv_row_scatter_indirect_into`] for `m` slots in one
    /// launch: row `r` of `src` goes into the cache at `table[r]` (a device array
    /// of `m` u64 pointers), at position `pos[r]`.
    pub fn gdn_kv_row_scatter_slots_into(
        &mut self,
        src: &GpuBuffer<f32>,
        table: u64,
        pos: &GpuBuffer<u32>,
        row: u32,
        m: u32,
    ) -> Result<(), GpuError> {
        let kernel = trueno_gpu::kernels::gdn::KvRowScatterIndirectKernel::new(row).slots();
        let kernel_type = KernelType::GdnKvRowScatterSlots { row };
        let cache_key = module_key!(self, "gdn_kv_row_scatter_slots_{}", row);
        let kernel_name = self.gdn_prepare(&kernel_type, &cache_key)?;
        let (gx, _, _) = kernel.grid();
        let (bx, _, _) = kernel.block();
        let config = LaunchConfig::grid_2d(gx, m, bx, 1);
        self.gdn_launch(
            &cache_key,
            kernel_name,
            config,
            &[src.as_ptr(), table, pos.as_ptr()],
        )
    }

    /// aprender#4273: [`Self::gdn_decode_attention_into`] split over the sequence —
    /// grid `(num_heads, ceil(seq_len / split_len))` writing per-split partials, then
    /// a per-head reduce. The unsplit kernel runs 16 blocks on Qwen3.5-4B whatever
    /// the context, so its time grew linearly on 16 of 128 SMs (measured in #4273).
    ///
    /// The partials live in `decode_attn_partials`, grown when a longer context
    /// needs more splits. The grid depends on `seq_len`, so this launch must not be
    /// replayed from a recorded graph at another position — neither can the unsplit
    /// one, whose `seq_len` scalar is baked into its recorded arguments.
    ///
    /// # Errors
    /// PTX compilation, allocation or kernel launch failure, or a null pointer.
    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_attention_split_into(
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
        let split = trueno_gpu::kernels::gdn::DecodeAttentionSplitKernel::new(
            num_heads,
            num_kv_heads,
            head_dim,
        );
        let split_len = split.split_len;
        let (acc_floats, ml_floats) = split.partial_floats(seq_len);
        let need = acc_floats + ml_floats;
        if self.decode_attn_partials.as_ref().map_or(0, GpuBuffer::len) < need {
            // Grow in steps of 64 splits' worth so a decode reallocates rarely; the
            // old buffer may still be read by an in-flight launch, so drain first.
            let rows_step = (num_heads * 64) as usize * (head_dim as usize + 2);
            let grown = need.div_ceil(rows_step) * rows_step;
            self.stream.synchronize()?;
            self.decode_attn_partials = Some(GpuBuffer::new(&self.context, grown)?);
        }
        let partials = self
            .decode_attn_partials
            .as_ref()
            .expect("allocated just above")
            .as_ptr();
        let part_acc = partials;
        let part_ml = partials + (acc_floats * 4) as u64;

        let split_type = KernelType::GdnDecodeAttentionSplit {
            num_heads,
            num_kv_heads,
            head_dim,
            split_len,
        };
        let split_key = module_key!(
            self,
            "gdn_decode_attention_split_{}_{}_{}_{}",
            num_heads,
            num_kv_heads,
            head_dim,
            split_len
        );
        let split_name = self.gdn_prepare(&split_type, &split_key)?;
        let (gx, gy, _) = split.grid(seq_len);
        let (bx, _, _) = split.block();
        self.gdn_launch_mixed(
            &split_key,
            split_name,
            LaunchConfig::grid_2d(gx, gy, bx, 1),
            &[
                q.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                part_acc,
                part_ml,
            ],
            &[u64::from(seq_len)],
        )?;

        let reduce = split.reduce();
        let reduce_type = KernelType::GdnDecodeAttentionReduce {
            num_heads,
            head_dim,
            split_len,
        };
        let reduce_key = module_key!(
            self,
            "gdn_decode_attention_reduce_{}_{}_{}",
            num_heads,
            head_dim,
            split_len
        );
        let reduce_name = self.gdn_prepare(&reduce_type, &reduce_key)?;
        let (rx, _, _) = reduce.grid();
        let (rbx, _, _) = reduce.block();
        self.gdn_launch_mixed(
            &reduce_key,
            reduce_name,
            LaunchConfig::grid_2d(rx, 1, rbx, 1),
            &[part_acc, part_ml, output.as_ptr()],
            &[u64::from(seq_len)],
        )
    }

    /// #5279: whether [`Self::gdn_decode_attention_slots_into`] can stand in
    /// for per-slot [`Self::gdn_decode_attention_into`] calls: the single call
    /// takes the split pair for this `head_dim`.
    #[must_use]
    pub fn gdn_decode_attention_has_slots(head_dim: u32) -> bool {
        head_dim % 128 == 0 && !decode_attention_unsplit()
    }

    /// #5279: the partials stride and buffer [`Self::gdn_decode_attention_slots_into`]
    /// launches with at `max_pos` for `m` slots, grown first if they would not
    /// fit: `(max_splits, part_acc)`.
    ///
    /// `max_splits` is only the stride between slots' partial rows (each slot
    /// derives its own split count from its position on the device, and a split
    /// past it exits), so it is rounded up to a power of two: a step keeps its
    /// launch arguments, and a captured graph stays valid, until the longest
    /// slot doubles.
    pub(crate) fn reserve_attention_slots(
        &mut self,
        heads: (u32, u32, u32),
        max_pos: u32,
        m: u32,
    ) -> Result<(u32, u64), GpuError> {
        let (num_heads, num_kv_heads, head_dim) = heads;
        let split = trueno_gpu::kernels::gdn::DecodeAttentionSplitKernel::new(
            num_heads,
            num_kv_heads,
            head_dim,
        )
        .slots();
        let max_splits = split
            .n_splits(max_pos.saturating_add(1))
            .next_power_of_two();
        let (slot_acc, slot_ml) = split.partial_floats(max_splits * split.split_len);
        let need = (slot_acc + slot_ml) * m as usize;
        if self.decode_attn_partials.as_ref().map_or(0, GpuBuffer::len) < need {
            // As in the single pair: grow in steps, drain the in-flight reads first.
            let rows_step = (num_heads * 64) as usize * (head_dim as usize + 2);
            let grown = need.div_ceil(rows_step) * rows_step;
            self.stream.synchronize()?;
            self.decode_attn_partials = Some(GpuBuffer::new(&self.context, grown)?);
        }
        let part_acc = self
            .decode_attn_partials
            .as_ref()
            .expect("allocated just above")
            .as_ptr();
        Ok((max_splits, part_acc))
    }

    /// #5279: decode attention for `m` slots in one split and one reduce
    /// launch. `k_table` and `v_table` are device addresses of `[m]` cache
    /// pointers, `pos` the slots' positions, `max_pos` the largest of them;
    /// row `r` of `output` is bitwise [`Self::gdn_decode_attention_into`] on
    /// row `r` of `q` against slot `r`'s caches at `pos[r] + 1` positions.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_decode_attention_slots_into(
        &mut self,
        q: &GpuBuffer<f32>,
        k_table: u64,
        v_table: u64,
        output: &GpuBuffer<f32>,
        pos: &GpuBuffer<u32>,
        max_pos: u32,
        m: u32,
        heads: (u32, u32, u32),
    ) -> Result<(), GpuError> {
        let (num_heads, num_kv_heads, head_dim) = heads;
        let split = trueno_gpu::kernels::gdn::DecodeAttentionSplitKernel::new(
            num_heads,
            num_kv_heads,
            head_dim,
        )
        .slots();
        let split_len = split.split_len;
        let (max_splits, part_acc) = self.reserve_attention_slots(heads, max_pos, m)?;
        let (slot_acc, _) = split.partial_floats(max_splits * split_len);
        let part_ml = part_acc + (slot_acc * m as usize * 4) as u64;

        let split_type = KernelType::GdnDecodeAttentionSplitSlots {
            num_heads,
            num_kv_heads,
            head_dim,
            split_len,
        };
        let split_key = module_key!(
            self,
            "gdn_decode_attention_split_slots_{}_{}_{}_{}",
            num_heads,
            num_kv_heads,
            head_dim,
            split_len
        );
        let split_name = self.gdn_prepare(&split_type, &split_key)?;
        let (bx, _, _) = split.block();
        let config = LaunchConfig {
            grid: (num_heads, max_splits, m),
            block: (bx, 1, 1),
            shared_mem: 0,
        };
        let tables = [
            q.as_ptr(),
            k_table,
            v_table,
            part_acc,
            part_ml,
            pos.as_ptr(),
        ];
        self.gdn_launch_mixed(
            &split_key,
            split_name,
            config,
            &tables,
            &[u64::from(max_splits)],
        )?;

        let reduce = split.reduce();
        let reduce_type = KernelType::GdnDecodeAttentionReduceSlots {
            num_heads,
            head_dim,
            split_len,
        };
        let reduce_key = module_key!(
            self,
            "gdn_decode_attention_reduce_slots_{}_{}_{}",
            num_heads,
            head_dim,
            split_len
        );
        let reduce_name = self.gdn_prepare(&reduce_type, &reduce_key)?;
        let (rbx, _, _) = reduce.block();
        self.gdn_launch_mixed(
            &reduce_key,
            reduce_name,
            LaunchConfig::grid_2d(num_heads, m, rbx, 1),
            &[part_acc, part_ml, output.as_ptr(), pos.as_ptr()],
            &[u64::from(max_splits)],
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
        let cache_key = module_key!(self, "gdn_causal_conv1d_silu_{}_{}", channels, kernel_size);
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
        // `{epsilon:e}` is not an integer: key it by its bits, format it once.
        let cache_key = self.module_key(
            (
                "gdn_per_head_l2_norm_{}_{}_{:e}",
                key_dims(&[
                    u64::from(head_dim),
                    u64::from(num_heads),
                    u64::from(epsilon.to_bits()),
                ]),
            ),
            || format!("gdn_per_head_l2_norm_{head_dim}_{num_heads}_{epsilon:e}"),
        );
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
        let cache_key = module_key!(self, "gdn_gates_{}", num_heads);
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
    /// (`delta_rule_recurrence_gqa`).
    ///
    /// `q` and `k` are `[num_k_heads * head_k_dim]`, `v` and `output` are
    /// `[num_v_heads * head_v_dim]`, `beta` and `gate` are per VALUE head, and
    /// `state` is `[num_v_heads * head_v_dim * head_k_dim]`, updated in place.
    /// Value head `h` reads key/query head `h % num_k_heads` — the tiled order
    /// the GGUF conversion writes (PMAT-3477, #3346/#3510).
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
        num_k_heads: u32,
        head_k_dim: u32,
        num_v_heads: u32,
        head_v_dim: u32,
    ) -> Result<(), GpuError> {
        let kernel = trueno_gpu::kernels::gdn::DeltaRuleRecurrenceKernel::new(
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
        );
        let kernel_type = KernelType::GdnDeltaRule {
            num_v_heads,
            head_v_dim,
            num_k_heads,
            head_k_dim,
        };
        let cache_key = module_key!(
            self,
            "gdn_delta_rule_{}_{}_{}_{}",
            num_v_heads,
            head_v_dim,
            num_k_heads,
            head_k_dim
        );
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
        // `{epsilon:e}` is not an integer: key it by its bits, format it once.
        let cache_key = self.module_key(
            (
                "gdn_gated_rmsnorm_{}_{}_{:e}",
                key_dims(&[
                    u64::from(head_dim),
                    u64::from(num_heads),
                    u64::from(epsilon.to_bits()),
                ]),
            ),
            || format!("gdn_gated_rmsnorm_{head_dim}_{num_heads}_{epsilon:e}"),
        );
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
        let cache_key = module_key!(self, "gdn_sigmoid_gate_{}", n);
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

/// aprender#4273: `APR_QWEN35_DECODE_ATTENTION=unsplit` selects the pre-#4273
/// single-pass-per-head kernel, for A/B receipts (nsys, token identity). Read once.
fn decode_attention_unsplit() -> bool {
    static UNSPLIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *UNSPLIT
        .get_or_init(|| std::env::var("APR_QWEN35_DECODE_ATTENTION").is_ok_and(|v| v == "unsplit"))
}
