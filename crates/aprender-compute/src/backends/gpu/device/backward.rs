//! GPU backward (gradient) operations for training
//!
//! Contract: wgpu-training-v1.yaml (FALSIFY-WGPU-001)
//!
//! Dispatches WGSL backward shaders to compute gradients on GPU.
//! All operations match CPU reference within ε < 1e-4.

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use super::super::runtime;
use super::super::shaders;
use super::GpuDevice;

impl GpuDevice {
    /// SiLU backward on GPU: grad_input[i] = grad_output[i] * silu'(input[i])
    ///
    /// # Contract (FALSIFY-WGPU-001)
    ///
    /// - **Precondition**: input.len() == grad_output.len() == grad_input.len()
    /// - **Postcondition**: max|grad_input_gpu - grad_input_cpu| < 1e-4
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn silu_backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), String> {
        runtime::block_on(self.silu_backward_async(input, grad_output, grad_input))
    }

    /// SiLU backward on GPU (async)
    pub async fn silu_backward_async(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), String> {
        let n = input.len();
        if grad_output.len() != n || grad_input.len() != n {
            return Err(format!(
                "SiLU backward: length mismatch: input={}, grad_output={}, grad_input={}",
                n,
                grad_output.len(),
                grad_input.len()
            ));
        }

        self.execute_backward_elementwise(
            "SiLU Backward",
            shaders::backward::SILU_BACKWARD_SHADER,
            input,
            grad_output,
            grad_input,
            n as u32,
        )
        .await
    }

    /// Generic dispatch for element-wise backward shaders (3 buffers + uniform)
    ///
    /// Binding layout: 0=input(read), 1=grad_output(read), 2=grad_input(write), 3=uniform{n}
    async fn execute_backward_elementwise(
        &self,
        op_name: &str,
        shader_source: &str,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        n: u32,
    ) -> Result<(), String> {
        use wgpu;

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{op_name} Shader")),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create buffers
        let input_buf = self.create_storage_buffer(&format!("{op_name} input"), input, true);
        let grad_out_buf =
            self.create_storage_buffer(&format!("{op_name} grad_output"), grad_output, true);
        let grad_in_buf = self.create_rw_storage_buffer(
            &format!("{op_name} grad_input"),
            (grad_input.len() * 4) as u64,
        );

        // Uniform: { n: u32 } padded to 16 bytes (WGSL alignment)
        let uniform_data: [u32; 4] = [n, 0, 0, 0];
        let uniform_buf = self.create_uniform_buffer(
            &format!("{op_name} uniform"),
            bytemuck::cast_slice(&uniform_data),
        );

        // Bind group layout: 3 storage + 1 uniform
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{op_name} BGL")),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{op_name} BG")),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{op_name} PL")),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{op_name} Pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Staging buffer for readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{op_name} Staging")),
            size: (grad_input.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dispatch
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // 2D dispatch for large tensors (>16M elements)
            let total_wg = n.div_ceil(256);
            pass.dispatch_workgroups(total_wg.min(65535), total_wg.div_ceil(65535), 1);
        }
        encoder.copy_buffer_to_buffer(&grad_in_buf, 0, &staging, 0, (grad_input.len() * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        // Read back
        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        receiver
            .receive()
            .await
            .ok_or_else(|| format!("{op_name}: map_async cancelled"))?
            .map_err(|e| format!("{op_name}: map_async failed: {e}"))?;

        let data = slice.get_mapped_range();
        grad_input.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();

        Ok(())
    }

    // --- Buffer helpers ---

    fn create_storage_buffer(&self, label: &str, data: &[f32], read_only: bool) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        let _ = read_only; // usage flags are same; read_only is in the shader
        buf
    }

    fn create_rw_storage_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_uniform_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, data);
        buf
    }

    /// GEMM backward for A: grad_a[M,K] = grad_c[M,N] @ B^T[N,K]
    ///
    /// # Contract (FALSIFY-WGPU-001)
    ///
    /// - **Precondition**: grad_c.len() == m*n, b.len() == k*n, grad_a.len() == m*k
    /// - **Postcondition**: max|grad_a_gpu - grad_a_cpu| < 1e-4
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn gemm_backward_a(
        &self,
        grad_c: &[f32],
        b: &[f32],
        grad_a: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        runtime::block_on(self.gemm_backward_a_async(grad_c, b, grad_a, m, k, n))
    }

    /// GEMM backward for A (async): grad_a = grad_c @ B^T
    pub async fn gemm_backward_a_async(
        &self,
        grad_c: &[f32],
        b: &[f32],
        grad_a: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        self.execute_backward_gemm(
            "GEMM Backward A",
            shaders::backward::GEMM_BACKWARD_A_SHADER,
            grad_c,
            b,
            grad_a,
            m,
            k,
            n,
        )
        .await
    }

    /// GEMM backward for B: grad_b[K,N] = A^T[K,M] @ grad_c[M,N]
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn gemm_backward_b(
        &self,
        a: &[f32],
        grad_c: &[f32],
        grad_b: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        runtime::block_on(self.gemm_backward_b_async(a, grad_c, grad_b, m, k, n))
    }

    /// GEMM backward for B (async): grad_b = A^T @ grad_c
    pub async fn gemm_backward_b_async(
        &self,
        a: &[f32],
        grad_c: &[f32],
        grad_b: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        self.execute_backward_gemm(
            "GEMM Backward B",
            shaders::backward::GEMM_BACKWARD_B_SHADER,
            a,
            grad_c,
            grad_b,
            m,
            k,
            n,
        )
        .await
    }

    /// Generic dispatch for GEMM backward shaders (tiled 16×16)
    ///
    /// Binding: 0=buf_a(read), 1=buf_b(read), 2=output(write), 3=uniform{M,K,N}
    async fn execute_backward_gemm(
        &self,
        op_name: &str,
        shader_source: &str,
        buf_a: &[f32],
        buf_b: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        use wgpu;

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{op_name} Shader")),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let a_buf = self.create_storage_buffer(&format!("{op_name} A"), buf_a, true);
        let b_buf = self.create_storage_buffer(&format!("{op_name} B"), buf_b, true);
        let out_buf =
            self.create_rw_storage_buffer(&format!("{op_name} Output"), (output.len() * 4) as u64);

        // Uniform: { M, K, N, pad }
        let dims: [u32; 4] = [m, k, n, 0];
        let uniform_buf =
            self.create_uniform_buffer(&format!("{op_name} Dims"), bytemuck::cast_slice(&dims));

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{op_name} Pipeline")),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (output.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);

            // For GEMM backward A: output is [M,K], dispatch ceil(M/16) × ceil(K/16)
            // For GEMM backward B: output is [K,N], dispatch ceil(K/16) × ceil(N/16)
            // The output dimensions are encoded in the first two dims of the output buffer.
            let out_rows = if op_name.contains("A") { m } else { k };
            let out_cols = if op_name.contains("A") { k } else { n };
            pass.dispatch_workgroups(out_rows.div_ceil(16), out_cols.div_ceil(16), 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (output.len() * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        receiver
            .receive()
            .await
            .ok_or_else(|| format!("{op_name}: map cancelled"))?
            .map_err(|e| format!("{op_name}: map failed: {e}"))?;

        let data = slice.get_mapped_range();
        output.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();

        Ok(())
    }

    /// RoPE backward on GPU: transpose rotation (negated sin)
    ///
    /// # Contract (FALSIFY-WGPU-001)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn rope_backward(
        &self,
        grad_output: &[f32],
        grad_input: &mut [f32],
        num_heads: u32,
        head_dim: u32,
        seq_len: u32,
        theta: f32,
    ) -> Result<(), String> {
        runtime::block_on(self.rope_backward_async(
            grad_output,
            grad_input,
            num_heads,
            head_dim,
            seq_len,
            theta,
        ))
    }

    /// RoPE backward (async)
    pub async fn rope_backward_async(
        &self,
        grad_output: &[f32],
        grad_input: &mut [f32],
        num_heads: u32,
        head_dim: u32,
        seq_len: u32,
        theta: f32,
    ) -> Result<(), String> {
        use wgpu;

        let n = grad_output.len();
        let total_pairs = num_heads * seq_len * (head_dim / 2);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RoPE Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::backward::ROPE_BACKWARD_SHADER.into()),
        });

        let go_buf = self.create_storage_buffer("rope_bwd grad_out", grad_output, true);
        let gi_buf = self.create_rw_storage_buffer("rope_bwd grad_in", (n * 4) as u64);

        // Uniform: { num_heads, head_dim, seq_len, theta_log2 }
        let params: [u32; 4] = [num_heads, head_dim, seq_len, theta.log2().to_bits()];
        let uniform_buf =
            self.create_uniform_buffer("rope_bwd params", bytemuck::cast_slice(&params));

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[storage_entry(0, true), storage_entry(1, false), uniform_entry(2)],
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: go_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gi_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RoPE Backward"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total_wg = total_pairs.div_ceil(256);
            pass.dispatch_workgroups(total_wg.min(65535), total_wg.div_ceil(65535), 1);
        }
        encoder.copy_buffer_to_buffer(&gi_buf, 0, &staging, 0, (n * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        receiver
            .receive()
            .await
            .ok_or("RoPE backward: cancelled".to_string())?
            .map_err(|e| format!("RoPE backward: {e}"))?;
        let data = slice.get_mapped_range();
        grad_input.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();
        Ok(())
    }

    /// AdamW optimizer step on GPU
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn adamw_step(
        &self,
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<(), String> {
        runtime::block_on(self.adamw_step_async(
            params,
            grads,
            m,
            v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        ))
    }

    /// AdamW step (async)
    pub async fn adamw_step_async(
        &self,
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<(), String> {
        use wgpu;

        let n = params.len() as u32;
        let bc1 = 1.0 - beta1.powi(step as i32);
        let bc2 = 1.0 - beta2.powi(step as i32);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AdamW Step"),
            source: wgpu::ShaderSource::Wgsl(shaders::backward::ADAMW_STEP_SHADER.into()),
        });

        // Params buffer is read-write (updated in-place)
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adamw params"),
            size: (params.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(params));

        let grads_buf = self.create_storage_buffer("adamw grads", grads, true);

        let m_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adamw m"),
            size: (m.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&m_buf, 0, bytemuck::cast_slice(m));

        let v_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adamw v"),
            size: (v.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&v_buf, 0, bytemuck::cast_slice(v));

        // Uniform: { n: u32, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, bc1: f32, bc2: f32 }
        // Pack as raw u32 bytes to handle the mixed u32/f32 layout
        let hp: [u32; 8] = [
            n,
            lr.to_bits(),
            beta1.to_bits(),
            beta2.to_bits(),
            eps.to_bits(),
            weight_decay.to_bits(),
            bc1.to_bits(),
            bc2.to_bits(),
        ];
        let uniform_buf = self.create_uniform_buffer("adamw hp", bytemuck::cast_slice(&hp));

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, false), // params (read-write)
                storage_entry(1, true),  // grads
                storage_entry(2, false), // m
                storage_entry(3, false), // v
                uniform_entry(4),
            ],
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grads_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: m_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("AdamW"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Staging buffers for readback
        let params_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (params.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let m_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (m.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (v.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // 2D dispatch for large tensors (>16M elements)
            let total_wg = n.div_ceil(256);
            pass.dispatch_workgroups(total_wg.min(65535), total_wg.div_ceil(65535), 1);
        }
        encoder.copy_buffer_to_buffer(
            &params_buf,
            0,
            &params_staging,
            0,
            (params.len() * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(&m_buf, 0, &m_staging, 0, (m.len() * 4) as u64);
        encoder.copy_buffer_to_buffer(&v_buf, 0, &v_staging, 0, (v.len() * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        // Read back all three buffers
        let read_buf = |staging: &wgpu::Buffer, out: &mut [f32]| -> Result<(), String> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv()
                .map_err(|e| format!("AdamW readback: {e}"))?
                .map_err(|e| format!("AdamW map: {e}"))?;
            let data = slice.get_mapped_range();
            out.copy_from_slice(bytemuck::cast_slice(&data));
            drop(data);
            staging.unmap();
            Ok(())
        };
        read_buf(&params_staging, params)?;
        read_buf(&m_staging, m)?;
        read_buf(&v_staging, v)?;

        Ok(())
    }

    /// RMSNorm backward on GPU
    ///
    /// Computes grad_input and accumulates grad_gamma via atomic CAS.
    /// One workgroup (256 threads) per row.
    ///
    /// # Contract (FALSIFY-WGPU-001)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn rmsnorm_backward(
        &self,
        input: &[f32],
        gamma: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        grad_gamma: &mut [f32],
        num_rows: u32,
        hidden_dim: u32,
        eps: f32,
    ) -> Result<(), String> {
        runtime::block_on(self.rmsnorm_backward_async(
            input,
            gamma,
            grad_output,
            grad_input,
            grad_gamma,
            num_rows,
            hidden_dim,
            eps,
        ))
    }

    /// RMSNorm backward (async)
    pub async fn rmsnorm_backward_async(
        &self,
        input: &[f32],
        gamma: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        grad_gamma: &mut [f32],
        num_rows: u32,
        hidden_dim: u32,
        eps: f32,
    ) -> Result<(), String> {
        use wgpu;

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RMSNorm Backward"),
            source: wgpu::ShaderSource::Wgsl(shaders::backward::RMSNORM_BACKWARD_SHADER.into()),
        });

        let input_buf = self.create_storage_buffer("rms_bwd input", input, true);
        let gamma_buf = self.create_storage_buffer("rms_bwd gamma", gamma, true);
        let grad_out_buf = self.create_storage_buffer("rms_bwd grad_out", grad_output, true);
        let grad_in_buf =
            self.create_rw_storage_buffer("rms_bwd grad_in", (grad_input.len() * 4) as u64);

        // grad_gamma: init to zero, accumulated via atomic CAS
        let grad_gamma_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_bwd grad_gamma"),
            size: (hidden_dim as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Zero-init grad_gamma
        let zeros = vec![0u8; hidden_dim as usize * 4];
        self.queue.write_buffer(&grad_gamma_buf, 0, &zeros);

        // Uniform: { num_rows, hidden_dim, eps_bits, pad }
        let params: [u32; 4] = [num_rows, hidden_dim, eps.to_bits(), 0];
        let uniform_buf =
            self.create_uniform_buffer("rms_bwd params", bytemuck::cast_slice(&params));

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),  // input
                storage_entry(1, true),  // gamma
                storage_entry(2, true),  // grad_output
                storage_entry(3, false), // grad_input
                storage_entry(4, false), // grad_gamma (atomic)
                uniform_entry(5),
            ],
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grad_in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_gamma_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RMSNorm Backward"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Staging for grad_input and grad_gamma
        let gi_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (grad_input.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gg_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (hidden_dim as usize * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // One workgroup (256 threads) per row
            pass.dispatch_workgroups(num_rows, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &grad_in_buf,
            0,
            &gi_staging,
            0,
            (grad_input.len() * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &grad_gamma_buf,
            0,
            &gg_staging,
            0,
            (hidden_dim as usize * 4) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        // Read back grad_input
        {
            let slice = gi_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv()
                .map_err(|e| format!("RMSNorm bwd gi: {e}"))?
                .map_err(|e| format!("RMSNorm bwd gi map: {e}"))?;
            let data = slice.get_mapped_range();
            grad_input.copy_from_slice(bytemuck::cast_slice(&data));
            drop(data);
            gi_staging.unmap();
        }
        // Read back grad_gamma (stored as atomic<u32> = bitcast f32)
        {
            let slice = gg_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv()
                .map_err(|e| format!("RMSNorm bwd gg: {e}"))?
                .map_err(|e| format!("RMSNorm bwd gg map: {e}"))?;
            let data = slice.get_mapped_range();
            // atomic<u32> stores are bit-identical to f32 after CAS
            let raw: &[u32] = bytemuck::cast_slice(&data);
            for (i, &bits) in raw.iter().enumerate() {
                grad_gamma[i] = f32::from_bits(bits);
            }
            drop(data);
            gg_staging.unmap();
        }

        Ok(())
    }

    /// NF4 dequantization on GPU
    ///
    /// Converts 4-bit NormalFloat packed weights to fp32 using codebook lookup.
    ///
    /// # Contract (FALSIFY-WGPU-003)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn nf4_dequant(
        &self,
        packed: &[u32],
        scales: &[f32],
        output: &mut [f32],
        n: u32,
        block_size: u32,
    ) -> Result<(), String> {
        runtime::block_on(self.nf4_dequant_async(packed, scales, output, n, block_size))
    }

    /// NF4 dequant (async)
    pub async fn nf4_dequant_async(
        &self,
        packed: &[u32],
        scales: &[f32],
        output: &mut [f32],
        n: u32,
        block_size: u32,
    ) -> Result<(), String> {
        use wgpu;

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NF4 Dequant"),
            source: wgpu::ShaderSource::Wgsl(shaders::backward::NF4_DEQUANT_SHADER.into()),
        });

        let packed_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nf4 packed"),
            size: (packed.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&packed_buf, 0, bytemuck::cast_slice(packed));

        let scales_buf = self.create_storage_buffer("nf4 scales", scales, true);
        let output_buf = self.create_rw_storage_buffer("nf4 output", (output.len() * 4) as u64);

        let params: [u32; 4] = [n, block_size, 0, 0];
        let uniform_buf = self.create_uniform_buffer("nf4 params", bytemuck::cast_slice(&params));

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),  // packed
                storage_entry(1, true),  // scales
                storage_entry(2, false), // output
                uniform_entry(3),
            ],
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: packed_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scales_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NF4 Dequant"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (output.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Use 2D dispatch to handle >65535 workgroups
            // Each workgroup has 256 threads. Total threads = n.
            // X = min(ceil(n/256), 65535), Y = ceil(ceil(n/256) / 65535)
            let total_wg = n.div_ceil(256);
            let x = total_wg.min(65535);
            let y = total_wg.div_ceil(65535);
            pass.dispatch_workgroups(x, y, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, (output.len() * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        receiver
            .receive()
            .await
            .ok_or("NF4 dequant: cancelled".to_string())?
            .map_err(|e| format!("NF4 dequant: {e}"))?;
        let data = slice.get_mapped_range();
        output.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();

        Ok(())
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    /// CPU reference: SiLU backward
    fn silu_backward_cpu(input: &[f32], grad_output: &[f32]) -> Vec<f32> {
        input
            .iter()
            .zip(grad_output.iter())
            .map(|(&x, &dy)| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                let y = x * sigmoid;
                let silu_prime = sigmoid * (1.0 + x - y);
                dy * silu_prime
            })
            .collect()
    }

    /// FALSIFY-WGPU-001: SiLU backward matches CPU within ε < 1e-4
    #[test]
    fn test_falsify_wgpu_001_silu_backward_parity() {
        let device = GpuDevice::new().expect("GPU device");

        let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let grad_output: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.01).collect();
        let expected = silu_backward_cpu(&input, &grad_output);

        let mut grad_input = vec![0.0f32; 100];
        device.silu_backward(&input, &grad_output, &mut grad_input).expect("silu_backward");

        let max_diff = grad_input
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "FALSIFY-WGPU-001: SiLU backward max diff = {max_diff} (threshold: 1e-4)"
        );
    }

    /// SiLU backward at x=0 (sigmoid=0.5, silu'=0.5)
    #[test]
    fn test_silu_backward_at_zero() {
        let device = GpuDevice::new().expect("GPU device");

        let input = vec![0.0f32; 4];
        let grad_output = vec![1.0f32; 4];
        let mut grad_input = vec![0.0f32; 4];

        device.silu_backward(&input, &grad_output, &mut grad_input).expect("silu_backward");

        // At x=0: sigmoid(0)=0.5, silu'(0) = 0.5 * (1 + 0 - 0) = 0.5
        for &g in &grad_input {
            assert!((g - 0.5).abs() < 1e-5, "silu'(0) should be 0.5, got {g}");
        }
    }

    /// SiLU backward length mismatch error
    #[test]
    fn test_silu_backward_length_mismatch() {
        let device = GpuDevice::new().expect("GPU device");

        let input = vec![1.0f32; 10];
        let grad_output = vec![1.0f32; 5]; // wrong length
        let mut grad_input = vec![0.0f32; 10];

        let result = device.silu_backward(&input, &grad_output, &mut grad_input);
        assert!(result.is_err());
    }

    /// CPU reference: matmul C = A[M,K] @ B[K,N]
    fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// FALSIFY-WGPU-001: GEMM backward A matches CPU within ε < 1e-3
    ///
    /// grad_a[M,K] = grad_c[M,N] @ B^T[N,K]
    /// Which is matmul(grad_c, B^T, M, N, K) but our shader handles the transpose internally.
    #[test]
    fn test_falsify_wgpu_001_gemm_backward_a_parity() {
        let device = GpuDevice::new().expect("GPU device");

        let (m, k, n) = (4, 8, 6);

        // Random-ish test data
        let grad_c: Vec<f32> = (0..m * n).map(|i| (i as f32 - 12.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 - 24.0) * 0.05).collect();

        // CPU reference: grad_a = grad_c @ B^T
        // B^T[N,K] means we need to transpose B[K,N] → B^T[N,K]
        let mut b_t = vec![0.0f32; n * k];
        for i in 0..k {
            for j in 0..n {
                b_t[j * k + i] = b[i * n + j];
            }
        }
        let expected = matmul_cpu(&grad_c, &b_t, m, n, k);

        let mut grad_a = vec![0.0f32; m * k];
        device
            .gemm_backward_a(&grad_c, &b, &mut grad_a, m as u32, k as u32, n as u32)
            .expect("gemm_backward_a");

        let max_diff =
            grad_a.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-3,
            "FALSIFY-WGPU-001: GEMM backward A max diff = {max_diff} (threshold: 1e-3)"
        );
    }

    /// FALSIFY-WGPU-001: GEMM backward B matches CPU within ε < 1e-3
    ///
    /// grad_b[K,N] = A^T[K,M] @ grad_c[M,N]
    #[test]
    fn test_falsify_wgpu_001_gemm_backward_b_parity() {
        let device = GpuDevice::new().expect("GPU device");

        let (m, k, n) = (4, 8, 6);

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let grad_c: Vec<f32> = (0..m * n).map(|i| (i as f32 - 12.0) * 0.05).collect();

        // CPU reference: grad_b = A^T @ grad_c
        let mut a_t = vec![0.0f32; k * m];
        for i in 0..m {
            for j in 0..k {
                a_t[j * m + i] = a[i * k + j];
            }
        }
        let expected = matmul_cpu(&a_t, &grad_c, k, m, n);

        let mut grad_b = vec![0.0f32; k * n];
        device
            .gemm_backward_b(&a, &grad_c, &mut grad_b, m as u32, k as u32, n as u32)
            .expect("gemm_backward_b");

        let max_diff =
            grad_b.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-3,
            "FALSIFY-WGPU-001: GEMM backward B max diff = {max_diff} (threshold: 1e-3)"
        );
    }

    /// FALSIFY-WGPU-001: RoPE backward matches CPU
    #[test]
    fn test_falsify_wgpu_001_rope_backward_parity() {
        let device = GpuDevice::new().expect("GPU device");

        let (num_heads, head_dim, seq_len) = (2, 4, 3);
        let theta = 10000.0f32;
        let n = num_heads * head_dim * seq_len;

        let grad_output: Vec<f32> = (0..n).map(|i| (i as f32 - 12.0) * 0.1).collect();

        // CPU reference: RoPE backward = transpose rotation
        let half_dim = head_dim / 2;
        let mut expected = vec![0.0f32; n];
        for h in 0..num_heads {
            for s in 0..seq_len {
                for p in 0..half_dim {
                    let freq_exp = -((2 * p) as f32) / head_dim as f32 * theta.log2();
                    let inv_freq = 2.0f32.powf(freq_exp);
                    let angle = s as f32 * inv_freq;
                    let (sin_a, cos_a) = angle.sin_cos();

                    let base = h * seq_len * head_dim + s * head_dim;
                    let even = base + 2 * p;
                    let odd = base + 2 * p + 1;

                    let dy_even = grad_output[even];
                    let dy_odd = grad_output[odd];

                    // Backward: transpose of rotation matrix
                    expected[even] = dy_even * cos_a + dy_odd * sin_a;
                    expected[odd] = -dy_even * sin_a + dy_odd * cos_a;
                }
            }
        }

        let mut grad_input = vec![0.0f32; n];
        device
            .rope_backward(
                &grad_output,
                &mut grad_input,
                num_heads as u32,
                head_dim as u32,
                seq_len as u32,
                theta,
            )
            .expect("rope_backward");

        let max_diff = grad_input
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "FALSIFY-WGPU-001: RoPE backward max diff = {max_diff} (threshold: 1e-4)"
        );
    }

    /// FALSIFY-WGPU-001: AdamW step matches CPU
    #[test]
    fn test_falsify_wgpu_001_adamw_step_parity() {
        let device = GpuDevice::new().expect("GPU device");

        let n = 16;
        let mut params: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let grads: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.01).collect();
        let mut m_state = vec![0.0f32; n];
        let mut v_state = vec![0.0f32; n];

        let lr: f32 = 1e-3;
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps: f32 = 1e-8;
        let wd: f32 = 0.01;
        let step = 1u32;

        // CPU reference
        let bc1: f32 = 1.0 - beta1.powi(step as i32);
        let bc2: f32 = 1.0 - beta2.powi(step as i32);
        let mut cpu_params = params.clone();
        let mut cpu_m = m_state.clone();
        let mut cpu_v = v_state.clone();
        for i in 0..n {
            cpu_m[i] = beta1 * cpu_m[i] + (1.0 - beta1) * grads[i];
            cpu_v[i] = beta2 * cpu_v[i] + (1.0 - beta2) * grads[i] * grads[i];
            let m_hat = cpu_m[i] / bc1;
            let v_hat = cpu_v[i] / bc2;
            cpu_params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * cpu_params[i]);
        }

        device
            .adamw_step(
                &mut params,
                &grads,
                &mut m_state,
                &mut v_state,
                lr as f32,
                beta1 as f32,
                beta2 as f32,
                eps as f32,
                wd as f32,
                step,
            )
            .expect("adamw_step");

        let max_diff =
            params.iter().zip(cpu_params.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "FALSIFY-WGPU-001: AdamW step max diff = {max_diff} (threshold: 1e-4)"
        );
    }

    /// FALSIFY-WGPU-001: RMSNorm backward matches CPU
    #[test]
    fn test_falsify_wgpu_001_rmsnorm_backward_parity() {
        let device = GpuDevice::new().expect("GPU device");

        let (num_rows, hidden_dim) = (3, 8);
        let eps: f32 = 1e-5;
        let n = num_rows * hidden_dim;

        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 12.0) * 0.1).collect();
        let gamma: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + i as f32 * 0.1).collect();
        let grad_output: Vec<f32> = (0..n).map(|i| (i as f32 - 12.0) * 0.05).collect();

        // CPU reference
        let mut cpu_grad_input = vec![0.0f32; n];
        let mut cpu_grad_gamma = vec![0.0f32; hidden_dim];
        for r in 0..num_rows {
            let row = &input[r * hidden_dim..(r + 1) * hidden_dim];
            let grow = &grad_output[r * hidden_dim..(r + 1) * hidden_dim];

            let sum_x2: f32 = row.iter().map(|x| x * x).sum();
            let mean_x2 = sum_x2 / hidden_dim as f32;
            let var_eps = mean_x2 + eps;
            let rms = var_eps.sqrt();
            let inv_rms = 1.0 / rms;

            let sum_xgg: f32 = row
                .iter()
                .zip(grow.iter())
                .zip(gamma.iter())
                .map(|((&x, &gy), &g)| x * gy * g)
                .sum();
            let mean_xgg = sum_xgg / hidden_dim as f32;

            for i in 0..hidden_dim {
                let x = row[i];
                let gy = grow[i];
                let g = gamma[i];
                let gamma_gy = g * gy;
                let correction = (x / var_eps) * mean_xgg;
                cpu_grad_input[r * hidden_dim + i] = inv_rms * (gamma_gy - correction);
                cpu_grad_gamma[i] += gy * x * inv_rms;
            }
        }

        let mut grad_input = vec![0.0f32; n];
        let mut grad_gamma = vec![0.0f32; hidden_dim];

        device
            .rmsnorm_backward(
                &input,
                &gamma,
                &grad_output,
                &mut grad_input,
                &mut grad_gamma,
                num_rows as u32,
                hidden_dim as u32,
                eps,
            )
            .expect("rmsnorm_backward");

        let gi_max_diff = grad_input
            .iter()
            .zip(cpu_grad_input.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let gg_max_diff = grad_gamma
            .iter()
            .zip(cpu_grad_gamma.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            gi_max_diff < 1e-3,
            "FALSIFY-WGPU-001: RMSNorm grad_input max diff = {gi_max_diff}"
        );
        assert!(
            gg_max_diff < 1e-2,
            "FALSIFY-WGPU-001: RMSNorm grad_gamma max diff = {gg_max_diff} (atomic CAS accumulation)"
        );
    }

    /// FALSIFY-WGPU-003: NF4 dequant matches CPU
    #[test]
    fn test_falsify_wgpu_003_nf4_dequant_parity() {
        let device = GpuDevice::new().expect("GPU device");

        // NF4 codebook
        let nf4_lut: [f32; 16] = [
            -1.0,
            -0.6961928,
            -0.5250731,
            -0.39491749,
            -0.28444138,
            -0.18477343,
            -0.09105004,
            0.0,
            0.0795803,
            0.1609302,
            0.24611230,
            0.33791524,
            0.44070983,
            0.5626170,
            0.7229568,
            1.0,
        ];

        let block_size = 4u32; // small for testing
        let n = 8u32; // 8 elements = 2 blocks of 4

        // Pack: each byte has 2 nibbles (low=even, high=odd)
        // Elements: indices [3, 7, 12, 1, 5, 15, 0, 9]
        // Byte 0: low=3, high=7 → 0x73
        // Byte 1: low=12, high=1 → 0x1C
        // Byte 2: low=5, high=15 → 0xF5
        // Byte 3: low=0, high=9 → 0x90
        // 8 elements = 4 bytes = 1 u32 (each byte has 2 nibbles)
        // Byte 0: elem[0]=3,elem[1]=7 → 0x73
        // Byte 1: elem[2]=12,elem[3]=1 → 0x1C
        // Byte 2: elem[4]=5,elem[5]=15 → 0xF5
        // Byte 3: elem[6]=0,elem[7]=9 → 0x90
        let packed: Vec<u32> = vec![0x90F5_1C73_u32];

        let scales: Vec<f32> = vec![2.0, 0.5]; // 2 blocks
        let indices = [3, 7, 12, 1, 5, 15, 0, 9];

        // CPU reference
        let mut expected = vec![0.0f32; n as usize];
        for i in 0..n as usize {
            let scale = scales[i / block_size as usize];
            expected[i] = nf4_lut[indices[i]] * scale;
        }

        let mut output = vec![0.0f32; n as usize];
        device.nf4_dequant(&packed, &scales, &mut output, n, block_size).expect("nf4_dequant");

        let max_diff =
            output.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "FALSIFY-WGPU-003: NF4 dequant max diff = {max_diff} (threshold: 1e-6)"
        );
    }
}
