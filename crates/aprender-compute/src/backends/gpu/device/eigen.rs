//! GPU eigendecomposition operations
//!
//! Symmetric eigendecomposition using Jacobi algorithm with GPU-accelerated
//! Givens rotations, plus CPU fallback for small matrices.

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use super::super::runtime;
use super::super::shaders;
use super::GpuDevice;

impl GpuDevice {
    /// Execute symmetric eigendecomposition on GPU (sync, native only)
    ///
    /// Computes eigenvalues and eigenvectors using Jacobi algorithm with GPU-accelerated
    /// Givens rotations. Returns (eigenvalues, eigenvector_data) where eigenvector_data
    /// is in row-major format.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn symmetric_eigen(
        &self,
        matrix: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        runtime::block_on(async { self.symmetric_eigen_async(matrix, n).await })
    }

    /// Execute symmetric eigendecomposition on GPU (async, works on all platforms)
    ///
    /// Computes eigenvalues and eigenvectors using Jacobi algorithm with GPU-accelerated
    /// Givens rotations.
    pub async fn symmetric_eigen_async(
        &self,
        matrix: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if matrix.len() != n * n {
            return Err(format!(
                "Matrix size mismatch: expected {} elements for {}x{} matrix, got {}",
                n * n,
                n,
                n,
                matrix.len()
            ));
        }

        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // For small matrices, use CPU (GPU overhead not worth it)
        if n < 64 {
            return self.symmetric_eigen_cpu(matrix, n);
        }

        // Create shader module for Jacobi rotation
        let rotation_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Jacobi Rotation Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::JACOBI_ROTATION_SHADER.into()),
        });

        // Create buffers
        let matrix_size = (n * n * std::mem::size_of::<f32>()) as u64;

        let matrix_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix Buffer"),
            size: matrix_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let eigenvectors_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Eigenvectors Buffer"),
            size: matrix_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // JacobiParams uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct JacobiParams {
            n: u32,
            p: u32,
            q: u32,
            c: f32,
            s: f32,
            _padding: [u32; 3],
        }

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Jacobi Params"),
            size: std::mem::size_of::<JacobiParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize eigenvectors to identity matrix
        let mut eigenvectors = vec![0.0f32; n * n];
        for i in 0..n {
            eigenvectors[i * n + i] = 1.0;
        }

        // Write initial data
        self.queue.write_buffer(&matrix_buffer, 0, bytemuck::cast_slice(matrix));
        self.queue.write_buffer(&eigenvectors_buffer, 0, bytemuck::cast_slice(&eigenvectors));

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Jacobi Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Jacobi Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: matrix_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: eigenvectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Jacobi Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let rotation_pipeline =
            self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Jacobi Rotation Pipeline"),
                layout: Some(&pipeline_layout),
                module: &rotation_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Jacobi iteration
        let max_sweeps = 50;
        let tolerance = 1e-7 * (matrix.iter().map(|x| x * x).sum::<f32>().sqrt()).max(1.0);

        // Working copy of matrix for CPU-side pivot selection
        let mut a = matrix.to_vec();

        for _sweep in 0..max_sweeps {
            let mut converged = true;

            // Cyclic Jacobi: process all pairs (i, j) where i < j
            for i in 0..n {
                for j in (i + 1)..n {
                    let aij = a[i * n + j];

                    if aij.abs() < tolerance {
                        continue;
                    }

                    converged = false;

                    // Compute rotation parameters
                    let aii = a[i * n + i];
                    let ajj = a[j * n + j];

                    let tau = (ajj - aii) / (2.0 * aij);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };

                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;

                    // Update params and dispatch GPU
                    let params = JacobiParams {
                        n: n as u32,
                        p: i as u32,
                        q: j as u32,
                        c,
                        s,
                        _padding: [0; 3],
                    };

                    self.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

                    // Create command encoder and dispatch
                    let mut encoder =
                        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Jacobi Rotation Encoder"),
                        });

                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Jacobi Rotation Pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&rotation_pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
                    }

                    self.queue.submit(Some(encoder.finish()));

                    // Update local copy of diagonal and off-diagonal
                    a[i * n + i] = aii - t * aij;
                    a[j * n + j] = ajj + t * aij;
                    a[i * n + j] = 0.0;
                    a[j * n + i] = 0.0;

                    // Update off-diagonal elements in rows/columns i and j
                    for k in 0..n {
                        if k != i && k != j {
                            let aki = a[k * n + i];
                            let akj = a[k * n + j];
                            a[k * n + i] = c * aki - s * akj;
                            a[i * n + k] = a[k * n + i];
                            a[k * n + j] = s * aki + c * akj;
                            a[j * n + k] = a[k * n + j];
                        }
                    }
                }
            }

            if converged {
                break;
            }
        }

        // Read back results
        let staging_matrix = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Matrix"),
            size: matrix_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_eigenvectors = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Eigenvectors"),
            size: matrix_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(&matrix_buffer, 0, &staging_matrix, 0, matrix_size);
        encoder.copy_buffer_to_buffer(
            &eigenvectors_buffer,
            0,
            &staging_eigenvectors,
            0,
            matrix_size,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read eigenvectors
        let eigenvector_slice = staging_eigenvectors.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        eigenvector_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("oneshot channel receiver dropped");
        });

        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();

        receiver
            .receive()
            .await
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        let mut result_eigenvectors = vec![0.0f32; n * n];
        {
            let data = eigenvector_slice.get_mapped_range();
            let output_data: &[f32] = bytemuck::cast_slice(&data);
            result_eigenvectors.copy_from_slice(output_data);
        }
        staging_eigenvectors.unmap();

        // Extract eigenvalues from diagonal of working matrix
        let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();

        Ok((eigenvalues, result_eigenvectors))
    }

    /// CPU fallback for small matrices (GPU overhead not worthwhile)
    pub(super) fn symmetric_eigen_cpu(
        &self,
        matrix: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let max_sweeps = 50;
        let tolerance = 1e-7 * (matrix.iter().map(|x| x * x).sum::<f32>().sqrt()).max(1.0);

        let mut a = matrix.to_vec();
        let mut v = vec![0.0f32; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        for _sweep in 0..max_sweeps {
            let mut converged = true;

            for i in 0..n {
                for j in (i + 1)..n {
                    let aij = a[i * n + j];

                    if aij.abs() < tolerance {
                        continue;
                    }

                    converged = false;

                    let aii = a[i * n + i];
                    let ajj = a[j * n + j];

                    let tau = (ajj - aii) / (2.0 * aij);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };

                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;

                    // Update diagonal
                    a[i * n + i] = aii - t * aij;
                    a[j * n + j] = ajj + t * aij;
                    a[i * n + j] = 0.0;
                    a[j * n + i] = 0.0;

                    // Update off-diagonal
                    for k in 0..n {
                        if k != i && k != j {
                            let aki = a[k * n + i];
                            let akj = a[k * n + j];
                            a[k * n + i] = c * aki - s * akj;
                            a[i * n + k] = a[k * n + i];
                            a[k * n + j] = s * aki + c * akj;
                            a[j * n + k] = a[k * n + j];
                        }
                    }

                    // Update eigenvectors
                    for k in 0..n {
                        let vki = v[k * n + i];
                        let vkj = v[k * n + j];
                        v[k * n + i] = c * vki - s * vkj;
                        v[k * n + j] = s * vki + c * vkj;
                    }
                }
            }

            if converged {
                break;
            }
        }

        let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();
        Ok((eigenvalues, v))
    }
}
