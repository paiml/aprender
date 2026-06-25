//! GPU execution backend.
//!
//! Executes compute tasks on GPU using wgpu (cross-platform GPU abstraction).
//!
//! This executor uses WebGPU/wgpu for cross-platform GPU compute across:
//! - Vulkan (Linux/Windows/Android)
//! - Metal (macOS/iOS)
//! - DirectX 12 (Windows)
//! - WebGPU (browsers)
//!
//! # Architecture
//!
//! For v1.1, GPU execution works by:
//! 1. Detecting available GPU adapters (discrete, integrated, software)
//! 2. Creating compute pipeline for shader execution
//! 3. Uploading task data to GPU buffers
//! 4. Dispatching compute shaders
//! 5. Reading results back from GPU
//!
//! # Limitations (v1.1)
//!
//! - No JIT compilation of Rust → SPIR-V (requires rust-gpu, deferred to v1.2)
//! - Currently executes pre-compiled compute shaders only
//! - Binary tasks are NOT supported on GPU (use CPU executor)
//! - Designed for data-parallel workloads (matrix ops, image processing)
//!
//! # Example
//!
//! ```no_run
//! use repartir::executor::gpu::GpuExecutor;
//! use repartir::executor::Executor;
//!
//! #[tokio::main]
//! async fn main() -> repartir::error::Result<()> {
//!     let executor = GpuExecutor::new().await?;
//!     println!("GPU: {}", executor.device_name());
//!     println!("Compute units: {}", executor.capacity());
//!     Ok(())
//! }
//! ```

use crate::error::{RepartirError, Result};
use crate::executor::{BoxFuture, Executor};
use crate::task::{ExecutionResult, Task};
use std::sync::Arc;
use tracing::{debug, info, warn};
use wgpu::{util::DeviceExt, Device, Instance, Queue};

/// Platform-appropriate wgpu backend mask for adapter enumeration.
///
/// PMAT-927 (class follow-up to PMAT-925): `Instance::default()` uses
/// `wgpu::Backends::all()`, which includes [`wgpu::Backends::GL`]. On Linux hosts
/// that expose both Vulkan and GLES/EGL (notably the intel AMD-RADV cross-silicon
/// baseline box) this instantiates a GLES adapter whose `EglContext::make_current`
/// **panics inside `Drop`**, aborting the whole process with SIGABRT ("panic in a
/// destructor during cleanup"); standalone enumeration can also spin/hang on the
/// broken EGL path.
///
/// We return [`wgpu::Backends::PRIMARY`], which in wgpu 23 (this crate's pin) is
/// `VULKAN | METAL | DX12 | BROWSER_WEBGPU` and **excludes** `GL` (GL lives only
/// in `Backends::SECONDARY`). This keeps the real GPU on every platform — Vulkan
/// on Linux/AMD-RADV, Metal on Apple, DX12 on Windows — while guaranteeing the
/// broken GLES/EGL adapter is never created.
#[must_use]
pub const fn gpu_backends() -> wgpu::Backends {
    // PRIMARY = VULKAN | METAL | DX12 | BROWSER_WEBGPU (never GL/GLES).
    wgpu::Backends::PRIMARY
}

/// GPU executor using wgpu for cross-platform compute.
///
/// This executor provides GPU-accelerated task execution using WebGPU/wgpu.
/// It automatically selects the best available GPU adapter.
pub struct GpuExecutor {
    /// GPU device.
    #[allow(dead_code)] // Used in v1.2 for compute shader execution
    device: Arc<Device>,
    /// Command queue for GPU.
    #[allow(dead_code)] // Used in v1.2 for compute shader execution
    queue: Arc<Queue>,
    /// Device name (for logging/debugging).
    device_name: String,
    /// Number of compute units (estimate of parallelism).
    compute_units: usize,
}

impl GpuExecutor {
    /// Creates a new GPU executor with automatic adapter selection.
    ///
    /// Selects the best available GPU in this order:
    /// 1. Discrete GPU (dedicated graphics card)
    /// 2. Integrated GPU (CPU-integrated graphics)
    /// 3. Software rasterizer (CPU fallback)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No GPU adapter is available
    /// - Device request fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use repartir::executor::gpu::GpuExecutor;
    ///
    /// #[tokio::main]
    /// async fn main() -> repartir::error::Result<()> {
    ///     let executor = GpuExecutor::new().await?;
    ///     println!("Using GPU: {}", executor.device_name());
    ///     Ok(())
    /// }
    /// ```
    pub async fn new() -> Result<Self> {
        info!("Initializing GPU executor with wgpu");

        // Create wgpu instance.
        // PMAT-927: constrain to a non-GLES backend mask (see `gpu_backends`) so
        // the broken GLES/EGL adapter (SIGABRT-in-Drop on Linux/AMD-RADV) is never
        // registered. `Instance::default()` would use `Backends::all()`.
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: gpu_backends(),
            ..Default::default()
        });

        // Request adapter (GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| RepartirError::InvalidTask {
                reason: "No GPU adapter found".to_string(),
            })?;

        let adapter_info = adapter.get_info();
        let device_name = adapter_info.name.clone();

        info!("Selected GPU adapter: {device_name}");
        info!("Backend: {:?}", adapter_info.backend);
        info!("Device type: {:?}", adapter_info.device_type);

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Repartir GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| RepartirError::InvalidTask {
                reason: format!("Failed to request GPU device: {e}"),
            })?;

        // Estimate compute units (conservative estimate)
        // Real value requires vendor-specific queries
        let compute_units = Self::estimate_compute_units(&adapter_info);

        info!("GPU executor initialized");
        info!("Estimated compute units: {compute_units}");

        Ok(Self { device: Arc::new(device), queue: Arc::new(queue), device_name, compute_units })
    }

    /// Returns the GPU device name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Estimates compute units based on adapter info.
    ///
    /// This is a conservative estimate. Real compute unit count
    /// requires vendor-specific queries (CUDA cores, stream processors, etc).
    fn estimate_compute_units(info: &wgpu::AdapterInfo) -> usize {
        use wgpu::DeviceType;

        match info.device_type {
            DeviceType::DiscreteGpu => {
                // Discrete GPU: assume high-end (2048 cores)
                debug!("Discrete GPU detected, estimating 2048 compute units");
                2048
            }
            DeviceType::IntegratedGpu => {
                // Integrated GPU: assume mid-range (512 cores)
                debug!("Integrated GPU detected, estimating 512 compute units");
                512
            }
            DeviceType::VirtualGpu => {
                // Virtual GPU: assume cloud instance (1024 cores)
                debug!("Virtual GPU detected, estimating 1024 compute units");
                1024
            }
            DeviceType::Cpu => {
                // Software rasterizer: minimal parallelism
                debug!("CPU rasterizer detected, estimating 16 compute units");
                16
            }
            DeviceType::Other => {
                // Unknown device: conservative estimate
                warn!("Unknown GPU type, using conservative estimate");
                256
            }
        }
    }

    /// Executes a compute shader on the GPU.
    ///
    /// # Implementation
    ///
    /// This method:
    /// 1. Validates the task has shader code
    /// 2. Creates GPU buffers for inputs and outputs
    /// 3. Uploads input data to GPU
    /// 4. Creates compute pipeline with the shader
    /// 5. Dispatches the compute workgroups
    /// 6. Reads back output buffers from GPU
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Task has no shader code (binary execution not supported on GPU)
    /// - Shader module creation fails
    /// - Buffer creation or mapping fails
    /// - Pipeline creation fails
    async fn execute_compute(&self, task: Task) -> Result<ExecutionResult> {
        let task_id = task.id();
        let start_time = std::time::Instant::now();

        // Check if task has shader code
        let shader_code = task.shader_code().ok_or_else(|| RepartirError::InvalidTask {
            reason: format!(
                "GPU task {task_id} has no shader code. \
                     Binary execution is not supported on GPU. \
                     Use .shader_code() to provide WGSL shader source code."
            ),
        })?;

        debug!("Task {task_id}: Creating shader module from {} bytes", shader_code.len());

        // Convert shader code bytes to string (WGSL format)
        let shader_source =
            std::str::from_utf8(shader_code).map_err(|e| RepartirError::InvalidTask {
                reason: format!("Shader code is not valid UTF-8 WGSL: {e}"),
            })?;

        // Create shader module from WGSL source
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Task {task_id} Shader")),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
        });

        // Create input buffers and upload data
        let input_buffers = self.create_input_buffers(&task);

        // Create output buffers
        let output_buffers = self.create_output_buffers(&task);

        // Create bind group layout
        let bind_group_layout =
            self.create_bind_group_layout(input_buffers.len(), output_buffers.len());

        // Create bind group
        let bind_group =
            self.create_bind_group(&bind_group_layout, &input_buffers, &output_buffers);

        // Create compute pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Task {task_id} Pipeline Layout")),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline =
            self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Task {task_id} Pipeline")),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Task {task_id} Encoder")),
        });

        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Task {task_id} Compute Pass")),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Get workgroup size (default to 1,1,1 if not specified)
            let (x, y, z) = task.workgroup_size().unwrap_or((1, 1, 1));
            debug!("Task {task_id}: Dispatching workgroups ({x}, {y}, {z})");
            compute_pass.dispatch_workgroups(x, y, z);
        }

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back output buffers
        let output_data = self.read_output_buffers(&output_buffers).await?;

        let duration = start_time.elapsed();
        info!(
            "Task {task_id}: GPU execution completed in {:.2}ms",
            duration.as_secs_f64() * 1000.0
        );

        Ok(ExecutionResult::new_gpu(
            task_id,
            0, // Success
            output_data,
            duration,
        ))
    }

    /// Creates input buffers and uploads data to GPU.
    fn create_input_buffers(&self, task: &Task) -> Vec<wgpu::Buffer> {
        task.input_buffers()
            .iter()
            .enumerate()
            .map(|(i, data)| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Task {} Input Buffer {i}", task.id())),
                    contents: data,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect()
    }

    /// Creates output buffers on GPU.
    fn create_output_buffers(&self, task: &Task) -> Vec<wgpu::Buffer> {
        task.output_buffer_sizes()
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Task {} Output Buffer {i}", task.id())),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect()
    }

    /// Creates bind group layout for buffers.
    #[allow(clippy::cast_possible_truncation)] // Buffer counts are always small
    fn create_bind_group_layout(
        &self,
        input_count: usize,
        output_count: usize,
    ) -> wgpu::BindGroupLayout {
        let mut entries = Vec::new();

        // Input buffers (read-only storage)
        for i in 0..input_count {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        // Output buffers (read-write storage)
        for i in 0..output_count {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (input_count + i) as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &entries,
        })
    }

    /// Creates bind group binding buffers.
    #[allow(clippy::cast_possible_truncation)] // Buffer counts are always small
    fn create_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        input_buffers: &[wgpu::Buffer],
        output_buffers: &[wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let mut entries = Vec::new();

        // Bind input buffers
        for (i, buffer) in input_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        // Bind output buffers
        for (i, buffer) in output_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (input_buffers.len() + i) as u32,
                resource: buffer.as_entire_binding(),
            });
        }

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout,
            entries: &entries,
        })
    }

    /// Reads output buffers from GPU back to CPU.
    async fn read_output_buffers(&self, buffers: &[wgpu::Buffer]) -> Result<Vec<Vec<u8>>> {
        let mut output_data = Vec::new();

        for buffer in buffers {
            let size = buffer.size();

            // Create staging buffer for readback
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Copy from GPU buffer to staging buffer
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Encoder"),
            });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
            self.queue.submit(std::iter::once(encoder.finish()));

            // Map staging buffer and read data
            let (sender, receiver) = futures::channel::oneshot::channel();
            staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });

            // Poll device until mapping completes
            self.device.poll(wgpu::Maintain::Wait);

            receiver
                .await
                .map_err(|_| RepartirError::InvalidTask {
                    reason: "Failed to receive buffer mapping result".to_string(),
                })?
                .map_err(|e| RepartirError::InvalidTask {
                    reason: format!("Failed to map output buffer: {e:?}"),
                })?;

            // Copy data from mapped buffer
            let data = staging_buffer.slice(..).get_mapped_range();
            output_data.push(data.to_vec());

            // Unmap staging buffer
            drop(data);
            staging_buffer.unmap();
        }

        Ok(output_data)
    }
}

impl Executor for GpuExecutor {
    fn execute(&self, task: Task) -> BoxFuture<'_, Result<ExecutionResult>> {
        Box::pin(async move {
            let task_id = task.id();
            debug!("Executing task {task_id} on GPU");

            // For v1.1, GPU executor validates but doesn't execute binary tasks
            self.execute_compute(task).await
        })
    }

    fn capacity(&self) -> usize {
        self.compute_units
    }

    fn name(&self) -> &'static str {
        "GPU"
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::manual_range_contains,
    clippy::panic
)]
mod tests {
    use super::*;

    /// PMAT-927 FALSIFIER: the wgpu adapter-enumeration backend mask used by the
    /// GPU executor MUST NOT contain GLES (`wgpu::Backends::GL`), and MUST contain
    /// the platform's real backend.
    ///
    /// RED on `Backends::all()` / `Instance::default()` (contains GL → GLES/EGL
    /// adapter → SIGABRT-in-Drop on Linux/AMD-RADV). GREEN on `Backends::PRIMARY`.
    /// Host-independent: it inspects the bitmask, it does not create any adapter.
    #[test]
    fn test_gpu_backends_excludes_gles() {
        let mask = gpu_backends();

        assert!(
            !mask.contains(wgpu::Backends::GL),
            "gpu_backends() must NOT include Backends::GL (GLES/EGL panics in Drop \
             on Linux/AMD-RADV → SIGABRT). mask = {mask:?}"
        );

        #[cfg(any(target_os = "linux", target_os = "android"))]
        assert!(
            mask.contains(wgpu::Backends::VULKAN),
            "gpu_backends() must include VULKAN on Linux (AMD-RADV/NVIDIA). mask = {mask:?}"
        );
        #[cfg(target_os = "macos")]
        assert!(
            mask.contains(wgpu::Backends::METAL),
            "gpu_backends() must include METAL on macOS (Apple Silicon). mask = {mask:?}"
        );
        #[cfg(target_os = "windows")]
        assert!(
            mask.contains(wgpu::Backends::VULKAN) || mask.contains(wgpu::Backends::DX12),
            "gpu_backends() must include VULKAN or DX12 on Windows. mask = {mask:?}"
        );
    }

    #[tokio::test]
    async fn test_gpu_executor_creation() {
        // This test may fail on CI without GPU
        match GpuExecutor::new().await {
            Ok(executor) => {
                assert_eq!(executor.name(), "GPU");
                assert!(executor.capacity() > 0);
                assert!(!executor.device_name().is_empty());
                println!("GPU found: {}", executor.device_name());
            }
            Err(e) => {
                println!("No GPU available (expected on CI): {e}");
                // Not a test failure - GPU may not be available
            }
        }
    }

    #[test]
    fn test_compute_units_estimation_discrete() {
        use wgpu::{AdapterInfo, Backend, DeviceType};

        let info = AdapterInfo {
            name: "Test GPU".to_string(),
            vendor: 0,
            device: 0,
            device_type: DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: Backend::Vulkan,
        };

        let units = GpuExecutor::estimate_compute_units(&info);
        assert_eq!(units, 2048); // Discrete GPU estimate
    }

    #[test]
    fn test_compute_units_estimation_integrated() {
        use wgpu::{AdapterInfo, Backend, DeviceType};

        let info = AdapterInfo {
            name: "Integrated GPU".to_string(),
            vendor: 0,
            device: 0,
            device_type: DeviceType::IntegratedGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: Backend::Vulkan,
        };

        let units = GpuExecutor::estimate_compute_units(&info);
        assert_eq!(units, 512); // Integrated GPU estimate
    }

    #[test]
    fn test_compute_units_estimation_virtual() {
        use wgpu::{AdapterInfo, Backend, DeviceType};

        let info = AdapterInfo {
            name: "Virtual GPU".to_string(),
            vendor: 0,
            device: 0,
            device_type: DeviceType::VirtualGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: Backend::Vulkan,
        };

        let units = GpuExecutor::estimate_compute_units(&info);
        assert_eq!(units, 1024); // Virtual GPU estimate
    }

    #[test]
    fn test_compute_units_estimation_cpu() {
        use wgpu::{AdapterInfo, Backend, DeviceType};

        let info = AdapterInfo {
            name: "CPU Rasterizer".to_string(),
            vendor: 0,
            device: 0,
            device_type: DeviceType::Cpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: Backend::Vulkan,
        };

        let units = GpuExecutor::estimate_compute_units(&info);
        assert_eq!(units, 16); // CPU rasterizer estimate
    }

    #[test]
    fn test_compute_units_estimation_other() {
        use wgpu::{AdapterInfo, Backend, DeviceType};

        let info = AdapterInfo {
            name: "Unknown GPU".to_string(),
            vendor: 0,
            device: 0,
            device_type: DeviceType::Other,
            driver: String::new(),
            driver_info: String::new(),
            backend: Backend::Vulkan,
        };

        let units = GpuExecutor::estimate_compute_units(&info);
        assert_eq!(units, 256); // Conservative estimate
    }

    #[tokio::test]
    async fn test_gpu_executor_not_ready_for_tasks() {
        // GPU executor should reject binary tasks in v1.1
        match GpuExecutor::new().await {
            Ok(executor) => {
                use crate::task::Backend;

                let task = Task::builder()
                    .binary("/bin/echo")
                    .arg("test")
                    .backend(Backend::Gpu)
                    .build()
                    .unwrap();

                let result = executor.execute(task).await;
                assert!(result.is_err());

                if let Err(RepartirError::InvalidTask { reason }) = result {
                    assert!(reason.contains("not support"));
                }
            }
            Err(_) => {
                // No GPU available - skip test
                println!("Skipping test: No GPU available");
            }
        }
    }
}
