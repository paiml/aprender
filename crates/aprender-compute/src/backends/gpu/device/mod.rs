//! GPU device initialization and management
//!
//! This module provides cross-platform GPU compute via wgpu (WebGPU).
//!
//! # Platform differences
//!
//! - **Native**: Sync wrappers available using `pollster::block_on`
//! - **WASM**: Sync wrappers unavailable (can't block main thread); use `*_async` methods
//!
//! Use `runtime::sync_available()` to check at runtime.

mod activations;
mod backward;
mod eigen;
pub(crate) mod linalg;
mod reductions;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use super::runtime;

/// Process-global lock serializing native wgpu instance/adapter/device creation.
///
/// PMAT-778: On hosts whose Vulkan ICD is unsafe to initialize concurrently
/// (notably the NVIDIA GB10 / aarch64, where the Mesa freedreno "Turnip" ICD
/// probes `/dev/dri/renderD128` and **segfaults** when multiple threads create
/// `wgpu::Instance`s and request adapters/devices simultaneously), every sync
/// device-creation entry point takes this lock. Device creation is a rare,
/// one-time-per-scheduler operation off the compute hot path, so serializing it
/// is free on healthy GPUs and turns a concurrent-init crash into ordered,
/// correct initialization. This is the companion to the adapter-probe `OnceLock`
/// cache (PMAT-773): the probe is memoized once, and actual device acquisition
/// is serialized here.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
static DEVICE_INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Process-global, lazily-created shared `wgpu::Instance`.
///
/// PMAT-778: Creating a fresh `wgpu::Instance` enumerates every installed Vulkan
/// ICD. On the NVIDIA GB10 / aarch64 the host ships ~11 Mesa ICDs (freedreno
/// "Turnip", panfrost, asahi, …) that are irrelevant to this hardware; the
/// freedreno ICD spawns its own background Vulkan threads and **segfaults** when
/// several instances enumerate it concurrently (each open of
/// `/dev/dri/renderD128` returns `VK_ERROR_INCOMPATIBLE_DRIVER`). Sharing one
/// instance for the whole process means the broken ICD is enumerated exactly
/// once, eliminating the concurrent-init race entirely. `wgpu::Instance` is
/// `Clone`/`Send`/`Sync`, so every adapter/device request can cheaply reuse it.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub(crate) fn shared_instance() -> wgpu::Instance {
    use std::sync::OnceLock;
    static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
    INSTANCE
        .get_or_init(|| {
            // Serialize the one-time enumeration against any other GPU init.
            let _guard = DEVICE_INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            wgpu::Instance::default()
        })
        .clone()
}

/// GPU device manager
#[derive(Clone)]
pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuDevice {
    /// Initialize GPU device (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn new() -> Result<Self, String> {
        // PMAT-778: serialize concurrent native device creation (freedreno ICD
        // segfaults on the GB10 when instances/devices are created in parallel).
        let _guard = DEVICE_INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        runtime::block_on(async { Self::new_async().await })
    }

    /// Initialize GPU device (async, works on all platforms)
    pub async fn new_async() -> Result<Self, String> {
        // Create instance
        let instance = shared_instance();

        // Request adapter (GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to find GPU adapter: {}", e))?;

        // Request device and queue with adapter's actual max buffer size
        // Default wgpu limits cap buffers at 256MB, which is too small for
        // 7B+ model weight matrices (e.g., FFN [18944, 3584] x f32 = 271MB)
        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = adapter.limits().max_buffer_size;
        limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Trueno GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        Ok(Self { device, queue })
    }

    /// Initialize GPU device with a specific adapter index (sync, native only)
    ///
    /// Use this to select a specific GPU when multiple are available.
    /// Adapter indices correspond to `Instance::enumerate_adapters()` ordering.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn new_with_adapter_index(index: u32) -> Result<Self, String> {
        // PMAT-778: serialize concurrent native device creation (see DEVICE_INIT_LOCK).
        let _guard = DEVICE_INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        runtime::block_on(async { Self::new_with_adapter_index_async(index).await })
    }

    /// Initialize GPU device with a specific adapter index (async, all platforms)
    ///
    /// Use this to select a specific GPU when multiple are available.
    /// Adapter indices correspond to `Instance::enumerate_adapters()` ordering.
    pub async fn new_with_adapter_index_async(index: u32) -> Result<Self, String> {
        let instance = shared_instance();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        if adapters.is_empty() {
            return Err("No GPU adapters found".to_string());
        }

        let adapter = adapters
            .into_iter()
            .nth(index as usize)
            .ok_or_else(|| format!("GPU adapter index {} out of range", index))?;

        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = adapter.limits().max_buffer_size;
        limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(&format!("Trueno GPU Device [{}]", index)),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("Failed to create device at index {}: {}", index, e))?;

        Ok(Self { device, queue })
    }

    /// List all available GPU adapters (sync, native only)
    ///
    /// Returns a list of (index, name, backend) tuples for each adapter.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn list_adapters() -> Vec<(u32, String, String)> {
        // PMAT-778: serialize concurrent native instance/adapter enumeration
        // (freedreno ICD segfaults on the GB10 under concurrent init).
        let _guard = DEVICE_INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        runtime::block_on(Self::list_adapters_async())
    }

    /// List all available GPU adapters (async, all platforms)
    pub async fn list_adapters_async() -> Vec<(u32, String, String)> {
        let instance = shared_instance();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        adapters
            .iter()
            .enumerate()
            .map(|(idx, adapter)| {
                let info = adapter.get_info();
                (idx as u32, info.name, format!("{:?}", info.backend))
            })
            .collect()
    }

    /// Check if GPU is available (sync, native only)
    ///
    /// PMAT-773: The adapter probe is cached for the lifetime of the process via a
    /// [`std::sync::OnceLock`]. On hosts WITHOUT a wgpu-compatible adapter (e.g.
    /// headless NVIDIA / Jetson / Blackwell GB10, where opening
    /// `/dev/dri/renderD128` fails with `VK_ERROR_INCOMPATIBLE_DRIVER`), the
    /// underlying `request_adapter` call is expensive and was being re-attempted by
    /// every test/caller, adding diffuse latency across a test suite. Caching the
    /// result means the failing Vulkan device open is attempted at most once per
    /// process; subsequent callers short-circuit.
    ///
    /// Behavior on hosts WITH a usable GPU is unchanged: the first probe succeeds,
    /// `true` is cached, and callers proceed to acquire a real device via
    /// [`Self::new`] exactly as before. The cache only memoizes whether an adapter
    /// is obtainable — it never holds a device, so real-GPU acquisition is never
    /// short-circuited.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn is_available() -> bool {
        use std::sync::OnceLock;
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| runtime::block_on(Self::is_available_async()))
    }

    /// Check if GPU is available (async, works on all platforms)
    pub async fn is_available_async() -> bool {
        let instance = shared_instance();
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .is_ok()
    }

    /// Generic helper for element-wise GPU operations
    ///
    /// This helper eliminates code duplication between element-wise operations
    /// (relu, clip, sigmoid, tanh, etc.) by abstracting the common GPU compute pattern.
    ///
    /// # Arguments
    ///
    /// * `op_name` - Operation name for labels (e.g., "ReLU", "Clip")
    /// * `shader_source` - WGSL shader source code
    /// * `input` - Input data
    /// * `result` - Output buffer
    /// * `uniform_data` - Optional uniform buffer data (e.g., clip parameters)
    pub(super) async fn execute_element_wise_op(
        &self,
        op_name: &str,
        shader_source: &str,
        input: &[f32],
        result: &mut [f32],
        uniform_data: Option<&[u8]>,
    ) -> Result<(), String> {
        let len = input.len();

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", op_name)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create input buffer
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Input", op_name)),
            size: std::mem::size_of_val(input) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Output", op_name)),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write input data
        self.queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

        // Create optional uniform buffer
        let uniform_buffer = uniform_data.map(|data| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Uniform", op_name)),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, data);
            buffer
        });

        // Create bind group layout entries (input + output + optional uniform)
        let mut bind_group_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        ];

        // Add uniform buffer binding if present
        if uniform_buffer.is_some() {
            bind_group_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} Bind Group Layout", op_name)),
                entries: &bind_group_entries,
            });

        // Create bind group entries
        let mut bind_entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
        ];

        // Add uniform buffer binding if present
        if let Some(ref uniform_buf) = uniform_buffer {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buf.as_entire_binding(),
            });
        }

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", op_name)),
            layout: &bind_group_layout,
            entries: &bind_entries,
        });

        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", op_name)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} Pipeline", op_name)),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Staging Buffer", op_name)),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{} Encoder", op_name)),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", op_name)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per workgroup)
            let workgroup_size = 256;
            let num_workgroups = (len as u32).div_ceil(workgroup_size);

            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(result) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Poll device to ensure GPU work completes and callbacks are invoked
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();

        receiver
            .receive()
            .await
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        {
            let data = buffer_slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&data));
        }

        staging_buffer.unmap();

        Ok(())
    }
}

#[cfg(all(test, feature = "gpu", not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn test_is_available_consistency() {
        // EXTREME TDD: Kill mutant that replaces is_available() with hardcoded false
        // Test that is_available() is consistent with GpuDevice::new()
        let available = GpuDevice::is_available();
        let device_result = GpuDevice::new();

        if available {
            // If is_available() returns true, device creation should succeed
            assert!(
                device_result.is_ok(),
                "is_available() returned true, but GpuDevice::new() failed"
            );
        } else {
            // If is_available() returns false, we can't make assertions about new()
            // (it might still succeed in some edge cases, but typically should fail)
            // The key test is: mutant always returns false, so on GPU systems this fails
            eprintln!(
                "GPU not available (is_available=false), device creation result: {:?}",
                device_result.is_err()
            );
        }
    }

    #[test]
    fn test_reduce_sum_not_hardcoded() {
        // EXTREME TDD: Kill mutant that replaces reduce_sum with Ok(-1.0)
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let device = GpuDevice::new().expect("Failed to create GPU device");
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // sum = 15.0

        // reduce_sum is async, so we use runtime::block_on
        let result = runtime::block_on(device.reduce_sum(&input)).expect("reduce_sum failed");

        // Kill mutant: verify result is NOT -1.0
        assert_ne!(result, -1.0, "reduce_sum returned hardcoded -1.0 (mutant not killed)");

        // Verify correct computation
        let expected: f32 = input.iter().sum();
        assert!(
            (result - expected).abs() < 1e-4,
            "reduce_sum({:?}) = {} (expected {})",
            input,
            result,
            expected
        );
    }
}
