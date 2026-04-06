//! GPU compute kernels (WGSL shaders)
//!
//! Parallel Reduction Algorithm (Harris 2007):
//! 1. Each thread loads one element
//! 2. Workgroup-local reduction using shared memory
//! 3. Global reduction of workgroup results
//!
//! Performance: O(N/P + log P) where P = num threads

use crate::{Error, Result};
use arrow::array::{Array, Float32Array, Int32Array};
use wgpu;
use wgpu::util::DeviceExt;

/// Workgroup size (256 threads = 8 warps on NVIDIA, optimal for most GPUs)
const WORKGROUP_SIZE: u32 = 256;

/// WGSL shader for parallel SUM reduction (i32)
const SUM_I32_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<i32>>;

var<workgroup> shared_data: array<i32, 256>;

@compute @workgroup_size(256)
fn sum_reduce(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let input_size = arrayLength(&input);

    // Load data into shared memory
    if (gid < input_size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0;
    }
    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride && gid + stride < input_size) {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread writes workgroup result
    if (tid == 0u) {
        atomicAdd(&output[0], shared_data[0]);
    }
}
";

/// WGSL shader for parallel SUM reduction (f32)
#[allow(dead_code)]
const SUM_F32_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_reduce(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let input_size = arrayLength(&input);

    // Load data into shared memory
    if (gid < input_size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }
    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride && gid + stride < input_size) {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread writes workgroup result
    if (tid == 0u) {
        output[0] += shared_data[0];
    }
}
";

/// WGSL shader for COUNT
#[allow(dead_code)]
const COUNT_SHADER: &str = r"
@group(0) @binding(0) var<storage, read_write> output: array<atomic<u32>>;

@compute @workgroup_size(256)
fn count_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let array_size: u32 = @ARRAY_SIZE@;

    if (global_id.x < array_size) {
        atomicAdd(&output[0], 1u);
    }
}
";

/// WGSL shader for MIN reduction (i32)
#[allow(dead_code)]
const MIN_I32_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<i32>>;

var<workgroup> shared_data: array<i32, 256>;

@compute @workgroup_size(256)
fn min_reduce(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let input_size = arrayLength(&input);

    // Load data into shared memory
    if (gid < input_size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 2147483647; // i32::MAX
    }
    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride && gid + stride < input_size) {
            shared_data[tid] = min(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread writes workgroup result
    if (tid == 0u) {
        atomicMin(&output[0], shared_data[0]);
    }
}
";

/// WGSL shader for MAX reduction (i32)
#[allow(dead_code)]
const MAX_I32_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<i32>>;

var<workgroup> shared_data: array<i32, 256>;

@compute @workgroup_size(256)
fn max_reduce(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let input_size = arrayLength(&input);

    // Load data into shared memory
    if (gid < input_size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = -2147483648; // i32::MIN
    }
    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride && gid + stride < input_size) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread writes workgroup result
    if (tid == 0u) {
        atomicMax(&output[0], shared_data[0]);
    }
}
";

/// Execute SUM aggregation on GPU (i32)
///
/// # Errors
/// Returns error if GPU execution fails
///
/// # Panics
/// May panic if buffer mapping fails (internal GPU error)
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_possible_truncation)]
pub async fn sum_i32(device: &wgpu::Device, queue: &wgpu::Queue, data: &Int32Array) -> Result<i32> {
    let input_data: Vec<i32> = data.values().to_vec();
    let input_size = input_data.len();

    if input_size == 0 {
        return Ok(0);
    }

    // Create input buffer
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create output buffer (initialized to 0)
    let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Output Buffer"),
        contents: bytemuck::cast_slice(&[0i32]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    // Create compute pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("SUM i32 Shader"),
        source: wgpu::ShaderSource::Wgsl(SUM_I32_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SUM i32 Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "sum_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
        ],
    });

    // Execute compute shader
    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_count = (input_size as u32).div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Read result buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: 4, // i32 = 4 bytes
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 4);
    queue.submit(Some(encoder.finish()));

    // Map buffer and read result
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("Failed to send buffer mapping result through channel");
    });
    device.poll(wgpu::Maintain::Wait);

    receiver
        .receive()
        .await
        .ok_or_else(|| Error::Other("Failed to receive mapping result".to_string()))?
        .map_err(|e| Error::Other(format!("Buffer mapping failed: {e:?}")))?;

    let data = buffer_slice.get_mapped_range();
    let result = i32::from_le_bytes(
        data[0..4].try_into().expect("Buffer must contain at least 4 bytes for i32 result"),
    );
    drop(data);
    staging_buffer.unmap();

    Ok(result)
}

/// Execute SUM aggregation on GPU (f32)
/// Placeholder - not yet implemented
///
/// # Errors
/// Returns error (not yet implemented)
#[allow(clippy::unused_async)]
pub async fn sum_f32(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    _data: &Float32Array,
) -> Result<f32> {
    // Placeholder implementation
    Err(Error::Other("f32 SUM not yet implemented".to_string()))
}

/// Execute COUNT aggregation on GPU
/// Trivial implementation - just returns array length
///
/// # Errors
/// Does not return errors in current implementation
#[allow(clippy::unused_async)]
pub async fn count(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    data: &dyn Array,
) -> Result<usize> {
    // COUNT is trivial - just return array length
    Ok(data.len())
}

/// Execute MIN aggregation on GPU (i32)
///
/// # Errors
/// Returns error if GPU execution fails
///
/// # Panics
/// May panic if buffer mapping fails (internal GPU error)
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_possible_truncation)]
pub async fn min_i32(device: &wgpu::Device, queue: &wgpu::Queue, data: &Int32Array) -> Result<i32> {
    let input_data: Vec<i32> = data.values().to_vec();
    let input_size = input_data.len();

    if input_size == 0 {
        return Ok(i32::MAX); // Empty array minimum is i32::MAX
    }

    // Create input buffer
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MIN Input Buffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create output buffer (initialized to i32::MAX)
    let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MIN Output Buffer"),
        contents: bytemuck::cast_slice(&[i32::MAX]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    // Create compute pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MIN i32 Shader"),
        source: wgpu::ShaderSource::Wgsl(MIN_I32_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MIN Bind Group Layout"),
        entries: &[
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MIN Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MIN i32 Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "min_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MIN Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MIN Compute Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MIN Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_count = (input_size as u32).div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Read result buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MIN Staging Buffer"),
        size: 4, // i32 = 4 bytes
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 4);
    queue.submit(Some(encoder.finish()));

    // Map buffer and read result
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("Failed to send buffer mapping result through channel");
    });
    device.poll(wgpu::Maintain::Wait);

    receiver
        .receive()
        .await
        .ok_or_else(|| Error::Other("Failed to receive mapping result".to_string()))?
        .map_err(|e| Error::Other(format!("Buffer mapping failed: {e:?}")))?;

    let data = buffer_slice.get_mapped_range();
    let result = i32::from_le_bytes(
        data[0..4].try_into().expect("Buffer must contain at least 4 bytes for i32 result"),
    );
    drop(data);
    staging_buffer.unmap();

    Ok(result)
}

/// Execute MAX aggregation on GPU (i32)
///
/// # Errors
/// Returns error if GPU execution fails
///
/// # Panics
/// May panic if buffer mapping fails (internal GPU error)
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_possible_truncation)]
pub async fn max_i32(device: &wgpu::Device, queue: &wgpu::Queue, data: &Int32Array) -> Result<i32> {
    let input_data: Vec<i32> = data.values().to_vec();
    let input_size = input_data.len();

    if input_size == 0 {
        return Ok(i32::MIN); // Empty array maximum is i32::MIN
    }

    // Create input buffer
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MAX Input Buffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create output buffer (initialized to i32::MIN)
    let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MAX Output Buffer"),
        contents: bytemuck::cast_slice(&[i32::MIN]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    // Create compute pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MAX i32 Shader"),
        source: wgpu::ShaderSource::Wgsl(MAX_I32_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MAX Bind Group Layout"),
        entries: &[
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MAX Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MAX i32 Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "max_reduce",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MAX Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
        ],
    });

    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MAX Compute Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MAX Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_count = (input_size as u32).div_ceil(WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Read result buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MAX Staging Buffer"),
        size: 4, // i32 = 4 bytes
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 4);
    queue.submit(Some(encoder.finish()));

    // Map buffer and read result
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("Failed to send buffer mapping result through channel");
    });
    device.poll(wgpu::Maintain::Wait);

    receiver
        .receive()
        .await
        .ok_or_else(|| Error::Other("Failed to receive mapping result".to_string()))?
        .map_err(|e| Error::Other(format!("Buffer mapping failed: {e:?}")))?;

    let data = buffer_slice.get_mapped_range();
    let result = i32::from_le_bytes(
        data[0..4].try_into().expect("Buffer must contain at least 4 bytes for i32 result"),
    );
    drop(data);
    staging_buffer.unmap();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;

    #[tokio::test]
    async fn test_count_returns_array_length() {
        // COUNT is trivial - doesn't need GPU
        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);

        // Create mock device/queue (not used by count())
        let instance = wgpu::Instance::default();
        let Some(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await
        else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };
        let Ok((device, queue)) =
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await
        else {
            eprintln!("Skipping GPU test (failed to create device)");
            return;
        };

        let result = count(&device, &queue, &data).await.unwrap();
        assert_eq!(result, 5);
    }

    #[tokio::test]
    async fn test_count_empty_array() {
        let data = Int32Array::from(vec![] as Vec<i32>);

        let instance = wgpu::Instance::default();
        let Some(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await
        else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };
        let Ok((device, queue)) =
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await
        else {
            eprintln!("Skipping GPU test (failed to create device)");
            return;
        };

        let result = count(&device, &queue, &data).await.unwrap();
        assert_eq!(result, 0);
    }

    #[tokio::test]
    async fn test_sum_f32_not_implemented() {
        // sum_f32 is placeholder - should return error
        let data = Float32Array::from(vec![1.0, 2.0, 3.0]);

        let instance = wgpu::Instance::default();
        let Some(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await
        else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };
        let Ok((device, queue)) =
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await
        else {
            eprintln!("Skipping GPU test (failed to create device)");
            return;
        };

        let result = sum_f32(&device, &queue, &data).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not yet implemented"));
    }
}
