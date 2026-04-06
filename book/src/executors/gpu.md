# GPU Executor

The GPU executor enables you to run compute shaders on GPUs using WGSL (WebGPU Shading Language).

## Overview

The `GpuExecutor` provides:

- **Cross-Platform**: Runs on Vulkan (Linux), Metal (macOS), DirectX 12 (Windows), WebGPU (browsers)
- **WGSL Shaders**: Modern, safe shader language
- **Buffer Management**: Automatic upload/download of data
- **Async Execution**: Non-blocking GPU operations

## Basic Usage

```rust
use repartir::executor::gpu::GpuExecutor;
use repartir::executor::Executor;
use repartir::task::{Task, Backend};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Create GPU executor
    let executor = GpuExecutor::new().await?;
    println!("GPU: {}", executor.device_name());
    println!("Compute units: {}", executor.capacity());

    // Define WGSL shader
    let shader = r#"
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            output[idx] = input[idx] * 2.0;
        }
    "#;

    // Prepare input data (1024 floats)
    let input_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let input_bytes: Vec<u8> = input_data.iter()
        .flat_map(|f| f.to_ne_bytes())
        .collect();

    // Create GPU task
    let task = Task::builder()
        .shader_code(shader.as_bytes().to_vec())
        .input_buffer(input_bytes)
        .output_buffer_size(1024 * 4) // 1024 floats * 4 bytes
        .workgroup_size(16, 1, 1) // 16 workgroups * 64 threads = 1024
        .backend(Backend::Gpu)
        .build()?;

    // Execute on GPU
    let result = executor.execute(task).await?;

    // Read output
    let output_bytes = &result.gpu_output_buffers()[0];
    let output_data: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    println!("Input[0]: {}", input_data[0]);
    println!("Output[0]: {}", output_data[0]);
    println!("Verified: {}", output_data[0] == input_data[0] * 2.0);

    Ok(())
}
```

## WGSL Shader Structure

A typical compute shader has:

1. **Storage Buffers**: Input/output data
2. **Workgroup Size**: Parallelism configuration
3. **Entry Point**: `main` function with compute attribute

```wgsl
// Binding 0: Input buffer (read-only)
@group(0) @binding(0)
var<storage, read> input: array<f32>;

// Binding 1: Output buffer (read-write)
@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

// Compute shader entry point
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        output[idx] = sqrt(input[idx]);
    }
}
```

## Buffer Bindings

Buffers are automatically bound in order:

- **Input buffers**: Bindings 0, 1, 2, ... (read-only storage)
- **Output buffers**: Bindings N, N+1, N+2, ... (read-write storage)

Where N = number of input buffers.

### Example with Multiple Buffers

```rust
let task = Task::builder()
    .shader_code(shader.as_bytes().to_vec())
    .input_buffer(input1_bytes)  // Binding 0
    .input_buffer(input2_bytes)  // Binding 1
    .output_buffer_size(output1_size)  // Binding 2
    .output_buffer_size(output2_size)  // Binding 3
    .workgroup_size(16, 1, 1)
    .backend(Backend::Gpu)
    .build()?;
```

Corresponding shader:

```wgsl
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output1: array<f32>;
@group(0) @binding(3) var<storage, read_write> output2: array<f32>;
```

## Workgroup Sizing

Workgroups determine GPU parallelism:

```rust
// For 1D data (1024 elements)
.workgroup_size(16, 1, 1)  // 16 workgroups * 64 threads = 1024

// For 2D data (512x512 image)
.workgroup_size(32, 32, 1)  // 32x32 workgroups * 16x16 threads = 512x512

// For 3D data (64x64x64 volume)
.workgroup_size(8, 8, 8)   // 8x8x8 workgroups * 8x8x8 threads = 64³
```

Workgroup size in shader (`@workgroup_size(x, y, z)`) must match data dimensions.

## Performance Tips

1. **Align Buffer Sizes**: Use multiples of 256 bytes for optimal GPU performance
2. **Minimize Transfers**: Batch operations to reduce CPU↔GPU data movement
3. **Workgroup Tuning**: Experiment with workgroup sizes (64, 128, 256 are common)
4. **Shared Memory**: Use `workgroup` storage for inter-thread communication

## Error Handling

GPU errors are returned as `RepartirError::InvalidTask`:

```rust
match executor.execute(task).await {
    Ok(result) => {
        println!("Success! {} output buffers", result.gpu_output_buffers().len());
    }
    Err(e) => {
        eprintln!("GPU error: {}", e);
        // Fallback to CPU?
    }
}
```

Common errors:

- **No GPU adapter found**: No GPU available (fallback to CPU)
- **Shader compilation failed**: WGSL syntax error
- **Buffer mapping failed**: GPU memory exhausted

## See Also

- **[GPU WGSL Guide](./gpu-wgsl.md)**: Complete WGSL language reference
- **[GPU Buffer Management](./gpu-buffers.md)**: Deep dive into buffer handling
- **[GPU Examples](./gpu-examples.md)**: Matrix multiply, image processing, more
