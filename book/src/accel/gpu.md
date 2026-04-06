# GPU Compute

GPU acceleration provides massive parallelism for large-scale visualization
tasks through the `gpu` feature.

## Enabling GPU Support

```toml
[dependencies]
trueno-viz = { version = "0.1", features = ["gpu"] }
```

## Device Selection

```rust
#[cfg(feature = "gpu")]
use trueno_viz::accel::gpu;

#[cfg(feature = "gpu")]
fn main() {
    // List available devices
    let devices = gpu::list_devices();
    for device in &devices {
        println!("{}: {}", device.name(), device.backend());
    }

    // Auto-select best device
    let device = gpu::default_device().unwrap();
    println!("Using: {}", device.name());
}
```

## GPU-Accelerated Operations

### Heatmap Rendering

```rust
#[cfg(feature = "gpu")]
{
    use trueno_viz::plots::Heatmap;

    let data: Vec<f32> = (0..4_000_000).map(|i| i as f32).collect();

    let heatmap = Heatmap::new(&data, 2000, 2000)
        .gpu(true)  // Enable GPU acceleration
        .build();

    heatmap.render_to_file("large_heatmap.png").unwrap();
}
```

### Batch Rendering

Render multiple plots in parallel on GPU:

```rust
#[cfg(feature = "gpu")]
{
    use trueno_viz::accel::gpu::GpuBatch;

    let batch = GpuBatch::new()
        .add(plot1)
        .add(plot2)
        .add(plot3);

    let results = batch.render_all(800, 600).unwrap();
}
```

## Supported Backends

| Backend | Platform | Status |
|---------|----------|--------|
| Vulkan | Linux, Windows | Full |
| Metal | macOS, iOS | Full |
| DX12 | Windows | Full |
| WebGPU | Browser | Experimental |

## Fallback Behavior

GPU operations fall back to CPU automatically:

```rust
#[cfg(feature = "gpu")]
{
    use trueno_viz::accel::gpu;

    if gpu::is_available() {
        println!("Using GPU");
    } else {
        println!("Falling back to CPU (SIMD)");
    }
}
```

## Performance Comparison

```text
Heatmap 2000x2000 (4M cells):
  CPU (scalar):  2500ms
  CPU (SIMD):     450ms
  GPU:             35ms
```

## Next Chapter

Continue to [WebAssembly](./wasm.md) for browser deployment.
