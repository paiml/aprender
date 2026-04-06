# GPU Pipeline

## Overview

The GPU pipeline uses CUDA for batch compression with async DMA.

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│   Host     │───▶│    H2D     │───▶│   Kernel   │───▶│    D2H     │
│   Memory   │    │  Transfer  │    │  Execution │    │  Transfer  │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
      ▲                                                      │
      └──────────────────────────────────────────────────────┘
```

## Pipeline Stages

### 1. Host-to-Device Transfer

```rust
fn transfer_to_device(&self, pages: &[[u8; PAGE_SIZE]]) -> Result<CudaSlice<u8>> {
    // Flatten pages into contiguous buffer
    let total_bytes = pages.len() * PAGE_SIZE;
    let mut flat_data = Vec::with_capacity(total_bytes);
    for page in pages {
        flat_data.extend_from_slice(page);
    }

    // Async copy to device
    let device_buffer = self.stream.clone_htod(&flat_data)?;
    Ok(device_buffer)
}
```

### 2. Kernel Execution

```rust
fn execute_kernel(&self, input: &CudaSlice<u8>, batch_size: u32) -> Result<CudaSlice<u8>> {
    // Allocate output buffer
    let mut output = self.stream.alloc_zeros::<u8>(batch_size as usize * PAGE_SIZE)?;
    let mut sizes = self.stream.alloc_zeros::<u32>(batch_size as usize)?;

    // Launch kernel
    // Grid: ceil(batch_size / 4) blocks
    // Block: 128 threads (4 warps)
    unsafe {
        self.stream
            .launch_builder(&self.kernel_fn)
            .arg(&input)
            .arg(&mut output)
            .arg(&mut sizes)
            .arg(&batch_size)
            .launch(cfg)?;
    }

    self.stream.synchronize()?;
    Ok(output)
}
```

### 3. Device-to-Host Transfer

```rust
fn transfer_from_device(&self, data: CudaSlice<u8>) -> Result<Vec<CompressedPage>> {
    let output = self.stream.clone_dtoh(&data)?;

    // Convert to CompressedPage structures
    // ...
}
```

## Warp-Cooperative Kernel

The LZ4 kernel uses warp-cooperative compression:

```
Block (128 threads)
├── Warp 0 (32 threads) → Page 0
├── Warp 1 (32 threads) → Page 1
├── Warp 2 (32 threads) → Page 2
└── Warp 3 (32 threads) → Page 3
```

### Shared Memory Layout

```
Shared Memory (48 KB per block)
├── Warp 0: 12 KB (hash table + working)
├── Warp 1: 12 KB
├── Warp 2: 12 KB
└── Warp 3: 12 KB
```

### PTX Generation

Using trueno-gpu for pure Rust PTX:

```rust
use trueno_gpu::kernels::lz4::Lz4WarpCompressKernel;
use trueno_gpu::kernels::Kernel;

let kernel = Lz4WarpCompressKernel::new(65536);
let ptx_string = kernel.emit_ptx();

// Load PTX into CUDA module
let ptx = Ptx::from(ptx_string);
let module = context.load_module(ptx)?;
```

## Async DMA Ring Buffer

```rust
struct AsyncPipeline {
    slots: Vec<PipelineSlot>,
    head: usize,
    tail: usize,
}

struct PipelineSlot {
    input_buffer: CudaSlice<u8>,
    output_buffer: CudaSlice<u8>,
    event: CudaEvent,
    state: SlotState,
}

enum SlotState {
    Free,
    H2DInProgress,
    KernelInProgress,
    D2HInProgress,
    Complete,
}
```

### Pipelining

```
Time ──────────────────────────────────────────────▶

Slot 0: [H2D][Kernel][D2H]
Slot 1:      [H2D][Kernel][D2H]
Slot 2:           [H2D][Kernel][D2H]
Slot 3:                [H2D][Kernel][D2H]
```

## Performance Optimization

### 1. Pinned Memory

```rust
// Use pinned (page-locked) memory for faster transfers
let pinned_buffer = cuda_malloc_host(size)?;
```

### 2. Stream Overlap

```rust
// Use separate streams for H2D, compute, D2H
let h2d_stream = context.create_stream()?;
let compute_stream = context.create_stream()?;
let d2h_stream = context.create_stream()?;
```

### 3. Kernel Occupancy

```rust
// Optimal: 4 warps per block = 128 threads
// Shared memory: 48 KB (12 KB per warp)
// Registers: ~32 per thread
```

## Error Handling

```rust
match kernel_result {
    Ok(output) => process_output(output),
    Err(CudaError::IllegalAddress) => {
        // Memory access violation
        fallback_to_cpu()?
    }
    Err(CudaError::LaunchFailed) => {
        // Kernel launch failed
        fallback_to_cpu()?
    }
    Err(e) => Err(Error::GpuNotAvailable(e.to_string())),
}
```
