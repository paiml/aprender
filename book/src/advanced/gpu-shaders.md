# GPU Compute Shaders

Trueno uses WGSL (WebGPU Shading Language) compute shaders for cross-platform GPU acceleration via wgpu. This chapter covers the shader architecture, memory hierarchy abstractions, and tiled reduction algorithms.

## Memory Hierarchy Abstractions

Based on the `cuda-tile-behavior.md` specification (Section 3.2), Trueno provides two key abstractions for efficient GPU memory access:

### TensorView

`TensorView<T>` provides a structured view into GPU buffer memory with shape, stride, and layout metadata. It enables zero-copy operations like slicing and transposition.

```rust
use trueno::backends::gpu::{TensorView, MemoryLayout};

// Create a 4D tensor view (batch=2, channels=3, height=32, width=32)
let view: TensorView<f32> = TensorView::new([2, 3, 32, 32]);

println!("Shape: {:?}", view.shape());     // [2, 3, 32, 32]
println!("Strides: {:?}", view.strides()); // [3072, 1024, 32, 1]
println!("Elements: {}", view.numel());    // 6144

// Create with explicit strides for non-contiguous views
let transposed = TensorView::<f32>::with_strides(
    [32, 32, 3, 2],
    [1, 32, 1024, 3072]
);
assert!(!transposed.is_contiguous());

// Change memory layout
let col_major = TensorView::new([4, 4, 1, 1])
    .with_layout(MemoryLayout::ColumnMajor);
```

### PartitionView

`PartitionView<T>` divides a tensor into tiles for efficient GPU workgroup distribution:

```rust
use trueno::backends::gpu::{TensorView, PartitionView};

// Partition a 64x64 tensor into 16x16 tiles
let tensor: TensorView<f32> = TensorView::new([64, 64, 1, 1]);
let partition: PartitionView<f32> = PartitionView::new(tensor, [16, 16, 1, 1]);

println!("Tile count: {:?}", partition.tile_count());  // [4, 4, 1, 1]
println!("Total tiles: {}", partition.total_tiles());  // 16

// Handle non-aligned dimensions (100x100 with 16x16 tiles)
let non_aligned: TensorView<f32> = TensorView::new([100, 100, 1, 1]);
let partition2: PartitionView<f32> = PartitionView::new(non_aligned, [16, 16, 1, 1]);

// Edge tiles are automatically detected
if let Some(tile_info) = partition2.get_tile([6, 6, 0, 0]) {
    println!("Edge tile size: {:?}", tile_info.size);  // [4, 4, 1, 1]
    println!("Is edge tile: {}", tile_info.is_edge);   // true
}
```

## Tiled Reduction Algorithms

Trueno implements 16x16 tile-based reduction algorithms inspired by GPU workgroup patterns:

### TILE_SIZE Constant

```rust
use trueno::backends::gpu::TILE_SIZE;

// TILE_SIZE = 16 matches standard GPU workgroup size
// This enables efficient shared memory usage and warp/wavefront alignment
```

### Tiled Sum, Max, Min

```rust
use trueno::backends::gpu::{tiled_sum_2d, tiled_max_2d, tiled_min_2d};

// 32x32 matrix data (row-major)
let data: Vec<f32> = (1..=1024).map(|x| x as f32).collect();

// Tiled sum reduction
let sum = tiled_sum_2d(&data, 32, 32);
assert!((sum - 524800.0).abs() < 1e-3);

// Tiled max reduction
let max_data = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 8.0, 4.0, 6.0];
let max = tiled_max_2d(&max_data, 3, 3);
assert!((max - 9.0).abs() < 1e-5);

// Tiled min reduction
let min_data = vec![5.0, 3.0, 7.0, -1.0, 9.0, 2.0];
let min = tiled_min_2d(&min_data, 2, 3);
assert!((min - (-1.0)).abs() < 1e-5);
```

### Reduction Algorithm

The tiled reduction uses a tree-based pattern:

1. **Load Phase**: Each workgroup loads a 16x16 tile into shared memory
2. **Row Reduction**: 16 -> 8 -> 4 -> 2 -> 1 (horizontal)
3. **Column Reduction**: 16 -> 8 -> 4 -> 2 -> 1 (vertical)
4. **Combine Phase**: Partial results from all tiles are combined

```
Tile (16x16 elements)
┌────────────────────────────────────────┐
│ Row reduction: 16 -> 8 -> 4 -> 2 -> 1  │
│                                        │
│  [x x x x x x x x x x x x x x x x]     │
│        │                               │
│  [x x x x x x x x]  (step 1: +8)       │
│        │                               │
│  [x x x x]          (step 2: +4)       │
│        │                               │
│  [x x]              (step 3: +2)       │
│        │                               │
│  [x]                (step 4: +1)       │
│                                        │
│ Then column reduction on first column  │
└────────────────────────────────────────┘
```

### Custom Reduction Operations

You can implement custom reductions using the `ReduceOp` trait:

```rust
use trueno::backends::gpu::{tiled_reduce_2d, ReduceOp, SumOp, MaxOp, MinOp};

// Built-in operations
let sum = tiled_reduce_2d::<SumOp>(&data, width, height);
let max = tiled_reduce_2d::<MaxOp>(&data, width, height);
let min = tiled_reduce_2d::<MinOp>(&data, width, height);

// ReduceOp trait for custom operations:
// - identity(): Starting value (0 for sum, -inf for max, inf for min)
// - combine(a, b): Binary operation to combine two values
```

## WGSL Shader Architecture

### Element-wise Operations

Element-wise shaders process one element per thread:

```wgsl
@compute @workgroup_size(256)
fn relu_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) {
        return;
    }
    output[idx] = max(0.0, input[idx]);
}
```

### Reduction Shaders

Reduction shaders use shared memory and tree reduction:

```wgsl
var<workgroup> tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn tiled_sum_kernel(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    // Load to shared memory
    let gx = global_id.x;
    let gy = global_id.y;
    let lx = local_id.x;
    let ly = local_id.y;

    if (gx < width && gy < height) {
        tile[ly][lx] = input[gy * width + gx];
    } else {
        tile[ly][lx] = 0.0;  // Identity for sum
    }
    workgroupBarrier();

    // Row reduction: 16 -> 8 -> 4 -> 2 -> 1
    if (lx < 8u) { tile[ly][lx] += tile[ly][lx + 8u]; }
    workgroupBarrier();
    if (lx < 4u) { tile[ly][lx] += tile[ly][lx + 4u]; }
    workgroupBarrier();
    if (lx < 2u) { tile[ly][lx] += tile[ly][lx + 2u]; }
    workgroupBarrier();
    if (lx < 1u) { tile[ly][lx] += tile[ly][lx + 1u]; }
    workgroupBarrier();

    // Column reduction on first column
    if (lx == 0u && ly < 8u) { tile[ly][0] += tile[ly + 8u][0]; }
    workgroupBarrier();
    // ... continue tree reduction

    // Write partial result
    if (lx == 0u && ly == 0u) {
        let tile_idx = wg_id.y * tiles_x + wg_id.x;
        partials[tile_idx] = tile[0][0];
    }
}
```

## Performance Characteristics

| Aspect | Value | Notes |
|--------|-------|-------|
| Tile size | 16x16 | Matches GPU workgroup size |
| Shared memory | 1KB per tile | 256 f32 values |
| Reduction depth | 4 steps per dimension | log2(16) = 4 |
| Memory access | Coalesced | Row-major within tiles |
| Bank conflicts | Zero | Power-of-two tile dimensions |

### Metal Validation Results (2026-01-03)

Validated on AMD Radeon Pro W5700X (Mac Pro 7,1):

| Size | GPU Throughput | Notes |
|------|----------------|-------|
| 1M elements | 121 Melem/s | 16x16 tile fits L2 cache |
| 10M elements | 149 Melem/s | Multiple tiles, good scaling |
| 32M elements | 149 Melem/s | Metal buffer limit (~128MB) |

**Key finding**: Consistent ~150 Melem/s throughput demonstrates efficient tiled reduction algorithm regardless of input size.

## Best Practices

1. **Use power-of-two tile sizes** - Enables efficient memory coalescing and avoids bank conflicts
2. **Prefer 16x16 workgroups** - Matches warp/wavefront size on most GPUs
3. **Minimize global memory access** - Load once to shared memory, compute locally
4. **Handle edge tiles** - Use identity elements for out-of-bounds values
5. **Use CPU fallback for validation** - The tiled reduction CPU implementation mirrors GPU algorithm

## Running Examples

```bash
# Run the GPU tiled reduction demo
cargo run --example gpu_tiled_reduction --features gpu --release

# Run GPU batch operations demo
cargo run --example gpu_batch_demo --features gpu --release

# Run tiled reduction benchmarks
cargo bench --features gpu --bench gpu_reduction
```

## Related Documentation

- [cuda-tile-behavior.md](../../../docs/specifications/cuda-tile-behavior.md) - Full specification
- [Performance Targets](../performance/targets.md) - Expected speedups
- [Backend Selection](../architecture/backend-selection.md) - When GPU is selected
