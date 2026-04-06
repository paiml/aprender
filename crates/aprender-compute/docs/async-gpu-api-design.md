# Async GPU API Design

## Motivation

Current GPU backend issues:
- **Each operation blocks**: Uses `pollster::block_on` for every call
- **Redundant transfers**: Data copied to/from GPU for each operation
- **No batching**: Operations can't be combined to amortize overhead
- **Example**: `vec.relu().scale(2.0).add(&other)` does 3 separate GPU transfers

**Goal**: Reduce GPU transfers by 2x through operation batching (v0.3.0 success criteria)

## Current Architecture

```rust
// Synchronous API (current)
pub fn relu(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
    pollster::block_on(async { self.relu_async(input, result).await })
}

// Each operation:
// 1. Creates buffers
// 2. Uploads data to GPU
// 3. Executes shader
// 4. Downloads result from GPU
// 5. Blocks until complete
```

**Problem**: 3 chained operations = 6 CPU↔GPU transfers (3 up, 3 down)

## Proposed Async Architecture

### Design 1: Command Builder Pattern (Recommended)

```rust
// New async API
pub struct GpuCommandBatch {
    device: Arc<GpuDevice>,
    operations: Vec<GpuOp>,
    intermediate_buffers: HashMap<BufferId, wgpu::Buffer>,
}

impl GpuCommandBatch {
    pub fn new(device: Arc<GpuDevice>) -> Self { ... }

    // Queue operations without executing
    pub fn relu(&mut self, input: BufferId) -> BufferId { ... }
    pub fn scale(&mut self, input: BufferId, scalar: f32) -> BufferId { ... }
    pub fn add(&mut self, a: BufferId, b: BufferId) -> BufferId { ... }

    // Execute all queued operations as single batch
    pub async fn execute(&mut self) -> Result<(), String> { ... }

    // Read result buffer back to CPU
    pub async fn read_buffer(&self, buffer_id: BufferId) -> Result<Vec<f32>, String> { ... }
}

// Usage
let mut batch = GpuCommandBatch::new(device.clone());
let input = batch.upload_data(&vec![1.0, 2.0, 3.0]);
let relu_out = batch.relu(input);
let scaled = batch.scale(relu_out, 2.0);
let final_out = batch.add(scaled, other_buffer);
batch.execute().await?;
let result = batch.read_buffer(final_out).await?;

// Result: 2 transfers (1 up, 1 down) instead of 6!
```

**Benefits**:
- ✅ Explicit batching control
- ✅ Buffer IDs prevent accidental reuse
- ✅ Clear execution point
- ✅ Can optimize buffer allocation
- ✅ Easy to implement incrementally

### Design 2: Futures-based API (Alternative)

```rust
pub struct GpuTensor {
    device: Arc<GpuDevice>,
    buffer: wgpu::Buffer,
    shape: Vec<usize>,
}

impl GpuTensor {
    pub async fn relu(&self) -> Result<Self, String> { ... }
    pub async fn scale(&self, scalar: f32) -> Result<Self, String> { ... }
    pub async fn add(&self, other: &Self) -> Result<Self, String> { ... }

    // Each returns future, can be chained
    pub async fn to_vec(&self) -> Result<Vec<f32>, String> { ... }
}

// Usage (with async/await)
let input = GpuTensor::from_vec(&device, vec![1.0, 2.0, 3.0]).await?;
let result = input
    .relu().await?
    .scale(2.0).await?
    .add(&other).await?
    .to_vec().await?;
```

**Issues**:
- ❌ Still executes each operation immediately
- ❌ Doesn't batch unless we add complex future combinators
- ❌ Harder to optimize buffer reuse

## Recommendation: Design 1 (Command Builder)

**Phase 1** (This implementation):
1. Create `GpuCommandBatch` struct
2. Implement buffer management (upload, allocate, read)
3. Add 3-5 operations (relu, scale, add, mul, dot)
4. Write tests validating 2x transfer reduction
5. Benchmark performance improvement

**Phase 2** (Future):
1. Add remaining operations (sigmoid, tanh, gelu, etc.)
2. Optimize buffer reuse (arena allocator)
3. Add compute pipeline caching
4. Implement operation fusion (relu+scale → single kernel)

## Implementation Plan

### Step 1: Core Infrastructure
- [ ] Create `src/backends/gpu/batch.rs`
- [ ] Define `GpuCommandBatch` struct
- [ ] Implement buffer ID management
- [ ] Add upload/read operations

### Step 2: Operation Support
- [ ] Add `relu` operation
- [ ] Add `scale` operation
- [ ] Add `add` operation
- [ ] Add `execute()` method

### Step 3: Testing & Validation
- [ ] Unit tests for each operation
- [ ] Integration test: chained operations
- [ ] Benchmark: compare vs synchronous API
- [ ] Validate 2x reduction in transfers

### Step 4: Documentation
- [ ] API documentation
- [ ] Usage examples
- [ ] Migration guide from sync API

## Success Criteria (v0.3.0)

✅ **2x fewer GPU transfers for chained operations**
- Before: `relu + scale + add` = 6 transfers (3 up, 3 down)
- After: 2 transfers (1 up, 1 down)

✅ **Performance improvement on chained operations**
- Target: ≥30% faster for 3+ operation chains
- Measure: End-to-end latency including transfers

✅ **Backward compatible**
- Keep existing synchronous API
- Add new async API alongside

## Open Questions

1. **Buffer lifetime management**: Who owns buffers? Arc? Rc?
2. **Error handling**: Fail fast or collect errors?
3. **Async runtime**: Require tokio or stay runtime-agnostic?
4. **Operator fusion**: Worth implementing in v1?

## References

- PyTorch JIT fusion: https://pytorch.org/docs/stable/jit.html
- TensorFlow XLA: https://www.tensorflow.org/xla
- wgpu compute examples: https://github.com/gfx-rs/wgpu/tree/trunk/examples
