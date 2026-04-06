# PTX Register Allocation Architecture

This chapter explains trueno-gpu's approach to register allocation, which delegates physical register assignment to NVIDIA's `ptxas` compiler. This is a pragmatic design that leverages 30+ years of GPU compiler optimization.

## The Traditional Compiler Problem

In traditional compilers (like LLVM for x86), you must map an infinite number of variables to a finite set of physical registers (e.g., `RAX`, `RDI`, `RSI` on x86-64). This requires complex algorithms:

- **Graph Coloring**: Model register interference as a graph, color with K colors (K = number of physical registers)
- **Linear Scan**: Faster but less optimal allocation for JIT compilers

These algorithms are complex to implement correctly and require significant engineering effort.

## Trueno's Strategy: Virtual Registers + ptxas

Trueno takes a different approach that leverages PTX's design as a **virtual ISA**:

```text
┌─────────────────────────────────────────────────────────────┐
│  Trueno PTX Builder (Rust)                                  │
│  - Allocates unlimited virtual registers (%f0, %f1, ...)    │
│  - Tracks liveness for pressure REPORTING                   │
│  - Emits SSA-style PTX                                      │
└─────────────────────────────────────────────────────────────┘
                             │
                        PTX Source
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  NVIDIA ptxas (JIT Compiler)                                │
│  - Graph coloring for physical register allocation          │
│  - Register spilling to local memory if needed              │
│  - Dead code elimination, constant folding, etc.            │
└─────────────────────────────────────────────────────────────┘
                             │
                        SASS Binary
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  GPU Execution                                              │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Virtual Register Allocation**: Each operation allocates a new virtual register with a monotonically increasing ID:

```rust
// In trueno-gpu's KernelBuilder
pub fn add_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
    // Allocate NEW virtual register (SSA style)
    let dst = self.registers.allocate_virtual(PtxType::F32);
    self.instructions.push(
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::Reg(dst))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b))
    );
    dst  // Return %f2, %f3, %f4, etc.
}
```

2. **Per-Type Namespaces**: PTX requires separate register namespaces per type:

| Type | Prefix | Example |
|------|--------|---------|
| `.f32` | `%f` | `%f0`, `%f1`, `%f2` |
| `.f64` | `%fd` | `%fd0`, `%fd1` |
| `.u32` | `%r` | `%r0`, `%r1` |
| `.u64` | `%rd` | `%rd0`, `%rd1` |
| `.pred` | `%p` | `%p0`, `%p1` |

3. **Emitted PTX**: The builder emits register declarations and instructions:

```ptx
.visible .entry vector_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
) {
    .reg .f32  %f<3>;    // Virtual registers %f0, %f1, %f2
    .reg .u32  %r<5>;    // Virtual registers %r0-4
    .reg .u64  %rd<7>;   // Virtual registers %rd0-6
    .reg .pred %p<1>;    // Predicate register %p0

    // Instructions use virtual registers
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;
    // ...
    add.rn.f32 %f2, %f0, %f1;
    // ...
}
```

4. **ptxas Does the Rest**: NVIDIA's `ptxas` compiler:
   - Builds an interference graph from virtual register liveness
   - Performs graph coloring to assign physical registers
   - Generates spill code if necessary (to `.local` memory)
   - Applies optimization passes

## Why This Design?

### 1. Pragmatism (Avoid Muda)

NVIDIA has invested 30+ years into GPU compiler optimization. Reimplementing graph coloring would be:
- Redundant (ptxas already does it)
- Inferior (we can't match NVIDIA's GPU-specific knowledge)
- Wasteful engineering effort (Muda in Toyota terms)

### 2. PTX is Designed for This

PTX (Parallel Thread Execution) is explicitly designed as a **virtual ISA**:
- Unlimited virtual registers
- SSA (Static Single Assignment) form
- Meant to be lowered by a backend compiler

From the [PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/):
> "PTX defines a virtual machine and ISA for general purpose parallel thread execution."

### 3. Focus on What Matters

Trueno focuses on:
- **Algorithm correctness**: Ensuring SIMD/GPU operations produce correct results
- **High-level optimization**: Tiling, kernel fusion, memory access patterns
- **Developer experience**: Safe, ergonomic Rust API

Low-level optimization (register allocation, instruction scheduling) is delegated to specialized tools.

## Register Pressure Monitoring

While we don't perform graph coloring, we DO track liveness for **diagnostics**:

```rust
pub struct RegisterAllocator {
    type_counters: HashMap<PtxType, u32>,
    live_ranges: HashMap<(PtxType, u32), LiveRange>,
    spill_count: usize,  // Muda tracking
}

impl RegisterAllocator {
    pub fn pressure_report(&self) -> RegisterPressure {
        RegisterPressure {
            max_live: self.allocated.len(),
            spill_count: self.spill_count,
            utilization: max_live as f64 / 256.0,
        }
    }
}
```

### Why Track Pressure?

1. **Developer Warnings**: Alert when kernels exceed 256 registers/thread
2. **Occupancy Estimation**: High register usage reduces concurrent threads
3. **Performance Debugging**: Identify kernels that may suffer from register spills

### GPU Register Limits

| Architecture | Registers/Thread | Registers/SM |
|--------------|------------------|--------------|
| Volta (sm_70) | 256 | 65,536 |
| Turing (sm_75) | 256 | 65,536 |
| Ampere (sm_80) | 256 | 65,536 |
| Ada (sm_89) | 256 | 65,536 |

**Occupancy Impact**: If a kernel uses 64 registers/thread, an SM with 65,536 registers can run 1024 threads. If it uses 128 registers/thread, only 512 threads can run concurrently.

## In-Place Operations for Register Reuse

For loops and accumulators, SSA-style allocation wastes registers:

```rust
// SSA style - allocates new register each iteration
for _ in 0..1000 {
    let new_sum = ctx.add_f32(sum, val);  // New register each time!
    sum = new_sum;
}
```

We provide **in-place operations** that reuse registers:

```rust
// In-place style - reuses existing register
let acc = ctx.mov_f32_imm(0.0);  // Allocate once
for _ in 0..1000 {
    ctx.add_f32_inplace(acc, val);  // Reuses %f0
}
```

### Available In-Place Operations

| Operation | Use Case |
|-----------|----------|
| `add_u32_inplace(dst, imm)` | Loop counters |
| `add_f32_inplace(dst, src)` | Accumulators |
| `fma_f32_inplace(dst, a, b)` | GEMM accumulation |
| `max_f32_inplace(dst, src)` | Online softmax |
| `mul_f32_inplace(dst, src)` | Scaling |
| `div_f32_inplace(dst, src)` | Normalization |
| `shr_u32_inplace(dst, imm)` | Stride halving |

## Potential Future Enhancements

The current design delegates all register allocation to ptxas. Potential future enhancements (tracked in [GitHub Issue #66](https://github.com/paiml/trueno/issues/66)):

### 1. Greedy Register Reuse

For kernels exceeding 256 registers, we could implement simple liveness-based reuse:

```rust
// Hypothetical future API
let allocator = RegisterAllocator::new()
    .with_reuse_strategy(ReuseStrategy::Greedy);
```

This would reuse `%r2` after its last use, reducing virtual register count.

### 2. ptxas Output Parsing

Parse `cuobjdump --dump-resource-usage` output to validate:
- Expected vs actual register usage
- Spill detection
- Occupancy calculation

### 3. Occupancy Calculator

Integrate NVIDIA's occupancy calculator to predict SM utilization before runtime.

## Best Practices

### 1. Use In-Place Operations for Loops

```rust
// Good - register reuse
let i = ctx.mov_u32_imm(0);
ctx.label("loop");
// ... loop body ...
ctx.add_u32_inplace(i, 1);  // Reuses %r0
ctx.branch("loop");

// Bad - register explosion
let mut i = ctx.mov_u32_imm(0);
ctx.label("loop");
// ... loop body ...
i = ctx.add_u32(i, 1);  // New register each iteration!
ctx.branch("loop");
```

### 2. Limit Unroll Factors

Each unrolled iteration adds registers. Balance throughput vs pressure:

```rust
// High pressure - 8x unroll
for i in 0..8 {
    let val = ctx.ld_global_f32(addr[i]);
    ctx.fma_f32_inplace(acc, val, weights[i]);
}

// Lower pressure - 4x unroll (often sufficient)
for i in 0..4 {
    let val = ctx.ld_global_f32(addr[i]);
    ctx.fma_f32_inplace(acc, val, weights[i]);
}
```

### 3. Use Shared Memory for Large Temporaries

Instead of keeping many values in registers, stage through shared memory:

```rust
// Use shared memory tile instead of many registers
let tile = ctx.alloc_shared::<f32>(TILE_SIZE * TILE_SIZE);
```

### 4. Monitor Kernel Complexity

For complex kernels, check register pressure:

```rust
let pressure = kernel.registers.pressure_report();
if pressure.utilization > 0.5 {
    eprintln!("Warning: High register pressure ({:.0}%)",
              pressure.utilization * 100.0);
}
```

## Running the Example

```bash
cargo run -p trueno-gpu --example register_allocation
```

This demonstrates:
1. Simple kernel with low register pressure
2. Complex kernel with higher pressure (unrolled dot product)
3. In-place operations for register reuse
4. Architectural trade-offs

## References

- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Occupancy Calculator](https://developer.nvidia.com/cuda-occupancy-calculator)
- [GitHub Issue #66: Liveness-Based Register Reuse](https://github.com/paiml/trueno/issues/66)
- Example: `trueno-gpu/examples/register_allocation.rs`
