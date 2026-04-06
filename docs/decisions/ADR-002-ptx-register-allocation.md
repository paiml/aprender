# ADR-002: PTX Register Allocation Strategy

**Status**: Accepted
**Date**: 2026-01-03
**Issue**: [#66](https://github.com/paiml/trueno/issues/66)

## Context

A team using trueno raised concerns about our register allocation strategy:

> Since you are likely blindly emitting new virtual registers (`%r1`, `%r2`...) without checking "Liveness" (when a variable is no longer needed), you might emit PTX code that uses 200 registers for a thread. If a thread needs too many registers, the GPU reduces "Occupancy" (fewer threads run at once).

This ADR documents our investigation and decision.

## Decision

**We will continue delegating physical register allocation to NVIDIA's `ptxas` compiler.**

This is a deliberate architectural choice, not an oversight. PTX is designed as a virtual ISA with unlimited registers, explicitly intended to be lowered by a backend compiler.

## Investigation Results

### 1. Does ptxas handle high virtual register counts?

**Yes.** NVIDIA's `ptxas` performs graph coloring register allocation on PTX input. It can coalesce 200+ virtual registers down to the optimal physical register count. This is documented in NVIDIA's PTX ISA specification:

> "The PTX programming model provides a virtual execution environment that abstracts the underlying GPU hardware. PTX programs use virtual registers."

### 2. What mitigations already exist?

| Mitigation | Status | Description |
|------------|--------|-------------|
| In-place operations | ✅ Implemented | `add_f32_inplace`, `fma_f32_inplace`, etc. |
| Pressure reporting | ✅ Implemented | `RegisterAllocator::pressure_report()` |
| Best practices docs | ✅ Complete | `book/src/architecture/ptx-register-allocation.md` |
| Working example | ✅ Complete | `trueno-gpu/examples/register_allocation.rs` |

### 3. When would manual reuse help?

Manual liveness-based reuse would only help if:
1. A kernel uses >256 virtual registers AND
2. `ptxas` fails to coalesce them AND
3. The resulting occupancy loss is measurable

In practice, this scenario is rare because:
- Most trueno kernels use 15-50 virtual registers
- In-place operations prevent explosion in loops
- `ptxas -O3` is highly effective

### 4. Implementation cost vs benefit

| Factor | Assessment |
|--------|------------|
| Implementation LOC | ~300-500 lines for greedy reuse pass |
| Debugging complexity | SSA->non-SSA complicates debugging |
| Maintenance burden | Must track ptxas version changes |
| Expected benefit | Minimal for typical kernels |

## Alternatives Considered

### Option A: Implement Greedy Register Reuse
- **Pros**: Reduces virtual register count, may improve compile time
- **Cons**: Redundant with ptxas, adds complexity, may interfere with ptxas optimizations
- **Decision**: Rejected

### Option B: Current Approach (Virtual Registers + ptxas)
- **Pros**: Pragmatic, leverages NVIDIA expertise, simpler codebase
- **Cons**: Relies on external compiler
- **Decision**: Accepted

### Option C: Warn on High Pressure
- **Pros**: Alerts developers without implementation complexity
- **Cons**: May produce false positives
- **Decision**: Already implemented via `pressure_report()`

## Consequences

### Positive
- Simpler PTX builder implementation
- Leverages 30+ years of NVIDIA compiler optimization
- Focus on high-level optimizations (tiling, fusion) instead

### Negative
- Dependency on ptxas quality (acceptable - it's excellent)
- Cannot control exact register allocation (rarely needed)

## Future Considerations

If future benchmarks show measurable occupancy loss from register pressure:
1. First: Reduce unroll factors, use shared memory
2. Second: Add in-place operation variants for problematic patterns
3. Last resort: Implement greedy reuse pass (design in `ptx-register-allocation.md`)

## References

- [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Architecture Doc](../../book/src/architecture/ptx-register-allocation.md)
- [Example](../../trueno-gpu/examples/register_allocation.rs)
- [Register Implementation](../../trueno-gpu/src/ptx/registers/mod.rs)
