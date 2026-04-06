# System Overview

Trueno-DB is a GPU-accelerated analytical database engine designed for high-performance aggregations and joins on large datasets.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Query Interface                       │
│  - SQL parsing (future)                                 │
│  - Query AST generation                                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│        JIT WGSL Compiler (CORE-003)                     │
│  - AST → WGSL shader generation                         │
│  - Kernel fusion                                        │
│  - Shader cache                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│     Cost-Based Backend Dispatcher (CORE-002) ✅         │
│  - Minimum data size: 10 MB                             │
│  - 5x rule: GPU if compute > 5x transfer                │
│  - PCIe Gen4 x16: 32 GB/s                               │
└─────────────────────────────────────────────────────────┘
           │                  │                  │
     ┌─────┴─────┐      ┌─────┴─────┐      ┌────┴─────┐
     │    GPU    │      │   SIMD    │      │  Scalar  │
     │  (CORE-004)      │ (CORE-005)       │          │
     │  wgpu     │      │  trueno   │      │  naive   │
     └───────────┘      └───────────┘      └──────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │ Arrow Storage Backend ✅      │
           │ (CORE-001)                   │
           │  - Parquet reader            │
           │  - Morsel iterator (128MB)   │
           │  - GPU transfer queue        │
           └──────────────────────────────┘
```

## Components

### Storage Layer (CORE-001) ✅

Handles data loading and morsel-based iteration.

**Key features**:
- Apache Arrow/Parquet integration
- Streaming RecordBatch reading
- 128MB morsel chunks (Poka-Yoke)
- Bounded GPU transfer queue (Heijunka)

See: [Arrow Storage Backend](../components/storage/arrow-backend.md)

### Backend Dispatcher (CORE-002) ✅

Automatically selects optimal execution backend.

**Selection algorithm**:
1. Data size ≥ 10 MB? → Continue
2. Compute > 5x transfer? → GPU
3. Otherwise → SIMD

See: [Cost-Based Backend Selection](./cost-based-backend.md)

### Query Engine (CORE-003) 🚧

JIT compiles WGSL shaders from query AST.

**Features**:
- Kernel fusion (combine operators)
- Shader cache
- Operator variants (GPU/SIMD/Scalar)

### GPU Kernels (CORE-004) 🚧

WGSL compute shaders for parallel operations.

**Operators**:
- Parallel reduction (sum, avg, count, min, max)
- Radix hash join
- Filter/projection

### SIMD Fallback (CORE-005) 🚧

CPU execution using trueno library.

**Features**:
- AVX2/AVX-512 SIMD
- spawn_blocking isolation
- Async interface

## Data Flow

1. **Load**: Parquet file → Arrow RecordBatch
2. **Morsel**: Split into 128MB chunks
3. **Dispatch**: Select backend (GPU/SIMD/Scalar)
4. **Execute**: Run query on selected backend
5. **Return**: Collect results

## Next Steps

- [Design Principles](./design-principles.md)
- [Out-of-Core Execution](./out-of-core-execution.md)
- [Heterogeneous Computing](./heterogeneous-computing.md)
