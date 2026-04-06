# Compiler-in-the-Loop (CITL)

This example demonstrates compiler-integrated training optimization.

## Running the Example

```bash
cargo run --example citl
```

## Code

```rust
{{#include ../../../examples/citl.rs}}
```

## CITL Concept

Compiler-in-the-Loop training uses compiler feedback to optimize:

1. **Graph optimization** - Fuse operations, eliminate redundancy
2. **Memory planning** - Optimal tensor allocation
3. **Kernel selection** - Choose best implementation per operation
4. **Quantization hints** - Identify quantization opportunities

## Pipeline

```
Training Loop
     │
     ▼
┌─────────────┐
│   Compiler  │ ◄── Profile data
│   Analysis  │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Optimization│
│   Hints     │
└─────────────┘
     │
     ▼
Next Training Iteration
```

## Benefits

| Optimization | Speedup |
|--------------|---------|
| Op fusion | 1.2-1.5x |
| Memory reuse | 20-30% less |
| Kernel tuning | 1.1-1.3x |
