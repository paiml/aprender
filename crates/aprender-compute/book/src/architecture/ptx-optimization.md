# PTX Optimization Passes

This chapter documents the PTX optimization passes in `trueno-gpu`, aligned with NVIDIA's official CUDA Tile IR (CUDA Toolkit 13.1).

## Overview

The `trueno_gpu::ptx::optimize` module provides four optimization passes:

| Pass | Description | Benefit |
|------|-------------|---------|
| **FMA Fusion** | `mul + add` → `fma` | Reduced latency, single rounding |
| **Loop Splitting** | Conditional loop splitting | Eliminates branch divergence |
| **Token-Based Ordering** | Memory dependency tracking | Barrier elimination |
| **Tile Validation** | Power-of-two constraints | Prevents register pressure |

## FMA Fusion Pass

The FMA (Fused Multiply-Add) fusion pass detects `mul` + `add` instruction patterns and fuses them into a single `fma` instruction.

### Benefits

- **Latency**: Single instruction instead of two
- **Precision**: Single rounding operation (IEEE 754 compliant)
- **Throughput**: Utilizes GPU FMA units efficiently

### Example

```rust
use trueno_gpu::ptx::optimize::fma_fusion;
use trueno_gpu::ptx::{Operand, PtxInstruction, PtxOp, PtxType, VirtualReg};

// Create mul + add pattern
let r0 = VirtualReg::new(0, PtxType::F32);
let r1 = VirtualReg::new(1, PtxType::F32);
let r2 = VirtualReg::new(2, PtxType::F32);
let r3 = VirtualReg::new(3, PtxType::F32);

let mul = PtxInstruction::new(PtxOp::Mul, PtxType::F32)
    .dst(Operand::Reg(r2.clone()))
    .src(Operand::Reg(r0.clone()))
    .src(Operand::Reg(r1.clone()));

let add = PtxInstruction::new(PtxOp::Add, PtxType::F32)
    .dst(Operand::Reg(r3))
    .src(Operand::Reg(r2))
    .src(Operand::ImmF32(1.0));

// Fuse to single FMA instruction
let fused = fma_fusion::pass(vec![mul, add]);
assert_eq!(fused.len(), 1); // mul + add → fma
```

### Academic Reference

Based on Click & Paleczny (1995) "A Simple Graph-Based Intermediate Representation" for SSA pattern matching.

## Loop Splitting Pass

The loop splitting pass analyzes conditional loops and identifies opportunities to split them at condition boundaries, eliminating branch divergence in GPU warps.

### Heavy Operations

The following operations trigger split profitability:

- `Ld` - Memory loads
- `St` - Memory stores
- `WmmaMma` - Tensor Core MMA
- `WmmaLoadA`, `WmmaLoadB`, `WmmaLoadC` - WMMA fragment loads
- `WmmaStoreD` - WMMA fragment stores

### Example

```rust
use trueno_gpu::ptx::optimize::loop_split;
use trueno_gpu::ptx::{PtxInstruction, PtxOp, PtxType, CmpOp};

// Check profitability
let heavy_op = PtxInstruction::new(PtxOp::Ld, PtxType::F32);
assert!(loop_split::is_split_profitable(&[heavy_op], 10));

let light_op = PtxInstruction::new(PtxOp::Add, PtxType::F32);
assert!(!loop_split::is_split_profitable(&[light_op], 10));

// Split point alignment for non-unit steps
assert_eq!(loop_split::align_split_point(5, 0, 4), 8);
assert_eq!(loop_split::align_split_point(8, 0, 4), 8);

// Loop predicate conversion
assert_eq!(
    loop_split::LoopPredicate::from_cmp_op(CmpOp::Lt),
    Some(loop_split::LoopPredicate::LessThan)
);
```

### NVIDIA Reference

Aligned with `LoopSplit.cpp` from NVIDIA CUDA Tile IR (CUDA Toolkit 13.1).

## Token-Based Ordering (TKO)

Token-Based Ordering provides explicit memory dependency tracking, enabling compiler-driven barrier elimination.

### Memory Ordering Semantics

| Ordering | PTX Modifier | Description |
|----------|--------------|-------------|
| `Weak` | `.weak` | No ordering guarantees |
| `Relaxed` | `.relaxed` | Relaxed consistency |
| `Acquire` | `.acquire` | Acquire semantics |
| `Release` | `.release` | Release semantics |

### Memory Scopes

| Scope | PTX Modifier | Description |
|-------|--------------|-------------|
| `Thread` | `.cta` | Thread-local |
| `Block` | `.cta` | Block-local |
| `Cluster` | `.cluster` | Cluster-local |
| `Device` | `.gpu` | Device-wide |
| `System` | `.sys` | System-wide |

### Example

```rust
use trueno_gpu::ptx::optimize::tko;

// Create tokens for memory operations
let t1 = tko::Token::new();
let t2 = tko::Token::new();
let t3 = tko::Token::new();

// Join tokens at synchronization point
let joined = tko::join_tokens(&[t1, t2, t3]);

// Memory ordering
let ordering = tko::MemoryOrdering::Acquire;
assert_eq!(ordering.to_ptx_modifier(), ".acquire");

// Memory scope
let scope = tko::MemoryScope::Device;
assert_eq!(scope.to_ptx_scope(), ".gpu");

// Token graph with cycle detection
let mut graph = tko::TokenGraph::new();
let ta = tko::Token::new();
let tb = tko::Token::new();
let tc = tko::Token::new();

graph.create_token(ta);
graph.create_token(tb);
graph.create_token(tc);
graph.add_dependency(tb, ta);
graph.add_dependency(tc, tb);

assert!(!graph.has_cycle()); // No deadlock

graph.add_dependency(ta, tc);
assert!(graph.has_cycle()); // DEADLOCK!
```

### NVIDIA Reference

Aligned with `memory_consistency_ops.mlir` from NVIDIA CUDA Tile IR.

## Tile Validation

Tile validation enforces constraints to prevent register pressure issues and compilation hangs.

### Constraints

1. **Power-of-two dimensions**: Required for efficient GPU scheduling
2. **Maximum tile elements**: 16M elements to prevent register spills
3. **Maximum single dimension**: 4096 to prevent degenerate shapes

### WMMA Valid Shapes

| Shape | Description |
|-------|-------------|
| `M16N16K16` | Standard 16×16×16 |
| `M8N32K16` | Alternate 8×32×16 |
| `M32N8K16` | Alternate 32×8×16 |

### Example

```rust
use trueno_gpu::ptx::optimize::tile_validation;
use trueno_gpu::ptx::WmmaShape;

// Valid shapes
assert!(tile_validation::validate_shape(&[16, 16]).is_ok());
assert!(tile_validation::validate_shape(&[32, 32]).is_ok());
assert!(tile_validation::validate_shape(&[64, 64]).is_ok());

// Invalid shapes
assert!(tile_validation::validate_shape(&[17, 16]).is_err()); // Not power of two
assert!(tile_validation::validate_shape(&[100, 100]).is_err());

// WMMA shapes
let valid_wmma = WmmaShape::M16N16K16;
assert!(tile_validation::validate_wmma_shape(&valid_wmma).is_ok());

let invalid_wmma = WmmaShape { m: 24, n: 24, k: 16 };
assert!(tile_validation::validate_wmma_shape(&invalid_wmma).is_err());
```

### Academic Reference

Based on Volkov & Demmel (2008) "Benchmarking GPUs to Tune Dense Linear Algebra".

## Running the Example

```bash
cargo run --example ptx_optimize
```

Output:

```
╔══════════════════════════════════════════════════════════════╗
║     PTX Optimization Passes (NVIDIA CUDA Tile IR Aligned)    ║
╚══════════════════════════════════════════════════════════════╝

1️⃣  FMA FUSION PASS
   Input:  2 instructions (mul + add)
   Output: 1 instruction (fma)

2️⃣  LOOP SPLITTING PASS
   Heavy ops trigger split: true
   Light ops trigger split: false

3️⃣  TOKEN-BASED ORDERING (TKO)
   Tokens created with unique IDs
   Cycle detection: working

4️⃣  TILE VALIDATION
   Power-of-two shapes: OK
   Invalid shapes: rejected

✅ All optimization demos completed successfully!
```

## Specification

Full specification: [cuda-tile-behavior.md](../../../docs/specifications/cuda-tile-behavior.md) (v1.4.0)

## Coverage

| Module | Coverage |
|--------|----------|
| `fma_fusion.rs` | 93.75% |
| `loop_split.rs` | 99.80% |
| `tko/mod.rs` | 94.29% |
| `tile_validation.rs` | 88.64% |
| **Total** | **94.28%** |
