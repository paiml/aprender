# The Orchestration Paradigm

> **"A single instrument cannot play a symphony. Neither can a single transpiler migrate a complex codebase."**

## The Problem with Simple Transpilation

Traditional transpilers make a fundamental mistake: they treat code migration as a **one-step translation problem**. This is like trying to move a house by picking it up and dropping it in a new location. It might work for a shed, but not for complex structures.

### Why Simple Transpilation Fails

**1. Loss of Semantic Meaning**

```python
# Python
x = [1, 2, 3]
y = x
y.append(4)
# x is now [1, 2, 3, 4] - shared reference
```

Simple transpilation to Rust:
```rust
// Naive transpilation
let mut x = vec![1, 2, 3];
let mut y = x;  // ❌ Moved! x is now invalid
y.push(4);
```

**Correct Batuta approach** (via Depyler):
```rust
// Semantic preservation
let mut x = vec![1, 2, 3];
let y = &mut x;  // ✓ Shared mutable reference
y.push(4);
// x is [1, 2, 3, 4] - semantics preserved
```

**2. Missing Optimizations**

Simple transpilers translate code literally. Batuta recognizes opportunities:

```python
# Python - CPU only
import numpy as np
result = np.dot(large_matrix_a, large_matrix_b)
```

**Batuta orchestration** (Depyler + Trueno):
```rust
// Automatic SIMD/GPU dispatch
use trueno::linalg::dot;
let result = dot(&matrix_a, &matrix_b)?;
// ✓ Dispatches to GPU if matrices > threshold
// ✓ Falls back to SIMD for smaller operations
```

**3. No Validation**

How do you know the transpiled code is correct? Simple transpilers say "it compiles, ship it!" Batuta says "prove it with syscall tracing, test execution, and benchmarks."

## The Orchestra Metaphor

Consider a symphony orchestra:

- **Conductor (Batuta)**: Coordinates all musicians, maintains tempo, ensures harmony
- **String Section (Transpilers)**: Decy, Depyler, Bashrs convert code to Rust
- **Brass Section (Foundation Libraries)**: Trueno, Aprender, Realizar provide runtime capabilities
- **Percussion (Support Tools)**: Ruchy, PMAT, Renacer provide quality and validation

Each instrument is virtuoso in its domain. But without coordination, you get noise, not music.

### The Conductor's Role

**Batuta coordinates:**

1. **Timing**: When to invoke which tool (5-phase workflow)
2. **Communication**: How tools share outputs (IR, AST, config)
3. **Quality**: Validation at each phase boundary
4. **Optimization**: Automatic selection of best tool for task

## Orchestration vs. Monolithic Tools

| Aspect | Monolithic Transpiler | Batuta Orchestration |
|--------|----------------------|---------------------|
| **Scope** | Single-language focus | Multi-language support |
| **Optimization** | Basic or none | Automatic SIMD/GPU |
| **Validation** | "It compiles" | Syscall tracing + tests |
| **ML Support** | External libraries | Native (Aprender/Realizar) |
| **Gradual Migration** | All-or-nothing | Ruchy scripting support |
| **Quality Metrics** | None | PMAT TDG scoring |
| **Workflow** | Linear | 5-phase Kanban |

## Core Principles

### 1. **Specialization**

Each tool excels at ONE thing:
- Decy: C/C++ ownership inference
- Trueno: Multi-backend compute dispatch
- Renacer: Syscall-level validation

**Do NOT try to make Depyler handle C code.** Use the right tool for the job.

### 2. **Composition**

Tools are composable building blocks:

```
Python + NumPy  →  Depyler + Trueno  →  Rust + SIMD/GPU
Python + sklearn → Depyler + Aprender → Rust + ML primitives
```

### 3. **State Management**

Orchestration requires tracking:
- Which phase are we in?
- What completed successfully?
- What failed and why?
- What's next?

This is why Batuta has a **workflow state machine** (`.batuta-state.json`).

### 4. **Incremental Progress**

Unlike monolithic transpilers, orchestration supports:
- Partial completion (Phase 1-2 done, 3-5 pending)
- Resume after errors
- Selective re-execution
- Caching of completed work

## Real-World Example

Consider migrating a Python ML web service:

```
project/
├── api.py            # Flask web server
├── model.py          # ML inference
├── preprocessing.py  # NumPy data transforms
├── utils.sh          # Deployment scripts
└── requirements.txt
```

### Monolithic Approach

```bash
# Try to transpile everything with one tool
some-transpiler --input project/ --output rust-project/
# ❌ Fails because:
# - Shell scripts not supported
# - NumPy performance poor
# - No validation of ML accuracy
# - No optimization
```

### Batuta Orchestration

```bash
# Phase 1: Analysis
batuta analyze --languages --dependencies --tdg
# ✓ Detects: Python (80%), Shell (20%)
# ✓ Identifies: Flask, NumPy, sklearn
# ✓ TDG Score: 73/100 (B)

# Phase 2: Transpilation
batuta transpile
# ✓ Depyler: api.py, model.py, preprocessing.py → Rust
# ✓ Bashrs: utils.sh → Rust CLI
# ✓ NumPy → Trueno: Automatic mapping
# ✓ sklearn → Aprender: Model conversion

# Phase 3: Optimization
batuta optimize --enable-gpu
# ✓ Trueno: SIMD for small matrices
# ✓ Trueno: GPU dispatch for large batch inference
# ✓ Memory layout optimization

# Phase 4: Validation
batuta validate --trace-syscalls --benchmark
# ✓ Renacer: Syscall equivalence check
# ✓ API tests: All passing
# ✓ Performance: 12x faster, 60% less memory

# Phase 5: Deployment
batuta build --release
# ✓ Optimized binary: 8MB (vs 200MB Python + deps)
# ✓ No interpreter, no GC pauses
```

## When NOT to Use Orchestration

Orchestration has overhead. Don't use Batuta if:

1. **Single file, simple logic**: Just hand-write Rust
2. **Already have Rust version**: You're done!
3. **Prototype/throwaway code**: Not worth the effort
4. **Actively changing code**: Finish development first

**Use Batuta when:**
- Multiple languages/files
- Complex dependencies
- Performance critical
- Need validation
- Long-term maintenance
- Team knowledge transfer

## Key Takeaways

**Orchestration is:**
- ✓ Systematic and repeatable
- ✓ Tool-agnostic (uses best tool for each task)
- ✓ Validatable at each step
- ✓ Optimizable automatically
- ✓ Recoverable from failures

**Orchestration is NOT:**
- ✗ Magic (it's systematic process)
- ✗ Perfect (tools have limitations)
- ✗ Instant (phases take time)
- ✗ Suitable for all projects

## Next Steps

Now that you understand the orchestration paradigm, let's explore how it embodies **Toyota Way principles** - the manufacturing philosophy that makes systematic code migration possible.

---

**Previous:** [Introduction](../introduction.md)
**Next:** [Toyota Way Principles](./toyota-way.md)
