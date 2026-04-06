# Example 2: C Library Migration

This walkthrough demonstrates transpiling a C numerical library into safe Rust
using `decy`, the C-to-Rust transpiler in the Sovereign AI Stack.

## Scenario

A team maintains `libvecmath`, a C99 numerical library providing vector
operations, matrix decomposition, and statistical functions. The library is
mature (10 years old, 8,000 lines) but suffers from periodic buffer overflows
reported through fuzzing. The goal is a memory-safe Rust port that preserves
the existing C API for downstream consumers during the transition.

## Source Project Layout

```
libvecmath/
  include/vecmath.h      # Public API (42 functions)
  src/vector.c           # Vector operations
  src/matrix.c           # Matrix operations
  src/stats.c            # Statistical functions
  src/alloc.c            # Custom allocator
  tests/test_suite.c     # CUnit test suite
  Makefile
```

## Step 1 -- Analyze

```bash
batuta analyze --languages --tdg ./libvecmath
```

```
Languages detected: C (95%), Shell (5%)
Functions: 42 public, 18 internal
Unsafe patterns: 23 raw pointer dereferences, 8 manual malloc/free pairs
TDG Score: C (58/100) — memory management complexity
```

Batuta flags every `malloc`/`free` pair, every raw pointer dereference, and
every buffer access without bounds checking. These become the primary targets
for safe Rust translation.

## Step 2 -- Transpile

```bash
batuta transpile ./libvecmath --tool decy --output ./vecmath_rs
```

Decy performs three sub-passes:

1. **Ownership inference**: Determines which pointers are owned, borrowed, or
   shared based on usage patterns (see [Ownership Inference](./c-ownership.md)).
2. **Memory translation**: Converts `malloc`/`free` to Rust ownership, arrays
   to `Vec<T>` or slices (see [Memory Management](./c-memory.md)).
3. **FFI boundary generation**: Creates safe wrappers for functions that must
   remain callable from C (see [FFI Boundaries](./c-ffi.md)).

## Step 3 -- Optimize

```bash
batuta optimize ./vecmath_rs --backend auto
```

Vector operations map to `trueno` SIMD kernels. The optimizer replaces
hand-written SIMD intrinsics in the original C with trueno's portable
abstractions that dispatch to AVX2, AVX-512, or NEON at runtime.

## Step 4 -- Validate

```bash
batuta validate ./vecmath_rs --reference ./libvecmath
```

Batuta compiles and runs both the C and Rust test suites, comparing numerical
outputs within tolerance. Syscall traces confirm identical file and network I/O
patterns.

## Step 5 -- Build

```bash
batuta build ./vecmath_rs --release
```

The output is a Rust crate with optional `cdylib` target for C consumers. The
Rust library can be used natively from Rust projects or linked as a drop-in
replacement for the original `.so`/`.a`.

## Result

| Metric          | C (libvecmath) | Rust (vecmath_rs) |
|-----------------|----------------|-------------------|
| Buffer overflows| 3 known CVEs   | 0 (by design)     |
| Test coverage   | 72%            | 96%               |
| Performance     | Baseline       | 1.05x (SIMD)      |
| Binary size     | 48 KB          | 52 KB              |

## Key Takeaways

- Decy infers Rust ownership from C usage patterns, converting the majority of
  pointer operations to safe references automatically.
- The FFI boundary layer lets C consumers link against the new Rust library
  without source changes, enabling gradual adoption.
- Buffer overflows are eliminated structurally by replacing raw pointer
  arithmetic with bounds-checked slices.
- The following sub-chapters detail each aspect: memory management, ownership
  inference, and FFI boundary design.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
