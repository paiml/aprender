# PTX Bug Detection

The `trueno-explain` crate provides static analysis for PTX (NVIDIA GPU assembly) to detect common bugs and performance issues before runtime.

## Overview

Hand-written PTX is error-prone. The PTX bug analyzer catches:

| Severity | Bug Class | Description |
|----------|-----------|-------------|
| P0 Critical | `SHARED_MEM_U64` | 64-bit addressing for shared memory (undefined behavior) |
| P0 Critical | `MISSING_BARRIER` | Missing `bar.sync` between shared memory operations |
| P0 Critical | `LOOP_BRANCH_END` | Unconditional branch to loop end (infinite loop) |
| P1 High | `HIGH_REG_PRESSURE` | >64 registers per thread (reduces occupancy) |
| P1 High | `PRED_OVERFLOW` | >8 predicates (causes spills) |
| P1 High | `PLACEHOLDER_CODE` | Incomplete code ("simplified", "omitted" comments) |
| P1 High | `EMPTY_LOOP` | Loop without computation |
| P1 High | `NO_BOUNDS_CHECK` | Missing thread bounds check |
| P1 High | `REG_SPILLS` | `.local` memory usage (register spills) |
| P2 Medium | `DEAD_CODE` | Unreachable code after `ret`/`bra` |
| P2 Medium | `UNOPT_MEM` | Non-vectorized memory access |
| P2 Medium | `REDUNDANT_MOVES` | Redundant register moves |

## Quick Start

```rust
use trueno_explain::{PtxBugAnalyzer, BugSeverity};

// Analyze PTX string
let ptx = include_str!("kernel.ptx");
let result = PtxBugAnalyzer::new().analyze(ptx);

// Check for bugs
if result.has_bugs() {
    println!("{}", result.format_report());
}

// Check specific severity
let critical = result.count_by_severity(BugSeverity::Critical);
assert_eq!(critical, 0, "No P0 bugs allowed!");
```

## Analyzer Modes

### Default Mode

Standard analysis - catches obvious bugs:

```rust
let analyzer = PtxBugAnalyzer::new();
let result = analyzer.analyze(ptx);
```

### Strict Mode

Catches more potential issues (may have false positives):

```rust
let analyzer = PtxBugAnalyzer::strict();
let result = analyzer.analyze(ptx);
```

### With Whitelist

Suppress known acceptable warnings:

```rust
use trueno_explain::PtxBugClass;

let analyzer = PtxBugAnalyzer::new()
    .with_whitelist("tensor_core*", PtxBugClass::HighRegisterPressure,
        "Tensor core kernels need high registers");
```

### Quantized Kernel Whitelist

Pre-configured for quantized kernels (q4k, q5k, q6k, q8k):

```rust
// Suppresses HighRegisterPressure for quantized kernels
let analyzer = PtxBugAnalyzer::with_quantized_whitelist();
```

## Examples

### Run Deep Bug Hunt

Analyze all trueno-gpu kernels:

```bash
cargo run -p trueno-explain --example deep_bug_hunt
```

Output:
```
SUMMARY: 30 kernels analyzed
  Total bugs: 16
  P0 Critical: 0
  P1 High: 16
  P2 Medium: 0

BUGS BY CLASS:
  HIGH_REG_PRESSURE         : 16
```

### Analyze External PTX

Analyze hand-rolled PTX from another project:

```bash
cargo run -p trueno-explain --example analyze_realizar
```

Output:
```
REALIZAR PTX SUMMARY
  Files analyzed: 4
  Total bugs: 18
  P0 Critical: 0
  P1 High: 15
  P2 Medium: 3
```

### Inspect PTX Details

Deep dive into specific kernel PTX:

```bash
cargo run -p trueno-explain --example ptx_inspector
```

## Bug Classes in Detail

### P0 Critical - Correctness Bugs

#### SharedMemU64Addressing

**Problem**: Using 64-bit registers for shared memory addressing.

```ptx
// BAD: %rd0 is 64-bit
st.shared.f32 [%rd0], %f0;

// GOOD: %r0 is 32-bit
st.shared.f32 [%r0], %f0;
```

**Impact**: Undefined behavior, potential silent corruption.

#### MissingBarrierSync

**Problem**: No `bar.sync` between shared memory write and read.

```ptx
// BAD: Race condition!
st.shared.f32 [%r0], %f0;
ld.shared.f32 %f1, [%r1];  // May read stale data

// GOOD: Barrier ensures visibility
st.shared.f32 [%r0], %f0;
bar.sync 0;
ld.shared.f32 %f1, [%r1];
```

**Impact**: Race condition, non-deterministic results.

### P1 High - Performance Bugs

#### HighRegisterPressure

**Problem**: >64 registers per thread reduces occupancy.

```
Register count: 120
Max occupancy: 65536 / (120 * 32) = 17 warps/SM (53%)
```

**Impact**: Reduced parallelism, lower throughput.

**Fix**: Reduce live variables, split kernel, or accept lower occupancy for compute-bound kernels.

#### PlaceholderCode

**Problem**: Comments indicate incomplete implementation.

```ptx
// Detected patterns:
// "simplified"
// "omitted"
// "placeholder"
// "for now"
// "TODO"
```

**Impact**: Kernel may produce incorrect results or have missing functionality.

### P2 Medium - Optimization Opportunities

#### DeadCode

**Problem**: Unreachable code after unconditional branch/return.

```ptx
// BAD: add.f32 is unreachable
ret;
add.f32 %f0, %f1, %f2;

// BAD: mul.f32 is unreachable
bra skip;
mul.f32 %f0, %f1, %f2;
skip:
```

**Impact**: Code bloat, wasted compilation time.

#### UnoptimizedMemoryPattern

**Problem**: Multiple single-element loads that could be vectorized.

```ptx
// BAD: 4 separate loads
ld.global.f32 %f0, [%rd0];
ld.global.f32 %f1, [%rd0+4];
ld.global.f32 %f2, [%rd0+8];
ld.global.f32 %f3, [%rd0+12];

// GOOD: Single vectorized load
ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0];
```

**Impact**: 4x memory bandwidth reduction.

## Integration with CI

Add PTX bug detection to your CI pipeline:

```yaml
# .github/workflows/ptx-analysis.yml
- name: PTX Bug Analysis
  run: |
    cargo run -p trueno-explain --example deep_bug_hunt
    # Fail if any P0 bugs found
    cargo test -p trueno-explain --test ptx_bug_hunting
```

## Writing Bug-Free PTX

Use `trueno-gpu` kernel generators instead of hand-writing PTX:

```rust
use trueno_gpu::kernels::{GemmKernel, Kernel};

// Generated PTX is verified bug-free
let kernel = GemmKernel::tiled(1024, 1024, 1024, 32);
let ptx = kernel.emit_ptx();

// Verify with analyzer
let result = PtxBugAnalyzer::new().analyze(&ptx);
assert!(result.is_valid());
```

## API Reference

### PtxBugAnalyzer

```rust
impl PtxBugAnalyzer {
    /// Create default analyzer
    pub fn new() -> Self;

    /// Create strict mode analyzer
    pub fn strict() -> Self;

    /// Pre-configured whitelist for quantized kernels
    pub fn with_quantized_whitelist() -> Self;

    /// Add whitelist entry
    pub fn with_whitelist(
        self,
        kernel_pattern: &str,  // e.g., "q4k*"
        bug_class: PtxBugClass,
        reason: &str
    ) -> Self;

    /// Analyze PTX and return report
    pub fn analyze(&self, ptx: &str) -> PtxBugReport;
}
```

### PtxBugReport

```rust
impl PtxBugReport {
    /// Check if any bugs found
    pub fn has_bugs(&self) -> bool;

    /// Check for specific bug class
    pub fn has_bug(&self, class: &PtxBugClass) -> bool;

    /// Check if kernel is valid (no P0/P1 bugs)
    pub fn is_valid(&self) -> bool;

    /// Count bugs by severity
    pub fn count_by_severity(&self, severity: BugSeverity) -> usize;

    /// Get formatted report string
    pub fn format_report(&self) -> String;
}
```

## See Also

- [PTX Best Practices](./ptx-best-practices.md)
- [PTX Register Allocation](../architecture/ptx-register-allocation.md)
- [PTX Code Generation](../architecture/ptx-generation.md)
