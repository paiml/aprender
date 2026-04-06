# trueno-cuda-edge: GPU Edge-Case Testing

`trueno-cuda-edge` is a GPU edge-case test framework implementing Popperian falsificationism for CUDA/GPU code. It provides 5 falsification frameworks with a 50-point verification checklist.

## Overview

GPU code is notoriously difficult to test due to:
- Non-deterministic behavior
- Hardware-dependent edge cases
- Complex lifecycle management
- Numerical precision variations

trueno-cuda-edge addresses these challenges with systematic falsification testing.

## Frameworks

### F1: Null Pointer Sentinel Fuzzer

Validates null pointer handling in GPU memory operations.

```rust
use trueno_cuda_edge::null_fuzzer::{
    NonNullDevicePtr, InjectionStrategy, NullFuzzerConfig
};

// Type-safe device pointer that rejects null at construction
let ptr = NonNullDevicePtr::<f32>::new(0x7f00_0000_0000)?;

// Configure injection strategy
let config = NullFuzzerConfig {
    strategy: InjectionStrategy::Periodic { interval: 10 },
    total_calls: 1000,
    fail_fast: false,
};
```

### F2: Shared Memory Boundary Prober

Validates shared memory allocation and access patterns.

```rust
use trueno_cuda_edge::shmem_prober::{
    ComputeCapability, check_allocation, shared_memory_limit
};

// Check shared memory limits by compute capability
let ampere = ComputeCapability::new(8, 0);
assert_eq!(shared_memory_limit(ampere), 164 * 1024); // 164 KB

// Validate allocation fits
check_allocation(ampere, 128 * 1024)?;
```

### F3: Context Lifecycle Chaos

Tests GPU context lifecycle edge cases.

```rust
use trueno_cuda_edge::lifecycle_chaos::{
    ChaosScenario, ContextLeakDetector, DestructionOrdering
};

// 8 chaos scenarios to test
let scenarios = ChaosScenario::all();
// DoubleDestroy, UseAfterDestroy, LeakedContext, etc.

// Leak detection with 1MB tolerance
let detector = ContextLeakDetector::new();
let report = detector.analyze(memory_before, memory_after);
```

### F4: Quantization Parity Oracle

Validates CPU/GPU numerical parity for quantized operations.

```rust
use trueno_cuda_edge::quant_oracle::{
    QuantFormat, BoundaryValueGenerator, check_values_parity, ParityConfig
};

// Format-specific tolerances
assert_eq!(QuantFormat::Q4K.tolerance(), 0.05);  // 5% for 4-bit
assert_eq!(QuantFormat::Q6K.tolerance(), 0.01);  // 1% for 6-bit

// Generate boundary values for edge case testing
let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);
let boundaries = gen.all_boundaries(); // 0, NaN, Inf, format levels

// Compare CPU and GPU results
let config = ParityConfig::new(QuantFormat::Q4K);
let report = check_values_parity(&cpu_values, &gpu_values, &config);
assert!(report.passed());
```

### F5: PTX Compilation Poison Trap

Mutation testing for PTX kernels.

```rust
use trueno_cuda_edge::ptx_poison::{
    PtxVerifier, PtxMutator, default_mutators
};

// Structural verification
let verifier = PtxVerifier::new();
let verified = verifier.verify(ptx_source)?;

// 8 mutation operators
let mutators = default_mutators();
// FlipAddSub, FlipMulDiv, InvertPredicate, RemoveBarrier, etc.

// Apply mutations
let mutated = PtxMutator::FlipAddSub.apply(ptx_source);
```

## Falsification Protocol

The 50-point falsification checklist tracks verification status:

```rust
use trueno_cuda_edge::falsification::{
    FalsificationReport, all_claims, Framework
};

// Initialize report with all 50 claims
let mut report = FalsificationReport::new();

// Mark claims as verified/violated during testing
report.mark_verified("NF-001");
report.mark_violated("SP-002");

// Track coverage
println!("Coverage: {:.1}%", report.coverage() * 100.0);
println!("Complete: {}", report.is_complete());
```

## Supervision Integration

Erlang OTP-style supervision for GPU workers:

```rust
use trueno_cuda_edge::supervisor::{
    SupervisionStrategy, SupervisionTree, GpuHealthMonitor
};

// One-for-one: isolated restarts
let mut tree = SupervisionTree::new(
    SupervisionStrategy::OneForOne,
    4 // workers
);

// Health monitoring
let monitor = GpuHealthMonitor::builder()
    .max_missed(3)
    .throttle_temp(85)
    .shutdown_temp(95)
    .build();
```

## Examples

Run the examples to see the framework in action:

```bash
# Null pointer fuzzing demo
cargo run -p trueno-cuda-edge --example null_fuzzer_demo

# Quantization boundary demo
cargo run -p trueno-cuda-edge --example quant_oracle_demo

# PTX verification demo
cargo run -p trueno-cuda-edge --example ptx_verifier_demo

# Falsification protocol demo
cargo run -p trueno-cuda-edge --example falsification_report_demo
```

## Test Coverage

trueno-cuda-edge has 98.86% test coverage with 292 tests:
- 176 unit tests
- 12 property-based tests (proptest)
- 76 integration tests
- 9 doc tests

## Integration

Add to your `Cargo.toml`:

```toml
[dev-dependencies]
trueno-cuda-edge = "0.1"
```

Then create integration tests using the falsification frameworks:

```rust
use trueno_cuda_edge::{
    null_fuzzer::NonNullDevicePtr,
    quant_oracle::{QuantFormat, check_values_parity, ParityConfig},
    ptx_poison::PtxVerifier,
};

#[test]
fn test_device_ptr_safety() {
    assert!(NonNullDevicePtr::<f32>::new(0).is_err());
}

#[test]
fn test_quantization_parity() {
    let config = ParityConfig::new(QuantFormat::Q4K);
    let report = check_values_parity(&cpu, &gpu, &config);
    assert!(report.passed());
}
```
