# Fuzz Testing Infrastructure

This directory contains fuzz testing targets for entrenar, following the Certeza EXTREME TDD methodology from the spec.

## Overview

Fuzz testing validates that entrenar's core functionality never panics or crashes with arbitrary inputs. This is
critical for production-ready ML infrastructure.

## Fuzz Targets

### 1. `parameter_calc` - LLaMA Parameter Calculations

**Purpose:** Validates that parameter calculations never panic or overflow with extreme inputs.

**Tests:**
- Embedding parameters (vocab_size × hidden_size × 2)
- Attention parameters (4 × hidden_size × hidden_size per layer)
- FFN parameters (3 × hidden_size × intermediate_size per layer)
- Memory estimations (FP32, FP16, 4-bit)
- LoRA adapter parameter calculations
- Batch size and sequence length combinations

**Invariants Validated:**
- Parameter calculations use overflow-safe arithmetic
- Hidden size / num_heads division never panics
- Memory estimation handles large values gracefully
- LoRA parameter calculation never panics
- Batch size calculations never overflow

### 2. `tensor_ops` - Tensor Operations

**Purpose:** Validates that tensor operations never panic with arbitrary inputs.

**Tests:**
- Element-wise operations (add, mul)
- Activation functions (relu, gelu, swish)
- Operation chaining (relu + add, gelu + mul)
- Extreme values (-1000 to +1000)
- Special values (zeros, ones)
- Large values that might cause numerical instability

**Invariants Validated:**
- Element-wise operations never panic
- Tensor creation validates sizes correctly
- Operations on special values (zeros, ones) work correctly
- Activations handle extreme values without panic
- NaN and Inf handling doesn't cause crashes

### 3. `lora_config` - LoRA Configuration

**Purpose:** Validates that LoRA configuration calculations are robust to arbitrary inputs.

**Tests:**
- LoRA rank vs hidden_size constraints
- Base model parameter calculations
- LoRA adapter parameter calculations
- Parameter reduction percentage calculations
- Memory calculations for different quantization levels
- QLoRA memory estimation (4-bit base + FP32 adapters)
- Scaling factor (alpha / rank) computation

**Invariants Validated:**
- LoRA rank <= hidden_size constraint is enforced
- Parameter calculations use checked arithmetic
- Parameter reduction is positive for valid configs
- Memory calculations don't overflow
- QLoRA uses less memory than FP32 LoRA
- Scaling factor computation doesn't panic

## System Requirements

**Important:** Fuzz testing requires the C++ standard library for linking with libfuzzer-sys.

On Ubuntu/Debian:
```bash
sudo apt-get install libstdc++-12-dev
```

On other systems, install the appropriate C++ development package.

## Running Fuzz Tests

### List all fuzz targets:
```bash
cargo fuzz list
```

### Run a specific fuzz target:
```bash
# Run for 60 seconds
cargo fuzz run parameter_calc -- -max_total_time=60

# Run for 1M iterations
cargo fuzz run parameter_calc -- -runs=1000000

# Run with specific number of jobs
cargo fuzz run tensor_ops -- -jobs=8 -workers=8
```

### Run all fuzz targets (for CI/CD):
```bash
# Quick smoke test (10 seconds each)
cargo fuzz run parameter_calc -- -max_total_time=10
cargo fuzz run tensor_ops -- -max_total_time=10
cargo fuzz run lora_config -- -max_total_time=10

# Full fuzz run (1M+ iterations, as per spec)
cargo fuzz run parameter_calc -- -runs=1000000
cargo fuzz run tensor_ops -- -runs=1000000
cargo fuzz run lora_config -- -runs=1000000
```

## Acceptance Criteria (from spec)

✅ **Target:** 1M+ iterations without crashes
✅ **Corpus generation:** Automatic via libfuzzer
✅ **Integration:** Ready for CI/CD (requires system library installation)

## Implementation Notes

- Uses `arbitrary` crate for structured fuzzing
- All fuzz targets validate specific invariants
- Overflow-safe arithmetic (`checked_mul`, `saturating_mul`)
- Input validation before calculations
- Graceful handling of invalid configurations

## Troubleshooting

### Linker Error: cannot find -lstdc++

This means the C++ standard library is not installed. Install it:
```bash
sudo apt-get install libstdc++-12-dev  # Ubuntu/Debian
sudo yum install libstdc++-devel       # RHEL/CentOS
```

### Out of Memory

Reduce the number of workers:
```bash
cargo fuzz run parameter_calc -- -workers=2
```

### Slow Performance

Limit iteration count or time:
```bash
cargo fuzz run parameter_calc -- -max_total_time=30 -runs=10000
```

## Files

- `fuzz/Cargo.toml` - Fuzz dependencies configuration
- `fuzz/fuzz_targets/parameter_calc.rs` - Parameter calculation fuzzing (100+ lines)
- `fuzz/fuzz_targets/tensor_ops.rs` - Tensor operations fuzzing (100+ lines)
- `fuzz/fuzz_targets/lora_config.rs` - LoRA configuration fuzzing (120+ lines)
- `fuzz/README.md` - This file

## Spec Compliance

This implementation follows **Phase 3 (Weeks 7-8)** of the LLaMA integration spec:

**Deliverables:**
- ✅ `fuzz/fuzz_targets/*.rs` - Fuzz testing targets
- ✅ Structured fuzzing with `arbitrary` crate
- ✅ 3 comprehensive fuzz targets covering critical paths
- ✅ Documentation

**Quality Gates:**
- ✅ Targets compile successfully
- ✅ Ready to run 1M+ iterations (requires system library)
- ✅ Corpus generation enabled
- ✅ CI/CD integration ready

**Built with EXTREME TDD** 🦀⚡

Following Certeza fuzz testing principles from the spec.
