# Fine-tune for Test Generation

This example demonstrates fine-tuning Qwen2-0.5B-Coder to generate Rust unit tests and property-based tests from
function implementations using QLoRA for memory efficiency.

## Running the Example

```bash
cargo run --example finetune_test_gen
```

## Code

```rust
{{#include ../../../examples/finetune_test_gen.rs}}
```

## Overview

The example shows a complete fine-tuning pipeline:

1. **Model Configuration** - Qwen2-0.5B-Coder with LoRA rank 16
2. **Dataset Preparation** - Function → test pairs
3. **QLoRA Setup** - 4-bit quantized base weights
4. **Training Loop** - Simulated training progress
5. **Inference** - Generate tests for new functions
6. **Evaluation** - Compile rate, mutation score, coverage

## QLoRA Memory Savings

| Configuration | VRAM Required | Reduction |
|---------------|---------------|-----------|
| FP32 fine-tuning | ~1500 MB | - |
| QLoRA fine-tuning | ~185 MB | **8x** |

## Training Data Format

Each training sample pairs a function with its tests:

**Input (function):**
```rust
/// Returns the absolute value of a number
pub fn abs(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}
```

**Output (unit tests):**
```rust
#[test]
fn test_abs_positive() {
    assert_eq!(abs(5), 5);
}

#[test]
fn test_abs_negative() {
    assert_eq!(abs(-5), 5);
}
```

**Output (property tests):**
```rust
proptest! {
    #[test]
    fn prop_abs_non_negative(x in any::<i32>()) {
        prop_assert!(abs(x) >= 0);
    }
}
```

## LoRA Configuration

```rust
let lora_config = LoRAConfig::new(16, 32.0)
    .target_qv_projections();  // q_proj and v_proj only
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rank (r) | 16 | Low-rank decomposition size |
| Alpha | 32.0 | Scaling parameter |
| Scale (α/r) | 2.0 | Effective learning rate multiplier |
| Targets | q_proj, v_proj | Attention projections |
| Trainable % | 0.26% | ~1.4M of 527M parameters |

## Evaluation Metrics

After fine-tuning on Rust function → test pairs:

| Metric | Baseline | Fine-tuned |
|--------|----------|------------|
| Compile rate | 62.0% | **94.0%** |
| Tests passing | 54.0% | **89.0%** |
| Coverage delta | +5.2% | **+12.4%** |
| Mutation score | 45.1% | **71.3%** |

## Test Type Distribution

The model learns to generate diverse test types:

- Happy path tests: 35.2%
- Edge case tests: 28.7%
- Error handling tests: 18.4%
- Property-based tests: 17.7%

## Use Cases

1. **IDE Integration** - Generate tests on-demand in editor
2. **CI Pipeline** - Auto-generate tests for new functions
3. **Test Coverage** - Fill coverage gaps automatically
4. **Code Review** - Suggest tests during PR review

## Related Examples

- [LoRA Fine-tuning](./lora-finetuning.md) - Basic LoRA setup
- [QLoRA Example](./qlora-example.md) - Memory-efficient fine-tuning
- [Training Loop](./training-loop.md) - Trainer API basics
