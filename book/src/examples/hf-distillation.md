# HuggingFace Model Distillation

This example demonstrates knowledge distillation from HuggingFace models to smaller student models.

## Running the Example

```bash
cargo run --example hf_distillation
```

## Code

```rust
{{#include ../../../examples/hf_distillation.rs}}
```

## Distillation Pipeline

1. **Load teacher model** from HuggingFace
2. **Create student model** with smaller architecture
3. **Generate soft targets** using temperature scaling
4. **Train student** on soft + hard targets

## Configuration

```rust
let config = DistillationConfig {
    temperature: 4.0,        // Softmax temperature
    alpha: 0.7,              // Weight for soft targets
    teacher_model: "bert-base-uncased",
    student_layers: 6,       // Half of teacher
};
```

## Memory Savings

| Model | Teacher | Student | Reduction |
|-------|---------|---------|-----------|
| BERT | 110M | 66M | 40% |
| GPT-2 | 124M | 82M | 34% |
| LLaMA | 7B | 1.3B | 81% |
