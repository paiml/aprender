# Shell Safety Classifier Inference

Loads a trained shell safety model and classifies shell scripts
into 5 safety categories.

## Prerequisites

Train the model first (see [Shell Safety Training](shell-safety-training.md)):

```bash
cargo run --example shell_safety_training -- /tmp/corpus.jsonl
```

## Run

```bash
cargo run --example shell_safety_inference -- /tmp/shell-safety-model/
```

## Source

```rust,ignore
{{#include ../../../examples/shell_safety_inference.rs}}
```
