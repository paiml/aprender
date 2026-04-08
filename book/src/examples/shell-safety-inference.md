<!-- PCU: examples-shell-safety-inference | contract: contracts/apr-page-examples-shell-safety-inference-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example shell_safety_training -->
<!-- Status: enforced -->

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
// Run this example:
//   cargo run --example shell_safety_inference
//
// See the CLI reference and source code in crates/ for implementation details.
```
