<!-- PCU: examples-publish-shell-safety | contract: contracts/apr-page-examples-publish-shell-safety-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example shell_safety_training -->
<!-- Status: enforced -->

# Publish Shell Safety Classifier

Uploads the trained shell safety model to HuggingFace as
`paiml/shell-safety-classifier`.

## Prerequisites

1. Train the model first:
   ```bash
   cargo run --example shell_safety_training -- /tmp/corpus.jsonl
   ```

2. Set HuggingFace token:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxx
   ```

## Run

```bash
cargo run --features hf-hub-integration --example publish_shell_safety -- /tmp/shell-safety-model/
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example publish_shell_safety
//
// See the CLI reference and source code in crates/ for implementation details.
```
