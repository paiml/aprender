# Provable Contracts Pipeline

Entrenar uses YAML-defined provable contracts to enforce training correctness invariants at compile time. Contracts are checked as `debug_assert!` in debug builds and compiled away in release builds.

## Contract Files

All contracts live in `contracts/`:

| Contract | Domain |
|----------|--------|
| `training-loop-v1.yaml` | Training loop invariants |
| `backward-pass-v1.yaml` | Backward pass gradient flow |
| `optimizer-v1.yaml` | Optimizer state updates |
| `matmul-v1.yaml` | Matrix multiplication |
| `gemm-v1.yaml` | General matrix multiply |
| `softmax-v1.yaml` | Softmax numerical stability |
| `lora-v1.yaml` | LoRA adapter constraints |
| `quantization-v1.yaml` | Quantization round-trip |
| `batch-v1.yaml` | Batch processing bounds |
| `checkpoint-v1.yaml` | Checkpoint save/load |
| `tokenizer-v1.yaml` | Tokenizer encode/decode |

## How the Pipeline Works

1. **YAML** -- Each contract defines preconditions, postconditions, and invariants
2. **build.rs** -- At compile time, `build.rs` reads the YAML files and generates Rust assertion code
3. **#[contract]** -- Functions annotated with `#[contract]` get the generated checks injected
4. **debug_assert!** -- Checks run in debug/test builds; zero overhead in release

## Running the Demo

```bash
cargo run --example contract_pipeline_demo
```

## Adding a New Contract

1. Create `contracts/my-feature-v1.yaml` with preconditions and postconditions
2. Add precondition checks to the target function
3. Annotate the function with `#[contract]`
4. Run tests: `cargo test`

For the full contract specification and YAML schema, see the
[provable-contracts integration guide](https://github.com/paiml/provable-contracts/blob/main/book/src/integration.md).
