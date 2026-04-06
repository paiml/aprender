# Provable Contracts Pipeline

Realizar uses YAML-defined provable contracts to enforce inference correctness invariants at compile time. Contracts are checked as `debug_assert!` in debug builds and compiled away in release builds.

## Contract Files

All contracts live in `contracts/`:

| Contract | Domain |
|----------|--------|
| `forward-pass-v1.yaml` | Forward pass invariants |
| `matmul-v1.yaml` | Matrix multiplication |
| `gemm-v1.yaml` | General matrix multiply |
| `quantization-v1.yaml` | Quantization round-trip |
| `tokenizer-v1.yaml` | Tokenizer encode/decode |
| `kv-cache-v1.yaml` | KV cache bounds |
| `kv-cache-management-v1.yaml` | KV cache lifecycle |
| `sampling-v1.yaml` | Sampling distribution |
| `kernel-dispatch-v1.yaml` | Kernel dispatch routing |
| `dispatch-v1.yaml` | Backend dispatch |
| `batch-inference-v1.yaml` | Batch inference bounds |

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
