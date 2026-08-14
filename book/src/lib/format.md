<!-- PCU: lib-format | contract: contracts/apr-page-lib-format-v1.yaml -->

# Module: `aprender::format`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/format/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/format/mod.rs) or directory.

## Example

<!-- example-cost: skip -->
```rust
use aprender::format::{ModelCard, AprConverter, ConvertOptions};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::format` is the model-file plumbing layer. It owns the `.apr`
binary format (v1 and v2), bidirectional conversion to/from GGUF, SafeTensors,
and ONNX, on-disk quantization, sharded checkpoints, golden-output references,
diff/lint tooling, and the layout-contract enforcement that keeps row-major
APR data safe from column-major imports. If a tensor lives on disk, something
in this module handles it.

## Key types

| Type | Description |
|------|-------------|
| `AprConverter` | The unified converter for GGUF / SafeTensors / ONNX → APR (and back). Driven by `ConvertOptions`. |
| `ConvertOptions`, `ConvertReport` | Configuration and result diagnostics for `AprConverter::convert`. |
| `ModelCard` | Structured model documentation: license, training data, intended use, citations. |
| `Gate`, `PokaYoke`, `PokaYokeResult` | Validation gates that catch malformed weight files at the import boundary. |
| `DiffReport`, `DiffEntry`, `DiffCategory` | Tensor-by-tensor diff of two model files. |
| `AprV2Header`, `AprV2Flags` | Low-level APR v2 header structures. |

The submodules `quantize`, `gguf`, `onnx`, `sharded`, `signing`, and
`layout_contract` each have their own focused public APIs — see the rustdoc.

## Usage patterns

### Pattern 1: Attach a model card to a checkpoint

<!-- example-cost: skip -->
```rust
use aprender::format::{ModelCard, TrainingDataInfo};

let card = ModelCard {
    name: "albor-370m-v1".to_string(),
    description: "370M-param distilled student.".to_string(),
    license: "Apache-2.0".to_string(),
    training_data: Some(TrainingDataInfo {
        dataset: "CSN-Python".to_string(),
        num_samples: 412_345,
        source: Some("https://huggingface.co/datasets/code_search_net".to_string()),
    }),
    citations: vec![],
    intended_use: Some("Code completion and small-scale generation.".to_string()),
};
println!("model card: {}", card.name);
```

### Pattern 2: Diff two checkpoints tensor-by-tensor

<!-- example-cost: skip -->
```rust
use aprender::format::{DiffOptions, DiffCategory};
// AprConverter / diff_models work on real on-disk artefacts;
// see `apr diff a.apr b.apr` for the CLI surface that wraps these.
//
//     let opts = DiffOptions::default();
//     let report = aprender::format::diff_models(&path_a, &path_b, &opts)?;
//     for entry in &report.entries {
//         if entry.category == DiffCategory::Numerical {
//             println!("{}: max_abs={}", entry.tensor_name, entry.max_abs_diff);
//         }
//     }
```

## See also

- [`serialization`](serialization.md) — the higher-level `AprReader` / `AprWriter` and SafeTensors helpers
- [`inspect`](inspect.md) — read-only inspection of model metadata + tensor stats
- [`models`](models.md) — model implementations that load weights from `format`
- [`primitives`](primitives.md) — the `Matrix` / `Vector` types that on-disk tensors deserialize into

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
