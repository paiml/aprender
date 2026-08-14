<!-- PCU: lib-serialization | contract: contracts/apr-page-lib-serialization-v1.yaml -->

# Module: `aprender::serialization`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/serialization/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/serialization/mod.rs) or directory.

## Example

<!-- example-cost: trivial -->
```rust
use aprender::serialization::{AprReader, AprWriter, SafeTensorsMetadata};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::serialization` is the high-level checkpoint I/O — read and write
`.apr` files and SafeTensors. Where [`format`](format.md) exposes the lower-
level format internals (validation gates, layout contracts, converters), this
module gives you ergonomic `AprReader` / `AprWriter` types for round-tripping
tensors and metadata. Use it whenever you need to persist trained weights,
load a checkpoint by name, or attach arbitrary JSON metadata alongside the
tensors.

## Key types

| Type | Description |
|------|-------------|
| `AprReader` | Reads `.apr` files. `open(path)`, `from_bytes(bytes)`, `open_filtered(path, predicate)` for partial loads. `read_tensor_f32(name)` for individual tensors. |
| `AprWriter` | Builds `.apr` files. `add_tensor_f32(name, shape, data)`, `set_metadata(key, value)`, then `write(path)` or `into_bytes()`. |
| `SafeTensorsMetadata` | Parsed metadata from a `.safetensors` header. |
| `AprTensorDescriptor` | Per-tensor descriptor (name, shape, dtype) used internally by the reader. |

## Usage patterns

### Pattern 1: Round-trip a small set of tensors

```rust
use aprender::serialization::{AprReader, AprWriter};
use serde_json::json;
use std::path::PathBuf;

let weights = [0.1_f32, 0.2, 0.3, 0.4];

// --- write
let mut w = AprWriter::new();
w.set_metadata("epoch", json!(5));
w.set_metadata("learning_rate", json!(1e-3));
w.add_tensor_f32("encoder.weight", vec![2, 2], &weights);

let bytes = w.to_bytes().expect("serialize");
assert!(!bytes.is_empty());

// --- read back from bytes
let r = AprReader::from_bytes(bytes).expect("parse");
let lr = r.get_metadata("learning_rate").cloned();
println!("learning_rate from file: {:?}", lr);

let read_back = r.read_tensor_f32("encoder.weight").expect("read tensor");
assert_eq!(read_back, weights.to_vec());
```

### Pattern 2: Filtered partial loads

```rust
use aprender::serialization::AprReader;

// Only load tensors whose names start with "embeddings."
// (Useful for large checkpoints when you only need a subset.)
//     let reader = AprReader::open_filtered(&path, |name| name.starts_with("embeddings."))?;
//     let emb = reader.read_tensor_f32("embeddings.tokens.weight")?;

// Inspect all metadata without touching tensor bytes:
//     for (key, value) in reader.all_metadata() {
//         println!("{} = {}", key, value);
//     }
```

## See also

- [`format`](format.md) — lower-level converter, validation, signing, sharding
- [`models`](models.md) — load Qwen2 / BERT weights via the `models::*` loaders that wrap this module
- [`inspect`](inspect.md) — read-only inspection of model metadata
- [`bundle`](bundle.md) — bundle model + tokenizer + config into a single artifact

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
