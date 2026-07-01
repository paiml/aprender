# apr-format

Sovereign `.apr` model container format — read + write, zero ML/GPU/tokenizer
dependencies.

`apr-format` is the leaf crate that owns the on-disk `.apr` container. It was
extracted from `aprender-core` (issue #2231 — "depend on the *format*, not the
*framework*") so that downstream consumers — realizar inference, `xpile`,
external tooling — can load and save `.apr` files **without** pulling the full
ML / GPU / tokenizer / quantization stack. `aprender-core` re-exports this
crate's public surface, so the extraction is API-compatible: existing
`aprender::format::*` paths keep working with no break.

## Installation

```toml
[dependencies]
apr-format = "0.50"

# Optional: zero-copy memory-mapped loading (v2 container)
apr-format = { version = "0.50", features = ["mmap"] }

# Optional: LZ4 + Zstd payload compression (v1 container)
apr-format = { version = "0.50", features = ["compression"] }

# Everything except the security seams
apr-format = { version = "0.50", features = ["full"] }
```

## What it is

- **Container I/O only.** Read and write the `.apr` model container in two
  versions: v1 (`APRN` magic) and the streaming, constant-memory v2 (`APR\0`).
- **Dependency-light.** Structural dependencies are just `serde`, `rmp-serde`,
  `bincode`, `serde_json`, `half`, and `thiserror`. No `trueno`, no autograd,
  no tokenizer.
- **std-only** (v1). `no_std` is an explicit deferred decision.
- **Structure vs. physics.** The byte-only structural validator
  (`validate_structure`) is separated from framework-level "physics" checks so
  a corrupt file can be diagnosed without loading tensors into an ML runtime.

The GGUF / SafeTensors / ONNX converter deliberately **stays in
`aprender-core`** — it needs `f32` physics and the ML stack. Only the container
moves here.

## Usage

```rust
use apr_format::{save, load, ModelType, SaveOptions};

// Write a model's weights to an .apr container
let weights: Vec<f32> = vec![1.0, 2.0, 3.0];
save(&weights, ModelType::LinearRegression, "model.apr", SaveOptions::default())?;

// Read them back (v1/v2 auto-detected)
let restored: Vec<f32> = load("model.apr", ModelType::LinearRegression)?;
assert_eq!(restored, weights);

// Inspect a container's header/metadata without deserializing tensors
let header = apr_format::inspect("model.apr")?;

// Byte-only structural validation (no ML runtime required)
let check = apr_format::validate_structure(&std::fs::read("model.apr")?);
# Ok::<(), apr_format::AprFormatError>(())
```

Zero-copy loading (with the `mmap` feature) and load-from-bytes helpers are also
exposed via `load_mmap`, `load_auto`, `load_from_bytes`, and `inspect_bytes`.

## Features

| Feature | Description |
|---------|-------------|
| `mmap` | Zero-copy memory-mapped loading of the v2 container (`memmap2`) |
| `compression` | LZ4 + Zstd payload compression for the v1 container |
| `encryption` | Placeholder seam for the sovereign-leaf security surface (Stage 2+) |
| `signing` | Placeholder seam for the sovereign-leaf security surface (Stage 2+) |
| `full` | Convenience meta-feature: `mmap` + `compression` (no security seams) |

## Public surface

- `save` / `load` / `load_auto` / `load_from_bytes` / `load_mmap` — container I/O
- `inspect` / `inspect_bytes` — header + metadata inspection
- `validate_structure` / `StructureCheck` — byte-only structural validation
- `Header`, `Metadata`, `Flags`, `ModelType`, `ModelInfo`, `SaveOptions`,
  `Compression` — the container types
- `ModelCard` / `TrainingDataInfo` — model-card metadata
- `crc32`, `f32_to_f16`, `f16_to_f32` — the deduplicated primitives
- `AprFormatError` / `Result` — the sovereign error seam (`aprender-core`
  `From`-wraps it)

## License

MIT OR Apache-2.0

---

Part of the [Aprender monorepo](https://github.com/paiml/aprender) — a
next-generation ML framework in pure Rust. See
[github.com/paiml/aprender](https://github.com/paiml/aprender) for the full
workspace, contracts, and book.
