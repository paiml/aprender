# Case Study: Sharded SafeTensors Serving (GH-213)

## The Bug

When running `apr serve` on a sharded SafeTensors model (3B+ parameters), the server crashed with:

```
SafeTensors header too large
```

The root cause: `start_realizar_server()` reads the first 8 bytes of the file for format detection. For a `.safetensors.index.json` file, those 8 bytes are `{"weight` — JSON text, not a binary header. The format detector interprets this as a SafeTensors header size (a massive number), triggering DOS protection.

Meanwhile, `apr run` already handled sharded models correctly via `run_sharded_safetensors_inference()` in realizar. The serve path simply lacked the same detection.

## The Fix

Two changes, following the existing realizar pattern:

### 1. Early Detection in `handlers.rs`

Before reading any bytes from the file, check if the path ends with `.safetensors.index.json`:

```rust
// GH-213: Detect sharded SafeTensors index.json BEFORE reading file bytes.
let path_str = model_path.to_string_lossy();
if path_str.ends_with(".safetensors.index.json") {
    return super::safetensors::start_sharded_safetensors_server(model_path, config);
}

// ... existing 8-byte format detection continues for non-sharded files
```

### 2. Sharded Server Function in `safetensors.rs`

The new `start_sharded_safetensors_server()` mirrors the single-file `start_safetensors_server()` but uses:

- `ShardedSafeTensorsModel::load_from_index()` instead of `std::fs::read()`
- `SafetensorsConfig::load_from_sibling()` for `config.json`
- `SafetensorsToAprConverter::convert_sharded()` instead of `convert()`

The rest (tokenizer loading, axum router, handler functions) is shared with the single-file path.

## Verification

MVP playbook tests confirmed the fix across all model sizes:

| Model | Shards | Serve CPU | Serve GPU |
|-------|--------|-----------|-----------|
| 0.5B | 1 (single file) | Pass | Pass |
| 3B | 2 | **Pass** (was crash) | **Pass** (was crash) |
| 7B | 4 | **Pass** (was crash) | **Pass** (was crash) |
| 14B | 6 | Timeout (resource) | Timeout (resource) |

The 14B timeouts are a resource limitation (56GB F32 model exceeds the 120s server-readiness timeout), not a code bug.

## Lessons

1. **Format detection must handle metadata files.** Binary magic-byte detection fails on JSON index files. Check file extensions first for known patterns before falling back to byte-level detection.

2. **Mirror existing patterns.** The `apr run` sharded path in realizar was the reference implementation. The serve fix reuses the same APIs (`ShardedSafeTensorsModel`, `SafetensorsToAprConverter::convert_sharded`) rather than reinventing.

3. **Test at every model size.** The bug only manifests with sharded models (3B+). Single-file models (0.5B) work fine. Without multi-model testing, this would have been missed.

## Related

- **Bug 205** in the [showcase spec](../../../docs/specifications/qwen2.5-coder-showcase-demo.md)
- `realizar/src/infer/mod.rs:1379` — reference sharded inference implementation
- `realizar/src/safetensors/mod.rs` — `ShardedSafeTensorsModel` API
