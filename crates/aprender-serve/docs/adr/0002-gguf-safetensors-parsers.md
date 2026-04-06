# ADR-0002: Pure Rust GGUF and Safetensors Parsers

## Status

Accepted

## Date

2024-11-15

## Context

Realizar needs to load models from popular formats used by llama.cpp (GGUF) and HuggingFace (Safetensors). Options:

1. **Use existing libraries** - gguf-rs, safetensors crate
2. **FFI bindings** - Bind to llama.cpp's parser
3. **Pure Rust from scratch** - Implement our own parsers

## Decision

Implement pure Rust parsers for both GGUF and Safetensors formats from scratch.

## Rationale

1. **Zero-copy memory mapping** - Optimized for large model files
2. **No Python dependencies** - Can load models without Python ecosystem
3. **Complete control** - Handle edge cases and format variations ourselves
4. **Precise error messages** - Debug issues without black-box libraries
5. **Quantization awareness** - Parse quantized tensor data directly

## Consequences

### Positive
- Zero external dependencies for model loading
- Optimized for our inference patterns
- Full control over memory management

### Negative
- Must track GGUF/Safetensors spec changes ourselves
- Initial development time investment
- Risk of format compatibility bugs

## Implementation

```rust
// GGUF parser
pub fn parse_gguf(path: &Path) -> Result<GGUFModel> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let header = GGUFHeader::parse(&mmap)?;
    let metadata = parse_metadata(&mmap, &header)?;
    let tensors = parse_tensor_info(&mmap, &header)?;

    Ok(GGUFModel { header, metadata, tensors, mmap })
}
```

## Validation

**Falsifiable claim**: Our GGUF parser loads models at equivalent or faster speed than llama.cpp's native loader.

**Test**: Benchmark load time for 1B, 7B, 13B parameter models against llama.cpp.

## References

- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Safetensors Format](https://github.com/huggingface/safetensors)
