# GGUF Module Migration Plan

## Status: Phase 1 - Module Structure Created

The GGUF monolith (54K lines) is being incrementally migrated into focused modules.

### Current Structure

```
src/gguf/
├── mod.rs                    # Main module (orchestrates re-exports)
├── MIGRATION.md              # This file
├── batch_scheduler.rs        # Batch scheduling (already extracted)
├── test_helpers.rs           # Test utilities (already extracted)
├── types.rs                  # Core types & constants (re-exports from monolith)
├── config.rs                 # GGUFConfig (re-exports from monolith)
├── model.rs                  # GGUFModel, MappedGGUFModel, GGUFTransformer (re-exports)
├── quantized.rs              # Quantized tensor types (re-exports)
└── owned.rs                  # OwnedQuantized* types (re-exports)
```

### Migration Strategy

**Phase 1: Structure** (COMPLETE)
- ✅ Create module files
- ✅ Re-export from monolith
- ✅ Verify builds work
- ✅ All imports remain compatible

**Phase 2: Extract Core Types** (TODO)
- Move type definitions from monolith to respective modules
- Move impl blocks with the types
- Keep monolith compiling by updating its imports
- Verify no breakage

**Phase 3: Extract Implementations** (TODO)
- Move helper functions
- Move trait implementations
- Extract parsing logic
- Extract inference logic

**Phase 4: Tests & Documentation** (TODO)
- Add comprehensive tests to each module
- Document each type with examples
- Remove code from monolith
- Verify 85%+ coverage maintained

### Key Types to Extract

#### types.rs
- `GGUF_MAGIC`, `GGUF_VERSION_V3`, quantization type constants
- `GGUF_ALIGNMENT`, buffer constants
- `TokenBuffer`, `AttentionBuffer`, `HiddenBuffer`
- `GGUFValue`, `GGUFHeader`, `TensorInfo`, `GGUFModel`

#### config.rs
- `GGUFConfig`
- `impl GGUFConfig::from_gguf()`

#### model.rs
- `MappedGGUFModel`
- `GGUFTransformer`, `GGUFTransformerLayer`
- Model loading and parsing implementations

#### quantized.rs
- `QuantizedTensorRef`
- `QKVWeights`, `OwnedQKVWeights`
- `OwnedQuantizedTensor`, `OwnedQuantizedLayer`

#### owned.rs
- `QuantizedGenerateConfig`
- `OwnedQuantizedKVCache`
- `OwnedQuantizedModel`
- `OwnedQuantizedModelCached`
- `OwnedQuantizedModelCachedSync`
- `OwnedQuantizedModelCuda` (feature = "cuda")

### Dependencies Between Modules

```
types (base types)
  ↓
model (uses types)
  ↓
config (uses model)
  ↓
quantized (uses types, model)
  ↓
owned (uses quantized, config)
  ↓
batch_scheduler (uses owned)
```

### Build Verification

```bash
# Verify lib builds
cargo build --lib

# Verify with CUDA feature
cargo build --lib --features cuda

# Run tests
cargo test --lib

# Check coverage impact
cargo llvm-cov --lib --no-report
```

### Import Compatibility

All existing imports remain valid:
```rust
use realizar::gguf::{GGUFModel, GGUFConfig, OwnedQuantizedModel};
```

Module-specific imports also work:
```rust
use realizar::gguf::types::{GGUF_MAGIC, GGUFHeader};
use realizar::gguf::config::GGUFConfig;
```

### Next Steps

1. Extract actual type definitions to respective modules
2. Update monolith to import from new modules
3. Move impl blocks to their types
4. Add comprehensive tests
5. Verify coverage remains ≥85%
6. Delete extracted code from monolith

### Notes

- Monolith remains as fallback during migration
- All re-exports maintain backward compatibility
- No breaking changes to public API
- Tests continue to pass throughout migration
