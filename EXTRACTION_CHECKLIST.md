# GGUF Module Extraction Checklist

## Phase 1: Module Structure ✅ COMPLETE

### Files Created
- [x] `src/gguf/types.rs` - Core types and constants (1.7KB)
- [x] `src/gguf/config.rs` - GGUFConfig extraction (292B)
- [x] `src/gguf/model.rs` - Model types (314B)
- [x] `src/gguf/quantized.rs` - Quantized tensor types (372B)
- [x] `src/gguf/owned.rs` - Owned model types (507B)
- [x] `src/gguf/MIGRATION.md` - Migration plan and tracking
- [x] `GGUF_EXTRACTION_SUMMARY.md` - Comprehensive summary

### Module Integration
- [x] Updated `src/gguf/mod.rs` to include new modules
- [x] Set up re-exports from monolith
- [x] Verified all imports remain compatible
- [x] No breaking changes to public API

### Verification
- [x] `cargo build --lib` passes
- [x] `cargo build --lib --features cuda` passes
- [x] All 5,684 tests pass
- [x] 26 tests ignored (as before)
- [x] No clippy errors introduced
- [x] Test coverage maintained at 80.97%

### Documentation
- [x] Created MIGRATION.md with detailed plan
- [x] Created GGUF_EXTRACTION_SUMMARY.md
- [x] Created EXTRACTION_CHECKLIST.md (this file)
- [x] Documented module dependencies
- [x] Listed all exported types per module

## Phase 2: Type Extraction (TODO)

### Constants Extraction (types.rs)
- [ ] Extract `GGUF_MAGIC` constant
- [ ] Extract `GGUF_VERSION_V3` constant
- [ ] Extract quantization type constants (Q4_0, Q4_K, etc.)
- [ ] Extract `GGUF_ALIGNMENT` constant
- [ ] Extract buffer constants (TOKEN_BUFFER_INLINE_CAP, etc.)
- [ ] Extract buffer type aliases (TokenBuffer, etc.)
- [ ] Update monolith to import from types module
- [ ] Verify builds and tests pass

### Enum Extraction (types.rs)
- [ ] Extract `GGUFValue` enum
- [ ] Extract all `GGUFValue` match arms
- [ ] Update monolith imports
- [ ] Verify builds and tests pass

### Struct Extraction (types.rs)
- [ ] Extract `GGUFHeader` struct
- [ ] Extract `TensorInfo` struct
- [ ] Extract `GGUFModel` struct
- [ ] Update monolith imports
- [ ] Verify builds and tests pass

### GGUFModel Methods (types.rs or separate parser.rs)
- [ ] Extract `from_bytes()` method
- [ ] Extract `parse_header()` method
- [ ] Extract `parse_metadata()` method
- [ ] Extract `parse_tensor_info()` method
- [ ] Extract helper read methods (read_u32, read_string, etc.)
- [ ] Update monolith imports
- [ ] Verify builds and tests pass

### Config Extraction (config.rs)
- [ ] Extract `GGUFConfig` struct definition
- [ ] Extract `GGUFConfig::from_gguf()` implementation
- [ ] Update monolith imports
- [ ] Add tests for config extraction
- [ ] Verify builds and tests pass

### Model Extraction (model.rs)
- [ ] Extract `MappedGGUFModel` struct
- [ ] Extract `MappedGGUFModel::from_path()` method
- [ ] Extract `MappedGGUFModel` helper methods
- [ ] Extract `GGUFTransformer` struct
- [ ] Extract `GGUFTransformerLayer` struct
- [ ] Extract transformer loading logic
- [ ] Update monolith imports
- [ ] Add tests for model loading
- [ ] Verify builds and tests pass

### Quantized Extraction (quantized.rs)
- [ ] Extract `QuantizedTensorRef` struct
- [ ] Extract `QKVWeights` enum
- [ ] Extract `OwnedQKVWeights` enum
- [ ] Extract `OwnedQuantizedTensor` struct
- [ ] Extract `OwnedQuantizedLayer` struct
- [ ] Extract all associated methods
- [ ] Update monolith imports
- [ ] Add tests for quantized types
- [ ] Verify builds and tests pass

### Owned Models Extraction (owned.rs)
- [ ] Extract `QuantizedGenerateConfig` struct
- [ ] Extract `OwnedQuantizedKVCache` struct
- [ ] Extract `OwnedQuantizedModel` struct
- [ ] Extract `OwnedQuantizedModelCached` struct
- [ ] Extract `OwnedQuantizedModelCachedSync` struct
- [ ] Extract `OwnedQuantizedModelCuda` struct (feature-gated)
- [ ] Extract all inference implementations
- [ ] Update monolith imports
- [ ] Add tests for owned models
- [ ] Verify builds and tests pass

## Phase 3: Testing & Documentation (TODO)

### Per-Module Tests
- [ ] types.rs - Test all constants and parsing
- [ ] config.rs - Test config extraction from various models
- [ ] model.rs - Test model loading and memory mapping
- [ ] quantized.rs - Test quantized tensor operations
- [ ] owned.rs - Test owned model inference

### Documentation
- [ ] Add rustdoc examples to all public types
- [ ] Document error cases
- [ ] Add module-level documentation
- [ ] Create architecture diagram
- [ ] Update CLAUDE.md with new structure

### Coverage Verification
- [ ] Run `cargo llvm-cov --lib`
- [ ] Verify coverage ≥ 85%
- [ ] Add tests for uncovered branches
- [ ] Run mutation testing on extracted modules

## Phase 4: Cleanup (TODO)

### Monolith Cleanup
- [ ] Remove extracted type definitions from monolith
- [ ] Remove extracted implementations from monolith
- [ ] Keep only unextracted code
- [ ] Verify monolith still compiles with imports

### Final Verification
- [ ] All builds pass (lib, cuda, all features)
- [ ] All tests pass (6000+ tests expected)
- [ ] No clippy warnings
- [ ] Coverage ≥ 85%
- [ ] No performance regressions
- [ ] Documentation complete

### Optional: Delete Monolith
- [ ] Verify all code extracted
- [ ] Remove `src/gguf_monolith.rs`
- [ ] Update `src/gguf/mod.rs` to not include monolith
- [ ] Final verification that everything works

## Current Status

**Phase**: 1 of 4
**Progress**: 100% of Phase 1 complete
**Tests**: 5,684 passing, 0 failing
**Coverage**: 80.97% (maintained)
**Next**: Begin Phase 2 - Extract constants and simple types

## Commands for Next Steps

```bash
# Before starting extraction
git status  # Ensure clean working directory
cargo test --lib  # Baseline test run

# After each extraction
cargo build --lib
cargo build --lib --features cuda
cargo test --lib
cargo clippy --lib

# Check coverage impact
cargo llvm-cov --lib --no-report
```

## Risk Assessment

**Phase 1 Risk**: ✅ NONE - All changes additive and backward compatible
**Phase 2 Risk**: 🟡 LOW-MEDIUM - Type extraction may introduce temporary breakage
**Phase 3 Risk**: 🟢 LOW - Testing and docs are additive
**Phase 4 Risk**: 🟡 MEDIUM - Monolith removal must be careful

## Success Metrics

- **Phase 1**: ✅ Module structure exists, all tests pass
- **Phase 2**: All types extracted, monolith imports from modules, tests pass
- **Phase 3**: ≥85% coverage, comprehensive docs, all tests pass
- **Phase 4**: Monolith deleted, codebase clean, no regressions

## Notes

- Take incremental approach - extract one type at a time
- Always verify builds and tests after each extraction
- Keep monolith compiling throughout (imports from new modules)
- Document as you extract (rustdoc examples)
- Measure coverage impact after each major extraction
- Use `git` to checkpoint after each successful extraction
