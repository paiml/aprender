# PMAT-44: APR Import Truncation

## Status: REALIZAR FIXES COMPLETE, APRENDER BUG FILED

## Issue
APR import from GGUF creates APR file with truncated layers. Original report: 9 out of 24 layers.

## Root Cause Analysis

### Realizar Issues (FIXED)
1. **Missing quantization type handlers** - Q4_0, Q4_1, Q5_0, Q5_1 defaulted to F32 byte calculation
2. **Missing CRC32 checksum** - APR v2 format requires header checksum
3. **Wrong quantization metadata format** - String instead of struct
4. **Unsorted tensor index** - APR v2 format requires sorted tensors

### Aprender Issues (BUG FILED)
1. **`apr tensors` shows only 100 tensors** - File contains 291 but display truncated
   - Root cause: Unknown - tensor index parsing stops early
   - Evidence: Python parsing shows all 291 tensors present and sorted correctly

## Fixes Applied to `src/convert/mod.rs`

### 1. Added CRC32 functions (lines 27-51)
```rust
fn crc32(data: &[u8]) -> u32 { ... }
fn compute_apr_header_checksum(header: &[u8]) -> u32 { ... }
```

### 2. Added missing byte_size handlers (lines 638-645)
```rust
2 => num_elements.div_ceil(32) * 18,   // Q4_0
3 => num_elements.div_ceil(32) * 20,   // Q4_1
6 => num_elements.div_ceil(32) * 22,   // Q5_0
7 => num_elements.div_ceil(32) * 24,   // Q5_1
```

### 3. Fixed quantization metadata format (lines 611-617)
```rust
"quantization": {
    "quant_type": "Q4_K",
    "bits": 4,
    "block_size": 256,
    "symmetric": true
},
```

### 4. Added tensor sorting (line 688)
```rust
raw_tensors.sort_by(|a, b| a.name.cmp(&b.name));
```

### 5. Added checksum computation (lines 758-760)
```rust
let checksum = compute_apr_header_checksum(&header);
header[40..44].copy_from_slice(&checksum.to_le_bytes());
```

## Verification

### Tests
- All 301 converter tests pass
- Zero clippy warnings

### File Integrity Check
```
APR Header:
- Magic: APR\0 ✓
- Version: 2.0 ✓
- Tensor count: 291 ✓
- Checksum: Valid ✓

Tensor Index (Python verification):
- All 291 tensors present ✓
- Tensors sorted alphabetically ✓
- All layers 0-23 present ✓
```

## Aprender Bug
Filed: See GitHub issue in aprender repo
