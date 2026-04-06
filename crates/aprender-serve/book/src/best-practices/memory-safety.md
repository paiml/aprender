# Memory Safety

Realizar prioritizes memory safety through Rust's ownership system, bounds checking, and defensive parsing.

## Allocation Attack Prevention

Untrusted input (GGUF files from the internet) can contain malicious values designed to exhaust memory. The T-COV-95 campaign discovered this vulnerability through proptest fuzzing.

### The Problem

A corrupted GGUF header could claim:
```
tensor_count: 18446744073709551615 (u64::MAX)
```

Naive code would attempt to allocate an 18 exabyte vector:
```rust
// VULNERABLE: Trusts untrusted input
let tensors = Vec::with_capacity(tensor_count as usize);  // OOM!
```

### The Solution

Bounds checks before any allocation based on untrusted data:

```rust
// src/gguf/loader.rs

/// Maximum number of tensors allowed (prevents allocation attacks)
const MAX_TENSOR_COUNT: u64 = 100_000;

/// Maximum metadata entries (reasonable for any model)
const MAX_METADATA_COUNT: u64 = 10_000;

/// Maximum tensor dimensions (GGUF spec: typically 1-4)
const MAX_DIMS: u32 = 8;

/// Maximum array length in metadata
const MAX_ARRAY_LEN: u64 = 10_000_000;

// In parse_header():
if tensor_count > MAX_TENSOR_COUNT {
    return Err(RealizarError::UnsupportedOperation {
        operation: "parse_gguf".to_string(),
        reason: format!(
            "tensor_count {} exceeds maximum allowed {} (corrupted header?)",
            tensor_count, MAX_TENSOR_COUNT
        ),
    });
}

if metadata_count > MAX_METADATA_COUNT {
    return Err(RealizarError::UnsupportedOperation {
        operation: "parse_gguf".to_string(),
        reason: format!(
            "metadata_count {} exceeds maximum allowed {} (corrupted header?)",
            metadata_count, MAX_METADATA_COUNT
        ),
    });
}
```

### Bounds Check Locations

| Location | Check | Limit |
|----------|-------|-------|
| `parse_header()` | `tensor_count` | 100,000 |
| `parse_header()` | `metadata_count` | 10,000 |
| `parse_tensor_info()` | `n_dims` | 8 |
| `parse_metadata()` | array length | 10,000,000 |
| `parse_metadata()` | string length | 10,000,000 |

## Zero Unsafe in Public API

Realizar's public API is 100% safe Rust. All `unsafe` code is isolated in the `trueno` dependency for SIMD operations.

```rust
// Public API: No unsafe
pub fn from_bytes(data: &[u8]) -> Result<Self> { ... }

// Internal: Unsafe isolated in trueno
use trueno::Vector;  // trueno handles unsafe SIMD internally
```

## Buffer Bounds Checking

All tensor data access validates bounds before indexing:

```rust
fn get_tensor_data(&self, info: &TensorInfo, data: &[u8]) -> Result<&[u8]> {
    let start = self.tensor_data_start + info.offset as usize;
    let end = start + info.byte_size();

    if end > data.len() {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "Tensor data out of bounds: {}..{} exceeds data length {}",
                start, end, data.len()
            ),
        });
    }

    Ok(&data[start..end])
}
```

## Integer Overflow Protection

Dimension calculations check for overflow before allocation:

```rust
fn calculate_byte_size(dims: &[u64], element_size: usize) -> Result<usize> {
    let mut total: u64 = 1;
    for dim in dims {
        total = total.checked_mul(*dim).ok_or_else(|| {
            RealizarError::InvalidShape {
                reason: "Dimension overflow".to_string(),
            }
        })?;
    }

    let byte_size = total.checked_mul(element_size as u64).ok_or_else(|| {
        RealizarError::InvalidShape {
            reason: "Byte size overflow".to_string(),
        }
    })?;

    usize::try_from(byte_size).map_err(|_| {
        RealizarError::InvalidShape {
            reason: format!("Byte size {} exceeds usize", byte_size),
        }
    })
}
```

## Discovery Through Fuzzing

The allocation attack vulnerability was discovered through property-based testing:

```rust
proptest! {
    #[test]
    fn fuzz_gguf_header(header in arb_gguf_header()) {
        // This test revealed OOM when tensor_count was u64::MAX
        let _ = GGUFModel::from_bytes(&header);
    }
}
```

When proptest generated `tensor_count = u64::MAX`, the process attempted to allocate ~150 exabytes and crashed. This led to the bounds check fix.

## Best Practices

1. **Validate before allocate**: Check all size fields against reasonable limits
2. **Use checked arithmetic**: `checked_mul()`, `checked_add()` for size calculations
3. **Prefer `try_from()`**: Convert sizes safely with proper error handling
4. **Test with fuzzing**: Property-based tests catch edge cases humans miss
5. **Document limits**: Make bounds explicit in code and documentation

## Related

- [Property-Based Testing](../tdd/property-based.md) - Fuzzing methodology
- [Zero Unsafe Code](./zero-unsafe.md) - Unsafe code policy
- [Error Handling](./error-handling.md) - Result-based error handling
