# Property-Based Testing

Realizar uses **proptest** for generative falsification—automatically generating millions of test inputs to find edge cases that handwritten tests miss.

## Philosophy: Generative Falsification

Traditional unit tests verify specific cases. Property-based tests define *invariants* that must hold for all inputs, then generate random inputs to find counterexamples.

```rust
// Traditional: Test specific case
#[test]
fn test_parse_valid_header() {
    let data = create_valid_gguf_header();
    assert!(GGUFModel::from_bytes(&data).is_ok());
}

// Property-based: Test invariant across all inputs
proptest! {
    #[test]
    fn fuzz_gguf_header(header in arb_gguf_header()) {
        // Should never panic, regardless of input
        let _ = GGUFModel::from_bytes(&header);
    }
}
```

## Proptest in Realizar

The T-COV-95 campaign added 33 proptest tests across three modules:

| Module | Tests | Cases/Test | Total Cases |
|--------|-------|------------|-------------|
| `gguf/tests/part_34.rs` | 12 | 200-1000 | ~4,000,000 |
| `api/tests/part_27.rs` | 12 | 100-500 | ~2,000,000 |
| `convert/tests_part_09.rs` | 9 | 100-300 | ~1,500,000 |

## Writing Property Tests

### 1. Define Strategies

Strategies generate arbitrary values with weighted distributions:

```rust
use proptest::prelude::*;

/// Generate arbitrary GGUF magic numbers (valid and invalid)
fn arb_magic() -> impl Strategy<Value = u32> {
    prop_oneof![
        3 => Just(GGUF_MAGIC),           // Valid magic (weighted)
        1 => Just(0x46554746),           // "FUFG" - almost valid
        1 => Just(0x47475546),           // "GGUF" wrong endian
        1 => Just(0x00000000),           // Zero
        1 => Just(0xFFFFFFFF),           // All ones
        1 => any::<u32>(),               // Random
    ]
}
```

### 2. Compose Complex Strategies

Build complex data structures from simple strategies:

```rust
fn arb_gguf_header() -> impl Strategy<Value = Vec<u8>> {
    (arb_magic(), arb_version(), arb_tensor_count(), arb_metadata_count())
        .prop_map(|(magic, version, tensor_count, metadata_count)| {
            let mut data = Vec::with_capacity(24);
            data.extend_from_slice(&magic.to_le_bytes());
            data.extend_from_slice(&version.to_le_bytes());
            data.extend_from_slice(&tensor_count.to_le_bytes());
            data.extend_from_slice(&metadata_count.to_le_bytes());
            data
        })
}
```

### 3. Define Property Tests

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn fuzz_gguf_header(header in arb_gguf_header()) {
        // Property: Parser never panics
        let result = GGUFModel::from_bytes(&header);

        // Property: Invalid magic always fails
        if header.len() >= 4 {
            let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
            if magic != GGUF_MAGIC {
                prop_assert!(result.is_err());
            }
        }
    }
}
```

## Testing Patterns

### Byte-Smasher: Bit-Flip Fuzzing

Flip individual bits in valid data to test corruption handling:

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn byte_smasher_single_bit_flip(
        byte_idx in 0usize..8,  // Only magic/version bytes
        bit_idx in 0u8..8
    ) {
        let mut data = create_valid_minimal_gguf();
        data[byte_idx] ^= 1 << bit_idx;

        let result = GGUFModel::from_bytes(&data);
        // Magic corruption should always fail
        if byte_idx < 4 {
            prop_assert!(result.is_err());
        }
    }
}
```

### Dimension Permutation

Test all combinations of tensor dimensions:

```rust
fn arb_tensor_dims() -> impl Strategy<Value = (u32, Vec<u64>)> {
    (0u32..=10).prop_flat_map(|n_dims| {
        let dims = prop::collection::vec(
            prop_oneof![
                3 => 1u64..100,           // Small (common)
                1 => Just(0u64),          // Zero dimension
                1 => Just(1u64),          // Singleton
                1 => 100u64..10000,       // Large
            ],
            n_dims as usize,
        );
        dims.prop_map(move |d| (n_dims, d))
    })
}
```

### Metadata Type Exhaustion

Test all valid and invalid metadata types:

```rust
const GGUF_TYPES: [u32; 13] = [
    0,  // UINT8
    1,  // INT8
    // ... all valid types
    12, // FLOAT64
];

proptest! {
    #[test]
    fn fuzz_invalid_metadata_type(invalid_type in 13u32..256) {
        // Create GGUF with invalid type
        let result = parse_with_type(invalid_type);
        prop_assert!(result.is_err());
    }
}
```

## Security Discovery: Allocation Attacks

Proptest discovered a critical vulnerability: corrupted `tensor_count` values could cause multi-terabyte allocation attempts.

**Before (vulnerable):**
```rust
let tensors = Vec::with_capacity(tensor_count as usize);  // OOM!
```

**After (fixed):**
```rust
const MAX_TENSOR_COUNT: u64 = 100_000;
if tensor_count > MAX_TENSOR_COUNT {
    return Err(/* bounds check error */);
}
```

This demonstrates the power of generative testing—it found a security bug that manual tests missed.

## Best Practices

1. **Bound extreme values**: Avoid `u64::MAX` in strategies; use bounded ranges
2. **Weight common cases**: Use `prop_oneof!` with weights for realistic distributions
3. **Test invariants, not implementations**: Focus on what must always be true
4. **Limit byte-smasher scope**: Only corrupt header bytes to avoid OOM from corrupted counts

## Running Property Tests

```bash
# Run all tests including proptest
cargo test

# Run with more cases (slower, more thorough)
PROPTEST_CASES=10000 cargo test

# Run specific proptest module
cargo test --lib gguf::tests::part_34
```

## Related

- [proptest Tool](../tools/proptest.md) - Installation and configuration
- [Coverage](../quality/coverage.md) - Coverage measurement
- [Mutation Testing](../quality/mutation.md) - Verifying test quality
