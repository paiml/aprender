# proptest

[proptest](https://crates.io/crates/proptest) is a property-based testing framework for Rust, inspired by Haskell's QuickCheck.

## Installation

Add to `Cargo.toml`:

```toml
[dev-dependencies]
proptest = "1.4"
```

## Basic Usage

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_addition_commutative(a in any::<i32>(), b in any::<i32>()) {
        prop_assert_eq!(a.wrapping_add(b), b.wrapping_add(a));
    }
}
```

## Configuration

### Test Case Count

```rust
proptest! {
    // Run 1000 cases instead of default 256
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn my_test(x in any::<u32>()) {
        // ...
    }
}
```

### Environment Variable

```bash
# Run 10,000 cases
PROPTEST_CASES=10000 cargo test

# Enable verbose output
PROPTEST_VERBOSE=1 cargo test
```

## Strategies

### Built-in Strategies

```rust
any::<u32>()                    // Any u32
0..100i32                       // Range
"[a-z]{3,10}"                   // Regex string
prop::collection::vec(any::<u8>(), 0..100)  // Vec
```

### Weighted Selection

```rust
fn arb_value() -> impl Strategy<Value = u32> {
    prop_oneof![
        5 => Just(0),           // 50% chance
        3 => 1..100u32,         // 30% chance
        2 => 100..1000u32,      // 20% chance
    ]
}
```

### Custom Strategies

```rust
fn arb_gguf_header() -> impl Strategy<Value = Vec<u8>> {
    (any::<u32>(), any::<u32>(), any::<u64>(), any::<u64>())
        .prop_map(|(magic, version, tensors, metadata)| {
            let mut data = Vec::new();
            data.extend_from_slice(&magic.to_le_bytes());
            data.extend_from_slice(&version.to_le_bytes());
            data.extend_from_slice(&tensors.to_le_bytes());
            data.extend_from_slice(&metadata.to_le_bytes());
            data
        })
}
```

## Assertions

```rust
proptest! {
    #[test]
    fn test_invariants(x in 0..100i32) {
        prop_assert!(x >= 0);
        prop_assert!(x < 100);
        prop_assert_eq!(x * 2, x + x);
    }
}
```

## Shrinking

When a test fails, proptest automatically **shrinks** the failing input to find the minimal counterexample:

```
test failed: assertion failed at line 42
minimal failing input: x = 0
```

## Best Practices

### 1. Avoid Extreme Values

```rust
// Bad: Can cause OOM
fn bad_strategy() -> impl Strategy<Value = u64> {
    any::<u64>()  // Includes u64::MAX
}

// Good: Bounded range
fn good_strategy() -> impl Strategy<Value = u64> {
    0u64..1_000_000
}
```

### 2. Test Invariants, Not Implementations

```rust
// Good: Tests a property that must always hold
#[test]
fn parser_never_panics(data in any::<Vec<u8>>()) {
    let _ = GGUFModel::from_bytes(&data);  // Never panics
}
```

### 3. Use `prop_assume!` for Preconditions

```rust
proptest! {
    #[test]
    fn test_division(a in any::<i32>(), b in any::<i32>()) {
        prop_assume!(b != 0);  // Skip if b is zero
        prop_assert_eq!(a / b * b + a % b, a);
    }
}
```

## Realizar Usage

The T-COV-95 campaign uses proptest for:

- **GGUF header fuzzing**: `src/gguf/tests/part_34.rs`
- **API request fuzzing**: `src/api/tests/part_27.rs`
- **Convert module fuzzing**: `src/convert/tests_part_09.rs`

Example from the codebase:

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn fuzz_gguf_header(header in arb_gguf_header()) {
        let result = GGUFModel::from_bytes(&header);
        match result {
            Ok(_) => { /* valid input */ }
            Err(_) => { /* expected for most random inputs */ }
        }
    }
}
```

## Related

- [Property-Based Testing](../tdd/property-based.md) - Testing philosophy
- [Coverage](../quality/coverage.md) - Coverage measurement
- [Official Documentation](https://docs.rs/proptest)
