# Case Study: Validated Tensors — Compile-Time Contract Enforcement

## Overview

Demonstrates the Poka-Yoke (mistake-proofing) pattern for tensor validation. This makes
it impossible to use invalid tensor data at compile time by encoding invariants in the
type system.

**Run command:**
```bash
cargo run --example validated_tensors
```

## Theoretical Foundation

- Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*
- Brady, E. (2017). *Type-Driven Development with Idris*
- Parsons, A. (2019). "Parse, Don't Validate"

## Key Types

| Type | Invariant | Validation |
|------|-----------|------------|
| `ValidatedEmbedding` | Non-zero density, finite values | Density > threshold, no NaN/Inf |
| `ValidatedWeight` | Proper dimensions, finite values | Shape matches config, no NaN |
| `ValidatedVector` | Non-empty, finite values | Length > 0, all values finite |

Inner data is private — there is no way to construct these types without passing validation.

## Usage

```rust
use aprender::format::{
    ValidatedEmbedding, ValidatedWeight, ValidatedVector,
    ContractValidationError,
};

fn main() {
    // Valid embedding passes all gates
    let data = vec![0.1, 0.2, 0.3, 0.4];
    let embedding = ValidatedEmbedding::new(data, 2, 2)
        .expect("validation passed");

    // Invalid embedding (all zeros) is rejected at construction
    let zeros = vec![0.0; 4];
    let result = ValidatedEmbedding::new(zeros, 2, 2);
    assert!(result.is_err()); // Density too low

    // ValidatedWeight enforces shape contract
    let weight = ValidatedWeight::new(data.clone(), 2, 2)
        .expect("valid weight matrix");

    // NaN values are rejected
    let bad = vec![f32::NAN, 0.1, 0.2, 0.3];
    let result = ValidatedWeight::new(bad, 2, 2);
    assert!(result.is_err()); // Contains NaN
}
```

## Why Poka-Yoke?

Traditional validation:
```rust
// BAD: validation at use site — easy to forget
fn inference(weights: &[f32]) {
    assert!(!weights.is_empty());           // runtime crash
    assert!(weights.iter().all(|v| v.is_finite())); // runtime crash
}
```

Poka-Yoke validation:
```rust
// GOOD: validation at construction — impossible to forget
fn inference(weights: &ValidatedWeight) {
    // weights are GUARANTEED valid by the type system
    // no runtime checks needed
}
```

## See Also

- [APR Poka-Yoke Validation](./poka-yoke-validation.md)
- [Type Safety Best Practices](../best-practices/type-safety.md)
- [APR Format Deep Dive](./apr-format-deep-dive.md)
