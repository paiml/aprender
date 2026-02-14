# Jidoka (Autonomation)

Jidoka is a Toyota Production System principle meaning "automation with a human touch."
When a defect is detected, the process stops immediately rather than producing more defective items.

In software, Jidoka means **making invalid states unrepresentable** at the type level.

## Theoretical Foundation

- **Shingo, S. (1986)**. *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press.
- **Brady, E. (2017)**. *Type-Driven Development with Idris*. Manning Publications.
- **Parsons, A. (2019)**. "Parse, Don't Validate" - https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/

## The Problem: Runtime Validation Can Be Bypassed

Traditional runtime validation has a fatal flaw: it can be bypassed.

```rust
// DANGEROUS: Nothing prevents using unvalidated data
fn process_embedding(data: Vec<f32>, vocab_size: usize, hidden_dim: usize) {
    // Validation can be skipped or forgotten
    if !validate_embedding(&data, vocab_size, hidden_dim) {
        return; // Easy to forget this check
    }
    // ... process data
}

// This compiles and runs, even with garbage data:
process_embedding(vec![0.0; 100], 151936, 896); // Wrong size!
```

## The Solution: Poka-Yoke via Newtypes

Poka-Yoke (mistake-proofing) makes the invalid state **impossible to represent**:

```rust
/// Validated embedding - inner data is PRIVATE
pub struct ValidatedEmbedding {
    data: Vec<f32>,      // PRIVATE - cannot be accessed directly
    vocab_size: usize,
    hidden_dim: usize,
}

impl ValidatedEmbedding {
    /// The ONLY way to create a ValidatedEmbedding
    pub fn new(data: Vec<f32>, vocab_size: usize, hidden_dim: usize)
        -> Result<Self, ContractValidationError>
    {
        // All validation gates run here
        if data.len() != vocab_size * hidden_dim {
            return Err(/* shape error */);
        }
        if zero_pct(&data) > 50.0 {
            return Err(/* density error - catches PMAT-234 bug */);
        }
        // ... more gates

        Ok(Self { data, vocab_size, hidden_dim })
    }

    /// Access validated data
    pub fn data(&self) -> &[f32] { &self.data }
}
```

Now invalid data **cannot exist**:

```rust
// This function REQUIRES validated data - cannot be bypassed
fn process_embedding(embedding: ValidatedEmbedding) {
    // Compiler guarantees data passed validation
    let data = embedding.data(); // Safe to use
}

// Cannot call with invalid data - compilation error!
process_embedding(vec![0.0; 100]); // ERROR: expected ValidatedEmbedding
```

## Validation Gates

The contract defines these validation gates (see `contracts/tensor-layout-v1.yaml`):

| Gate ID | Rule | Threshold |
|---------|------|-----------|
| F-DATA-QUALITY-001 | Embedding density | < 50% zeros |
| F-DATA-QUALITY-002 | No NaN/Inf | count = 0 |
| F-DATA-QUALITY-003 | Non-degenerate | L2 > 1e-6 |
| F-DATA-QUALITY-004 | Spot check | tokens at 10/50/90% non-zero |

## Real Bug: PMAT-234

This pattern caught a real bug where SafeTensors data had 94.5% leading zeros due to an offset calculation error:

```
# Trace output showing the bug
embedding len: 136134656 floats
first non-zero at index 128684288: value=-0.0131
leading zeros: 128684288 (94.5% of total)
```

Without validation, this data would have been used for inference, producing garbage output (Hebrew characters instead of English).

With `ValidatedEmbedding`, the load fails immediately with:

```
[F-DATA-QUALITY-001] Tensor 'embedding': DENSITY FAILURE: 94.5% zeros
(max 50%). Data likely loaded from wrong offset!
```

## Popperian Falsification Tests

Per Popper (1959), each validation rule has explicit falsification tests that attempt to **disprove** the contract works:

```rust
#[test]
fn falsify_001_embedding_rejects_all_zeros() {
    let bad_data = vec![0.0f32; 100 * 64]; // 100% zeros
    let result = ValidatedEmbedding::new(bad_data, 100, 64);
    assert!(result.is_err(), "Should reject 100% zeros");
    assert!(result.unwrap_err().message.contains("DENSITY"));
}
```

If any falsification test passes when it should fail, the contract implementation is broken.

## Running the Example

```bash
cargo run --example validated_tensors
```

Output:

```
═══════════════════════════════════════════════════════════════════
     PMAT-235: Validated Tensors - Compile-Time Enforcement
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│ Demo 1: Valid Embedding (Passes All Gates)                      │
└─────────────────────────────────────────────────────────────────┘

  ✅ ValidatedEmbedding created successfully!
     vocab_size: 100
     hidden_dim: 64
     Statistics:
       elements: 6400
       zero_pct: 0.0%
       min: -0.0998, max: 0.0998
       L2 norm: 6.3489

┌─────────────────────────────────────────────────────────────────┐
│ Demo 2: Density Rejection (Catches PMAT-234 Bug)                │
└─────────────────────────────────────────────────────────────────┘

  Creating embedding with 94.5% zeros (simulates offset bug)...
  ✅ Correctly rejected!
     Rule: F-DATA-QUALITY-001
     Tensor: embedding
     Error: DENSITY FAILURE: 94.5% zeros (max 50%)
```

## Key Takeaways

1. **Make invalid states unrepresentable** - Use newtypes with private fields
2. **Validation at construction** - The only way to create the type runs validation
3. **Compiler enforcement** - Cannot bypass validation because the type requires it
4. **Popperian testing** - Write tests that attempt to disprove correctness

**See also:**
- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)
- [Poka-Yoke Validation Example](../../../examples/poka_yoke_validation.rs)
- [Contract Specification](../../../contracts/tensor-layout-v1.yaml)
