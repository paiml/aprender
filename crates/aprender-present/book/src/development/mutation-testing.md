# Mutation Testing

Measure test effectiveness by introducing bugs.

## Concept

```
Original: x + y
Mutant 1: x - y  // Does a test fail?
Mutant 2: x * y  // Does a test fail?
```

If tests pass with a mutant, tests are weak.

## Running

```bash
cargo mutants --timeout 300
```

## Mutation Operators

| Operator | Original | Mutant |
|----------|----------|--------|
| Arithmetic | `+` | `-`, `*`, `/` |
| Comparison | `<` | `<=`, `>`, `>=` |
| Boolean | `true` | `false` |
| Return | `return x` | `return default` |
| Boundary | `< 10` | `<= 10` |

## Example Output

```
Found 150 mutants

Killed: 142 (94.7%)
Survived: 8 (5.3%)

Survived mutants:
- src/button.rs:45: changed `+` to `-`
- src/slider.rs:78: changed `<` to `<=`
```

## Fixing Survivors

Add tests that catch the mutation:

```rust
// Original (mutation survived)
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min { min }
    else if x > max { max }
    else { x }
}

// Add boundary test
#[test]
fn test_clamp_boundary() {
    assert_eq!(clamp(10.0, 0.0, 10.0), 10.0);  // Exactly at max
    assert_eq!(clamp(0.0, 0.0, 10.0), 0.0);    // Exactly at min
}
```

## Target Score

**Minimum: 80%** mutation coverage

## Integration

```makefile
# In tier3
tier3: tier2 coverage
    @cargo mutants --timeout 300
```

## Verified Test

```rust
#[test]
fn test_mutation_detection() {
    // This test catches arithmetic mutations
    fn add(a: i32, b: i32) -> i32 { a + b }

    assert_eq!(add(2, 3), 5);
    assert_eq!(add(-1, 1), 0);
    assert_eq!(add(0, 0), 0);
    // All mutations (-, *, /) would fail these tests
}
```
