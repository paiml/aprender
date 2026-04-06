# Property Based Testing

Generate test inputs automatically to find edge cases.

## Concept

| Approach | Input | Coverage |
|----------|-------|----------|
| Unit test | Hand-picked | Limited |
| Property test | Generated | Extensive |

## Key Properties

### 1. Idempotence

```rust
// Laying out twice gives same result
fn property_layout_idempotent(widget: &mut Widget, size: Size) {
    widget.layout(size);
    let result1 = widget.bounds();
    widget.layout(size);
    let result2 = widget.bounds();
    assert_eq!(result1, result2);
}
```

### 2. Commutativity

```rust
// Order of sibling widgets doesn't affect total size
fn property_children_order(children: Vec<Widget>) {
    let sum1: f32 = children.iter().map(|c| c.width()).sum();
    let reversed: Vec<_> = children.into_iter().rev().collect();
    let sum2: f32 = reversed.iter().map(|c| c.width()).sum();
    assert_eq!(sum1, sum2);
}
```

### 3. Bounds Respect Constraints

```rust
// Widget size stays within constraints
fn property_respects_constraints(constraints: Constraints, size: Size) {
    let result = constraints.constrain(size);
    assert!(result.width >= constraints.min_width);
    assert!(result.width <= constraints.max_width);
    assert!(result.height >= constraints.min_height);
    assert!(result.height <= constraints.max_height);
}
```

## Generator Pattern

```rust
fn arbitrary_size() -> Size {
    // Deterministic pseudo-random generation
    let seed = 12345u64;
    let width = (seed % 1000) as f32;
    let height = ((seed >> 16) % 1000) as f32;
    Size::new(width, height)
}
```

## Test Structure

| Phase | Action |
|-------|--------|
| Generate | Create random inputs |
| Execute | Run function under test |
| Assert | Check property holds |
| Shrink | Find minimal failing case |

## Verified Test

```rust
#[test]
fn test_property_constraints_bounds() {
    use presentar_core::{Constraints, Size};

    // Property: constrain always returns value within bounds
    let test_cases = [
        (Size::new(50.0, 50.0), Constraints::new(0.0, 100.0, 0.0, 100.0)),
        (Size::new(150.0, 150.0), Constraints::new(0.0, 100.0, 0.0, 100.0)),
        (Size::new(0.0, 0.0), Constraints::new(10.0, 100.0, 10.0, 100.0)),
    ];

    for (size, constraints) in test_cases {
        let result = constraints.constrain(size);
        assert!(result.width >= constraints.min_width);
        assert!(result.width <= constraints.max_width);
        assert!(result.height >= constraints.min_height);
        assert!(result.height <= constraints.max_height);
    }
}
```
