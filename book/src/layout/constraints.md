# Constraints

Layout constraints define minimum and maximum sizes for widgets.

## Structure

```rust
pub struct Constraints {
    pub min_width: f32,
    pub max_width: f32,
    pub min_height: f32,
    pub max_height: f32,
}
```

## Constructors

### Tight (Exact Size)

```rust
use presentar_core::{Constraints, Size};

let c = Constraints::tight(Size::new(100.0, 50.0));
// min_width = max_width = 100.0
// min_height = max_height = 50.0
```

### Loose (Up to Maximum)

```rust
let c = Constraints::loose(Size::new(400.0, 300.0));
// min_width = 0, max_width = 400.0
// min_height = 0, max_height = 300.0
```

### Unbounded

```rust
let c = Constraints::unbounded();
// min = 0, max = INFINITY
```

## Methods

| Method | Description |
|--------|-------------|
| `constrain(size)` | Clamp size to constraints |
| `is_tight()` | True if min == max |
| `is_bounded()` | True if max is finite |
| `biggest()` | Maximum allowed size |
| `smallest()` | Minimum allowed size |
| `deflate(h, v)` | Subtract from all bounds |

## Constrain Usage

```rust
fn measure(&self, constraints: Constraints) -> Size {
    let desired = Size::new(200.0, 100.0);
    constraints.constrain(desired) // Clamp to bounds
}
```

## Builder Methods

```rust
let c = Constraints::unbounded()
    .with_min_width(100.0)
    .with_max_width(500.0)
    .with_min_height(50.0)
    .with_max_height(200.0);
```

## Verified Test

```rust
#[test]
fn test_constraints_constrain() {
    use presentar_core::{Constraints, Size};

    let c = Constraints::new(10.0, 100.0, 20.0, 80.0);

    // Within bounds
    assert_eq!(c.constrain(Size::new(50.0, 50.0)), Size::new(50.0, 50.0));

    // Below minimum
    assert_eq!(c.constrain(Size::new(5.0, 5.0)), Size::new(10.0, 20.0));

    // Above maximum
    assert_eq!(c.constrain(Size::new(200.0, 200.0)), Size::new(100.0, 80.0));
}
```
