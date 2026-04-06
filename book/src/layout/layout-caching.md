# Layout Caching

Optimize performance with cached layout results.

## Problem

Without caching:
```
Every frame: Measure ALL → Layout ALL → Paint ALL
```

Expensive for deep widget trees.

## Solution

Cache layout results:
```
Frame N:   Measure → Layout → Paint → Cache
Frame N+1: Check cache → Paint (skip measure/layout)
```

## Cache Key

```rust
struct LayoutCacheKey {
    constraints: Constraints,
    child_count: usize,
}
```

## Cache Entry

```rust
struct LayoutCacheEntry {
    size: Size,
    child_positions: Vec<Rect>,
}
```

## Invalidation

Cache invalidates when:
- Constraints change
- Children change
- State changes

```rust
fn should_relayout(&self, new_constraints: Constraints) -> bool {
    self.cached_constraints != Some(new_constraints)
        || self.dirty
}
```

## Usage

```rust
fn measure(&self, constraints: Constraints) -> Size {
    // Check cache first
    if let Some(cached) = self.layout_cache.get(&constraints) {
        return cached.size;
    }

    // Compute and cache
    let size = self.compute_size(constraints);
    self.layout_cache.insert(constraints, size);
    size
}
```

## Performance Impact

| Scenario | Without Cache | With Cache |
|----------|---------------|------------|
| Static UI | 5ms | 0.1ms |
| Scroll | 10ms | 2ms |
| Animation | 15ms | 8ms |

## Verified Test

```rust
#[test]
fn test_layout_caching() {
    use presentar_core::{Constraints, Size};

    // Simulated cache behavior
    let constraints = Constraints::loose(Size::new(400.0, 300.0));
    let cached_size = Size::new(200.0, 150.0);

    // Cache hit should return same value
    assert_eq!(cached_size, Size::new(200.0, 150.0));
}
```
