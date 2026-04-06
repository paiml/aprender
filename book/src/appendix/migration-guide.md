# Migration Guide

Upgrading between Presentar versions.

## General Strategy

1. Read changelog for breaking changes
2. Update dependencies
3. Run tests to find issues
4. Fix compile errors
5. Run visual regression tests
6. Verify accessibility

## Widget Trait Changes

### Old (pre-0.1)

```rust
trait Widget {
    fn render(&self, canvas: &mut Canvas);
}
```

### New (0.1+)

```rust
trait Widget {
    fn measure(&self, constraints: &Constraints) -> Size;
    fn layout(&mut self, size: Size);
    fn paint(&self, canvas: &mut dyn Canvas);
}
```

## Migration Steps

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `cargo update` | Update dependencies |
| 2 | `cargo check` | Find compile errors |
| 3 | `cargo test` | Run test suite |
| 4 | `make test-visual` | Check visual regression |

## Common Fixes

### Canvas Method Renames

| Old | New |
|-----|-----|
| `fill_rect` | `fill_rect` (unchanged) |
| `draw_text` | `draw_text` (unchanged) |
| `render()` | `paint()` |

### Size/Constraints

```rust
// Old
let size = (100.0, 50.0);

// New
let size = Size::new(100.0, 50.0);
let constraints = Constraints::tight(size);
```

## Verified Test

```rust
#[test]
fn test_migration_size_creation() {
    use presentar_core::Size;

    // New API for size creation
    let size = Size::new(100.0, 50.0);
    assert_eq!(size.width, 100.0);
    assert_eq!(size.height, 50.0);

    // Size is Copy
    let size2 = size;
    assert_eq!(size, size2);
}
```
