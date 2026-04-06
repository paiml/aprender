# Memory Management

Efficient memory use in WASM.

## WASM Memory Model

- Linear memory (one big array)
- No garbage collector
- Manual/RAII management

## Allocation Strategy

```
Prefer:
1. Stack allocation (no cost)
2. Arena allocation (bulk free)
3. Heap allocation (last resort)
```

## Stack Allocation

```rust
// Good: Stack allocated
let size = Size::new(100.0, 50.0);

// Avoid: Unnecessary heap
let size = Box::new(Size::new(100.0, 50.0));
```

## Reuse Buffers

```rust
// Good: Reuse
let mut canvas = RecordingCanvas::new();
for frame in frames {
    canvas.clear();
    widget.paint(&mut canvas);
}

// Bad: Allocate per frame
for frame in frames {
    let mut canvas = RecordingCanvas::new();
    widget.paint(&mut canvas);
}
```

## Widget Memory

| Widget | Stack | Heap |
|--------|-------|------|
| Text | Label, style | String content |
| Button | State, colors | Label string |
| Column | Alignment | Children Vec |

## Minimizing Allocations

```rust
// Use &str when possible
fn new(label: &str) -> Self;

// Use SmallVec for small collections
use smallvec::SmallVec;
children: SmallVec<[Box<dyn Widget>; 4]>
```

## Profiling

```rust
// Track allocations in tests
let before = std::alloc::get_allocations();
widget.paint(&mut canvas);
let after = std::alloc::get_allocations();
assert!(after - before < 100);
```

## Verified Test

```rust
#[test]
fn test_memory_efficiency() {
    use std::mem::size_of;
    use presentar_core::{Size, Point, Rect};

    // Core types are small
    assert!(size_of::<Size>() <= 16);
    assert!(size_of::<Point>() <= 16);
    assert!(size_of::<Rect>() <= 32);
}
```
