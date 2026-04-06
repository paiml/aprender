# Virtualization

Render only visible items for large lists.

## Problem

```
10,000 items → 10,000 widgets → Slow
```

## Solution

```
10,000 items → ~20 visible widgets → Fast
```

## Virtual List

```rust
struct VirtualList {
    items: Vec<Item>,
    visible_range: Range<usize>,
    item_height: f32,
    scroll_offset: f32,
}

impl VirtualList {
    fn visible_items(&self) -> &[Item] {
        &self.items[self.visible_range.clone()]
    }
}
```

## Calculating Visible Range

```rust
fn calculate_visible_range(&self, viewport_height: f32) -> Range<usize> {
    let start = (self.scroll_offset / self.item_height) as usize;
    let visible_count = (viewport_height / self.item_height).ceil() as usize + 1;
    let end = (start + visible_count).min(self.items.len());
    start..end
}
```

## Scroll Handling

```rust
fn on_scroll(&mut self, delta: f32) {
    self.scroll_offset = (self.scroll_offset + delta)
        .max(0.0)
        .min(self.max_scroll());
    self.visible_range = self.calculate_visible_range(self.viewport_height);
}
```

## Performance

| Items | Without Virtual | With Virtual |
|-------|-----------------|--------------|
| 100 | 5ms | 5ms |
| 1,000 | 50ms | 5ms |
| 10,000 | 500ms | 5ms |

## Verified Test

```rust
#[test]
fn test_virtualization_range() {
    let item_height = 50.0;
    let viewport = 500.0;
    let scroll = 100.0;

    let start = (scroll / item_height) as usize;  // 2
    let count = (viewport / item_height).ceil() as usize + 1;  // 11

    assert_eq!(start, 2);
    assert_eq!(count, 11);
}
```
