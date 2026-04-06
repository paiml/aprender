# Grid System

CSS Grid-inspired layout for complex arrangements.

## Basic Grid

```yaml
widgets:
  - type: Grid
    columns: 3
    gap: 16
    children:
      - type: Text
        value: "Cell 1"
      - type: Text
        value: "Cell 2"
      - type: Text
        value: "Cell 3"
```

## Column Templates

| Template | Description |
|----------|-------------|
| `"1fr 1fr 1fr"` | 3 equal columns |
| `"200px 1fr"` | Fixed + flexible |
| `"auto 1fr auto"` | Content-sized edges |
| `"repeat(4, 1fr)"` | 4 equal columns |

## Spanning

```yaml
widgets:
  - type: Grid
    columns: "1fr 1fr 1fr"
    children:
      - type: Container
        grid_column: "1 / 3"  # Span 2 columns
        children: [...]
      - type: Container
        grid_row: "1 / 3"     # Span 2 rows
        children: [...]
```

## Responsive Grid

```yaml
widgets:
  - type: Grid
    columns:
      mobile: 1
      tablet: 2
      desktop: 3
    gap: 16
```

## Alignment

| Property | Values |
|----------|--------|
| `justify_items` | start, center, end, stretch |
| `align_items` | start, center, end, stretch |
| `justify_content` | start, center, end, space-between |
| `align_content` | start, center, end, space-between |

## Auto-fill vs Auto-fit

```yaml
# Auto-fill: maintains column count even if empty
columns: "repeat(auto-fill, minmax(200px, 1fr))"

# Auto-fit: collapses empty columns
columns: "repeat(auto-fit, minmax(200px, 1fr))"
```

## Verified Test

```rust
#[test]
fn test_grid_column_widths() {
    // Grid with 3 equal columns
    let container_width = 600.0;
    let gap = 16.0;
    let columns = 3;

    // Total gap space
    let total_gap = gap * (columns as f32 - 1.0);
    let available = container_width - total_gap;
    let column_width = available / columns as f32;

    assert_eq!(total_gap, 32.0);
    assert!((column_width - 189.33).abs() < 0.1);

    // Verify all columns fit
    let total = column_width * columns as f32 + total_gap;
    assert!((total - container_width).abs() < 0.1);
}
```
