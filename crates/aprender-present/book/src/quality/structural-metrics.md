# Structural Metrics

Code and widget tree quality measures.

## Widget Tree Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Max depth | ≤10 | Tree nesting levels |
| Max children | ≤50 | Direct children count |
| Total nodes | ≤1000 | Without virtualization |

## Code Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Cyclomatic complexity | ≤10 | Branches per function |
| Cognitive complexity | ≤15 | Mental burden |
| Function length | ≤50 lines | Readability |
| File length | ≤500 lines | Maintainability |

## Tree Depth Analysis

```rust
fn max_depth(widget: &dyn Widget) -> usize {
    let child_depths: Vec<usize> = widget.children()
        .iter()
        .map(|c| max_depth(c.as_ref()))
        .collect();

    1 + child_depths.into_iter().max().unwrap_or(0)
}
```

## Node Count

```rust
fn node_count(widget: &dyn Widget) -> usize {
    1 + widget.children()
        .iter()
        .map(|c| node_count(c.as_ref()))
        .sum::<usize>()
}
```

## Complexity Warning

| Depth | Status |
|-------|--------|
| 1-5 | Good |
| 6-10 | Acceptable |
| 11-15 | Warning |
| >15 | Refactor |

## Refactoring Patterns

```rust
// Before: Deep nesting
Column {
    children: vec![Row { children: vec![Column { ... }] }]
}

// After: Extract component
struct MyComponent { ... }
impl Widget for MyComponent { ... }
```

## Verified Test

```rust
#[test]
fn test_structural_tree_depth() {
    // Recursive depth calculation
    fn depth(levels: &[usize]) -> usize {
        if levels.is_empty() {
            0
        } else {
            1 + levels.iter().max().copied().unwrap_or(0)
        }
    }

    // Flat tree: depth 1
    assert_eq!(depth(&[]), 0);
    assert_eq!(depth(&[0, 0, 0]), 1);

    // Nested tree: depth = 1 + max child depth
    assert_eq!(depth(&[1, 2, 1]), 3);  // 1 + 2

    // Deep tree warning threshold
    let max_recommended = 10;
    let actual_depth = 8;
    assert!(actual_depth <= max_recommended);
}
```
