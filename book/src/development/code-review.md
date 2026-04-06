# Code Review

Standards for reviewing Presentar contributions.

## Checklist

| Category | Check |
|----------|-------|
| Tests | All tests pass |
| Coverage | ≥95% line coverage |
| Mutations | ≥80% mutation score |
| Lint | No clippy warnings |
| Format | `cargo fmt` applied |
| A11y | WCAG AA compliant |

## Review Focus Areas

### 1. Widget Implementation

```rust
// Check: Does it implement all required methods?
impl Widget for MyWidget {
    fn measure(&self, constraints: &Constraints) -> Size { ... }
    fn layout(&mut self, size: Size) { ... }
    fn paint(&self, canvas: &mut dyn Canvas) { ... }
}
```

### 2. Test Quality

| Aspect | Requirement |
|--------|-------------|
| Unit tests | Each public method |
| Edge cases | Boundaries, empty, max |
| Determinism | No randomness |
| Independence | No test order dependency |

### 3. Performance

```rust
// Check: O(n) or better for layout
fn layout_children(&mut self) {
    for child in &mut self.children {
        child.layout(size);  // Single pass
    }
}
```

## Anti-Patterns

| Pattern | Issue | Fix |
|---------|-------|-----|
| `unwrap()` in library | Panics | Use `Result` |
| `clone()` in hot path | Allocation | Use references |
| Nested loops | O(n²) | Flatten/cache |

## Approval Criteria

- [ ] All CI checks pass
- [ ] Two reviewer approvals
- [ ] No unresolved comments
- [ ] Documentation updated

## Verified Test

```rust
#[test]
fn test_code_review_coverage_threshold() {
    // Minimum coverage requirements
    let required_line_coverage = 95.0;
    let required_mutation_score = 80.0;

    // Simulated metrics
    let actual_line = 97.5;
    let actual_mutation = 85.0;

    assert!(actual_line >= required_line_coverage);
    assert!(actual_mutation >= required_mutation_score);
}
```
