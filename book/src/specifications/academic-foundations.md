# Academic Foundations

Research principles behind Presentar.

## Core Papers

| Topic | Foundation |
|-------|------------|
| Layout | CSS Flexbox Specification |
| Rendering | Retained Mode Graphics |
| Testing | Mutation Testing (DeMillo 1978) |
| Quality | Software Metrics (Chidamber-Kemerer) |

## Layout Algorithm

Based on CSS Flexbox with simplifications:

1. **Measure Phase**: Bottom-up intrinsic sizing
2. **Layout Phase**: Top-down constraint solving
3. **Paint Phase**: Sequential draw command generation

```
constraints(parent) → measure(child) → size(child) → layout(child)
```

## Constraint Propagation

From Flutter's RenderObject model:

```rust
// Constraints flow down
fn layout(&mut self, constraints: &Constraints) -> Size {
    let child_constraints = self.compute_child_constraints(constraints);
    let child_size = self.child.layout(&child_constraints);
    // Size flows up
    self.compute_size(child_size, constraints)
}
```

## Accessibility

WCAG 2.1 guidelines with focus on:

- Perceivable (contrast ratios)
- Operable (keyboard navigation)
- Understandable (predictable behavior)
- Robust (valid semantics)

## Quality Scoring

Inspired by GQM (Goal-Question-Metric):

```
Goal: Ship high-quality UI
├─ Question: Is it accessible?
│  └─ Metric: WCAG AA compliance
├─ Question: Is it performant?
│  └─ Metric: Frame time <16ms
└─ Question: Is it tested?
   └─ Metric: >95% coverage
```

## Verified Test

```rust
#[test]
fn test_academic_constraint_propagation() {
    use presentar_core::{Constraints, Size};

    // Parent provides constraints
    let parent_constraints = Constraints::new(0.0, 200.0, 0.0, 100.0);

    // Child measures within constraints
    let child_intrinsic = Size::new(150.0, 80.0);
    let child_size = parent_constraints.constrain(child_intrinsic);

    assert_eq!(child_size.width, 150.0);  // Within bounds
    assert_eq!(child_size.height, 80.0);  // Within bounds

    // Overflow case
    let overflow = Size::new(300.0, 150.0);
    let clamped = parent_constraints.constrain(overflow);
    assert_eq!(clamped.width, 200.0);  // Clamped to max
    assert_eq!(clamped.height, 100.0); // Clamped to max
}
```
