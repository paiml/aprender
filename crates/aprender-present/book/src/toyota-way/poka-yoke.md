# Poka Yoke

Mistake-proofing through design.

## Principle

| Japanese | English | Application |
|----------|---------|-------------|
| Poka | Inadvertent mistake | User/developer error |
| Yoke | Prevention | Type system, validation |

## Types of Prevention

| Level | Mechanism | Example |
|-------|-----------|---------|
| Compile | Type system | `Size` vs `f32` |
| Runtime | Validation | Constraint bounds |
| Design | API shape | Builder pattern |

## Type-Level Poka Yoke

```rust
// Bad: Can mix up width and height
fn create_rect(width: f32, height: f32) -> Rect

// Good: Named types prevent mistakes
fn create_rect(size: Size) -> Rect
```

## State Machine Poka Yoke

```rust
// Bad: Can call methods in wrong order
widget.paint();  // Before layout!

// Good: Type states enforce order
fn layout(self) -> LayoutWidget { ... }
fn paint(&self) { ... }  // Only on LayoutWidget
```

## Builder Pattern

```rust
// Prevents forgetting required fields
Button::builder()
    .label("Click")  // Required
    .on_click(handler)  // Required
    .build()  // Only compiles with required fields
```

## Validation

```rust
impl Constraints {
    pub fn new(min_w: f32, max_w: f32, min_h: f32, max_h: f32) -> Self {
        // Poka-yoke: prevent invalid constraints
        assert!(min_w <= max_w, "min_width must be <= max_width");
        assert!(min_h <= max_h, "min_height must be <= max_height");
        Self { min_w, max_w, min_h, max_h }
    }
}
```

## Verified Test

```rust
#[test]
fn test_poka_yoke_type_safety() {
    use presentar_core::Size;

    // Poka-yoke: Size type prevents mixing up width/height
    let size = Size::new(100.0, 50.0);

    // Clear which is which
    assert_eq!(size.width, 100.0);
    assert_eq!(size.height, 50.0);

    // Can't accidentally swap values like with (f32, f32)
    fn requires_size(s: Size) -> f32 {
        s.width + s.height
    }

    assert_eq!(requires_size(size), 150.0);
}

#[test]
#[should_panic(expected = "min_width must be <= max_width")]
fn test_poka_yoke_constraint_validation() {
    use presentar_core::Constraints;

    // Invalid constraints panic at construction
    let _invalid = Constraints::new(100.0, 50.0, 0.0, 100.0);
}
```
