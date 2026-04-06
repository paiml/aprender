# Brick Architecture (PROBAR-SPEC-009)

All Presentar widgets implement the `Brick` trait, enabling "tests define interface" philosophy. This ensures widgets declare their contracts as falsifiable assertions before rendering.

## Overview

The Brick Architecture provides:
- **Assertions**: What the widget promises (TextVisible, ContrastRatio, etc.)
- **Budget**: Performance budget (default 16ms for 60fps)
- **Verification**: Runtime validation before rendering
- **JIDOKA**: Rendering blocked if verification fails

## The Brick Trait

```rust
use presentar::{Brick, BrickAssertion, BrickBudget, BrickVerification};

pub trait Brick {
    /// Name of this brick for debugging
    fn brick_name(&self) -> &'static str;

    /// Assertions this brick promises to satisfy
    fn assertions(&self) -> &[BrickAssertion];

    /// Performance budget for this brick
    fn budget(&self) -> BrickBudget;

    /// Verify all assertions pass
    fn verify(&self) -> BrickVerification;

    /// Returns true if all assertions pass
    fn can_render(&self) -> bool {
        self.verify().is_valid()
    }
}
```

## Built-in Assertions

| Assertion | Description |
|-----------|-------------|
| `TextVisible` | Text content is visible and readable |
| `ContrastRatio(f64)` | WCAG contrast ratio minimum (4.5 for AA) |
| `MinSize { w, h }` | Minimum dimensions in pixels |
| `Accessible` | Meets accessibility requirements |
| `Custom { name, validator_id }` | Custom validation logic |

## Performance Budgets

```rust
// Uniform budget: all operations share 16ms (60fps)
BrickBudget::uniform(16)

// Tiered budget: layout gets more time than paint
BrickBudget {
    layout_ms: 8,
    paint_ms: 4,
    total_ms: 16,
}
```

## Example: Button Widget

```rust
use presentar::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_widgets::Button;

let button = Button::new("Submit");

// Check brick properties
println!("Brick: {}", button.brick_name());      // "Button"
println!("Assertions: {:?}", button.assertions()); // [TextVisible, ContrastRatio(4.5)]
println!("Budget: {:?}", button.budget());        // 16ms

// JIDOKA: Verify before rendering
if button.can_render() {
    // Safe to render
    frame.render(button, area);
} else {
    // Handle verification failure
    let verification = button.verify();
    for (assertion, reason) in &verification.failed {
        eprintln!("Failed: {:?} - {}", assertion, reason);
    }
}
```

## SimpleBrick Helper

For simple widgets, use `SimpleBrick`:

```rust
use presentar_core::brick_widget::{SimpleBrick, BrickWidgetExt};

struct MyWidget {
    text: String,
    brick: SimpleBrick,
}

impl MyWidget {
    fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            brick: SimpleBrick::new("MyWidget")
                .with_assertion(BrickAssertion::TextVisible)
                .with_assertion(BrickAssertion::ContrastRatio(4.5))
                .with_budget(BrickBudget::uniform(16)),
        }
    }
}
```

## JIDOKA Enforcement

The Brick Architecture enforces JIDOKA (built-in quality):

1. **Before Render**: `can_render()` must return `true`
2. **Verification**: All assertions checked
3. **Block on Failure**: Rendering stops if verification fails
4. **Debug Info**: Failed assertions report reason

This prevents bugs from shipping by catching issues at render time, not in production.

## Best Practices

1. **Declare assertions upfront** - Define what each widget must satisfy
2. **Use realistic budgets** - 16ms for 60fps, 8ms for 120fps
3. **Test with `can_render()`** - Verify constraints before rendering
4. **Use custom assertions** - For domain-specific requirements
5. **Monitor verification time** - Keep under 1ms per widget

## See Also

- [Widget Trait](./widget-trait.md)
- [Custom Widgets](./custom-widgets.md)
- [Accessibility Checking](../testing/accessibility-checking.md)
