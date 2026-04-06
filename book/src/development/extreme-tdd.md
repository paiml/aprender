# EXTREME TDD

Presentar is developed using **EXTREME TDD** methodology—tests are written FIRST, implementation follows.

## The RED-GREEN-REFACTOR Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│    ┌───────────┐                                                │
│    │           │                                                │
│    │    RED    │◀─────────────────────────────────┐             │
│    │           │                                   │             │
│    └─────┬─────┘                                   │             │
│          │                                         │             │
│          │ Write failing test                      │             │
│          ▼                                         │             │
│    ┌───────────┐                                   │             │
│    │           │                                   │             │
│    │   GREEN   │                                   │             │
│    │           │                                   │             │
│    └─────┬─────┘                                   │             │
│          │                                         │             │
│          │ Make it pass (minimal)                  │             │
│          ▼                                         │             │
│    ┌───────────┐                                   │             │
│    │           │                                   │             │
│    │ REFACTOR  │───────────────────────────────────┘             │
│    │           │                                                 │
│    └───────────┘                                                 │
│                                                                 │
│          Improve code, tests still pass                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Example: Implementing a Button Widget

### Step 1: RED - Write the Failing Test

```rust
// tests/button_test.rs
use presentar_test::*;

#[test]
fn test_button_renders_label() {
    let button = Button::new("Click Me");
    let harness = Harness::new(button);

    harness.assert_text("[data-testid='button']", "Click Me");
}

#[test]
fn test_button_is_interactive() {
    let button = Button::new("Submit");

    assert!(button.is_interactive());
    assert!(button.is_focusable());
}

#[test]
fn test_button_disabled_not_focusable() {
    let button = Button::new("Submit").disabled(true);

    assert!(button.is_interactive());
    assert!(!button.is_focusable());
}

#[test]
fn test_button_accessible() {
    let button = Button::new("Submit");

    assert_eq!(button.accessible_name(), Some("Submit"));
    assert_eq!(button.accessible_role(), AccessibleRole::Button);
}
```

Run tests - they FAIL (no Button exists yet):

```bash
cargo test
# error[E0433]: failed to resolve: use of undeclared type `Button`
```

### Step 2: GREEN - Minimal Implementation

```rust
// src/button.rs
pub struct Button {
    label: String,
    disabled: bool,
}

impl Button {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            disabled: false,
        }
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

impl Widget for Button {
    fn type_id(&self) -> TypeId { TypeId::of::<Self>() }

    fn measure(&self, c: Constraints) -> Size {
        c.constrain(Size::new(100.0, 40.0))
    }

    fn layout(&mut self, b: Rect) -> LayoutResult {
        LayoutResult { size: b.size() }
    }

    fn paint(&self, _: &mut dyn Canvas) {}

    fn event(&mut self, _: &Event) -> Option<Box<dyn Any + Send>> { None }

    fn children(&self) -> &[Box<dyn Widget>] { &[] }
    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] { &mut [] }

    fn is_interactive(&self) -> bool { true }
    fn is_focusable(&self) -> bool { !self.disabled }
    fn accessible_name(&self) -> Option<&str> { Some(&self.label) }
    fn accessible_role(&self) -> AccessibleRole { AccessibleRole::Button }
    fn test_id(&self) -> Option<&str> { Some("button") }
}
```

Run tests - they PASS:

```bash
cargo test
# test test_button_renders_label ... ok
# test test_button_is_interactive ... ok
# test test_button_disabled_not_focusable ... ok
# test test_button_accessible ... ok
```

### Step 3: REFACTOR - Improve Code Quality

```rust
// Add builder pattern, proper paint implementation
impl Button {
    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }
}

impl Widget for Button {
    fn paint(&self, canvas: &mut dyn Canvas) {
        // Background
        canvas.fill_rect(self.bounds, self.background_color());

        // Label
        let style = TextStyle {
            color: self.text_color(),
            size: 14.0,
            ..Default::default()
        };
        canvas.draw_text(&self.label, self.bounds.center(), &style);
    }

    fn background_color(&self) -> Color {
        if self.disabled {
            Color::GRAY
        } else if self.hovered {
            Color::rgb(0.4, 0.4, 0.9)
        } else {
            Color::rgb(0.3, 0.3, 0.8)
        }
    }
}
```

Run tests again - still PASS:

```bash
cargo test
# All tests pass
cargo clippy -- -D warnings
# No warnings
```

## Quality Gates

Every commit must pass the three-tier quality pipeline:

```makefile
# TIER 1: On-save (<1 second)
tier1:
    @cargo check --workspace

# TIER 2: Pre-commit (1-5 minutes)
tier2: fmt lint test score
    @presentar gate-check .presentar-gates.toml

# TIER 3: Nightly (hours)
tier3: tier2 coverage
    @cargo mutants --timeout 300
```

## Test Categories

| Category | Purpose | Tool |
|----------|---------|------|
| Unit tests | Individual function behavior | `cargo test` |
| Property tests | Invariants across inputs | `proptest` |
| Integration tests | Cross-module behavior | `cargo test --test '*'` |
| Visual regression | Pixel-perfect rendering | `presentar-test` |
| Accessibility | WCAG compliance | `A11yChecker` |
| Performance | Frame time <16ms | `criterion` |
| Mutation | Test effectiveness | `cargo mutants` |

## Toyota Way: Zero Tolerance

> **Jidoka:** Stop the line when defects are detected.

- Tests must pass before any commit
- Clippy warnings are errors
- Coverage must not decrease
- Mutation score targets enforced

## Next Steps

- [RED-GREEN-REFACTOR](./red-green-refactor.md) - Detailed cycle explanation
- [Testing Strategy](./testing-strategy.md) - Comprehensive testing approach
- [Mutation Testing](./mutation-testing.md) - Measuring test effectiveness
