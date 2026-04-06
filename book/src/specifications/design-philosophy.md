# Design Philosophy

Guiding principles for Presentar.

## Core Principles

| Principle | Description |
|-----------|-------------|
| WASM-first | Target WebAssembly primarily |
| Zero-dependency | Minimal external crates |
| Sovereign | No cloud dependencies |
| Testable | Every feature verifiable |

## Layered Architecture

```
App (YAML)
    ↓
Presentar (Widgets)
    ↓
Trueno-Viz (Graphics)
    ↓
Trueno (SIMD/GPU)
```

## Constraints Over Configuration

```yaml
# Bad: Many options
button:
  corner_radius: 4
  shadow_offset: 2
  shadow_blur: 8
  border_width: 1

# Good: Semantic choice
button:
  style: primary
```

## Unidirectional Data Flow

```
Event → State → Widget → Paint → Frame
  ↑_________________________________|
```

## Composition Over Inheritance

```rust
// Composition
struct Card {
    content: Box<dyn Widget>,
    padding: Padding,
}

// Not inheritance
// struct Card extends Container { ... }
```

## Make Invalid States Unrepresentable

```rust
// Bad
struct Button {
    is_loading: bool,
    is_disabled: bool,
    // Can be both loading AND disabled?
}

// Good
enum ButtonState {
    Idle,
    Loading,
    Disabled,
}
```

## Explicit Over Implicit

```rust
// Explicit sizing
let size = widget.measure(&constraints);
widget.layout(size);

// Not implicit
// widget.auto_layout();
```

## Verified Test

```rust
#[test]
fn test_design_philosophy_states() {
    // Make invalid states unrepresentable
    #[derive(Debug, PartialEq)]
    enum ButtonState {
        Idle,
        Hovered,
        Pressed,
        Loading,
        Disabled,
    }

    let state = ButtonState::Loading;

    // Can only be ONE state at a time
    assert_eq!(state, ButtonState::Loading);
    assert_ne!(state, ButtonState::Disabled);

    // State transitions are explicit
    fn can_click(state: &ButtonState) -> bool {
        matches!(state, ButtonState::Idle | ButtonState::Hovered)
    }

    assert!(!can_click(&ButtonState::Loading));
    assert!(!can_click(&ButtonState::Disabled));
    assert!(can_click(&ButtonState::Idle));
}
```
