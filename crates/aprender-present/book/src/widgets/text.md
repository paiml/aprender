# Text

The `Text` widget displays styled text content with configurable font properties.

## Basic Usage

```rust
use presentar_widgets::Text;

let text = Text::new("Hello, World!");
```

## Builder Pattern

Text supports a fluent builder pattern:

```rust
use presentar_widgets::Text;
use presentar_core::{Color, widget::FontWeight};

let text = Text::new("Welcome")
    .font_size(24.0)
    .color(Color::from_hex("#1f2937").unwrap())
    .weight(FontWeight::Bold)
    .line_height(1.5)
    .with_test_id("welcome-text");
```

## Customization Options

| Method | Description | Default |
|--------|-------------|---------|
| `font_size(f32)` | Text size in pixels | `14.0` |
| `color(Color)` | Text color | `BLACK` |
| `weight(FontWeight)` | Font weight | `Normal` |
| `line_height(f32)` | Line height multiplier | `1.2` |
| `max_width(f32)` | Maximum width for wrapping | `None` |

## Font Weights

Available font weights:

```rust
use presentar_core::widget::FontWeight;

let thin = Text::new("Thin").weight(FontWeight::Thin);
let light = Text::new("Light").weight(FontWeight::Light);
let normal = Text::new("Normal").weight(FontWeight::Normal);
let medium = Text::new("Medium").weight(FontWeight::Medium);
let semibold = Text::new("SemiBold").weight(FontWeight::SemiBold);
let bold = Text::new("Bold").weight(FontWeight::Bold);
let black = Text::new("Black").weight(FontWeight::Black);
```

## Text Wrapping

Set `max_width` to enable text wrapping:

```rust
let paragraph = Text::new("This is a long paragraph that will wrap...")
    .max_width(300.0)
    .line_height(1.6);
```

## Accessibility

Text provides accessible content:

- Role: `AccessibleRole::Text`
- `accessible_name` defaults to the text content
- Not focusable (static content)

```rust
let text = Text::new("Important notice")
    .with_accessible_name("Alert: Important notice");
```

## Testing

Use `test_id` for testing:

```rust
let text = Text::new("Status: Ready").with_test_id("status-text");

// In tests
let content = harness.find("[data-testid='status-text']").text();
assert_eq!(content, "Status: Ready");
```

## YAML Definition

```yaml
- Text:
    content: "Hello, World!"
    font_size: 16
    color: "#1f2937"
    weight: bold
```

## Common Patterns

### Headings

```rust
let h1 = Text::new("Page Title")
    .font_size(32.0)
    .weight(FontWeight::Bold);

let h2 = Text::new("Section Title")
    .font_size(24.0)
    .weight(FontWeight::SemiBold);
```

### Body Text

```rust
let body = Text::new("Lorem ipsum dolor sit amet...")
    .font_size(16.0)
    .line_height(1.6)
    .max_width(600.0);
```

### Captions

```rust
let caption = Text::new("Figure 1: Architecture diagram")
    .font_size(12.0)
    .color(Color::from_hex("#6b7280").unwrap());
```
