# Theming

Visual customization in YAML manifests.

## Theme Structure

```yaml
theme:
  colors:
    primary: "#4285f4"
    secondary: "#34a853"
    error: "#ea4335"
    background: "#ffffff"
    surface: "#f5f5f5"
    text: "#202124"

  fonts:
    body: "Inter"
    heading: "Inter"
    mono: "JetBrains Mono"

  spacing:
    xs: 4
    sm: 8
    md: 16
    lg: 24
    xl: 32
```

## Color Tokens

| Token | Usage |
|-------|-------|
| `primary` | Primary actions, links |
| `secondary` | Secondary actions |
| `error` | Error states, warnings |
| `background` | Page background |
| `surface` | Card backgrounds |
| `text` | Body text |

## Using Theme Values

```yaml
widgets:
  - type: Button
    label: "Primary"
    color: "{{ theme.colors.primary }}"

  - type: Text
    value: "Hello"
    font: "{{ theme.fonts.heading }}"
```

## Dark Mode

```yaml
theme:
  mode: auto  # or "light" / "dark"

  colors:
    light:
      background: "#ffffff"
      text: "#202124"
    dark:
      background: "#1a1a1a"
      text: "#e8e8e8"
```

## Typography Scale

```yaml
theme:
  typography:
    h1: { size: 32, weight: 700 }
    h2: { size: 24, weight: 600 }
    body: { size: 16, weight: 400 }
    caption: { size: 12, weight: 400 }
```

## Component Variants

```yaml
theme:
  components:
    button:
      primary:
        background: "{{ colors.primary }}"
        color: "#ffffff"
      secondary:
        background: "transparent"
        border: "{{ colors.primary }}"
```

## Verified Test

```rust
#[test]
fn test_theming_color_parsing() {
    use presentar_core::Color;

    // Parse hex color
    fn parse_hex(s: &str) -> Option<Color> {
        let s = s.trim_start_matches('#');
        if s.len() != 6 { return None; }

        let r = u8::from_str_radix(&s[0..2], 16).ok()? as f32 / 255.0;
        let g = u8::from_str_radix(&s[2..4], 16).ok()? as f32 / 255.0;
        let b = u8::from_str_radix(&s[4..6], 16).ok()? as f32 / 255.0;

        Some(Color::new(r, g, b, 1.0))
    }

    // Primary blue
    let primary = parse_hex("#4285f4").unwrap();
    assert!((primary.r - 0.259).abs() < 0.01);
    assert!((primary.g - 0.522).abs() < 0.01);
    assert!((primary.b - 0.957).abs() < 0.01);

    // White
    let white = parse_hex("#ffffff").unwrap();
    assert_eq!(white.r, 1.0);
    assert_eq!(white.g, 1.0);
    assert_eq!(white.b, 1.0);

    // Black
    let black = parse_hex("#000000").unwrap();
    assert_eq!(black.r, 0.0);
    assert_eq!(black.g, 0.0);
    assert_eq!(black.b, 0.0);
}
```
