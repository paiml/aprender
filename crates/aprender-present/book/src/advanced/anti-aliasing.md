# Anti-Aliasing

Smooth rendering of edges and text.

## Modes

| Mode | Quality | Performance |
|------|---------|-------------|
| None | Jagged | Fastest |
| Grayscale | Good | Fast |
| Subpixel | Best | Slower |

## Test Determinism

For reproducible tests, use grayscale only:

```rust
// Test configuration
let config = RenderConfig {
    antialiasing: Antialiasing::Grayscale,
    dpi: 1.0,  // Fixed DPI
};
```

## Text Rendering

```rust
// Grayscale antialiasing for text
canvas.draw_text_aa(&text, position, &style, Antialiasing::Grayscale);
```

## Shape Edges

```rust
// Antialiased rectangle
canvas.fill_rect_aa(bounds, color, Antialiasing::Grayscale);
```

## Why Grayscale for Tests?

| Mode | Cross-Platform | Deterministic |
|------|----------------|---------------|
| None | Yes | Yes |
| Grayscale | Yes | Yes |
| Subpixel | No (RGB order varies) | No |

## Verified Test

```rust
#[test]
fn test_antialiasing_determinism() {
    // Grayscale AA is deterministic
    let config_a = presentar_core::Color::new(0.5, 0.5, 0.5, 1.0);
    let config_b = presentar_core::Color::new(0.5, 0.5, 0.5, 1.0);

    assert_eq!(config_a, config_b);  // Same gray = deterministic
}
```
