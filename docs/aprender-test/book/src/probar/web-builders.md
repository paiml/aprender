# Web Asset Builders

Probar provides type-safe builders for generating HTML, CSS, and JavaScript assets. These builders enforce best practices and accessibility compliance at compile time.

## Overview

The web builder system consists of:

| Builder | Purpose |
|---------|---------|
| `HtmlBuilder` | Generate accessible HTML documents |
| `CssBuilder` | Generate structured stylesheets |
| `CssRule` | Define individual CSS rules |
| `JsBuilder` | Generate minimal WASM loaders (max 20 lines) |
| `WebBundle` | Combine and validate all assets |

## HtmlBuilder

Generate accessible HTML documents with WCAG-compliant attributes.

### Basic Usage

```rust
use jugar_probar::web::HtmlBuilder;

let html = HtmlBuilder::new()
    .title("My WASM Game")
    .canvas("game-canvas", 800, 600)
    .build()?;

println!("Generated {} bytes", html.content.len());
```

### Generated HTML

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My WASM Game</title>
</head>
<body>
    <canvas id="game-canvas" width="800" height="600"
            role="application" aria-label="Application canvas"
            tabindex="0"></canvas>
</body>
</html>
```

### Auto-Generated Features

- DOCTYPE declaration
- `lang="en"` for accessibility
- charset and viewport meta tags
- WCAG-compliant canvas attributes (role, aria-label, tabindex)

## CssBuilder

Generate structured CSS stylesheets with preset helpers.

### Basic Usage

```rust
use jugar_probar::web::{CssBuilder, CssRule};

let css = CssBuilder::new()
    .reset()                      // Modern CSS reset
    .fullscreen_body()            // Body fills viewport
    .responsive_canvas("game")    // Canvas sizing
    .rule(
        CssRule::new(".btn")
            .declaration("padding", "12px 24px")
            .declaration("cursor", "pointer"),
    )
    .build()?;
```

### Preset Methods

| Method | Description |
|--------|-------------|
| `.reset()` | Modern CSS reset (box-sizing, margins) |
| `.fullscreen_body()` | Body fills entire viewport |
| `.responsive_canvas(id)` | Canvas scales to viewport |

### CssRule Builder

```rust
let rule = CssRule::new("#my-element")
    .declaration("background", "rgba(0, 0, 0, 0.8)")
    .declaration("border-radius", "8px")
    .declaration("padding", "16px");
```

## JsBuilder

Generate minimal WASM loaders. **Enforces a strict 20-line limit** to comply with Zero-JS policy.

### Basic Usage

```rust
use jugar_probar::web::JsBuilder;

let js = JsBuilder::new("game.wasm", "game-canvas")
    .memory(256, 1024)        // Memory pages (64KB each)
    .entry_point("main")      // WASM export to call
    .build()?;

assert!(js.within_limit());   // Check < 20 lines
```

### Generated JavaScript

```javascript
(async()=>{
const c=document.getElementById('game-canvas');
const m=new WebAssembly.Memory({initial:256,maximum:1024});
const i={env:{memory:m,canvas:c}};
const{instance:w}=await WebAssembly.instantiateStreaming(fetch('game.wasm'),i);
w.exports.main();
})();
```

### Line Limit Enforcement

The builder fails if the generated JS exceeds 20 lines:

```rust
let result = JsBuilder::new("app.wasm", "canvas")
    // ... many configurations ...
    .build();

if let Err(e) = result {
    println!("JS exceeded line limit: {}", e);
}
```

## WebBundle

Combine HTML, CSS, and JS into a validated bundle.

### Basic Usage

```rust
use jugar_probar::web::{HtmlBuilder, CssBuilder, JsBuilder, WebBundle};

let html = HtmlBuilder::new()
    .title("Game")
    .canvas("game", 800, 600)
    .build()?;

let css = CssBuilder::new()
    .reset()
    .responsive_canvas("game")
    .build()?;

let js = JsBuilder::new("game.wasm", "game")
    .build()?;

let bundle = WebBundle::new(html, css, js);

// Check validation
assert!(bundle.is_valid());
println!("Errors: {}", bundle.validation.error_count());
```

### Single-File Output

Generate a self-contained HTML file with inline CSS and JS:

```rust
let output = bundle.to_single_file();
std::fs::write("index.html", output)?;
```

### Validation Report

The bundle automatically validates all assets:

```rust
let bundle = WebBundle::new(html, css, js);

println!("HTML valid: {}", bundle.validation.html.is_valid());
println!("CSS valid: {}", bundle.validation.css.is_valid());
println!("JS valid: {}", bundle.validation.js.is_valid());
println!("Total errors: {}", bundle.validation.error_count());
println!("Total warnings: {}", bundle.validation.warning_count());
```

## Example

Run the full demo:

```bash
cargo run --example web_builders_demo -p jugar-probar
```

## See Also

- [Web Validation](./web-validation.md) - Validation and linting APIs
- [Zero-JS Validation](./zero-js-validation.md) - Unauthorized JS detection
- [Compliance Checking](./compliance.md) - Full compliance validation
