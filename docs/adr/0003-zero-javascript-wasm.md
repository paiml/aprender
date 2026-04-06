# ADR-003: Zero-JavaScript WASM Architecture

## Status

Accepted

## Context

Web deployment of simular requires a strategy for browser compatibility. Traditional approaches use JavaScript for:
- DOM manipulation
- Event handling
- Canvas rendering
- State management

### Problems with JavaScript
1. **Type safety**: Runtime errors from type coercion
2. **Testing complexity**: Requires browser environment
3. **Code duplication**: Logic split between Rust and JS
4. **Security**: XSS attack surface

### Requirement
Deploy simular to web browsers with minimal JavaScript surface area.

## Decision

Adopt a **Zero-JavaScript** policy where:

1. **All logic lives in Rust/WASM**
2. **JavaScript limited to one initialization line**
3. **DOM manipulation via web-sys**
4. **Event handling via Rust closures**

### Architecture

```
HTML                    Rust/WASM
┌─────────────────┐    ┌─────────────────┐
│ <script>        │    │ init_app()      │
│   import init   │───►│   setup_dom()   │
│   init();       │    │   bind_events() │
│ </script>       │    │   render_loop() │
└─────────────────┘    └─────────────────┘
     1 line JS              All logic
```

### Implementation

```rust
// Rust handles everything
#[wasm_bindgen(js_name = initTspApp)]
pub fn init_tsp_app() -> Result<(), JsValue> {
    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");

    // DOM manipulation
    let canvas = document.get_element_by_id("canvas")...;

    // Event binding
    setup_button(&document, "btn-run", move || { ... })?;

    Ok(())
}
```

```html
<!-- HTML: Single JS line -->
<script type="module">
  import init from './pkg/simular.js'; init();
</script>
```

## Consequences

### Positive
- **Type safety**: All logic type-checked at compile time
- **Testability**: Can test with wasm-bindgen-test
- **Single source**: No JS/Rust code duplication
- **Security**: Minimal XSS surface (no eval, no dynamic code)
- **Performance**: No JS↔WASM marshaling overhead

### Negative
- **Learning curve**: web-sys API is verbose
- **Bundle size**: Full DOM bindings increase WASM size
- **Debugging**: Stack traces harder to read
- **Browser APIs**: Not all available via web-sys

### Metrics

| Metric | Traditional | Zero-JS |
|--------|------------|---------|
| JS lines | 500+ | 1 |
| Type errors | Runtime | Compile-time |
| Test coverage | ~60% | 95%+ |
| XSS surface | High | Minimal |

## References

- [wasm-bindgen Guide](https://rustwasm.github.io/docs/wasm-bindgen/)
- [web-sys API](https://docs.rs/web-sys/latest/web_sys/)
