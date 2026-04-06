# FAQ

Frequently asked questions.

## General

### What is Presentar?

A WASM-first visualization framework built on the Sovereign AI Stack. It eliminates Python/CUDA dependencies for self-hosted AI workloads.

### Why not React/Vue/Svelte?

- No JavaScript runtime overhead
- Type-safe at compile time
- Deterministic rendering
- Zero-dependency testing

### Why not Streamlit/Gradio?

- No Python GIL
- 60fps GPU rendering
- Type safety
- Deterministic tests

## Technical

### What's the minimum Rust version?

Rust 1.75+ with `wasm32-unknown-unknown` target.

### How do I add a custom widget?

Implement the `Widget` trait. See [Custom Widgets](../widgets/custom-widgets.md).

### How do I test widgets?

Use the zero-dependency test harness:
```rust
let harness = Harness::new(widget);
harness.assert_exists("[data-testid='btn']");
```

### What's the bundle size?

Approximately 100KB for a basic app.

### How do I deploy?

Build to WASM and serve statically:
```bash
cargo build --target wasm32-unknown-unknown --release
```

## Testing

### Why no Playwright/Selenium?

Zero external dependencies policy. We build our own test harness in pure Rust.

### How do I run tests?

```bash
make test       # All tests
make test-fast  # Unit tests only
```

### How do I do visual regression?

```rust
Snapshot::assert_match("name", &screenshot, 0.001);
```

## Performance

### What's the frame budget?

16ms for 60fps. Typical paint is <8ms.

### How do I optimize?

- Use layout caching
- Minimize draw commands
- Avoid deep nesting
