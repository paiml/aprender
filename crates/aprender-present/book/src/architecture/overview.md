# Architecture Overview

Presentar's layered architecture.

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 9: App Runtime                                           │
│  - YAML parser, .apr/.ald loaders, Pacha integration            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 8: Presentar (Reactive UI Framework)                     │
│  - Widget tree, layout engine, event dispatch, state            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 7: Trueno-Viz (GPU Rendering Primitives)                 │
│  - Paths, fills, strokes, text, charts, WGSL shaders            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: Trueno (SIMD/GPU Compute)                             │
│  - Tensor ops, backend dispatch, memory management              │
└─────────────────────────────────────────────────────────────────┘
```

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `presentar` | Main entry point, re-exports |
| `presentar-core` | Widget trait, geometry, events |
| `presentar-widgets` | Built-in widget library |
| `presentar-layout` | Layout engine, caching |
| `presentar-yaml` | YAML manifest parsing |
| `presentar-test` | Zero-dep test harness |

## Data Flow

```
User Input → Event → Widget Tree → State Update
    ↑                                    │
    └────────────────────────────────────┘
                  Repaint
```

## Key Components

### Widget Tree
- Retained-mode UI hierarchy
- Measure → Layout → Paint cycle
- Event propagation

### Layout Engine
- Flexbox-inspired constraints
- Caching for performance
- Bottom-up measure, top-down layout

### Canvas Abstraction
- `Canvas` trait for rendering
- `RecordingCanvas` for testing
- GPU backend via Trueno-Viz

## Dependencies

**80% Sovereign Stack:**
- Trueno (SIMD ops)
- Trueno-Viz (GPU rendering)

**20% External:**
- winit (windowing)
- fontdue (font rasterization)

## Verified Test

```rust
#[test]
fn test_architecture_layers() {
    // presentar re-exports presentar-core
    use presentar::{Widget, Constraints, Size};
    use presentar_widgets::Button;

    let button = Button::new("Test");
    let size = button.measure(Constraints::unbounded());
    assert!(size.width > 0.0);
}
```
