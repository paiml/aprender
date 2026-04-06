# Layer Hierarchy

Presentar's vertical architecture.

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

## Layer 6: Trueno

**Foundation layer**
- SIMD-accelerated tensor operations
- Memory management
- Backend abstraction (CPU/GPU)

## Layer 7: Trueno-Viz

**Rendering primitives**
- Paths, fills, strokes
- Text rendering
- WGSL shaders

## Layer 8: Presentar

**UI Framework**
- Widget trait and tree
- Layout engine
- Event system
- State management

## Layer 9: App Runtime

**Application layer**
- YAML manifest parsing
- Model loading (.apr)
- Dataset loading (.ald)
- Pacha registry integration

## Dependencies Flow

```
App Runtime
    ↓ uses
Presentar
    ↓ uses
Trueno-Viz
    ↓ uses
Trueno
```

## Verified Test

```rust
#[test]
fn test_layer_independence() {
    // Each layer can be tested independently
    use presentar_core::{Size, Constraints};

    // Core layer works without higher layers
    let c = Constraints::loose(Size::new(100.0, 100.0));
    assert_eq!(c.biggest(), Size::new(100.0, 100.0));
}
```
