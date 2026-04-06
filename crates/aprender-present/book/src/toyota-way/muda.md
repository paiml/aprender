# Muda (Waste Elimination)

Eliminate everything that doesn't add value.

## The Seven Wastes

| Waste | Software Equivalent | Presentar Solution |
|-------|---------------------|-------------------|
| Overproduction | Unused features | YAGNI principle |
| Waiting | Slow builds | Incremental compilation |
| Transport | Data copying | Zero-copy where possible |
| Processing | Runtime overhead | Compile-time optimization |
| Inventory | Bloated deps | Minimal dependencies |
| Motion | Context switching | Focused codebase |
| Defects | Bugs | TDD, type safety |

## Waste Eliminated

### No Python GIL

```rust
// Pure Rust = no GIL overhead
// Concurrent rendering without locks
```

### No Runtime Interpretation

```rust
// YAML → Rust structs at build time
// Not parsed at runtime
```

### Minimal Dependencies

```toml
# 80% Sovereign Stack (Trueno ecosystem)
# 20% External (winit, fontdue only)
```

### No Unnecessary Abstraction

```rust
// Direct widget implementation
// No virtual DOM diffing
```

## Bundle Size

| Framework | Bundle Size |
|-----------|-------------|
| React + DOM | ~200KB |
| Presentar | ~100KB |

## Performance

| Metric | Target | Presentar |
|--------|--------|-----------|
| Frame time | <16ms | ~8ms |
| First paint | <100ms | ~50ms |
| Memory | Minimal | ~10MB |

## Verified Test

```rust
#[test]
fn test_muda_elimination() {
    use std::mem::size_of;
    use presentar_widgets::Button;

    // Button is small (no waste)
    let button_size = size_of::<Button>();
    assert!(button_size < 256);  // Bytes, not KB
}
```
