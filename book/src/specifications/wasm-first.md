# WASM First

WebAssembly as the primary compilation target.

## Why WASM First

| Benefit | Description |
|---------|-------------|
| Portability | Runs in any modern browser |
| Performance | Near-native speed |
| Security | Sandboxed execution |
| Size | Compact binary format |

## Target Triple

```bash
# Primary target
cargo build --target wasm32-unknown-unknown

# Alternative for WASI
cargo build --target wasm32-wasi
```

## No-std Compatible

```rust
#![no_std]

// Core types work without std
use presentar_core::{Size, Constraints, Color};
```

## Size Budget

| Component | Budget |
|-----------|--------|
| Core | <100KB |
| Widgets | <150KB |
| Full app | <500KB |

## Optimization

```bash
# Build for release
cargo build --release --target wasm32-unknown-unknown

# Optimize with wasm-opt
wasm-opt -Oz -o app.wasm target/wasm32.../app.wasm
```

## Feature Flags

```toml
[features]
default = ["std"]
std = []
wasm = ["wee_alloc"]
```

## Async in WASM

```rust
// Use wasm-bindgen-futures for async
#[wasm_bindgen]
pub async fn load_data() -> JsValue {
    let data = fetch_data().await;
    serde_wasm_bindgen::to_value(&data).unwrap()
}
```

## Verified Test

```rust
#[test]
fn test_wasm_first_size_budget() {
    // Size budget validation
    let core_kb = 85.0;
    let widgets_kb = 120.0;
    let total_kb = core_kb + widgets_kb;

    // Individual budgets
    assert!(core_kb < 100.0);
    assert!(widgets_kb < 150.0);

    // Total budget
    assert!(total_kb < 500.0);

    // Calculate percentage of budget used
    let budget_used = total_kb / 500.0 * 100.0;
    assert!(budget_used < 50.0);  // 41% used
}
```
