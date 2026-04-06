# Quick Start

Build your first Presentar app in 5 minutes.

## Create the Project

```bash
cargo new hello-presentar
cd hello-presentar
```

## Add Dependencies

```toml
# Cargo.toml
[package]
name = "hello-presentar"
version = "0.1.0"
edition = "2021"

[dependencies]
presentar = "0.1"

[lib]
crate-type = ["cdylib"]
```

## Create the App Manifest

```yaml
# app.yaml
presentar: "0.1"
name: "hello-presentar"
version: "1.0.0"

layout:
  type: "app"
  sections:
    - id: "main"
      widgets:
        - type: "text"
          content: "Hello, Presentar!"
          style: "heading-1"

        - type: "button"
          label: "Click Me"
          on_click: "greet"

interactions:
  - trigger: "greet"
    action: "update_text"
    script: |
      set_state("greeting", "You clicked the button!")
```

## Write the Rust Code

```rust
// src/lib.rs
use presentar::prelude::*;

#[presentar::main]
pub fn app() -> App<AppState> {
    App::from_yaml(include_str!("../app.yaml"))
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct AppState {
    greeting: String,
}

impl State for AppState {
    type Message = AppMessage;

    fn update(&mut self, msg: Self::Message) -> Command<Self::Message> {
        match msg {
            AppMessage::Greet => {
                self.greeting = "You clicked the button!".to_string();
            }
        }
        Command::None
    }
}

pub enum AppMessage {
    Greet,
}
```

## Build and Run

```bash
# Development build
cargo build --target wasm32-unknown-unknown

# Generate JS bindings
wasm-bindgen target/wasm32-unknown-unknown/debug/hello_presentar.wasm \
    --out-dir pkg --target web

# Serve locally
python3 -m http.server 8080 -d pkg
```

Open http://localhost:8080 in your browser.

## Production Build

```bash
# Optimized release build
cargo build --target wasm32-unknown-unknown --release

# Generate bindings
wasm-bindgen target/wasm32-unknown-unknown/release/hello_presentar.wasm \
    --out-dir pkg --target web

# Optimize WASM (reduces size by ~30%)
wasm-opt -O3 -o pkg/hello_presentar_bg_opt.wasm pkg/hello_presentar_bg.wasm
```

## Using the Makefile

Presentar projects include a Makefile for common tasks:

```bash
make dev      # Start development server with hot reload
make build    # Production build
make test     # Run all tests
make tier2    # Pre-commit quality gates
```

## Next Steps

- [First App](./first-app.md) - Build a complete application
- [Core Concepts](./core-concepts.md) - Understand the architecture
- [YAML Configuration](./yaml-configuration.md) - Master the manifest format
