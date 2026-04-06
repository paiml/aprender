# probador

[![Crates.io](https://img.shields.io/crates/v/probador.svg)](https://crates.io/crates/probador)
[![CI](https://github.com/paiml/probar/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/probar/actions/workflows/ci.yml)

**probador** (Spanish: "tester") is the CLI tool for [Probar](https://github.com/paiml/probar) - a Playwright-compatible testing framework for WASM games and applications.

> **Note:** The library is published separately as [jugar-probar](https://crates.io/crates/jugar-probar).

## Installation

```bash
cargo install probador
```

## Quick Start

```bash
# Validate a playbook state machine
probador playbook login.yaml --validate

# Run with mutation testing (M1-M5 falsification)
probador playbook login.yaml --mutate

# Export state diagram
probador playbook login.yaml --export svg -o diagram.svg

# Run tests
probador test

# Run tests with coverage
probador coverage --html

# Watch mode for development
probador watch tests/

# Start dev server for WASM
probador serve --port 8080
```

## Commands

| Command | Description |
|---------|-------------|
| `test` | Run tests with optional filtering |
| `playbook` | YAML-driven state machine testing |
| `coverage` | Generate coverage reports |
| `record` | Record test execution as GIF/MP4 |
| `report` | Generate HTML/JSON/JUnit reports |
| `serve` | Development server for WASM |
| `watch` | Watch mode with hot reload |
| `init` | Initialize new test project |
| `config` | View/manage configuration |

## Playbook Testing

probador supports YAML-driven state machine testing with mutation testing:

```yaml
# login.yaml
version: "1.0"
name: "Login Flow"
machine:
  id: "login"
  initial: "logged_out"
  states:
    logged_out:
      id: "logged_out"
    logging_in:
      id: "logging_in"
    logged_in:
      id: "logged_in"
      final_state: true
  transitions:
    - id: "start_login"
      from: "logged_out"
      to: "logging_in"
      event: "submit"
    - id: "complete_login"
      from: "logging_in"
      to: "logged_in"
      event: "success"
```

```bash
# Validate
probador playbook login.yaml --validate

# Export as SVG diagram
probador playbook login.yaml --export svg -o login.svg

# Mutation testing (M1-M5 classes)
probador playbook login.yaml --mutate
```

### Mutation Classes

| Class | Description |
|-------|-------------|
| M1 | State removal |
| M2 | Transition removal |
| M3 | Event swap |
| M4 | Target swap |
| M5 | Guard negation |

## Library Usage

For programmatic usage in Rust tests, add the library crate:

```bash
cargo add jugar-probar --dev
```

```rust
use jugar_probar::prelude::*;

#[test]
fn test_game() {
    let mut gui = gui_coverage! {
        buttons: ["start", "quit"],
        screens: ["menu", "game"]
    };

    gui.click("start");
    gui.visit("game");

    assert!(gui.meets(50.0));
}
```

## Documentation

- [Book](https://paiml.github.io/probar/) - Comprehensive guide
- [API Docs](https://docs.rs/jugar-probar) - Library documentation
- [CLI Reference](https://paiml.github.io/probar/probar/cli-reference.html) - Full command reference

## License

MIT OR Apache-2.0
