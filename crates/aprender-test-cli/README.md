# aprender-test-cli

[![Crates.io](https://img.shields.io/crates/v/aprender-test-cli.svg)](https://crates.io/crates/aprender-test-cli)
[![CI](https://github.com/paiml/aprender/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/aprender/actions/workflows/ci.yml)

**aprender-test-cli** is the CLI tool for [Probar](https://github.com/paiml/aprender/tree/main/crates/aprender-test) - a Playwright-compatible testing framework for WASM games and applications.

> **Note:** The library is published separately as [aprender-test-lib](https://crates.io/crates/aprender-test-lib).

## Installation

```bash
cargo install aprender-test-cli
```

## Quick Start

```bash
# Validate a playbook state machine
aprender-test-cli playbook login.yaml --validate

# Run with mutation testing (M1-M5 falsification)
aprender-test-cli playbook login.yaml --mutate

# Export state diagram
aprender-test-cli playbook login.yaml --export svg -o diagram.svg

# Run tests
aprender-test-cli test

# Run tests with coverage
aprender-test-cli coverage --html

# Watch mode for development
aprender-test-cli watch tests/

# Start dev server for WASM
aprender-test-cli serve --port 8080
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

aprender-test-cli supports YAML-driven state machine testing with mutation testing:

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
aprender-test-cli playbook login.yaml --validate

# Export as SVG diagram
aprender-test-cli playbook login.yaml --export svg -o login.svg

# Mutation testing (M1-M5 classes)
aprender-test-cli playbook login.yaml --mutate
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
cargo add aprender-test-lib --dev
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

- [Book](https://github.com/paiml/aprender/tree/main/crates/aprender-test/book) - Comprehensive guide
- [API Docs](https://docs.rs/aprender-test-lib) - Library documentation
- [CLI Reference](https://github.com/paiml/aprender/tree/main/crates/aprender-test/book) - Full command reference

## License

MIT OR Apache-2.0

---

Part of the [Aprender monorepo](https://github.com/paiml/aprender) — 70 workspace crates.
