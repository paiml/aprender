# Probar Quick Start

Get started with Probar testing in 5 minutes.

![Coverage Visualization](../assets/coverage_viridis.png)

## Installation

Probar is distributed as two crates:

| Crate | Purpose | Install |
|-------|---------|---------|
| **jugar-probar** | Library for writing tests | `cargo add jugar-probar --dev` |
| **probador** | CLI tool | `cargo install probador` |

### Add the Library

```toml
[dev-dependencies]
jugar-probar = "0.3"
```

### Install the CLI (Optional)

```bash
cargo install probador
```

## Write Your First Test

```rust
use jugar_probar::prelude::*;

#[test]
fn test_game_initializes() {
    // Create test platform
    let config = WebConfig::new(800, 600);
    let mut platform = WebPlatform::new_for_test(config);

    // Run initial frame
    let output = platform.frame(0.0, "[]");

    // Verify game started
    assert!(output.contains("commands"));
}
```

## Run Tests

```bash
# Run all tests
cargo test

# With verbose output
cargo test -- --nocapture

# Using probador CLI
probador test
```

## Test Structure

### Basic Assertions

```rust
use jugar_probar::Assertion;

#[test]
fn test_assertions() {
    // Equality
    let eq = Assertion::equals(&42, &42);
    assert!(eq.passed);

    // Range
    let range = Assertion::in_range(50.0, 0.0, 100.0);
    assert!(range.passed);

    // Boolean
    let truthy = Assertion::is_true(true);
    assert!(truthy.passed);

    // Approximate equality (for floats)
    let approx = Assertion::approx_eq(3.14, 3.14159, 0.01);
    assert!(approx.passed);
}
```

### GUI Coverage

```rust
use jugar_probar::gui_coverage;

#[test]
fn test_gui_coverage() {
    let mut gui = gui_coverage! {
        buttons: ["start", "pause", "quit"],
        screens: ["menu", "game", "game_over"]
    };

    // Record interactions
    gui.click("start");
    gui.visit("menu");
    gui.visit("game");

    // Check coverage
    println!("{}", gui.summary());
    assert!(gui.meets(50.0));  // At least 50% coverage
}
```

### Testing Game Logic

```rust
#[test]
fn test_ball_movement() {
    let mut platform = WebPlatform::new_for_test(WebConfig::default());

    // Get initial position
    let initial_pos = platform.get_ball_position();

    // Advance 60 frames (1 second)
    for _ in 0..60 {
        platform.advance_frame(1.0 / 60.0);
    }

    // Ball should have moved
    let new_pos = platform.get_ball_position();
    assert_ne!(initial_pos, new_pos);
}
```

### Testing Input

```rust
#[test]
fn test_paddle_responds_to_input() {
    let mut platform = WebPlatform::new_for_test(WebConfig::default());

    let initial_y = platform.get_paddle_y(Player::Left);

    // Simulate pressing W key
    platform.key_down("KeyW");
    for _ in 0..30 {
        platform.advance_frame(1.0 / 60.0);
    }
    platform.key_up("KeyW");

    // Paddle should have moved up
    let new_y = platform.get_paddle_y(Player::Left);
    assert!(new_y < initial_y);
}
```

## Using probador CLI

```bash
# Validate playbook state machines
probador playbook login.yaml --validate

# Export state diagram as SVG
probador playbook login.yaml --export svg -o diagram.svg

# Run mutation testing
probador playbook login.yaml --mutate

# Generate coverage reports
probador coverage --html

# Watch mode with hot reload
probador watch tests/

# Start dev server for WASM
probador serve --port 8080
```

## Examples

Run the included examples:

```bash
# Deterministic simulation with replay
cargo run --example pong_simulation -p jugar-probar

# Locator API demo
cargo run --example locator_demo -p jugar-probar

# Accessibility checking
cargo run --example accessibility_demo -p jugar-probar

# GUI coverage demo
cargo run --example gui_coverage -p jugar-probar
```

## Example Output

```
=== Probar Pong Simulation Demo ===

--- Demo 1: Pong Simulation ---
Initial state:
  Ball: (400.0, 300.0)
  Paddles: P1=300.0, P2=300.0
  Score: 0 - 0

Simulating 300 frames...

Final state after 300 frames:
  Ball: (234.5, 412.3)
  Paddles: P1=180.0, P2=398.2
  Score: 2 - 1
  State valid: true

--- Demo 2: Deterministic Replay ---
Recording simulation (seed=42, frames=500)...
  Completed: true
  Final hash: 6233835744931225727

Replaying simulation...
  Determinism verified: true
  Hashes match: true
```

## Next Steps

- [Assertions](./assertions.md) - All assertion types
- [Simulation](./simulation.md) - Deterministic simulation
- [Fuzzing](./fuzzing.md) - Random testing
- [Coverage Tooling](./coverage-tooling.md) - Code coverage
- [CLI Reference](./cli-reference.md) - Full probador command reference
