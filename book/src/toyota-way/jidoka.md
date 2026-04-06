# Jidoka (Built-in Quality)

Stop the line when defects are detected.

## Principle

> Build quality in, don't inspect it out.

## Application in Presentar

### Compile-Time Checks

```rust
// Type system catches errors
let size: Size = button.measure(constraints);
// let size: String = button.measure(constraints);  // Won't compile!
```

### Test Failures Stop the Build

```bash
cargo test || exit 1  # No deployment if tests fail
```

### Clippy as Andon Cord

```bash
cargo clippy -- -D warnings  # Warnings = errors
```

## Quality Gates

| Gate | Trigger | Action |
|------|---------|--------|
| Type Check | Every compile | Block if fails |
| Tests | Every commit | Block if fails |
| Clippy | Pre-commit | Block on warnings |
| Coverage | Nightly | Alert if decreases |

## Andon Light System

```
GREEN  → All tests pass, continue
YELLOW → Warning, investigate
RED    → Failure, stop and fix
```

## Implementation

```rust
// Tests are the quality gate
#[test]
fn test_button_behavior() {
    let button = Button::new("OK");
    assert!(button.is_interactive());
    // If this fails, deployment stops
}
```

## Benefits

| Benefit | Result |
|---------|--------|
| Early detection | Fix bugs when cheap |
| No bad deployments | Quality guaranteed |
| Developer confidence | Safe to refactor |

## Verified Test

```rust
#[test]
fn test_jidoka_quality_gate() {
    // This test IS the quality gate
    // If it fails, CI stops
    assert!(true);
}
```
