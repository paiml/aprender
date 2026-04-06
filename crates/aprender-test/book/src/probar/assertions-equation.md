# Equation Verification

> **Toyota Way**: Poka-Yoke (Mistake-Proofing) - Mathematical correctness guarantees

Equation verification validates physics and game math invariants with floating-point tolerance handling.

## Basic Usage

```rust
use probar::prelude::*;

fn test_physics() -> ProbarResult<()> {
    let mut verifier = EquationVerifier::new("physics_test");

    // Verify kinematics equation: v = v0 + at
    let v0 = 10.0;
    let a = 5.0;
    let t = 2.0;
    let v = v0 + a * t;

    verifier.verify_eq("v = v0 + at", 20.0, v);
    verifier.verify_in_range("speed", v, 0.0, 100.0);

    assert!(verifier.all_passed());
    Ok(())
}
```

## Running the Example

```bash
cargo run --example equation_verify
```
