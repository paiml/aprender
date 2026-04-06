# Soft Assertions

> **Toyota Way**: Kaizen (Continuous Improvement) - Collect all failures before stopping

Soft assertions allow you to collect multiple assertion failures without immediately stopping the test. This is useful for validating multiple related conditions in a single test run.

## Basic Usage

```rust
use probar::prelude::*;

fn test_form_validation() -> ProbarResult<()> {
    let mut soft = SoftAssertions::new();

    // Collect all validation failures
    soft.assert_eq("username", "alice", expected_username);
    soft.assert_eq("email", "alice@example.com", expected_email);
    soft.assert_eq("role", "admin", expected_role);

    // Check all assertions at once
    soft.verify()?;
    Ok(())
}
```

## Running the Example

```bash
cargo run --example soft_assertions
```
