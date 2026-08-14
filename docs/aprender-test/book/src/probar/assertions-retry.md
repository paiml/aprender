# Retry Assertions

> **Toyota Way**: Jidoka (Built-in Quality) - Automatic retry with intelligent backoff

Retry assertions automatically retry failed conditions with configurable timeout and intervals, perfect for testing asynchronous state changes.

## Basic Usage

```rust
use probar::prelude::*;

fn test_async_state() -> ProbarResult<()> {
    let retry = RetryAssertion::new()
        .with_timeout(Duration::from_secs(5))
        .with_interval(Duration::from_millis(100));

    retry.retry_true(|| {
        // Check condition that may take time to become true
        check_element_visible()
    })?;

    Ok(())
}
```

## Running the Example

```bash
cargo run --example retry_assertions
```
