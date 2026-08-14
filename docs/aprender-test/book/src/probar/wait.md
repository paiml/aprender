# Wait Mechanisms

> **Toyota Way**: Jidoka (Automation with Human Touch) - Automatic detection of ready state

Probar provides Playwright-compatible wait mechanisms for synchronization in tests.

## Running the Example

```bash
cargo run --example wait_mechanisms
```

## Load States

Wait for specific page load states:

```rust
use probar::prelude::*;

// Available load states
let load = LoadState::Load;              // window.onload event
let dom = LoadState::DomContentLoaded;   // DOMContentLoaded event
let idle = LoadState::NetworkIdle;       // No requests for 500ms

// Each state has a default timeout
assert_eq!(LoadState::Load.default_timeout_ms(), 30_000);
assert_eq!(LoadState::NetworkIdle.default_timeout_ms(), 60_000);

// Get event name for JavaScript
assert_eq!(LoadState::Load.event_name(), "load");
assert_eq!(LoadState::DomContentLoaded.event_name(), "DOMContentLoaded");
```

## Wait Options

Configure wait behavior with `WaitOptions`:

```rust
use probar::prelude::*;

// Default options (30s timeout, 50ms polling)
let default_opts = WaitOptions::default();

// Custom options with builder pattern
let opts = WaitOptions::new()
    .with_timeout(10_000)           // 10 second timeout
    .with_poll_interval(100)        // Poll every 100ms
    .with_wait_until(LoadState::NetworkIdle);

// Access as Duration
let timeout: Duration = opts.timeout();
let poll: Duration = opts.poll_interval();
```

## Navigation Options

Configure navigation-specific waits:

```rust
use probar::prelude::*;

let nav_opts = NavigationOptions::new()
    .with_timeout(5000)
    .with_wait_until(LoadState::DomContentLoaded)
    .with_url(UrlPattern::Contains("dashboard".into()));
```

## Page Events

Wait for specific page events (Playwright parity):

```rust
use probar::prelude::*;

// All available page events
let events = [
    PageEvent::Load,
    PageEvent::DomContentLoaded,
    PageEvent::Close,
    PageEvent::Console,
    PageEvent::Dialog,
    PageEvent::Download,
    PageEvent::Popup,
    PageEvent::Request,
    PageEvent::Response,
    PageEvent::PageError,
    PageEvent::WebSocket,
    PageEvent::Worker,
];

// Get event name string
assert_eq!(PageEvent::Load.as_str(), "load");
assert_eq!(PageEvent::Popup.as_str(), "popup");
```

## Using the Waiter

### Wait for URL Pattern

```rust
use probar::prelude::*;

let mut waiter = Waiter::new();
waiter.set_url("https://example.com/dashboard");

let options = WaitOptions::new().with_timeout(5000);

// Wait for URL to match pattern
let result = waiter.wait_for_url(
    &UrlPattern::Contains("dashboard".into()),
    &options,
)?;

println!("Waited for: {}", result.waited_for);
println!("Elapsed: {:?}", result.elapsed);
```

### Wait for Load State

```rust
use probar::prelude::*;

let mut waiter = Waiter::new();
waiter.set_load_state(LoadState::Load);

let options = WaitOptions::new().with_timeout(30_000);

// Wait for page to be fully loaded
waiter.wait_for_load_state(LoadState::Load, &options)?;

// DomContentLoaded is satisfied by Load state
waiter.wait_for_load_state(LoadState::DomContentLoaded, &options)?;
```

### Wait for Navigation

```rust
use probar::prelude::*;

let mut waiter = Waiter::new();
waiter.set_url("https://example.com/app");
waiter.set_load_state(LoadState::Load);

let nav_opts = NavigationOptions::new()
    .with_timeout(10_000)
    .with_wait_until(LoadState::NetworkIdle)
    .with_url(UrlPattern::Contains("app".into()));

let result = waiter.wait_for_navigation(&nav_opts)?;
```

### Wait for Custom Function

```rust
use probar::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

let waiter = Waiter::new();
let options = WaitOptions::new()
    .with_timeout(5000)
    .with_poll_interval(50);

// Wait for counter to reach threshold
let counter = Arc::new(AtomicUsize::new(0));
let counter_clone = counter.clone();

// Simulate async updates
std::thread::spawn(move || {
    for _ in 0..10 {
        std::thread::sleep(Duration::from_millis(100));
        counter_clone.fetch_add(1, Ordering::SeqCst);
    }
});

// Wait until counter >= 5
waiter.wait_for_function(
    || counter.load(Ordering::SeqCst) >= 5,
    &options,
)?;
```

### Wait for Events

```rust
use probar::prelude::*;

let mut waiter = Waiter::new();
let options = WaitOptions::new().with_timeout(5000);

// Record events as they occur
waiter.record_event(PageEvent::Load);
waiter.record_event(PageEvent::DomContentLoaded);

// Wait for specific event
waiter.wait_for_event(&PageEvent::Load, &options)?;

// Clear recorded events
waiter.clear_events();
```

## Convenience Functions

```rust
use probar::prelude::*;

// Wait for condition with timeout
wait_until(|| some_condition(), 5000)?;

// Simple timeout (discouraged - use conditions instead)
wait_timeout(100);  // Sleep for 100ms
```

## Custom Wait Conditions

Implement the `WaitCondition` trait for custom logic:

```rust
use probar::prelude::*;

// Using FnCondition helper
let condition = FnCondition::new(
    || check_some_state(),
    "waiting for state to be ready",
);

let waiter = Waiter::new();
let options = WaitOptions::new().with_timeout(5000);

waiter.wait_for(&condition, &options)?;
```

## Network Idle Detection

NetworkIdle waits for no network requests for 500ms:

```rust
use probar::prelude::*;

let mut waiter = Waiter::new();

// Simulate pending requests
waiter.set_pending_requests(3);  // 3 active requests

// Network is NOT idle
assert!(!waiter.is_network_idle());

// All requests complete
waiter.set_pending_requests(0);

// After 500ms of no activity, network is idle
// (In real usage, this is tracked automatically)
```

## Error Handling

Wait operations return `ProbarResult` with timeout errors:

```rust
use probar::prelude::*;

let waiter = Waiter::new();
let options = WaitOptions::new()
    .with_timeout(100)
    .with_poll_interval(10);

match waiter.wait_for_function(|| false, &options) {
    Ok(result) => println!("Success: {:?}", result.elapsed),
    Err(ProbarError::Timeout { ms }) => {
        println!("Timed out after {}ms", ms);
    }
    Err(e) => println!("Other error: {}", e),
}
```

## Best Practices

1. **Prefer explicit waits over timeouts**
   ```rust
   // Good: Wait for specific condition
   waiter.wait_for_load_state(LoadState::NetworkIdle, &options)?;

   // Avoid: Fixed sleep
   wait_timeout(5000);
   ```

2. **Use appropriate polling intervals**
   ```rust
   // Fast polling for quick checks
   let fast = WaitOptions::new().with_poll_interval(10);

   // Slower polling for resource-intensive checks
   let slow = WaitOptions::new().with_poll_interval(200);
   ```

3. **Set realistic timeouts**
   ```rust
   // Navigation can be slow
   let nav = NavigationOptions::new().with_timeout(30_000);

   // UI updates should be fast
   let ui = WaitOptions::new().with_timeout(5000);
   ```

4. **Combine with assertions**
   ```rust
   // Wait then assert
   waiter.wait_for_load_state(LoadState::Load, &options)?;
   expect(locator).to_be_visible();
   ```

## Example: Full Page Load Flow

```rust
use probar::prelude::*;

fn wait_for_page_ready() -> ProbarResult<()> {
    let mut waiter = Waiter::new();

    // 1. Wait for navigation to target URL
    let nav_opts = NavigationOptions::new()
        .with_timeout(30_000)
        .with_url(UrlPattern::Contains("/app".into()));

    waiter.set_url("https://example.com/app");
    waiter.wait_for_navigation(&nav_opts)?;

    // 2. Wait for DOM to be ready
    waiter.set_load_state(LoadState::DomContentLoaded);
    let opts = WaitOptions::new().with_timeout(10_000);
    waiter.wait_for_load_state(LoadState::DomContentLoaded, &opts)?;

    // 3. Wait for network to settle
    waiter.set_load_state(LoadState::NetworkIdle);
    waiter.wait_for_load_state(LoadState::NetworkIdle, &opts)?;

    // 4. Wait for app-specific ready state
    waiter.wait_for_function(
        || app_is_initialized(),
        &opts,
    )?;

    Ok(())
}
```
