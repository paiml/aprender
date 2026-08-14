# Browser Contexts

> **Toyota Way**: Heijunka (Level Loading) - Balanced resource allocation

Manage isolated browser contexts for parallel testing with independent storage, cookies, and sessions.

## Running the Example

```bash
cargo run --example multi_context
```

## Quick Start

```rust
use probar::{BrowserContext, ContextConfig};

// Create a context with default settings
let context = BrowserContext::new(ContextConfig::default());

// Create a context with custom settings
let custom = BrowserContext::new(
    ContextConfig::default()
        .with_viewport(1920, 1080)
        .with_locale("en-US")
        .with_timezone("America/New_York")
);
```

## Context Configuration

```rust
use probar::{ContextConfig, StorageState, Cookie};

// Full configuration
let config = ContextConfig::default()
    .with_viewport(1280, 720)
    .with_device_scale_factor(2.0)
    .with_mobile(false)
    .with_touch_enabled(false)
    .with_locale("en-GB")
    .with_timezone("Europe/London")
    .with_user_agent("Mozilla/5.0 (Custom Agent)")
    .with_offline(false)
    .with_javascript_enabled(true)
    .with_ignore_https_errors(false);

println!("Viewport: {}x{}",
    config.viewport_width,
    config.viewport_height);
```

## Storage State

```rust
use probar::{StorageState, Cookie, SameSite};
use std::collections::HashMap;

// Create storage state
let mut storage = StorageState::new();

// Add local storage
storage.set_local_storage("session", "abc123");
storage.set_local_storage("theme", "dark");

// Add session storage
storage.set_session_storage("cart", "[1,2,3]");

// Add cookies
let cookie = Cookie::new("auth_token", "xyz789")
    .with_domain(".example.com")
    .with_path("/")
    .with_secure(true)
    .with_http_only(true)
    .with_same_site(SameSite::Strict);

storage.add_cookie(cookie);

// Check storage contents
println!("Local storage items: {}", storage.local_storage_count());
println!("Session storage items: {}", storage.session_storage_count());
println!("Cookies: {}", storage.cookies().len());
```

## Cookie Management

```rust
use probar::{Cookie, SameSite};

// Create a basic cookie
let basic = Cookie::new("user_id", "12345");

// Create a full cookie
let secure = Cookie::new("session", "abc123xyz")
    .with_domain(".example.com")
    .with_path("/app")
    .with_expires(1735689600)  // Unix timestamp
    .with_secure(true)
    .with_http_only(true)
    .with_same_site(SameSite::Lax);

// Check cookie properties
println!("Name: {}", secure.name());
println!("Value: {}", secure.value());
println!("Domain: {:?}", secure.domain());
println!("Secure: {}", secure.secure());
println!("HttpOnly: {}", secure.http_only());
println!("SameSite: {:?}", secure.same_site());
```

## Context Pool for Parallel Testing

```rust
use probar::{ContextPool, ContextConfig};

// Create a pool of contexts
let pool = ContextPool::new(4);  // 4 parallel contexts

// Acquire a context for testing
let context = pool.acquire();

// Run test with context
// ...

// Context is returned to pool when dropped

// Get pool statistics
let stats = pool.stats();
println!("Total contexts: {}", stats.total);
println!("Available: {}", stats.available);
println!("In use: {}", stats.in_use);
```

## Context State Management

```rust
use probar::{BrowserContext, ContextState};

// Create context
let context = BrowserContext::default();

// Check state
match context.state() {
    ContextState::New => println!("Fresh context"),
    ContextState::Active => println!("Context is running"),
    ContextState::Closed => println!("Context was closed"),
}

// Context lifecycle
// context.start()?;
// ... run tests ...
// context.close()?;
```

## Multi-User Testing

```rust
use probar::{BrowserContext, ContextConfig, StorageState, Cookie};

fn create_user_context(user_id: &str, auth_token: &str) -> BrowserContext {
    let mut storage = StorageState::new();

    // Set user-specific storage
    storage.set_local_storage("user_id", user_id);

    // Set auth cookie
    storage.add_cookie(
        Cookie::new("auth", auth_token)
            .with_domain(".example.com")
            .with_secure(true)
    );

    let config = ContextConfig::default()
        .with_storage_state(storage);

    BrowserContext::new(config)
}

// Create contexts for different users
let admin = create_user_context("admin", "admin_token_xyz");
let user1 = create_user_context("user1", "user1_token_abc");
let user2 = create_user_context("user2", "user2_token_def");

// Run parallel tests with different users
// Each context is completely isolated
```

## Geolocation in Contexts

```rust
use probar::{ContextConfig, Geolocation};

// Set geolocation for context
let config = ContextConfig::default()
    .with_geolocation(Geolocation {
        latitude: 37.7749,
        longitude: -122.4194,
        accuracy: Some(10.0),
    })
    .with_permission("geolocation", "granted");

// Test location-based features
```

## Context Manager

```rust
use probar::ContextManager;

// Create context manager
let manager = ContextManager::new();

// Create named contexts
manager.create("admin", ContextConfig::default());
manager.create("user", ContextConfig::default());

// Get context by name
if let Some(ctx) = manager.get("admin") {
    // Use admin context
}

// List all contexts
for name in manager.context_names() {
    println!("Context: {}", name);
}

// Close specific context
manager.close("admin");

// Close all contexts
manager.close_all();
```

## Saving and Restoring State

```rust
use probar::{BrowserContext, StorageState};

// Save context state after login
fn save_authenticated_state(context: &BrowserContext) -> StorageState {
    context.storage_state()
}

// Restore state in new context
fn restore_state(storage: StorageState) -> BrowserContext {
    let config = probar::ContextConfig::default()
        .with_storage_state(storage);
    BrowserContext::new(config)
}

// Example: Login once, reuse state
// let login_context = BrowserContext::default();
// ... perform login ...
// let state = save_authenticated_state(&login_context);
//
// // Fast test setup - no login needed
// let test_context = restore_state(state);
```

## Best Practices

1. **Isolation**: Use separate contexts for tests that shouldn't share state
2. **Pool Sizing**: Match pool size to available system resources
3. **State Reuse**: Save auth state to avoid repeated logins
4. **Clean Slate**: Use fresh contexts for tests requiring clean state
5. **Parallel Safe**: Each test should use its own context
6. **Resource Cleanup**: Ensure contexts are properly closed
7. **Timeout Handling**: Configure appropriate timeouts per context
