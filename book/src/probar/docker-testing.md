# Docker Cross-Browser Testing

Probar provides Docker-based infrastructure for cross-browser WASM testing, enabling consistent test execution across Chrome, Firefox, and WebKit with proper COOP/COEP header configuration for SharedArrayBuffer support.

## Overview

Docker testing solves several key challenges:

1. **Cross-Browser Consistency**: Test WASM applications across Chrome, Firefox, and WebKit
2. **CI/CD Integration**: Consistent environments in GitHub Actions, GitLab CI, etc.
3. **SharedArrayBuffer Support**: Pre-configured COOP/COEP headers
4. **Parallel Execution**: Run tests across all browsers simultaneously

## Quick Start

Enable the `docker` feature in your `Cargo.toml`:

```toml
[dev-dependencies]
jugar-probar = { version = "0.4", features = ["docker"] }
```

### Single Browser Testing

```rust
use probar::docker::{DockerTestRunner, Browser};
use std::time::Duration;

let mut runner = DockerTestRunner::builder()
    .browser(Browser::Chrome)
    .with_coop_coep(true)
    .timeout(Duration::from_secs(60))
    .build()?;

runner.simulate_start()?;
let results = runner.simulate_run_tests(&["tests/e2e.rs"])?;
runner.simulate_stop()?;

assert!(results.all_passed());
```

### Parallel Cross-Browser Testing

```rust
use probar::docker::{ParallelRunner, Browser};

let mut runner = ParallelRunner::builder()
    .browsers(&Browser::all())  // Chrome, Firefox, WebKit
    .tests(&["tests/worker.rs", "tests/atomics.rs"])
    .build()?;

runner.simulate_run()?;

assert!(runner.all_passed());
println!("Total: {} passed, {} failed",
    runner.aggregate_stats().0,
    runner.aggregate_stats().1);
```

## Browser Support

| Browser | CDP Port | Docker Image | Use Case |
|---------|----------|--------------|----------|
| Chrome | 9222 | `probar-chrome:latest` | Primary testing |
| Firefox | 9223 | `probar-firefox:latest` | Cross-browser validation |
| WebKit | 9224 | `probar-webkit:latest` | Safari compatibility |

### Browser Configuration

```rust
use probar::docker::Browser;

// Parse browser from string
let browser = Browser::from_str("firefox").unwrap();

// Get browser properties
println!("CDP Port: {}", browser.default_cdp_port());
println!("Image: {}", browser.image_name());
println!("Container: {}", browser.container_prefix());
```

## COOP/COEP Headers

For SharedArrayBuffer to work in browsers, your server must send:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### Configuration

```rust
use probar::docker::CoopCoepConfig;

// Default: enables SharedArrayBuffer
let config = CoopCoepConfig::default();
assert!(config.shared_array_buffer_available());

// Disable for testing without isolation
let disabled = CoopCoepConfig::disabled();
assert!(!disabled.shared_array_buffer_available());
```

### Header Validation

```rust
use probar::docker::validate_coop_coep_headers;
use std::collections::HashMap;

let mut headers = HashMap::new();
headers.insert("Cross-Origin-Opener-Policy".to_string(), "same-origin".to_string());
headers.insert("Cross-Origin-Embedder-Policy".to_string(), "require-corp".to_string());

match validate_coop_coep_headers(&headers) {
    Ok(true) => println!("SharedArrayBuffer enabled"),
    Ok(false) => println!("Headers present but invalid"),
    Err(e) => println!("Error: {}", e),
}
```

## Container Lifecycle

### States

```rust
use probar::docker::ContainerState;

// Container state machine:
// NotCreated -> Creating -> Starting -> Running -> Stopping -> Stopped
//                                    -> HealthChecking
//                                    -> Error
```

### Full Lifecycle Example

```rust
use probar::docker::{DockerTestRunner, Browser, ContainerState};

let mut runner = DockerTestRunner::builder()
    .browser(Browser::Firefox)
    .cleanup(true)
    .capture_logs(true)
    .build()?;

assert_eq!(runner.state(), ContainerState::NotCreated);

runner.simulate_start()?;
assert_eq!(runner.state(), ContainerState::Running);
assert!(runner.container_id().is_some());

let results = runner.simulate_run_tests(&["tests/e2e.rs"])?;
println!("Results: {}", results);

runner.simulate_stop()?;
assert_eq!(runner.state(), ContainerState::Stopped);

// Access captured logs
for log in runner.logs() {
    println!("{}", log);
}
```

## Container Configuration

### Custom Configuration

```rust
use probar::docker::{DockerTestRunner, Browser};
use std::path::PathBuf;
use std::time::Duration;

let runner = DockerTestRunner::builder()
    .browser(Browser::Chrome)
    .with_coop_coep(true)
    .timeout(Duration::from_secs(120))
    .parallel(4)  // Max parallel containers
    .pull_images(true)
    .cleanup(true)
    .capture_logs(true)
    .docker_socket("/var/run/docker.sock".to_string())
    .volume(PathBuf::from("./tests"), "/app/tests".to_string())
    .env("DEBUG".to_string(), "1".to_string())
    .build()?;
```

### Browser-Specific Defaults

```rust
use probar::docker::ContainerConfig;

// Get browser-specific container configuration
let chrome_config = ContainerConfig::for_browser(Browser::Chrome);
assert_eq!(chrome_config.ports, vec![(9222, 9222)]);

let firefox_config = ContainerConfig::for_browser(Browser::Firefox);
assert_eq!(firefox_config.ports, vec![(9223, 9223)]);
```

## Docker Compose Integration

Probar provides Docker Compose configuration for parallel testing:

```yaml
# docker/docker-compose.test.yml
version: "3.9"

services:
  chrome:
    build:
      context: ..
      dockerfile: docker/Dockerfile.wasm-test
      target: chrome
    ports:
      - "9222:9222"
    environment:
      - PROBAR_BROWSER=chrome
      - PROBAR_COOP_COEP=true

  firefox:
    build:
      context: ..
      dockerfile: docker/Dockerfile.wasm-test
      target: firefox
    ports:
      - "9223:9223"

  webkit:
    build:
      context: ..
      dockerfile: docker/Dockerfile.wasm-test
      target: webkit
    ports:
      - "9224:9224"
```

### Running Parallel Tests

```bash
# Start all browser containers
docker-compose -f docker/docker-compose.test.yml up -d

# Run tests
cargo test --features docker

# Cleanup
docker-compose -f docker/docker-compose.test.yml down
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Cross-Browser Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build test containers
        run: docker-compose -f docker/docker-compose.test.yml build

      - name: Run cross-browser tests
        run: |
          docker-compose -f docker/docker-compose.test.yml up -d
          cargo test --features docker
          docker-compose -f docker/docker-compose.test.yml down
```

### GitLab CI

```yaml
cross-browser-tests:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker-compose -f docker/docker-compose.test.yml up -d
    - cargo test --features docker
    - docker-compose -f docker/docker-compose.test.yml down
```

## Test Results

### Result Aggregation

```rust
use probar::docker::{TestResult, TestResults, Browser};
use std::time::Duration;

let mut results = TestResults::new(Browser::Chrome);

// Add passing test
results.add_result(TestResult::passed(
    "test_worker_lifecycle".to_string(),
    Duration::from_millis(150),
));

// Add failing test
results.add_result(TestResult::failed(
    "test_shared_memory".to_string(),
    Duration::from_millis(200),
    "assertion failed: expected 42, got 0".to_string(),
));

println!("{}", results);
// Output: chrome: 1 passed, 1 failed (50.0%)

assert!(!results.all_passed());
assert_eq!(results.total(), 2);
assert_eq!(results.pass_rate(), 50.0);
```

### Cross-Browser Comparison

```rust
use probar::docker::{ParallelRunner, Browser};

let mut runner = ParallelRunner::builder()
    .browsers(&Browser::all())
    .tests(&["tests/e2e.rs"])
    .build()?;

runner.simulate_run()?;

// Compare results across browsers
for (browser, results) in runner.results_by_browser() {
    println!("{}: {} passed, {} failed",
        browser, results.passed, results.failed);
}

// Aggregate statistics
let (passed, failed, duration) = runner.aggregate_stats();
println!("Total: {} passed, {} failed in {:?}", passed, failed, duration);
```

## Error Handling

```rust
use probar::docker::{DockerError, DockerTestRunner};

let result = DockerTestRunner::builder()
    .docker_socket("/nonexistent/docker.sock".to_string())
    .build();

match result {
    Ok(_) => println!("Runner created"),
    Err(DockerError::DaemonUnavailable(msg)) => {
        println!("Docker not available: {}", msg);
    }
    Err(DockerError::ConfigError(msg)) => {
        println!("Configuration error: {}", msg);
    }
    Err(e) => println!("Other error: {}", e),
}
```

## Example

Run the Docker demo:

```bash
cargo run --example docker_demo -p jugar-probar --features docker
```

Output:

```
╔══════════════════════════════════════════════════════════════╗
║     Docker Cross-Browser WASM Testing (PROBAR-SPEC-014)      ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Browser Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Supported browsers for Docker-based testing:

    Browser: chrome
      ├─ CDP Port: 9222
      ├─ Image: probar-chrome:latest
      └─ Container Prefix: probar-chrome
...
```

## Best Practices

1. **Use Parallel Testing**: Run all browsers simultaneously to reduce test time
2. **Enable COOP/COEP**: Always enable for SharedArrayBuffer-dependent code
3. **Capture Logs**: Enable `capture_logs(true)` for debugging failures
4. **Set Timeouts**: Configure appropriate timeouts for CI environments
5. **Clean Up**: Use `cleanup(true)` to remove containers after tests
6. **Resource Limits**: Configure memory/CPU limits in container config

## See Also

- [WASM Threading](./wasm-threading.md) - Thread capability detection
- [Compliance Checking](./compliance.md) - Zero-JS validation
- [Streaming UX Validation](./streaming-validation.md) - Real-time testing
