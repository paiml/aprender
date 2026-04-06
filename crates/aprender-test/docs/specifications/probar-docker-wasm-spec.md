# Probar Docker WASM Testing Specification

**Version**: 1.0.0
**Status**: IMPLEMENTING
**Ticket**: PROBAR-SPEC-014
**Target**: Cross-Browser WASM Testing via Docker
**Toyota Principle**: Heijunka (Level Loading) + Jidoka (Built-in Quality)
**PMAT Compliance**: Required (bashrs validation)

---

## Executive Summary

This specification defines Docker-based testing infrastructure for WASM applications,
enabling cross-browser testing (Chrome, Firefox, WebKit) with proper COOP/COEP header
configuration for SharedArrayBuffer support.

### Key Requirements

1. **Cross-Browser Testing**: Chrome, Firefox, WebKit in isolated containers
2. **SharedArrayBuffer Support**: Pre-configured COOP/COEP headers
3. **CI/CD Integration**: GitHub Actions, GitLab CI compatible
4. **Zero-JS Compliance**: All testing via Rust/probar (no Playwright/Puppeteer)
5. **PMAT Quality Gates**: 95% coverage, 85% mutation score

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Host Machine                                 │
├─────────────────────────────────────────────────────────────────┤
│  cargo test -p probar --features docker                         │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              DockerTestRunner                                ││
│  │  ┌─────────────┬─────────────┬─────────────┐                ││
│  │  │   Chrome    │   Firefox   │   WebKit    │                ││
│  │  │  Container  │  Container  │  Container  │                ││
│  │  └──────┬──────┴──────┬──────┴──────┬──────┘                ││
│  │         │             │             │                        ││
│  │         ▼             ▼             ▼                        ││
│  │  ┌─────────────────────────────────────────┐                ││
│  │  │         probar serve (per container)     │                ││
│  │  │  • COOP: same-origin                     │                ││
│  │  │  • COEP: require-corp                    │                ││
│  │  │  • SharedArrayBuffer: enabled            │                ││
│  │  └─────────────────────────────────────────┘                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Docker Components

### 1. Base Image: `probar-wasm-base`

```dockerfile
# Multi-stage build for minimal image size
FROM rust:1.83-slim-bookworm AS builder
# ... build probar ...

FROM debian:bookworm-slim AS runtime
# Install browsers + dependencies
```

**Requirements**:
- Rust 1.83+ toolchain
- Chrome 120+
- Firefox 120+
- WebKit (via playwright-webkit or wpewebkit)
- probar CLI pre-installed

### 2. Browser Containers

| Container | Browser | CDP Port | Purpose |
|-----------|---------|----------|---------|
| `probar-chrome` | Chromium 120+ | 9222 | Primary testing |
| `probar-firefox` | Firefox 120+ | 9223 | Cross-browser |
| `probar-webkit` | WebKit | 9224 | Safari compat |

### 3. Security Headers (COOP/COEP)

All containers MUST serve with:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

This enables:
- `SharedArrayBuffer` construction
- `Atomics.wait()` / `Atomics.notify()`
- Web Worker with shared memory
- AudioWorklet with SharedArrayBuffer

---

## API Design

### DockerTestRunner

```rust
use probar::docker::{DockerTestRunner, Browser, DockerConfig};

// Create runner with specific browser
let runner = DockerTestRunner::new()
    .browser(Browser::Firefox)
    .with_coop_coep(true)
    .timeout(Duration::from_secs(60))
    .build()?;

// Run tests
let results = runner.run_tests(&[
    "tests/worker_tests.rs",
    "tests/shared_memory_tests.rs",
]).await?;

assert!(results.all_passed());
```

### Multi-Browser Parallel Testing

```rust
use probar::docker::{DockerTestRunner, Browser, ParallelRunner};

// Run same tests across all browsers in parallel
let runner = ParallelRunner::new()
    .browsers(&[Browser::Chrome, Browser::Firefox, Browser::WebKit])
    .tests(&["tests/e2e_tests.rs"])
    .build()?;

let results = runner.run_parallel().await?;

// Check each browser passed
for (browser, result) in results.by_browser() {
    println!("{}: {} passed, {} failed", browser, result.passed, result.failed);
    assert!(result.all_passed());
}
```

### Container Lifecycle

```rust
use probar::docker::{Container, ContainerConfig};

// Manual container management
let container = Container::start(ContainerConfig {
    image: "probar-chrome:latest",
    ports: vec![("9222", "9222")],
    environment: vec![
        ("DISPLAY", ":99"),
        ("PROBAR_COOP_COEP", "true"),
    ],
    ..Default::default()
})?;

// Use container
let page = container.new_page("http://localhost:8080").await?;
// ... run tests ...

// Cleanup
container.stop()?;
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROBAR_DOCKER_IMAGE` | `probar-wasm:latest` | Base image to use |
| `PROBAR_BROWSER` | `chrome` | Default browser |
| `PROBAR_CDP_PORT` | `9222` | CDP debug port |
| `PROBAR_COOP_COEP` | `true` | Enable security headers |
| `PROBAR_TIMEOUT` | `60` | Test timeout (seconds) |
| `PROBAR_PARALLEL` | `4` | Max parallel containers |

### probar.toml Configuration

```toml
[docker]
enabled = true
image = "probar-wasm:latest"
browsers = ["chrome", "firefox"]
parallel = 4
timeout = 60

[docker.chrome]
image = "probar-chrome:latest"
cdp_port = 9222

[docker.firefox]
image = "probar-firefox:latest"
cdp_port = 9223

[docker.webkit]
image = "probar-webkit:latest"
cdp_port = 9224

[docker.headers]
coop = "same-origin"
coep = "require-corp"
```

---

## PMAT Quality Requirements

### Coverage Targets

| Metric | Minimum | Target |
|--------|---------|--------|
| Line Coverage | 95% | 98% |
| Branch Coverage | 90% | 95% |
| Mutation Score | 85% | 90% |

### Falsification Tests (100-Point Checklist)

| # | Claim | Test Method | Falsifier |
|---|-------|-------------|-----------|
| 1 | Chrome container starts | Integration | Find startup failure |
| 2 | Firefox container starts | Integration | Find startup failure |
| 3 | WebKit container starts | Integration | Find startup failure |
| 4 | COOP header set correctly | HTTP check | Find missing header |
| 5 | COEP header set correctly | HTTP check | Find missing header |
| 6 | SharedArrayBuffer available | JS check | Find unavailable SAB |
| 7 | Atomics available | JS check | Find unavailable Atomics |
| 8 | CDP connection works | Protocol test | Find connection failure |
| 9 | Tests run in container | E2E test | Find execution failure |
| 10 | Results collected correctly | Unit test | Find result mismatch |
| 11 | Container cleanup works | Integration | Find orphan container |
| 12 | Parallel execution works | Load test | Find race condition |
| 13 | Timeout handling works | Timeout test | Find hung container |
| 14 | Error propagation works | Error test | Find swallowed error |
| 15 | Cross-browser results match | Comparison | Find browser difference |

---

## Implementation Phases

### Phase 1: Core Infrastructure (This PR)
- [ ] DockerTestRunner struct
- [ ] Container lifecycle management
- [ ] Chrome container support
- [ ] COOP/COEP header configuration
- [ ] Basic test execution

### Phase 2: Multi-Browser Support
- [ ] Firefox container
- [ ] WebKit container
- [ ] Browser-specific configurations
- [ ] Cross-browser result comparison

### Phase 3: CI/CD Integration
- [ ] GitHub Actions workflow
- [ ] GitLab CI configuration
- [ ] Parallel test execution
- [ ] Result aggregation

### Phase 4: Advanced Features
- [ ] Container caching
- [ ] Incremental testing
- [ ] Visual regression in containers
- [ ] Network simulation

---

## Security Considerations

1. **Container Isolation**: Each test run in isolated container
2. **No Privileged Mode**: Containers run without --privileged
3. **Network Isolation**: Optional network=none for security tests
4. **Resource Limits**: CPU/memory limits to prevent DoS
5. **Image Verification**: SHA256 digest verification for images

---

## References

1. [Docker SDK for Rust](https://docs.rs/bollard)
2. [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
3. [SharedArrayBuffer Requirements](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer)
4. [COOP/COEP Headers](https://web.dev/cross-origin-isolation-guide/)
5. [whisper.apr WASM Architecture](../whisper.apr/docs/specifications/)

---

_Last updated: 2025-01-06_
_Author: probar automated specification system_
