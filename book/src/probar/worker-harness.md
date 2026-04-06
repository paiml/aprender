# Worker Harness Testing

Probar provides comprehensive Web Worker testing for WASM applications (PROBAR-SPEC-013), including lifecycle validation, message ordering, ring buffers, and shared memory testing.

## Overview

The Worker Harness tests critical Web Worker functionality:

1. **Lifecycle State Machine** - Valid state transitions
2. **Lamport Clock Ordering** - Message causality verification
3. **Ring Buffer Testing** - Audio streaming buffers
4. **Shared Memory Testing** - SharedArrayBuffer and Atomics
5. **Memory Leak Detection** - Memory growth analysis

## Quick Start

```rust
use jugar_probar::worker_harness::{
    WorkerTestHarness, WorkerTestConfig,
    RingBufferTestConfig, SharedMemoryTestConfig,
};

let harness = WorkerTestHarness::new();

// Test lifecycle transitions
let failures = harness.test_lifecycle_transitions();
assert!(failures.is_empty(), "Invalid transitions found");

// Verify message ordering
let timestamps: Vec<u64> = (0..100).map(|i| i * 10).collect();
assert!(harness.verify_message_ordering(&timestamps).is_empty());
```

## Configuration Presets

### Default

```rust
let config = WorkerTestConfig::default();
// init_timeout: 10s
// command_timeout: 30s
// stress_iterations: 100
// All tests enabled
```

### Minimal (Fast CI)

```rust
let config = WorkerTestConfig::minimal();
// init_timeout: 5s
// stress_iterations: 10
// Error recovery and memory leak tests disabled
```

### Comprehensive

```rust
let config = WorkerTestConfig::comprehensive();
// init_timeout: 30s
// stress_iterations: 1000
// All tests enabled with thorough coverage
```

## Worker Lifecycle States

The harness validates the worker state machine:

```
NotCreated ──► Loading ──► Initializing ──► Ready
                                             │
                                             ▼
               Terminated ◄── Error ◄── Processing
```

### State Definitions

| State | Description |
|-------|-------------|
| `NotCreated` | Worker not yet instantiated |
| `Loading` | Script being loaded |
| `Initializing` | Running initialization code |
| `Ready` | Ready to process messages |
| `Processing` | Actively handling a message |
| `Error` | Encountered an error |
| `Terminated` | Worker has been terminated |

### Testing Transitions

```rust
let harness = WorkerTestHarness::new();
let failures = harness.test_lifecycle_transitions();

for failure in &failures {
    println!("Invalid transition: {}", failure);
}
```

## Lamport Clock Ordering

Verifies message causality using Lamport timestamps:

```rust
// Valid: monotonically increasing
let valid = vec![1, 2, 3, 4, 5];
assert!(harness.verify_message_ordering(&valid).is_empty());

// Invalid: timestamp regression
let invalid = vec![1, 2, 5, 3, 6]; // 3 < 5 violates ordering
let failures = harness.verify_message_ordering(&invalid);
assert!(!failures.is_empty());
```

### Disable Ordering Check

```rust
let config = WorkerTestConfig {
    verify_lamport_ordering: false,
    ..Default::default()
};
let harness = WorkerTestHarness::with_config(config);
```

## Ring Buffer Testing

Tests SPSC (Single Producer, Single Consumer) ring buffers for audio streaming:

```rust
let config = RingBufferTestConfig {
    buffer_size: 65536,  // 64KB
    sample_size: 512,    // 512 bytes per sample
    num_samples: 1000,   // Total samples to test
    test_overflow: true,
    test_underrun: true,
    test_concurrent: true,
};

let result = harness.test_ring_buffer(&config);

println!("Writes: {}", result.writes_succeeded);
println!("Reads: {}", result.reads_succeeded);
println!("Overflows: {}", result.overflows_detected);
println!("Underruns: {}", result.underruns_detected);
assert!(result.passed);
```

### Audio Worklet Configuration

```rust
// Optimal for 16kHz audio (whisper.apr)
let audio_config = RingBufferTestConfig {
    buffer_size: 16384,  // ~1 second at 16kHz
    sample_size: 512,    // 32ms chunks
    num_samples: 500,
    test_overflow: true,
    test_underrun: true,
    test_concurrent: true,
};
```

## Shared Memory Testing

Tests SharedArrayBuffer and Atomics operations:

```rust
let config = SharedMemoryTestConfig {
    buffer_size: 4096,
    num_atomic_ops: 1000,
    test_wait_notify: true,
    test_concurrent_writes: true,
    wait_timeout: Duration::from_millis(100),
};

let result = harness.test_shared_memory(&config);

assert!(result.atomics_correct);
assert!(result.wait_notify_works);
assert_eq!(result.race_conditions_detected, 0);
```

## Memory Leak Detection

Detects memory growth exceeding 10%:

```rust
let metrics = WorkerMetrics {
    memory_start: 1024 * 1024,  // 1MB
    memory_end: 1024 * 1024 + 200 * 1024,  // 1.2MB
    ..Default::default()
};

if metrics.has_memory_leak() {
    println!("Memory grew by {} bytes ({}%)",
        metrics.memory_growth(),
        metrics.memory_growth() as f64 / metrics.memory_start as f64 * 100.0
    );
}
```

## Full Test Result

```rust
let result = WorkerTestResult {
    passed: true,
    lifecycle_passed: true,
    ordering_passed: true,
    shared_memory_passed: true,
    ring_buffer_passed: true,
    error_recovery_passed: true,
    memory_leak_passed: true,
    failures: vec![],
    metrics: WorkerMetrics::default(),
};

println!("{}", result);
// Output:
// ══════════════════════════════════════════════
// Worker Test Result: PASSED
// ══════════════════════════════════════════════
// ├─ Lifecycle: ✓
// ├─ Ordering: ✓
// ├─ Shared Memory: ✓
// ├─ Ring Buffer: ✓
// ├─ Error Recovery: ✓
// └─ Memory Leak: ✓
```

## CDP JavaScript Injection

Generate JavaScript for browser injection:

```rust
// Lifecycle state tracking
let lifecycle_js = WorkerTestHarness::lifecycle_test_js();

// Ring buffer testing
let ring_js = WorkerTestHarness::ring_buffer_test_js(16384);

// Shared memory testing
let shared_js = WorkerTestHarness::shared_memory_test_js(4096);
```

## Example

Run the demo:

```bash
cargo run --example worker_harness_demo -p jugar-probar
```

## Integration with Docker Testing

Combine with Docker cross-browser testing:

```rust
use jugar_probar::docker::{DockerTestRunner, Browser};
use jugar_probar::worker_harness::WorkerTestHarness;

// Test workers across all browsers
let mut runner = DockerTestRunner::builder()
    .browser(Browser::Chrome)
    .with_coop_coep(true)  // Required for SharedArrayBuffer
    .build()?;

runner.simulate_start()?;

// Inject worker test harness
let harness = WorkerTestHarness::new();
let lifecycle_js = WorkerTestHarness::lifecycle_test_js();
// ... inject and verify
```

## WorkerBrick Code Generation

Generate both JavaScript worker code and Rust bindings from a single definition:

```rust
use jugar_probar::brick::worker::{
    WorkerBrick, BrickWorkerMessage, BrickWorkerMessageDirection, FieldType,
};

let audio_worker = WorkerBrick::new("audio_processor")
    // Messages TO the worker
    .message(
        BrickWorkerMessage::new("init", BrickWorkerMessageDirection::ToWorker)
            .field("sampleRate", FieldType::Number)
            .field("sharedBuffer", FieldType::SharedArrayBuffer),
    )
    // Messages FROM the worker
    .message(
        BrickWorkerMessage::new("ready", BrickWorkerMessageDirection::FromWorker)
    )
    // State machine transitions
    .transition("uninitialized", "init", "initializing")
    .transition("initializing", "ready", "ready");

// Generate JavaScript (zero hand-written JS!)
let js = audio_worker.to_worker_js();

// Generate Rust web_sys bindings
let rust = audio_worker.to_rust_bindings();

// Generate TypeScript definitions
let ts = audio_worker.to_typescript_defs();
```

### Supported Field Types

| FieldType | TypeScript | Rust |
|-----------|------------|------|
| `String` | `string` | `String` |
| `Number` | `number` | `f64` |
| `Boolean` | `boolean` | `bool` |
| `SharedArrayBuffer` | `SharedArrayBuffer` | `js_sys::SharedArrayBuffer` |
| `Float32Array` | `Float32Array` | `js_sys::Float32Array` |
| `Optional(T)` | `T \| undefined` | `Option<T>` |

Run the code generation demo:

```bash
cargo run --example worker_brick_demo -p jugar-probar
```

## See Also

- [Zero-JS Validation](./zero-js-validation.md) - WASM-first validation
- [Docker Cross-Browser Testing](./docker-testing.md) - Multi-browser testing
- [WASM Threading](./wasm-threading.md) - Thread capability detection
- [Web Builders](./web-builders.md) - HTML/CSS/JS asset generation
