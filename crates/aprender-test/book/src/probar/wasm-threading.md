# WASM Threading Capabilities

Probar provides comprehensive detection and testing of WASM threading capabilities, ensuring your application handles various browser configurations correctly.

## Overview

Web applications using `SharedArrayBuffer` for threading require specific HTTP headers (COOP/COEP). Probar helps you:

- Detect threading capability availability
- Verify COOP/COEP header configuration
- Test fallback paths for single-threaded mode
- Validate thread-safe code paths

## Capability Detection

### Available Capabilities

```rust
use jugar_probar::capabilities::{WasmCapability, WasmThreadCapabilities};

let capabilities = [
    WasmCapability::SharedArrayBuffer,  // Shared memory between workers
    WasmCapability::Atomics,            // Atomic operations
    WasmCapability::BulkMemory,         // Bulk memory operations
    WasmCapability::Simd128,            // 128-bit SIMD
    WasmCapability::Threads,            // Web Worker threading
    WasmCapability::ExceptionHandling,  // Native exceptions
    WasmCapability::TailCall,           // Tail call optimization
    WasmCapability::MultiMemory,        // Multiple memories
    WasmCapability::Memory64,           // 64-bit addressing
];
```

### Threading Modes

```rust
use jugar_probar::capabilities::ThreadingMode;

// Available modes based on browser capabilities
let modes = [
    ThreadingMode::SingleThreaded,  // Main thread only
    ThreadingMode::WorkerBased,     // Web Workers with message passing
    ThreadingMode::SharedMemory,    // SharedArrayBuffer support
    ThreadingMode::Atomics,         // Full atomic operations
];
```

## COOP/COEP Headers

### Understanding Cross-Origin Isolation

For `SharedArrayBuffer` to be available, your server must send:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### Checking Header Status

```rust
use jugar_probar::capabilities::CoopCoepStatus;

let status = CoopCoepStatus {
    coop_value: Some("same-origin".to_string()),
    coep_value: Some("require-corp".to_string()),
    cross_origin_isolated: true,
};

assert!(status.cross_origin_isolated);
```

## Building Capability Configurations

Use the builder pattern for test scenarios:

```rust
let full_threading = WasmThreadCapabilities::builder()
    .with_shared_array_buffer(true)
    .with_atomics(true)
    .with_cross_origin_isolated(true)
    .with_hardware_concurrency(8)
    .build();

assert!(full_threading.can_use_threads());
assert_eq!(
    full_threading.recommended_mode(),
    ThreadingMode::Atomics
);
```

### Testing Fallback Paths

```rust
// Simulate browser without SharedArrayBuffer
let no_sab = WasmThreadCapabilities::builder()
    .with_shared_array_buffer(false)
    .with_atomics(false)
    .with_cross_origin_isolated(false)
    .with_hardware_concurrency(4)
    .build();

assert!(!no_sab.can_use_threads());
assert_eq!(
    no_sab.recommended_mode(),
    ThreadingMode::WorkerBased
);
```

## Browser Testing

### Verify Threading Availability

```rust
#[tokio::test]
async fn test_threading_detection() {
    let browser = Browser::new().await?;
    let page = browser.new_page().await?;

    page.goto("http://localhost:8080").await?;

    // Check if app correctly detects threading
    let is_threaded: bool = page
        .evaluate("window.isThreadedAvailable()")
        .await?;

    // Verify UI reflects capability
    if is_threaded {
        page.wait_for_selector("#parallel-mode").await?;
    } else {
        page.wait_for_selector("#sequential-mode").await?;
    }
}
```

### Test COOP/COEP Compliance

```rust
#[tokio::test]
async fn test_headers_configured() {
    let browser = Browser::new().await?;
    let page = browser.new_page().await?;

    page.goto("http://localhost:8080").await?;

    // Check cross-origin isolation
    let isolated: bool = page
        .evaluate("window.crossOriginIsolated")
        .await?;

    assert!(isolated, "COOP/COEP headers not configured");
}
```

## CLI Compliance Check

Use `probador comply` to verify COOP/COEP:

```bash
probador comply . --checks C006

# Output:
# [✓] C006: COOP/COEP headers configured correctly
```

## Distributed Worker Execution

For multi-worker scenarios, Probar provides work-stealing and data locality tracking:

### Backend Selection

```rust
use jugar_probar::brick::distributed::{BackendSelector, Backend};

let selector = BackendSelector::new()
    .with_gpu_threshold(1_000_000)   // 1M elements for GPU
    .with_simd_threshold(10_000)     // 10K for SIMD
    .with_cpu_max_threshold(100_000_000); // 100M max local

let backend = selector.select(50_000, true);
assert_eq!(backend, Backend::Simd);
```

### Data Locality Tracking

```rust
use jugar_probar::brick::distributed::{BrickDataTracker, WorkerId};

let tracker = BrickDataTracker::new();

// Register data locations
tracker.track_data("model_weights", WorkerId::new(0), 100 * 1024 * 1024);
tracker.track_data("embeddings", WorkerId::new(0), 50 * 1024 * 1024);

// Calculate worker affinity for a task
let deps = vec!["model_weights".into(), "embeddings".into()];
let affinity = tracker.calculate_affinity(&deps);
// Worker 0 has highest affinity (has both datasets)
```

### Work-Stealing Scheduler

```rust
use jugar_probar::brick::distributed::{
    WorkStealingScheduler, DistributedBrick, Backend,
};
use std::sync::Arc;

let data_tracker = Arc::new(BrickDataTracker::new());
let scheduler = WorkStealingScheduler::new(data_tracker);

// Submit tasks with priority
let task_id = scheduler.submit_priority(
    distributed_brick.to_task_spec(),
    "input_key".into(),
    10, // priority
);

let stats = scheduler.stats();
println!("Submitted: {}, Completed: {}",
    stats.total_submitted, stats.total_completed);
```

### PUB/SUB Coordination

```rust
use jugar_probar::brick::distributed::{BrickCoordinator, BrickMessage};

let coordinator = BrickCoordinator::new();

// Subscribe to weight updates
let sub = coordinator.subscribe("weights");

// Publish update
coordinator.publish("weights", BrickMessage::WeightUpdate {
    brick_name: "encoder".into(),
    weights: vec![0u8; 100],
    version: 1,
});

// Receive messages
let messages = sub.drain();
```

Run the distributed demo:

```bash
cargo run --example distributed_worker_demo -p jugar-probar
```

## Examples

Run the WASM capabilities demos:

```bash
# Threading capabilities
cargo run --example wasm_capabilities -p jugar-probar

# Worker harness testing
cargo run --example worker_harness_demo -p jugar-probar

# Worker code generation
cargo run --example worker_brick_demo -p jugar-probar

# Distributed execution
cargo run --example distributed_worker_demo -p jugar-probar
```

## See Also

- [Worker Harness Testing](./worker-harness.md) - Comprehensive worker testing
- [Web Builders](./web-builders.md) - Asset generation
