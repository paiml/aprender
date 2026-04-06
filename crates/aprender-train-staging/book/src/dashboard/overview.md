# Dashboard Overview

The Dashboard module provides real-time training monitoring capabilities with support for both native and browser-based
dashboards.

## Features

- **DashboardSource trait** - Unified interface for training data access
- **Trend analysis** - Automatic detection of metric trends (Rising, Falling, Stable)
- **Resource monitoring** - GPU, CPU, and memory utilization tracking
- **WASM support** - Browser-compatible dashboard bindings

## Quick Start

```rust
use std::sync::{Arc, Mutex};
use entrenar::storage::{InMemoryStorage, ExperimentStorage};
use entrenar::run::{Run, TracingConfig};
use entrenar::dashboard::{DashboardSource, Trend};

// Create storage and run
let mut storage = InMemoryStorage::new();
let exp_id = storage.create_experiment("my-exp", None).unwrap();
let storage = Arc::new(Mutex::new(storage));

let mut run = Run::new(&exp_id, storage.clone(), TracingConfig::disabled()).unwrap();

// Log some metrics
run.log_metric("loss", 0.5).unwrap();
run.log_metric("loss", 0.4).unwrap();
run.log_metric("loss", 0.3).unwrap();

// Get dashboard data
let status = run.status();
let metrics = run.recent_metrics(10);
let resources = run.resource_usage();

// Analyze trends
if let Some(loss) = metrics.get("loss") {
    println!("Loss trend: {} {}", loss.trend, loss.trend.emoji());
    println!("Latest: {:?}", loss.latest());
    println!("Min: {:?}", loss.min());
    println!("Max: {:?}", loss.max());
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard Module                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │  DashboardSource │    │  MetricSnapshot  │              │
│  │      trait       │───▶│    + Trend       │              │
│  └──────────────────┘    └──────────────────┘              │
│           │                                                  │
│           │              ┌──────────────────┐              │
│           │              │ ResourceSnapshot │              │
│           └─────────────▶│  GPU/CPU/Memory  │              │
│                          └──────────────────┘              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  WASM Module (optional)               │  │
│  │  ┌────────────────┐    ┌────────────────────┐       │  │
│  │  │ IndexedDbStorage│    │     WasmRun        │       │  │
│  │  │ ExperimentStorage│    │  wasm_bindgen API │       │  │
│  │  └────────────────┘    └────────────────────┘       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Types

| Type | Description |
|------|-------------|
| `DashboardSource` | Trait for providing dashboard data |
| `MetricSnapshot` | Time-series metric data with trend |
| `ResourceSnapshot` | System resource utilization |
| `Trend` | Metric direction (Rising, Falling, Stable) |
| `IndexedDbStorage` | Browser-compatible storage (WASM) |
| `WasmRun` | JavaScript-friendly run wrapper (WASM) |

## Use Cases

### Terminal Dashboard

Monitor training progress in the terminal with real-time updates:

```rust
use entrenar::dashboard::DashboardSource;

loop {
    let metrics = run.recent_metrics(50);
    let resources = run.resource_usage();

    // Update terminal display
    print!("\r");
    for (key, snapshot) in &metrics {
        print!("{}: {:.4} {} | ", key,
            snapshot.latest().unwrap_or(0.0),
            snapshot.trend.emoji());
    }
    print!("GPU: {:.1}%", resources.gpu_util * 100.0);

    std::thread::sleep(std::time::Duration::from_secs(1));
}
```

### Browser Dashboard

Use WASM bindings for interactive web dashboards:

```javascript
import { WasmRun } from 'entrenar';

const run = await WasmRun.new('my-experiment');

// Log metrics during training
run.log_metric('loss', 0.5);
run.log_metric('accuracy', 0.85);

// Get all metrics as JSON
const metrics = JSON.parse(run.get_metrics_json());
console.log(metrics);

// Finish the run
run.finish();
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `wasm` | Enable WASM bindings for browser support |

```toml
[dependencies]
entrenar = { version = "0.2", features = ["wasm"] }
```

## See Also

- [DashboardSource Trait](./dashboard-source.md) - Detailed trait documentation
- [WASM Bindings](./wasm.md) - Browser dashboard setup
- [Real-Time Monitoring](../monitor/overview.md) - Terminal monitoring features
