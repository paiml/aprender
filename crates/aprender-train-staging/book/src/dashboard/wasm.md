# WASM Bindings

The dashboard module provides WebAssembly bindings for browser-based training dashboards.

## Feature Flag

Enable WASM support in your `Cargo.toml`:

```toml
[dependencies]
entrenar = { version = "0.2", features = ["wasm"] }
```

## IndexedDbStorage

Browser-compatible storage backend that implements `ExperimentStorage`:

```rust
pub struct IndexedDbStorage {
    // In-memory implementation mimicking IndexedDB behavior
}

impl ExperimentStorage for IndexedDbStorage {
    fn create_experiment(&mut self, name: &str, config: Option<Value>) -> Result<String>;
    fn create_run(&mut self, experiment_id: &str) -> Result<String>;
    fn start_run(&mut self, run_id: &str) -> Result<()>;
    fn complete_run(&mut self, run_id: &str, status: RunStatus) -> Result<()>;
    fn log_metric(&mut self, run_id: &str, key: &str, step: u64, value: f64) -> Result<()>;
    fn log_artifact(&mut self, run_id: &str, key: &str, data: &[u8]) -> Result<String>;
    fn get_metrics(&self, run_id: &str, key: &str) -> Result<Vec<MetricPoint>>;
    fn get_run_status(&self, run_id: &str) -> Result<RunStatus>;
    fn set_span_id(&mut self, run_id: &str, span_id: &str) -> Result<()>;
    fn get_span_id(&self, run_id: &str) -> Result<Option<String>>;
}
```

### Additional Methods

```rust
impl IndexedDbStorage {
    /// List all experiments
    pub fn list_experiments(&self) -> Vec<String>;

    /// List all runs for an experiment
    pub fn list_runs(&self, experiment_id: &str) -> Vec<String>;

    /// List all metric keys for a run
    pub fn list_metric_keys(&self, run_id: &str) -> Vec<String>;
}
```

## WasmRun

JavaScript-friendly training run wrapper with `wasm_bindgen`:

```rust
#[wasm_bindgen]
pub struct WasmRun {
    run_id: String,
    experiment_id: String,
    storage: Arc<Mutex<IndexedDbStorage>>,
    step_counters: HashMap<String, u64>,
    finished: bool,
}
```

### JavaScript API

```javascript
// Create a new run
const run = await WasmRun.new('my-experiment');

// Log metrics (auto-incrementing step)
run.log_metric('loss', 0.5);
run.log_metric('loss', 0.4);
run.log_metric('accuracy', 0.85);

// Log at specific step
run.log_metric_at('learning_rate', 100, 0.001);

// Get all metrics as JSON
const metrics = JSON.parse(run.get_metrics_json());
// {
//   "loss": [
//     {"step": 0, "value": 0.5, "timestamp": "..."},
//     {"step": 1, "value": 0.4, "timestamp": "..."}
//   ],
//   "accuracy": [
//     {"step": 0, "value": 0.85, "timestamp": "..."}
//   ]
// }

// Subscribe to metric updates
run.subscribe_metrics((key, value) => {
    console.log(`${key}: ${value}`);
});

// Get run info
console.log(run.run_id());        // "run-0"
console.log(run.experiment_id()); // "exp-0"
console.log(run.current_step('loss')); // 2
console.log(run.is_finished());   // false

// Finish the run
run.finish();  // Success status
// or
run.fail();    // Failed status
```

## Building for WASM

### Prerequisites

```bash
# Install wasm-pack
cargo install wasm-pack

# Install wasm32 target
rustup target add wasm32-unknown-unknown
```

### Build

```bash
# Build for bundler (webpack, etc.)
wasm-pack build --target bundler --features wasm

# Build for web (no bundler)
wasm-pack build --target web --features wasm

# Build for Node.js
wasm-pack build --target nodejs --features wasm
```

### Output

```
pkg/
├── entrenar.js           # JavaScript bindings
├── entrenar.d.ts         # TypeScript definitions
├── entrenar_bg.wasm      # WebAssembly module
├── entrenar_bg.wasm.d.ts # WASM TypeScript defs
└── package.json          # npm package
```

## Usage in Web Applications

### With a Bundler (Webpack/Vite)

```javascript
// Install: npm install ./pkg
import init, { WasmRun } from 'entrenar';

async function main() {
    await init();

    const run = WasmRun.new('training-session');

    // Training loop
    for (let epoch = 0; epoch < 100; epoch++) {
        const loss = trainEpoch();
        run.log_metric('loss', loss);

        updateChart(JSON.parse(run.get_metrics_json()));
    }

    run.finish();
}

main();
```

### Without a Bundler

```html
<script type="module">
    import init, { WasmRun } from './pkg/entrenar.js';

    async function main() {
        await init();

        const run = WasmRun.new('browser-training');
        run.log_metric('loss', 0.5);

        document.getElementById('metrics').textContent =
            run.get_metrics_json();
    }

    main();
</script>
```

### React Example

```jsx
import { useEffect, useState, useRef } from 'react';
import init, { WasmRun } from 'entrenar';

function TrainingDashboard() {
    const [metrics, setMetrics] = useState({});
    const runRef = useRef(null);

    useEffect(() => {
        async function setup() {
            await init();
            runRef.current = WasmRun.new('react-training');
        }
        setup();

        return () => {
            if (runRef.current && !runRef.current.is_finished()) {
                runRef.current.finish();
            }
        };
    }, []);

    const logMetric = (key, value) => {
        if (runRef.current) {
            runRef.current.log_metric(key, value);
            setMetrics(JSON.parse(runRef.current.get_metrics_json()));
        }
    };

    return (
        <div>
            <button onClick={() => logMetric('loss', Math.random())}>
                Log Random Loss
            </button>
            <pre>{JSON.stringify(metrics, null, 2)}</pre>
        </div>
    );
}
```

## Error Handling

WASM methods return `Result<T, JsValue>` which throws on error:

```javascript
try {
    const run = WasmRun.new('my-experiment');
    run.log_metric('loss', 0.5);
} catch (error) {
    console.error('WASM error:', error);
}
```

## Limitations

- **No real IndexedDB** - Current implementation uses in-memory storage
- **Single-threaded** - WebAssembly runs on the main thread
- **No persistence** - Data is lost on page refresh
- **Subscribe placeholder** - `subscribe_metrics` is a placeholder API

## See Also

- [Dashboard Overview](./overview.md)
- [DashboardSource Trait](./dashboard-source.md)
