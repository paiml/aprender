# MLOps Overview

Entrenar provides a comprehensive MLOps toolkit for production machine learning workflows, following Toyota Production
System principles.

## Toyota Way Principles

Each component implements specific TPS principles:

| Component | Toyota Principle | Description |
|-----------|-----------------|-------------|
| Preflight Validation | Jidoka (自働化) | Built-in quality through automatic defect detection |
| Experiment Tracking | Genchi Genbutsu | Go and see - understand through observation |
| Model Registry | Kanban | Visual workflow management for model stages |
| GPU Monitoring | Andon | Visual alerting system for issues |
| HPO | Kaizen | Continuous improvement through optimization |
| Differential Privacy | Jidoka | Protect quality of privacy guarantees |

## Components

### Experiment Tracking

Track experiments, runs, metrics, and artifacts with SQLite-based storage:

```rust
use entrenar::storage::{SqliteBackend, ExperimentStorage, RunStatus};

let mut backend = SqliteBackend::open("experiments.db")?;
let exp_id = backend.create_experiment("my-experiment", None)?;
let run_id = backend.create_run(&exp_id)?;
backend.start_run(&run_id)?;
backend.log_metric(&run_id, "loss", 0, 0.5)?;
backend.complete_run(&run_id, RunStatus::Success)?;
```

### Preflight Validation

Catch data issues before training starts (30-50% of failures):

```rust
use entrenar::storage::{Preflight, PreflightCheck};

let preflight = Preflight::standard()
    .add_check(PreflightCheck::min_samples(1000))
    .add_check(PreflightCheck::disk_space_mb(10240));

let results = preflight.run(&data);
if !results.all_passed() {
    eprintln!("{}", results.report());
}
```

### Model Registry

Manage model lifecycle with staging workflows:

```rust
use entrenar::storage::{InMemoryRegistry, ModelRegistry, ModelStage};

let mut registry = InMemoryRegistry::new();
registry.register_model("my-model", 1, metrics, artifact_path)?;
registry.transition_stage("my-model", 1, ModelStage::Staging)?;
registry.transition_stage("my-model", 1, ModelStage::Production)?;
```

### Hyperparameter Optimization

Bayesian optimization with TPE sampler:

```rust
use entrenar::optim::hpo::{BayesianOptimizer, SearchSpace, ParameterDef};

let space = SearchSpace::new()
    .add("lr", ParameterDef::LogUniform(1e-5, 1e-2))
    .add("hidden_dim", ParameterDef::Discrete(vec![64, 128, 256, 512]));

let optimizer = BayesianOptimizer::new(space, 50);
```

### Differential Privacy

Privacy-preserving training with DP-SGD:

```rust
use entrenar::optim::dp::{DPOptimizer, PrivacyEngine};

let engine = PrivacyEngine::new()
    .with_noise_multiplier(1.0)
    .with_max_grad_norm(1.0)
    .with_target_epsilon(1.0)
    .with_target_delta(1e-5);

let dp_optimizer = DPOptimizer::new(base_optimizer, engine);
```

### GPU Monitoring

Real-time GPU metrics with Andon alerting:

```rust
use entrenar::monitor::gpu::{GpuMonitor, AndonSystem, GpuAlert};

let monitor = GpuMonitor::new()?;
let metrics = monitor.collect_metrics()?;

let andon = AndonSystem::default();
let alerts = andon.check(&metrics);
for alert in alerts {
    eprintln!("ALERT: {}", alert.message());
}
```

### REST API Server

HTTP API for remote experiment tracking:

```rust
use entrenar::server::{TrackingServer, ServerConfig};

let config = ServerConfig::default().with_port(5000);
let server = TrackingServer::new(config);
server.run().await?;
```

Endpoints:
- `GET /health` - Health check
- `POST /api/v1/experiments` - Create experiment
- `POST /api/v1/runs` - Create run
- `POST /api/v1/runs/{id}/metrics` - Log metrics

### Cloud Storage

Store artifacts in S3, Azure Blob, or GCS:

```rust
use entrenar::storage::{S3Config, BackendConfig, LocalBackend};

// Local storage
let backend = LocalBackend::new("./artifacts");

// S3 configuration
let s3_config = S3Config {
    bucket: "my-bucket".to_string(),
    region: Some("us-east-1".to_string()),
    endpoint: None,
    access_key: None,
    secret_key: None,
};
```

### LLM Evaluation

Evaluate LLM outputs for quality:

```rust
use entrenar::monitor::llm::{InMemoryLLMEvaluator, LLMEvaluator};

let mut evaluator = InMemoryLLMEvaluator::new();
let result = evaluator.evaluate_response(
    &run_id,
    "What is machine learning?",
    "Machine learning is...",
    Some("ML is a subset of AI...")
)?;

println!("Relevance: {}", result.relevance);
println!("Coherence: {}", result.coherence);
println!("Groundedness: {}", result.groundedness);
```

## Cargo Run Examples

```bash
# Start REST API server
cargo run --features server -- server --port 5000

# Run with experiment tracking
cargo run --example experiment_tracking

# Run with GPU monitoring
cargo run --example gpu_monitor

# Run HPO sweep
cargo run --example hpo_sweep
```

## See Also

- [Experiment Tracking](./experiment-tracking.md)
- [Preflight Validation](./preflight.md)
- [Model Registry](./model-registry.md)
- [Hyperparameter Optimization](./hpo.md)
- [Differential Privacy](./differential-privacy.md)
- [GPU Monitoring](./gpu-monitoring.md)
- [REST API](./rest-api.md)
- [Cloud Storage](./cloud-storage.md)
- [LLM Evaluation](./llm-evaluation.md)
