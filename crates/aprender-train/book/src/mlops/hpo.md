# Hyperparameter Optimization

Bayesian hyperparameter optimization with Tree-structured Parzen Estimator (TPE) for efficient search.

## Toyota Principle: Kaizen

Continuous improvement through systematic optimization. HPO automates the search for better hyperparameters.

## Quick Start

```rust
use entrenar::optim::hpo::{
    BayesianOptimizer, SearchSpace, ParameterDef, Trial, TrialState
};

// Define search space
let space = SearchSpace::new()
    .add("learning_rate", ParameterDef::LogUniform(1e-5, 1e-1))
    .add("batch_size", ParameterDef::Discrete(vec![16, 32, 64, 128]))
    .add("hidden_dim", ParameterDef::Uniform(64.0, 512.0))
    .add("dropout", ParameterDef::Uniform(0.0, 0.5));

// Create optimizer
let mut optimizer = BayesianOptimizer::new(space, 100); // 100 trials

// Optimization loop
while let Some(trial) = optimizer.ask()? {
    let params = &trial.params;

    // Train with these parameters
    let accuracy = train_model(
        params.get("learning_rate").unwrap().as_f64(),
        params.get("batch_size").unwrap().as_i64() as usize,
        params.get("hidden_dim").unwrap().as_f64() as usize,
        params.get("dropout").unwrap().as_f64(),
    )?;

    // Report result
    optimizer.tell(trial.id, accuracy, TrialState::Complete)?;
}

// Get best parameters
let best = optimizer.best_trial()?;
println!("Best accuracy: {}", best.value.unwrap());
println!("Best params: {:?}", best.params);
```

## Parameter Types

```rust
use entrenar::optim::hpo::ParameterDef;

// Continuous uniform [low, high]
ParameterDef::Uniform(0.0, 1.0)

// Log-uniform (good for learning rates)
ParameterDef::LogUniform(1e-5, 1e-1)

// Discrete choices
ParameterDef::Discrete(vec![32, 64, 128, 256])

// Categorical (strings)
ParameterDef::Categorical(vec![
    "relu".to_string(),
    "gelu".to_string(),
    "swish".to_string(),
])

// Integer range
ParameterDef::IntUniform(1, 10)
```

## Grid Search

For exhaustive search over discrete spaces:

```rust
use entrenar::optim::hpo::GridSearch;

let grid = GridSearch::new()
    .add("lr", vec![0.001, 0.01, 0.1])
    .add("batch_size", vec![32, 64])
    .add("optimizer", vec!["adam", "sgd"]);

for config in grid.iter() {
    let result = train_with_config(&config)?;
    println!("{:?} -> {}", config, result);
}
```

## Random Search

For baseline comparison:

```rust
use entrenar::optim::hpo::RandomSearch;

let search = RandomSearch::new(space, 50); // 50 random trials

for trial in search.iter() {
    let result = train_with_params(&trial.params)?;
    println!("Trial {}: {}", trial.id, result);
}
```

## Early Stopping

Prune unpromising trials:

```rust
use entrenar::optim::hpo::{BayesianOptimizer, MedianPruner};

let pruner = MedianPruner::new()
    .with_n_startup_trials(5)
    .with_n_warmup_steps(10);

let mut optimizer = BayesianOptimizer::new(space, 100)
    .with_pruner(pruner);

// During training
for epoch in 0..100 {
    let loss = train_epoch(&model);

    // Check if should prune
    if optimizer.should_prune(trial_id, epoch, loss)? {
        optimizer.tell(trial_id, loss, TrialState::Pruned)?;
        break;
    }
}
```

## Multi-Objective Optimization

Optimize multiple objectives simultaneously:

```rust
use entrenar::optim::hpo::MultiObjectiveOptimizer;

let mut optimizer = MultiObjectiveOptimizer::new(space)
    .add_objective("accuracy", true)   // maximize
    .add_objective("latency", false);  // minimize

while let Some(trial) = optimizer.ask()? {
    let (accuracy, latency) = evaluate(&trial.params)?;
    optimizer.tell(trial.id, vec![accuracy, latency])?;
}

// Get Pareto front
let pareto_front = optimizer.pareto_front()?;
```

## Parallel Trials

Run multiple trials concurrently:

```rust
use entrenar::optim::hpo::BayesianOptimizer;
use std::thread;

let optimizer = Arc::new(Mutex::new(BayesianOptimizer::new(space, 100)));

let handles: Vec<_> = (0..4).map(|_| {
    let opt = Arc::clone(&optimizer);
    thread::spawn(move || {
        loop {
            let trial = {
                let mut opt = opt.lock().unwrap();
                opt.ask()
            };

            if let Some(trial) = trial {
                let result = train_with_params(&trial.params);
                let mut opt = opt.lock().unwrap();
                opt.tell(trial.id, result, TrialState::Complete);
            } else {
                break;
            }
        }
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

## Cargo Run Example

```bash
# Run HPO sweep
cargo run --example hpo_sweep

# With specific search space
cargo run --example hpo_sweep -- --trials 50 --space config/search_space.yaml

# Resume from checkpoint
cargo run --example hpo_sweep -- --resume hpo_checkpoint.json
```

## Visualization

```rust
// Export trials for visualization
let history = optimizer.history()?;

// Save as JSON for plotting
let json = serde_json::to_string_pretty(&history)?;
std::fs::write("hpo_history.json", json)?;
```

## Integration with Experiment Tracking

```rust
use entrenar::storage::SqliteBackend;
use entrenar::optim::hpo::BayesianOptimizer;

let mut storage = SqliteBackend::open("experiments.db")?;
let exp_id = storage.create_experiment("hpo-sweep", None)?;

while let Some(trial) = optimizer.ask()? {
    let run_id = storage.create_run(&exp_id)?;
    storage.start_run(&run_id)?;

    // Log hyperparameters
    for (name, value) in &trial.params {
        storage.log_param(&run_id, name, value.clone().into())?;
    }

    // Train and evaluate
    let result = train_with_params(&trial.params)?;
    storage.log_metric(&run_id, "final_accuracy", 0, result)?;

    storage.complete_run(&run_id, RunStatus::Success)?;
    optimizer.tell(trial.id, result, TrialState::Complete)?;
}
```

## Best Practices

1. **Start with random search** - Establish baseline
2. **Use log-uniform for learning rates** - Spans orders of magnitude
3. **Enable early stopping** - Save compute on bad trials
4. **Run enough trials** - TPE needs ~20 trials to model well
5. **Log everything** - Integrate with experiment tracking

## See Also

- [MLOps Overview](./overview.md)
- [Experiment Tracking](./experiment-tracking.md)
- [Learning Rate Tuning](../best-practices/lr-tuning.md)
