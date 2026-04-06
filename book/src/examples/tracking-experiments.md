# Tracking Experiments

This example demonstrates experiment tracking in Pacha.

## Running the Example

```bash
cargo run --example experiment_tracking
```

## Experiment Tracking Features

- **Run tracking** - Track individual training executions
- **Metric logging** - Log loss, accuracy, and custom metrics over time
- **Hyperparameter storage** - Record the exact parameters used
- **Run comparison** - Find the best run by any metric

## Creating a Training Recipe

```rust
let recipe = TrainingRecipe::builder()
    .name("fraud-training")
    .version(RecipeVersion::new(1, 0, 0))
    .description("Fraud detection training recipe")
    .hyperparameters(
        Hyperparameters::builder()
            .learning_rate(1e-3)
            .batch_size(32)
            .epochs(10)
            .build(),
    )
    .build();

registry.register_recipe(&recipe)?;
```

## Creating an Experiment Run

```rust
let hyperparams = Hyperparameters::builder()
    .learning_rate(1e-4)
    .batch_size(64)
    .epochs(20)
    .build();

let mut run = ExperimentRun::from_recipe(recipe.reference(), hyperparams);
run.start();
```

## Logging Metrics

```rust
// During training loop
for epoch in 0..epochs {
    let loss = train_epoch(&model, &data);
    run.log_metric("loss", loss, epoch as u64);
    run.log_metric("accuracy", accuracy, epoch as u64);
}
```

## Completing the Run

```rust
run.complete();  // On success
run.fail("Out of memory");  // On failure
run.cancel();  // On cancellation
```

## Finding the Best Run

```rust
let runs = registry.list_runs(&recipe.reference())?;

let best = runs
    .iter()
    .filter(|r| r.status == RunStatus::Completed)
    .max_by(|a, b| {
        let auc_a = a.get_metric("auc").unwrap_or(0.0);
        let auc_b = b.get_metric("auc").unwrap_or(0.0);
        auc_a.partial_cmp(&auc_b).unwrap()
    });
```
