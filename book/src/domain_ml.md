# ML Training Simulation

The ML domain provides training simulation, prediction, and multi-turn evaluation capabilities.

## Training Simulation

Simulate ML training runs with anomaly detection:

```rust,ignore
use simular::domains::ml::{
    TrainingSimulation, TrainingConfig, TrainingState,
};

let config = TrainingConfig {
    learning_rate: 0.001,
    batch_size: 32,
    max_epochs: 100,
    early_stopping_patience: 10,
    ..Default::default()
};

let mut sim = TrainingSimulation::new(config);

// Simulate training loop
while !sim.is_complete() {
    let state: &TrainingState = sim.step();

    println!("Epoch {}: loss = {:.4}, val_loss = {:.4}",
        state.epoch, state.train_loss, state.val_loss);

    if state.should_stop {
        println!("Early stopping triggered");
        break;
    }
}

let metrics = sim.final_metrics();
println!("Final: best_val_loss = {:.4}", metrics.best_val_loss);
```

## Training Metrics

```rust,ignore
pub struct TrainingMetrics {
    pub epochs_completed: usize,
    pub best_val_loss: f64,
    pub best_epoch: usize,
    pub final_train_loss: f64,
    pub final_val_loss: f64,
    pub training_time_estimate: f64,
}
```

## Training Trajectory

Record and analyze training history:

```rust,ignore
use simular::domains::ml::TrainingTrajectory;

let trajectory = TrainingTrajectory::new();

// Add epoch results
trajectory.add_epoch(1, 0.5, 0.52);
trajectory.add_epoch(2, 0.4, 0.43);
trajectory.add_epoch(3, 0.35, 0.38);

// Analyze
println!("Best epoch: {}", trajectory.best_epoch());
println!("Converged: {}", trajectory.has_converged(0.01));
println!("Overfitting: {}", trajectory.is_overfitting());
```

## Anomaly Detection

Detect training anomalies with Jidoka:

```rust,ignore
use simular::domains::ml::{AnomalyDetector, TrainingAnomaly};

let mut detector = AnomalyDetector::new();

// Monitor training
detector.observe(epoch, train_loss, val_loss);

// Check for anomalies
match detector.check() {
    Some(TrainingAnomaly::LossExplosion) => {
        println!("Loss explosion detected!");
    }
    Some(TrainingAnomaly::GradientVanishing) => {
        println!("Vanishing gradients detected!");
    }
    Some(TrainingAnomaly::Overfitting { gap }) => {
        println!("Overfitting detected, gap = {:.4}", gap);
    }
    None => {}
}
```

## Rolling Statistics

Track statistics over a window:

```rust,ignore
use simular::domains::ml::RollingStats;

let mut stats = RollingStats::new(10);  // Window size 10

for loss in losses {
    stats.add(loss);
    println!("Mean: {:.4}, Std: {:.4}",
        stats.mean(), stats.std());
}
```

## Prediction Simulation

Simulate inference:

```rust,ignore
use simular::domains::ml::{PredictionSimulation, PredictionState, InferenceConfig};

let config = InferenceConfig {
    batch_size: 64,
    use_gpu: false,
    ..Default::default()
};

let mut sim = PredictionSimulation::new(config);

// Run inference
let state: &PredictionState = sim.predict(&input_data);
println!("Latency: {:.2}ms", state.latency_ms);
println!("Throughput: {:.0} samples/sec", state.throughput);
```

## Multi-Turn Evaluation

Evaluate multi-turn interactions:

```rust,ignore
use simular::domains::ml::{MultiTurnSimulation, Turn, TurnMetrics};

let mut sim = MultiTurnSimulation::new();

// Add turns
sim.add_turn(Turn {
    input: "Hello".to_string(),
    output: "Hi there!".to_string(),
    latency_ms: 150.0,
    token_count: 5,
});

sim.add_turn(Turn {
    input: "How are you?".to_string(),
    output: "I'm doing well, thanks!".to_string(),
    latency_ms: 200.0,
    token_count: 8,
});

// Evaluate
let eval = sim.evaluate();
println!("Total turns: {}", eval.total_turns);
println!("Avg latency: {:.2}ms", eval.avg_latency_ms);
println!("Total tokens: {}", eval.total_tokens);
```

## Pareto Analysis

Analyze trade-offs:

```rust,ignore
use simular::domains::ml::{ParetoPoint, ParetoAnalysis};

let points = vec![
    ParetoPoint { accuracy: 0.95, latency: 100.0 },
    ParetoPoint { accuracy: 0.92, latency: 50.0 },
    ParetoPoint { accuracy: 0.98, latency: 200.0 },
    ParetoPoint { accuracy: 0.90, latency: 30.0 },
];

let analysis = ParetoAnalysis::new(points);
let frontier = analysis.pareto_frontier();

println!("Pareto-optimal configurations:");
for point in frontier {
    println!("  Accuracy: {:.2}%, Latency: {:.0}ms",
        point.accuracy * 100.0, point.latency);
}
```

## Jidoka ML Feedback

Automatic rule patching for ML pipelines:

```rust,ignore
use simular::domains::ml::{JidokaMLFeedback, RulePatch, AnomalyPattern};

let feedback = JidokaMLFeedback::new();

// Observe anomaly
let pattern = AnomalyPattern {
    anomaly_type: AnomalyType::LossExplosion,
    frequency: 0.05,
    context: "high learning rate".to_string(),
};

// Get suggested patch
let patch: RulePatch = feedback.suggest_patch(&pattern);
println!("Suggested fix: {:?}", patch);
```

## Example: Training Run Analysis

```rust,ignore
use simular::domains::ml::{
    TrainingSimulation, TrainingConfig, AnomalyDetector,
};

fn main() {
    let config = TrainingConfig {
        learning_rate: 0.01,
        batch_size: 32,
        max_epochs: 50,
        early_stopping_patience: 5,
        ..Default::default()
    };

    let mut sim = TrainingSimulation::new(config);
    let mut detector = AnomalyDetector::new();

    while !sim.is_complete() {
        let state = sim.step();

        // Monitor for anomalies
        detector.observe(state.epoch, state.train_loss, state.val_loss);

        if let Some(anomaly) = detector.check() {
            println!("Anomaly detected at epoch {}: {:?}",
                state.epoch, anomaly);

            // Could adjust learning rate, stop training, etc.
        }

        if state.epoch % 10 == 0 {
            println!("Epoch {:>3}: train={:.4}, val={:.4}",
                state.epoch, state.train_loss, state.val_loss);
        }
    }

    let metrics = sim.final_metrics();
    println!("\nTraining complete:");
    println!("  Best val loss: {:.4} at epoch {}",
        metrics.best_val_loss, metrics.best_epoch);
}
```

## Next Steps

- [Jidoka Guards](./engine_jidoka.md) - Anomaly detection in detail
- [Bayesian Optimization](./domain_optimization.md) - Hyperparameter tuning
