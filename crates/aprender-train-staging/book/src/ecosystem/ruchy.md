# Ruchy Session Bridge

The Ruchy session bridge preserves training history from interactive
Ruchy sessions, converting them to Entrenar artifacts for
reproducibility and archival.

## Feature Flag

Enable the session bridge:

```toml
[dependencies]
entrenar = { version = "0.2", features = ["ruchy-sessions"] }
```

## EntrenarSession

Represents a training session with metrics and code history:

```rust
use entrenar::ecosystem::EntrenarSession;

let session = EntrenarSession::new("sess-001", "LoRA Fine-tuning")
    .with_user("alice")
    .with_architecture("llama-7b")
    .with_dataset("custom-dataset")
    .with_config("batch_size", "32")
    .with_config("learning_rate", "1e-4")
    .with_tag("fine-tuning")
    .with_tag("lora")
    .with_notes("Initial experiment with rank 64");
```

## SessionMetrics

Track training metrics over time:

```rust
let mut session = EntrenarSession::new("sess-001", "Training");

// Log metrics
session.metrics.add_loss(0.5);
session.metrics.add_loss(0.3);
session.metrics.add_loss(0.2);

session.metrics.add_accuracy(75.0);
session.metrics.add_accuracy(85.0);

session.metrics.add_lr(0.001);
session.metrics.add_grad_norm(1.5);

// Custom metrics
session.metrics.add_custom("f1_score", 0.82);
session.metrics.add_custom("bleu", 0.45);

// Statistics
println!("Steps: {}", session.metrics.total_steps());
println!("Final loss: {:?}", session.metrics.final_loss());
println!("Best loss: {:?}", session.metrics.best_loss());
println!("Final accuracy: {:?}", session.metrics.final_accuracy());
println!("Best accuracy: {:?}", session.metrics.best_accuracy());
```

## Code History

Capture executed code cells:

```rust
use entrenar::ecosystem::CodeCell;

let cell = CodeCell {
    execution_order: 1,
    source: r#"
model = load_model("llama-7b")
trainer = Trainer(model, lr=1e-4)
trainer.train(epochs=10)
    "#.to_string(),
    output: Some("Training completed. Final loss: 0.2".to_string()),
    timestamp: chrono::Utc::now(),
    duration_ms: Some(45000),
};

session.add_code_cell(cell);
```

## Session Lifecycle

```rust
// Create and track session
let mut session = EntrenarSession::new("sess-001", "Training")
    .with_user("bob");

// Log during training
for epoch in 0..10 {
    let loss = train_epoch();
    session.metrics.add_loss(loss);
}

// Check if session has training data
if session.has_training_data() {
    println!("Recorded {} steps", session.metrics.total_steps());
}

// Mark session as ended
session.end();

// Get duration
if let Some(duration) = session.duration() {
    println!("Session lasted {} hours", duration.num_hours());
}
```

## Converting from Ruchy

Convert a Ruchy session to EntrenarSession:

```rust
use entrenar::ecosystem::{EntrenarSession, RuchySession};

// RuchySession comes from the Ruchy crate
let ruchy_session: RuchySession = /* ... */;

// Convert to EntrenarSession
let session: EntrenarSession = ruchy_session.into();

println!("Session: {}", session.name);
println!("User: {:?}", session.user);
println!("Steps: {}", session.metrics.total_steps());
```

## Converting to Research Artifact

Preserve session as a research artifact:

```rust
use entrenar::ecosystem::session_to_artifact;

let mut session = EntrenarSession::new("sess-001", "LoRA Experiment")
    .with_user("alice")
    .with_architecture("llama-7b")
    .with_tag("lora")
    .with_tag("fine-tuning");

session.metrics.add_loss(0.5);
session.metrics.add_loss(0.2);

// Convert to artifact
let artifact = session_to_artifact(&session)?;

println!("Artifact ID: {}", artifact.id);
println!("Type: {}", artifact.artifact_type);  // Notebook
println!("Authors: {:?}", artifact.authors);
println!("Keywords: {:?}", artifact.keywords);
println!("Version: {}", artifact.version);  // "1.0.0+steps2"
```

### Artifact Properties

The conversion:

- Sets artifact type to `Notebook`
- Adds user as author with `Software` and `Investigation` roles
- Generates description from session metrics
- Copies tags as keywords (or defaults to ["training", "experiment", "entrenar"])
- Sets version with step count suffix

## Error Handling

```rust
use entrenar::ecosystem::RuchyBridgeError;

let session = EntrenarSession::new("empty", "Empty Session");

match session_to_artifact(&session) {
    Ok(artifact) => println!("Created: {}", artifact.id),
    Err(RuchyBridgeError::NoTrainingHistory) => {
        eprintln!("Session has no training data or code");
    }
    Err(e) => eprintln!("Conversion failed: {}", e),
}
```

## Full Workflow Example

```rust
use entrenar::ecosystem::{EntrenarSession, CodeCell, session_to_artifact};
use entrenar::research::{CitationMetadata, ArchiveDeposit, ZenodoConfig};

// 1. Create session
let mut session = EntrenarSession::new("exp-2024-001", "Temperature Ablation Study")
    .with_user("researcher@university.edu")
    .with_architecture("llama-2-7b")
    .with_dataset("alpaca-clean")
    .with_config("temperature", "4.0")
    .with_config("alpha", "0.7")
    .with_tag("distillation")
    .with_tag("ablation");

// 2. Log training progress
for epoch in 0..50 {
    let loss = train_epoch();
    session.metrics.add_loss(loss);

    if epoch % 10 == 0 {
        let accuracy = evaluate();
        session.metrics.add_accuracy(accuracy);
    }
}

// 3. Capture final code
session.add_code_cell(CodeCell {
    execution_order: 1,
    source: "# Training code...".to_string(),
    output: Some("Training complete".to_string()),
    timestamp: chrono::Utc::now(),
    duration_ms: Some(3600000),
});

// 4. End session
session.end();

// 5. Convert to artifact
let artifact = session_to_artifact(&session)?;

// 6. Generate citation
let citation = CitationMetadata::from_artifact(&artifact, 2024);
println!("{}", citation.to_bibtex());

// 7. Optionally deposit to archive
// let deposit = ArchiveDeposit::new(ZenodoConfig::new("your-token"));
// deposit.prepare(&artifact)?;
```

## See Also

- [Ecosystem Overview](./overview.md)
- [Research Artifacts](../research/overview.md)
- [Academic Research](../research/overview.md)
