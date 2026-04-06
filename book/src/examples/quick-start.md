# Quick Start

This example demonstrates the basic operations of the Pacha registry.

## Running the Example

```bash
cargo run --example quick_start
```

## What It Does

1. **Creates a temporary registry** - Uses `tempfile` for isolated testing
2. **Registers a model** - Creates a fraud detector with metrics and documentation
3. **Registers a dataset** - Adds transaction data with a datasheet
4. **Queries artifacts** - Retrieves and displays model information
5. **Transitions stages** - Promotes the model from development to staging
6. **Shows statistics** - Displays registry usage metrics

## Code Walkthrough

### Creating the Registry

```rust
let temp_dir = TempDir::new().expect("Failed to create temp dir");
let config = RegistryConfig::new(temp_dir.path());
let registry = Registry::open(config)?;
```

### Registering a Model

```rust
let card = ModelCard::builder()
    .description("Fraud detection model")
    .metrics([("auc", 0.95), ("f1", 0.88)])
    .primary_uses(["Fraud detection in payment transactions"])
    .build();

let model_id = registry.register_model(
    "fraud-detector",
    &ModelVersion::new(1, 0, 0),
    model_data,
    card,
)?;
```

### Registering a Dataset

```rust
let datasheet = Datasheet::builder()
    .purpose("Transaction data for fraud detection")
    .creators(["Data Engineering Team"])
    .instance_count(1_000_000)
    .license("Internal Use Only")
    .build();

let dataset_id = registry.register_dataset(
    "transactions",
    &DatasetVersion::new(1, 0, 0),
    dataset_data,
    datasheet,
)?;
```

### Stage Transition

```rust
registry.transition_model_stage(
    "fraud-detector",
    &ModelVersion::new(1, 0, 0),
    ModelStage::Staging,
)?;
```

## Expected Output

```
=== Pacha Quick Start ===

1. Registering a model...
   Registered model ID: 1cd6809b-6b55-48f5-b619-a5ac4930339b

2. Registering a dataset...
   Registered dataset ID: b6cd66ab-5b48-4e7d-ab75-4dd229edc6c8

3. Querying the model...
   Model: fraud-detector:1.0.0
   Stage: development

4. Promoting model to staging...
   New stage: staging

5. Registry statistics:
   Models: 1
   Datasets: 1
   Objects: 2
   Total size: 89 bytes

✅ Quick start complete!
```
