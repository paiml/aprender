# Metrics and Model Evaluation

Aprender v0.10 introduces comprehensive metrics for evaluating ML models, including regression metrics, classification metrics, model comparison, and drift detection.

## Regression Metrics

### R² (Coefficient of Determination)

```rust
use aprender::metrics::r_squared;
use aprender::primitives::Vector;

let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);

let r2 = r_squared(&y_pred, &y_true);
println!("R² Score: {:.4}", r2);  // ~0.9486
```

### MSE, RMSE, MAE

```rust
use aprender::metrics::{mse, rmse, mae};

let y_true = Vector::from_slice(&[3.0, -0.5, 2.0, 7.0]);
let y_pred = Vector::from_slice(&[2.5, 0.0, 2.0, 8.0]);

println!("MSE:  {:.4}", mse(&y_pred, &y_true));   // 0.375
println!("RMSE: {:.4}", rmse(&y_pred, &y_true));  // 0.6124
println!("MAE:  {:.4}", mae(&y_pred, &y_true));   // 0.5
```

## Classification Metrics

### Accuracy

```rust
use aprender::metrics::classification::accuracy;

let y_true = vec![0, 1, 2, 0, 1, 2];
let y_pred = vec![0, 1, 2, 0, 2, 1];  // 4/6 correct

let acc = accuracy(&y_pred, &y_true);
println!("Accuracy: {:.2}%", acc * 100.0);  // 66.67%
```

### Precision, Recall, F1-Score

```rust
use aprender::metrics::classification::{precision, recall, f1_score, Average};

let y_true = vec![0, 1, 1, 0, 1, 0];
let y_pred = vec![0, 1, 0, 0, 1, 1];

// Macro averaging (unweighted mean across classes)
let prec = precision(&y_pred, &y_true, Average::Macro);
let rec = recall(&y_pred, &y_true, Average::Macro);
let f1 = f1_score(&y_pred, &y_true, Average::Macro);

println!("Precision: {:.4}", prec);
println!("Recall:    {:.4}", rec);
println!("F1-Score:  {:.4}", f1);
```

### Averaging Strategies

| Strategy | Description |
|----------|-------------|
| `Average::Macro` | Unweighted mean across classes |
| `Average::Micro` | Global TP, FP, FN counts |
| `Average::Weighted` | Weighted by class support |

## Model Evaluation

### ModelEvaluator

Compare multiple models with cross-validation:

```rust
use aprender::metrics::evaluator::{ModelEvaluator, TaskType};
use aprender::linear::LinearRegression;
use aprender::tree::DecisionTreeRegressor;

// Prepare data
let X = Matrix::from_slice(100, 4, &features);
let y = Vector::from_slice(&targets);

// Create evaluator
let evaluator = ModelEvaluator::new(TaskType::Regression)
    .with_cv_folds(5)
    .with_metric("r2");

// Evaluate models
let lr = LinearRegression::new();
let dt = DecisionTreeRegressor::new(5);

let results = evaluator.compare(vec![
    ("Linear Regression", &lr),
    ("Decision Tree", &dt),
], &X, &y)?;

// Get best model
if let Some(best) = results.best_model() {
    println!("Best model: {} (R² = {:.4})", best.name, best.mean_score);
}
```

### ModelResult

```rust
pub struct ModelResult {
    pub name: String,
    pub cv_scores: Vec<f32>,     // Score per fold
    pub mean_score: f32,         // Mean CV score
    pub std_score: f32,          // Standard deviation
    pub train_time_ms: Option<u64>,
    pub metrics: HashMap<String, f32>,
}
```

### ComparisonResult

```rust
pub struct ComparisonResult {
    pub models: Vec<ModelResult>,
    pub task_type: TaskType,
    pub primary_metric: String,
}

impl ComparisonResult {
    pub fn best_model(&self) -> Option<&ModelResult>;
    pub fn ranked_models(&self) -> Vec<&ModelResult>;
    pub fn to_markdown(&self) -> String;
}
```

## Integration with Realizar

Use aprender metrics in Lambda handlers:

```rust
use realizar::lambda::{LambdaHandler, LambdaRequest, LambdaMetrics};
use aprender::metrics::classification::{accuracy, f1_score, Average};

// Track prediction quality
let mut predictions = Vec::new();
let mut actuals = Vec::new();

// After collecting feedback...
let acc = accuracy(&predictions, &actuals);
let f1 = f1_score(&predictions, &actuals, Average::Macro);

// Export to Prometheus
let mut metrics = LambdaMetrics::new();
metrics.add_custom("model_accuracy", acc);
metrics.add_custom("model_f1_score", f1);
```

## Best Practices

### Metric Selection

| Task | Primary Metric | Secondary Metrics |
|------|----------------|-------------------|
| Regression | R² | RMSE, MAE |
| Binary Classification | F1-Score | Precision, Recall, AUC |
| Multi-class | Macro F1 | Accuracy, Weighted F1 |
| Imbalanced | Weighted F1 | Precision per class |

### Cross-Validation

```rust
// Use 5-fold CV for stable estimates
let evaluator = ModelEvaluator::new(TaskType::Classification)
    .with_cv_folds(5)
    .with_stratified(true);  // Preserve class distribution
```

### Reporting

```rust
// Generate markdown report
let report = results.to_markdown();
println!("{}", report);

// Output:
// | Model | Mean Score | Std | Train Time |
// |-------|------------|-----|------------|
// | Random Forest | 0.9234 | 0.0123 | 45ms |
// | Logistic Regression | 0.8876 | 0.0089 | 12ms |
```
