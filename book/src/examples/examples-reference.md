# Examples Reference

This page provides a complete reference for all `cargo run --example` commands available in Aprender.

## Quick Reference

| Example | Description | Category |
|---------|-------------|----------|
| `linear_regression` | Basic linear regression | Supervised |
| `logistic_regression` | Binary classification | Supervised |
| `decision_tree_iris` | Decision tree classifier | Supervised |
| `random_forest_iris` | Random forest classifier | Supervised |
| `gbm_iris` | Gradient boosting classifier | Supervised |
| `naive_bayes_iris` | Naive Bayes classifier | Supervised |
| `knn_iris` | K-nearest neighbors | Supervised |
| `svm_iris` | Support vector machine | Supervised |
| `kmeans_clustering` | K-means unsupervised | Unsupervised |
| `pca_iris` | Dimensionality reduction | Unsupervised |
| `time_series_forecasting` | ARIMA forecasting | Time Series |
| `text_preprocessing` | NLP text processing | NLP |
| `qwen_inference` | LLM inference | Deep Learning |

## Running Examples

### Basic Usage

```bash
# Run with default settings
cargo run --example <name>

# Run in release mode (10-20x faster)
cargo run --example <name> --release

# With feature flags
cargo run --example <name> --features inference

# With arguments
cargo run --example <name> -- arg1 arg2
```

## Supervised Learning

### Linear Regression

```bash
cargo run --example linear_regression --release
cargo run --example regularized_regression --release
cargo run --example boston_housing --release
```

### Classification

```bash
cargo run --example logistic_regression --release
cargo run --example decision_tree_iris --release
cargo run --example random_forest_iris --release
cargo run --example gbm_iris --release
cargo run --example naive_bayes_iris --release
cargo run --example knn_iris --release
cargo run --example svm_iris --release
```

### Bayesian Inference

```bash
cargo run --example bayesian_linear_regression --release
cargo run --example bayesian_logistic_regression --release
cargo run --example beta_binomial_inference --release
cargo run --example gamma_poisson_inference --release
cargo run --example dirichlet_multinomial_inference --release
cargo run --example normal_inverse_gamma_inference --release
```

### Generalized Linear Models

```bash
cargo run --example negative_binomial_glm --release
```

## Unsupervised Learning

### Clustering

```bash
cargo run --example iris_clustering --release
cargo run --example dbscan_clustering --release
cargo run --example hierarchical_clustering --release
cargo run --example gmm_clustering --release
cargo run --example spectral_clustering --release
```

### Dimensionality Reduction

```bash
cargo run --example pca_iris --release
cargo run --example tsne_visualization --release
```

### Anomaly Detection

```bash
cargo run --example isolation_forest_anomaly --release
cargo run --example lof_anomaly --release
```

## Deep Learning

### Neural Networks

```bash
cargo run --example xor_training --release
cargo run --example neural_network_training --release
cargo run --example classification_training --release
```

### LLM Inference

```bash
# Qwen model inference (requires model file)
cargo run --example qwen_inference --release --features inference

# Whisper transcription
cargo run --example whisper_transcribe --release --features inference

# HuggingFace model import
cargo run --example phi_hf_import --release
```

## Time Series

```bash
cargo run --example time_series_forecasting --release
```

## NLP / Text Processing

```bash
cargo run --example text_preprocessing --release
cargo run --example text_classification --release
cargo run --example nlp_advanced --release
cargo run --example topic_sentiment_analysis --release
```

## Graph Algorithms

```bash
cargo run --example graph_algorithms_comprehensive --release
cargo run --example graph_social_network --release
cargo run --example community_detection --release
```

## Optimization

### Gradient-Based

```bash
cargo run --example optimizer_demo --release
cargo run --example batch_optimization --release
cargo run --example convex_optimization --release
cargo run --example constrained_optimization --release
cargo run --example admm_optimization --release
```

### Metaheuristics

```bash
cargo run --example metaheuristics_optimization --release
cargo run --example aco_tsp --release
cargo run --example tabu_tsp --release
cargo run --example predator_prey_optimization --release
```

## Model Operations

### APR Format

```bash
cargo run --example apr_loading_modes --release
cargo run --example apr_inspection --release
cargo run --example apr_scoring --release
cargo run --example apr_cache --release
cargo run --example apr_embed --release
cargo run --example apr_with_metadata --release
cargo run --example apr_cli_commands --release
cargo run --example create_test_apr --release
```

### Model Serialization

```bash
cargo run --example model_serialization --release
cargo run --example shell_model_format --release
cargo run --example shell_encryption_demo --release
```

## Data Processing

```bash
cargo run --example dataframe_basics --release
cargo run --example data_preprocessing_scalers --release
cargo run --example synthetic_data_generation --release
cargo run --example descriptive_statistics --release
cargo run --example bayesian_blocks_histogram --release
```

## Recommendations

```bash
cargo run --example recommend_content --release
```

## Pattern Mining

```bash
cargo run --example market_basket_apriori --release
```

## AutoML

```bash
cargo run --example automl_clustering --release
cargo run --example grid_search_tuning --release
cargo run --example cross_validation --release
```

## GPU / CUDA

```bash
cargo run --example cuda_backend --release
cargo run --example trueno_compute_integration --release
```

## Model Zoo

```bash
cargo run --example model_zoo --release
```

## Sovereign AI Stack

```bash
cargo run --example sovereign_stack --release
cargo run --example sovereign_offline --release
```

## Pipeline & Validation

```bash
cargo run --example pipeline_verification --release
cargo run --example poka_yoke_validation --release
```

## Advanced

### Mixture of Experts

```bash
cargo run --example mixture_of_experts --release
```

### Online Learning

```bash
cargo run --example online_learning --release
```

### Code Analysis

```bash
cargo run --example code_analysis --release
```

### Logic Programming

```bash
cargo run --example logic_family_tree --release
```

## See Also

- [Case Studies](./linear-regression.md) - Detailed walkthroughs
- [APR CLI Tool](../tools/apr-cli.md) - Command-line interface
- [APR Format Specification](../tools/apr-spec.md) - Model format details
