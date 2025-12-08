# Case Study: Model Zoo

This example demonstrates the Model Zoo protocol for model sharing and discovery, providing standardized metadata and quality scoring.

## Overview

The Model Zoo provides:
- **Standardized model metadata format**
- **Quality score caching for quick filtering**
- **Version management**
- **Popularity metrics**
- **Search and discovery**

## Running the Example

```bash
cargo run --example model_zoo
```

## Model Zoo Entry

Create comprehensive model entries:

```rust
let entry = ModelZooEntry::new("housing-price-predictor", "Housing Price Predictor")
    .with_description("Linear regression model trained on Boston Housing dataset")
    .with_version("2.1.0")
    .with_author(
        AuthorInfo::new("Jane Doe", "jane@example.com")
            .with_organization("Acme ML Labs")
            .with_url("https://jane.example.com"),
    )
    .with_model_type(ModelZooType::LinearRegression)
    .with_quality_score(87.5)
    .with_tag("regression")
    .with_tag("housing")
    .with_tag("tabular")
    .with_download_url("https://models.example.com/housing-v2.1.0.apr")
    .with_size(1024 * 1024 * 5)  // 5 MB
    .with_sha256("abc123def456...")
    .with_license("Apache-2.0")
    .with_timestamps("2024-01-15T10:30:00Z", "2024-12-01T14:22:00Z")
    .with_metadata("dataset", "boston_housing")
    .with_metadata("r2_score", "0.91");

println!("{}", entry);
println!("Quality Grade: {}", entry.quality_grade());
println!("Human Size: {}", entry.human_size());
println!("Has Tag 'regression': {}", entry.has_tag("regression"));
println!("Matches 'housing': {}", entry.matches_query("housing"));
```

## Model Types

Supported model categories:

| Type | Category |
|------|----------|
| LinearRegression | Regression |
| LogisticRegression | Classification |
| DecisionTree | Classification |
| RandomForest | Classification |
| GradientBoosting | Classification |
| Knn | Classification |
| KMeans | Clustering |
| Svm | Classification |
| NaiveBayes | Classification |
| NeuralNetwork | DeepLearning |
| TimeSeries | TimeSeries |

## Author Information

```rust
// Basic author
let basic = AuthorInfo::new("John Smith", "john@example.com");

// Full author info
let full = AuthorInfo::new("Alice Johnson", "alice@mlcompany.com")
    .with_organization("ML Company Inc.")
    .with_url("https://alice.mlcompany.com");
```

## Model Zoo Index

Manage collections of models:

```rust
let mut index = ModelZooIndex::new("1.0.0");

// Add models
let models = vec![
    ModelZooEntry::new("iris-classifier", "Iris Flower Classifier")
        .with_model_type(ModelZooType::RandomForest)
        .with_quality_score(92.0)
        .with_tag("classification"),
    ModelZooEntry::new("sentiment-analyzer", "Sentiment Analyzer")
        .with_model_type(ModelZooType::LogisticRegression)
        .with_quality_score(85.0)
        .with_tag("nlp"),
    // ...
];

for model in models {
    index.add_model(model);
}

// Feature models
index.feature_model("iris-classifier");

println!("All Tags: {:?}", index.all_tags());

// Get featured models
for entry in index.get_featured() {
    println!("Featured: {} ({})", entry.name, entry.quality_grade());
}
```

## Search and Filter

### Search by Query

```rust
for entry in index.search("classifier") {
    println!("{} ({:.0})", entry.name, entry.quality_score);
}
```

### Filter by Tag

```rust
for entry in index.filter_by_tag("classification") {
    println!("{}", entry.name);
}
```

### Filter by Category

```rust
for entry in index.filter_by_category(ModelCategory::Clustering) {
    println!("{}", entry.name);
}
```

### Filter by Quality

```rust
// High quality models (>= 85)
for entry in index.filter_by_quality(85.0) {
    println!("{} (grade {})", entry.name, entry.quality_grade());
}
```

### Most Popular

```rust
for entry in index.most_popular(3) {
    println!("{} ({} downloads)", entry.name, entry.downloads);
}
```

### Highest Quality

```rust
for entry in index.highest_quality(3) {
    println!("{} ({:.0})", entry.name, entry.quality_score);
}
```

## Zoo Statistics

```rust
let stats = index.stats();

println!("Total Models: {}", stats.total_models);
println!("Total Downloads: {}", stats.total_downloads);
println!("Total Size: {}", stats.human_total_size());
println!("Average Quality: {:.1}", stats.avg_quality_score);

println!("Category Breakdown:");
for (category, count) in &stats.category_counts {
    println!("  {}: {}", category.name(), count);
}

println!("Top Tags:");
let mut tags: Vec<_> = stats.tag_counts.iter().collect();
tags.sort_by(|a, b| b.1.cmp(a.1));
for (tag, count) in tags.iter().take(5) {
    println!("  {}: {}", tag, count);
}
```

## Quality Grades

Based on the 100-point scoring system:

| Grade | Score Range |
|-------|-------------|
| A+ | 97-100 |
| A | 93-96 |
| A- | 90-92 |
| B+ | 87-89 |
| B | 83-86 |
| B- | 80-82 |
| C+ | 77-79 |
| C | 73-76 |
| C- | 70-72 |
| D | 60-69 |
| F | <60 |

## Source Code

- Example: `examples/model_zoo.rs`
- Module: `src/zoo/mod.rs`
