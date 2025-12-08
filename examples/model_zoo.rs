//! Model Zoo Example
//!
//! Demonstrates the Model Zoo protocol for model sharing and discovery:
//! - Standardized model metadata format
//! - Quality score caching for quick filtering
//! - Version management and popularity metrics
//!
//! Run with: `cargo run --example model_zoo`

use aprender::zoo::{AuthorInfo, ModelCategory, ModelZooEntry, ModelZooIndex, ModelZooType};

fn main() {
    println!("=== Model Zoo Demo ===\n");

    // Part 1: Model Zoo Entry
    model_entry_demo();

    // Part 2: Model Types
    model_types_demo();

    // Part 3: Author Info
    author_info_demo();

    // Part 4: Model Zoo Index
    model_zoo_index_demo();

    // Part 5: Search and Filter
    search_filter_demo();

    // Part 6: Zoo Statistics
    zoo_stats_demo();

    println!("\n=== Model Zoo Demo Complete! ===");
}

fn model_entry_demo() {
    println!("--- Part 1: Model Zoo Entry ---\n");

    // Create a comprehensive model entry
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
    println!("  Quality Grade: {}", entry.quality_grade());
    println!("  Human Size: {}", entry.human_size());
    println!("  Has Tag 'regression': {}", entry.has_tag("regression"));
    println!("  Matches 'housing': {}", entry.matches_query("housing"));
    println!();
}

fn model_types_demo() {
    println!("--- Part 2: Model Types ---\n");

    let types = [
        ModelZooType::LinearRegression,
        ModelZooType::LogisticRegression,
        ModelZooType::DecisionTree,
        ModelZooType::RandomForest,
        ModelZooType::GradientBoosting,
        ModelZooType::Knn,
        ModelZooType::KMeans,
        ModelZooType::Svm,
        ModelZooType::NaiveBayes,
        ModelZooType::NeuralNetwork,
        ModelZooType::TimeSeries,
    ];

    println!("{:<20} {:<15}", "Model Type", "Category");
    println!("{}", "-".repeat(40));

    for model_type in &types {
        println!(
            "{:<20} {:<15}",
            model_type.name(),
            model_type.category().name()
        );
    }
    println!();
}

fn author_info_demo() {
    println!("--- Part 3: Author Info ---\n");

    // Basic author
    let basic = AuthorInfo::new("John Smith", "john@example.com");
    println!("Basic Author: {}", basic);

    // Full author info
    let full = AuthorInfo::new("Alice Johnson", "alice@mlcompany.com")
        .with_organization("ML Company Inc.")
        .with_url("https://alice.mlcompany.com");

    println!("Full Author: {}", full);
    println!("  Organization: {:?}", full.organization);
    println!("  URL: {:?}", full.url);
    println!();
}

fn model_zoo_index_demo() {
    println!("--- Part 4: Model Zoo Index ---\n");

    let mut index = ModelZooIndex::new("1.0.0");

    // Add various models
    let models = vec![
        ModelZooEntry::new("iris-classifier", "Iris Flower Classifier")
            .with_model_type(ModelZooType::RandomForest)
            .with_quality_score(92.0)
            .with_tag("classification")
            .with_tag("beginner"),
        ModelZooEntry::new("sentiment-analyzer", "Sentiment Analyzer")
            .with_model_type(ModelZooType::LogisticRegression)
            .with_quality_score(85.0)
            .with_tag("nlp")
            .with_tag("classification"),
        ModelZooEntry::new("customer-segmentation", "Customer Segmentation")
            .with_model_type(ModelZooType::KMeans)
            .with_quality_score(78.0)
            .with_tag("clustering")
            .with_tag("marketing"),
        ModelZooEntry::new("stock-predictor", "Stock Price Predictor")
            .with_model_type(ModelZooType::TimeSeries)
            .with_quality_score(71.0)
            .with_tag("finance")
            .with_tag("time-series"),
        ModelZooEntry::new("image-classifier", "Image Classifier (ResNet)")
            .with_model_type(ModelZooType::NeuralNetwork)
            .with_quality_score(94.5)
            .with_tag("deep-learning")
            .with_tag("computer-vision"),
    ];

    for model in models {
        index.add_model(model);
    }

    // Feature some models
    index.feature_model("iris-classifier");
    index.feature_model("image-classifier");

    println!("{}", index);

    println!("All Tags: {:?}", index.all_tags());

    // Get featured models
    println!("\nFeatured Models:");
    for entry in index.get_featured() {
        println!("  - {} ({})", entry.name, entry.quality_grade());
    }
    println!();
}

fn search_filter_demo() {
    println!("--- Part 5: Search and Filter ---\n");

    let index = create_sample_index();

    // Search by query
    println!("Search 'classifier':");
    for entry in index.search("classifier") {
        println!("  - {} ({:.0})", entry.name, entry.quality_score);
    }

    // Filter by tag
    println!("\nFilter by tag 'classification':");
    for entry in index.filter_by_tag("classification") {
        println!("  - {} ({:.0})", entry.name, entry.quality_score);
    }

    // Filter by category
    println!("\nFilter by category 'Clustering':");
    for entry in index.filter_by_category(ModelCategory::Clustering) {
        println!("  - {} ({:.0})", entry.name, entry.quality_score);
    }

    // Filter by quality
    println!("\nHigh quality (>= 85):");
    for entry in index.filter_by_quality(85.0) {
        println!(
            "  - {} ({:.0}, grade {})",
            entry.name,
            entry.quality_score,
            entry.quality_grade()
        );
    }

    // Most popular
    println!("\nMost Popular (top 3):");
    for entry in index.most_popular(3) {
        println!("  - {} ({} downloads)", entry.name, entry.downloads);
    }

    // Highest quality
    println!("\nHighest Quality (top 3):");
    for entry in index.highest_quality(3) {
        println!(
            "  - {} ({:.0}, grade {})",
            entry.name,
            entry.quality_score,
            entry.quality_grade()
        );
    }
    println!();
}

fn zoo_stats_demo() {
    println!("--- Part 6: Zoo Statistics ---\n");

    let index = create_sample_index();
    let stats = index.stats();

    println!("Model Zoo Statistics:");
    println!("  Total Models: {}", stats.total_models);
    println!("  Total Downloads: {}", stats.total_downloads);
    println!("  Total Size: {}", stats.human_total_size());
    println!("  Average Quality: {:.1}", stats.avg_quality_score);

    println!("\nCategory Breakdown:");
    for (category, count) in &stats.category_counts {
        println!("  {}: {}", category.name(), count);
    }

    println!("\nTop Tags:");
    let mut tags: Vec<_> = stats.tag_counts.iter().collect();
    tags.sort_by(|a, b| b.1.cmp(a.1));
    for (tag, count) in tags.iter().take(5) {
        println!("  {}: {}", tag, count);
    }
    println!();
}

/// Create a sample index for demos
fn create_sample_index() -> ModelZooIndex {
    let mut index = ModelZooIndex::new("1.0.0");

    let models = vec![
        create_model(
            "iris-classifier",
            "Iris Classifier",
            ModelZooType::RandomForest,
            92.0,
            1500,
            2_500_000,
            &["classification", "beginner"],
        ),
        create_model(
            "sentiment-analyzer",
            "Sentiment Analyzer",
            ModelZooType::LogisticRegression,
            85.0,
            3200,
            1_200_000,
            &["nlp", "classification"],
        ),
        create_model(
            "customer-segments",
            "Customer Segmentation",
            ModelZooType::KMeans,
            78.0,
            800,
            500_000,
            &["clustering", "marketing"],
        ),
        create_model(
            "stock-lstm",
            "Stock Predictor LSTM",
            ModelZooType::NeuralNetwork,
            71.0,
            500,
            15_000_000,
            &["finance", "deep-learning", "time-series"],
        ),
        create_model(
            "resnet-50",
            "ResNet-50 ImageNet",
            ModelZooType::NeuralNetwork,
            94.5,
            12000,
            98_000_000,
            &["deep-learning", "computer-vision", "classification"],
        ),
        create_model(
            "xgboost-fraud",
            "Fraud Detection XGBoost",
            ModelZooType::GradientBoosting,
            89.0,
            2100,
            8_500_000,
            &["classification", "finance", "fraud"],
        ),
        create_model(
            "arima-sales",
            "Sales Forecasting ARIMA",
            ModelZooType::TimeSeries,
            82.0,
            650,
            250_000,
            &["time-series", "forecasting", "retail"],
        ),
        create_model(
            "knn-recommender",
            "KNN Recommender",
            ModelZooType::Knn,
            76.0,
            420,
            3_200_000,
            &["recommendation", "e-commerce"],
        ),
    ];

    for model in models {
        index.add_model(model);
    }

    index.feature_model("resnet-50");
    index.feature_model("iris-classifier");

    index
}

fn create_model(
    id: &str,
    name: &str,
    model_type: ModelZooType,
    quality: f32,
    downloads: u64,
    size: u64,
    tags: &[&str],
) -> ModelZooEntry {
    let mut entry = ModelZooEntry::new(id, name)
        .with_model_type(model_type)
        .with_quality_score(quality)
        .with_size(size);

    entry.downloads = downloads;

    for tag in tags {
        entry = entry.with_tag(*tag);
    }

    entry
}
