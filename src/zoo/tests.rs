use super::*;

#[test]
fn test_model_zoo_entry_new() {
    let entry = ModelZooEntry::new("test-model", "Test Model");
    assert_eq!(entry.id, "test-model");
    assert_eq!(entry.name, "Test Model");
    assert_eq!(entry.version, "1.0.0");
}

#[test]
fn test_model_zoo_entry_builder() {
    let entry = ModelZooEntry::new("lr-housing", "Linear Regression Housing")
        .with_description("Predicts housing prices")
        .with_version("2.0.0")
        .with_model_type(ModelZooType::LinearRegression)
        .with_quality_score(85.5)
        .with_tag("regression")
        .with_tag("housing")
        .with_license("Apache-2.0")
        .with_size(1024 * 1024);

    assert_eq!(entry.description, "Predicts housing prices");
    assert_eq!(entry.version, "2.0.0");
    assert_eq!(entry.model_type, ModelZooType::LinearRegression);
    assert!((entry.quality_score - 85.5).abs() < 0.01);
    assert_eq!(entry.tags.len(), 2);
    assert_eq!(entry.license, "Apache-2.0");
}

#[test]
fn test_model_zoo_entry_quality_grade() {
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(98.0)
            .quality_grade(),
        "A+"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(85.0)
            .quality_grade(),
        "B"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(70.0)
            .quality_grade(),
        "C-"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(50.0)
            .quality_grade(),
        "F"
    );
}

#[test]
fn test_model_zoo_entry_matches_query() {
    let entry = ModelZooEntry::new("linear-regression", "Linear Regression")
        .with_description("A linear model")
        .with_tag("regression");

    assert!(entry.matches_query("linear"));
    assert!(entry.matches_query("LINEAR")); // Case insensitive
    assert!(entry.matches_query("regression"));
    assert!(!entry.matches_query("forest"));
}

#[test]
fn test_model_zoo_entry_has_tag() {
    let entry = ModelZooEntry::new("test", "Test")
        .with_tag("machine-learning")
        .with_tag("regression");

    assert!(entry.has_tag("machine-learning"));
    assert!(entry.has_tag("REGRESSION")); // Case insensitive
    assert!(!entry.has_tag("classification"));
}

#[test]
fn test_model_zoo_entry_human_size() {
    let entry = ModelZooEntry::new("test", "Test").with_size(1024 * 1024 * 5); // 5 MB
    assert!(entry.human_size().contains("MB"));

    let entry_kb = ModelZooEntry::new("test", "Test").with_size(1024 * 500); // 500 KB
    assert!(entry_kb.human_size().contains("KB"));
}

#[test]
fn test_author_info() {
    let author = AuthorInfo::new("John Doe", "john@example.com")
        .with_organization("Acme Corp")
        .with_url("https://john.example.com");

    assert_eq!(author.name, "John Doe");
    assert_eq!(author.email, "john@example.com");
    assert_eq!(author.organization, Some("Acme Corp".to_string()));

    let display = format!("{author}");
    assert!(display.contains("John Doe"));
    assert!(display.contains("john@example.com"));
}

#[test]
fn test_model_zoo_type() {
    assert_eq!(ModelZooType::LinearRegression.name(), "Linear Regression");
    assert_eq!(
        ModelZooType::LinearRegression.category(),
        ModelCategory::Regression
    );
    assert_eq!(ModelZooType::KMeans.category(), ModelCategory::Clustering);
}

#[test]
fn test_model_zoo_index_new() {
    let index = ModelZooIndex::new("1.0.0");
    assert_eq!(index.version, "1.0.0");
    assert!(index.models.is_empty());
}

#[test]
fn test_model_zoo_index_add_model() {
    let mut index = ModelZooIndex::new("1.0.0");

    let entry = ModelZooEntry::new("test-model", "Test Model")
        .with_model_type(ModelZooType::LinearRegression)
        .with_tag("test");

    index.add_model(entry);

    assert_eq!(index.total_models(), 1);
    assert_eq!(index.categories.get(&ModelCategory::Regression), Some(&1));
    assert_eq!(index.tags.get("test"), Some(&1));
}

#[test]
fn test_model_zoo_index_search() {
    let mut index = ModelZooIndex::new("1.0.0");

    index.add_model(ModelZooEntry::new("linear-reg", "Linear Regression").with_tag("regression"));
    index
        .add_model(ModelZooEntry::new("random-forest", "Random Forest").with_tag("classification"));

    let results = index.search("linear");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "linear-reg");
}

#[test]
fn test_model_zoo_index_filter_by_tag() {
    let mut index = ModelZooIndex::new("1.0.0");

    index.add_model(ModelZooEntry::new("m1", "Model 1").with_tag("regression"));
    index.add_model(ModelZooEntry::new("m2", "Model 2").with_tag("regression"));
    index.add_model(ModelZooEntry::new("m3", "Model 3").with_tag("classification"));

    let results = index.filter_by_tag("regression");
    assert_eq!(results.len(), 2);
}

#[test]
fn test_model_zoo_index_filter_by_quality() {
    let mut index = ModelZooIndex::new("1.0.0");

    index.add_model(ModelZooEntry::new("m1", "High Quality").with_quality_score(90.0));
    index.add_model(ModelZooEntry::new("m2", "Medium Quality").with_quality_score(75.0));
    index.add_model(ModelZooEntry::new("m3", "Low Quality").with_quality_score(50.0));

    let high_quality = index.filter_by_quality(80.0);
    assert_eq!(high_quality.len(), 1);
    assert_eq!(high_quality[0].id, "m1");
}

#[test]
fn test_model_zoo_index_most_popular() {
    let mut index = ModelZooIndex::new("1.0.0");

    let mut m1 = ModelZooEntry::new("m1", "Model 1");
    m1.downloads = 100;

    let mut m2 = ModelZooEntry::new("m2", "Model 2");
    m2.downloads = 500;

    let mut m3 = ModelZooEntry::new("m3", "Model 3");
    m3.downloads = 50;

    index.add_model(m1);
    index.add_model(m2);
    index.add_model(m3);

    let popular = index.most_popular(2);
    assert_eq!(popular.len(), 2);
    assert_eq!(popular[0].id, "m2"); // Most downloads first
    assert_eq!(popular[1].id, "m1");
}

#[test]
fn test_model_zoo_index_featured() {
    let mut index = ModelZooIndex::new("1.0.0");

    index.add_model(ModelZooEntry::new("m1", "Model 1"));
    index.add_model(ModelZooEntry::new("m2", "Model 2"));

    index.feature_model("m1");
    index.feature_model("nonexistent"); // Should not panic

    assert_eq!(index.featured.len(), 1);
    assert_eq!(index.get_featured().len(), 1);
}

#[test]
fn test_model_zoo_index_stats() {
    let mut index = ModelZooIndex::new("1.0.0");

    let mut m1 = ModelZooEntry::new("m1", "Model 1")
        .with_quality_score(80.0)
        .with_model_type(ModelZooType::LinearRegression);
    m1.downloads = 100;
    m1.size_bytes = 1024;

    let mut m2 = ModelZooEntry::new("m2", "Model 2")
        .with_quality_score(90.0)
        .with_model_type(ModelZooType::RandomForest);
    m2.downloads = 200;
    m2.size_bytes = 2048;

    index.add_model(m1);
    index.add_model(m2);

    let stats = index.stats();
    assert_eq!(stats.total_models, 2);
    assert_eq!(stats.total_downloads, 300);
    assert_eq!(stats.total_size_bytes, 3072);
    assert!((stats.avg_quality_score - 85.0).abs() < 0.01);
}

#[test]
fn test_human_bytes() {
    assert_eq!(human_bytes(500), "500 B");
    assert!(human_bytes(1024 * 5).contains("KB"));
    assert!(human_bytes(1024 * 1024 * 10).contains("MB"));
    assert!(human_bytes(1024 * 1024 * 1024 * 2).contains("GB"));
}

#[test]
fn test_model_zoo_entry_display() {
    let entry = ModelZooEntry::new("test-model", "Test Model")
        .with_description("A test model")
        .with_quality_score(85.0)
        .with_tag("test");

    let display = format!("{entry}");
    assert!(display.contains("Test Model"));
    assert!(display.contains("test-model"));
    assert!(display.contains("85")); // Quality score rounded to integer
}

#[test]
fn test_all_tags() {
    let mut index = ModelZooIndex::new("1.0.0");
    index.add_model(
        ModelZooEntry::new("m1", "M1")
            .with_tag("b-tag")
            .with_tag("a-tag"),
    );
    index.add_model(ModelZooEntry::new("m2", "M2").with_tag("c-tag"));

    let tags = index.all_tags();
    assert_eq!(tags.len(), 3);
    assert_eq!(tags[0], "a-tag"); // Sorted
}

#[test]
fn test_model_zoo_entry_with_download_url() {
    let entry =
        ModelZooEntry::new("test", "Test").with_download_url("https://example.com/model.apr");
    assert_eq!(entry.download_url, "https://example.com/model.apr");
}

#[test]
fn test_model_zoo_entry_with_sha256() {
    let hash = "abc123def456";
    let entry = ModelZooEntry::new("test", "Test").with_sha256(hash);
    assert_eq!(entry.sha256, hash);
}

#[test]
fn test_model_zoo_entry_with_timestamps() {
    let entry = ModelZooEntry::new("test", "Test")
        .with_timestamps("2024-01-01T00:00:00Z", "2024-06-01T00:00:00Z");
    assert_eq!(entry.created_at, "2024-01-01T00:00:00Z");
    assert_eq!(entry.updated_at, "2024-06-01T00:00:00Z");
}

#[test]
fn test_model_zoo_entry_with_metadata() {
    let entry = ModelZooEntry::new("test", "Test")
        .with_metadata("custom_key", "custom_value")
        .with_metadata("another", "data");
    assert_eq!(
        entry.metadata.get("custom_key"),
        Some(&"custom_value".to_string())
    );
    assert_eq!(entry.metadata.get("another"), Some(&"data".to_string()));
}

#[test]
fn test_model_zoo_entry_quality_score_clamping() {
    let entry_high = ModelZooEntry::new("test", "Test").with_quality_score(150.0);
    assert_eq!(entry_high.quality_score, 100.0);

    let entry_low = ModelZooEntry::new("test", "Test").with_quality_score(-10.0);
    assert_eq!(entry_low.quality_score, 0.0);
}

#[test]
fn test_model_zoo_entry_quality_grade_all_levels() {
    // A grades
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(97.0)
            .quality_grade(),
        "A+"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(93.0)
            .quality_grade(),
        "A"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(90.0)
            .quality_grade(),
        "A-"
    );
    // B grades
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(87.0)
            .quality_grade(),
        "B+"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(83.0)
            .quality_grade(),
        "B"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(80.0)
            .quality_grade(),
        "B-"
    );
    // C grades
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(77.0)
            .quality_grade(),
        "C+"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(73.0)
            .quality_grade(),
        "C"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(70.0)
            .quality_grade(),
        "C-"
    );
    // D and F
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(60.0)
            .quality_grade(),
        "D"
    );
    assert_eq!(
        ModelZooEntry::new("a", "A")
            .with_quality_score(59.0)
            .quality_grade(),
        "F"
    );
}

#[test]
fn test_model_zoo_entry_matches_query_description() {
    let entry = ModelZooEntry::new("test", "Test").with_description("housing price prediction");
    assert!(entry.matches_query("housing"));
    assert!(entry.matches_query("PREDICTION"));
}

#[test]
fn test_author_info_default() {
    let author = AuthorInfo::default();
    assert_eq!(author.name, "");
    assert_eq!(author.email, "");
    assert!(author.url.is_none());
    assert!(author.organization.is_none());
}

#[test]
fn test_author_info_display_minimal() {
    let author = AuthorInfo::new("Alice", "");
    let display = format!("{}", author);
    assert!(display.contains("Alice"));
    assert!(!display.contains("<")); // No email brackets
}

#[test]
fn test_model_zoo_type_all_variants_name() {
    assert_eq!(
        ModelZooType::LogisticRegression.name(),
        "Logistic Regression"
    );
    assert_eq!(ModelZooType::DecisionTree.name(), "Decision Tree");
    assert_eq!(ModelZooType::RandomForest.name(), "Random Forest");
    assert_eq!(ModelZooType::GradientBoosting.name(), "Gradient Boosting");
    assert_eq!(ModelZooType::Knn.name(), "K-Nearest Neighbors");
    assert_eq!(ModelZooType::Svm.name(), "SVM");
    assert_eq!(ModelZooType::NaiveBayes.name(), "Naive Bayes");
    assert_eq!(ModelZooType::NeuralNetwork.name(), "Neural Network");
    assert_eq!(ModelZooType::Ensemble.name(), "Ensemble");
    assert_eq!(ModelZooType::TimeSeries.name(), "Time Series");
    assert_eq!(ModelZooType::Other.name(), "Other");
}

#[test]
fn test_model_zoo_type_all_variants_category() {
    assert_eq!(
        ModelZooType::LogisticRegression.category(),
        ModelCategory::Classification
    );
    assert_eq!(
        ModelZooType::DecisionTree.category(),
        ModelCategory::Classification
    );
    assert_eq!(
        ModelZooType::RandomForest.category(),
        ModelCategory::Classification
    );
    assert_eq!(
        ModelZooType::GradientBoosting.category(),
        ModelCategory::Classification
    );
    assert_eq!(ModelZooType::Knn.category(), ModelCategory::Classification);
    assert_eq!(ModelZooType::Svm.category(), ModelCategory::Classification);
    assert_eq!(
        ModelZooType::NaiveBayes.category(),
        ModelCategory::Classification
    );
    assert_eq!(
        ModelZooType::NeuralNetwork.category(),
        ModelCategory::DeepLearning
    );
    assert_eq!(ModelZooType::Ensemble.category(), ModelCategory::Ensemble);
    assert_eq!(
        ModelZooType::TimeSeries.category(),
        ModelCategory::TimeSeries
    );
    assert_eq!(ModelZooType::Other.category(), ModelCategory::Other);
}

#[test]
fn test_model_zoo_type_display() {
    let t = ModelZooType::LinearRegression;
    assert_eq!(format!("{}", t), "Linear Regression");
}

#[test]
fn test_model_category_all_names() {
    assert_eq!(ModelCategory::Regression.name(), "Regression");
    assert_eq!(ModelCategory::Classification.name(), "Classification");
    assert_eq!(ModelCategory::Clustering.name(), "Clustering");
    assert_eq!(ModelCategory::DeepLearning.name(), "Deep Learning");
    assert_eq!(ModelCategory::Ensemble.name(), "Ensemble");
    assert_eq!(ModelCategory::TimeSeries.name(), "Time Series");
    assert_eq!(ModelCategory::Other.name(), "Other");
}

#[test]
fn test_model_category_display() {
    assert_eq!(format!("{}", ModelCategory::Regression), "Regression");
    assert_eq!(format!("{}", ModelCategory::DeepLearning), "Deep Learning");
}

#[test]
fn test_model_zoo_index_get_model() {
    let mut index = ModelZooIndex::new("1.0.0");
    index.add_model(ModelZooEntry::new("m1", "Model 1"));
    index.add_model(ModelZooEntry::new("m2", "Model 2"));

    assert!(index.get_model("m1").is_some());
    assert!(index.get_model("m2").is_some());
    assert!(index.get_model("nonexistent").is_none());
}

#[test]
fn test_model_zoo_index_filter_by_category() {
    let mut index = ModelZooIndex::new("1.0.0");
    index.add_model(ModelZooEntry::new("lr", "LR").with_model_type(ModelZooType::LinearRegression));
    index.add_model(ModelZooEntry::new("rf", "RF").with_model_type(ModelZooType::RandomForest));
    index.add_model(ModelZooEntry::new("km", "KM").with_model_type(ModelZooType::KMeans));

    let regression = index.filter_by_category(ModelCategory::Regression);
    assert_eq!(regression.len(), 1);
    assert_eq!(regression[0].id, "lr");

    let classification = index.filter_by_category(ModelCategory::Classification);
    assert_eq!(classification.len(), 1);
    assert_eq!(classification[0].id, "rf");

    let clustering = index.filter_by_category(ModelCategory::Clustering);
    assert_eq!(clustering.len(), 1);
    assert_eq!(clustering[0].id, "km");
}

#[test]
fn test_model_zoo_index_highest_quality() {
    let mut index = ModelZooIndex::new("1.0.0");
    index.add_model(ModelZooEntry::new("m1", "M1").with_quality_score(70.0));
    index.add_model(ModelZooEntry::new("m2", "M2").with_quality_score(95.0));
    index.add_model(ModelZooEntry::new("m3", "M3").with_quality_score(85.0));

    let highest = index.highest_quality(2);
    assert_eq!(highest.len(), 2);
    assert_eq!(highest[0].id, "m2"); // Highest first
    assert_eq!(highest[1].id, "m3");
}

#[test]
fn test_model_zoo_index_most_recent() {
    let mut index = ModelZooIndex::new("1.0.0");
    index.add_model(ModelZooEntry::new("m1", "M1").with_timestamps("2023-01-01", "2023-01-01"));
    index.add_model(ModelZooEntry::new("m2", "M2").with_timestamps("2024-06-01", "2024-06-01"));
    index.add_model(ModelZooEntry::new("m3", "M3").with_timestamps("2024-01-01", "2024-01-01"));

    let recent = index.most_recent(2);
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].id, "m2"); // Most recent first
    assert_eq!(recent[1].id, "m3");
}

#[test]
fn test_model_zoo_index_feature_model_duplicate() {
    let mut index = ModelZooIndex::new("1.0.0");
    index.add_model(ModelZooEntry::new("m1", "Model 1"));

    index.feature_model("m1");
    index.feature_model("m1"); // Should not add duplicate

    assert_eq!(index.featured.len(), 1);
}

#[test]
fn test_model_zoo_index_display() {
    let mut index = ModelZooIndex::new("2.0.0");
    index.add_model(
        ModelZooEntry::new("m1", "M1")
            .with_model_type(ModelZooType::LinearRegression)
            .with_tag("test"),
    );
    index.feature_model("m1");

    let display = format!("{}", index);
    assert!(display.contains("2.0.0"));
    assert!(display.contains("Total Models: 1"));
    assert!(display.contains("Featured: 1"));
}

#[test]
fn test_model_zoo_entry_display_with_tags() {
    let entry = ModelZooEntry::new("test", "Test Model")
        .with_description("Description")
        .with_quality_score(85.0)
        .with_tag("ml")
        .with_tag("regression");

    let display = format!("{}", entry);
    assert!(display.contains("Test Model"));
    assert!(display.contains("Description"));
    assert!(display.contains("Tags: ml, regression"));
}

#[test]
fn test_model_zoo_entry_display_no_description() {
    let entry = ModelZooEntry::new("test", "Test Model").with_quality_score(85.0);

    let display = format!("{}", entry);
    assert!(display.contains("Test Model"));
    assert!(!display.contains("  \n  \n")); // No double newline from empty description
}

#[test]
fn test_zoo_stats_human_total_size() {
    let mut index = ModelZooIndex::new("1.0.0");
    let mut m1 = ModelZooEntry::new("m1", "M1");
    m1.size_bytes = 1024 * 1024 * 500; // 500 MB
    index.add_model(m1);

    let stats = index.stats();
    let human = stats.human_total_size();
    assert!(human.contains("MB"));
}

#[test]
fn test_zoo_stats_empty_index() {
    let index = ModelZooIndex::new("1.0.0");
    let stats = index.stats();

    assert_eq!(stats.total_models, 0);
    assert_eq!(stats.total_downloads, 0);
    assert_eq!(stats.total_size_bytes, 0);
    assert_eq!(stats.avg_quality_score, 0.0);
}

#[test]
fn test_human_bytes_boundaries() {
    assert_eq!(human_bytes(0), "0 B");
    assert_eq!(human_bytes(1023), "1023 B");
    assert!(human_bytes(1024).contains("KB"));
    assert!(human_bytes(1024 * 1024 - 1).contains("KB"));
    assert!(human_bytes(1024 * 1024).contains("MB"));
    assert!(human_bytes(1024 * 1024 * 1024 - 1).contains("MB"));
    assert!(human_bytes(1024 * 1024 * 1024).contains("GB"));
}

#[test]
fn test_model_zoo_entry_with_author() {
    let author = AuthorInfo::new("Test Author", "test@example.com").with_organization("Test Org");
    let entry = ModelZooEntry::new("test", "Test").with_author(author);

    assert_eq!(entry.author.name, "Test Author");
    assert_eq!(entry.author.organization, Some("Test Org".to_string()));
}

#[test]
fn test_model_zoo_entry_with_model_type() {
    let entry = ModelZooEntry::new("test", "Test").with_model_type(ModelZooType::NeuralNetwork);
    assert_eq!(entry.model_type, ModelZooType::NeuralNetwork);
}
