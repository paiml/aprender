
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
