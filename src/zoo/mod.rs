//! Model Zoo Protocol (spec §8)
//!
//! Defines the protocol for model sharing and discovery in the Aprender ecosystem.
//! Supports interactive.paiml.com, presentar, and the model zoo repository.
//!
//! # Overview
//!
//! The Model Zoo provides:
//! - Standardized model metadata format for discovery
//! - Quality score caching for quick filtering
//! - Version management for model evolution
//! - Download tracking and popularity metrics
//!
//! # Example
//!
//! ```
//! use aprender::zoo::{ModelZooEntry, ModelZooIndex, AuthorInfo};
//!
//! let entry = ModelZooEntry::new(
//!     "linear-regression-housing",
//!     "Housing Price Predictor",
//! )
//! .with_description("Linear regression model for predicting housing prices")
//! .with_author(AuthorInfo::new("Jane Doe", "jane@example.com"))
//! .with_quality_score(85.5)
//! .with_tag("regression")
//! .with_tag("housing");
//!
//! assert_eq!(entry.id, "linear-regression-housing");
//! ```

use std::collections::HashMap;
use std::fmt;

/// Model zoo entry for sharing and discovery
#[derive(Debug, Clone)]
pub struct ModelZooEntry {
    /// Unique model identifier (slug format)
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Model description
    pub description: String,

    /// Version string (semver)
    pub version: String,

    /// Author information
    pub author: AuthorInfo,

    /// Model type identifier
    pub model_type: ModelZooType,

    /// Quality score (cached, 0-100)
    pub quality_score: f32,

    /// Tags for discovery
    pub tags: Vec<String>,

    /// Download URL
    pub download_url: String,

    /// File size in bytes
    pub size_bytes: u64,

    /// SHA-256 hash for verification
    pub sha256: String,

    /// License identifier (SPDX)
    pub license: String,

    /// Creation timestamp (ISO 8601)
    pub created_at: String,

    /// Last updated timestamp (ISO 8601)
    pub updated_at: String,

    /// Download count
    pub downloads: u64,

    /// Star count (user favorites)
    pub stars: u64,

    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl ModelZooEntry {
    /// Create a new model zoo entry
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            version: "1.0.0".to_string(),
            author: AuthorInfo::default(),
            model_type: ModelZooType::Other,
            quality_score: 0.0,
            tags: Vec::new(),
            download_url: String::new(),
            size_bytes: 0,
            sha256: String::new(),
            license: "MIT".to_string(),
            created_at: String::new(),
            updated_at: String::new(),
            downloads: 0,
            stars: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set description
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set version
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set author
    #[must_use]
    pub fn with_author(mut self, author: AuthorInfo) -> Self {
        self.author = author;
        self
    }

    /// Set model type
    #[must_use]
    pub fn with_model_type(mut self, model_type: ModelZooType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set quality score
    #[must_use]
    pub fn with_quality_score(mut self, score: f32) -> Self {
        self.quality_score = score.clamp(0.0, 100.0);
        self
    }

    /// Add a tag
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set download URL
    #[must_use]
    pub fn with_download_url(mut self, url: impl Into<String>) -> Self {
        self.download_url = url.into();
        self
    }

    /// Set file size
    #[must_use]
    pub fn with_size(mut self, size_bytes: u64) -> Self {
        self.size_bytes = size_bytes;
        self
    }

    /// Set SHA-256 hash
    #[must_use]
    pub fn with_sha256(mut self, hash: impl Into<String>) -> Self {
        self.sha256 = hash.into();
        self
    }

    /// Set license
    #[must_use]
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = license.into();
        self
    }

    /// Set timestamps
    #[must_use]
    pub fn with_timestamps(
        mut self,
        created: impl Into<String>,
        updated: impl Into<String>,
    ) -> Self {
        self.created_at = created.into();
        self.updated_at = updated.into();
        self
    }

    /// Add custom metadata
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get human-readable file size
    #[must_use]
    pub fn human_size(&self) -> String {
        human_bytes(self.size_bytes)
    }

    /// Check if entry matches search query
    #[must_use]
    pub fn matches_query(&self, query: &str) -> bool {
        let query_lower = query.to_lowercase();
        self.id.to_lowercase().contains(&query_lower)
            || self.name.to_lowercase().contains(&query_lower)
            || self.description.to_lowercase().contains(&query_lower)
            || self
                .tags
                .iter()
                .any(|t| t.to_lowercase().contains(&query_lower))
    }

    /// Check if entry has tag
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        let tag_lower = tag.to_lowercase();
        self.tags.iter().any(|t| t.to_lowercase() == tag_lower)
    }

    /// Get quality grade letter
    #[must_use]
    pub fn quality_grade(&self) -> &'static str {
        match self.quality_score {
            s if s >= 97.0 => "A+",
            s if s >= 93.0 => "A",
            s if s >= 90.0 => "A-",
            s if s >= 87.0 => "B+",
            s if s >= 83.0 => "B",
            s if s >= 80.0 => "B-",
            s if s >= 77.0 => "C+",
            s if s >= 73.0 => "C",
            s if s >= 70.0 => "C-",
            s if s >= 60.0 => "D",
            _ => "F",
        }
    }
}

impl fmt::Display for ModelZooEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} ({})", self.name, self.id)?;
        writeln!(
            f,
            "  Version: {} | Size: {} | Quality: {:.0} ({})",
            self.version,
            self.human_size(),
            self.quality_score,
            self.quality_grade()
        )?;
        if !self.description.is_empty() {
            writeln!(f, "  {}", self.description)?;
        }
        if !self.tags.is_empty() {
            writeln!(f, "  Tags: {}", self.tags.join(", "))?;
        }
        Ok(())
    }
}

/// Author information
#[derive(Debug, Clone, Default)]
pub struct AuthorInfo {
    /// Author name
    pub name: String,
    /// Author email
    pub email: String,
    /// Author URL (optional)
    pub url: Option<String>,
    /// Organization (optional)
    pub organization: Option<String>,
}

impl AuthorInfo {
    /// Create new author info
    #[must_use]
    pub fn new(name: impl Into<String>, email: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            email: email.into(),
            url: None,
            organization: None,
        }
    }

    /// Set URL
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set organization
    #[must_use]
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }
}

impl fmt::Display for AuthorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.email.is_empty() {
            write!(f, " <{}>", self.email)?;
        }
        if let Some(org) = &self.organization {
            write!(f, " ({org})")?;
        }
        Ok(())
    }
}

/// Model type for zoo classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModelZooType {
    /// Linear regression
    LinearRegression,
    /// Logistic regression
    LogisticRegression,
    /// Decision tree
    DecisionTree,
    /// Random forest
    RandomForest,
    /// Gradient boosting
    GradientBoosting,
    /// K-Nearest Neighbors
    Knn,
    /// K-Means clustering
    KMeans,
    /// Support Vector Machine
    Svm,
    /// Naive Bayes
    NaiveBayes,
    /// Neural network
    NeuralNetwork,
    /// Ensemble model
    Ensemble,
    /// Time series (ARIMA, etc.)
    TimeSeries,
    /// Other/custom
    #[default]
    Other,
}

impl ModelZooType {
    /// Get type name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::LinearRegression => "Linear Regression",
            Self::LogisticRegression => "Logistic Regression",
            Self::DecisionTree => "Decision Tree",
            Self::RandomForest => "Random Forest",
            Self::GradientBoosting => "Gradient Boosting",
            Self::Knn => "K-Nearest Neighbors",
            Self::KMeans => "K-Means",
            Self::Svm => "SVM",
            Self::NaiveBayes => "Naive Bayes",
            Self::NeuralNetwork => "Neural Network",
            Self::Ensemble => "Ensemble",
            Self::TimeSeries => "Time Series",
            Self::Other => "Other",
        }
    }

    /// Get category
    #[must_use]
    pub const fn category(&self) -> ModelCategory {
        match self {
            Self::LinearRegression => ModelCategory::Regression,
            Self::LogisticRegression
            | Self::DecisionTree
            | Self::RandomForest
            | Self::GradientBoosting
            | Self::Knn
            | Self::Svm
            | Self::NaiveBayes => ModelCategory::Classification,
            Self::KMeans => ModelCategory::Clustering,
            Self::NeuralNetwork => ModelCategory::DeepLearning,
            Self::Ensemble => ModelCategory::Ensemble,
            Self::TimeSeries => ModelCategory::TimeSeries,
            Self::Other => ModelCategory::Other,
        }
    }
}

impl fmt::Display for ModelZooType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Model category for filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCategory {
    /// Regression models
    Regression,
    /// Classification models
    Classification,
    /// Clustering models
    Clustering,
    /// Deep learning models
    DeepLearning,
    /// Ensemble models
    Ensemble,
    /// Time series models
    TimeSeries,
    /// Other
    Other,
}

impl ModelCategory {
    /// Get category name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Regression => "Regression",
            Self::Classification => "Classification",
            Self::Clustering => "Clustering",
            Self::DeepLearning => "Deep Learning",
            Self::Ensemble => "Ensemble",
            Self::TimeSeries => "Time Series",
            Self::Other => "Other",
        }
    }
}

impl fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Model zoo index for discovery
#[derive(Debug, Clone, Default)]
pub struct ModelZooIndex {
    /// Index version
    pub version: String,

    /// Last updated timestamp (ISO 8601)
    pub updated_at: String,

    /// Models in the zoo
    pub models: Vec<ModelZooEntry>,

    /// Featured model IDs (for homepage display)
    pub featured: Vec<String>,

    /// Categories with counts
    pub categories: HashMap<ModelCategory, usize>,

    /// Tags with counts
    pub tags: HashMap<String, usize>,
}

impl ModelZooIndex {
    /// Create a new empty index
    #[must_use]
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            updated_at: String::new(),
            models: Vec::new(),
            featured: Vec::new(),
            categories: HashMap::new(),
            tags: HashMap::new(),
        }
    }

    /// Add a model to the index
    pub fn add_model(&mut self, entry: ModelZooEntry) {
        // Update category counts
        let category = entry.model_type.category();
        *self.categories.entry(category).or_insert(0) += 1;

        // Update tag counts
        for tag in &entry.tags {
            *self.tags.entry(tag.clone()).or_insert(0) += 1;
        }

        self.models.push(entry);
    }

    /// Mark model as featured
    pub fn feature_model(&mut self, id: impl Into<String>) {
        let id = id.into();
        if self.models.iter().any(|m| m.id == id) && !self.featured.contains(&id) {
            self.featured.push(id);
        }
    }

    /// Get model by ID
    #[must_use]
    pub fn get_model(&self, id: &str) -> Option<&ModelZooEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Get featured models
    #[must_use]
    pub fn get_featured(&self) -> Vec<&ModelZooEntry> {
        self.featured
            .iter()
            .filter_map(|id| self.get_model(id))
            .collect()
    }

    /// Search models by query
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&ModelZooEntry> {
        self.models
            .iter()
            .filter(|m| m.matches_query(query))
            .collect()
    }

    /// Filter models by tag
    #[must_use]
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&ModelZooEntry> {
        self.models.iter().filter(|m| m.has_tag(tag)).collect()
    }

    /// Filter models by category
    #[must_use]
    pub fn filter_by_category(&self, category: ModelCategory) -> Vec<&ModelZooEntry> {
        self.models
            .iter()
            .filter(|m| m.model_type.category() == category)
            .collect()
    }

    /// Filter models by minimum quality score
    #[must_use]
    pub fn filter_by_quality(&self, min_score: f32) -> Vec<&ModelZooEntry> {
        self.models
            .iter()
            .filter(|m| m.quality_score >= min_score)
            .collect()
    }

    /// Get models sorted by downloads (most popular)
    #[must_use]
    pub fn most_popular(&self, limit: usize) -> Vec<&ModelZooEntry> {
        let mut models: Vec<_> = self.models.iter().collect();
        models.sort_by(|a, b| b.downloads.cmp(&a.downloads));
        models.into_iter().take(limit).collect()
    }

    /// Get models sorted by quality score
    #[must_use]
    pub fn highest_quality(&self, limit: usize) -> Vec<&ModelZooEntry> {
        let mut models: Vec<_> = self.models.iter().collect();
        models.sort_by(|a, b| {
            b.quality_score
                .partial_cmp(&a.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        models.into_iter().take(limit).collect()
    }

    /// Get recent models
    #[must_use]
    pub fn most_recent(&self, limit: usize) -> Vec<&ModelZooEntry> {
        let mut models: Vec<_> = self.models.iter().collect();
        models.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        models.into_iter().take(limit).collect()
    }

    /// Total model count
    #[must_use]
    pub fn total_models(&self) -> usize {
        self.models.len()
    }

    /// Get all unique tags
    #[must_use]
    pub fn all_tags(&self) -> Vec<&str> {
        let mut tags: Vec<_> = self.tags.keys().map(String::as_str).collect();
        tags.sort_unstable();
        tags
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> ZooStats {
        let total_downloads: u64 = self.models.iter().map(|m| m.downloads).sum();
        let total_size: u64 = self.models.iter().map(|m| m.size_bytes).sum();
        let avg_quality = if self.models.is_empty() {
            0.0
        } else {
            self.models.iter().map(|m| m.quality_score).sum::<f32>() / self.models.len() as f32
        };

        ZooStats {
            total_models: self.models.len(),
            total_downloads,
            total_size_bytes: total_size,
            avg_quality_score: avg_quality,
            category_counts: self.categories.clone(),
            tag_counts: self.tags.clone(),
        }
    }
}

impl fmt::Display for ModelZooIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model Zoo Index v{}", self.version)?;
        writeln!(f, "Total Models: {}", self.models.len())?;
        writeln!(f, "Featured: {}", self.featured.len())?;
        writeln!(
            f,
            "Categories: {:?}",
            self.categories.keys().collect::<Vec<_>>()
        )?;
        writeln!(f, "Tags: {}", self.tags.len())?;
        Ok(())
    }
}

/// Zoo statistics
#[derive(Debug, Clone)]
pub struct ZooStats {
    /// Total number of models
    pub total_models: usize,
    /// Total downloads across all models
    pub total_downloads: u64,
    /// Total size of all models in bytes
    pub total_size_bytes: u64,
    /// Average quality score
    pub avg_quality_score: f32,
    /// Category counts
    pub category_counts: HashMap<ModelCategory, usize>,
    /// Tag counts
    pub tag_counts: HashMap<String, usize>,
}

impl ZooStats {
    /// Get human-readable total size
    #[must_use]
    pub fn human_total_size(&self) -> String {
        human_bytes(self.total_size_bytes)
    }
}

/// Convert bytes to human-readable string
fn human_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
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

        index.add_model(
            ModelZooEntry::new("linear-reg", "Linear Regression").with_tag("regression"),
        );
        index.add_model(
            ModelZooEntry::new("random-forest", "Random Forest").with_tag("classification"),
        );

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
        index.add_model(
            ModelZooEntry::new("lr", "LR").with_model_type(ModelZooType::LinearRegression),
        );
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
        let author =
            AuthorInfo::new("Test Author", "test@example.com").with_organization("Test Org");
        let entry = ModelZooEntry::new("test", "Test").with_author(author);

        assert_eq!(entry.author.name, "Test Author");
        assert_eq!(entry.author.organization, Some("Test Org".to_string()));
    }

    #[test]
    fn test_model_zoo_entry_with_model_type() {
        let entry = ModelZooEntry::new("test", "Test").with_model_type(ModelZooType::NeuralNetwork);
        assert_eq!(entry.model_type, ModelZooType::NeuralNetwork);
    }
}
