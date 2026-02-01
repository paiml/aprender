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
mod tests;
