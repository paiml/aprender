
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
